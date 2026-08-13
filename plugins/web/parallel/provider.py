"""Parallel.ai web search + content extraction — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider` and uses the
native ``AsyncParallel`` SDK client for both search and extraction.

Config keys this provider responds to::

    web:
      search_backend: "parallel"      # explicit per-capability
      extract_backend: "parallel"     # explicit per-capability
      backend: "parallel"             # shared fallback
      # Optional: search mode (default "agentic"; also "fast" or "one-shot")
      # via the PARALLEL_SEARCH_MODE env var.

Env vars::

    PARALLEL_API_KEY=...             # https://parallel.ai (required)
    PARALLEL_SEARCH_MODE=agentic     # optional: agentic|fast|one-shot
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import logging
import os
import threading
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List

import aiofiles.os

from agent.web_search_provider import WebSearchProvider
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Keep the optional client construction lazy while loading the installed SDK
# with the plugin module instead of during the first awaited search.
try:
    import parallel as _ParallelBootstrap  # noqa: F401
except Exception:
    _ParallelBootstrap = None


@dataclass
class _ParallelClientEntry:
    fingerprint: str
    client: Any
    active_leases: int = 0
    retired: bool = False
    closed: bool = False


@dataclass
class _ParallelScopeState:
    profile_home: str
    entries: dict[str, _ParallelClientEntry] = field(default_factory=dict)
    consumers: weakref.WeakSet[object] = field(default_factory=weakref.WeakSet)
    lock_ref: weakref.ReferenceType[asyncio.Lock] | None = None


_ParallelScope = tuple[asyncio.AbstractEventLoop, str]
_parallel_scope_states: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, _ParallelScopeState]
] = weakref.WeakKeyDictionary()
_parallel_scope_aliases: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, str]
] = weakref.WeakKeyDictionary()
_parallel_owner_scopes: weakref.WeakKeyDictionary[
    object, tuple[weakref.ReferenceType[asyncio.AbstractEventLoop], str]
] = weakref.WeakKeyDictionary()
_parallel_scope_guard = threading.RLock()
_parallel_scope_context: contextvars.ContextVar[
    tuple[str, str] | None
] = contextvars.ContextVar("parallel_web_profile_scope", default=None)
_parallel_reset_profiles: set[str] = set()


async def _finish_owned_task(task: asyncio.Task[Any]) -> Any:
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised below
            if task.cancelled():
                raise
            if cancellation is None:
                cancellation = exc
        except Exception as exc:
            if cancellation is not None:
                raise cancellation from exc
            raise
    if cancellation is not None:
        raise cancellation
    return result


def _lexical_parallel_home() -> str:
    return os.path.normcase(os.fspath(get_hermes_home()))


def _prune_closed_parallel_loops() -> None:
    with _parallel_scope_guard:
        known_loops = set(_parallel_scope_states) | set(_parallel_scope_aliases)
        for loop in known_loops:
            if loop.is_closed():
                _parallel_scope_states.pop(loop, None)
                _parallel_scope_aliases.pop(loop, None)
        for owner, (loop_ref, _profile) in tuple(_parallel_owner_scopes.items()):
            loop = loop_ref()
            if loop is None or loop.is_closed():
                _parallel_owner_scopes.pop(owner, None)


async def _activate_parallel_scope() -> _ParallelScope:
    _prune_closed_parallel_loops()
    loop = asyncio.get_running_loop()
    lexical = _lexical_parallel_home()
    active = _parallel_scope_context.get()
    if active is not None and active[0] == lexical:
        canonical = active[1]
    else:
        with _parallel_scope_guard:
            aliases = _parallel_scope_aliases.get(loop)
            canonical = aliases.get(lexical) if aliases is not None else None
        if canonical is None:
            expanduser = aiofiles.os.wrap(os.path.expanduser)
            expanded = str(await expanduser(lexical))
            is_absolute = (
                expanded.startswith(("/", "\\\\"))
                or (
                    len(expanded) >= 3
                    and expanded[1] == ":"
                    and expanded[2] in "/\\"
                )
            )
            if not is_absolute:
                expanded = str(await aiofiles.os.getcwd()) + os.sep + expanded
            realpath = aiofiles.os.wrap(os.path.realpath)
            canonical = os.path.normcase(str(await realpath(expanded)))
        with _parallel_scope_guard:
            _parallel_scope_aliases.setdefault(loop, {})[lexical] = canonical
    _parallel_scope_context.set((lexical, canonical))
    return loop, canonical


def _state_for_scope(scope: _ParallelScope) -> _ParallelScopeState:
    loop, profile = scope
    with _parallel_scope_guard:
        states = _parallel_scope_states.setdefault(loop, {})
        return states.setdefault(profile, _ParallelScopeState(profile))


def _existing_state_for_scope(
    scope: _ParallelScope,
) -> _ParallelScopeState | None:
    loop, profile = scope
    with _parallel_scope_guard:
        states = _parallel_scope_states.get(loop)
        return states.get(profile) if states is not None else None


def _state_lock(state: _ParallelScopeState) -> asyncio.Lock:
    with _parallel_scope_guard:
        lock = state.lock_ref() if state.lock_ref is not None else None
        if lock is None:
            lock = asyncio.Lock()
            state.lock_ref = weakref.ref(lock)
        return lock


def _discard_parallel_scope_if_idle(
    scope: _ParallelScope,
    state: _ParallelScopeState,
) -> None:
    lock = state.lock_ref() if state.lock_ref is not None else None
    if state.entries or state.consumers or (lock is not None and lock.locked()):
        return
    loop, profile = scope
    with _parallel_scope_guard:
        states = _parallel_scope_states.get(loop)
        if states is None or states.get(profile) is not state:
            return
        states.pop(profile, None)
        if not states:
            _parallel_scope_states.pop(loop, None)


def _api_key_fingerprint(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


async def _close_parallel_entry(entry: _ParallelClientEntry) -> None:
    if entry.closed:
        return
    entry.closed = True
    try:
        await entry.client.close()
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("Parallel async client close failed", exc_info=True)


async def _close_parallel_entries(entries: list[_ParallelClientEntry]) -> None:
    for entry in entries:
        await _close_parallel_entry(entry)


async def _release_parallel_entry(
    scope: _ParallelScope,
    state: _ParallelScopeState,
    entry: _ParallelClientEntry,
) -> None:
    close_entry = False
    async with _state_lock(state):
        entry.active_leases = max(0, entry.active_leases - 1)
        if entry.active_leases == 0 and (entry.retired or not state.consumers):
            if state.entries.get(entry.fingerprint) is entry:
                state.entries.pop(entry.fingerprint, None)
            close_entry = True
    if close_entry:
        await _close_parallel_entry(entry)
    _discard_parallel_scope_if_idle(scope, state)


async def _acquire_parallel_entry() -> tuple[
    _ParallelScope,
    _ParallelScopeState,
    _ParallelClientEntry,
]:
    scope = await _activate_parallel_scope()
    state = _state_for_scope(scope)
    from agent.web_search_provider import get_provider_env

    try:
        api_key = await get_provider_env("PARALLEL_API_KEY")
    except BaseException:
        _discard_parallel_scope_if_idle(scope, state)
        raise
    if not api_key:
        _discard_parallel_scope_if_idle(scope, state)
        raise ValueError(
            "PARALLEL_API_KEY environment variable not set. "
            "Get your API key at https://parallel.ai"
        )

    try:
        from parallel import AsyncParallel  # noqa: WPS433 — deliberately lazy
    except ImportError as exc:
        _discard_parallel_scope_if_idle(scope, state)
        raise ImportError(
            "The optional Parallel SDK is not installed. "
            "Install async-hermes-agent[parallel-web]."
        ) from exc

    fingerprint = _api_key_fingerprint(api_key)
    to_close: list[_ParallelClientEntry] = []
    try:
        async with _state_lock(state):
            reset_requested = (
                state.profile_home in _parallel_reset_profiles
                or _lexical_parallel_home() in _parallel_reset_profiles
            )
            if reset_requested:
                _parallel_reset_profiles.discard(state.profile_home)
                _parallel_reset_profiles.discard(_lexical_parallel_home())
                for old in state.entries.values():
                    old.retired = True
                    if old.active_leases == 0:
                        to_close.append(old)
                state.entries.clear()

            entry = state.entries.get(fingerprint)
            if entry is None or entry.retired or entry.closed:
                for old_fingerprint, old in tuple(state.entries.items()):
                    if old_fingerprint == fingerprint:
                        continue
                    old.retired = True
                    state.entries.pop(old_fingerprint, None)
                    if old.active_leases == 0:
                        to_close.append(old)
                entry = _ParallelClientEntry(
                    fingerprint=fingerprint,
                    client=AsyncParallel(api_key=api_key),
                )
                state.entries[fingerprint] = entry
            entry.active_leases += 1
    except BaseException:
        if to_close:
            cleanup = asyncio.create_task(
                _close_parallel_entries(to_close),
                name="parallel-web-failed-acquire-close",
            )
            await _finish_owned_task(cleanup)
        _discard_parallel_scope_if_idle(scope, state)
        raise

    if to_close:
        cleanup = asyncio.create_task(
            _close_parallel_entries(to_close),
            name="parallel-web-retired-client-close",
        )
        try:
            await _finish_owned_task(cleanup)
        except BaseException:
            release = asyncio.create_task(
                _release_parallel_entry(scope, state, entry),
                name="parallel-web-acquire-rollback",
            )
            await _finish_owned_task(release)
            raise
    return scope, state, entry


@asynccontextmanager
async def _parallel_client_lease():
    scope, state, entry = await _acquire_parallel_entry()
    try:
        yield entry.client
    finally:
        cleanup = asyncio.create_task(
            _release_parallel_entry(scope, state, entry),
            name="parallel-web-client-lease-release",
        )
        await _finish_owned_task(cleanup)


async def _get_parallel_client() -> Any:
    """Lazy-load and cache the active profile's native async client."""
    _scope, state, entry = await _acquire_parallel_entry()
    # This retained private helper historically returns a cached client rather
    # than a lease. Balance the request counter without closing an agent-owned
    # cache; standalone runtime calls use ``_parallel_client_lease`` below.
    async with _state_lock(state):
        entry.active_leases = max(0, entry.active_leases - 1)
    return entry.client


def _reset_clients_for_tests() -> None:
    """Drop the cached client so tests can re-instantiate cleanly."""
    with _parallel_scope_guard:
        _parallel_reset_profiles.add(_lexical_parallel_home())


async def _retain_parallel_lifecycle(owner: object) -> None:
    scope = await _activate_parallel_scope()
    state = _state_for_scope(scope)
    try:
        async with _state_lock(state):
            state.consumers.add(owner)
            with _parallel_scope_guard:
                _parallel_owner_scopes[owner] = (
                    weakref.ref(scope[0]),
                    scope[1],
                )
    except BaseException:
        _discard_parallel_scope_if_idle(scope, state)
        raise


async def _release_parallel_lifecycle(owner: object) -> None:
    loop = asyncio.get_running_loop()
    with _parallel_scope_guard:
        retained = _parallel_owner_scopes.get(owner)
    if retained is None:
        return
    if retained[0]() is not loop:
        raise RuntimeError(
            "The Parallel web client lifecycle lease belongs to another event "
            "loop; release it on its owning loop"
        )
    scope = (loop, retained[1])
    state = _existing_state_for_scope(scope)
    if state is None:
        with _parallel_scope_guard:
            _parallel_owner_scopes.pop(owner, None)
        return
    cleanup = asyncio.create_task(
        _release_parallel_lifecycle_owned(owner, scope, state),
        name="parallel-web-lifecycle-release",
    )
    await _finish_owned_task(cleanup)


async def _release_parallel_lifecycle_owned(
    owner: object,
    scope: _ParallelScope,
    state: _ParallelScopeState,
) -> None:
    to_close: list[_ParallelClientEntry] = []
    async with _state_lock(state):
        state.consumers.discard(owner)
        with _parallel_scope_guard:
            _parallel_owner_scopes.pop(owner, None)
        if state.consumers:
            return
        for fingerprint, entry in tuple(state.entries.items()):
            entry.retired = True
            state.entries.pop(fingerprint, None)
            if entry.active_leases == 0:
                to_close.append(entry)
    await _close_parallel_entries(to_close)
    _discard_parallel_scope_if_idle(scope, state)


async def _resolve_search_mode() -> str:
    """Return the validated PARALLEL_SEARCH_MODE value (default "agentic")."""
    from agent.web_search_provider import get_provider_env

    mode = (
        await get_provider_env("PARALLEL_SEARCH_MODE") or "agentic"
    ).lower().strip()
    if mode not in {"fast", "one-shot", "agentic"}:
        mode = "agentic"
    return mode


class ParallelWebSearchProvider(WebSearchProvider):
    """Parallel.ai search + async extract provider."""

    @property
    def name(self) -> str:
        return "parallel"

    @property
    def display_name(self) -> str:
        return "Parallel"

    async def is_available(self) -> bool:
        """Return True when ``PARALLEL_API_KEY`` is set to a non-empty value."""
        from agent.web_search_provider import get_provider_env

        return bool(await get_provider_env("PARALLEL_API_KEY"))

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    async def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a Parallel search through the native async SDK.

        Uses the ``beta.search`` endpoint with the configured mode
        (``PARALLEL_SEARCH_MODE`` env var, default "agentic"). Limit is
        capped at 20 server-side.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            mode = await _resolve_search_mode()
            logger.info(
                "Parallel search: '%s' (mode=%s, limit=%d)", query, mode, limit
            )
            async with _parallel_client_lease() as client:
                response = await client.beta.search(
                    search_queries=[query],
                    objective=query,
                    mode=mode,
                    max_results=min(limit, 20),
                )

            web_results = []
            for i, result in enumerate(response.results or []):
                excerpts = result.excerpts or []
                web_results.append(
                    {
                        "url": result.url or "",
                        "title": result.title or "",
                        "description": " ".join(excerpts) if excerpts else "",
                        "position": i + 1,
                    }
                )

            return {"success": True, "data": {"web": web_results}}
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except ImportError as exc:
            return {
                "success": False,
                "error": f"Parallel SDK not installed: {exc}",
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Parallel search error: %s", exc)
            return {"success": False, "error": f"Parallel search failed: {exc}"}

    async def extract(
        self, urls: List[str], **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Extract content from one or more URLs via the async SDK.

        Returns the legacy list-of-results shape that
        :func:`tools.web_tools.web_extract_tool` expects: one entry per
        successful URL plus one entry per failed URL with an ``error``
        field. Errors are not raised — they're returned as per-URL items.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [
                    {"url": u, "error": "Interrupted", "title": ""} for u in urls
                ]

            logger.info("Parallel extract: %d URL(s)", len(urls))
            async with _parallel_client_lease() as client:
                response = await client.beta.extract(
                    urls=urls,
                    full_content=True,
                )

            results: List[Dict[str, Any]] = []
            for result in response.results or []:
                content = result.full_content or ""
                if not content:
                    content = "\n\n".join(result.excerpts or [])
                url = result.url or ""
                title = result.title or ""
                results.append(
                    {
                        "url": url,
                        "title": title,
                        "content": content,
                        "raw_content": content,
                        "metadata": {"sourceURL": url, "title": title},
                    }
                )

            for error in response.errors or []:
                results.append(
                    {
                        "url": error.url or "",
                        "title": "",
                        "content": "",
                        "error": error.content or error.error_type or "extraction failed",
                        "metadata": {"sourceURL": error.url or ""},
                    }
                )

            return results
        except ValueError as exc:
            return [{"url": u, "title": "", "content": "", "error": str(exc)} for u in urls]
        except ImportError as exc:
            return [
                {"url": u, "title": "", "content": "", "error": f"Parallel SDK not installed: {exc}"}
                for u in urls
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Parallel extract error: %s", exc)
            return [
                {"url": u, "title": "", "content": "", "error": f"Parallel extract failed: {exc}"}
                for u in urls
            ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Parallel",
            "badge": "paid",
            "tag": "Objective-tuned search + parallel page extraction.",
            "env_vars": [
                {
                    "key": "PARALLEL_API_KEY",
                    "prompt": "Parallel API key",
                    "url": "https://parallel.ai",
                },
            ],
        }
