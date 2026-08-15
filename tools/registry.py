"""Central registry for all hermes-agent tools.

Each tool file calls ``registry.register()`` at module level to declare its
schema, handler, toolset membership, and availability check.  ``model_tools.py``
queries the registry instead of maintaining its own parallel data structures.

Import chain (circular-import safe):
    tools/registry.py  (no imports from model_tools or tool files)
           ^
    tools/*.py  (import from tools.registry at module level)
           ^
    model_tools.py  (imports tools.registry + all tool modules)
           ^
    run_agent.py, cli.py, batch_runner.py, etc.
"""

import ast
import asyncio
import concurrent.futures
import functools
import inspect
import json
import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from collections.abc import Callable

import aiofiles
import aiofiles.os

from hermes_cli.async_source_loader import (
    _locate_source_module,
    _load_source_module,
    _unload_source_finder,
)

logger = logging.getLogger(__name__)
_realpath = aiofiles.os.wrap(os.path.realpath)
_builtin_import_guard = threading.RLock()
_builtin_import_claim: concurrent.futures.Future[bool] | None = None


def _claim_builtin_import() -> tuple[concurrent.futures.Future[bool], bool]:
    """Claim the process-global module import phase without binding a loop."""
    global _builtin_import_claim
    with _builtin_import_guard:
        claim = _builtin_import_claim
        if claim is None:
            claim = concurrent.futures.Future()
            _builtin_import_claim = claim
            return claim, True
        return claim, False


def _finish_builtin_import_claim(
    claim: concurrent.futures.Future[bool],
    *,
    completed: bool,
) -> None:
    """Publish import completion and release the claim for future refreshes."""
    global _builtin_import_claim
    with _builtin_import_guard:
        if _builtin_import_claim is claim:
            _builtin_import_claim = None
        if not claim.done():
            claim.set_result(completed)


def _active_mcp_registry_scope() -> object | None:
    """Return the loaded MCP loop/profile scope without importing eagerly."""
    mcp_module = sys.modules.get("tools.mcp_tool")
    current_scope = getattr(mcp_module, "_current_mcp_scope_key", None)
    if not callable(current_scope):
        return None
    try:
        return current_scope()
    except RuntimeError:
        return None


def _active_plugin_registry_scope(*, registration: bool = False) -> object | None:
    """Return the current plugin profile scope without importing eagerly."""
    plugin_module = sys.modules.get("hermes_cli.plugins")
    current_scope = getattr(plugin_module, "_current_plugin_registry_scope", None)
    if not callable(current_scope):
        return None
    try:
        return current_scope(registration=registration)
    except RuntimeError:
        return None


def _is_registry_register_call(node: ast.AST) -> bool:
    """Return True when *node* is a ``registry.register(...)`` expression."""
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    function = node.value.func
    return (
        isinstance(function, ast.Attribute)
        and function.attr == "register"
        and isinstance(function.value, ast.Name)
        and function.value.id == "registry"
    )


async def _module_registers_tools(module_path: Path) -> bool:
    """Return whether a module has a top-level registry registration."""
    try:
        async with aiofiles.open(module_path, encoding="utf-8") as handle:
            source = await handle.read()
    except OSError:
        return False
    if "registry" not in source or "register" not in source:
        return False
    try:
        tree = ast.parse(source, filename=str(module_path))
    except SyntaxError:
        return False
    return any(_is_registry_register_call(statement) for statement in tree.body)


async def discover_builtin_tools(
    tools_dir: Path | None = None,
) -> list[str]:
    """Import built-in self-registering modules and return their names.

    Filesystem discovery and its verdict cache use awaited file APIs. Module
    imports happen only at this explicit lazy boundary; importing this module
    itself remains state-only.
    """
    tools_path = Path(tools_dir) if tools_dir is not None else Path(__file__).parent
    cache = await _load_discovery_cache()
    fresh_cache: dict[str, list] = {}
    cache_dirty = False

    try:
        filenames = sorted(await aiofiles.os.listdir(tools_path))
    except OSError:
        filenames = []
    module_names: list[str] = []
    for filename in filenames:
        path = tools_path / filename
        if path.suffix != ".py" or filename in {
            "__init__.py",
            "registry.py",
            "mcp_tool.py",
        }:
            continue
        absolute_path = await _realpath(path)
        try:
            stat_result = await aiofiles.os.stat(path)
        except OSError:
            continue
        stat_key = (stat_result.st_mtime_ns, stat_result.st_size)
        cached = cache.get(absolute_path)
        if (
            isinstance(cached, (list, tuple))
            and len(cached) == 3
            and (cached[0], cached[1]) == stat_key
        ):
            registers = bool(cached[2])
        else:
            registers = await _module_registers_tools(path)
            cache_dirty = True
        fresh_cache[absolute_path] = [stat_key[0], stat_key[1], registers]
        if registers:
            module_names.append(f"tools.{path.stem}")

    if cache_dirty or set(fresh_cache) != set(cache):
        await _save_discovery_cache(fresh_cache)

    claim, owner = _claim_builtin_import()
    if not owner:
        completed = await asyncio.shield(asyncio.wrap_future(claim))
        if not completed:
            # The owner was cancelled before completing its import phase.
            # Retry as a fresh claimant; no thread or executor is involved.
            return await discover_builtin_tools(tools_dir)
        return [name for name in module_names if name in sys.modules]

    imported: list[str] = []
    completed = False
    try:
        package_finder_owner = None
        for module_name in module_names:
            await asyncio.sleep(0)
            existing = sys.modules.get(module_name)
            if existing is not None:
                imported.append(module_name)
                continue
            try:
                located = await _locate_source_module(module_name)
                if located is None:
                    raise ModuleNotFoundError(f"No module named {module_name!r}")
                source_file, is_package = located
                if is_package:
                    raise ImportError(
                        f"Expected source module for built-in tool {module_name!r}"
                    )
                module = await _load_source_module(
                    module_name,
                    source_file,
                    package_dir=(
                        source_file.parent
                        if package_finder_owner is None
                        else None
                    ),
                )
                if package_finder_owner is None:
                    # The first successful load owns a finder containing the
                    # full tools source tree. Keep it for later lazy imports;
                    # one-module finders created afterward can be removed.
                    package_finder_owner = module
                else:
                    _unload_source_finder(module)
                imported.append(module_name)
            except Exception as exc:
                logger.warning(
                    "Could not import tool module %s: %s",
                    module_name,
                    exc,
                )
        completed = True
    finally:
        _finish_builtin_import_claim(claim, completed=completed)
    return imported


def _discovery_cache_path() -> Path | None:
    """Return the discovery verdict cache path, or None if unavailable."""
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home()) / "cache" / "tool_discovery_cache.json"
    except Exception:
        return None


async def _load_discovery_cache() -> dict[str, list]:
    """Read the discovery cache; malformed or unavailable data is a miss."""
    path = _discovery_cache_path()
    if path is None:
        return {}
    try:
        async with aiofiles.open(path, encoding="utf-8") as handle:
            data = json.loads(await handle.read())
    except (OSError, ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


async def _save_discovery_cache(cache: dict[str, list]) -> None:
    """Best-effort atomic persistence for discovery verdicts."""
    path = _discovery_cache_path()
    if path is None:
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        await aiofiles.os.makedirs(path.parent, exist_ok=True)
        async with aiofiles.open(temporary, "w", encoding="utf-8") as handle:
            await handle.write(json.dumps(cache, indent=0))
            await handle.flush()
            await aiofiles.os.wrap(os.fsync)(handle.fileno())
        await aiofiles.os.replace(temporary, path)
    except Exception as exc:
        logger.debug("Could not write tool discovery cache %s: %s", path, exc)
    finally:
        try:
            await aiofiles.os.remove(temporary)
        except OSError:
            pass


class ToolEntry:
    """Metadata for a single registered tool."""

    __slots__ = (
        "name", "toolset", "schema", "handler", "check_fn",
        "requires_env", "is_async", "description", "emoji",
        "max_result_size_chars", "dynamic_schema_overrides",
    )

    def __init__(self, name, toolset, schema, handler, check_fn,
                 requires_env, is_async, description, emoji,
                 max_result_size_chars=None, dynamic_schema_overrides=None):
        self.name = name
        self.toolset = toolset
        self.schema = schema
        self.handler = handler
        self.check_fn = check_fn
        self.requires_env = requires_env
        self.is_async = is_async
        self.description = description
        self.emoji = emoji
        self.max_result_size_chars = max_result_size_chars
        # Optional zero-arg callable returning a dict of schema overrides
        # applied at get_definitions() time. Use for fields that depend on
        # runtime config (e.g. delegate_task's description must reflect the
        # user's current delegation.max_concurrent_children / max_spawn_depth
        # so the model isn't told the wrong limits). The callable is invoked
        # on every get_definitions() call; results are merged shallow on top
        # of the base schema before the {"type": "function", ...} wrap.
        self.dynamic_schema_overrides = dynamic_schema_overrides


# ---------------------------------------------------------------------------
# check_fn TTL cache
#
# check_fn callables like tools/terminal_tool.check_terminal_requirements
# probe external state (Docker daemon, Modal SDK install, playwright binary
# availability). For a long-lived CLI or gateway process, calling them on
# every get_definitions() is pure waste — external state changes on human
# timescales. Cache results for ~30 s so env-var flips via ``hermes tools``
# or live credential file changes propagate within a turn or two without
# requiring any explicit invalidation.
#
# Transient-failure suppression (issue #21658 / #5304): these probes can flap.
# A single ``subprocess.run([docker, "version"], timeout=5)`` that times out
# under load returns False for one call, which would silently strip the entire
# terminal+file toolset from whatever agent is being built at that instant —
# most visibly a delegate_task subagent, which then reports "Tool read_file
# does not exist". To absorb such flakes WITHOUT pinning a permanently-stale
# "available" verdict, we remember the last time each check returned True and,
# when a fresh probe fails within a short grace window of that last success,
# we serve the last-good True instead of caching the failure. A failure that
# persists past the grace window is honored normally, so a backend that really
# went down stops advertising its tools.
# ---------------------------------------------------------------------------

_CHECK_FN_TTL_SECONDS = 30.0
# How long after a successful check a subsequent transient failure is treated
# as a flake (last-good True is served) rather than a real outage. Kept short
# so a genuinely-down backend is reflected within a couple of turns.
_CHECK_FN_FAILURE_GRACE_SECONDS = 60.0
_CHECK_FN_CACHE_MAX = 512
_check_fn_cache: dict[tuple[Callable, str | None], tuple[float, bool]] = {}
# Monotonic timestamp of the most recent True result per check_fn.
_check_fn_last_good: dict[tuple[Callable, str | None], float] = {}
_check_fn_cache_lock = threading.Lock()
CHECK_FN_CACHE_BYPASS = ""


def _prune_check_fn_caches(now: float) -> None:
    """Expire stale entries and cap profile-dimensional cache growth."""
    for key, (timestamp, _) in list(_check_fn_cache.items()):
        if now - timestamp >= _CHECK_FN_TTL_SECONDS:
            _check_fn_cache.pop(key, None)
    for key, timestamp in list(_check_fn_last_good.items()):
        if now - timestamp >= _CHECK_FN_FAILURE_GRACE_SECONDS:
            _check_fn_last_good.pop(key, None)
    while len(_check_fn_cache) >= _CHECK_FN_CACHE_MAX:
        _check_fn_cache.pop(next(iter(_check_fn_cache)))
    while len(_check_fn_last_good) >= _CHECK_FN_CACHE_MAX:
        _check_fn_last_good.pop(next(iter(_check_fn_last_good)))


async def check_fn_cache_scope() -> str | None:
    """Return the active profile key for multiplexed availability checks."""
    try:
        from agent.secret_scope import is_multiplex_active

        if not is_multiplex_active():
            return None
        from hermes_constants import get_hermes_home_override

        override = get_hermes_home_override()
        if not override:
            return CHECK_FN_CACHE_BYPASS
        return await _realpath(os.path.expanduser(str(override)))
    except Exception:
        return CHECK_FN_CACHE_BYPASS


async def _check_fn_cached(fn: Callable) -> bool:
    """Return bool(fn()), TTL-cached across calls."""
    now = time.monotonic()
    scope = await check_fn_cache_scope()
    if scope == CHECK_FN_CACHE_BYPASS:
        try:
            value = fn()
            if inspect.isawaitable(value):
                value = await value
            return bool(value)
        except Exception:
            logger.warning(
                "check_fn %s raised while profile cache scope was unresolved",
                getattr(fn, "__qualname__", fn),
                exc_info=True,
            )
            return False
    cache_key = (fn, scope)
    with _check_fn_cache_lock:
        _prune_check_fn_caches(now)
        cached = _check_fn_cache.get(cache_key)
        if cached is not None:
            ts, value = cached
            if now - ts < _CHECK_FN_TTL_SECONDS:
                return value

    raised = False
    try:
        value = fn()
        if inspect.isawaitable(value):
            value = await value
        value = bool(value)
    except Exception:
        value = False
        raised = True

    with _check_fn_cache_lock:
        _prune_check_fn_caches(now)
        if value:
            _check_fn_last_good[cache_key] = now
            _check_fn_cache[cache_key] = (now, True)
            return True
        last_good = _check_fn_last_good.get(cache_key)
        if last_good is not None and now - last_good < _CHECK_FN_FAILURE_GRACE_SECONDS:
            logger.warning(
                "check_fn %s failed (%s) within %.0fs of last success; "
                "treating as transient and keeping tool(s) available",
                getattr(fn, "__qualname__", fn),
                "raised" if raised else "returned False",
                _CHECK_FN_FAILURE_GRACE_SECONDS,
            )
            return True
        logger.warning(
            "check_fn %s %s; dependent tools will be unavailable this turn",
            getattr(fn, "__qualname__", fn),
            "raised" if raised else "returned False",
        )
        _check_fn_cache[cache_key] = (now, False)
        return False


async def get_cached_check_fn_result(fn: Callable) -> bool | None:
    """Return a fresh cached verdict for *fn* without executing the probe."""
    now = time.monotonic()
    scope = await check_fn_cache_scope()
    if scope == CHECK_FN_CACHE_BYPASS:
        return None
    with _check_fn_cache_lock:
        cached = _check_fn_cache.get((fn, scope))
        if cached is None:
            return None
        timestamp, value = cached
        return value if now - timestamp < _CHECK_FN_TTL_SECONDS else None


def invalidate_check_fn_cache() -> None:
    """Drop all cached ``check_fn`` results. Call after config changes that
    affect tool availability (e.g. ``hermes tools enable``)."""
    with _check_fn_cache_lock:
        _check_fn_cache.clear()
        _check_fn_last_good.clear()


class ToolRegistry:
    """Singleton registry that collects tool schemas + handlers from tool files."""

    def __init__(self):
        self._tools: dict[str, ToolEntry] = {}
        self._mcp_tools: dict[object, dict[str, ToolEntry]] = {}
        self._plugin_tools: dict[object, dict[str, ToolEntry]] = {}
        # Plugin module namespace (handler.__globals__["__name__"]) -> operator
        # opt-in for built-in override. It remains valid for the owning
        # manager's lifetime, so delayed callbacks retain the same authority;
        # manager cleanup retires both this policy and its profile scope.
        self._plugin_override_policy: dict[str, bool] = {}
        self._plugin_module_scopes: dict[str, object | None] = {}
        self._toolset_checks: dict[str, Callable] = {}
        self._plugin_toolset_checks: dict[object, dict[str, Callable]] = {}
        self._toolset_aliases: dict[str, str] = {}
        self._plugin_toolset_aliases: dict[object, dict[str, str]] = {}
        self._mcp_toolset_aliases: dict[object, dict[str, str]] = {}
        # MCP dynamic refresh can mutate the registry while other threads are
        # reading tool metadata, so keep mutations serialized and readers on
        # stable snapshots.
        self._lock = threading.RLock()
        # Monotonically-increasing generation counter. Bumped on every
        # mutation (register / deregister / register_toolset_alias / MCP
        # refresh). External callers (e.g. get_tool_definitions) can memoize
        # against it: a cache entry keyed on the generation is valid for as
        # long as the generation hasn't changed.
        self._generation: int = 0

    def _snapshot_state(self) -> tuple[list[ToolEntry], dict[str, Callable]]:
        """Return a coherent snapshot of registry entries and toolset checks."""
        with self._lock:
            effective = dict(self._tools)
            plugin_scope = _active_plugin_registry_scope()
            if plugin_scope is not None:
                effective.update(self._plugin_tools.get(plugin_scope, {}))
            entries = list(effective.values())
            scope = _active_mcp_registry_scope()
            if scope is not None:
                entries.extend(
                    entry
                    for name, entry in self._mcp_tools.get(scope, {}).items()
                    if name not in effective
                )
            elif not self._tools:
                # Pure synchronous compatibility inspection (not a runtime
                # dispatch boundary) may occur just after ``asyncio.run``.
                # Expose the one populated MCP scope when it is unambiguous.
                populated = [values for values in self._mcp_tools.values() if values]
                if len(populated) == 1:
                    entries.extend(populated[0].values())
            checks = dict(self._toolset_checks)
            if plugin_scope is not None:
                checks.update(self._plugin_toolset_checks.get(plugin_scope, {}))
            return entries, checks

    def _snapshot_entries(self) -> list[ToolEntry]:
        """Return a stable snapshot of registered tool entries."""
        return self._snapshot_state()[0]

    async def _toolset_has_exposable_tools(
        self,
        toolset: str,
        entries: list[ToolEntry],
    ) -> bool:
        """Return True when at least one tool in *toolset* would be exposed.

        Mirrors :meth:`get_tool_definitions` per-tool filtering so doctor,
        banners, and other toolset-level surfaces agree with runtime exposure.
        Mixed toolsets (e.g. ``terminal`` plus desktop-only ``read_terminal``)
        must not be gated solely by the first registered ``check_fn``.
        """
        check_results: dict[Callable, bool] = {}
        for entry in entries:
            if entry.toolset != toolset:
                continue
            if not entry.check_fn:
                return True
            if entry.check_fn not in check_results:
                check_results[entry.check_fn] = await _check_fn_cached(entry.check_fn)
            if check_results[entry.check_fn]:
                return True
        return False

    def get_entry(
        self,
        name: str,
        *,
        scope: str | None = None,
    ) -> ToolEntry | None:
        """Return a registered tool entry by name, or None."""
        with self._lock:
            plugin_scope = (
                scope if scope is not None else _active_plugin_registry_scope()
            )
            if plugin_scope is not None:
                plugin = self._plugin_tools.get(plugin_scope, {}).get(name)
                if plugin is not None:
                    return plugin
            builtin = self._tools.get(name)
            if builtin is not None:
                return builtin
            mcp_scope = (
                scope if scope is not None else _active_mcp_registry_scope()
            )
            if mcp_scope is not None:
                entry = self._mcp_tools.get(mcp_scope, {}).get(name)
                if entry is not None:
                    return entry
            return None

    def get_registered_toolset_names(self) -> list[str]:
        """Return sorted unique toolset names present in the registry."""
        return sorted({entry.toolset for entry in self._snapshot_entries()})

    def get_tool_names_for_toolset(self, toolset: str) -> list[str]:
        """Return sorted tool names registered under a given toolset."""
        return sorted(
            entry.name for entry in self._snapshot_entries()
            if entry.toolset == toolset
        )

    def register_toolset_alias(self, alias: str, toolset: str) -> None:
        """Register an explicit alias for a canonical toolset name."""
        with self._lock:
            mcp_scope = (
                _active_mcp_registry_scope() if toolset.startswith("mcp-") else None
            )
            plugin_scope = (
                _active_plugin_registry_scope(registration=True)
                if mcp_scope is None
                else None
            )
            if plugin_scope is None and mcp_scope is None:
                caller_module = self._caller_module()
                caller_namespace = self._plugin_namespace_for_module(caller_module)
                if caller_module.startswith("hermes_plugins.") and (
                    caller_namespace is None
                ):
                    raise RuntimeError(
                        f"Plugin module {caller_module!r} is no longer attached "
                        "to an active plugin manager"
                    )
                if caller_namespace is not None:
                    plugin_scope = self._plugin_module_scopes.get(caller_namespace)
            if mcp_scope is not None:
                aliases = self._mcp_toolset_aliases.setdefault(mcp_scope, {})
            elif plugin_scope is not None:
                aliases = self._plugin_toolset_aliases.setdefault(plugin_scope, {})
            else:
                aliases = self._toolset_aliases
            existing = aliases.get(alias)
            if existing and existing != toolset:
                logger.warning(
                    "Toolset alias collision: '%s' (%s) overwritten by %s",
                    alias, existing, toolset,
                )
            aliases[alias] = toolset
            self._generation += 1

    def get_registered_toolset_aliases(self) -> dict[str, str]:
        """Return a snapshot of ``{alias: canonical_toolset}`` mappings."""
        with self._lock:
            aliases = dict(self._toolset_aliases)
            plugin_scope = _active_plugin_registry_scope()
            if plugin_scope is not None:
                aliases.update(self._plugin_toolset_aliases.get(plugin_scope, {}))
            scope = _active_mcp_registry_scope()
            if scope is not None:
                aliases.update(self._mcp_toolset_aliases.get(scope, {}))
            return aliases

    def get_toolset_alias_target(self, alias: str) -> str | None:
        """Return the canonical toolset name for an alias, or None."""
        with self._lock:
            plugin_scope = _active_plugin_registry_scope()
            if plugin_scope is not None:
                target = self._plugin_toolset_aliases.get(plugin_scope, {}).get(alias)
                if target is not None:
                    return target
            scope = _active_mcp_registry_scope()
            if scope is not None:
                target = self._mcp_toolset_aliases.get(scope, {}).get(alias)
                if target is not None:
                    return target
            return self._toolset_aliases.get(alias)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_plugin_override_policy(
        self,
        module_namespace: str,
        allowed: bool,
        *,
        scope: str | None = None,
    ) -> None:
        """Bind a plugin module namespace to its operator opt-in for built-in
        override. Called once per plugin at load time. Later delayed
        ``register()`` calls remain gated until the owning manager is cleaned
        up, after which registrations from the retired module fail closed.
        """
        # The active plugin manager already binds the namespace to its profile
        # scope.  Keep that ownership map as the source of truth while
        # accepting upstream's explicit scope argument for compatibility.
        del scope
        with self._lock:
            self._plugin_override_policy[module_namespace] = bool(allowed)

    def _bind_plugin_scope(
        self,
        module_namespace: str,
        scope: object | None,
    ) -> None:
        """Associate one loaded plugin namespace with its profile overlay."""
        with self._lock:
            self._plugin_module_scopes[module_namespace] = scope

    def _unbind_plugin_namespaces(self, module_namespaces: set[str]) -> None:
        """Retire module namespaces whose owning manager was cleaned up."""
        with self._lock:
            for namespace in module_namespaces:
                self._plugin_module_scopes.pop(namespace, None)
                self._plugin_override_policy.pop(namespace, None)

    def _plugin_namespace_for_module(self, module_name: str) -> str | None:
        matches = (
            namespace
            for namespace in self._plugin_override_policy
            if module_name == namespace or module_name.startswith(f"{namespace}.")
        )
        return max(matches, key=len, default=None)

    def _clear_plugin_scope(self, scope: object) -> None:
        """Remove all registry state owned by one plugin profile."""
        with self._lock:
            self._plugin_tools.pop(scope, None)
            self._plugin_toolset_checks.pop(scope, None)
            self._plugin_toolset_aliases.pop(scope, None)
            namespaces = [
                namespace
                for namespace, owner_scope in self._plugin_module_scopes.items()
                if owner_scope is scope
            ]
            for namespace in namespaces:
                self._plugin_module_scopes.pop(namespace, None)
                self._plugin_override_policy.pop(namespace, None)
            self._generation += 1

    def _plugin_owner_of(self, handler: Callable) -> str | None:
        """Return the plugin module namespace that defined *handler*, or None
        if it was not defined in a loaded plugin module.

        Authorization is bound to where the handler was DEFINED
        (``handler.__globals__["__name__"]``), which is fixed at definition
        time and cannot drift with the call site, thread, or timing. Lambdas
        and nested functions inherit the defining module's globals, so a
        plugin cannot launder an override through a callback. Built-in/MCP
        handlers live outside the plugin namespace and return None (unchanged
        behavior).
        """
        target = handler
        while isinstance(target, functools.partial):
            target = target.func
        target = getattr(target, "__func__", target)
        try:
            mod = target.__globals__.get("__name__", "")  # type: ignore[attr-defined]
        except AttributeError:
            return None
        namespace = self._plugin_namespace_for_module(mod)
        if namespace is not None:
            return namespace
        # Also gate plugin modules currently loading but not yet policy-recorded
        # (defensive: a handler defined in the plugin namespace is plugin code).
        if isinstance(mod, str) and mod.startswith("hermes_plugins."):
            return mod
        return None

    @staticmethod
    def _caller_module() -> str:
        """Best-effort module name of whoever called the registry method that
        invoked this helper (two frames up: this helper, then the registry
        method itself, then the actual caller).

        ``deregister()`` takes only a tool name — unlike ``register()`` it has
        no handler argument to bind authorization to via ``_plugin_owner_of``.
        Frame inspection is the only way to know who is asking.
        """
        try:
            frame = sys._getframe(2)
            return frame.f_globals.get("__name__", "") or ""
        except Exception:
            return ""

    def register(
        self,
        name: str,
        toolset: str,
        schema: dict,
        handler: Callable,
        check_fn: Callable | None = None,
        requires_env: list | None = None,
        is_async: bool = False,
        description: str = "",
        emoji: str = "",
        max_result_size_chars: int | float | None = None,
        dynamic_schema_overrides: Callable | None = None,
        override: bool = False,
        scope: str | None = None,
    ):
        """Register a tool.  Called at module-import time by each tool file.

        ``override=True`` is an explicit opt-in for plugins that intend to
        replace an existing built-in tool implementation (e.g. swap the
        default browser tool for a headed-Chrome CDP backend). Without it,
        registrations that would shadow an existing tool from a different
        toolset are rejected to prevent accidental overwrites.
        """
        if not inspect.iscoroutinefunction(handler):
            raise TypeError(
                f"Tool '{name}' must use an async handler in async-hermes-agent"
            )

        with self._lock:
            mcp_scope = (
                (scope if scope is not None else _active_mcp_registry_scope())
                if toolset.startswith("mcp-")
                else None
            )
            owner = self._plugin_owner_of(handler)
            if owner is not None and owner not in self._plugin_override_policy:
                raise RuntimeError(
                    f"Plugin module {owner!r} is no longer attached to an "
                    "active plugin manager"
                )
            plugin_scope = (
                (
                    scope
                    if scope is not None
                    else _active_plugin_registry_scope(registration=True)
                )
                if mcp_scope is None
                else None
            )
            if owner is not None and mcp_scope is None and plugin_scope is None:
                plugin_scope = self._plugin_module_scopes.get(owner)
            if mcp_scope is not None:
                target = self._mcp_tools.setdefault(mcp_scope, {})
            elif plugin_scope is not None:
                target = self._plugin_tools.setdefault(plugin_scope, {})
            else:
                target = self._tools
            existing = target.get(name)
            if existing is None and (mcp_scope is not None or plugin_scope is not None):
                # MCP tools never shadow retained built-ins, even though MCP
                # entries themselves are isolated by profile.
                existing = self._tools.get(name)
            if existing and existing.toolset != toolset:
                if override:
                    _owner = owner
                    if _owner is not None and not self._plugin_override_policy.get(_owner, False):
                        logger.error(
                            "Tool registration REJECTED: plugin %r attempted to "
                            "override built-in tool %r (existing toolset %r) without "
                            "operator opt-in. Set "
                            "plugins.entries.<plugin_id>.allow_tool_override: true "
                            "in config.yaml to allow it.",
                            _owner, name, existing.toolset,
                        )
                        raise PermissionError(
                            f"Plugin module {_owner!r} cannot override built-in "
                            f"tool {name!r} without operator opt-in "
                            f"(allow_tool_override)."
                        )
                    # Explicit opt-in (or non-plugin caller): replace the tool.
                    # Logged at INFO so the override is auditable in agent.log.
                    logger.info(
                        "Tool '%s': toolset '%s' overriding existing toolset '%s' "
                        "(override=True opt-in)",
                        name, toolset, existing.toolset,
                    )
                else:
                    # Reject every cross-toolset shadow, including MCP-to-MCP
                    # collisions. Legitimate MCP reconnect/refresh re-registers
                    # within the same canonical toolset and remains allowed.
                    logger.error(
                        "Tool registration REJECTED: '%s' (toolset '%s') would "
                        "shadow existing tool from toolset '%s'. Pass "
                        "override=True to register() if the replacement is "
                        "intentional, or deregister the existing tool first.",
                        name, toolset, existing.toolset,
                    )
                    return
            target[name] = ToolEntry(
                name=name,
                toolset=toolset,
                schema=schema,
                handler=handler,
                check_fn=check_fn,
                requires_env=requires_env or [],
                # The async-only registry validates the handler itself; retain
                # the upstream field and argument as compatibility metadata.
                is_async=True,
                description=description or schema.get("description", ""),
                emoji=emoji,
                max_result_size_chars=max_result_size_chars,
                dynamic_schema_overrides=dynamic_schema_overrides,
            )
            # Availability is now derived per-tool (_toolset_has_exposable_tools),
            # so this map no longer gates a toolset. It is still consumed by
            # get_toolset_requirements -> TOOLSET_REQUIREMENTS["check_fn"], which
            # banner.py reads (presence only, never called) to classify an
            # already-unavailable toolset as lazy-init vs disabled. Keep the
            # write path for that classification.
            checks = (
                self._plugin_toolset_checks.setdefault(plugin_scope, {})
                if plugin_scope is not None
                else self._toolset_checks
            )
            if check_fn and toolset not in checks:
                checks[toolset] = check_fn
            self._generation += 1

    def deregister(self, name: str) -> None:
        """Remove a tool from the registry.

        Also cleans up the toolset check if no other tools remain in the
        same toolset.  Used by MCP dynamic tool discovery to nuke-and-repave
        when a server sends ``notifications/tools/list_changed``.

        Gated by the same operator opt-in policy ``register(override=True)``
        enforces. Without this, a plugin could bypass that gate entirely by
        deregistering a tool it doesn't own and then calling plain
        ``register()`` over the now-empty slot — ``register()`` only runs its
        override check when an ``existing`` entry is present, so removing it
        first skips the check altogether. MCP toolsets (``mcp-*``) are exempt:
        dynamic tool discovery legitimately nukes-and-repaves its own tools on
        every refresh and has no plugin-override concept.
        """
        with self._lock:
            mcp_scope = _active_mcp_registry_scope()
            mcp_tools = (
                self._mcp_tools.get(mcp_scope, {})
                if mcp_scope is not None
                else {}
            )
            plugin_scope = _active_plugin_registry_scope()
            plugin_tools = (
                self._plugin_tools.get(plugin_scope, {})
                if plugin_scope is not None
                else {}
            )
            entry = plugin_tools.get(name) or mcp_tools.get(name) or self._tools.get(name)
            if entry is None:
                return
            if not entry.toolset.startswith("mcp-"):
                caller_mod = self._caller_module()
                owner = self._plugin_owner_of(entry.handler)
                # Ownership check: bind to the plugin package root
                # (``hermes_plugins.{name}``), not the exact module string.
                # A handler defined in ``hermes_plugins.pkg.handlers`` is
                # still owned by the ``hermes_plugins.pkg`` package — exact
                # string equality would wrongly block root-module cleanup code
                # from removing tools registered by a submodule of the same
                # plugin (egilewski review on #55840).
                caller_root = self._plugin_namespace_for_module(caller_mod)
                same_plugin = bool(owner and caller_root == owner)
                if (
                    caller_mod.startswith("hermes_plugins.")
                    and not same_plugin
                    and not self._plugin_override_policy.get(caller_root or "", False)
                ):
                    logger.error(
                        "Tool deregistration REJECTED: plugin %r attempted to "
                        "remove tool %r (toolset %r) it does not own, without "
                        "operator opt-in. Set "
                        "plugins.entries.%s.allow_tool_override: true in "
                        "config.yaml to allow it.",
                        caller_mod, name, entry.toolset, caller_mod,
                    )
                    raise PermissionError(
                        f"Plugin module {caller_mod!r} cannot deregister tool "
                        f"{name!r} (toolset {entry.toolset!r}) without operator "
                        f"opt-in (allow_tool_override)."
                    )
            if name in plugin_tools:
                target = plugin_tools
                target_kind = "plugin"
            elif name in mcp_tools:
                target = mcp_tools
                target_kind = "mcp"
            else:
                target = self._tools
                target_kind = "shared"
            del target[name]
            # Drop the toolset check and aliases if this was the last tool in
            # that toolset.
            toolset_still_exists = any(
                e.toolset == entry.toolset for e in target.values()
            )
            if not toolset_still_exists:
                if target_kind == "plugin" and plugin_scope is not None:
                    self._plugin_toolset_checks.get(plugin_scope, {}).pop(
                        entry.toolset, None
                    )
                    aliases = self._plugin_toolset_aliases.get(plugin_scope, {})
                    self._plugin_toolset_aliases[plugin_scope] = {
                        alias: alias_target
                        for alias, alias_target in aliases.items()
                        if alias_target != entry.toolset
                    }
                else:
                    self._toolset_checks.pop(entry.toolset, None)
                if target_kind == "mcp" and mcp_scope is not None:
                    aliases = self._mcp_toolset_aliases.get(mcp_scope, {})
                    self._mcp_toolset_aliases[mcp_scope] = {
                        alias: alias_target
                        for alias, alias_target in aliases.items()
                        if alias_target != entry.toolset
                    }
                elif target_kind == "shared":
                    self._toolset_aliases = {
                        alias: alias_target
                        for alias, alias_target in self._toolset_aliases.items()
                        if alias_target != entry.toolset
                    }
            if mcp_scope is not None and not mcp_tools:
                self._mcp_tools.pop(mcp_scope, None)
                if not self._mcp_toolset_aliases.get(mcp_scope):
                    self._mcp_toolset_aliases.pop(mcp_scope, None)
            if plugin_scope is not None and not plugin_tools:
                self._plugin_tools.pop(plugin_scope, None)
                self._plugin_toolset_checks.pop(plugin_scope, None)
                if not self._plugin_toolset_aliases.get(plugin_scope):
                    self._plugin_toolset_aliases.pop(plugin_scope, None)
            self._generation += 1
        logger.debug("Deregistered tool: %s", name)

    # ------------------------------------------------------------------
    # Schema retrieval
    # ------------------------------------------------------------------

    async def get_definitions(
        self,
        tool_names: set[str],
        quiet: bool = False,
    ) -> list[dict]:
        """Return OpenAI-format tool schemas for the requested tool names.

        Only tools whose ``check_fn()`` returns True (or have no check_fn)
        are included. ``check_fn()`` results are cached for ~30 s via
        :func:`_check_fn_cached` to amortize repeat probes (check_terminal_
        requirements probes modal/docker, browser checks probe playwright,
        etc.); TTL chosen so env-var changes (``hermes tools enable foo``)
        still take effect in near-real-time without forcing a full cache
        flush on every call.
        """
        result = []
        # Per-call cache on top of the 30 s TTL — handles repeat probes of the
        # same check_fn within one definitions pass without re-reading the
        # TTL clock.
        check_results: dict[Callable, bool] = {}
        entries_by_name = {entry.name: entry for entry in self._snapshot_entries()}
        for name in sorted(tool_names):
            entry = entries_by_name.get(name)
            if not entry:
                continue
            if entry.check_fn:
                if entry.check_fn not in check_results:
                    check_results[entry.check_fn] = await _check_fn_cached(entry.check_fn)
                if not check_results[entry.check_fn]:
                    if not quiet:
                        logger.debug("Tool %s unavailable (check failed)", name)
                    continue
            # Ensure schema always has a "name" field — use entry.name as fallback
            schema_with_name = {**entry.schema, "name": entry.name}
            # Apply runtime-dynamic overrides (e.g. delegate_task description
            # depends on current delegation.max_concurrent_children /
            # max_spawn_depth). Caller side (model_tools.get_tool_definitions)
            # already keys its memo on config.yaml mtime + size, so changes
            # to delegation.* in config invalidate the cache automatically.
            if entry.dynamic_schema_overrides is not None:
                try:
                    overrides = entry.dynamic_schema_overrides()
                    if inspect.isawaitable(overrides):
                        overrides = await overrides
                    if isinstance(overrides, dict):
                        schema_with_name.update(overrides)
                except Exception as exc:
                    logger.warning(
                        "dynamic_schema_overrides for tool %s raised %s; "
                        "using static schema",
                        name, exc,
                    )
            result.append({"type": "function", "function": schema_with_name})
        return result

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_handler_result(name: str, result):
        """Enforce the result shapes supported by the agent tool pipeline.

        Normal tool results are strings.  The sole structured exception is the
        multimodal envelope consumed by the agent executor.  Returning every
        other value as a string error keeps logging, hooks, budgeting, and
        persistence from receiving values they cannot safely slice or size.
        """
        if isinstance(result, str):
            return result
        if (
            isinstance(result, dict)
            and result.get("_multimodal") is True
            and isinstance(result.get("content"), list)
        ):
            return result

        result_type = type(result).__name__
        logger.error(
            "Tool %s handler returned unsupported result type: %s",
            name,
            result_type,
        )
        return tool_error(
            f"Tool handler returned unsupported result type: {result_type}",
            error_type="tool_result_contract",
            tool=name,
            result_type=result_type,
        )

    async def dispatch(
        self,
        name: str,
        args: dict,
        *,
        scope: str | None = None,
        **kwargs,
    ) -> str | dict:
        """Execute a tool handler by name.

        * Every active handler is awaited directly.
        * Handler results are normalized to a string or supported multimodal
          envelope before leaving the registry.
        * All exceptions are caught and returned as ``{"error": "..."}``
          for consistent error format.
        """
        entry = self.get_entry(name, scope=scope)
        if not entry:
            return tool_error(f"Unknown tool: {name}")
        try:
            result = await entry.handler(args, **kwargs)
            return self._normalize_handler_result(name, result)
        except Exception as e:
            logger.exception("Tool %s dispatch error: %s", name, e)
            # Route through the sanitizer so framing tokens / CDATA / fences
            # in exception strings don't reach the model as structural noise.
            # See model_tools._sanitize_tool_error for rationale.
            raw = f"Tool execution failed: {type(e).__name__}: {e}"
            try:
                from model_tools import _sanitize_tool_error
                sanitized = _sanitize_tool_error(raw)
            except Exception:
                sanitized = raw  # defensive: never let the sanitizer block error propagation
            return tool_error(sanitized)

    # ------------------------------------------------------------------
    # Query helpers  (replace redundant dicts in model_tools.py)
    # ------------------------------------------------------------------

    def get_max_result_size(self, name: str, default: int | float | None = None) -> int | float:
        """Return per-tool max result size, or *default* (or global default)."""
        entry = self.get_entry(name)
        if entry and entry.max_result_size_chars is not None:
            return entry.max_result_size_chars
        if default is not None:
            return default
        from tools.budget_config import DEFAULT_RESULT_SIZE_CHARS
        return DEFAULT_RESULT_SIZE_CHARS

    def get_all_tool_names(self) -> list[str]:
        """Return sorted list of all registered tool names."""
        return sorted(entry.name for entry in self._snapshot_entries())

    def get_schema(self, name: str) -> dict | None:
        """Return a tool's raw schema dict, bypassing check_fn filtering.

        Useful for token estimation and introspection where availability
        doesn't matter — only the schema content does.
        """
        entry = self.get_entry(name)
        return entry.schema if entry else None

    def get_toolset_for_tool(self, name: str) -> str | None:
        """Return the toolset a tool belongs to, or None."""
        entry = self.get_entry(name)
        return entry.toolset if entry else None

    def get_emoji(self, name: str, default: str = "⚡") -> str:
        """Return the emoji for a tool, or *default* if unset."""
        entry = self.get_entry(name)
        return (entry.emoji if entry and entry.emoji else default)

    def get_tool_to_toolset_map(self) -> dict[str, str]:
        """Return ``{tool_name: toolset_name}`` for every registered tool."""
        return {entry.name: entry.toolset for entry in self._snapshot_entries()}

    async def is_toolset_available(self, toolset: str) -> bool:
        """Check if a toolset has at least one exposable tool.

        Returns False (rather than crashing) when a per-tool check raises
        an unexpected exception (e.g. network error, missing import, bad config).
        """
        entries, _ = self._snapshot_state()
        return await self._toolset_has_exposable_tools(toolset, entries)

    async def check_toolset_requirements(self) -> dict[str, bool]:
        """Return ``{toolset: available_bool}`` for every toolset."""
        entries, _ = self._snapshot_state()
        toolsets = sorted({entry.toolset for entry in entries})
        return {
            toolset: await self._toolset_has_exposable_tools(toolset, entries)
            for toolset in toolsets
        }

    async def get_available_toolsets(self) -> dict[str, dict]:
        """Return toolset metadata for UI display."""
        toolsets: dict[str, dict] = {}
        entries, _ = self._snapshot_state()
        for entry in entries:
            ts = entry.toolset
            if ts not in toolsets:
                toolsets[ts] = {
                    "available": await self._toolset_has_exposable_tools(ts, entries),
                    "tools": [],
                    "description": "",
                    "requirements": [],
                }
            toolsets[ts]["tools"].append(entry.name)
            if entry.requires_env:
                for env in entry.requires_env:
                    if env not in toolsets[ts]["requirements"]:
                        toolsets[ts]["requirements"].append(env)
        return toolsets

    def get_toolset_requirements(self) -> dict[str, dict]:
        """Build a TOOLSET_REQUIREMENTS-compatible dict for backward compat."""
        result: dict[str, dict] = {}
        entries, toolset_checks = self._snapshot_state()
        for entry in entries:
            ts = entry.toolset
            if ts not in result:
                result[ts] = {
                    "name": ts,
                    "env_vars": [],
                    "check_fn": toolset_checks.get(ts),
                    "setup_url": None,
                    "tools": [],
                }
            if entry.name not in result[ts]["tools"]:
                result[ts]["tools"].append(entry.name)
            for env in entry.requires_env:
                if env not in result[ts]["env_vars"]:
                    result[ts]["env_vars"].append(env)
        return result

    async def check_tool_availability(self, quiet: bool = False):
        """Return (available_toolsets, unavailable_info) like the old function."""
        available = []
        unavailable = []
        entries, _ = self._snapshot_state()
        for ts in sorted({entry.toolset for entry in entries}):
            ts_entries = [entry for entry in entries if entry.toolset == ts]
            if await self._toolset_has_exposable_tools(ts, entries):
                available.append(ts)
            else:
                unavailable.append({
                    "name": ts,
                    "env_vars": ts_entries[0].requires_env if ts_entries else [],
                    "tools": [entry.name for entry in ts_entries],
                })
        return available, unavailable


# Module-level singleton
registry = ToolRegistry()


# ---------------------------------------------------------------------------
# Helpers for tool response serialization
# ---------------------------------------------------------------------------
# Every tool handler must return a JSON string.  These helpers eliminate the
# boilerplate ``json.dumps({"error": msg}, ensure_ascii=False)`` that appears
# hundreds of times across tool files.
#
# Usage:
#   from tools.registry import registry, tool_error, tool_result
#
#   return tool_error("something went wrong")
#   return tool_error("not found", code=404)
#   return tool_result(success=True, data=payload)
#   return tool_result(items)            # pass a dict directly


def tool_error(message, **extra) -> str:
    """Return a JSON error string for tool handlers.

    >>> tool_error("file not found")
    '{"error": "file not found"}'
    >>> tool_error("bad input", success=False)
    '{"error": "bad input", "success": false}'
    """
    result = {"error": str(message)}
    if extra:
        result.update(extra)
    return json.dumps(result, ensure_ascii=False)


def tool_result(data=None, **kwargs) -> str:
    """Return a JSON result string for tool handlers.

    Accepts a dict positional arg *or* keyword arguments (not both):

    >>> tool_result(success=True, count=42)
    '{"success": true, "count": 42}'
    >>> tool_result({"key": "value"})
    '{"key": "value"}'
    """
    if data is not None:
        return json.dumps(data, ensure_ascii=False)
    return json.dumps(kwargs, ensure_ascii=False)
