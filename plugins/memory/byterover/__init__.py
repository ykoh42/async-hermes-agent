"""ByteRover memory plugin — MemoryProvider interface.

Persistent memory via the ByteRover CLI (``brv``). Organizes knowledge into
a hierarchical context tree with tiered retrieval (fuzzy text → LLM-driven
search). Local-first with optional cloud sync.

Original PR #3499 by hieuntg81, adapted to MemoryProvider ABC.

Requires: ``brv`` CLI installed (npm install -g byterover-cli or
curl -fsSL https://byterover.dev/install.sh | sh).

Config via environment variables (profile-scoped via each profile's .env):
  BRV_API_KEY   — ByteRover API key (for cloud features, optional for local)

Config via config.yaml:
  memory:
    byterover:
      auto_extract: false  # disable automatic brv curate hooks

Working directory: $HERMES_HOME/byterover/ (profile-scoped context tree)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import asyncio
import contextvars
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiofiles.os

from agent.memory_provider import MemoryProvider
from agent.secret_scope import get_secret
from hermes_constants import get_hermes_home
from tools.environments.local import build_subprocess_env, _terminate_process
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Timeouts
_QUERY_TIMEOUT = 10   # brv query — should be fast
_CURATE_TIMEOUT = 120  # brv curate — may involve LLM processing

# Minimum lengths to filter noise
_MIN_QUERY_LEN = 10
_MIN_OUTPUT_LEN = 20


async def _finish_owned_task(task: asyncio.Task[Any]) -> Any:
    """Finish one owned task before propagating repeated cancellation."""
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            output = await asyncio.shield(task)
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
    return output


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return default


async def _load_plugin_config() -> dict[str, Any]:
    """Read ByteRover's profile-scoped memory config.

    New memory-provider setup stores non-secret provider settings under
    ``memory.<provider>``.  Some users also set ``memory.provider_config`` from
    early docs/issues, so accept it as a compatibility fallback.
    """
    try:
        from hermes_cli.config import load_config_readonly

        config = await load_config_readonly()
        memory_config = config.get("memory", {})
        if not isinstance(memory_config, dict):
            return {}

        provider_config = memory_config.get("byterover", {})
        if isinstance(provider_config, dict) and provider_config:
            return dict(provider_config)

        legacy_config = memory_config.get("provider_config", {})
        if isinstance(legacy_config, dict):
            return dict(legacy_config)
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# brv binary resolution (cached per loop and canonical profile)
# ---------------------------------------------------------------------------

# Retained private test seam. Runtime resolution is stored in the scoped cache
# below; a non-None value deliberately overrides it for focused subprocess
# tests that historically monkeypatched this name.
_cached_brv_path: str | None = None
_which = aiofiles.os.wrap(shutil.which)


@dataclass
class _BrvPathState:
    signature: tuple[str, str] | None = None
    resolved: str | None = None
    lock_ref: weakref.ReferenceType[asyncio.Lock] | None = None


_BrvScope = tuple[asyncio.AbstractEventLoop, str]
_brv_path_states: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, _BrvPathState]
] = weakref.WeakKeyDictionary()
_brv_scope_aliases: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, str]
] = weakref.WeakKeyDictionary()
_brv_scope_context: contextvars.ContextVar[tuple[str, str] | None] = (
    contextvars.ContextVar("byterover_profile_scope", default=None)
)
_brv_cache_guard = threading.RLock()


def _lexical_brv_profile() -> str:
    return os.path.normcase(os.fspath(get_hermes_home()))


def _prune_closed_brv_loops() -> None:
    with _brv_cache_guard:
        for loop in set(_brv_path_states) | set(_brv_scope_aliases):
            if loop.is_closed():
                _brv_path_states.pop(loop, None)
                _brv_scope_aliases.pop(loop, None)


async def _activate_brv_scope() -> _BrvScope:
    _prune_closed_brv_loops()
    loop = asyncio.get_running_loop()
    lexical = _lexical_brv_profile()
    active = _brv_scope_context.get()
    if active is not None and active[0] == lexical:
        canonical = active[1]
    else:
        with _brv_cache_guard:
            aliases = _brv_scope_aliases.get(loop)
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
        with _brv_cache_guard:
            _brv_scope_aliases.setdefault(loop, {})[lexical] = canonical
    _brv_scope_context.set((lexical, canonical))
    return loop, canonical


def _brv_state(scope: _BrvScope) -> _BrvPathState:
    loop, profile = scope
    with _brv_cache_guard:
        return _brv_path_states.setdefault(loop, {}).setdefault(
            profile,
            _BrvPathState(),
        )


def _brv_state_lock(state: _BrvPathState) -> asyncio.Lock:
    with _brv_cache_guard:
        lock = state.lock_ref() if state.lock_ref is not None else None
        if lock is None:
            lock = asyncio.Lock()
            state.lock_ref = weakref.ref(lock)
        return lock


async def _resolve_brv_path() -> str | None:
    """Find the brv binary on PATH or well-known install locations."""
    if _cached_brv_path is not None:
        return _cached_brv_path if _cached_brv_path != "" else None
    scope = await _activate_brv_scope()
    state = _brv_state(scope)
    path_value = os.environ.get("PATH", "")
    home_value = os.environ.get("HOME", "").strip()
    if not home_value:
        expanduser = aiofiles.os.wrap(os.path.expanduser)
        home_value = str(await expanduser("~"))
    signature = (path_value, home_value)

    async with _brv_state_lock(state):
        if state.signature == signature and state.resolved is not None:
            return state.resolved or None

        found = await _which("brv", path=path_value)
        if not found:
            home = Path(home_value)
            candidates = [
                home / ".brv-cli" / "bin" / "brv",
                Path("/usr/local/bin/brv"),
                home / ".npm-global" / "bin" / "brv",
            ]
            for candidate in candidates:
                if await aiofiles.os.path.exists(candidate):
                    found = str(candidate)
                    break

        state.signature = signature
        state.resolved = found or ""
        return found


async def _invalidate_brv_path(path: str) -> None:
    global _cached_brv_path
    if _cached_brv_path == path:
        _cached_brv_path = None
        return
    scope = await _activate_brv_scope()
    state = _brv_state(scope)
    async with _brv_state_lock(state):
        if state.resolved == path:
            state.signature = None
            state.resolved = None


async def _build_brv_env(brv_path: str) -> dict[str, str]:
    """Build a profile-scoped child env containing only ByteRover's key."""
    api_key = get_secret("BRV_API_KEY")
    base_env = os.environ.copy()
    base_env.pop("BRV_API_KEY", None)
    extra = (
        {"_HERMES_FORCE_BRV_API_KEY": str(api_key)}
        if api_key is not None
        else None
    )
    env = await build_subprocess_env(
        base=base_env,
        scrub_secrets=True,
        extra=extra,
    )
    brv_bin_dir = str(Path(brv_path).parent)
    env["PATH"] = brv_bin_dir + os.pathsep + env.get("PATH", "")
    return env


async def _terminate_and_drain_brv(
    process: asyncio.subprocess.Process,
    communicate_task: asyncio.Task[tuple[bytes, bytes]],
) -> tuple[bytes, bytes]:
    if hasattr(process, "pid"):
        await _terminate_process(process)
    else:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await process.wait()
    return await communicate_task


async def _run_brv(
    args: list[str],
    timeout: int = _QUERY_TIMEOUT,  # noqa: ASYNC109 - upstream argument name
    cwd: str | None = None,
) -> dict:
    """Run a brv CLI command. Returns {success, output, error}."""
    brv_path = await _resolve_brv_path()
    if not brv_path:
        return {"success": False, "error": "brv CLI not found. Install: npm install -g byterover-cli"}

    cmd = [brv_path] + args
    effective_cwd = cwd or str(_get_brv_cwd())
    await aiofiles.os.makedirs(effective_cwd, exist_ok=True)

    try:
        env = await _build_brv_env(brv_path)
        subprocess_kwargs: dict[str, Any] = {}
        if os.name != "nt":
            subprocess_kwargs["start_new_session"] = True
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
            cwd=effective_cwd,
            env=env,
            **subprocess_kwargs,
        )
        if os.name != "nt" and hasattr(process, "pid"):
            process._hermes_pgid = process.pid
        communicate_task = asyncio.create_task(process.communicate())
        timeout_error: TimeoutError | None = None
        communication_error: Exception | None = None
        try:
            async with asyncio.timeout(timeout):
                stdout_bytes, stderr_bytes = await asyncio.shield(communicate_task)
        except TimeoutError as exc:
            timeout_error = exc
        except asyncio.CancelledError:
            cleanup_task = asyncio.create_task(
                _terminate_and_drain_brv(process, communicate_task),
                name="byterover-cancelled-process-cleanup",
            )
            try:
                await _finish_owned_task(cleanup_task)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("brv cleanup after cancellation failed", exc_info=True)
            raise
        except Exception as exc:
            communication_error = exc

        if communication_error is not None:
            cleanup_task = asyncio.create_task(
                _terminate_and_drain_brv(process, communicate_task),
                name="byterover-failed-process-cleanup",
            )
            await _finish_owned_task(cleanup_task)
            raise communication_error

        if timeout_error is not None:
            cleanup_task = asyncio.create_task(
                _terminate_and_drain_brv(process, communicate_task),
                name="byterover-timed-out-process-cleanup",
            )
            await _finish_owned_task(cleanup_task)
            return {"success": False, "error": f"brv timed out after {timeout}s"}

        stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
        stderr = stderr_bytes.decode("utf-8", errors="replace").strip()

        if process.returncode == 0:
            return {"success": True, "output": stdout}
        return {
            "success": False,
            "error": stderr or stdout or f"brv exited {process.returncode}",
        }

    except FileNotFoundError:
        await _invalidate_brv_path(brv_path)
        return {"success": False, "error": "brv CLI not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _get_brv_cwd() -> Path:
    """Profile-scoped working directory for the brv context tree."""
    return get_hermes_home() / "byterover"


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

QUERY_SCHEMA = {
    "name": "brv_query",
    "description": (
        "Search ByteRover's persistent knowledge tree for relevant context. "
        "Returns memories, project knowledge, architectural decisions, and "
        "patterns from previous sessions. Use for any question where past "
        "context would help."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
        },
        "required": ["query"],
    },
}

CURATE_SCHEMA = {
    "name": "brv_curate",
    "description": (
        "Store important information in ByteRover's persistent knowledge tree. "
        "Use for architectural decisions, bug fixes, user preferences, project "
        "patterns — anything worth remembering across sessions. ByteRover's LLM "
        "automatically categorizes and organizes the memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The information to remember."},
        },
        "required": ["content"],
    },
}

STATUS_SCHEMA = {
    "name": "brv_status",
    "description": "Check ByteRover status — CLI version, context tree stats, cloud sync state.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class ByteRoverMemoryProvider(MemoryProvider):
    """ByteRover persistent memory via the brv CLI."""

    def __init__(self, config: dict[str, Any] | None = None):
        self._config = dict(config) if config is not None else None
        config_snapshot = self._config or {}
        self._auto_extract = _coerce_bool(config_snapshot.get("auto_extract"), True)
        self._cwd = ""
        self._session_id = ""
        self._turn_count = 0
        self._brv_available = False

    @property
    def name(self) -> str:
        return "byterover"

    async def is_available(self) -> bool:
        """Check if brv CLI is installed. No network calls."""
        self._brv_available = await _resolve_brv_path() is not None
        return self._brv_available

    def get_config_schema(self):
        return [
            {
                "key": "api_key",
                "description": "ByteRover API key (optional, for cloud sync)",
                "secret": True,
                "env_var": "BRV_API_KEY",
                "url": "https://app.byterover.dev",
            },
            {
                "key": "auto_extract",
                "description": "Automatically curate completed turns and compression/memory hooks",
                "default": "true",
                "choices": ["true", "false"],
            },
        ]

    async def initialize(self, session_id: str, **kwargs) -> None:
        if self._config is None:
            self._config = await _load_plugin_config()
            self._auto_extract = _coerce_bool(
                self._config.get("auto_extract"), True
            )
        self._cwd = str(_get_brv_cwd())
        self._session_id = session_id
        self._turn_count = 0
        await aiofiles.os.makedirs(self._cwd, exist_ok=True)

    def system_prompt_block(self) -> str:
        if not self._brv_available and not _cached_brv_path:
            return ""
        return (
            "# ByteRover Memory\n"
            "Active. Persistent knowledge tree with hierarchical context.\n"
            "Use brv_query to search past knowledge, brv_curate to store "
            "important facts, brv_status to check state."
        )

    async def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Run brv query before the agent's first LLM call."""
        if not query or len(query.strip()) < _MIN_QUERY_LEN:
            return ""
        result = await _run_brv(
            ["query", "--", query.strip()[:5000]],
            timeout=_QUERY_TIMEOUT, cwd=self._cwd,
        )
        if result["success"] and result.get("output"):
            output = result["output"].strip()
            if len(output) > _MIN_OUTPUT_LEN:
                return f"## ByteRover Context\n{output}"
        return ""

    async def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """No-op: prefetch() runs at turn start."""

    async def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
    ) -> None:
        """Curate a substantive completed conversation turn."""
        self._turn_count += 1
        if not self._auto_extract:
            logger.debug("ByteRover sync_turn skipped (auto_extract disabled)")
            return

        # Only curate substantive turns
        if len(user_content.strip()) < _MIN_QUERY_LEN:
            return

        combined = (
            f"User: {user_content[:2000]}\n"
            f"Assistant: {assistant_content[:2000]}"
        )
        await _run_brv(
            ["curate", "--", combined],
            timeout=_CURATE_TIMEOUT,
            cwd=self._cwd,
        )

    async def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
    ) -> None:
        """Mirror built-in memory writes to ByteRover."""
        if not self._auto_extract:
            logger.debug("ByteRover memory mirror skipped (auto_extract disabled)")
            return
        if action not in {"add", "replace"} or not content:
            return

        label = "User profile" if target == "user" else "Agent memory"
        await _run_brv(
            ["curate", "--", f"[{label}] {content}"],
            timeout=_CURATE_TIMEOUT,
            cwd=self._cwd,
        )

    async def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        """Extract insights before context compression discards turns."""
        if not self._auto_extract:
            logger.debug("ByteRover pre-compression flush skipped (auto_extract disabled)")
            return ""
        if not messages:
            return ""

        # Build a summary of messages about to be compressed
        parts = []
        for msg in messages[-10:]:  # last 10 messages
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip() and role in {"user", "assistant"}:
                parts.append(f"{role}: {content[:500]}")

        if not parts:
            return ""

        combined = "\n".join(parts)

        await _run_brv(
            ["curate", "--", f"[Pre-compression context]\n{combined}"],
            timeout=_CURATE_TIMEOUT,
            cwd=self._cwd,
        )
        logger.info("ByteRover pre-compression flush: %d messages", len(parts))
        return ""

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [QUERY_SCHEMA, CURATE_SCHEMA, STATUS_SCHEMA]

    async def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if tool_name == "brv_query":
            return await self._tool_query(args)
        elif tool_name == "brv_curate":
            return await self._tool_curate(args)
        elif tool_name == "brv_status":
            return await self._tool_status()
        return tool_error(f"Unknown tool: {tool_name}")

    async def shutdown(self) -> None:
        """ByteRover subprocesses are awaited by their owning operations."""

    # -- Tool implementations ------------------------------------------------

    async def _tool_query(self, args: dict) -> str:
        query = args.get("query", "")
        if not query:
            return tool_error("query is required")

        result = await _run_brv(
            ["query", "--", query.strip()[:5000]],
            timeout=_QUERY_TIMEOUT, cwd=self._cwd,
        )

        if not result["success"]:
            return tool_error(result.get("error", "Query failed"))

        output = result.get("output", "").strip()
        if not output or len(output) < _MIN_OUTPUT_LEN:
            return json.dumps({"result": "No relevant memories found."})

        # Truncate very long results
        if len(output) > 8000:
            output = output[:8000] + "\n\n[... truncated]"

        return json.dumps({"result": output})

    async def _tool_curate(self, args: dict) -> str:
        content = args.get("content", "")
        if not content:
            return tool_error("content is required")

        result = await _run_brv(
            ["curate", "--", content],
            timeout=_CURATE_TIMEOUT, cwd=self._cwd,
        )

        if not result["success"]:
            return tool_error(result.get("error", "Curate failed"))

        return json.dumps({"result": "Memory curated successfully."})

    async def _tool_status(self) -> str:
        result = await _run_brv(["status"], timeout=15, cwd=self._cwd)
        if not result["success"]:
            return tool_error(result.get("error", "Status check failed"))
        return json.dumps({"status": result.get("output", "")})


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register ByteRover as a memory provider plugin."""
    ctx.register_memory_provider(ByteRoverMemoryProvider())
