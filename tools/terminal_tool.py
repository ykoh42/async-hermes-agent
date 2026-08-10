"""Local terminal tool used by the async training harness.

Hermes' original terminal module also contained Docker, SSH, Modal, Daytona,
and Vercel backends.  Those transports are intentionally outside this fork's
training scope.  This module keeps the stable helper names used by file,
code-execution, delegation, and batch-runner code while providing one
predictable local backend.
"""

from __future__ import annotations

import os
import json
import re
import asyncio
import concurrent.futures.thread as _thread_backend_bootstrap  # noqa: F401
import contextvars
import logging
import threading
import time
import weakref
from typing import Any, Callable, List, Optional

import aiofiles
import aiofiles.os

# Preserve the original local import patch points after priming their modules.
from agent import redact as _redact_bootstrap  # noqa: F401
from agent import verification_evidence as _verification_evidence_bootstrap  # noqa: F401
from hermes_cli import managed_scope as _managed_scope_bootstrap  # noqa: F401
from tools import ansi_strip as _ansi_strip_bootstrap  # noqa: F401
from tools import file_tools as _file_tools_bootstrap  # noqa: F401
from tools import process_registry as _process_registry_bootstrap  # noqa: F401
from tools import tool_output_limits as _tool_output_limits_bootstrap  # noqa: F401
from tools.environments.local import (
    LocalEnvironment,
    _cwd_usable,
    _get_sudo_password_callback,
    _read_shell_token,
    _resolve_safe_cwd,
    _rewrite_real_sudo_invocations,
    _transform_sudo_command,
    build_subprocess_env,
    _set_sudo_password_callback as set_sudo_password_callback,
)
from tools.approval import check_all_command_guards
from tools.registry import registry

logger = logging.getLogger(__name__)

_active_environments: dict[str, "LocalEnvironment"] = {}
_env_lock = threading.RLock()
_last_activity: dict[str, float] = {}
_creation_locks_lock = threading.Lock()
_creation_locks: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, asyncio.Lock]
] = weakref.WeakKeyDictionary()
_task_env_overrides: dict[str, dict[str, Any]] = {}
_session_cwds: dict[str, str] = {}
_CONTAINER_BACKENDS = frozenset()
_approval_callback: contextvars.ContextVar[Callable[..., Any] | None] = (
    contextvars.ContextVar("terminal_approval_callback", default=None)
)
def _safe_parse_import_env(name: str, default: Any, converter: Callable[[str], Any], type_label: str) -> Any:
    """Parse module-level numeric env vars without breaking import.

    Terminal tool is imported by library callers, tests, and tool discovery.
    A malformed env var must not make the whole module unloadable.
    """
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return converter(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid value for %s: %r (expected %s). Falling back to %r.",
            name,
            raw,
            type_label,
            default,
        )
        return default


FOREGROUND_MAX_TIMEOUT = _safe_parse_import_env(
    "TERMINAL_MAX_FOREGROUND_TIMEOUT", 600, int, "integer"
)
DISK_USAGE_WARNING_THRESHOLD_GB = _safe_parse_import_env(
    "TERMINAL_DISK_WARNING_GB", 500.0, float, "number"
)

_SHELL_LEVEL_BACKGROUND_RE = re.compile(
    r"(?:^|[;&|]\s*|&&\s*|\|\|\s*|\$\(\s*)(?:nohup|disown|setsid)\b",
    re.IGNORECASE | re.MULTILINE,
)
_INLINE_BACKGROUND_AMP_RE = re.compile(r"\s&\s")
_TRAILING_BACKGROUND_AMP_RE = re.compile(r"\s&\s*(?:#.*)?$")
_LONG_LIVED_FOREGROUND_PATTERNS = (
    re.compile(
        r"\b(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?(?:dev|start|serve|watch)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bdocker\s+compose\s+up\b", re.IGNORECASE),
    re.compile(r"\bnext\s+dev\b", re.IGNORECASE),
    re.compile(r"\bvite(?:\s|$)", re.IGNORECASE),
    re.compile(r"\bnodemon\b", re.IGNORECASE),
    re.compile(r"\buvicorn\b", re.IGNORECASE),
    re.compile(r"\bgunicorn\b", re.IGNORECASE),
    re.compile(r"\bpython(?:3)?\s+-m\s+http\.server\b", re.IGNORECASE),
)


def _strip_quotes(command: str) -> str:
    result = re.sub(r"'[^']*'", "''", command)
    result = re.sub(r'"(?:[^"\\]|\\.)*"', '""', result)
    return re.sub(r"`[^`]*`", "``", result)


def _looks_like_help_or_version_command(command: str) -> bool:
    normalized = " ".join(command.lower().split())
    return (
        " --help" in normalized
        or normalized.endswith(" -h")
        or " --version" in normalized
        or normalized.endswith(" -v")
    )


def _command_requires_pipe_stdin(command: str) -> bool:
    """Return True when PTY mode would break an EOF-driven command."""
    normalized = " ".join(command.lower().split())
    return normalized.startswith("gh auth login") and "--with-token" in normalized


def _foreground_background_guidance(command: str) -> str | None:
    """Suggest background mode when a foreground command looks long-lived.

    Prevents workflows that start a server/watch process and then stall before
    follow-up checks or test commands run.
    """
    if _looks_like_help_or_version_command(command):
        return None
    unquoted = _strip_quotes(command)
    if _SHELL_LEVEL_BACKGROUND_RE.search(unquoted):
        return (
            "Foreground command uses shell-level background wrappers (nohup/disown/setsid). "
            "Re-send WITHOUT the wrapper as terminal(command=\"<cmd>\", background=true, "
            "notify_on_complete=true) so Hermes tracks the process, then run readiness "
            "checks and tests in separate commands."
        )
    if _INLINE_BACKGROUND_AMP_RE.search(unquoted) or _TRAILING_BACKGROUND_AMP_RE.search(unquoted):
        return (
            "Foreground command uses '&' backgrounding. Re-send WITHOUT the '&' as "
            "terminal(command=\"<cmd>\", background=true) — add notify_on_complete=true "
            "for bounded jobs — then run health checks and tests in follow-up terminal calls."
        )
    if any(pattern.search(unquoted) for pattern in _LONG_LIVED_FOREGROUND_PATTERNS):
        return (
            "This foreground command appears to start a long-lived server/watch process. "
            "Run it with background=true, verify readiness (health endpoint/log signal), "
            "then execute tests in a separate command."
        )
    return None


def _parse_env_var(
    name: str,
    default: str,
    converter: Callable[[str], Any] = int,
    type_label: str = "integer",
) -> Any:
    """Parse an environment variable with a clear error on bad values.

    Without this wrapper, a single malformed env var causes an unhandled
    conversion error that kills every terminal command.
    """
    raw = os.getenv(name, default)
    try:
        return converter(raw)
    except (ValueError, json.JSONDecodeError):
        raise ValueError(
            f"Invalid value for {name}: {raw!r} (expected {type_label}). "
            "Check ~/.hermes/.env or environment variables."
        )


async def _get_env_config() -> dict[str, Any]:
    """Read local terminal configuration without synchronous filesystem I/O."""
    raw_cwd = os.getenv("TERMINAL_CWD", "").strip()
    expanduser = aiofiles.os.wrap(os.path.expanduser)
    cwd = await expanduser(raw_cwd) if raw_cwd else await aiofiles.os.getcwd()
    return {
        "env_type": "local",
        "cwd": os.path.normpath(cwd),  # noqa: ASYNC240 - lexical only
        "timeout": _parse_env_var("TERMINAL_TIMEOUT", "120"),
        "local_persistent": _parse_env_var(
            "TERMINAL_LOCAL_PERSISTENT",
            "false",
            lambda value: value.lower() in {"true", "1", "yes"},
            "boolean",
        ),
    }


def _resolve_container_task_id(task_id: str | None) -> str:
    """Normalize task ids without collapsing independent local sessions."""
    return str(task_id or "default")


def _get_approval_callback() -> Callable[..., Any] | None:
    return _approval_callback.get()


def set_approval_callback(cb):
    _approval_callback.set(cb)


def _rewrite_compound_background(command: str) -> str:
    """Wrap top-level ``A && B &`` as ``A && { B & }``.

    Bash otherwise backgrounds the whole compound in a subshell, which can
    keep pipes open and wait forever for a long-running ``B``. Quoted text,
    redirects, subshells, and existing brace groups are left unchanged.
    """
    length = len(command)
    index = 0
    paren_depth = 0
    brace_depth = 0
    last_chain_end = -1
    rewrites: list[tuple[int, int]] = []

    while index < length:
        char = command[index]

        if char == "\n" and paren_depth == 0 and brace_depth == 0:
            last_chain_end = -1
            index += 1
            continue
        if char.isspace():
            index += 1
            continue
        if char == "#":
            newline = command.find("\n", index)
            if newline == -1:
                break
            index = newline
            continue
        if char == "\\" and index + 1 < length:
            index += 2
            continue
        if char in ("'", '"'):
            _, next_index = _read_shell_token(command, index)
            index = max(next_index, index + 1)
            continue
        if char == "(":
            paren_depth += 1
            index += 1
            continue
        if char == ")":
            paren_depth = max(0, paren_depth - 1)
            index += 1
            continue
        if char == "{" and index + 1 < length and (
            command[index + 1].isspace() or command[index + 1] == "\n"
        ):
            brace_depth += 1
            index += 1
            continue
        if char == "}" and brace_depth > 0:
            brace_depth -= 1
            last_chain_end = -1
            index += 1
            continue
        if paren_depth > 0 or brace_depth > 0:
            index += 1
            continue
        if command.startswith("&&", index) or command.startswith("||", index):
            last_chain_end = index + 2
            index += 2
            continue
        if char == ";":
            last_chain_end = -1
            index += 1
            continue
        if char == "|":
            last_chain_end = -1
            index += 1
            continue
        if char == "&":
            if index + 1 < length and command[index + 1] == ">":
                index += 2
                continue
            previous = index - 1
            while previous >= 0 and command[previous].isspace():
                previous -= 1
            if previous >= 0 and command[previous] in "<>":
                index += 1
                continue
            if last_chain_end >= 0:
                rewrites.append((last_chain_end, index))
            last_chain_end = -1
            index += 1
            continue

        _, next_index = _read_shell_token(command, index)
        index = max(next_index, index + 1)

    result = command
    for chain_end, ampersand in reversed(rewrites):
        insert_at = chain_end
        while insert_at < ampersand and result[insert_at].isspace():
            insert_at += 1
        result = (
            result[:insert_at]
            + "{ "
            + result[insert_at:ampersand]
            + "& }"
            + result[ampersand + 1 :]
        )
    return result


def resolve_task_overrides(task_id: str | None) -> dict[str, Any]:
    return dict(_task_env_overrides.get(str(task_id or "default"), {}))


async def register_task_env_overrides(task_id: str, overrides: dict[str, Any]) -> None:
    key = str(task_id or "default")
    _task_env_overrides[key] = dict(overrides or {})
    cwd = _task_env_overrides[key].get("cwd")
    if (
        isinstance(cwd, str)
        and os.path.isabs(cwd)
        and await aiofiles.os.path.isdir(cwd)
    ):
        await record_session_cwd(key, cwd)


def clear_task_env_overrides(task_id: str) -> None:
    """Remove one task's local environment overrides and cwd anchor."""
    key = str(task_id or "default")
    _task_env_overrides.pop(key, None)
    clear_session_cwd(key)


async def get_session_cwd(  # noqa: ASYNC124 - public await-only state API
    session_key: Optional[str],
) -> Optional[str]:
    """Return the recorded working directory for a session, if any."""
    key = str(session_key or "default")
    with _env_lock:
        return _session_cwds.get(key)


async def record_session_cwd(  # noqa: ASYNC124 - public await-only state API
    session_key: Optional[str], cwd: Optional[str]
) -> None:
    if not isinstance(cwd, str) or not cwd.strip():
        return
    key = str(session_key or "default")
    with _env_lock:
        _session_cwds[key] = cwd


def clear_session_cwd(session_key: str) -> None:
    """Forget the durable working-directory anchor for one local session."""
    with _env_lock:
        _session_cwds.pop(session_key, None)


def _create_environment(
    *, cwd: str | None = None, timeout: int | None = None, **_kwargs: Any
) -> LocalEnvironment:
    """Construct the local environment without performing external I/O."""
    return LocalEnvironment(
        cwd or "",
        int(timeout or _parse_env_var("TERMINAL_TIMEOUT", "120")),
    )


def get_active_env(task_id: str | None) -> LocalEnvironment | None:
    key = _resolve_container_task_id(task_id)
    with _env_lock:
        return _active_environments.get(key)


def is_persistent_env(task_id: str) -> bool:
    env = get_active_env(task_id)
    if env is None:
        return False
    return bool(getattr(env, "_persistent", False))


def _get_creation_lock(task_id: str) -> asyncio.Lock:
    """Get the per-task async environment creation lock."""
    loop = asyncio.get_running_loop()
    with _creation_locks_lock:
        per_loop = _creation_locks.setdefault(loop, {})
        return per_loop.setdefault(task_id, asyncio.Lock())


async def _get_or_create_environment(
    task_id: str | None = None,
) -> LocalEnvironment:
    """Get or create a local environment without blocking filesystem calls."""
    raw_key = _resolve_container_task_id(task_id)
    creation_lock = _get_creation_lock(raw_key)
    async with creation_lock:
        with _env_lock:
            env = _active_environments.get(raw_key)
            if env is not None:
                _last_activity[raw_key] = time.time()
                return env
        overrides = resolve_task_overrides(raw_key)
        config = await _get_env_config()
        cwd = overrides.get("cwd") or await get_session_cwd(raw_key)
        env = LocalEnvironment(cwd or config["cwd"], int(config["timeout"]))
        await env._ensure_initialized()
        env._persistent = bool(config.get("local_persistent", False))
        with _env_lock:
            # Another turn may have created the environment while async config
            # was being read. Reuse it rather than replacing its cwd/state.
            existing = _active_environments.get(raw_key)
            if existing is not None:
                env = existing
            else:
                _active_environments[raw_key] = env
                _last_activity[raw_key] = time.time()
        await record_session_cwd(raw_key, env.cwd)
        return env


async def cleanup_vm(task_id: str, *, force_remove: bool = False) -> None:
    """Terminate and reap native background commands for one task.

    Native asyncio subprocesses need an awaited lifecycle so they cannot outlive a
    closed agent or remain as zombies after their parent request is cancelled.
    """
    del force_remove
    key = _resolve_container_task_id(task_id)
    from tools.process_registry import process_registry

    await process_registry.kill_all(key, source="agent.close", consume_output=False)
    with _env_lock:
        env = _active_environments.pop(key, None)
        _last_activity.pop(key, None)
    with _creation_locks_lock:
        for per_loop in list(_creation_locks.values()):
            per_loop.pop(key, None)
    if env is not None:
        await record_session_cwd(key, env.cwd)
    try:
        from tools.file_tools import clear_file_ops_cache

        clear_file_ops_cache(key)
    except Exception:
        logger.debug("Unable to clear file-operations cache for %s", key, exc_info=True)


async def cleanup_all_environments() -> int:  # noqa: ASYNC124 - lifecycle API
    """Clean up every active local environment and its tracked processes."""
    with _env_lock:
        task_ids = tuple(_active_environments)
    cleaned = 0
    for task_id in task_ids:
        try:
            await cleanup_vm(task_id)
            cleaned += 1
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("Error cleaning %s", task_id, exc_info=True)
    return cleaned




async def terminal_tool(  # noqa: ASYNC109 - upstream public API names timeout
    command: str,
    background: bool = False,
    timeout: Optional[int] = None,  # noqa: ASYNC109 - upstream public API
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    force: bool = False,
    workdir: Optional[str] = None,
    pty: bool = False,
    notify_on_complete: bool = False,
    watch_patterns: Optional[List[str]] = None,
) -> str:
    """Run a local command asynchronously and preserve Hermes' JSON result contract."""
    from tools.tool_output_limits import _refresh_tool_output_limits

    await _refresh_tool_output_limits()

    if not isinstance(command, str):
        return json.dumps(
            {
                "output": "",
                "exit_code": -1,
                "error": f"Invalid command: expected string, got {type(command).__name__}",
                "status": "error",
            },
            ensure_ascii=False,
        )
    if not command.strip():
        return json.dumps(
            {
                "output": "",
                "exit_code": -1,
                "error": "Invalid command: expected a non-empty string",
                "status": "error",
            },
            ensure_ascii=False,
        )

    if not background:
        guidance = _foreground_background_guidance(command)
        if guidance:
            return json.dumps(
                {
                    "output": "",
                    "exit_code": -1,
                    "error": guidance,
                    "status": "error",
                },
                ensure_ascii=False,
            )

    if timeout is not None and not background and timeout > FOREGROUND_MAX_TIMEOUT:
        return json.dumps(
            {
                "output": "",
                "exit_code": -1,
                "error": (
                    f"Foreground timeout {timeout}s exceeds the {FOREGROUND_MAX_TIMEOUT}s "
                    "maximum; use background=true for longer commands."
                ),
                "status": "error",
            },
            ensure_ascii=False,
        )
    if not force:
        guard = await check_all_command_guards(
            command,
            "local",
            approval_callback=_get_approval_callback(),
        )
        if not guard.get("approved", False):
            return json.dumps(
                {
                    "output": "",
                    "exit_code": -1,
                    "error": (
                        guard.get("message") or "Terminal command blocked by policy."
                    ),
                    "status": "denied",
                },
                ensure_ascii=False,
            )

    try:
        env = await _get_or_create_environment(task_id)
        cwd = workdir or env.cwd
        expanduser = aiofiles.os.wrap(os.path.expanduser)
        cwd = os.path.normpath(  # noqa: ASYNC240 - lexical only
            await expanduser(str(cwd))
        )
        if workdir and not os.path.isabs(await expanduser(workdir)):
            return json.dumps(
                {
                    "output": "",
                    "exit_code": -1,
                    "error": "workdir must be an absolute path.",
                    "status": "error",
                },
                ensure_ascii=False,
            )
        if not await _cwd_usable(cwd):
            if workdir:
                return json.dumps(
                    {
                        "output": "",
                        "exit_code": -1,
                        "error": (
                            f"Working directory does not exist or is inaccessible: {cwd}"
                        ),
                        "status": "error",
                    },
                    ensure_ascii=False,
                )
            recovered = await _resolve_safe_cwd(cwd)
            logger.warning(
                "Terminal working directory %s is missing or inaccessible; "
                "recovered to %s",
                cwd,
                recovered,
            )
            cwd = recovered
            env.cwd = recovered
            await record_session_cwd(task_id, recovered)
        if background:
            from tools.process_registry import process_registry

            effective_pty = pty and not _command_requires_pipe_stdin(command)
            process_session = await process_registry.spawn_local(
                command=command,
                cwd=cwd,
                task_id=_resolve_container_task_id(task_id),
                session_key=str(session_id or task_id or ""),
                env_vars=env.env,
                use_pty=effective_pty,
            )
            payload = {
                "output": "Background process started",
                "session_id": process_session.id,
                "pid": process_session.pid,
                "exit_code": 0,
                "error": None,
                "hint": (
                    "This process runs silently. Use process(action='poll') or "
                    "process(action='wait') to observe completion."
                ),
            }
            if pty and not effective_pty:
                payload["pty_note"] = (
                    "PTY disabled for this command because it expects piped stdin/EOF."
                )
            if notify_on_complete and watch_patterns:
                watch_patterns = None
                payload["watch_patterns_ignored"] = (
                    "watch_patterns ignored because notify_on_complete=True; "
                    "the two notification modes are mutually exclusive."
                )
            if notify_on_complete:
                process_session.notify_on_complete = True
                payload["notify_on_complete"] = True
                payload["hint"] = (
                    "Hermes will queue exactly one notification when this process "
                    "finishes. You can continue with other work."
                )
                process_registry.pending_watchers.append(
                    {
                        "session_id": process_session.id,
                        "check_interval": 5,
                        "session_key": process_session.session_key,
                        "notify_on_complete": True,
                    }
                )
            elif watch_patterns:
                process_session.watch_patterns = [
                    str(pattern)
                    for pattern in watch_patterns
                    if isinstance(pattern, str) and pattern
                ]
                payload["watch_patterns"] = process_session.watch_patterns
            if notify_on_complete or process_session.watch_patterns:
                if process_session.exited and notify_on_complete:
                    process_registry._enqueue_completion(process_session)
                await process_registry._write_checkpoint()
            return json.dumps(payload, ensure_ascii=False)

        starting_cwd = env.cwd
        result = await env.execute(command, cwd=cwd, timeout=timeout)
        if workdir:
            # ``workdir`` applies to one command only. LocalEnvironment tracks
            # the shell's final cwd, so restore the durable session cwd after
            # a transient override.
            env.cwd = starting_cwd
        else:
            await record_session_cwd(task_id, env.cwd)
        exit_code = int(result.get("returncode", 0))
        raw_output = str(result.get("output", ""))
        output, truncation = await _prepare_terminal_output(raw_output, command)
        payload: dict[str, Any] = {
            "output": output,
            "exit_code": exit_code,
            "error": None,
        }
        if not workdir and exit_code == 0:
            try:
                realpath = aiofiles.os.wrap(os.path.realpath)
                if await realpath(env.cwd) != await realpath(starting_cwd):
                    payload["cwd"] = env.cwd
            except (OSError, TypeError, ValueError):
                pass
        payload.update(truncation)
        exit_note = _interpret_exit_code(command, exit_code)
        if exit_note:
            payload["exit_code_meaning"] = exit_note
        try:
            from agent.verification_evidence import record_terminal_result

            evidence = await record_terminal_result(
                command=command,
                cwd=cwd,
                session_id=session_id or task_id or "default",
                exit_code=exit_code,
                output=output,
            )
            if evidence:
                payload["verification_evidence"] = {
                    "status": evidence.get("status"),
                    "kind": evidence.get("kind"),
                    "scope": evidence.get("scope"),
                    "canonical_command": evidence.get("canonical_command"),
                }
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("verification evidence recording failed", exc_info=True)
        return json.dumps(payload, ensure_ascii=False)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return json.dumps(
            {
                "output": "",
                "exit_code": -1,
                "error": f"Failed to execute command: {exc}",
                "status": "error",
            },
            ensure_ascii=False,
        )


async def _prepare_terminal_output(
    output: str,
    command: str,
) -> tuple[str, dict[str, Any]]:
    """Canonicalize model-facing output and spill an oversized full result."""
    from agent.redact import redact_terminal_output
    from hermes_constants import get_hermes_home
    from tools.ansi_strip import strip_ansi
    from tools.tool_output_limits import get_max_bytes

    clean_output = strip_ansi(output)
    limit = get_max_bytes()
    metadata: dict[str, Any] = {}
    if len(clean_output) > limit:
        spill_dir = get_hermes_home() / "cache" / "terminal-output"
        try:
            await aiofiles.os.makedirs(spill_dir, exist_ok=True)
            await aiofiles.os.wrap(os.chmod)(spill_dir, 0o700)
            cutoff = time.time() - 7 * 86400
            for name in await aiofiles.os.listdir(spill_dir):
                if not name.startswith("out-") or not name.endswith(".log"):
                    continue
                candidate = spill_dir / name
                try:
                    if (await aiofiles.os.stat(candidate)).st_mtime < cutoff:
                        await aiofiles.os.remove(candidate)
                except OSError:
                    pass

            spill_path = spill_dir / f"out-{time.time_ns()}-{os.getpid()}.log"
            redacted_full = redact_terminal_output(clean_output, command, force=True)
            async with aiofiles.open(spill_path, "w", encoding="utf-8") as handle:
                await handle.write(redacted_full)
                await handle.flush()
                await aiofiles.os.wrap(os.fsync)(handle.fileno())
            await aiofiles.os.wrap(os.chmod)(spill_path, 0o600)
            metadata = {
                "output_total_chars": len(clean_output),
                "full_output_path": str(spill_path),
                "truncation_note": (
                    f"Full output ({len(clean_output):,} chars) saved to {spill_path} — "
                    "search it with search_files or page it with read_file instead of "
                    "re-running the command."
                ),
            }
        except Exception:
            logger.debug("Unable to persist truncated terminal output", exc_info=True)

        head_chars = int(limit * 0.4)
        tail_chars = limit - head_chars
        omitted = len(clean_output) - head_chars - tail_chars
        notice = (
            f"\n\n... [OUTPUT TRUNCATED - {omitted} chars omitted "
            f"out of {len(clean_output)} total] ...\n\n"
        )
        clean_output = clean_output[:head_chars] + notice + clean_output[-tail_chars:]

    return redact_terminal_output(clean_output.strip(), command), metadata


def _interpret_exit_code(command: str, exit_code: int) -> str | None:
    """Return a note when a non-zero exit code is conventional, not erroneous."""
    if exit_code == 0:
        return None
    segments = re.split(r"\s*(?:\|\||&&|[|;])\s*", command)
    words = (segments[-1] if segments else command).strip().split()
    base_cmd = ""
    for word in words:
        if "=" in word and not word.startswith("-"):
            continue
        base_cmd = word.rsplit("/", 1)[-1]
        break
    if not base_cmd:
        return None
    semantics = {
        "grep": {1: "No matches found (not an error)"},
        "egrep": {1: "No matches found (not an error)"},
        "fgrep": {1: "No matches found (not an error)"},
        "rg": {1: "No matches found (not an error)"},
        "ag": {1: "No matches found (not an error)"},
        "ack": {1: "No matches found (not an error)"},
        "diff": {1: "Files differ (expected, not an error)"},
        "colordiff": {1: "Files differ (expected, not an error)"},
        "find": {
            1: "Some directories were inaccessible (partial results may still be valid)"
        },
        "test": {1: "Condition evaluated to false (expected, not an error)"},
        "[": {1: "Condition evaluated to false (expected, not an error)"},
        "curl": {
            6: "Could not resolve host",
            7: "Failed to connect to host",
            22: "HTTP response code indicated error (e.g. 404, 500)",
            28: "Operation timed out",
        },
        "git": {
            1: "Non-zero exit (often normal — e.g. 'git diff' returns 1 when files differ)"
        },
    }
    return semantics.get(base_cmd, {}).get(exit_code)


def check_terminal_requirements() -> bool:
    return True


TERMINAL_TOOL_DESCRIPTION = """Execute shell commands on a Linux environment. Filesystem, current working directory, and exported environment variables persist between calls.

Do NOT use cat/head/tail (use read_file), grep/rg/find/ls (use search_files), sed/awk (use patch), or echo/heredoc file creation (use write_file). Reserve terminal for: builds, installs, git, processes, scripts, network, package managers, and anything that needs a shell.
Environment state persists: activate a virtualenv or export variables once per session, not before every command.

Foreground (default): returns INSTANTLY when the command finishes, even with a high timeout — set timeout generously for long builds.
Background: set background=true (returns a session_id). Pair with notify_on_complete=true for bounded tasks; leave silent only for servers/daemons that never exit. Never use nohup/setsid/trailing '&' — use background=true so Hermes tracks the process. After starting a server, verify readiness with a health check, then act in a separate call; no blind sleep loops. Manage with process(action="poll"/"wait").
Working directory: use 'workdir' for per-command cwd. When a command changes the session cwd (cd, pushd), the result includes a "cwd" field — trust it instead of prefixing every command with 'cd'.
PTY: set pty=true for interactive CLIs (they hang without it). Pipe git output to cat if it might page.
"""


TERMINAL_SCHEMA = {
    "name": "terminal",
    "description": TERMINAL_TOOL_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The command to execute on the VM",
            },
            "background": {
                "type": "boolean",
                "description": (
                    "Run in the background, returning a session_id. Pair with "
                    "notify_on_complete=true for anything with a defined end "
                    "(tests, builds, deploys) — without it the process runs silently. "
                    "Only servers/watchers/daemons that never exit should stay silent. "
                    "Short commands: prefer foreground with a generous timeout."
                ),
                "default": False,
            },
            "timeout": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    f"Max seconds to wait (default: 180, foreground max: "
                    f"{FOREGROUND_MAX_TIMEOUT}). Returns INSTANTLY when command "
                    "finishes — set high for long tasks, you won't wait unnecessarily. "
                    f"Foreground timeout above {FOREGROUND_MAX_TIMEOUT}s is rejected; "
                    "use background=true for longer commands."
                ),
            },
            "workdir": {
                "type": "string",
                "description": (
                    "Working directory for this command (absolute path). Defaults "
                    "to the session working directory."
                ),
            },
            "pty": {
                "type": "boolean",
                "description": (
                    "Run in pseudo-terminal (PTY) mode for interactive CLI tools "
                    "like Codex, Claude Code, or Python REPL. Default: false."
                ),
                "default": False,
            },
            "notify_on_complete": {
                "type": "boolean",
                "description": (
                    "With background=true: get exactly one notification when the "
                    "process exits. The right choice for nearly every bounded long "
                    "task — set it and keep working. MUTUALLY EXCLUSIVE with "
                    "watch_patterns (watch_patterns is dropped when both are set)."
                ),
                "default": False,
            },
            "watch_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Strings to watch for in background output. ONLY for rare "
                    "one-shot mid-process signals on processes that never exit "
                    "(e.g. ['Application startup complete'] on a server). NOT "
                    "for end-of-run markers (use notify_on_complete) and NOT "
                    "for per-iteration patterns like 'ERROR' in loops — "
                    "rate-limited to 1 notification/15s; repeated over-firing "
                    "auto-disables it and falls back to notify-on-exit. When in "
                    "doubt, use notify_on_complete. MUTUALLY EXCLUSIVE with "
                    "notify_on_complete."
                ),
            },
        },
        "required": ["command"],
    },
}


async def _handle_terminal(args: dict, **kwargs) -> str:
    """Adapt the registry's JSON-object contract to ``terminal_tool``."""
    return await terminal_tool(
        command=args.get("command", ""),
        timeout=args.get("timeout"),
        background=bool(args.get("background", False)),
        task_id=kwargs.get("task_id"),
        session_id=kwargs.get("session_id"),
        workdir=args.get("workdir"),
        pty=bool(args.get("pty", False)),
        notify_on_complete=bool(args.get("notify_on_complete", False)),
        watch_patterns=args.get("watch_patterns"),
    )


registry.register(
    name="terminal",
    toolset="terminal",
    schema=TERMINAL_SCHEMA,
    handler=_handle_terminal,
    check_fn=check_terminal_requirements,
    emoji="💻",
    max_result_size_chars=100_000,
)
