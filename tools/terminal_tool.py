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
import shlex
import asyncio
import contextvars
import inspect
import logging
import signal
import threading
import time
import weakref
from pathlib import Path
from typing import Any, Callable, Optional

import aiofiles
import aiofiles.os

from tools.environments.local import build_subprocess_env
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
# Native ``background=true`` commands are owned by the running event loop,
# not the legacy ProcessRegistry (which is backed by blocking Popen readers).
# Keeping their handles here lets ``AIAgent.close()`` terminate and reap them
# without leaking child processes when a service request/session ends.
_background_processes: dict[str, set[asyncio.subprocess.Process]] = {}
_background_reapers: dict[asyncio.subprocess.Process, asyncio.Task] = {}

_approval_callback: contextvars.ContextVar[Callable[..., Any] | None] = (
    contextvars.ContextVar("terminal_approval_callback", default=None)
)
_sudo_password_callback: Callable[..., Any] | None = None


def _safe_parse_import_env(name: str, default: Any, converter: Callable[[str], Any], type_label: str) -> Any:
    """Parse an optional import-time setting without breaking tool discovery."""
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return converter(raw)
    except (TypeError, ValueError):
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


def _foreground_background_guidance(command: str) -> str | None:
    """Return a recovery recipe for foreground commands that would stall a turn."""
    if _looks_like_help_or_version_command(command):
        return None
    unquoted = _strip_quotes(command)
    if _SHELL_LEVEL_BACKGROUND_RE.search(unquoted):
        return (
            "Foreground command uses shell-level background wrappers. Re-send WITHOUT "
            "the wrapper as terminal(command=\"<cmd>\", background=true), then run "
            "readiness checks in follow-up terminal calls."
        )
    if _INLINE_BACKGROUND_AMP_RE.search(unquoted) or _TRAILING_BACKGROUND_AMP_RE.search(unquoted):
        return (
            "Foreground command uses '&' backgrounding. Re-send WITHOUT the '&' as "
            "terminal(command=\"<cmd>\", background=true), then run readiness checks "
            "in follow-up terminal calls."
        )
    if any(pattern.search(unquoted) for pattern in _LONG_LIVED_FOREGROUND_PATTERNS):
        return (
            "This command appears to start a long-lived server or watcher. Run it with "
            "background=true, verify readiness, then execute tests separately."
        )
    return None


def _parse_env_var(
    name: str,
    default: str,
    converter: Callable[[str], Any] = int,
    type_label: str = "integer",
) -> Any:
    """Parse a terminal environment variable with a useful error message."""
    raw = os.getenv(name)
    if raw in (None, ""):
        return converter(default) if converter is not str else default
    try:
        return converter(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be valid {type_label}") from exc


async def _get_env_config() -> dict[str, Any]:
    """Read local terminal configuration without synchronous filesystem I/O."""
    raw_cwd = os.getenv("TERMINAL_CWD", "").strip()
    expanded_cwd = os.path.expanduser(raw_cwd) if raw_cwd else ""
    if expanded_cwd and os.path.isabs(expanded_cwd):
        cwd = expanded_cwd if await aiofiles.os.path.isdir(expanded_cwd) else os.getcwd()
    else:
        cwd = os.getcwd()
    return {
        "env_type": "local",
        "cwd": os.path.abspath(cwd),
        "timeout": _parse_env_var("TERMINAL_TIMEOUT", "120"),
        "docker_forward_env": _parse_env_var(
            "TERMINAL_DOCKER_FORWARD_ENV", "[]", json.loads, "valid JSON"
        ),
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
        "local_persistent": True,
    }


def _resolve_container_task_id(task_id: str | None = None) -> str:
    """Normalize task ids without collapsing independent local sessions."""
    return str(task_id or "default")


def _is_unusable_container_cwd(_cwd: str | None) -> bool:
    return False


def _docker_has_host_access(_config: dict[str, Any] | None = None) -> bool:
    return False


def _check_vercel_sandbox_requirements(_config: dict[str, Any] | None = None) -> bool:
    return False


def _start_cleanup_thread() -> None:
    """Compatibility hook; local environments live until explicit cleanup."""


def _get_approval_callback() -> Callable[..., Any] | None:
    return _approval_callback.get()


def set_approval_callback(callback: Callable[..., Any] | None) -> None:
    _approval_callback.set(callback)


def _get_sudo_password_callback() -> Callable[..., Any] | None:
    return _sudo_password_callback


def set_sudo_password_callback(callback: Callable[..., Any] | None) -> None:
    global _sudo_password_callback
    _sudo_password_callback = callback


def _read_shell_token(command: str, start: int) -> tuple[str, int]:
    """Read one shell token, preserving quotes and escapes."""
    index = start
    length = len(command)

    while index < length:
        char = command[index]
        if char.isspace() or char in ";|&()":
            break
        if char == "'":
            index += 1
            while index < length and command[index] != "'":
                index += 1
            if index < length:
                index += 1
            continue
        if char == '"':
            index += 1
            while index < length:
                inner = command[index]
                if inner == "\\" and index + 1 < length:
                    index += 2
                    continue
                if inner == '"':
                    index += 1
                    break
                index += 1
            continue
        if char == "\\" and index + 1 < length:
            index += 2
            continue
        index += 1

    return command[start:index], index


def _rewrite_compound_background(command: str, *_args: Any, **_kwargs: Any) -> str:
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


def _transform_sudo_command(command: str, *_args: Any, **_kwargs: Any) -> tuple[str, str | None]:
    # BaseEnvironment expects the historical ``(command, sudo_stdin)`` pair.
    # The lean local backend does not inject a password, so the second item is
    # always empty while the protocol remains compatible with file/process
    # helpers and the local shell snapshot tests.
    return command, None


def resolve_task_overrides(task_id: str | None = None) -> dict[str, Any]:
    return dict(_task_env_overrides.get(str(task_id or "default"), {}))


def register_task_env_overrides(task_id: str, overrides: dict[str, Any]) -> None:
    key = str(task_id or "default")
    _task_env_overrides[key] = dict(overrides or {})
    cwd = _task_env_overrides[key].get("cwd")
    if isinstance(cwd, str) and os.path.isabs(cwd) and os.path.isdir(cwd):
        record_session_cwd(key, cwd)


def clear_task_env_overrides(task_id: str) -> None:
    """Remove one task's local environment overrides and cwd anchor."""
    key = str(task_id or "default")
    _task_env_overrides.pop(key, None)
    clear_session_cwd(key)


def get_session_cwd(task_id: str | None = None) -> str:
    """Return the in-memory cwd anchor for a session.

    This lookup performs no filesystem I/O, so it remains a synchronous state
    accessor even though environment creation and command execution are async.
    """
    key = str(task_id or "default")
    with _env_lock:
        env = _active_environments.get(key)
        if env is not None and env.cwd:
            return env.cwd
        recorded = _session_cwds.get(key)
    if recorded:
        return recorded
    configured = os.getenv("TERMINAL_CWD", "").strip()
    if configured:
        configured = os.path.expanduser(configured)
        if os.path.isabs(configured):
            return os.path.abspath(configured)
    return os.getcwd()


def record_session_cwd(task_id: str | None, cwd: str) -> None:
    if cwd:
        _session_cwds[str(task_id or "default")] = os.path.abspath(os.path.expanduser(cwd))


def clear_session_cwd(task_id: str | None = None) -> None:
    """Forget the durable working-directory anchor for one local session."""
    _session_cwds.pop(str(task_id or "default"), None)


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    """Terminate and reap a foreground shell process.

    Commands run in their own POSIX process group so a shell that launched a
    child (for example ``sleep`` or a compiler) cannot survive a timeout or a
    cancelled turn.  Windows has no portable process-group equivalent here,
    so it uses the asyncio process primitives directly.
    """
    if process.returncode is not None:
        return
    try:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except TimeoutError:
            if process.returncode is None:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
            await process.wait()
    except ProcessLookupError:
        # The child exited between the return-code check and the signal.
        return


class LocalEnvironment:
    """Small shell-backed environment implementing the file-op protocol."""

    def __init__(self, cwd: str, timeout: int = 120):
        self.cwd = os.path.abspath(cwd)
        self.timeout = timeout
        self._lock = asyncio.Lock()

    async def execute(
        self,
        command: str,
        cwd: str | None = None,
        timeout: int | float | None = None,
        stdin_data: str | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """Run a local shell command without blocking the agent event loop."""
        workdir = os.path.abspath(os.path.expanduser(cwd or self.cwd))
        if not await aiofiles.os.path.isdir(workdir):
            recovered = await _nearest_existing_directory(workdir)
            if recovered is None:
                return {
                    "output": f"Working directory does not exist: {workdir}",
                    "returncode": 1,
                }
            logger.warning(
                "Terminal working directory %s is missing on disk; recovered to %s",
                workdir,
                recovered,
            )
            workdir = recovered
            async with self._lock:
                self.cwd = recovered
        limit = float(timeout or self.timeout)
        shell = os.environ.get("SHELL") or "/bin/sh"
        marker = "__HERMES_LOCAL_CWD_7F3A__"
        wrapped = (
            f"cd {shlex.quote(workdir)} && {{ {command}; }}; _rc=$?; "
            f"printf '\\n{marker}%s\\n' \"$PWD\"; exit $_rc"
        )
        try:
            process = await asyncio.create_subprocess_shell(
                wrapped,
                executable=shell,
                cwd=workdir,
                **({"start_new_session": True} if os.name == "posix" else {}),
                stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=build_subprocess_env(
                    scrub_secrets=True,
                ),
            )
            stdout_chunks: list[bytes] = []
            stderr_chunks: list[bytes] = []

            async def _drain(
                stream: asyncio.StreamReader | None,
                chunks: list[bytes],
            ) -> None:
                if stream is None:
                    return
                while chunk := await stream.read(64 * 1024):
                    chunks.append(chunk)

            readers = [
                asyncio.create_task(_drain(process.stdout, stdout_chunks)),
                asyncio.create_task(_drain(process.stderr, stderr_chunks)),
            ]
            if process.stdin is not None:
                try:
                    process.stdin.write(stdin_data.encode())
                    await process.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), timeout=limit)
            except TimeoutError:
                await _terminate_process(process)
                await _finish_stream_readers(readers)
                output = _decode_process_output(stdout_chunks, stderr_chunks)
                timeout_message = f"Command timed out after {limit:g}s"
                return {
                    "output": (
                        f"{output.rstrip()}\n{timeout_message}" if output else timeout_message
                    ),
                    "returncode": 124,
                }
            except asyncio.CancelledError:
                # A cancelled turn must not leave the shell (or its pipe
                # reader) alive after the caller has moved on.  Re-raise only
                # after the child has been reaped so cancellation cannot leak
                # a process or file descriptors into the next turn.
                await _terminate_process(process)
                await _finish_stream_readers(readers)
                raise
            await _finish_stream_readers(readers)
        except OSError as exc:
            return {"output": f"Failed to execute command: {exc}", "returncode": 1}

        output = _decode_process_output(stdout_chunks, stderr_chunks)
        match = re.search(rf"\n{re.escape(marker)}([^\n]*)\n?$", output)
        if match:
            new_cwd = match.group(1).strip()
            output = output[: match.start()].rstrip("\n")
            if new_cwd and await aiofiles.os.path.isdir(new_cwd):
                async with self._lock:
                    self.cwd = new_cwd
        return {"output": output, "returncode": process.returncode}


async def _finish_stream_readers(readers: list[asyncio.Task[None]]) -> None:
    """Collect available output without waiting on pipes inherited by children."""
    _, pending = await asyncio.wait(readers, timeout=0.25)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


def _decode_process_output(
    stdout_chunks: list[bytes],
    stderr_chunks: list[bytes],
) -> str:
    return b"".join(stdout_chunks).decode(errors="replace") + b"".join(
        stderr_chunks
    ).decode(errors="replace")


async def _nearest_existing_directory(path: str) -> str | None:
    """Return the closest existing directory at or above *path*."""
    candidate = Path(os.path.abspath(os.path.expanduser(path)))
    for current in (candidate, *candidate.parents):
        if await aiofiles.os.path.isdir(current):
            return str(current)
    return None


def _create_environment(
    *, cwd: str | None = None, timeout: int | None = None, **_kwargs: Any
) -> LocalEnvironment:
    """Construct the local environment without performing external I/O."""
    return LocalEnvironment(
        cwd or os.getcwd(),
        int(timeout or _parse_env_var("TERMINAL_TIMEOUT", "120")),
    )


def get_active_env(task_id: str | None = None) -> LocalEnvironment | None:
    key = _resolve_container_task_id(task_id)
    with _env_lock:
        return _active_environments.get(key)


def is_persistent_env(_task_id: str | None = None) -> bool:
    return True


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
        cwd = overrides.get("cwd") or get_session_cwd(raw_key)
        env = LocalEnvironment(cwd or config["cwd"], int(config["timeout"]))
        with _env_lock:
            # Another turn may have created the environment while async config
            # was being read. Reuse it rather than replacing its cwd/state.
            existing = _active_environments.get(raw_key)
            if existing is not None:
                env = existing
            else:
                _active_environments[raw_key] = env
                _last_activity[raw_key] = time.time()
        record_session_cwd(raw_key, env.cwd)
        return env


async def cleanup_vm(task_id: str | None = None) -> None:
    """Terminate and reap native background commands for one task.

    Native asyncio subprocesses need an awaited lifecycle so they cannot outlive a
    closed agent or remain as zombies after their parent request is cancelled.
    """
    key = _resolve_container_task_id(task_id)
    processes = list(_background_processes.pop(key, set()))
    reapers = [
        _background_reapers.pop(process, None)
        for process in processes
    ]
    if processes:
        # Use the same process-group aware termination path as foreground
        # commands; terminating only the shell can strand its child command.
        await asyncio.gather(
            *(_terminate_process(process) for process in processes),
            return_exceptions=True,
        )
    for reaper in reapers:
        if reaper is not None and not reaper.done():
            reaper.cancel()
    if reapers:
        await asyncio.gather(
            *(reaper for reaper in reapers if reaper is not None),
            return_exceptions=True,
        )
    with _env_lock:
        env = _active_environments.pop(key, None)
        if env is not None:
            record_session_cwd(key, env.cwd)
        _last_activity.pop(key, None)


def _track_background_process(
    task_id: str | None,
    process: asyncio.subprocess.Process,
) -> None:
    """Register a native child and remove it once its awaited reaper exits."""
    key = _resolve_container_task_id(task_id)
    _background_processes.setdefault(key, set()).add(process)

    async def _reap() -> None:
        try:
            await process.wait()
        finally:
            processes = _background_processes.get(key)
            if processes is not None:
                processes.discard(process)
                if not processes:
                    _background_processes.pop(key, None)
            _background_reapers.pop(process, None)

    _background_reapers[process] = asyncio.create_task(_reap())


def cleanup_all_environments() -> None:
    with _env_lock:
        for key, env in list(_active_environments.items()):
            record_session_cwd(key, env.cwd)
        _active_environments.clear()
        _last_activity.clear()




async def terminal_tool(
    command: str,
    background: bool = False,
    timeout: int | None = None,
    task_id: str | None = None,
    session_id: str | None = None,
    force: bool = False,
    workdir: str | None = None,
    pty: bool = False,
    notify_on_complete: bool = False,
    watch_patterns: list[str] | None = None,
    **_kwargs: Any,
) -> str:
    """Run a local command asynchronously and preserve Hermes' JSON result contract."""
    from tools.tool_output_limits import refresh_tool_output_limits

    await refresh_tool_output_limits()

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
    if pty:
        return json.dumps(
            {
                "output": "",
                "exit_code": -1,
                "error": "PTY execution is not supported by the native async local terminal.",
                "status": "error",
            },
            ensure_ascii=False,
        )
    if notify_on_complete or watch_patterns:
        return json.dumps(
            {
                "output": "",
                "exit_code": -1,
                "error": (
                    "Background notification flags require the removed messaging gateway "
                    "and are not supported by the async library runtime."
                ),
                "status": "error",
            },
            ensure_ascii=False,
        )

    from tools.approval import validate_terminal_command

    guard = await validate_terminal_command(command)
    if not guard.get("approved", False):
        return json.dumps(
            {
                "output": "",
                "exit_code": -1,
                "error": guard.get("message") or "Terminal command blocked by policy.",
                "status": "denied",
            },
            ensure_ascii=False,
        )

    callback = _get_approval_callback()
    if callback is not None and not force:
        if not inspect.iscoroutinefunction(callback):
            raise RuntimeError(
                "Async Hermes requires a coroutine terminal approval callback"
            )
        decision = await callback(
            command=command,
            task_id=task_id,
            session_id=session_id,
            background=background,
        )
        approved = decision is True or str(decision).strip().lower() in {
            "approve",
            "approved",
            "allow",
            "yes",
        }
        if not approved:
            return json.dumps(
                {
                    "output": "",
                    "exit_code": -1,
                    "error": "Terminal command denied by the approval callback.",
                    "status": "denied",
                },
                ensure_ascii=False,
            )

    try:
        env = await _get_or_create_environment(task_id)
        cwd = workdir or env.cwd
        cwd = os.path.abspath(os.path.expanduser(str(cwd)))
        if workdir and not os.path.isabs(os.path.expanduser(workdir)):
            return json.dumps(
                {
                    "output": "",
                    "exit_code": -1,
                    "error": "workdir must be an absolute path.",
                    "status": "error",
                },
                ensure_ascii=False,
            )
        if not await aiofiles.os.path.isdir(cwd):
            if workdir:
                return json.dumps(
                    {
                        "output": "",
                        "exit_code": -1,
                        "error": f"Working directory does not exist: {cwd}",
                        "status": "error",
                    },
                    ensure_ascii=False,
                )
            recovered = await _nearest_existing_directory(cwd)
            if recovered is None:
                return json.dumps(
                    {
                        "output": "",
                        "exit_code": -1,
                        "error": f"Working directory does not exist: {cwd}",
                        "status": "error",
                    },
                    ensure_ascii=False,
                )
            logger.warning(
                "Terminal working directory %s is missing on disk; recovered to %s",
                cwd,
                recovered,
            )
            cwd = recovered
            env.cwd = recovered
            record_session_cwd(task_id, recovered)
        if background:
            process = await asyncio.create_subprocess_shell(
                command,
                executable=os.environ.get("SHELL") or "/bin/sh",
                cwd=cwd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env=build_subprocess_env(
                    scrub_secrets=True,
                ),
                **({"start_new_session": True} if os.name == "posix" else {}),
            )
            _track_background_process(task_id, process)
            return json.dumps(
                {
                    "output": "Background process started",
                    "session_id": str(process.pid),
                    "pid": process.pid,
                    "exit_code": 0,
                    "error": None,
                },
                ensure_ascii=False,
            )

        starting_cwd = env.cwd
        result = await env.execute(command, cwd=cwd, timeout=timeout)
        if workdir:
            # ``workdir`` applies to one command only. LocalEnvironment tracks
            # the shell's final cwd, so restore the durable session cwd after
            # a transient override.
            env.cwd = starting_cwd
        else:
            record_session_cwd(task_id, env.cwd)
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
                if os.path.realpath(env.cwd) != os.path.realpath(starting_cwd):
                    payload["cwd"] = env.cwd
            except (OSError, TypeError, ValueError):
                pass
        payload.update(truncation)
        exit_note = _interpret_exit_code(command, exit_code)
        if exit_note:
            payload["exit_code_meaning"] = exit_note
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
    """Explain conventional non-zero statuses that are not actual failures."""
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
        "test": {1: "Condition evaluated to false (expected, not an error)"},
        "[": {1: "Condition evaluated to false (expected, not an error)"},
    }
    return semantics.get(base_cmd, {}).get(exit_code)


def check_terminal_requirements() -> bool:
    return True


TERMINAL_SCHEMA = {
    "name": "terminal",
    "description": "Run a local shell command in the session working directory.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute."},
            "timeout": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    f"Timeout in seconds. Foreground maximum is {FOREGROUND_MAX_TIMEOUT}; "
                    "use background=true for longer commands."
                ),
            },
            "background": {"type": "boolean", "description": "Start without waiting for completion."},
            "workdir": {
                "type": "string",
                "description": "Absolute working directory for this command.",
            },
            "pty": {"type": "boolean", "description": "Unsupported in the async local runtime."},
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
