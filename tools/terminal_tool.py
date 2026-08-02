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
import signal
import threading
import time
import weakref
from pathlib import Path
from typing import Any, Callable, Optional

import aiofiles.os

from tools.environments.local import build_subprocess_env
from tools.registry import registry, tool_error

_active_environments: dict[str, "LocalEnvironment"] = {}
_env_lock = threading.RLock()
_last_activity: dict[str, float] = {}
_creation_locks: dict[str, threading.Lock] = {}
_creation_locks_lock = threading.Lock()
_async_creation_locks: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, asyncio.Lock]
] = weakref.WeakKeyDictionary()
_task_env_overrides: dict[str, dict[str, Any]] = {}
_session_cwds: dict[str, str] = {}
_CONTAINER_BACKENDS = frozenset()
# Native ``background=true`` commands are owned by the running event loop,
# not the legacy ProcessRegistry (which is backed by blocking Popen readers).
# Keeping their handles here lets ``AIAgent.close()`` terminate and reap them
# without leaking child processes when a service request/session ends.
_async_background_processes: dict[str, set[asyncio.subprocess.Process]] = {}
_async_background_reapers: dict[asyncio.subprocess.Process, asyncio.Task] = {}

_approval_callback: Callable[..., Any] | None = None
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


def _get_env_config() -> dict[str, Any]:
    """Return the local-only terminal configuration."""
    raw_cwd = os.getenv("TERMINAL_CWD", "").strip()
    cwd = raw_cwd if raw_cwd and os.path.isabs(os.path.expanduser(raw_cwd)) else os.getcwd()
    cwd = os.path.abspath(os.path.expanduser(cwd))
    if not os.path.isdir(cwd):
        cwd = os.getcwd()
    return {
        "env_type": "local",
        "cwd": cwd,
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


async def _get_env_config_async() -> dict[str, Any]:
    """Read local terminal configuration without synchronous filesystem I/O.

    The synchronous configuration helper remains for legacy code-execution
    callers.  The registered terminal tool uses this coroutine so a first
    tool call cannot perform ``os.path.isdir`` while the event loop is serving
    another agent.
    """
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
    return _approval_callback


def set_approval_callback(callback: Callable[..., Any] | None) -> None:
    global _approval_callback
    _approval_callback = callback


def _get_sudo_password_callback() -> Callable[..., Any] | None:
    return _sudo_password_callback


def set_sudo_password_callback(callback: Callable[..., Any] | None) -> None:
    global _sudo_password_callback
    _sudo_password_callback = callback


def _rewrite_compound_background(command: str, *_args: Any, **_kwargs: Any) -> str:
    return command


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


def get_session_cwd(task_id: str | None = None) -> str:
    key = str(task_id or "default")
    with _env_lock:
        env = _active_environments.get(key)
        if env is not None and env.cwd:
            return env.cwd
        return _session_cwds.get(key, _get_env_config()["cwd"])


async def get_session_cwd_async(task_id: str | None = None) -> str:
    """Return a session cwd using the async terminal configuration path."""
    key = str(task_id or "default")
    with _env_lock:
        env = _active_environments.get(key)
        if env is not None and env.cwd:
            return env.cwd
        recorded = _session_cwds.get(key)
    if recorded:
        return recorded
    return (await _get_env_config_async())["cwd"]


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
            return {"output": f"Working directory does not exist: {workdir}", "returncode": 1}
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
                    scrub_secrets=False,
                    inherit_profile_home=False,
                ),
            )
            input_bytes = stdin_data.encode() if stdin_data is not None else None
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input_bytes), timeout=limit
                )
            except TimeoutError:
                await _terminate_process(process)
                return {
                    "output": f"Command timed out after {limit:g}s",
                    "returncode": 124,
                }
            except asyncio.CancelledError:
                # A cancelled turn must not leave the shell (or its pipe
                # reader) alive after the caller has moved on.  Re-raise only
                # after the child has been reaped so cancellation cannot leak
                # a process or file descriptors into the next turn.
                await _terminate_process(process)
                raise
        except OSError as exc:
            return {"output": f"Failed to execute command: {exc}", "returncode": 1}

        output = (stdout or b"").decode(errors="replace") + (stderr or b"").decode(errors="replace")
        match = re.search(rf"\n{re.escape(marker)}([^\n]*)\n?$", output)
        if match:
            new_cwd = match.group(1).strip()
            output = output[: match.start()].rstrip("\n")
            if new_cwd and await aiofiles.os.path.isdir(new_cwd):
                async with self._lock:
                    self.cwd = new_cwd
        return {"output": output, "returncode": process.returncode}


def _create_environment(*, cwd: str | None = None, timeout: int | None = None, **_kwargs: Any) -> LocalEnvironment:
    config = _get_env_config()
    return LocalEnvironment(cwd or config["cwd"], int(timeout or config["timeout"]))


def get_active_env(task_id: str | None = None) -> LocalEnvironment | None:
    key = _resolve_container_task_id(task_id)
    with _env_lock:
        return _active_environments.get(key)


def is_persistent_env(_task_id: str | None = None) -> bool:
    return True


def _get_or_create_environment(task_id: str | None = None) -> LocalEnvironment:
    raw_key = _resolve_container_task_id(task_id)
    with _env_lock:
        env = _active_environments.get(raw_key)
        if env is not None:
            _last_activity[raw_key] = time.time()
            return env
    overrides = resolve_task_overrides(raw_key)
    cwd = overrides.get("cwd") or get_session_cwd(raw_key)
    env = _create_environment(cwd=cwd, timeout=_get_env_config()["timeout"], task_id=raw_key)
    with _env_lock:
        _active_environments[raw_key] = env
        _last_activity[raw_key] = time.time()
    record_session_cwd(raw_key, env.cwd)
    return env


def _get_async_creation_lock(task_id: str) -> asyncio.Lock:
    """Get the per-task async environment creation lock."""
    loop = asyncio.get_running_loop()
    with _creation_locks_lock:
        per_loop = _async_creation_locks.setdefault(loop, {})
        return per_loop.setdefault(task_id, asyncio.Lock())


async def _get_or_create_environment_async(
    task_id: str | None = None,
) -> LocalEnvironment:
    """Get or create a local environment without blocking filesystem calls."""
    raw_key = _resolve_container_task_id(task_id)
    creation_lock = _get_async_creation_lock(raw_key)
    async with creation_lock:
        with _env_lock:
            env = _active_environments.get(raw_key)
            if env is not None:
                _last_activity[raw_key] = time.time()
                return env
        overrides = resolve_task_overrides(raw_key)
        config = await _get_env_config_async()
        cwd = overrides.get("cwd") or await get_session_cwd_async(raw_key)
        env = LocalEnvironment(cwd or config["cwd"], int(config["timeout"]))
        with _env_lock:
            # A legacy synchronous caller may have created the environment
            # while the async config was being read. Reuse that object rather
            # than replacing its cwd/state.
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
    processes = list(_async_background_processes.pop(key, set()))
    reapers = [
        _async_background_reapers.pop(process, None)
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


def _track_async_background_process(
    task_id: str | None,
    process: asyncio.subprocess.Process,
) -> None:
    """Register a native child and remove it once its awaited reaper exits."""
    key = _resolve_container_task_id(task_id)
    _async_background_processes.setdefault(key, set()).add(process)

    async def _reap() -> None:
        try:
            await process.wait()
        finally:
            processes = _async_background_processes.get(key)
            if processes is not None:
                processes.discard(process)
                if not processes:
                    _async_background_processes.pop(key, None)
            _async_background_reapers.pop(process, None)

    _async_background_reapers[process] = asyncio.create_task(_reap())


def cleanup_all_environments() -> None:
    with _env_lock:
        for key, env in list(_active_environments.items()):
            record_session_cwd(key, env.cwd)
        _active_environments.clear()
        _last_activity.clear()




async def terminal_tool(
    command: str,
    timeout: int | None = None,
    background: bool = False,
    task_id: str | None = None,
    **_kwargs: Any,
) -> str:
    """Native-async terminal handler used by the async conversation loop."""
    if not isinstance(command, str) or not command.strip():
        return tool_error("terminal requires a non-empty command")
    env = await _get_or_create_environment_async(task_id)
    if background:
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                executable=os.environ.get("SHELL") or "/bin/sh",
                cwd=env.cwd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
            _track_async_background_process(task_id, process)
            return f"Background process started (pid={process.pid})"
        except OSError as exc:
            return tool_error(f"Failed to start background command: {exc}")
    result = await env.execute(command, timeout=timeout)
    return str(result.get("output", "")) if result.get("returncode", 0) == 0 else tool_error(
        f"Command exited with code {result.get('returncode')}:\n{result.get('output', '')}"
    )


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
            "timeout": {"type": "integer", "minimum": 1, "description": "Timeout in seconds."},
            "background": {"type": "boolean", "description": "Start without waiting for completion."},
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
