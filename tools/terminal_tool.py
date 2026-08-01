"""Local terminal tool used by the synchronous training harness.

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
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from tools.registry import registry, tool_error

_active_environments: dict[str, "LocalEnvironment"] = {}
_env_lock = threading.RLock()
_last_activity: dict[str, float] = {}
_creation_locks: dict[str, threading.Lock] = {}
_creation_locks_lock = threading.Lock()
_task_env_overrides: dict[str, dict[str, Any]] = {}
_session_cwds: dict[str, str] = {}
_CONTAINER_BACKENDS = frozenset()

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


def record_session_cwd(task_id: str | None, cwd: str) -> None:
    if cwd:
        _session_cwds[str(task_id or "default")] = os.path.abspath(os.path.expanduser(cwd))


class LocalEnvironment:
    """Small shell-backed environment implementing the file-op protocol."""

    def __init__(self, cwd: str, timeout: int = 120):
        self.cwd = os.path.abspath(cwd)
        self.timeout = timeout
        self._lock = threading.RLock()

    def execute(
        self,
        command: str,
        cwd: str | None = None,
        timeout: int | float | None = None,
        stdin_data: str | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        workdir = os.path.abspath(os.path.expanduser(cwd or self.cwd))
        if not os.path.isdir(workdir):
            return {"output": f"Working directory does not exist: {workdir}", "returncode": 1}
        limit = float(timeout or self.timeout)
        shell = os.environ.get("SHELL") or "/bin/sh"
        marker = "__HERMES_LOCAL_CWD_7F3A__"
        wrapped = f"cd {shlex.quote(workdir)} && {{ {command}; }}; _rc=$?; printf '\\n{marker}%s\\n' \"$PWD\"; exit $_rc"
        try:
            completed = subprocess.run(
                wrapped,
                shell=True,
                executable=shell,
                cwd=workdir,
                input=stdin_data,
                text=True,
                capture_output=True,
                timeout=limit,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            return {"output": output + f"\nCommand timed out after {limit:g}s", "returncode": 124}
        except OSError as exc:
            return {"output": f"Failed to execute command: {exc}", "returncode": 1}

        output = (completed.stdout or "") + (completed.stderr or "")
        match = re.search(rf"\n{re.escape(marker)}([^\n]*)\n?$", output)
        if match:
            new_cwd = match.group(1).strip()
            output = output[: match.start()].rstrip("\n")
            if new_cwd and os.path.isdir(new_cwd):
                with self._lock:
                    self.cwd = new_cwd
        return {"output": output, "returncode": completed.returncode}


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


def cleanup_vm(task_id: str | None = None) -> None:
    key = _resolve_container_task_id(task_id)
    with _env_lock:
        env = _active_environments.pop(key, None)
        if env is not None:
            record_session_cwd(key, env.cwd)
        _last_activity.pop(key, None)


def cleanup_all_environments() -> None:
    with _env_lock:
        for key, env in list(_active_environments.items()):
            record_session_cwd(key, env.cwd)
        _active_environments.clear()
        _last_activity.clear()


def terminal_tool(
    command: str,
    timeout: int | None = None,
    background: bool = False,
    task_id: str | None = None,
    **_kwargs: Any,
) -> str:
    """Execute one local shell command and return a JSON-compatible result."""
    if not isinstance(command, str) or not command.strip():
        return tool_error("terminal requires a non-empty command")
    env = _get_or_create_environment(task_id)
    if background:
        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                executable=os.environ.get("SHELL") or "/bin/sh",
                cwd=env.cwd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return f"Background process started (pid={proc.pid})"
        except OSError as exc:
            return tool_error(f"Failed to start background command: {exc}")
    result = env.execute(command, timeout=timeout)
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


registry.register(
    name="terminal",
    toolset="terminal",
    schema=TERMINAL_SCHEMA,
    handler=lambda args, **kw: terminal_tool(
        command=args.get("command", ""),
        timeout=args.get("timeout"),
        background=bool(args.get("background", False)),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_terminal_requirements,
    emoji="💻",
    max_result_size_chars=100_000,
)
