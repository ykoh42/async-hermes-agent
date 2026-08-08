"""Pure environment helpers for the native asyncio subprocess backend."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
import ntpath
import os
import platform
import re
import shlex
import signal
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Awaitable, Callable

import aiofiles
import aiofiles.os


logger = logging.getLogger(__name__)


_IS_WINDOWS = platform.system() == "Windows"
_HERMES_PROVIDER_ENV_FORCE_PREFIX = "_HERMES_FORCE_"
_ACTIVE_VENV_MARKER_VARS = ("VIRTUAL_ENV", "CONDA_PREFIX")


def _msys_to_windows_path(path: str) -> str:
    """Translate Git Bash, Cygwin, or WSL drive paths on Windows."""
    if not _IS_WINDOWS or not path:
        return path
    match = re.match(r"^/(?:(?:cygdrive|mnt)/)?([a-zA-Z])(/.*)?$", path)
    if not match:
        return path
    drive = match.group(1).upper()
    tail = (match.group(2) or "").replace("/", "\\")
    return f"{drive}:{tail or chr(92)}"


async def _resolve_local_initial_cwd(cwd: str) -> str:
    """Resolve the local backend's initial cwd to an absolute host path."""
    if cwd:
        expanduser = aiofiles.os.wrap(os.path.expanduser)
        expanded = await expanduser(cwd)
    else:
        expanded = await aiofiles.os.getcwd()
    if _IS_WINDOWS:
        expanded = _msys_to_windows_path(expanded)
        if ntpath.isabs(expanded):
            return expanded
    if os.path.isabs(expanded):
        return expanded

    candidate = await aiofiles.os.path.abspath(expanded)
    current = await aiofiles.os.getcwd()

    # Recover config values such as ``hermes-agent`` when Hermes was launched
    # from that directory already. The lexical absolute path would otherwise
    # point at a missing nested ``./hermes-agent``.
    if not await aiofiles.os.path.isdir(candidate):
        wanted_parts = Path(expanded).parts
        current_parts = Path(current).parts
        if wanted_parts and len(wanted_parts) <= len(current_parts):
            if current_parts[-len(wanted_parts) :] == wanted_parts:
                return current

    return candidate


def _build_provider_env_blocklist() -> frozenset[str]:
    blocked = {
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_ORG_ID",
        "OPENAI_ORGANIZATION",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "LLM_MODEL",
        "GOOGLE_API_KEY",
        "VERTEX_CREDENTIALS_PATH",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "AWS_BEARER_TOKEN_BEDROCK",
        "SUDO_PASSWORD",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "HERMES_DASHBOARD_SESSION_TOKEN",
    }
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY

        for provider in PROVIDER_REGISTRY.values():
            blocked.update(provider.api_key_env_vars)
            if provider.base_url_env_var:
                blocked.add(provider.base_url_env_var)
    except ImportError:
        pass
    blocked.discard("CLAUDE_CODE_OAUTH_TOKEN")
    return frozenset(blocked)


_HERMES_PROVIDER_ENV_BLOCKLIST = _build_provider_env_blocklist()

_ALWAYS_STRIP_KEYS = frozenset(
    {
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "GITHUB_APP_ID",
        "GITHUB_APP_PRIVATE_KEY_PATH",
        "GITHUB_APP_INSTALLATION_ID",
        "TELEGRAM_BOT_TOKEN",
        "DISCORD_BOT_TOKEN",
        "SLACK_BOT_TOKEN",
        "SLACK_APP_TOKEN",
        "SLACK_SIGNING_SECRET",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_RELAY_ID",
        "GATEWAY_RELAY_SECRET",
        "GATEWAY_RELAY_DELIVERY_KEY",
        "HASS_TOKEN",
        "EMAIL_PASSWORD",
        "HERMES_DASHBOARD_SESSION_TOKEN",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "DAYTONA_API_KEY",
    }
)


def _is_hermes_internal_secret(key: str) -> bool:
    upper = key.upper()
    return (
        upper.startswith("AUXILIARY_")
        and upper.endswith(("_API_KEY", "_BASE_URL"))
    ) or (
        upper.startswith("GATEWAY_RELAY_")
        and upper.endswith(("_SECRET", "_KEY", "_TOKEN"))
    )


def _inject_session_context_env(env: dict) -> None:
    """Bridge task-local session identity without leaking sibling sessions."""
    try:
        from gateway.session_context import (
            _UNSET,
            _VAR_MAP,
            session_context_engaged,
        )
    except Exception:
        return

    engaged = session_context_engaged()
    for variable_name, variable in _VAR_MAP.items():
        value = variable.get()
        if value is not _UNSET:
            env[variable_name] = "" if value is None else str(value)
        elif engaged:
            env.pop(variable_name, None)


async def _sanitize_subprocess_env(
    base_env: Mapping[str, str] | None,
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Strip Hermes inference credentials from a model-driven subprocess."""
    try:
        from tools.env_passthrough import (
            get_all_passthrough,
            resolve_passthrough_value,
        )
    except Exception:
        passthrough_names: frozenset[str] = frozenset()

        def resolve_passthrough_value(
            _name: str,
            fallback: str | None = None,
        ) -> str | None:
            return fallback

    else:
        passthrough_names = await get_all_passthrough()

    sanitized: dict[str, str] = {}
    for key, value in dict(base_env or {}).items():
        if key.startswith(_HERMES_PROVIDER_ENV_FORCE_PREFIX):
            continue
        if _is_hermes_internal_secret(key):
            continue
        passthrough = key in passthrough_names
        if key in _HERMES_PROVIDER_ENV_BLOCKLIST and not passthrough:
            continue
        resolved = resolve_passthrough_value(key, value) if passthrough else value
        if resolved is not None:
            sanitized[key] = resolved
    for key, value in dict(extra_env or {}).items():
        if key.startswith(_HERMES_PROVIDER_ENV_FORCE_PREFIX):
            real_key = key[len(_HERMES_PROVIDER_ENV_FORCE_PREFIX) :]
            if not _is_hermes_internal_secret(real_key):
                sanitized[real_key] = value
            continue
        if _is_hermes_internal_secret(key):
            continue
        passthrough = key in passthrough_names
        if key in _HERMES_PROVIDER_ENV_BLOCKLIST and not passthrough:
            continue
        resolved = resolve_passthrough_value(key, value) if passthrough else value
        if resolved is not None:
            sanitized[key] = resolved

    try:
        from hermes_constants import (
            apply_subprocess_home_env,
            get_hermes_home_override,
        )

        override = get_hermes_home_override()
        if override:
            sanitized["HERMES_HOME"] = override
        await apply_subprocess_home_env(sanitized)
    except Exception:
        pass
    _inject_session_context_env(sanitized)
    for marker in _ACTIVE_VENV_MARKER_VARS:
        sanitized.pop(marker, None)
    if _IS_WINDOWS:
        sanitized.setdefault("PYTHONUTF8", "1")
    return sanitized


async def build_subprocess_env(
    base: Mapping[str, str] | None = None,
    *,
    inherit_profile_home: bool = True,
    scrub_secrets: bool = True,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the environment for ``asyncio.create_subprocess_exec``."""
    source = dict(base) if base is not None else os.environ.copy()
    if scrub_secrets:
        return await _sanitize_subprocess_env(source, extra)
    if inherit_profile_home:
        try:
            from hermes_constants import (
                apply_subprocess_home_env,
                get_hermes_home_override,
            )

            override = get_hermes_home_override()
            if override:
                source["HERMES_HOME"] = override
            await apply_subprocess_home_env(source)
        except Exception:
            pass
    if extra:
        source.update(extra)
    return source


async def hermes_subprocess_env(*, inherit_credentials: bool = False) -> dict[str, str]:
    """Build a sanitized environment for a non-terminal child process.

    Tier-1 gateway, GitHub, and infrastructure credentials are always removed.
    Provider credentials are retained only for explicitly model-driving CLIs.
    """
    env = os.environ.copy()
    for key in _ALWAYS_STRIP_KEYS:
        env.pop(key, None)
    for key in list(env):
        if key.startswith(_HERMES_PROVIDER_ENV_FORCE_PREFIX) or _is_hermes_internal_secret(
            key
        ):
            env.pop(key, None)
    if not inherit_credentials:
        for key in _HERMES_PROVIDER_ENV_BLOCKLIST:
            env.pop(key, None)
    env.setdefault("PYTHONUTF8", "1")
    try:
        from hermes_constants import (
            apply_subprocess_home_env,
            get_hermes_home_override,
        )

        override = get_hermes_home_override()
        if override:
            env["HERMES_HOME"] = override
        await apply_subprocess_home_env(env)
    except Exception:
        pass
    _inject_session_context_env(env)
    for marker in _ACTIVE_VENV_MARKER_VARS:
        env.pop(marker, None)
    return env


_sudo_password_callback: contextvars.ContextVar[
    Callable[[], Awaitable[str | None]] | None
] = contextvars.ContextVar("terminal_sudo_password_callback", default=None)


def _get_sudo_password_callback() -> Callable[[], Awaitable[str | None]] | None:
    return _sudo_password_callback.get()


def _set_sudo_password_callback(cb) -> None:
    _sudo_password_callback.set(cb)


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


def _looks_like_env_assignment(token: str) -> bool:
    if "=" not in token or token.startswith("="):
        return False
    name, _value = token.split("=", 1)
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name))


def _rewrite_real_sudo_invocations(command: str) -> tuple[str, int]:
    """Rewrite unquoted sudo command words and return their count."""
    output: list[str] = []
    index = 0
    length = len(command)
    command_start = True
    sudo_count = 0

    while index < length:
        char = command[index]
        if char.isspace():
            output.append(char)
            if char == "\n":
                command_start = True
            index += 1
            continue
        if char == "#" and command_start:
            newline = command.find("\n", index)
            if newline == -1:
                output.append(command[index:])
                break
            output.append(command[index:newline])
            index = newline
            continue
        if command.startswith(("&&", "||", ";;"), index):
            output.append(command[index : index + 2])
            index += 2
            command_start = True
            continue
        if char in ";|&(":
            output.append(char)
            index += 1
            command_start = True
            continue
        if char == ")":
            output.append(char)
            index += 1
            command_start = False
            continue

        token, next_index = _read_shell_token(command, index)
        if command_start and token == "sudo":
            output.append("sudo -S -p ''")
            sudo_count += 1
        else:
            output.append(token)
        command_start = command_start and _looks_like_env_assignment(token)
        index = next_index

    return "".join(output), sudo_count


async def _transform_sudo_command(
    command: str | None, *_args: Any, **_kwargs: Any
) -> tuple[str | None, str | None]:
    """Prepare real sudo invocations for configured or async-supplied input."""
    if command is None:
        return None, None
    transformed, sudo_count = _rewrite_real_sudo_invocations(command)
    if sudo_count == 0:
        return command, None

    try:
        from agent.secret_scope import get_secret
    except ImportError:
        configured_password = os.environ.get("SUDO_PASSWORD")
    else:
        configured_password = get_secret("SUDO_PASSWORD")

    if configured_password is not None:
        return transformed, (f"{configured_password}\n" * sudo_count)

    password = None
    if (callback := _get_sudo_password_callback()) is not None:
        if not inspect.iscoroutinefunction(callback):
            raise RuntimeError(
                "Async Hermes requires a coroutine sudo password callback"
            )
        password = await callback()

    if password:
        return transformed, (f"{password}\n" * sudo_count)
    return command, None


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    """Terminate and reap a foreground shell process and its process group."""
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
        return


class LocalEnvironment:
    """Small shell-backed environment implementing the file-op protocol."""

    def __init__(self, cwd: str = "", timeout: int = 60, env: dict = None):
        self.cwd = cwd
        self.timeout = timeout
        initial_env = os.environ.copy()
        if env:
            initial_env.update(env)
        self.env = initial_env
        self._persistent = False
        self._initialized = False
        self._lock = asyncio.Lock()

    async def get_temp_dir(self) -> str:
        """Return a shell-safe writable temp dir for local execution."""
        if _IS_WINDOWS:
            try:
                from hermes_constants import get_hermes_home

                cache_dir = get_hermes_home() / "cache" / "terminal"
            except Exception:
                gettempdir = aiofiles.os.wrap(tempfile.gettempdir)
                cache_dir = Path(await gettempdir()) / "hermes_terminal"
            await aiofiles.os.makedirs(cache_dir, exist_ok=True)
            return str(cache_dir).replace("\\", "/")

        for env_var in ("TMPDIR", "TMP", "TEMP"):
            candidate = self.env.get(env_var) or os.environ.get(env_var)
            if candidate and candidate.startswith("/"):
                return candidate.rstrip("/") or "/"

        if await aiofiles.os.path.isdir("/tmp") and await aiofiles.os.access(
            "/tmp", os.W_OK | os.X_OK
        ):
            return "/tmp"

        gettempdir = aiofiles.os.wrap(tempfile.gettempdir)
        candidate = await gettempdir()
        if candidate.startswith("/"):
            return candidate.rstrip("/") or "/"
        return "/tmp"

    async def _ensure_initialized(self) -> None:
        """Resolve filesystem-backed state lazily on first async use."""
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return
            cwd = os.path.normpath(  # noqa: ASYNC240 - lexical only
                await _resolve_local_initial_cwd(self.cwd)
            )
            env = await build_subprocess_env(base=self.env, scrub_secrets=True)
            self.cwd = cwd
            self.env = env
            self._initialized = True

    async def execute(  # noqa: ASYNC109 - upstream public API names timeout
        self,
        command: str,
        cwd: str | None = None,
        timeout: int | float | None = None,  # noqa: ASYNC109 - public API
        stdin_data: str | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """Run a local shell command without blocking the agent event loop."""
        await self._ensure_initialized()
        prepared_command, sudo_stdin = await _transform_sudo_command(command)
        if prepared_command is None:
            return {"output": "Command must be a string", "returncode": 1}
        if sudo_stdin is not None:
            stdin_data = sudo_stdin + (stdin_data or "")
        expanduser = aiofiles.os.wrap(os.path.expanduser)
        workdir = os.path.normpath(  # noqa: ASYNC240 - lexical only
            await expanduser(cwd or self.cwd)
        )
        if not await _cwd_usable(workdir):
            recovered = await _resolve_safe_cwd(workdir)
            logger.warning(
                "Terminal working directory %s is missing or inaccessible; "
                "recovered to %s",
                workdir,
                recovered,
            )
            workdir = recovered
            async with self._lock:
                self.cwd = recovered
        limit = float(timeout or self.timeout)
        shell = os.environ.get("SHELL") or "/bin/sh"
        marker = f"__HERMES_LOCAL_STATE_{uuid.uuid4().hex}__"
        cwd_marker = f"\n{marker}_CWD="
        env_marker = f"\n{marker}_ENV\n"
        end_marker = f"\n{marker}_END\n"
        wrapped = (
            f"cd {shlex.quote(workdir)} && {{ {prepared_command}; }}; _rc=$?; "
            f"command printf {shlex.quote(cwd_marker + '%s' + env_marker)} \"$PWD\"; "
            f"/usr/bin/env -0; command printf {shlex.quote(end_marker)}; exit $_rc"
        )
        try:
            process = await asyncio.create_subprocess_shell(
                wrapped,
                executable=shell,
                cwd=workdir,
                start_new_session=os.name == "posix",
                stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=await build_subprocess_env(base=self.env, scrub_secrets=False),
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
            if stdin_data is not None and process.stdin is not None:
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
                        f"{output.rstrip()}\n{timeout_message}"
                        if output
                        else timeout_message
                    ),
                    "returncode": 124,
                }
            except asyncio.CancelledError:
                await _terminate_process(process)
                await _finish_stream_readers(readers)
                raise
            await _finish_stream_readers(readers)
        except OSError as exc:
            return {"output": f"Failed to execute command: {exc}", "returncode": 1}

        stdout = b"".join(stdout_chunks)
        cwd_token = cwd_marker.encode()
        env_token = env_marker.encode()
        end_token = end_marker.encode()
        cwd_start = stdout.rfind(cwd_token)
        env_start = stdout.find(env_token, cwd_start + len(cwd_token))
        state_end = stdout.find(end_token, env_start + len(env_token))
        if cwd_start >= 0:
            if env_start >= 0 and state_end >= 0:
                new_cwd = os.fsdecode(stdout[cwd_start + len(cwd_token) : env_start])
                captured_env: dict[str, str] = {}
                for entry in stdout[
                    env_start + len(env_token) : state_end
                ].split(b"\0"):
                    if b"=" not in entry:
                        continue
                    key, value = entry.split(b"=", 1)
                    captured_env[os.fsdecode(key)] = os.fsdecode(value)
                if captured_env:
                    async with self._lock:
                        self.env = captured_env
                        if new_cwd and await aiofiles.os.path.isdir(new_cwd):
                            self.cwd = new_cwd
            stdout = stdout[:cwd_start].rstrip(b"\n")
        output = stdout.decode(errors="replace") + b"".join(stderr_chunks).decode(
            errors="replace"
        )
        return {"output": output, "returncode": process.returncode}

    async def cleanup(self) -> None:
        """Release environment resources after all subprocesses are reaped."""


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


async def _cwd_usable(path: str) -> bool:
    """Return whether *path* is a directory the subprocess can enter."""
    return await aiofiles.os.path.isdir(path) and await aiofiles.os.access(
        path, os.X_OK
    )


async def _resolve_safe_cwd(path: str) -> str:
    """Return the closest existing, accessible directory at or above *path*."""
    if not path:
        return await aiofiles.os.wrap(tempfile.gettempdir)()
    expanduser = aiofiles.os.wrap(os.path.expanduser)
    expanded = await expanduser(path)
    if not os.path.isabs(expanded):
        expanded = os.path.join(await aiofiles.os.getcwd(), expanded)
    candidate = Path(os.path.normpath(expanded))  # noqa: ASYNC240 - lexical only
    for current in (candidate, *candidate.parents):
        if await _cwd_usable(str(current)):
            return str(current)
    return await aiofiles.os.wrap(tempfile.gettempdir)()
