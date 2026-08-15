"""Pure environment helpers for the native asyncio subprocess backend."""

from __future__ import annotations

import asyncio
import codecs
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
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from collections.abc import Awaitable, Callable

import aiofiles
import aiofiles.os

# Prime modules used by retained local imports before subprocess async paths run.
import hermes_constants as _hermes_constants  # noqa: F401
from tools import self_repo_guard as _self_repo_guard_bootstrap  # noqa: F401

from agent.delegation_context import delegated_child_subprocess_env
from tools.environments.base import (
    BaseEnvironment,
    _BoundedOutputCollector,
    _UNBOUNDED_CAPTURE_CHARS,
    touch_activity_if_due,
)

try:
    from gateway import session_context as _session_context_bootstrap  # noqa: F401
except Exception:
    _session_context_bootstrap = None
try:
    from tools import env_passthrough as _env_passthrough_bootstrap  # noqa: F401
except Exception:
    _env_passthrough_bootstrap = None


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


def _windows_to_msys_path(path: str) -> str:
    """Translate a native Windows drive path to its Git Bash form."""
    if not _IS_WINDOWS or not path:
        return path
    match = re.match(r"^([a-zA-Z]):[\\/]*(.*)$", path)
    if not match:
        return path
    drive = match.group(1).lower()
    tail = (match.group(2) or "").replace("\\", "/").lstrip("/")
    return f"/{drive}/{tail}" if tail else f"/{drive}/"


def _bash_safe_path(path: str) -> str:
    """Return *path* in the form expected by Git Bash on Windows."""
    if not _IS_WINDOWS or not path:
        return path
    path = _windows_to_msys_path(path)
    return path.replace("\\", "/")


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
        "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
        "AWS_ROLE_SESSION_NAME",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "AZURE_CLIENT_ID",
        "AZURE_CLIENT_SECRET",
        "AZURE_TENANT_ID",
        "AZURE_CLIENT_CERTIFICATE_PATH",
        "AZURE_CLIENT_CERTIFICATE_PASSWORD",
        "AZURE_CLIENT_SEND_CERTIFICATE_CHAIN",
        "AZURE_FEDERATED_TOKEN_FILE",
        "AZURE_TOKEN_CREDENTIALS",
        "AZURE_USERNAME",
        "AZURE_PASSWORD",
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
    delegated_env = delegated_child_subprocess_env(sanitized)
    assert delegated_env is not None
    return delegated_env


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
    delegated_env = delegated_child_subprocess_env(source)
    assert delegated_env is not None
    return delegated_env


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
    else:
        from agent.secret_scope import (
            UnscopedSecretError,
            current_secret_scope,
            is_multiplex_active,
        )

        if is_multiplex_active():
            scope = current_secret_scope()
            if scope is None:
                raise UnscopedSecretError(
                    "A model-driving subprocess requested credentials without "
                    "an active profile secret scope while multiplexing is enabled."
                )
            # Replace, never overlay, the process credential set. The process
            # environment may still contain the last profile loaded by a
            # legacy host; only this task's profile may reach the child.
            for key in _HERMES_PROVIDER_ENV_BLOCKLIST:
                env.pop(key, None)
                if key not in _ALWAYS_STRIP_KEYS and key in scope:
                    env[key] = scope[key]
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
    delegated_env = delegated_child_subprocess_env(env)
    assert delegated_env is not None
    return delegated_env


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


async def _terminate_and_reap(process: asyncio.subprocess.Process) -> None:
    """Terminate and reap a foreground shell process and its process group."""
    if not _IS_WINDOWS:
        pgid = getattr(process, "_hermes_pgid", process.pid)

        def _group_alive() -> bool:
            try:
                os.killpg(pgid, 0)
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                return True

        async def _wait_for_group_exit(grace_seconds: float) -> bool:
            deadline = asyncio.get_running_loop().time() + grace_seconds
            while _group_alive():
                if process.returncode is not None:
                    await process.wait()
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    return not _group_alive()
                await asyncio.sleep(min(0.05, remaining))
            if process.returncode is not None:
                await process.wait()
            return True

        try:
            if _group_alive():
                os.killpg(pgid, signal.SIGTERM)
                if not await _wait_for_group_exit(1.0):
                    os.killpg(pgid, signal.SIGKILL)
                    await _wait_for_group_exit(2.0)
            if process.returncode is None:
                await process.wait()
        except (ProcessLookupError, PermissionError, OSError):
            if process.returncode is None:
                process.kill()
            await process.wait()
        return

    if process.returncode is not None:
        await process.wait()
        return
    try:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except TimeoutError:
            if process.returncode is None:
                process.kill()
            await process.wait()
    except ProcessLookupError:
        return


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    """Finish owned process termination before propagating cancellation."""
    terminate_task = asyncio.create_task(_terminate_and_reap(process))
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            await asyncio.shield(terminate_task)
            break
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised below
            if terminate_task.cancelled():
                raise
            if cancellation is None:
                cancellation = exc
        except Exception as exc:
            if cancellation is not None:
                raise cancellation from exc
            raise
    if cancellation is not None:
        raise cancellation


async def _finish_process_cleanup(
    process: asyncio.subprocess.Process,
    readers: list[asyncio.Task[Any]],
) -> None:
    """Terminate a child and collect its readers as one owned cleanup."""

    async def _cleanup() -> None:
        try:
            await _terminate_process(process)
        finally:
            await _finish_stream_readers(readers)

    cleanup_task = asyncio.create_task(_cleanup())
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            await asyncio.shield(cleanup_task)
            break
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised below
            if cleanup_task.cancelled():
                raise
            if cancellation is None:
                cancellation = exc
        except Exception as exc:
            if cancellation is not None:
                raise cancellation from exc
            raise
    if cancellation is not None:
        raise cancellation


async def _find_bash() -> str:
    """Locate bash using native async filesystem probes."""
    candidates: list[str] = []
    for directory in os.getenv("PATH", os.defpath).split(os.pathsep):
        candidates.append(os.path.join(directory or ".", "bash"))
    candidates.extend(("/bin/bash", "/usr/bin/bash", "/opt/homebrew/bin/bash"))
    for candidate in candidates:
        if await aiofiles.os.path.isfile(candidate) and await aiofiles.os.access(
            candidate,
            os.X_OK,
        ):
            return candidate
    raise FileNotFoundError("bash executable not found")


async def _resolve_shell_init_files() -> list[str]:
    """Resolve the upstream shell-init configuration without blocking I/O."""
    explicit: list[str] = []
    auto_source = True
    try:
        from hermes_cli.config import load_config_readonly

        config = await load_config_readonly()
        terminal = config.get("terminal") or {}
        configured = terminal.get("shell_init_files") or []
        if isinstance(configured, list):
            explicit = [str(path) for path in configured if path]
        auto_source = bool(terminal.get("auto_source_bashrc", True))
    except Exception:
        pass
    candidates = (
        explicit
        if explicit
        else ["~/.profile", "~/.bash_profile", "~/.bashrc"]
        if auto_source and not _IS_WINDOWS
        else []
    )
    expanduser = aiofiles.os.wrap(os.path.expanduser)
    resolved: list[str] = []
    for raw in candidates:
        try:
            path = os.path.expandvars(  # noqa: ASYNC240 - pure env-string transform
                await expanduser(raw)
            )
            if path and await aiofiles.os.path.isfile(path):
                resolved.append(path)
        except asyncio.CancelledError:
            raise
        except Exception:
            continue
    return resolved


def _prepend_shell_init(command: str, files: list[str]) -> str:
    if not files:
        return command
    prelude = ["set +e"]
    for path in files:
        quoted = shlex.quote(path)
        prelude.append(f"[ -r {quoted} ] && . {quoted} 2>/dev/null || true")
    return "\n".join(prelude) + "\n" + command


async def _drain_process_output(
    stream: asyncio.StreamReader | None,
    collector: _BoundedOutputCollector,
    decoder,
) -> None:
    if stream is None:
        return
    while chunk := await stream.read(64 * 1024):
        collector.append(decoder.decode(chunk))
    collector.append(decoder.decode(b"", final=True))


class LocalEnvironment(BaseEnvironment):
    """Run host commands through the shared upstream session-snapshot contract."""

    _profile_scoped_passthrough = True

    def _additional_profile_scoped_passthrough_names(self) -> tuple[str, ...]:
        """Keep task-local profile-home values out of shared snapshots."""
        return ("HERMES_HOME", "HERMES_REAL_HOME", "HOME")

    def __init__(self, cwd: str = "", timeout: int = 60, env: dict | None = None):
        super().__init__(cwd=cwd, timeout=timeout, env=env)
        initial_env = os.environ.copy()
        if env:
            initial_env.update(env)
        self.env = initial_env
        self._persistent = False
        self._initialized = False
        self._lock = asyncio.Lock()
        self._cwd_candidate: contextvars.ContextVar[str | None] = (
            contextvars.ContextVar(
                f"local_cwd_candidate_{self._session_id}",
                default=None,
            )
        )

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
        """Resolve host state and capture the login-shell snapshot lazily."""
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
            await BaseEnvironment.init_session(self)
            await self._validate_cwd_update(cwd)

    @staticmethod
    def _quote_cwd_for_cd(cwd: str) -> str:
        return BaseEnvironment._quote_cwd_for_cd(_windows_to_msys_path(cwd))

    def _quote_shell_path(self, path: str) -> str:
        return shlex.quote(_bash_safe_path(path))

    def _extract_cwd_from_output(self, result: dict[str, Any]) -> None:
        """Normalize Git Bash cwd markers before their async validation."""
        previous = self.cwd
        super()._extract_cwd_from_output(result)
        candidate = self.cwd
        self.cwd = previous
        if candidate != previous and _IS_WINDOWS:
            candidate = _msys_to_windows_path(candidate)
        self._cwd_candidate.set(candidate if candidate != previous else None)

    async def _validate_cwd_update(self, previous: str) -> None:
        """Roll back a cwd marker that no longer names a real directory."""
        candidate = self._cwd_candidate.get()
        self._cwd_candidate.set(None)
        if not candidate or candidate == previous:
            return
        if await aiofiles.os.path.isdir(candidate):
            self.cwd = candidate

    async def execute(  # noqa: ASYNC109 - upstream public API names timeout
        self,
        command: str,
        cwd: str = "",
        *,
        timeout: int | None = None,  # noqa: ASYNC109 - upstream public API
        stdin_data: str | None = None,
        rewrite_compound_background: bool = True,
        bounded_capture: bool = False,
    ) -> dict:
        """Execute and validate the cwd marker through native async metadata I/O."""
        await self._ensure_initialized()
        previous = self.cwd
        result = await super().execute(
            command,
            cwd,
            timeout=timeout,
            stdin_data=stdin_data,
            rewrite_compound_background=rewrite_compound_background,
            bounded_capture=bounded_capture,
        )
        await self._validate_cwd_update(previous)
        return result

    async def _run_bash(  # noqa: ASYNC109 - upstream public API names timeout
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int | float = 120,  # noqa: ASYNC109 - public API
        stdin_data: str | None = None,
        bounded_capture: bool = False,
    ) -> dict[str, Any]:
        """Spawn bash and stream its output without a worker thread."""
        if login:
            init_files = await _resolve_shell_init_files()
            cmd_string = _prepend_shell_init(cmd_string, init_files)
        if not await _cwd_usable(self.cwd):
            recovered = await _resolve_safe_cwd(self.cwd)
            logger.warning(
                "LocalEnvironment cwd %r is missing on disk; falling back "
                "to %r so terminal commands keep working.",
                self.cwd,
                recovered,
            )
            self.cwd = recovered
        workdir = self.cwd
        shell = await _find_bash()
        argv = [shell, "-l", "-c", cmd_string] if login else [shell, "-c", cmd_string]
        collector = (
            await self._bounded_output_collector()
            if bounded_capture
            else _BoundedOutputCollector(_UNBOUNDED_CAPTURE_CHARS)
        )
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        activity_state = {
            "start": time.monotonic(),
            "last_touch": time.monotonic(),
            "interval": 10.0,
        }
        try:
            # Upstream overlays the environment snapshot on the current host
            # environment for every spawn.  Preserve that ordering so benign
            # runtime additions stay visible while the sanitizer below still
            # re-resolves profile passthrough and strips protected secrets.
            current_env = os.environ.copy()
            current_env.update(self.env)
            run_env = await build_subprocess_env(
                base=current_env,
                # Re-resolve allowlisted values against the active profile on
                # every spawn.  ``self.env`` is the sanitized initialization
                # snapshot and may have been created under a different
                # multiplexed profile; treating it as already final would
                # leak that profile's passthrough value into later turns.
                scrub_secrets=True,
            )
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=workdir,
                start_new_session=os.name == "posix",
                stdin=(
                    asyncio.subprocess.PIPE
                    if stdin_data is not None
                    else asyncio.subprocess.DEVNULL
                ),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=run_env,
            )
            if not _IS_WINDOWS:
                process._hermes_pgid = process.pid
            reader = asyncio.create_task(
                _drain_process_output(process.stdout, collector, decoder),
                name=f"local-output-{process.pid}",
            )
            wait_task = asyncio.create_task(
                process.wait(),
                name=f"local-wait-{process.pid}",
            )
            owned_tasks = [reader, wait_task]
            try:
                deadline = asyncio.get_running_loop().time() + float(timeout)
                if stdin_data is not None and process.stdin is not None:
                    try:
                        process.stdin.write(stdin_data.encode())
                        async with asyncio.timeout_at(deadline):
                            await process.stdin.drain()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    finally:
                        process.stdin.close()
                while not wait_task.done():
                    from tools.interrupt import is_interrupted

                    if is_interrupted():
                        await _finish_process_cleanup(process, owned_tasks)
                        suffix = "\n[Command interrupted]"
                        output = collector.render(suffix=suffix)
                        if collector.total_chars == 0:
                            output = output.lstrip()
                        return await self._finalize_wait_result(
                            collector,
                            output,
                            130,
                        )
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise TimeoutError
                    await asyncio.wait(
                        (wait_task,),
                        timeout=min(0.2, remaining),
                    )
                    if not wait_task.done():
                        touch_activity_if_due(
                            activity_state,
                            "terminal command running",
                        )
                await wait_task
            except TimeoutError:
                await _finish_process_cleanup(process, owned_tasks)
                suffix = f"\n[Command timed out after {timeout}s]"
                output = collector.render(suffix=suffix)
                if collector.total_chars == 0:
                    output = output.lstrip()
                return await self._finalize_wait_result(
                    collector,
                    output,
                    124,
                )
            except asyncio.CancelledError:
                await _finish_process_cleanup(process, owned_tasks)
                raise
            except BaseException:
                await _finish_process_cleanup(  # noqa: ASYNC120 - cleanup must finish
                    process,
                    owned_tasks,
                )
                raise
            await _finish_stream_readers(owned_tasks)
        except OSError as exc:
            return {"output": f"Failed to execute command: {exc}", "returncode": 1}
        return await self._finalize_wait_result(
            collector,
            collector.render(),
            process.returncode,
        )

    async def cleanup(self) -> None:
        """Remove the session snapshot and any orphaned atomic temp files."""
        for path in (self._snapshot_path, self._cwd_file):
            try:
                await aiofiles.os.remove(path)
            except OSError:
                pass
        parent = Path(self._snapshot_path).parent
        prefix = Path(self._snapshot_path).name + ".tmp."
        try:
            for name in await aiofiles.os.listdir(parent):
                if name.startswith(prefix):
                    try:
                        await aiofiles.os.remove(parent / name)
                    except OSError:
                        pass
        except OSError:
            pass


async def _finish_stream_readers(readers: list[asyncio.Task[Any]]) -> None:
    """Collect available output without waiting on pipes inherited by children."""
    async def _collect() -> None:
        _, pending = await asyncio.wait(readers, timeout=0.25)
        for task in pending:
            task.cancel()
        await asyncio.gather(*readers, return_exceptions=True)

    collect_task = asyncio.create_task(_collect(), name="local-reader-cleanup")
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            await asyncio.shield(collect_task)
            break
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised below
            if collect_task.cancelled():
                raise
            if cancellation is None:
                cancellation = exc
        except Exception as exc:
            if cancellation is not None:
                raise cancellation from exc
            raise
    if cancellation is not None:
        raise cancellation


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
