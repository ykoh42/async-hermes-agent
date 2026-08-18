"""Terminal tool for local, container, SSH, and cloud sandbox execution.

The retained environment selection follows upstream ``TERMINAL_ENV`` behavior
for local, Docker, SSH, Singularity, Modal, Daytona, and Vercel Sandbox
backends. Environment creation, command execution, and cleanup are awaited at
their existing public boundaries.
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
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any
from collections.abc import Callable

import aiofiles
import aiofiles.os

# Preserve the original local import patch points after priming their modules.
from agent import redact as _redact_bootstrap  # noqa: F401
from agent import secret_scope as _secret_scope_bootstrap  # noqa: F401
from agent import verification_evidence as _verification_evidence_bootstrap  # noqa: F401
from hermes_cli import lifecycle as _lifecycle_bootstrap
# The retained foreground path consults ``lifecycle.has_hook`` after the
# subprocess has completed.  That helper intentionally keeps the plugin
# module import lazy, so prime the stable plugin registration graph beside the
# terminal module rather than letting importlib walk it from an active event
# loop on the first direct ``terminal_tool`` call.
from hermes_cli import plugins as _plugins_bootstrap  # noqa: F401
from hermes_cli import managed_scope as _managed_scope_bootstrap  # noqa: F401
from tools import ansi_strip as _ansi_strip_bootstrap  # noqa: F401
from tools import file_tools as _file_tools_bootstrap  # noqa: F401
from tools import interrupt as _interrupt_bootstrap
from tools import process_registry as _process_registry_bootstrap  # noqa: F401
from tools import shell_heredoc as _shell_heredoc_bootstrap  # noqa: F401
from tools import self_repo_guard as _self_repo_guard_bootstrap  # noqa: F401
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
from tools.environments.base import BaseEnvironment, EnvironmentConnectionError
from tools.environments import file_sync as _file_sync_bootstrap  # noqa: F401
from tools.environments.singularity import SingularityEnvironment
from tools.approval import check_all_command_guards
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)


def _redact_terminal_error_text(value: Any) -> str:
    """Force-redact exception text before returning a terminal error envelope."""
    from agent.redact import redact_sensitive_text

    return redact_sensitive_text("" if value is None else str(value), force=True)

_TerminalScopeKey = tuple[asyncio.AbstractEventLoop | object, str]
_TERMINAL_NO_LOOP = object()
_terminal_scope_context: contextvars.ContextVar[
    tuple[str, str] | None
] = contextvars.ContextVar("terminal_profile_scope", default=None)
_terminal_scope_aliases: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, str]
] = weakref.WeakKeyDictionary()
_terminal_scope_aliases_lock = threading.RLock()


def _lexical_terminal_profile_identity() -> str:
    from hermes_constants import get_hermes_home

    return os.path.normcase(os.fspath(get_hermes_home()))


def _current_terminal_scope_key() -> _TerminalScopeKey:
    _prune_closed_terminal_loops()
    lexical = _lexical_terminal_profile_identity()
    try:
        loop: object = asyncio.get_running_loop()
    except RuntimeError:
        loop = _TERMINAL_NO_LOOP
    active = _terminal_scope_context.get()
    if active is not None and active[0] == lexical and loop is not _TERMINAL_NO_LOOP:
        return loop, active[1]
    if loop is _TERMINAL_NO_LOOP:
        return loop, lexical
    with _terminal_scope_aliases_lock:
        aliases = _terminal_scope_aliases.get(loop)
        canonical = aliases.get(lexical, lexical) if aliases is not None else lexical
    return loop, canonical


async def _activate_terminal_scope() -> _TerminalScopeKey:
    _prune_closed_terminal_loops()
    loop = asyncio.get_running_loop()
    lexical = _lexical_terminal_profile_identity()
    active = _terminal_scope_context.get()
    if active is not None and active[0] == lexical:
        canonical = active[1]
    else:
        expanduser = aiofiles.os.wrap(os.path.expanduser)
        expanded = str(await expanduser(lexical))
        is_absolute = (
            expanded.startswith(("/", "\\\\"))
            or (len(expanded) >= 3 and expanded[1] == ":" and expanded[2] in "/\\")
        )
        if not is_absolute:
            expanded = str(await aiofiles.os.getcwd()) + os.sep + expanded
        realpath = aiofiles.os.wrap(os.path.realpath)
        canonical = os.path.normcase(str(await realpath(expanded)))
    scope = (loop, canonical)
    with _terminal_scope_aliases_lock:
        _terminal_scope_aliases.setdefault(loop, {})[lexical] = canonical
    _terminal_scope_context.set((lexical, canonical))
    for source in ((loop, lexical), (_TERMINAL_NO_LOOP, lexical)):
        for scoped in (
            _active_environments,
            _last_activity,
            _task_env_overrides,
            _session_cwds,
            _container_aliases,
        ):
            migrate = getattr(scoped, "migrate", None)
            if callable(migrate):
                migrate(source, scope)
    return scope


class _ScopedTerminalDict(MutableMapping):
    """Dict-compatible active-profile view retained for private test hooks."""

    def __init__(self) -> None:
        self._loop_values: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, dict[str, dict]
        ] = weakref.WeakKeyDictionary()
        self._staged_values: dict[str, dict] = {}

    @property
    def _values(self) -> dict[_TerminalScopeKey, dict]:
        values = {
            (loop, profile): scoped
            for loop, profiles in self._loop_values.items()
            for profile, scoped in profiles.items()
        }
        values.update(
            ((_TERMINAL_NO_LOOP, profile), scoped)
            for profile, scoped in self._staged_values.items()
        )
        return values

    def prune_closed_loops(self) -> None:
        for loop in tuple(self._loop_values):
            if loop.is_closed():
                self._loop_values.pop(loop, None)

    def scoped(self, scope: _TerminalScopeKey) -> dict:
        loop, profile = scope
        if loop is _TERMINAL_NO_LOOP:
            return self._staged_values.setdefault(profile, {})
        return self._loop_values.setdefault(loop, {}).setdefault(profile, {})

    def scoped_if_present(self, scope: _TerminalScopeKey) -> dict | None:
        loop, profile = scope
        if loop is _TERMINAL_NO_LOOP:
            return self._staged_values.get(profile)
        profiles = self._loop_values.get(loop)
        return profiles.get(profile) if profiles is not None else None

    def discard_scope(self, scope: _TerminalScopeKey) -> None:
        loop, profile = scope
        if loop is _TERMINAL_NO_LOOP:
            self._staged_values.pop(profile, None)
            return
        profiles = self._loop_values.get(loop)
        if profiles is None:
            return
        profiles.pop(profile, None)
        if not profiles:
            self._loop_values.pop(loop, None)

    def _active(self) -> dict:
        return self.scoped(_current_terminal_scope_key())

    def __getitem__(self, key):
        return self._active()[key]

    def __setitem__(self, key, value) -> None:
        self._active()[key] = value

    def __delitem__(self, key) -> None:
        del self._active()[key]

    def __iter__(self):
        return iter(tuple(self._active()))

    def __len__(self) -> int:
        return len(self._active())

    def clear(self) -> None:
        self.discard_scope(_current_terminal_scope_key())

    def migrate(self, source: _TerminalScopeKey, target: _TerminalScopeKey) -> None:
        if source == target:
            return
        staged = self.scoped_if_present(source)
        if staged is None:
            return
        self.discard_scope(source)
        self.scoped(target).update(staged)


_active_environments: MutableMapping[str, BaseEnvironment] = _ScopedTerminalDict()
_env_lock = threading.RLock()
_last_activity: MutableMapping[str, float] = _ScopedTerminalDict()
_creation_locks_lock = threading.Lock()
_creation_locks: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, dict[str, asyncio.Lock]]
] = weakref.WeakKeyDictionary()
_cleanup_tasks: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, asyncio.Task[None]]
] = weakref.WeakKeyDictionary()
_cleanup_handles: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, asyncio.TimerHandle]
] = weakref.WeakKeyDictionary()
_cleanup_lifetimes: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, int]
] = weakref.WeakKeyDictionary()
_task_env_overrides: MutableMapping[str, dict[str, Any]] = _ScopedTerminalDict()
_session_cwds: MutableMapping[str, str] = _ScopedTerminalDict()
_container_aliases: MutableMapping[str, str] = _ScopedTerminalDict()
_container_alias_lock = threading.RLock()
_CONTAINER_BACKENDS = frozenset(
    {"docker", "singularity", "modal", "daytona", "vercel_sandbox"}
)
_SUPPORTED_BACKENDS = _CONTAINER_BACKENDS | {"local", "ssh", "managed_modal"}
_approval_callback: contextvars.ContextVar[Callable[..., Any] | None] = (
    contextvars.ContextVar("terminal_approval_callback", default=None)
)


def _prune_closed_terminal_loops() -> None:
    for scoped in (
        _active_environments,
        _last_activity,
        _task_env_overrides,
        _session_cwds,
        _container_aliases,
    ):
        prune = getattr(scoped, "prune_closed_loops", None)
        if callable(prune):
            prune()
    with _terminal_scope_aliases_lock:
        for loop in tuple(_terminal_scope_aliases):
            if loop.is_closed():
                _terminal_scope_aliases.pop(loop, None)
    with _creation_locks_lock:
        for loop in tuple(_creation_locks):
            if loop.is_closed():
                _creation_locks.pop(loop, None)
    for mapping in (_cleanup_tasks, _cleanup_handles, _cleanup_lifetimes):
        for loop in tuple(mapping):
            if loop.is_closed():
                mapping.pop(loop, None)


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
_VERCEL_SANDBOX_DEFAULT_CWD = "/vercel/sandbox"
_SUPPORTED_VERCEL_RUNTIMES = ("node24", "node22", "python3.13")


def _is_supported_vercel_runtime(runtime: str) -> bool:
    return not runtime or runtime in _SUPPORTED_VERCEL_RUNTIMES


async def _check_vercel_sandbox_requirements(config: dict[str, Any]) -> bool:
    """Validate Vercel Sandbox terminal backend requirements."""
    runtime = str(config.get("vercel_runtime") or "").strip()
    if not _is_supported_vercel_runtime(runtime):
        supported = ", ".join(_SUPPORTED_VERCEL_RUNTIMES)
        logger.error(
            "Vercel Sandbox runtime %r is not supported. "
            "Set TERMINAL_VERCEL_RUNTIME to one of: %s.",
            runtime,
            supported,
        )
        return False

    disk = config.get("container_disk", 51200)
    if disk not in {0, 51200}:
        logger.error(
            "Vercel Sandbox does not support custom TERMINAL_CONTAINER_DISK=%s. "
            "Use the default shared setting (51200 MB).",
            disk,
        )
        return False

    from hermes_cli.async_source_loader import _locate_source_module

    if await _locate_source_module("vercel") is None:
        logger.error(
            "vercel is required for the Vercel Sandbox terminal backend: "
            "pip install vercel"
        )
        return False

    from agent.secret_scope import get_secret

    has_oidc = bool(get_secret("VERCEL_OIDC_TOKEN"))
    has_token = bool(get_secret("VERCEL_TOKEN"))
    has_project = bool(get_secret("VERCEL_PROJECT_ID"))
    has_team = bool(get_secret("VERCEL_TEAM_ID"))
    if has_oidc:
        return True
    if has_token or has_project or has_team:
        if has_token and has_project and has_team:
            return True
        logger.error(
            "Vercel Sandbox backend selected with token auth, but "
            "VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID must all "
            "be set together. VERCEL_OIDC_TOKEN is supported for one-off "
            "local development only."
        )
        return False
    logger.error(
        "Vercel Sandbox backend selected but no supported auth configuration "
        "was found. Set VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID "
        "for normal use. VERCEL_OIDC_TOKEN is supported for one-off local "
        "development only."
    )
    return False

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
    try:
        from tools.shell_heredoc import strip_inert_heredoc_bodies

        command = strip_inert_heredoc_bodies(command)
    except Exception:
        # The quote scanner remains the conservative fallback if the optional
        # heredoc helper cannot be imported during process bootstrap.
        pass
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


def _resolve_notification_flag_conflict(
    *,
    notify_on_complete: bool,
    watch_patterns: Any,
    background: bool,
) -> tuple[Any, str]:
    """Preserve upstream's model-facing conflict note for background flags."""
    if background and notify_on_complete and watch_patterns:
        return (
            None,
            "watch_patterns ignored because notify_on_complete=True; "
            "these two flags produce duplicate notifications when combined",
        )
    return watch_patterns, ""


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
    from agent.secret_scope import get_secret

    raw = get_secret(name, default)
    try:
        return converter(raw)
    except (ValueError, json.JSONDecodeError):
        raise ValueError(
            f"Invalid value for {name}: {raw!r} (expected {type_label}). "
            "Check ~/.hermes/.env or environment variables."
        )


async def _get_env_config() -> dict[str, Any]:
    """Get terminal configuration with upstream config/env precedence."""
    from agent.secret_scope import get_secret
    from hermes_cli.config import (
        TERMINAL_CONFIG_ENV_MAP,
        load_config_readonly,
        read_user_config_raw,
    )

    try:
        raw_config, merged_config = await asyncio.gather(
            read_user_config_raw(),
            load_config_readonly(),
        )
    except Exception:
        logger.debug("terminal config → env fallback bridge failed", exc_info=True)
        raw_config, merged_config = {}, {}
    raw_terminal = raw_config.get("terminal", {})
    if not isinstance(raw_terminal, dict):
        raw_terminal = {}
    merged_terminal = merged_config.get("terminal", {})
    if not isinstance(merged_terminal, dict):
        merged_terminal = {}

    def setting(config_key: str, default: Any) -> Any:
        env_name = TERMINAL_CONFIG_ENV_MAP.get(config_key)
        if config_key in raw_terminal:
            value = raw_terminal[config_key]
            if config_key != "cwd" or str(value or "").strip() not in {
                ".",
                "auto",
                "cwd",
            }:
                return value
        if env_name:
            env_value = get_secret(env_name)
            if env_value is not None:
                return env_value
        value = merged_terminal.get(config_key, default)
        if config_key == "cwd" and str(value or "").strip() in {
            ".",
            "auto",
            "cwd",
        }:
            return default
        return value

    env_type = str(setting("backend", "local"))
    default_image = "nikolaik/python-nodejs:python3.11-nodejs20"
    default_cwd = (
        await _safe_getcwd()
        if env_type == "local"
        else "/vercel/sandbox"
        if env_type == "vercel_sandbox"
        else "~"
        if env_type == "ssh"
        else "/root"
    )
    raw_cwd = str(setting("cwd", default_cwd))
    expanduser = aiofiles.os.wrap(os.path.expanduser)
    cwd = (
        raw_cwd
        if env_type == "ssh" and (raw_cwd == "~" or raw_cwd.startswith("~/"))
        else await expanduser(raw_cwd)
    )
    container_backend = env_type in _CONTAINER_BACKENDS
    docker_backend = env_type == "docker"
    mount_docker_cwd = str(
        setting("docker_mount_cwd_to_workspace", "false")
    ).lower() in {"true", "1", "yes"}
    host_cwd = None
    if env_type == "docker" and mount_docker_cwd:
        source = str(setting("cwd", await _safe_getcwd()))
        candidate = os.path.abspath(await expanduser(source))  # noqa: ASYNC240
        if (
            any(candidate.startswith(prefix) for prefix in _HOST_CWD_PREFIXES)
            or (
                os.path.isabs(candidate)
                and await aiofiles.os.path.isdir(candidate)
                and not candidate.startswith(("/workspace", "/root"))
            )
        ):
            host_cwd = candidate
            cwd = "/workspace"
    elif container_backend and cwd:
        if _is_unusable_container_cwd(cwd) and cwd != default_cwd:
            logger.info(
                "Ignoring TERMINAL_CWD=%r for %s backend "
                "(host/relative path won't work in sandbox). Using %r instead.",
                cwd,
                env_type,
                default_cwd,
            )
            cwd = default_cwd

    def env_text(config_key: str, default: str) -> str:
        value = setting(config_key, default)
        if isinstance(value, (list, dict)):
            return json.dumps(value)
        return str(value)

    def parsed(
        config_key: str,
        env_name: str,
        default: str,
        converter: Callable[[str], Any] = int,
        type_label: str = "integer",
    ) -> Any:
        raw = env_text(config_key, default)
        try:
            return converter(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            raise ValueError(
                f"Invalid value for {env_name}: {raw!r} "
                f"(expected {type_label}). Check ~/.hermes/.env or "
                "environment variables."
            )

    return {
        "env_type": env_type,
        "modal_mode": env_text("modal_mode", "auto"),
        "cwd": os.path.normpath(cwd) if cwd else cwd,  # noqa: ASYNC240
        "host_cwd": host_cwd,
        "docker_mount_cwd_to_workspace": mount_docker_cwd,
        "timeout": parsed("timeout", "TERMINAL_TIMEOUT", "180"),
        "lifetime_seconds": parsed(
            "lifetime_seconds", "TERMINAL_LIFETIME_SECONDS", "300"
        ),
        "docker_image": env_text("docker_image", default_image),
        "singularity_image": env_text(
            "singularity_image", f"docker://{default_image}"
        ),
        "modal_image": env_text("modal_image", default_image),
        "daytona_image": env_text("daytona_image", default_image),
        "vercel_runtime": env_text("vercel_runtime", "").strip(),
        "ssh_host": env_text("ssh_host", ""),
        "ssh_user": env_text("ssh_user", ""),
        "ssh_port": parsed("ssh_port", "TERMINAL_SSH_PORT", "22"),
        "ssh_key": env_text("ssh_key", ""),
        "ssh_persistent": str(
            get_secret(
                "TERMINAL_SSH_PERSISTENT",
                env_text("persistent_shell", "true"),
            )
        ).lower()
        in {"true", "1", "yes"},
        "container_cpu": (
            parsed(
                "container_cpu", "TERMINAL_CONTAINER_CPU", "1", float, "number"
            )
            if container_backend
            else 1.0
        ),
        "container_memory": (
            parsed(
                "container_memory", "TERMINAL_CONTAINER_MEMORY", "5120"
            )
            if container_backend
            else 5120
        ),
        "container_disk": (
            parsed("container_disk", "TERMINAL_CONTAINER_DISK", "51200")
            if container_backend
            else 51200
        ),
        "container_persistent": env_text(
            "container_persistent", "true"
        ).lower()
        in {"true", "1", "yes"},
        "docker_forward_env": (
            parsed(
                "docker_forward_env",
                "TERMINAL_DOCKER_FORWARD_ENV",
                "[]",
                json.loads,
                "valid JSON",
            )
            if docker_backend
            else []
        ),
        "docker_volumes": (
            parsed(
                "docker_volumes",
                "TERMINAL_DOCKER_VOLUMES",
                "[]",
                json.loads,
                "valid JSON",
            )
            if docker_backend
            else []
        ),
        "docker_env": (
            parsed(
                "docker_env",
                "TERMINAL_DOCKER_ENV",
                "{}",
                json.loads,
                "valid JSON",
            )
            if docker_backend
            else {}
        ),
        "docker_extra_args": (
            parsed(
                "docker_extra_args",
                "TERMINAL_DOCKER_EXTRA_ARGS",
                "[]",
                json.loads,
                "valid JSON",
            )
            if docker_backend
            else []
        ),
        "docker_run_as_host_user": env_text(
            "docker_run_as_host_user", "false"
        ).lower()
        in {"true", "1", "yes"},
        "docker_network": env_text("docker_network", "true").lower()
        in {"true", "1", "yes"},
        "docker_shm_size": env_text("docker_shm_size", "1g"),
        "docker_persist_across_processes": env_text(
            "docker_persist_across_processes", "true"
        ).lower()
        in {"true", "1", "yes"},
        "docker_orphan_reaper": env_text(
            "docker_orphan_reaper", "true"
        ).lower()
        in {"true", "1", "yes"},
    }


async def _safe_getcwd() -> str:
    try:
        return await aiofiles.os.getcwd()
    except FileNotFoundError:
        from agent.secret_scope import get_secret

        fallback = get_secret("TERMINAL_CWD") or os.getenv("HOME", "/root")
        return await aiofiles.os.wrap(os.path.expanduser)(fallback)


_HOST_CWD_PREFIXES = ("/Users/", "/home/", "C:\\", "C:/")


def _is_unusable_container_cwd(cwd: str) -> bool:
    if not cwd:
        return False
    if any(cwd.startswith(prefix) for prefix in _HOST_CWD_PREFIXES):
        return True
    return not os.path.isabs(cwd)  # noqa: ASYNC240 - lexical only


def _docker_volume_uses_host_path(volume: Any) -> bool:
    if isinstance(volume, str):
        source = volume.split(":", 1)[0].strip()
        return bool(source) and (
            source.startswith(("/", "~", ".", "\\\\"))
            or bool(re.match(r"^[A-Za-z]:[\\/]", source))
        )
    if isinstance(volume, dict):
        source = volume.get("source") or volume.get("host_path")
        return isinstance(source, str) and bool(source.strip())
    return False


def _docker_has_host_access(config: dict[str, Any]) -> bool:
    """Return True when Docker exposes host paths through bind mounts."""
    if config.get("env_type") != "docker":
        return False
    if config.get("host_cwd") and config.get("docker_mount_cwd_to_workspace"):
        return True
    return any(
        _docker_volume_uses_host_path(volume)
        for volume in config.get("docker_volumes", [])
    )


_WORKDIR_SAFE_ASCII_CHARS = frozenset('/\\:_-.~ +@=,')


def _is_safe_workdir_char(character: str) -> bool:
    if not character:
        return False
    if ord(character) < 32 or ord(character) == 127:
        return False
    return character.isalnum() or character in _WORKDIR_SAFE_ASCII_CHARS


def _validate_workdir(workdir: str) -> str | None:
    """Reject workdir values containing shell metacharacters or controls."""
    if not workdir:
        return None
    for character in workdir:
        if not _is_safe_workdir_char(character):
            return (
                "Blocked: workdir contains disallowed character "
                f"{character!r}. Use a simple filesystem path without shell "
                "metacharacters."
            )
    return None


def _safe_command_preview(command: Any, limit: int = 200) -> str:
    """Return a bounded, forcibly redacted command preview for logs."""
    if command is None:
        return "<None>"
    if isinstance(command, str):
        value = command
    else:
        try:
            value = repr(command)
        except Exception:
            return f"<{type(command).__name__}>"
    from agent.redact import redact_sensitive_text

    return redact_sensitive_text(
        value,
        force=True,
        redact_url_credentials=True,
    )[:limit]


def _resolve_terminal_spill_path(spill_file_path: Any) -> Any:
    if not spill_file_path:
        return None
    from hermes_constants import get_hermes_home

    value = str(spill_file_path)
    return (
        get_hermes_home() / value
        if not os.path.isabs(value)
        else Path(value)
    )


async def _discard_terminal_spill_artifacts(spill_file_path: Any) -> None:
    """Remove one foreground command's unpublished raw spill artifacts."""
    spill_path = _resolve_terminal_spill_path(spill_file_path)
    if spill_path is None:
        return
    targets = [spill_path]
    try:
        names = await aiofiles.os.listdir(spill_path.parent)
    except OSError:
        names = []
    temporary_prefix = f".{spill_path.name}."
    targets.extend(
        spill_path.parent / name
        for name in names
        if name.startswith(temporary_prefix) and name.endswith(".tmp")
    )
    for target in targets:
        try:
            await aiofiles.os.remove(target)
        except OSError:
            pass


async def _discard_terminal_spill_before_reraise(spill_file_path: Any) -> None:
    """Finish owned spill deletion through repeated caller cancellation."""
    if not spill_file_path:
        return
    from tools.environments.file_sync import _await_owned

    cleanup_task = asyncio.create_task(
        _discard_terminal_spill_artifacts(spill_file_path),
        name="terminal-raw-spill-discard",
    )
    try:
        await _await_owned(cleanup_task)
    except BaseException:  # noqa: ASYNC103 - caller re-raises the active error
        # This helper only runs while another BaseException is already active.
        # Cleanup cancellation/failure must not replace that original error.
        pass


async def _get_modal_backend_state(modal_mode: object | None) -> dict[str, Any]:
    from tools.managed_tool_gateway import is_managed_tool_gateway_ready
    from tools.tool_backend_helpers import (
        has_direct_modal_credentials,
        managed_nous_tools_enabled,
        resolve_modal_backend_state,
    )

    has_direct, managed_ready, managed_enabled = await asyncio.gather(
        has_direct_modal_credentials(),
        is_managed_tool_gateway_ready("modal"),
        managed_nous_tools_enabled(),
    )
    return resolve_modal_backend_state(
        modal_mode,
        has_direct=has_direct,
        managed_ready=managed_ready,
        managed_enabled=managed_enabled,
    )


def _resolve_container_task_id(task_id: str | None) -> str:
    """Map a task id to its shared or per-session sandbox key.

    Backend/image overrides remain the explicit isolation signal used by the
    existing rollout API.  Docker with ``container_persistent=false`` also
    gets upstream's retained per-session isolation; subagent aliases resolve
    back to their parent's container without changing their task identity.
    """
    isolation_keys = frozenset(
        {
            "docker_image",
            "modal_image",
            "singularity_image",
            "daytona_image",
            "env_type",
        }
    )
    if task_id and task_id in _task_env_overrides:
        overrides = _task_env_overrides[task_id]
        if set(overrides) & isolation_keys:
            return task_id
    if task_id and _docker_session_isolation_enabled():
        return _resolve_container_alias(task_id)
    return "default"


def _docker_session_isolation_enabled() -> bool:
    """Whether non-persistent Docker sessions get distinct containers."""
    from agent.secret_scope import get_secret

    env_type = get_secret("TERMINAL_ENV", "local")
    persistent = get_secret("TERMINAL_CONTAINER_PERSISTENT", "true")
    return str(env_type or "local").lower() == "docker" and str(
        persistent or "true"
    ).lower() not in {"true", "1", "yes"}


def register_container_alias(
    child_task_id: str, parent_task_id: str | None
) -> None:
    """Make a delegated child resolve to its parent's sandbox."""
    if not child_task_id:
        return
    with _container_alias_lock:
        _container_aliases[child_task_id] = str(parent_task_id or "default")


def _resolve_container_alias(task_id: str) -> str:
    """Follow child→parent aliases without looping on malformed input."""
    seen: set[str] = set()
    key = task_id
    with _container_alias_lock:
        while key in _container_aliases and key not in seen:
            seen.add(key)
            key = _container_aliases[key]
    return key


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
    raw_key = task_id or "default"
    return dict(
        _task_env_overrides.get(raw_key)
        or _task_env_overrides.get(_resolve_container_task_id(raw_key))
        or {}
    )


def register_task_env_overrides(task_id: str, overrides: dict[str, Any]):
    key = str(task_id or "default")
    _task_env_overrides[key] = dict(overrides or {})
    cwd = _task_env_overrides[key].get("cwd")
    if isinstance(cwd, str) and cwd.strip():
        record_session_cwd(key, cwd)
        container_key = _resolve_container_task_id(key)
        with _env_lock:
            env = _active_environments.get(key) or _active_environments.get(
                container_key
            )
        if env is not None and getattr(env, "cwd", None) is not None:
            env.cwd = cwd


def clear_task_env_overrides(task_id: str) -> None:
    """Remove one task's local environment overrides and cwd anchor."""
    key = str(task_id or "default")
    _task_env_overrides.pop(key, None)
    clear_session_cwd(key)
    with _container_alias_lock:
        _container_aliases.pop(key, None)


def get_session_cwd(session_key: str | None) -> str | None:
    """Return the recorded working directory for a session, if any."""
    key = str(session_key or "default")
    with _env_lock:
        return _session_cwds.get(key)


def record_session_cwd(
    session_key: str | None, cwd: str | None
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


def _resolve_task_host_cwd(
    config: dict[str, Any], task_id: str | None
) -> str | None:
    """Resolve the host path allowed for a Docker ``/workspace`` mount.

    In per-session Docker mode a process-global ``TERMINAL_CWD``/config path
    is not a valid mount source for a fresh session.  Only a cwd explicitly
    attached to that session is accepted; the legacy shared-container path is
    preserved for the default task and persistent containers.
    """
    if config.get("env_type") != "docker":
        return None
    if not config.get("docker_mount_cwd_to_workspace"):
        return None
    if not _docker_session_isolation_enabled():
        return config.get("host_cwd")
    if _resolve_container_task_id(task_id) == "default":
        return config.get("host_cwd")
    overrides = resolve_task_overrides(task_id)
    if overrides.get("cwd_source") == "process":
        return None
    candidate = overrides.get("cwd")
    if not isinstance(candidate, str) or not candidate.strip():
        return None
    candidate = os.path.abspath(os.path.expanduser(candidate))
    if not os.path.isdir(candidate) or candidate.startswith(("/workspace", "/root")):
        return None
    return candidate


def _create_environment(
    env_type: str,
    image: str,
    cwd: str,
    timeout: int,
    ssh_config: dict | None = None,
    container_config: dict | None = None,
    local_config: dict | None = None,
    task_id: str = "default",
    host_cwd: str | None = None,
) -> BaseEnvironment:
    """Construct the selected backend without performing external I/O."""
    del local_config
    container = container_config or {}
    cpu = container.get("container_cpu", 1)
    memory = container.get("container_memory", 5120)
    disk = container.get("container_disk", 51200)
    persistent = container.get("container_persistent", True)
    if env_type == "local":
        return LocalEnvironment(cwd, timeout)
    if env_type == "docker":
        from tools.environments.docker import DockerEnvironment

        session_scoped = (
            _docker_session_isolation_enabled()
            and task_id != "default"
            and not (
                task_id
                and set(resolve_task_overrides(task_id))
                & {"docker_image", "modal_image", "singularity_image", "daytona_image", "env_type"}
            )
        )
        docker_env = DockerEnvironment(
            image=image,
            cwd=cwd,
            timeout=timeout,
            cpu=cpu,
            memory=memory,
            disk=disk,
            persistent_filesystem=persistent,
            task_id=task_id,
            volumes=container.get("docker_volumes", []),
            host_cwd=host_cwd,
            auto_mount_cwd=container.get(
                "docker_mount_cwd_to_workspace", False
            ),
            forward_env=container.get("docker_forward_env", []),
            env=container.get("docker_env", {}),
            run_as_host_user=container.get("docker_run_as_host_user", False),
            network=container.get("docker_network", True),
            extra_args=container.get("docker_extra_args", []),
            persist_across_processes=(
                False
                if session_scoped
                else container.get("docker_persist_across_processes", True)
            ),
            shm_size=container.get("docker_shm_size", "1g"),
        )
        if session_scoped:
            try:
                docker_env._session_scoped = True
            except AttributeError:
                pass
        return docker_env
    if env_type == "singularity":
        return SingularityEnvironment(
            image=image,
            cwd=cwd,
            timeout=timeout,
            cpu=cpu,
            memory=memory,
            disk=disk,
            persistent_filesystem=persistent,
            task_id=task_id,
        )
    if env_type == "ssh":
        if not ssh_config or not ssh_config.get("host") or not ssh_config.get("user"):
            raise ValueError(
                "SSH environment requires ssh_host and ssh_user to be configured"
            )
        from tools.environments.ssh import SSHEnvironment

        return SSHEnvironment(
            host=ssh_config["host"],
            user=ssh_config["user"],
            port=ssh_config.get("port", 22),
            key_path=ssh_config.get("key", ""),
            cwd=cwd,
            timeout=timeout,
        )
    if env_type == "managed_modal":
        from tools.environments.managed_modal import ManagedModalEnvironment

        sandbox_kwargs = {
            "cpu": cpu,
            "memory": memory,
            "ephemeral_disk": disk,
        }
        return ManagedModalEnvironment(
            image=image,
            cwd=cwd,
            timeout=timeout,
            modal_sandbox_kwargs=sandbox_kwargs,
            persistent_filesystem=persistent,
            task_id=task_id,
        )
    if env_type == "modal":
        from tools.environments.modal import ModalEnvironment

        sandbox_kwargs = {
            "cpu": cpu,
            "memory": memory,
            "ephemeral_disk": disk,
        }
        return ModalEnvironment(
            image=image,
            cwd=cwd,
            timeout=timeout,
            modal_sandbox_kwargs=sandbox_kwargs,
            persistent_filesystem=persistent,
            task_id=task_id,
        )
    if env_type == "vercel_sandbox":
        from tools.environments.vercel_sandbox import VercelSandboxEnvironment

        return VercelSandboxEnvironment(
            runtime=container.get("vercel_runtime") or None,
            cwd=cwd,
            timeout=timeout,
            cpu=cpu,
            memory=memory,
            disk=disk,
            persistent_filesystem=persistent,
            task_id=task_id,
        )
    if env_type == "daytona":
        from tools.environments.daytona import DaytonaEnvironment

        return DaytonaEnvironment(
            image=image,
            cwd=cwd,
            timeout=timeout,
            cpu=int(cpu),
            memory=memory,
            disk=disk,
            persistent_filesystem=persistent,
            task_id=task_id,
        )
    if env_type not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown environment type: {env_type}. Use 'local', 'docker', "
            "'singularity', 'modal', 'daytona', 'vercel_sandbox', or 'ssh'"
        )
    raise RuntimeError(
        f"TERMINAL_ENV={env_type!r} requires its native-async backend, "
        "which is not available in this build"
    )


def get_active_env(task_id: str):
    key = _resolve_container_task_id(task_id)
    with _env_lock:
        return _active_environments.get(key)


async def ensure_task_env(task_id: str | None = None):
    """Lazily create the selected sandbox for a task when it is non-local.

    Image and media tools may need a remote sandbox before the first terminal
    command.  Local paths intentionally remain host-side, so the local
    backend is a no-op.  Bring-up is best-effort: a failed native backend
    leaves the caller's existing fail-closed path intact.
    """
    config = await _get_env_config()
    if str(config.get("env_type") or "local") == "local":
        return None
    try:
        return await _get_or_create_environment(task_id)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - best-effort lazy bring-up
        effective_task_id = _resolve_container_task_id(task_id)
        logger.warning(
            "Lazy terminal environment init failed for task %s: %s",
            effective_task_id[:8],
            exc,
        )
        return None


def is_persistent_env(task_id: str) -> bool:
    env = get_active_env(task_id)
    if env is None:
        return False
    return bool(
        getattr(env, "_persistent", False)
        or getattr(env, "_session_scoped", False)
    )


def _get_creation_lock(task_id: str) -> asyncio.Lock:
    """Get the per-task async environment creation lock."""
    loop, profile = _current_terminal_scope_key()
    assert isinstance(loop, asyncio.AbstractEventLoop)
    with _creation_locks_lock:
        per_scope = _creation_locks.setdefault(loop, {}).setdefault(profile, {})
        return per_scope.setdefault(task_id, asyncio.Lock())


async def _get_or_create_environment(
    task_id: str | None = None,
) -> BaseEnvironment:
    """Get or create the selected environment without blocking I/O."""
    await _activate_terminal_scope()
    raw_key = _resolve_container_task_id(task_id)
    creation_lock = _get_creation_lock(raw_key)
    async with creation_lock:
        with _env_lock:
            env = _active_environments.get(raw_key)
            if env is not None:
                _last_activity[raw_key] = time.time()
                return env
        overrides = resolve_task_overrides(task_id)
        config = await _get_env_config()
        cwd = (
            overrides.get("cwd")
            or get_session_cwd(task_id)
            or get_session_cwd(raw_key)
        )
        env_type = str(overrides.get("env_type") or config["env_type"])
        image_env_type = env_type
        if env_type == "modal":
            modal_state = await _get_modal_backend_state(config.get("modal_mode"))
            selected = modal_state["selected_backend"]
            if selected == "managed":
                env_type = "managed_modal"
            elif selected != "direct":
                raise ValueError(
                    "Modal backend selected but no direct Modal credentials/config "
                    "or managed tool gateway was found."
                )
        image = str(
            overrides.get(f"{image_env_type}_image")
            or config.get(f"{image_env_type}_image")
            or config.get("docker_image")
            or ""
        )
        env = _create_environment(
            env_type,
            image,
            cwd or config["cwd"],
            int(config["timeout"]),
            ssh_config={
                "host": config.get("ssh_host"),
                "user": config.get("ssh_user"),
                "port": config.get("ssh_port"),
                "key": config.get("ssh_key"),
            },
            container_config=config,
            local_config=config,
            task_id=raw_key,
            host_cwd=_resolve_task_host_cwd(config, task_id),
        )
        await env._ensure_initialized()
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
        _start_cleanup_thread(int(config["lifetime_seconds"]))
        return env


async def _cleanup_inactive_envs(lifetime_seconds: int = 300) -> int:
    """Clean environments whose upstream idle lifetime has elapsed."""
    now = time.time()
    from tools.process_registry import process_registry

    with _env_lock:
        candidates = tuple(_last_activity.items())
    cleaned = 0
    for task_id, last_activity in candidates:
        if await process_registry.has_active_processes(task_id):
            with _env_lock:
                if task_id in _active_environments:
                    _last_activity[task_id] = now
            continue
        if now - last_activity <= lifetime_seconds:
            continue
        await cleanup_vm(task_id)
        cleaned += 1
    return cleaned


async def _cleanup_reaper(scope: _TerminalScopeKey) -> None:
    loop, profile = scope
    assert isinstance(loop, asyncio.AbstractEventLoop)
    tasks = _cleanup_tasks.setdefault(loop, {})
    lifetimes = _cleanup_lifetimes.setdefault(loop, {})
    try:
        lifetime = lifetimes.get(profile, 300)
        try:
            await _cleanup_inactive_envs(lifetime)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("Environment idle reaper failed", exc_info=True)
    finally:
        if tasks.get(profile) is asyncio.current_task():
            tasks.pop(profile, None)
        with _env_lock:
            scoped_values = getattr(_active_environments, "scoped_if_present", None)
            has_active = (
                bool(scoped_values(scope) or {})
                if scoped_values is not None
                else bool(_active_environments)
            )
        if has_active and profile in lifetimes:
            _schedule_cleanup_reaper(scope)
        else:
            lifetimes.pop(profile, None)


def _schedule_cleanup_reaper(scope: _TerminalScopeKey) -> None:
    """Schedule one idle sweep without retaining a sleeping asyncio task."""
    loop, profile = scope
    assert isinstance(loop, asyncio.AbstractEventLoop)
    handles = _cleanup_handles.setdefault(loop, {})
    existing = handles.get(profile)
    if existing is not None and not existing.cancelled():
        return

    def dispatch() -> None:
        handles.pop(profile, None)
        lifetimes = _cleanup_lifetimes.get(loop)
        if lifetimes is None or profile not in lifetimes:
            return
        task = asyncio.create_task(
            _cleanup_reaper(scope),
            name="terminal-environment-reaper",
        )
        _cleanup_tasks.setdefault(loop, {})[profile] = task

    handles[profile] = loop.call_later(60, dispatch)


def _start_cleanup_thread(lifetime_seconds: int = 300) -> None:
    """Start the upstream idle reaper as an event-loop task, never a thread."""
    scope = _current_terminal_scope_key()
    loop, profile = scope
    assert isinstance(loop, asyncio.AbstractEventLoop)
    _cleanup_lifetimes.setdefault(loop, {})[profile] = max(1, int(lifetime_seconds))
    tasks = _cleanup_tasks.get(loop)
    active = tasks.get(profile) if tasks is not None else None
    if active is not None and not active.done():
        return
    _schedule_cleanup_reaper(scope)


async def _stop_cleanup_thread() -> None:
    """Stop and await the native async idle reaper for the current loop."""
    await _activate_terminal_scope()
    loop, profile = _current_terminal_scope_key()
    assert isinstance(loop, asyncio.AbstractEventLoop)
    handles = _cleanup_handles.get(loop)
    handle = handles.pop(profile, None) if handles is not None else None
    if handles is not None and not handles:
        _cleanup_handles.pop(loop, None)
    if handle is not None:
        handle.cancel()
    tasks = _cleanup_tasks.get(loop)
    task = tasks.pop(profile, None) if tasks is not None else None
    if tasks is not None and not tasks:
        _cleanup_tasks.pop(loop, None)
    lifetimes = _cleanup_lifetimes.get(loop)
    if lifetimes is not None:
        lifetimes.pop(profile, None)
        if not lifetimes:
            _cleanup_lifetimes.pop(loop, None)
    if task is None or task is asyncio.current_task():
        return
    task.cancel()
    from tools.environments.file_sync import _await_owned

    async def reap() -> None:
        await asyncio.gather(task, return_exceptions=True)

    await _await_owned(asyncio.create_task(reap()))


async def cleanup_vm(task_id: str, *, force_remove: bool = False) -> None:
    """Terminate and reap native background commands for one task.

    Native asyncio subprocesses need an awaited lifecycle so they cannot outlive a
    closed agent or remain as zombies after their parent request is cancelled.
    """
    scope = await _activate_terminal_scope()
    loop, profile = scope
    assert isinstance(loop, asyncio.AbstractEventLoop)
    key = _resolve_container_task_id(task_id)
    from tools.process_registry import process_registry
    from tools.environments.file_sync import _await_owned

    async def _cleanup_owned() -> None:
        env: BaseEnvironment | None = None
        try:
            await process_registry.kill_all(
                key,
                source="agent.close",
                consume_output=False,
            )
        finally:
            with _env_lock:
                env = _active_environments.pop(key, None)
                _last_activity.pop(key, None)
            with _creation_locks_lock:
                per_loop = _creation_locks.get(loop)
                per_scope = per_loop.get(profile) if per_loop is not None else None
                if per_scope is not None:
                    per_scope.pop(key, None)
                    if not per_scope:
                        per_loop.pop(profile, None)
                        if not per_loop:
                            _creation_locks.pop(loop, None)

            try:
                from tools.file_tools import clear_file_ops_cache

                clear_file_ops_cache(key)
            except Exception:
                logger.debug(
                    "Unable to clear file-operations cache for %s",
                    key,
                    exc_info=True,
                )

            if env is not None:
                record_session_cwd(key, env.cwd)
                try:
                    import inspect

                    if "force_remove" in inspect.signature(env.cleanup).parameters:
                        await env.cleanup(force_remove=force_remove)
                    else:
                        await env.cleanup()
                    logger.info("Manually cleaned up environment for task: %s", key)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    error_str = str(exc)
                    if "404" in error_str or "not found" in error_str.lower():
                        logger.info("Environment for task %s already cleaned up", key)
                    else:
                        logger.warning(
                            "Error cleaning up environment for task %s: %s",
                            key,
                            exc,
                        )

    try:
        await _await_owned(asyncio.create_task(_cleanup_owned()))
    finally:
        with _env_lock:
            idle = not _active_environments
        if idle:
            await _stop_cleanup_thread()


async def cleanup_all_environments() -> int:  # noqa: ASYNC124 - lifecycle API
    """Clean up every active local environment and its tracked processes."""
    await _activate_terminal_scope()
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
    await _stop_cleanup_thread()
    return cleaned


def _resolve_command_cwd(
    *,
    workdir: str | None,
    default_cwd: str,
    session_key: str | None = None,
    env_type: str | None = None,
) -> str:
    """Resolve one command's cwd with container-host path protection."""
    if workdir:
        return workdir
    recorded = get_session_cwd(session_key)
    if (
        recorded
        and env_type in _CONTAINER_BACKENDS
        and _is_unusable_container_cwd(recorded)
    ):
        logger.info(
            "Ignoring recorded session cwd %r for %s backend; using %r",
            recorded,
            env_type,
            default_cwd,
        )
        return default_cwd
    return recorded or default_cwd




async def terminal_tool(  # noqa: ASYNC109 - upstream public API names timeout
    command: str,
    background: bool = False,
    timeout: int | None = None,  # noqa: ASYNC109 - upstream public API
    task_id: str | None = None,
    session_id: str | None = None,
    force: bool = False,
    workdir: str | None = None,
    pty: bool = False,
    notify_on_complete: bool = False,
    watch_patterns: list[str] | None = None,
) -> str:
    """Run a local command asynchronously and preserve Hermes' JSON result contract."""
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
    try:
        config = await _get_env_config()
        overrides = resolve_task_overrides(task_id)
        env_type = str(overrides.get("env_type") or config["env_type"])

        if timeout is not None and timeout <= 0:
            return tool_error(
                f"timeout must be a positive number of seconds (got {timeout})."
            )

        if (
            timeout is not None
            and not background
            and timeout > FOREGROUND_MAX_TIMEOUT
        ):
            return tool_error(
                f"Foreground timeout {timeout}s exceeds the maximum of "
                f"{FOREGROUND_MAX_TIMEOUT}s. Use background=true with "
                "notify_on_complete=true for long-running commands."
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

        try:
            env = await _get_or_create_environment(task_id)
        except ImportError as exc:
            return json.dumps(
                {
                    "output": "",
                    "exit_code": -1,
                    "error": _redact_terminal_error_text(
                        "Terminal tool disabled: environment creation failed "
                        f"({exc})"
                    ),
                    "status": "disabled",
                },
                ensure_ascii=False,
            )

        from tools.approval import get_current_session_key

        session_key = get_current_session_key(default="") or (task_id or "")
        cwd = _resolve_command_cwd(
            workdir=workdir,
            default_cwd=str(env.cwd),
            session_key=session_key or task_id,
            env_type=env_type,
        )
        # On Windows, rewriting the checkout backing this interpreter can
        # corrupt loaded files. POSIX keeps old inodes alive, so the guard is
        # platform-scoped and does not apply to remote sandboxes.
        if env_type == "local":
            from tools.self_repo_guard import (
                detect_self_repo_git_mutation,
                guard_active,
            )

            self_repo_hit, self_repo_message = (
                detect_self_repo_git_mutation(command, cwd)
                if guard_active()
                else (False, None)
            )
            if self_repo_hit:
                return json.dumps(
                    {
                        "output": "",
                        "exit_code": 1,
                        "error": self_repo_message,
                        "status": "blocked",
                    },
                    ensure_ascii=False,
                )
        if env_type == "local" and not workdir and not await _cwd_usable(cwd):
            recovered = await _resolve_safe_cwd(cwd)
            logger.warning(
                "Terminal working directory %s is missing or inaccessible; "
                "recovered to %s",
                cwd,
                recovered,
            )
            cwd = recovered
            env.cwd = recovered
            record_session_cwd(session_key, recovered)

        approval_note: str | None = None
        approved_run = bool(force)
        if not force:
            guard = await check_all_command_guards(
                command,
                env_type,
                approval_callback=_get_approval_callback(),
                has_host_access=_docker_has_host_access(config),
            )
            if not guard.get("approved", False):
                description = guard.get("description", "command flagged")
                fallback_message = (
                    f"Command denied: {description}. "
                    "Use the approval prompt to allow it, or rephrase the command."
                )
                return json.dumps(
                    {
                        "output": "",
                        "exit_code": -1,
                        "error": guard.get("message", fallback_message),
                        "status": "blocked",
                    },
                    ensure_ascii=False,
                )
            if guard.get("user_approved"):
                description = guard.get("description", "flagged as dangerous")
                approval_note = (
                    f"Command required approval ({description}) and was approved "
                    "by the user."
                )
                approved_run = True
            elif guard.get("smart_approved"):
                description = guard.get("description", "flagged as dangerous")
                approval_note = (
                    f"Command was flagged ({description}) and auto-approved by "
                    "smart approval."
                )

        if workdir:
            workdir_error = _validate_workdir(workdir)
            if workdir_error:
                logger.warning(
                    "Blocked dangerous workdir: %s (command: %s)",
                    workdir[:200],
                    _safe_command_preview(command),
                )
                return json.dumps(
                    {
                        "output": "",
                        "exit_code": -1,
                        "error": workdir_error,
                        "status": "blocked",
                    },
                    ensure_ascii=False,
                )
        if background:
            from tools.process_registry import process_registry

            effective_pty = pty and not _command_requires_pipe_stdin(command)
            pty_disabled_reason = None
            if pty and not effective_pty:
                pty_disabled_reason = (
                    "PTY disabled for this command because it expects piped stdin/EOF "
                    "(for example gh auth login --with-token). For local background "
                    "processes, call process(action='close') after writing so it "
                    "receives EOF."
                )
            try:
                if env_type == "local":
                    background_env = dict(env.env)
                    if getattr(env, "_snapshot_ready", False):
                        background_env["_HERMES_SESSION_SNAPSHOT_PATH"] = (
                            env._snapshot_path
                        )
                    process_session = await process_registry.spawn_local(
                        command=command,
                        cwd=cwd,
                        task_id=_resolve_container_task_id(task_id),
                        session_key=session_key,
                        env_vars=background_env,
                        use_pty=effective_pty,
                    )
                else:
                    process_session = await process_registry.spawn_via_env(
                        env=env,
                        command=command,
                        cwd=cwd,
                        task_id=_resolve_container_task_id(task_id),
                        session_key=session_key,
                    )
                payload = {
                    "output": "Background process started",
                    "session_id": process_session.id,
                    "pid": process_session.pid,
                    "exit_code": 0,
                    "error": None,
                }
                if approval_note:
                    payload["approval"] = approval_note
                if pty_disabled_reason:
                    payload["pty_note"] = pty_disabled_reason
                if not notify_on_complete and not watch_patterns:
                    payload["hint"] = (
                        "background=true without notify_on_complete=true means "
                        "this process runs SILENTLY — you will not be told when "
                        "it exits. If this is a bounded task (test suite, build, "
                        "CI poller, deploy, anything with a defined end), you "
                        "almost certainly wanted notify_on_complete=true so the "
                        "system pings you on exit. Re-launch with "
                        "notify_on_complete=true, or call process(action='poll') "
                        "/ process(action='wait') yourself to learn the outcome. "
                        "Only ignore this hint for genuine long-lived processes "
                        "that never exit (servers, watchers, daemons)."
                    )
                watch_patterns, conflict_note = (
                    _resolve_notification_flag_conflict(
                        notify_on_complete=bool(notify_on_complete),
                        watch_patterns=watch_patterns,
                        background=True,
                    )
                )
                if conflict_note:
                    payload["watch_patterns_ignored"] = conflict_note
                if notify_on_complete:
                    process_session.notify_on_complete = True
                    payload["notify_on_complete"] = True
                    process_registry.pending_watchers.append(
                        {
                            "session_id": process_session.id,
                            "check_interval": 5,
                            "session_key": process_session.session_key,
                            "notify_on_complete": True,
                        }
                    )
                elif watch_patterns:
                    process_session.watch_patterns = list(watch_patterns)
                    payload["watch_patterns"] = process_session.watch_patterns
                if notify_on_complete or process_session.watch_patterns:
                    if process_session.exited and notify_on_complete:
                        process_registry._enqueue_completion(process_session)
                    await process_registry._write_checkpoint()
                return json.dumps(payload, ensure_ascii=False)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                return json.dumps(
                    {
                        "output": "",
                        "exit_code": -1,
                        "error": _redact_terminal_error_text(
                            f"Failed to start background process: {exc}"
                        ),
                    },
                    ensure_ascii=False,
                )

        starting_cwd = env.cwd
        effective_timeout = (
            timeout
            if timeout is not None
            else int(config.get("timeout", getattr(env, "timeout", 180)))
        )
        max_retries = 3
        retry_count = 0

        # An interrupt that landed while an approved command was waiting for
        # consent must not kill the just-approved run. Clear once before the
        # retry loop; a real interrupt during backoff remains observable.
        if approved_run:
            _interrupt_bootstrap.clear_current_thread_interrupt()

        while True:
            try:
                result = await env.execute(
                    command,
                    cwd=cwd,
                    timeout=effective_timeout,
                    # This is the model-facing foreground path. Bound retention while
                    # the backend drains the command stream; internal environment
                    # consumers intentionally keep BaseEnvironment's False default.
                    bounded_capture=True,
                )
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if "timeout" in str(exc).lower():
                    return json.dumps(
                        {
                            "output": "",
                            "exit_code": 124,
                            "error": (
                                f"Command timed out after {effective_timeout} seconds"
                            ),
                        },
                        ensure_ascii=False,
                    )
                if retry_count >= max_retries:
                    logger.error(
                        "Execution failed after %d retries - Command: %s - "
                        "Error: %s: %s - Task: %s, Backend: %s",
                        max_retries,
                        _safe_command_preview(command),
                        type(exc).__name__,
                        exc,
                        task_id,
                        env_type,
                    )
                    return json.dumps(
                        {
                            "output": "",
                            "exit_code": -1,
                            "error": _redact_terminal_error_text(
                                "Command execution failed: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                        },
                        ensure_ascii=False,
                    )
                retry_count += 1
                wait_time = 2**retry_count
                logger.warning(
                    "Execution error, retrying in %ds (attempt %d/%d) - "
                    "Command: %s - Error: %s: %s - Task: %s, Backend: %s",
                    wait_time,
                    retry_count,
                    max_retries,
                    _safe_command_preview(command),
                    type(exc).__name__,
                    exc,
                    task_id,
                    env_type,
                )
                await asyncio.sleep(wait_time)
        spill_file_path = result.get("full_output_path")
        try:
            post_cwd = getattr(env, "cwd", None)
            if workdir:
                # ``workdir`` applies to one command only. LocalEnvironment tracks
                # the shell's final cwd, so restore the durable session cwd after
                # a transient override.
                env.cwd = starting_cwd
            else:
                record_session_cwd(session_key, post_cwd)
            exit_code = int(result.get("returncode", 0))
            raw_output = str(result.get("output", ""))

            sudo_auth_failed = _sudo_wrong_password_failure(raw_output)

            # Plugins see the bounded transport result before ANSI stripping,
            # redaction, and the final cap. The first string replacement wins.
            try:
                hook_results = await _lifecycle_bootstrap.invoke_hook(
                    "transform_terminal_output",
                    command=command,
                    output=raw_output,
                    returncode=exit_code,
                    task_id=task_id or "",
                    env_type=env_type,
                )
                for hook_result in hook_results:
                    if isinstance(hook_result, str):
                        raw_output = hook_result
                        break
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

            from tools.tool_output_limits import _refresh_tool_output_limits

            await _refresh_tool_output_limits()
            output, truncation = await _prepare_terminal_output(
                raw_output,
                command,
                spill_total_chars=result.get("output_total_chars"),
                spill_file_path=spill_file_path,
            )
        except BaseException:
            await _discard_terminal_spill_before_reraise(spill_file_path)
            raise
        payload: dict[str, Any] = {
            "output": output,
            "exit_code": exit_code,
            "error": None,
        }
        if post_cwd and cwd:
            try:
                realpath = aiofiles.os.wrap(os.path.realpath)
                if await realpath(str(post_cwd)) != await realpath(str(cwd)):
                    payload["cwd"] = str(post_cwd)
            except Exception:
                pass
        payload.update(truncation)
        exit_note = _interpret_exit_code(command, exit_code)
        if exit_note:
            payload["exit_code_meaning"] = exit_note
        elif exit_code != 0:
            try:
                from tools.terminal_hints import annotate_failure

                failure_hint = annotate_failure(command, exit_code, output)
            except Exception:
                failure_hint = None
            if failure_hint:
                payload["hint"] = failure_hint
        if approval_note:
            if exit_code == 130 and "[Command interrupted]" in output:
                payload["approval"] = (
                    approval_note.rstrip(".") + ", then interrupted."
                )
            else:
                payload["approval"] = approval_note
        if sudo_auth_failed:
            payload["sudo_auth_failed"] = True
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
    except EnvironmentConnectionError as exc:
        degraded_mode = os.getenv("TERMINAL_DEGRADED_MODE", "warn").strip().lower()
        if degraded_mode == "fail":
            import traceback

            traceback_text = traceback.format_exc()
            logger.error("terminal_tool exception:\n%s", traceback_text)
            return json.dumps(
                {
                    "output": "",
                    "exit_code": -1,
                    "error": _redact_terminal_error_text(
                        f"Failed to execute command: {exc}"
                    ),
                    "traceback": _redact_terminal_error_text(traceback_text),
                    "status": "error",
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "output": "",
                "exit_code": -1,
                "status": "degraded",
                "reason": exc.reason,
                "retry_hint": exc.retry_hint,
                "error": f"Terminal backend degraded: {exc.reason}",
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        import traceback

        traceback_text = traceback.format_exc()
        logger.error("terminal_tool exception:\n%s", traceback_text)
        return json.dumps(
            {
                "output": "",
                "exit_code": -1,
                "error": _redact_terminal_error_text(
                    f"Failed to execute command: {exc}"
                ),
                "traceback": _redact_terminal_error_text(traceback_text),
                "status": "error",
            },
            ensure_ascii=False,
        )


async def _prepare_terminal_output(
    output: str,
    command: str,
    *,
    spill_total_chars: Any = None,
    spill_file_path: Any = None,
) -> tuple[str, dict[str, Any]]:
    """Finalize upstream's bounded raw capture for the model-facing payload.

    The transport collector owns the first head/tail cap and reports the raw
    stream size plus spill path.  Keep that ordering: apply the same final cap
    to a backend or plugin replacement that is still oversized, then strip
    ANSI from both the visible window and the recoverable spill.
    """
    from agent.redact import redact_terminal_output
    from hermes_constants import get_hermes_home
    from tools.ansi_strip import strip_ansi
    from tools.tool_output_limits import get_max_bytes

    limit = get_max_bytes()
    metadata: dict[str, Any] = {}
    raw_total_chars = (
        spill_total_chars
        if isinstance(spill_total_chars, int) and spill_total_chars >= 0
        else None
    )
    spill_path = (
        get_hermes_home() / str(spill_file_path)
        if spill_file_path and not os.path.isabs(str(spill_file_path))
        else spill_file_path
    )

    # Non-streaming SDK backends can still return an unbounded replacement.
    # Preserve the upstream final safety cap and make that full output
    # recoverable, while streaming backends normally arrive with spill_path.
    if len(output) > limit and not spill_path:
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
            raw_total_chars = len(output)
            redacted_full = redact_terminal_output(strip_ansi(output), command)
            async with aiofiles.open(spill_path, "w", encoding="utf-8") as handle:
                await handle.write(redacted_full)
                await handle.flush()
                await aiofiles.os.wrap(os.fsync)(handle.fileno())
            await aiofiles.os.wrap(os.chmod)(spill_path, 0o600)
        except Exception:
            spill_path = None
            logger.debug("Unable to persist truncated terminal output", exc_info=True)

    if len(output) > limit:
        head_chars = int(limit * 0.4)
        tail_chars = limit - head_chars
        omitted = len(output) - head_chars - tail_chars
        notice = (
            f"\n\n... [OUTPUT TRUNCATED - {omitted:,} chars omitted "
            f"out of {len(output):,} total] ...\n\n"
        )
        output = output[:head_chars] + notice + output[-tail_chars:]

    if spill_path:
        try:
            async with aiofiles.open(
                spill_path,
                encoding="utf-8",
                errors="replace",
            ) as handle:
                raw_spill = await handle.read()
            sanitized_spill = redact_terminal_output(strip_ansi(raw_spill), command)
            from tools.spill_safety import _write_text_exclusive_async

            await _write_text_exclusive_async(
                spill_path,
                sanitized_spill,
                private=True,
                overwrite=True,
                encoding="utf-8",
                errors="replace",
            )
            await aiofiles.os.wrap(os.chmod)(spill_path, 0o600)
            total = raw_total_chars if raw_total_chars is not None else len(raw_spill)
            metadata = {
                "output_total_chars": total,
                "full_output_path": str(spill_path),
                "truncation_note": (
                    "Output exceeded the capture window (head+tail shown). "
                    f"Full output ({total:,} chars) saved to {spill_path} — "
                    "search it with search_files or page it with read_file instead of "
                    "re-running the command."
                ),
            }
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("spill redaction failed; dropping spill handle", exc_info=True)
            try:
                await aiofiles.os.remove(spill_path)
            except OSError:
                pass

    clean_output = strip_ansi(output)
    return redact_terminal_output(clean_output.strip(), command), metadata


# Signal-death notes for the lethal signals seen in practice. Keyed by
# signum; used for both the ``-signum`` (subprocess) and ``128+signum``
# (shell) encodings. Curated rather than exhaustive so we never mislabel a
# legitimate application exit code (e.g. 130/SIGINT is handled by the
# executor's interrupt-marker path and excluded here).
_SIGNAL_EXIT_NOTES: dict[int, str] = {
    3: "SIGQUIT (quit from keyboard)",
    4: "SIGILL (illegal instruction — corrupt binary or wrong architecture)",
    6: "SIGABRT (abort — assertion failure, fatal runtime error, or glibc abort)",
    7: "SIGBUS (bus error — misaligned or unmapped memory access)",
    8: "SIGFPE (fatal arithmetic error, e.g. integer division by zero)",
    9: "SIGKILL — often the kernel OOM killer on memory exhaustion, or an explicit kill -9",
    11: "SIGSEGV (segmentation fault — the program crashed)",
    13: "SIGPIPE (wrote to a closed pipe — e.g. output piped to a reader that exited)",
    15: "SIGTERM (terminated — kill/timeout or shutdown requested it to stop)",
    24: "SIGXCPU (CPU time limit exceeded)",
    25: "SIGXFSZ (file size limit exceeded)",
}


def _interpret_signal_exit(exit_code: int) -> str | None:
    """Map signal-termination exit codes to a human-readable note."""
    if exit_code < 0:
        signum = -exit_code
        if signum == 2:  # SIGINT is owned by the interrupt-marker path.
            return None
        note = _SIGNAL_EXIT_NOTES.get(signum)
        if note:
            return f"Command terminated by signal {signum}: {note}"
        try:
            import signal as _signal

            name = _signal.Signals(signum).name
        except (ValueError, ImportError):
            name = f"signal {signum}"
        return f"Command terminated by {name} (signal {signum})"
    if exit_code > 128:
        signum = exit_code - 128
        note = _SIGNAL_EXIT_NOTES.get(signum)
        if note:
            return (
                f"Exit code {exit_code} usually means the command was terminated "
                f"by signal {signum}: {note}"
            )
    return None


def _interpret_exit_code(command: str, exit_code: int) -> str | None:
    """Return a note when a non-zero exit code is conventional, not erroneous."""
    if exit_code == 0:
        return None
    signal_note = _interpret_signal_exit(exit_code)
    if signal_note is not None:
        return signal_note
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


_SUDO_WRONG_PASSWORD_MARKERS = (
    "sudo: authentication failed",
    "sudo: incorrect password attempt",
    "sudo: maximum 3 incorrect authentication attempts",
    "sudo: 3 incorrect password attempts",
)


def _sudo_wrong_password_failure(output: str) -> bool:
    """Return True when sudo rejected a password supplied on stdin."""
    if not output:
        return False
    lowered = output.lower()
    return any(marker in lowered for marker in _SUDO_WRONG_PASSWORD_MARKERS)


async def check_terminal_requirements() -> bool:
    """Check if all requirements for the configured terminal backend are met."""
    try:
        config = await _get_env_config()
        env_type = config["env_type"]
        if env_type == "local":
            return True
        if env_type == "docker":
            from tools.environments.docker import (
                _run_docker_command,
                find_docker,
            )

            executable = await find_docker()
            if not executable:
                logger.error(
                    "Docker executable not found in PATH or common install locations"
                )
                return False
            result = await _run_docker_command(
                [executable, "version"],
                timeout=5,
            )
            return result.returncode == 0
        if env_type == "singularity":
            from tools.environments.singularity import (
                _ensure_singularity_available,
            )

            try:
                await _ensure_singularity_available()
            except (OSError, RuntimeError, TimeoutError):
                return False
            return True
        if env_type == "ssh":
            if not config.get("ssh_host") or not config.get("ssh_user"):
                logger.error(
                    "SSH backend selected but TERMINAL_SSH_HOST and "
                    "TERMINAL_SSH_USER are not both set. Configure both or "
                    "switch TERMINAL_ENV to 'local'."
                )
                return False
            return True
        if env_type == "modal":
            modal_state = await _get_modal_backend_state(config.get("modal_mode"))
            if modal_state["selected_backend"] == "managed":
                return True
            if modal_state["selected_backend"] != "direct":
                logger.error(
                    "Modal backend selected but no usable direct or managed "
                    "backend was found."
                )
                return False
            from hermes_cli.async_source_loader import _locate_source_module

            if await _locate_source_module("modal") is None:
                logger.error(
                    "modal is required for direct modal terminal backend: "
                    "pip install modal"
                )
                return False
            return True
        if env_type == "vercel_sandbox":
            return await _check_vercel_sandbox_requirements(config)
        if env_type == "daytona":
            from agent.secret_scope import get_secret
            from hermes_cli.async_source_loader import _locate_source_module

            if await _locate_source_module("daytona") is None:
                return False
            return get_secret("DAYTONA_API_KEY") is not None
        logger.error(
            "Unknown TERMINAL_ENV '%s'. Use one of: local, docker, singularity, "
            "modal, daytona, vercel_sandbox, ssh.",
            env_type,
        )
        return False
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("Terminal requirements check failed: %s", exc, exc_info=True)
        return False


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
                "description": (
                    f"Max seconds to wait (default: 180, foreground max: "
                    f"{FOREGROUND_MAX_TIMEOUT}). Returns INSTANTLY when command "
                    "finishes — set high for long tasks, you won't wait unnecessarily. "
                    f"Foreground timeout above {FOREGROUND_MAX_TIMEOUT}s is rejected; "
                    "use background=true for longer commands."
                ),
                "minimum": 1,
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
                    "like Codex, Claude Code, or Python REPL. Only works with "
                    "local and SSH backends. Default: false."
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


async def _handle_terminal(args, **kw):
    return await terminal_tool(
        command=args.get("command"),
        background=args.get("background", False),
        timeout=args.get("timeout"),
        task_id=kw.get("task_id"),
        session_id=kw.get("session_id"),
        workdir=args.get("workdir"),
        pty=args.get("pty", False),
        notify_on_complete=args.get("notify_on_complete", False),
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
