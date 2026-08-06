"""Pure environment helpers for the native asyncio subprocess backend."""

from __future__ import annotations

import os
import platform
import re
from collections.abc import Mapping


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


def _sanitize_subprocess_env(
    base_env: Mapping[str, str] | None,
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Strip Hermes inference credentials from a model-driven subprocess."""
    try:
        from tools.env_passthrough import is_env_passthrough
    except ImportError:
        is_env_passthrough = lambda _name: False

    sanitized: dict[str, str] = {}
    for key, value in dict(base_env or {}).items():
        if key.startswith(_HERMES_PROVIDER_ENV_FORCE_PREFIX):
            continue
        if _is_hermes_internal_secret(key):
            continue
        if key not in _HERMES_PROVIDER_ENV_BLOCKLIST or is_env_passthrough(key):
            sanitized[key] = value
    for key, value in dict(extra_env or {}).items():
        if key.startswith(_HERMES_PROVIDER_ENV_FORCE_PREFIX):
            key = key[len(_HERMES_PROVIDER_ENV_FORCE_PREFIX):]
        if _is_hermes_internal_secret(key):
            continue
        if key not in _HERMES_PROVIDER_ENV_BLOCKLIST or is_env_passthrough(key):
            sanitized[key] = value

    try:
        from hermes_constants import apply_subprocess_home_env, get_hermes_home_override

        override = get_hermes_home_override()
        if override:
            sanitized["HERMES_HOME"] = override
        apply_subprocess_home_env(sanitized)
    except Exception:
        pass
    for marker in _ACTIVE_VENV_MARKER_VARS:
        sanitized.pop(marker, None)
    if _IS_WINDOWS:
        sanitized.setdefault("PYTHONUTF8", "1")
    return sanitized


def build_subprocess_env(
    base: Mapping[str, str] | None = None,
    *,
    inherit_profile_home: bool = True,
    scrub_secrets: bool = True,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the environment for ``asyncio.create_subprocess_exec``."""
    source = dict(base) if base is not None else os.environ.copy()
    if scrub_secrets:
        return _sanitize_subprocess_env(source, extra)
    if inherit_profile_home:
        try:
            from hermes_constants import (
                apply_subprocess_home_env,
                get_hermes_home_override,
            )

            override = get_hermes_home_override()
            if override:
                source["HERMES_HOME"] = override
            apply_subprocess_home_env(source)
        except Exception:
            pass
    if extra:
        source.update(extra)
    return source


def hermes_subprocess_env(*, inherit_credentials: bool = False) -> dict[str, str]:
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
        from hermes_constants import apply_subprocess_home_env, get_hermes_home_override

        override = get_hermes_home_override()
        if override:
            env["HERMES_HOME"] = override
        apply_subprocess_home_env(env)
    except Exception:
        pass
    for marker in _ACTIVE_VENV_MARKER_VARS:
        env.pop(marker, None)
    return env
