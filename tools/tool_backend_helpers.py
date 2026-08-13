"""Native-async backend selection helpers used by retained provider tools."""

from __future__ import annotations

import os
from typing import Any

import aiofiles.os

from agent.secret_scope import get_secret, is_multiplex_active
from utils import is_truthy_value

_DEFAULT_BROWSER_PROVIDER = "local"
_DEFAULT_MODAL_MODE = "auto"
_VALID_MODAL_MODES = {"auto", "direct", "managed"}


async def resolve_provider_secret(
    env_var: str,
    provider_id: str,
    config_value: str = "",
    env_getter=None,
) -> str:
    """Resolve a voice-provider key: config, scoped env/dotenv, then pool."""
    value = str(config_value or "").strip()
    if value:
        return value

    try:
        from agent.secret_scope import get_secret, is_multiplex_active

        value = str(get_secret(env_var, "") or "").strip()
        if value:
            return value
        if is_multiplex_active():
            return ""
    except Exception:
        try:
            from agent.secret_scope import is_multiplex_active

            if is_multiplex_active():
                return ""
        except Exception:
            pass
        value = str(os.getenv(env_var, "") or "").strip()
        if value:
            return value

    try:
        if env_getter is not None:
            value = env_getter(env_var)
            if hasattr(value, "__await__"):
                value = await value
        else:
            from hermes_cli.config import get_env_value_prefer_dotenv

            value = await get_env_value_prefer_dotenv(env_var)
        value = str(value or "").strip()
    except Exception:
        value = ""
    if value or not provider_id:
        return value

    try:
        from agent.credential_pool import load_pool

        for pool_key in (provider_id, f"custom:{provider_id}"):
            pool = await load_pool(pool_key)
            if pool is None or not pool.has_credentials():
                continue
            entry = await pool.peek()
            if entry is None:
                continue
            value = str(
                getattr(entry, "runtime_api_key", "")
                or getattr(entry, "access_token", "")
                or ""
            ).strip()
            if value:
                return value
    except Exception:
        return ""
    return ""


async def resolve_openai_audio_api_key() -> str:
    """Prefer the voice-tools override, then the standard OpenAI pool key."""
    return (
        await resolve_provider_secret("VOICE_TOOLS_OPENAI_KEY", "")
        or await resolve_provider_secret("OPENAI_API_KEY", "openai-api")
    )


async def managed_nous_tools_enabled(*, force_fresh: bool = False) -> bool:
    """Return whether the active Nous account may use the Tool Gateway."""
    try:
        from hermes_cli.nous_account import get_nous_portal_account_info

        account_info = await get_nous_portal_account_info(
            force_fresh=force_fresh,
        )
        if not account_info.logged_in:
            return False
        return account_info.tool_gateway_entitled
    except Exception:
        return False


async def nous_tool_gateway_unavailable_message(
    capability: str = "the Nous Tool Gateway",
    *,
    force_fresh: bool = False,
) -> str:
    """Return account-aware guidance for an unavailable managed tool."""
    try:
        from hermes_cli.nous_account import (
            format_nous_portal_entitlement_message,
            get_nous_portal_account_info,
        )

        account_info = await get_nous_portal_account_info(
            force_fresh=force_fresh,
        )
        message = format_nous_portal_entitlement_message(
            account_info,
            capability=capability,
        )
        if message:
            return message
    except Exception:
        pass
    return (
        f"{capability} is unavailable. Run `hermes model` to refresh your "
        "Nous Portal login and billing status."
    )


def coerce_modal_mode(value: object | None) -> str:
    """Return the requested Modal mode when valid, else the default."""
    mode = str(value or _DEFAULT_MODAL_MODE).strip().lower()
    return mode if mode in _VALID_MODAL_MODES else _DEFAULT_MODAL_MODE


def normalize_modal_mode(value: object | None) -> str:
    return coerce_modal_mode(value)


async def has_direct_modal_credentials() -> bool:
    """Return True when direct Modal credentials/config are available."""
    if get_secret("MODAL_TOKEN_ID") and get_secret("MODAL_TOKEN_SECRET"):
        return True
    if is_multiplex_active():
        # ~/.modal.toml is OS-user global and cannot be attributed to the
        # active Hermes profile. Direct Modal itself rejects this fallback in
        # multiplex mode, so do not advertise an unusable foreign account.
        return False
    try:
        home = await aiofiles.os.wrap(os.path.expanduser)("~")
        return await aiofiles.os.path.exists(os.path.join(home, ".modal.toml"))
    except (PermissionError, OSError):
        return False


def resolve_modal_backend_state(
    modal_mode: object | None,
    *,
    has_direct: bool,
    managed_ready: bool,
    managed_enabled: bool | None = None,
) -> dict[str, Any]:
    """Resolve direct vs managed Modal backend selection."""
    requested_mode = coerce_modal_mode(modal_mode)
    if managed_enabled is None:
        managed_enabled = managed_ready
    managed_mode_blocked = requested_mode == "managed" and not managed_enabled
    if requested_mode == "managed":
        selected = "managed" if managed_enabled and managed_ready else None
    elif requested_mode == "direct":
        selected = "direct" if has_direct else None
    else:
        selected = (
            "managed"
            if managed_enabled and managed_ready
            else "direct"
            if has_direct
            else None
        )
    return {
        "requested_mode": requested_mode,
        "mode": requested_mode,
        "has_direct": has_direct,
        "managed_ready": managed_ready,
        "managed_mode_blocked": managed_mode_blocked,
        "selected_backend": selected,
    }


def normalize_browser_cloud_provider(value: object | None) -> str:
    """Return a normalized browser provider key."""
    provider = str(value or _DEFAULT_BROWSER_PROVIDER).strip().lower()
    return provider or _DEFAULT_BROWSER_PROVIDER


async def prefers_gateway(config_section: str) -> bool:
    """Return whether ``<section>.use_gateway`` is enabled in config."""
    try:
        from hermes_cli.config import load_config_readonly

        config = await load_config_readonly()
        section = (config or {}).get(config_section)
        if isinstance(section, dict):
            return is_truthy_value(section.get("use_gateway"), default=False)
    except Exception:
        pass
    return False


async def fal_key_is_configured() -> bool:
    """Return whether FAL_KEY exists in the active secret scope or dotenv."""
    value: Any = None
    try:
        from agent.secret_scope import get_secret

        value = get_secret("FAL_KEY", "")
    except Exception:
        pass
    if not str(value or "").strip():
        try:
            from agent.secret_scope import is_multiplex_active

            if is_multiplex_active():
                return False
            from hermes_cli.config import get_env_value_prefer_dotenv

            value = await get_env_value_prefer_dotenv("FAL_KEY")
        except Exception:
            value = None
    return bool(str(value or "").strip())
