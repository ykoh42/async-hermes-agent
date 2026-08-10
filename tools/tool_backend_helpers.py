"""Native-async backend selection helpers used by retained provider tools."""

from __future__ import annotations

from typing import Any

from utils import is_truthy_value

_DEFAULT_BROWSER_PROVIDER = "local"


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
