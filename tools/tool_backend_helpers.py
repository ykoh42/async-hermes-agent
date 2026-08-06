"""Native-async backend selection helpers used by retained provider tools."""

from __future__ import annotations

from typing import Any

from utils import is_truthy_value


async def managed_nous_tools_enabled(*, force_fresh: bool = False) -> bool:
    """Return whether the current Nous account can use a managed tool gateway."""
    try:
        from hermes_cli.nous_account import get_nous_portal_account_info

        info = await get_nous_portal_account_info(force_fresh=force_fresh)
        return bool(info.logged_in and info.tool_gateway_entitled)
    except Exception:
        return False


async def nous_tool_gateway_unavailable_message(
    capability: str = "the Nous Tool Gateway",
    *,
    force_fresh: bool = False,
) -> str:
    """Return account-aware guidance for an unavailable managed tool path."""
    try:
        from hermes_cli.nous_account import (
            format_nous_portal_entitlement_message,
            get_nous_portal_account_info,
        )

        account_info = await get_nous_portal_account_info(force_fresh=force_fresh)
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


async def prefers_gateway(config_section: str) -> bool:
    """Return the ``<section>.use_gateway`` preference from config.yaml."""
    try:
        from hermes_cli.config import load_config_readonly

        section = (await load_config_readonly()).get(config_section)
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
            from hermes_cli.config import get_env_value_prefer_dotenv

            value = await get_env_value_prefer_dotenv("FAL_KEY")
        except Exception:
            value = None
    return bool(str(value or "").strip())
