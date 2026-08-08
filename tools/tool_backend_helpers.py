"""Native-async backend selection helpers used by retained provider tools."""

from __future__ import annotations

from typing import Any

_DEFAULT_BROWSER_PROVIDER = "local"


def normalize_browser_cloud_provider(value: object | None) -> str:
    """Return a normalized browser provider key."""
    provider = str(value or _DEFAULT_BROWSER_PROVIDER).strip().lower()
    return provider or _DEFAULT_BROWSER_PROVIDER


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
