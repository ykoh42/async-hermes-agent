"""Native-async resolution for Nous-hosted vendor passthroughs."""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Callable, Optional

from tools.tool_backend_helpers import managed_nous_tools_enabled

logger = logging.getLogger(__name__)

_DEFAULT_TOOL_GATEWAY_DOMAIN = "nousresearch.com"
_DEFAULT_TOOL_GATEWAY_SCHEME = "https"
_NOUS_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120


@dataclass(frozen=True)
class ManagedToolGatewayConfig:
    vendor: str
    gateway_origin: str
    nous_user_token: str
    managed_mode: bool


async def _read_user_token_override() -> Optional[str]:
    try:
        from agent.secret_scope import UnscopedSecretError, get_secret

        try:
            value = get_secret("TOOL_GATEWAY_USER_TOKEN")
        except UnscopedSecretError:
            from hermes_cli.config import get_env_value_prefer_dotenv

            value = await get_env_value_prefer_dotenv("TOOL_GATEWAY_USER_TOKEN")
    except Exception:
        try:
            from hermes_cli.config import get_env_value_prefer_dotenv

            value = await get_env_value_prefer_dotenv("TOOL_GATEWAY_USER_TOKEN")
        except Exception:
            value = ""
    if not str(value or "").strip():
        try:
            from agent.secret_scope import is_multiplex_active

            if is_multiplex_active():
                return None
            from hermes_cli.config import get_env_value_prefer_dotenv

            value = await get_env_value_prefer_dotenv("TOOL_GATEWAY_USER_TOKEN")
        except Exception:
            value = ""
    return str(value or "").strip() or None


async def read_nous_access_token() -> Optional[str]:
    """Resolve a current Nous OAuth token or explicit gateway override."""
    explicit = await _read_user_token_override()
    if explicit:
        return explicit
    cached_token = await peek_nous_access_token()
    try:
        from hermes_cli.auth import resolve_nous_access_token

        value = await resolve_nous_access_token(
            refresh_skew_seconds=_NOUS_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
        )
        return str(value).strip() or None
    except Exception as exc:
        logger.debug("Nous access token refresh failed: %s", exc)
        return cached_token


async def peek_nous_access_token() -> Optional[str]:
    """Return a cached Nous token without refreshing it over the network."""
    explicit = await _read_user_token_override()
    if explicit:
        return explicit
    try:
        from hermes_cli.auth import get_provider_auth_state

        state = await get_provider_auth_state("nous") or {}
        value = state.get("access_token")
        return str(value).strip() or None
    except Exception:
        return None


def get_tool_gateway_scheme() -> str:
    scheme = os.getenv("TOOL_GATEWAY_SCHEME", "").strip().lower()
    if not scheme:
        return _DEFAULT_TOOL_GATEWAY_SCHEME
    if scheme in {"http", "https"}:
        return scheme
    raise ValueError("TOOL_GATEWAY_SCHEME must be 'http' or 'https'")


def build_vendor_gateway_url(vendor: str) -> str:
    vendor_key = f"{vendor.upper().replace('-', '_')}_GATEWAY_URL"
    explicit = os.getenv(vendor_key, "").strip().rstrip("/")
    if explicit:
        return explicit
    domain = os.getenv("TOOL_GATEWAY_DOMAIN", "").strip().strip("/")
    return f"{get_tool_gateway_scheme()}://{vendor}-gateway.{domain or _DEFAULT_TOOL_GATEWAY_DOMAIN}"


async def resolve_managed_tool_gateway(
    vendor: str,
    gateway_builder: Optional[Callable[[str], str]] = None,
    token_reader=None,
) -> Optional[ManagedToolGatewayConfig]:
    """Resolve managed gateway configuration after entitlement and token checks."""
    if not await managed_nous_tools_enabled():
        return None
    builder = gateway_builder or build_vendor_gateway_url
    reader = token_reader or read_nous_access_token
    gateway_origin = builder(vendor)
    token = await reader()
    if not gateway_origin or not token:
        return None
    return ManagedToolGatewayConfig(
        vendor=vendor,
        gateway_origin=gateway_origin,
        nous_user_token=token,
        managed_mode=True,
    )


async def is_managed_tool_gateway_ready(
    vendor: str,
    gateway_builder: Optional[Callable[[str], str]] = None,
    token_reader=None,
) -> bool:
    """Return whether a managed gateway can be resolved without token refresh."""
    return (
        await resolve_managed_tool_gateway(
            vendor,
            gateway_builder=gateway_builder,
            token_reader=token_reader or peek_nous_access_token,
        )
        is not None
    )
