"""Disabled compatibility boundary for Hermes-managed tool gateways.

The training runtime uses configured external MCP servers instead of the
vendor-managed browser, media, and sandbox gateway services.  Terminal
environment imports retain this tiny module so selecting the removed managed
Modal backend fails cleanly rather than breaking agent startup.
"""

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class ManagedToolGatewayConfig:
    gateway_origin: str
    nous_user_token: str


def build_vendor_gateway_url(vendor: str) -> str:
    """Return the conventional vendor URL for compatibility callers.

    The training checkout never enables managed gateways, but web-provider
    modules import this helper while deciding whether a direct provider is
    configured. Keeping the pure URL helper avoids making those imports
    conditional without re-enabling the managed transport.
    """
    import os

    explicit = os.getenv(f"{str(vendor).upper().replace('-', '_')}_GATEWAY_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    domain = os.getenv("TOOL_GATEWAY_DOMAIN", "nousresearch.com").strip().strip("/")
    scheme = os.getenv("TOOL_GATEWAY_SCHEME", "https").strip().lower()
    if scheme not in {"http", "https"}:
        scheme = "https"
    return f"{scheme}://{vendor}-gateway.{domain}"


def peek_nous_access_token() -> Optional[str]:
    """Read an explicit token for compatibility checks without refreshing."""
    import os
    token = os.getenv("TOOL_GATEWAY_USER_TOKEN", "").strip()
    return token or None


def read_nous_access_token() -> Optional[str]:
    """Read an explicit token; OAuth refresh is intentionally not supported."""
    return peek_nous_access_token()


def resolve_managed_tool_gateway(
    vendor: str,
    gateway_builder: Optional[Callable[[str], str]] = None,
    token_reader: Optional[Callable[[], Optional[str]]] = None,
) -> ManagedToolGatewayConfig | None:
    """Managed gateways are intentionally unavailable in this runtime."""
    return None


def is_managed_tool_gateway_ready(vendor: str) -> bool:
    """Return false so optional managed backends remain disabled."""
    return False
