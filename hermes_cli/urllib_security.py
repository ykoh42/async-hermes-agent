"""Security policy for credential-bearing async HTTP requests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

import httpx

# Headers safe to forward to a different origin. Everything else is dropped:
# custom provider headers routinely carry credentials under arbitrary names.
_CROSS_ORIGIN_SAFE_HEADERS = frozenset({"accept", "host", "user-agent"})
_DEFAULT_PORTS = {"http": 80, "https": 443}


def url_origin(url: str) -> tuple[str, str, int | None]:
    """Return a normalized (scheme, hostname, effective port) origin."""
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    # Accessing ``parsed.port`` validates malformed/non-numeric ports. Let the
    # ValueError fail the request closed instead of collapsing it to a default.
    port = parsed.port
    return (
        scheme,
        (parsed.hostname or "").lower().rstrip("."),
        port if port is not None else _DEFAULT_PORTS.get(scheme),
    )


async def open_credentialed_url(
    request: httpx.Request,
    *,
    timeout: float,
    opener_factory: Callable[..., Any] | None = None,
) -> httpx.Response:
    """Open a request without forwarding credentials across origins."""
    original_origin = url_origin(str(request.url))

    async def strip_cross_origin_credentials(outgoing: httpx.Request) -> None:
        if url_origin(str(outgoing.url)) == original_origin:
            return
        for name in list(outgoing.headers):
            if name.lower() not in _CROSS_ORIGIN_SAFE_HEADERS:
                del outgoing.headers[name]

    client_factory = opener_factory or httpx.AsyncClient
    async with client_factory(
        timeout=timeout,
        follow_redirects=True,
        event_hooks={"request": [strip_cross_origin_credentials]},
    ) as client:
        response = await client.send(request)
        response.raise_for_status()
        return response


__all__ = ["open_credentialed_url", "url_origin"]
