"""Security policy for credential-bearing async HTTP requests."""

from __future__ import annotations

import io
import urllib.error
import urllib.parse
import urllib.request
import urllib.response
from collections.abc import Callable, Iterable
from email.message import Message
from typing import Any
from urllib.parse import urlparse

import httpx

# Headers safe to forward to a different origin. Everything else is dropped:
# custom provider headers routinely carry credentials under arbitrary names.
_CROSS_ORIGIN_SAFE_HEADERS = frozenset({"accept", "user-agent"})
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


def _response_headers(response: httpx.Response) -> Message:
    headers = Message()
    for name, value in response.headers.multi_items():
        headers.add_header(name, value)
    return headers


def _urllib_response(response: httpx.Response):
    return urllib.response.addinfourl(
        io.BytesIO(response.content),
        _response_headers(response),
        str(response.url),
        response.status_code,
    )


class SafeCredentialRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Preserve request headers only while redirects stay on one origin."""

    def __init__(
        self,
        original_url: str,
        *,
        cross_origin_safe_headers: Iterable[str] = _CROSS_ORIGIN_SAFE_HEADERS,
    ) -> None:
        self._original_origin = url_origin(original_url)
        self._cross_origin_safe_headers = frozenset(
            str(name).lower() for name in cross_origin_safe_headers
        )

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None

        resolved_url = urllib.parse.urljoin(req.full_url, newurl)
        if url_origin(resolved_url) != self._original_origin:
            for name, _value in list(redirected.header_items()):
                if name.lower() not in self._cross_origin_safe_headers:
                    redirected.remove_header(name)
        return redirected


async def open_credentialed_url(
    request: urllib.request.Request,
    *,
    timeout: float,
    opener_factory: Callable[..., Any] | None = None,
):
    """Open a request without forwarding credentials across origins."""
    if not isinstance(request, urllib.request.Request):
        raise TypeError("request must be urllib.request.Request")

    outgoing_request = httpx.Request(
        request.get_method(),
        request.full_url,
        headers=dict(request.header_items()),
        content=request.data,
    )
    original_origin = url_origin(request.full_url)

    async def strip_cross_origin_credentials(outgoing: httpx.Request) -> None:
        if url_origin(str(outgoing.url)) == original_origin:
            return
        for name in list(outgoing.headers):
            if name.lower() not in _CROSS_ORIGIN_SAFE_HEADERS:
                del outgoing.headers[name]
        outgoing.headers["Host"] = outgoing.url.netloc.decode("ascii")

    async def enforce_urllib_redirect_semantics(response: httpx.Response) -> None:
        code = response.status_code
        method = response.request.method
        allowed = (
            code in {301, 302, 303, 307, 308} and method in {"GET", "HEAD"}
        ) or (code in {301, 302, 303} and method == "POST")
        if response.is_redirect and not allowed:
            raise urllib.error.HTTPError(
                str(response.request.url),
                code,
                response.reason_phrase,
                response.headers,
                None,
            )

    client_kwargs = {
        "timeout": timeout,
        "follow_redirects": True,
        "event_hooks": {
            "request": [strip_cross_origin_credentials],
            "response": [enforce_urllib_redirect_semantics],
        },
    }
    if opener_factory is None:
        from agent.ssl_verify import _create_httpx_client

        client_context = await _create_httpx_client(**client_kwargs)
    else:
        client_context = opener_factory(**client_kwargs)
    try:
        async with client_context as client:
            response = await client.send(outgoing_request)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        response = exc.response
        raise urllib.error.HTTPError(
            str(response.url),
            response.status_code,
            response.reason_phrase,
            _response_headers(response),
            io.BytesIO(response.content),
        ) from exc
    except httpx.RequestError as exc:
        raise urllib.error.URLError(exc) from exc

    return _urllib_response(response)


__all__ = [
    "SafeCredentialRedirectHandler",
    "open_credentialed_url",
    "url_origin",
]
