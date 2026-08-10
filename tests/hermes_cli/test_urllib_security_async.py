from unittest.mock import patch
import urllib.error
import urllib.request

import httpx
import pytest
from blockbuster import BlockBuster

from hermes_cli.urllib_security import (
    SafeCredentialRedirectHandler,
    open_credentialed_url,
)


@pytest.mark.asyncio
async def test_default_credentialed_client_setup_does_not_block(monkeypatch):
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        monkeypatch.delenv(name, raising=False)

    request = urllib.request.Request("https://catalog.example.test/models")
    request.add_header("Authorization", "Bearer secret")

    async def send(_client, outgoing):
        return httpx.Response(200, json={"data": []}, request=outgoing)

    blocker = BlockBuster()
    blocker.activate()
    try:
        with patch.object(httpx.AsyncClient, "send", new=send):
            response = await open_credentialed_url(request, timeout=5.0)
    finally:
        blocker.deactivate()

    with response:
        assert response.status == 200
        assert response.read() == b'{"data":[]}'


@pytest.mark.asyncio
async def test_cross_origin_redirect_drops_credential_headers():
    observed = []

    async def handler(request):
        observed.append((str(request.url), dict(request.headers)))
        if request.url.host == "catalog.example.test":
            return httpx.Response(
                302,
                headers={"Location": "https://redirect.example.test/models"},
                request=request,
            )
        return httpx.Response(200, content=b"ok", request=request)

    def client_factory(**kwargs):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)

    request = urllib.request.Request("https://catalog.example.test/models")
    request.add_header("Authorization", "Bearer secret")
    request.add_header("X-Provider-Key", "secret-two")
    request.add_header("Accept", "application/json")

    response = await open_credentialed_url(
        request,
        timeout=5.0,
        opener_factory=client_factory,
    )

    with response:
        assert response.read() == b"ok"
    assert observed[0][1]["authorization"] == "Bearer secret"
    redirected_headers = observed[1][1]
    assert "authorization" not in redirected_headers
    assert "x-provider-key" not in redirected_headers
    assert redirected_headers["accept"] == "application/json"


def test_post_307_remains_rejected_by_upstream_redirect_handler():
    request = urllib.request.Request(
        "https://models.example.test/load",
        data=b"{}",
        headers={"Authorization": "Bearer secret"},
        method="POST",
    )
    handler = SafeCredentialRedirectHandler(request.full_url)
    with pytest.raises(urllib.error.HTTPError):
        handler.redirect_request(
            request,
            None,
            307,
            "Temporary Redirect",
            {},
            "https://other.example.test/load",
        )


@pytest.mark.asyncio
async def test_native_transport_preserves_urllib_post_307_rejection():
    async def handler(request):
        return httpx.Response(
            307,
            headers={"Location": "https://other.example.test/load"},
            request=request,
        )

    def client_factory(**kwargs):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)

    request = urllib.request.Request(
        "https://models.example.test/load",
        data=b"{}",
        headers={"Authorization": "Bearer secret"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        await open_credentialed_url(
            request,
            timeout=5.0,
            opener_factory=client_factory,
        )
    assert exc_info.value.code == 307


@pytest.mark.asyncio
async def test_native_transport_preserves_urllib_http_error_shape():
    async def handler(request):
        return httpx.Response(401, content=b"denied", request=request)

    def client_factory(**kwargs):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)

    request = urllib.request.Request("https://models.example.test/models")
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        await open_credentialed_url(
            request,
            timeout=5.0,
            opener_factory=client_factory,
        )
    assert exc_info.value.code == 401
    assert exc_info.value.read() == b"denied"


@pytest.mark.asyncio
async def test_native_transport_preserves_urllib_transport_error_shape():
    async def handler(request):
        raise httpx.ConnectError("unreachable", request=request)

    def client_factory(**kwargs):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)

    request = urllib.request.Request("https://models.example.test/models")
    with pytest.raises(urllib.error.URLError):
        await open_credentialed_url(
            request,
            timeout=5.0,
            opener_factory=client_factory,
        )
