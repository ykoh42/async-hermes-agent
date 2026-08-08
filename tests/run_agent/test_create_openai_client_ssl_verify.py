"""Regression: keepalive httpx client must honor custom CA bundles for HTTPS providers."""

import ssl
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import certifi
import httpx
import pytest

from agent.ssl_verify import resolve_httpx_verify
from agent.process_bootstrap import build_keepalive_http_client

_CA_ENV_VARS = (
    "HERMES_CA_BUNDLE",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "ALL_PROXY",
    "https_proxy",
    "http_proxy",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def clean_tls_env(monkeypatch):
    for var in _CA_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


async def test_build_keepalive_http_client_uses_hermes_ca_bundle(
    clean_tls_env,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_CA_BUNDLE", certifi.where())
    verify = await resolve_httpx_verify()
    client = await build_keepalive_http_client(
        "https://ollama.example.com/v1", verify=verify,
    )
    try:
        assert isinstance(client, httpx.AsyncClient)
        assert isinstance(client._transport._pool._ssl_context, ssl.SSLContext)
    finally:
        await client.aclose()


async def test_build_keepalive_http_client_honors_per_provider_ssl_ca_cert(
    clean_tls_env,
):
    verify = await resolve_httpx_verify(ca_bundle=certifi.where())
    client = await build_keepalive_http_client(
        "https://ollama.example.com/v1", verify=verify,
    )
    try:
        assert isinstance(client, httpx.AsyncClient)
        assert client._transport._pool._ssl_context is verify
    finally:
        await client.aclose()


async def test_build_keepalive_http_client_ssl_verify_false(clean_tls_env):
    verify = await resolve_httpx_verify(ssl_verify=False)
    client = await build_keepalive_http_client(
        "https://ollama.example.com/v1", verify=verify,
    )
    try:
        assert isinstance(client, httpx.AsyncClient)
        assert client._transport._pool._ssl_context.check_hostname is False
    finally:
        await client.aclose()


async def test_https_proxy_uses_separate_prebuilt_ssl_context(
    clean_tls_env,
    monkeypatch,
):
    target_context = await resolve_httpx_verify()
    proxy_context = await resolve_httpx_verify()
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.example.com:8443")

    with patch(
        "agent.ssl_verify.resolve_httpx_verify",
        AsyncMock(return_value=proxy_context),
    ) as resolve_proxy_verify:
        client = await build_keepalive_http_client(
            "https://provider.example.com/v1",
            verify=target_context,
        )

    try:
        pool = client._transport._pool
        assert pool._ssl_context is target_context
        assert pool._proxy_ssl_context is proxy_context
        resolve_proxy_verify.assert_awaited_once_with()
    finally:
        await client.aclose()


@pytest.mark.parametrize("error", [RuntimeError("boom"), asyncio.CancelledError()])
async def test_keepalive_builder_closes_transport_on_client_constructor_failure(
    clean_tls_env,
    error,
):
    transport = SimpleNamespace(aclose=AsyncMock())
    target_context = await resolve_httpx_verify()

    with (
        patch("httpx.AsyncHTTPTransport", return_value=transport),
        patch("httpx.AsyncClient", side_effect=error),
        pytest.raises(type(error)),
    ):
        await build_keepalive_http_client(
            "https://provider.example.com/v1",
            verify=target_context,
        )

    transport.aclose.assert_awaited_once_with()
