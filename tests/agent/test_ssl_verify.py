"""Tests for agent.ssl_verify.resolve_httpx_verify."""

import asyncio
import ssl
import threading
from unittest.mock import AsyncMock

import certifi
import httpx
import pytest
from blockbuster import BlockBuster

from agent.ssl_verify import (
    _create_httpx_client,
    _create_openai_sdk_client,
    _materialize_httpx_verify,
    _resolve_httpx_client_verify,
    resolve_httpx_verify,
)

_CA_ENV_VARS = (
    "HERMES_CA_BUNDLE",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def clean_ca_env(monkeypatch):
    for var in _CA_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


async def test_hermes_ca_bundle_returns_ssl_context(clean_ca_env, monkeypatch):
    monkeypatch.setenv("HERMES_CA_BUNDLE", certifi.where())
    result = await resolve_httpx_verify()
    assert isinstance(result, ssl.SSLContext)


async def test_default_without_env_is_true(clean_ca_env):
    assert await resolve_httpx_verify() is True


async def test_ca_env_priority_matches_upstream_requests_resolver(
    clean_ca_env,
    monkeypatch,
    tmp_path,
):
    requests_bundle = tmp_path / "requests.pem"
    ssl_bundle = tmp_path / "ssl.pem"
    requests_bundle.write_text("requests")
    ssl_bundle.write_text("ssl")
    seen = []

    def create_default_context(*, cafile=None, capath=None):
        seen.append((cafile, capath))
        return object()

    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(requests_bundle))
    monkeypatch.setenv("SSL_CERT_FILE", str(ssl_bundle))
    monkeypatch.setattr(ssl, "create_default_context", create_default_context)

    await resolve_httpx_verify()

    assert seen == [(str(requests_bundle), None)]


async def test_invalid_higher_priority_ca_env_falls_through_to_valid_candidate(
    clean_ca_env,
    monkeypatch,
    tmp_path,
):
    requests_bundle = tmp_path / "requests.pem"
    requests_bundle.write_text("requests")
    seen = []

    def create_default_context(*, cafile=None, capath=None):
        seen.append((cafile, capath))
        return object()

    monkeypatch.setenv("HERMES_CA_BUNDLE", str(tmp_path / "missing.pem"))
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(requests_bundle))
    monkeypatch.setattr(ssl, "create_default_context", create_default_context)

    await resolve_httpx_verify()

    assert seen == [(str(requests_bundle), None)]


async def test_ca_directory_falls_back_to_upstream_true(
    clean_ca_env,
    tmp_path,
):
    assert await resolve_httpx_verify(ca_bundle=str(tmp_path)) is True


async def test_ssl_cert_dir_does_not_change_upstream_resolution(
    clean_ca_env,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path))
    assert await resolve_httpx_verify() is True


async def test_client_materializer_preserves_httpx_ssl_cert_dir(
    clean_ca_env,
    tmp_path,
    monkeypatch,
):
    calls = []
    create_default_context = ssl.create_default_context

    def tracked_create_default_context(*args, **kwargs):
        calls.append(kwargs)
        return create_default_context(*args, **kwargs)

    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path))
    monkeypatch.setattr(ssl, "create_default_context", tracked_create_default_context)

    result = await _resolve_httpx_client_verify()

    assert isinstance(result, ssl.SSLContext)
    assert calls == [{"capath": str(tmp_path)}]


async def test_client_materializer_ignores_ca_env_when_trust_env_is_false(
    clean_ca_env,
    tmp_path,
    monkeypatch,
):
    calls = []
    create_default_context = ssl.create_default_context

    def tracked_create_default_context(*args, **kwargs):
        calls.append(kwargs)
        return create_default_context(*args, **kwargs)

    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path))
    monkeypatch.setattr(ssl, "create_default_context", tracked_create_default_context)

    result = await _resolve_httpx_client_verify(trust_env=False)

    assert isinstance(result, ssl.SSLContext)
    assert calls == [{"cafile": certifi.where()}]


async def test_default_context_creation_does_not_block_event_loop(
    clean_ca_env,
    monkeypatch,
):
    event_loop_thread = threading.get_ident()
    create_default_context = ssl.create_default_context
    certifi_where = certifi.where

    def tracked_create_default_context(*args, **kwargs):
        assert threading.get_ident() != event_loop_thread
        return create_default_context(*args, **kwargs)

    def tracked_certifi_where():
        assert threading.get_ident() != event_loop_thread
        return certifi_where()

    monkeypatch.setattr(ssl, "create_default_context", tracked_create_default_context)
    monkeypatch.setattr(certifi, "where", tracked_certifi_where)
    blocker = BlockBuster()
    blocker.activate()
    try:
        result = await _resolve_httpx_client_verify()
    finally:
        blocker.deactivate()

    assert isinstance(result, ssl.SSLContext)


async def test_bare_httpx_materializer_does_not_apply_hermes_ca_env(
    clean_ca_env,
    monkeypatch,
):
    calls = []
    create_default_context = ssl.create_default_context
    monkeypatch.setenv("HERMES_CA_BUNDLE", "/not-an-httpx-ca-setting")

    def tracked_create_default_context(*args, **kwargs):
        calls.append(kwargs)
        return create_default_context(*args, **kwargs)

    monkeypatch.setattr(ssl, "create_default_context", tracked_create_default_context)

    result = await _materialize_httpx_verify()

    assert isinstance(result, ssl.SSLContext)
    assert calls == [{"cafile": certifi.where()}]


async def test_httpx_client_preserves_env_proxy_mounts_and_trust_env(
    clean_ca_env,
    monkeypatch,
):
    monkeypatch.setattr(
        "httpx._utils.get_environment_proxies",
        lambda: {
            "https://": "https://proxy.example:8443",
            "all://localhost": None,
        },
    )

    client = await _create_httpx_client(timeout=1.0)
    try:
        mounts = {
            pattern.pattern: transport
            for pattern, transport in client._mounts.items()
        }
        proxy_transport = mounts["https://"]

        assert client._trust_env is True
        assert mounts["all://localhost"] is None
        assert isinstance(client._transport, httpx.AsyncHTTPTransport)
        assert isinstance(proxy_transport, httpx.AsyncHTTPTransport)
        assert isinstance(
            proxy_transport._pool._proxy_ssl_context,
            ssl.SSLContext,
        )
    finally:
        await client.aclose()


async def test_httpx_client_runs_proxy_discovery_and_tls_setup_off_loop(
    clean_ca_env,
    monkeypatch,
):
    import httpcore

    event_loop_thread = threading.get_ident()
    default_proxy_context = httpcore.default_ssl_context

    def tracked_environment_proxies():
        assert threading.get_ident() != event_loop_thread
        return {"https://": "https://proxy.example:8443"}

    def tracked_proxy_context():
        assert threading.get_ident() != event_loop_thread
        return default_proxy_context()

    monkeypatch.setattr(
        "httpx._utils.get_environment_proxies",
        tracked_environment_proxies,
    )
    monkeypatch.setattr(
        "httpcore.default_ssl_context",
        tracked_proxy_context,
    )

    client = await _create_httpx_client(timeout=1.0)
    await client.aclose()


async def test_httpx_client_materializes_explicit_proxy_mount(clean_ca_env):
    client = await _create_httpx_client(
        proxy="https://proxy.example:8443",
        trust_env=False,
    )
    try:
        mounts = {
            pattern.pattern: transport
            for pattern, transport in client._mounts.items()
        }
        proxy_transport = mounts["all://"]

        assert isinstance(proxy_transport, httpx.AsyncHTTPTransport)
        assert isinstance(
            proxy_transport._pool._proxy_ssl_context,
            ssl.SSLContext,
        )
    finally:
        await client.aclose()


async def test_httpx_client_closes_owned_transport_on_constructor_failure(
    clean_ca_env,
    monkeypatch,
):
    transport = httpx.AsyncHTTPTransport(
        verify=await _materialize_httpx_verify(),
    )
    close = AsyncMock(wraps=transport.aclose)
    monkeypatch.setattr(transport, "aclose", close)
    monkeypatch.setattr(
        "httpx.AsyncHTTPTransport",
        lambda **_kwargs: transport,
    )
    monkeypatch.setattr(
        "httpx._utils.get_environment_proxies",
        lambda: {},
    )
    monkeypatch.setattr(
        "httpx.AsyncClient",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("constructor failed")),
    )

    with pytest.raises(RuntimeError, match="constructor failed"):
        await _create_httpx_client()

    close.assert_awaited_once()


async def test_httpx_client_closes_owned_explicit_proxy_transport_on_failure(
    clean_ca_env,
    monkeypatch,
):
    custom_transport = httpx.MockTransport(
        lambda request: httpx.Response(200, request=request)
    )
    custom_close = AsyncMock(wraps=custom_transport.aclose)
    monkeypatch.setattr(custom_transport, "aclose", custom_close)
    proxy_transport = httpx.AsyncHTTPTransport(
        verify=await _materialize_httpx_verify(),
    )
    proxy_close = AsyncMock(wraps=proxy_transport.aclose)
    monkeypatch.setattr(proxy_transport, "aclose", proxy_close)
    monkeypatch.setattr(
        "httpx.AsyncHTTPTransport",
        lambda **_kwargs: proxy_transport,
    )
    monkeypatch.setattr(
        "httpx.AsyncClient",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("constructor failed")),
    )

    with pytest.raises(RuntimeError, match="constructor failed"):
        await _create_httpx_client(
            transport=custom_transport,
            proxy="http://proxy.example:8080",
        )

    proxy_close.assert_awaited_once()
    custom_close.assert_not_awaited()


async def test_httpx_client_cleanup_survives_repeated_cancellation(
    clean_ca_env,
    monkeypatch,
):
    close_started = asyncio.Event()
    allow_close = asyncio.Event()

    class Transport:
        async def aclose(self):
            close_started.set()
            await allow_close.wait()

    monkeypatch.setattr(
        "httpx.AsyncHTTPTransport",
        lambda **_kwargs: Transport(),
    )
    monkeypatch.setattr(
        "httpx._utils.get_environment_proxies",
        lambda: {},
    )
    monkeypatch.setattr(
        "httpx.AsyncClient",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("constructor failed")),
    )

    create = asyncio.create_task(_create_httpx_client())
    await close_started.wait()
    create.cancel()
    await asyncio.sleep(0)
    create.cancel()
    await asyncio.sleep(0)

    assert create.done() is False
    allow_close.set()
    with pytest.raises(asyncio.CancelledError):
        await create


async def test_openai_sdk_client_preserves_defaults_without_blocking(
    clean_ca_env,
    monkeypatch,
):
    import openai
    from openai import _base_client

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

    blocker = BlockBuster()
    blocker.activate()
    try:
        client = await _create_openai_sdk_client(
            openai.AsyncOpenAI,
            api_key="test-key",
            base_url="https://api.example.com/v1",
        )
    finally:
        blocker.deactivate()

    try:
        http_client = client._client
        pool = http_client._transport._pool
        assert isinstance(http_client, _base_client.AsyncHttpxClientWrapper)
        assert http_client.follow_redirects is True
        assert http_client.base_url == client.base_url
        assert client.timeout == _base_client.DEFAULT_TIMEOUT
        assert pool._max_connections == 1000
        assert pool._max_keepalive_connections == 100
        assert pool._keepalive_expiry == 5.0
    finally:
        await client.close()


async def test_openai_sdk_constructor_cleanup_survives_repeated_cancellation(
    monkeypatch,
):
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    close_finished = asyncio.Event()

    class HttpClient:
        async def aclose(self):
            close_started.set()
            await allow_close.wait()
            close_finished.set()

    class FailingClient:
        def __init__(self, **_kwargs):
            raise RuntimeError("constructor failed")

    monkeypatch.setattr(
        "agent.ssl_verify._create_httpx_client",
        AsyncMock(return_value=HttpClient()),
    )

    create = asyncio.create_task(
        _create_openai_sdk_client(
            FailingClient,
            api_key="test-key",
        )
    )
    await close_started.wait()
    create.cancel()
    await asyncio.sleep(0)
    create.cancel()
    await asyncio.sleep(0)

    assert create.done() is False
    allow_close.set()
    with pytest.raises(asyncio.CancelledError):
        await create
    assert close_finished.is_set()
