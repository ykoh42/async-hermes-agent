"""Tests for agent.ssl_verify.resolve_httpx_verify."""

import asyncio
import inspect
import ssl
import threading
from unittest.mock import AsyncMock

import certifi
import httpx
import pytest
from blockbuster import BlockBuster

from agent import ssl_verify as ssl_verify_module
from agent.secret_scope import (
    UnscopedSecretError,
    is_multiplex_active,
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from agent.ssl_verify import (
    _context_from_ca_directories,
    _create_httpx_client,
    _create_openai_sdk_client,
    _default_proxy_context,
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


@pytest.fixture
def upstream_certifi_context(clean_ca_env):
    return ssl.create_default_context(cafile=certifi.where())


@pytest.fixture
def upstream_proxy_context(clean_ca_env):
    import httpcore

    return httpcore.default_ssl_context()


@pytest.fixture
def upstream_empty_capath_context(clean_ca_env, tmp_path):
    return ssl.create_default_context(capath=str(tmp_path))


async def test_hermes_ca_bundle_returns_ssl_context(clean_ca_env, monkeypatch):
    monkeypatch.setenv("HERMES_CA_BUNDLE", certifi.where())
    result = await resolve_httpx_verify()
    assert isinstance(result, ssl.SSLContext)


async def test_in_memory_certifi_context_preserves_tls_and_trust_store(
    clean_ca_env,
    upstream_certifi_context,
):
    result = await _resolve_httpx_client_verify(trust_env=False)

    assert result.options == upstream_certifi_context.options
    assert result.verify_flags == upstream_certifi_context.verify_flags
    assert result.minimum_version == upstream_certifi_context.minimum_version
    assert set(result.get_ca_certs(binary_form=True)) == set(
        upstream_certifi_context.get_ca_certs(binary_form=True)
    )


async def test_proxy_context_preserves_httpcore_default_trust_store(
    clean_ca_env,
    upstream_proxy_context,
):
    result = await _default_proxy_context()

    # OpenSSL keeps ``capath`` certificates lazy until certificate-chain
    # lookup, so httpcore's fresh context does not report them here.  The
    # async materializer reads those same hashed entries up front; include
    # them when comparing the effective trust stores.
    _, default_capath = ssl_verify_module._raw_default_verify_paths()
    capath_context = await _context_from_ca_directories(default_capath or "")
    expected_certificates = set(
        upstream_proxy_context.get_ca_certs(binary_form=True)
    ) | set(capath_context.get_ca_certs(binary_form=True))

    assert result.options == upstream_proxy_context.options
    assert result.verify_flags == upstream_proxy_context.verify_flags
    assert result.minimum_version == upstream_proxy_context.minimum_version
    assert set(result.get_ca_certs(binary_form=True)) == expected_certificates


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

    def create_default_context(*, cadata=None):
        seen.append(cadata)
        return object()

    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(requests_bundle))
    monkeypatch.setenv("SSL_CERT_FILE", str(ssl_bundle))
    monkeypatch.setattr(ssl, "create_default_context", create_default_context)

    await resolve_httpx_verify()

    assert seen == [b"requests"]


async def test_invalid_higher_priority_ca_env_falls_through_to_valid_candidate(
    clean_ca_env,
    monkeypatch,
    tmp_path,
):
    requests_bundle = tmp_path / "requests.pem"
    requests_bundle.write_text("requests")
    seen = []

    def create_default_context(*, cadata=None):
        seen.append(cadata)
        return object()

    monkeypatch.setenv("HERMES_CA_BUNDLE", str(tmp_path / "missing.pem"))
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(requests_bundle))
    monkeypatch.setattr(ssl, "create_default_context", create_default_context)

    await resolve_httpx_verify()

    assert seen == [b"requests"]


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
    upstream_empty_capath_context,
):
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path))

    result = await _resolve_httpx_client_verify()

    assert isinstance(result, ssl.SSLContext)
    assert result.check_hostname is True
    assert result.verify_mode is ssl.CERT_REQUIRED
    assert result.options == upstream_empty_capath_context.options
    assert result.verify_flags == upstream_empty_capath_context.verify_flags
    assert result.minimum_version == upstream_empty_capath_context.minimum_version
    assert result.cert_store_stats() == upstream_empty_capath_context.cert_store_stats()


async def test_ca_directory_only_trusts_openssl_hashed_entries(
    clean_ca_env,
    tmp_path,
):
    bundle = await ssl_verify_module._read_file_bytes(certifi.where())
    (tmp_path / "unhashed.pem").write_bytes(bundle)

    empty = await _context_from_ca_directories(str(tmp_path))
    assert empty.cert_store_stats()["x509"] == 0

    (tmp_path / "0123abcd.0").write_bytes(bundle)
    populated = await _context_from_ca_directories(str(tmp_path))
    assert populated.cert_store_stats()["x509"] > 0


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
    assert len(calls) == 1
    assert set(calls[0]) == {"cadata"}
    assert isinstance(calls[0]["cadata"], str)


async def test_default_context_parsing_stays_on_event_loop_without_file_blocking(
    clean_ca_env,
    monkeypatch,
):
    event_loop_thread = threading.get_ident()
    create_default_context = ssl.create_default_context
    certifi_where = certifi.where

    def tracked_create_default_context(*args, **kwargs):
        assert threading.get_ident() == event_loop_thread
        assert set(kwargs) == {"cadata"}
        return create_default_context(*args, **kwargs)

    def tracked_certifi_where():
        assert threading.get_ident() == event_loop_thread
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


async def test_ca_read_cancellation_propagates_without_leaking_a_task(
    clean_ca_env,
    monkeypatch,
    tmp_path,
):
    ca_path = tmp_path / "ca.pem"
    ca_path.write_bytes(b"unused")
    started = asyncio.Event()
    wait_forever = asyncio.Event()

    async def stalled_read(_path):
        started.set()
        await wait_forever.wait()
        return b"unused"

    monkeypatch.setattr(ssl_verify_module, "_read_file_bytes", stalled_read)
    resolver = asyncio.create_task(resolve_httpx_verify(ca_bundle=str(ca_path)))
    await started.wait()
    resolver.cancel()

    with pytest.raises(asyncio.CancelledError):
        await resolver
    assert resolver.cancelled()


async def test_ssl_verify_has_no_generic_worker_bridge():
    source = inspect.getsource(ssl_verify_module)
    assert "aiofiles.os.wrap" not in source
    assert "asyncio.to_thread" not in source
    assert "run_in_executor" not in source


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
    assert len(calls) == 1
    assert set(calls[0]) == {"cadata"}
    assert isinstance(calls[0]["cadata"], str)


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


async def test_httpx_client_proxy_discovery_and_tls_parsing_do_not_use_workers(
    clean_ca_env,
    monkeypatch,
):
    event_loop_thread = threading.get_ident()
    create_default_context = ssl.create_default_context
    context_calls = []

    def tracked_environment_proxies():
        assert threading.get_ident() == event_loop_thread
        return {"https://": "https://proxy.example:8443"}

    def tracked_create_default_context(*args, **kwargs):
        assert threading.get_ident() == event_loop_thread
        context_calls.append(kwargs)
        return create_default_context(*args, **kwargs)

    monkeypatch.setattr(
        "httpx._utils.get_environment_proxies",
        tracked_environment_proxies,
    )
    monkeypatch.setattr(
        ssl,
        "create_default_context",
        tracked_create_default_context,
    )

    blocker = BlockBuster()
    blocker.activate()
    try:
        client = await _create_httpx_client(timeout=1.0)
    finally:
        blocker.deactivate()
    try:
        assert context_calls
        assert all(set(call) == {"cadata"} for call in context_calls)
    finally:
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


async def test_openai_sdk_default_base_url_is_profile_scoped(monkeypatch):
    previous = is_multiplex_active()
    set_multiplex_active(True)
    captured: list[str] = []

    class HttpClient:
        async def aclose(self):
            return None

    class Client:
        def __init__(self, **kwargs):
            self.http_client = kwargs["http_client"]

    async def create_http_client(**kwargs):
        await asyncio.sleep(0)
        captured.append(kwargs["base_url"])
        return HttpClient()

    async def resolve(name: str):
        token = set_secret_scope(
            {"OPENAI_BASE_URL": f"https://{name}.example/v1"}
        )
        try:
            return await _create_openai_sdk_client(Client, api_key="test")
        finally:
            reset_secret_scope(token)

    monkeypatch.setenv("OPENAI_BASE_URL", "https://foreign.example/v1")
    monkeypatch.setattr(
        "agent.ssl_verify._create_httpx_client",
        create_http_client,
    )
    try:
        await asyncio.gather(resolve("alpha"), resolve("beta"))
    finally:
        set_multiplex_active(previous)

    assert set(captured) == {
        "https://alpha.example/v1",
        "https://beta.example/v1",
    }


async def test_openai_sdk_default_base_url_fails_closed_without_scope(
    monkeypatch,
):
    previous = is_multiplex_active()
    set_multiplex_active(True)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://foreign.example/v1")
    create_http_client = AsyncMock()
    monkeypatch.setattr(
        "agent.ssl_verify._create_httpx_client",
        create_http_client,
    )
    try:
        with pytest.raises(UnscopedSecretError, match="OPENAI_BASE_URL"):
            await _create_openai_sdk_client(object, api_key="test")
    finally:
        set_multiplex_active(previous)

    create_http_client.assert_not_awaited()
