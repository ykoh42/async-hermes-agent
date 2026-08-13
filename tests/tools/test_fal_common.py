import asyncio
import sys
from types import SimpleNamespace

import pytest
from blockbuster import BlockBuster

from tools.fal_common import (
    _ManagedFalClient,
    _close_fal_client,
    _create_fal_client,
    _extract_http_status,
    _normalize_fal_queue_url_format,
)


@pytest.mark.asyncio
async def test_import_fal_client_never_cold_imports_on_running_loop(monkeypatch):
    from tools.fal_common import import_fal_client

    monkeypatch.delitem(sys.modules, "fal_client", raising=False)

    with pytest.raises(ImportError, match="before the async runtime starts"):
        import_fal_client()


@pytest.mark.asyncio
async def test_import_fal_client_returns_preloaded_module(monkeypatch):
    from tools.fal_common import import_fal_client

    preloaded = SimpleNamespace(AsyncClient=object)
    monkeypatch.setitem(sys.modules, "fal_client", preloaded)

    assert import_fal_client() is preloaded


class _FakeFALClient:
    default_timeout = 120.0

    def _get_auth(self):
        return SimpleNamespace(header_value="Key test-id:test-secret")


@pytest.mark.asyncio
async def test_create_fal_client_materializes_http_transport_without_blocking(
    monkeypatch,
):
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

    fal_client = SimpleNamespace(
        AsyncClient=_FakeFALClient,
        client=SimpleNamespace(USER_AGENT="fal-client/0.13.1 (python)"),
    )
    blocker = BlockBuster()
    blocker.activate()
    try:
        client = await _create_fal_client(fal_client)
    finally:
        blocker.deactivate()

    try:
        assert client._client.timeout.connect == 120.0
        assert client._client.timeout.read == 120.0
        assert client._client.headers["authorization"] == "Key test-id:test-secret"
        assert client._client.headers["user-agent"] == "fal-client/0.13.1 (python)"
    finally:
        await client._client.aclose()


@pytest.mark.asyncio
async def test_close_fal_client_survives_repeated_cancellation():
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    close_finished = asyncio.Event()

    class HttpClient:
        async def aclose(self):
            close_started.set()
            await allow_close.wait()
            close_finished.set()

    client = SimpleNamespace(_client=HttpClient())
    task = asyncio.create_task(_close_fal_client(client))
    await close_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)

    assert task.done() is False
    allow_close.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert close_finished.is_set()


@pytest.mark.asyncio
async def test_close_fal_client_preserves_cancellation_when_close_fails():
    close_started = asyncio.Event()
    allow_close = asyncio.Event()

    class HttpClient:
        async def aclose(self):
            close_started.set()
            await allow_close.wait()
            raise RuntimeError("close failed")

    client = SimpleNamespace(_client=HttpClient())
    task = asyncio.create_task(_close_fal_client(client))
    await close_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    allow_close.set()

    with pytest.raises(asyncio.CancelledError):
        await task


def test_managed_fal_helpers_preserve_upstream_shapes():
    assert _normalize_fal_queue_url_format(" https://gateway.test/// ") == (
        "https://gateway.test/"
    )
    with pytest.raises(ValueError, match="origin is required"):
        _normalize_fal_queue_url_format("  ")

    response_error = RuntimeError("response error")
    response_error.response = SimpleNamespace(status_code=403)
    assert _extract_http_status(response_error) == 403

    direct_error = RuntimeError("direct error")
    direct_error.status_code = 429
    assert _extract_http_status(direct_error) == 429
    assert _extract_http_status(RuntimeError("plain")) is None


@pytest.mark.asyncio
async def test_managed_fal_client_uses_async_sdk_queue_contract(monkeypatch):
    captured = {}

    class FakeHTTPClient:
        async def aclose(self):
            captured["closed"] = True

    http_client = FakeHTTPClient()

    class FakeAsyncClient:
        default_timeout = 120.0

        def __init__(self, key=None):
            captured["key"] = key

        def _get_auth(self):
            return SimpleNamespace(header_value=f"Key {captured['key']}")

    class FakeResponse:
        def json(self):
            return {
                "request_id": "req-123",
                "response_url": "https://gateway.test/requests/req-123",
                "status_url": "https://gateway.test/requests/req-123/status",
                "cancel_url": "https://gateway.test/requests/req-123/cancel",
            }

    async def maybe_retry(client, method, url, **kwargs):
        captured.update(
            http_client=client,
            method=method,
            url=url,
            request=kwargs,
        )
        return FakeResponse()

    class FakeHandle:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    async def create_httpx_client(**kwargs):
        captured["client_headers"] = kwargs["headers"]
        captured["client_timeout"] = kwargs["timeout"]
        return http_client

    monkeypatch.setattr(
        "agent.ssl_verify._create_httpx_client",
        create_httpx_client,
    )
    fal_client = SimpleNamespace(
        AsyncClient=FakeAsyncClient,
        client=SimpleNamespace(
            USER_AGENT="fal-client/0.13.1 (python)",
            _async_maybe_retry_request=maybe_retry,
            _raise_for_status=lambda response: None,
            AsyncRequestHandle=FakeHandle,
            add_hint_header=lambda value, headers: headers.update(
                {"x-fal-runner-hint": value}
            ),
            add_priority_header=lambda value, headers: headers.update(
                {"x-fal-queue-priority": value}
            ),
            add_timeout_header=lambda value, headers: headers.update(
                {"x-fal-request-timeout": str(value)}
            ),
        ),
    )
    client = _ManagedFalClient(
        fal_client,
        key="nous-token",
        queue_run_origin="https://gateway.test/",
    )

    blocker = BlockBuster()
    blocker.activate()
    try:
        handle = await client.submit(
            "fal-ai/model",
            {"prompt": "hello"},
            path="edit",
            hint="fast",
            priority="normal",
            start_timeout=17,
            webhook_url="https://callback.test/hook",
            headers={"x-idempotency-key": "call-123"},
        )
        await client.close()
    finally:
        blocker.deactivate()

    assert captured["key"] == "nous-token"
    assert captured["client_headers"] == {
        "Authorization": "Key nous-token",
        "User-Agent": "fal-client/0.13.1 (python)",
    }
    assert captured["client_timeout"] == 120.0
    assert captured["http_client"] is http_client
    assert captured["method"] == "POST"
    assert captured["url"] == (
        "https://gateway.test/fal-ai/model/edit?"
        "fal_webhook=https%3A%2F%2Fcallback.test%2Fhook"
    )
    assert captured["request"]["json"] == {"prompt": "hello"}
    assert captured["request"]["timeout"] == 120.0
    assert captured["request"]["headers"] == {
        "x-idempotency-key": "call-123",
        "x-fal-runner-hint": "fast",
        "x-fal-queue-priority": "normal",
        "x-fal-request-timeout": "17",
    }
    assert handle.request_id == "req-123"
    assert handle.client is http_client
    assert captured["closed"] is True
