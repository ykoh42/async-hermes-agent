"""Native-async Copilot token exchange tests."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest


@pytest.fixture(autouse=True)
def _clear_jwt_cache(tmp_path, monkeypatch):
    import hermes_cli.copilot_auth as module

    module._jwt_cache.clear()
    module._exchange_failure_cache.clear()
    monkeypatch.setattr(
        module,
        "_jwt_disk_path",
        lambda: tmp_path / module._JWT_DISK_FILENAME,
    )
    yield
    module._jwt_cache.clear()
    module._exchange_failure_cache.clear()


class _Client:
    payload: dict = {}
    request_headers: dict[str, str] = {}

    def __init__(self, **_kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def get(self, url: str, *, headers: dict[str, str]):
        type(self).request_headers = headers
        request = httpx.Request("GET", url)
        return httpx.Response(200, request=request, json=type(self).payload)


@pytest.mark.asyncio
async def test_exchanges_token_successfully(monkeypatch):
    from hermes_cli import copilot_auth

    _Client.payload = {
        "token": "tid=abc;exp=999",
        "expires_at": time.time() + 1800,
    }
    monkeypatch.setattr(copilot_auth.httpx, "AsyncClient", _Client)

    api_token, expires_at, base_url = await copilot_auth.exchange_copilot_token(
        "gho_test123"
    )

    assert api_token == "tid=abc;exp=999"
    assert isinstance(expires_at, float)
    assert base_url is None
    assert _Client.request_headers["Authorization"] == "token gho_test123"
    assert "GitHubCopilotChat" in _Client.request_headers["User-Agent"]


@pytest.mark.asyncio
async def test_exchange_rejects_empty_token(monkeypatch):
    from hermes_cli import copilot_auth

    _Client.payload = {"token": "", "expires_at": 0}
    monkeypatch.setattr(copilot_auth.httpx, "AsyncClient", _Client)

    with pytest.raises(ValueError, match="empty token"):
        await copilot_auth.exchange_copilot_token("gho_test123")


@pytest.mark.asyncio
async def test_get_copilot_api_token_returns_exchange():
    from hermes_cli.copilot_auth import get_copilot_api_token

    with patch(
        "hermes_cli.copilot_auth.exchange_copilot_token",
        new=AsyncMock(return_value=("exchanged", time.time() + 1800, None)),
    ):
        assert await get_copilot_api_token("gho_raw") == ("exchanged", None)


@pytest.mark.asyncio
async def test_runtime_credentials_use_exchange(monkeypatch):
    from hermes_cli import auth

    monkeypatch.setattr(
        "hermes_cli.copilot_auth.resolve_copilot_token",
        AsyncMock(return_value=("gho_raw", "GH_TOKEN")),
    )
    exchange = AsyncMock(return_value=("exchanged", "https://enterprise.example/v1"))
    monkeypatch.setattr("hermes_cli.copilot_auth.get_copilot_api_token", exchange)

    credentials = await auth.resolve_api_key_provider_credentials("copilot")

    assert credentials["api_key"] == "exchanged"
    assert credentials["base_url"] == "https://enterprise.example/v1"
    assert credentials["source"] == "GH_TOKEN"
    exchange.assert_awaited_once_with("gho_raw")


def test_token_fingerprint_is_stable():
    from hermes_cli.copilot_auth import _token_fingerprint

    assert _token_fingerprint("gho_abc") == _token_fingerprint("gho_abc")


def test_derive_enterprise_base_url():
    from hermes_cli.copilot_auth import _derive_base_url_from_proxy_ep

    token = "tid=abc;proxy-ep=proxy.enterprise.githubcopilot.com;sku=enterprise"
    assert _derive_base_url_from_proxy_ep(token) == "https://api.enterprise.githubcopilot.com"


@pytest.mark.asyncio
async def test_disk_cache_survives_memory_cache_clear(monkeypatch):
    from hermes_cli import copilot_auth

    _Client.payload = {
        "token": "tid=persisted;exp=999",
        "expires_at": time.time() + 1800,
    }

    async def successful_get(url: str, *, headers: dict[str, str]):
        request = httpx.Request("GET", url)
        return httpx.Response(200, request=request, json=_Client.payload)

    get = AsyncMock(side_effect=successful_get)
    monkeypatch.setattr(_Client, "get", get)
    monkeypatch.setattr(copilot_auth.httpx, "AsyncClient", _Client)

    first = await copilot_auth.exchange_copilot_token("gho_persisted")
    copilot_auth._jwt_cache.clear()
    second = await copilot_auth.exchange_copilot_token("gho_persisted")

    assert second == first
    assert get.await_count == 1


@pytest.mark.asyncio
async def test_evict_removes_memory_disk_and_negative_cache():
    from hermes_cli import copilot_auth

    token = "gho_stale"
    fingerprint = copilot_auth._token_fingerprint(token)
    copilot_auth._jwt_cache[fingerprint] = (
        "tid=stale",
        time.time() + 1800,
        None,
    )
    copilot_auth._exchange_failure_cache[fingerprint] = time.time() + 60
    await copilot_auth._save_jwt_to_disk(
        fingerprint,
        "tid=stale",
        time.time() + 1800,
        None,
    )

    await copilot_auth.evict_cached_exchanged_token(token)

    assert fingerprint not in copilot_auth._jwt_cache
    assert fingerprint not in copilot_auth._exchange_failure_cache
    assert await copilot_auth._load_jwt_from_disk(fingerprint) is None


@pytest.mark.asyncio
async def test_oversized_disk_store_is_ignored(tmp_path):
    from hermes_cli import copilot_auth

    path = tmp_path / copilot_auth._JWT_DISK_FILENAME
    path.write_text("x" * (copilot_auth._JWT_DISK_MAX_BYTES + 1))

    assert await copilot_auth._read_jwt_store(path) is None


@pytest.mark.asyncio
async def test_permanent_exchange_failure_is_not_retried(monkeypatch):
    from hermes_cli import copilot_auth

    class RejectedClient(_Client):
        pass

    async def rejected_get(url: str, *, headers: dict[str, str]):
        request = httpx.Request("GET", url)
        response = httpx.Response(403, request=request)
        raise httpx.HTTPStatusError("rejected", request=request, response=response)

    get = AsyncMock(side_effect=rejected_get)
    monkeypatch.setattr(RejectedClient, "get", get)
    monkeypatch.setattr(copilot_auth.httpx, "AsyncClient", RejectedClient)
    sleep = AsyncMock()
    monkeypatch.setattr(copilot_auth.asyncio, "sleep", sleep)

    with pytest.raises(ValueError):
        await copilot_auth.exchange_copilot_token("gho_rejected")
    with pytest.raises(ValueError, match="recently failed"):
        await copilot_auth.exchange_copilot_token("gho_rejected")

    assert get.await_count == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_transient_exchange_failure_retries_without_blocking(monkeypatch):
    from hermes_cli import copilot_auth

    class UnavailableClient(_Client):
        pass

    async def unavailable_get(_url: str, *, headers: dict[str, str]):
        raise httpx.ConnectError("offline")

    get = AsyncMock(side_effect=unavailable_get)
    monkeypatch.setattr(UnavailableClient, "get", get)
    monkeypatch.setattr(copilot_auth.httpx, "AsyncClient", UnavailableClient)
    sleep = AsyncMock()
    monkeypatch.setattr(copilot_auth.asyncio, "sleep", sleep)

    with pytest.raises(ValueError):
        await copilot_auth.exchange_copilot_token("gho_offline")

    assert get.await_count == copilot_auth._EXCHANGE_MAX_ATTEMPTS
    assert sleep.await_count == copilot_auth._EXCHANGE_MAX_ATTEMPTS - 1


@pytest.mark.asyncio
async def test_exchange_cancellation_propagates(monkeypatch):
    from hermes_cli import copilot_auth

    class CancelledClient(_Client):
        async def get(self, _url: str, *, headers: dict[str, str]):
            raise asyncio.CancelledError

    import asyncio

    monkeypatch.setattr(copilot_auth.httpx, "AsyncClient", CancelledClient)
    with pytest.raises(asyncio.CancelledError):
        await copilot_auth.exchange_copilot_token("gho_cancelled")
