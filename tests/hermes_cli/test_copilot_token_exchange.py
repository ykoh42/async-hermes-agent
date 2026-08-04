"""Native-async Copilot token exchange tests."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest


@pytest.fixture(autouse=True)
def _clear_jwt_cache():
    import hermes_cli.copilot_auth as module

    module._jwt_cache.clear()
    yield
    module._jwt_cache.clear()


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
