from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
from unittest.mock import AsyncMock

from hermes_cli import auth


def _state(**overrides):
    state = {
        "provider": "minimax-oauth",
        "portal_base_url": "https://api.minimax.io",
        "inference_base_url": "https://api.minimax.io/anthropic",
        "client_id": auth.MINIMAX_OAUTH_CLIENT_ID,
        "access_token": "access-token",
        "refresh_token": "refresh-token",
        "expires_at": datetime.fromtimestamp(
            time.time() + 3600,
            tz=timezone.utc,
        ).isoformat(),
    }
    state.update(overrides)
    return state


@pytest.fixture
def minimax_auth_store(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    auth_file = hermes_home / "auth.json"

    def write(state):
        auth_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "active_provider": "minimax-oauth",
                    "providers": {"minimax-oauth": state},
                }
            )
        )

    return auth_file, write


@pytest.mark.asyncio
async def test_resolve_minimax_oauth_uses_fresh_stored_token(
    minimax_auth_store,
):
    _auth_file, write = minimax_auth_store
    write(_state(access_token="fresh-token"))

    credentials = await auth.resolve_minimax_oauth_runtime_credentials()

    assert credentials == {
        "provider": "minimax-oauth",
        "api_key": "fresh-token",
        "base_url": "https://api.minimax.io/anthropic",
        "source": "oauth",
    }


@pytest.mark.asyncio
async def test_resolve_minimax_oauth_refreshes_without_blocking_event_loop(
    minimax_auth_store,
    monkeypatch,
):
    auth_file, write = minimax_auth_store
    write(_state(expires_at="1970-01-01T00:00:00+00:00"))
    request_seen = asyncio.Event()
    release_response = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        request_seen.set()
        await release_response.wait()
        assert b"refresh_token=refresh-token" in request.content
        return httpx.Response(
            200,
            json={
                "status": "success",
                "access_token": "rotated-access",
                "refresh_token": "rotated-refresh",
                "expired_in": 900,
            },
        )

    async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        auth.httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=transport, **kwargs),
    )

    task = asyncio.create_task(auth.resolve_minimax_oauth_runtime_credentials())
    await asyncio.wait_for(request_seen.wait(), timeout=1)
    await asyncio.sleep(0)
    assert not task.done()
    release_response.set()

    credentials = await task

    assert credentials["api_key"] == "rotated-access"
    persisted = json.loads(auth_file.read_text())["providers"]["minimax-oauth"]
    assert persisted["refresh_token"] == "rotated-refresh"


@pytest.mark.asyncio
async def test_terminal_minimax_refresh_failure_quarantines_tokens(
    minimax_auth_store,
    monkeypatch,
):
    auth_file, write = minimax_auth_store
    write(_state(expires_at="1970-01-01T00:00:00+00:00"))
    async_client = httpx.AsyncClient
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(401, text="invalid_grant")
    )
    monkeypatch.setattr(
        auth.httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=transport, **kwargs),
    )

    with pytest.raises(auth.AuthError) as exc_info:
        await auth.resolve_minimax_oauth_runtime_credentials()

    assert exc_info.value.relogin_required is True
    persisted = json.loads(auth_file.read_text())["providers"]["minimax-oauth"]
    assert "access_token" not in persisted
    assert "refresh_token" not in persisted
    assert persisted["last_auth_error"]["relogin_required"] is True


@pytest.mark.asyncio
async def test_minimax_refresh_error_body_is_bounded():
    body = "invalid_grant " + "x" * (64 * 1024)
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(400, text=body)
    )

    async with httpx.AsyncClient(transport=transport) as client:
        response = await auth._minimax_post_form(
            client,
            "https://api.minimax.io/oauth/token",
            data={},
            headers={},
        )
        text = await auth._minimax_response_error_text(response)

    assert text.startswith("invalid_grant")
    assert text.endswith("...[truncated]")
    assert len(text) <= auth._MINIMAX_OAUTH_ERROR_BODY_LIMIT + len("...[truncated]")


@pytest.mark.asyncio
async def test_minimax_token_provider_is_awaitable(minimax_auth_store):
    _auth_file, write = minimax_auth_store
    write(_state(access_token="fresh-token"))

    provider = auth.build_minimax_oauth_token_provider()

    assert await provider() == "fresh-token"


@pytest.mark.asyncio
async def test_minimax_pool_refreshes_the_selected_singleton(
    minimax_auth_store,
    monkeypatch,
):
    _auth_file, write = minimax_auth_store
    write(_state(access_token="old-access"))
    from agent.credential_pool import load_pool

    pool = await load_pool("minimax-oauth")
    selected = await pool.select()
    assert selected is not None
    assert selected.runtime_api_key == "old-access"

    async_client = httpx.AsyncClient
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            json={
                "status": "success",
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expired_in": 900,
            },
        )
    )
    monkeypatch.setattr(
        auth.httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=transport, **kwargs),
    )

    refreshed = await pool.try_refresh_current()

    assert refreshed is not None
    assert refreshed.runtime_api_key == "new-access"


@pytest.mark.asyncio
async def test_main_anthropic_transport_refreshes_minimax_before_request(
    monkeypatch,
):
    from agent import anthropic_adapter
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.api_mode = "anthropic_messages"
    agent.provider = "minimax-oauth"
    agent.api_key = "old-access"
    agent._anthropic_api_key = "old-access"
    agent._anthropic_base_url = "https://api.minimax.io/anthropic"
    agent._anthropic_client = object()
    agent._anthropic_client_source = (
        "old-access",
        agent._anthropic_base_url,
        False,
    )
    agent._oauth_1m_beta_disabled = False
    agent._provider_request_timeout = None
    agent.log_prefix = ""
    agent._capture_anthropic_response_headers = lambda _response: None
    entry = SimpleNamespace(source="oauth", runtime_api_key="old-access")
    agent._credential_pool = SimpleNamespace(
        provider="minimax-oauth",
        entries=lambda: [entry],
    )

    resolve = AsyncMock(
        return_value={
            "provider": "minimax-oauth",
            "api_key": "new-access",
            "base_url": agent._anthropic_base_url,
            "source": "oauth",
        }
    )
    monkeypatch.setattr(auth, "resolve_minimax_oauth_runtime_credentials", resolve)
    fresh_client = object()
    monkeypatch.setattr(
        anthropic_adapter,
        "build_anthropic_client",
        lambda *_args, **_kwargs: fresh_client,
    )
    create = AsyncMock(return_value="response")
    monkeypatch.setattr(anthropic_adapter, "create_anthropic_message", create)

    result = await agent._execute_model_request({"model": "MiniMax-M2.7"})

    assert result == "response"
    assert agent.api_key == "new-access"
    assert agent._anthropic_api_key == "new-access"
    assert agent._anthropic_client is fresh_client
    resolve.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_explicit_minimax_key_is_not_replaced_by_oauth_singleton(
    monkeypatch,
):
    from agent import anthropic_adapter
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.api_mode = "anthropic_messages"
    agent.provider = "minimax-oauth"
    agent.api_key = "explicit-key"
    agent._anthropic_api_key = "explicit-key"
    agent._anthropic_base_url = "https://api.minimax.io/anthropic"
    agent._anthropic_client = native_client = object()
    agent._anthropic_client_source = (
        "explicit-key",
        agent._anthropic_base_url,
        False,
    )
    agent._oauth_1m_beta_disabled = False
    agent._provider_request_timeout = None
    agent._credential_pool = None
    agent.log_prefix = ""
    agent._capture_anthropic_response_headers = lambda _response: None

    resolve = AsyncMock(side_effect=AssertionError("must not read OAuth singleton"))
    monkeypatch.setattr(auth, "resolve_minimax_oauth_runtime_credentials", resolve)
    create = AsyncMock(return_value="response")
    monkeypatch.setattr(anthropic_adapter, "create_anthropic_message", create)

    result = await agent._execute_model_request({"model": "MiniMax-M2.7"})

    assert result == "response"
    assert agent._anthropic_client is native_client
    resolve.assert_not_awaited()


@pytest.mark.asyncio
async def test_auxiliary_minimax_uses_native_anthropic_client(monkeypatch):
    from agent import anthropic_adapter, auxiliary_client

    resolve = AsyncMock(
        return_value={
            "provider": "minimax-oauth",
            "api_key": "oauth-access",
            "base_url": "https://api.minimax.io/anthropic",
            "source": "oauth",
        }
    )
    monkeypatch.setattr(auth, "resolve_minimax_oauth_runtime_credentials", resolve)
    native_client = object()
    monkeypatch.setattr(
        anthropic_adapter,
        "build_anthropic_client",
        lambda *_args, **_kwargs: native_client,
    )

    client, model = await auxiliary_client.resolve_provider_client(
        "minimax-oauth",
        "MiniMax-M2.7",
        config={},
    )

    assert isinstance(client, auxiliary_client.AnthropicAuxiliaryClient)
    assert client._real_client is native_client
    assert client.api_key == "oauth-access"
    assert model == "MiniMax-M2.7"
    resolve.assert_awaited_once_with()
