from __future__ import annotations

import asyncio
import base64
import json
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import httpx
import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli import auth


def _jwt(*, expires_in: int = 3600) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    payload = (
        base64
        .urlsafe_b64encode(
            json.dumps({
                "exp": int(time.time()) + expires_in,
                "scope": auth.NOUS_INFERENCE_INVOKE_SCOPE,
            }).encode()
        )
        .decode()
        .rstrip("=")
    )
    return f"{header}.{payload}.signature"


def _state(**overrides):
    state = {
        "access_token": _jwt(),
        "refresh_token": "refresh-token",
        "client_id": auth.DEFAULT_NOUS_CLIENT_ID,
        "portal_base_url": auth.DEFAULT_NOUS_PORTAL_URL,
        "inference_base_url": auth.DEFAULT_NOUS_INFERENCE_URL,
        "token_type": "Bearer",
        "scope": auth.DEFAULT_NOUS_SCOPE,
        "obtained_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": datetime.fromtimestamp(
            time.time() + 3600,
            tz=timezone.utc,
        ).isoformat(),
        "tls": {"insecure": False, "ca_bundle": None},
        "label": "Nous account",
    }
    state.update(overrides)
    return state


@pytest.fixture
def nous_auth_store(tmp_path, monkeypatch):
    auth._RESOLVE_TOKEN_CACHE = None
    hermes_home = tmp_path / "hermes"
    shared_home = tmp_path / "shared"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(shared_home))
    auth_file = hermes_home / "auth.json"

    def write(state, *, pool_entries=None):
        payload = {
            "version": 1,
            "active_provider": "nous",
            "providers": {"nous": state},
        }
        if pool_entries is not None:
            payload["credential_pool"] = {"nous": pool_entries}
        auth_file.write_text(json.dumps(payload))

    return auth_file, shared_home / auth.NOUS_SHARED_STORE_FILENAME, write


@pytest.mark.asyncio
async def test_access_token_cache_is_scoped_to_active_profile(tmp_path, monkeypatch):
    auth._RESOLVE_TOKEN_CACHE = None

    async def no_shared_store():
        return None

    monkeypatch.setattr(auth, "_nous_shared_store_path", no_shared_store)
    token_a = _jwt() + "a"
    token_b = _jwt() + "b"
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    for home, access_token in ((home_a, token_a), (home_b, token_b)):
        home.mkdir()
        (home / "auth.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "active_provider": "nous",
                    "providers": {"nous": _state(access_token=access_token)},
                }
            )
        )

    async def resolve(home):
        profile_token = set_hermes_home_override(home)
        try:
            return await auth.resolve_nous_access_token()
        finally:
            reset_hermes_home_override(profile_token)

    assert await resolve(home_a) == token_a
    assert await resolve(home_b) == token_b


@pytest.mark.asyncio
async def test_resolve_nous_access_token_returns_portal_token_not_agent_key(
    nous_auth_store,
    monkeypatch,
):
    _auth_file, _shared_file, write = nous_auth_store
    access_token = _jwt()
    write(_state(access_token=access_token, agent_key="inference-agent-key"))
    async_client = httpx.AsyncClient
    transport = httpx.MockTransport(
        lambda _request: pytest.fail(
            "fresh access token must not call the token endpoint"
        )
    )
    monkeypatch.setattr(
        auth.httpx,
        "AsyncClient",
        lambda **kwargs: async_client(
            transport=transport,
            **{
                key: value
                for key, value in kwargs.items()
                if key not in {"transport", "mounts"}
            },
        ),
    )

    resolved = await auth.resolve_nous_access_token()

    assert resolved == access_token
    assert resolved != "inference-agent-key"


@pytest.mark.asyncio
async def test_resolve_nous_access_token_refreshes_and_returns_portal_token(
    nous_auth_store,
    monkeypatch,
):
    auth_file, shared_file, write = nous_auth_store
    write(
        _state(
            access_token=_jwt(expires_in=-60),
            expires_at="1970-01-01T00:00:00+00:00",
            agent_key="stale-inference-agent-key",
        )
    )
    refreshed_access = _jwt(expires_in=7200)
    async_client = httpx.AsyncClient
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "access_token": refreshed_access,
                "refresh_token": "rotated-refresh",
                "expires_in": 7200,
                "scope": auth.DEFAULT_NOUS_SCOPE,
            },
        )
    )
    monkeypatch.setattr(
        auth.httpx,
        "AsyncClient",
        lambda **kwargs: async_client(
            transport=transport,
            **{
                key: value
                for key, value in kwargs.items()
                if key not in {"transport", "mounts"}
            },
        ),
    )

    resolved = await auth.resolve_nous_access_token()

    assert resolved == refreshed_access
    persisted = json.loads(auth_file.read_text())["providers"]["nous"]
    assert persisted["access_token"] == refreshed_access
    assert persisted["refresh_token"] == "rotated-refresh"
    assert persisted["agent_key"] == "stale-inference-agent-key"
    assert json.loads(shared_file.read_text())["access_token"] == refreshed_access


@pytest.mark.asyncio
async def test_resolve_nous_uses_fresh_invoke_jwt_and_preserves_metadata(
    nous_auth_store,
    monkeypatch,
):
    auth_file, shared_file, write = nous_auth_store
    access_token = _jwt()
    write(_state(access_token=access_token))
    async_client = httpx.AsyncClient
    transport = httpx.MockTransport(
        lambda _request: pytest.fail("fresh JWT must not call the token endpoint")
    )
    monkeypatch.setattr(
        auth.httpx,
        "AsyncClient",
        lambda **kwargs: async_client(
            transport=transport,
            **{
                key: value
                for key, value in kwargs.items()
                if key not in {"transport", "mounts"}
            },
        ),
    )

    credentials = await auth.resolve_nous_runtime_credentials()

    assert credentials["api_key"] == access_token
    assert credentials["base_url"] == auth.DEFAULT_NOUS_INFERENCE_URL
    assert credentials["source"] == auth.NOUS_AUTH_PATH_INVOKE_JWT
    persisted = json.loads(auth_file.read_text())["providers"]["nous"]
    assert persisted["label"] == "Nous account"
    assert persisted["agent_key"] == access_token
    assert json.loads(shared_file.read_text())["refresh_token"] == "refresh-token"


@pytest.mark.asyncio
async def test_resolve_nous_refreshes_without_blocking_and_persists_rotation(
    nous_auth_store,
    monkeypatch,
):
    auth_file, shared_file, write = nous_auth_store
    write(
        _state(
            access_token=_jwt(expires_in=-60),
            expires_at="1970-01-01T00:00:00+00:00",
        )
    )
    request_seen = asyncio.Event()
    release_response = asyncio.Event()
    new_access = _jwt(expires_in=7200)

    async def handler(request: httpx.Request) -> httpx.Response:
        request_seen.set()
        await release_response.wait()
        assert request.headers["x-nous-refresh-token"] == "refresh-token"
        return httpx.Response(
            200,
            json={
                "access_token": new_access,
                "refresh_token": "rotated-refresh",
                "expires_in": 7200,
                "scope": auth.DEFAULT_NOUS_SCOPE,
                "inference_base_url": auth.DEFAULT_NOUS_INFERENCE_URL,
            },
        )

    async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        auth.httpx,
        "AsyncClient",
        lambda **kwargs: async_client(
            transport=transport,
            **{
                key: value
                for key, value in kwargs.items()
                if key not in {"transport", "mounts"}
            },
        ),
    )

    task = asyncio.create_task(auth.resolve_nous_runtime_credentials())
    await asyncio.wait_for(request_seen.wait(), timeout=1)
    await asyncio.sleep(0)
    assert not task.done()
    release_response.set()
    credentials = await task

    assert credentials["api_key"] == new_access
    persisted = json.loads(auth_file.read_text())["providers"]["nous"]
    assert persisted["refresh_token"] == "rotated-refresh"
    assert persisted["agent_key"] == new_access
    assert json.loads(shared_file.read_text())["refresh_token"] == "rotated-refresh"


@pytest.mark.asyncio
async def test_nous_refresh_rejects_network_supplied_inference_host(
    nous_auth_store,
    monkeypatch,
):
    auth_file, _shared_file, write = nous_auth_store
    write(_state(access_token=_jwt(expires_in=-60)))
    new_access = _jwt()
    async_client = httpx.AsyncClient
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            json={
                "access_token": new_access,
                "refresh_token": "rotated-refresh",
                "expires_in": 3600,
                "scope": auth.DEFAULT_NOUS_SCOPE,
                "inference_base_url": "https://attacker.example/v1",
            },
        )
    )
    monkeypatch.setattr(
        auth.httpx,
        "AsyncClient",
        lambda **kwargs: async_client(
            transport=transport,
            **{
                key: value
                for key, value in kwargs.items()
                if key not in {"transport", "mounts"}
            },
        ),
    )

    credentials = await auth.resolve_nous_runtime_credentials()

    assert credentials["base_url"] == auth.DEFAULT_NOUS_INFERENCE_URL
    persisted = json.loads(auth_file.read_text())["providers"]["nous"]
    assert persisted["inference_base_url"] == auth.DEFAULT_NOUS_INFERENCE_URL


@pytest.mark.asyncio
async def test_terminal_nous_refresh_quarantines_singleton_and_shared_store(
    nous_auth_store,
    monkeypatch,
):
    auth_file, shared_file, write = nous_auth_store
    state = _state(access_token=_jwt(expires_in=-60))
    pool_entry = {
        "id": "singleton",
        "label": "Nous account",
        "auth_type": "oauth",
        "priority": 0,
        "source": "device_code",
        "access_token": state["access_token"],
        "refresh_token": state["refresh_token"],
    }
    write(state, pool_entries=[pool_entry])
    shared_file.parent.mkdir(parents=True)
    shared_file.write_text(json.dumps(state))
    async_client = httpx.AsyncClient
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            400,
            json={
                "error": "invalid_grant",
                "error_description": "refresh token expired",
            },
        )
    )
    monkeypatch.setattr(
        auth.httpx,
        "AsyncClient",
        lambda **kwargs: async_client(
            transport=transport,
            **{
                key: value
                for key, value in kwargs.items()
                if key not in {"transport", "mounts"}
            },
        ),
    )

    with pytest.raises(auth.AuthError) as exc_info:
        await auth.resolve_nous_runtime_credentials()

    assert exc_info.value.relogin_required is True
    store = json.loads(auth_file.read_text())
    persisted = store["providers"]["nous"]
    assert "access_token" not in persisted
    assert "refresh_token" not in persisted
    assert persisted["last_auth_error"]["code"] == "invalid_grant"
    assert store["credential_pool"]["nous"] == []
    assert not shared_file.exists()


@pytest.mark.asyncio
async def test_nous_pool_refreshes_selected_singleton(
    nous_auth_store,
    monkeypatch,
):
    _auth_file, _shared_file, write = nous_auth_store
    old_access = _jwt(expires_in=-60)
    write(_state(access_token=old_access))
    from agent.credential_pool import load_pool

    pool = await load_pool("nous")
    selected = await pool.select()
    assert selected is not None
    new_access = _jwt(expires_in=7200)
    async_client = httpx.AsyncClient
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            json={
                "access_token": new_access,
                "refresh_token": "rotated-refresh",
                "expires_in": 7200,
                "scope": auth.DEFAULT_NOUS_SCOPE,
                "inference_base_url": auth.DEFAULT_NOUS_INFERENCE_URL,
            },
        )
    )
    monkeypatch.setattr(
        auth.httpx,
        "AsyncClient",
        lambda **kwargs: async_client(
            transport=transport,
            **{
                key: value
                for key, value in kwargs.items()
                if key not in {"transport", "mounts"}
            },
        ),
    )

    refreshed = await pool.try_refresh_current()

    assert refreshed is not None
    assert refreshed.runtime_api_key == new_access
    assert refreshed.refresh_token == "rotated-refresh"


@pytest.mark.asyncio
async def test_main_agent_nous_refresh_rebuilds_deferred_runtime(monkeypatch):
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.provider = "nous"
    agent.api_mode = "chat_completions"
    agent.model = "model"
    agent.api_key = "old-key"
    agent._provider_request_timeout = None
    agent._provider_stale_timeout = None
    agent._ensure_provider_runtime = AsyncMock(return_value=True)
    resolve = AsyncMock(
        return_value={
            "api_key": "new-key",
            "base_url": auth.DEFAULT_NOUS_INFERENCE_URL,
        }
    )
    monkeypatch.setattr(auth, "resolve_nous_runtime_credentials", resolve)

    refreshed = await agent._try_refresh_nous_client_credentials(force=True)

    assert refreshed is True
    resolve.assert_awaited_once_with(timeout_seconds=15.0, force_refresh=True)
    assert agent._deferred_provider_runtime["api_key"] == "new-key"
    agent._ensure_provider_runtime.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("api_mode", ["chat_completions", "anthropic_messages"])
async def test_main_agent_nous_refresh_rebuilds_when_token_is_unchanged(
    monkeypatch,
    api_mode,
):
    """A forced Nous refresh also repairs a stale transport with the same JWT."""
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.provider = "nous"
    agent.api_mode = api_mode
    agent.model = "anthropic/claude-opus-4.8"
    agent.api_key = "same-key"
    agent._provider_request_timeout = None
    agent._provider_stale_timeout = None
    agent._ensure_provider_runtime = AsyncMock(return_value=True)
    resolve = AsyncMock(
        return_value={
            "api_key": "same-key",
            "base_url": auth.DEFAULT_NOUS_INFERENCE_URL,
        }
    )
    monkeypatch.setattr(auth, "resolve_nous_runtime_credentials", resolve)

    refreshed = await agent._try_refresh_nous_client_credentials(force=True)

    assert refreshed is True
    assert agent._deferred_provider_runtime == {
        "provider": "nous",
        "model": "anthropic/claude-opus-4.8",
        "api_key": "same-key",
        "base_url": auth.DEFAULT_NOUS_INFERENCE_URL,
        "api_mode": api_mode,
        "request_timeout": None,
        "stale_timeout": None,
        "update_primary": False,
    }
    agent._ensure_provider_runtime.assert_awaited_once_with()
