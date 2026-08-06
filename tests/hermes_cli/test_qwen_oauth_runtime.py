from __future__ import annotations

import asyncio
import json
import stat
import time

import httpx
import pytest

from hermes_cli import auth


@pytest.fixture
def qwen_auth_path(tmp_path, monkeypatch):
    path = tmp_path / ".qwen" / "oauth_creds.json"
    monkeypatch.setattr(auth, "_qwen_cli_auth_path", lambda: path)
    return path


def _tokens(**overrides):
    tokens = {
        "access_token": "access-token",
        "refresh_token": "refresh-token",
        "token_type": "Bearer",
        "resource_url": "portal.qwen.ai",
        "expiry_date": int((time.time() + 3600) * 1000),
    }
    tokens.update(overrides)
    return tokens


@pytest.mark.asyncio
async def test_resolve_qwen_runtime_credentials_reads_fresh_cli_token(
    qwen_auth_path,
) -> None:
    qwen_auth_path.parent.mkdir(parents=True)
    qwen_auth_path.write_text(json.dumps(_tokens(access_token="fresh")))

    resolved = await auth.resolve_qwen_runtime_credentials()

    assert resolved["provider"] == "qwen-oauth"
    assert resolved["api_key"] == "fresh"
    assert resolved["base_url"] == auth.DEFAULT_QWEN_BASE_URL
    assert resolved["source"] == "qwen-cli"


@pytest.mark.asyncio
async def test_resolve_qwen_runtime_credentials_refreshes_and_persists(
    qwen_auth_path,
    monkeypatch,
) -> None:
    qwen_auth_path.parent.mkdir(parents=True)
    qwen_auth_path.write_text(
        json.dumps(_tokens(expiry_date=int((time.time() - 60) * 1000)))
    )
    request_seen = asyncio.Event()
    release_response = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        request_seen.set()
        await release_response.wait()
        assert b"refresh_token=refresh-token" in request.content
        return httpx.Response(
            200,
            json={
                "access_token": "rotated-access",
                "refresh_token": "rotated-refresh",
                "expires_in": 7200,
            },
        )

    async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        auth.httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=transport, **kwargs),
    )
    resolve_task = asyncio.create_task(auth.resolve_qwen_runtime_credentials())
    await asyncio.wait_for(request_seen.wait(), timeout=1)
    await asyncio.sleep(0)
    assert not resolve_task.done()
    release_response.set()

    resolved = await resolve_task

    assert resolved["api_key"] == "rotated-access"
    persisted = json.loads(qwen_auth_path.read_text())
    assert persisted["refresh_token"] == "rotated-refresh"
    assert stat.S_IMODE(qwen_auth_path.stat().st_mode) == 0o600


@pytest.mark.asyncio
async def test_qwen_refresh_preserves_upstream_error_contract(
    qwen_auth_path,
    monkeypatch,
) -> None:
    async_client = httpx.AsyncClient
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(401, text="invalid refresh token")
    )
    monkeypatch.setattr(
        auth.httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=transport, **kwargs),
    )

    with pytest.raises(auth.AuthError) as exc_info:
        await auth._refresh_qwen_cli_tokens(_tokens())

    assert exc_info.value.provider == "qwen-oauth"
    assert exc_info.value.code == "qwen_refresh_failed"
    assert "invalid refresh token" in str(exc_info.value)
    assert not qwen_auth_path.exists()
