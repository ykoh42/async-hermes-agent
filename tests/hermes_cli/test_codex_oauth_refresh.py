from __future__ import annotations

import asyncio
import base64
import json
import time
from unittest.mock import AsyncMock

import httpx
import pytest

from hermes_cli import auth


def _install_transport(monkeypatch, handler) -> None:
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


@pytest.mark.asyncio
async def test_refresh_codex_oauth_rotates_tokens_without_blocking(monkeypatch) -> None:
    request_seen = asyncio.Event()
    release_response = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        request_seen.set()
        await release_response.wait()
        assert request.headers["user-agent"] == auth.CODEX_OAUTH_USER_AGENT
        assert b"refresh_token=old-refresh" in request.content
        return httpx.Response(
            200,
            json={
                "access_token": "new-access",
                "refresh_token": "new-refresh",
            },
        )

    _install_transport(monkeypatch, handler)
    refresh_task = asyncio.create_task(
        auth.refresh_codex_oauth_pure("old-access", "old-refresh")
    )
    await asyncio.wait_for(request_seen.wait(), timeout=1)
    await asyncio.sleep(0)
    assert not refresh_task.done()
    release_response.set()

    refreshed = await refresh_task

    assert refreshed["access_token"] == "new-access"
    assert refreshed["refresh_token"] == "new-refresh"
    assert refreshed["last_refresh"].endswith("Z")


@pytest.mark.asyncio
async def test_refresh_codex_oauth_preserves_rate_limit_classification(monkeypatch) -> None:
    _install_transport(
        monkeypatch,
        lambda _request: httpx.Response(
            429,
            headers={"Retry-After": "17"},
            json={"error": "rate_limit_exceeded"},
        ),
    )

    with pytest.raises(auth.AuthError) as exc_info:
        await auth.refresh_codex_oauth_pure("old-access", "old-refresh")

    assert exc_info.value.code == auth.CODEX_RATE_LIMITED_CODE
    assert exc_info.value.relogin_required is False
    assert "retry after 17s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_refresh_codex_oauth_preserves_invalid_grant_classification(monkeypatch) -> None:
    _install_transport(
        monkeypatch,
        lambda _request: httpx.Response(
            400,
            json={
                "error": "invalid_grant",
                "error_description": "refresh token expired",
            },
        ),
    )

    with pytest.raises(auth.AuthError) as exc_info:
        await auth.refresh_codex_oauth_pure("old-access", "old-refresh")

    assert exc_info.value.code == "invalid_grant"
    assert exc_info.value.relogin_required is True
    assert "refresh token expired" in str(exc_info.value)


@pytest.mark.asyncio
async def test_resolve_codex_runtime_credentials_preserves_public_contract(
    monkeypatch,
) -> None:
    class _Entry:
        id = "codex-entry"
        access_token = "codex-access"
        runtime_api_key = "codex-access"
        runtime_base_url = auth.DEFAULT_CODEX_BASE_URL
        last_refresh = "2026-08-08T00:00:00Z"

    class _Pool:
        def has_credentials(self):
            return True

        async def select(self):
            return _Entry()

    monkeypatch.setattr(
        "agent.credential_pool.load_pool",
        AsyncMock(return_value=_Pool()),
    )
    monkeypatch.setattr(
        auth,
        "get_provider_auth_state",
        AsyncMock(
            return_value={"tokens": {"access_token": "codex-access"}}
        ),
    )

    credentials = await auth.resolve_codex_runtime_credentials()

    assert credentials == {
        "provider": "openai-codex",
        "base_url": auth.DEFAULT_CODEX_BASE_URL,
        "api_key": "codex-access",
        "source": "hermes-auth-store",
        "last_refresh": "2026-08-08T00:00:00Z",
        "auth_mode": "chatgpt",
    }


@pytest.mark.asyncio
async def test_codex_quota_probe_uses_native_async_transport(monkeypatch) -> None:
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "https://api.openai.com/auth": {
                    "chatgpt_account_id": "account-123",
                }
            }
        ).encode()
    ).decode().rstrip("=")
    token = f"e30.{payload}.signature"
    auth._codex_quota_probe_cache.clear()

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/backend-api/wham/usage"
        assert request.headers["chatgpt-account-id"] == "account-123"
        return httpx.Response(
            200,
            json={
                "rate_limit": {
                    "primary_window": {"used_percent": 25},
                    "secondary_window": {"used_percent": 90},
                }
            },
        )

    _install_transport(monkeypatch, handler)

    assert await auth._probe_codex_quota_restored(token) is True


@pytest.mark.asyncio
async def test_codex_quota_probe_cancellation_does_not_cache_indeterminate(
    monkeypatch,
) -> None:
    payload = base64.urlsafe_b64encode(b'{"sub":"cancel-test"}').decode().rstrip("=")
    token = f"e30.{payload}.signature"
    cache_key = auth.hashlib.sha256(token.encode()).hexdigest()[:16]
    auth._codex_quota_probe_cache.clear()
    request_started = asyncio.Event()

    async def handler(_request: httpx.Request) -> httpx.Response:
        request_started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    _install_transport(monkeypatch, handler)
    task = asyncio.create_task(auth._probe_codex_quota_restored(token))
    await asyncio.wait_for(request_started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cache_key not in auth._codex_quota_probe_cache


@pytest.mark.asyncio
async def test_resolve_codex_clears_stale_quota_cooldown(monkeypatch) -> None:
    class _Entry:
        id = "codex-entry"
        access_token = "codex-access"
        runtime_api_key = "codex-access"
        runtime_base_url = auth.DEFAULT_CODEX_BASE_URL
        last_refresh = "2026-08-08T00:00:00Z"
        last_status = "exhausted"
        last_status_at = time.time()
        last_error_code = 429
        last_error_reason = "usage_limit"
        last_error_message = "quota exhausted"
        last_error_reset_at = time.time() + 3600

    class _ExhaustedPool:
        def has_credentials(self):
            return True

        async def select(self):
            return None

        def entries(self):
            return [_Entry()]

    class _RecoveredPool:
        def has_credentials(self):
            return True

        async def select(self):
            return _Entry()

    load_pool = AsyncMock(side_effect=[_ExhaustedPool(), _RecoveredPool()])
    clear_cooldowns = AsyncMock(return_value=1)
    monkeypatch.setattr("agent.credential_pool.load_pool", load_pool)
    monkeypatch.setattr(
        auth,
        "_probe_codex_quota_restored",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        auth,
        "clear_codex_pool_quota_cooldowns",
        clear_cooldowns,
    )
    monkeypatch.setattr(
        auth,
        "get_provider_auth_state",
        AsyncMock(return_value={}),
    )

    credentials = await auth.resolve_codex_runtime_credentials()

    assert credentials["api_key"] == "codex-access"
    assert credentials["source"] == "credential_pool"
    assert load_pool.await_count == 2
    clear_cooldowns.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_clear_codex_pool_quota_cooldowns_persists_only_rate_limits(
    monkeypatch,
    tmp_path,
) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "version": 1,
                "credential_pool": {
                    "openai-codex": [
                        {
                            "id": "quota-entry",
                            "access_token": "quota-token",
                            "last_status": "exhausted",
                            "last_error_code": 429,
                            "last_error_reason": "usage_limit",
                            "last_error_message": "quota exhausted",
                            "last_error_reset_at": time.time() + 3600,
                        },
                        {
                            "id": "auth-entry",
                            "access_token": "auth-token",
                            "last_status": "exhausted",
                            "last_error_code": 401,
                            "last_error_reason": "invalid_token",
                            "last_error_message": "login required",
                            "last_error_reset_at": time.time() + 3600,
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(auth, "_auth_file_path", AsyncMock(return_value=auth_path))
    monkeypatch.setattr(auth, "_global_auth_file_path", AsyncMock(return_value=None))

    assert await auth.clear_codex_pool_quota_cooldowns() == 1

    stored = json.loads(auth_path.read_text(encoding="utf-8"))
    quota_entry, auth_entry = stored["credential_pool"]["openai-codex"]
    assert quota_entry["last_status"] is None
    assert quota_entry["last_error_code"] is None
    assert auth_entry["last_status"] == "exhausted"
    assert auth_entry["last_error_code"] == 401
