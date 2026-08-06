from __future__ import annotations

import asyncio

import httpx
import pytest

from hermes_cli import auth


def _install_transport(monkeypatch, handler) -> None:
    async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        auth.httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=transport, **kwargs),
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
