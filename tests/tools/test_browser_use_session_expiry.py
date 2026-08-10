"""Regression coverage for provider-authoritative cloud browser expiry."""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

import tools.browser_tool as browser_tool
from plugins.browser.browser_use import provider as browser_use_provider


def _isolate_browser_state(monkeypatch):
    monkeypatch.setattr(browser_tool, "_active_sessions", {})
    monkeypatch.setattr(browser_tool, "_session_last_activity", {})
    monkeypatch.setattr(
        browser_tool, "_start_browser_cleanup_thread", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        browser_tool, "_ensure_cdp_supervisor", AsyncMock(return_value=None)
    )


@pytest.mark.asyncio
async def test_browser_use_preserves_provider_timeout(monkeypatch):
    provider = browser_use_provider.BrowserUseBrowserProvider()
    monkeypatch.setattr(
        provider,
        "_get_config",
        AsyncMock(return_value={
            "api_key": "test-key",
            "base_url": "https://api.browser-use.example/api/v3",
            "managed_mode": False,
        }),
    )
    real_async_client = httpx.AsyncClient

    async def handler(request):
        return httpx.Response(
            200,
            json={
                "id": "browser-session-1",
                "cdpUrl": "ws://browser-use.example/devtools/browser/1",
                "timeoutAt": "2030-01-01T00:05:00Z",
            },
            request=request,
        )

    def client_factory(**kwargs):
        kwargs.pop("transport", None)
        kwargs.pop("mounts", None)
        return real_async_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(browser_use_provider.httpx, "AsyncClient", client_factory)
    session = await provider.create_session("task-1")
    assert session["expires_at"] == "2030-01-01T00:05:00Z"


@pytest.mark.asyncio
async def test_live_cloud_session_is_reused(monkeypatch):
    _isolate_browser_state(monkeypatch)
    existing = {
        "session_name": "existing",
        "bb_session_id": "browser-session-1",
        "cdp_url": "ws://browser-use.example/devtools/browser/1",
        "expires_at": "2999-01-01T00:05:00Z",
    }
    browser_tool._active_sessions["task-1"] = existing
    provider = Mock()
    provider.create_session = AsyncMock()
    monkeypatch.setattr(
        browser_tool, "_get_cloud_provider", AsyncMock(return_value=provider)
    )

    session = await browser_tool._get_session_info("task-1")
    assert session is existing
    provider.create_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_expired_cloud_session_is_replaced_without_reusing_dead_cdp(
    monkeypatch,
):
    _isolate_browser_state(monkeypatch)
    browser_tool._active_sessions["task-1"] = {
        "session_name": "expired",
        "bb_session_id": "browser-session-old",
        "cdp_url": "ws://browser-use.example/devtools/browser/old",
        "expires_at": "2020-01-01T00:05:00Z",
    }
    browser_tool._session_last_activity["task-1"] = 1.0

    provider = Mock()
    provider.create_session = AsyncMock(return_value={
        "session_name": "replacement",
        "bb_session_id": "browser-session-new",
        "cdp_url": "ws://browser-use.example/devtools/browser/new",
        "expires_at": "2999-01-01T00:05:00Z",
    })
    provider.close_session = AsyncMock(return_value=True)
    monkeypatch.setattr(
        browser_tool, "_get_cloud_provider", AsyncMock(return_value=provider)
    )
    monkeypatch.setattr(
        browser_tool, "_get_cdp_override", AsyncMock(return_value="")
    )
    monkeypatch.setattr(
        browser_tool, "_stop_cdp_supervisor", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        browser_tool, "_maybe_stop_recording", AsyncMock(return_value=None)
    )
    run_command = AsyncMock()
    monkeypatch.setattr(browser_tool, "_run_browser_command", run_command)
    monkeypatch.setattr(
        browser_tool, "_is_camofox_mode", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(
        browser_tool.aiofiles.os.path, "exists", AsyncMock(return_value=False)
    )

    session = await browser_tool._get_session_info("task-1")
    assert session["bb_session_id"] == "browser-session-new"
    assert browser_tool._active_sessions["task-1"] is session
    assert "task-1" in browser_tool._session_last_activity
    provider.close_session.assert_awaited_once_with("browser-session-old")
    provider.create_session.assert_awaited_once_with("task-1")
    run_command.assert_not_awaited()
