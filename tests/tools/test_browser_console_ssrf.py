"""Tests that browser_console blocks console messages and errors from eval-navigated private pages.

browser_snapshot, browser_vision, _browser_eval, and browser_get_images all re-check
the page URL before returning content. browser_console (in console output mode) must
do the same to prevent leakage of console log messages and exception details.
"""

import json
from unittest.mock import AsyncMock

import pytest

from tools import browser_tool

PRIVATE_URL = "http://127.0.0.1:8080/internal"


@pytest.fixture(autouse=True)
def _patches(monkeypatch):
    monkeypatch.setattr(
        browser_tool, "_is_camofox_mode", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(browser_tool, "_last_session_key", lambda key: key)


def _mock_run_success(monkeypatch):
    async def _run(task_id, command, args=None, **kwargs):
        if command == "console":
            return {
                "success": True,
                "data": {
                    "messages": [
                        {"type": "log", "text": "secret internal message"}
                    ]
                }
            }
        elif command == "errors":
            return {
                "success": True,
                "data": {
                    "errors": [
                        {"message": "internal exception info"}
                    ]
                }
            }
        return {"success": True, "data": {}}
    monkeypatch.setattr(browser_tool, "_run_browser_command", _run)


@pytest.mark.asyncio
async def test_blocks_console_on_private_page(monkeypatch):
    _mock_run_success(monkeypatch)
    monkeypatch.setattr(
        browser_tool, "_eval_ssrf_guard_active", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        browser_tool, "_current_page_private_url", AsyncMock(return_value=PRIVATE_URL)
    )

    result = json.loads(await browser_tool.browser_console(task_id="test"))
    assert result["success"] is False
    assert "private or internal address" in result["error"]
    assert PRIVATE_URL in result["error"]


@pytest.mark.asyncio
async def test_allows_console_on_public_page(monkeypatch):
    _mock_run_success(monkeypatch)
    monkeypatch.setattr(
        browser_tool, "_eval_ssrf_guard_active", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        browser_tool, "_current_page_private_url", AsyncMock(return_value=None)
    )

    result = json.loads(await browser_tool.browser_console(task_id="test"))
    assert result["success"] is True
    assert result["total_messages"] == 1
    assert result["console_messages"][0]["text"] == "secret internal message"


@pytest.mark.asyncio
async def test_skips_guard_for_local_backend(monkeypatch):
    _mock_run_success(monkeypatch)
    monkeypatch.setattr(
        browser_tool, "_eval_ssrf_guard_active", AsyncMock(return_value=False)
    )

    result = json.loads(await browser_tool.browser_console(task_id="test"))
    assert result["success"] is True
    assert result["total_messages"] == 1


@pytest.mark.asyncio
async def test_skips_guard_when_private_urls_allowed(monkeypatch):
    _mock_run_success(monkeypatch)
    monkeypatch.setattr(
        browser_tool, "_eval_ssrf_guard_active", AsyncMock(return_value=False)
    )

    result = json.loads(await browser_tool.browser_console(task_id="test"))
    assert result["success"] is True
    assert result["total_messages"] == 1


@pytest.mark.asyncio
async def test_guard_does_not_block_on_failed_console_command(monkeypatch):
    """If the console command itself fails, browser_console returns the error naturally."""
    async def _run(task_id, command, args=None, **kwargs):
        return {"success": False, "error": "console fetch failed"}
    monkeypatch.setattr(browser_tool, "_run_browser_command", _run)
    monkeypatch.setattr(
        browser_tool, "_eval_ssrf_guard_active", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        browser_tool, "_current_page_private_url", AsyncMock(return_value=PRIVATE_URL)
    )

    result = json.loads(await browser_tool.browser_console(task_id="test"))
    # When the page is private, the guard checks _current_page_private_url first.
    # Because it checks _current_page_private_url BEFORE running the command, it should block it.
    assert result["success"] is False
    assert "private or internal address" in result["error"]
