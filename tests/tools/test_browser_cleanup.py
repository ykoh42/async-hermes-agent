"""Regression tests for browser session cleanup and screenshot recovery."""

from unittest.mock import AsyncMock, patch

import pytest


class TestScreenshotPathRecovery:
    def test_extracts_standard_absolute_path(self):
        from tools.browser_tool import _extract_screenshot_path_from_text

        assert (
            _extract_screenshot_path_from_text("Screenshot saved to /tmp/foo.png")
            == "/tmp/foo.png"
        )

    def test_extracts_quoted_absolute_path(self):
        from tools.browser_tool import _extract_screenshot_path_from_text

        assert (
            _extract_screenshot_path_from_text(
                "Screenshot saved to '/Users/david/.hermes/browser_screenshots/shot.png'"
            )
            == "/Users/david/.hermes/browser_screenshots/shot.png"
        )


class TestBrowserCleanup:
    pytestmark = pytest.mark.asyncio

    def setup_method(self):
        from tools import browser_tool

        self.browser_tool = browser_tool
        self.orig_active_sessions = browser_tool._active_sessions.copy()
        self.orig_session_last_activity = browser_tool._session_last_activity.copy()
        self.orig_recording_sessions = browser_tool._recording_sessions.copy()
        self.orig_cleanup_done = browser_tool._cleanup_done

    def teardown_method(self):
        self.browser_tool._active_sessions.clear()
        self.browser_tool._active_sessions.update(self.orig_active_sessions)
        self.browser_tool._session_last_activity.clear()
        self.browser_tool._session_last_activity.update(self.orig_session_last_activity)
        self.browser_tool._recording_sessions.clear()
        self.browser_tool._recording_sessions.update(self.orig_recording_sessions)
        self.browser_tool._cleanup_done = self.orig_cleanup_done

    async def test_cleanup_browser_clears_tracking_state(self):
        browser_tool = self.browser_tool
        browser_tool._active_sessions["task-1"] = {
            "session_name": "sess-1",
            "bb_session_id": None,
        }
        browser_tool._session_last_activity["task-1"] = 123.0

        with (
            patch(
                "tools.browser_tool._maybe_stop_recording", new_callable=AsyncMock
            ) as mock_stop,
            patch(
                "tools.browser_tool._run_browser_command",
                new=AsyncMock(return_value={"success": True}),
            ) as mock_run,
            patch(
                "tools.browser_tool.aiofiles.os.path.exists",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "tools.browser_tool._stop_cdp_supervisor", new_callable=AsyncMock
            ),
            patch(
                "tools.browser_tool._is_camofox_mode",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "tools.browser_tool._stop_browser_cleanup_thread",
                new_callable=AsyncMock,
            ),
        ):
            await browser_tool.cleanup_browser("task-1")

        assert "task-1" not in browser_tool._active_sessions
        assert "task-1" not in browser_tool._session_last_activity
        mock_stop.assert_awaited_once_with("task-1")
        mock_run.assert_awaited_once_with("task-1", "close", [], timeout=10)


    async def test_emergency_cleanup_clears_all_tracking_state(self):
        browser_tool = self.browser_tool
        browser_tool._cleanup_done = False
        browser_tool._active_sessions["task-1"] = {"session_name": "sess-1"}
        browser_tool._active_sessions["task-2"] = {"session_name": "sess-2"}
        browser_tool._session_last_activity["task-1"] = 1.0
        browser_tool._session_last_activity["task-2"] = 2.0
        browser_tool._recording_sessions.update({"task-1", "task-2"})

        with patch(
            "tools.browser_tool.cleanup_all_browsers", new_callable=AsyncMock
        ) as mock_cleanup_all:
            await browser_tool._emergency_cleanup_all_sessions()

        mock_cleanup_all.assert_awaited_once_with()
        assert browser_tool._active_sessions == {}
        assert browser_tool._session_last_activity == {}
        assert browser_tool._recording_sessions == set()
        assert browser_tool._cleanup_done is True
