"""Tests for Chromium-presence detection in browser_tool.

Regression guard for the "browser tool advertised but Chromium missing"
class of bug — where ``agent-browser`` CLI is discoverable but no
Chromium build is on disk, causing every browser_* tool call to hang
for the full command timeout before surfacing a useless error.
"""

import os

import pytest

from tools import browser_tool as bt


@pytest.fixture(autouse=True)
def _reset_chromium_cache():
    bt._cached_chromium_installed = None
    yield
    bt._cached_chromium_installed = None


class TestChromiumSearchRoots:
    def test_respects_playwright_browsers_path_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        roots = bt._chromium_search_roots()
        assert str(tmp_path) == roots[0]


    def test_always_includes_default_ms_playwright_cache(self, monkeypatch):
        monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH", raising=False)
        roots = bt._chromium_search_roots()
        home = os.path.expanduser("~")
        assert any(r == os.path.join(home, ".cache", "ms-playwright") for r in roots)


class TestChromiumInstalled:
    pytestmark = pytest.mark.asyncio

    async def test_true_when_plain_chromium_on_path(self, monkeypatch):
        monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
        monkeypatch.setattr(
            bt.shutil,
            "which",
            lambda name: "/usr/bin/chromium" if name == "chromium" else None,
        )

        assert await bt._chromium_installed() is True


    async def test_result_cached(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        (tmp_path / "chromium-1208").mkdir()
        assert await bt._chromium_installed() is True
        # Delete after first call — cached True should still return True.
        (tmp_path / "chromium-1208").rmdir()
        assert await bt._chromium_installed() is True


class TestCheckBrowserRequirementsChromium:
    pytestmark = pytest.mark.asyncio

    async def test_local_mode_with_chromium_returns_true(self, monkeypatch, tmp_path):
        from unittest.mock import AsyncMock

        monkeypatch.setattr(bt, "_is_camofox_mode", AsyncMock(return_value=False))
        monkeypatch.setattr(
            bt,
            "_find_agent_browser",
            AsyncMock(return_value="/usr/local/bin/agent-browser"),
        )
        monkeypatch.setattr(
            bt, "_requires_real_termux_browser_install", AsyncMock(return_value=False)
        )
        monkeypatch.setattr(bt, "_get_cloud_provider", AsyncMock(return_value=None))
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        (tmp_path / "chromium-1208").mkdir()

        assert await bt.check_browser_requirements() is True


    async def test_camofox_mode_does_not_require_chromium(self, monkeypatch, tmp_path):
        from unittest.mock import AsyncMock

        monkeypatch.setattr(bt, "_is_camofox_mode", AsyncMock(return_value=True))
        # Even with no chromium on disk, camofox drives its own backend.
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
        monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path / "fakehome"))

        assert await bt.check_browser_requirements() is True


class TestRunBrowserCommandChromiumGuard:
    """Verify _run_browser_command fails fast (no timeout hang) when
    Chromium is missing in local mode.
    """
