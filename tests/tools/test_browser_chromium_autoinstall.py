"""Tests for gated native-async Chromium auto-install."""

from unittest.mock import AsyncMock

import pytest

import tools.browser_tool as browser_tool


class _Process:
    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.returncode = returncode
        self._result = (stdout, stderr)

    async def communicate(self):
        return self._result

    def kill(self):
        self.returncode = -9

    async def wait(self):
        return self.returncode


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    browser_tool._chromium_autoinstall_attempted = False
    browser_tool._cached_chromium_installed = None
    yield
    browser_tool._chromium_autoinstall_attempted = False
    browser_tool._cached_chromium_installed = None


class TestGating:
    pytestmark = pytest.mark.asyncio

    async def test_disabled_lazy_installs_skips(self, monkeypatch):
        monkeypatch.setattr(
            browser_tool, "_running_in_docker", AsyncMock(return_value=False)
        )
        monkeypatch.setattr(
            browser_tool,
            "load_config_readonly",
            AsyncMock(return_value={"security": {"allow_lazy_installs": False}}),
        )
        create = AsyncMock()
        monkeypatch.setattr(browser_tool.asyncio, "create_subprocess_exec", create)
        assert await browser_tool._maybe_autoinstall_chromium() is False
        create.assert_not_awaited()

    async def test_docker_skips(self, monkeypatch):
        monkeypatch.setattr(
            browser_tool, "_running_in_docker", AsyncMock(return_value=True)
        )
        create = AsyncMock()
        monkeypatch.setattr(browser_tool.asyncio, "create_subprocess_exec", create)
        assert await browser_tool._maybe_autoinstall_chromium() is False
        create.assert_not_awaited()


class TestInstall:
    pytestmark = pytest.mark.asyncio

    async def test_success_installs_binary_only_and_rechecks(self, monkeypatch):
        monkeypatch.setattr(
            browser_tool, "_running_in_docker", AsyncMock(return_value=False)
        )
        monkeypatch.setattr(
            browser_tool, "load_config_readonly", AsyncMock(return_value={})
        )
        monkeypatch.setattr(
            browser_tool,
            "_find_agent_browser",
            AsyncMock(return_value="/x/agent-browser"),
        )
        installed = AsyncMock(return_value=True)
        monkeypatch.setattr(browser_tool, "_chromium_installed", installed)
        create = AsyncMock(return_value=_Process())
        monkeypatch.setattr(browser_tool.asyncio, "create_subprocess_exec", create)

        assert await browser_tool._maybe_autoinstall_chromium() is True
        command = create.await_args.args
        assert command[:2] == ("/x/agent-browser", "install")
        assert "--with-deps" not in command
        installed.assert_awaited_once()

    async def test_nonzero_exit_returns_false(self, monkeypatch):
        monkeypatch.setattr(
            browser_tool, "_running_in_docker", AsyncMock(return_value=False)
        )
        monkeypatch.setattr(
            browser_tool, "load_config_readonly", AsyncMock(return_value={})
        )
        monkeypatch.setattr(
            browser_tool,
            "_find_agent_browser",
            AsyncMock(return_value="/x/agent-browser"),
        )
        monkeypatch.setattr(
            browser_tool.asyncio,
            "create_subprocess_exec",
            AsyncMock(return_value=_Process(returncode=1, stderr=b"boom")),
        )
        assert await browser_tool._maybe_autoinstall_chromium() is False


class TestOneShot:
    pytestmark = pytest.mark.asyncio

    async def test_second_call_does_not_reinstall(self, monkeypatch):
        monkeypatch.setattr(
            browser_tool, "_running_in_docker", AsyncMock(return_value=False)
        )
        monkeypatch.setattr(
            browser_tool, "load_config_readonly", AsyncMock(return_value={})
        )
        monkeypatch.setattr(
            browser_tool,
            "_find_agent_browser",
            AsyncMock(return_value="/x/agent-browser"),
        )
        monkeypatch.setattr(
            browser_tool, "_chromium_installed", AsyncMock(return_value=True)
        )
        create = AsyncMock(return_value=_Process())
        monkeypatch.setattr(browser_tool.asyncio, "create_subprocess_exec", create)

        assert await browser_tool._maybe_autoinstall_chromium() is True
        assert await browser_tool._maybe_autoinstall_chromium() is True
        assert create.await_count == 1
