"""Tests for macOS Homebrew PATH discovery in browser_tool.py."""

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.browser_tool import (
    _discover_homebrew_node_dirs,
    _find_agent_browser,
    _run_browser_command,
    _SANE_PATH,
    check_browser_requirements,
)
import tools.browser_tool as _bt


pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _clear_browser_caches():
    """Clear browser discovery caches between tests."""
    _bt._cached_homebrew_node_dirs = None
    _bt._cached_agent_browser = None
    _bt._agent_browser_resolved = False
    yield
    _bt._cached_homebrew_node_dirs = None
    _bt._cached_agent_browser = None
    _bt._agent_browser_resolved = False


class TestSanePath:
    """Verify _SANE_PATH includes fallback directories used by browser_tool."""

    async def test_includes_termux_bin(self):
        assert "/data/data/com.termux/files/usr/bin" in _SANE_PATH.split(os.pathsep)


    async def test_includes_standard_dirs(self):
        path_parts = _SANE_PATH.split(os.pathsep)
        assert "/usr/local/bin" in path_parts
        assert "/usr/bin" in path_parts
        assert "/bin" in path_parts


class TestDiscoverHomebrewNodeDirs:
    """Tests for _discover_homebrew_node_dirs()."""

    async def test_returns_empty_when_no_homebrew(self):
        """Non-macOS systems without /opt/homebrew/opt should return empty."""
        with patch(
            "aiofiles.os.path.isdir", new=AsyncMock(return_value=False)
        ):
            assert await _discover_homebrew_node_dirs() == ()


    async def test_excludes_plain_node(self):
        """'node' (unversioned) should be excluded — covered by /opt/homebrew/bin."""
        with patch(
            "aiofiles.os.path.isdir", new=AsyncMock(return_value=True)
        ), patch("aiofiles.os.listdir", new=AsyncMock(return_value=["node"])):
            result = await _discover_homebrew_node_dirs()
        assert result == ()

    async def test_handles_oserror_gracefully(self):
        """Should return empty list if listdir raises OSError."""
        with patch(
            "aiofiles.os.path.isdir", new=AsyncMock(return_value=True)
        ), patch(
            "aiofiles.os.listdir",
            new=AsyncMock(side_effect=OSError("Permission denied")),
        ):
            assert await _discover_homebrew_node_dirs() == ()


class TestFindAgentBrowser:
    """Tests for _find_agent_browser() Homebrew path search."""

    async def test_finds_in_current_path(self):
        """Should return result from shutil.which if available on current PATH."""
        with patch("shutil.which", return_value="/usr/local/bin/agent-browser"), \
             patch(
                 "tools.browser_tool.agent_browser_runnable",
                 new_callable=AsyncMock,
                 return_value=True,
             ):
            assert await _find_agent_browser() == "/usr/local/bin/agent-browser"


    async def test_raises_when_not_found(self):
        """Should raise FileNotFoundError when nothing works."""
        original_path_exists = Path.exists

        def mock_path_exists(self):
            if "node_modules" in str(self) and "agent-browser" in str(self):
                return False
            return original_path_exists(self)

        with patch("shutil.which", return_value=None), \
             patch("aiofiles.os.path.isdir", new=AsyncMock(return_value=False)), \
             patch.object(Path, "exists", mock_path_exists), \
             patch(
                 "tools.browser_tool._discover_homebrew_node_dirs",
                 new=AsyncMock(return_value=[]),
             ):
            with pytest.raises(FileNotFoundError, match="agent-browser CLI not found"):
                await _find_agent_browser()


class TestBrowserRequirements:
    async def test_cdp_override_does_not_require_agent_browser_cli(self, monkeypatch):
        monkeypatch.setenv("BROWSER_CDP_URL", "ws://127.0.0.1:9222/devtools/browser/test")
        monkeypatch.setattr("tools.browser_tool._is_camofox_mode", AsyncMock(return_value=False))
        monkeypatch.setattr(
            "tools.browser_tool._find_agent_browser",
            AsyncMock(side_effect=FileNotFoundError("not found")),
        )

        assert await check_browser_requirements() is True

    async def test_termux_requires_real_agent_browser_install_not_npx_fallback(self, monkeypatch):
        monkeypatch.setenv("TERMUX_VERSION", "0.118.3")
        monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
        monkeypatch.setattr("tools.browser_tool._is_camofox_mode", AsyncMock(return_value=False))
        monkeypatch.setattr("tools.browser_tool._get_cloud_provider", AsyncMock(return_value=None))
        monkeypatch.setattr("tools.browser_tool._find_agent_browser", AsyncMock(return_value="npx agent-browser"))

        assert await check_browser_requirements() is False


class TestRunBrowserCommandTermuxFallback:
    async def test_termux_local_mode_rejects_bare_npx_fallback(self, monkeypatch):
        monkeypatch.setenv("TERMUX_VERSION", "0.118.3")
        monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
        monkeypatch.setattr("tools.browser_tool._find_agent_browser", AsyncMock(return_value="npx agent-browser"))
        monkeypatch.setattr("tools.browser_tool._get_cloud_provider", AsyncMock(return_value=None))

        result = await _run_browser_command("task-1", "navigate", ["https://example.com"])

        assert result["success"] is False
        assert "bare npx fallback" in result["error"]
        assert "agent-browser install" in result["error"]


async def test_run_browser_command_repeated_cancellation_reaps_and_unlinks(
    tmp_path, monkeypatch
):
    wait_started = asyncio.Event()
    release_wait = asyncio.Event()
    wait_completed = asyncio.Event()

    class BlockingProcess:
        returncode = None
        killed = False

        async def wait(self):
            wait_started.set()
            await release_wait.wait()
            wait_completed.set()
            return self.returncode

        def kill(self):
            self.killed = True
            self.returncode = -9

    process = BlockingProcess()

    async def create_process(*_args, **_kwargs):
        return process

    monkeypatch.setattr(
        _bt, "_find_agent_browser", AsyncMock(return_value="/x/agent-browser")
    )
    monkeypatch.setattr(
        _bt, "_requires_real_termux_browser_install", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(_bt, "_is_local_mode", AsyncMock(return_value=False))
    monkeypatch.setattr(
        _bt,
        "_get_session_info",
        AsyncMock(return_value={"session_name": "cancel-session"}),
    )
    monkeypatch.setattr(_bt, "_is_headed_mode", AsyncMock(return_value=False))
    monkeypatch.setattr(_bt, "_get_browser_engine", AsyncMock(return_value="auto"))
    monkeypatch.setattr(_bt, "_is_camofox_mode", AsyncMock(return_value=False))
    monkeypatch.setattr(_bt, "_write_owner_pid", AsyncMock())
    monkeypatch.setattr(
        _bt, "_build_browser_env", AsyncMock(return_value={"PATH": "/usr/bin"})
    )
    monkeypatch.setattr(
        _bt, "_merge_browser_path", AsyncMock(return_value="/usr/bin")
    )
    monkeypatch.setattr(
        _bt, "_needs_chromium_sandbox_bypass", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(_bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

    task = asyncio.create_task(
        _run_browser_command("cancel-task", "snapshot", [], timeout=60)
    )
    await wait_started.wait()
    task.cancel()
    while not process.killed:
        await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)

    assert task.done() is False
    release_wait.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert wait_completed.is_set()
    output_dir = tmp_path / "agent-browser-cancel-session"
    assert not (output_dir / "_stdout_snapshot").exists()
    assert not (output_dir / "_stderr_snapshot").exists()


class TestRunBrowserCommandPathConstruction:
    """Verify _run_browser_command() includes Homebrew node dirs in subprocess PATH."""

    async def test_subprocess_preserves_executable_path_with_spaces(self, tmp_path):
        """A local agent-browser path containing spaces must stay one argv entry."""
        captured_cmd = None

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.wait = AsyncMock(return_value=0)

        async def capture_popen(*cmd, **kwargs):
            nonlocal captured_cmd
            captured_cmd = list(cmd)
            return mock_proc

        fake_session = {
            "session_name": "test-session",
            "session_id": "test-id",
            "cdp_url": None,
        }
        browser_path = "/Users/test/Library/Application Support/hermes/node_modules/.bin/agent-browser"
        hermes_home = str(tmp_path / "hermes-home")

        with patch("tools.browser_tool._find_agent_browser", new_callable=AsyncMock, return_value=browser_path), \
             patch("tools.browser_tool._is_local_mode", new_callable=AsyncMock, return_value=False), \
             patch("tools.browser_tool._get_session_info", new_callable=AsyncMock, return_value=fake_session), \
             patch("tools.browser_tool._is_headed_mode", new_callable=AsyncMock, return_value=False), \
             patch("tools.browser_tool._get_browser_engine", new_callable=AsyncMock, return_value="auto"), \
             patch("tools.browser_tool._is_camofox_mode", new_callable=AsyncMock, return_value=False), \
             patch("tools.browser_tool._needs_chromium_sandbox_bypass", new_callable=AsyncMock, return_value=False), \
             patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)), \
             patch("tools.browser_tool._discover_homebrew_node_dirs", new=AsyncMock(return_value=[])), \
             patch("hermes_constants.Path.home", return_value=tmp_path), \
             patch("asyncio.create_subprocess_exec", side_effect=capture_popen), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch.dict(
                 os.environ,
                 {
                     "PATH": "/usr/bin:/bin",
                     "HOME": "/home/test",
                     "HERMES_HOME": hermes_home,
                 },
                 clear=True,
             ):
            await _run_browser_command("test-task", "navigate", ["https://example.com"])

        assert captured_cmd is not None
        assert captured_cmd[0] == browser_path
        assert captured_cmd[1:5] == [
            "--session",
            "test-session",
            "--json",
            "navigate",
        ]


    async def test_subprocess_path_includes_termux_fallback_dirs(self, tmp_path):
        """Termux fallback dirs should survive browser PATH rebuilding."""
        captured_env = {}

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.wait = AsyncMock(return_value=0)

        async def capture_popen(*cmd, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            return mock_proc

        fake_session = {
            "session_name": "test-session",
            "session_id": "test-id",
            "cdp_url": None,
        }

        real_isdir = os.path.isdir

        def selective_isdir(path):
            if path in {
                "/data/data/com.termux/files/usr/bin",
                "/data/data/com.termux/files/usr/sbin",
            }:
                return True
            if path.startswith(str(tmp_path)):
                return True
            return real_isdir(path)

        with patch("tools.browser_tool._find_agent_browser", new_callable=AsyncMock, return_value="/usr/local/bin/agent-browser"), \
             patch("tools.browser_tool._is_local_mode", new_callable=AsyncMock, return_value=False), \
             patch("tools.browser_tool._get_session_info", new_callable=AsyncMock, return_value=fake_session), \
             patch("tools.browser_tool._is_headed_mode", new_callable=AsyncMock, return_value=False), \
             patch("tools.browser_tool._get_browser_engine", new_callable=AsyncMock, return_value="auto"), \
             patch("tools.browser_tool._is_camofox_mode", new_callable=AsyncMock, return_value=False), \
             patch("tools.browser_tool._needs_chromium_sandbox_bypass", new_callable=AsyncMock, return_value=False), \
             patch("tools.browser_tool._socket_safe_tmpdir", return_value=str(tmp_path)), \
             patch("tools.browser_tool._discover_homebrew_node_dirs", new=AsyncMock(return_value=[])), \
             patch("aiofiles.os.path.isdir", new=AsyncMock(side_effect=selective_isdir)), \
             patch("asyncio.create_subprocess_exec", side_effect=capture_popen), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch.dict(os.environ, {"PATH": "/usr/bin:/bin", "HOME": "/home/test"}, clear=True):
            await _run_browser_command("test-task", "navigate", ["https://example.com"])

        result_path = captured_env.get("PATH", "")
        assert "/data/data/com.termux/files/usr/bin" in result_path
        assert "/data/data/com.termux/files/usr/sbin" in result_path
