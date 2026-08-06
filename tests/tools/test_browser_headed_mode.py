"""Tests for headed browser mode: config/env resolution, --headed injection,
and the per-turn cleanup skip that keeps headed sessions alive between turns.

Salvaged from PR #24064 (fixes #11020 lead bug).
"""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


pytestmark = pytest.mark.asyncio


def _reset_headed_cache():
    """Reset the module-level headed-mode cache so tests start clean."""
    import tools.browser_tool as bt
    bt._cached_headed_mode = None
    bt._headed_mode_resolved = False


@pytest.fixture(autouse=True)
def _clean_headed_cache():
    _reset_headed_cache()
    yield
    _reset_headed_cache()


# ---------------------------------------------------------------------------
# _is_headed_mode resolution
# ---------------------------------------------------------------------------

class TestIsHeadedMode:
    async def test_default_is_false(self):
        from tools.browser_tool import _is_headed_mode
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AGENT_BROWSER_HEADED", None)
            with patch(
                "tools.browser_tool.load_config_readonly",
                new_callable=AsyncMock,
                return_value={},
            ):
                assert await _is_headed_mode() is False

    async def test_config_true(self):
        from tools.browser_tool import _is_headed_mode
        cfg = {"browser": {"headed": True}}
        with patch(
            "tools.browser_tool.load_config_readonly",
            new_callable=AsyncMock,
            return_value=cfg,
        ):
            assert await _is_headed_mode() is True


    async def test_caching(self):
        from tools.browser_tool import _is_headed_mode
        cfg = {"browser": {"headed": True}}
        with patch(
            "tools.browser_tool.load_config_readonly",
            new_callable=AsyncMock,
            return_value=cfg,
        ) as mock_read:
            assert await _is_headed_mode() is True
            assert await _is_headed_mode() is True
            mock_read.assert_awaited_once()


# ---------------------------------------------------------------------------
# Per-turn cleanup skip (agent/chat_completion_helpers.cleanup_task_resources)
# ---------------------------------------------------------------------------

def _make_agent(verbose=False):
    return SimpleNamespace(verbose_logging=verbose)


class TestCleanupTaskResourcesHeadedSkip:
    async def test_headless_still_cleans_browser(self):
        from agent.chat_completion_helpers import cleanup_task_resources
        with (
            patch("tools.browser_tool._is_headed_mode", new_callable=AsyncMock, return_value=False),
            patch("agent.chat_completion_helpers.cleanup_vm", new_callable=AsyncMock),
            patch("tools.browser_tool.cleanup_browser", new_callable=AsyncMock) as mock_cb,
            patch(
                "agent.chat_completion_helpers.is_persistent_env",
                return_value=False,
            ),
        ):
            await cleanup_task_resources(_make_agent(), "task-x")
            mock_cb.assert_awaited_once_with("task-x")


    async def test_headed_does_not_skip_vm_cleanup(self):
        """Headed mode only affects the browser; VM teardown is untouched."""
        from agent.chat_completion_helpers import cleanup_task_resources
        with (
            patch("tools.browser_tool._is_headed_mode", new_callable=AsyncMock, return_value=True),
            patch("agent.chat_completion_helpers.cleanup_vm", new_callable=AsyncMock) as mock_vm,
            patch("tools.browser_tool.cleanup_browser", new_callable=AsyncMock),
            patch(
                "agent.chat_completion_helpers.is_persistent_env",
                return_value=False,
            ),
        ):
            await cleanup_task_resources(_make_agent(), "task-x")
            mock_vm.assert_awaited_once_with("task-x")


# ---------------------------------------------------------------------------
# --headed flag injection in local mode
# ---------------------------------------------------------------------------

class TestHeadedFlagInjection:
    async def _run_and_capture(self, bt):
        """Run a snapshot command with subprocess creation mocked."""
        captured_cmds = []
        mock_proc = MagicMock()
        mock_proc.wait = AsyncMock(return_value=None)
        mock_proc.returncode = 0

        async def capture_subprocess(*cmd, **kwargs):
            captured_cmds.append(list(cmd))
            return mock_proc

        with patch("asyncio.create_subprocess_exec", side_effect=capture_subprocess), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("tools.browser_tool._write_owner_pid", new_callable=AsyncMock), \
             patch("tools.browser_tool._needs_chromium_sandbox_bypass", new_callable=AsyncMock, return_value=False):
            await bt._run_browser_command("task1", "snapshot", [], _engine_override="auto")
        return captured_cmds

    @patch("tools.browser_tool._get_session_info", new_callable=AsyncMock)
    @patch("tools.browser_tool._find_agent_browser", new_callable=AsyncMock, return_value="/usr/bin/agent-browser")
    @patch("tools.browser_tool._is_local_mode", new_callable=AsyncMock, return_value=False)
    @patch("tools.browser_tool._get_browser_engine", new_callable=AsyncMock, return_value="auto")
    @patch("tools.browser_tool._is_camofox_mode", new_callable=AsyncMock, return_value=False)
    async def test_headed_flag_added_in_local_mode(
        self, _camofox, _engine, _local, _find, _session
    ):
        import tools.browser_tool as bt
        bt._cached_headed_mode = True
        bt._headed_mode_resolved = True
        _session.return_value = {"session_name": "test-sess"}

        captured = await self._run_and_capture(bt)
        assert len(captured) == 1
        assert "--headed" in captured[0]


    @patch("tools.browser_tool._get_session_info", new_callable=AsyncMock)
    @patch("tools.browser_tool._find_agent_browser", new_callable=AsyncMock, return_value="/usr/bin/agent-browser")
    @patch("tools.browser_tool._is_local_mode", new_callable=AsyncMock, return_value=False)
    @patch("tools.browser_tool._get_browser_engine", new_callable=AsyncMock, return_value="auto")
    @patch("tools.browser_tool._is_camofox_mode", new_callable=AsyncMock, return_value=False)
    async def test_headed_flag_not_added_in_cloud_mode(
        self, _camofox, _engine, _local, _find, _session
    ):
        """Cloud (CDP) sessions never get --headed — it's a local-only flag."""
        import tools.browser_tool as bt
        bt._cached_headed_mode = True
        bt._headed_mode_resolved = True
        _session.return_value = {
            "session_name": "test-sess",
            "cdp_url": "wss://example.invalid/cdp",
        }

        captured = await self._run_and_capture(bt)
        assert len(captured) == 1
        assert "--headed" not in captured[0]
        assert "--cdp" in captured[0]
