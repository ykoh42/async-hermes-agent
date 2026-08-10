"""Tests for Lightpanda engine support in browser_tool.py."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from blockbuster import BlockBuster


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_engine_cache():
    """Reset the module-level engine cache so tests start clean."""
    import tools.browser_tool as bt
    bt._cached_browser_engine = None
    bt._browser_engine_resolved = False


@pytest.fixture(autouse=True)
def _clean_engine_cache():
    """Reset engine cache before and after each test."""
    _reset_engine_cache()
    yield
    _reset_engine_cache()


# ---------------------------------------------------------------------------
# _get_browser_engine
# ---------------------------------------------------------------------------

class TestGetBrowserEngine:
    """Test engine resolution from config and env vars."""

    async def test_default_is_auto(self):
        """With no config or env var, engine defaults to 'auto'."""
        from tools.browser_tool import _get_browser_engine
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AGENT_BROWSER_ENGINE", None)
            with patch("tools.browser_tool.load_config_readonly", new_callable=AsyncMock, return_value={}):
                assert await _get_browser_engine() == "auto"

    async def test_config_lightpanda(self):
        """Config browser.engine = 'lightpanda' is respected."""
        from tools.browser_tool import _get_browser_engine
        cfg = {"browser": {"engine": "lightpanda"}}
        with patch("tools.browser_tool.load_config_readonly", new_callable=AsyncMock, return_value=cfg):
            assert await _get_browser_engine() == "lightpanda"


    async def test_caching(self):
        """Result is cached — second call doesn't re-read config."""
        from tools.browser_tool import _get_browser_engine
        mock_read = AsyncMock(return_value={"browser": {"engine": "lightpanda"}})
        with patch("tools.browser_tool.load_config_readonly", mock_read):
            assert await _get_browser_engine() == "lightpanda"
            assert await _get_browser_engine() == "lightpanda"
            mock_read.assert_awaited_once()


# ---------------------------------------------------------------------------
# _should_inject_engine
# ---------------------------------------------------------------------------

class TestShouldInjectEngine:
    """Test whether --engine flag is injected based on mode."""

    async def test_auto_never_injects(self):
        from tools.browser_tool import _should_inject_engine
        assert await _should_inject_engine("auto") is False

    async def test_lightpanda_injects_in_local_mode(self):
        from tools.browser_tool import _should_inject_engine
        with patch("tools.browser_tool._is_camofox_mode", new_callable=AsyncMock, return_value=False), \
             patch("tools.browser_tool._get_cdp_override", new_callable=AsyncMock, return_value=""), \
             patch("tools.browser_tool._get_cloud_provider", new_callable=AsyncMock, return_value=None):
            assert await _should_inject_engine("lightpanda") is True

    async def test_chrome_injects_in_local_mode(self):
        from tools.browser_tool import _should_inject_engine
        with patch("tools.browser_tool._is_camofox_mode", new_callable=AsyncMock, return_value=False), \
             patch("tools.browser_tool._get_cdp_override", new_callable=AsyncMock, return_value=""), \
             patch("tools.browser_tool._get_cloud_provider", new_callable=AsyncMock, return_value=None):
            assert await _should_inject_engine("chrome") is True

    async def test_no_inject_in_camofox_mode(self):
        from tools.browser_tool import _should_inject_engine
        with patch("tools.browser_tool._is_camofox_mode", new_callable=AsyncMock, return_value=True):
            assert await _should_inject_engine("lightpanda") is False

    async def test_no_inject_with_cdp_override(self):
        from tools.browser_tool import _should_inject_engine
        with patch("tools.browser_tool._is_camofox_mode", new_callable=AsyncMock, return_value=False), \
             patch("tools.browser_tool._get_cdp_override_raw", new_callable=AsyncMock, return_value="ws://localhost:9222"):
            assert await _should_inject_engine("lightpanda") is False


# ---------------------------------------------------------------------------
# _needs_lightpanda_fallback
# ---------------------------------------------------------------------------

class TestNeedsLightpandaFallback:
    """Test fallback detection for Lightpanda results."""

    async def test_non_lightpanda_never_falls_back(self):
        from tools.browser_tool import _needs_lightpanda_fallback
        result = {"success": False, "error": "timeout"}
        assert await _needs_lightpanda_fallback("chrome", "open", result) is False
        assert await _needs_lightpanda_fallback("auto", "open", result) is False

    async def test_failed_command_triggers_fallback(self):
        from tools.browser_tool import _needs_lightpanda_fallback
        result = {"success": False, "error": "page.goto: Timeout"}
        assert await _needs_lightpanda_fallback("lightpanda", "open", result) is True


    async def test_empty_snapshot_triggers_fallback(self):
        from tools.browser_tool import _needs_lightpanda_fallback
        result = {"success": True, "data": {"snapshot": ""}}
        assert await _needs_lightpanda_fallback("lightpanda", "snapshot", result) is True


    async def test_unknown_command_does_not_trigger_fallback(self):
        """Commands not in the whitelist should not trigger fallback."""
        from tools.browser_tool import _needs_lightpanda_fallback
        result = {"success": False, "error": "nope"}
        assert await _needs_lightpanda_fallback("lightpanda", "some_future_cmd", result) is False

    async def test_small_screenshot_probe_does_not_block(self, tmp_path):
        from tools.browser_tool import _lightpanda_fallback_reason

        screenshot = tmp_path / "lightpanda.png"
        screenshot.write_bytes(b"x" * 1024)
        result = {"success": True, "data": {"path": str(screenshot)}}

        blocker = BlockBuster()
        blocker.activate()
        try:
            reason = await _lightpanda_fallback_reason(
                "lightpanda", "screenshot", result
            )
        finally:
            blocker.deactivate()

        assert "suspiciously small" in reason


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    """Verify engine config is in DEFAULT_CONFIG."""

    async def test_engine_in_default_config(self):
        from hermes_cli.config import DEFAULT_CONFIG
        assert "engine" in DEFAULT_CONFIG["browser"]
        assert DEFAULT_CONFIG["browser"]["engine"] == "auto"

    async def test_env_var_registered(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS
        assert "AGENT_BROWSER_ENGINE" in OPTIONAL_ENV_VARS
        entry = OPTIONAL_ENV_VARS["AGENT_BROWSER_ENGINE"]
        assert entry["category"] == "tool"
        assert entry["advanced"] is True


class TestLightpandaRequirements:
    """Lightpanda should expose browser tools without local Chromium."""

    async def test_lightpanda_local_mode_does_not_require_chromium(self):
        import tools.browser_tool as bt

        with patch("tools.browser_tool._is_camofox_mode", new_callable=AsyncMock, return_value=False), \
             patch("tools.browser_tool._get_cdp_override", new_callable=AsyncMock, return_value=""), \
             patch("tools.browser_tool._find_agent_browser", new_callable=AsyncMock, return_value="/usr/bin/agent-browser"), \
             patch("tools.browser_tool._requires_real_termux_browser_install", new_callable=AsyncMock, return_value=False), \
             patch("tools.browser_tool._get_cloud_provider", new_callable=AsyncMock, return_value=None), \
             patch("tools.browser_tool._get_browser_engine", new_callable=AsyncMock, return_value="lightpanda"), \
             patch("tools.browser_tool._chromium_installed", new_callable=AsyncMock, return_value=False):
            assert await bt.check_browser_requirements() is True

    async def test_chrome_local_mode_still_requires_chromium(self):
        import tools.browser_tool as bt

        with patch("tools.browser_tool._is_camofox_mode", new_callable=AsyncMock, return_value=False), \
             patch("tools.browser_tool._get_cdp_override", new_callable=AsyncMock, return_value=""), \
             patch("tools.browser_tool._find_agent_browser", new_callable=AsyncMock, return_value="/usr/bin/agent-browser"), \
             patch("tools.browser_tool._requires_real_termux_browser_install", new_callable=AsyncMock, return_value=False), \
             patch("tools.browser_tool._get_cloud_provider", new_callable=AsyncMock, return_value=None), \
             patch("tools.browser_tool._get_browser_engine", new_callable=AsyncMock, return_value="auto"), \
             patch("tools.browser_tool._chromium_installed", new_callable=AsyncMock, return_value=False):
            assert await bt.check_browser_requirements() is False


# ---------------------------------------------------------------------------
# cleanup_all_browsers resets engine cache
# ---------------------------------------------------------------------------

class TestCleanupResetsEngineCache:
    """Verify cleanup_all_browsers resets engine-related globals."""

    async def test_engine_cache_reset(self):
        import tools.browser_tool as bt
        # Seed the cache
        bt._cached_browser_engine = "lightpanda"
        bt._browser_engine_resolved = True
        bt._cached_homebrew_node_dirs = ("/opt/homebrew/opt/node@24/bin",)
        # cleanup should reset them
        await bt.cleanup_all_browsers()
        assert bt._cached_browser_engine is None
        assert bt._browser_engine_resolved is False
        assert bt._cached_homebrew_node_dirs is None


# ---------------------------------------------------------------------------
# fallback warning annotation
# ---------------------------------------------------------------------------

class TestLightpandaFallbackWarning:
    """Verify Chrome fallback results are annotated for users."""

    async def test_fallback_result_gets_user_visible_warning(self):
        from tools.browser_tool import _annotate_lightpanda_fallback

        result = {"success": True, "data": {"snapshot": "- heading \"Hello\" [ref=e1]"}}
        annotated = _annotate_lightpanda_fallback(
            result,
            "Lightpanda returned an empty/too-short snapshot; retried with Chrome.",
        )

        assert annotated["browser_engine"] == "chrome"
        assert "Lightpanda fallback" in annotated["fallback_warning"]
        assert annotated["browser_engine_fallback"] == {
            "from": "lightpanda",
            "to": "chrome",
            "reason": "Lightpanda returned an empty/too-short snapshot; retried with Chrome.",
        }
        assert annotated["data"]["fallback_warning"] == annotated["fallback_warning"]
        assert annotated["data"]["browser_engine"] == "chrome"


    async def test_browser_navigate_surfaces_fallback_warning(self):
        import json
        import tools.browser_tool as bt

        result = bt._annotate_lightpanda_fallback(
            {"success": True, "data": {"title": "Fallback OK", "url": "https://example.com/"}},
            "synthetic Lightpanda failure; retried with Chrome.",
        )

        with patch("tools.browser_tool._is_local_backend", new_callable=AsyncMock, return_value=True), \
             patch("tools.browser_tool._get_cloud_provider", new_callable=AsyncMock, return_value=None), \
             patch("tools.browser_tool._get_session_info", new_callable=AsyncMock, return_value={
                 "session_name": "test", "_first_nav": False, "features": {"local": True, "proxies": True}
             }), \
             patch("tools.browser_tool._run_browser_command", new_callable=AsyncMock, side_effect=[
                 result,
                 {"success": True, "data": {"snapshot": "- heading \"Fallback OK\" [ref=e1]", "refs": {"e1": {}}}},
             ]):
            response = json.loads(await bt.browser_navigate("https://example.com", task_id="warn-test"))

        assert response["success"] is True
        assert response["browser_engine"] == "chrome"
        assert "Lightpanda fallback" in response["fallback_warning"]
        assert response["browser_engine_fallback"]["from"] == "lightpanda"
        assert response["browser_engine_fallback"]["to"] == "chrome"
        bt._last_active_session_key.pop("warn-test", None)


    async def test_browser_vision_lightpanda_response_has_structured_fallback(self, tmp_path):
        import json
        import tools.browser_tool as bt

        chrome_shot = tmp_path / "chrome-structured.png"
        chrome_shot.write_bytes(b"\x89PNG" + b"0" * 128)

        class _Msg:
            content = "Example Domain screenshot"

        class _Choice:
            message = _Msg()

        class _Response:
            choices = [_Choice()]

        with patch("tools.browser_tool._get_browser_engine", new_callable=AsyncMock, return_value="lightpanda"), \
             patch("tools.browser_tool._should_inject_engine", new_callable=AsyncMock, return_value=True), \
             patch("tools.browser_tool._chrome_fallback_screenshot", new_callable=AsyncMock, return_value={
                 "success": True, "data": {"path": str(chrome_shot)}
             }), \
             patch("hermes_constants.get_hermes_dir", return_value=tmp_path), \
             patch("tools.browser_tool._lazy_call_llm", new_callable=AsyncMock, return_value=_Response()):
            response = json.loads(await bt.browser_vision("what is this?", task_id="vision-structured"))

        assert response["success"] is True
        assert response["browser_engine"] == "chrome"
        assert response["browser_engine_fallback"] == {
            "from": "lightpanda",
            "to": "chrome",
            "reason": "Lightpanda has no graphical renderer for screenshots; used Chrome for vision capture.",
        }

# ---------------------------------------------------------------------------
# _engine_override parameter
# ---------------------------------------------------------------------------

class TestEngineOverride:
    """Verify _engine_override bypasses the cached engine."""

    @patch("tools.browser_tool._get_session_info", new_callable=AsyncMock)
    @patch("tools.browser_tool._find_agent_browser", new_callable=AsyncMock, return_value="/usr/bin/agent-browser")
    @patch("tools.browser_tool._is_local_mode", new_callable=AsyncMock, return_value=True)
    @patch("tools.browser_tool._chromium_installed", new_callable=AsyncMock, return_value=True)
    @patch("tools.browser_tool._get_cloud_provider", new_callable=AsyncMock, return_value=None)
    @patch("tools.browser_tool._get_cdp_override", new_callable=AsyncMock, return_value="")
    @patch("tools.browser_tool._is_camofox_mode", new_callable=AsyncMock, return_value=False)
    async def test_override_prevents_engine_injection(
        self, _camofox, _cdp, _cloud, _chromium, _local, _find, _session
    ):
        """When _engine_override='auto', --engine flag is NOT injected."""
        import tools.browser_tool as bt

        # Set the global cache to lightpanda
        bt._cached_browser_engine = "lightpanda"
        bt._browser_engine_resolved = True

        _session.return_value = {"session_name": "test-sess"}

        # Track the cmd_parts that Popen receives
        captured_cmds = []
        mock_proc = MagicMock()
        mock_proc.wait = AsyncMock(return_value=None)
        mock_proc.returncode = 0

        async def capture_popen(*cmd, **kwargs):
            captured_cmds.append(list(cmd))
            return mock_proc

        # We need to mock the file operations too
        with patch("asyncio.create_subprocess_exec", side_effect=capture_popen), \
             patch("os.open", return_value=99), \
             patch("os.close"), \
             patch("os.unlink"), \
             patch("os.makedirs"), \
             patch("builtins.open", MagicMock(return_value=MagicMock(
                 __enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value='{"success": true, "data": {}}'))),
                 __exit__=MagicMock(return_value=False),
             ))), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("tools.browser_tool._write_owner_pid", new_callable=AsyncMock), \
             patch("tools.browser_tool._needs_chromium_sandbox_bypass", new_callable=AsyncMock, return_value=False):
            blocker = BlockBuster()
            blocker.activate()
            try:
                await bt._run_browser_command(
                    "task1", "snapshot", [], _engine_override="auto"
                )
            finally:
                blocker.deactivate()

        # Should NOT contain "--engine" since override is "auto"
        assert captured_cmds
        assert "--engine" not in captured_cmds[0]

    @patch("tools.browser_tool._get_session_info", new_callable=AsyncMock)
    @patch("tools.browser_tool._find_agent_browser", new_callable=AsyncMock, return_value="/usr/bin/agent-browser")
    @patch("tools.browser_tool._is_local_mode", new_callable=AsyncMock, return_value=True)
    @patch("tools.browser_tool._chromium_installed", new_callable=AsyncMock, return_value=True)
    @patch("tools.browser_tool._get_cloud_provider", new_callable=AsyncMock, return_value=None)
    @patch("tools.browser_tool._get_cdp_override", new_callable=AsyncMock, return_value="")
    @patch("tools.browser_tool._is_camofox_mode", new_callable=AsyncMock, return_value=False)
    async def test_no_override_uses_cached_engine(
        self, _camofox, _cdp, _cloud, _chromium, _local, _find, _session
    ):
        """Without _engine_override, the cached engine is used."""
        import tools.browser_tool as bt

        bt._cached_browser_engine = "lightpanda"
        bt._browser_engine_resolved = True

        _session.return_value = {"session_name": "test-sess"}

        captured_cmds = []
        mock_proc = MagicMock()
        mock_proc.wait = AsyncMock(return_value=None)
        mock_proc.returncode = 0

        async def capture_popen(*cmd, **kwargs):
            captured_cmds.append(list(cmd))
            return mock_proc

        # Return a substantive snapshot so the LP fallback does NOT trigger.
        mock_stdout = '{"success": true, "data": {"snapshot": "- heading \\"Hello\\" [ref=e1]", "refs": {"e1": {}}}}'
        with patch("asyncio.create_subprocess_exec", side_effect=capture_popen), \
             patch("os.open", return_value=99), \
             patch("os.close"), \
             patch("os.unlink"), \
             patch("os.makedirs"), \
             patch("builtins.open", MagicMock(return_value=MagicMock(
                 __enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value=mock_stdout))),
                 __exit__=MagicMock(return_value=False),
             ))), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("tools.browser_tool._write_owner_pid", new_callable=AsyncMock), \
             patch("tools.browser_tool._needs_chromium_sandbox_bypass", new_callable=AsyncMock, return_value=False):
            await bt._run_browser_command("task1", "snapshot", [])

        # SHOULD contain "--engine lightpanda"
        assert captured_cmds
        assert "--engine" in captured_cmds[0]
        engine_idx = captured_cmds[0].index("--engine")
        assert captured_cmds[0][engine_idx + 1] == "lightpanda"

    async def test_hybrid_local_sidecar_injects_engine_even_with_cloud_provider(self):
        """A task::local sidecar is local even when global cloud config exists."""
        import tools.browser_tool as bt

        bt._cached_browser_engine = "lightpanda"
        bt._browser_engine_resolved = True
        captured_cmds = []
        mock_provider = MagicMock()

        mock_proc = MagicMock()
        mock_proc.wait = AsyncMock(return_value=None)
        mock_proc.returncode = 0

        async def capture_popen(*cmd, **kwargs):
            captured_cmds.append(list(cmd))
            return mock_proc

        mock_stdout = json.dumps({
            "success": True,
            "data": {"snapshot": '- heading "Hello" [ref=e1]', "refs": {"e1": {}}},
        })
        with patch("tools.browser_tool._get_session_info", new_callable=AsyncMock, return_value={"session_name": "local-sidecar"}), \
             patch("tools.browser_tool._find_agent_browser", new_callable=AsyncMock, return_value="/usr/bin/agent-browser"), \
             patch("tools.browser_tool._is_local_mode", new_callable=AsyncMock, return_value=False), \
             patch("tools.browser_tool._chromium_installed", new_callable=AsyncMock, return_value=True), \
             patch("tools.browser_tool._get_cloud_provider", new_callable=AsyncMock, return_value=mock_provider), \
             patch("tools.browser_tool._get_cdp_override", new_callable=AsyncMock, return_value=""), \
             patch("tools.browser_tool._is_camofox_mode", new_callable=AsyncMock, return_value=False), \
             patch("asyncio.create_subprocess_exec", side_effect=capture_popen), \
             patch("os.open", return_value=99), \
             patch("os.close"), \
             patch("os.unlink"), \
             patch("os.makedirs"), \
             patch("builtins.open", MagicMock(return_value=MagicMock(
                 __enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value=mock_stdout))),
                 __exit__=MagicMock(return_value=False),
             ))), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("tools.browser_tool._write_owner_pid", new_callable=AsyncMock), \
             patch("tools.browser_tool._needs_chromium_sandbox_bypass", new_callable=AsyncMock, return_value=False):
            await bt._run_browser_command("task::local", "snapshot", [])

        assert captured_cmds
        assert "--engine" in captured_cmds[0]
        assert captured_cmds[0][captured_cmds[0].index("--engine") + 1] == "lightpanda"
