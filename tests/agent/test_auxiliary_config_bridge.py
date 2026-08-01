"""Tests for auxiliary model config bridging — verifies that config.yaml values
are properly mapped to environment variables by both CLI and gateway loaders.

Also tests the vision_tools and browser_tool model override env vars.
"""

import os
import sys
from unittest.mock import patch, MagicMock, AsyncMock

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _run_auxiliary_bridge(config_dict, monkeypatch):
    """Simulate the auxiliary config → env var bridging logic shared by CLI and gateway.

    This mirrors the code in cli.py load_cli_config() and gateway/run.py.
    Both use the same pattern; we test it once here.
    """
    # Clear env vars
    for key in (
        "AUXILIARY_VISION_PROVIDER", "AUXILIARY_VISION_MODEL",
        "AUXILIARY_VISION_BASE_URL", "AUXILIARY_VISION_API_KEY",
        "AUXILIARY_WEB_EXTRACT_PROVIDER", "AUXILIARY_WEB_EXTRACT_MODEL",
        "AUXILIARY_WEB_EXTRACT_BASE_URL", "AUXILIARY_WEB_EXTRACT_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    # Compression config is read directly from config.yaml — no env var bridging.

    # Auxiliary bridge
    auxiliary_cfg = config_dict.get("auxiliary", {})
    if auxiliary_cfg and isinstance(auxiliary_cfg, dict):
        aux_task_env = {
            "vision": {
                "provider": "AUXILIARY_VISION_PROVIDER",
                "model": "AUXILIARY_VISION_MODEL",
                "base_url": "AUXILIARY_VISION_BASE_URL",
                "api_key": "AUXILIARY_VISION_API_KEY",
            },
            "web_extract": {
                "provider": "AUXILIARY_WEB_EXTRACT_PROVIDER",
                "model": "AUXILIARY_WEB_EXTRACT_MODEL",
                "base_url": "AUXILIARY_WEB_EXTRACT_BASE_URL",
                "api_key": "AUXILIARY_WEB_EXTRACT_API_KEY",
            },
        }
        for task_key, env_map in aux_task_env.items():
            task_cfg = auxiliary_cfg.get(task_key, {})
            if not isinstance(task_cfg, dict):
                continue
            prov = str(task_cfg.get("provider", "")).strip()
            model = str(task_cfg.get("model", "")).strip()
            base_url = str(task_cfg.get("base_url", "")).strip()
            api_key = str(task_cfg.get("api_key", "")).strip()
            if prov and prov != "auto":
                os.environ[env_map["provider"]] = prov
            if model:
                os.environ[env_map["model"]] = model
            if base_url:
                os.environ[env_map["base_url"]] = base_url
            if api_key:
                os.environ[env_map["api_key"]] = api_key


# ── Config bridging tests ────────────────────────────────────────────────────


class TestAuxiliaryConfigBridge:
    """Verify the config.yaml → env var bridging logic used by CLI and gateway."""


    def test_vision_model_bridged(self, monkeypatch):
        config = {
            "auxiliary": {
                "vision": {"provider": "auto", "model": "openai/gpt-4o"},
            }
        }
        _run_auxiliary_bridge(config, monkeypatch)
        assert os.environ.get("AUXILIARY_VISION_MODEL") == "openai/gpt-4o"
        # auto provider should not be set
        assert os.environ.get("AUXILIARY_VISION_PROVIDER") is None

    def test_web_extract_bridged(self, monkeypatch):
        config = {
            "auxiliary": {
                "web_extract": {"provider": "nous", "model": "gemini-2.5-flash"},
            }
        }
        _run_auxiliary_bridge(config, monkeypatch)
        assert os.environ.get("AUXILIARY_WEB_EXTRACT_PROVIDER") == "nous"
        assert os.environ.get("AUXILIARY_WEB_EXTRACT_MODEL") == "gemini-2.5-flash"





    def test_mixed_tasks(self, monkeypatch):
        config = {
            "auxiliary": {
                "vision": {"provider": "openrouter", "model": ""},
                "web_extract": {"provider": "auto", "model": "custom-llm"},
            }
        }
        _run_auxiliary_bridge(config, monkeypatch)
        assert os.environ.get("AUXILIARY_VISION_PROVIDER") == "openrouter"
        assert os.environ.get("AUXILIARY_VISION_MODEL") is None
        assert os.environ.get("AUXILIARY_WEB_EXTRACT_PROVIDER") is None
        assert os.environ.get("AUXILIARY_WEB_EXTRACT_MODEL") == "custom-llm"





# ── Vision model override tests ──────────────────────────────────────────────


class TestVisionModelOverride:
    """Test that AUXILIARY_VISION_MODEL env var overrides the default model in the handler."""

    @pytest.mark.asyncio
    async def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv("AUXILIARY_VISION_MODEL", "openai/gpt-4o")
        from tools.vision_tools import _handle_vision_analyze
        with (
            patch("tools.vision_tools.vision_analyze_tool", new_callable=AsyncMock) as mock_tool,
            patch("tools.vision_tools._should_use_native_vision_fast_path", return_value=False),
        ):
            mock_tool.return_value = '{"success": true}'
            await _handle_vision_analyze({"image_url": "http://test.jpg", "question": "test"})
            call_args = mock_tool.call_args
            # 3rd positional arg = model
            assert call_args[0][2] == "openai/gpt-4o"

    @pytest.mark.asyncio
    async def test_default_model_when_no_override(self, monkeypatch):
        monkeypatch.delenv("AUXILIARY_VISION_MODEL", raising=False)
        from tools.vision_tools import _handle_vision_analyze
        with (
            patch("tools.vision_tools.vision_analyze_tool", new_callable=AsyncMock) as mock_tool,
            patch("tools.vision_tools._should_use_native_vision_fast_path", return_value=False),
        ):
            mock_tool.return_value = '{"success": true}'
            await _handle_vision_analyze({"image_url": "http://test.jpg", "question": "test"})
            call_args = mock_tool.call_args
            # With no AUXILIARY_VISION_MODEL env var, model should be None
            # (the centralized call_llm router picks the provider default)
            assert call_args[0][2] is None


# ── DEFAULT_CONFIG shape tests ───────────────────────────────────────────────


class TestDefaultConfigShape:
    """Verify the DEFAULT_CONFIG in hermes_cli/config.py has correct auxiliary structure."""

    def test_auxiliary_section_exists(self):
        from hermes_cli.config import DEFAULT_CONFIG
        assert "auxiliary" in DEFAULT_CONFIG

    def test_vision_task_structure(self):
        from hermes_cli.config import DEFAULT_CONFIG
        vision = DEFAULT_CONFIG["auxiliary"]["vision"]
        assert "provider" in vision
        assert "model" in vision
        assert vision["provider"] == "auto"
        assert vision["model"] == ""

    def test_web_extract_task_structure(self):
        from hermes_cli.config import DEFAULT_CONFIG
        web = DEFAULT_CONFIG["auxiliary"]["web_extract"]
        assert "provider" in web
        assert "model" in web
        assert web["provider"] == "auto"
        assert web["model"] == ""
