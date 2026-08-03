"""Tests for Ollama Cloud provider integration."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider, resolve_api_key_provider_credentials
from hermes_cli.models import _PROVIDER_MODELS, _PROVIDER_LABELS, _PROVIDER_ALIASES, normalize_provider
from hermes_cli.model_normalize import normalize_model_for_provider
from agent.model_metadata import _URL_TO_PROVIDER, _PROVIDER_PREFIXES
from agent.models_dev import PROVIDER_TO_MODELS_DEV, list_agentic_models


# ── Provider Registry ──

class TestOllamaCloudProviderRegistry:
    def test_ollama_cloud_in_registry(self):
        assert "ollama-cloud" in PROVIDER_REGISTRY

    def test_ollama_cloud_config(self):
        pconfig = PROVIDER_REGISTRY["ollama-cloud"]
        assert pconfig.id == "ollama-cloud"
        assert pconfig.name == "Ollama Cloud"
        assert pconfig.auth_type == "api_key"
        assert pconfig.inference_base_url == "https://ollama.com/v1"


# ── Provider Aliases ──

PROVIDER_ENV_VARS = (
    "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY", "GEMINI_API_KEY", "OLLAMA_API_KEY",
    "GLM_API_KEY", "ZAI_API_KEY", "KIMI_API_KEY",
    "MINIMAX_API_KEY", "DEEPSEEK_API_KEY",
)

@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch):
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class TestOllamaCloudAliases:

    @pytest.mark.asyncio
    async def test_alias_ollama_underscore(self):
        """ollama_cloud (underscore) is the unambiguous cloud alias."""
        assert await resolve_provider("ollama_cloud") == "ollama-cloud"


    def test_models_py_aliases(self):
        assert _PROVIDER_ALIASES.get("ollama_cloud") == "ollama-cloud"
        # bare "ollama" stays local
        assert _PROVIDER_ALIASES.get("ollama") == "custom"


# ── Auto-detection ──

class TestOllamaCloudAutoDetection:
    @pytest.mark.asyncio
    async def test_auto_detects_ollama_api_key(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "test-ollama-key")
        assert await resolve_provider("auto") == "ollama-cloud"


# ── Credential Resolution ──

class TestOllamaCloudCredentials:
    @pytest.mark.asyncio
    async def test_resolve_with_ollama_api_key(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "ollama-secret")
        creds = await resolve_api_key_provider_credentials("ollama-cloud")
        assert creds["provider"] == "ollama-cloud"
        assert creds["api_key"] == "ollama-secret"
        assert creds["base_url"] == "https://ollama.com/v1"


    @pytest.mark.asyncio
    async def test_runtime_ollama_cloud(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "ollama-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = await resolve_runtime_provider(requested="ollama-cloud")
        assert result["provider"] == "ollama-cloud"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "ollama-key"
        assert result["base_url"] == "https://ollama.com/v1"



# ── Model Normalization ──

class TestOllamaCloudModelNormalization:


    def test_passthrough_no_tag(self):
        assert normalize_model_for_provider("glm-5", "ollama-cloud") == "glm-5"


# ── URL-to-Provider Mapping ──


# ── models.dev Integration ──

class TestOllamaCloudModelsDev:
    def test_ollama_cloud_mapped(self):
        assert PROVIDER_TO_MODELS_DEV.get("ollama-cloud") == "ollama-cloud"

    @pytest.mark.asyncio
    async def test_list_agentic_models_with_mock_data(self):
        """list_agentic_models filters correctly from mock models.dev data."""
        mock_data = {
            "ollama-cloud": {
                "models": {
                    "qwen3.5:397b": {"tool_call": True},
                    "glm-5": {"tool_call": True},
                    "nemotron-3-nano:30b": {"tool_call": True},
                    "some-embedding:latest": {"tool_call": False},
                }
            }
        }
        with patch("agent.models_dev.fetch_models_dev", new=AsyncMock(return_value=mock_data)):
            result = await list_agentic_models("ollama-cloud")
        assert "qwen3.5:397b" in result
        assert "glm-5" in result
        assert "nemotron-3-nano:30b" in result
        assert "some-embedding:latest" not in result  # no tool_call


# ── Agent Init (no SyntaxError) ──

class TestOllamaCloudAgentInit:
    def test_ollama_cloud_agent_uses_chat_completions(self, monkeypatch):
        """Ollama Cloud falls through to chat_completions — no special elif needed."""
        monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
        with patch("run_agent.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from run_agent import AIAgent
            agent = AIAgent(
                model="qwen3.5:397b",
                provider="ollama-cloud",
                api_key="test-key",
                base_url="https://ollama.com/v1",
            )
            assert agent.api_mode == "chat_completions"
            assert agent.provider == "ollama-cloud"


# ── providers.py New System ──

class TestOllamaCloudProvidersNew:
    def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS
        assert "ollama-cloud" in HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS["ollama-cloud"]
        assert overlay.transport == "openai_chat"
        assert overlay.base_url_env_var == "OLLAMA_BASE_URL"

    def test_alias_resolves(self):
        from hermes_cli.providers import normalize_provider as np
        assert np("ollama") == "custom"  # bare "ollama" = local
        assert np("ollama-cloud") == "ollama-cloud"


    def test_get_provider(self):
        from hermes_cli.providers import get_provider
        pdef = get_provider("ollama-cloud")
        assert pdef is not None
        assert pdef.id == "ollama-cloud"
        assert pdef.transport == "openai_chat"
