"""Tests for API-key provider support (z.ai/GLM, Kimi, MiniMax, AI Gateway)."""

import os

import pytest
from unittest.mock import AsyncMock

from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    resolve_provider,
    get_api_key_provider_status,
    resolve_api_key_provider_credentials,
    AuthError,
    KIMI_CODE_BASE_URL,
    STEPFUN_STEP_PLAN_INTL_BASE_URL,
    STEPFUN_STEP_PLAN_CN_BASE_URL,
    _resolve_kimi_base_url,
)
from hermes_cli.copilot_auth import _try_gh_cli_token


# =============================================================================
# Provider Registry tests
# =============================================================================

class TestProviderRegistry:
    """Test that new providers are correctly registered."""

    @pytest.mark.parametrize("provider_id,name,auth_type", [
        ("copilot-acp", "GitHub Copilot ACP", "external_process"),
        ("copilot", "GitHub Copilot", "api_key"),
        ("huggingface", "Hugging Face", "api_key"),
        ("zai", "Z.AI / GLM", "api_key"),
        ("xai", "xAI", "api_key"),
        ("nvidia", "NVIDIA NIM", "api_key"),
        ("kimi-coding", "Kimi / Moonshot", "api_key"),
        ("stepfun", "StepFun Step Plan", "api_key"),
        ("minimax", "MiniMax", "api_key"),
        ("minimax-cn", "MiniMax (China)", "api_key"),
        ("ai-gateway", "Vercel AI Gateway", "api_key"),
        ("kilocode", "Kilo Code", "api_key"),
        ("gmi", "GMI Cloud", "api_key"),
    ])
    def test_provider_registered(self, provider_id, name, auth_type):
        assert provider_id in PROVIDER_REGISTRY
        pconfig = PROVIDER_REGISTRY[provider_id]
        assert pconfig.name == name
        assert pconfig.auth_type == auth_type
        assert pconfig.inference_base_url  # must have a default base URL




    def test_copilot_env_vars(self):
        pconfig = PROVIDER_REGISTRY["copilot"]
        assert pconfig.api_key_env_vars == ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")
        assert pconfig.base_url_env_var == "COPILOT_API_BASE_URL"

    def test_kimi_env_vars(self):
        pconfig = PROVIDER_REGISTRY["kimi-coding"]
        # KIMI_API_KEY is the primary env var; KIMI_CODING_API_KEY is a
        # secondary fallback for Kimi Code sk-kimi- keys so users don't
        # have to overload the same variable.
        assert "KIMI_API_KEY" in pconfig.api_key_env_vars
        assert "KIMI_CODING_API_KEY" in pconfig.api_key_env_vars
        assert pconfig.base_url_env_var == "KIMI_BASE_URL"

    def test_minimax_env_vars(self):
        pconfig = PROVIDER_REGISTRY["minimax"]
        assert pconfig.api_key_env_vars == ("MINIMAX_API_KEY",)
        assert pconfig.base_url_env_var == "MINIMAX_BASE_URL"

    def test_stepfun_env_vars(self):
        pconfig = PROVIDER_REGISTRY["stepfun"]
        assert pconfig.api_key_env_vars == ("STEPFUN_API_KEY",)
        assert pconfig.base_url_env_var == "STEPFUN_BASE_URL"

    def test_minimax_cn_env_vars(self):
        pconfig = PROVIDER_REGISTRY["minimax-cn"]
        assert pconfig.api_key_env_vars == ("MINIMAX_CN_API_KEY",)
        assert pconfig.base_url_env_var == "MINIMAX_CN_BASE_URL"

    def test_ai_gateway_env_vars(self):
        pconfig = PROVIDER_REGISTRY["ai-gateway"]
        assert pconfig.api_key_env_vars == ("AI_GATEWAY_API_KEY",)
        assert pconfig.base_url_env_var == "AI_GATEWAY_BASE_URL"

    def test_kilocode_env_vars(self):
        pconfig = PROVIDER_REGISTRY["kilocode"]
        assert pconfig.api_key_env_vars == ("KILOCODE_API_KEY",)
        assert pconfig.base_url_env_var == "KILOCODE_BASE_URL"

    def test_gmi_env_vars(self):
        pconfig = PROVIDER_REGISTRY["gmi"]
        assert pconfig.api_key_env_vars == ("GMI_API_KEY",)
        assert pconfig.base_url_env_var == "GMI_BASE_URL"

    def test_huggingface_env_vars(self):
        pconfig = PROVIDER_REGISTRY["huggingface"]
        assert pconfig.api_key_env_vars == ("HF_TOKEN",)
        assert pconfig.base_url_env_var == "HF_BASE_URL"

    def test_base_urls(self):
        assert PROVIDER_REGISTRY["copilot"].inference_base_url == "https://api.githubcopilot.com"
        assert PROVIDER_REGISTRY["copilot-acp"].inference_base_url == "acp://copilot"
        assert PROVIDER_REGISTRY["zai"].inference_base_url == "https://api.z.ai/api/paas/v4"
        assert PROVIDER_REGISTRY["kimi-coding"].inference_base_url == "https://api.moonshot.ai/v1"
        assert PROVIDER_REGISTRY["stepfun"].inference_base_url == STEPFUN_STEP_PLAN_INTL_BASE_URL
        assert PROVIDER_REGISTRY["minimax"].inference_base_url == "https://api.minimax.io/anthropic"
        assert PROVIDER_REGISTRY["minimax-cn"].inference_base_url == "https://api.minimaxi.com/anthropic"
        assert PROVIDER_REGISTRY["ai-gateway"].inference_base_url == "https://ai-gateway.vercel.sh/v1"
        assert PROVIDER_REGISTRY["kilocode"].inference_base_url == "https://api.kilo.ai/api/gateway"
        assert PROVIDER_REGISTRY["gmi"].inference_base_url == "https://api.gmi-serving.com/v1"
        assert PROVIDER_REGISTRY["huggingface"].inference_base_url == "https://router.huggingface.co/v1"

    def test_oauth_providers_unchanged(self):
        """Ensure we didn't break the existing OAuth providers."""
        assert "nous" in PROVIDER_REGISTRY
        assert PROVIDER_REGISTRY["nous"].auth_type == "oauth_device_code"
        assert "openai-codex" in PROVIDER_REGISTRY
        assert PROVIDER_REGISTRY["openai-codex"].auth_type == "oauth_external"


# =============================================================================
# Provider Resolution tests
# =============================================================================

# Derived from the live PROVIDER_REGISTRY so the list can never drift when a
# new provider (and its env var) is added — a hand-maintained tuple here was
# missing HF_TOKEN/DEEPINFRA_API_KEY, which made the auto-detection tests
# env-dependent (they failed on any machine with HF_TOKEN exported).
from hermes_cli.auth import PROVIDER_REGISTRY as _REGISTRY

_EXTRA_ENV_VARS = (
    # Checked directly in resolve_provider("auto"), not via the registry.
    "OPENROUTER_API_KEY", "NOUS_API_KEY",
    # Base URLs / paths that influence detection but aren't api_key_env_vars.
    "LM_BASE_URL", "KIMI_BASE_URL", "STEPFUN_BASE_URL", "KILOCODE_BASE_URL",
    "GMI_BASE_URL", "OPENAI_BASE_URL",
    "HERMES_COPILOT_ACP_COMMAND", "COPILOT_CLI_PATH",
    "HERMES_COPILOT_ACP_ARGS", "COPILOT_ACP_BASE_URL",
)

PROVIDER_ENV_VARS = tuple(
    dict.fromkeys(
        [var for cfg in _REGISTRY.values() for var in cfg.api_key_env_vars]
        + list(_EXTRA_ENV_VARS)
    )
)


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    for key in PROVIDER_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("hermes_cli.auth._load_auth_store", AsyncMock(return_value={}))


class TestResolveProvider:
    """Test resolve_provider() with new providers."""

    @pytest.mark.asyncio
    async def test_explicit_zai(self):
        assert await resolve_provider("zai") == "zai"





    @pytest.mark.asyncio
    async def test_explicit_ai_gateway(self):
        assert await resolve_provider("ai-gateway") == "ai-gateway"

    @pytest.mark.asyncio
    async def test_explicit_gmi(self):
        assert await resolve_provider("gmi") == "gmi"



    @pytest.mark.asyncio
    async def test_alias_zhipu(self):
        assert await resolve_provider("zhipu") == "zai"

    @pytest.mark.asyncio
    async def test_alias_kimi(self):
        assert await resolve_provider("kimi") == "kimi-coding"

    @pytest.mark.asyncio
    async def test_alias_moonshot(self):
        assert await resolve_provider("moonshot") == "kimi-coding"

    @pytest.mark.asyncio
    async def test_alias_step(self):
        assert await resolve_provider("step") == "stepfun"

    @pytest.mark.asyncio
    async def test_alias_minimax_underscore(self):
        assert await resolve_provider("minimax_cn") == "minimax-cn"

    @pytest.mark.asyncio
    async def test_alias_aigateway(self):
        assert await resolve_provider("aigateway") == "ai-gateway"

    @pytest.mark.asyncio
    async def test_alias_vercel(self):
        assert await resolve_provider("vercel") == "ai-gateway"

    @pytest.mark.asyncio
    async def test_alias_gmi_cloud(self):
        assert await resolve_provider("gmi-cloud") == "gmi"

    @pytest.mark.asyncio
    async def test_explicit_kilocode(self):
        assert await resolve_provider("kilocode") == "kilocode"

    @pytest.mark.asyncio
    async def test_alias_kilo(self):
        assert await resolve_provider("kilo") == "kilocode"

    @pytest.mark.asyncio
    async def test_alias_kilo_code(self):
        assert await resolve_provider("kilo-code") == "kilocode"

    @pytest.mark.asyncio
    async def test_alias_kilo_gateway(self):
        assert await resolve_provider("kilo-gateway") == "kilocode"

    @pytest.mark.asyncio
    async def test_alias_case_insensitive(self):
        assert await resolve_provider("GLM") == "zai"
        assert await resolve_provider("Z-AI") == "zai"
        assert await resolve_provider("Kimi") == "kimi-coding"

    @pytest.mark.asyncio
    async def test_alias_github_copilot(self):
        assert await resolve_provider("github-copilot") == "copilot"

    @pytest.mark.asyncio
    async def test_alias_github_models(self):
        assert await resolve_provider("github-models") == "copilot"

    @pytest.mark.asyncio
    async def test_alias_github_copilot_acp(self):
        assert await resolve_provider("github-copilot-acp") == "copilot-acp"
        assert await resolve_provider("copilot-acp-agent") == "copilot-acp"

    @pytest.mark.asyncio
    async def test_explicit_huggingface(self):
        assert await resolve_provider("huggingface") == "huggingface"

    @pytest.mark.asyncio
    async def test_alias_hf(self):
        assert await resolve_provider("hf") == "huggingface"

    @pytest.mark.asyncio
    async def test_alias_hugging_face(self):
        assert await resolve_provider("hugging-face") == "huggingface"

    @pytest.mark.asyncio
    async def test_alias_huggingface_hub(self):
        assert await resolve_provider("huggingface-hub") == "huggingface"

    @pytest.mark.asyncio
    async def test_unknown_provider_raises(self):
        with pytest.raises(AuthError):
            await resolve_provider("nonexistent-provider-xyz")

    @pytest.mark.asyncio
    async def test_auto_detects_glm_key(self, monkeypatch):
        monkeypatch.setenv("GLM_API_KEY", "test-glm-key")
        assert await resolve_provider("auto") == "zai"

    @pytest.mark.asyncio
    async def test_auto_detects_zai_key(self, monkeypatch):
        monkeypatch.setenv("ZAI_API_KEY", "test-zai-key")
        assert await resolve_provider("auto") == "zai"

    @pytest.mark.asyncio
    async def test_auto_detects_z_ai_key(self, monkeypatch):
        monkeypatch.setenv("Z_AI_API_KEY", "test-z-ai-key")
        assert await resolve_provider("auto") == "zai"

    @pytest.mark.asyncio
    async def test_auto_detects_kimi_key(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "test-kimi-key")
        assert await resolve_provider("auto") == "kimi-coding"

    @pytest.mark.asyncio
    async def test_auto_detects_stepfun_key(self, monkeypatch):
        monkeypatch.setenv("STEPFUN_API_KEY", "test-stepfun-key")
        assert await resolve_provider("auto") == "stepfun"

    @pytest.mark.asyncio
    async def test_auto_detects_minimax_key(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-mm-key")
        assert await resolve_provider("auto") == "minimax"

    @pytest.mark.asyncio
    async def test_auto_detects_minimax_cn_key(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "test-mm-cn-key")
        assert await resolve_provider("auto") == "minimax-cn"

    @pytest.mark.asyncio
    async def test_auto_detects_ai_gateway_key(self, monkeypatch):
        monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-gw-key")
        assert await resolve_provider("auto") == "ai-gateway"

    @pytest.mark.asyncio
    async def test_auto_detects_gmi_key(self, monkeypatch):
        monkeypatch.setenv("GMI_API_KEY", "test-gmi-key")
        assert await resolve_provider("auto") == "gmi"

    @pytest.mark.asyncio
    async def test_auto_detects_kilocode_key(self, monkeypatch):
        monkeypatch.setenv("KILOCODE_API_KEY", "test-kilo-key")
        assert await resolve_provider("auto") == "kilocode"

    @pytest.mark.asyncio
    async def test_auto_detects_hf_token(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_test_token")
        assert await resolve_provider("auto") == "huggingface"

    @pytest.mark.asyncio
    async def test_openrouter_takes_priority_over_glm(self, monkeypatch):
        """OpenRouter API key should win over GLM in auto-detection."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.setenv("GLM_API_KEY", "glm-key")
        assert await resolve_provider("auto") == "openrouter"

    @pytest.mark.asyncio
    async def test_auto_does_not_select_copilot_from_github_token(self, monkeypatch):
        # AWS Bedrock auto-detection (via boto3's credential chain) runs at
        # the tail of resolve_provider("auto") and will silently pick up
        # ~/.aws/credentials on developer machines that aren't blanked by
        # the hermetic conftest. Force-disable it so this test exercises
        # the specific "GitHub token alone shouldn't auto-pick copilot"
        # behavior, not the Bedrock fallback.
        monkeypatch.setattr(
            "agent.bedrock_adapter.has_aws_credentials",
            AsyncMock(return_value=False),
        )
        monkeypatch.setenv("GITHUB_TOKEN", "gh-test-token")
        with pytest.raises(AuthError, match="No inference provider configured"):
            await resolve_provider("auto")


# =============================================================================
# API Key Provider Status tests
# =============================================================================

class TestApiKeyProviderStatus:

    @pytest.mark.asyncio
    async def test_unconfigured_provider(self):
        status = await get_api_key_provider_status("zai")
        assert status["configured"] is False
        assert status["logged_in"] is False

    @pytest.mark.asyncio
    async def test_configured_provider(self, monkeypatch):
        monkeypatch.setenv("GLM_API_KEY", "test-key-123")
        status = await get_api_key_provider_status("zai")
        assert status["configured"] is True
        assert status["logged_in"] is True
        assert status["key_source"] == "GLM_API_KEY"
        assert "z.ai" in status["base_url"].lower() or "api.z.ai" in status["base_url"]


# =============================================================================
# Credential Resolution tests
# =============================================================================

class TestResolveApiKeyProviderCredentials:





    @pytest.mark.asyncio
    async def test_try_gh_cli_token_uses_homebrew_path_when_not_on_path(self, monkeypatch):
        async def _is_homebrew_file(path):
            return path == "/opt/homebrew/bin/gh"

        async def _is_homebrew_executable(path, mode):
            return path == "/opt/homebrew/bin/gh" and mode == os.X_OK

        monkeypatch.setattr("hermes_cli.copilot_auth.shutil.which", lambda command: None)
        monkeypatch.setattr(
            "hermes_cli.copilot_auth.aiofiles.os.path.isfile",
            _is_homebrew_file,
        )
        monkeypatch.setattr(
            "hermes_cli.copilot_auth.aiofiles.os.access",
            _is_homebrew_executable,
        )

        calls = []

        class _Process:
            returncode = 0

            async def communicate(self):
                return b"gh-cli-secret\n", b""

        async def _create_process(*cmd, **kwargs):
            calls.append(list(cmd))
            return _Process()

        monkeypatch.setattr(
            "hermes_cli.copilot_auth.asyncio.create_subprocess_exec",
            _create_process,
        )

        assert await _try_gh_cli_token() == "gh-cli-secret"
        assert calls == [["/opt/homebrew/bin/gh", "auth", "token"]]



    @pytest.mark.asyncio
    async def test_resolve_stepfun_with_key(self, monkeypatch):
        monkeypatch.setenv("STEPFUN_API_KEY", "stepfun-secret-key")
        creds = await resolve_api_key_provider_credentials("stepfun")
        assert creds["provider"] == "stepfun"
        assert creds["api_key"] == "stepfun-secret-key"
        assert creds["base_url"] == STEPFUN_STEP_PLAN_INTL_BASE_URL




    @pytest.mark.asyncio
    async def test_resolve_ai_gateway_with_key(self, monkeypatch):
        monkeypatch.setenv("AI_GATEWAY_API_KEY", "gw-secret-key")
        creds = await resolve_api_key_provider_credentials("ai-gateway")
        assert creds["provider"] == "ai-gateway"
        assert creds["api_key"] == "gw-secret-key"
        assert creds["base_url"] == "https://ai-gateway.vercel.sh/v1"

    @pytest.mark.asyncio
    async def test_resolve_kilocode_with_key(self, monkeypatch):
        monkeypatch.setenv("KILOCODE_API_KEY", "kilo-secret-key")
        creds = await resolve_api_key_provider_credentials("kilocode")
        assert creds["provider"] == "kilocode"
        assert creds["api_key"] == "kilo-secret-key"
        assert creds["base_url"] == "https://api.kilo.ai/api/gateway"





# =============================================================================
# Runtime Provider Resolution tests
# =============================================================================

class TestRuntimeProviderResolution:

    @pytest.mark.asyncio
    async def test_runtime_zai(self, monkeypatch):
        monkeypatch.setenv("GLM_API_KEY", "glm-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = await resolve_runtime_provider(requested="zai")
        assert result["provider"] == "zai"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "glm-key"
        assert "z.ai" in result["base_url"] or "api.z.ai" in result["base_url"]



    @pytest.mark.asyncio
    async def test_runtime_minimax(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "mm-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = await resolve_runtime_provider(requested="minimax")
        assert result["provider"] == "minimax"
        assert result["api_key"] == "mm-key"

    @pytest.mark.asyncio
    async def test_runtime_ai_gateway(self, monkeypatch):
        monkeypatch.setenv("AI_GATEWAY_API_KEY", "gw-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = await resolve_runtime_provider(requested="ai-gateway")
        assert result["provider"] == "ai-gateway"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "gw-key"
        assert "ai-gateway.vercel.sh" in result["base_url"]

    @pytest.mark.asyncio
    async def test_runtime_kilocode(self, monkeypatch):
        monkeypatch.setenv("KILOCODE_API_KEY", "kilo-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = await resolve_runtime_provider(requested="kilocode")
        assert result["provider"] == "kilocode"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "kilo-key"
        assert "kilo.ai" in result["base_url"]

    @pytest.mark.asyncio
    async def test_runtime_gmi(self, monkeypatch):
        monkeypatch.setenv("GMI_API_KEY", "gmi-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = await resolve_runtime_provider(requested="gmi")
        assert result["provider"] == "gmi"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "gmi-key"
        assert result["base_url"] == "https://api.gmi-serving.com/v1"

    @pytest.mark.asyncio
    async def test_runtime_auto_detects_api_key_provider(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "auto-kimi-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = await resolve_runtime_provider(requested="auto")
        assert result["provider"] == "kimi-coding"
        assert result["api_key"] == "auto-kimi-key"

    @pytest.mark.asyncio
    async def test_runtime_copilot_uses_gh_cli_token(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.copilot_auth._try_gh_cli_token",
            AsyncMock(return_value="gho_cli_secret"),
        )
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = await resolve_runtime_provider(requested="copilot")
        assert result["provider"] == "copilot"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "gho_cli_secret"
        assert result["base_url"] == "https://api.githubcopilot.com"

    @pytest.mark.asyncio
    async def test_runtime_copilot_uses_responses_for_gpt_5_4(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.copilot_auth._try_gh_cli_token",
            AsyncMock(return_value="gho_cli_secret"),
        )
        monkeypatch.setattr(
            "hermes_cli.runtime_provider._get_model_config",
            lambda *args, **kwargs: {"provider": "copilot", "default": "gpt-5.4"},
        )
        from hermes_cli.runtime_provider import resolve_runtime_provider

        result = await resolve_runtime_provider(requested="copilot")

        assert result["provider"] == "copilot"
        assert result["api_mode"] == "codex_responses"

    @pytest.mark.asyncio
    async def test_runtime_copilot_acp_uses_process_runtime(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.auth.shutil.which", lambda command: f"/usr/local/bin/{command}")
        monkeypatch.setenv("HERMES_COPILOT_ACP_ARGS", "--acp --stdio --debug")

        from hermes_cli.runtime_provider import resolve_runtime_provider

        result = await resolve_runtime_provider(requested="copilot-acp")

        assert result["provider"] == "copilot-acp"
        assert result["api_mode"] == "chat_completions"
        assert result["command"] == "/usr/local/bin/copilot"
        assert result["args"] == ["--acp", "--stdio", "--debug"]


# =============================================================================
# Kimi Code auto-detection tests
# =============================================================================

MOONSHOT_DEFAULT_URL = "https://api.moonshot.ai/v1"


class TestResolveKimiBaseUrl:
    """Test _resolve_kimi_base_url() helper for key-prefix auto-detection."""

    def test_sk_kimi_prefix_routes_to_kimi_code(self):
        url = _resolve_kimi_base_url("sk-kimi-abc123", MOONSHOT_DEFAULT_URL, "")
        assert url == KIMI_CODE_BASE_URL


    def test_env_override_wins_over_legacy(self):
        custom = "https://custom.example.com/v1"
        url = _resolve_kimi_base_url("sk-abc123", MOONSHOT_DEFAULT_URL, custom)
        assert url == custom


class TestKimiCodeStatusAutoDetect:
    """Test that get_api_key_provider_status auto-detects sk-kimi- keys."""


    @pytest.mark.asyncio
    async def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "sk-kimi-test-key")
        monkeypatch.setenv("KIMI_BASE_URL", "https://override.example/v1")
        status = await get_api_key_provider_status("kimi-coding")
        assert status["base_url"] == "https://override.example/v1"


class TestKimiCodeCredentialAutoDetect:
    """Test that resolve_api_key_provider_credentials auto-detects sk-kimi- keys."""


    @pytest.mark.asyncio
    async def test_legacy_key_gets_moonshot_url(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "sk-legacy-secret-key")
        creds = await resolve_api_key_provider_credentials("kimi-coding")
        assert creds["api_key"] == "sk-legacy-secret-key"
        assert creds["base_url"] == MOONSHOT_DEFAULT_URL


    @pytest.mark.asyncio
    async def test_non_kimi_providers_unaffected(self, monkeypatch):
        """Ensure the auto-detect logic doesn't leak to other providers."""
        monkeypatch.setenv("GLM_API_KEY", "sk-kim...isnt")
        async def _no_endpoint(*args, **kwargs):
            return None

        monkeypatch.setattr("hermes_cli.auth.detect_zai_endpoint", _no_endpoint)
        creds = await resolve_api_key_provider_credentials("zai")
        assert creds["base_url"] == "https://api.z.ai/api/paas/v4"


class TestZaiEndpointAutoDetect:
    """Test that resolve_api_key_provider_credentials auto-detects Z.AI endpoints."""


    @pytest.mark.asyncio
    async def test_env_override_skips_probe(self, monkeypatch):
        """GLM_BASE_URL should always win without probing."""
        monkeypatch.setenv("GLM_API_KEY", "glm-key")
        monkeypatch.setenv("GLM_BASE_URL", "https://custom.example/v4")
        probe_called = False

        def _never_called(*a, **kw):
            nonlocal probe_called
            probe_called = True
            return None

        monkeypatch.setattr("hermes_cli.auth.detect_zai_endpoint", _never_called)
        creds = await resolve_api_key_provider_credentials("zai")
        assert creds["base_url"] == "https://custom.example/v4"
        assert not probe_called

    @pytest.mark.asyncio
    async def test_no_key_skips_probe(self, monkeypatch):
        """Without an API key, no probe should occur."""
        async def _no_endpoint(*args, **kwargs):
            return None

        monkeypatch.setattr("hermes_cli.auth.detect_zai_endpoint", _no_endpoint)
        creds = await resolve_api_key_provider_credentials("zai")
        assert creds["api_key"] == ""


# =============================================================================
# Hugging Face provider model list tests
# =============================================================================

class TestHuggingFaceModels:
    """Verify Hugging Face model lists are consistent across all locations."""


    def test_model_metadata_has_context_lengths(self):
        """Every HF model should have a context length entry."""
        from hermes_cli.models import _PROVIDER_MODELS
        from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS
        lower_keys = {k.lower() for k in DEFAULT_CONTEXT_LENGTHS}
        hf_models = _PROVIDER_MODELS["huggingface"]
        for model in hf_models:
            assert model.lower() in lower_keys, (
                f"HF model {model!r} missing from DEFAULT_CONTEXT_LENGTHS"
            )


# =============================================================================
# NovitaAI provider tests (added by feat/add-novita-provider)
# =============================================================================

class TestNovitaProvider:
    """Tests for NovitaAI — an OpenAI-compatible multi-model aggregator."""

    def test_novita_profile_loads(self):
        from providers import get_provider_profile
        profile = get_provider_profile("novita")
        assert profile is not None
        assert profile.name == "novita"
        assert profile.display_name == "NovitaAI"
        assert profile.base_url == "https://api.novita.ai/openai/v1"
        assert "NOVITA_API_KEY" in profile.env_vars





# =============================================================================
# MiniMax OAuth provider tests (added by feat/minimax-oauth-provider)
# =============================================================================

class TestMinimaxOAuthProvider:
    """Tests for the minimax-oauth OAuth provider."""

    def test_minimax_oauth_in_provider_registry(self):
        assert "minimax-oauth" in PROVIDER_REGISTRY
        pconfig = PROVIDER_REGISTRY["minimax-oauth"]
        assert pconfig.auth_type == "oauth_minimax"
        assert pconfig.id == "minimax-oauth"


    def test_minimax_oauth_aux_model_registered(self):
        # Aux model for the minimax-oauth provider now lives on the
        # ProviderProfile (plugins/model-providers/minimax/__init__.py),
        # not the legacy _API_KEY_PROVIDER_AUX_MODELS dict in
        # agent/auxiliary_client.py. The profile layer is the source
        # of truth; _get_aux_model_for_provider() reads from it first
        # and only falls back to the dict when no profile is registered.
        import model_tools  # noqa: F401  -- triggers plugin discovery
        import providers

        profile = providers.get_provider_profile("minimax-oauth")
        assert profile is not None, "minimax-oauth provider profile must be registered"
        assert profile.default_aux_model, (
            "minimax-oauth profile must advertise a non-empty default_aux_model "
            "so the auxiliary client (compression / vision / session-search) "
            "doesn't fire the 'No auxiliary LLM provider configured' warning "
            "for every minimax-oauth session."
        )


# =============================================================================
# DeepInfra provider tests
# =============================================================================
# Registration / alias / env-var invariants are asserted in
# TestProviderRegistry + TestResolveProvider above. The classes below
# cover the catalog/tag/pricing/profile machinery added on top of the
# baseline provider wiring.


class TestDeepInfraProviderProfile:
    """plugins/model-providers/deepinfra registration + aux resolution."""

    @pytest.mark.asyncio
    async def test_profile_registered_with_alias_and_aux(self):
        from providers import get_provider_profile
        from agent.auxiliary_client import _get_aux_model_for_provider
        from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider
        from hermes_cli.config import OPTIONAL_ENV_VARS
        from hermes_cli.models import CANONICAL_PROVIDERS

        profile = get_provider_profile("deepinfra")
        assert profile is not None
        assert profile.name == "deepinfra"
        assert profile.auth_type == "api_key"
        # Alias resolves to the same profile.
        assert get_provider_profile("deep-infra") is profile
        assert await resolve_provider("deep-infra") == "deepinfra"
        assert PROVIDER_REGISTRY["deepinfra"].inference_base_url == profile.base_url
        assert any(entry.slug == "deepinfra" for entry in CANONICAL_PROVIDERS)
        assert OPTIONAL_ENV_VARS["DEEPINFRA_API_KEY"]["password"] is True
        assert OPTIONAL_ENV_VARS["DEEPINFRA_BASE_URL"]["password"] is False
        # Aux model is resolved via the profile (not via the legacy
        # _API_KEY_PROVIDER_AUX_MODELS_FALLBACK dict, which has no
        # deepinfra entry).
        assert _get_aux_model_for_provider("deepinfra")
        # Fallback list intentionally empty — live catalog is the source
        # of truth. Pin the shape only, not contents.
        assert isinstance(profile.fallback_models, tuple)
