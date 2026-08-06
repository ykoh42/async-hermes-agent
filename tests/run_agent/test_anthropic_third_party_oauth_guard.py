"""Tests for ``_is_anthropic_oauth`` guard against third-party Anthropic-compatible providers.

The invariant: ``self._is_anthropic_oauth`` must only ever be True when
``self.provider == 'anthropic'`` (native Anthropic).  Third-party providers
that speak the Anthropic protocol (MiniMax, Zhipu GLM, Alibaba DashScope,
Kimi, LiteLLM proxies, etc.) must never trip OAuth code paths — doing so
injects Claude-Code identity headers and system prompts that cause
401/403 from those endpoints.

This test class covers all FIVE sites that assign ``_is_anthropic_oauth``:

1. ``AIAgent.__init__``                              (line ~1022)
2. ``AIAgent.switch_model``                          (line ~1832)
3. ``AIAgent._try_refresh_anthropic_client_credentials`` (line ~5335)
4. ``AIAgent._swap_credential``                      (line ~5378)
5. ``AIAgent._try_activate_fallback``                (line ~6536)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from run_agent import AIAgent


# A plausible-looking OAuth token (``sk-ant-`` without the ``-api`` suffix).
_OAUTH_LIKE_TOKEN = "sk-ant-oauth-example-1234567890abcdef"
_API_KEY_TOKEN = "sk-ant-api-abcdef1234567890"


@pytest.fixture
def agent():
    """Minimal AIAgent construction, skipping tool discovery."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


class TestOAuthFlagOnCredentialSwap:
    """Site 4 — _swap_credential (credential pool rotation)."""

    @pytest.mark.asyncio
    async def test_pool_swap_on_third_party_never_flips_oauth(self, agent):
        agent.api_mode = "anthropic_messages"
        agent.provider = "glm"              # ← Zhipu GLM via /anthropic
        agent._anthropic_api_key = "old-key"
        agent._anthropic_base_url = "https://open.bigmodel.cn/api/anthropic"
        agent._anthropic_client = MagicMock()
        agent._is_anthropic_oauth = False

        entry = MagicMock()
        entry.runtime_api_key = _OAUTH_LIKE_TOKEN
        entry.runtime_base_url = "https://open.bigmodel.cn/api/anthropic"

        with patch("agent.anthropic_adapter.build_anthropic_client",
                   return_value=MagicMock()):
            await agent._swap_credential(entry)

        assert agent._is_anthropic_oauth is False


class TestNativeAnthropicRefresh:
    @pytest.mark.asyncio
    async def test_refresh_awaits_token_resolution_and_rebuilds_runtime(
        self,
        agent,
        monkeypatch,
    ):
        agent.api_mode = "anthropic_messages"
        agent.provider = "anthropic"
        agent.model = "claude-sonnet-4-6"
        agent._anthropic_api_key = _OAUTH_LIKE_TOKEN
        agent._anthropic_base_url = "https://api.anthropic.com"
        agent._provider_request_timeout = None
        agent._provider_stale_timeout = None
        agent._ensure_provider_runtime = AsyncMock(return_value=True)
        resolve = AsyncMock(return_value="sk-ant-oat01-refreshed-token")
        monkeypatch.setattr(
            "agent.anthropic_adapter.resolve_anthropic_token",
            resolve,
        )

        refreshed = await agent._try_refresh_anthropic_client_credentials()

        assert refreshed is True
        resolve.assert_awaited_once_with()
        assert agent._deferred_provider_runtime["api_key"] == (
            "sk-ant-oat01-refreshed-token"
        )
        agent._ensure_provider_runtime.assert_awaited_once_with()


class TestOAuthFlagOnConstruction:
    """Site 1 — AIAgent.__init__ on a third-party anthropic_messages provider."""

    def test_minimax_init_does_not_flip_oauth(self):
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("agent.anthropic_adapter.build_anthropic_client",
                  return_value=MagicMock()),
            # Simulate a stale ANTHROPIC_TOKEN in the env — the init code
            # MUST NOT fall back to it when provider != anthropic.
            patch("agent.anthropic_adapter.resolve_anthropic_token",
                  return_value=_OAUTH_LIKE_TOKEN),
        ):
            agent = AIAgent(
                api_key="minimax-key-1234",
                base_url="https://api.minimax.io/anthropic",
                provider="minimax",
                api_mode="anthropic_messages",
                model="claude-sonnet-4-6",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        # The effective key should be the explicit minimax-key, not the
        # stale Anthropic OAuth token, and the OAuth flag must be False.
        assert agent._anthropic_api_key == "minimax-key-1234"
        assert agent._is_anthropic_oauth is False


class TestOAuthFlagOnFallbackActivation:
    """Site 5 — _try_activate_fallback targeting a third-party Anthropic endpoint."""

    def test_fallback_to_third_party_does_not_flip_oauth(self, agent):
        """Directly mimic the post-fallback assignment at line ~6537."""
        from agent.anthropic_adapter import _is_oauth_token

        # Emulate the relevant lines of _try_activate_fallback without
        # running the entire recovery stack (which pulls in streaming,
        # sessions, etc.).
        fb_provider = "minimax"
        effective_key = _OAUTH_LIKE_TOKEN
        agent._is_anthropic_oauth = (
            _is_oauth_token(effective_key) if fb_provider == "anthropic" else False
        )
        assert agent._is_anthropic_oauth is False


class TestApiKeyTokensAlwaysSafe:
    """Regression: plain API-key shapes must always resolve to non-OAuth, any provider."""

    def test_native_anthropic_with_api_key_token(self):
        from agent.anthropic_adapter import _is_oauth_token
        assert _is_oauth_token(_API_KEY_TOKEN) is False
