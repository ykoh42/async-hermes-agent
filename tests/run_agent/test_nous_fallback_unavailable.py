"""Tests for Nous fallback local-availability suppression.

Blocker if Nous token material is missing locally: the fallback chain
should not repeatedly attempt Nous resolution; it must skip and continue
to the next provider.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from run_agent import AIAgent


def _make_agent(fallback_model=None):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = None
        return agent


class TestNousFallbackLocalAvailability:
    @pytest.mark.asyncio
    async def test_missing_nous_token_is_skipped_once(self):
        """Nous fallback is skipped when no access/refresh token is stored."""
        agent = _make_agent(
            fallback_model=[
                {"provider": "nous", "model": "anthropic/claude-sonnet-4.6"},
                {
                    "provider": "custom",
                    "model": "gpt-5.5",
                    "base_url": "https://fallback.example/v1",
                    "api_key": "fb-key",
                },
            ]
        )
        with patch("run_agent.OpenAI", return_value=MagicMock()):
            activated = await agent._try_activate_fallback(None)
        assert activated is True
        assert agent.model == "gpt-5.5"

    @pytest.mark.asyncio
    async def test_nous_unavailable_not_retried_in_same_session(self):
        """After Nous is skipped once, subsequent activations continue further."""
        agent = _make_agent(
            fallback_model=[
                {"provider": "nous", "model": "anthropic/claude-sonnet-4.6"},
                {"provider": "openai-codex", "model": "gpt-5.5"},
            ]
        )
        await agent._try_activate_fallback(None)
        key = (
            "nous",
            "anthropic/claude-sonnet-4.6",
            "",
        )
        assert key in getattr(agent, "_unavailable_fallback_keys", set())

    @pytest.mark.asyncio
    async def test_present_nous_token_allows_activation(self):
        """Nous is considered when token material exists."""
        agent = _make_agent(
            fallback_model=[
                {
                    "provider": "nous",
                    "model": "anthropic/claude-sonnet-4.6",
                    "base_url": "https://inference-api.nousresearch.com/v1",
                    "api_key": "portal-jwt",
                },
                {"provider": "openai-codex", "model": "gpt-5.5"},
            ]
        )
        with patch(
            "agent.anthropic_adapter.build_anthropic_client",
            return_value=object(),
        ):
            activated = await agent._try_activate_fallback(None)
        assert activated is True
        assert agent.provider == "nous"
