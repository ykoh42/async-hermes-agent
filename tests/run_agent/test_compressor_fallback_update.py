"""Tests that _try_activate_fallback updates the context compressor."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from run_agent import AIAgent
from agent.context_compressor import ContextCompressor


def _make_agent_with_compressor() -> AIAgent:
    """Build a minimal AIAgent with a context_compressor, skipping __init__."""
    agent = AIAgent.__new__(AIAgent)

    # Primary model settings
    agent.model = "primary-model"
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "sk-primary"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock()
    agent.quiet_mode = True

    # Fallback config
    agent._fallback_activated = False
    agent._fallback_model = {
        "provider": "openai",
        "model": "gpt-4o",
    }
    agent._fallback_chain = [agent._fallback_model]
    agent._fallback_index = 0

    # Context compressor with primary model values
    compressor = ContextCompressor(
        model="primary-model",
        threshold_percent=0.50,
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-primary",
        provider="openrouter",
        quiet_mode=True,
    )
    agent.context_compressor = compressor

    return agent


@patch("agent.model_metadata.get_static_context_length", return_value=128_000)
@pytest.mark.asyncio
async def test_compressor_updated_on_fallback(mock_ctx_len):
    """After fallback activation, the compressor must reflect the fallback model."""
    agent = _make_agent_with_compressor()

    assert agent.context_compressor.model == "primary-model"

    fb_client = MagicMock()
    fb_client.base_url = "https://api.openai.com/v1"
    fb_client.api_key = "sk-fallback"
    async def initialize_fallback_runtime():
        agent.client = fb_client
        agent.base_url = str(fb_client.base_url)
        agent.api_key = str(fb_client.api_key)
        agent.api_mode = "chat_completions"
        agent.provider = "openai"
        agent.model = "gpt-4o"
        return True

    agent._ensure_provider_runtime = AsyncMock(side_effect=initialize_fallback_runtime)
    agent._is_direct_openai_url = lambda url: "api.openai.com" in url
    agent._is_azure_openai_url = lambda _url: False
    agent._provider_model_requires_responses_api = lambda *_args, **_kwargs: False
    agent._anthropic_prompt_cache_policy = lambda **_kwargs: (False, False)
    agent._buffer_status = lambda _msg: None
    agent._emit_status = lambda msg: None

    with patch("agent.chat_completion_helpers.rewrite_prompt_model_identity"):
        result = await agent._try_activate_fallback()

    assert result is True
    assert agent._fallback_activated is True

    c = agent.context_compressor
    assert c.model == "gpt-4o"
    assert c.base_url == "https://api.openai.com/v1"
    assert c.api_key == "sk-fallback"
    assert c.provider == "openai"
    assert c.context_length == 128_000
    assert c.threshold_tokens == int(128_000 * c.threshold_percent)
