"""Live context-window parity at the first awaited runtime boundary."""

from unittest.mock import AsyncMock, patch

import pytest

from run_agent import AIAgent


def _runtime_patches(context_length: int):
    config = {}
    return (
        patch("hermes_cli.config.load_config", return_value=config),
        patch(
            "hermes_cli.config.load_config_readonly",
            new_callable=AsyncMock,
            return_value=config,
        ),
        patch(
            "agent.model_metadata.get_model_context_length",
            new_callable=AsyncMock,
            return_value=context_length,
        ),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    )


def _new_agent() -> AIAgent:
    return AIAgent(
        model="unknown-live-context-model",
        provider="openrouter",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


@pytest.mark.asyncio
async def test_builtin_engine_awaits_live_context_window_before_first_turn():
    patches = _runtime_patches(128_000)
    with patches[0], patches[1], patches[2] as context_length, patches[3], patches[4], patches[5]:
        agent = _new_agent()
        try:
            # Construction uses the non-I/O fallback only; the first awaited
            # boundary must replace it with authoritative provider metadata.
            assert agent.context_compressor.context_length == 256_000

            await agent._ensure_provider_runtime()

            context_length.assert_awaited_once_with(
                "unknown-live-context-model",
                base_url="https://openrouter.ai/api/v1",
                api_key="test-key",
                config_context_length=None,
                provider="openrouter",
                custom_providers=[],
            )
            assert agent.context_compressor.context_length == 128_000
            assert agent.context_compressor.threshold_percent == 0.75
            assert agent.context_compressor.threshold_tokens == 96_000
            assert agent._primary_runtime["compressor_context_length"] == 128_000
            assert agent._primary_runtime["compressor_threshold_tokens"] == 96_000
        finally:
            await agent.close()


@pytest.mark.asyncio
async def test_live_subminimum_window_is_rejected_at_awaited_startup():
    patches = _runtime_patches(32_768)
    with patches[0], patches[1], patches[2] as context_length, patches[3], patches[4], patches[5]:
        agent = _new_agent()
        try:
            with pytest.raises(
                ValueError,
                match=(
                    r"Model unknown-live-context-model has a context window of "
                    r"32,768 tokens, which is below the minimum 64,000 required"
                ),
            ):
                await agent._ensure_provider_runtime()
            context_length.assert_awaited_once()
        finally:
            await agent.close()
