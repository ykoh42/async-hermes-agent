"""Follow-up for the cross-turn stream-stale circuit breaker (#58962).

The breaker latches: once ``_consecutive_stale_streams`` reaches the give-up
threshold, ``interruptible_streaming_api_call`` raises BEFORE any stream is
attempted — so the "reset on successful stream" path can never run again on
its own. The breaker's error message tells the user to "switch models …
then retry", and the provider-fallback chain swaps providers on the same
agent object, so BOTH swap paths must clear the streak or a healthy new
provider would keep short-circuiting forever:

- ``switch_model()``   (user-initiated /model swap)
- ``try_activate_fallback()``  (automatic provider fallback)
- ``restore_primary_runtime()``  (turn-start restore back to the primary)

"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from run_agent import AIAgent


def _make_agent_openrouter():
    """Minimal openrouter agent (skips __init__), mirroring
    tests/run_agent/test_switch_model_rollback.py."""
    agent = AIAgent.__new__(AIAgent)

    agent.provider = "openrouter"
    agent.model = "x-ai/grok-4"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "or-key-original"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock(name="OriginalClient")
    agent._client_kwargs = {
        "api_key": "or-key-original",
        "base_url": "https://openrouter.ai/api/v1",
    }
    agent.context_compressor = None
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = None
    agent._anthropic_client = None
    agent._is_anthropic_oauth = False
    agent._cached_system_prompt = "cached"
    agent._primary_runtime = {}
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = []
    agent._fallback_model = None
    agent._config_context_length = None

    return agent


def _make_fallback_agent(fallback_model):
    """Full-constructor agent for the fallback path, mirroring
    tests/run_agent/test_24996_fallback_exhaustion_cooldown.py."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _mock_client(base_url="https://openrouter.ai/api/v1", api_key="fb-key"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = api_key
    return mock


@pytest.mark.asyncio
async def test_switch_model_resets_stale_streak():
    """A user-initiated /model swap must clear the latched streak so the new
    provider gets a real stream attempt instead of an instant short-circuit."""
    agent = _make_agent_openrouter()
    agent._consecutive_stale_streams = 7  # past any reasonable threshold

    async def initialize_runtime():
        pending = agent._deferred_provider_runtime
        agent.provider = agent.requested_provider = pending["provider"]
        agent.model = pending["model"]
        agent.api_key = pending["api_key"]
        agent.base_url = pending["base_url"]
        agent.api_mode = pending["api_mode"]
        agent._deferred_provider_runtime = None

    agent._ensure_provider_runtime = AsyncMock(side_effect=initialize_runtime)
    agent._persist_pending_billing_route = AsyncMock()

    with patch("hermes_cli.config.load_config_readonly", new_callable=AsyncMock, return_value={}):
        await agent.switch_model(
            new_model="openai/gpt-5",
            new_provider="openrouter",
            api_key="or-key-new",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
        )

    assert agent._consecutive_stale_streams == 0


@pytest.mark.asyncio
async def test_switch_model_failure_does_not_reset_streak():
    """A failed swap rolls back — the agent is still on the wedged provider,
    so the breaker must stay latched (reset happens after the rebuild)."""
    agent = _make_agent_openrouter()
    agent._consecutive_stale_streams = 7

    agent._ensure_provider_runtime = AsyncMock(
        side_effect=RuntimeError("simulated client build failure")
    )

    with patch("hermes_cli.config.load_config_readonly", new_callable=AsyncMock, return_value={}):
        try:
            await agent.switch_model(
                new_model="openai/gpt-5",
                new_provider="openrouter",
                api_key="or-key-new",
                base_url="https://openrouter.ai/api/v1",
                api_mode="chat_completions",
            )
        except RuntimeError:
            pass

    assert agent._consecutive_stale_streams == 7


@pytest.mark.asyncio
async def test_fallback_activation_resets_stale_streak():
    """Automatic provider fallback swaps to a different backend; the streak
    measured the OLD provider and must not wedge the new one."""
    fbs = [{
        "provider": "openai",
        "model": "gpt-4o",
        "api_key": "fb-key",
        "base_url": "https://api.openai.com/v1",
    }]
    agent = _make_fallback_agent(fallback_model=fbs)
    agent._consecutive_stale_streams = 7

    with patch("openai.AsyncOpenAI", return_value=_mock_client()):
        assert await agent._try_activate_fallback() is True

    assert agent._consecutive_stale_streams == 0


@pytest.mark.asyncio
async def test_fallback_exhaustion_keeps_stale_streak():
    """When the chain is exhausted (no swap happened), the streak stays
    latched — the session is still wedged on the same provider."""
    agent = _make_fallback_agent(fallback_model=[])
    agent._consecutive_stale_streams = 7

    assert await agent._try_activate_fallback() is False
    assert agent._consecutive_stale_streams == 7
