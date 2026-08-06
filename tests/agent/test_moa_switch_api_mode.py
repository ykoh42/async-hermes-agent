"""MoA switching uses the native async facade."""

from __future__ import annotations

import types

import pytest


def _make_fake_agent():
    """A minimal stand-in carrying only the attributes switch_model touches."""
    agent = types.SimpleNamespace()
    agent.model = "minimax-m3"
    agent.provider = "opencode-go"
    agent.api_mode = "anthropic_messages"
    agent.api_key = "old-key"
    agent.base_url = "https://old.example/v1"
    agent.client = object()
    agent._client_kwargs = {"base_url": "https://old.example/v1"}
    agent._config_context_length = 123456
    agent._transport_cache = {}
    agent.quiet_mode = True
    agent._runtime_config_loaded = True
    agent._context_engine_started = True
    agent._anthropic_prompt_cache_policy = lambda: (False, False)

    async def persist_pending_billing_route():
        return None

    agent._persist_pending_billing_route = persist_pending_billing_route
    return agent


@pytest.mark.asyncio
async def test_switch_to_moa_uses_native_async_facade():
    from agent import agent_runtime_helpers as arh

    agent = _make_fake_agent()

    async def ensure_provider_runtime():
        from agent.agent_init import initialize_deferred_runtime

        return await initialize_deferred_runtime(agent)

    agent._ensure_provider_runtime = ensure_provider_runtime
    await arh.switch_model(
        agent,
        new_model="frontier",
        new_provider="moa",
        api_key="moa-virtual-provider",
        base_url="moa://local",
        api_mode="chat_completions",
    )

    assert agent.provider == "moa"
    assert agent.model == "frontier"
    assert agent.api_mode == "chat_completions"
    assert agent.base_url == "moa://local"
    assert hasattr(agent.client.chat, "completions")
