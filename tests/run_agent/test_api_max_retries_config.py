"""Tests for agent.api_max_retries config surface.

Closes #11616 — make the hardcoded ``max_retries = 3`` in the agent's API
retry loop user-configurable so fallback-provider setups can fail over
faster on flaky primaries instead of burning ~3x180s on the same stall.
"""
from unittest.mock import AsyncMock, patch

import pytest

from run_agent import AIAgent


async def _make_agent(api_max_retries=None):
    """Build an AIAgent with a mocked config.load_config that returns a
    config tree containing the given agent.api_max_retries (or default)."""
    cfg = {"agent": {}}
    if api_max_retries is not None:
        cfg["agent"]["api_max_retries"] = api_max_retries

    with patch("run_agent.OpenAI"), \
         patch("hermes_cli.config.load_config", return_value=cfg), \
         patch(
             "hermes_cli.config.load_config_readonly",
             new_callable=AsyncMock,
             return_value=cfg,
         ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        await agent._ensure_provider_runtime()
        return agent


@pytest.mark.asyncio
async def test_default_api_max_retries_is_three():
    """No config override → legacy default of 3 retries preserved."""
    agent = await _make_agent()
    assert agent._api_max_retries == 3


@pytest.mark.asyncio
async def test_api_max_retries_honors_config_override():
    """Setting agent.api_max_retries in config propagates to the agent."""
    agent = await _make_agent(api_max_retries=1)
    assert agent._api_max_retries == 1

    agent2 = await _make_agent(api_max_retries=5)
    assert agent2._api_max_retries == 5



