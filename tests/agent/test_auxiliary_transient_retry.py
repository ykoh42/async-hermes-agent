"""Per-model auxiliary client-cache isolation.

Two auxiliary calls to the same provider/base_url/key but different models
must get distinct cache keys so a concurrent MoA fan-out never shares one
client entry across models.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_transient_retry_count_default():
    from agent import auxiliary_client as ac

    with patch(
        "hermes_cli.config.load_config_readonly",
        new=AsyncMock(return_value={}),
    ):
        assert await ac._transient_retry_count() == ac._DEFAULT_TRANSIENT_RETRIES


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configured", "expected"),
    [(-1, 0), (0, 0), (4, 4), (20, 6), ("3", 3), ("invalid", 2)],
)
async def test_transient_retry_count_config_and_bounds(configured, expected):
    from agent import auxiliary_client as ac

    config = {"auxiliary": {"transient_retries": configured}}
    assert await ac._transient_retry_count(config) == expected


@pytest.mark.asyncio
async def test_model_participates_in_client_cache_key():
    """Same provider/base_url/key, different model -> different cache key.

    This is what stops two concurrent advisors from sharing (and racing on)
    one cached client entry."""
    from agent.auxiliary_client import _client_cache_key

    k_opus = await _client_cache_key(
        "openrouter", base_url="https://openrouter.ai/api/v1",
        api_key="K", model="anthropic/claude-opus-4.8",
    )
    k_gpt = await _client_cache_key(
        "openrouter", base_url="https://openrouter.ai/api/v1",
        api_key="K", model="openai/gpt-5.5",
    )
    assert k_opus != k_gpt
    # Same model still collides (cache still works for reuse).
    k_opus2 = await _client_cache_key(
        "openrouter", base_url="https://openrouter.ai/api/v1",
        api_key="K", model="anthropic/claude-opus-4.8",
    )
    assert k_opus == k_opus2
