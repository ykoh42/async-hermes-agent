"""Per-model auxiliary client-cache isolation.

Two auxiliary calls to the same provider/base_url/key but different models
must get distinct cache keys so a concurrent MoA fan-out never shares one
client entry across models.
"""

from __future__ import annotations

import pytest


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
