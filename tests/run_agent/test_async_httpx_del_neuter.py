"""Tests for event-loop ownership in the auxiliary client cache."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestClientCacheBoundedGrowth:
    @pytest.mark.asyncio
    async def test_same_key_replaces_stale_loop_entry(self):
        from agent.auxiliary_client import (
            _client_cache,
            _client_cache_key,
            _client_cache_lock,
            _get_cached_client,
        )

        key = await _client_cache_key("test_replace", task="")
        old_loop = asyncio.new_event_loop()
        old_loop.close()
        async with _client_cache_lock:
            _client_cache[key] = (MagicMock(), "old-model", old_loop)

        new_client = MagicMock()
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            new=AsyncMock(return_value=(new_client, "new-model")),
        ):
            client, model = await _get_cached_client("test_replace")

        assert client is new_client
        assert model == "new-model"
        async with _client_cache_lock:
            assert _client_cache[key][1] == "new-model"
            _client_cache.pop(key, None)
