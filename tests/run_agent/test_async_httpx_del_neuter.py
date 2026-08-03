"""Tests for the AsyncHttpxClientWrapper.__del__ neuter fix.

The OpenAI SDK's ``AsyncHttpxClientWrapper.__del__`` schedules
``aclose()`` via ``asyncio.get_running_loop().create_task()``.  When GC
fires during CLI idle time, prompt_toolkit's event loop picks up the task
and crashes with "Event loop is closed" because the underlying TCP
transport is bound to a dead worker loop.

The three-layer defence:
1. ``neuter_async_httpx_del()`` replaces ``__del__`` with a no-op.
2. A custom asyncio exception handler silences residual errors.
3. ``cleanup_stale_clients()`` evicts stale cache entries.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Layer 1: neuter_async_httpx_del
# ---------------------------------------------------------------------------

class TestNeuterAsyncHttpxDel:
    """Verify neuter_async_httpx_del replaces __del__ on the SDK class."""

    def test_del_becomes_noop(self):
        """After neuter, __del__ should do nothing (no RuntimeError)."""
        from agent.auxiliary_client import neuter_async_httpx_del

        try:
            from openai._base_client import AsyncHttpxClientWrapper
        except ImportError:
            pytest.skip("openai SDK not installed")

        # Save original so we can restore
        original_del = AsyncHttpxClientWrapper.__del__
        try:
            neuter_async_httpx_del()
            # The patched __del__ should be a no-op lambda
            assert AsyncHttpxClientWrapper.__del__ is not original_del
            # Calling it should not raise, even without a running loop
            wrapper = MagicMock(spec=AsyncHttpxClientWrapper)
            AsyncHttpxClientWrapper.__del__(wrapper)  # Should be silent
        finally:
            # Restore original to avoid leaking into other tests
            AsyncHttpxClientWrapper.__del__ = original_del

    def test_neuter_idempotent(self):
        """Calling neuter twice doesn't break anything."""
        from agent.auxiliary_client import neuter_async_httpx_del

        try:
            from openai._base_client import AsyncHttpxClientWrapper
        except ImportError:
            pytest.skip("openai SDK not installed")

        original_del = AsyncHttpxClientWrapper.__del__
        try:
            neuter_async_httpx_del()
            first_del = AsyncHttpxClientWrapper.__del__
            neuter_async_httpx_del()
            second_del = AsyncHttpxClientWrapper.__del__
            # Both calls should succeed; the class should have a no-op
            assert first_del is not original_del
            assert second_del is not original_del
        finally:
            AsyncHttpxClientWrapper.__del__ = original_del



# ---------------------------------------------------------------------------
# Layer 3: cleanup_stale_clients
# ---------------------------------------------------------------------------

class TestCleanupStaleAsyncClients:
    """Verify that cache lifecycle remains on the event loop."""

    @pytest.mark.asyncio
    async def test_removes_stale_entries(self):
        from agent.auxiliary_client import (
            _client_cache,
            _client_cache_lock,
            cleanup_stale_clients,
        )

        closed_loop = asyncio.new_event_loop()
        closed_loop.close()
        key = ("test_stale", "", "", "", (), False, "", "", "test-model")
        client = MagicMock()
        client._client = MagicMock(is_closed=False)
        async with _client_cache_lock:
            _client_cache[key] = (client, "test-model", closed_loop)

        await cleanup_stale_clients()

        async with _client_cache_lock:
            assert key not in _client_cache

    @pytest.mark.asyncio
    async def test_keeps_current_loop_entries(self):
        from agent.auxiliary_client import (
            _client_cache,
            _client_cache_lock,
            cleanup_stale_clients,
        )

        key = ("test_live", "", "", "", (), False, "", "", "test-model")
        async with _client_cache_lock:
            _client_cache[key] = (
                MagicMock(),
                "test-model",
                asyncio.get_running_loop(),
            )

        await cleanup_stale_clients()

        async with _client_cache_lock:
            assert key in _client_cache
            _client_cache.pop(key, None)


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
