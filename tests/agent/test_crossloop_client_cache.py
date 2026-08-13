"""Tests for cross-loop client cache isolation fix (#2681).

Verifies that _get_cached_client() returns different AsyncOpenAI clients
when called from different event loops, preventing the httpx deadlock
that occurs when a cached async client bound to loop A is reused on loop B.

This test file is self-contained and does not import the full tool chain,
so it can run without optional dependencies like firecrawl.
"""

import asyncio
import threading
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Minimal stubs so we can import _get_cached_client without the full tree
# ---------------------------------------------------------------------------

async def _stub_resolve_provider_client(provider, model, **kw):
    """Return a unique mock client each time, simulating AsyncOpenAI creation."""
    client = MagicMock(name=f"client-{provider}-async")
    client.api_key = "test"
    client.base_url = kw.get("explicit_base_url", "http://localhost:8081/v1")
    return client, model or "test-model"


@pytest.fixture(autouse=True)
def _clean_client_cache():
    """Clear the client cache before each test."""
    # We need to patch before importing
    with patch.dict("sys.modules", {}):
        pass
    # Import and clear
    import agent.auxiliary_client as ac
    ac._client_cache.clear()
    yield
    ac._client_cache.clear()


class TestCrossLoopCacheIsolation:
    """Verify async clients are cached per-event-loop, not globally."""


    def test_different_loops_get_different_clients(self):
        """Different event loops must get separate client instances."""
        from agent.auxiliary_client import _get_cached_client

        results = {}

        def _get_client_on_new_loop(name):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                with patch("agent.auxiliary_client.resolve_provider_client",
                            side_effect=_stub_resolve_provider_client):
                    client, _ = loop.run_until_complete(
                        _get_cached_client(
                            "custom", "m1", base_url="http://localhost:8081/v1"
                        )
                    )
                results[name] = (client, loop)
                from agent.auxiliary_client import shutdown_cached_clients

                loop.run_until_complete(
                    shutdown_cached_clients()
                )
            finally:
                loop.close()

        t1 = threading.Thread(target=_get_client_on_new_loop, args=("a",))
        t2 = threading.Thread(target=_get_client_on_new_loop, args=("b",))
        t1.start(); t1.join()
        t2.start(); t2.join()

        client_a, loop_a = results["a"]
        client_b, loop_b = results["b"]

        assert loop_a is not loop_b, "Test setup error: same loop on both threads"
        assert client_a is not client_b, (
            "Different event loops got the SAME cached client — this causes "
            "httpx cross-loop deadlocks in gateway mode (#2681)"
        )


    def test_gateway_simulation_no_deadlock(self):
        """Simulate two independent event loops with asyncio.run(),
        which creates a new loop. The cached client must be created on THAT loop,
        not reused from a different one."""
        from agent.auxiliary_client import _get_cached_client

        # Simulate: first call on "gateway loop"
        gateway_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(gateway_loop)

        with patch("agent.auxiliary_client.resolve_provider_client",
                    side_effect=_stub_resolve_provider_client):
            gateway_client, _ = gateway_loop.run_until_complete(
                _get_cached_client("custom", "m1", base_url="http://localhost:8081/v1")
            )

        # Simulate a worker thread that owns its own asyncio.run() loop.
        worker_client_id = [None]
        def _worker():
            async def _inner():
                from agent.auxiliary_client import shutdown_cached_clients

                try:
                    with patch("agent.auxiliary_client.resolve_provider_client",
                                side_effect=_stub_resolve_provider_client):
                        client, _ = await _get_cached_client(
                            "custom", "m1", base_url="http://localhost:8081/v1"
                        )
                    worker_client_id[0] = id(client)
                finally:
                    await shutdown_cached_clients()
            asyncio.run(_inner())

        t = threading.Thread(target=_worker)
        t.start()
        t.join()

        assert worker_client_id[0] != id(gateway_client), (
            "Worker thread (asyncio.run) got the gateway's cached client — "
            "this is the exact cross-loop scenario that causes httpx deadlocks. "
            "The cache key must include the event loop identity (#2681)"
        )
        from agent.auxiliary_client import shutdown_cached_clients

        gateway_loop.run_until_complete(shutdown_cached_clients())
        gateway_loop.close()
