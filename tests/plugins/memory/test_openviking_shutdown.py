"""Tests for OpenViking memory-provider shutdown teardown.

The runtime-autostart waiter is an owned asyncio task that waits on network
health probes. These tests assert that it short-circuits on shutdown and that
``shutdown()`` cancels and awaits it without leaking the task.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

import plugins.memory.openviking as openviking_module
from plugins.memory.openviking import OpenVikingMemoryProvider


pytestmark = pytest.mark.asyncio


async def test_wait_for_health_short_circuits_on_should_stop():
    """The health waiter returns False without probing when should_stop is set,
    so its owned task can finish promptly at shutdown."""
    probes: list[str] = []

    def _reach(endpoint):
        probes.append(endpoint)
        return (False, "down")

    with patch.object(
        openviking_module, "_validate_openviking_reachability", _reach
    ):
        result = await openviking_module._wait_for_openviking_health(
            "http://example.invalid",
            timeout_seconds=60.0,
            should_stop=lambda: True,
        )

    assert result is False
    assert probes == []  # bailed before the first network probe


async def test_shutdown_cancels_and_awaits_runtime_start_task():
    """shutdown() cancels and awaits the owned runtime-autostart task."""
    provider = OpenVikingMemoryProvider()
    started = asyncio.Event()
    finished = asyncio.Event()

    async def _runtime():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            finished.set()

    task = asyncio.create_task(_runtime(), name="openviking-runtime-start")
    provider._runtime_start_task = task
    provider._track_task(task)
    await asyncio.wait_for(started.wait(), timeout=2.0)

    await provider.shutdown()

    assert finished.is_set()
    assert task.done()
    assert task.cancelled()


async def test_cancelled_shutdown_finishes_cleanup_before_propagating_cancellation():
    """External cancellation cannot interrupt owned client/task cleanup."""
    provider = OpenVikingMemoryProvider()
    owned_started = asyncio.Event()
    client_close_started = asyncio.Event()
    release_client_close = asyncio.Event()

    async def _owned_work():
        owned_started.set()
        await asyncio.Event().wait()

    class _Client:
        closed = False

        async def close(self):
            client_close_started.set()
            await release_client_close.wait()
            self.closed = True

    client = _Client()
    provider._client = client
    owned_task = provider._track_task(asyncio.create_task(_owned_work()))
    await asyncio.wait_for(owned_started.wait(), timeout=1.0)

    shutdown_task = asyncio.create_task(provider.shutdown())
    await asyncio.wait_for(client_close_started.wait(), timeout=1.0)
    shutdown_task.cancel()
    release_client_close.set()

    with pytest.raises(asyncio.CancelledError):
        await shutdown_task

    assert owned_task.cancelled()
    assert client.closed is True
    assert provider._client is None
