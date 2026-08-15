"""Native-async authorization-gate regressions.

The upstream file also covered blocking CLI/gateway human-wait accounting.
Those transports are deliberately removed from this library distribution, so
their source-level tests are not restored.  The retained gate invariants are
the bounded native lock and the rule that arbitrary callback residency never
extends a batch deadline.
"""

import asyncio
import time

import pytest

from agent.tool_executor import _ConcurrentToolAuthorizationGate


@pytest.mark.asyncio
async def test_serializes_callbacks():
    gate = _ConcurrentToolAuthorizationGate(lock_timeout=1.0)
    active = 0
    maximum = 0
    guard = asyncio.Lock()

    async def _callback():
        nonlocal active, maximum
        async with guard:
            active += 1
            maximum = max(maximum, active)
        try:
            await asyncio.sleep(0.03)
        finally:
            async with guard:
                active -= 1

    await asyncio.gather(*(gate.run(_callback) for _ in range(4)))
    assert maximum == 1


@pytest.mark.asyncio
async def test_lock_timeout_degrades_to_unserialized():
    gate = _ConcurrentToolAuthorizationGate(lock_timeout=0.1)
    holder_started = asyncio.Event()
    release = asyncio.Event()

    async def _wedged():
        holder_started.set()
        await release.wait()

    holder = asyncio.create_task(gate.run(_wedged))
    await asyncio.wait_for(holder_started.wait(), timeout=1)
    started = time.monotonic()
    assert await gate.run(lambda: "ran-unserialized") == "ran-unserialized"
    assert time.monotonic() - started < 1.0
    release.set()
    await asyncio.wait_for(holder, timeout=1)


@pytest.mark.asyncio
async def test_wedged_callback_contributes_nothing_to_exclusion():
    gate = _ConcurrentToolAuthorizationGate(lock_timeout=0.1)
    started = asyncio.Event()
    release = asyncio.Event()

    async def _wedged():
        started.set()
        await release.wait()

    holder = asyncio.create_task(gate.run(_wedged))
    await asyncio.wait_for(started.wait(), timeout=1)
    deadline = time.monotonic() + 0.3
    first = deadline + gate.excluded_seconds() - time.monotonic()
    await asyncio.sleep(0.15)
    second = deadline + gate.excluded_seconds() - time.monotonic()
    assert second < first
    await asyncio.sleep(0.2)
    assert deadline + gate.excluded_seconds() - time.monotonic() <= 0
    release.set()
    await asyncio.wait_for(holder, timeout=1)
