"""Native-async regression tests for the concurrent start-order gate.

The upstream fix bounded the wait behind a wedged dispatch.  The retained
runtime expresses the same gate with asyncio tasks, so these tests exercise
the exact ordering/timeout invariant at that boundary rather than recreating
the removed synchronous worker executor.
"""

import asyncio
import time

import pytest

from agent.tool_executor import _OrderedToolStartGate


@pytest.mark.asyncio
async def test_wedged_dispatch_does_not_starve_later_tools(monkeypatch):
    gate = _OrderedToolStartGate()
    monkeypatch.setattr("agent.tool_executor._START_ORDER_GATE_TIMEOUT_S", 0.1)
    started = asyncio.Event()
    release = asyncio.Event()
    dispatched: list[str] = []

    async def _wedged():
        started.set()
        await release.wait()

    async def _dispatch(name: str):
        dispatched.append(name)

    first = asyncio.create_task(gate.advance(0, _wedged))
    await asyncio.wait_for(started.wait(), timeout=1)
    later = [
        asyncio.create_task(gate.advance(index, lambda n=name: _dispatch(n)))
        for index, name in ((1, "tool_b"), (2, "tool_c"))
    ]
    await asyncio.wait_for(asyncio.gather(*later), timeout=1)
    assert dispatched == ["tool_b", "tool_c"]
    release.set()
    await asyncio.wait_for(first, timeout=1)


@pytest.mark.asyncio
async def test_gate_timeout_stays_under_the_batch_deadline(monkeypatch):
    gate = _OrderedToolStartGate()
    monkeypatch.setattr("agent.tool_executor._START_ORDER_GATE_TIMEOUT_S", 0.1)
    started = asyncio.Event()
    release = asyncio.Event()
    dispatched: list[str] = []

    async def _wedged():
        started.set()
        await release.wait()

    first = asyncio.create_task(gate.advance(0, _wedged))
    await asyncio.wait_for(started.wait(), timeout=1)
    deadline = time.monotonic() + 0.5
    await asyncio.wait_for(
        gate.advance(1, lambda: dispatched.append("tool_b")),
        timeout=0.4,
    )
    assert dispatched == ["tool_b"]
    assert time.monotonic() < deadline
    release.set()
    await asyncio.wait_for(first, timeout=1)


@pytest.mark.asyncio
async def test_abandoned_batch_does_not_dispatch_late(monkeypatch):
    gate = _OrderedToolStartGate()
    monkeypatch.setattr("agent.tool_executor._START_ORDER_GATE_TIMEOUT_S", 30.0)
    started = asyncio.Event()
    release = asyncio.Event()
    dispatched: list[str] = []

    async def _wedged():
        started.set()
        await release.wait()

    first = asyncio.create_task(gate.advance(0, _wedged))
    await asyncio.wait_for(started.wait(), timeout=1)
    late = asyncio.create_task(
        gate.advance(1, lambda: dispatched.append("late-tool"))
    )
    await asyncio.sleep(0)
    late.cancel()
    await asyncio.gather(late, return_exceptions=True)
    assert dispatched == []
    release.set()
    await asyncio.wait_for(first, timeout=1)
