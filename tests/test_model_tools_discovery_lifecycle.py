"""Event-loop-neutral ownership of the lazy built-in discovery boundary."""

from __future__ import annotations

import asyncio

import model_tools


def test_discovery_can_run_across_sequential_event_loops(monkeypatch):
    calls = 0

    async def discover():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return []

    monkeypatch.setattr(model_tools, "discover_builtin_tools", discover)
    monkeypatch.setattr(model_tools, "_builtin_tools_discovered", False)
    monkeypatch.setattr(
        model_tools,
        "_public_maps_generation",
        model_tools.registry._generation,
    )

    asyncio.run(model_tools._ensure_builtin_tools_discovered())
    asyncio.run(model_tools._ensure_builtin_tools_discovered())

    assert calls == 1
