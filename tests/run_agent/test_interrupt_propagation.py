"""Behavior tests for task-local agent interrupt propagation."""

from __future__ import annotations

import asyncio

import pytest

from tools.interrupt import (
    _bind_interrupt_event,
    is_interrupted,
    _reset_interrupt_event,
)


def _make_bare_agent():
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._interrupt_event = asyncio.Event()
    agent._pending_redirect = None
    agent._pending_steer = None
    agent._active_children = []
    agent.quiet_mode = True
    agent.provider = "openrouter"
    agent.model = "test/model"
    agent._base_url = "http://localhost:1234"
    return agent


def test_parent_interrupt_sets_child_event_and_flag():
    parent = _make_bare_agent()
    child = _make_bare_agent()
    parent._active_children.append(child)

    parent.interrupt("new user message")

    assert parent._interrupt_requested is True
    assert parent._interrupt_event.is_set()
    assert child._interrupt_requested is True
    assert child._interrupt_event.is_set()
    assert child._interrupt_message == "new user message"


def test_prestart_interrupt_is_preserved_when_turn_context_binds():
    agent = _make_bare_agent()
    agent.interrupt("stop before start")

    token = _bind_interrupt_event(agent._interrupt_event)
    try:
        assert is_interrupted() is True
    finally:
        _reset_interrupt_event(token)


@pytest.mark.asyncio
async def test_concurrent_agents_have_isolated_interrupt_contexts():
    first = _make_bare_agent()
    second = _make_bare_agent()
    ready = [asyncio.Event(), asyncio.Event()]
    inspect = asyncio.Event()

    async def observe(agent, slot):
        token = _bind_interrupt_event(agent._interrupt_event)
        try:
            ready[slot].set()
            await inspect.wait()
            return is_interrupted()
        finally:
            _reset_interrupt_event(token)

    tasks = [
        asyncio.create_task(observe(first, 0)),
        asyncio.create_task(observe(second, 1)),
    ]
    await asyncio.gather(*(event.wait() for event in ready))
    first.interrupt("stop only first")
    inspect.set()

    assert await asyncio.gather(*tasks) == [True, False]


@pytest.mark.asyncio
async def test_clearing_one_agent_does_not_clear_another():
    first = _make_bare_agent()
    second = _make_bare_agent()
    first.interrupt()
    second.interrupt()

    first.clear_interrupt()

    assert first._interrupt_event.is_set() is False
    assert second._interrupt_event.is_set() is True
