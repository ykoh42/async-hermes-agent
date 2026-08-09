"""Native-async durable session activity parity tests."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import run_agent
from agent.session_activity import ActivityProvenance


def _agent_with_db(session_id: str = "sess-1"):
    agent = SimpleNamespace(
        session_id=session_id,
        _session_db=SimpleNamespace(
            touch_session_activity=AsyncMock(),
            clear_session_activity_labels=AsyncMock(),
        ),
        _last_activity_ts=0.0,
        _last_activity_desc="",
        _last_activity_provenance=ActivityProvenance.UNKNOWN,
        _session_activity_last_persist_mono=0.0,
        _current_tool=None,
        _api_call_count=0,
        max_iterations=10,
        iteration_budget=SimpleNamespace(used=0, max_total=10),
    )
    for name in (
        "_touch_activity",
        "_persist_session_activity_if_due",
        "_drain_session_activity_persist",
        "_reset_activity_labels_after_turn",
        "get_activity_summary",
    ):
        setattr(
            agent,
            name,
            getattr(run_agent.AIAgent, name).__get__(agent, SimpleNamespace),
        )
    return agent


@pytest.mark.asyncio
async def test_touch_activity_persists_once_per_upstream_cadence(monkeypatch):
    agent = _agent_with_db()
    monotonic = {"value": 1000.0}
    monkeypatch.setattr(run_agent.time, "time", lambda: 1_700_000_000.0)
    monkeypatch.setattr(
        run_agent.time,
        "monotonic",
        lambda: monotonic["value"],
    )

    agent._touch_activity("starting API call #1")
    await agent._drain_session_activity_persist()
    agent._session_db.touch_session_activity.assert_awaited_once_with(
        "sess-1",
        1_700_000_000.0,
        description="starting API call #1",
        provenance=ActivityProvenance.UNKNOWN,
    )

    agent._session_db.touch_session_activity.reset_mock()
    monotonic["value"] = 1030.0
    agent._touch_activity("receiving stream response")
    await agent._drain_session_activity_persist()
    agent._session_db.touch_session_activity.assert_not_awaited()

    monotonic["value"] = 1061.0
    agent._touch_activity("API call #1 completed")
    await agent._drain_session_activity_persist()
    agent._session_db.touch_session_activity.assert_awaited_once_with(
        "sess-1",
        1_700_000_000.0,
        description="API call #1 completed",
        provenance=ActivityProvenance.UNKNOWN,
    )


@pytest.mark.asyncio
async def test_force_persist_bypasses_cadence_and_preserves_provenance(monkeypatch):
    agent = _agent_with_db()
    monkeypatch.setattr(run_agent.time, "time", lambda: 1_700_000_000.0)
    monkeypatch.setattr(run_agent.time, "monotonic", lambda: 1000.0)

    agent._session_activity_last_persist_mono = 999.0
    agent._touch_activity(
        "context compression completed",
        provenance=ActivityProvenance.AGENT_COMPRESSION,
        force_persist=True,
    )
    await agent._drain_session_activity_persist()

    assert agent._last_activity_provenance is ActivityProvenance.AGENT_COMPRESSION
    agent._session_db.touch_session_activity.assert_awaited_once_with(
        "sess-1",
        1_700_000_000.0,
        description="context compression completed",
        provenance=ActivityProvenance.AGENT_COMPRESSION,
    )


@pytest.mark.asyncio
async def test_activity_persist_failure_is_fail_open(monkeypatch):
    agent = _agent_with_db()
    agent._session_db.touch_session_activity.side_effect = OSError("disk gone")
    monkeypatch.setattr(run_agent.time, "time", lambda: 1.0)
    monkeypatch.setattr(run_agent.time, "monotonic", lambda: 1000.0)

    agent._touch_activity("tool completed: terminal")
    await agent._drain_session_activity_persist()

    assert agent._last_activity_desc == "tool completed: terminal"


def test_sync_activity_backend_is_not_called(monkeypatch):
    agent = _agent_with_db()
    sync_touch = MagicMock()
    agent._session_db.touch_session_activity = sync_touch
    monkeypatch.setattr(run_agent.time, "time", lambda: 1.0)
    monkeypatch.setattr(run_agent.time, "monotonic", lambda: 1000.0)

    agent._touch_activity("starting API call #1")

    sync_touch.assert_not_called()
    assert not hasattr(agent, "_session_activity_persist_task")


@pytest.mark.asyncio
async def test_activity_persist_finishes_through_repeated_cancellation(monkeypatch):
    agent = _agent_with_db()
    started = asyncio.Event()
    release = asyncio.Event()
    completed = False

    async def touch(*args, **kwargs):
        nonlocal completed
        started.set()
        await release.wait()
        completed = True

    agent._session_db.touch_session_activity.side_effect = touch
    monkeypatch.setattr(run_agent.time, "time", lambda: 1.0)
    monkeypatch.setattr(run_agent.time, "monotonic", lambda: 1000.0)

    agent._touch_activity("starting API call #1")
    drain_task = asyncio.create_task(agent._drain_session_activity_persist())
    await started.wait()
    drain_task.cancel()
    await asyncio.sleep(0)
    drain_task.cancel()

    await asyncio.sleep(0)
    assert drain_task.done() is False

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await drain_task
    assert completed is True


@pytest.mark.asyncio
async def test_turn_reset_orders_pending_touch_before_durable_clear(monkeypatch):
    agent = _agent_with_db()
    order: list[str] = []

    async def touch(*args, **kwargs):
        order.append("touch")

    async def clear(*args, **kwargs):
        order.append("clear")

    agent._session_db.touch_session_activity.side_effect = touch
    agent._session_db.clear_session_activity_labels.side_effect = clear
    monkeypatch.setattr(run_agent.time, "time", lambda: 1.0)
    monkeypatch.setattr(run_agent.time, "monotonic", lambda: 1000.0)

    agent._touch_activity(
        "compressing context",
        provenance=ActivityProvenance.AGENT_COMPRESSION,
    )
    timestamp = agent._last_activity_ts
    await agent._reset_activity_labels_after_turn()

    assert order == ["touch", "clear"]
    assert agent._last_activity_ts == timestamp
    assert agent._last_activity_desc == ""
    assert agent._last_activity_provenance is ActivityProvenance.UNKNOWN


def test_get_activity_summary_exposes_upstream_contract(monkeypatch):
    agent = _agent_with_db()
    monkeypatch.setattr(run_agent.time, "time", lambda: 1_700_000_010.0)
    agent._last_activity_ts = 1_700_000_000.0
    agent._last_activity_desc = "executing tool: terminal"

    summary = agent.get_activity_summary()

    assert summary["last_activity_at"] == 1_700_000_000.0
    assert summary["last_activity_description"] == "executing tool: terminal"
    assert summary["last_activity_provenance"] == "unknown"
    assert summary["seconds_since_activity"] == 10.0
    assert summary["last_activity_ts"] == 1_700_000_000.0
    assert summary["last_activity_desc"] == "executing tool: terminal"


@pytest.mark.asyncio
async def test_context_overflow_warning_stamps_cooldown(monkeypatch):
    agent = _agent_with_db()
    agent._last_ctx_overflow_warn = None
    agent._emit_warning = MagicMock()
    monkeypatch.setattr(run_agent.time, "time", lambda: 1_700_000_100.0)
    monkeypatch.setattr(run_agent.time, "monotonic", lambda: 2000.0)

    run_agent.AIAgent._warn_context_overflow_blocked(
        agent,
        "cooldown: 30s remaining",
        80_000,
        40_000,
    )
    await agent._drain_session_activity_persist()

    assert (
        agent._last_activity_provenance
        is ActivityProvenance.AGENT_COMPRESSION_COOLDOWN
    )
    assert "compression blocked" in agent._last_activity_desc
    agent._emit_warning.assert_called_once()
