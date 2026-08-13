"""Restored completions require positive session ownership proof."""

from __future__ import annotations

import asyncio
import json

import pytest

from tools import async_delegation as ad
from tools.process_registry import ProcessRegistry

pytestmark = pytest.mark.asyncio


def _make_registry():
    registry = ProcessRegistry.__new__(ProcessRegistry)
    registry._running = {}
    registry._finished = {}
    registry.completion_queue = asyncio.Queue()
    registry._completion_consumed = set()
    registry._poll_observed = set()
    return registry


def _event(session_key="", restored=False):
    event = {
        "type": "async_delegation",
        "delegation_id": "d-old",
        "session_key": session_key,
        "origin_ui_session_id": "",
        "goal": "secret goal",
        "status": "success",
        "summary": "SECRET RESULT",
        "api_calls": 3,
        "duration_seconds": 1.5,
        "dispatched_at": 1.0,
        "completed_at": 2.0,
    }
    if restored:
        event["restored"] = True
    return event


async def test_restore_stamps_flag_without_mutating_durable_payload(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    await ad._reset_for_tests()
    record = {
        "delegation_id": "d-old",
        "goal": "secret goal",
        "context": None,
        "toolsets": None,
        "role": "leaf",
        "model": "m",
        "session_key": "OWNER",
        "origin_ui_session_id": "",
        "parent_session_id": "OWNER",
        "status": "running",
        "dispatched_at": 1.0,
        "completed_at": None,
    }
    await ad._persist_dispatch(record)
    await ad._persist_completion(_event("OWNER"), {"summary": "SECRET RESULT"})
    queue = asyncio.Queue()
    assert await ad.restore_undelivered_completions(queue) == 1
    assert queue.get_nowait()["restored"] is True
    async with ad._transaction() as connection:
        cursor = await connection.execute(
            "SELECT event_json FROM async_delegations WHERE delegation_id='d-old'"
        )
        durable_event = json.loads((await cursor.fetchone())[0])
    assert "restored" not in durable_event


async def test_unfiltered_drain_requeues_restored_event():
    registry = _make_registry()
    registry.completion_queue.put_nowait(_event("OWNER", restored=True))
    assert registry.drain_notifications() == []
    assert registry.completion_queue.qsize() == 1


async def test_positive_owner_consumes_restored_event():
    registry = _make_registry()
    registry.completion_queue.put_nowait(_event("OWNER", restored=True))
    results = registry.drain_notifications(
        owns_event=lambda event: event.get("session_key") == "OWNER"
    )
    assert len(results) == 1
    assert registry.completion_queue.empty()


async def test_ownerless_same_process_event_keeps_legacy_delivery():
    registry = _make_registry()
    registry.completion_queue.put_nowait(_event())
    assert len(registry.drain_notifications()) == 1
