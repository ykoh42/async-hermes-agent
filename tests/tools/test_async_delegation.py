"""Native-async parity tests for durable background delegation."""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock

import aiosqlite
import pytest
import pytest_asyncio

from tools import async_delegation as ad
from tools.process_registry import format_process_notification, process_registry

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture(autouse=True)
async def _clean_state(monkeypatch, tmp_path):
    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    await ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    await ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


async def _drain_for(delegation_id: str, timeout: float = 3.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        try:
            event = process_registry.completion_queue.get_nowait()
        except asyncio.QueueEmpty:
            await asyncio.sleep(0.01)
            continue
        if event.get("delegation_id") == delegation_id:
            return event
    return None


async def test_dispatch_returns_before_native_async_runner_finishes():
    started = asyncio.Event()
    release = asyncio.Event()

    async def runner():
        started.set()
        await release.wait()
        return {"status": "completed", "summary": "done"}

    result = await ad.dispatch_async_delegation(
        goal="g",
        context=None,
        toolsets=None,
        role="leaf",
        model="m",
        session_key="owner",
        runner=runner,
    )
    assert result["status"] == "dispatched"
    await asyncio.wait_for(started.wait(), timeout=1)
    assert ad.active_count() == 1
    assert process_registry.completion_queue.empty()
    release.set()
    event = await _drain_for(result["delegation_id"])
    assert event["summary"] == "done"


async def test_sync_runner_fails_closed_without_thread_fallback():
    result = await ad.dispatch_async_delegation(
        goal="g",
        context=None,
        toolsets=None,
        role="leaf",
        model="m",
        session_key="owner",
        runner=lambda: {"status": "completed"},
    )
    assert result == {
        "status": "rejected",
        "error": "Async delegation runner must be a native async callable",
    }
    assert ad.active_count() == 0


async def test_completion_event_and_durable_result_preserve_upstream_shape():
    async def runner():
        return {
            "status": "completed",
            "summary": "the result",
            "api_calls": 3,
            "duration_seconds": 2.0,
            "model": "test-model",
        }

    result = await ad.dispatch_async_delegation(
        goal="compute X",
        context="some context",
        toolsets=["web", "file"],
        role="leaf",
        model="test-model",
        session_key="agent:main:cli:dm:local",
        parent_session_id="parent-session",
        runner=runner,
    )
    event = await _drain_for(result["delegation_id"])
    assert event["type"] == "async_delegation"
    assert event["summary"] == "the result"
    assert event["session_key"] == "agent:main:cli:dm:local"
    assert event["parent_session_id"] == "parent-session"
    durable = await ad.get_durable_delegation(result["delegation_id"])
    assert durable["state"] == "completed"
    assert durable["delivery_state"] == "pending"
    assert durable["result"]["summary"] == "the result"


async def test_rich_reinjection_block_is_self_contained():
    async def runner():
        return {
            "status": "completed",
            "summary": "The answer is 42.",
            "api_calls": 7,
            "duration_seconds": 3.5,
            "model": "test-model",
        }

    result = await ad.dispatch_async_delegation(
        goal="Compute the meaning of life",
        context="User is a philosopher. Respond tersely.",
        toolsets=["web"],
        role="leaf",
        model="test-model",
        session_key="owner",
        runner=runner,
    )
    event = await _drain_for(result["delegation_id"])
    text = format_process_notification(event)
    for needle in (
        "ASYNC DELEGATION COMPLETE",
        "Compute the meaning of life",
        "User is a philosopher",
        "Toolsets: web",
        "The answer is 42.",
        "Status: completed",
        "API calls: 7",
    ):
        assert needle in text


async def test_capacity_check_is_atomic_on_one_event_loop():
    release = asyncio.Event()

    async def blocker():
        await release.wait()
        return {"status": "completed", "summary": "x"}

    first, second = await asyncio.gather(
        ad.dispatch_async_delegation(
            goal="one",
            context=None,
            toolsets=None,
            role="leaf",
            model="m",
            session_key="owner",
            runner=blocker,
            max_async_children=1,
        ),
        ad.dispatch_async_delegation(
            goal="two",
            context=None,
            toolsets=None,
            role="leaf",
            model="m",
            session_key="owner",
            runner=blocker,
            max_async_children=1,
        ),
    )
    assert sorted((first["status"], second["status"])) == [
        "dispatched",
        "rejected",
    ]
    release.set()


async def test_batch_completion_keeps_input_result_order_and_single_event():
    async def runner():
        return {
            "results": [
                {"task_index": 0, "status": "completed", "summary": "first"},
                {"task_index": 1, "status": "completed", "summary": "second"},
            ],
            "total_duration_seconds": 0.2,
        }

    result = await ad.dispatch_async_delegation_batch(
        goals=["one", "two"],
        context="ctx",
        toolsets=None,
        role="leaf",
        model="m",
        session_key="owner",
        runner=runner,
    )
    event = await _drain_for(result["delegation_id"])
    assert event["is_batch"] is True
    assert event["goals"] == ["one", "two"]
    assert [entry["summary"] for entry in event["results"]] == [
        "first",
        "second",
    ]
    assert process_registry.completion_queue.empty()


async def test_interrupt_for_session_cancels_and_persists_terminal_event():
    started = asyncio.Event()
    finalized = asyncio.Event()

    async def runner():
        started.set()
        try:
            await asyncio.Future()
        finally:
            finalized.set()

    result = await ad.dispatch_async_delegation(
        goal="long task",
        context=None,
        toolsets=None,
        role="leaf",
        model="m",
        session_key="owner",
        parent_session_id="parent",
        runner=runner,
    )
    await started.wait()
    count = await ad.interrupt_for_session(parent_session_id="parent")
    assert count == 1
    await finalized.wait()
    event = await _drain_for(result["delegation_id"])
    assert event["status"] == "interrupted"
    durable = await ad.get_durable_delegation(result["delegation_id"])
    assert durable["state"] == "interrupted"
    assert ad.active_count() == 0


async def test_runner_exception_is_a_durable_error_completion():
    async def runner():
        raise RuntimeError("child crashed")

    result = await ad.dispatch_async_delegation(
        goal="crash",
        context=None,
        toolsets=None,
        role="leaf",
        model="m",
        session_key="owner",
        runner=runner,
    )
    event = await _drain_for(result["delegation_id"])
    assert event["status"] == "error"
    assert event["error"] == "RuntimeError: child crashed"


async def test_stale_runner_interrupts_then_force_finalizes(monkeypatch):
    monkeypatch.setattr(ad, "_STALE_CHECK_INTERVAL", 0.01)
    monkeypatch.setattr(ad, "_STALE_IDLE_SECONDS", 0.03)
    monkeypatch.setattr(ad, "_STALL_GRACE_SECONDS", 0.03)
    interrupted = 0

    async def runner():
        await asyncio.Future()

    def interrupt():
        nonlocal interrupted
        interrupted += 1

    result = await ad.dispatch_async_delegation(
        goal="stuck",
        context=None,
        toolsets=None,
        role="leaf",
        model="m",
        session_key="owner",
        runner=runner,
        interrupt_fn=interrupt,
        progress_fn=lambda: ((0, None), False),
    )
    event = await _drain_for(result["delegation_id"])
    assert event["status"] == "stalled"
    assert event["stall_phase"] == "idle"
    assert interrupted >= 1
    await asyncio.sleep(0)
    assert process_registry.completion_queue.empty()


async def test_live_listing_exposes_activity_without_callables():
    release = asyncio.Event()
    activity_at = time.time() - 12

    async def runner():
        await release.wait()
        return {"status": "completed", "summary": "done"}

    result = await ad.dispatch_async_delegation(
        goal="live listing",
        context=None,
        toolsets=None,
        role="leaf",
        model="m",
        session_key="owner",
        runner=runner,
        progress_fn=lambda: (((3, "web_search", activity_at),), True),
    )
    item = next(
        entry
        for entry in ad.list_async_delegations()
        if entry["delegation_id"] == result["delegation_id"]
    )
    assert item["in_tool"] is True
    assert item["children_activity"][0]["api_calls"] == 3
    assert "progress_fn" not in item
    assert "interrupt_fn" not in item
    release.set()
    await _drain_for(result["delegation_id"])
    assert ad._monitor_task is None or ad._monitor_task.done()


async def test_claim_release_complete_and_drop_contract():
    async def runner():
        return {"status": "completed", "summary": "done"}

    result = await ad.dispatch_async_delegation(
        goal="delivery",
        context=None,
        toolsets=None,
        role="leaf",
        model="m",
        session_key="owner",
        runner=runner,
    )
    event = await _drain_for(result["delegation_id"])
    claim = await ad.claim_event_delivery(event, "consumer-a")
    assert claim
    assert await ad.claim_event_delivery(event, "consumer-b") is None
    await ad.release_event_delivery(event, claim)
    second_claim = await ad.claim_event_delivery(event, "consumer-b")
    assert second_claim
    await ad.complete_event_delivery(event, second_claim)
    durable = await ad.get_durable_delegation(result["delegation_id"])
    assert durable["delivery_state"] == "delivered"
    assert durable["delivery_attempts"] == 2


async def test_cancelled_claim_commits_then_releases_before_propagating(
    monkeypatch,
):
    async def runner():
        await asyncio.sleep(0)
        return {"status": "completed", "summary": "done"}

    result = await ad.dispatch_async_delegation(
        goal="delivery",
        context=None,
        toolsets=None,
        role="leaf",
        model="m",
        session_key="owner",
        runner=runner,
    )
    event = await _drain_for(result["delegation_id"])
    claim_commit_started = asyncio.Event()
    allow_claim_commit = asyncio.Event()
    original_commit = aiosqlite.Connection.commit

    async def delayed_claim_commit(connection) -> None:
        claim_commit_started.set()
        await allow_claim_commit.wait()
        await original_commit(connection)

    monkeypatch.setattr(aiosqlite.Connection, "commit", delayed_claim_commit)
    claim_task = asyncio.create_task(ad.claim_event_delivery(event, "cancelled-claim"))
    try:
        await asyncio.wait_for(claim_commit_started.wait(), timeout=1)
        claim_task.cancel()
        await asyncio.sleep(0)
        assert not claim_task.done()
        claim_task.cancel()
        await asyncio.sleep(0)
        assert not claim_task.done()
        allow_claim_commit.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(claim_task, timeout=1)

        durable = await ad.get_durable_delegation(result["delegation_id"])
        assert durable is not None
        assert durable["delivery_state"] == "pending"
        assert durable["delivery_attempts"] == 1
        replacement = await ad.claim_event_delivery(event, "replacement")
        assert replacement is not None
        await ad.complete_event_delivery(event, replacement)
        durable = await ad.get_durable_delegation(result["delegation_id"])
        assert durable is not None
        assert durable["delivery_state"] == "delivered"
        assert durable["delivery_attempts"] == 2
    finally:
        allow_claim_commit.set()
        if not claim_task.done():
            claim_task.cancel()
            await asyncio.gather(claim_task, return_exceptions=True)


async def test_abandoned_running_record_becomes_unknown(monkeypatch):
    record = {
        "delegation_id": "d-old",
        "goal": "old goal",
        "context": None,
        "toolsets": None,
        "role": "leaf",
        "model": "m",
        "session_key": "owner",
        "origin_ui_session_id": "",
        "parent_session_id": "parent",
        "status": "running",
        "dispatched_at": 1.0,
        "completed_at": None,
    }
    await ad._persist_dispatch(record)
    monkeypatch.setattr("gateway.status._pid_exists", AsyncMock(return_value=False))
    assert await ad.recover_abandoned_delegations() == 1
    durable = await ad.get_durable_delegation("d-old")
    assert durable["state"] == "unknown"
    assert durable["result"]["status"] == "unknown"


async def test_restore_stamps_in_memory_only_and_unfiltered_drain_fails_closed():
    record = {
        "delegation_id": "d-old",
        "goal": "old goal",
        "context": None,
        "toolsets": None,
        "role": "leaf",
        "model": "m",
        "session_key": "owner",
        "origin_ui_session_id": "",
        "parent_session_id": "parent",
        "status": "running",
        "dispatched_at": 1.0,
        "completed_at": None,
    }
    await ad._persist_dispatch(record)
    event = {
        "type": "async_delegation",
        "delegation_id": "d-old",
        "session_key": "owner",
        "goal": "old goal",
        "status": "completed",
        "summary": "secret",
        "dispatched_at": 1.0,
        "completed_at": 2.0,
    }
    await ad._persist_completion(event, {"summary": "secret"})
    queue = asyncio.Queue()
    assert await ad.restore_undelivered_completions(queue) == 1
    restored = queue.get_nowait()
    assert restored["restored"] is True

    process_registry.completion_queue.put_nowait(restored)
    assert process_registry.drain_notifications() == []
    assert process_registry.completion_queue.qsize() == 1
    claimed = process_registry.drain_notifications(session_key="owner")
    assert claimed[0][0]["summary"] == "secret"

    async with ad._transaction() as conn:
        cursor = await conn.execute(
            "SELECT event_json FROM async_delegations WHERE delegation_id='d-old'"
        )
        payload = json.loads((await cursor.fetchone())[0])
    assert "restored" not in payload


async def test_concurrent_restore_enqueues_each_pending_event_once():
    record = {
        "delegation_id": "d-once",
        "goal": "once",
        "context": None,
        "toolsets": None,
        "role": "leaf",
        "model": "m",
        "session_key": "owner",
        "status": "running",
        "dispatched_at": 1.0,
    }
    await ad._persist_dispatch(record)
    event = {
        "type": "async_delegation",
        "delegation_id": "d-once",
        "session_key": "owner",
        "goal": "once",
        "status": "completed",
        "summary": "one result",
        "dispatched_at": 1.0,
        "completed_at": 2.0,
    }
    await ad._persist_completion(event, {"summary": "one result"})
    queue = asyncio.Queue()
    counts = await asyncio.gather(
        ad.restore_undelivered_completions(queue),
        ad.restore_undelivered_completions(queue),
    )
    assert sorted(counts) == [0, 1]
    assert queue.qsize() == 1


async def test_restore_once_guard_is_isolated_per_profile_path(monkeypatch, tmp_path):
    current_path = tmp_path / "profile-a.db"
    monkeypatch.setattr(ad, "_db_path", lambda: current_path)
    queue = asyncio.Queue()

    async def persist(delegation_id: str):
        await ad._persist_dispatch(
            {
                "delegation_id": delegation_id,
                "goal": delegation_id,
                "session_key": delegation_id,
                "status": "running",
                "dispatched_at": 1.0,
            }
        )
        await ad._persist_completion(
            {
                "type": "async_delegation",
                "delegation_id": delegation_id,
                "session_key": delegation_id,
                "goal": delegation_id,
                "status": "completed",
                "summary": delegation_id,
                "dispatched_at": 1.0,
                "completed_at": 2.0,
            },
            {"summary": delegation_id},
        )

    await persist("profile-a")
    assert await ad.restore_undelivered_completions(queue) == 1
    current_path = tmp_path / "profile-b.db"
    await persist("profile-b")
    assert await ad.restore_undelivered_completions(queue) == 1
    assert [queue.get_nowait()["summary"] for _ in range(2)] == [
        "profile-a",
        "profile-b",
    ]
