"""Deterministic cancellation coverage for queued subagent delivery."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import aiosqlite
import pytest

from tools import async_delegation as ad
from tools.process_registry import process_registry


async def _run_claimed_parent_turn(
    event: dict[str, Any],
    *,
    consumer: str,
    parent_turn: Callable[[], Awaitable[str]],
) -> str:
    claim = await ad.claim_event_delivery(event, consumer)
    assert claim is not None
    return await _run_parent_turn_with_claim(event, claim, parent_turn)


async def _run_parent_turn_with_claim(
    event: dict[str, Any],
    claim: str,
    parent_turn: Callable[[], Awaitable[str]],
) -> str:
    try:
        response = await parent_turn()
    except BaseException:
        await ad.release_event_delivery(event, claim)
        raise
    else:
        await ad.complete_event_delivery(event, claim)
        return response


@pytest.mark.asyncio
async def test_cancelled_parent_turn_releases_and_reclaims_delivery_exactly_once(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    monkeypatch.setattr(process_registry, "completion_queue", asyncio.Queue())
    await ad._reset_for_tests()
    allow_release_commit = asyncio.Event()
    delivery: asyncio.Task[str] | None = None

    async def child_runner() -> dict[str, Any]:
        await asyncio.sleep(0)
        return {"status": "completed", "summary": "child result"}

    try:
        dispatch = await ad.dispatch_async_delegation(
            goal="complete deterministically",
            context=None,
            toolsets=None,
            role="leaf",
            model="test-model",
            session_key="test-session",
            runner=child_runner,
        )
        event = await asyncio.wait_for(
            process_registry.completion_queue.get(), timeout=1
        )
        assert event["delegation_id"] == dispatch["delegation_id"]

        release_calls: list[tuple[str, str]] = []
        release_event_delivery = ad.release_event_delivery

        async def record_release(released_event: dict[str, Any], claim: str) -> None:
            release_calls.append((released_event["delegation_id"], claim))
            await release_event_delivery(released_event, claim)

        monkeypatch.setattr(ad, "release_event_delivery", record_release)
        parent_started = asyncio.Event()
        release_commit_started = asyncio.Event()
        original_commit = aiosqlite.Connection.commit

        async def delayed_release_commit(connection) -> None:
            if release_calls:
                release_commit_started.set()
                await allow_release_commit.wait()
            await original_commit(connection)

        monkeypatch.setattr(aiosqlite.Connection, "commit", delayed_release_commit)

        async def cancelled_parent_turn() -> str:
            parent_started.set()
            await asyncio.Future()
            raise AssertionError("unreachable")

        delivery = asyncio.create_task(
            _run_claimed_parent_turn(
                event,
                consumer="cancelled-parent",
                parent_turn=cancelled_parent_turn,
            )
        )
        await asyncio.wait_for(parent_started.wait(), timeout=1)
        delivery.cancel()
        await asyncio.wait_for(release_commit_started.wait(), timeout=1)
        delivery.cancel()
        await asyncio.sleep(0)
        assert not delivery.done()
        allow_release_commit.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(delivery, timeout=1)

        assert len(release_calls) == 1
        durable = await ad.get_durable_delegation(dispatch["delegation_id"])
        assert durable is not None
        assert durable["delivery_state"] == "pending"
        assert durable["delivery_attempts"] == 1

        competing_claims = await asyncio.gather(
            ad.claim_event_delivery(event, "replacement-parent"),
            ad.claim_event_delivery(event, "competing-parent"),
        )
        reclaimed = [claim for claim in competing_claims if claim is not None]
        assert len(reclaimed) == 1
        await ad.complete_event_delivery(event, reclaimed[0])

        durable = await ad.get_durable_delegation(dispatch["delegation_id"])
        assert durable is not None
        assert durable["delivery_state"] == "delivered"
        assert durable["delivery_attempts"] == 2
        assert await ad.claim_event_delivery(event, "late-parent") is None
    finally:
        allow_release_commit.set()
        if delivery is not None and not delivery.done():
            delivery.cancel()
            await asyncio.gather(delivery, return_exceptions=True)
        await ad._reset_for_tests()


@pytest.mark.asyncio
async def test_cancelled_completion_waits_for_commit_without_releasing_claim(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    monkeypatch.setattr(process_registry, "completion_queue", asyncio.Queue())
    await ad._reset_for_tests()
    allow_completion_commit = asyncio.Event()
    delivery: asyncio.Task[str] | None = None

    async def child_runner() -> dict[str, Any]:
        await asyncio.sleep(0)
        return {"status": "completed", "summary": "child result"}

    try:
        dispatch = await ad.dispatch_async_delegation(
            goal="complete once",
            context=None,
            toolsets=None,
            role="leaf",
            model="test-model",
            session_key="test-session",
            runner=child_runner,
        )
        event = await asyncio.wait_for(
            process_registry.completion_queue.get(), timeout=1
        )
        claim = await ad.claim_event_delivery(event, "cancelled-completion")
        assert claim is not None

        completion_commit_started = asyncio.Event()
        original_commit = aiosqlite.Connection.commit

        async def delayed_completion_commit(connection) -> None:
            completion_commit_started.set()
            await allow_completion_commit.wait()
            await original_commit(connection)

        monkeypatch.setattr(
            aiosqlite.Connection,
            "commit",
            delayed_completion_commit,
        )
        delivery = asyncio.create_task(
            _run_parent_turn_with_claim(
                event,
                claim,
                parent_turn=lambda: asyncio.sleep(0, result="accepted"),
            )
        )
        await asyncio.wait_for(completion_commit_started.wait(), timeout=1)
        delivery.cancel()
        await asyncio.sleep(0)
        assert not delivery.done()
        delivery.cancel()
        await asyncio.sleep(0)
        assert not delivery.done()
        allow_completion_commit.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(delivery, timeout=1)

        durable = await ad.get_durable_delegation(dispatch["delegation_id"])
        assert durable is not None
        assert durable["delivery_state"] == "delivered"
        assert durable["delivery_attempts"] == 1
        assert await ad.claim_event_delivery(event, "late-parent") is None
    finally:
        allow_completion_commit.set()
        if delivery is not None and not delivery.done():
            delivery.cancel()
            await asyncio.gather(delivery, return_exceptions=True)
        await ad._reset_for_tests()
