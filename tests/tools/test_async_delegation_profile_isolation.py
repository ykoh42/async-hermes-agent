"""Profile and event-loop isolation for durable async delegations."""

from __future__ import annotations

import asyncio
import gc
import os
import weakref
from contextlib import asynccontextmanager
from pathlib import Path

import aiofiles.os
import pytest
import pytest_asyncio

from hermes_constants import (
    reset_hermes_home_override,
    set_hermes_home_override,
)
from tools import async_delegation as ad
from tools.process_registry import process_registry


@asynccontextmanager
async def _profile(home: Path):
    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


@pytest_asyncio.fixture(autouse=True)
async def _clean_delegations():
    await ad._reset_for_tests()
    yield
    await ad._reset_for_tests()


async def _profile_queue(home: Path) -> asyncio.Queue[dict]:
    async with _profile(home):
        await process_registry._activate_profile_state()
        queue: asyncio.Queue[dict] = asyncio.Queue()
        process_registry.completion_queue = queue
        return queue


async def _dispatch_blocked(
    home: Path,
    *,
    started: asyncio.Event,
    release: asyncio.Event,
    session_key: str = "same",
    progress_fn=None,
) -> dict:
    async with _profile(home):

        async def runner():
            started.set()
            await release.wait()
            return {"status": "completed", "summary": str(home)}

        return await ad.dispatch_async_delegation(
            goal=str(home),
            context=None,
            toolsets=None,
            role="leaf",
            model="m",
            session_key=session_key,
            runner=runner,
            progress_fn=progress_fn,
        )


@pytest.mark.asyncio
async def test_same_session_and_delegation_id_are_profile_isolated(
    monkeypatch, tmp_path
):
    profile_a = tmp_path / "a"
    profile_b = tmp_path / "b"
    queue_a, queue_b = await asyncio.gather(
        _profile_queue(profile_a), _profile_queue(profile_b)
    )
    started_a = asyncio.Event()
    started_b = asyncio.Event()
    release_a = asyncio.Event()
    release_b = asyncio.Event()
    monkeypatch.setattr(ad, "_new_delegation_id", lambda: "deleg_same")

    result_a, result_b = await asyncio.gather(
        _dispatch_blocked(
            profile_a, started=started_a, release=release_a
        ),
        _dispatch_blocked(
            profile_b, started=started_b, release=release_b
        ),
    )
    assert result_a == result_b == {
        "status": "dispatched",
        "delegation_id": "deleg_same",
    }
    await asyncio.gather(started_a.wait(), started_b.wait())

    async with _profile(profile_a):
        assert await ad.interrupt_for_session(session_key="same") == 1
        assert ad.active_count() == 0
        durable_a = await ad.get_durable_delegation("deleg_same")
    assert durable_a is not None
    assert durable_a["state"] == "interrupted"
    event_a = await asyncio.wait_for(queue_a.get(), timeout=2)
    assert event_a["status"] == "interrupted"
    assert queue_b.empty()

    async with _profile(profile_b):
        assert ad.active_count() == 1
        assert ad.has_live_for_session(session_key="same")
    release_b.set()
    event_b = await asyncio.wait_for(queue_b.get(), timeout=2)
    assert event_b["status"] == "completed"
    assert event_b["summary"] == str(profile_b)
    assert queue_a.empty()


@pytest.mark.asyncio
async def test_canonical_symlink_profiles_share_capacity_and_state(tmp_path):
    profile = tmp_path / "profile"
    alias = tmp_path / "alias"
    await aiofiles.os.makedirs(profile)
    symlink = aiofiles.os.wrap(os.symlink)
    await symlink(profile, alias, target_is_directory=True)
    queue = await _profile_queue(profile)
    async with _profile(alias):
        await process_registry._activate_profile_state()
        assert process_registry.completion_queue is queue
    started = asyncio.Event()
    release = asyncio.Event()

    first = await _dispatch_blocked(
        profile, started=started, release=release
    )
    await started.wait()
    async with _profile(alias):

        async def runner():
            return {"status": "completed", "summary": "unexpected"}

        second = await ad.dispatch_async_delegation(
            goal="second",
            context=None,
            toolsets=None,
            role="leaf",
            model="m",
            session_key="same",
            runner=runner,
            max_async_children=1,
        )
        assert ad.active_count() == 1
    assert first["status"] == "dispatched"
    assert second["status"] == "rejected"
    release.set()
    assert (await asyncio.wait_for(queue.get(), timeout=2))["status"] == "completed"


@pytest.mark.asyncio
async def test_canonical_alias_restores_one_durable_event_once(tmp_path):
    profile = tmp_path / "profile"
    alias = tmp_path / "alias"
    await aiofiles.os.makedirs(profile)
    symlink = aiofiles.os.wrap(os.symlink)
    await symlink(profile, alias, target_is_directory=True)
    queue: asyncio.Queue[dict] = asyncio.Queue()
    async with _profile(profile):
        record = {
            "delegation_id": "deleg_restore",
            "goal": "restore",
            "session_key": "same",
            "status": "running",
            "dispatched_at": 1.0,
        }
        event = {
            "type": "async_delegation",
            "delegation_id": "deleg_restore",
            "session_key": "same",
            "goal": "restore",
            "status": "completed",
            "summary": "once",
            "dispatched_at": 1.0,
            "completed_at": 2.0,
        }
        await ad._persist_dispatch(record)
        await ad._persist_completion(event, {"summary": "once"})
        assert await ad.restore_undelivered_completions(queue) == 1
    async with _profile(alias):
        assert await ad.restore_undelivered_completions(queue) == 0
    assert queue.qsize() == 1


@pytest.mark.asyncio
async def test_stale_monitor_only_finalizes_its_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(ad, "_STALE_CHECK_INTERVAL", 0.01)
    monkeypatch.setattr(ad, "_STALE_IDLE_SECONDS", 0.03)
    monkeypatch.setattr(ad, "_STALL_GRACE_SECONDS", 0.03)
    profile_a = tmp_path / "a"
    profile_b = tmp_path / "b"
    queue_a, queue_b = await asyncio.gather(
        _profile_queue(profile_a), _profile_queue(profile_b)
    )
    started_a = asyncio.Event()
    started_b = asyncio.Event()
    release_a = asyncio.Event()
    release_b = asyncio.Event()

    result_a, result_b = await asyncio.gather(
        _dispatch_blocked(
            profile_a,
            started=started_a,
            release=release_a,
            progress_fn=lambda: ((0, None), False),
        ),
        _dispatch_blocked(
            profile_b, started=started_b, release=release_b
        ),
    )
    await asyncio.gather(started_a.wait(), started_b.wait())
    event_a = await asyncio.wait_for(queue_a.get(), timeout=2)
    assert event_a["delegation_id"] == result_a["delegation_id"]
    assert event_a["status"] == "stalled"
    assert queue_b.empty()
    async with _profile(profile_a):
        assert (await ad._activate_scope_state()).monitor_task is None
    async with _profile(profile_b):
        assert ad.active_count() == 1
    release_b.set()
    event_b = await asyncio.wait_for(queue_b.get(), timeout=2)
    assert event_b["delegation_id"] == result_b["delegation_id"]
    assert event_b["status"] == "completed"


@pytest.mark.asyncio
async def test_runner_context_mutation_cannot_redirect_completion(tmp_path):
    profile_a = tmp_path / "a"
    profile_b = tmp_path / "b"
    queue_a, queue_b = await asyncio.gather(
        _profile_queue(profile_a), _profile_queue(profile_b)
    )
    async with _profile(profile_a):

        async def runner():
            set_hermes_home_override(profile_b)
            return {"status": "completed", "summary": "A owned"}

        result = await ad.dispatch_async_delegation(
            goal="owner boundary",
            context=None,
            toolsets=None,
            role="leaf",
            model="m",
            session_key="same",
            runner=runner,
        )
    event = await asyncio.wait_for(queue_a.get(), timeout=2)
    assert event["summary"] == "A owned"
    assert queue_b.empty()
    async with _profile(profile_a):
        durable_a = await ad.get_durable_delegation(result["delegation_id"])
    async with _profile(profile_b):
        durable_b = await ad.get_durable_delegation(result["delegation_id"])
    assert durable_a is not None
    assert durable_a["state"] == "completed"
    assert durable_b is None


@pytest.mark.asyncio
async def test_durable_restore_and_claim_are_profile_isolated(tmp_path):
    async def seed(home: Path, summary: str):
        async with _profile(home):
            record = {
                "delegation_id": "deleg_same",
                "goal": summary,
                "session_key": "same",
                "status": "running",
                "dispatched_at": 1.0,
            }
            event = {
                "type": "async_delegation",
                "delegation_id": "deleg_same",
                "session_key": "same",
                "goal": summary,
                "status": "completed",
                "summary": summary,
                "dispatched_at": 1.0,
                "completed_at": 2.0,
            }
            await ad._persist_dispatch(record)
            await ad._persist_completion(event, {"summary": summary})
            queue: asyncio.Queue[dict] = asyncio.Queue()
            assert await ad.restore_undelivered_completions(queue) == 1
            return queue.get_nowait()

    profile_a = tmp_path / "a"
    profile_b = tmp_path / "b"
    event_a, event_b = await asyncio.gather(
        seed(profile_a, "A only"), seed(profile_b, "B only")
    )
    assert event_a["summary"] == "A only"
    assert event_b["summary"] == "B only"

    async with _profile(profile_a):
        claim = await ad.claim_event_delivery(event_a, "consumer")
        assert claim
        await ad.complete_event_delivery(event_a, claim)
        assert (await ad.get_durable_delegation("deleg_same"))["delivery_state"] == (
            "delivered"
        )
    async with _profile(profile_b):
        durable_b = await ad.get_durable_delegation("deleg_same")
        assert durable_b is not None
        assert durable_b["delivery_state"] == "pending"


@pytest.mark.asyncio
async def test_repeated_cancellation_finishes_profile_owned_finalization(
    monkeypatch, tmp_path
):
    profile = tmp_path / "profile"
    queue = await _profile_queue(profile)
    runner_started = asyncio.Event()
    persist_started = asyncio.Event()
    allow_persist = asyncio.Event()
    original_persist = ad._persist_completion

    async def delayed_persist(event, result):
        persist_started.set()
        await allow_persist.wait()
        await original_persist(event, result)

    monkeypatch.setattr(ad, "_persist_completion", delayed_persist)
    async with _profile(profile):

        async def runner():
            runner_started.set()
            await asyncio.Future()

        result = await ad.dispatch_async_delegation(
            goal="cancel",
            context=None,
            toolsets=None,
            role="leaf",
            model="m",
            session_key="same",
            runner=runner,
        )
        state = await ad._activate_scope_state()
        task = state.tasks[result["delegation_id"]]
    await runner_started.wait()
    task.cancel()
    await asyncio.wait_for(persist_started.wait(), timeout=1)
    task.cancel()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    allow_persist.set()
    await asyncio.gather(task, return_exceptions=True)
    event = await asyncio.wait_for(queue.get(), timeout=2)
    assert event["status"] == "interrupted"
    async with _profile(profile):
        durable = await ad.get_durable_delegation(result["delegation_id"])
        assert durable is not None
        assert durable["state"] == "interrupted"
        await asyncio.sleep(0)
        assert not state.tasks


def test_completed_scope_does_not_retain_closed_event_loops(tmp_path):
    loop_refs: list[weakref.ReferenceType[asyncio.AbstractEventLoop]] = []

    async def cycle(home: Path) -> None:
        token = set_hermes_home_override(home)
        try:
            loop_refs.append(weakref.ref(asyncio.get_running_loop()))
            await asyncio.gather(
                ad.get_durable_delegation("missing-a"),
                ad.get_durable_delegation("missing-b"),
            )
            queue: asyncio.Queue[dict] = asyncio.Queue()
            assert sorted(
                await asyncio.gather(
                    ad.restore_undelivered_completions(queue),
                    ad.restore_undelivered_completions(queue),
                )
            ) == [0, 0]
            state = await ad._activate_scope_state()
            assert state.db_lock is None
            assert state.restore_lock is None
        finally:
            reset_hermes_home_override(token)

    asyncio.run(cycle(tmp_path / "first"))
    asyncio.run(cycle(tmp_path / "second"))
    gc.collect()
    assert [reference() for reference in loop_refs] == [None, None]
