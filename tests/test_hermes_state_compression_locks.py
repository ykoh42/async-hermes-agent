"""Native-async compression lease behavior."""

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

import hermes_state
from hermes_state import SessionDB


@pytest_asyncio.fixture
async def db(tmp_path: Path):
    database = SessionDB(tmp_path / "state.db")
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_single_holder_and_release(db):
    assert await db.try_acquire_compression_lock("s1", "holder-1") is True
    assert await db.try_acquire_compression_lock("s1", "holder-2") is False
    assert await db.get_compression_lock_holder("s1") == "holder-1"
    await db.release_compression_lock("s1", "holder-1")
    assert await db.try_acquire_compression_lock("s1", "holder-2") is True


@pytest.mark.asyncio
async def test_leases_are_isolated_per_session(db):
    assert await db.try_acquire_compression_lock("s1", "holder-1") is True
    assert await db.try_acquire_compression_lock("s2", "holder-2") is True
    assert await db.get_compression_lock_holder("s1") == "holder-1"
    assert await db.get_compression_lock_holder("s2") == "holder-2"


@pytest.mark.asyncio
async def test_expired_lease_is_reclaimable(db):
    assert await db.try_acquire_compression_lock("s1", "stale", ttl_seconds=0.01)
    await asyncio.sleep(0.02)
    assert await db.get_compression_lock_holder("s1") is None
    assert await db.try_acquire_compression_lock("s1", "fresh") is True


@pytest.mark.asyncio
async def test_dead_pid_lease_is_reclaimed(db, monkeypatch):
    holder = "pid=424242:tid=1:agent=a:nonce=dead"
    assert await db.try_acquire_compression_lock("s1", holder, ttl_seconds=300)
    monkeypatch.setattr(
        hermes_state,
        "psutil",
        SimpleNamespace(pid_exists=lambda pid: pid != 424242),
    )
    assert await db.try_acquire_compression_lock("s1", "fresh", ttl_seconds=300)


@pytest.mark.asyncio
async def test_live_or_unstructured_lease_waits_for_ttl(db):
    live = f"pid={os.getpid()}:tid=1:agent=a:nonce=live"
    assert await db.try_acquire_compression_lock("s1", live, ttl_seconds=300)
    assert await db.try_acquire_compression_lock("s1", "other") is False
    assert await db.try_acquire_compression_lock("s2", "legacy", ttl_seconds=300)
    assert await db.try_acquire_compression_lock("s2", "other") is False


@pytest.mark.asyncio
async def test_concurrent_acquire_has_exactly_one_winner(db):
    async def acquire(index: int) -> bool:
        return await db.try_acquire_compression_lock("contended", f"holder-{index}")

    tasks = []
    async with asyncio.TaskGroup() as group:
        for index in range(16):
            tasks.append(group.create_task(acquire(index)))

    assert sum(task.result() for task in tasks) == 1
