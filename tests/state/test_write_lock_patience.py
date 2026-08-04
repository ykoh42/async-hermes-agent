"""Native-async SQLite write contention behavior."""

import asyncio
import sqlite3
import threading
import time

import pytest
import pytest_asyncio

from hermes_state import SessionDB


def _hold_write_lock(db_path, hold_seconds, started):
    connection = sqlite3.connect(str(db_path), timeout=1.0, isolation_level=None)
    try:
        connection.execute("BEGIN IMMEDIATE")
        started.set()
        time.sleep(hold_seconds)
        connection.execute("COMMIT")
    finally:
        connection.close()


async def _wait_for_thread(started):
    for _ in range(500):
        if started.is_set():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("lock holder did not start")


@pytest_asyncio.fixture
async def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    await database.create_session("seed", source="library")
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_write_waits_without_blocking_event_loop(db):
    started = threading.Event()
    holder = threading.Thread(target=_hold_write_lock, args=(db.db_path, 1.5, started))
    holder.start()
    await _wait_for_thread(started)

    heartbeats = 0

    async def heartbeat():
        nonlocal heartbeats
        while holder.is_alive():
            heartbeats += 1
            await asyncio.sleep(0.01)

    async with asyncio.TaskGroup() as group:
        write = group.create_task(
            db.append_message("seed", role="user", content="survived contention")
        )
        group.create_task(heartbeat())

    holder.join(timeout=2)
    assert isinstance(write.result(), int)
    assert heartbeats >= 20
    assert (await db.get_messages("seed"))[0]["content"] == "survived contention"


@pytest.mark.asyncio
async def test_exhausted_patience_reports_contention(db, monkeypatch):
    monkeypatch.setattr(SessionDB, "_WRITE_PATIENCE_S", 0.05)
    started = threading.Event()
    holder = threading.Thread(target=_hold_write_lock, args=(db.db_path, 1.5, started))
    holder.start()
    await _wait_for_thread(started)
    try:
        with pytest.raises(sqlite3.OperationalError, match="another Hermes process"):
            await db.set_meta("key", "value")
    finally:
        holder.join(timeout=2)


@pytest.mark.asyncio
async def test_lazy_open_propagates_non_database_path_error(tmp_path):
    path = tmp_path / "state.db"
    path.mkdir()
    db = SessionDB(path)
    with pytest.raises(sqlite3.Error):
        await db.get_session("missing")
