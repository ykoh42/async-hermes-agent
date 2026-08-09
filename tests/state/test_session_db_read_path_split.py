"""Native-async parity tests for the SessionDB WAL read-path split."""

import asyncio
import sqlite3

import pytest
import pytest_asyncio

import hermes_state
from hermes_state import SessionDB


@pytest_asyncio.fixture
async def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    await database.create_session("committed", source="library")
    yield database
    await database.close()


@pytest_asyncio.fixture
async def wal_db(tmp_path, monkeypatch):
    monkeypatch.setattr(
        hermes_state, "is_sqlite_wal_reset_vulnerable", lambda: False
    )
    database = SessionDB(tmp_path / "wal-state.db")
    await database.create_session("committed", source="library")
    await database.append_message(
        "committed", role="user", content="hello graphiti world"
    )
    await database.append_message(
        "committed", role="assistant", content="memory answer"
    )
    assert database._wal_active
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_read_connection_is_reused_and_distinct_from_writer(wal_db):
    db = wal_db
    writer = await db._get_connection()
    first = await db._get_read_conn()
    second = await db._get_read_conn()

    assert first is second
    assert first is not writer


@pytest.mark.asyncio
async def test_wal_read_completes_during_write_and_hides_uncommitted_rows(
    wal_db,
):
    db = wal_db
    write_started = asyncio.Event()
    release_write = asyncio.Event()

    async def held_write(connection):
        await connection.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
            ("pending", "library", 1.0),
        )
        write_started.set()
        await release_write.wait()

    write_task = asyncio.create_task(db._write(held_write))
    await write_started.wait()
    try:
        committed = await asyncio.wait_for(db.get_session("committed"), 0.5)
        pending = await asyncio.wait_for(db.get_session("pending"), 0.5)
        messages = await asyncio.wait_for(db.get_messages("committed"), 0.5)
        conversation = await asyncio.wait_for(
            db.get_messages_as_conversation("committed"), 0.5
        )
        matches = await asyncio.wait_for(
            db.search_messages("graphiti", limit=5), 0.5
        )
        listed = await asyncio.wait_for(
            db.list_sessions_rich(source="library"), 0.5
        )
        assert committed["id"] == "committed"
        assert pending is None
        assert [message["role"] for message in messages] == [
            "user",
            "assistant",
        ]
        assert [message["role"] for message in conversation] == [
            "user",
            "assistant",
        ]
        assert matches and matches[0]["session_id"] == "committed"
        assert any(session["id"] == "committed" for session in listed)
    finally:
        release_write.set()
        await write_task

    assert (await db.get_session("pending"))["id"] == "pending"


@pytest.mark.asyncio
async def test_non_wal_read_fallback_waits_for_writer_lock(db):
    db._wal_active = False
    writer_lock = db._get_write_lock()
    await writer_lock.acquire()
    read_task = asyncio.create_task(db.get_session("committed"))
    try:
        await asyncio.sleep(0)
        assert not read_task.done()
    finally:
        writer_lock.release()

    assert (await read_task)["id"] == "committed"


@pytest.mark.asyncio
async def test_read_connection_open_failure_is_remembered(
    wal_db, monkeypatch
):
    db = wal_db
    real_connect = hermes_state.aiosqlite.connect
    calls = 0

    async def failing_connect(database, *args, **kwargs):
        nonlocal calls
        if isinstance(database, str) and "mode=ro" in database:
            calls += 1
            raise sqlite3.OperationalError("simulated open failure")
        return await real_connect(database, *args, **kwargs)

    monkeypatch.setattr(hermes_state.aiosqlite, "connect", failing_connect)

    assert (await db.get_session("committed"))["id"] == "committed"
    assert (await db.get_session("committed"))["id"] == "committed"
    assert calls == 1


@pytest.mark.asyncio
async def test_close_drains_read_and_writer_connections(db):
    writer = await db._get_connection()
    reader = await db._get_read_conn()

    await db.close()

    with pytest.raises(ValueError, match="no active connection"):
        await writer.execute("SELECT 1")
    if reader is not None:
        with pytest.raises(ValueError, match="no active connection"):
            await reader.execute("SELECT 1")


@pytest.mark.asyncio
async def test_close_waits_for_active_read_context(wal_db, monkeypatch):
    db = wal_db
    reader = await db._get_read_conn()
    original_execute = reader.execute
    fetch_started = asyncio.Event()
    release_fetch = asyncio.Event()

    class ControlledCursor:
        def __init__(self, cursor):
            self._cursor = cursor

        async def fetchall(self):
            fetch_started.set()
            await release_fetch.wait()
            return await self._cursor.fetchall()

        async def close(self):
            await self._cursor.close()

    async def controlled_execute(sql, *args, **kwargs):
        cursor = await original_execute(sql, *args, **kwargs)
        if sql.startswith("SELECT * FROM messages"):
            return ControlledCursor(cursor)
        return cursor

    monkeypatch.setattr(reader, "execute", controlled_execute)
    read_task = asyncio.create_task(db.get_messages("committed"))
    await fetch_started.wait()
    close_task = asyncio.create_task(db.close())
    await asyncio.sleep(0)
    assert not close_task.done()

    release_fetch.set()
    messages = await read_task
    await close_task

    assert len(messages) == 2
