"""Async parity coverage for upstream WAL and bounded FTS maintenance."""

import asyncio
import logging
import sqlite3

import pytest
import pytest_asyncio

from hermes_state import SessionDB


@pytest_asyncio.fixture
async def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_periodic_checkpoint_uses_passive_not_truncate(db):
    connection = await db._get_connection()
    statements = []
    await connection.set_trace_callback(statements.append)
    try:
        await db._try_wal_checkpoint()
    finally:
        await connection.set_trace_callback(None)

    assert sum("wal_checkpoint(PASSIVE)" in sql for sql in statements) == 1
    assert not any("wal_checkpoint(TRUNCATE)" in sql for sql in statements)


@pytest.mark.asyncio
async def test_periodic_checkpoint_logs_warning_on_failure(
    db, monkeypatch, caplog
):
    connection = await db._get_connection()
    original_execute = connection.execute

    async def fail_passive(sql, *args, **kwargs):
        if "wal_checkpoint(PASSIVE)" in sql:
            raise sqlite3.OperationalError("disk I/O error")
        return await original_execute(sql, *args, **kwargs)

    monkeypatch.setattr(connection, "execute", fail_passive)
    with caplog.at_level(logging.WARNING):
        await db._try_wal_checkpoint()

    assert "WAL checkpoint (PASSIVE) failed" in caplog.text


@pytest.mark.asyncio
async def test_successful_writes_trigger_checkpoint_and_merge_at_cadence(
    db, monkeypatch
):
    checkpoints = []
    merges = []

    async def checkpoint():
        checkpoints.append(db._write_count)

    async def merge():
        merges.append(db._write_count)

    monkeypatch.setattr(db, "_CHECKPOINT_EVERY_N_WRITES", 3)
    monkeypatch.setattr(db, "_FTS_MERGE_EVERY_N_WRITES", 5)
    monkeypatch.setattr(db, "_try_wal_checkpoint", checkpoint)
    monkeypatch.setattr(db, "_try_incremental_merge_fts", merge)

    await db.create_session("s1", source="library")
    for index in range(4):
        await db.append_message("s1", role="user", content=str(index))

    assert checkpoints == [3]
    assert merges == [5]


@pytest.mark.asyncio
async def test_incremental_fts_merge_is_bounded_and_never_optimizes(db):
    await db.create_session("s1", source="library")
    await db.append_message("s1", role="user", content="bounded merge")
    connection = await db._get_connection()
    statements = []
    await connection.set_trace_callback(statements.append)
    try:
        executed = await db._merge_fts_incrementally(max_pages=37)
    finally:
        await connection.set_trace_callback(None)

    present = [
        table_name
        for table_name in db._FTS_TABLES
        if await db._fts_table_exists(table_name)
    ]
    merge_sql = [sql for sql in statements if "VALUES('merge', 37)" in sql]
    assert present
    assert len(merge_sql) == executed
    assert len(present) <= executed <= (
        len(present) * db._FTS_MERGE_COMMANDS_PER_PASS
    )
    for table_name in present:
        assert 1 <= sum(
            f"{table_name}({table_name}, rank)" in sql for sql in merge_sql
        ) <= db._FTS_MERGE_COMMANDS_PER_PASS
    assert any("VALUES('usermerge', 2)" in sql for sql in statements)
    assert not any("'optimize'" in sql for sql in statements)


@pytest.mark.asyncio
async def test_close_uses_truncate_and_finishes_connection_shutdown(db):
    connection = await db._get_connection()
    statements = []
    await connection.set_trace_callback(statements.append)

    await db.close()

    assert sum("wal_checkpoint(TRUNCATE)" in sql for sql in statements) == 1
    with pytest.raises(ValueError, match="no active connection"):
        await connection.execute("SELECT 1")


@pytest.mark.asyncio
async def test_close_logs_debug_when_truncate_fails(db, monkeypatch, caplog):
    connection = await db._get_connection()
    original_execute = connection.execute

    async def fail_truncate(sql, *args, **kwargs):
        if "wal_checkpoint(TRUNCATE)" in sql:
            raise sqlite3.OperationalError("database is locked")
        return await original_execute(sql, *args, **kwargs)

    monkeypatch.setattr(connection, "execute", fail_truncate)
    with caplog.at_level(logging.DEBUG):
        await db.close()

    assert "WAL checkpoint (TRUNCATE) at close failed" in caplog.text
    with pytest.raises(ValueError, match="no active connection"):
        await original_execute("SELECT 1")


@pytest.mark.asyncio
async def test_close_cancellation_still_closes_owned_connection(
    db, monkeypatch
):
    connection = await db._get_connection()
    original_execute = connection.execute
    checkpoint_started = asyncio.Event()

    async def block_truncate(sql, *args, **kwargs):
        if "wal_checkpoint(TRUNCATE)" in sql:
            checkpoint_started.set()
            await asyncio.sleep(0.05)
        return await original_execute(sql, *args, **kwargs)

    monkeypatch.setattr(connection, "execute", block_truncate)
    close_task = asyncio.create_task(db.close())
    await checkpoint_started.wait()
    close_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await close_task
    with pytest.raises(ValueError, match="no active connection"):
        await original_execute("SELECT 1")
