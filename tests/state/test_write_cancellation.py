"""Cancellation safety for SQLite write transactions."""

import asyncio

import pytest

from hermes_state import SessionDB


@pytest.mark.asyncio
async def test_repeated_cancellation_rolls_back_begin_before_unlock(
    tmp_path,
    monkeypatch,
):
    database = SessionDB(tmp_path / "state.db")
    await database.create_session("seed", source="test")
    connection = await database._get_connection()
    original_execute = connection.execute
    original_rollback = connection.rollback
    begin_executed = asyncio.Event()
    rollback_started = asyncio.Event()
    release_rollback = asyncio.Event()
    rollback_completed = asyncio.Event()

    async def controlled_execute(sql, *args, **kwargs):
        result = await original_execute(sql, *args, **kwargs)
        if sql == "BEGIN IMMEDIATE":
            begin_executed.set()
            await asyncio.Event().wait()
        return result

    async def controlled_rollback():
        rollback_started.set()
        await release_rollback.wait()
        await original_rollback()
        rollback_completed.set()

    monkeypatch.setattr(connection, "execute", controlled_execute)
    monkeypatch.setattr(connection, "rollback", controlled_rollback)

    async def write_row(active_connection):
        await active_connection.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
            ("cancelled", "test", 0.0),
        )

    writing = asyncio.create_task(database._write(write_row))
    await begin_executed.wait()
    writing.cancel()
    await rollback_started.wait()
    writing.cancel()
    await asyncio.sleep(0)

    try:
        assert writing.done() is False
    finally:
        release_rollback.set()
        with pytest.raises(asyncio.CancelledError):
            await writing
        await asyncio.wait_for(rollback_completed.wait(), timeout=1.0)

    monkeypatch.setattr(connection, "execute", original_execute)
    monkeypatch.setattr(connection, "rollback", original_rollback)
    await database.create_session("after-cancellation", source="test")
    assert await database.get_session("after-cancellation") is not None
    await database.close()
