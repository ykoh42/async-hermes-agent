"""Native-async state.db connection self-healing regressions."""

import sqlite3
from unittest.mock import AsyncMock

import pytest

from hermes_state import SessionDB, _is_not_a_database_error, _on_disk_journal_mode


class _BrokenConnection:
    def __init__(self, connection):
        self._connection = connection

    async def execute(self, *_args, **_kwargs):
        raise sqlite3.DatabaseError("file is not a database")

    async def close(self):
        await self._connection.close()

    def __getattr__(self, name):
        return getattr(self._connection, name)


def test_not_a_database_classifier_is_narrow():
    assert _is_not_a_database_error(
        sqlite3.DatabaseError("file is not a database")
    )
    assert not _is_not_a_database_error(
        sqlite3.DatabaseError("database disk image is malformed")
    )
    assert not _is_not_a_database_error(ValueError("file is not a database"))


@pytest.mark.asyncio
async def test_write_reopens_broken_connection_once(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        await db.create_session("first", source="library")
        connection = db._connection
        db._connection = _BrokenConnection(connection)
        await db.create_session("second", source="library")
        assert db._notadb_reconnect_attempted is True
        assert await db.get_session("first") is not None
        assert await db.get_session("second") is not None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_not_a_database_reconnect_is_one_shot(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        await db.create_session("first", source="library")
        db._notadb_reconnect_attempted = True
        connection = db._connection
        broken = _BrokenConnection(connection)
        db._connection = broken
        with pytest.raises(sqlite3.DatabaseError, match="not a database"):
            await db.create_session("second", source="library")
    finally:
        if "broken" in locals():
            await broken.close()
        db._connection = None
        await db.close()


class _AsyncCursor:
    def __init__(self, row):
        self.fetchone = AsyncMock(return_value=row)


@pytest.mark.asyncio
async def test_journal_mode_retries_transient_disk_io_error():
    connection = AsyncMock()
    connection.execute.side_effect = [
        sqlite3.OperationalError("disk i/o error"),
        sqlite3.OperationalError("disk i/o error"),
        _AsyncCursor(("wal",)),
    ]
    assert await _on_disk_journal_mode(connection) == "wal"
    assert connection.execute.await_count == 3


@pytest.mark.asyncio
async def test_journal_mode_bounds_persistent_disk_io_retries():
    connection = AsyncMock()
    connection.execute.side_effect = sqlite3.OperationalError("disk i/o error")
    assert await _on_disk_journal_mode(connection) is None
    assert connection.execute.await_count == 4
