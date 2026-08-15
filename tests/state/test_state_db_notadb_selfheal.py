"""Async port of upstream state.db connection self-heal coverage."""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from hermes_state import (
    SessionDB,
    _is_not_a_database_error,
    _on_disk_journal_mode,
)


class _NotADbOnce:
    def __init__(self, real_connection):
        self._real = real_connection

    async def execute(self, *args, **kwargs):
        raise sqlite3.DatabaseError("file is not a database")

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.mark.asyncio
async def test_not_a_database_classifier_matches_only_sqlite_error():
    assert _is_not_a_database_error(
        sqlite3.DatabaseError("file is not a database")
    )
    assert not _is_not_a_database_error(
        sqlite3.DatabaseError("database disk image is malformed")
    )
    assert not _is_not_a_database_error(ValueError("file is not a database"))


@pytest.mark.asyncio
async def test_write_self_heals_when_connection_breaks(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        await db.create_session(session_id="s1", source="cli", model="test")
        db._connection = _NotADbOnce(db._connection)

        await db.create_session(session_id="s2", source="cli", model="test")

        assert db._notadb_reconnect_attempted is True
        assert await db.get_session("s1") is not None
        assert await db.get_session("s2") is not None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_reconnect_is_one_shot(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        await db.create_session(session_id="s1", source="cli", model="test")
        db._notadb_reconnect_attempted = True
        db._connection = _NotADbOnce(db._connection)
        with pytest.raises(sqlite3.DatabaseError, match="not a database"):
            await db.create_session(session_id="s2", source="cli", model="test")
    finally:
        broken = db._connection
        if broken is not None:
            await broken._real.close()
        db._connection = None
        await db.close()


class _FakeCursor:
    def __init__(self, row):
        self.row = row

    async def fetchone(self):
        return self.row


class _JournalConnection:
    def __init__(self, failures, row=("wal",)):
        self.failures = list(failures)
        self.row = row
        self.calls = 0

    async def execute(self, _sql):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return _FakeCursor(self.row)


@pytest.mark.asyncio
async def test_transient_journal_eio_clears_on_retry():
    conn = _JournalConnection(
        [sqlite3.OperationalError("disk i/o error")] * 2
    )
    assert await _on_disk_journal_mode(conn) == "wal"
    assert conn.calls == 3


@pytest.mark.asyncio
async def test_persistent_journal_eio_is_bounded():
    conn = _JournalConnection(
        [sqlite3.OperationalError("disk i/o error")] * 10
    )
    assert await _on_disk_journal_mode(conn) is None
    assert conn.calls == 4


@pytest.mark.asyncio
async def test_non_eio_journal_error_fails_fast():
    conn = _JournalConnection([sqlite3.OperationalError("database is locked")])
    assert await _on_disk_journal_mode(conn) is None
    assert conn.calls == 1
