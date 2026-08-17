"""Native-async recovery for malformed state database schema and indexes."""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pytest
from blockbuster import BlockBuster

import hermes_state
from hermes_state import SessionDB


async def _build_healthy_db(db_path: Path) -> str:
    database = SessionDB(db_path)
    session_id = await database.create_session(
        session_id=str(uuid.uuid4()),
        source="library",
    )
    for index in range(5):
        await database.append_message(
            session_id,
            role="user",
            content=f"hello world {index}",
        )
        await database.append_message(
            session_id,
            role="assistant",
            content=f"reply about pizza {index}",
        )
    await database.close()
    return session_id


def _corrupt_duplicate_fts(db_path: Path) -> None:
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("PRAGMA writable_schema=ON")
        connection.execute(
            "INSERT INTO sqlite_master (type, name, tbl_name, rootpage, sql) "
            "SELECT type, name, tbl_name, rootpage, sql FROM sqlite_master "
            "WHERE name='messages_fts'"
        )
        connection.commit()
    finally:
        connection.close()


def _corrupt_fts_index_data(db_path: Path) -> None:
    connection = sqlite3.connect(db_path, isolation_level=None)
    try:
        connection.execute(
            "UPDATE messages_fts_data SET block = X'DEADBEEFDEADBEEF'"
        )
    finally:
        connection.close()


def _corrupt_btree_index(db_path: Path, index_name: str) -> None:
    connection = sqlite3.connect(db_path)
    original_sql = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND name=?",
        (index_name,),
    ).fetchone()[0]

    def set_index_sql(sql: str) -> None:
        connection.execute("PRAGMA writable_schema=ON")
        connection.execute(
            "UPDATE sqlite_master SET sql=? WHERE type='index' AND name=?",
            (sql, index_name),
        )
        version = connection.execute("PRAGMA schema_version").fetchone()[0]
        connection.execute(f"PRAGMA schema_version={version + 1}")
        connection.execute("PRAGMA writable_schema=OFF")
        connection.commit()

    set_index_sql(original_sql + " WHERE 0")
    connection.close()

    connection = sqlite3.connect(db_path)
    try:
        connection.execute(f"REINDEX {index_name}")
        connection.commit()
        connection.execute("PRAGMA writable_schema=ON")
        connection.execute(
            "UPDATE sqlite_master SET sql=? WHERE type='index' AND name=?",
            (original_sql, index_name),
        )
        version = connection.execute("PRAGMA schema_version").fetchone()[0]
        connection.execute(f"PRAGMA schema_version={version + 1}")
        connection.execute("PRAGMA writable_schema=OFF")
        connection.commit()
    finally:
        connection.close()


@pytest.mark.asyncio
async def test_duplicate_fts_schema_is_repaired_without_blocking(tmp_path):
    path = tmp_path / "state.db"
    await _build_healthy_db(path)
    _corrupt_duplicate_fts(path)

    blocker = BlockBuster()
    blocker.activate()
    try:
        report = await hermes_state.repair_state_db_schema(path)
    finally:
        blocker.deactivate()

    assert report["repaired"] is True
    assert report["strategy"] == "dedup_schema"
    assert report["backup_path"]
    assert Path(report["backup_path"]).exists()

    database = SessionDB(path)
    try:
        connection = await database._get_connection()
        row = await (
            await connection.execute(
                "SELECT COUNT(*) FROM messages_fts "
                "WHERE messages_fts MATCH 'pizza'"
            )
        ).fetchone()
        assert row[0] == 5
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_sessiondb_auto_repairs_malformed_schema(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    session_id = await _build_healthy_db(path)
    _corrupt_duplicate_fts(path)
    monkeypatch.setattr(hermes_state, "_repair_attempted_paths", set())

    database = SessionDB(path)
    blocker = BlockBuster()
    blocker.activate()
    try:
        assert (await database.get_session(session_id))["id"] == session_id
    finally:
        blocker.deactivate()
        await database.close()


@pytest.mark.asyncio
async def test_read_only_sessiondb_does_not_repair_malformed_schema(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "state.db"
    await _build_healthy_db(path)
    _corrupt_duplicate_fts(path)
    monkeypatch.setattr(hermes_state, "_repair_attempted_paths", set())
    repair_called = False

    async def unexpected_repair(db_path, *, backup=True):
        nonlocal repair_called
        repair_called = True
        raise AssertionError("read-only SessionDB attempted a database repair")

    monkeypatch.setattr(
        hermes_state,
        "repair_state_db_schema",
        unexpected_repair,
    )
    database = SessionDB(path, read_only=True)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            await database.get_session("missing")
    finally:
        await database.close()

    assert repair_called is False


@pytest.mark.asyncio
async def test_auto_repair_is_attempted_once_per_process(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    await _build_healthy_db(path)
    _corrupt_duplicate_fts(path)
    monkeypatch.setattr(hermes_state, "_repair_attempted_paths", set())
    calls = 0

    async def failed_repair(db_path, *, backup=True):
        nonlocal calls
        calls += 1
        return {
            "repaired": False,
            "strategy": None,
            "backup_path": None,
            "error": "still malformed",
        }

    monkeypatch.setattr(hermes_state, "repair_state_db_schema", failed_repair)
    for _ in range(2):
        database = SessionDB(path)
        with pytest.raises(sqlite3.DatabaseError):
            await database.get_session("missing")
        await database.close()
    assert calls == 1


@pytest.mark.asyncio
async def test_unrepairable_file_keeps_backup_and_fails_safely(tmp_path):
    path = tmp_path / "state.db"
    path.write_bytes(b"SQLite format 3\x00" + b"\x00\xde\xad\xbe\xef" * 200)

    blocker = BlockBuster()
    blocker.activate()
    try:
        report = await hermes_state.repair_state_db_schema(path)
    finally:
        blocker.deactivate()

    assert report["repaired"] is False
    assert report["error"]
    assert report["backup_path"]
    assert Path(report["backup_path"]).exists()


@pytest.mark.asyncio
async def test_fts_write_corruption_is_rebuilt_in_place(tmp_path):
    path = tmp_path / "state.db"
    session_id = await _build_healthy_db(path)
    _corrupt_fts_index_data(path)
    assert await hermes_state._db_opens_cleanly(path) is not None

    blocker = BlockBuster()
    blocker.activate()
    try:
        report = await hermes_state.repair_state_db_schema(path)
    finally:
        blocker.deactivate()

    assert report["repaired"] is True
    assert report["strategy"] == "rebuild_fts"
    assert await hermes_state._db_opens_cleanly(path) is None
    database = SessionDB(path)
    try:
        await database.append_message(
            session_id,
            role="user",
            content="post repair pizza message",
        )
        assert len(await database.get_messages(session_id)) == 11
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_stale_btree_index_is_reindexed_without_losing_rows(tmp_path):
    path = tmp_path / "state.db"
    session_id = await _build_healthy_db(path)
    _corrupt_btree_index(path, "idx_messages_session")
    reason = await hermes_state._db_opens_cleanly(path)
    assert reason is not None
    assert "wrong # of entries in index idx_messages_session" in reason

    blocker = BlockBuster()
    blocker.activate()
    try:
        report = await hermes_state.repair_state_db_schema(path, backup=False)
    finally:
        blocker.deactivate()

    assert report["strategy"] == "reindex_btree"
    assert await hermes_state._db_opens_cleanly(path) is None
    database = SessionDB(path)
    try:
        assert len(await database.get_messages(session_id)) == 10
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_repair_ledger_stops_repeating_same_failed_surgery(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    path.write_bytes(b"not a recoverable sqlite database")

    for _ in range(hermes_state._MAX_PERSISTENT_REPAIR_ATTEMPTS):
        await hermes_state._record_repair_outcome(path, repaired=False)

    assert await hermes_state._persistent_repair_attempts_exhausted(path)

    async def unexpected_probe(db_path):
        raise AssertionError("an exhausted repair budget must fail closed")

    monkeypatch.setattr(hermes_state, "_db_opens_cleanly", unexpected_probe)
    report = await hermes_state.repair_state_db_schema(path)
    assert report["repaired"] is False
    assert "manual recovery" in report["error"]


@pytest.mark.asyncio
async def test_repair_ledger_resets_when_database_fingerprint_changes(tmp_path):
    path = tmp_path / "state.db"
    path.write_bytes(b"first damaged file")
    for _ in range(hermes_state._MAX_PERSISTENT_REPAIR_ATTEMPTS):
        await hermes_state._record_repair_outcome(path, repaired=False)
    assert await hermes_state._persistent_repair_attempts_exhausted(path)

    path.write_bytes(b"a different damaged file with a new size")
    assert not await hermes_state._persistent_repair_attempts_exhausted(path)


@pytest.mark.asyncio
async def test_successful_repair_clears_persistent_ledger(tmp_path):
    path = tmp_path / "state.db"
    path.write_bytes(b"damaged")
    await hermes_state._record_repair_outcome(path, repaired=False)
    assert hermes_state._repair_ledger_path(path).exists()

    await hermes_state._record_repair_outcome(path, repaired=True)
    assert not hermes_state._repair_ledger_path(path).exists()


@pytest.mark.asyncio
async def test_malformed_backup_is_deduplicated_and_retained(tmp_path):
    path = tmp_path / "state.db"
    path.write_bytes(b"damaged sqlite bytes")

    first = await hermes_state._backup_db_file(path)
    second = await hermes_state._backup_db_file(path)
    assert first is not None
    assert second == first
    assert len(await hermes_state._existing_malformed_backups(path)) == 1

    for index in range(4):
        backup = path.with_name(f"{path.name}.malformed-backup-2026010{index}")
        backup.write_bytes(b"backup")
        backup.with_name(backup.name + "-wal").write_bytes(b"wal")
        backup.with_name(backup.name + "-shm").write_bytes(b"shm")

    await hermes_state._prune_malformed_backups(path, keep=3)
    retained = await hermes_state._existing_malformed_backups(path)
    assert len(retained) == 3
    assert not path.with_name(
        f"{path.name}.malformed-backup-20260100-wal"
    ).exists()
