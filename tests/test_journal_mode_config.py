"""Native-async coverage for Hermes' SQLite journal-mode policy."""

from __future__ import annotations

import sqlite3
from unittest.mock import AsyncMock

import aiosqlite
import pytest
import yaml
from blockbuster import BlockBuster

import hermes_state


def _write_config(monkeypatch: pytest.MonkeyPatch, tmp_path, config: object) -> None:
    home = tmp_path / "hermes-home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_resolve_journal_mode_uses_database_config(monkeypatch, tmp_path):
    _write_config(monkeypatch, tmp_path, {"database": {"journal_mode": "DELETE"}})

    assert await hermes_state.resolve_journal_mode() == "delete"


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["bogus", "truncate", None, 42, {"bad": "shape"}])
async def test_invalid_journal_mode_falls_back_to_wal(monkeypatch, tmp_path, value):
    _write_config(monkeypatch, tmp_path, {"database": {"journal_mode": value}})

    assert await hermes_state.resolve_journal_mode() == "wal"


@pytest.mark.asyncio
async def test_apply_wal_honors_configured_delete(monkeypatch, tmp_path):
    _write_config(monkeypatch, tmp_path, {"database": {"journal_mode": "delete"}})
    monkeypatch.setattr(
        hermes_state,
        "is_sqlite_wal_reset_vulnerable",
        lambda version_info=None: False,
    )
    connection = await aiosqlite.connect(tmp_path / "configured.db")
    blockbuster = BlockBuster()
    blockbuster.activate()
    try:
        assert (
            await hermes_state.apply_wal_with_fallback(
                connection,
                db_label="configured.db",
            )
            == "delete"
        )
        row = await (await connection.execute("PRAGMA journal_mode")).fetchone()
        assert row[0].lower() == "delete"
    finally:
        blockbuster.deactivate()
        await connection.close()


@pytest.mark.asyncio
async def test_vulnerable_sqlite_does_not_enable_wal(monkeypatch, tmp_path):
    _write_config(monkeypatch, tmp_path, {"database": {"journal_mode": "wal"}})
    monkeypatch.setattr(
        hermes_state,
        "is_sqlite_wal_reset_vulnerable",
        lambda version_info=None: True,
    )
    connection = await aiosqlite.connect(tmp_path / "fresh.db")
    try:
        mode = await hermes_state.apply_wal_with_fallback(
            connection,
            db_label="fresh.db",
        )
        row = await (await connection.execute("PRAGMA journal_mode")).fetchone()
        assert mode == "delete"
        assert row[0].lower() == "delete"
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_vulnerable_sqlite_preserves_existing_wal(monkeypatch, tmp_path):
    _write_config(monkeypatch, tmp_path, {"database": {"journal_mode": "wal"}})
    path = tmp_path / "existing-wal.db"
    seed = await aiosqlite.connect(path)
    try:
        row = await (await seed.execute("PRAGMA journal_mode=WAL")).fetchone()
        assert row[0].lower() == "wal"
        await seed.execute("CREATE TABLE t (x INTEGER)")
        await seed.execute("INSERT INTO t VALUES (42)")
        await seed.commit()
    finally:
        await seed.close()

    monkeypatch.setattr(
        hermes_state,
        "is_sqlite_wal_reset_vulnerable",
        lambda version_info=None: True,
    )
    connection = await aiosqlite.connect(path)
    try:
        assert await hermes_state.apply_wal_with_fallback(connection) == "wal"
        row = await (await connection.execute("SELECT x FROM t")).fetchone()
        assert row[0] == 42
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_database_pragmas_are_applied(monkeypatch, tmp_path):
    _write_config(
        monkeypatch,
        tmp_path,
        {
            "database": {
                "cache_size": -4096,
                "temp_store": 2,
                "wal_autocheckpoint": 23,
                "journal_size_limit": 65536,
            }
        },
    )
    connection = await aiosqlite.connect(tmp_path / "pragmas.db")
    try:
        await hermes_state.apply_database_pragmas(connection, db_label="test.db")
        for pragma, expected in (
            ("cache_size", -4096),
            ("temp_store", 2),
            ("wal_autocheckpoint", 23),
            ("journal_size_limit", 65536),
        ):
            row = await (await connection.execute(f"PRAGMA {pragma}")).fetchone()
            assert row[0] == expected
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_sessiondb_surfaces_lazy_initialization_failure(
    monkeypatch,
    tmp_path,
):
    hermes_state._set_last_init_error(None)
    failure = sqlite3.OperationalError("locking protocol")
    monkeypatch.setattr(
        hermes_state,
        "apply_wal_with_fallback",
        AsyncMock(side_effect=failure),
    )
    database = hermes_state.SessionDB(tmp_path / "state.db")

    with pytest.raises(sqlite3.OperationalError, match="locking protocol"):
        await database.get_session("missing")

    assert hermes_state.get_last_init_error() == (
        "OperationalError: locking protocol"
    )
    assert "NFS/SMB/FUSE/ZFS" in hermes_state.format_session_db_unavailable()


@pytest.mark.parametrize(
    "version_info,expected",
    [
        ((3, 6, 23), False),
        ((3, 7, 0), True),
        ((3, 44, 6), False),
        ((3, 45, 0), True),
        ((3, 50, 7), False),
        ((3, 51, 2), True),
        ((3, 51, 3), False),
    ],
)
def test_sqlite_wal_reset_vulnerability_matrix(version_info, expected):
    assert hermes_state.is_sqlite_wal_reset_vulnerable(version_info) is expected
