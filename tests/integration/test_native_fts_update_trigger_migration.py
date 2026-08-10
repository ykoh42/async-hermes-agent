"""Native-async parity coverage for upstream FTS UPDATE OF migration."""

from __future__ import annotations

import sqlite3

import aiosqlite
import pytest
from blockbuster import BlockBuster
from pyleak import no_task_leaks
from pyleak.eventloop import LeakAction

from hermes_state import SessionDB
from hermes_state_common import FTS_CJK_STALE_KEY
from hermes_state_schema import SessionSchemaMixin


pytestmark = pytest.mark.asyncio


async def _execute(connection, sql: str, params=()) -> None:
    cursor = await connection.execute(sql, params)
    await cursor.close()


async def _trigger_sql(connection, name: str) -> str | None:
    async with connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?",
        (name,),
    ) as cursor:
        row = await cursor.fetchone()
    return row[0] if row else None


async def _install_legacy_inline_base_fts(database: SessionDB, connection) -> None:
    """Replace v23 FTS with the broad inline shape shipped by v11..v22."""
    await database._drop_fts_triggers(connection)
    cursor = await connection.executescript(
        """
        DROP TABLE IF EXISTS messages_fts;
        DROP TABLE IF EXISTS messages_fts_trigram;
        DROP VIEW IF EXISTS messages_fts_trigram_src;

        CREATE VIRTUAL TABLE messages_fts USING fts5(content);
        CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, content) VALUES (
                new.id,
                COALESCE(new.content, '') || ' ' ||
                COALESCE(new.tool_name, '') || ' ' ||
                COALESCE(new.tool_calls, '')
            );
        END;
        CREATE TRIGGER messages_fts_delete AFTER DELETE ON messages BEGIN
            DELETE FROM messages_fts WHERE rowid = old.id;
        END;
        CREATE TRIGGER messages_fts_update AFTER UPDATE ON messages BEGIN
            DELETE FROM messages_fts WHERE rowid = old.id;
            INSERT INTO messages_fts(rowid, content) VALUES (
                new.id,
                COALESCE(new.content, '') || ' ' ||
                COALESCE(new.tool_name, '') || ' ' ||
                COALESCE(new.tool_calls, '')
            );
        END;
        """
    )
    await cursor.close()


async def test_fresh_database_installs_narrow_update_triggers(tmp_path):
    database = SessionDB(tmp_path / "fresh.db")
    try:
        connection = await database._get_connection()
        sql = await _trigger_sql(connection, "messages_fts_update")
        if sql is None:
            pytest.skip("SQLite build has no FTS5")
        compact = " ".join(sql.split()).upper()
        assert "AFTER UPDATE OF " in compact
        assert "CONTENT" in compact
        assert "TOOL_NAME" in compact
        assert "TOOL_CALLS" in compact

        trigram = await _trigger_sql(
            connection,
            "messages_fts_trigram_update",
        )
        if trigram is not None:
            assert "AFTER UPDATE OF " in " ".join(trigram.split()).upper()
    finally:
        await database.close()


async def test_reopen_migrates_broad_trigger_and_bypasses_status_updates(
    tmp_path,
):
    path = tmp_path / "restart.db"
    original = SessionDB(path)
    connection = await original._get_connection()
    if await _trigger_sql(connection, "messages_fts_update") is None:
        await original.close()
        pytest.skip("SQLite build has no FTS5")
    await original.close()

    migration_connection = await aiosqlite.connect(path, isolation_level=None)
    try:
        await _execute(
            migration_connection,
            "DROP TRIGGER IF EXISTS messages_fts_update",
        )
        await _execute(
            migration_connection,
            "CREATE TRIGGER messages_fts_update AFTER UPDATE ON messages "
            "BEGIN SELECT 1; END",
        )
    finally:
        await migration_connection.close()

    restored = SessionDB(path)
    blocker = BlockBuster()
    async with no_task_leaks(action=LeakAction.RAISE):
        blocker.activate()
        try:
            connection = await restored._get_connection()
            migrated = await _trigger_sql(connection, "messages_fts_update")
            assert migrated is not None
            assert "AFTER UPDATE OF " in " ".join(migrated.split()).upper()

            await restored.create_session("s", source="test")
            message_id = await restored.append_message(
                "s",
                role="user",
                content="searchable",
            )
            await _execute(connection, "DROP TABLE messages_fts")

            await _execute(
                connection,
                "UPDATE messages SET active = 0, compacted = 1, observed = 1 "
                "WHERE id = ?",
                (message_id,),
            )
            with pytest.raises(
                sqlite3.OperationalError,
                match=r"no such table.*messages_fts",
            ):
                await _execute(
                    connection,
                    "UPDATE messages SET content = 'changed' WHERE id = ?",
                    (message_id,),
                )
        finally:
            await restored.close()
            blocker.deactivate()


async def test_cjk_soft_failure_is_quarantined_durably(tmp_path, monkeypatch):
    database = SessionDB(tmp_path / "cjk-soft.db")
    try:
        connection = await database._get_connection()
        if await database._db_has_legacy_inline_fts(connection):
            pytest.skip("fresh database unexpectedly uses legacy FTS")
        await _execute(
            connection,
            "DROP TRIGGER IF EXISTS messages_fts_cjk_update",
        )
        await _execute(
            connection,
            "CREATE TRIGGER messages_fts_cjk_update "
            "AFTER UPDATE ON messages BEGIN SELECT 1; END",
        )
        database._fts_cjk_available = True

        async def _soft_fail(_connection):
            database._fts_cjk_available = False

        monkeypatch.setattr(database, "_ensure_fts_cjk_schema", _soft_fail)
        dropped = await database._migrate_broad_fts_update_triggers(connection)

        assert dropped >= 1
        assert database._fts_cjk_available is False
        assert await _trigger_sql(connection, "messages_fts_cjk_update") is None
        assert await database.get_meta(FTS_CJK_STALE_KEY) == "1"
    finally:
        await database.close()


async def test_cjk_ensure_exception_is_quarantined_and_propagated(
    tmp_path,
    monkeypatch,
):
    database = SessionDB(tmp_path / "cjk-error.db")
    try:
        connection = await database._get_connection()
        if await database._db_has_legacy_inline_fts(connection):
            pytest.skip("fresh database unexpectedly uses legacy FTS")
        await _execute(
            connection,
            "DROP TRIGGER IF EXISTS messages_fts_cjk_update",
        )
        await _execute(
            connection,
            "CREATE TRIGGER messages_fts_cjk_update "
            "AFTER UPDATE ON messages BEGIN SELECT 1; END",
        )

        async def _fail(_connection):
            raise sqlite3.DatabaseError("injected CJK ensure failure")

        monkeypatch.setattr(database, "_ensure_fts_cjk_schema", _fail)
        with pytest.raises(
            sqlite3.DatabaseError,
            match="injected CJK ensure failure",
        ):
            await database._migrate_broad_fts_update_triggers(connection)

        assert database._fts_cjk_available is False
        assert await _trigger_sql(connection, "messages_fts_cjk_update") is None
        assert await database.get_meta(FTS_CJK_STALE_KEY) == "1"
    finally:
        await database.close()


async def test_cjk_broad_trigger_is_restored_as_update_of(tmp_path, monkeypatch):
    database = SessionDB(tmp_path / "cjk-restored.db")
    try:
        connection = await database._get_connection()
        if await database._db_has_legacy_inline_fts(connection):
            pytest.skip("fresh database unexpectedly uses legacy FTS")
        await _execute(
            connection,
            "DROP TRIGGER IF EXISTS messages_fts_cjk_update",
        )
        await _execute(
            connection,
            "CREATE TRIGGER messages_fts_cjk_update "
            "AFTER UPDATE ON messages BEGIN SELECT 1; END",
        )

        async def _restore(_connection):
            await _execute(
                _connection,
                "CREATE TRIGGER messages_fts_cjk_update "
                "AFTER UPDATE OF content, tool_name, tool_calls ON messages "
                "BEGIN SELECT 1; END",
            )

        monkeypatch.setattr(database, "_ensure_fts_cjk_schema", _restore)
        assert await database._migrate_broad_fts_update_triggers(connection) == 1
        cjk_sql = await _trigger_sql(connection, "messages_fts_cjk_update")
        assert cjk_sql is not None
        assert "AFTER UPDATE OF " in " ".join(cjk_sql.split()).upper()
    finally:
        await database.close()


async def test_legacy_migration_does_not_drop_cjk_trigger(tmp_path):
    database = SessionDB(tmp_path / "legacy.db")
    try:
        connection = await database._get_connection()
        if not database._fts_enabled:
            pytest.skip("SQLite build has no FTS5")
        await _install_legacy_inline_base_fts(database, connection)
        await _execute(
            connection,
            "DROP TRIGGER IF EXISTS messages_fts_cjk_update",
        )
        await _execute(
            connection,
            "CREATE TRIGGER messages_fts_cjk_update "
            "AFTER UPDATE ON messages BEGIN SELECT 1; END",
        )

        assert await database._db_has_legacy_inline_fts(connection)
        assert await database._migrate_broad_fts_update_triggers(connection) >= 1
        base_sql = await _trigger_sql(connection, "messages_fts_update")
        cjk_sql = await _trigger_sql(connection, "messages_fts_cjk_update")
        assert base_sql is not None
        assert "AFTER UPDATE OF " in " ".join(base_sql.split()).upper()
        assert cjk_sql is not None
        assert "AFTER UPDATE OF " not in " ".join(cjk_sql.split()).upper()
    finally:
        await database.close()


async def test_trigger_narrowing_helper_matches_upstream():
    assert SessionSchemaMixin._fts_update_trigger_needs_narrowing(
        "CREATE TRIGGER t AFTER UPDATE ON messages BEGIN SELECT 1; END"
    )
    assert not SessionSchemaMixin._fts_update_trigger_needs_narrowing(
        "CREATE TRIGGER t AFTER UPDATE OF content ON messages BEGIN SELECT 1; END"
    )
    assert not SessionSchemaMixin._fts_update_trigger_needs_narrowing(None)
