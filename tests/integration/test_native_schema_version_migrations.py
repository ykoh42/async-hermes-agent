"""Native-async E2E coverage for retained SessionDB schema migrations."""

from __future__ import annotations

import asyncio
import inspect
import json

import pytest
from blockbuster import BlockBuster
from pyleak import no_task_leaks
from pyleak.eventloop import LeakAction

from hermes_state import SCHEMA_VERSION, SessionDB
from hermes_state_common import FTS_STORAGE_VERSION
from hermes_state_schema import SessionSchemaMixin


pytestmark = pytest.mark.asyncio


async def _execute(connection, sql: str, params=()) -> None:
    statement = await connection.execute(sql, params)
    await statement.close()


async def _fetchone(connection, sql: str, params=()):
    async with connection.execute(sql, params) as cursor:
        return await cursor.fetchone()


async def _fetchall(connection, sql: str, params=()):
    async with connection.execute(sql, params) as cursor:
        return await cursor.fetchall()


async def test_v15_migration_tags_existing_ephemeral_subagent_children(tmp_path):
    path = tmp_path / "subagent.db"
    original = SessionDB(path)
    await original.create_session("parent", source="cli")
    await original.create_session(
        "child",
        source="delegate",
        parent_session_id="parent",
    )
    connection = await original._get_connection()
    await _execute(connection, "UPDATE schema_version SET version = 15")
    await original.close()

    restored = SessionDB(path)
    try:
        child = await restored.get_session("child")
        assert child is not None
        assert json.loads(child["model_config"])["_delegate_from"] == "parent"
        connection = await restored._get_connection()
        assert (
            await _fetchone(connection, "SELECT version FROM schema_version")
        )[0] == SCHEMA_VERSION
    finally:
        await restored.close()


async def test_v19_migration_seeds_historical_per_model_usage(tmp_path):
    path = tmp_path / "usage.db"
    original = SessionDB(path)
    await original.create_session(
        "historical",
        source="cli",
        model="test/model",
    )
    connection = await original._get_connection()
    await _execute(
        connection,
        "UPDATE sessions SET billing_provider = 'provider', "
        "billing_base_url = 'https://example.test/v1', billing_mode = 'api_key', "
        "api_call_count = 2, input_tokens = 11, output_tokens = 7, "
        "cache_read_tokens = 3, cache_write_tokens = 2, reasoning_tokens = 5, "
        "estimated_cost_usd = 0.25 WHERE id = 'historical'",
    )
    await _execute(
        connection,
        "DELETE FROM session_model_usage WHERE session_id = 'historical'",
    )
    await _execute(connection, "UPDATE schema_version SET version = 19")
    await original.close()

    restored = SessionDB(path)
    try:
        connection = await restored._get_connection()
        row = await _fetchone(
            connection,
            "SELECT model, billing_provider, billing_base_url, billing_mode, "
            "task, api_call_count, input_tokens, output_tokens, "
            "cache_read_tokens, cache_write_tokens, reasoning_tokens, "
            "estimated_cost_usd FROM session_model_usage "
            "WHERE session_id = 'historical'",
        )
        assert tuple(row) == (
            "test/model",
            "provider",
            "https://example.test/v1",
            "api_key",
            "",
            2,
            11,
            7,
            3,
            2,
            5,
            0.25,
        )
    finally:
        await restored.close()


async def test_v24_inline_prompt_replaces_stale_existing_hash(tmp_path):
    path = tmp_path / "prompt.db"
    original = SessionDB(path)
    await original.create_session(
        "session",
        source="cli",
        system_prompt="old addressed prompt",
    )
    connection = await original._get_connection()
    await _execute(
        connection,
        "UPDATE sessions SET system_prompt = 'authoritative inline prompt' "
        "WHERE id = 'session'",
    )
    await _execute(connection, "UPDATE schema_version SET version = 24")
    await original.close()

    restored = SessionDB(path)
    try:
        session = await restored.get_session("session")
        assert session is not None
        assert session["system_prompt"] == "authoritative inline prompt"
        connection = await restored._get_connection()
        raw = await _fetchone(
            connection,
            "SELECT s.system_prompt, sp.prompt FROM sessions s "
            "JOIN system_prompts sp ON sp.hash = s.system_prompt_hash "
            "WHERE s.id = 'session'",
        )
        assert tuple(raw) == (None, "authoritative inline prompt")
    finally:
        await restored.close()


async def test_duplicate_titles_are_repaired_before_unique_index_restore(tmp_path):
    path = tmp_path / "titles.db"
    original = SessionDB(path)
    await original.create_session("older", source="cli")
    await original.create_session("newer", source="cli")
    connection = await original._get_connection()
    await _execute(
        connection,
        "DROP INDEX IF EXISTS idx_sessions_title_unique",
    )
    await _execute(
        connection,
        "UPDATE sessions SET title = 'duplicate' WHERE id IN ('older', 'newer')",
    )
    await original.close()

    restored = SessionDB(path)
    try:
        connection = await restored._get_connection()
        rows = await _fetchall(
            connection,
            "SELECT id, title FROM sessions ORDER BY rowid",
        )
        assert [(row["id"], row["title"]) for row in rows] == [
            ("older", None),
            ("newer", "duplicate"),
        ]
        assert await _fetchone(
            connection,
            "SELECT 1 FROM sqlite_master WHERE type = 'index' "
            "AND name = 'idx_sessions_title_unique'",
        )
    finally:
        await restored.close()


async def test_existing_schema_version_does_not_advance_without_fts5(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "no-fts.db"
    original = SessionDB(path)
    await original.session_count()
    connection = await original._get_connection()
    await _execute(connection, "UPDATE schema_version SET version = 24")
    await original.close()

    restored = SessionDB(path)

    async def _no_fts(_cursor):
        return False

    monkeypatch.setattr(restored, "_sqlite_supports_fts5", _no_fts)
    try:
        connection = await restored._get_connection()
        assert (
            await _fetchone(connection, "SELECT version FROM schema_version")
        )[0] == 24
        assert restored._fts_enabled is False
    finally:
        await restored.close()


async def test_legacy_fts_sets_optimize_marker_and_advances_main_schema(tmp_path):
    path = tmp_path / "legacy-fts.db"
    original = SessionDB(path)
    connection = await original._get_connection()
    if not original._fts_enabled:
        await original.close()
        pytest.skip("SQLite build has no FTS5")
    await original._drop_fts_triggers(connection)
    script_cursor = await connection.executescript(
        """
        DROP TABLE IF EXISTS messages_fts;
        DROP TABLE IF EXISTS messages_fts_trigram;
        DROP VIEW IF EXISTS messages_fts_trigram_src;
        CREATE VIRTUAL TABLE messages_fts USING fts5(content);
        """
    )
    await script_cursor.close()
    await _execute(connection, "UPDATE schema_version SET version = 22")
    await _execute(
        connection,
        "DELETE FROM state_meta WHERE key IN "
        "('fts_optimize_available', 'fts_storage_version')",
    )
    await original.close()

    restored = SessionDB(path)
    try:
        connection = await restored._get_connection()
        assert (
            await _fetchone(
                connection,
                "SELECT value FROM state_meta "
                "WHERE key = 'fts_optimize_available'",
            )
        )[0] == "1"
        assert (
            await _fetchone(connection, "SELECT version FROM schema_version")
        )[0] == SCHEMA_VERSION
        assert await restored.fts_optimize_available()
    finally:
        await restored.close()


async def test_external_fts_layout_is_stamped_on_reopen(tmp_path):
    path = tmp_path / "external-fts.db"
    first = SessionDB(path)
    await first.session_count()
    if not first._fts_enabled:
        await first.close()
        pytest.skip("SQLite build has no FTS5")
    await first.close()

    second = SessionDB(path)
    blocker = BlockBuster()
    async with no_task_leaks(action=LeakAction.RAISE):
        blocker.activate()
        try:
            connection = await second._get_connection()
            marker = await _fetchone(
                connection,
                "SELECT value FROM state_meta "
                "WHERE key = 'fts_storage_version'",
            )
            assert marker[0] == str(FTS_STORAGE_VERSION)
        finally:
            await second.close()
            blocker.deactivate()


async def test_cancelled_schema_initialization_cleans_up_and_can_retry(
    tmp_path,
    monkeypatch,
):
    database = SessionDB(tmp_path / "cancelled-init.db")
    reconcile_started = asyncio.Event()
    release_reconcile = asyncio.Event()
    original_reconcile = database._reconcile_columns

    async def _stalled_reconcile(cursor):
        reconcile_started.set()
        await release_reconcile.wait()
        await original_reconcile(cursor)

    monkeypatch.setattr(database, "_reconcile_columns", _stalled_reconcile)
    async with no_task_leaks(action=LeakAction.RAISE):
        task = asyncio.create_task(database.session_count())
        await asyncio.wait_for(reconcile_started.wait(), timeout=1.0)
        task.cancel()
        release_reconcile.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert database._connection is None
    assert database._initializing_connection is None
    assert database._schema_ready is False

    monkeypatch.setattr(database, "_reconcile_columns", original_reconcile)
    try:
        assert await database.session_count() == 0
        assert database._schema_ready is True
    finally:
        await database.close()


async def test_schema_migration_module_preserves_coroutine_api_shape():
    assert issubclass(SessionDB, SessionSchemaMixin)
    for name in (
        "_dedupe_legacy_system_prompts",
        "_reconcile_columns",
        "_init_schema",
    ):
        assert inspect.iscoroutinefunction(getattr(SessionDB, name))
