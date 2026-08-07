"""Native-async runtime recovery from corrupt FTS shadow tables."""

import sqlite3

import aiosqlite
import pytest
import pytest_asyncio
from blockbuster import BlockBuster
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

from hermes_state import SessionDB


@pytest_asyncio.fixture
async def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    yield database
    await database.close()


async def _corrupt_fts(db_path, table="messages_fts"):
    connection = await aiosqlite.connect(db_path)
    try:
        await connection.execute(
            f"UPDATE {table}_data "
            "SET block = X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'"
        )
        await connection.commit()
    finally:
        await connection.close()


async def _message_contents(db_path):
    connection = await aiosqlite.connect(db_path)
    try:
        cursor = await connection.execute(
            "SELECT content FROM messages ORDER BY id"
        )
        try:
            return [row[0] for row in await cursor.fetchall()]
        finally:
            await cursor.close()
    finally:
        await connection.close()


def test_corruption_error_classification_covers_sqlite_variants():
    assert SessionDB._is_fts_write_corruption_error(
        sqlite3.DatabaseError("database disk image is malformed")
    )
    assert SessionDB._is_fts_write_corruption_error(
        sqlite3.DatabaseError(
            'fts5: corrupt structure record for table "messages_fts"'
        )
    )
    assert not SessionDB._is_fts_write_corruption_error(
        sqlite3.DatabaseError("no such table: nothing_fts_related")
    )


@pytest.mark.asyncio
async def test_append_self_heals_without_blocking_or_leaking(db, tmp_path):
    await db.create_session("s1", source="test")
    if not db._fts_enabled:
        pytest.skip("FTS5 unavailable in this build")
    await db.append_message("s1", "user", "hello world")
    await _corrupt_fts(tmp_path / "state.db")

    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        blockbuster = BlockBuster()
        blockbuster.activate()
        try:
            message_id = await db.append_message(
                "s1", "user", "healed append"
            )
        finally:
            blockbuster.deactivate()

    assert message_id is not None
    assert await _message_contents(tmp_path / "state.db") == [
        "hello world",
        "healed append",
    ]


@pytest.mark.asyncio
async def test_base_search_self_heals_without_blocking_or_leaking(db, tmp_path):
    await db.create_session("s1", source="test")
    if not db._fts_enabled:
        pytest.skip("FTS5 unavailable in this build")
    await db.append_message("s1", "user", "a searchable needle here")
    await _corrupt_fts(tmp_path / "state.db")

    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        blockbuster = BlockBuster()
        blockbuster.activate()
        try:
            results = await db.search_messages("needle")
        finally:
            blockbuster.deactivate()

    assert db._fts_runtime_rebuild_attempted is True
    assert any("needle" in (row.get("snippet") or "") for row in results)


@pytest.mark.asyncio
async def test_trigram_search_self_heals(db, tmp_path):
    await db.create_session("s1", source="test")
    if not db._fts_enabled:
        pytest.skip("FTS5 unavailable in this build")
    if not db._trigram_available:
        pytest.skip("trigram tokenizer unavailable in this build")
    await db.append_message("s1", "user", "关于大别山项目的进展报告")
    await _corrupt_fts(tmp_path / "state.db", "messages_fts_trigram")

    results = await db.search_messages("大别山项目")

    assert db._fts_runtime_rebuild_attempted is True
    assert any(">>>" in (row.get("snippet") or "") for row in results)


@pytest.mark.asyncio
async def test_runtime_rebuild_is_one_shot_per_instance(db, tmp_path):
    await db.create_session("s1", source="test")
    if not db._fts_enabled:
        pytest.skip("FTS5 unavailable in this build")
    await db.append_message("s1", "user", "seed")
    await _corrupt_fts(tmp_path / "state.db")
    await db.append_message("s1", "user", "first heal")
    assert db._fts_runtime_rebuild_attempted is True

    await _corrupt_fts(tmp_path / "state.db")
    with pytest.raises(sqlite3.DatabaseError):
        await db.append_message("s1", "user", "second corruption")
