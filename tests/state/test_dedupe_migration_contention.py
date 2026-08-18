"""v25 system-prompt dedupe resumes safely after SQLite contention."""

import sqlite3

import aiosqlite
import pytest

from hermes_state import SessionDB


async def _legacy_db(tmp_path: object, rows: int = 5):
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    await db.create_session("bootstrap", "test")
    await db.close()
    async with aiosqlite.connect(path) as conn:
        for index in range(rows):
            await conn.execute(
                "INSERT OR IGNORE INTO sessions (id, source, started_at) "
                "VALUES (?, 'test', 1.0)",
                (f"sess-{index}",),
            )
            await conn.execute(
                "UPDATE sessions SET system_prompt = ?, "
                "system_prompt_hash = NULL WHERE id = ?",
                (f"legacy prompt {index}", f"sess-{index}"),
            )
        await conn.commit()
    return path


@pytest.mark.asyncio
async def test_mid_loop_lock_error_keeps_partial_rows_readable(tmp_path, monkeypatch):
    path = await _legacy_db(tmp_path)
    db = SessionDB(db_path=path)
    calls = 0
    original = db._store_system_prompt

    async def fail_after_two(connection, prompt):
        nonlocal calls
        if calls >= 2:
            raise sqlite3.OperationalError("database is locked")
        calls += 1
        return await original(connection, prompt)

    monkeypatch.setattr(db, "_store_system_prompt", fail_after_two)
    async with aiosqlite.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        await db._dedupe_legacy_system_prompts(conn)
        await conn.commit()
        async with conn.execute(
            "SELECT system_prompt, system_prompt_hash FROM sessions "
            "WHERE id LIKE 'sess-%' ORDER BY id"
        ) as cursor:
            rows = await cursor.fetchall()
    await db.close()
    assert any(row["system_prompt"] is None for row in rows)
    assert any(row["system_prompt"] is not None for row in rows)


@pytest.mark.asyncio
async def test_later_dedupe_run_completes_remainder(tmp_path, monkeypatch):
    path = await _legacy_db(tmp_path)
    db = SessionDB(db_path=path)
    calls = 0
    original = db._store_system_prompt

    async def fail_after_two(connection, prompt):
        nonlocal calls
        if calls >= 2:
            raise sqlite3.OperationalError("database is locked")
        calls += 1
        return await original(connection, prompt)

    monkeypatch.setattr(db, "_store_system_prompt", fail_after_two)
    async with aiosqlite.connect(path) as conn:
        await db._dedupe_legacy_system_prompts(conn)
        await conn.commit()
        monkeypatch.setattr(db, "_store_system_prompt", original)
        await db._dedupe_legacy_system_prompts(conn)
        await conn.commit()
        async with conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE id LIKE 'sess-%' "
            "AND system_prompt IS NOT NULL"
        ) as cursor:
            remaining = (await cursor.fetchone())[0]
    await db.close()
    assert remaining == 0
