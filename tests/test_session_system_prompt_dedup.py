"""Behavior coverage for content-addressed session system prompts."""

from __future__ import annotations

import json
import time

import pytest
import pytest_asyncio

from hermes_state import SCHEMA_VERSION, SessionDB


@pytest_asyncio.fixture()
async def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    await session_db.close()


async def _prompt_count(db: SessionDB) -> int:
    connection = await db._get_connection()
    row = await (await connection.execute(
        "SELECT COUNT(*) FROM system_prompts"
    )).fetchone()
    return int(row[0])


@pytest.mark.asyncio
async def test_prompt_snapshots_are_deduplicated_and_hydrated_for_readers(db):
    prompt = "You are Hermes.\n" + ("Follow the profile policy.\n" * 5)
    await db.create_session(
        "s1",
        "telegram",
        session_key="agent:main:telegram:dm:c1",
        chat_id="c1",
        chat_type="dm",
        system_prompt=prompt,
    )
    await db.create_session("s2", "cli", system_prompt=prompt)

    connection = await db._get_connection()
    stored = await (await connection.execute(
        "SELECT hash, prompt FROM system_prompts"
    )).fetchall()
    assert len(stored) == 1
    assert stored[0]["prompt"] == prompt
    raw_sessions = await (await connection.execute(
        "SELECT system_prompt, system_prompt_hash FROM sessions ORDER BY id"
    )).fetchall()
    assert [row["system_prompt"] for row in raw_sessions] == [None, None]
    assert {row["system_prompt_hash"] for row in raw_sessions} == {
        stored[0]["hash"]
    }

    assert (await db.get_session("s1"))["system_prompt"] == prompt
    assert all(
        row["system_prompt"] == prompt for row in await db.list_sessions_rich()
    )


@pytest.mark.asyncio
async def test_prompt_replacement_and_route_changes_collect_only_orphans(db):
    shared_prompt = "Model: x-ai/grok-4.5\nProvider: nous"
    await db.create_session(
        "s1",
        "hermes_browser",
        model="x-ai/grok-4.5",
        model_config={"_branched_from": "parent"},
        system_prompt=shared_prompt,
    )
    await db.create_session("s2", "cli", system_prompt=shared_prompt)

    await db.update_session_runtime_lock(
        "s1",
        model="anthropic/claude-opus-4.8",
        provider="anthropic",
        confirmed=True,
    )
    s1 = await db.get_session("s1")
    assert s1["system_prompt"] is None
    assert json.loads(s1["model_config"])["_branched_from"] == "parent"
    assert (await db.get_session("s2"))["system_prompt"] == shared_prompt
    assert await _prompt_count(db) == 1

    await db.update_session_billing_route(
        "s2",
        provider="openrouter",
        base_url="https://example.test/v1",
    )
    assert (await db.get_session("s2"))["system_prompt"] is None
    assert await _prompt_count(db) == 0

    await db.update_system_prompt("s2", "replacement")
    assert (await db.get_session("s2"))["system_prompt"] == "replacement"
    await db.update_system_prompt("s2", None)
    assert await _prompt_count(db) == 0


@pytest.mark.asyncio
async def test_existing_session_enrichment_does_not_leak_unused_prompt(db):
    await db.create_session("s1", "cli", system_prompt="original prompt")
    await db.create_session("s1", "cli", system_prompt="unused prompt")

    connection = await db._get_connection()
    prompts = [
        row["prompt"]
        for row in await (await connection.execute(
            "SELECT prompt FROM system_prompts"
        )).fetchall()
    ]
    assert prompts == ["original prompt"]
    assert (await db.get_session("s1"))["system_prompt"] == "original prompt"


@pytest.mark.asyncio
async def test_every_session_deletion_path_reclaims_final_prompt_reference(db):
    async def seed(session_id: str, *, source: str = "cli") -> None:
        await db.create_session(
            session_id,
            source,
            system_prompt=f"unique prompt for {session_id}",
        )
        assert await _prompt_count(db) == 1

    await seed("single-empty")
    assert await db.delete_session_if_empty("single-empty") is True
    assert await _prompt_count(db) == 0

    await seed("bulk")
    assert await db.delete_sessions(["bulk"]) == 1
    assert await _prompt_count(db) == 0

    await seed("ended-empty")
    await db.end_session("ended-empty", "user_exit")
    assert await db.delete_empty_sessions() == 1
    assert await _prompt_count(db) == 0

    await seed("pruned")
    await db.end_session("pruned", "user_exit")
    assert await db.prune_sessions(
        older_than_days=None,
        started_before=time.time() + 1,
    ) == 1
    assert await _prompt_count(db) == 0

    await seed("ghost", source="tui")
    await db.end_session("ghost", "user_exit")
    connection = await db._get_connection()
    await connection.execute("UPDATE sessions SET started_at = 0 WHERE id = 'ghost'")
    await connection.commit()
    assert await db.prune_empty_ghost_sessions() == 1
    assert await _prompt_count(db) == 0


@pytest.mark.asyncio
async def test_deleting_one_shared_session_preserves_prompt_until_final_reference(db):
    prompt = "shared deletion prompt"
    await db.create_session("s1", "cli", system_prompt=prompt)
    await db.create_session("s2", "cli", system_prompt=prompt)

    assert await db.delete_session("s1") is True
    assert await _prompt_count(db) == 1
    assert (await db.get_session("s2"))["system_prompt"] == prompt

    assert await db.delete_session("s2") is True
    assert await _prompt_count(db) == 0


@pytest.mark.asyncio
async def test_compression_child_uses_content_addressed_prompt(db):
    prompt = "compressed child prompt"
    await db.create_session("parent", "webui")
    await db.append_message("parent", "user", "original")
    assert await db.try_acquire_compression_lock(
        "parent", "holder", ttl_seconds=60
    )

    await db.publish_compression_child(
        parent_session_id="parent",
        child_session_id="child",
        source="webui",
        system_prompt=prompt,
        messages=[{"role": "user", "content": "summary"}],
        compression_lock_holder="holder",
    )

    connection = await db._get_connection()
    raw = await (await connection.execute(
        "SELECT system_prompt, system_prompt_hash FROM sessions WHERE id = 'child'"
    )).fetchone()
    assert raw["system_prompt"] is None
    assert raw["system_prompt_hash"] is not None
    assert (await db.get_session("child"))["system_prompt"] == prompt
    assert await _prompt_count(db) == 1


@pytest.mark.asyncio
async def test_v24_inline_prompts_migrate_once_to_content_addressed_storage(tmp_path):
    db_path = tmp_path / "legacy-prompts.db"
    legacy_prompt = "Legacy system prompt\n" + ("same policy\n" * 20)

    db = SessionDB(db_path=db_path)
    await db.create_session("s1", "cli")
    await db.create_session("s2", "telegram")
    connection = await db._get_connection()
    await connection.execute(
        "UPDATE sessions SET system_prompt = ?, system_prompt_hash = NULL",
        (legacy_prompt,),
    )
    await connection.execute("UPDATE schema_version SET version = 24")
    await connection.commit()
    await db.close()

    migrated = SessionDB(db_path=db_path)
    try:
        assert (await migrated.get_session("s1"))["system_prompt"] == legacy_prompt
        assert (await migrated.get_session("s2"))["system_prompt"] == legacy_prompt
        assert await _prompt_count(migrated) == 1
        migrated_connection = await migrated._get_connection()
        raw_sessions = await (await migrated_connection.execute(
            "SELECT system_prompt, system_prompt_hash FROM sessions ORDER BY id"
        )).fetchall()
        assert [row["system_prompt"] for row in raw_sessions] == [None, None]
        assert len({row["system_prompt_hash"] for row in raw_sessions}) == 1
        assert (await (await migrated_connection.execute(
            "SELECT version FROM schema_version LIMIT 1"
        )).fetchone())[0] == SCHEMA_VERSION
    finally:
        await migrated.close()


@pytest.mark.asyncio
async def test_compact_rows_omit_prompt_fields(db):
    await db.create_session("s1", "cli", system_prompt="never materialize me")
    rows = await db.list_sessions_rich(
        compact_rows=True,
        order_by_last_active=True,
    )

    assert rows[0]["id"] == "s1"
    assert "system_prompt" not in rows[0]
    assert "system_prompt_hash" not in rows[0]
