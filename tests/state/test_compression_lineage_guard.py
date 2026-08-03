"""Native-async compression handoff and lineage guards."""

import sqlite3

import pytest
import pytest_asyncio

from hermes_state import SessionDB


@pytest_asyncio.fixture
async def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    yield database
    await database.close()


async def _closed_parent(db, session_id="parent"):
    await db.create_session(session_id, source="library")
    await db.append_message(session_id, "user", "before split")
    await db.end_session(session_id, "compression")


@pytest.mark.asyncio
async def test_find_live_compression_child_requires_one_canonical_child(db):
    await _closed_parent(db)
    await db.create_session("child", source="library", parent_session_id="parent")
    child = await db.find_live_compression_child("parent")
    assert child is not None and child["id"] == "child"

    await db.create_session("ambiguous", source="library", parent_session_id="parent")
    assert await db.find_live_compression_child("parent") is None


@pytest.mark.asyncio
async def test_non_continuation_children_do_not_make_lineage_ambiguous(db):
    await _closed_parent(db)
    await db.create_session("canonical", source="library", parent_session_id="parent")
    await db.create_session(
        "branch",
        source="library",
        parent_session_id="parent",
        model_config={"_branched_from": "parent"},
    )
    await db.create_session(
        "delegate",
        source="library",
        parent_session_id="parent",
        model_config={"_delegate_from": "parent"},
    )
    await db.create_session("tool-child", source="tool", parent_session_id="parent")
    child = await db.find_live_compression_child("parent")
    assert child is not None and child["id"] == "canonical"


@pytest.mark.asyncio
async def test_publish_compression_child_is_atomic_on_insert_failure(db):
    await db.create_session("parent", source="library")
    await db.append_message("parent", "user", "original")
    await db.create_session("child", source="library")
    assert await db.try_acquire_compression_lock("parent", "winner", ttl_seconds=60)

    with pytest.raises(sqlite3.IntegrityError):
        await db.publish_compression_child(
            parent_session_id="parent",
            child_session_id="child",
            source="library",
            messages=[{"role": "user", "content": "summary"}],
            compression_lock_holder="winner",
        )

    assert (await db.get_session("parent"))["ended_at"] is None
    assert [m["content"] for m in await db.get_messages("parent")] == ["original"]


@pytest.mark.asyncio
async def test_publish_exposes_only_complete_child(db):
    await db.create_session("parent", source="library")
    await db.append_message("parent", "user", "original")
    assert await db.try_acquire_compression_lock("parent", "winner", ttl_seconds=60)
    await db.publish_compression_child(
        parent_session_id="parent",
        child_session_id="child",
        source="library",
        system_prompt="compressed system",
        messages=[{"role": "user", "content": "summary"}],
        compression_lock_holder="winner",
    )

    assert (await db.get_session("parent"))["end_reason"] == "compression"
    child = await db.find_live_compression_child("parent")
    assert child is not None and child["system_prompt"] == "compressed system"
    assert [m["content"] for m in await db.get_messages("child")] == ["summary"]


@pytest.mark.asyncio
async def test_publish_rejects_lost_lease_without_mutation(db):
    await db.create_session("parent", source="library")
    await db.append_message("parent", "user", "durable")
    assert await db.try_acquire_compression_lock("parent", "winner", ttl_seconds=60)
    with pytest.raises(RuntimeError, match="lease lost"):
        await db.publish_compression_child(
            parent_session_id="parent",
            child_session_id="stale",
            source="library",
            messages=[{"role": "user", "content": "stale"}],
            compression_lock_holder="loser",
        )
    assert (await db.get_session("parent"))["ended_at"] is None
    assert await db.get_session("stale") is None


@pytest.mark.asyncio
async def test_lease_blocks_non_owner_but_allows_owner_flush(db):
    await db.create_session("leased", source="library")
    assert await db.try_acquire_compression_lock("leased", "winner", ttl_seconds=60)
    with pytest.raises(RuntimeError, match="being compressed"):
        await db.append_message("leased", "user", "stale")
    await db.append_message(
        "leased",
        "assistant",
        "winner flush",
        compression_lock_holder="winner",
    )
    assert [m["content"] for m in await db.get_messages("leased")] == ["winner flush"]
