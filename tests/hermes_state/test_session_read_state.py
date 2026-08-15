import asyncio
import time

import pytest
import pytest_asyncio

from hermes_state import SessionDB


@pytest_asyncio.fixture
async def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        await database.close()


async def _last_read(db, sid):
    row = await db.get_session(sid)
    return row.get("last_read_at") if row is not None else None


async def _row(db, sid):
    rows = await db.list_sessions_rich(include_archived=True)
    return next(s for s in rows if s["id"] == sid)


@pytest.mark.asyncio
async def test_untracked_sessions_are_read(db):
    """NULL watermark = never tracked = read, so shipping the column doesn't
    badge a user's entire pre-feature history at once."""
    await db.create_session(session_id="s1", source="cli")
    await db.append_message(session_id="s1", role="user", content="hi")

    assert await _last_read(db, "s1") is None
    assert (await _row(db, "s1"))["unread"] is False


@pytest.mark.asyncio
async def test_mark_read_then_new_activity_flips_back_to_unread(db):
    await db.create_session(session_id="s1", source="cli")
    await db.append_message(session_id="s1", role="user", content="hi")

    assert await db.set_session_read("s1") is True
    assert (await _row(db, "s1"))["unread"] is False

    # New activity postdating the watermark makes it unread again without
    # any write on the message path.
    await asyncio.sleep(0.01)
    await db.append_message(session_id="s1", role="assistant", content="reply")
    assert (await _row(db, "s1"))["unread"] is True


@pytest.mark.asyncio
async def test_mark_unread_explicitly(db):
    await db.create_session(session_id="s1", source="cli")
    await db.append_message(session_id="s1", role="user", content="hi")
    await db.set_session_read("s1")

    assert await db.set_session_read("s1", read=False) is True
    assert await _last_read(db, "s1") == 0.0
    assert (await _row(db, "s1"))["unread"] is True


@pytest.mark.asyncio
async def test_missing_session_returns_false(db):
    assert await db.set_session_read("nope") is False


async def _compression_pair(db: SessionDB):
    base = time.time() - 100
    await db.create_session("root", source="cli")
    await db.create_session("tip", source="cli", parent_session_id="root")

    async def _update(conn):
        await conn.execute(
            "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression', message_count = 1 WHERE id = 'root'",
            (base, base + 10),
        )
        await conn.execute(
            "UPDATE sessions SET started_at = ?, message_count = 1 WHERE id = 'tip'",
            (base + 20,),
        )

    await db._execute_write(_update)


@pytest.mark.asyncio
async def test_reading_compression_tip_stamps_whole_lineage(db):
    await _compression_pair(db)

    assert await db.set_session_read("tip") is True

    root_read = await _last_read(db, "root")
    assert root_read is not None and root_read > 0
    assert root_read == await _last_read(db, "tip")

    # The projected conversation row (root surfaced as tip) derives read.
    rows = await db.list_sessions_rich(order_by_last_active=True)
    assert [s["id"] for s in rows] == ["tip"]
    assert rows[0]["unread"] is False


@pytest.mark.asyncio
async def test_marking_root_unread_marks_projected_conversation(db):
    await _compression_pair(db)
    await db.set_session_read("tip")

    assert await db.set_session_read("root", read=False) is True

    rows = await db.list_sessions_rich(order_by_last_active=True)
    assert [s["id"] for s in rows] == ["tip"]
    assert rows[0]["unread"] is True
