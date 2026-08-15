"""Recovery for legacy compression parents with no continuation child."""

from types import SimpleNamespace

import pytest

from agent.conversation_compression import recover_rotated_compression_session
from hermes_state import CompressionSessionClosedError, SessionDB


@pytest.mark.asyncio
async def test_recover_rotated_compression_session_reopens_legacy_orphan(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        await db.create_session("orphan", source="cli")
        await db.append_message("orphan", "user", "before compression")
        await db.end_session("orphan", "compression")
        agent = SimpleNamespace(_session_db=db, session_id="orphan")

        assert await recover_rotated_compression_session(agent) is None
        await db.append_message("orphan", "user", "after recovery")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_recover_rotated_compression_session_keeps_parent_closed_with_child(
    tmp_path,
):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        await db.create_session("parent", source="cli")
        await db.append_message("parent", "user", "before compression")
        await db.end_session("parent", "compression")
        await db.create_session("child", source="cli", parent_session_id="parent")
        agent = SimpleNamespace(_session_db=db, session_id="parent")

        assert await recover_rotated_compression_session(agent) is None
        try:
            await db.append_message("parent", "user", "must stay closed")
        except CompressionSessionClosedError:
            pass
        else:
            raise AssertionError("compression parent with child was reopened")
    finally:
        await db.close()
