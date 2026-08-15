"""Native-async parity tests for transcript pagination and safety bounds."""

import pytest

import hermes_state
from hermes_state import SessionDB


pytestmark = pytest.mark.asyncio


async def _seed(db: SessionDB, session_id: str = "s1", count: int = 10) -> None:
    await db.create_session(session_id, source="cli")
    await db.append_messages_batch(
        session_id,
        [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"msg-{index}",
            }
            for index in range(count)
        ],
    )


async def test_latest_pages_count_back_from_newest_but_remain_chronological(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        await _seed(db)
        page1 = await db.get_messages("s1", limit=4, offset=0, latest=True)
        page2 = await db.get_messages("s1", limit=4, offset=4, latest=True)
        page3 = await db.get_messages("s1", limit=4, offset=8, latest=True)

        assert [row["content"] for row in page1] == [
            "msg-6",
            "msg-7",
            "msg-8",
            "msg-9",
        ]
        assert [row["content"] for row in page2] == [
            "msg-2",
            "msg-3",
            "msg-4",
            "msg-5",
        ]
        assert [row["content"] for row in page3] == ["msg-0", "msg-1"]
    finally:
        await db.close()


async def test_resume_safety_counts_active_rows_across_lineage(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        await _seed(db, "root", 3)
        await db.end_session("root", "compression")
        await db.create_session("tip", source="cli", parent_session_id="root")
        await db.append_messages_batch(
            "tip",
            [{"role": "assistant", "content": f"tip-{i}"} for i in range(2)],
        )

        assert await db.get_resume_message_count("tip") == 5
        with pytest.raises(hermes_state.SessionResumeTooLargeError) as exc_info:
            await db.assert_resume_safe("tip", max_messages=4)
        assert exc_info.value.message_count == 5
        assert exc_info.value.limit == 4
    finally:
        await db.close()


async def test_export_safety_is_bounded_to_requested_active_segment(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        await _seed(db, "root", 3)
        await db.end_session("root", "compression")
        await db.create_session("tip", source="cli", parent_session_id="root")
        await db.append_messages_batch(
            "tip",
            [{"role": "assistant", "content": f"tip-{i}"} for i in range(2)],
        )

        assert await db.assert_export_safe("tip", max_messages=2) == 2
        with pytest.raises(hermes_state.SessionExportTooLargeError) as exc_info:
            await db.assert_export_safe("root", max_messages=2)
        assert exc_info.value.session_id == "root"
        assert exc_info.value.message_count == 3
        assert exc_info.value.limit == 2
    finally:
        await db.close()


async def test_transcript_safety_zero_disables_guard_and_returns_count(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        await _seed(db, count=3)
        assert await db.assert_resume_safe("s1", max_messages=0) == 3
        assert await db.assert_export_safe("s1", max_messages=0) == 3
    finally:
        await db.close()


async def test_resume_and_export_boundaries_use_resolved_limits(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        await _seed(db, count=3)

        async def _limit():
            return 2

        monkeypatch.setattr(hermes_state, "resolved_max_resume_messages", _limit)
        monkeypatch.setattr(hermes_state, "resolved_max_export_messages", _limit)

        with pytest.raises(hermes_state.SessionResumeTooLargeError):
            await db.get_resume_conversations("s1")
        with pytest.raises(hermes_state.SessionExportTooLargeError):
            await db.export_session("s1")
    finally:
        await db.close()
