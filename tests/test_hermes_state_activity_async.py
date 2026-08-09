"""Native-async SessionDB activity projection parity tests."""

import pytest

from agent.session_activity import ActivityProvenance
from hermes_state import SessionDB


@pytest.mark.asyncio
async def test_session_activity_never_moves_backwards(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        await db.create_session("s1", "cli")
        heartbeat = 1_700_000_500.0
        await db.touch_session_activity(
            "s1",
            heartbeat,
            description="starting API call #1",
            provenance=ActivityProvenance.UNKNOWN,
        )

        row = await db.get_session("s1")
        assert row["last_activity_at"] == heartbeat
        assert row["last_activity_description"] == "starting API call #1"
        assert row["last_activity_provenance"] == "unknown"

        activity = await db.get_session_activity("s1")
        assert activity["last_activity_at"] == heartbeat
        assert activity["last_activity_description"] == "starting API call #1"
        assert activity["last_activity_provenance"] == "unknown"
        sessions = await db.list_sessions_rich()
        assert sessions[0]["last_active"] == heartbeat

        await db.touch_session_activity(
            "s1",
            heartbeat - 100,
            description="ignored",
        )
        row = await db.get_session("s1")
        assert row["last_activity_at"] == heartbeat
        assert row["last_activity_description"] == "starting API call #1"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_clear_session_activity_labels_keeps_timestamp(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        await db.create_session("s1", "cli")
        heartbeat = 1_700_000_500.0
        await db.touch_session_activity(
            "s1",
            heartbeat,
            description="compressing context",
            provenance=ActivityProvenance.AGENT_COMPRESSION,
        )

        await db.clear_session_activity_labels("s1")

        row = await db.get_session("s1")
        assert row["last_activity_at"] == heartbeat
        assert row["last_activity_description"] == ""
        assert row["last_activity_provenance"] == "unknown"
        activity = await db.get_session_activity("s1")
        assert activity["last_activity_at"] == heartbeat
        assert activity["last_activity_description"] == ""
        assert activity["last_activity_provenance"] == "unknown"
    finally:
        await db.close()
