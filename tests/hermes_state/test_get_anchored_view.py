"""Tests for SessionDB.get_anchored_view — anchored window + session bookends.

Used by the discovery shape of session_search: an FTS5 match becomes the
anchor, the call returns goal (bookend_start) + match (window) + resolution
(bookend_end) in a single round trip, no LLM.
"""
import pytest
import pytest_asyncio

from hermes_state import SessionDB


@pytest_asyncio.fixture
async def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    yield database
    await database.close()


async def _seed_long_session(db, sid="s1", n=30):
    """Create a long session with alternating user/assistant prose. Returns ids ascending."""
    await db.create_session(sid, source="cli")
    ids = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        mid = await db.append_message(sid, role=role, content=f"prose msg {i}")
        ids.append(mid)
    return ids


class TestWindowAndBookendShape:
    @pytest.mark.asyncio
    async def test_returns_window_with_bookend_start_and_end(self, db):
        ids = await _seed_long_session(db, n=30)
        # Anchor mid-session
        anchor = ids[15]
        view = await db.get_anchored_view("s1", anchor, window=3, bookend=3)
        assert len(view["window"]) == 7  # ±3 + anchor
        assert len(view["bookend_start"]) == 3
        assert len(view["bookend_end"]) == 3
        # bookend_start is the first 3 ids of the session
        assert [m["id"] for m in view["bookend_start"]] == ids[:3]
        # bookend_end is the last 3 ids of the session
        assert [m["id"] for m in view["bookend_end"]] == ids[-3:]

    @pytest.mark.asyncio
    async def test_window_anchor_marked_correctly(self, db):
        ids = await _seed_long_session(db, n=20)
        anchor = ids[10]
        view = await db.get_anchored_view("s1", anchor, window=2, bookend=3)
        # Anchor message is present in the window
        anchor_msgs = [m for m in view["window"] if m["id"] == anchor]
        assert len(anchor_msgs) == 1




class TestRoleFiltering:
    @pytest.mark.asyncio
    async def test_tool_role_filtered_from_window(self, db):
        await db.create_session("s1", source="cli")
        user_ids = []
        for i in range(5):
            user_ids.append(await db.append_message("s1", role="user", content=f"u{i}"))
            await db.append_message("s1", role="tool", content=f"tool output {i}", tool_name="x")
        # Anchor on user message
        view = await db.get_anchored_view("s1", user_ids[2], window=5, bookend=0)
        # No tool messages should appear in the window
        roles = [m.get("role") for m in view["window"]]
        assert "tool" not in roles

    @pytest.mark.asyncio
    async def test_anchor_preserved_even_when_tool_role(self, db):
        await db.create_session("s1", source="cli")
        await db.append_message("s1", role="user", content="ask")
        tool_id = await db.append_message("s1", role="tool", content="tool output", tool_name="x")
        await db.append_message("s1", role="user", content="follow-up")
        # Anchor on the tool message — should still appear despite default filter
        view = await db.get_anchored_view("s1", tool_id, window=5, bookend=0)
        ids_in_window = [m["id"] for m in view["window"]]
        assert tool_id in ids_in_window







class TestSessionIsolation:
    """Bookends must not cross session boundaries."""

    @pytest.mark.asyncio
    async def test_bookends_only_from_anchor_session(self, db):
        ids1 = await _seed_long_session(db, sid="s1", n=20)
        await _seed_long_session(db, sid="s2", n=20)
        view = await db.get_anchored_view("s1", ids1[10], window=2, bookend=3)
        # All bookend messages should have session_id = s1 (or session_id col)
        for m in view["bookend_start"] + view["bookend_end"]:
            assert m.get("session_id") == "s1"
