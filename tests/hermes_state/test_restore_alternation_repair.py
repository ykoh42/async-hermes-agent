"""get_messages_as_conversation(repair_alternation=True) — heal durable
alternation violations at the restore boundary.

A turn that persists a user row but no assistant row (e.g. its reply was
suppressed, or two concurrent turns interleaved their flushes) leaves a
``user;user`` pair in state.db. Without repair at restore, the defensive
pre-request ``repair_message_sequence`` re-fires on EVERY request for the
rest of the session's life, because it mutates only the per-request list.

Default (``repair_alternation=False``) must stay verbatim: inspection and
export consumers (trace upload, context guard) read the transcript as-is.
"""

import pytest
import pytest_asyncio

from hermes_state import SessionDB


@pytest_asyncio.fixture()
async def db(tmp_path):
    db_path = tmp_path / "test_state.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    await session_db.close()


async def _seed_wedged_session(db, session_id="s1"):
    """assistant → user → user (no assistant row between): the durable wedge."""
    await db.create_session(session_id, "system prompt")
    await db.append_message(session_id=session_id, role="user", content="first ask")
    await db.append_message(session_id=session_id, role="assistant", content="first reply")
    await db.append_message(session_id=session_id, role="user", content="unanswered turn")
    await db.append_message(session_id=session_id, role="user", content="next turn")
    await db.append_message(session_id=session_id, role="assistant", content="next reply")




@pytest.mark.asyncio
async def test_repair_alternation_merges_user_pair(db):
    await _seed_wedged_session(db)
    messages = await db.get_messages_as_conversation("s1", repair_alternation=True)
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    # Both user texts survive, merged in order — no user input is lost.
    merged = messages[2]["content"]
    assert "unanswered turn" in merged and "next turn" in merged
    assert merged.index("unanswered turn") < merged.index("next turn")


@pytest.mark.asyncio
async def test_repaired_load_is_stable_under_prerequest_repair(db):
    """The restored list must yield ZERO further repairs — this is the whole
    point: the pre-request defensive repair stops firing every turn."""
    from agent.agent_runtime_helpers import repair_message_sequence

    await _seed_wedged_session(db)
    messages = await db.get_messages_as_conversation("s1", repair_alternation=True)
    assert repair_message_sequence(None, messages) == 0
