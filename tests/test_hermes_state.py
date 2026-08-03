"""Behavior contracts for the native-async ``SessionDB`` interface."""

import inspect
import sqlite3

import pytest
import pytest_asyncio

from hermes_state import SessionDB


@pytest_asyncio.fixture
async def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    yield database
    await database.close()


def test_public_session_interface_is_async():
    for name in (
        "create_session",
        "append_message",
        "get_session",
        "get_messages",
        "get_messages_as_conversation",
        "search_messages",
        "close",
    ):
        assert inspect.iscoroutinefunction(getattr(SessionDB, name)), name


@pytest.mark.asyncio
async def test_session_lifecycle_and_transcript_round_trip(db):
    assert await db.create_session("s1", source="library", model="test-model") == "s1"
    user_id = await db.append_message(
        "s1",
        role="user",
        content=[{"type": "text", "text": "hello"}],
        display_kind="prompt",
        display_metadata={"request_id": "r1"},
        timestamp=123.5,
    )
    assistant_id = await db.append_message(
        "s1",
        role="assistant",
        content="answer",
        reasoning="reason",
        reasoning_details=[{"type": "summary", "text": "thought"}],
    )

    session = await db.get_session("s1")
    assert session["model"] == "test-model"
    assert session["message_count"] == 2
    messages = await db.get_messages("s1")
    assert [message["id"] for message in messages] == [user_id, assistant_id]
    assert messages[0]["content"] == [{"type": "text", "text": "hello"}]
    assert messages[0]["display_metadata"] == {"request_id": "r1"}
    assert messages[0]["timestamp"] == 123.5
    assert messages[1]["reasoning"] == "reason"

    await db.end_session("s1", "completed")
    assert (await db.get_session("s1"))["end_reason"] == "completed"


@pytest.mark.asyncio
async def test_conversation_replay_preserves_tool_order_and_reasoning(db):
    await db.create_session("s1", source="library")
    await db.append_message("s1", role="user", content="calculate")
    await db.append_message(
        "s1",
        role="assistant",
        content="",
        reasoning="use tool",
        tool_calls=[
            {"id": "call-1", "type": "function", "function": {"name": "calc", "arguments": "{}"}}
        ],
    )
    await db.append_message(
        "s1",
        role="tool",
        content="42",
        tool_name="calc",
        tool_call_id="call-1",
    )
    await db.append_message("s1", role="assistant", content="42")

    replay = await db.get_messages_as_conversation("s1")
    assert [message["role"] for message in replay] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert replay[1]["reasoning"] == "use tool"
    assert replay[1]["tool_calls"][0]["id"] == "call-1"
    assert replay[2]["tool_call_id"] == "call-1"


@pytest.mark.asyncio
async def test_titles_meta_search_and_recent_listing(db):
    await db.create_session("s1", source="library")
    await db.append_message("s1", role="user", content="remember cobalt project")
    assert await db.set_session_title("s1", "Cobalt") is True
    assert await db.get_session_title("s1") == "Cobalt"
    assert await db.resolve_session_by_title("Cobalt") == "s1"
    assert await db.get_next_title_in_lineage("Cobalt") == "Cobalt #2"

    await db.set_meta("checkpoint", "ready")
    assert await db.get_meta("checkpoint") == "ready"
    assert await db.delete_meta("checkpoint") is True
    assert await db.get_meta("checkpoint") is None

    sessions = await db.list_sessions_rich(source="library")
    assert [session["id"] for session in sessions] == ["s1"]
    assert "cobalt project" in sessions[0]["preview"].lower()
    matches = await db.search_messages("cobalt")
    assert [match["session_id"] for match in matches] == ["s1"]


@pytest.mark.asyncio
async def test_rewind_is_auditable_and_excluded_from_live_replay(db):
    await db.create_session("s1", source="library")
    await db.append_message("s1", role="user", content="first")
    await db.append_message("s1", role="assistant", content="reply")
    target = await db.append_message("s1", role="user", content="retry this")
    await db.append_message("s1", role="assistant", content="discarded")

    result = await db.rewind_to_message("s1", target)
    assert result["rewound_count"] == 2
    assert result["target_message"]["content"] == "retry this"
    assert [message["content"] for message in await db.get_messages("s1")] == [
        "first",
        "reply",
    ]
    assert len(await db.get_messages("s1", include_inactive=True)) == 4


@pytest.mark.asyncio
async def test_lone_surrogate_is_safely_persisted(db):
    await db.create_session("s1", source="library")
    await db.append_message("s1", role="user", content="bad\ud800text")
    messages = await db.get_messages("s1")
    assert messages[0]["content"] == "bad\ufffdtext"


@pytest.mark.asyncio
async def test_close_is_idempotent_and_rejects_new_io(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    await db.create_session("s1", source="library")
    await db.close()
    await db.close()
    with pytest.raises(RuntimeError, match="closed"):
        await db.get_session("s1")


@pytest.mark.asyncio
async def test_read_only_connection_uses_the_same_async_interface(tmp_path):
    path = tmp_path / "state.db"
    writer = SessionDB(path)
    await writer.create_session("s1", source="library")
    await writer.close()

    reader = SessionDB(path, read_only=True)
    try:
        assert (await reader.get_session("s1"))["id"] == "s1"
        with pytest.raises(sqlite3.OperationalError):
            await reader.set_meta("forbidden", "write")
    finally:
        await reader.close()
