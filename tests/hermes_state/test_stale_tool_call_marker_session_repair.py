"""Async port of the upstream stale tool-call marker repair tests."""

from __future__ import annotations

import pytest

from hermes_state import (
    SessionDB,
    _is_stale_tool_call_marker_message,
    _strip_stale_tool_call_markers,
)


def test_matches_bare_marker_with_tool_calls():
    assert _is_stale_tool_call_marker_message(
        {
            "role": "assistant",
            "content": "[memory]",
            "tool_calls": [{"id": "1", "function": {"name": "skill_manage"}}],
        }
    ) is True


def test_matches_dotted_marker():
    assert _is_stale_tool_call_marker_message(
        {
            "role": "assistant",
            "content": "[foo.bar]",
            "tool_calls": [{"id": "1", "function": {"name": "foo.bar"}}],
        }
    ) is True


def test_ignores_marker_without_tool_calls():
    assert _is_stale_tool_call_marker_message(
        {"role": "assistant", "content": "[memory]"}
    ) is False


def test_ignores_real_content_with_tool_calls():
    assert _is_stale_tool_call_marker_message(
        {
            "role": "assistant",
            "content": "I'll check that for you.",
            "tool_calls": [{"id": "1"}],
        }
    ) is False


def test_ignores_user_role():
    assert _is_stale_tool_call_marker_message(
        {"role": "user", "content": "[memory]", "tool_calls": [{"id": "1"}]}
    ) is False


def test_clears_contaminated_content_keeps_tool_calls():
    calls = [{"id": "1", "function": {"name": "skill_manage", "arguments": "{}"}}]
    messages = [
        {"role": "user", "content": "do the full task"},
        {"role": "assistant", "content": "[memory]", "tool_calls": calls},
        {"role": "tool", "content": "ok", "tool_call_id": "1"},
    ]
    out = _strip_stale_tool_call_markers(messages)
    assert out[1]["content"] == ""
    assert out[1]["tool_calls"] == calls


def test_unaffected_session_passes_through_unchanged():
    messages = [
        {"role": "user", "content": "What's the weather?"},
        {"role": "assistant", "content": "It's sunny."},
    ]
    assert _strip_stale_tool_call_markers(messages) == messages


@pytest.mark.asyncio
async def test_polluted_session_resumes_without_marker(tmp_path):
    db = SessionDB(db_path=tmp_path / "t.db")
    try:
        await db.create_session(session_id="s1", source="library")
        await db.append_message("s1", role="user", content="do the full task")
        await db.append_message(
            "s1",
            role="assistant",
            content="[memory]",
            tool_calls=[{"id": "1", "function": {"name": "skill_manage", "arguments": "{}"}}],
        )
        await db.append_message("s1", role="tool", content="ok", tool_call_id="1")
        await db.append_message("s1", role="assistant", content="Here is the result.")

        conv = await db.get_messages_as_conversation("s1")
        contents = [m.get("content") for m in conv if m.get("role") == "assistant"]
        assert "[memory]" not in contents
        assert "Here is the result." in contents
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_clean_session_resumes_unaffected(tmp_path):
    db = SessionDB(db_path=tmp_path / "t.db")
    try:
        await db.create_session(session_id="s1", source="library")
        await db.append_message("s1", role="user", content="What's the weather?")
        await db.append_message("s1", role="assistant", content="It's sunny.")
        conv = await db.get_messages_as_conversation("s1")
        assert [m.get("content") for m in conv] == ["What's the weather?", "It's sunny."]
    finally:
        await db.close()
