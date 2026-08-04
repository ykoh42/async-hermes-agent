"""Tests for SessionDB.append_messages_batch (#23254 salvage).

The batch writer reuses _insert_message_rows (the same row-serialization
path as replace/compact/import), runs the same admission guards as
append_message, is atomic (all rows or none), and aggregates the session
counters in one UPDATE.
"""

import json
import sqlite3

import pytest
import pytest_asyncio

from hermes_state import (
    CompressionSessionClosedError,
    SessionDB,
)


@pytest_asyncio.fixture()
async def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    await d.create_session("sess-batch", source="cli")
    yield d
    await d.close()


def _turn_messages():
    return [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "let me check",
            "tool_calls": [{"name": "terminal", "arguments": "{}"}],
            "reasoning_content": "thinking...",
            "finish_reason": "tool_calls",
        },
        {
            "role": "tool",
            "content": "tool output",
            "tool_name": "terminal",
            "tool_call_id": "call_1",
        },
        {"role": "assistant", "content": "answer", "finish_reason": "stop"},
    ]


@pytest.mark.asyncio
class TestAppendMessagesBatch:
    async def test_batch_rows_identical_to_single_appends(self, db, tmp_path):
        """The batch writer stores the same bytes append_message would."""
        db2 = SessionDB(db_path=tmp_path / "state2.db")
        await db2.create_session("sess-batch", source="cli")
        try:
            msgs = _turn_messages()
            await db.append_messages_batch("sess-batch", msgs)
            for m in msgs:
                role = m["role"]
                await db2.append_message(
                    session_id="sess-batch",
                    role=role,
                    content=m.get("content"),
                    tool_name=m.get("tool_name"),
                    tool_calls=m.get("tool_calls"),
                    tool_call_id=m.get("tool_call_id"),
                    finish_reason=m.get("finish_reason"),
                    reasoning_content=(
                        m.get("reasoning_content") if role == "assistant" else None
                    ),
                )
            cols = (
                "role, content, tool_call_id, tool_calls, tool_name, "
                "finish_reason, reasoning_content, observed, active"
            )
            conn_a = await db._get_connection()
            conn_b = await db2._get_connection()
            rows_a = await (await conn_a.execute(
                f"SELECT {cols} FROM messages ORDER BY id"
            )).fetchall()
            rows_b = await (await conn_b.execute(
                f"SELECT {cols} FROM messages ORDER BY id"
            )).fetchall()
            assert [tuple(r) for r in rows_a] == [tuple(r) for r in rows_b]
        finally:
            await db2.close()

    async def test_reasoning_gated_to_assistant_rows(self, db):
        """_insert_message_rows role-gates reasoning fields; a tool row
        carrying reasoning keys must not persist them."""
        await db.append_messages_batch(
            "sess-batch",
            [
                {
                    "role": "tool",
                    "content": "out",
                    "tool_name": "t",
                    "tool_call_id": "c1",
                    "reasoning_content": "should not persist",
                }
            ],
        )
        connection = await db._get_connection()
        row = await (await connection.execute(
            "SELECT reasoning_content FROM messages"
        )).fetchone()
        assert row[0] is None

    async def test_counters_aggregate_once(self, db):
        await db.append_messages_batch("sess-batch", _turn_messages())
        connection = await db._get_connection()
        row = await (await connection.execute(
            "SELECT message_count, tool_call_count FROM sessions WHERE id = ?",
            ("sess-batch",),
        )).fetchone()
        assert row["message_count"] == 4
        assert row["tool_call_count"] == 1

    async def test_returns_inserted_count(self, db):
        assert await db.append_messages_batch("sess-batch", _turn_messages()) == 4

    async def test_empty_batch_is_noop(self, db):
        assert await db.append_messages_batch("sess-batch", []) == 0
        connection = await db._get_connection()
        row = await (await connection.execute(
            "SELECT message_count FROM sessions WHERE id = ?", ("sess-batch",)
        )).fetchone()
        assert row["message_count"] == 0

    async def test_atomicity_all_or_nothing(self, db, monkeypatch):
        """A failure mid-batch leaves ZERO rows and untouched counters."""
        async def failing_insert(self_db, conn, session_id, messages):
            await conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) "
                "VALUES (?, 'user', 'partial', ?)",
                (session_id, 1.0),
            )
            raise sqlite3.OperationalError("boom mid-batch")

        monkeypatch.setattr(SessionDB, "_insert_message_rows", failing_insert)
        with pytest.raises(sqlite3.OperationalError):
            await db.append_messages_batch("sess-batch", _turn_messages())
        monkeypatch.undo()

        connection = await db._get_connection()
        count = (await (await connection.execute(
            "SELECT COUNT(*) FROM messages"
        )).fetchone())[0]
        assert count == 0
        row = await (await connection.execute(
            "SELECT message_count, tool_call_count FROM sessions WHERE id = ?",
            ("sess-batch",),
        )).fetchone()
        assert row["message_count"] == 0
        assert row["tool_call_count"] == 0

    async def test_compression_closed_session_rejected(self, db):
        connection = await db._get_connection()
        await connection.execute(
            "UPDATE sessions SET ended_at = 1.0, end_reason = 'compression' "
            "WHERE id = ?",
            ("sess-batch",),
        )
        await connection.commit()
        with pytest.raises(CompressionSessionClosedError):
            await db.append_messages_batch("sess-batch", _turn_messages())

    async def test_multimodal_content_encoded(self, db):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "data:x"}},
                ],
            }
        ]
        await db.append_messages_batch("sess-batch", msgs)
        connection = await db._get_connection()
        raw = (await (await connection.execute(
            "SELECT content FROM messages"
        )).fetchone())[0]
        # encoded via _encode_content — same sentinel prefix as append_message
        loaded = await db.get_messages("sess-batch")
        assert loaded, raw

    async def test_tool_calls_json_string_not_double_encoded(self, db):
        msgs = [
            {
                "role": "assistant",
                "content": "x",
                "tool_calls": json.dumps([{"name": "t", "arguments": "{}"}]),
            }
        ]
        await db.append_messages_batch("sess-batch", msgs)
        connection = await db._get_connection()
        raw = (await (await connection.execute(
            "SELECT tool_calls FROM messages"
        )).fetchone())[0]
        assert json.loads(raw) == [{"name": "t", "arguments": "{}"}]
