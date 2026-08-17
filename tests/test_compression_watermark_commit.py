"""Watermark commit: concurrent appends survive in-place compaction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import pytest_asyncio

from hermes_state import SessionCompressionInProgressError, SessionDB


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> SessionDB:
    database = SessionDB(tmp_path / "state.db")
    await database.create_session("sess1", source="test")
    yield database
    await database.close()


async def _seed(database: SessionDB, count: int = 6) -> None:
    for index in range(count):
        role = "user" if index % 2 == 0 else "assistant"
        await database.append_message("sess1", role=role, content=f"turn {index}")


SUMMARY = [
    {"role": "user", "content": "[CONTEXT COMPACTION] summary of turns 0-5"},
    {"role": "assistant", "content": "Continuing from the summary."},
]


@pytest.mark.asyncio
async def test_concurrent_tail_survives_compaction(db: SessionDB) -> None:
    await _seed(db)
    watermark = await db.get_active_message_watermark("sess1")
    await db.append_message("sess1", role="user", content="mid-compression steer")
    await db.append_message("sess1", role="assistant", content="mid-compression reply")

    count = await db.archive_and_compact("sess1", SUMMARY, watermark=watermark)

    live = await db.get_messages("sess1")
    assert [row["content"] for row in live] == [
        SUMMARY[0]["content"],
        SUMMARY[1]["content"],
        "mid-compression steer",
        "mid-compression reply",
    ]
    assert count == 4


@pytest.mark.asyncio
async def test_tail_clone_preserves_tool_columns(db: SessionDB) -> None:
    await _seed(db, 2)
    watermark = await db.get_active_message_watermark("sess1")
    await db.append_message(
        "sess1",
        role="assistant",
        content="tool caller",
        tool_calls=[
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "terminal", "arguments": "{}"},
            }
        ],
    )
    await db.append_message(
        "sess1",
        role="tool",
        content="tool output",
        tool_call_id="c1",
        tool_name="terminal",
    )

    await db.archive_and_compact("sess1", SUMMARY, watermark=watermark)

    live = {row["content"]: row for row in await db.get_messages("sess1")}
    parsed = live["tool caller"]["tool_calls"]
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    assert parsed[0]["id"] == "c1"
    assert live["tool output"]["tool_call_id"] == "c1"
    assert live["tool output"]["tool_name"] == "terminal"


@pytest.mark.asyncio
async def test_conversation_load_includes_summary_and_tail(db: SessionDB) -> None:
    await _seed(db)
    watermark = await db.get_active_message_watermark("sess1")
    await db.append_message("sess1", role="user", content="late arrival")
    await db.archive_and_compact("sess1", SUMMARY, watermark=watermark)

    conversation = await db.get_messages_as_conversation("sess1")
    assert [message["content"] for message in conversation] == [
        SUMMARY[0]["content"],
        SUMMARY[1]["content"],
        "late arrival",
    ]


@pytest.mark.asyncio
async def test_none_watermark_preserves_historical_behavior(db: SessionDB) -> None:
    await _seed(db)
    await db.append_message("sess1", role="user", content="gets archived")
    assert await db.archive_and_compact("sess1", SUMMARY, watermark=None) == 2
    assert "gets archived" not in [
        row["content"] for row in await db.get_messages("sess1")
    ]


@pytest.mark.asyncio
async def test_archived_tail_remains_recoverable(db: SessionDB) -> None:
    await _seed(db, 4)
    watermark = await db.get_active_message_watermark("sess1")
    await db.append_message("sess1", role="user", content="tail row")
    await db.archive_and_compact("sess1", SUMMARY, watermark=watermark)

    rows = await db.get_messages("sess1", include_inactive=True)
    tail_rows = [row for row in rows if row["content"] == "tail row"]
    assert sorted(bool(row["active"]) for row in tail_rows) == [False, True]


@pytest.mark.asyncio
async def test_session_counters_include_tail(db: SessionDB) -> None:
    await _seed(db)
    watermark = await db.get_active_message_watermark("sess1")
    await db.append_message(
        "sess1",
        role="assistant",
        content="tail with tools",
        tool_calls=[
            {
                "id": "t1",
                "type": "function",
                "function": {"name": "x", "arguments": "{}"},
            }
        ],
    )
    await db.archive_and_compact("sess1", SUMMARY, watermark=watermark)
    session = await db.get_session("sess1")
    assert session["message_count"] == 3
    assert session["tool_call_count"] == 1


@pytest.mark.asyncio
async def test_commit_refused_when_lease_lost(db: SessionDB) -> None:
    await _seed(db)
    watermark = await db.get_active_message_watermark("sess1")
    assert await db.try_acquire_compression_lock("sess1", "worker-A") is True
    await db.release_compression_lock("sess1", "worker-A")
    assert await db.try_acquire_compression_lock("sess1", "worker-B") is True

    with pytest.raises(SessionCompressionInProgressError):
        await db.archive_and_compact(
            "sess1", SUMMARY, watermark=watermark, lock_holder="worker-A"
        )
    assert [row["content"] for row in await db.get_messages("sess1")] == [
        f"turn {index}" for index in range(6)
    ]


@pytest.mark.asyncio
async def test_commit_allowed_for_live_holder(db: SessionDB) -> None:
    await _seed(db)
    watermark = await db.get_active_message_watermark("sess1")
    assert await db.try_acquire_compression_lock("sess1", "worker-A") is True
    assert (
        await db.archive_and_compact(
            "sess1", SUMMARY, watermark=watermark, lock_holder="worker-A"
        )
        == 2
    )


@pytest.mark.asyncio
async def test_rotation_tail_follows_child(db: SessionDB) -> None:
    await _seed(db)
    watermark = await db.get_active_message_watermark("sess1")
    assert await db.try_acquire_compression_lock("sess1", "rotator") is True
    await db.append_message("sess1", role="user", content="mid-rotation steer")

    await db.publish_compression_child(
        parent_session_id="sess1",
        child_session_id="child1",
        source="test",
        messages=SUMMARY,
        compression_lock_holder="rotator",
        require_compression_lease=True,
        watermark=watermark,
    )

    child = await db.get_messages_as_conversation("child1")
    assert [message["content"] for message in child] == [
        SUMMARY[0]["content"],
        SUMMARY[1]["content"],
        "mid-rotation steer",
    ]
    child_info = await db.get_session("child1")
    assert child_info["message_count"] == 3
    parent_info = await db.get_session("sess1")
    assert parent_info["end_reason"] == "compression"
