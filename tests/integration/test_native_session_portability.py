"""Native-async E2E coverage for retained session export and import."""

from __future__ import annotations

import asyncio
import inspect
import json

import pytest
from blockbuster import BlockBuster
from pyleak import no_task_leaks
from pyleak.eventloop import LeakAction

from agent.session_activity import ActivityProvenance
from hermes_state import SessionDB
from hermes_state_portability import SessionPortabilityMixin


pytestmark = pytest.mark.asyncio


async def _close_all(*databases: SessionDB) -> None:
    for database in databases:
        await database.close()


async def test_session_portability_real_sqlite_round_trip_preserves_trajectory(
    tmp_path,
):
    source = SessionDB(tmp_path / "source.db")
    restored = SessionDB(tmp_path / "restored.db")
    blocker = BlockBuster()

    async with no_task_leaks(action=LeakAction.RAISE):
        blocker.activate()
        try:
            await source.create_session(
                "root",
                source="cli",
                model="test/model",
                model_config={"reasoning": True},
                system_prompt="stable prompt",
                cwd="/workspace/project",
            )
            await source.append_messages_batch(
                "root",
                [
                    {"role": "user", "content": "inspect"},
                    {
                        "role": "assistant",
                        "content": "running",
                        "reasoning": "inspect first",
                        "reasoning_content": "inspect first",
                        "reasoning_details": [
                            {"type": "reasoning.text", "text": "signed state"}
                        ],
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "terminal",
                                    "arguments": json.dumps(
                                        {"command": "printf observed"}
                                    ),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "content": "observed",
                        "tool_name": "terminal",
                        "tool_call_id": "call-1",
                    },
                    {
                        "role": "assistant",
                        "content": "finished",
                        "reasoning": "use observation",
                    },
                ],
            )
            await source.touch_session_activity(
                "root",
                ts=1234.0,
                description="working",
                provenance=ActivityProvenance.AGENT_COMPRESSION,
            )
            await source.end_session("root", "compression")
            await source.create_session(
                "tip",
                source="cli",
                parent_session_id="root",
                model="test/model",
            )
            await source.append_message(
                "tip",
                role="assistant",
                content="continued",
            )

            exported_root = await source.export_session("root")
            exported_lineage = await source.export_session_lineage("tip")
            exported_all = await source.export_all(source="cli")

            assert exported_root is not None
            assert exported_root["system_prompt"] == "stable prompt"
            assert exported_root["last_activity_at"] == 1234.0
            assert exported_lineage is not None
            assert exported_lineage["lineage_session_ids"] == ["root", "tip"]
            assert [
                message["role"] for message in exported_lineage["messages"]
            ] == ["user", "assistant", "tool", "assistant", "assistant"]

            import_result = await restored.import_sessions(exported_all)
            restored_root = await restored.export_session("root")
            restored_tip = await restored.get_session("tip")
        finally:
            await _close_all(source, restored)
            blocker.deactivate()

    assert import_result == {
        "ok": True,
        "imported": 2,
        "skipped": 0,
        "detached": 0,
        "imported_ids": [item["id"] for item in exported_all],
        "skipped_ids": [],
        "errors": [],
    }
    assert restored_root is not None
    assert restored_tip is not None
    assert restored_tip["parent_session_id"] == "root"
    assert restored_root["system_prompt"] == "stable prompt"
    assert restored_root["last_activity_at"] is None
    assert restored_root["last_activity_description"] is None
    assert restored_root["last_activity_provenance"] is None

    original_messages = exported_root["messages"]
    restored_messages = restored_root["messages"]
    assert [message["role"] for message in restored_messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    for original, imported in zip(original_messages, restored_messages, strict=True):
        for field in (
            "role",
            "content",
            "tool_calls",
            "tool_call_id",
            "tool_name",
            "reasoning",
            "reasoning_content",
            "reasoning_details",
        ):
            assert imported.get(field) == original.get(field)


async def test_session_import_validation_is_atomic(tmp_path):
    database = SessionDB(tmp_path / "validation.db")
    try:
        result = await database.import_sessions(
            [
                {"id": "valid-looking", "messages": []},
                {"id": "invalid", "messages": [{"role": 3, "content": "x"}]},
            ]
        )

        assert result == {
            "ok": False,
            "imported": 0,
            "skipped": 0,
            "detached": 0,
            "errors": [
                {
                    "index": 1,
                    "session_id": "invalid",
                    "error": "messages[0].role must be a non-empty string",
                }
            ],
        }
        assert await database.get_session("valid-looking") is None
        assert await database.get_session("invalid") is None
    finally:
        await database.close()


async def test_session_import_enforces_upstream_payload_limits(
    tmp_path,
    monkeypatch,
):
    database = SessionDB(tmp_path / "limits.db")
    monkeypatch.setattr(SessionDB, "_IMPORT_MAX_SESSIONS", 1)
    with pytest.raises(ValueError, match="at most 1 entries"):
        await database.import_sessions(
            [{"id": "one", "messages": []}, {"id": "two", "messages": []}]
        )

    monkeypatch.setattr(SessionDB, "_IMPORT_MAX_SESSIONS", 500)
    monkeypatch.setattr(SessionDB, "_IMPORT_MAX_MESSAGES_PER_SESSION", 1)
    try:
        result = await database.import_sessions(
            [
                {
                    "id": "too-many-messages",
                    "messages": [
                        {"role": "user", "content": "one"},
                        {"role": "assistant", "content": "two"},
                    ],
                }
            ]
        )
        assert result["ok"] is False
        assert result["errors"] == [
            {
                "index": 0,
                "session_id": "too-many-messages",
                "error": "messages exceeds the per-session import limit",
            }
        ]
        assert await database.get_session("too-many-messages") is None
    finally:
        await database.close()


async def test_session_import_detaches_missing_and_cyclic_parents(tmp_path):
    database = SessionDB(tmp_path / "parents.db")
    payload = [
        {"id": "missing", "parent_session_id": "absent", "messages": []},
        {"id": "a", "parent_session_id": "b", "messages": []},
        {"id": "b", "parent_session_id": "a", "messages": []},
    ]
    try:
        result = await database.import_sessions(payload)

        assert result["ok"] is True
        assert result["imported"] == 3
        assert result["detached"] == 2
        missing = await database.get_session("missing")
        a = await database.get_session("a")
        b = await database.get_session("b")
        assert missing is not None and missing["parent_session_id"] is None
        assert a is not None and a["parent_session_id"] is None
        assert b is not None and b["parent_session_id"] == "a"

        repeated = await database.import_sessions(payload)
        assert repeated["imported"] == 0
        assert repeated["skipped"] == 3
    finally:
        await database.close()


async def test_cancelled_session_import_rolls_back_partial_transaction(
    tmp_path,
    monkeypatch,
):
    database = SessionDB(tmp_path / "cancel.db")
    insert_started = asyncio.Event()
    release_insert = asyncio.Event()
    original_insert = database._insert_message_rows

    async def _stalled_insert(connection, session_id, messages):
        insert_started.set()
        await release_insert.wait()
        return await original_insert(connection, session_id, messages)

    monkeypatch.setattr(database, "_insert_message_rows", _stalled_insert)

    async with no_task_leaks(action=LeakAction.RAISE):
        task = asyncio.create_task(
            database.import_sessions(
                [
                    {
                        "id": "cancelled",
                        "messages": [{"role": "user", "content": "partial"}],
                    }
                ]
            )
        )
        await asyncio.wait_for(insert_started.wait(), timeout=1.0)
        task.cancel()
        release_insert.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        assert await database.get_session("cancelled") is None
        assert await database.message_count() == 0
    finally:
        await database.close()


async def test_skill_session_helpers_and_distinct_cwds(tmp_path):
    database = SessionDB(tmp_path / "helpers.db")
    try:
        await database.create_session(
            "skill",
            source="cli",
            cwd="/workspace/a",
        )
        await database.append_message(
            "skill",
            role="user",
            content="[IMPORTANT: The user has invoked the demo skill] request",
        )
        await database.append_message(
            "skill",
            role="assistant",
            content="first reply",
        )
        await database.set_session_title("skill", "Skill title")
        await database.create_session(
            "other",
            source="cli",
            cwd="/workspace/a",
        )
        await database.create_session(
            "archived",
            source="cli",
            cwd="/workspace/b",
        )
        await database.set_session_archived("archived", True)

        scaffolded = await database.list_skill_scaffolded_sessions()
        first_reply = await database.get_first_assistant_text("skill")
        active_cwds = await database.distinct_session_cwds()
        all_cwds = await database.distinct_session_cwds(include_archived=True)
    finally:
        await database.close()

    assert scaffolded == [
        {
            "id": "skill",
            "title": "Skill title",
            "content": "[IMPORTANT: The user has invoked the demo skill] request",
        }
    ]
    assert first_reply == "first reply"
    assert [(row["cwd"], row["sessions"]) for row in active_cwds] == [
        ("/workspace/a", 2)
    ]
    assert {row["cwd"] for row in all_cwds} == {
        "/workspace/a",
        "/workspace/b",
    }


async def test_portability_module_preserves_coroutine_api_shape():
    assert issubclass(SessionDB, SessionPortabilityMixin)
    for name in (
        "distinct_session_cwds",
        "list_skill_scaffolded_sessions",
        "get_first_assistant_text",
        "export_session",
        "export_session_lineage",
        "export_all",
        "import_sessions",
    ):
        assert inspect.iscoroutinefunction(getattr(SessionDB, name))
