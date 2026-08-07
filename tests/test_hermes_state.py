"""Behavior contracts for the native-async ``SessionDB`` interface."""

import inspect
import sqlite3
import time

import pytest
import pytest_asyncio

from hermes_state import CompressionSessionClosedError, SessionDB, workspace_key


@pytest_asyncio.fixture
async def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    yield database
    await database.close()


def test_public_session_interface_is_async():
    for name in (
        "create_session",
        "ensure_session",
        "append_message",
        "replace_messages",
        "clear_messages",
        "message_count",
        "latest_message_row_id",
        "latest_user_message_row_id",
        "get_message_role",
        "update_session_meta",
        "update_session_model",
        "get_session",
        "resolve_session_id",
        "session_count_ge",
        "count_empty_sessions",
        "has_archived_messages",
        "search_sessions",
        "session_count",
        "session_count_by_source",
        "has_platform_message_id",
        "get_messages",
        "get_messages_as_conversation",
        "search_messages",
        "list_sessions_rich",
        "get_session_rich_row",
        "get_compression_tip",
        "set_session_title",
        "set_auto_title_if_empty",
        "set_session_archived",
        "set_session_pinned",
        "close",
    ):
        assert inspect.iscoroutinefunction(getattr(SessionDB, name)), name


@pytest.mark.asyncio
async def test_session_compatibility_primitives_preserve_upstream_contract(db):
    assert await db.ensure_session(
        "session-alpha",
        source="library",
        model="model-a",
        cwd="/workspace",
    ) == "session-alpha"
    assert await db.ensure_session("session-alpha", source="ignored") == (
        "session-alpha"
    )
    assert (await db.get_session("session-alpha"))["source"] == "library"

    first = await db.append_message(
        "session-alpha", role="user", content="first"
    )
    second = await db.append_message(
        "session-alpha", role="assistant", content="second"
    )
    tool_call_only = await db.append_message(
        "session-alpha",
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "terminal", "arguments": "{}"},
            }
        ],
    )
    assert await db.message_count("session-alpha") == 3
    assert await db.message_count() == 3
    assert await db.latest_user_message_row_id("session-alpha") == first
    assert await db.latest_message_row_id(
        "session-alpha", role="assistant"
    ) == second
    assert await db.latest_message_row_id(
        "session-alpha", role="assistant", require_text=False
    ) == tool_call_only
    assert await db.get_message_role("session-alpha", second) == "assistant"

    await db.replace_messages(
        "session-alpha",
        [
            {"role": "user", "content": "replacement"},
            {"role": "assistant", "content": "answer"},
        ],
    )
    assert [
        message["content"] for message in await db.get_messages("session-alpha")
    ] == ["replacement", "answer"]
    assert await db.message_count("session-alpha") == 2

    await db.update_session_meta(
        "session-alpha",
        '{"custom":"kept","browser_model_lock":{"model":"old"}}',
        model="model-b",
    )
    await db.update_session_model("session-alpha", "model-c")
    session = await db.get_session("session-alpha")
    assert session["model"] == "model-c"
    assert session["model_config"] == '{"custom":"kept"}'

    assert await db.resolve_session_id("session-alpha") == "session-alpha"
    assert await db.resolve_session_id("session-al") == "session-alpha"
    await db.ensure_session("session-alpine", source="library")
    assert await db.resolve_session_id("session-al") is None
    assert await db.session_count_ge(2) is True

    await db.clear_messages("session-alpha")
    assert await db.message_count("session-alpha") == 0
    assert (await db.get_session("session-alpha"))["tool_call_count"] == 0
    await db.end_session("session-alpha", "done")
    assert await db.count_empty_sessions() == 1


@pytest.mark.asyncio
async def test_replace_active_messages_preserves_archived_transcript(db):
    await db.ensure_session("session", source="library")
    archived_id = await db.append_message(
        "session", role="user", content="archived"
    )
    await db.append_message("session", role="assistant", content="live")
    connection = await db._get_connection()
    await connection.execute(
        "UPDATE messages SET active = 0, compacted = 1 WHERE id = ?",
        (archived_id,),
    )
    await connection.commit()

    await db.replace_messages(
        "session",
        [{"role": "user", "content": "new live"}],
        active_only=True,
    )

    all_messages = await db.get_messages("session", include_inactive=True)
    assert [message["content"] for message in all_messages] == [
        "archived",
        "new live",
    ]
    assert (await db.get_session("session"))["message_count"] == 1


@pytest.mark.asyncio
async def test_replace_rejects_closed_compression_session_atomically(db):
    await db.ensure_session("session", source="library")
    await db.append_message("session", role="user", content="preserved")
    await db.end_session("session", "compression")

    with pytest.raises(CompressionSessionClosedError):
        await db.replace_messages(
            "session", [{"role": "user", "content": "replacement"}]
        )

    assert [
        message["content"] for message in await db.get_messages("session")
    ] == ["preserved"]


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

    addressed = await db.get_messages_as_conversation("s1", include_row_ids=True)
    assert [message["_row_id"] for message in addressed] == [
        message["id"] for message in await db.get_messages("s1")
    ]


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


def test_sanitize_title_preserves_upstream_validation_contract():
    assert SessionDB.sanitize_title("  My\n Project  ") == "My Project"
    assert SessionDB.sanitize_title("hello\x00world\u200b") == "helloworld"
    assert SessionDB.sanitize_title(" \t\n ") is None
    with pytest.raises(ValueError, match="too long"):
        SessionDB.sanitize_title("A" * 101)


@pytest.mark.asyncio
async def test_session_title_uniqueness_and_auto_title_are_atomic(db):
    await db.create_session("manual", source="library")
    await db.create_session("automatic", source="library")
    await db.create_session("duplicate", source="library")

    assert await db.set_session_title("manual", "  Shared\n Title  ") is True
    assert await db.get_session_title("manual") == "Shared Title"
    with pytest.raises(ValueError, match="already in use"):
        await db.set_session_title("duplicate", "Shared Title")

    assert await db.set_auto_title_if_empty("automatic", "Generated") is True
    assert await db.set_auto_title_if_empty("automatic", "Overwrite") is False
    assert await db.get_session_title("automatic") == "Generated"


@pytest.mark.asyncio
async def test_compression_tip_can_reclaim_hidden_ancestor_title(db):
    await db.create_session("root", source="library")
    await db.set_session_title("root", "fingerprint-scanner")
    await db.end_session("root", "compression")
    await db.create_session(
        "tip", source="library", parent_session_id="root"
    )
    await db.set_session_title("tip", "fingerprint-scanner #2")

    assert await db.set_session_title("tip", "fingerprint-scanner") is True
    assert await db.get_session_title("root") is None
    assert await db.get_session_title("tip") == "fingerprint-scanner"


@pytest.mark.asyncio
async def test_archive_and_pin_update_the_whole_compression_lineage(db):
    await db.create_session("root", source="library")
    await db.end_session("root", "compression")
    await db.create_session(
        "tip", source="library", parent_session_id="root"
    )

    assert await db.set_session_archived("tip", True) is True
    assert await db.set_session_pinned("root", True) is True
    assert (await db.get_session("root"))["archived"] == 1
    assert (await db.get_session("tip"))["archived"] == 1
    assert (await db.get_session("root"))["pinned"] == 1
    assert (await db.get_session("tip"))["pinned"] == 1

    assert await db.set_session_archived("root", False) is True
    assert await db.set_session_pinned("tip", False) is True
    assert (await db.get_session("root"))["archived"] == 0
    assert (await db.get_session("tip"))["archived"] == 0
    assert (await db.get_session("root"))["pinned"] == 0
    assert (await db.get_session("tip"))["pinned"] == 0


def test_workspace_key_matches_upstream_grouping_contract():
    assert workspace_key(
        {
            "git_repo_root": "/workspace/repo",
            "cwd": "/workspace/repo/src",
            "git_branch": "feature",
        }
    ) == "/workspace/repo"
    assert workspace_key({"cwd": "/workspace/notes"}) == "/workspace/notes"
    assert workspace_key({"git_repo_root": "", "cwd": "   "}) is None


@pytest.mark.asyncio
async def test_search_sessions_preserves_workspace_order_and_pagination(db):
    await db.create_session(
        "repo-root",
        source="library",
        cwd="/workspace/repo",
        git_repo_root="/workspace/repo",
    )
    await db.create_session(
        "repo-child-dir",
        source="library",
        cwd="/workspace/repo/src",
    )
    await db.create_session(
        "other",
        source="other",
        cwd="/workspace/other",
    )
    connection = await db._get_connection()
    await connection.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?",
        (1.0, "repo-root"),
    )
    await connection.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?",
        (2.0, "repo-child-dir"),
    )
    await connection.commit()

    matches = await db.search_sessions(
        source="library", workspace_key="/workspace/repo"
    )
    assert [session["id"] for session in matches] == [
        "repo-child-dir",
        "repo-root",
    ]
    assert [session["last_active"] for session in matches] == [2.0, 1.0]
    assert [
        session["id"]
        for session in await db.search_sessions(
            source="library", limit=1, offset=1
        )
    ] == ["repo-root"]


@pytest.mark.asyncio
async def test_session_counts_preserve_visibility_filters(db):
    await db.create_session("cli-root", source="cli", cwd="/workspace/repo")
    await db.append_message("cli-root", role="user", content="hello")
    await db.create_session("telegram-root", source="telegram")
    await db.create_session("archived", source="cli")
    await db.set_session_archived("archived", True)
    await db.end_session("cli-root", "compression")
    await db.create_session(
        "compression-tip",
        source="cli",
        parent_session_id="cli-root",
    )

    assert await db.session_count() == 3
    assert await db.session_count(source="cli") == 2
    assert await db.session_count(sources=["cli", "telegram"]) == 3
    assert await db.session_count(exclude_sources=["telegram"]) == 2
    assert await db.session_count(cwd_prefix="/workspace") == 2
    assert await db.session_count(min_message_count=1) == 1
    assert await db.session_count(include_archived=True) == 4
    assert await db.session_count(archived_only=True) == 1
    assert await db.session_count(exclude_children=True) == 2
    assert await db.session_count_by_source() == {"cli": 2, "telegram": 1}
    assert await db.session_count_by_source(include_archived=True) == {
        "cli": 3,
        "telegram": 1,
    }
    assert await db.session_count_by_source(exclude_children=True) == {
        "cli": 1,
        "telegram": 1,
    }


@pytest.mark.asyncio
async def test_session_count_by_source_normalizes_empty_source_to_cli(db):
    await db.create_session("implicit-cli", source="")
    await db.create_session("explicit-cli", source="cli")

    assert await db.session_count_by_source() == {"cli": 2}


@pytest.mark.asyncio
async def test_message_existence_probes_preserve_session_scope(db):
    await db.create_session("one", source="library")
    await db.create_session("two", source="library")
    archived_id = await db.append_message(
        "one",
        role="user",
        content="archived",
        platform_message_id="platform-1",
    )
    connection = await db._get_connection()
    await connection.execute(
        "UPDATE messages SET active = 0 WHERE id = ?",
        (archived_id,),
    )
    await connection.commit()

    assert await db.has_archived_messages("one") is True
    assert await db.has_archived_messages("two") is False
    assert await db.has_platform_message_id("one", "platform-1") is True
    assert await db.has_platform_message_id("two", "platform-1") is False


@pytest.mark.asyncio
async def test_rich_listing_honors_search_id_and_session_scope(db):
    await db.create_session(
        "root-an94",
        source="library",
        session_key="scope:a",
    )
    await db.set_session_title("root-an94", "AN-94 Notes")
    await db.create_session(
        "other-session",
        source="library",
        session_key="scope:b",
    )
    await db.set_session_title("other-session", "Unrelated")

    scoped = await db.list_sessions_rich(
        session_key="scope:a",
        order_by_last_active=True,
    )
    assert [session["id"] for session in scoped] == ["root-an94"]

    by_id = await db.list_sessions_rich(
        id_query="ROOT-AN",
        order_by_last_active=True,
    )
    assert [session["id"] for session in by_id] == ["root-an94"]

    by_title = await db.list_sessions_rich(
        search_query="an94",
        order_by_last_active=True,
    )
    assert [session["id"] for session in by_title] == ["root-an94"]


@pytest.mark.asyncio
async def test_rich_listing_backfills_pinned_rows_beyond_page(db):
    await db.create_session("old-pinned", source="library")
    await db.create_session("new-recent", source="library")
    connection = await db._get_connection()
    await connection.execute(
        "UPDATE sessions SET pinned = 1, started_at = 1 WHERE id = ?",
        ("old-pinned",),
    )
    await connection.execute(
        "UPDATE sessions SET started_at = 2 WHERE id = ?",
        ("new-recent",),
    )
    await connection.commit()

    page = await db.list_sessions_rich(limit=1, include_pinned=True)
    assert [session["id"] for session in page] == ["new-recent", "old-pinned"]


@pytest.mark.asyncio
async def test_rich_listing_projects_compression_root_to_tip(db):
    await db.create_session("root", source="library")
    await db.append_message("root", role="user", content="old preview")
    await db.end_session("root", "compression")
    await db.create_session(
        "middle",
        source="library",
        parent_session_id="root",
    )
    await db.end_session("middle", "compression")
    await db.create_session(
        "tip",
        source="library",
        parent_session_id="middle",
    )
    await db.append_message("tip", role="user", content="live preview")

    sessions = await db.list_sessions_rich(source="library")

    assert await db.get_compression_tip("root") == "tip"
    assert await db.get_compression_tip("middle") == "tip"
    assert [session["id"] for session in sessions] == ["tip"]
    assert sessions[0]["_lineage_root_id"] == "root"
    assert sessions[0]["preview"] == "live preview"


@pytest.mark.asyncio
async def test_rich_listing_uses_freshest_activity_and_compact_projection(db):
    await db.create_session(
        "activity",
        source="library",
        system_prompt="large prompt payload",
    )
    await db.append_message("activity", role="user", content="hello", timestamp=20)
    connection = await db._get_connection()
    await connection.execute(
        "UPDATE sessions SET last_activity_at = 10 WHERE id = ?",
        ("activity",),
    )
    await connection.commit()

    row = await db.get_session_rich_row("activity", compact_rows=True)

    assert row is not None
    assert row["last_active"] == 20
    assert "system_prompt" not in row
    assert "system_prompt_hash" not in row


@pytest.mark.asyncio
async def test_meta_cursor_and_token_flush_keep_upstream_arguments(db):
    connection = await db._get_connection()
    cursor = await connection.cursor()

    await db.set_meta("inline", "ready", cursor=cursor)

    assert await db.get_meta("inline") == "ready"
    assert await db.flush_token_counts(0.01) is True


@pytest.mark.asyncio
async def test_compression_cooldown_raw_snapshot_round_trip(db):
    await db.create_session("cooldown", source="library")
    original = await db.get_compression_failure_cooldown_row("cooldown")
    assert original == {
        "session_exists": True,
        "cooldown_until": None,
        "error": None,
    }

    await db.record_compression_failure_cooldown(
        "cooldown", time.time() + 60, "temporary"
    )
    assert (await db.get_compression_failure_cooldown_row("cooldown"))["error"] == (
        "temporary"
    )

    await db.restore_compression_failure_cooldown_row("cooldown", original)
    assert await db.get_compression_failure_cooldown_row("cooldown") == original


@pytest.mark.asyncio
async def test_delete_session_preserves_upstream_cascade_and_orphan_contract(
    db, tmp_path
):
    await db.create_session("parent", source="library")
    await db.create_session(
        "delegate",
        source="tool",
        parent_session_id="parent",
        model_config={"_delegate_from": "parent"},
    )
    await db.create_session(
        "branch",
        source="library",
        parent_session_id="parent",
        model_config={"_branched_from": "parent"},
    )
    await db.append_message("parent", role="user", content="root")
    await db.append_message("delegate", role="assistant", content="child")
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    for name in (
        "parent.jsonl",
        "delegate.json",
        "request_dump_delegate_1.json",
    ):
        (sessions_dir / name).write_text("{}", encoding="utf-8")

    expected = await db.get_session_delete_targets("parent")
    assert expected == ["parent", "delegate"]
    await db.create_session(
        "late-delegate",
        source="tool",
        parent_session_id="parent",
        model_config={"_delegate_from": "parent"},
    )
    assert not await db.delete_session(
        "parent", sessions_dir=sessions_dir, expected_delete_ids=expected
    )

    assert await db.delete_session("parent", sessions_dir=sessions_dir)
    assert await db.get_session("parent") is None
    assert await db.get_session("delegate") is None
    assert await db.get_session("late-delegate") is None
    assert (await db.get_session("branch"))["parent_session_id"] is None
    assert not any(sessions_dir.iterdir())


@pytest.mark.asyncio
async def test_single_empty_guard_and_bulk_sweep_keep_distinct_contracts(
    db, tmp_path
):
    await db.create_session("empty", source="library")
    await db.end_session("empty", "done")
    await db.create_session("live", source="library")
    await db.create_session("titled", source="library")
    await db.set_session_title("titled", "Keep")
    await db.end_session("titled", "done")
    await db.create_session("archived", source="library")
    await db.end_session("archived", "done")
    connection = await db._get_connection()
    await connection.execute(
        "UPDATE sessions SET archived = 1 WHERE id = 'archived'"
    )
    await connection.commit()
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / "empty.jsonl").write_text("", encoding="utf-8")

    assert await db.delete_session_if_empty("titled", sessions_dir) is False
    assert await db.delete_empty_sessions(sessions_dir) == 2
    assert await db.get_session("empty") is None
    assert await db.get_session("titled") is None
    assert await db.get_session("live") is not None
    assert await db.get_session("archived") is not None
    assert not (sessions_dir / "empty.jsonl").exists()


@pytest.mark.asyncio
async def test_prune_sessions_uses_last_activity_and_upstream_filters(db):
    old = time.time() - 120 * 86_400
    recent = time.time() - 1 * 86_400
    for session_id, title in (("stale", "Batch stale"), ("active", "Batch active")):
        await db.create_session(session_id, source="library")
        await db.set_session_title(session_id, title)
        await db.append_message(session_id, role="user", content=session_id)
        await db.end_session(session_id, "done")

    connection = await db._get_connection()
    await connection.execute(
        "UPDATE sessions SET started_at = ? WHERE id IN ('stale', 'active')", (old,)
    )
    await connection.execute(
        "UPDATE messages SET timestamp = ? WHERE session_id = 'stale'", (old,)
    )
    await connection.execute(
        "UPDATE messages SET timestamp = ? WHERE session_id = 'active'", (recent,)
    )
    await connection.commit()

    assert await db.prune_sessions(title_like="batch", max_messages=1) == 1
    assert await db.get_session("stale") is None
    assert await db.get_session("active") is not None


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
