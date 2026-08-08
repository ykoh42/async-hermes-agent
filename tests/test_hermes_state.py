"""Behavior contracts for the native-async ``SessionDB`` interface."""

import inspect
import sqlite3
import time

import pytest
import pytest_asyncio
from blockbuster import BlockBuster

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
        "queue_token_counts",
        "reopen_session",
        "finalize_orphaned_compression_sessions",
        "backfill_repo_roots",
        "list_prune_candidates",
        "archive_sessions",
        "archive_stale_sessions",
        "logical_size_bytes",
        "optimize_fts",
        "rebuild_fts",
        "vacuum",
        "maybe_auto_prune_and_vacuum",
        "maybe_auto_archive",
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
        "get_resume_conversations",
        "get_ancestor_display_prefix",
        "search_messages",
        "list_recent_user_messages",
        "search_sessions_by_id",
        "list_sessions_rich",
        "get_session_rich_row",
        "get_compression_tip",
        "get_compression_lineage",
        "resolve_resume_session_id",
        "restore_rewound",
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
async def test_queue_token_counts_preserves_delta_order_and_absolute_barrier(db):
    await db.create_session("accounting", source="library")

    await db.queue_token_counts(
        "accounting", input_tokens=5, output_tokens=2, api_call_count=1
    )
    await db.queue_token_counts(
        "accounting",
        input_tokens=100,
        output_tokens=20,
        api_call_count=1,
        absolute=True,
    )
    await db.queue_token_counts(
        "accounting", input_tokens=7, output_tokens=3, api_call_count=1
    )

    assert await db.flush_token_counts() is True
    session = await db.get_session("accounting")
    assert session["input_tokens"] == 107
    assert session["output_tokens"] == 23
    assert session["api_call_count"] == 2


@pytest.mark.asyncio
async def test_reopen_session_clears_the_existing_end_boundary(db):
    await db.create_session("resumable", source="library")
    await db.end_session("resumable", "completed")

    await db.reopen_session("resumable")

    session = await db.get_session("resumable")
    assert session["ended_at"] is None
    assert session["end_reason"] is None


@pytest.mark.asyncio
async def test_finalize_orphaned_compression_sessions_preserves_other_children(db):
    old = time.time() - 800_000
    await db.create_session("parent", source="library")
    await db.end_session("parent", "compression")

    for session_id in ("orphan", "used", "empty", "recent"):
        await db.create_session(
            session_id,
            source="library",
            parent_session_id="parent",
        )
    await db.append_message("orphan", role="user", content="unfinished")
    await db.append_message("used", role="user", content="completed call")
    await db.append_message("recent", role="user", content="still active")

    connection = await db._get_connection()
    await connection.execute(
        "UPDATE sessions SET started_at = ? "
        "WHERE id IN ('orphan', 'used', 'empty')",
        (old,),
    )
    await connection.execute(
        "UPDATE sessions SET api_call_count = 1 WHERE id = 'used'"
    )
    await connection.commit()

    assert await db.finalize_orphaned_compression_sessions() == 1
    assert (await db.get_session("orphan"))["end_reason"] == (
        "orphaned_compression"
    )
    for session_id in ("used", "empty", "recent"):
        session = await db.get_session(session_id)
        assert session["ended_at"] is None
        assert session["end_reason"] is None


@pytest.mark.asyncio
async def test_backfill_repo_roots_only_fills_missing_roots(db):
    await db.create_session("missing", source="library", cwd="/work/repo")
    await db.create_session(
        "recorded",
        source="library",
        cwd="/work/repo",
        git_repo_root="/original/root",
    )
    await db.create_session("not-git", source="library", cwd="/work/plain")

    await db.backfill_repo_roots(
        {
            "/work/repo": "/resolved/root",
            "/work/plain": "",
            "": "/ignored/root",
        }
    )

    assert (await db.get_session("missing"))["git_repo_root"] == (
        "/resolved/root"
    )
    assert (await db.get_session("recorded"))["git_repo_root"] == (
        "/original/root"
    )
    assert (await db.get_session("not-git"))["git_repo_root"] is None


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


@pytest.mark.asyncio
async def test_resume_resolution_follows_the_message_bearing_continuation(db):
    await db.create_session("root", source="library")
    await db.append_message("root", role="user", content="before compression")
    await db.end_session("root", "compression")
    await db.create_session(
        "continuation",
        source="library",
        parent_session_id="root",
    )
    await db.append_message(
        "continuation",
        role="assistant",
        content="after compression",
    )
    await db.create_session(
        "branch",
        source="library",
        parent_session_id="root",
        model_config={"_branched_from": "root"},
    )
    await db.append_message("branch", role="user", content="unrelated")

    connection = await db._get_connection()
    await connection.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = ?",
        (1.0, 2.0, "root"),
    )
    await connection.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?",
        (3.0, "continuation"),
    )
    await connection.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?",
        (4.0, "branch"),
    )
    await connection.commit()

    assert await db.resolve_resume_session_id("root") == "continuation"


@pytest.mark.asyncio
async def test_resume_resolution_walks_from_the_middle_of_a_plain_chain(db):
    parent = None
    for index, session_id in enumerate(("a", "b", "c", "d")):
        await db.create_session(
            session_id,
            source="library",
            parent_session_id=parent,
        )
        connection = await db._get_connection()
        await connection.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (float(index), session_id),
        )
        parent = session_id
    await connection.commit()
    await db.append_message("d", role="user", content="latest")

    assert await db.resolve_resume_session_id("b") == "d"
    assert await db.resolve_resume_session_id("c") == "d"


@pytest.mark.asyncio
async def test_resume_resolution_keeps_message_bearing_parent(db):
    await db.create_session("root", source="library")
    await db.append_message("root", role="user", content="only message")
    await db.create_session(
        "empty-child", source="library", parent_session_id="root"
    )

    assert await db.resolve_resume_session_id("root") == "root"


@pytest.mark.asyncio
async def test_resume_resolution_skips_non_conversation_children(db):
    await db.create_session("root", source="library")
    await db.create_session(
        "continuation", source="library", parent_session_id="root"
    )
    await db.create_session(
        "branch",
        source="library",
        parent_session_id="root",
        model_config={"_branched_from": "root"},
    )
    await db.create_session(
        "delegate",
        source="library",
        parent_session_id="root",
        model_config={"_delegate_from": "root"},
    )
    await db.create_session(
        "tool-child", source="tool", parent_session_id="root"
    )
    connection = await db._get_connection()
    for started_at, session_id in enumerate(
        ("continuation", "branch", "delegate", "tool-child"), start=1
    ):
        await connection.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (float(started_at), session_id),
        )
    await connection.commit()
    await db.append_message(
        "continuation", role="user", content="resumable"
    )
    await db.append_message("branch", role="user", content="branch")
    await db.append_message("delegate", role="user", content="delegate")
    await db.append_message("tool-child", role="tool", content="tool")

    assert await db.resolve_resume_session_id("root") == "continuation"


@pytest.mark.asyncio
async def test_resume_conversations_match_separate_reads_and_prefix(db):
    await db.create_session("root", source="library")
    await db.append_message("root", role="user", content="root question")
    await db.append_message("root", role="assistant", content="root answer")
    await db.end_session("root", "compression")
    await db.create_session(
        "tip", source="library", parent_session_id="root"
    )
    await db.append_message("tip", role="user", content="tip question")
    await db.append_message("tip", role="assistant", content="tip answer")

    model_history, display_history = await db.get_resume_conversations("tip")

    assert model_history == await db.get_messages_as_conversation(
        "tip", repair_alternation=True, include_row_ids=True
    )
    assert display_history == await db.get_messages_as_conversation(
        "tip", include_ancestors=True, include_row_ids=True
    )
    assert [
        message["content"]
        for message in await db.get_ancestor_display_prefix("tip")
    ] == ["root question", "root answer"]


@pytest.mark.asyncio
async def test_compression_lineage_excludes_branch_children(db):
    await db.create_session("root", source="library")
    await db.end_session("root", "compression")
    await db.create_session(
        "child", source="library", parent_session_id="root"
    )
    await db.end_session("child", "compression")
    await db.create_session(
        "tip", source="library", parent_session_id="child"
    )
    await db.create_session(
        "branch",
        source="library",
        parent_session_id="root",
        model_config={"_branched_from": "root"},
    )

    assert await db.get_compression_lineage("tip") == [
        "root",
        "child",
        "tip",
    ]
    assert await db.get_compression_lineage("branch") == ["branch"]


@pytest.mark.asyncio
async def test_restore_rewound_reactivates_the_archived_tail(db):
    await db.create_session("session", source="library")
    await db.append_message("session", role="user", content="first")
    await db.append_message("session", role="assistant", content="answer")
    target_id = await db.append_message(
        "session", role="user", content="second"
    )
    await db.append_message("session", role="assistant", content="later")

    assert (await db.rewind_to_message("session", target_id))["rewound_count"] == 2
    assert await db.restore_rewound("session", target_id) == 2
    assert [
        message["content"] for message in await db.get_messages("session")
    ] == ["first", "answer", "second", "later"]


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
async def test_list_and_archive_sessions_preserve_filters_order_and_lineage(db):
    now = time.time()
    await db.create_session("old", source="library")
    await db.set_session_title("old", "Archive old")
    await db.append_message("old", role="user", content="old")
    await db.end_session("old", "compression")
    await db.create_session(
        "tip", source="library", parent_session_id="old"
    )
    await db.set_session_title("tip", "Archive tip")
    await db.append_message("tip", role="assistant", content="tip")
    await db.end_session("tip", "done")
    await db.create_session("recent", source="library")
    await db.set_session_title("recent", "Archive recent")
    await db.append_message("recent", role="user", content="recent")
    await db.end_session("recent", "done")

    connection = await db._get_connection()
    await connection.execute(
        "UPDATE sessions SET started_at = ? WHERE id IN ('old', 'tip')",
        (now - 20 * 86_400,),
    )
    await connection.execute(
        "UPDATE messages SET timestamp = ? WHERE session_id IN ('old', 'tip')",
        (now - 10 * 86_400,),
    )
    await connection.execute(
        "UPDATE sessions SET started_at = ? WHERE id = 'recent'",
        (now - 2 * 86_400,),
    )
    await connection.execute(
        "UPDATE messages SET timestamp = ? WHERE session_id = 'recent'",
        (now - 86_400,),
    )
    await connection.commit()

    rows = await db.list_prune_candidates(
        older_than_days=5,
        title_like="archive",
        archived=False,
    )
    assert [row["id"] for row in rows] == ["old", "tip"]
    assert set(rows[0]) == {
        "id",
        "source",
        "title",
        "model",
        "started_at",
        "last_active",
        "ended_at",
        "message_count",
        "archived",
    }

    assert await db.archive_sessions(
        older_than_days=5, title_like="archive", end_reason="done"
    ) == 1
    assert (await db.get_session("old"))["archived"] == 1
    assert (await db.get_session("tip"))["archived"] == 1
    assert (await db.get_session("recent"))["archived"] == 0
    assert await db.archive_sessions(
        older_than_days=5, title_like="archive", end_reason="done"
    ) == 0


@pytest.mark.asyncio
async def test_archive_stale_sessions_honors_activity_pins_and_lineage_tips(db):
    old = time.time() - 10 * 86_400
    recent = time.time() - 86_400
    for session_id in ("stale", "pinned", "root", "tip"):
        await db.create_session(session_id, source="library")
        await db.append_message(session_id, role="user", content=session_id)
    await db.set_session_pinned("pinned", True)
    await db.end_session("root", "compression")
    connection = await db._get_connection()
    await connection.execute(
        "UPDATE sessions SET parent_session_id = 'root' WHERE id = 'tip'"
    )
    await connection.execute(
        "UPDATE sessions SET started_at = ?", (old,)
    )
    await connection.execute(
        "UPDATE messages SET timestamp = ?", (old,)
    )
    await connection.execute(
        "UPDATE messages SET timestamp = ? WHERE session_id = 'tip'", (recent,)
    )
    await connection.commit()

    assert await db.archive_stale_sessions(3) == 1
    assert (await db.get_session("stale"))["archived"] == 1
    assert (await db.get_session("pinned"))["archived"] == 0
    assert (await db.get_session("root"))["archived"] == 0
    assert (await db.get_session("tip"))["archived"] == 0

    assert await db.archive_stale_sessions(3, exclude_pinned=False) == 1
    assert (await db.get_session("pinned"))["archived"] == 1
    assert await db.archive_stale_sessions(-1) == 0


@pytest.mark.asyncio
async def test_auto_archive_is_throttled_and_records_zero_result(db):
    first = await db.maybe_auto_archive(idle_days=3, min_interval_hours=24)
    assert first == {"skipped": False, "archived": 0}
    assert await db.get_meta("last_auto_archive") is not None
    second = await db.maybe_auto_archive(idle_days=3, min_interval_hours=24)
    assert second == {"skipped": True, "archived": 0}


@pytest.mark.asyncio
async def test_auto_prune_removes_transcripts_and_throttles_vacuum(
    db, tmp_path, monkeypatch
):
    old = time.time() - 100 * 86_400
    await db.create_session("old", source="library")
    await db.append_message("old", role="user", content="old")
    await db.end_session("old", "done")
    connection = await db._get_connection()
    await connection.execute(
        "UPDATE sessions SET started_at = ? WHERE id = 'old'", (old,)
    )
    await connection.execute(
        "UPDATE messages SET timestamp = ? WHERE session_id = 'old'", (old,)
    )
    await connection.commit()
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / "old.jsonl").write_text("{}\n", encoding="utf-8")
    (sessions_dir / "request_dump_old_1.json").write_text("{}", encoding="utf-8")

    vacuum_calls = 0

    async def fake_vacuum():
        nonlocal vacuum_calls
        vacuum_calls += 1
        return 0

    monkeypatch.setattr(db, "vacuum", fake_vacuum)
    result = await db.maybe_auto_prune_and_vacuum(
        retention_days=90,
        min_interval_hours=0,
        sessions_dir=sessions_dir,
    )
    assert result == {"skipped": False, "pruned": 1, "vacuumed": True}
    assert vacuum_calls == 1
    assert not (sessions_dir / "old.jsonl").exists()
    assert not (sessions_dir / "request_dump_old_1.json").exists()
    assert await db.get_meta("last_vacuum") is not None

    await db.set_meta("last_auto_prune", str(time.time()))
    skipped = await db.maybe_auto_prune_and_vacuum(min_interval_hours=24)
    assert skipped == {"skipped": True, "pruned": 0, "vacuumed": False}


@pytest.mark.asyncio
async def test_auto_maintenance_preserves_best_effort_error_contracts(
    db, monkeypatch
):
    async def fail_prune(**_kwargs):
        raise RuntimeError("prune failed")

    monkeypatch.setattr(db, "prune_sessions", fail_prune)
    failed = await db.maybe_auto_prune_and_vacuum(min_interval_hours=0)
    assert failed == {
        "skipped": False,
        "pruned": 0,
        "vacuumed": False,
        "error": "prune failed",
    }
    assert await db.get_meta("last_auto_prune") is None

    async def one_pruned(**_kwargs):
        return 1

    async def fail_vacuum():
        raise RuntimeError("vacuum failed")

    monkeypatch.setattr(db, "prune_sessions", one_pruned)
    monkeypatch.setattr(db, "vacuum", fail_vacuum)
    recovered = await db.maybe_auto_prune_and_vacuum(min_interval_hours=0)
    assert recovered == {"skipped": False, "pruned": 1, "vacuumed": False}
    assert await db.get_meta("last_vacuum") is None
    assert await db.get_meta("last_auto_prune") is not None


@pytest.mark.asyncio
async def test_logical_size_optimize_and_vacuum_use_native_async_sqlite(db):
    await db.create_session("session", source="library")
    await db.append_message("session", role="user", content="hello")
    connection = await db._get_connection()
    await connection.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(content)"
    )
    await connection.commit()
    cursor = await connection.execute(
        "SELECT COUNT(*) FROM sqlite_master "
        "WHERE type = 'table' AND name IN (?, ?, ?)",
        db._FTS_TABLES,
    )
    present = await cursor.fetchone()
    await cursor.close()

    assert (await db.logical_size_bytes()) > 0
    assert present[0] >= 1
    assert await db.optimize_fts() == present[0]
    assert await db.vacuum() == present[0]


@pytest.mark.asyncio
async def test_fresh_database_creates_search_indexes_and_indexes_new_messages(db):
    await db.create_session("searchable", source="library")
    await db.append_message(
        "searchable", role="user", content="alpha filler beta"
    )

    connection = await db._get_connection()
    rows = await (
        await connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name IN (?, ?) ORDER BY name",
            ("messages_fts", "messages_fts_trigram"),
        )
    ).fetchall()
    trigger_count = await (
        await connection.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type = 'trigger' AND name IN (?, ?, ?, ?, ?, ?)",
            (
                "messages_fts_insert",
                "messages_fts_delete",
                "messages_fts_update",
                "messages_fts_trigram_insert",
                "messages_fts_trigram_delete",
                "messages_fts_trigram_update",
            ),
        )
    ).fetchone()

    assert [row["name"] for row in rows] == [
        "messages_fts",
        "messages_fts_trigram",
    ]
    assert trigger_count[0] == 6
    # A literal LIKE fallback cannot match this boolean expression. This proves
    # the fresh-database path is using the populated FTS index.
    assert [
        row["session_id"] for row in await db.search_messages("alpha AND beta")
    ] == ["searchable"]


@pytest.mark.asyncio
async def test_reopening_repairs_missing_fts_triggers_and_backfills_the_gap(tmp_path):
    path = tmp_path / "state.db"
    database = SessionDB(path)
    await database.create_session("repair", source="library")
    connection = await database._get_connection()
    for trigger_name in (
        "messages_fts_insert",
        "messages_fts_delete",
        "messages_fts_update",
        "messages_fts_trigram_insert",
        "messages_fts_trigram_delete",
        "messages_fts_trigram_update",
    ):
        await connection.execute(f"DROP TRIGGER {trigger_name}")
    await connection.commit()
    await database.append_message(
        "repair", role="user", content="triggerless woodpecker"
    )
    await database.close()

    reopened = SessionDB(path)
    try:
        assert [
            row["session_id"]
            for row in await reopened.search_messages("woodpecker")
        ] == ["repair"]
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_rebuild_fts_restores_index_from_canonical_messages(db):
    await db.create_session("rebuild", source="library")
    await db.append_message(
        "rebuild", role="user", content="recoverable kingfisher"
    )
    connection = await db._get_connection()
    for table_name in ("messages_fts", "messages_fts_trigram"):
        await connection.execute(
            f"INSERT INTO {table_name}({table_name}) VALUES('delete-all')"
        )
    await connection.commit()

    assert await db.search_messages("kingfisher") == []
    assert await db.rebuild_fts() == 2
    assert [
        row["session_id"] for row in await db.search_messages("kingfisher")
    ] == ["rebuild"]


@pytest.mark.asyncio
async def test_recent_user_messages_and_session_id_search_preserve_contract(db):
    await db.create_session("20260807_120000_abcd12", source="library")
    first = await db.append_message(
        "20260807_120000_abcd12", role="user", content="first real turn"
    )
    await db.append_message(
        "20260807_120000_abcd12",
        role="user",
        content="timeline marker",
        display_kind="model_switch",
    )
    latest = await db.append_message(
        "20260807_120000_abcd12",
        role="user",
        content=[{"type": "text", "text": "latest multimodal turn"}],
    )
    await db.rewind_to_message("20260807_120000_abcd12", first)

    recent = await db.list_recent_user_messages("20260807_120000_abcd12")
    assert recent == []
    recent_with_inactive = await db.list_recent_user_messages(
        "20260807_120000_abcd12", include_inactive=True
    )
    assert [row["id"] for row in recent_with_inactive] == [latest, first]
    assert recent_with_inactive[0]["preview"] == "latest multimodal turn"

    assert [
        row["id"] for row in await db.search_sessions_by_id("ABCD12")
    ] == ["20260807_120000_abcd12"]


@pytest.mark.asyncio
async def test_search_messages_returns_context_and_honors_projection(db):
    await db.create_session("context", source="library")
    await db.append_message("context", role="user", content="before")
    await db.append_message(
        "context", role="assistant", content="projectionneedle"
    )
    await db.append_message("context", role="user", content="after")

    default = await db.search_messages("projectionneedle")
    projected = await db.search_messages(
        "projectionneedle", fields=("session_id", "role", "snippet")
    )
    context_only = await db.search_messages(
        "projectionneedle", fields=("session_id", "context")
    )

    assert [message["content"] for message in default[0]["context"]] == [
        "before",
        "projectionneedle",
        "after",
    ]
    assert set(projected[0]) == {"session_id", "role", "snippet"}
    assert "context" not in projected[0]
    assert context_only == [
        {
            "session_id": "context",
            "context": default[0]["context"],
        }
    ]


@pytest.mark.asyncio
async def test_search_messages_preserves_cjk_substring_routes(db):
    await db.create_session("cjk", source="library")
    await db.append_message(
        "cjk", role="user", content="错误日志：数据库连接超时"
    )
    await db.append_message(
        "cjk", role="assistant", content="讨论Agent通信协议"
    )
    await db.append_message(
        "cjk", role="assistant", content="修改youer服务端"
    )

    assert len(await db.search_messages("数据库连接")) == 1
    assert len(await db.search_messages("连接")) == 1
    assert len(await db.search_messages("Agent通信")) == 1
    assert len(await db.search_messages("youer")) == 1


@pytest.mark.asyncio
async def test_search_messages_cjk_like_treats_wildcards_as_literals(db):
    await db.create_session("literal", source="library")
    await db.create_session("wildcard", source="library")
    await db.append_message(
        "literal", role="user", content="达成100%完成率"
    )
    await db.append_message(
        "wildcard", role="user", content="达成100完成率是目标"
    )

    assert [
        row["session_id"] for row in await db.search_messages("100%完成")
    ] == ["literal"]


@pytest.mark.asyncio
async def test_search_messages_cjk_or_and_tool_role_use_like_fallback(db):
    await db.create_session("guangxi", source="library")
    await db.create_session("guilin", source="library")
    await db.create_session("tool", source="library")
    await db.append_message("guangxi", role="user", content="广西旅行计划")
    await db.append_message("guilin", role="user", content="桂林旅行计划")
    await db.append_message(
        "tool",
        role="tool",
        content="数据库连接超时",
        tool_name="terminal",
    )

    assert {
        row["session_id"] for row in await db.search_messages("广西 OR 桂林")
    } == {"guangxi", "guilin"}
    tool_hits = await db.search_messages(
        "数据库连接", role_filter=["tool"]
    )
    assert [(row["session_id"], row["role"]) for row in tool_hits] == [
        ("tool", "tool")
    ]


@pytest.mark.asyncio
async def test_search_messages_normalizes_sort_and_preserves_empty_filters(db):
    await db.create_session("older", source="library")
    await db.create_session("newer", source="library")
    older_id = await db.append_message(
        "older", role="user", content="sortneedle"
    )
    newer_id = await db.append_message(
        "newer", role="user", content="sortneedle"
    )
    connection = await db._get_connection()
    await connection.execute(
        "UPDATE messages SET timestamp = ? WHERE id = ?", (10.0, older_id)
    )
    await connection.execute(
        "UPDATE messages SET timestamp = ? WHERE id = ?", (20.0, newer_id)
    )
    await connection.commit()

    assert [
        row["session_id"]
        for row in await db.search_messages("sortneedle", sort=" NEWEST ")
    ] == ["newer", "older"]
    assert await db.search_messages("sortneedle", source_filter=[]) == []
    assert {
        row["session_id"]
        for row in await db.search_messages("sortneedle", exclude_sources=[])
    } == {"older", "newer"}


@pytest.mark.asyncio
async def test_search_messages_visibility_matches_rewind_and_compaction(db):
    await db.create_session("visibility", source="library")
    rewound_id = await db.append_message(
        "visibility", role="user", content="rewoundneedle"
    )
    compacted_id = await db.append_message(
        "visibility", role="assistant", content="compactedneedle"
    )
    connection = await db._get_connection()
    await connection.execute(
        "UPDATE messages SET active = 0, compacted = 0 WHERE id = ?",
        (rewound_id,),
    )
    await connection.execute(
        "UPDATE messages SET active = 0, compacted = 1 WHERE id = ?",
        (compacted_id,),
    )
    await connection.commit()

    assert await db.search_messages("rewoundneedle") == []
    assert len(await db.search_messages("rewoundneedle", include_inactive=True)) == 1
    assert len(await db.search_messages("compactedneedle")) == 1


@pytest.mark.asyncio
async def test_search_messages_supplements_deferred_rebuild_gap(db):
    await db.create_session("gap", source="library")
    connection = await db._get_connection()
    await connection.execute("DROP TRIGGER messages_fts_insert")
    await connection.execute("DROP TRIGGER messages_fts_trigram_insert")
    await connection.commit()
    message_id = await db.append_message(
        "gap", role="user", content="deferredgapneedle"
    )
    await db.set_meta("fts_rebuild_high_water", str(message_id))
    await db.set_meta("fts_rebuild_progress", "0")

    hits = await db.search_messages("deferredgapneedle")
    assert [row["session_id"] for row in hits] == ["gap"]


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
async def test_read_only_connection_uses_the_same_async_interface(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "state.db"
    writer = SessionDB(path)
    await writer.create_session("s1", source="library")
    await writer.append_message(
        "s1", role="user", content="readonly woodpecker"
    )
    await writer.close()

    monkeypatch.chdir(tmp_path)
    reader = SessionDB("state.db", read_only=True)
    blocker = BlockBuster()
    blocker.activate()
    try:
        assert (await reader.get_session("s1"))["id"] == "s1"
        assert reader._fts_enabled is True
        assert reader._trigram_available is True
        assert [
            row["session_id"]
            for row in await reader.search_messages("woodpecker")
        ] == ["s1"]
        with pytest.raises(sqlite3.OperationalError):
            await reader.set_meta("forbidden", "write")
    finally:
        blocker.deactivate()
        await reader.close()
