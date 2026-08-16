"""Behavior-level differential checks for the SQLite and PostgreSQL backends.

These tests deliberately compare the retained user-visible contract instead of
comparing SQL or private schema details.  The SQLite backend is the retained
upstream behavior oracle.  The explicit backend-difference allowlist is limited
to generated clocks/row identifiers, FTS score and tool-call headline text,
and low-level driver error classes; everything else in the selected scenarios
is compared directly so an accidental backend drift fails loudly.
"""

from __future__ import annotations

import ast
import json
import os
import uuid
from pathlib import Path
from typing import Any

import pytest

from hermes_state import SessionDB as SQLiteSessionDB
from hermes_state_postgres import SessionDB as PostgresSessionDB


_REPO_ROOT = Path(__file__).resolve().parents[2]
# These are the async-retained equivalents of the upstream v2026.8.13
# SessionDB tests; the differential cases must remain anchored to them.
_RETAINED_UPSTREAM_ANCHORS = {
    "crud": ("tests/test_hermes_state.py", "test_session_lifecycle_and_transcript_round_trip"),
    "messages": (
        "tests/test_hermes_state.py",
        "test_conversation_replay_preserves_tool_order_and_reasoning",
    ),
    "search": (
        "tests/test_hermes_state.py",
        "test_search_messages_returns_context_and_honors_projection",
    ),
    "title": (
        "tests/test_hermes_state.py",
        "test_session_title_uniqueness_and_auto_title_are_atomic",
    ),
    "export": (
        "tests/hermes_state/test_transcript_safety.py",
        "test_export_safety_is_bounded_to_requested_active_segment",
    ),
    "tool_calls": (
        "tests/test_hermes_state.py",
        "test_search_messages_cjk_or_and_tool_role_use_like_fallback",
    ),
}


def _postgres_dsn() -> str:
    dsn = os.environ.get("HERMES_POSTGRES_TEST_DSN")
    if not dsn:
        pytest.skip("set HERMES_POSTGRES_TEST_DSN for a real PostgreSQL run")
    return dsn


def _canonical_id(value: Any, ids: dict[str, str]) -> Any:
    if isinstance(value, str):
        return next(
            (label for label, session_id in ids.items() if value == session_id),
            value,
        )
    if isinstance(value, list):
        return [_canonical_id(item, ids) for item in value]
    if isinstance(value, dict):
        return {key: _canonical_id(item, ids) for key, item in value.items()}
    return value


def _canonical_session(row: dict[str, Any], ids: dict[str, str]) -> dict[str, Any]:
    """Keep stable session semantics and omit clock/driver-generated values."""
    fields = (
        "source",
        "model",
        "model_config",
        "system_prompt",
        "title",
        "title_source",
        "message_count",
        "tool_call_count",
        "archived",
        "pinned",
        "end_reason",
        "ended_at",
        "parent_session_id",
        "cwd",
        "git_repo_root",
        "session_key",
        "user_id",
    )
    result = {}
    for field in fields:
        value = row.get(field)
        if field == "model_config" and isinstance(value, str):
            value = json.loads(value)
        result[field] = _canonical_id(value, ids)
    return result


def _canonical_message(row: dict[str, Any], ids: dict[str, str]) -> dict[str, Any]:
    fields = (
        "role",
        "content",
        "tool_name",
        "tool_calls",
        "active",
        "compacted",
        "observed",
        "reasoning",
        "reasoning_content",
        "reasoning_details",
        "finish_reason",
        "token_count",
    )
    return {
        field: _canonical_id(row.get(field), ids) for field in fields
    }


def _canonical_conversation(
    rows: list[dict[str, Any]], ids: dict[str, str]
) -> list[dict[str, Any]]:
    return [
        {
            key: _canonical_id(row.get(key), ids)
            for key in ("role", "content", "tool_name", "tool_calls")
        }
        for row in rows
    ]


def _canonical_search_row(
    row: dict[str, Any], ids: dict[str, str], *, include_fts_text: bool
) -> dict[str, Any]:
    result = {
        key: _canonical_id(row.get(key), ids)
        for key in ("role", "source", "model", "tool_name", "session_id")
    }
    if include_fts_text:
        result["snippet"] = row.get("snippet")
        result["context"] = _canonical_id(row.get("context"), ids)
    else:
        # SQLite FTS5 and PostgreSQL tsvector produce different snippets for a
        # tool_calls-only hit.  Membership, role, source, and marker shape are
        # the portable contract; ranking/snippet text is intentionally not.
        snippet = row.get("snippet") or ""
        result["snippet_has_markers"] = ">>>" in snippet and "<<<" in snippet
    return result


async def _run_backends(tmp_path: Path, case):
    """Run one behavior case against independent SQLite and PostgreSQL stores."""
    dsn = _postgres_dsn()
    sqlite = SQLiteSessionDB(tmp_path / f"state-{uuid.uuid4().hex}.db")
    postgres = PostgresSessionDB(dsn)
    prefix = f"diff-{uuid.uuid4().hex}"
    resources = [[sqlite, []], [postgres, []]]
    try:
        sqlite_result, resources[0][1] = await case(sqlite, prefix)
        postgres_result, resources[1][1] = await case(postgres, prefix)
        assert postgres_result == sqlite_result
    finally:
        for database, session_ids in resources:
            if session_ids:
                await database.delete_sessions(session_ids)
            await database.close()


async def _crud_case(database, prefix: str):
    ids = {"root": f"{prefix}-crud-root"}
    session_id = ids["root"]
    await database.create_session(
        session_id,
        "differential",
        model="model-a",
        model_config={"temperature": 0, "nested": {"keep": True}},
        system_prompt="stable system prompt",
    )
    await database.append_message(session_id, "user", "hello differential")
    await database.append_message(
        session_id,
        "assistant",
        {"answer": "structured"},
        tool_calls=[{"id": "call-1", "type": "function"}],
    )
    await database.append_message(
        session_id, "tool", "tool output", tool_name="shell"
    )
    assert await database.set_session_title(session_id, "Differential title")
    session = await database.get_session(session_id)
    messages = await database.get_messages(session_id)
    conversation = await database.get_messages_as_conversation(session_id)
    exported = await database.export_session(session_id)
    assert session is not None
    assert exported is not None
    return (
        {
            "session": _canonical_session(session, ids),
            "messages": [_canonical_message(row, ids) for row in messages],
            "conversation": _canonical_conversation(conversation, ids),
            "export": {
                "session": _canonical_session(exported, ids),
                "messages": [
                    _canonical_message(row, ids)
                    for row in exported["messages"]
                ],
            },
        },
        list(ids.values()),
    )


async def _search_case(database, prefix: str):
    marker = prefix.replace("-", "")
    source = f"search-source-{marker}"
    content_query = f"{marker} needle phrase"
    tool_query = f"tool-only-{marker}"
    ids = {
        "needle": f"{prefix}-search-needle",
        "tool": f"{prefix}-search-tool",
    }
    await database.create_session(ids["needle"], source, model="m")
    await database.append_message(
        ids["needle"], "user", f"portable {content_query}"
    )
    await database.append_message(ids["needle"], "assistant", "answer")
    await database.create_session(ids["tool"], source, model="m")
    await database.append_message(
        ids["tool"],
        "assistant",
        None,
        tool_calls=[{"function": {"arguments": tool_query}}],
    )

    content_hits = await database.search_messages(content_query)
    tool_hits = await database.search_messages(tool_query)
    session_hits = await database.search_sessions(source=source)
    by_id = await database.search_sessions_by_id(
        source=source, query=ids["needle"]
    )
    listed = await database.list_sessions_rich(
        source=source, project_compression_tips=False
    )
    return (
        {
            "content_hits": [
                _canonical_search_row(row, ids, include_fts_text=True)
                for row in content_hits
            ],
            "tool_hits": [
                _canonical_search_row(row, ids, include_fts_text=False)
                for row in tool_hits
            ],
            "session_hits": [
                _canonical_session(row, ids) | {
                    "id": _canonical_id(row["id"], ids)
                }
                for row in session_hits
            ],
            "by_id": [
                _canonical_session(row, ids) | {
                    "id": _canonical_id(row["id"], ids)
                }
                for row in by_id
            ],
            "listed_ids": [
                _canonical_id(row["id"], ids) for row in listed
            ],
        },
        list(ids.values()),
    )


async def _runtime_lock_case(database, prefix: str):
    ids = {"root": f"{prefix}-runtime-root"}
    session_id = ids["root"]
    await database.create_session(
        session_id,
        "runtime-source",
        model="original",
        model_config={"unrelated": {"keep": True}},
        system_prompt="must be invalidated",
    )
    await database.update_session_runtime_lock(
        session_id,
        provider="provider-a",
        model_options={"temperature": 0.1},
        route_source="runtime",
        confirmed=True,
    )
    session = await database.get_session(session_id)
    assert session is not None
    config = json.loads(session["model_config"])
    config["browser_model_lock"].pop("updated_at", None)
    lock_value = await database.get_session_model_config_value(
        session_id, "browser_model_lock"
    )
    if isinstance(lock_value, dict):
        lock_value = dict(lock_value)
        lock_value.pop("updated_at", None)
    return (
        {
            "model": session["model"],
            "model_config": config,
            "system_prompt": session["system_prompt"],
            "system_prompt_hash": session["system_prompt_hash"],
            "lock_value": lock_value,
        },
        list(ids.values()),
    )


async def _lock_and_meta_case(database, prefix: str):
    ids = {"root": f"{prefix}-lock-root"}
    session_id = ids["root"]
    await database.create_session(session_id, "lock-source")
    await database.set_meta("differential-key", "differential-value")
    acquired = await database.try_acquire_compression_lock(session_id, "owner")
    rejected = await database.try_acquire_compression_lock(session_id, "other")
    holder = await database.get_compression_lock_holder(session_id)
    await database.release_compression_lock(session_id, "owner")
    released = await database.get_compression_lock_holder(session_id)
    return (
        {
            "meta": await database.get_meta("differential-key"),
            "acquired": acquired,
            "rejected": rejected,
            "holder": holder,
            "released": released,
        },
        list(ids.values()),
    )


@pytest.mark.asyncio
async def test_sqlite_and_postgres_crud_differential(tmp_path):
    await _run_backends(tmp_path, _crud_case)


@pytest.mark.asyncio
async def test_sqlite_and_postgres_search_differential(tmp_path):
    await _run_backends(tmp_path, _search_case)


@pytest.mark.asyncio
async def test_sqlite_and_postgres_runtime_lock_differential(tmp_path):
    await _run_backends(tmp_path, _runtime_lock_case)


@pytest.mark.asyncio
async def test_sqlite_and_postgres_compression_lock_and_meta_differential(tmp_path):
    await _run_backends(tmp_path, _lock_and_meta_case)


def test_differential_cases_remain_anchored_to_retained_upstream_tests():
    """Prevent the differential suite from silently drifting from upstream tests."""
    for relative_path, test_name in _RETAINED_UPSTREAM_ANCHORS.values():
        source = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")
        tree = ast.parse(source)
        test_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        }
        assert test_name in test_names, (relative_path, test_name)
