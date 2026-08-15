"""Async-state ports of the upstream SQL LIKE wildcard regression tests."""

from __future__ import annotations

import pytest

from hermes_state import SessionDB, _cwd_prefix_clause
from hermes_state_common import escape_like


def test_escape_like_quotes_sql_wildcards_and_backslashes():
    assert escape_like(r"user_%\draft") == r"user\_\%\\draft"


def test_cwd_prefix_clause_escapes_path_wildcards():
    clause, params = _cwd_prefix_clause("/tmp/my_project")
    assert "ESCAPE '\\'" in clause
    assert params == [
        "/tmp/my_project",
        "/tmp/my\\_project/%",
        "/tmp/my\\_project\\\\%",
    ]


def test_prune_filter_where_escapes_substring_filters():
    clause, params = SessionDB._prune_filter_where(
        title_like="user_auth",
        model_like="model_mini",
        branch_like="session_prune",
    )
    assert clause.count("ESCAPE '\\'") == 3
    assert params == ["%user\\_auth%", "%model\\_mini%", "%session\\_prune%"]


@pytest.mark.asyncio
async def test_title_like_underscore_is_literal(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        for sid, title in (
            ("target", "user_auth refactor"),
            ("bystander1", "user-auth review"),
            ("bystander2", "userXauth notes"),
        ):
            await db.create_session(session_id=sid, source="library")
            await db.set_session_title(sid, title)
            await db.end_session(sid, end_reason="done")
        rows = await db.list_prune_candidates(title_like="user_auth")
        assert {row["id"] for row in rows} == {"target"}
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_percent_filter_does_not_select_everything(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        for sid, title in (("a", "alpha"), ("b", "beta"), ("pct", "100% coverage")):
            await db.create_session(session_id=sid, source="library")
            await db.set_session_title(sid, title)
            await db.end_session(sid, end_reason="done")
        rows = await db.list_prune_candidates(title_like="%")
        assert {row["id"] for row in rows} == {"pct"}
    finally:
        await db.close()
