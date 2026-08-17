"""Explicit branch sessions own their copied transcript."""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio

from hermes_state import SessionDB


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> SessionDB:
    database = SessionDB(tmp_path / "state.db")
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_branch_resume_does_not_include_parent_rows_added_after_fork(
    db: SessionDB,
) -> None:
    await db.create_session("parent", source="tui")
    await db.append_message("parent", role="user", content="before branch")
    await db.append_message("parent", role="assistant", content="initial answer")
    await db.create_session(
        "branch",
        source="tui",
        parent_session_id="parent",
        model_config={"_branched_from": "parent"},
    )
    await db.append_message("branch", role="user", content="before branch")
    await db.append_message("branch", role="assistant", content="initial answer")
    await db.append_message("parent", role="user", content="after branch")
    await db.append_message("parent", role="assistant", content="later answer")

    _, display_history = await db.get_resume_conversations("branch")

    assert [message["content"] for message in display_history] == [
        "before branch",
        "initial answer",
    ]
    assert [
        message["content"]
        for message in await db.get_messages_as_conversation(
            "branch", include_ancestors=True
        )
    ] == ["before branch", "initial answer"]
    assert await db.get_ancestor_display_prefix("branch") == []
