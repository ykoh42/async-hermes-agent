"""Parity tests for the upstream generic hidden-session flag."""

import pytest
import pytest_asyncio

from hermes_state import SessionDB


@pytest_asyncio.fixture
async def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_hidden_excluded_by_default_included_on_request(db):
    await db.create_session("visible", source="cli")
    await db.create_session("secret", source="cli")
    for session_id in ("visible", "secret"):
        await db.append_message(
            session_id=session_id,
            role="user",
            content="hello",
        )

    assert await db.set_session_hidden("secret", True) is True
    assert (await db.get_session("secret"))["hidden"] == 1
    assert (await db.get_session("visible"))["hidden"] == 0

    default_ids = {
        row["id"]
        for row in await db.list_sessions_rich(min_message_count=1)
    }
    assert default_ids == {"visible"}

    all_ids = {
        row["id"]
        for row in await db.list_sessions_rich(
            min_message_count=1,
            include_hidden=True,
        )
    }
    assert all_ids == {"visible", "secret"}

    assert await db.set_session_hidden("secret", False) is True
    assert (await db.get_session("secret"))["hidden"] == 0
    unhidden_ids = {
        row["id"]
        for row in await db.list_sessions_rich(min_message_count=1)
    }
    assert unhidden_ids == {"visible", "secret"}
