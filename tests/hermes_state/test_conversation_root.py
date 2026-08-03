"""Tests for SessionDB.get_conversation_root — stable conversation id resolution.

The conversation root is the Nous Portal ``conversation=`` tag value: one
stable id per user-facing conversation, surviving context-compression
session rotation and covering delegate subagent trees.
"""
import pytest
import pytest_asyncio

from hermes_state import SessionDB


@pytest_asyncio.fixture
async def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_root_of_standalone_session_is_itself(db):
    await db.create_session("solo", source="cli")
    assert await db.get_conversation_root("solo") == "solo"






@pytest.mark.asyncio
async def test_root_covers_delegate_child_sessions(db):
    await db.create_session("parent", source="cli")
    await db.create_session("child", source="delegate", parent_session_id="parent")
    assert await db.get_conversation_root("child") == "parent"



