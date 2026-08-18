"""Async ownership semantics for the retained SessionDB context protocol."""

from pathlib import Path

import pytest

from hermes_state import SessionDB, _live_connection_counts


@pytest.mark.asyncio
async def test_async_context_manager_returns_and_closes(tmp_path: Path):
    db_path = tmp_path / "state.db"
    async with SessionDB(db_path) as database:
        assert database is not None
        await database.create_session("session-1", "test", model="test")
        tracking_key = database._connection_tracking_key
        assert tracking_key is not None
    assert database._closed
    assert database._connection is None
    assert tracking_key not in _live_connection_counts


@pytest.mark.asyncio
async def test_async_context_manager_preserves_exception(tmp_path: Path):
    database = SessionDB(tmp_path / "state.db")
    with pytest.raises(RuntimeError, match="preserved"):
        async with database:
            await database.session_count()
            raise RuntimeError("preserved")
    assert database._closed


@pytest.mark.asyncio
async def test_async_context_manager_is_safe_after_early_close(tmp_path: Path):
    async with SessionDB(tmp_path / "state.db") as database:
        await database.session_count()
        await database.close()
    assert database._closed
