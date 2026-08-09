"""Cancellation contracts for SQLite connection ownership."""

import asyncio

import pytest

from hermes_state import SessionDB, _live_connection_counts


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_connection_close():
    """A repeated caller cancellation cannot detach connection.close()."""
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    close_completed = asyncio.Event()

    class ControlledConnection:
        async def close(self):
            close_started.set()
            await release_close.wait()
            close_completed.set()

    tracking_key = "repeated-cancel-close"
    database = SessionDB("unused-state.db")
    database._connection = ControlledConnection()
    database._connection_tracking_key = tracking_key
    _live_connection_counts[tracking_key] = 1

    task = asyncio.create_task(database.close())
    await close_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)

    try:
        assert task.done() is False
    finally:
        release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(close_completed.wait(), timeout=1.0)
        _live_connection_counts.pop(tracking_key, None)

    assert database._connection is None
    assert tracking_key not in _live_connection_counts
