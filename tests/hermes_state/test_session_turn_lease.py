"""Cross-process turn lease behavior for the native-async SQLite backend."""

from __future__ import annotations

import asyncio

import pytest

from hermes_state import SessionDB


@pytest.mark.asyncio
async def test_session_turn_lease_acquire_refresh_and_release(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        await db.create_session("session", source="cli")

        assert await db.try_acquire_session_turn_lease("session", "holder-a") is True
        assert await db.try_acquire_session_turn_lease("session", "holder-b") is False
        assert await db.refresh_session_turn_lease("session", "holder-b") is False
        assert await db.refresh_session_turn_lease("session", "holder-a") is True

        await db.release_session_turn_lease("session", "holder-b")
        assert await db.try_acquire_session_turn_lease("session", "holder-b") is False
        await db.release_session_turn_lease("session", "holder-a")
        assert await db.try_acquire_session_turn_lease("session", "holder-b") is True
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_session_turn_lease_wait_is_cancellable(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        await db.create_session("session", source="cli")
        assert await db.try_acquire_session_turn_lease("session", "holder-a") is True

        waiter = asyncio.create_task(
            db.acquire_session_turn_lease(
                "session",
                "holder-b",
                wait_seconds=30.0,
                poll_interval_seconds=0.01,
            )
        )
        await asyncio.sleep(0.03)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
    finally:
        await db.release_session_turn_lease("session", "holder-a")
        await db.close()


@pytest.mark.asyncio
async def test_session_turn_lease_uses_compression_lineage_root(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        await db.create_session("root", source="cli")
        await db.create_session("child", source="cli", parent_session_id="root")
        await db.end_session("root", "compression")

        assert await db.try_acquire_session_turn_lease("child", "holder-a") is True
        assert await db.try_acquire_session_turn_lease("root", "holder-b") is False

        await db.release_session_turn_lease("root", "holder-a")
    finally:
        await db.close()
