"""Appends flow freely during compression; the commit preserves them (#75316)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import pytest_asyncio

from hermes_state import (
    CompressionSessionBusyError,
    SessionCompressionInProgressError,
    SessionDB,
)


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> SessionDB:
    database = SessionDB(tmp_path / "state.db")
    await database.create_session("sess1", source="test")
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_append_is_never_blocked_by_a_foreign_compression_lock(
    db: SessionDB,
) -> None:
    """A steer landing during compression persists immediately."""
    assert await db.try_acquire_compression_lock("sess1", "compressor") is True
    started = time.monotonic()
    await db.append_message("sess1", role="user", content="steered mid-compression")
    assert time.monotonic() - started < 0.5
    assert any(
        row["content"] == "steered mid-compression"
        for row in await db.get_messages("sess1")
    )


@pytest.mark.asyncio
async def test_append_is_never_blocked_by_a_stale_lock(db: SessionDB) -> None:
    """A crashed compressor's unexpired lock does not fence writes."""
    assert (
        await db.try_acquire_compression_lock(
            "sess1", "pid-9999999-long-gone", ttl_seconds=3600
        )
        is True
    )
    started = time.monotonic()
    await db.append_message("sess1", role="user", content="lands despite stale lock")
    assert time.monotonic() - started < 0.5
    assert any(
        row["content"] == "lands despite stale lock"
        for row in await db.get_messages("sess1")
    )


@pytest.mark.asyncio
async def test_the_lock_owner_append_still_works(db: SessionDB) -> None:
    assert await db.try_acquire_compression_lock("sess1", "compressor") is True
    await db.append_message(
        "sess1",
        role="assistant",
        content="written by the compressor",
        compression_lock_holder="compressor",
    )
    assert any(
        row["content"] == "written by the compressor"
        for row in await db.get_messages("sess1")
    )


def test_transient_error_is_a_subclass_of_the_original() -> None:
    assert issubclass(SessionCompressionInProgressError, CompressionSessionBusyError)


@pytest.mark.asyncio
async def test_no_lock_means_no_delay(db: SessionDB) -> None:
    started = time.monotonic()
    await db.append_message("sess1", role="user", content="uncontended")
    assert time.monotonic() - started < 0.5


@pytest.mark.asyncio
async def test_a_lost_compression_lease_still_fails_fast(db: SessionDB) -> None:
    """A lost compression lease remains a permanent failure."""
    started = time.monotonic()
    with pytest.raises(CompressionSessionBusyError):
        await db.publish_compression_child(
            parent_session_id="sess1",
            child_session_id="child1",
            source="test",
            messages=[{"role": "user", "content": "compacted"}],
            compression_lock_holder="not-the-holder",
            require_compression_lease=True,
        )
    assert time.monotonic() - started < 0.5
