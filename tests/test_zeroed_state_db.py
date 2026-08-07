"""Native-async coverage for zeroed state database quarantine."""

from __future__ import annotations

import asyncio
import sqlite3

import pytest
from blockbuster import BlockBuster

import hermes_state


@pytest.mark.asyncio
async def test_is_zeroed_state_db_and_quarantine(tmp_path):
    path = tmp_path / "state.db"
    original = bytes(1024)
    path.write_bytes(original)

    blockbuster = BlockBuster()
    blockbuster.activate()
    try:
        assert await hermes_state.is_zeroed_state_db(path) is True
        quarantine = await hermes_state.quarantine_zeroed_state_db(path)
    finally:
        blockbuster.deactivate()

    assert quarantine is not None
    assert quarantine.exists()
    assert not path.exists()
    assert quarantine.read_bytes() == original


@pytest.mark.asyncio
async def test_sessiondb_opens_fresh_after_zeroed_quarantine(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "state.db"
    path.write_bytes(bytes(4096))
    database = hermes_state.SessionDB(path)

    blockbuster = BlockBuster()
    blockbuster.activate()
    try:
        await database.ensure_session("fresh")
        assert await hermes_state.is_zeroed_state_db(path) is False
    finally:
        blockbuster.deactivate()
        await database.close()

    backups = list(tmp_path.glob("state.db.zeroed-*.bak"))
    assert len(backups) == 1
    assert backups[0].stat().st_size == 4096


@pytest.mark.asyncio
async def test_concurrent_sessiondb_quarantine_does_not_clobber(tmp_path):
    path = tmp_path / "state.db"
    path.write_bytes(bytes(4096))

    async def open_database(session_id: str) -> None:
        database = hermes_state.SessionDB(path)
        try:
            await database.ensure_session(session_id)
        finally:
            await database.close()

    blockbuster = BlockBuster()
    blockbuster.activate()
    try:
        await asyncio.gather(open_database("one"), open_database("two"))
        assert await hermes_state.is_zeroed_state_db(path) is False
    finally:
        blockbuster.deactivate()

    backups = list(tmp_path.glob("state.db.zeroed-*.bak"))
    assert len(backups) == 1
    assert backups[0].stat().st_size == 4096
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT 1").fetchone() == (1,)
    finally:
        connection.close()


@pytest.mark.asyncio
async def test_open_connection_skips_zeroed_probe(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    database = hermes_state.SessionDB(path)
    await database.ensure_session("live")

    async def fail_open(*args, **kwargs):
        raise AssertionError("live database must not be probed as a raw file")

    monkeypatch.setattr(hermes_state.aiofiles, "open", fail_open)
    try:
        assert await hermes_state.is_zeroed_state_db(path) is False
    finally:
        await database.close()
