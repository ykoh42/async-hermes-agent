"""Every durable delegation operation closes its aiosqlite connection."""

from __future__ import annotations

import asyncio

import pytest

from tools import async_delegation as ad

pytestmark = pytest.mark.asyncio


class _TrackingConnection:
    def __init__(self, real, closed):
        self._real = real
        self._closed = closed

    async def close(self):
        self._closed.append(id(self._real))
        await self._real.close()

    def __getattr__(self, name):
        return getattr(self._real, name)


async def test_ledger_operations_close_every_connection(monkeypatch, tmp_path):
    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    opened: list[int] = []
    closed: list[int] = []
    real_connect = ad.aiosqlite.connect

    async def tracking_connect(*args, **kwargs):
        connection = await real_connect(*args, **kwargs)
        opened.append(id(connection))
        return _TrackingConnection(connection, closed)

    monkeypatch.setattr(ad.aiosqlite, "connect", tracking_connect)
    await ad.get_durable_delegation("nope")
    await ad.recover_abandoned_delegations()
    await ad.restore_undelivered_completions(asyncio.Queue())
    await ad.mark_completion_delivered("nope")
    await ad.claim_completion_delivery("nope", "claim-1")
    assert opened
    assert opened == closed


async def test_schema_init_failure_closes_connection(monkeypatch, tmp_path):
    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    opened: list[int] = []
    closed: list[int] = []
    real_connect = ad.aiosqlite.connect

    async def tracking_connect(*args, **kwargs):
        connection = await real_connect(*args, **kwargs)
        opened.append(id(connection))
        return _TrackingConnection(connection, closed)

    async def fail_schema(_connection):
        raise RuntimeError("simulated schema init failure")

    monkeypatch.setattr(ad.aiosqlite, "connect", tracking_connect)
    monkeypatch.setattr(ad, "_initialize_schema", fail_schema)
    with pytest.raises(RuntimeError, match="simulated schema"):
        await ad._connect()
    assert opened == closed
