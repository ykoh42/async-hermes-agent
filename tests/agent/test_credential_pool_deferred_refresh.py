"""Thread-safety of the deferred single-use-token refresh path (#71775).

The deferred path deliberately runs OAuth network I/O outside the pool
lock. These tests pin the two invariants that make that safe:

1. `select()` does NOT hold the pool lock while the deferred refresh's
   network call runs (the whole point of the PR).
2. The pool mutations that follow the network call (`_replace_entry`,
   `_persist`) DO re-serialize under the pool lock, so a concurrent
   `select()`/rotation cannot tear `self._entries` or double-write
   auth.json.
"""

import asyncio
from dataclasses import replace

import pytest

from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    CredentialPool,
    PooledCredential,
)


def _codex_entry(entry_id: str = "codex-1") -> PooledCredential:
    return PooledCredential(
        provider="openai-codex",
        id=entry_id,
        label="test codex",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="device_code",
        access_token="at-stale",
        refresh_token="rt-stale",
        expires_at_ms=1,  # long expired -> needs refresh
    )


@pytest.mark.asyncio
async def test_select_does_not_hold_pool_lock_during_deferred_refresh(monkeypatch):
    pool = CredentialPool("openai-codex", [_codex_entry()])
    lock_free_during_refresh = {}

    async def _fake_refresh(entry, *, force):
        async with pool._lock:
            lock_free_during_refresh["value"] = True
        refreshed = replace(entry, access_token="at-fresh", expires_at_ms=2**53)
        pool._replace_entry(entry, refreshed)
        return refreshed

    monkeypatch.setattr(
        pool, "_entry_needs_refresh", lambda e: e.access_token == "at-stale"
    )
    monkeypatch.setattr(pool, "_refresh_entry", _fake_refresh)
    async def _no_persist(**_kwargs):
        return None

    monkeypatch.setattr(pool, "_persist", _no_persist)

    selected = await pool.select()

    assert lock_free_during_refresh.get("value") is True, (
        "select() held the pool lock during the deferred refresh network window"
    )
    assert selected is not None
    assert selected.access_token == "at-fresh"


@pytest.mark.asyncio
async def test_deferred_refreshes_are_serialized(monkeypatch):
    """Concurrent selectors share one refresh and observe one final entry."""
    pool = CredentialPool("openai-codex", [_codex_entry()])
    calls = 0

    async def _fake_refresh(entry, *, force):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        refreshed = replace(entry, access_token="at-fresh", expires_at_ms=2**53)
        pool._replace_entry(entry, refreshed)
        return refreshed

    monkeypatch.setattr(
        pool, "_entry_needs_refresh", lambda entry: entry.access_token == "at-stale"
    )
    monkeypatch.setattr(pool, "_refresh_entry", _fake_refresh)

    first, second = await asyncio.gather(pool.select(), pool.select())

    assert calls == 1
    assert first is not None and first.access_token == "at-fresh"
    assert second is not None and second.access_token == "at-fresh"
    assert pool._entries[0].access_token == "at-fresh"
