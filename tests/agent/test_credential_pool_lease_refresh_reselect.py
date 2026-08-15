"""acquire_lease must wait for a native single-use-token refresh.

The upstream regression was found when refresh moved outside its synchronous
pool lock.  This fork performs the refresh at an awaited
``_available_entries`` boundary, so the tests preserve the behavior assertion
while using the native async seam.
"""

import asyncio
import threading

import pytest

from agent.credential_pool import CredentialPool, PooledCredential


def _entry(entry_id: str) -> PooledCredential:
    return PooledCredential(
        id=entry_id,
        provider="anthropic",
        auth_type="oauth",
        access_token="tok",
        label=entry_id,
        source="oauth",
        priority=0,
    )


def _bare_pool(entries):
    """Minimal pool shell — avoids disk/keyring I/O in __init__."""
    pool = CredentialPool.__new__(CredentialPool)
    pool._lock = asyncio.Lock()
    pool._entries = list(entries)
    pool._active_leases = {}
    pool._current_id = None
    pool._max_concurrent = 2
    pool._unmatched_rotation_streak = 0
    pool.provider = "anthropic"
    return pool


def _wire_native_refresh(pool, *, refresh_succeeds: bool = True):
    """Model the native async refresh contract with an explicit state flag.

    The upstream implementation deferred single-use refreshes outside a
    synchronous pool lock.  This fork performs the same refresh at the
    awaited ``_available_entries`` boundary, so the test keeps the behavior
    assertion while adapting the private seam to the native implementation.
    """
    state = {"needs_refresh": True, "refresh_calls": 0}

    async def fake_available(
        *,
        clear_expired=False,
        allow_refresh=True,
        lock_held=False,
    ):
        if state["needs_refresh"]:
            state["refresh_calls"] += 1
            await asyncio.sleep(0)
            if refresh_succeeds and allow_refresh:
                state["needs_refresh"] = False
            else:
                return []
        return list(pool._entries)

    pool._available_entries = fake_available
    return state


@pytest.mark.asyncio
async def test_acquire_lease_waits_for_native_refresh_before_leasing():
    """The only entry needs a refresh; once refreshed it is available, so a
    lease MUST be granted rather than reporting no credentials."""
    pool = _bare_pool([_entry("e1")])
    state = _wire_native_refresh(pool)

    lease = await pool.acquire_lease()

    assert state["refresh_calls"] == 1, "the deferred refresh should run once"
    assert state["needs_refresh"] is False, "entry is available post-refresh"
    assert lease == "e1", (
        "acquire_lease returned None despite a successfully refreshed, "
        "available entry — the caller would fail an answerable request"
    )
    assert pool._active_leases.get("e1") == 1, "the lease must be recorded"


@pytest.mark.asyncio
async def test_acquire_lease_without_refresh_does_not_double_select():
    """No pending refresh -> exactly one selection pass (no wasted work)."""
    pool = _bare_pool([_entry("e1")])
    state = _wire_native_refresh(pool)
    state["needs_refresh"] = False  # already healthy

    passes = {"n": 0}
    original = pool._available_entries

    async def counting(**kwargs):
        passes["n"] += 1
        return await original(**kwargs)

    pool._available_entries = counting

    lease = await pool.acquire_lease()

    assert lease == "e1"
    assert passes["n"] == 1, "healthy pool must not trigger a retry path"
    assert state["refresh_calls"] == 0


@pytest.mark.asyncio
async def test_acquire_lease_still_none_when_refresh_does_not_help():
    """If the refresh leaves nothing available, None is still the answer —
    the retry must not loop or invent a credential."""
    pool = _bare_pool([_entry("e1")])
    state = _wire_native_refresh(pool, refresh_succeeds=False)

    assert await pool.acquire_lease() is None
    assert state["refresh_calls"] == 1, "refresh must run once"
    assert pool._active_leases == {}
