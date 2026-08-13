"""Event-loop and physical-path ownership for auth-store locks."""

from __future__ import annotations

import asyncio
import gc
import weakref

import pytest

from hermes_cli import auth


def test_auth_store_lock_does_not_retain_closed_event_loop(tmp_path):
    loop_refs = []
    lock_refs = []

    async def use_lock():
        loop_refs.append(weakref.ref(asyncio.get_running_loop()))
        lock = await auth._auth_store_lock_for(tmp_path / "auth.json")
        lock_refs.append(weakref.ref(lock))
        async with lock:
            await asyncio.sleep(0)

    asyncio.run(use_lock())
    asyncio.run(use_lock())
    gc.collect()

    assert loop_refs[0]() is None
    assert lock_refs[0]() is None


@pytest.mark.asyncio
async def test_auth_store_lock_canonicalizes_symlink_aliases(tmp_path):
    store = tmp_path / "auth.json"
    alias = tmp_path / "auth-alias.json"
    store.write_text("{}", encoding="utf-8")
    alias.symlink_to(store)

    first = await auth._auth_store_lock_for(store)
    second = await auth._auth_store_lock_for(alias)

    assert first is second
