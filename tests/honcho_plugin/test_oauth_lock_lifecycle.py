"""Event-loop and path ownership for Honcho OAuth refresh serialization."""

from __future__ import annotations

import asyncio
import gc
import weakref

import pytest

from plugins.memory.honcho import oauth


def test_refresh_lock_does_not_cross_or_retain_event_loops(tmp_path):
    loop_refs = []
    lock_refs = []

    async def use_lock():
        loop_refs.append(weakref.ref(asyncio.get_running_loop()))
        lock = await oauth._refresh_lock(tmp_path / "honcho.json")
        lock_refs.append(weakref.ref(lock))
        async with lock:
            await asyncio.sleep(0)

    asyncio.run(use_lock())
    asyncio.run(use_lock())
    gc.collect()

    assert loop_refs[0]() is None
    assert lock_refs[0]() is None


@pytest.mark.asyncio
async def test_refresh_lock_canonicalizes_symlink_aliases(tmp_path):
    config = tmp_path / "honcho.json"
    alias = tmp_path / "honcho-alias.json"
    config.write_text("{}", encoding="utf-8")
    alias.symlink_to(config)

    first = await oauth._refresh_lock(config)
    second = await oauth._refresh_lock(alias)

    assert first is second


@pytest.mark.asyncio
async def test_different_config_files_do_not_share_refresh_lock(tmp_path):
    first = await oauth._refresh_lock(tmp_path / "a.json")
    second = await oauth._refresh_lock(tmp_path / "b.json")

    assert first is not second
