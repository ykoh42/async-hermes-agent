"""Event-loop ownership for the cua-driver feature-probe lock."""

from __future__ import annotations

import asyncio
import gc
import weakref

from tools.computer_use import cua_backend


def test_support_probe_lock_does_not_cross_or_retain_event_loops():
    loop_refs = []
    lock_refs = []

    async def use_lock():
        loop_refs.append(weakref.ref(asyncio.get_running_loop()))
        lock = cua_backend._no_overlay_lock("cua-driver")
        lock_refs.append(weakref.ref(lock))
        async with lock:
            await asyncio.sleep(0)

    asyncio.run(use_lock())
    asyncio.run(use_lock())
    gc.collect()

    assert loop_refs[0]() is None
    assert lock_refs[0]() is None
