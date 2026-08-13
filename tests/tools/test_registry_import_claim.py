"""Cross-event-loop coordination for built-in tool imports."""

from __future__ import annotations

import asyncio
import threading

from tools import registry as registry_module


def test_builtin_import_claim_coordinates_two_event_loops():
    started = threading.Event()
    waiter_ready = threading.Event()
    release = threading.Event()
    outcomes = []

    async def owner():
        claim, owns_claim = registry_module._claim_builtin_import()
        assert owns_claim is True
        started.set()
        while not release.is_set():
            await asyncio.sleep(0)
        registry_module._finish_builtin_import_claim(claim, completed=True)

    async def waiter():
        claim, owns_claim = registry_module._claim_builtin_import()
        assert owns_claim is False
        waiter_ready.set()
        outcomes.append(await asyncio.shield(asyncio.wrap_future(claim)))

    owner_thread = threading.Thread(target=lambda: asyncio.run(owner()))
    owner_thread.start()
    assert started.wait(timeout=2)
    waiter_thread = threading.Thread(target=lambda: asyncio.run(waiter()))
    waiter_thread.start()
    assert waiter_ready.wait(timeout=2)
    release.set()
    owner_thread.join(timeout=2)
    waiter_thread.join(timeout=2)

    assert owner_thread.is_alive() is False
    assert waiter_thread.is_alive() is False
    assert outcomes == [True]
    assert registry_module._builtin_import_claim is None


def test_cancelled_owner_releases_claim_for_retry():
    claim, owner = registry_module._claim_builtin_import()
    assert owner is True
    registry_module._finish_builtin_import_claim(claim, completed=False)

    replacement, replacement_owner = registry_module._claim_builtin_import()
    try:
        assert replacement_owner is True
        assert replacement is not claim
    finally:
        registry_module._finish_builtin_import_claim(replacement, completed=True)
