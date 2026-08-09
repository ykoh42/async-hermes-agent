"""Tests for native-async plugin singleton helpers."""

import asyncio

import pytest

from plugins.plugin_utils import SingletonSlot, lazy_singleton


@pytest.mark.asyncio
async def test_lazy_singleton_builds_once_and_returns_same_instance():
    calls = []

    @lazy_singleton
    async def get():
        calls.append(1)
        return object()

    first = await get()
    second = await get()
    assert first is second
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_lazy_singleton_reset_rebuilds():
    counter = {"n": 0}

    @lazy_singleton
    async def get():
        counter["n"] += 1
        return counter["n"]

    assert await get() == 1
    assert await get() == 1
    await get.reset()
    assert await get() == 2


@pytest.mark.asyncio
async def test_lazy_singleton_concurrent_first_call_builds_once():
    build_count = 0
    release_build = asyncio.Event()

    @lazy_singleton
    async def get():
        nonlocal build_count
        build_count += 1
        await release_build.wait()
        return object()

    tasks = [asyncio.create_task(get()) for _ in range(16)]
    await asyncio.sleep(0)
    release_build.set()
    results = await asyncio.gather(*tasks)

    assert build_count == 1
    assert all(result is results[0] for result in results)


def test_lazy_singleton_rejects_sync_factory():
    with pytest.raises(TypeError, match="must be async"):

        @lazy_singleton
        def get():
            return object()


@pytest.mark.asyncio
async def test_slot_caches_first_value():
    slot: SingletonSlot[str] = SingletonSlot()
    assert slot.peek() is None

    async def first():
        return "first"

    first_value = await slot.get(first)
    assert slot.peek() == "first"

    async def second():
        return "second"

    second_value = await slot.get(second)
    assert first_value == second_value == "first"


@pytest.mark.asyncio
async def test_slot_factory_exception_not_cached():
    slot: SingletonSlot[str] = SingletonSlot()

    async def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        await slot.get(boom)
    assert slot.peek() is None

    async def recovered():
        return "recovered"

    assert await slot.get(recovered) == "recovered"


@pytest.mark.asyncio
async def test_slot_factory_cancellation_not_cached():
    slot: SingletonSlot[str] = SingletonSlot()
    started = asyncio.Event()

    async def wait_forever():
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    task = asyncio.create_task(slot.get(wait_forever))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert slot.peek() is None

    async def recovered():
        return "recovered"

    assert await slot.get(recovered) == "recovered"


@pytest.mark.asyncio
async def test_slot_concurrent_first_call_builds_once():
    build_count = 0
    release_build = asyncio.Event()
    slot: SingletonSlot[object] = SingletonSlot()

    async def factory():
        nonlocal build_count
        build_count += 1
        await release_build.wait()
        return object()

    tasks = [asyncio.create_task(slot.get(factory)) for _ in range(16)]
    await asyncio.sleep(0)
    release_build.set()
    results = await asyncio.gather(*tasks)

    assert build_count == 1
    assert all(result is results[0] for result in results)


@pytest.mark.asyncio
async def test_slot_reset_waits_for_inflight_build():
    slot: SingletonSlot[str] = SingletonSlot()
    started = asyncio.Event()
    release_build = asyncio.Event()

    async def factory():
        started.set()
        await release_build.wait()
        return "built"

    build_task = asyncio.create_task(slot.get(factory))
    await started.wait()
    reset_task = asyncio.create_task(slot.reset())
    await asyncio.sleep(0)
    assert reset_task.done() is False

    release_build.set()
    assert await build_task == "built"
    await reset_task
    assert slot.peek() is None
