"""Tests for native-async plugin singleton helpers."""

import asyncio
import queue
import threading

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


@pytest.mark.asyncio
async def test_slot_reset_finishes_after_repeated_cancellation():
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
    reset_task.cancel()
    await asyncio.sleep(0)
    reset_task.cancel()
    assert not reset_task.done()

    release_build.set()
    assert await build_task == "built"
    with pytest.raises(asyncio.CancelledError):
        await reset_task
    assert slot.peek() is None


def test_slot_serializes_concurrent_first_call_across_event_loops():
    slot: SingletonSlot[object] = SingletonSlot()
    owner_started = threading.Event()
    release_owner = threading.Event()
    results: queue.Queue[tuple[object | None, BaseException | None]] = queue.Queue()
    build_count = 0
    count_guard = threading.Lock()

    async def factory():
        nonlocal build_count
        with count_guard:
            build_count += 1
        owner_started.set()
        while not release_owner.is_set():
            await asyncio.sleep(0.001)
        return object()

    def run() -> None:
        try:
            results.put((asyncio.run(slot.get(factory)), None))
        except BaseException as exc:  # pragma: no cover - assertion reports it
            results.put((None, exc))

    first = threading.Thread(target=run)
    second = threading.Thread(target=run)
    first.start()
    assert owner_started.wait(timeout=2)
    second.start()
    release_owner.set()
    first.join(timeout=2)
    second.join(timeout=2)

    first_result, first_error = results.get_nowait()
    second_result, second_error = results.get_nowait()
    assert first_error is second_error is None
    assert first_result is second_result
    assert build_count == 1
