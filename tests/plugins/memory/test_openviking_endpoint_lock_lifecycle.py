"""OpenViking endpoint-safety lock lifecycle and profile semantics."""

import asyncio
import gc
import weakref

import pytest

import plugins.memory.openviking as openviking
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture(autouse=True)
def _clear_endpoint_safety_cache():
    openviking._clear_openviking_endpoint_safety_cache()
    yield
    openviking._clear_openviking_endpoint_safety_cache()


@pytest.mark.asyncio
async def test_endpoint_safety_waiter_cancellation_does_not_poison_cache(monkeypatch):
    import tools.url_safety as url_safety

    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def allow_url(_value):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return False

    monkeypatch.setattr(url_safety, "is_always_blocked_url", allow_url)
    endpoint = "https://cancelled-waiter.openviking.example.test"
    owner = asyncio.create_task(
        openviking._openviking_endpoint_is_always_blocked(endpoint)
    )
    await started.wait()
    waiter = asyncio.create_task(
        openviking._openviking_endpoint_is_always_blocked(endpoint)
    )
    await asyncio.sleep(0)

    waiter.cancel()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    release.set()
    assert await owner is False
    assert await openviking._openviking_endpoint_is_always_blocked(endpoint) is False
    assert calls == 1


@pytest.mark.asyncio
async def test_endpoint_safety_owner_cancellation_releases_lock_and_retries(monkeypatch):
    import tools.url_safety as url_safety

    first_started = asyncio.Event()
    calls = 0

    async def allow_url(_value):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_started.set()
            await asyncio.Future()
        return False

    monkeypatch.setattr(url_safety, "is_always_blocked_url", allow_url)
    endpoint = "https://cancelled-owner.openviking.example.test"
    owner = asyncio.create_task(
        openviking._openviking_endpoint_is_always_blocked(endpoint)
    )
    await first_started.wait()
    waiter = asyncio.create_task(
        openviking._openviking_endpoint_is_always_blocked(endpoint)
    )
    await asyncio.sleep(0)

    owner.cancel()
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner

    assert await waiter is False
    assert await openviking._openviking_endpoint_is_always_blocked(endpoint) is False
    assert calls == 2


def test_endpoint_safety_lock_survives_contended_sequential_asyncio_runs(monkeypatch):
    import tools.url_safety as url_safety

    calls = []

    async def allow_url(value):
        calls.append(value)
        await asyncio.sleep(0)
        return False

    async def contend(endpoint):
        return await asyncio.gather(
            openviking._openviking_endpoint_is_always_blocked(endpoint),
            openviking._openviking_endpoint_is_always_blocked(endpoint),
        )

    monkeypatch.setattr(url_safety, "is_always_blocked_url", allow_url)
    first = "https://first-loop.openviking.example.test"
    second = "https://second-loop.openviking.example.test"

    assert asyncio.run(contend(first)) == [False, False]
    assert asyncio.run(contend(second)) == [False, False]
    assert calls == [first, second]


@pytest.mark.asyncio
async def test_endpoint_safety_cache_is_shared_across_profiles(monkeypatch, tmp_path):
    import tools.url_safety as url_safety

    calls = []

    async def allow_url(value):
        calls.append(value)
        await asyncio.sleep(0)
        return False

    async def check_from(profile, endpoint):
        token = set_hermes_home_override(profile)
        try:
            return await openviking._openviking_endpoint_is_always_blocked(endpoint)
        finally:
            reset_hermes_home_override(token)

    monkeypatch.setattr(url_safety, "is_always_blocked_url", allow_url)
    endpoint = "https://shared.openviking.example.test"

    assert await asyncio.gather(
        check_from(tmp_path / "profile-a", endpoint),
        check_from(tmp_path / "profile-b", endpoint),
    ) == [False, False]
    assert calls == [endpoint]


def test_endpoint_safety_lock_does_not_retain_closed_loop():
    loop = asyncio.new_event_loop()
    loop_ref = weakref.ref(loop)

    async def contend_lock():
        lock = openviking._endpoint_safety_lock()
        await lock.acquire()
        waiter = asyncio.create_task(lock.acquire())
        await asyncio.sleep(0)
        lock.release()
        await waiter
        lock.release()

    loop.run_until_complete(contend_lock())
    loop.close()
    del loop
    gc.collect()

    assert loop_ref() is None
