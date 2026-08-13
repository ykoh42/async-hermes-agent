"""Profile and event-loop ownership for the models.dev cache."""

from __future__ import annotations

import asyncio
import gc
import json
import threading
import time
import weakref
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agent import models_dev as md
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _reset_models_dev_state() -> None:
    with md._models_dev_update_guard:
        md._models_dev_profile_states.clear()
        md._models_dev_profile_aliases.clear()
    md._models_dev_profile_context.set(None)
    md._models_dev_cache = {}
    md._models_dev_cache_time = 0
    md._models_dev_retry_after = 0
    md._models_dev_lock = None
    md._models_dev_refresh_task = None
    md._models_dev_update_claim = None
    md._models_dev_legacy_snapshot = (
        id(md._models_dev_cache),
        0,
        0,
        None,
        None,
    )


@pytest.fixture(autouse=True)
def isolated_models_dev_state():
    _reset_models_dev_state()
    yield
    for state in md._models_dev_profile_states.values():
        assert state.refresh_task is None or state.refresh_task.done()
        assert state.refresh_claim is None
        assert state.update_claim is None or state.update_claim.done()
    _reset_models_dev_state()


@pytest.fixture
def profiles(tmp_path):
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    profile_a.mkdir()
    profile_b.mkdir()
    alias_a = tmp_path / "profile-a-alias"
    alias_a.symlink_to(profile_a, target_is_directory=True)
    return profile_a, profile_b, alias_a


async def _run_in_profile(home: Path, awaitable):
    token = set_hermes_home_override(home)
    try:
        return await awaitable()
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_sequential_and_concurrent_disk_reads_are_profile_local(profiles):
    profile_a, profile_b, _ = profiles
    registry_a = {"profile-a": {}}
    registry_b = {"profile-b": {}}
    (profile_a / "models_dev_cache.json").write_text(
        json.dumps(registry_a),
        encoding="utf-8",
    )
    (profile_b / "models_dev_cache.json").write_text(
        json.dumps(registry_b),
        encoding="utf-8",
    )

    async def load():
        return await md.fetch_models_dev(allow_network=False)

    assert await _run_in_profile(profile_a, load) == registry_a
    assert await _run_in_profile(profile_b, load) == registry_b

    _reset_models_dev_state()
    result_a, result_b = await asyncio.gather(
        _run_in_profile(profile_a, load),
        _run_in_profile(profile_b, load),
    )
    assert result_a == registry_a
    assert result_b == registry_b


@pytest.mark.asyncio
async def test_distinct_profiles_refresh_concurrently_and_write_own_disks(profiles):
    profile_a, profile_b, _ = profiles
    entered = {"profile-a": asyncio.Event(), "profile-b": asyncio.Event()}
    release = asyncio.Event()

    async def fetch_network():
        from hermes_constants import get_hermes_home

        name = get_hermes_home().name
        entered[name].set()
        await release.wait()
        return {name: {}}

    async def refresh():
        return await md.fetch_models_dev(force_refresh=True)

    with patch.object(md, "_fetch_models_dev_from_network", new=fetch_network):
        task_a = asyncio.create_task(_run_in_profile(profile_a, refresh))
        task_b = asyncio.create_task(_run_in_profile(profile_b, refresh))
        await asyncio.gather(*(event.wait() for event in entered.values()))
        release.set()
        result_a, result_b = await asyncio.gather(task_a, task_b)

    assert result_a == {"profile-a": {}}
    assert result_b == {"profile-b": {}}
    assert json.loads(
        (profile_a / "models_dev_cache.json").read_text(encoding="utf-8")
    ) == result_a
    assert json.loads(
        (profile_b / "models_dev_cache.json").read_text(encoding="utf-8")
    ) == result_b


@pytest.mark.asyncio
async def test_owner_cancellation_is_profile_local_and_same_profile_retries(profiles):
    profile_a, profile_b, _ = profiles
    entered_a = asyncio.Event()
    release_b = asyncio.Event()
    calls = {"profile-a": 0, "profile-b": 0}

    async def fetch_network():
        from hermes_constants import get_hermes_home

        name = get_hermes_home().name
        calls[name] += 1
        if name == "profile-a" and calls[name] == 1:
            entered_a.set()
            await asyncio.Future()
        if name == "profile-b":
            await release_b.wait()
        return {name: {}}

    async def refresh():
        return await md.fetch_models_dev(force_refresh=True)

    with patch.object(md, "_fetch_models_dev_from_network", new=fetch_network):
        owner_a = asyncio.create_task(_run_in_profile(profile_a, refresh))
        await entered_a.wait()
        waiter_a = asyncio.create_task(_run_in_profile(profile_a, refresh))
        sibling_b = asyncio.create_task(_run_in_profile(profile_b, refresh))
        await asyncio.sleep(0)
        owner_a.cancel()
        owner_a.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner_a
        release_b.set()
        result_a, result_b = await asyncio.gather(waiter_a, sibling_b)

    assert result_a == {"profile-a": {}}
    assert result_b == {"profile-b": {}}
    assert calls == {"profile-a": 2, "profile-b": 1}


@pytest.mark.asyncio
async def test_waiter_cancellation_does_not_cancel_same_profile_owner(profiles):
    profile_a, _, _ = profiles
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def fetch_network():
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return {"profile-a": {}}

    async def refresh():
        return await md.fetch_models_dev()

    with (
        patch.object(md, "_disk_cache_age_seconds", new=AsyncMock(return_value=None)),
        patch.object(md, "_load_disk_cache", new=AsyncMock(return_value={})),
        patch.object(md, "_save_disk_cache", new=AsyncMock()),
        patch.object(md, "_fetch_models_dev_from_network", new=fetch_network),
    ):
        owner = asyncio.create_task(_run_in_profile(profile_a, refresh))
        await entered.wait()
        waiter = asyncio.create_task(_run_in_profile(profile_a, refresh))
        await asyncio.sleep(0)
        waiter.cancel()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not owner.done()
        release.set()
        assert await owner == {"profile-a": {}}

    assert calls == 1


@pytest.mark.asyncio
async def test_failure_backoff_does_not_suppress_sibling_profile(profiles):
    profile_a, profile_b, _ = profiles

    async def fetch_network():
        from hermes_constants import get_hermes_home

        name = get_hermes_home().name
        if name == "profile-a":
            raise OSError("profile A offline")
        return {name: {}}

    async def refresh():
        return await md.fetch_models_dev()

    with (
        patch.object(md, "_disk_cache_age_seconds", new=AsyncMock(return_value=None)),
        patch.object(md, "_load_disk_cache", new=AsyncMock(return_value={})),
        patch.object(md, "_fetch_models_dev_from_network", new=fetch_network),
    ):
        result_a = await _run_in_profile(profile_a, refresh)
        result_b = await _run_in_profile(profile_b, refresh)

    assert result_a == {}
    assert result_b == {"profile-b": {}}

    async def state():
        return await md._activate_models_dev_profile()

    state_a = await _run_in_profile(profile_a, state)
    state_b = await _run_in_profile(profile_b, state)
    assert state_a.retry_after > time.time()
    assert state_b.retry_after == 0


@pytest.mark.asyncio
async def test_symlink_aliases_share_profile_cache_and_singleflight(profiles):
    profile_a, _, alias_a = profiles
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def fetch_network():
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return {"shared": {}}

    async def refresh():
        return await md.fetch_models_dev()

    with (
        patch.object(md, "_disk_cache_age_seconds", new=AsyncMock(return_value=None)),
        patch.object(md, "_load_disk_cache", new=AsyncMock(return_value={})),
        patch.object(md, "_save_disk_cache", new=AsyncMock()),
        patch.object(md, "_fetch_models_dev_from_network", new=fetch_network),
    ):
        direct = asyncio.create_task(_run_in_profile(profile_a, refresh))
        await entered.wait()
        alias = asyncio.create_task(_run_in_profile(alias_a, refresh))
        await asyncio.sleep(0)
        release.set()
        assert await asyncio.gather(direct, alias) == [
            {"shared": {}},
            {"shared": {}},
        ]

    assert calls == 1


def test_finished_background_refresh_does_not_retain_closed_loop(profiles):
    profile_a, _, _ = profiles
    loop_refs = []
    task_refs = []

    async def run_refresh():
        loop_refs.append(weakref.ref(asyncio.get_running_loop()))
        token = set_hermes_home_override(profile_a)
        try:
            state = await md._activate_models_dev_profile()
            state.cache = {"stale": {}}
            state.cache_time = time.time() - md._MODELS_DEV_CACHE_TTL - 1
            md._publish_models_dev_legacy_state(state)
            assert await md.fetch_models_dev() == {"stale": {}}
            task = state.refresh_task
            assert task is not None
            task_refs.append(weakref.ref(task))
            await task
            await asyncio.sleep(0)
            assert state.refresh_task is None
            assert state.update_claim is None
        finally:
            reset_hermes_home_override(token)

    with (
        patch.object(md, "_fetch_models_dev_from_network", new=AsyncMock(return_value={"fresh": {}})),
        patch.object(md, "_save_disk_cache", new=AsyncMock()),
    ):
        asyncio.run(run_refresh())

    gc.collect()
    assert loop_refs[0]() is None
    assert task_refs[0]() is None


@pytest.mark.asyncio
async def test_background_cancelled_before_first_step_releases_its_claim(profiles):
    profile_a, _, _ = profiles
    token = set_hermes_home_override(profile_a)
    try:
        state = await md._activate_models_dev_profile()
        state.cache = {"stale": {}}
        state.cache_time = time.time() - md._MODELS_DEV_CACHE_TTL - 1
        md._publish_models_dev_legacy_state(state)
        assert await md.fetch_models_dev() == {"stale": {}}
        task = state.refresh_task
        assert task is not None
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
        assert state.refresh_task is None
        assert state.refresh_claim is None
        assert state.update_claim is None
    finally:
        reset_hermes_home_override(token)


def test_same_profile_remains_singleflight_across_distinct_loops(profiles):
    profile_a, _, _ = profiles
    entered = threading.Event()
    release = threading.Event()
    calls = 0
    calls_guard = threading.Lock()
    results = []
    errors = []

    async def fetch_network():
        nonlocal calls
        with calls_guard:
            calls += 1
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.001)
        return {"shared": {}}

    async def refresh():
        return await _run_in_profile(profile_a, md.fetch_models_dev)

    def runner():
        try:
            results.append(asyncio.run(refresh()))
        except BaseException as exc:
            errors.append(exc)

    with (
        patch.object(md, "_disk_cache_age_seconds", new=AsyncMock(return_value=None)),
        patch.object(md, "_load_disk_cache", new=AsyncMock(return_value={})),
        patch.object(md, "_save_disk_cache", new=AsyncMock()),
        patch.object(md, "_fetch_models_dev_from_network", new=fetch_network),
    ):
        first = threading.Thread(target=runner)
        second = threading.Thread(target=runner)
        first.start()
        assert entered.wait(timeout=1)
        second.start()
        time.sleep(0.02)
        release.set()
        first.join(timeout=2)
        second.join(timeout=2)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert results == [{"shared": {}}, {"shared": {}}]
    assert calls == 1
