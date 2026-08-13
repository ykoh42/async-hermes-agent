"""Concurrency and loop ownership for external-secret source snapshots."""

from __future__ import annotations

import asyncio
import gc
import weakref
from types import SimpleNamespace

import pytest

from hermes_cli import env_loader


@pytest.fixture(autouse=True)
def _reset_secret_cache():
    env_loader.reset_secret_source_cache()
    yield
    env_loader.reset_secret_source_cache()


def _report(home, environ):
    key = "PROFILE_API_KEY"
    environ[key] = f"secret-{home.name}"
    applied = SimpleNamespace(source="slow")
    return SimpleNamespace(
        sources=[SimpleNamespace()],
        provenance={key: applied},
    )


@pytest.mark.asyncio
async def test_same_profile_hydration_is_serialized_once(tmp_path, monkeypatch):
    profile = tmp_path / "profile"
    profile.mkdir()
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def apply_all(_config, home, *, environ):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return _report(home, environ)

    monkeypatch.setattr(env_loader, "_load_secrets_config", _config)
    monkeypatch.setattr("agent.secret_sources.registry.apply_all", apply_all)

    first = asyncio.create_task(
        env_loader.hydrate_profile_secret_sources(profile)
    )
    await entered.wait()
    second = asyncio.create_task(
        env_loader.hydrate_profile_secret_sources(profile)
    )
    await asyncio.sleep(0)

    assert calls == 1
    assert not second.done()
    release.set()
    first_result, second_result = await asyncio.gather(first, second)
    assert first_result == second_result == {
        "PROFILE_API_KEY": "secret-profile"
    }
    assert calls == 1


@pytest.mark.asyncio
async def test_different_profiles_hydrate_concurrently(tmp_path, monkeypatch):
    profiles = [tmp_path / "a", tmp_path / "b"]
    for profile in profiles:
        profile.mkdir()
    both_entered = asyncio.Event()
    release = asyncio.Event()
    entered: set[str] = set()

    async def apply_all(_config, home, *, environ):
        entered.add(home.name)
        if len(entered) == 2:
            both_entered.set()
        await release.wait()
        return _report(home, environ)

    monkeypatch.setattr(env_loader, "_load_secrets_config", _config)
    monkeypatch.setattr("agent.secret_sources.registry.apply_all", apply_all)

    tasks = [
        asyncio.create_task(
            env_loader.hydrate_profile_secret_sources(profile)
        )
        for profile in profiles
    ]
    await asyncio.wait_for(both_entered.wait(), timeout=1)
    release.set()
    results = await asyncio.gather(*tasks)

    assert results == [
        {"PROFILE_API_KEY": "secret-a"},
        {"PROFILE_API_KEY": "secret-b"},
    ]


@pytest.mark.asyncio
async def test_cancelled_hydration_rolls_back_cache_and_retries(
    tmp_path,
    monkeypatch,
):
    profile = tmp_path / "profile"
    profile.mkdir()
    entered = asyncio.Event()
    calls = 0

    async def apply_all(_config, home, *, environ):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await asyncio.Future()
        return _report(home, environ)

    monkeypatch.setattr(env_loader, "_load_secrets_config", _config)
    monkeypatch.setattr("agent.secret_sources.registry.apply_all", apply_all)

    cancelled = asyncio.create_task(
        env_loader.hydrate_profile_secret_sources(profile)
    )
    await entered.wait()
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled

    result = await env_loader.hydrate_profile_secret_sources(profile)
    assert result == {"PROFILE_API_KEY": "secret-profile"}
    assert calls == 2


@pytest.mark.asyncio
async def test_canonical_alias_uses_one_lock_and_snapshot(tmp_path, monkeypatch):
    profile = tmp_path / "profile"
    alias = tmp_path / "profile-alias"
    profile.mkdir()
    alias.symlink_to(profile, target_is_directory=True)
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def apply_all(_config, home, *, environ):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return _report(profile, environ)

    monkeypatch.setattr(env_loader, "_load_secrets_config", _config)
    monkeypatch.setattr("agent.secret_sources.registry.apply_all", apply_all)

    direct = asyncio.create_task(
        env_loader.hydrate_profile_secret_sources(profile)
    )
    await entered.wait()
    through_alias = asyncio.create_task(
        env_loader.hydrate_profile_secret_sources(alias)
    )
    await asyncio.sleep(0)
    assert calls == 1

    release.set()
    assert await direct == await through_alias
    assert calls == 1


def test_sequential_event_loops_are_not_retained(tmp_path, monkeypatch):
    profile = tmp_path / "profile"
    profile.mkdir()
    calls = 0

    async def apply_all(_config, home, *, environ):
        nonlocal calls
        calls += 1
        return _report(home, environ)

    monkeypatch.setattr(env_loader, "_load_secrets_config", _config)
    monkeypatch.setattr("agent.secret_sources.registry.apply_all", apply_all)
    loop_refs: list[weakref.ReferenceType[asyncio.AbstractEventLoop]] = []

    async def hydrate(profile):
        loop_refs.append(weakref.ref(asyncio.get_running_loop()))
        return await env_loader.hydrate_profile_secret_sources(profile)

    for _ in range(2):
        assert asyncio.run(hydrate(profile))
        gc.collect()

    assert calls == 1
    assert all(loop_ref() is None for loop_ref in loop_refs)
    assert not env_loader._SECRET_SOURCE_CACHE_LOCK._locks


async def _config(_home):
    return {"slow": {"enabled": True}}
