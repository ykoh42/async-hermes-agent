"""Profile, loop, and lease ownership of cached auxiliary clients."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock

import pytest

from agent import auxiliary_client as aux
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)


@pytest.fixture(autouse=True)
def _reset_auxiliary_cache_scopes():
    aux._client_cache.clear()
    aux._retired_auxiliary_clients.clear()
    with aux._auxiliary_cache_guard:
        aux._auxiliary_cache_aliases.clear()
        aux._auxiliary_lifecycle_consumers.clear()
        aux._auxiliary_lifecycle_locks.clear()
        aux._auxiliary_owner_scopes.clear()
    yield
    aux._client_cache.clear()
    aux._retired_auxiliary_clients.clear()
    with aux._auxiliary_cache_guard:
        aux._auxiliary_cache_aliases.clear()
        aux._auxiliary_lifecycle_consumers.clear()
        aux._auxiliary_lifecycle_locks.clear()
        aux._auxiliary_owner_scopes.clear()


@dataclass
class _FakeClient:
    home: str
    close_started: asyncio.Event | None = None
    allow_close: asyncio.Event | None = None
    closed: int = 0
    calls: list[str] = field(default_factory=list)

    async def aclose(self) -> None:
        if self.close_started is not None:
            self.close_started.set()
        if self.allow_close is not None:
            await self.allow_close.wait()
        self.closed += 1


async def _under_profile(home, operation, *args, **kwargs):
    token = set_hermes_home_override(home)
    try:
        return await operation(*args, **kwargs)
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_same_key_isolated_between_concurrent_profiles(tmp_path, monkeypatch):
    homes = [tmp_path / "a", tmp_path / "b"]
    for home in homes:
        home.mkdir()
    built: list[_FakeClient] = []

    async def resolve(_provider, model, **_kwargs):
        client = _FakeClient(str(get_hermes_home()))
        built.append(client)
        await asyncio.sleep(0)
        return client, model

    monkeypatch.setattr(aux, "resolve_provider_client", resolve)
    client_a, client_b = await asyncio.gather(
        _under_profile(homes[0], aux._get_cached_client, "custom", "m"),
        _under_profile(homes[1], aux._get_cached_client, "custom", "m"),
    )

    assert client_a[0] is not client_b[0]
    assert {client_a[0].home, client_b[0].home} == {
        str(homes[0]),
        str(homes[1]),
    }
    assert await _under_profile(
        homes[0], aux._get_cached_client, "custom", "m"
    ) == client_a
    assert await _under_profile(
        homes[1], aux._get_cached_client, "custom", "m"
    ) == client_b
    assert len(built) == 2


@pytest.mark.asyncio
async def test_final_profile_lease_closes_only_its_cached_client(
    tmp_path,
    monkeypatch,
):
    class Owner:
        pass

    home_a = tmp_path / "a"
    home_b = tmp_path / "b"
    home_a.mkdir()
    home_b.mkdir()
    owner_a = Owner()
    owner_b = Owner()

    async def resolve(_provider, model, **_kwargs):
        return _FakeClient(str(get_hermes_home())), model

    monkeypatch.setattr(aux, "resolve_provider_client", resolve)
    await _under_profile(home_a, aux._retain_auxiliary_lifecycle, owner_a)
    client_a, _ = await _under_profile(
        home_a, aux._get_cached_client, "custom", "m"
    )
    await _under_profile(home_b, aux._retain_auxiliary_lifecycle, owner_b)
    client_b, _ = await _under_profile(
        home_b, aux._get_cached_client, "custom", "m"
    )

    await _under_profile(home_b, aux._release_auxiliary_lifecycle, owner_a)

    assert client_a.closed == 1
    assert client_b.closed == 0
    assert await _under_profile(
        home_b, aux._get_cached_client, "custom", "m"
    ) == (client_b, "m")

    await _under_profile(home_b, aux._release_auxiliary_lifecycle, owner_b)
    assert client_b.closed == 1


@pytest.mark.asyncio
async def test_concurrent_same_profile_cache_miss_closes_unexposed_loser(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "profile"
    home.mkdir()
    release_builds = asyncio.Event()
    both_started = asyncio.Event()
    built: list[_FakeClient] = []

    async def resolve(_provider, model, **_kwargs):
        client = _FakeClient(str(home))
        built.append(client)
        if len(built) == 2:
            both_started.set()
        await release_builds.wait()
        return client, model

    monkeypatch.setattr(aux, "resolve_provider_client", resolve)
    first = asyncio.create_task(
        _under_profile(home, aux._get_cached_client, "custom", "m")
    )
    second = asyncio.create_task(
        _under_profile(home, aux._get_cached_client, "custom", "m")
    )
    await both_started.wait()
    release_builds.set()
    result_a, result_b = await asyncio.gather(first, second)

    assert result_a[0] is result_b[0]
    winner = result_a[0]
    loser = next(client for client in built if client is not winner)
    assert winner.closed == 0
    assert loser.closed == 1
    await _under_profile(home, aux.shutdown_cached_clients)
    assert winner.closed == 1


@pytest.mark.asyncio
async def test_sibling_lease_protects_active_call_until_final_release(
    tmp_path,
    monkeypatch,
):
    class Owner:
        pass

    home = tmp_path / "profile"
    home.mkdir()
    first_owner = Owner()
    sibling_owner = Owner()
    call_started = asyncio.Event()
    finish_call = asyncio.Event()
    client = _FakeClient(str(home))

    async def resolve(_provider, model, **_kwargs):
        return client, model

    async def active_call():
        call_started.set()
        await finish_call.wait()

    monkeypatch.setattr(aux, "resolve_provider_client", resolve)
    await _under_profile(home, aux._retain_auxiliary_lifecycle, first_owner)
    await _under_profile(home, aux._retain_auxiliary_lifecycle, sibling_owner)
    assert await _under_profile(
        home, aux._get_cached_client, "custom", "m"
    ) == (client, "m")
    request = asyncio.create_task(active_call())
    await call_started.wait()

    await _under_profile(home, aux._release_auxiliary_lifecycle, first_owner)
    assert client.closed == 0

    finish_call.set()
    await request
    await _under_profile(home, aux._release_auxiliary_lifecycle, sibling_owner)
    assert client.closed == 1


@pytest.mark.asyncio
async def test_final_release_finishes_close_through_repeated_cancellation(
    tmp_path,
    monkeypatch,
):
    class Owner:
        pass

    home = tmp_path / "profile"
    home.mkdir()
    owner = Owner()
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    client = _FakeClient(str(home), close_started, allow_close)

    monkeypatch.setattr(
        aux,
        "resolve_provider_client",
        AsyncMock(return_value=(client, "m")),
    )
    await _under_profile(home, aux._retain_auxiliary_lifecycle, owner)
    await _under_profile(home, aux._get_cached_client, "custom", "m")

    release = asyncio.create_task(
        _under_profile(home, aux._release_auxiliary_lifecycle, owner)
    )
    await close_started.wait()
    release.cancel()
    await asyncio.sleep(0)
    release.cancel()
    await asyncio.sleep(0)
    assert not release.done()

    allow_close.set()
    with pytest.raises(asyncio.CancelledError):
        await release
    assert client.closed == 1
    assert not aux._client_cache


def test_sequential_event_loops_never_reuse_native_client(monkeypatch):
    built: list[_FakeClient] = []

    async def resolve(_provider, model, **_kwargs):
        client = _FakeClient(str(get_hermes_home()))
        built.append(client)
        return client, model

    monkeypatch.setattr(aux, "resolve_provider_client", resolve)

    async def once():
        client, _ = await aux._get_cached_client("custom", "m")
        await aux.shutdown_cached_clients()
        return client

    first = asyncio.run(once())
    second = asyncio.run(once())
    assert first is not second
    assert first.closed == second.closed == 1
