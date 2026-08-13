"""Profile and event-loop isolation for the native-async LSP service."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from agent import lsp as lsp_module
from agent.lsp import install as lsp_install
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)


@pytest.fixture(autouse=True)
def _reset_lsp_scopes():
    with lsp_module._lsp_state_guard:
        lsp_module._lsp_loop_states.clear()
        lsp_module._lsp_scope_aliases.clear()
        lsp_module._lsp_owner_scopes.clear()
        lsp_install._install_locks.clear()
        lsp_install._install_results.clear()
    yield
    with lsp_module._lsp_state_guard:
        lsp_module._lsp_loop_states.clear()
        lsp_module._lsp_scope_aliases.clear()
        lsp_module._lsp_owner_scopes.clear()
        lsp_install._install_locks.clear()
        lsp_install._install_results.clear()


@dataclass
class _FakeService:
    home: str
    workspace: str = "same-workspace"
    shutdown_homes: list[str] = field(default_factory=list)

    def is_active(self) -> bool:
        return True

    async def shutdown(self) -> None:
        self.shutdown_homes.append(str(get_hermes_home()))


async def _under_profile(home, operation, *args):
    token = set_hermes_home_override(home)
    try:
        return await operation(*args)
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_concurrent_profiles_create_distinct_same_workspace_services(
    tmp_path,
    monkeypatch,
):
    homes = [tmp_path / "a", tmp_path / "b"]
    for home in homes:
        home.mkdir()
    created: list[_FakeService] = []

    async def create():
        service = _FakeService(str(get_hermes_home()))
        created.append(service)
        await asyncio.sleep(0)
        return service

    monkeypatch.setattr(lsp_module.LSPService, "create_from_config", create)
    service_a, service_b = await asyncio.gather(
        _under_profile(homes[0], lsp_module.get_service),
        _under_profile(homes[1], lsp_module.get_service),
    )

    assert service_a is not service_b
    assert service_a.workspace == service_b.workspace == "same-workspace"
    assert {service_a.home, service_b.home} == {str(home) for home in homes}
    assert len(created) == 2
    assert await _under_profile(homes[0], lsp_module.get_service) is service_a
    assert await _under_profile(homes[1], lsp_module.get_service) is service_b


@pytest.mark.asyncio
async def test_canonical_profile_alias_shares_service(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    alias = tmp_path / "profile-alias"
    alias.symlink_to(home, target_is_directory=True)
    created: list[_FakeService] = []

    async def create():
        service = _FakeService(str(get_hermes_home()))
        created.append(service)
        return service

    monkeypatch.setattr(lsp_module.LSPService, "create_from_config", create)

    direct = await _under_profile(home, lsp_module.get_service)
    through_alias = await _under_profile(alias, lsp_module.get_service)

    assert through_alias is direct
    assert len(created) == 1


@pytest.mark.asyncio
async def test_shutdown_is_limited_to_active_profile(tmp_path, monkeypatch):
    home_a = tmp_path / "a"
    home_b = tmp_path / "b"
    home_a.mkdir()
    home_b.mkdir()

    async def create():
        return _FakeService(str(get_hermes_home()))

    monkeypatch.setattr(lsp_module.LSPService, "create_from_config", create)
    service_a = await _under_profile(home_a, lsp_module.get_service)
    service_b = await _under_profile(home_b, lsp_module.get_service)

    await _under_profile(home_a, lsp_module.shutdown_service)

    assert service_a.shutdown_homes == [str(home_a)]
    assert service_b.shutdown_homes == []
    assert await _under_profile(home_b, lsp_module.get_service) is service_b


@pytest.mark.asyncio
async def test_owner_release_targets_retained_profile_not_current_context(
    tmp_path,
    monkeypatch,
):
    class Owner:
        pass

    owner_a = Owner()
    owner_b = Owner()
    home_a = tmp_path / "a"
    home_b = tmp_path / "b"
    home_a.mkdir()
    home_b.mkdir()

    async def create():
        return _FakeService(str(get_hermes_home()))

    monkeypatch.setattr(lsp_module.LSPService, "create_from_config", create)
    await _under_profile(home_a, lsp_module._retain_lsp_lifecycle, owner_a)
    service_a = await _under_profile(home_a, lsp_module.get_service)
    await _under_profile(home_b, lsp_module._retain_lsp_lifecycle, owner_b)
    service_b = await _under_profile(home_b, lsp_module.get_service)

    # A downstream app may close an owner while another profile is active.
    await _under_profile(home_b, lsp_module._release_lsp_lifecycle, owner_a)

    assert service_a.shutdown_homes == [str(home_a)]
    assert service_b.shutdown_homes == []
    assert await _under_profile(home_b, lsp_module.get_service) is service_b

    await _under_profile(home_b, lsp_module._release_lsp_lifecycle, owner_b)
    assert service_b.shutdown_homes == [str(home_b)]


@pytest.mark.asyncio
async def test_final_owner_release_finishes_under_repeated_cancellation(
    tmp_path,
    monkeypatch,
):
    class Owner:
        pass

    class SlowService(_FakeService):
        async def shutdown(self) -> None:
            entered.set()
            await allow_shutdown.wait()
            await super().shutdown()

    owner = Owner()
    home = tmp_path / "profile"
    home.mkdir()
    entered = asyncio.Event()
    allow_shutdown = asyncio.Event()
    service = SlowService(str(home))

    async def create():
        return service

    monkeypatch.setattr(lsp_module.LSPService, "create_from_config", create)
    await _under_profile(home, lsp_module._retain_lsp_lifecycle, owner)
    assert await _under_profile(home, lsp_module.get_service) is service

    release = asyncio.create_task(
        _under_profile(home, lsp_module._release_lsp_lifecycle, owner)
    )
    await entered.wait()
    release.cancel()
    await asyncio.sleep(0)
    release.cancel()
    await asyncio.sleep(0)
    assert not release.done()

    allow_shutdown.set()
    with pytest.raises(asyncio.CancelledError):
        await release

    assert service.shutdown_homes == [str(home)]
    scope = await _under_profile(home, lsp_module._activate_lsp_scope)
    state = lsp_module._existing_state_for_scope(scope)
    assert state is not None
    assert state.service is None
    assert state.shutdown_task is None
    assert not lsp_module._lsp_owner_scopes
    await _under_profile(home, lsp_module._release_lsp_lifecycle, owner)


@pytest.mark.asyncio
async def test_new_same_profile_consumer_waits_for_final_release_cleanup(
    tmp_path,
    monkeypatch,
):
    class Owner:
        pass

    class SlowService(_FakeService):
        async def shutdown(self) -> None:
            entered.set()
            await allow_shutdown.wait()
            await super().shutdown()

    first_owner = Owner()
    next_owner = Owner()
    home = tmp_path / "profile"
    home.mkdir()
    entered = asyncio.Event()
    allow_shutdown = asyncio.Event()
    first_service = SlowService(str(home))
    next_service = _FakeService(str(home))
    services = iter([first_service, next_service])

    async def create():
        return next(services)

    monkeypatch.setattr(lsp_module.LSPService, "create_from_config", create)
    await _under_profile(home, lsp_module._retain_lsp_lifecycle, first_owner)
    assert await _under_profile(home, lsp_module.get_service) is first_service

    release = asyncio.create_task(
        _under_profile(
            home,
            lsp_module._release_lsp_lifecycle,
            first_owner,
        )
    )
    await entered.wait()
    retain = asyncio.create_task(
        _under_profile(home, lsp_module._retain_lsp_lifecycle, next_owner)
    )
    await asyncio.sleep(0)
    assert not retain.done()

    allow_shutdown.set()
    await release
    await retain

    assert await _under_profile(home, lsp_module.get_service) is next_service
    await _under_profile(home, lsp_module._release_lsp_lifecycle, next_owner)


@pytest.mark.asyncio
async def test_failed_and_cancelled_creation_roll_back_for_retry(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "profile"
    home.mkdir()
    entered = asyncio.Event()
    release = asyncio.Event()
    service = _FakeService(str(home))
    calls = 0

    async def create():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("config failed")
        if calls == 2:
            entered.set()
            await release.wait()
        return service

    monkeypatch.setattr(lsp_module.LSPService, "create_from_config", create)

    with pytest.raises(RuntimeError, match="config failed"):
        await _under_profile(home, lsp_module.get_service)

    cancelled_create = asyncio.create_task(
        _under_profile(home, lsp_module.get_service)
    )
    await entered.wait()
    cancelled_create.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_create

    release.set()
    assert await _under_profile(home, lsp_module.get_service) is service
    assert calls == 3


@pytest.mark.asyncio
async def test_binary_install_cache_and_lock_are_profile_scoped(
    tmp_path,
    monkeypatch,
):
    home_a = tmp_path / "a"
    home_b = tmp_path / "b"
    home_a.mkdir()
    home_b.mkdir()
    calls: list[str] = []

    async def install(_pkg):
        home = str(get_hermes_home())
        calls.append(home)
        await asyncio.sleep(0)
        return f"{home}/lsp/bin/pyright"

    monkeypatch.setattr(lsp_install, "_do_install", install)

    a_first, a_second, b_first = await asyncio.gather(
        _under_profile(home_a, lsp_install.try_install, "pyright"),
        _under_profile(home_a, lsp_install.try_install, "pyright"),
        _under_profile(home_b, lsp_install.try_install, "pyright"),
    )

    assert a_first == a_second == f"{home_a}/lsp/bin/pyright"
    assert b_first == f"{home_b}/lsp/bin/pyright"
    assert sorted(calls) == sorted([str(home_a), str(home_b)])
