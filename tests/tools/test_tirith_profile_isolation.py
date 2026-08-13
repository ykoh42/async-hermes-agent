"""Profile and event-loop ownership for retained Tirith runtime state."""

from __future__ import annotations

import asyncio
import gc
import weakref
from pathlib import Path
from unittest.mock import AsyncMock

import aiofiles
import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import tirith_security as tirith


_CFG = {
    "tirith_enabled": True,
    "tirith_path": "tirith",
    "tirith_timeout": 5,
    "tirith_fail_open": True,
}


def _reset_scoped_state() -> None:
    with tirith._tirith_scope_guard:
        tirith._tirith_loop_states.clear()
        tirith._tirith_scope_aliases.clear()
        tirith._tirith_staged_states.clear()
    tirith._tirith_scope_context.set(None)
    tirith._resolved_path = None
    tirith._install_failure_reason = ""
    tirith._crash_count = 0
    tirith._circuit_open = False
    tirith._install_task = None
    tirith._warned_messages = set()
    tirith._legacy_state_snapshot = (
        None,
        "",
        0,
        False,
        None,
        frozenset(),
    )


@pytest.fixture(autouse=True)
def isolated_tirith_state():
    _reset_scoped_state()
    yield
    for profiles in list(tirith._tirith_loop_states.values()):
        assert all(
            state.install_task is None or state.install_task.done()
            for state in profiles.values()
        )
    _reset_scoped_state()


@pytest.fixture
def profile_homes(tmp_path):
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    profile_a.mkdir()
    profile_b.mkdir()
    alias_a = tmp_path / "profile-a-alias"
    alias_a.symlink_to(profile_a, target_is_directory=True)
    return profile_a, profile_b, alias_a


async def _profile_state(home: Path) -> tirith._TirithProfileState:
    token = set_hermes_home_override(home)
    try:
        return await tirith._activate_tirith_state()
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_path_crash_and_warning_state_is_isolated_by_profile(
    monkeypatch,
    profile_homes,
    caplog,
):
    profile_a, profile_b, _ = profile_homes

    async def mutate(home: Path, path: str, crashes: int) -> None:
        token = set_hermes_home_override(home)
        try:
            state = await tirith._activate_tirith_state()
            tirith._update_tirith_state(state, resolved_path=path)
            for _ in range(crashes):
                tirith._record_tirith_crash()
            tirith._warn_once("spawn", "spawn failed for %s", home.name)
            tirith._warn_once("spawn", "duplicate for %s", home.name)
        finally:
            reset_hermes_home_override(token)

    await asyncio.gather(
        mutate(profile_a, "/profile-a/tirith", tirith._CRASH_LIMIT),
        mutate(profile_b, "/profile-b/tirith", 1),
    )

    state_a = await _profile_state(profile_a)
    state_b = await _profile_state(profile_b)
    assert state_a.resolved_path == "/profile-a/tirith"
    assert state_b.resolved_path == "/profile-b/tirith"
    assert state_a.circuit_open is True
    assert state_b.circuit_open is False
    assert caplog.messages.count("spawn failed for profile-a") == 1
    assert caplog.messages.count("spawn failed for profile-b") == 1
    assert not any(message.startswith("duplicate") for message in caplog.messages)


@pytest.mark.asyncio
async def test_same_profile_install_is_once_and_sibling_profiles_do_not_share_lock(
    monkeypatch,
    profile_homes,
):
    profile_a, profile_b, _ = profile_homes
    monkeypatch.setattr(tirith.shutil, "which", lambda _name: None)
    monkeypatch.setattr(tirith.aiofiles.os.path, "isfile", AsyncMock(return_value=False))
    monkeypatch.setattr(tirith, "_clear_install_failed", AsyncMock())
    monkeypatch.setattr(tirith, "_mark_install_failed", AsyncMock())
    entered = {"profile-a": asyncio.Event(), "profile-b": asyncio.Event()}
    release = {"profile-a": asyncio.Event(), "profile-b": asyncio.Event()}
    calls: list[str] = []

    async def install(*, log_failures=True):
        del log_failures
        from hermes_constants import get_hermes_home

        profile = get_hermes_home().name
        calls.append(profile)
        entered[profile].set()
        await release[profile].wait()
        return f"/{profile}/tirith", ""

    monkeypatch.setattr(tirith, "_install_tirith", install)

    async def run(home: Path):
        token = set_hermes_home_override(home)
        try:
            return await tirith._background_install()
        finally:
            reset_hermes_home_override(token)

    first_a = asyncio.create_task(run(profile_a))
    second_a = asyncio.create_task(run(profile_a))
    only_b = asyncio.create_task(run(profile_b))
    await asyncio.gather(entered["profile-a"].wait(), entered["profile-b"].wait())
    assert sorted(calls) == ["profile-a", "profile-b"]

    release["profile-b"].set()
    assert await only_b == "/profile-b/tirith"
    assert not first_a.done()
    release["profile-a"].set()
    assert await asyncio.gather(first_a, second_a) == [
        "/profile-a/tirith",
        "/profile-a/tirith",
    ]
    assert calls.count("profile-a") == 1


@pytest.mark.asyncio
async def test_install_failure_reason_and_disk_marker_are_profile_local(
    monkeypatch,
    profile_homes,
):
    profile_a, profile_b, _ = profile_homes
    monkeypatch.setattr(tirith.shutil, "which", lambda _name: None)
    monkeypatch.setattr(tirith.aiofiles.os.path, "isfile", AsyncMock(return_value=False))

    async def fail_install(*, log_failures=True):
        del log_failures
        from hermes_constants import get_hermes_home

        reason = f"failure-{get_hermes_home().name}"
        return None, reason

    monkeypatch.setattr(tirith, "_install_tirith", fail_install)

    async def run(home: Path):
        token = set_hermes_home_override(home)
        try:
            assert await tirith._background_install() is None
            state = await tirith._activate_tirith_state()
            marker = tirith._failure_marker_path()
            async with aiofiles.open(marker, encoding="utf-8") as marker_file:
                return state.install_failure_reason, await marker_file.read()
        finally:
            reset_hermes_home_override(token)

    result_a, result_b = await asyncio.gather(run(profile_a), run(profile_b))

    assert result_a == ("failure-profile-a", "failure-profile-a")
    assert result_b == ("failure-profile-b", "failure-profile-b")


@pytest.mark.asyncio
async def test_cancelled_profile_install_does_not_cancel_sibling_or_block_retry(
    monkeypatch,
    profile_homes,
):
    profile_a, profile_b, _ = profile_homes
    monkeypatch.setattr(tirith, "_load_security_config", AsyncMock(return_value=_CFG))
    monkeypatch.setattr(tirith, "is_platform_supported", lambda: True)
    monkeypatch.setattr(tirith.shutil, "which", lambda _name: None)
    monkeypatch.setattr(tirith.aiofiles.os.path, "isfile", AsyncMock(return_value=False))
    monkeypatch.setattr(tirith, "_read_failure_reason", AsyncMock(return_value=None))
    started = {"profile-a": asyncio.Event(), "profile-b": asyncio.Event()}
    release_b = asyncio.Event()
    cancelled_a = asyncio.Event()

    async def install(*, log_failures=True):
        del log_failures
        from hermes_constants import get_hermes_home

        profile = get_hermes_home().name
        started[profile].set()
        try:
            if profile == "profile-a":
                await asyncio.Future()
            await release_b.wait()
            return "/profile-b/tirith"
        finally:
            if profile == "profile-a":
                cancelled_a.set()

    monkeypatch.setattr(tirith, "_background_install", install)

    async def start(home: Path) -> tirith._TirithProfileState:
        token = set_hermes_home_override(home)
        try:
            assert await tirith.ensure_installed() is None
            return await tirith._activate_tirith_state()
        finally:
            reset_hermes_home_override(token)

    state_a, state_b = await asyncio.gather(start(profile_a), start(profile_b))
    await asyncio.gather(started["profile-a"].wait(), started["profile-b"].wait())
    task_a = state_a.install_task
    task_b = state_b.install_task
    assert task_a is not None and task_b is not None and task_a is not task_b

    task_a.cancel()
    task_a.cancel()
    await asyncio.gather(task_a, return_exceptions=True)
    await cancelled_a.wait()
    await asyncio.sleep(0)
    assert state_a.install_task is None
    assert not task_b.done()

    release_b.set()
    assert await task_b == "/profile-b/tirith"
    await asyncio.sleep(0)
    assert state_b.install_task is None

    retry_a = await start(profile_a)
    assert retry_a.install_task is not None
    retry_a.install_task.cancel()
    await asyncio.gather(retry_a.install_task, return_exceptions=True)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_symlink_aliases_share_one_profile_state(profile_homes):
    profile_a, _, alias_a = profile_homes
    direct = await _profile_state(profile_a)
    alias = await _profile_state(alias_a)

    assert alias is direct


def test_closed_loops_and_scoped_states_are_collectible(profile_homes):
    profile_a, profile_b, _ = profile_homes
    loop_refs = []
    state_refs = []

    async def activate(home: Path) -> None:
        loop_refs.append(weakref.ref(asyncio.get_running_loop()))
        state_refs.append(weakref.ref(await _profile_state(home)))

    asyncio.run(activate(profile_a))
    asyncio.run(activate(profile_b))
    gc.collect()

    assert loop_refs[0]() is None
    assert loop_refs[1]() is None
    assert state_refs[0]() is None
    assert state_refs[1]() is None


def test_private_reset_hook_seeds_each_fresh_event_loop(profile_homes):
    profile_a, _, _ = profile_homes

    async def reset_and_read() -> str | None | bool:
        token = set_hermes_home_override(profile_a)
        try:
            tirith._resolved_path = "/seeded/tirith"
            tirith._install_failure_reason = ""
            tirith._crash_count = 0
            tirith._circuit_open = False
            tirith._install_task = None
            tirith._reset_spawn_warning_state()
            return (await tirith._activate_tirith_state()).resolved_path
        finally:
            reset_hermes_home_override(token)

    assert asyncio.run(reset_and_read()) == "/seeded/tirith"
    # The assigned value equals the previous legacy snapshot here, but the
    # second loop still needs its own seeded scoped state.
    assert asyncio.run(reset_and_read()) == "/seeded/tirith"
