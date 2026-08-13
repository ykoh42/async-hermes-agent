"""Profile isolation for retained background-process runtime state."""

from __future__ import annotations

import asyncio
import gc
import json
import time
import weakref

import aiofiles
import pytest

from agent import secret_scope
from hermes_constants import (
    reset_hermes_home_override,
    set_hermes_home_override,
)
from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.mark.asyncio
async def test_same_task_and_session_are_isolated_across_concurrent_profiles(
    tmp_path,
):
    registry = ProcessRegistry()
    ready = asyncio.Event()
    entered = 0

    async def populate(profile, label):
        nonlocal entered
        token = set_hermes_home_override(profile)
        try:
            await registry._activate_profile_state()
            session = ProcessSession(
                id="proc_same",
                command=f"echo {label}",
                task_id="same-task",
                session_key="same-session",
                started_at=time.time(),
            )
            registry._running[session.id] = session
            registry.completion_queue.put_nowait({"type": label})
            entered += 1
            if entered == 2:
                ready.set()
            await ready.wait()
            visible = await registry.list_sessions(
                task_id="same-task",
                session_key="same-session",
            )
            queued = registry.completion_queue.get_nowait()
            await registry._write_checkpoint()
            return visible, queued
        finally:
            reset_hermes_home_override(token)

    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    (visible_a, queued_a), (visible_b, queued_b) = await asyncio.gather(
        populate(profile_a, "alpha"),
        populate(profile_b, "beta"),
    )

    assert [entry["command"] for entry in visible_a] == ["echo alpha"]
    assert [entry["command"] for entry in visible_b] == ["echo beta"]
    assert queued_a == {"type": "alpha"}
    assert queued_b == {"type": "beta"}
    async with aiofiles.open(profile_a / "processes.json", encoding="utf-8") as handle:
        checkpoint_a = json.loads(await handle.read())
    async with aiofiles.open(profile_b / "processes.json", encoding="utf-8") as handle:
        checkpoint_b = json.loads(await handle.read())
    assert [entry["command"] for entry in checkpoint_a] == ["echo alpha"]
    assert [entry["command"] for entry in checkpoint_b] == ["echo beta"]


@pytest.mark.asyncio
async def test_wait_timeout_limit_uses_active_profile_scope(
    monkeypatch,
    tmp_path,
):
    registry = ProcessRegistry()
    monkeypatch.setenv("TERMINAL_TIMEOUT", "999")
    previous_multiplex = secret_scope.is_multiplex_active()
    outer_scope = secret_scope.set_secret_scope(None)
    secret_scope.set_multiplex_active(True)
    profiles = {label: tmp_path / label for label in ("a", "b")}

    async def wait_for_profile(label: str, timeout: str):
        home_token = set_hermes_home_override(profiles[label])
        scope_token = secret_scope.set_secret_scope(
            {"TERMINAL_TIMEOUT": timeout}
        )
        try:
            await registry._activate_profile_state()
            session = ProcessSession(
                id="proc_same",
                command=f"echo {label}",
                exited=True,
                exit_code=0,
            )
            registry._finished[session.id] = session
            await asyncio.sleep(0)
            return await registry.wait(session.id, timeout=100)
        finally:
            secret_scope.reset_secret_scope(scope_token)
            reset_hermes_home_override(home_token)

    try:
        result_a, result_b = await asyncio.gather(
            wait_for_profile("a", "5"),
            wait_for_profile("b", "7"),
        )
        assert result_a["timeout_note"] == (
            "Requested wait of 100s was clamped to configured limit of 5s"
        )
        assert result_b["timeout_note"] == (
            "Requested wait of 100s was clamped to configured limit of 7s"
        )

        home_token = set_hermes_home_override(profiles["a"])
        try:
            with pytest.raises(secret_scope.UnscopedSecretError):
                await registry.wait("proc_same", timeout=100)
        finally:
            reset_hermes_home_override(home_token)
    finally:
        secret_scope.set_multiplex_active(previous_multiplex)
        secret_scope.reset_secret_scope(outer_scope)


@pytest.mark.asyncio
async def test_canonical_profile_alias_reuses_the_same_registry_state(tmp_path):
    registry = ProcessRegistry()
    profile = tmp_path / "profile"
    profile.mkdir()
    alias = tmp_path / "profile-alias"
    alias.symlink_to(profile, target_is_directory=True)

    token = set_hermes_home_override(profile)
    try:
        await registry._activate_profile_state()
        registry._running["proc_alias"] = ProcessSession(
            id="proc_alias",
            command="echo canonical",
        )
    finally:
        reset_hermes_home_override(token)

    token = set_hermes_home_override(alias)
    try:
        assert (await registry.get("proc_alias")).command == "echo canonical"
        assert registry.count_running() == 1
    finally:
        reset_hermes_home_override(token)


def test_no_loop_private_state_staging_migrates_at_first_await(tmp_path):
    registry = ProcessRegistry()
    token = set_hermes_home_override(tmp_path / "profile")
    try:
        registry._running["proc_staged"] = ProcessSession(
            id="proc_staged",
            command="echo staged",
        )

        async def read_staged():
            return await registry.get("proc_staged")

        session = asyncio.run(read_staged())
    finally:
        reset_hermes_home_override(token)

    assert session is not None
    assert session.command == "echo staged"


def test_closed_event_loop_drops_process_registry_state(tmp_path):
    registry = ProcessRegistry()
    profile = tmp_path / "profile"

    async def populate():
        token = set_hermes_home_override(profile)
        try:
            await registry._activate_profile_state()
            registry._running["loop-owned"] = ProcessSession(
                id="loop-owned",
                command="echo loop-owned",
            )
            return weakref.ref(asyncio.get_running_loop())
        finally:
            reset_hermes_home_override(token)

    loop_ref = asyncio.run(populate())
    token = set_hermes_home_override(profile)
    try:
        async def read_new_loop():
            return await registry.get("loop-owned")

        assert asyncio.run(read_new_loop()) is None
    finally:
        reset_hermes_home_override(token)
    gc.collect()
    assert loop_ref() is None
    assert not registry._loop_profile_states
