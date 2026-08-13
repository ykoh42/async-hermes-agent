"""Profile isolation for retained terminal environment state."""

from __future__ import annotations

import asyncio
import gc
import weakref

import pytest

from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from tools import terminal_tool as terminal


class _FakeEnvironment:
    def __init__(self, label: str, image: str, cwd: str) -> None:
        self.label = label
        self.image = image
        self.cwd = cwd
        self.cleaned = False

    async def _ensure_initialized(self) -> None:
        return None

    async def cleanup(self, *, force_remove: bool = False) -> None:
        self.cleaned = True


@pytest.fixture
def isolated_terminal_profiles(monkeypatch):
    async def get_config():
        label = get_hermes_home().name
        return {
            "env_type": "docker",
            "cwd": f"/workspace/{label}",
            "timeout": 30,
            "lifetime_seconds": 300,
            "docker_image": f"base-{label}",
        }

    def create_environment(
        env_type,
        image,
        cwd,
        timeout,
        **kwargs,
    ):
        return _FakeEnvironment(get_hermes_home().name, image, cwd)

    monkeypatch.setattr(terminal, "_get_env_config", get_config)
    monkeypatch.setattr(terminal, "_create_environment", create_environment)


@pytest.mark.asyncio
async def test_same_task_is_sequentially_isolated_across_profiles(
    isolated_terminal_profiles,
    tmp_path,
):
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"

    async def get(profile, label):
        token = set_hermes_home_override(profile)
        try:
            await terminal._activate_terminal_scope()
            terminal.register_task_env_overrides(
                "same-task",
                {
                    "env_type": "docker",
                    "docker_image": f"image-{label}",
                    "cwd": f"/cwd/{label}",
                },
            )
            return await terminal._get_or_create_environment("same-task")
        finally:
            reset_hermes_home_override(token)

    env_a = await get(profile_a, "alpha")
    env_b = await get(profile_b, "beta")
    assert env_a is not env_b
    assert (env_a.label, env_a.image, env_a.cwd) == (
        "profile-a",
        "image-alpha",
        "/cwd/alpha",
    )
    assert (env_b.label, env_b.image, env_b.cwd) == (
        "profile-b",
        "image-beta",
        "/cwd/beta",
    )

    for profile, expected, env in (
        (profile_a, "alpha", env_a),
        (profile_b, "beta", env_b),
    ):
        token = set_hermes_home_override(profile)
        try:
            await terminal._activate_terminal_scope()
            assert terminal.get_active_env("same-task") is env
            assert terminal.resolve_task_overrides("same-task") == {
                "env_type": "docker",
                "docker_image": f"image-{expected}",
                "cwd": f"/cwd/{expected}",
            }
            assert terminal.get_session_cwd("same-task") == f"/cwd/{expected}"
            await terminal.cleanup_vm("same-task")
            assert env.cleaned is True
        finally:
            reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_same_task_is_concurrently_isolated_and_cleanup_is_per_profile(
    isolated_terminal_profiles,
    tmp_path,
):
    ready = asyncio.Event()
    entered = 0

    async def use(profile, label):
        nonlocal entered
        token = set_hermes_home_override(profile)
        try:
            await terminal._activate_terminal_scope()
            terminal.register_task_env_overrides(
                "same-task",
                {"env_type": "docker", "docker_image": f"image-{label}"},
            )
            env = await terminal._get_or_create_environment("same-task")
            entered += 1
            if entered == 2:
                ready.set()
            await ready.wait()
            assert terminal.get_active_env("same-task") is env
            assert terminal.resolve_task_overrides("same-task")[
                "docker_image"
            ] == f"image-{label}"
            await terminal.cleanup_all_environments()
            assert terminal.get_active_env("same-task") is None
            return env
        finally:
            reset_hermes_home_override(token)

    env_a, env_b = await asyncio.gather(
        use(tmp_path / "profile-a", "alpha"),
        use(tmp_path / "profile-b", "beta"),
    )
    assert env_a is not env_b
    assert env_a.cleaned is True
    assert env_b.cleaned is True
    assert not terminal._cleanup_tasks
    assert not terminal._cleanup_handles


def test_no_loop_overrides_and_cwd_migrate_at_first_await(tmp_path):
    token = set_hermes_home_override(tmp_path / "profile")
    try:
        terminal.register_task_env_overrides(
            "staged-task",
            {"env_type": "ssh", "cwd": "/remote/staged"},
        )

        async def read_staged():
            await terminal._activate_terminal_scope()
            return (
                terminal.resolve_task_overrides("staged-task"),
                terminal.get_session_cwd("staged-task"),
            )

        overrides, cwd = asyncio.run(read_staged())
    finally:
        reset_hermes_home_override(token)

    assert overrides == {"env_type": "ssh", "cwd": "/remote/staged"}
    assert cwd == "/remote/staged"


@pytest.mark.asyncio
async def test_same_loop_pre_activation_overrides_migrate_to_canonical_profile(
    tmp_path,
):
    profile = tmp_path / "profile"
    profile.mkdir()
    alias = tmp_path / "profile-alias"
    alias.symlink_to(profile, target_is_directory=True)
    token = set_hermes_home_override(alias)
    try:
        terminal.register_task_env_overrides(
            "staged-task",
            {"env_type": "docker", "docker_image": "row-image"},
        )
        await terminal._activate_terminal_scope()
        assert terminal.resolve_task_overrides("staged-task") == {
            "env_type": "docker",
            "docker_image": "row-image",
        }
    finally:
        terminal.clear_task_env_overrides("staged-task")
        reset_hermes_home_override(token)


def test_closed_event_loop_drops_terminal_scoped_runtime_state(tmp_path):
    profile = tmp_path / "profile"

    async def populate():
        token = set_hermes_home_override(profile)
        try:
            await terminal._activate_terminal_scope()
            terminal.register_task_env_overrides(
                "loop-owned",
                {"env_type": "local", "cwd": "/loop-owned"},
            )
            terminal._start_cleanup_thread(300)
            return weakref.ref(asyncio.get_running_loop())
        finally:
            reset_hermes_home_override(token)

    loop_ref = asyncio.run(populate())
    token = set_hermes_home_override(profile)
    try:
        async def read_new_loop():
            await terminal._activate_terminal_scope()
            return terminal.resolve_task_overrides("loop-owned")

        assert asyncio.run(read_new_loop()) == {}
    finally:
        reset_hermes_home_override(token)
    gc.collect()
    assert loop_ref() is None
    assert not terminal._terminal_scope_aliases
    assert not terminal._cleanup_handles


@pytest.mark.asyncio
async def test_canonical_profile_alias_reuses_terminal_environment(
    isolated_terminal_profiles,
    tmp_path,
):
    profile = tmp_path / "profile"
    profile.mkdir()
    alias = tmp_path / "profile-alias"
    alias.symlink_to(profile, target_is_directory=True)

    token = set_hermes_home_override(profile)
    try:
        env = await terminal._get_or_create_environment("same-task")
    finally:
        reset_hermes_home_override(token)

    token = set_hermes_home_override(alias)
    try:
        assert await terminal._get_or_create_environment("same-task") is env
        await terminal.cleanup_vm("same-task")
    finally:
        reset_hermes_home_override(token)
