"""Profile isolation and lifecycle contracts for retained approval state."""

from __future__ import annotations

import asyncio
import gc
import weakref

import aiofiles
import aiofiles.os
import pytest
import yaml

from agent import secret_scope
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from gateway.session_context import reset_session_vars, set_current_session_id
from tools import approval


@pytest.fixture(autouse=True)
def isolated_approval_profiles(monkeypatch):
    approval._approval_profile_states.clear()
    approval._approval_profile_aliases.clear()
    context_token = approval._approval_profile_context.set(None)

    async def release(_session_key):
        return None

    monkeypatch.setattr(approval, "_release_permission_mode_dependents", release)
    yield
    approval._approval_profile_context.reset(context_token)
    approval._approval_profile_states.clear()
    approval._approval_profile_aliases.clear()


@pytest.mark.asyncio
async def test_same_session_state_is_isolated_across_concurrent_profiles(
    monkeypatch,
    tmp_path,
):
    ready = asyncio.Event()
    entered = 0

    async def load_config_readonly():
        label = get_hermes_home().name
        return {
            "approvals": {
                "mode": "off" if label == "profile-a" else "manual",
            },
            "command_allowlist": [f"allow-{label}"],
        }

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        load_config_readonly,
    )

    async def populate(profile, own_key, other_key):
        nonlocal entered
        token = set_hermes_home_override(profile)
        try:
            await approval._load_approval_config_snapshot()
            approval.approve_session("same-session", own_key)
            approval.approve_permanent(f"permanent-{own_key}")
            await approval.enable_session_yolo("same-session")
            entered += 1
            if entered == 2:
                ready.set()
            await ready.wait()
            return {
                "own": approval.is_approved("same-session", own_key),
                "other": approval.is_approved("same-session", other_key),
                "yolo": approval.is_session_yolo_enabled("same-session"),
                "mode": approval._get_approval_mode(),
                "permanent": set(approval._permanent_approved),
            }
        finally:
            reset_hermes_home_override(token)

    state_a, state_b = await asyncio.gather(
        populate(tmp_path / "profile-a", "alpha", "beta"),
        populate(tmp_path / "profile-b", "beta", "alpha"),
    )

    assert state_a == {
        "own": True,
        "other": False,
        "yolo": True,
        "mode": "off",
        "permanent": {"allow-profile-a", "permanent-alpha"},
    }
    assert state_b == {
        "own": True,
        "other": False,
        "yolo": True,
        "mode": "manual",
        "permanent": {"allow-profile-b", "permanent-beta"},
    }


@pytest.mark.asyncio
async def test_same_session_state_is_isolated_across_sequential_profiles(tmp_path):
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"

    token = set_hermes_home_override(profile_a)
    try:
        await approval._activate_approval_profile()
        approval.approve_session("same-session", "alpha")
        await approval.enable_session_yolo("same-session")
    finally:
        reset_hermes_home_override(token)

    token = set_hermes_home_override(profile_b)
    try:
        await approval._activate_approval_profile()
        assert approval.is_approved("same-session", "alpha") is False
        assert approval.is_session_yolo_enabled("same-session") is False
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_distinct_library_agent_session_ids_do_not_share_approval(tmp_path):
    profile = tmp_path / "profile"
    ready = asyncio.Event()
    entered = 0

    async def use_agent_session(session_id, approve):
        nonlocal entered
        token = set_hermes_home_override(profile)
        try:
            set_current_session_id(session_id)
            await approval._activate_approval_profile()
            if approve:
                approval.approve_session(
                    approval.get_current_session_key(),
                    "recursive delete",
                )
                await approval.enable_session_yolo(
                    approval.get_current_session_key()
                )
            entered += 1
            if entered == 2:
                ready.set()
            await ready.wait()
            key = approval.get_current_session_key()
            return (
                key,
                approval.is_approved(key, "recursive delete"),
                approval.is_current_session_yolo_enabled(),
            )
        finally:
            reset_session_vars()
            reset_hermes_home_override(token)

    approved, unapproved = await asyncio.gather(
        use_agent_session("agent-session-a", True),
        use_agent_session("agent-session-b", False),
    )
    assert approved == ("agent-session-a", True, True)
    assert unapproved == ("agent-session-b", False, False)


def test_explicit_gateway_session_key_precedes_library_session_id():
    from gateway.session_context import clear_session_vars, set_session_vars

    tokens = set_session_vars(
        session_key="gateway-key",
        session_id="library-session-id",
    )
    try:
        assert approval.get_current_session_key() == "gateway-key"
    finally:
        clear_session_vars(tokens)
        reset_session_vars()


@pytest.mark.asyncio
async def test_canonical_profile_alias_reuses_approval_state(tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    alias = tmp_path / "profile-alias"
    alias.symlink_to(profile, target_is_directory=True)

    token = set_hermes_home_override(profile)
    try:
        await approval._activate_approval_profile()
        approval.approve_session("same-session", "canonical")
    finally:
        reset_hermes_home_override(token)

    token = set_hermes_home_override(alias)
    try:
        await approval._activate_approval_profile()
        assert approval.is_approved("same-session", "canonical") is True
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_permanent_allowlist_round_trips_to_each_profile_only(tmp_path):
    async def round_trip(profile, pattern):
        token = set_hermes_home_override(profile)
        try:
            await approval.save_permanent_allowlist({pattern})
            loaded = await approval.load_permanent_allowlist()
            return loaded, set(approval._permanent_approved)
        finally:
            reset_hermes_home_override(token)

    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    result_a, result_b = await asyncio.gather(
        round_trip(profile_a, "allow-alpha"),
        round_trip(profile_b, "allow-beta"),
    )

    assert result_a == ({"allow-alpha"}, {"allow-alpha"})
    assert result_b == ({"allow-beta"}, {"allow-beta"})
    async with aiofiles.open(profile_a / "config.yaml", encoding="utf-8") as handle:
        config_a = yaml.safe_load(await handle.read())
    async with aiofiles.open(profile_b / "config.yaml", encoding="utf-8") as handle:
        config_b = yaml.safe_load(await handle.read())
    assert config_a["command_allowlist"] == ["allow-alpha"]
    assert config_b["command_allowlist"] == ["allow-beta"]


@pytest.mark.asyncio
async def test_command_boundary_does_not_use_other_profiles_allowlist(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    async def load_config_readonly():
        label = get_hermes_home().name
        return {
            "approvals": {"mode": "manual"},
            "command_allowlist": (
                ["rm -rf build"] if label == "profile-a" else []
            ),
        }

    async def deny(*_args, **_kwargs):
        return "deny"

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        load_config_readonly,
    )

    async def check(profile):
        token = set_hermes_home_override(profile)
        try:
            return await approval.check_dangerous_command(
                "rm -rf build",
                "local",
                approval_callback=deny,
            )
        finally:
            reset_hermes_home_override(token)

    allowed, denied = await asyncio.gather(
        check(tmp_path / "profile-a"),
        check(tmp_path / "profile-b"),
    )
    assert allowed["approved"] is True
    assert denied["approved"] is False
    assert denied["outcome"] == "denied"


@pytest.mark.asyncio
async def test_cancelled_config_load_does_not_publish_partial_profile_state(
    monkeypatch,
    tmp_path,
):
    started = asyncio.Event()
    release = asyncio.Event()
    token = set_hermes_home_override(tmp_path / "profile")
    try:
        await approval._activate_approval_profile()
        approval._replace_approval_config({"mode": "smart"})
        approval._replace_permanent({"old-pattern"})

        async def load_config_readonly():
            started.set()
            await release.wait()
            return {
                "approvals": {"mode": "off"},
                "command_allowlist": ["new-pattern"],
            }

        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            load_config_readonly,
        )
        task = asyncio.create_task(approval._load_approval_config_snapshot())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert approval._get_approval_mode() == "smart"
        assert set(approval._permanent_approved) == {"old-pattern"}
    finally:
        release.set()
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_repeated_cancellation_finishes_atomic_allowlist_save(
    monkeypatch,
    tmp_path,
):
    entered_replace = asyncio.Event()
    release_replace = asyncio.Event()
    original_replace = aiofiles.os.replace

    async def delayed_replace(source, destination):
        entered_replace.set()
        await release_replace.wait()
        await original_replace(source, destination)

    monkeypatch.setattr(aiofiles.os, "replace", delayed_replace)
    profile = tmp_path / "profile"
    token = set_hermes_home_override(profile)
    try:
        task = asyncio.create_task(
            approval.save_permanent_allowlist({"durable-pattern"})
        )
        await entered_replace.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        release_replace.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release_replace.set()
        reset_hermes_home_override(token)

    async with aiofiles.open(profile / "config.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(await handle.read())
    assert config["command_allowlist"] == ["durable-pattern"]
    assert not [
        path
        for path in profile.iterdir()
        if path.name.startswith(".config_") and path.suffix == ".tmp"
    ]


@pytest.mark.asyncio
async def test_approval_callback_cancellation_propagates_without_caching(
    monkeypatch,
    tmp_path,
):
    entered = asyncio.Event()

    async def load_config_readonly():
        return {"approvals": {"mode": "manual"}}

    async def wait_for_user(*_args, **_kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        load_config_readonly,
    )
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    token = set_hermes_home_override(tmp_path / "profile")
    session_token = approval.set_current_session_key("cancelled-session")
    try:
        task = asyncio.create_task(
            approval.check_dangerous_command(
                "rm -rf build",
                "local",
                approval_callback=wait_for_user,
            )
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert approval.is_approved(
            "cancelled-session",
            "recursive delete",
        ) is False
    finally:
        approval.reset_current_session_key(session_token)
        reset_hermes_home_override(token)


def test_allowlist_lock_does_not_retain_closed_event_loops(tmp_path):
    profile = tmp_path / "profile"

    async def save_and_capture_loop():
        token = set_hermes_home_override(profile)
        try:
            await approval.save_permanent_allowlist({"loop-owned"})
            return weakref.ref(asyncio.get_running_loop())
        finally:
            reset_hermes_home_override(token)

    loop_ref = asyncio.run(save_and_capture_loop())
    gc.collect()
    assert loop_ref() is None


def test_contended_allowlist_lock_does_not_retain_closed_event_loop(
    monkeypatch,
    tmp_path,
):
    profile = tmp_path / "profile"
    first_replace_entered: asyncio.Event | None = None
    release_first_replace: asyncio.Event | None = None
    original_replace = aiofiles.os.replace

    async def save_concurrently():
        nonlocal first_replace_entered, release_first_replace
        first_replace_entered = asyncio.Event()
        release_first_replace = asyncio.Event()
        replace_calls = 0

        async def delayed_first_replace(source, destination):
            nonlocal replace_calls
            replace_calls += 1
            if replace_calls == 1:
                first_replace_entered.set()
                await release_first_replace.wait()
            await original_replace(source, destination)

        monkeypatch.setattr(aiofiles.os, "replace", delayed_first_replace)
        token = set_hermes_home_override(profile)
        try:
            first = asyncio.create_task(
                approval.save_permanent_allowlist({"first"})
            )
            await first_replace_entered.wait()
            second = asyncio.create_task(
                approval.save_permanent_allowlist({"second"})
            )
            await asyncio.sleep(0)
            release_first_replace.set()
            await asyncio.gather(first, second)
            return weakref.ref(asyncio.get_running_loop())
        finally:
            reset_hermes_home_override(token)

    loop_ref = asyncio.run(save_concurrently())
    monkeypatch.setattr(aiofiles.os, "replace", original_replace)
    first_replace_entered = None
    release_first_replace = None
    gc.collect()
    assert loop_ref() is None


def test_sync_private_monkeypatch_contract_is_preserved(monkeypatch):
    monkeypatch.setattr(approval, "_approval_config_snapshot", {"mode": False})
    monkeypatch.setattr(approval, "_session_approved", {})
    monkeypatch.setattr(approval, "_session_yolo", set())
    monkeypatch.setattr(approval, "_permanent_approved", set())

    approval.approve_session("session", "pattern")
    approval.approve_permanent("permanent")
    approval._session_yolo.add("session")

    assert approval._get_approval_mode() == "off"
    assert approval.is_approved("session", "pattern") is True
    assert approval.is_approved("other", "permanent") is True
    assert approval.is_session_yolo_enabled("session") is True


@pytest.mark.asyncio
async def test_sudo_stdin_guard_uses_only_active_profile_password(monkeypatch):
    async def load_config_readonly():
        return {"approvals": {"mode": "manual"}}

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        load_config_readonly,
    )
    monkeypatch.setenv("HERMES_INTERACTIVE", "0")

    # Preserve upstream's single-profile presence contract: an explicitly
    # empty process value is still configured, while absence triggers the
    # unconditional stdin-password guard.
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    assert await approval._check_sudo_stdin_guard("sudo -S whoami") == (
        True,
        "sudo password guessing via stdin (sudo -S)",
    )
    monkeypatch.setenv("SUDO_PASSWORD", "")
    assert await approval._check_sudo_stdin_guard("sudo -S whoami") == (
        False,
        None,
    )

    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.setenv("SUDO_PASSWORD", "foreign-profile-password")

    missing_token = secret_scope.set_secret_scope({})
    try:
        blocked = await approval.check_all_command_guards(
            "echo guessed | sudo -S whoami",
            "local",
        )
        mounted_docker = await approval.check_all_command_guards(
            "sudo -S whoami",
            "docker",
            has_host_access=True,
        )
        isolated_docker = await approval.check_all_command_guards(
            "sudo -S whoami",
            "docker",
        )
    finally:
        secret_scope.reset_secret_scope(missing_token)

    configured_token = secret_scope.set_secret_scope({"SUDO_PASSWORD": ""})
    try:
        allowed = await approval.check_all_command_guards(
            "sudo -S whoami",
            "local",
        )
    finally:
        secret_scope.reset_secret_scope(configured_token)

    assert blocked["approved"] is False
    assert "sudo password guessing" in blocked["message"]
    assert mounted_docker["approved"] is False
    assert "sudo password guessing" in mounted_docker["message"]
    assert isolated_docker == {"approved": True, "message": None}
    assert allowed == {"approved": True, "message": None}


@pytest.mark.asyncio
async def test_sudo_stdin_guard_fails_closed_without_multiplex_scope(monkeypatch):
    async def load_config_readonly():
        return {"approvals": {"mode": "manual"}}

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        load_config_readonly,
    )
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.setenv("SUDO_PASSWORD", "foreign-profile-password")

    with pytest.raises(secret_scope.UnscopedSecretError):
        await approval.check_all_command_guards("sudo -S whoami", "local")
