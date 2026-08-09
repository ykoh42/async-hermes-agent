"""Tests for tools.environments.local.build_subprocess_env — the single
factory for child-process environments (profile-home + secret-scrub owner).
"""

import asyncio
import json
import os
import sys

import pytest
from blockbuster import BlockBuster
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

from tools.environments.local import build_subprocess_env


# ---------------------------------------------------------------------------
# Unit: scrub path delegates to _sanitize_subprocess_env semantics
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.asyncio


async def test_scrub_on_strips_provider_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    env = await build_subprocess_env()
    assert "ANTHROPIC_API_KEY" not in env


async def test_scrub_on_strips_dynamic_internal_secret(monkeypatch):
    monkeypatch.setenv("AUXILIARY_VISION_API_KEY", "sk-aux")
    monkeypatch.setenv("GATEWAY_RELAY_FOO_TOKEN", "tok")
    env = await build_subprocess_env()
    assert "AUXILIARY_VISION_API_KEY" not in env
    assert "GATEWAY_RELAY_FOO_TOKEN" not in env


async def test_scrub_on_forwards_extra_like_sanitize_extra_env(monkeypatch):
    env = await build_subprocess_env(extra={"MY_HARMLESS_VAR": "1"})
    assert env.get("MY_HARMLESS_VAR") == "1"
    # extra still goes through the blocklist on the scrub path
    env2 = await build_subprocess_env(extra={"ANTHROPIC_API_KEY": "sk"})
    assert "ANTHROPIC_API_KEY" not in env2


async def test_force_prefix_only_overrides_on_explicit_extra():
    base = {
        "PATH": "/bin",
        "_HERMES_FORCE_OPENAI_API_KEY": "must-not-leak",
    }

    without_extra = await build_subprocess_env(base)
    forced = await build_subprocess_env(
        base,
        extra={"_HERMES_FORCE_OPENAI_API_KEY": "explicit"},
    )
    dynamic_secret = await build_subprocess_env(
        base,
        extra={"_HERMES_FORCE_AUXILIARY_VISION_API_KEY": "must-not-leak"},
    )

    assert "_HERMES_FORCE_OPENAI_API_KEY" not in without_extra
    assert "OPENAI_API_KEY" not in without_extra
    assert forced["OPENAI_API_KEY"] == "explicit"
    assert "AUXILIARY_VISION_API_KEY" not in dynamic_secret


# ---------------------------------------------------------------------------
# Unit: no-scrub path preserves content exactly
# ---------------------------------------------------------------------------


async def test_no_scrub_inherit_profile_home_bridges_context_override(tmp_path):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    token = set_hermes_home_override(str(tmp_path))
    try:
        env = await build_subprocess_env(
            {"PATH": "/bin"}, scrub_secrets=False, inherit_profile_home=True
        )
    finally:
        reset_hermes_home_override(token)
    assert env["HERMES_HOME"] == str(tmp_path)


# ---------------------------------------------------------------------------
# E2E: real subprocess sees the factory's contract
# ---------------------------------------------------------------------------


async def test_e2e_child_sees_hermes_home_and_no_planted_secret(tmp_path, monkeypatch):
    """A real child spawned with a factory-built env must see HERMES_HOME
    propagated and (with scrub on) a planted provider-style key absent."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-FAKE-planted")
    monkeypatch.setenv("AUXILIARY_FAKE_API_KEY", "sk-FAKE-aux")

    env = await build_subprocess_env()  # scrub on (default)

    code = (
        "import os, json; "
        "print(json.dumps({'home': os.environ.get('HERMES_HOME'), "
        "'k1': 'ANTHROPIC_API_KEY' in os.environ, "
        "'k2': 'AUXILIARY_FAKE_API_KEY' in os.environ}))"
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        code,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)

    assert process.returncode == 0, stderr.decode(errors="replace")
    result = json.loads(stdout)
    assert result["home"] == str(hermes_home)
    assert result["k1"] is False
    assert result["k2"] is False


async def test_e2e_no_scrub_child_keeps_planted_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-FAKE-planted")
    env = await build_subprocess_env(
        scrub_secrets=False,
        inherit_profile_home=False,
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import os; print(os.environ.get('ANTHROPIC_API_KEY', ''))",
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
    assert process.returncode == 0, stderr.decode(errors="replace")
    assert stdout.decode().strip() == "sk-FAKE-planted"


@pytest.mark.parametrize("scrub_secrets", [True, False])
async def test_delegated_child_scrubs_kanban_env_on_both_build_paths(
    scrub_secrets,
):
    from agent.delegation_context import (
        DELEGATED_CHILD_ENV_MARKER,
        delegated_child_context,
    )

    base = {
        "PATH": "/bin",
        "HERMES_KANBAN_TASK": "parent-task",
        "HERMES_KANBAN_RUN_ID": "parent-run",
    }
    with delegated_child_context():
        env = await build_subprocess_env(
            base,
            scrub_secrets=scrub_secrets,
            inherit_profile_home=False,
        )

    assert "HERMES_KANBAN_TASK" not in env
    assert "HERMES_KANBAN_RUN_ID" not in env
    assert env[DELEGATED_CHILD_ENV_MARKER] == "1"


async def test_home_application_errors_remain_best_effort(monkeypatch):
    async def fail_home(_env):
        raise RuntimeError("synthetic HOME failure")

    monkeypatch.setattr("hermes_constants.apply_subprocess_home_env", fail_home)

    env = await build_subprocess_env({"PATH": "/bin"})

    assert env["PATH"] == "/bin"
    assert "HOME" not in env


async def test_home_application_cancellation_propagates(monkeypatch):
    entered = asyncio.Event()

    async def wait_forever(_env):
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr("hermes_constants.apply_subprocess_home_env", wait_forever)
    task = asyncio.create_task(build_subprocess_env({"PATH": "/bin"}))
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


async def test_concurrent_profiles_resolve_passthrough_values_independently(
    tmp_path,
):
    from agent.secret_scope import (
        reset_secret_scope,
        set_multiplex_active,
        set_secret_scope,
    )
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )
    from tools.env_passthrough import clear_env_passthrough, register_env_passthrough

    register_env_passthrough(["PROFILE_SCOPED_TOKEN"])
    set_multiplex_active(True)

    async def build_for(profile: str, value: str) -> dict[str, str]:
        profile_home = tmp_path / profile
        profile_home.mkdir()
        home_token = set_hermes_home_override(profile_home)
        secret_token = set_secret_scope({"PROFILE_SCOPED_TOKEN": value})
        try:
            return await build_subprocess_env({
                "HOME": str(tmp_path),
                "HERMES_REAL_HOME": str(tmp_path),
                "PATH": "/bin",
                "PROFILE_SCOPED_TOKEN": "foreign-global-value",
            })
        finally:
            reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)

    try:
        alpha, beta = await asyncio.gather(
            build_for("alpha", "alpha-secret"),
            build_for("beta", "beta-secret"),
        )
    finally:
        set_multiplex_active(False)
        clear_env_passthrough()

    assert alpha["HERMES_HOME"] == str(tmp_path / "alpha")
    assert beta["HERMES_HOME"] == str(tmp_path / "beta")
    assert alpha["PROFILE_SCOPED_TOKEN"] == "alpha-secret"
    assert beta["PROFILE_SCOPED_TOKEN"] == "beta-secret"


async def test_build_subprocess_env_is_nonblocking_and_leak_free(tmp_path):
    base = {
        "HOME": str(tmp_path),
        "HERMES_REAL_HOME": str(tmp_path),
        "PATH": "/bin",
    }
    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        blocker = BlockBuster()
        blocker.activate()
        try:
            env = await build_subprocess_env(base)
        finally:
            blocker.deactivate()

    assert env["HOME"] == str(tmp_path)
