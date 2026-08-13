"""Profile isolation for skill required-environment readiness."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import aiofiles
import pytest

from agent import secret_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import env_passthrough, skills_tool


_REQUIRED_NAME = "PROFILE_SKILL_REQUIRED_TOKEN"
_SKILL_NAME = "profile-required-env"


@pytest.fixture(autouse=True)
def _restore_profile_readiness_state(monkeypatch):
    previous_multiplex = secret_scope.is_multiplex_active()
    scope_token = secret_scope.set_secret_scope(None)
    passthrough_token = env_passthrough._allowed_env_vars_var.set(set())
    previous_capture = skills_tool._secret_capture_callback
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    skills_tool.set_secret_capture_callback(None)
    skills_tool._SKILLS_CACHE.clear()
    try:
        yield
    finally:
        skills_tool.set_secret_capture_callback(previous_capture)
        skills_tool._SKILLS_CACHE.clear()
        env_passthrough._allowed_env_vars_var.reset(passthrough_token)
        secret_scope.reset_secret_scope(scope_token)
        secret_scope.set_multiplex_active(previous_multiplex)


def _write_skill(home: Path, *, required_name: str = _REQUIRED_NAME) -> None:
    skill_dir = home / "skills" / _SKILL_NAME
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {_SKILL_NAME}\n"
        "description: Profile-scoped readiness fixture.\n"
        "required_environment_variables:\n"
        f"  - name: {required_name}\n"
        "    prompt: Enter the profile token\n"
        "---\n\n"
        "# Profile readiness\n\n"
        "Use the configured profile token.\n",
        encoding="utf-8",
    )


async def _view_in_profile(
    home: Path,
    secrets: dict[str, str],
) -> tuple[dict, bool]:
    home_token = set_hermes_home_override(home)
    secret_token = secret_scope.set_secret_scope(secrets)
    passthrough_token = env_passthrough._allowed_env_vars_var.set(set())
    try:
        payload = json.loads(
            await skills_tool.skill_view(_SKILL_NAME, preprocess=False)
        )
        registered = await env_passthrough.is_env_passthrough(_REQUIRED_NAME)
        return payload, registered
    finally:
        env_passthrough._allowed_env_vars_var.reset(passthrough_token)
        secret_scope.reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


@pytest.mark.asyncio
async def test_concurrent_profiles_do_not_borrow_foreign_process_secret(
    monkeypatch,
    tmp_path,
):
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    _write_skill(profile_a)
    _write_skill(profile_b)
    monkeypatch.setenv(_REQUIRED_NAME, "foreign-process-poison")
    secret_scope.set_multiplex_active(True)

    (result_a, registered_a), (result_b, registered_b) = await asyncio.gather(
        _view_in_profile(profile_a, {_REQUIRED_NAME: "profile-a-value"}),
        _view_in_profile(profile_b, {_REQUIRED_NAME: ""}),
    )

    assert result_a["readiness_status"] == "available"
    assert result_a["missing_required_environment_variables"] == []
    assert registered_a is True
    assert result_b["readiness_status"] == "setup_needed"
    assert result_b["missing_required_environment_variables"] == [
        _REQUIRED_NAME
    ]
    assert registered_b is False


@pytest.mark.asyncio
async def test_unscoped_multiplex_readiness_fails_closed(
    monkeypatch,
    tmp_path,
):
    home = tmp_path / "unscoped"
    _write_skill(home)
    monkeypatch.setenv(_REQUIRED_NAME, "foreign-process-poison")
    secret_scope.set_multiplex_active(True)
    home_token = set_hermes_home_override(home)
    try:
        with pytest.raises(
            secret_scope.UnscopedSecretError,
            match=_REQUIRED_NAME,
        ):
            await skills_tool.skill_view(_SKILL_NAME, preprocess=False)
    finally:
        reset_hermes_home_override(home_token)

    assert await env_passthrough.is_env_passthrough(_REQUIRED_NAME) is False


@pytest.mark.asyncio
async def test_single_profile_empty_dotenv_retains_process_fallback(
    monkeypatch,
    tmp_path,
):
    home = tmp_path / "legacy"
    _write_skill(home)
    (home / ".env").write_text(f"{_REQUIRED_NAME}=\n", encoding="utf-8")
    monkeypatch.setenv(_REQUIRED_NAME, "legacy-process-value")

    result, registered = await _view_in_profile(home, {})

    assert result["readiness_status"] == "available"
    assert result["missing_required_environment_variables"] == []
    assert registered is True


@pytest.mark.asyncio
async def test_single_profile_nonempty_dotenv_remains_available(
    monkeypatch,
    tmp_path,
):
    home = tmp_path / "dotenv"
    _write_skill(home)
    (home / ".env").write_text(
        f"{_REQUIRED_NAME}=dotenv-value\n",
        encoding="utf-8",
    )
    monkeypatch.delenv(_REQUIRED_NAME, raising=False)

    result, registered = await _view_in_profile(home, {})

    assert result["readiness_status"] == "available"
    assert result["missing_required_environment_variables"] == []
    assert registered is True


@pytest.mark.asyncio
async def test_capture_callback_reloads_profile_dotenv_and_registers_passthrough(
    monkeypatch,
    tmp_path,
):
    home = tmp_path / "captured"
    _write_skill(home)
    monkeypatch.setenv(_REQUIRED_NAME, "foreign-process-poison")
    secret_scope.set_multiplex_active(True)
    calls = []

    async def capture(name, prompt, metadata):
        calls.append((name, prompt, metadata))
        async with aiofiles.open(home / ".env", "w", encoding="utf-8") as handle:
            await handle.write(f"{name}=captured-profile-value\n")
        return {"success": True, "skipped": False}

    skills_tool.set_secret_capture_callback(capture)
    result, registered = await _view_in_profile(home, {})

    assert calls == [
        (
            _REQUIRED_NAME,
            "Enter the profile token",
            {"skill_name": _SKILL_NAME},
        )
    ]
    assert result["readiness_status"] == "available"
    assert result["missing_required_environment_variables"] == []
    assert registered is True


@pytest.mark.asyncio
async def test_capture_callback_cancellation_still_propagates(
    monkeypatch,
    tmp_path,
):
    home = tmp_path / "cancelled"
    _write_skill(home)
    monkeypatch.setenv(_REQUIRED_NAME, "foreign-process-poison")
    secret_scope.set_multiplex_active(True)

    async def cancelled(*_args):  # noqa: ASYNC124 - cancellation test double
        raise asyncio.CancelledError

    skills_tool.set_secret_capture_callback(cancelled)
    home_token = set_hermes_home_override(home)
    secret_token = secret_scope.set_secret_scope({})
    try:
        with pytest.raises(asyncio.CancelledError):
            await skills_tool.skill_view(_SKILL_NAME, preprocess=False)
    finally:
        secret_scope.reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)

    assert await env_passthrough.is_env_passthrough(_REQUIRED_NAME) is False
