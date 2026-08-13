"""Profile isolation for trusted skill inline-shell snippets."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agent import secret_scope, skill_preprocessing
from hermes_constants import (
    reset_hermes_home_override,
    set_hermes_home_override,
)
from tools import env_passthrough, skills_tool


_SKILL_NAME = "profile-inline-shell"
_REQUIRED_NAME = "PROFILE_INLINE_REQUIRED_TOKEN"
_UNDECLARED_NAME = "PROFILE_INLINE_UNDECLARED_SECRET"


@pytest.fixture(autouse=True)
def _restore_profile_environment_state(monkeypatch):
    previous_multiplex = secret_scope.is_multiplex_active()
    scope_token = secret_scope.set_secret_scope(None)
    passthrough_token = env_passthrough._allowed_env_vars_var.set(set())

    async def _no_config_passthrough() -> frozenset[str]:
        return frozenset()

    async def _inline_shell_config() -> dict[str, object]:
        return {"inline_shell": True, "inline_shell_timeout": 5}

    monkeypatch.setattr(
        env_passthrough,
        "_load_config_passthrough",
        _no_config_passthrough,
    )
    monkeypatch.setattr(
        skill_preprocessing,
        "load_skills_config",
        _inline_shell_config,
    )
    skills_tool._SKILLS_CACHE.clear()
    try:
        yield
    finally:
        skills_tool._SKILLS_CACHE.clear()
        env_passthrough._allowed_env_vars_var.reset(passthrough_token)
        secret_scope.reset_secret_scope(scope_token)
        secret_scope.set_multiplex_active(previous_multiplex)


def _write_inline_skill(home: Path) -> None:
    skill_dir = home / "skills" / _SKILL_NAME
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {_SKILL_NAME}\n"
        "description: Profile-scoped inline-shell fixture.\n"
        "required_environment_variables:\n"
        f"  - name: {_REQUIRED_NAME}\n"
        "  - name: OPENAI_API_KEY\n"
        "---\n\n"
        "# Inline shell profile fixture\n\n"
        f"required=!`printf '%s' \"${{{_REQUIRED_NAME}-<missing>}}\"`\n"
        f"undeclared=!`printf '%s' \"${{{_UNDECLARED_NAME}-<missing>}}\"`\n"
        "provider=!`printf '%s' \"${OPENAI_API_KEY-<missing>}\"`\n",
        encoding="utf-8",
    )


async def _view_profile_skill(home: Path, label: str) -> dict:
    home_token = set_hermes_home_override(home)
    scope_token = secret_scope.set_secret_scope(
        {
            _REQUIRED_NAME: f"profile-{label}-required",
            _UNDECLARED_NAME: f"profile-{label}-undeclared",
            "OPENAI_API_KEY": f"profile-{label}-provider",
        }
    )
    passthrough_token = env_passthrough._allowed_env_vars_var.set(set())
    try:
        return json.loads(await skills_tool.skill_view(_SKILL_NAME))
    finally:
        env_passthrough._allowed_env_vars_var.reset(passthrough_token)
        secret_scope.reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)


@pytest.mark.asyncio
async def test_concurrent_profile_skills_receive_only_scoped_required_env(
    monkeypatch,
    tmp_path: Path,
):
    homes = {
        label: tmp_path / f"profile-{label}"
        for label in ("a", "b")
    }
    for home in homes.values():
        _write_inline_skill(home)

    monkeypatch.setenv(_REQUIRED_NAME, "profile-a-process-poison")
    monkeypatch.setenv(_UNDECLARED_NAME, "profile-a-undeclared-poison")
    monkeypatch.setenv("OPENAI_API_KEY", "profile-a-provider-poison")
    secret_scope.set_multiplex_active(True)

    result_a, result_b = await asyncio.gather(
        _view_profile_skill(homes["a"], "a"),
        _view_profile_skill(homes["b"], "b"),
    )

    for label, result in (("a", result_a), ("b", result_b)):
        assert result["success"] is True
        assert result["readiness_status"] == "available"
        assert f"required=profile-{label}-required" in result["content"]
        assert "undeclared=<missing>" in result["content"]
        assert "provider=<missing>" in result["content"]
        assert "process-poison" not in result["content"]


@pytest.mark.asyncio
async def test_unscoped_multiplex_inline_shell_fails_closed(
    monkeypatch,
):
    monkeypatch.setenv(_REQUIRED_NAME, "foreign-process-poison")
    env_passthrough.register_env_passthrough([_REQUIRED_NAME])
    secret_scope.set_multiplex_active(True)

    with pytest.raises(
        secret_scope.UnscopedSecretError,
        match="inline-shell subprocess environment",
    ):
        await skill_preprocessing.run_inline_shell(
            f"printf '%s' \"${_REQUIRED_NAME}\"",
            None,
            timeout=5,
        )


@pytest.mark.asyncio
async def test_multiplex_global_passthrough_keeps_process_value(
    monkeypatch,
):
    process_locale = "process-global-locale"
    monkeypatch.setenv("LANG", process_locale)
    env_passthrough.register_env_passthrough(["LANG"])
    secret_scope.set_multiplex_active(True)
    scope_token = secret_scope.set_secret_scope({"LANG": "profile-poison"})
    try:
        output = await skill_preprocessing.run_inline_shell(
            "printf '%s' \"$LANG\"",
            None,
            timeout=5,
        )
    finally:
        secret_scope.reset_secret_scope(scope_token)

    assert output == process_locale


@pytest.mark.asyncio
async def test_multiplex_explicit_empty_passthrough_does_not_borrow_process_value(
    monkeypatch,
):
    monkeypatch.setenv(_REQUIRED_NAME, "foreign-process-poison")
    env_passthrough.register_env_passthrough([_REQUIRED_NAME])
    secret_scope.set_multiplex_active(True)
    scope_token = secret_scope.set_secret_scope({_REQUIRED_NAME: ""})
    try:
        output = await skill_preprocessing.run_inline_shell(
            f"printf '<%s>' \"${_REQUIRED_NAME}\"",
            None,
            timeout=5,
        )
    finally:
        secret_scope.reset_secret_scope(scope_token)

    assert output == "<>"


@pytest.mark.asyncio
async def test_single_profile_inline_shell_preserves_process_environment(
    monkeypatch,
):
    monkeypatch.setenv(_REQUIRED_NAME, "legacy-required")
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-provider")

    output = await skill_preprocessing.run_inline_shell(
        f"printf '%s|%s' \"${_REQUIRED_NAME}\" \"$OPENAI_API_KEY\"",
        None,
        timeout=5,
    )

    assert output == "legacy-required|legacy-provider"
