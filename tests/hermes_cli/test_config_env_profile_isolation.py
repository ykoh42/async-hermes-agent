from __future__ import annotations

import asyncio

import pytest

from agent import secret_scope
from hermes_cli import config, managed_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture(autouse=True)
def _reset_config_profile_state(monkeypatch: pytest.MonkeyPatch):
    scope_token = secret_scope.set_secret_scope(None)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    config._LOAD_CONFIG_CACHE.clear()
    config._LAST_EXPANDED_CONFIG_BY_PATH.clear()
    config._LAST_READONLY_CONFIG_SOURCES_BY_PATH.clear()
    managed_scope.invalidate_managed_cache()
    yield
    secret_scope.reset_secret_scope(scope_token)


def _write_profile_config(home, value: str = "${PROFILE_API_KEY}") -> None:
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n"
        "  default: custom/model\n"
        "auxiliary:\n"
        "  vision:\n"
        f"    api_key: {value}\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_user_config_env_refs_are_isolated_by_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("PROFILE_API_KEY", "foreign-process-key")
    homes = {name: tmp_path / name for name in ("a", "b")}
    for home in homes.values():
        _write_profile_config(home)

    async def load(name: str) -> str:
        home_token = set_hermes_home_override(homes[name])
        scope_token = secret_scope.set_secret_scope(
            {"PROFILE_API_KEY": f"profile-{name}-key"}
        )
        try:
            loaded = await config.load_config_readonly()
            cached = await config.load_config_readonly()
            assert cached is loaded
            return loaded["auxiliary"]["vision"]["api_key"]
        finally:
            secret_scope.reset_secret_scope(scope_token)
            reset_hermes_home_override(home_token)

    assert await asyncio.gather(load("a"), load("b")) == [
        "profile-a-key",
        "profile-b-key",
    ]


@pytest.mark.asyncio
async def test_profile_secret_rotation_invalidates_only_its_cached_expansion(
    tmp_path,
) -> None:
    home = tmp_path / "profile"
    _write_profile_config(home)
    home_token = set_hermes_home_override(home)
    try:
        first_token = secret_scope.set_secret_scope({"PROFILE_API_KEY": "first"})
        try:
            first = await config.load_config_readonly()
        finally:
            secret_scope.reset_secret_scope(first_token)
        second_token = secret_scope.set_secret_scope({"PROFILE_API_KEY": "second"})
        try:
            second = await config.load_config_readonly()
        finally:
            secret_scope.reset_secret_scope(second_token)
    finally:
        reset_hermes_home_override(home_token)

    assert first["auxiliary"]["vision"]["api_key"] == "first"
    assert second["auxiliary"]["vision"]["api_key"] == "second"
    assert second is not first


@pytest.mark.asyncio
async def test_user_config_expansion_fails_closed_without_profile_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    home = tmp_path / "profile"
    _write_profile_config(home)
    monkeypatch.setenv("PROFILE_API_KEY", "foreign-process-key")
    home_token = set_hermes_home_override(home)
    try:
        with pytest.raises(secret_scope.UnscopedSecretError):
            await config.load_config_readonly()
    finally:
        reset_hermes_home_override(home_token)


@pytest.mark.asyncio
async def test_managed_env_refs_remain_process_scoped_and_invalidate_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    home = tmp_path / "profile"
    managed = tmp_path / "managed"
    _write_profile_config(home, "user-literal")
    managed.mkdir()
    (managed / "config.yaml").write_text(
        "auxiliary:\n  vision:\n    api_key: ${MANAGED_API_KEY}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("MANAGED_API_KEY", "managed-first")
    home_token = set_hermes_home_override(home)
    scope_token = secret_scope.set_secret_scope(
        {"MANAGED_API_KEY": "profile-must-not-win"}
    )
    try:
        first = await config.load_config_readonly()
        monkeypatch.setenv("MANAGED_API_KEY", "managed-second")
        second = await config.load_config_readonly()
    finally:
        secret_scope.reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)

    assert first["auxiliary"]["vision"]["api_key"] == "managed-first"
    assert second["auxiliary"]["vision"]["api_key"] == "managed-second"
    assert second is not first


@pytest.mark.asyncio
async def test_malformed_config_lkg_reexpands_for_current_profile(
    tmp_path,
) -> None:
    home = tmp_path / "profile"
    _write_profile_config(home)
    home_token = set_hermes_home_override(home)
    try:
        first_token = secret_scope.set_secret_scope(
            {"PROFILE_API_KEY": "profile-a-secret"}
        )
        try:
            first = await config.load_config_readonly()
        finally:
            secret_scope.reset_secret_scope(first_token)

        (home / "config.yaml").write_text(
            "auxiliary:\n  vision: [unterminated\n",
            encoding="utf-8",
        )
        second_token = secret_scope.set_secret_scope(
            {"PROFILE_API_KEY": "profile-b-secret"}
        )
        try:
            second = await config.load_config_readonly()
            cached = await config.load_config_readonly()
        finally:
            secret_scope.reset_secret_scope(second_token)
    finally:
        reset_hermes_home_override(home_token)

    assert first["auxiliary"]["vision"]["api_key"] == "profile-a-secret"
    assert second["auxiliary"]["vision"]["api_key"] == "profile-b-secret"
    assert cached is second
