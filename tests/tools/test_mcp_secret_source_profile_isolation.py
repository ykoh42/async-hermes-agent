"""Security regressions for external-secret MCP subprocess environments."""

from __future__ import annotations

import asyncio
import os

import pytest

from hermes_cli import env_loader
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.mcp_tool import _build_safe_env


@pytest.fixture(autouse=True)
def _reset_source_state():
    env_loader.reset_secret_source_cache()
    yield
    env_loader.reset_secret_source_cache()


def _seed_source(profile, value: str) -> None:
    lexical = env_loader._lexical_home_key(profile)
    canonical = os.path.normcase(os.path.realpath(profile))
    env_loader._SECRET_SOURCE_HOME_ALIASES[lexical] = canonical
    env_loader._SECRET_SOURCES_BY_HOME[canonical] = {
        "MCP_API_TOKEN": "command",
    }
    env_loader._SECRET_SOURCE_VALUES_BY_HOME[canonical] = {
        "MCP_API_TOKEN": value,
    }


@pytest.mark.asyncio
async def test_mcp_source_values_are_isolated_between_concurrent_profiles(
    tmp_path,
    monkeypatch,
):
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    profile_a.mkdir()
    profile_b.mkdir()
    _seed_source(profile_a, "alpha-token")
    _seed_source(profile_b, "beta-token")
    # Simulate the process environment retaining the first-loaded profile.
    monkeypatch.setenv("MCP_API_TOKEN", "alpha-stale-process-token")
    monkeypatch.setenv("PATH", "/usr/bin")

    async def build(profile) -> dict:
        token = set_hermes_home_override(profile)
        try:
            await asyncio.sleep(0)
            return _build_safe_env(None)
        finally:
            reset_hermes_home_override(token)

    env_a, env_b = await asyncio.gather(build(profile_a), build(profile_b))

    assert env_a["MCP_API_TOKEN"] == "alpha-token"
    assert env_b["MCP_API_TOKEN"] == "beta-token"
    assert "alpha-stale-process-token" not in env_b.values()


@pytest.mark.asyncio
async def test_profile_without_source_does_not_inherit_process_provenance(
    tmp_path,
    monkeypatch,
):
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    profile_a.mkdir()
    profile_b.mkdir()
    _seed_source(profile_a, "alpha-token")
    monkeypatch.setenv("MCP_API_TOKEN", "alpha-token")

    token = set_hermes_home_override(profile_b)
    try:
        result = _build_safe_env(None)
        assert env_loader.get_secret_source("MCP_API_TOKEN") is None
    finally:
        reset_hermes_home_override(token)

    assert "MCP_API_TOKEN" not in result


def test_unassociated_legacy_snapshot_cannot_authorize_process_secret(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "process-profile"))
    monkeypatch.setenv("MCP_API_TOKEN", "other-profile-token")
    env_loader._SECRET_SOURCES["MCP_API_TOKEN"] = "command"

    result = _build_safe_env(None)

    assert "MCP_API_TOKEN" not in result


@pytest.mark.asyncio
async def test_canonical_profile_alias_reuses_only_its_source_snapshot(
    tmp_path,
):
    profile = tmp_path / "profile"
    alias = tmp_path / "profile-alias"
    profile.mkdir()
    alias.symlink_to(profile, target_is_directory=True)
    _seed_source(profile, "profile-token")
    # This awaited boundary records the alias-to-canonical relationship.
    await env_loader.get_secret_source_values(alias)

    token = set_hermes_home_override(alias)
    try:
        result = _build_safe_env(None)
        source = env_loader.get_secret_source("MCP_API_TOKEN")
    finally:
        reset_hermes_home_override(token)

    assert source == "command"
    assert result["MCP_API_TOKEN"] == "profile-token"
