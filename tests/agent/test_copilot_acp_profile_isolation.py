"""Profile-scoped subprocess credentials for Copilot ACP."""

from __future__ import annotations

import asyncio

import pytest

from agent import secret_scope
from agent.copilot_acp_client import CopilotACPClient, _build_subprocess_env


@pytest.fixture(autouse=True)
def _restore_secret_scope():
    previous = secret_scope.is_multiplex_active()
    token = secret_scope.set_secret_scope(None)
    yield
    secret_scope.reset_secret_scope(token)
    secret_scope.set_multiplex_active(previous)


@pytest.mark.asyncio
async def test_acp_child_replaces_process_provider_keys_with_profile_scope(
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "process-profile-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "process-anthropic-key")
    monkeypatch.setenv("GH_TOKEN", "process-github-token")
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({
        "OPENAI_API_KEY": "scoped-profile-key",
        "GH_TOKEN": "scoped-github-token",
    })
    try:
        env = await _build_subprocess_env()
    finally:
        secret_scope.reset_secret_scope(token)

    assert env["OPENAI_API_KEY"] == "scoped-profile-key"
    assert "ANTHROPIC_API_KEY" not in env
    assert "GH_TOKEN" not in env


@pytest.mark.asyncio
async def test_acp_child_empty_profile_does_not_inherit_process_provider_keys(
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "process-profile-key")
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({})
    try:
        env = await _build_subprocess_env()
    finally:
        secret_scope.reset_secret_scope(token)

    assert "OPENAI_API_KEY" not in env


@pytest.mark.asyncio
async def test_acp_child_unscoped_multiplex_build_fails_closed():
    secret_scope.set_multiplex_active(True)

    with pytest.raises(secret_scope.UnscopedSecretError):
        await _build_subprocess_env()


async def _construct_in_scope(secrets: dict[str, str]) -> CopilotACPClient:
    token = secret_scope.set_secret_scope(secrets)
    try:
        await asyncio.sleep(0)
        return CopilotACPClient()
    finally:
        secret_scope.reset_secret_scope(token)


@pytest.mark.asyncio
async def test_direct_client_constructor_isolates_concurrent_profile_config(
    monkeypatch,
):
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", "foreign-copilot")
    monkeypatch.setenv("COPILOT_CLI_PATH", "foreign-cli-path")
    monkeypatch.setenv("HERMES_COPILOT_ACP_ARGS", "--foreign")
    secret_scope.set_multiplex_active(True)

    profile_a, profile_b = await asyncio.gather(
        _construct_in_scope(
            {
                "HERMES_COPILOT_ACP_COMMAND": "copilot-a",
                "COPILOT_CLI_PATH": "",
                "HERMES_COPILOT_ACP_ARGS": "--profile a",
            }
        ),
        _construct_in_scope(
            {
                "HERMES_COPILOT_ACP_COMMAND": "",
                "COPILOT_CLI_PATH": "copilot-b",
                "HERMES_COPILOT_ACP_ARGS": "--profile b",
            }
        ),
    )

    assert profile_a._acp_command == "copilot-a"
    assert profile_a._acp_args == ["--profile", "a"]
    assert profile_b._acp_command == "copilot-b"
    assert profile_b._acp_args == ["--profile", "b"]


def test_direct_client_constructor_fails_closed_without_profile_scope(
    monkeypatch,
):
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", "foreign-copilot")
    secret_scope.set_multiplex_active(True)

    with pytest.raises(
        secret_scope.UnscopedSecretError,
        match="HERMES_COPILOT_ACP_COMMAND",
    ):
        CopilotACPClient()


def test_direct_client_explicit_config_bypasses_profile_env_lookup():
    secret_scope.set_multiplex_active(True)

    client = CopilotACPClient(
        acp_command="explicit-copilot",
        acp_args=["--explicit"],
    )

    assert client._acp_command == "explicit-copilot"
    assert client._acp_args == ["--explicit"]


def test_direct_client_single_profile_env_precedence_is_unchanged(monkeypatch):
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", "single-profile-command")
    monkeypatch.setenv("COPILOT_CLI_PATH", "lower-priority-command")
    monkeypatch.setenv("HERMES_COPILOT_ACP_ARGS", "--single profile")

    client = CopilotACPClient()

    assert client._acp_command == "single-profile-command"
    assert client._acp_args == ["--single", "profile"]
