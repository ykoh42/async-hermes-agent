"""Whole-script approval behavior retained for execute_code."""

from __future__ import annotations

import pytest

from tools import approval
from tools.terminal_tool import (
    _docker_has_host_access,
    set_approval_callback,
)


@pytest.fixture(autouse=True)
def _approval_state(monkeypatch):
    monkeypatch.setattr(
        approval,
        "_approval_config_snapshot",
        {"mode": "manual", "cron_mode": "deny"},
    )
    approval._session_approved.clear()
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)


@pytest.mark.asyncio
async def test_execute_code_skips_isolated_docker_without_host_mounts(monkeypatch):
    async def config_snapshot():
        approval._approval_config_snapshot = {
            "mode": "manual",
            "cron_mode": "deny",
        }

    monkeypatch.setattr(approval, "_load_approval_config_snapshot", config_snapshot)
    result = await approval.check_execute_code_guard(
        "import os",
        "docker",
        has_host_access=False,
    )
    assert result == {"approved": True, "message": None}


@pytest.mark.asyncio
async def test_execute_code_cron_deny_fails_closed(monkeypatch):
    async def config_snapshot():
        approval._approval_config_snapshot = {
            "mode": "manual",
            "cron_mode": "deny",
        }

    monkeypatch.setattr(approval, "_load_approval_config_snapshot", config_snapshot)
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    result = await approval.check_execute_code_guard("print('x')", "local")
    assert result["approved"] is False
    assert result["pattern_key"] == "execute_code"
    assert result["outcome"] == "blocked"


@pytest.mark.asyncio
async def test_execute_code_interactive_uses_per_call_terminal_guards(monkeypatch):
    async def config_snapshot():
        approval._approval_config_snapshot = {
            "mode": "manual",
            "cron_mode": "deny",
        }

    monkeypatch.setattr(approval, "_load_approval_config_snapshot", config_snapshot)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    result = await approval.check_execute_code_guard("print('x')", "local")
    assert result == {"approved": True, "message": None}


@pytest.mark.asyncio
async def test_execute_code_ask_uses_native_async_callback(monkeypatch):
    async def config_snapshot():
        approval._approval_config_snapshot = {
            "mode": "manual",
            "cron_mode": "deny",
        }

    async def approve_once(command, _description, **_kwargs):
        assert command.startswith("execute_code <<'PY'")
        return "once"

    monkeypatch.setattr(approval, "_load_approval_config_snapshot", config_snapshot)
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    set_approval_callback(approve_once)
    try:
        result = await approval.check_execute_code_guard(
            "print('x')",
            "local",
        )
    finally:
        set_approval_callback(None)
    assert result["approved"] is True
    assert result["user_approved"] is True


def test_docker_host_access_detects_bind_mount_shapes():
    assert _docker_has_host_access({"env_type": "docker", "docker_volumes": []}) is False
    assert _docker_has_host_access(
        {"env_type": "docker", "docker_volumes": ["/host:/workspace"]}
    ) is True
    assert _docker_has_host_access(
        {
            "env_type": "docker",
            "host_cwd": "/host/project",
            "docker_mount_cwd_to_workspace": True,
        }
    ) is True
