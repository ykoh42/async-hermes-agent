"""Every retained native cua-driver spawn path receives a sanitized env."""

import json
from unittest.mock import AsyncMock

import pytest


pytestmark = pytest.mark.asyncio
SECRET = "sk-super-secret-should-not-leak"


def _assert_sanitized(env):
    assert env is not None
    assert "ANTHROPIC_API_KEY" not in env
    assert env.get("PATH") == "/usr/bin:/bin"
    assert env.get("CUA_DRIVER_RS_TELEMETRY_ENABLED") == "0"


async def test_resolve_mcp_invocation_sanitizes_env(monkeypatch):
    from tools.computer_use import cua_backend

    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    captured = {}
    manifest = json.dumps(
        {"mcp_invocation": {"command": "cua-driver", "args": ["mcp"]}}
    )

    async def fake_run(command, **kwargs):
        captured["env"] = kwargs.get("env")
        return 0, manifest, ""

    monkeypatch.setattr(cua_backend, "_run_command", fake_run)
    command, _args = await cua_backend._resolve_mcp_invocation("cua-driver")
    assert command == "cua-driver"
    _assert_sanitized(captured["env"])


async def test_update_check_sanitizes_env(monkeypatch):
    from tools.computer_use import cua_backend

    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(
        cua_backend,
        "resolve_cua_driver_cmd",
        AsyncMock(return_value="cua-driver"),
    )
    captured = {}
    payload = json.dumps(
        {
            "current_version": "1.0.0",
            "latest_version": "1.0.0",
            "update_available": False,
        }
    )

    async def fake_run(command, **kwargs):
        captured["env"] = kwargs.get("env")
        return 0, payload, ""

    monkeypatch.setattr(cua_backend, "_run_command", fake_run)
    result = await cua_backend.cua_driver_update_check(timeout=1.0)
    assert result["update_available"] is False
    _assert_sanitized(captured["env"])


async def test_cli_fallback_sanitizes_env(monkeypatch):
    from tools.computer_use import cua_backend

    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(
        cua_backend,
        "resolve_cua_driver_cmd",
        AsyncMock(return_value="cua-driver"),
    )
    captured = {}

    async def fake_run(command, **kwargs):
        captured["env"] = kwargs.get("env")
        return 0, json.dumps({"tree_markdown": "root"}), ""

    monkeypatch.setattr(cua_backend, "_run_command", fake_run)
    session = object.__new__(cua_backend._CuaDriverSession)
    session._embedded_daemon = None
    result = await session._call_tool_via_cli("list_windows", {}, timeout=5.0)
    assert result["isError"] is False
    _assert_sanitized(captured["env"])
