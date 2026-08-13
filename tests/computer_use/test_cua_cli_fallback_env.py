"""The native CLI fallback sanitizes its subprocess environment."""

import json
from unittest.mock import AsyncMock

import pytest

from tools.computer_use.cua_backend import _CuaDriverSession


pytestmark = pytest.mark.asyncio


async def test_cli_fallback_strips_provider_secret_from_subprocess_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "redacted")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(
        "tools.computer_use.cua_backend.resolve_cua_driver_cmd",
        AsyncMock(return_value="/resolved/cua-driver"),
    )
    captured = {}

    async def fake_run(command, **kwargs):
        captured["env"] = kwargs.get("env")
        return 0, json.dumps({"tree_markdown": "root"}), ""

    monkeypatch.setattr("tools.computer_use.cua_backend._run_command", fake_run)
    session = object.__new__(_CuaDriverSession)
    session._embedded_daemon = None
    result = await session._call_tool_via_cli("list_windows", {}, timeout=5.0)
    assert result["isError"] is False
    assert captured["env"] is not None
    assert "ANTHROPIC_API_KEY" not in captured["env"]
    assert captured["env"].get("PATH") == "/usr/bin:/bin"
