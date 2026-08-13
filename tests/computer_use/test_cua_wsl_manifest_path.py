from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from tools.computer_use import cua_backend


pytestmark = pytest.mark.asyncio


async def test_wsl_windows_manifest_path_translates_to_drvfs():
    with patch("hermes_constants.is_wsl", return_value=True):
        assert cua_backend._wsl_windows_path_to_posix(
            r"C:\Users\Fernando\AppData\Local\cua-driver\cua-driver.exe"
        ) == "/mnt/c/Users/Fernando/AppData/Local/cua-driver/cua-driver.exe"


async def test_resolve_mcp_invocation_normalizes_windows_manifest_command_in_wsl(
    monkeypatch,
):
    manifest = {
        "mcp_invocation": {
            "command": r"C:\Users\Fernando\AppData\Local\cua-driver\cua-driver.exe",
            "args": ["mcp"],
        }
    }
    monkeypatch.setattr(
        cua_backend,
        "_run_command",
        AsyncMock(return_value=(0, json.dumps(manifest), "")),
    )
    monkeypatch.setattr(
        cua_backend,
        "_mcp_args_with_overlay_flag",
        AsyncMock(side_effect=lambda args, driver_cmd="cua-driver": list(args)),
    )
    with patch("hermes_constants.is_wsl", return_value=True):
        command, args = await cua_backend._resolve_mcp_invocation("cua-driver")
    assert command == "/mnt/c/Users/Fernando/AppData/Local/cua-driver/cua-driver.exe"
    assert args == ["mcp"]
