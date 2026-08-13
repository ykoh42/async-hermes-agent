"""Native-async behavior contracts for the cua-driver --no-overlay policy."""

import json
import sys
from unittest.mock import AsyncMock, patch

import pytest

from tools.computer_use import cua_backend


pytestmark = pytest.mark.asyncio


async def test_explicit_true_overrides():
    cfg = {"computer_use": {"no_overlay": True}}
    with patch(
        "hermes_cli.config.load_config_readonly", AsyncMock(return_value=cfg)
    ):
        assert await cua_backend._cua_no_overlay() is True


async def test_config_load_failure_falls_through_to_auto_detect():
    with patch(
        "hermes_cli.config.load_config_readonly",
        AsyncMock(side_effect=RuntimeError("boom")),
    ), patch.object(sys, "platform", "darwin"):
        assert await cua_backend._cua_no_overlay() is True


async def test_returns_true_when_help_shows_flag(monkeypatch):
    cua_backend._no_overlay_support.clear()
    monkeypatch.setattr(
        cua_backend,
        "_run_command",
        AsyncMock(
            return_value=(
                0,
                "Usage: cua-driver [OPTIONS]\n--no-overlay Disable cursor overlay",
                "",
            )
        ),
    )
    assert await cua_backend._cua_driver_supports_no_overlay("cua-driver") is True


async def test_help_probe_passes_sanitized_env(monkeypatch):
    cua_backend._no_overlay_support.clear()
    run = AsyncMock(return_value=(0, "--no-overlay in help", ""))
    monkeypatch.setattr(cua_backend, "_run_command", run)
    await cua_backend._cua_driver_supports_no_overlay("cua-driver")
    assert "env" in run.await_args.kwargs
    assert run.await_args.kwargs["env"] is not None


async def test_manifest_command_drives_support_probe(monkeypatch):
    manifest = {
        "mcp_invocation": {
            "command": "/opt/relocated/cua-driver",
            "args": ["mcp"],
        }
    }
    monkeypatch.setattr(
        cua_backend,
        "_run_command",
        AsyncMock(return_value=(0, json.dumps(manifest), "")),
    )
    monkeypatch.setattr(
        cua_backend, "_cua_no_overlay", AsyncMock(return_value=True)
    )
    probe = AsyncMock(return_value=True)
    monkeypatch.setattr(cua_backend, "_cua_driver_supports_no_overlay", probe)
    command, args = await cua_backend._resolve_mcp_invocation("/usr/bin/cua-driver")
    assert command == "/opt/relocated/cua-driver"
    probe.assert_awaited_with("/opt/relocated/cua-driver")
    assert args == ["mcp", "--no-overlay"]


async def test_probe_distinguishes_support_between_binaries(monkeypatch):
    monkeypatch.setattr(
        cua_backend, "_cua_no_overlay", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        cua_backend,
        "_cua_driver_supports_no_overlay",
        AsyncMock(side_effect=lambda cmd: cmd == "/opt/relocated/cua-driver"),
    )
    system = await cua_backend._mcp_args_with_overlay_flag(
        ["mcp"], driver_cmd="/usr/bin/cua-driver"
    )
    relocated = await cua_backend._mcp_args_with_overlay_flag(
        ["mcp"], driver_cmd="/opt/relocated/cua-driver"
    )
    assert "--no-overlay" not in system
    assert "--no-overlay" in relocated


async def test_appended_when_enabled_and_supported(monkeypatch):
    monkeypatch.setattr(
        cua_backend, "_cua_no_overlay", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        cua_backend,
        "_cua_driver_supports_no_overlay",
        AsyncMock(return_value=True),
    )
    assert await cua_backend._mcp_args_with_overlay_flag(["mcp"]) == [
        "mcp",
        "--no-overlay",
    ]


async def test_not_appended_when_disabled(monkeypatch):
    monkeypatch.setattr(
        cua_backend, "_cua_no_overlay", AsyncMock(return_value=False)
    )
    probe = AsyncMock(return_value=True)
    monkeypatch.setattr(cua_backend, "_cua_driver_supports_no_overlay", probe)
    assert await cua_backend._mcp_args_with_overlay_flag(["mcp"]) == ["mcp"]
    probe.assert_not_awaited()


async def test_does_not_mutate_original_list(monkeypatch):
    original = ["mcp"]
    monkeypatch.setattr(
        cua_backend, "_cua_no_overlay", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        cua_backend,
        "_cua_driver_supports_no_overlay",
        AsyncMock(return_value=True),
    )
    result = await cua_backend._mcp_args_with_overlay_flag(original)
    assert "--no-overlay" in result
    assert "--no-overlay" not in original
