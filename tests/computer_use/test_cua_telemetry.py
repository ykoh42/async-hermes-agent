"""Native-async contracts for cua-driver telemetry opt-in policy."""

from unittest.mock import AsyncMock, patch

import pytest

from tools.computer_use import cua_backend


pytestmark = pytest.mark.asyncio
_VAR = "CUA_DRIVER_RS_TELEMETRY_ENABLED"


async def test_explicit_false_disables():
    cfg = {"computer_use": {"cua_telemetry": False}}
    with patch(
        "hermes_cli.config.load_config_readonly", AsyncMock(return_value=cfg)
    ):
        assert await cua_backend._cua_telemetry_disabled() is True


async def test_config_load_failure_fails_safe():
    with patch(
        "hermes_cli.config.load_config_readonly",
        AsyncMock(side_effect=RuntimeError("boom")),
    ):
        assert await cua_backend._cua_telemetry_disabled() is True


async def test_disabled_injects_var_zero():
    with patch.object(
        cua_backend, "_cua_telemetry_disabled", AsyncMock(return_value=True)
    ):
        env = await cua_backend.cua_driver_child_env({"PATH": "/usr/bin"})
    assert env[_VAR] == "0"
    assert env["PATH"] == "/usr/bin"


async def test_disabled_overrides_inherited_enabled():
    with patch.object(
        cua_backend, "_cua_telemetry_disabled", AsyncMock(return_value=True)
    ):
        env = await cua_backend.cua_driver_child_env({_VAR: "1"})
    assert env[_VAR] == "0"
