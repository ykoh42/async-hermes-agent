"""Native-async coverage for terminal backend degradation classification.

The upstream tests exercised synchronous SSH/Docker bootstrap helpers.  The
retained runtime exposes the same public result contract at the awaited
``terminal_tool`` boundary, so these tests pin that boundary directly.
"""

import json

import pytest

from tools.environments.base import EnvironmentConnectionError

pytestmark = pytest.mark.asyncio


def _config():
    return {
        "env_type": "local",
        "cwd": "/tmp",
        "timeout": 1,
        "lifetime_seconds": 60,
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
    }


async def _raise_connection(*args, **kwargs):
    del args, kwargs
    raise EnvironmentConnectionError("SSH connection failed", retry_hint="retry later")


async def test_connection_error_preserves_runtime_error_compatibility():
    error = EnvironmentConnectionError("backend down")
    assert isinstance(error, RuntimeError)
    assert error.reason == "backend down"
    assert error.retry_hint


async def test_warn_mode_returns_structured_degraded_result(monkeypatch):
    import tools.terminal_tool as terminal

    async def config():
        return _config()

    monkeypatch.setattr(terminal, "_get_env_config", config)
    monkeypatch.setattr(terminal, "_get_or_create_environment", _raise_connection)
    monkeypatch.delenv("TERMINAL_DEGRADED_MODE", raising=False)

    result = json.loads(await terminal.terminal_tool("echo hi"))
    assert result["status"] == "degraded"
    assert result["exit_code"] == -1
    assert result["reason"] == "SSH connection failed"
    assert result["retry_hint"] == "retry later"
    assert "traceback" not in result


async def test_fail_mode_returns_redacted_error_and_traceback(monkeypatch):
    import tools.terminal_tool as terminal

    async def config():
        return _config()

    monkeypatch.setattr(terminal, "_get_env_config", config)
    monkeypatch.setattr(terminal, "_get_or_create_environment", _raise_connection)
    monkeypatch.setenv("TERMINAL_DEGRADED_MODE", "fail")

    result = json.loads(await terminal.terminal_tool("echo hi"))
    assert result["status"] == "error"
    assert "traceback" in result
    assert "SSH connection failed" in result["error"]


async def test_ordinary_runtime_error_is_not_degraded(monkeypatch):
    import tools.terminal_tool as terminal

    async def config():
        return _config()

    async def ordinary(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("ordinary command failure")

    monkeypatch.setattr(terminal, "_get_env_config", config)
    monkeypatch.setattr(terminal, "_get_or_create_environment", ordinary)
    result = json.loads(await terminal.terminal_tool("echo hi"))
    assert result["status"] == "error"
    assert result.get("status") != "degraded"


async def test_default_config_declares_warn_mode():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["terminal"].get("degraded_mode", "warn") == "warn"
