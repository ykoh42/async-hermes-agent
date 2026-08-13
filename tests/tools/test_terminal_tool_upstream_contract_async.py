"""Behavior-level async ports of retained upstream terminal JSON contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tools import terminal_tool as terminal


def _config(**overrides):
    return {"env_type": "local", "timeout": 7, **overrides}


def _environment(*, cwd: str = "/tmp"):
    return SimpleNamespace(cwd=cwd, env={}, _snapshot_ready=False)


async def _install_foreground_result(
    monkeypatch: pytest.MonkeyPatch,
    environment: object,
    result: dict,
) -> AsyncMock:
    execute = AsyncMock(return_value=result)
    environment.execute = execute
    monkeypatch.setattr(
        terminal,
        "_get_env_config",
        AsyncMock(return_value=_config()),
    )
    monkeypatch.setattr(
        terminal,
        "_get_or_create_environment",
        AsyncMock(return_value=environment),
    )
    monkeypatch.setattr(
        "tools.tool_output_limits._refresh_tool_output_limits",
        AsyncMock(),
    )
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        "agent.verification_evidence.record_terminal_result",
        AsyncMock(return_value=None),
    )
    return execute


@pytest.mark.asyncio
async def test_non_string_rejection_precedes_async_config_io(monkeypatch):
    get_config = AsyncMock(side_effect=AssertionError("config must not be read"))
    refresh_limits = AsyncMock(
        side_effect=AssertionError("output limits must not be read")
    )
    monkeypatch.setattr(terminal, "_get_env_config", get_config)
    monkeypatch.setattr(
        "tools.tool_output_limits._refresh_tool_output_limits",
        refresh_limits,
    )

    result = json.loads(await terminal.terminal_tool(None))

    assert result == {
        "output": "",
        "exit_code": -1,
        "error": "Invalid command: expected string, got NoneType",
        "status": "error",
    }
    get_config.assert_not_awaited()
    refresh_limits.assert_not_awaited()


@pytest.mark.asyncio
async def test_timeout_preflight_reads_config_and_cap_precedes_guidance(monkeypatch):
    events: list[str] = []

    async def get_config():
        events.append("config")
        return _config()

    get_environment = AsyncMock()
    guidance = Mock(side_effect=AssertionError("timeout cap must win"))
    monkeypatch.setattr(terminal, "_get_env_config", get_config)
    monkeypatch.setattr(terminal, "_get_or_create_environment", get_environment)
    monkeypatch.setattr(terminal, "_foreground_background_guidance", guidance)

    invalid = json.loads(await terminal.terminal_tool("printf x", timeout=0))
    capped = json.loads(
        await terminal.terminal_tool(
            "uvicorn app:app",
            timeout=terminal.FOREGROUND_MAX_TIMEOUT + 1,
        )
    )

    assert invalid == {
        "error": "timeout must be a positive number of seconds (got 0)."
    }
    assert capped == {
        "error": (
            f"Foreground timeout {terminal.FOREGROUND_MAX_TIMEOUT + 1}s "
            f"exceeds the maximum of {terminal.FOREGROUND_MAX_TIMEOUT}s. "
            "Use background=true with notify_on_complete=true for long-running "
            "commands."
        )
    }
    assert events == ["config", "config"]
    guidance.assert_not_called()
    get_environment.assert_not_awaited()


@pytest.mark.asyncio
async def test_environment_creation_precedes_guard_and_denial_is_blocked(monkeypatch):
    events: list[str] = []

    async def get_config():
        events.append("config")
        return _config(env_type="docker")

    async def get_environment(_task_id):
        events.append("environment")
        return _environment()

    async def guard(*_args, **_kwargs):
        events.append("guard")
        return {"approved": False, "description": "recursive delete"}

    monkeypatch.setattr(terminal, "_get_env_config", get_config)
    monkeypatch.setattr(terminal, "_get_or_create_environment", get_environment)
    monkeypatch.setattr(terminal, "check_all_command_guards", guard)

    result = json.loads(await terminal.terminal_tool("rm -rf build"))

    assert events == ["config", "environment", "guard"]
    assert result == {
        "output": "",
        "exit_code": -1,
        "error": (
            "Command denied: recursive delete. Use the approval prompt to allow "
            "it, or rephrase the command."
        ),
        "status": "blocked",
    }


@pytest.mark.asyncio
async def test_workdir_guard_runs_after_environment_and_command_guard(monkeypatch):
    events: list[str] = []

    async def get_environment(_task_id):
        events.append("environment")
        return _environment()

    async def guard(*_args, **_kwargs):
        events.append("guard")
        return {"approved": True}

    monkeypatch.setattr(
        terminal,
        "_get_env_config",
        AsyncMock(return_value=_config()),
    )
    monkeypatch.setattr(terminal, "_get_or_create_environment", get_environment)
    monkeypatch.setattr(terminal, "check_all_command_guards", guard)

    result = json.loads(
        await terminal.terminal_tool(
            "printf safe",
            workdir="/tmp/project;touch-pwned",
        )
    )

    assert events == ["environment", "guard"]
    assert result["status"] == "blocked"
    assert "disallowed character ';'" in result["error"]


@pytest.mark.asyncio
async def test_relative_workdir_is_forwarded_like_upstream(monkeypatch):
    environment = _environment(cwd="/tmp/base")
    execute = await _install_foreground_result(
        monkeypatch,
        environment,
        {"output": "relative-ok", "returncode": 0},
    )

    result = json.loads(
        await terminal.terminal_tool(
            "printf relative-ok",
            workdir="relative-project",
            force=True,
        )
    )

    assert result["output"] == "relative-ok"
    assert execute.await_args.kwargs["cwd"] == "relative-project"


@pytest.mark.asyncio
async def test_cwd_echo_is_retained_for_nonzero_transient_workdir(monkeypatch):
    class Environment:
        cwd = "/tmp/base"
        env = {}
        _snapshot_ready = False

        async def execute(self, *_args, **_kwargs):
            self.cwd = "/tmp/end"
            return {"output": "failed after cd", "returncode": 1}

    environment = Environment()
    await _install_foreground_result(
        monkeypatch,
        environment,
        {"output": "unused", "returncode": 0},
    )
    environment.execute = Environment.execute.__get__(environment)

    result = json.loads(
        await terminal.terminal_tool(
            "cd /tmp/end && false",
            workdir="/tmp/start",
            force=True,
        )
    )

    assert result["exit_code"] == 1
    assert result["cwd"] == "/tmp/end"
    assert environment.cwd == "/tmp/base"


@pytest.mark.asyncio
async def test_environment_import_error_uses_upstream_disabled_shape(monkeypatch):
    monkeypatch.setattr(
        terminal,
        "_get_env_config",
        AsyncMock(return_value=_config()),
    )
    monkeypatch.setattr(
        terminal,
        "_get_or_create_environment",
        AsyncMock(side_effect=ImportError("missing backend")),
    )

    result = json.loads(await terminal.terminal_tool("printf x", force=True))

    assert result == {
        "output": "",
        "exit_code": -1,
        "error": (
            "Terminal tool disabled: environment creation failed (missing backend)"
        ),
        "status": "disabled",
    }


@pytest.mark.asyncio
async def test_background_failure_does_not_fall_through_outer_error_shape(monkeypatch):
    monkeypatch.setattr(
        terminal,
        "_get_env_config",
        AsyncMock(return_value=_config()),
    )
    monkeypatch.setattr(
        terminal,
        "_get_or_create_environment",
        AsyncMock(return_value=_environment()),
    )
    monkeypatch.setattr(
        "tools.process_registry.process_registry.spawn_local",
        AsyncMock(side_effect=RuntimeError("spawn failed")),
    )

    result = json.loads(
        await terminal.terminal_tool("printf x", background=True, force=True)
    )

    assert result == {
        "output": "",
        "exit_code": -1,
        "error": "Failed to start background process: spawn failed",
    }


@pytest.mark.asyncio
async def test_background_model_facing_notes_match_upstream(monkeypatch):
    process_session = SimpleNamespace(
        id="process-1",
        pid=123,
        session_key="session-1",
        watch_patterns=[],
        notify_on_complete=False,
        exited=False,
    )
    monkeypatch.setattr(
        terminal,
        "_get_env_config",
        AsyncMock(return_value=_config()),
    )
    monkeypatch.setattr(
        terminal,
        "_get_or_create_environment",
        AsyncMock(return_value=_environment()),
    )
    monkeypatch.setattr(
        "tools.process_registry.process_registry.spawn_local",
        AsyncMock(return_value=process_session),
    )
    monkeypatch.setattr(
        "tools.process_registry.process_registry._write_checkpoint",
        AsyncMock(),
    )
    from tools.process_registry import process_registry

    prior_watcher_count = len(process_registry.pending_watchers)
    try:
        silent = json.loads(
            await terminal.terminal_tool(
                "printf x",
                background=True,
                force=True,
            )
        )
        result = json.loads(
            await terminal.terminal_tool(
                "gh auth login --with-token",
                background=True,
                pty=True,
                notify_on_complete=True,
                watch_patterns=["ready"],
                force=True,
            )
        )
    finally:
        del process_registry.pending_watchers[prior_watcher_count:]

    assert silent["hint"] == (
        "background=true without notify_on_complete=true means this process runs "
        "SILENTLY — you will not be told when it exits. If this is a bounded task "
        "(test suite, build, CI poller, deploy, anything with a defined end), you "
        "almost certainly wanted notify_on_complete=true so the system pings you "
        "on exit. Re-launch with notify_on_complete=true, or call "
        "process(action='poll') / process(action='wait') yourself to learn the "
        "outcome. Only ignore this hint for genuine long-lived processes that "
        "never exit (servers, watchers, daemons)."
    )
    assert "hint" not in result
    assert result["pty_note"] == (
        "PTY disabled for this command because it expects piped stdin/EOF "
        "(for example gh auth login --with-token). For local background "
        "processes, call process(action='close') after writing so it receives EOF."
    )
    assert result["watch_patterns_ignored"] == (
        "watch_patterns ignored because notify_on_complete=True; these two flags "
        "produce duplicate notifications when combined"
    )
    assert result["notify_on_complete"] is True


@pytest.mark.asyncio
async def test_outer_exception_retains_upstream_traceback_field(monkeypatch):
    monkeypatch.setattr(
        terminal,
        "_get_env_config",
        AsyncMock(side_effect=RuntimeError("config exploded")),
    )

    result = json.loads(await terminal.terminal_tool("printf x"))

    assert result["output"] == ""
    assert result["exit_code"] == -1
    assert result["error"] == "Failed to execute command: config exploded"
    assert result["status"] == "error"
    assert "Traceback (most recent call last):" in result["traceback"]
    assert "RuntimeError: config exploded" in result["traceback"]
