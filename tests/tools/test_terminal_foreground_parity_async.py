"""Async ports of the upstream foreground terminal pipeline contracts."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest

from tools import terminal_tool as terminal


async def _install_foreground_fakes(
    monkeypatch: pytest.MonkeyPatch,
    environment: object,
    *,
    guard: dict | None = None,
    hook_results: list[object] | None = None,
) -> AsyncMock:
    monkeypatch.setattr(
        terminal,
        "_get_env_config",
        AsyncMock(return_value={"env_type": "local", "timeout": 7}),
    )
    monkeypatch.setattr(
        terminal,
        "_get_or_create_environment",
        AsyncMock(return_value=environment),
    )
    monkeypatch.setattr(
        terminal,
        "check_all_command_guards",
        AsyncMock(return_value=guard or {"approved": True}),
    )
    monkeypatch.setattr(
        "tools.tool_output_limits._refresh_tool_output_limits",
        AsyncMock(),
    )
    monkeypatch.setattr(
        "agent.verification_evidence.record_terminal_result",
        AsyncMock(return_value=None),
    )
    hook = AsyncMock(return_value=hook_results or [])
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", hook)
    return hook


@pytest.mark.asyncio
async def test_foreground_retries_three_times_with_async_backoff(monkeypatch):
    sleeps: list[int] = []

    async def sleep(delay: int) -> None:
        sleeps.append(delay)

    class Environment:
        cwd = "/tmp"
        calls = 0

        async def execute(self, *_args, **_kwargs):
            self.calls += 1
            raise ConnectionError("transport reset")

    environment = Environment()
    await _install_foreground_fakes(monkeypatch, environment)
    monkeypatch.setattr(terminal.asyncio, "sleep", sleep)

    result = json.loads(
        await terminal.terminal_tool("printf retry", force=True, task_id="retry")
    )

    assert environment.calls == 4
    assert sleeps == [2, 4, 8]
    assert result == {
        "output": "",
        "exit_code": -1,
        "error": "Command execution failed: ConnectionError: transport reset",
    }


@pytest.mark.asyncio
async def test_foreground_retry_recovers_on_next_attempt(monkeypatch):
    sleeps: list[int] = []

    async def sleep(delay: int) -> None:
        sleeps.append(delay)

    class Environment:
        cwd = "/tmp"
        calls = 0

        async def execute(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("transport reset")
            return {"output": "recovered", "returncode": 0}

    environment = Environment()
    await _install_foreground_fakes(monkeypatch, environment)
    monkeypatch.setattr(terminal.asyncio, "sleep", sleep)

    result = json.loads(
        await terminal.terminal_tool("printf recovered", force=True)
    )

    assert environment.calls == 2
    assert sleeps == [2]
    assert result["output"] == "recovered"


@pytest.mark.asyncio
async def test_foreground_timeout_is_not_retried(monkeypatch):
    class Environment:
        cwd = "/tmp"
        calls = 0

        async def execute(self, *_args, **_kwargs):
            self.calls += 1
            raise TimeoutError("backend timeout")

    environment = Environment()
    await _install_foreground_fakes(monkeypatch, environment)

    result = json.loads(await terminal.terminal_tool("sleep 9", force=True))

    assert environment.calls == 1
    assert result == {
        "output": "",
        "exit_code": 124,
        "error": "Command timed out after 7 seconds",
    }


@pytest.mark.asyncio
async def test_foreground_cancellation_is_never_retried(monkeypatch):
    class Environment:
        cwd = "/tmp"
        calls = 0

        async def execute(self, *_args, **_kwargs):
            self.calls += 1
            raise asyncio.CancelledError

    environment = Environment()
    await _install_foreground_fakes(monkeypatch, environment)

    with pytest.raises(asyncio.CancelledError):
        await terminal.terminal_tool("sleep 9", force=True)
    assert environment.calls == 1


@pytest.mark.asyncio
async def test_transform_hook_precedes_sanitization_and_first_string_wins(
    monkeypatch,
):
    class Environment:
        cwd = "/tmp"

        async def execute(self, *_args, **_kwargs):
            return {"output": "\x1b[31mraw\x1b[0m", "returncode": 0}

    hook = await _install_foreground_fakes(
        monkeypatch,
        Environment(),
        hook_results=[None, "\x1b[32mhooked\x1b[0m", "ignored"],
    )

    result = json.loads(await terminal.terminal_tool("printf raw", force=True))

    assert result["output"] == "hooked"
    assert hook.await_args.kwargs["output"] == "\x1b[31mraw\x1b[0m"


@pytest.mark.asyncio
async def test_failure_hint_and_sudo_failure_flags_match_upstream(monkeypatch):
    class Environment:
        cwd = "/tmp"

        async def execute(self, *_args, **_kwargs):
            return {
                "output": "bash: python: command not found\n"
                "sudo: 3 incorrect password attempts",
                "returncode": 127,
            }

    await _install_foreground_fakes(monkeypatch, Environment())

    result = json.loads(await terminal.terminal_tool("python x.py", force=True))

    assert "no bare `python`" in result["hint"]
    assert result["sudo_auth_failed"] is True
    assert "sudo_cache_cleared" not in result


@pytest.mark.asyncio
async def test_user_approval_note_and_interrupt_slate_are_preserved(monkeypatch):
    class Environment:
        cwd = "/tmp"

        async def execute(self, *_args, **_kwargs):
            return {
                "output": "[Command interrupted]",
                "returncode": 130,
            }

    await _install_foreground_fakes(
        monkeypatch,
        Environment(),
        guard={
            "approved": True,
            "user_approved": True,
            "description": "recursive delete",
        },
    )
    clear_interrupt = Mock()
    monkeypatch.setattr(
        "tools.interrupt.clear_current_thread_interrupt",
        clear_interrupt,
    )

    result = json.loads(await terminal.terminal_tool("rm -r target"))

    clear_interrupt.assert_called_once_with()
    assert result["approval"] == (
        "Command required approval (recursive delete) and was approved by the "
        "user, then interrupted."
    )


@pytest.mark.asyncio
async def test_docker_bind_mount_reaches_command_guard(monkeypatch):
    guard = AsyncMock(
        return_value={"approved": False, "message": "blocked for test"}
    )
    get_environment = AsyncMock()
    monkeypatch.setattr(
        terminal,
        "_get_env_config",
        AsyncMock(
            return_value={
                "env_type": "docker",
                "timeout": 7,
                "docker_volumes": ["/host/project:/workspace"],
            }
        ),
    )
    monkeypatch.setattr(terminal, "check_all_command_guards", guard)
    monkeypatch.setattr(terminal, "_get_or_create_environment", get_environment)
    monkeypatch.setattr(
        "tools.tool_output_limits._refresh_tool_output_limits",
        AsyncMock(),
    )

    result = json.loads(await terminal.terminal_tool("rm -rf build"))

    assert result["status"] == "blocked"
    assert guard.await_args.kwargs["has_host_access"] is True
    get_environment.assert_awaited_once_with(None)


@pytest.mark.asyncio
async def test_workdir_allowlist_blocks_shell_metacharacters_before_execution(
    monkeypatch,
):
    get_environment = AsyncMock()
    monkeypatch.setattr(
        terminal,
        "_get_env_config",
        AsyncMock(return_value={"env_type": "local", "timeout": 7}),
    )
    monkeypatch.setattr(terminal, "_get_or_create_environment", get_environment)
    monkeypatch.setattr(
        "tools.tool_output_limits._refresh_tool_output_limits",
        AsyncMock(),
    )

    result = json.loads(
        await terminal.terminal_tool(
            "printf safe",
            workdir="/tmp/project;touch-pwned",
            force=True,
        )
    )

    assert result["status"] == "blocked"
    assert "disallowed character ';'" in result["error"]
    get_environment.assert_awaited_once_with(None)


def test_workdir_allowlist_keeps_unicode_paths_and_log_preview_redacts_secrets():
    assert terminal._validate_workdir("/tmp/프로젝트 자료,+@=") is None
    preview = terminal._safe_command_preview(
        "curl 'https://example.test/?access_token=opaque-secret-value' "
        "-H 'Authorization: Bearer sk-test_abcdefghijklmnopqrstuvwxyz'"
    )
    assert "opaque-secret-value" not in preview
    assert "sk-test_abcdefghijklmnopqrstuvwxyz" not in preview


@pytest.mark.asyncio
async def test_hook_cancellation_discards_raw_spill_through_repeated_cancel(
    monkeypatch,
    tmp_path,
):
    spill = tmp_path / "out-hook.log"
    temporary = tmp_path / ".out-hook.log.pending.tmp"
    spill.write_text("OPENAI_API_KEY=raw-secret", encoding="utf-8")
    temporary.write_text("raw-temporary-secret", encoding="utf-8")
    hook_started = asyncio.Event()
    removal_started = asyncio.Event()
    allow_removal = asyncio.Event()

    class Environment:
        cwd = str(tmp_path)

        async def execute(self, *_args, **_kwargs):
            return {
                "output": "visible",
                "returncode": 0,
                "output_total_chars": 25,
                "full_output_path": str(spill),
            }

    await _install_foreground_fakes(monkeypatch, Environment())

    async def cancelling_hook(*_args, **_kwargs):
        hook_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", cancelling_hook)
    original_remove = terminal.aiofiles.os.remove

    async def delayed_remove(path):
        if str(path) == str(spill):
            removal_started.set()
            await allow_removal.wait()
        return await original_remove(path)

    monkeypatch.setattr(terminal.aiofiles.os, "remove", delayed_remove)
    task = asyncio.create_task(
        terminal.terminal_tool("printf visible", force=True)
    )
    await hook_started.wait()
    task.cancel()
    await removal_started.wait()
    task.cancel()
    allow_removal.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert not spill.exists()
    assert not temporary.exists()


@pytest.mark.asyncio
async def test_redaction_write_cancellation_discards_raw_spill_and_temporary(
    monkeypatch,
    tmp_path,
):
    spill = tmp_path / "out-redaction.log"
    temporary = tmp_path / ".out-redaction.log.pending.tmp"
    spill.write_text("OPENAI_API_KEY=raw-secret", encoding="utf-8")
    temporary.write_text("raw-temporary-secret", encoding="utf-8")
    write_started = asyncio.Event()
    removal_started = asyncio.Event()
    allow_removal = asyncio.Event()

    class Environment:
        cwd = str(tmp_path)

        async def execute(self, *_args, **_kwargs):
            return {
                "output": "visible",
                "returncode": 0,
                "output_total_chars": 25,
                "full_output_path": str(spill),
            }

    await _install_foreground_fakes(monkeypatch, Environment())
    original_open = terminal.aiofiles.open

    class BlockingHandle:
        def __init__(self, handle):
            self._handle = handle

        def __getattr__(self, name):
            return getattr(self._handle, name)

        async def write(self, value):
            write_started.set()
            await asyncio.Event().wait()
            return await self._handle.write(value)

    class BlockingOpen:
        def __init__(self, manager):
            self._manager = manager

        async def __aenter__(self):
            handle = await self._manager.__aenter__()
            return BlockingHandle(handle)

        async def __aexit__(self, exc_type, exc, traceback):
            return await self._manager.__aexit__(exc_type, exc, traceback)

    def open_with_blocked_write(file, mode="r", *args, **kwargs):
        manager = original_open(file, mode, *args, **kwargs)
        if str(file) == str(spill) and mode in {"w", "x", "a"}:
            return BlockingOpen(manager)
        return manager

    monkeypatch.setattr(terminal.aiofiles, "open", open_with_blocked_write)
    original_remove = terminal.aiofiles.os.remove

    async def delayed_remove(path):
        if str(path) == str(spill):
            removal_started.set()
            await allow_removal.wait()
        return await original_remove(path)

    monkeypatch.setattr(terminal.aiofiles.os, "remove", delayed_remove)
    task = asyncio.create_task(
        terminal.terminal_tool("printf visible", force=True)
    )
    await write_started.wait()
    task.cancel()
    await removal_started.wait()
    task.cancel()
    allow_removal.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert not spill.exists()
    assert not temporary.exists()
