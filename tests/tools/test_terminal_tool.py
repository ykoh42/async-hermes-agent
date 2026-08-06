"""Native-async local terminal behavior contracts."""

from __future__ import annotations

import json
import os
import shlex
import sys

import pytest
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

import tools.terminal_tool as terminal


@pytest.mark.asyncio
async def test_executes_local_command_and_reports_exit_code(tmp_path):
    result = json.loads(
        await terminal.terminal_tool(
            "printf hello", workdir=str(tmp_path), task_id="exec"
        )
    )
    assert result["exit_code"] == 0
    assert result["output"] == "hello"


@pytest.mark.asyncio
async def test_exported_environment_persists_between_calls(tmp_path):
    task_id = "persistent-environment"
    try:
        async with (
            no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
            no_task_leaks(action=LeakAction.RAISE),
        ):
            first = json.loads(
                await terminal.terminal_tool(
                    "export HERMES_STICKY_ENV_PROBE=sticky; "
                    'printf "first=%s" "$HERMES_STICKY_ENV_PROBE"',
                    workdir=str(tmp_path),
                    task_id=task_id,
                )
            )
            second = json.loads(
                await terminal.terminal_tool(
                    'printf "second=%s" "$HERMES_STICKY_ENV_PROBE"',
                    task_id=task_id,
                )
            )
    finally:
        await terminal.cleanup_vm(task_id)

    assert first["output"] == "first=sticky"
    assert second["output"] == "second=sticky"


@pytest.mark.asyncio
async def test_background_process_inherits_session_environment(tmp_path):
    from tools.process_registry import process_registry

    task_id = "background-environment"
    await terminal.terminal_tool(
        "export HERMES_BACKGROUND_ENV_PROBE=from-session",
        workdir=str(tmp_path),
        task_id=task_id,
    )
    started = json.loads(
        await terminal.terminal_tool(
            'printf "%s" "$HERMES_BACKGROUND_ENV_PROBE"',
            background=True,
            task_id=task_id,
        )
    )
    try:
        result = await process_registry.wait(started["session_id"], timeout=2)
    finally:
        await terminal.cleanup_vm(task_id)

    assert result["status"] == "exited"
    assert result["output"] == "from-session"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "rg --line-number 'sudo' .",
        "printf '%s\\n' sudo",
        "grep -n sudo README.md",
    ],
)
async def test_sudo_mentions_are_not_rewritten(command, monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)

    transformed, sudo_stdin = await terminal._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


@pytest.mark.asyncio
async def test_configured_sudo_password_rewrites_real_invocations(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")

    transformed, sudo_stdin = await terminal._transform_sudo_command(
        "sudo first && VALUE=1 sudo second"
    )

    assert transformed == "sudo -S -p '' first && VALUE=1 sudo -S -p '' second"
    assert sudo_stdin == "testpass\ntestpass\n"


@pytest.mark.asyncio
async def test_profile_scoped_sudo_password_wins_over_process_env(monkeypatch):
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    monkeypatch.setenv("SUDO_PASSWORD", "wrong-global-pass")
    token = set_secret_scope({"SUDO_PASSWORD": "scoped-pass"})
    try:
        _transformed, sudo_stdin = await terminal._transform_sudo_command("sudo true")
    finally:
        reset_secret_scope(token)

    assert sudo_stdin == "scoped-pass\n"


@pytest.mark.asyncio
async def test_explicit_empty_sudo_password_is_still_configured(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "")

    transformed, sudo_stdin = await terminal._transform_sudo_command("sudo true")

    assert transformed == "sudo -S -p '' true"
    assert sudo_stdin == "\n"


@pytest.mark.asyncio
async def test_async_sudo_callback_supplies_password(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)

    async def password_callback():
        return "callback-pass"

    terminal.set_sudo_password_callback(password_callback)
    try:
        transformed, sudo_stdin = await terminal._transform_sudo_command("sudo true")
    finally:
        terminal.set_sudo_password_callback(None)

    assert transformed == "sudo -S -p '' true"
    assert sudo_stdin == "callback-pass\n"


@pytest.mark.asyncio
async def test_sync_sudo_callback_fails_fast(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    terminal.set_sudo_password_callback(lambda: "legacy-pass")
    try:
        with pytest.raises(RuntimeError, match="coroutine sudo password callback"):
            await terminal._transform_sudo_command("sudo true")
    finally:
        terminal.set_sudo_password_callback(None)


@pytest.mark.asyncio
async def test_local_environment_pipes_sudo_password_before_stdin(tmp_path, monkeypatch):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sudo = fake_bin / "sudo"
    fake_sudo.write_text(
        "#!/bin/sh\n"
        "IFS= read -r password\n"
        "IFS= read -r payload\n"
        "printf 'password=%s\\npayload=%s\\n' \"$password\" \"$payload\"\n"
    )
    fake_sudo.chmod(0o755)
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")

    environment = terminal.LocalEnvironment(str(tmp_path))
    environment.env["PATH"] = f"{fake_bin}{os.pathsep}{environment.env['PATH']}"
    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        result = await environment.execute("sudo target", stdin_data="payload-data\n")

    assert result["returncode"] == 0
    assert result["output"].splitlines() == [
        "password=testpass",
        "payload=payload-data",
    ]
    assert "SUDO_PASSWORD" not in environment.env


@pytest.mark.asyncio
@pytest.mark.parametrize("command", [None, "", "   "])
async def test_rejects_invalid_commands(command):
    result = json.loads(await terminal.terminal_tool(command))
    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_async_approval_callback_can_allow_and_deny(tmp_path):
    decisions: list[str] = []

    async def callback(**kwargs):
        decisions.append(kwargs["command"])
        return kwargs["command"] == "printf allowed"

    terminal.set_approval_callback(callback)
    try:
        allowed = json.loads(
            await terminal.terminal_tool("printf allowed", workdir=str(tmp_path))
        )
        denied = json.loads(
            await terminal.terminal_tool("printf denied", workdir=str(tmp_path))
        )
    finally:
        terminal.set_approval_callback(None)

    assert allowed["exit_code"] == 0
    assert denied["status"] == "denied"
    assert decisions == ["printf allowed", "printf denied"]


@pytest.mark.asyncio
async def test_force_bypasses_registered_approval_callback(tmp_path):
    async def deny(**_kwargs):
        return False

    terminal.set_approval_callback(deny)
    try:
        result = json.loads(
            await terminal.terminal_tool(
                "printf forced", force=True, workdir=str(tmp_path)
            )
        )
    finally:
        terminal.set_approval_callback(None)
    assert result["output"] == "forced"


def test_schema_preserves_stable_local_arguments():
    properties = terminal.TERMINAL_SCHEMA["parameters"]["properties"]
    assert {
        "command",
        "background",
        "timeout",
        "workdir",
        "pty",
        "notify_on_complete",
        "watch_patterns",
    } <= properties.keys()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="stdlib PTY transport is POSIX-only")
async def test_public_terminal_pty_session_is_managed_by_process_tool(tmp_path):
    from tools.process_registry import process_registry

    source = shlex.quote("print(input())")
    command = f"{shlex.quote(sys.executable)} -u -c {source}"
    started = json.loads(
        await terminal.terminal_tool(
            command,
            background=True,
            pty=True,
            notify_on_complete=True,
            workdir=str(tmp_path),
            task_id="pty-integration",
        )
    )
    try:
        submitted = await process_registry.submit_stdin(
            started["session_id"], "from-terminal"
        )
        finished = await process_registry.wait(started["session_id"], timeout=2)
    finally:
        await process_registry.kill_all("pty-integration")

    assert started["output"] == "Background process started"
    assert started["notify_on_complete"] is False
    assert "process(action='poll')" in started["notify_unsupported"]
    assert submitted["status"] == "ok"
    assert finished["status"] == "exited"
    assert "from-terminal" in finished["output"]
