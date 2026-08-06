"""Native-async local terminal behavior contracts."""

from __future__ import annotations

import json
import os
import shlex
import sys

import pytest

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
