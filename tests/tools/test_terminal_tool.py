"""Native-async local terminal behavior contracts."""
from __future__ import annotations

import json

import pytest

import tools.terminal_tool as terminal


@pytest.mark.asyncio
async def test_executes_local_command_and_reports_exit_code(tmp_path):
    result = json.loads(
        await terminal.terminal_tool("printf hello", workdir=str(tmp_path), task_id="exec")
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
            await terminal.terminal_tool("printf forced", force=True, workdir=str(tmp_path))
        )
    finally:
        terminal.set_approval_callback(None)
    assert result["output"] == "forced"


def test_schema_preserves_stable_local_arguments():
    properties = terminal.TERMINAL_SCHEMA["parameters"]["properties"]
    assert {"command", "background", "timeout", "workdir", "pty"} <= properties.keys()
