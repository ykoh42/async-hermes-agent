"""Cancellation of terminal execution reaps the shell process group."""

import asyncio

import pytest

import tools.terminal_tool as terminal


@pytest.mark.asyncio
async def test_cancelled_foreground_command_does_not_leak_process(tmp_path):
    task = asyncio.create_task(
        terminal.terminal_tool("sleep 30", workdir=str(tmp_path), task_id="cancel")
    )
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await terminal.cleanup_vm("cancel")
    assert terminal.get_active_env("cancel") is None
