"""Task-scoped working directories remain isolated in the async terminal."""

import asyncio
import json

import pytest

import tools.terminal_tool as terminal


@pytest.mark.asyncio
async def test_registered_task_cwd_is_used(tmp_path):
    task_id = "cwd-task"
    await terminal.register_task_env_overrides(task_id, {"cwd": str(tmp_path)})
    try:
        result = json.loads(await terminal.terminal_tool("pwd", task_id=task_id))
        assert result["output"] == str(tmp_path)
    finally:
        await terminal.cleanup_vm(task_id)
        terminal.clear_session_cwd(task_id)


@pytest.mark.asyncio
async def test_explicit_workdir_wins_over_task_cwd(tmp_path):
    configured = tmp_path / "configured"
    explicit = tmp_path / "explicit"
    configured.mkdir()
    explicit.mkdir()
    task_id = "cwd-override"
    await terminal.register_task_env_overrides(task_id, {"cwd": str(configured)})
    try:
        result = json.loads(
            await terminal.terminal_tool("pwd", task_id=task_id, workdir=str(explicit))
        )
        assert result["output"] == str(explicit)
    finally:
        await terminal.cleanup_vm(task_id)
        terminal.clear_session_cwd(task_id)


@pytest.mark.asyncio
async def test_different_tasks_keep_independent_cwds(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    await terminal.register_task_env_overrides("one", {"cwd": str(first)})
    await terminal.register_task_env_overrides("two", {"cwd": str(second)})
    try:
        one, two = await asyncio.gather(
            terminal.terminal_tool("pwd", task_id="one"),
            terminal.terminal_tool("pwd", task_id="two"),
        )
        assert json.loads(one)["output"] == str(first)
        assert json.loads(two)["output"] == str(second)
    finally:
        await terminal.cleanup_vm("one")
        await terminal.cleanup_vm("two")
