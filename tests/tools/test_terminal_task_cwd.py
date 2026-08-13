"""Task-scoped working directories remain isolated in the async terminal."""

import asyncio
import json

import pytest

import tools.terminal_tool as terminal


@pytest.mark.asyncio
async def test_public_session_cwd_keywords_match_upstream_contract(
    tmp_path, monkeypatch
):
    terminal.record_session_cwd(session_key="keyword-session", cwd=str(tmp_path))
    try:
        assert terminal.get_session_cwd(session_key="keyword-session") == str(
            tmp_path
        )
        assert terminal.is_persistent_env(task_id="keyword-session") is False
        await terminal._get_or_create_environment("keyword-session")
        assert terminal.is_persistent_env(task_id="keyword-session") is False
        await terminal.cleanup_vm(task_id="keyword-session", force_remove=True)
    finally:
        terminal.clear_session_cwd(session_key="keyword-session")


@pytest.mark.asyncio
async def test_registered_task_cwd_is_used(tmp_path):
    task_id = "cwd-task"
    terminal.register_task_env_overrides(task_id, {"cwd": str(tmp_path)})
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
    terminal.register_task_env_overrides(task_id, {"cwd": str(configured)})
    try:
        result = json.loads(
            await terminal.terminal_tool("pwd", task_id=task_id, workdir=str(explicit))
        )
        assert result["output"] == str(explicit)
    finally:
        await terminal.cleanup_vm(task_id)
        terminal.clear_session_cwd(task_id)


@pytest.mark.asyncio
async def test_relative_workdir_remains_relative_to_environment_cwd(tmp_path):
    configured = tmp_path / "configured"
    relative = configured / "relative"
    relative.mkdir(parents=True)
    task_id = "cwd-relative-override"
    terminal.register_task_env_overrides(task_id, {"cwd": str(configured)})
    try:
        result = json.loads(
            await terminal.terminal_tool(
                "pwd",
                task_id=task_id,
                workdir="relative",
                force=True,
            )
        )
        assert result["output"] == str(relative)
    finally:
        await terminal.cleanup_vm(task_id)
        terminal.clear_session_cwd(task_id)


@pytest.mark.asyncio
async def test_different_tasks_keep_independent_cwds(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    terminal.register_task_env_overrides("one", {"cwd": str(first)})
    terminal.register_task_env_overrides("two", {"cwd": str(second)})
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
