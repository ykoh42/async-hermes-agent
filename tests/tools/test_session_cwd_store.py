"""Per-session working directories stay isolated in the native async terminal."""

import json

import pytest

import tools.file_tools as file_tools
import tools.terminal_tool as terminal


@pytest.fixture(autouse=True)
def clean_terminal_state(monkeypatch):
    monkeypatch.setattr(terminal, "_session_cwds", {})
    monkeypatch.setattr(terminal, "_task_env_overrides", {})
    monkeypatch.setattr(terminal, "_active_environments", {})
    monkeypatch.setattr(terminal, "_last_activity", {})


@pytest.mark.asyncio
async def test_records_are_keyed_and_cleared_by_session(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    await terminal.record_session_cwd("one", str(first))
    await terminal.record_session_cwd("two", str(second))

    assert await terminal.get_session_cwd("one") == str(first)
    assert await terminal.get_session_cwd("two") == str(second)
    terminal.clear_session_cwd("one")
    assert await terminal.get_session_cwd("one") != str(first)
    assert await terminal.get_session_cwd("two") == str(second)


@pytest.mark.asyncio
async def test_registered_override_seeds_and_updates_record(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    await terminal.register_task_env_overrides("session", {"cwd": str(first)})
    assert await terminal.get_session_cwd("session") == str(first)
    await terminal.register_task_env_overrides("session", {"cwd": str(second)})
    assert await terminal.get_session_cwd("session") == str(second)


@pytest.mark.asyncio
async def test_file_resolution_uses_each_sessions_record(tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    await terminal.record_session_cwd("one", str(first))
    await terminal.record_session_cwd("two", str(second))

    assert await file_tools._resolve_path_for_task("f.py", "one") == first / "f.py"
    assert await file_tools._resolve_path_for_task("f.py", "two") == second / "f.py"


@pytest.mark.asyncio
async def test_cd_updates_only_its_session_record(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    destination = tmp_path / "destination"
    first.mkdir()
    second.mkdir()
    destination.mkdir()
    await terminal.register_task_env_overrides("one", {"cwd": str(first)})
    await terminal.register_task_env_overrides("two", {"cwd": str(second)})

    result = json.loads(
        await terminal.terminal_tool(f"cd {destination}", task_id="one")
    )

    assert result["exit_code"] == 0
    assert await terminal.get_session_cwd("one") == str(destination)
    assert await terminal.get_session_cwd("two") == str(second)


@pytest.mark.asyncio
async def test_next_command_runs_from_updated_cwd(tmp_path):
    start = tmp_path / "start"
    destination = tmp_path / "destination"
    start.mkdir()
    destination.mkdir()
    await terminal.register_task_env_overrides("session", {"cwd": str(start)})

    await terminal.terminal_tool(f"cd {destination}", task_id="session")
    result = json.loads(await terminal.terminal_tool("pwd", task_id="session"))

    assert result["output"] == str(destination)


@pytest.mark.asyncio
async def test_child_record_can_diverge_without_mutating_parent(tmp_path):
    parent = tmp_path / "parent"
    child = tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    await terminal.record_session_cwd("parent", str(parent))
    await terminal.record_session_cwd(
        "child", await terminal.get_session_cwd("parent")
    )
    await terminal.record_session_cwd("child", str(child))

    assert await terminal.get_session_cwd("parent") == str(parent)
    assert await terminal.get_session_cwd("child") == str(child)
