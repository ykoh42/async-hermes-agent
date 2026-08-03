"""Native async terminal recovery when a session deletes its own cwd."""

import json
import shutil

import pytest

from tools.terminal_tool import LocalEnvironment, terminal_tool


@pytest.mark.asyncio
async def test_environment_recovers_to_closest_existing_parent(tmp_path, caplog):
    missing = tmp_path / "workspace" / "nested"
    missing.mkdir(parents=True)
    environment = LocalEnvironment(str(missing), timeout=5)
    shutil.rmtree(missing)

    result = await environment.execute("pwd")

    assert result["returncode"] == 0
    assert result["output"] == str(missing.parent)
    assert environment.cwd == str(missing.parent)
    assert "missing on disk" in caplog.text


@pytest.mark.asyncio
async def test_environment_keeps_existing_cwd(tmp_path, caplog):
    environment = LocalEnvironment(str(tmp_path), timeout=5)

    result = await environment.execute("pwd")

    assert result["returncode"] == 0
    assert result["output"] == str(tmp_path)
    assert environment.cwd == str(tmp_path)
    assert "missing on disk" not in caplog.text


@pytest.mark.asyncio
async def test_terminal_session_recovers_and_persists_new_anchor(tmp_path):
    missing = tmp_path / "workspace" / "nested"
    missing.mkdir(parents=True)
    task_id = "recovery"
    import tools.terminal_tool as module

    module.register_task_env_overrides(task_id, {"cwd": str(missing)})
    await terminal_tool("pwd", task_id=task_id)
    shutil.rmtree(missing)

    result = json.loads(await terminal_tool("pwd", task_id=task_id))

    assert result["exit_code"] == 0
    assert result["output"] == str(missing.parent)
    assert module.get_session_cwd(task_id) == str(missing.parent)
    await module.cleanup_vm(task_id)
    module.clear_task_env_overrides(task_id)


@pytest.mark.asyncio
async def test_missing_explicit_workdir_is_not_silently_recovered(tmp_path):
    missing = tmp_path / "missing"
    result = json.loads(await terminal_tool("pwd", workdir=str(missing)))

    assert result["status"] == "error"
    assert "does not exist" in result["error"]
