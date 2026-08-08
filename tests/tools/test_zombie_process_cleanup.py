"""Native background subprocesses are reaped at the agent boundary."""

import json

import pytest

import tools.terminal_tool as terminal
from tools.process_registry import process_registry


@pytest.mark.asyncio
async def test_cleanup_vm_terminates_background_process(tmp_path):
    task_id = "background-cleanup"
    result = json.loads(
        await terminal.terminal_tool(
            "sleep 30", background=True, workdir=str(tmp_path), task_id=task_id
        )
    )
    pid = result["pid"]
    session = await process_registry.get(result["session_id"])
    assert session is not None
    assert session.pid == pid
    assert session.process is not None

    await terminal.cleanup_vm(task_id)

    assert result["session_id"] not in process_registry.snapshot_running_ids(task_id)
    assert session.process.returncode is not None


@pytest.mark.asyncio
async def test_background_process_reaper_removes_natural_exit(tmp_path):
    task_id = "background-natural"
    result = json.loads(
        await terminal.terminal_tool(
            "sleep 0.05", background=True, workdir=str(tmp_path), task_id=task_id
        )
    )
    finished = await process_registry.wait(result["session_id"], timeout=2)

    assert finished["status"] == "exited"
    assert result["session_id"] not in process_registry.snapshot_running_ids(task_id)
    await terminal.cleanup_vm(task_id)


@pytest.mark.asyncio
async def test_cleanup_all_environments_reaps_every_active_task(tmp_path):
    task_ids = {"background-cleanup-all-a", "background-cleanup-all-b"}
    with terminal._env_lock:
        previously_active = set(terminal._active_environments)
    started = {}
    for task_id in task_ids:
        started[task_id] = (
            json.loads(
                await terminal.terminal_tool(
                    "sleep 30",
                    background=True,
                    workdir=str(tmp_path),
                    task_id=task_id,
                )
            )
        )

    try:
        cleaned = await terminal.cleanup_all_environments()
        assert cleaned == len(previously_active | task_ids)
        with terminal._env_lock:
            assert not terminal._active_environments
        assert all(
            item["session_id"] not in process_registry.snapshot_running_ids(task_id)
            for task_id, item in started.items()
        )
    finally:
        for task_id in task_ids:
            await terminal.cleanup_vm(task_id)
