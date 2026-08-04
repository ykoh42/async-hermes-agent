"""Native background subprocesses are reaped at the agent boundary."""

import asyncio
import json

import pytest

import tools.terminal_tool as terminal


@pytest.mark.asyncio
async def test_cleanup_vm_terminates_background_process(tmp_path):
    task_id = "background-cleanup"
    result = json.loads(
        await terminal.terminal_tool(
            "sleep 30", background=True, workdir=str(tmp_path), task_id=task_id
        )
    )
    pid = result["pid"]
    processes = list(terminal._background_processes[task_id])
    assert processes and processes[0].pid == pid

    await terminal.cleanup_vm(task_id)

    assert task_id not in terminal._background_processes
    assert all(process.returncode is not None for process in processes)


@pytest.mark.asyncio
async def test_background_process_reaper_removes_natural_exit(tmp_path):
    task_id = "background-natural"
    await terminal.terminal_tool(
        "sleep 0.05", background=True, workdir=str(tmp_path), task_id=task_id
    )
    await asyncio.sleep(0.15)
    assert task_id not in terminal._background_processes
    await terminal.cleanup_vm(task_id)
