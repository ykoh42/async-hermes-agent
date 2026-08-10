"""Native-async process-status helper behavior."""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

from gateway import status

pytestmark = pytest.mark.asyncio


async def test_psutil_process_probe_preserves_identity_and_tree_shapes():
    pid = os.getpid()

    snapshot = (await status._inspect_processes((pid,)))[pid]
    tree = await status._process_tree_pids(pid)

    assert isinstance(snapshot["name"], str)
    assert isinstance(snapshot["cmdline"], list)
    assert snapshot["cmdline"]
    assert tree[-1] == pid
    assert await status._pid_exists_including_zombie(pid) is True


async def test_process_signal_probe_refuses_recycled_pid_identity():
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(60)",
    )
    try:
        identities = await status._process_tree_identities(process.pid)
        own_identity = identities[-1]
        recycled_identity = {
            "pid": own_identity["pid"],
            "create_time": float(own_identity["create_time"]) + 1.0,
        }

        await status._signal_process_identities(
            (recycled_identity,), "terminate"
        )
        await asyncio.sleep(0.05)
        assert process.returncode is None

        await status._signal_process_identities((own_identity,), "terminate")
        await asyncio.wait_for(process.wait(), timeout=5.0)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


async def test_ps_status_repeated_cancellation_drains_process(monkeypatch):
    communicate_started = asyncio.Event()
    release_communicate = asyncio.Event()
    communicate_completed = asyncio.Event()

    class BlockingProcess:
        returncode = None
        killed = False

        async def communicate(self):
            communicate_started.set()
            await release_communicate.wait()
            communicate_completed.set()
            self.returncode = -9
            return b"", None

        async def wait(self):
            return self.returncode

        def kill(self):
            self.killed = True

    process = BlockingProcess()

    async def create_process(*_args, **_kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

    task = asyncio.create_task(status._ps_process_status(123))
    await communicate_started.wait()
    task.cancel()
    while not process.killed:
        await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)

    assert task.done() is False
    release_communicate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert communicate_completed.is_set()
