"""Native-async process-status helper behavior."""

from __future__ import annotations

import asyncio

import pytest

from gateway import status

pytestmark = pytest.mark.asyncio


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
