"""Cancellation contracts for native skill preprocessing subprocesses."""

import asyncio

import pytest

from agent.skill_preprocessing import run_inline_shell


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_inline_shell_reap(monkeypatch):
    communicate_started = asyncio.Event()
    stop_started = asyncio.Event()
    release_stop = asyncio.Event()
    stop_completed = asyncio.Event()

    class ControlledProcess:
        returncode = None

        async def communicate(self):
            communicate_started.set()
            await asyncio.Event().wait()

        def kill(self):
            self.returncode = -9

        async def wait(self):
            stop_started.set()
            await release_stop.wait()
            stop_completed.set()
            return self.returncode

    process = ControlledProcess()

    async def create_process(*_args, **_kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

    task = asyncio.create_task(run_inline_shell("command", None, timeout=60))
    await communicate_started.wait()
    task.cancel()
    await stop_started.wait()
    task.cancel()
    await asyncio.sleep(0)

    try:
        assert task.done() is False
    finally:
        release_stop.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(stop_completed.wait(), timeout=1.0)
