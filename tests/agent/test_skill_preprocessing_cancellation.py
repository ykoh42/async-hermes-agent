"""Cancellation contracts for native skill preprocessing subprocesses."""

import asyncio

import pytest

from agent.skill_preprocessing import run_inline_shell


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_inline_shell_reap(monkeypatch):
    communicate_started = asyncio.Event()
    release_communicate = asyncio.Event()
    communicate_completed = asyncio.Event()

    class ControlledProcess:
        returncode = None

        async def communicate(self):
            communicate_started.set()
            await release_communicate.wait()
            communicate_completed.set()
            self.returncode = -9
            return b"", b""

        def kill(self):
            self.returncode = -9

        async def wait(self):
            return self.returncode

    process = ControlledProcess()

    async def create_process(*_args, **_kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

    task = asyncio.create_task(run_inline_shell("command", None, timeout=60))
    await communicate_started.wait()
    task.cancel()
    while process.returncode is None:
        await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)

    try:
        assert task.done() is False
    finally:
        release_communicate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(communicate_completed.wait(), timeout=1.0)


@pytest.mark.asyncio
async def test_inline_shell_timeout_drains_process(monkeypatch):
    communicate_completed = asyncio.Event()
    killed = asyncio.Event()

    class ControlledProcess:
        returncode = None

        async def communicate(self):
            await killed.wait()
            communicate_completed.set()
            return b"", b""

        def kill(self):
            self.returncode = -9
            killed.set()

        async def wait(self):
            return self.returncode

    async def create_process(*_args, **_kwargs):
        return ControlledProcess()

    async def timeout(*_args, **_kwargs):
        raise TimeoutError

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(asyncio, "wait_for", timeout)

    assert (
        await run_inline_shell("command", None, timeout=60)
        == "[inline-shell timeout after 60s: command]"
    )
    assert communicate_completed.is_set()


@pytest.mark.asyncio
async def test_inline_shell_communicate_error_preserves_result(monkeypatch):
    class FailedProcess:
        returncode = None

        async def communicate(self):
            raise RuntimeError("pipe failed")

        def kill(self):
            self.returncode = -9

        async def wait(self):
            return self.returncode

    async def create_process(*_args, **_kwargs):
        return FailedProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

    assert (
        await run_inline_shell("command", None, timeout=60)
        == "[inline-shell error: pipe failed]"
    )
