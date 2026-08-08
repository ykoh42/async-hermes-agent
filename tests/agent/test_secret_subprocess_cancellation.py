"""Cancellation contracts for native secret-source subprocesses."""

import asyncio

import pytest

from agent.secret_sources.base import communicate_subprocess


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_secret_helper_reap():
    communicate_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_completed = asyncio.Event()
    killed = asyncio.Event()

    class ControlledProcess:
        returncode = None

        async def communicate(self):
            communicate_started.set()
            await killed.wait()
            cleanup_started.set()
            await release_cleanup.wait()
            cleanup_completed.set()
            return b"", b""

        def kill(self):
            self.returncode = -9
            killed.set()

    helper = asyncio.create_task(
        communicate_subprocess(
            ControlledProcess(),
            timeout=60,
            timeout_message="timed out",
        )
    )
    await communicate_started.wait()
    helper.cancel()
    await cleanup_started.wait()
    helper.cancel()
    await asyncio.sleep(0)

    try:
        assert helper.done() is False
    finally:
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await helper
        await asyncio.wait_for(cleanup_completed.wait(), timeout=1.0)
