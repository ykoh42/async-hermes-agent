"""Cancellation contracts for the append-only trajectory writer."""

import asyncio

import pytest

from agent.trajectory import save_trajectory


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_owned_trajectory_write(monkeypatch):
    """Repeated cancellation cannot detach the shielded JSONL write task."""
    write_started = asyncio.Event()
    release_write = asyncio.Event()
    write_completed = asyncio.Event()

    class ControlledFile:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def write(self, _line):
            write_started.set()
            await release_write.wait()
            write_completed.set()

        async def flush(self):
            return None

    monkeypatch.setattr(
        "agent.trajectory.aiofiles.open",
        lambda *_args, **_kwargs: ControlledFile(),
    )

    task = asyncio.create_task(
        save_trajectory(
            [{"from": "human", "value": "cancel me"}],
            "test-model",
            completed=False,
        )
    )
    await write_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)

    try:
        assert task.done() is False
    finally:
        release_write.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(write_completed.wait(), timeout=1.0)
