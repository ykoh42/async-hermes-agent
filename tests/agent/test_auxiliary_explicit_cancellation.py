"""Native-async explicit cancellation at the compression boundary."""

import asyncio

import pytest

from agent.auxiliary_client import AuxiliaryExplicitCancellation
from agent.conversation_compression import (
    CompressionCommitFence,
    run_compress_context_with_progress_timeout,
)


@pytest.mark.asyncio
async def test_hard_cancel_stops_silent_compression_without_thread_fallback():
    entered = asyncio.Event()
    hard_cancel = asyncio.Event()

    async def worker(_fence: CompressionCommitFence):
        entered.set()
        await asyncio.Event().wait()

    running = asyncio.create_task(
        run_compress_context_with_progress_timeout(
            worker=worker,
            messages=[],
            system_prompt_fallback="system",
            idle_timeout_seconds=30,
            total_ceiling_seconds=60,
            hard_cancel_event=hard_cancel,
        )
    )
    await entered.wait()
    hard_cancel.set()

    with pytest.raises(AuxiliaryExplicitCancellation):
        await asyncio.wait_for(running, timeout=1)


@pytest.mark.asyncio
async def test_hard_cancel_waits_for_started_commit():
    commit_started = asyncio.Event()
    release_commit = asyncio.Event()
    hard_cancel = asyncio.Event()

    async def worker(fence: CompressionCommitFence):
        assert await fence.begin_commit()
        commit_started.set()
        await release_commit.wait()
        fence.finish_commit()
        return ["committed"], "system"

    running = asyncio.create_task(
        run_compress_context_with_progress_timeout(
            worker=worker,
            messages=[],
            system_prompt_fallback="system",
            idle_timeout_seconds=30,
            total_ceiling_seconds=60,
            hard_cancel_event=hard_cancel,
        )
    )
    await commit_started.wait()
    hard_cancel.set()
    await asyncio.sleep(0)
    assert not running.done()
    release_commit.set()

    assert await running == (["committed"], "system")
