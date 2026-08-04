"""Native-async progress timeout and durable commit-fence contracts."""

from __future__ import annotations

import asyncio

import pytest

from agent.conversation_compression import (
    CompressionCommitFence,
    resolve_context_compression_timeouts,
    run_compress_context_with_progress_timeout,
)


def test_resolve_context_compression_timeouts():
    assert resolve_context_compression_timeouts({}) == (120.0, 600.0)
    assert resolve_context_compression_timeouts(
        {"context_timeout_seconds": 0}
    ) == (0.0, 600.0)
    assert resolve_context_compression_timeouts(
        {
            "context_timeout_seconds": 90,
            "context_total_ceiling_seconds": 30,
        }
    ) == (90.0, 90.0)


@pytest.mark.asyncio
async def test_silent_compression_is_cancelled_before_commit():
    original = [{"role": "user", "content": "keep-me"}]
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def worker(fence: CompressionCommitFence):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    result = await run_compress_context_with_progress_timeout(
        worker=worker,
        messages=original,
        system_prompt_fallback="fallback",
        idle_timeout_seconds=0.02,
        total_ceiling_seconds=0.1,
    )

    assert started.is_set()
    assert cancelled.is_set()
    assert result == (original, "fallback")


@pytest.mark.asyncio
async def test_progress_extends_idle_budget_until_success():
    compressed = [{"role": "user", "content": "summary"}]

    async def worker(fence: CompressionCommitFence):
        for _ in range(4):
            await asyncio.sleep(0.015)
            fence.touch_progress()
        assert await fence.begin_commit()
        try:
            return compressed, "updated"
        finally:
            fence.finish_commit()

    result = await run_compress_context_with_progress_timeout(
        worker=worker,
        messages=[],
        system_prompt_fallback="fallback",
        idle_timeout_seconds=0.03,
        total_ceiling_seconds=0.2,
    )

    assert result == (compressed, "updated")


@pytest.mark.asyncio
async def test_commit_started_before_timeout_is_awaited():
    compressed = [{"role": "assistant", "content": "committed"}]

    async def worker(fence: CompressionCommitFence):
        assert await fence.begin_commit()
        try:
            await asyncio.sleep(0.05)
            return compressed, "updated"
        finally:
            fence.finish_commit()

    result = await run_compress_context_with_progress_timeout(
        worker=worker,
        messages=[],
        system_prompt_fallback="fallback",
        idle_timeout_seconds=0.01,
        total_ceiling_seconds=0.01,
    )

    assert result == (compressed, "updated")


@pytest.mark.asyncio
async def test_host_cancellation_revokes_commit_and_propagates():
    started = asyncio.Event()
    child_cancelled = asyncio.Event()

    async def worker(fence: CompressionCommitFence):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            child_cancelled.set()
            raise

    host = asyncio.create_task(
        run_compress_context_with_progress_timeout(
            worker=worker,
            messages=[],
            system_prompt_fallback="fallback",
            idle_timeout_seconds=30,
            total_ceiling_seconds=60,
        )
    )
    await started.wait()
    host.cancel()

    with pytest.raises(asyncio.CancelledError):
        await host
    assert child_cancelled.is_set()
