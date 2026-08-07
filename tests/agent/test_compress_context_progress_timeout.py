"""Native-async progress timeout and durable commit-fence contracts."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from agent.conversation_compression import (
    CompressionCommitFence,
    resolve_context_compression_timeouts,
    run_compress_context_with_progress_timeout,
)


def test_resolve_context_compression_timeouts():
    assert resolve_context_compression_timeouts() == (120.0, 600.0)
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
    assert resolve_context_compression_timeouts(
        {"context_timeout_seconds": -1}
    ) == (-1.0, 600.0)


@pytest.mark.asyncio
async def test_disabled_timeout_requires_direct_worker_call():
    async def worker(_fence):
        return [], "unused"

    with pytest.raises(ValueError, match="idle_timeout_seconds > 0"):
        await run_compress_context_with_progress_timeout(
            worker=worker,
            messages=[],
            system_prompt_fallback="fallback",
            idle_timeout_seconds=0,
            total_ceiling_seconds=1,
        )


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


@pytest.mark.asyncio
async def test_timeout_resolves_callable_fallback_only_after_cancellation():
    fallback = Mock(return_value="lazy-fallback")

    async def worker(_fence):
        await asyncio.Event().wait()

    result = await run_compress_context_with_progress_timeout(
        worker=worker,
        messages=["original"],
        system_prompt_fallback=fallback,
        idle_timeout_seconds=0.01,
        total_ceiling_seconds=0.05,
    )

    assert result == (["original"], "lazy-fallback")
    fallback.assert_called_once_with()


@pytest.mark.asyncio
async def test_success_does_not_resolve_callable_fallback():
    fallback = Mock(side_effect=AssertionError("fallback must remain lazy"))

    async def worker(fence):
        assert await fence.begin_commit()
        try:
            return ["compressed"], "updated"
        finally:
            fence.finish_commit()

    result = await run_compress_context_with_progress_timeout(
        worker=worker,
        messages=[],
        system_prompt_fallback=fallback,
        idle_timeout_seconds=0.05,
        total_ceiling_seconds=0.1,
    )

    assert result == (["compressed"], "updated")
    fallback.assert_not_called()


@pytest.mark.asyncio
async def test_commit_overrun_is_surfaced_while_commit_is_running():
    commit_started = asyncio.Event()
    release_commit = asyncio.Event()
    overrun_seen = asyncio.Event()
    overruns = []

    async def worker(fence):
        assert await fence.begin_commit()
        commit_started.set()
        try:
            await release_commit.wait()
            return ["committed"], "updated"
        finally:
            fence.finish_commit()

    def on_commit_overrun(waited, ceiling):
        overruns.append((waited, ceiling))
        overrun_seen.set()

    running = asyncio.create_task(
        run_compress_context_with_progress_timeout(
            worker=worker,
            messages=[],
            system_prompt_fallback="fallback",
            idle_timeout_seconds=0.01,
            total_ceiling_seconds=0.02,
            on_commit_overrun=on_commit_overrun,
        )
    )
    await commit_started.wait()
    await asyncio.wait_for(overrun_seen.wait(), timeout=0.5)
    assert not running.done()

    release_commit.set()
    assert await running == (["committed"], "updated")
    assert len(overruns) == 1


@pytest.mark.asyncio
async def test_agent_owned_timeout_records_cooldown_after_worker_rollback(
    monkeypatch,
):
    from agent.context_compressor import ContextCompressor
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.session_id = "s1"
    agent._session_db = None
    agent._cached_system_prompt = "sys"
    agent._emit_warning = MagicMock()
    agent._touch_activity = MagicMock()
    agent._build_system_prompt = AsyncMock(return_value="sys")
    agent._conversation_root_id = AsyncMock(return_value=None)
    agent.context_compressor = MagicMock()
    agent.context_compressor._consecutive_timeout_failures = 0
    agent.context_compressor.record_timeout_failure = (
        ContextCompressor.record_timeout_failure.__get__(
            agent.context_compressor, MagicMock
        )
    )
    agent.context_compressor._record_compression_failure_cooldown = MagicMock()

    async def hung_compress(_agent, _messages, _system_message, **_kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(
        "agent.conversation_compression.compress_context", hung_compress
    )
    monkeypatch.setattr(
        "agent.conversation_compression.resolve_context_compression_timeouts",
        lambda _config=None: (0.01, 0.05),
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly", AsyncMock(return_value={})
    )

    original = [{"role": "user", "content": "stay"}]
    with patch("agent.portal_tags.get_conversation_context", return_value=object()):
        result = await AIAgent._compress_context(agent, original, "sys")

    assert result == (original, "sys")
    assert agent.context_compressor._consecutive_timeout_failures == 1
    cooldown_args = (
        agent.context_compressor._record_compression_failure_cooldown.call_args.args
    )
    assert cooldown_args[0] == 60.0
    assert "host compress_context timeout" in cooldown_args[1]


@pytest.mark.asyncio
async def test_timeout_guard_state_is_persisted_natively(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    await db.create_session("timeout-session", source="library")
    compressor = SimpleNamespace(
        _summary_failure_cooldown_until=0.0,
        _last_summary_error=None,
        _fallback_compression_streak=0,
        _ineffective_compression_count=0,
    )
    agent = SimpleNamespace(
        context_compressor=compressor,
        _session_db=db,
        session_id="timeout-session",
    )

    async def worker(_fence):
        await asyncio.Event().wait()

    def on_timeout(*_args):
        compressor._summary_failure_cooldown_until = time.monotonic() + 60
        compressor._last_summary_error = "host timeout"

    try:
        await run_compress_context_with_progress_timeout(
            worker=worker,
            messages=[],
            system_prompt_fallback="system",
            idle_timeout_seconds=0.01,
            total_ceiling_seconds=0.05,
            on_timeout=on_timeout,
            telemetry_agent=agent,
        )
        persisted = await db.get_compression_failure_cooldown("timeout-session")
        assert persisted is not None
        assert persisted["error"] == "host timeout"
    finally:
        await db.close()
