"""Async parity coverage for compression activity heartbeat state."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.conversation_compression import (
    CompressionCommitFence,
    _CompressionActivityHeartbeat,
)
from agent.session_activity import ActivityProvenance


def _activity_agent(provenance=ActivityProvenance.UNKNOWN):
    agent = SimpleNamespace(
        _last_activity_provenance=provenance,
        _last_activity_desc="",
        touches=[],
        _drain_session_activity_persist=AsyncMock(),
    )

    def touch(desc, *, provenance=None, force_persist=False):
        agent.touches.append((desc, provenance, force_persist))
        agent._last_activity_provenance = provenance
        agent._last_activity_desc = desc

    agent._touch_activity = touch
    return agent


@pytest.mark.asyncio
async def test_heartbeat_stop_force_persists_terminal_state():
    agent = _activity_agent()
    heartbeat = _CompressionActivityHeartbeat(agent, interval_seconds=3600.0)

    heartbeat.start()
    await heartbeat.stop("context compression completed")

    assert agent.touches == [
        (
            "context compression started",
            ActivityProvenance.AGENT_COMPRESSION,
            False,
        ),
        (
            "context compression completed",
            ActivityProvenance.AGENT_COMPRESSION,
            True,
        ),
    ]
    agent._drain_session_activity_persist.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_provenance",
    [
        ActivityProvenance.AGENT_COMPRESSION_TIMEOUT,
        ActivityProvenance.AGENT_COMPRESSION_COOLDOWN,
    ],
)
async def test_heartbeat_does_not_clobber_terminal_provenance(
    terminal_provenance,
):
    agent = _activity_agent(terminal_provenance)
    agent._last_activity_desc = "terminal compression state"
    heartbeat = _CompressionActivityHeartbeat(agent, interval_seconds=60.0)

    heartbeat._touch("context compression in progress")
    await heartbeat.stop("context compression completed")

    assert agent.touches == []
    assert agent._last_activity_provenance is terminal_provenance
    assert agent._last_activity_desc == "terminal compression state"


@pytest.mark.asyncio
async def test_heartbeat_start_republishes_new_compression_episode():
    agent = _activity_agent(ActivityProvenance.AGENT_COMPRESSION_TIMEOUT)
    heartbeat = _CompressionActivityHeartbeat(agent, interval_seconds=60.0)

    heartbeat.start()
    await heartbeat.stop()

    assert agent.touches[0][:2] == (
        "context compression started",
        ActivityProvenance.AGENT_COMPRESSION,
    )
    assert agent.touches[-1][:2] == (
        "context compression completed",
        ActivityProvenance.AGENT_COMPRESSION,
    )


@pytest.mark.asyncio
async def test_heartbeat_latches_suppression_after_terminal_state():
    agent = _activity_agent(ActivityProvenance.AGENT_COMPRESSION_TIMEOUT)
    heartbeat = _CompressionActivityHeartbeat(agent, interval_seconds=60.0)

    heartbeat._touch("context compression in progress")
    assert heartbeat._suppressed is True
    agent._last_activity_provenance = ActivityProvenance.UNKNOWN
    heartbeat._touch("context compression in progress")
    await heartbeat.stop("context compression completed")

    assert agent.touches == []


@pytest.mark.asyncio
async def test_cancelled_commit_fence_suppresses_heartbeat_and_stop():
    agent = _activity_agent(ActivityProvenance.AGENT_COMPRESSION)
    fence = CompressionCommitFence()
    assert await fence.cancel_before_commit() is True
    heartbeat = _CompressionActivityHeartbeat(
        agent,
        interval_seconds=60.0,
        commit_fence=fence,
    )

    heartbeat._touch("context compression in progress")
    await heartbeat.stop("context compression completed")

    assert agent.touches == []
    assert heartbeat._suppressed is True
