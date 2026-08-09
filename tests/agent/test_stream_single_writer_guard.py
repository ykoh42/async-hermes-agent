"""Regression tests for the best-effort single-writer fence accessors."""

import gc
import weakref

import run_agent
from agent.stream_single_writer import claim_stream_writer, stream_writer_is_current


class _RaisingFenceAgent:
    def _claim_stream_writer(self):
        raise RuntimeError("boom")

    def _stream_writer_is_current(self, token):
        raise RuntimeError("boom")


def _real_agent():
    """Build a real AIAgent without running provider initialization."""
    return object.__new__(run_agent.AIAgent)


def test_claim_swallows_fence_exceptions():
    assert claim_stream_writer(_RaisingFenceAgent()) == 0


def test_real_agent_fence_still_supersedes_and_preserves_sole_writer():
    agent = _real_agent()

    first = claim_stream_writer(agent)
    assert first > 0
    assert stream_writer_is_current(agent, first) is True

    second = claim_stream_writer(agent)
    assert second > first
    assert stream_writer_is_current(agent, first) is False
    assert stream_writer_is_current(agent, second) is True


def test_writer_tokens_are_isolated_between_agent_instances():
    first_agent = _real_agent()
    second_agent = _real_agent()

    first_token = claim_stream_writer(first_agent)
    second_token = claim_stream_writer(second_agent)

    assert stream_writer_is_current(first_agent, first_token) is True
    assert stream_writer_is_current(second_agent, second_token) is True
    assert first_agent._stream_writer_superseded() is False
    assert second_agent._stream_writer_superseded() is False


def test_task_token_does_not_extend_agent_lifetime():
    agent = _real_agent()
    claim_stream_writer(agent)
    agent_ref = weakref.ref(agent)

    del agent
    gc.collect()

    assert agent_ref() is None
