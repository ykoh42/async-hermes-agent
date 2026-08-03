"""Regression tests for the notice-spine (AgentNotice + emitter callbacks).

Covers:
  A. _emit_notice / _emit_notice_clear emitter behaviour (bare AIAgent via
     object.__new__ — same pattern as test_steer.py and test_file_mutation_verifier.py).
  B. Constructor / init_agent signature threading.
  C. TUI _agent_cbs notice binding — mirrors the status_callback tests already
     in tests/test_tui_gateway_server.py.
"""
from __future__ import annotations

import inspect
from unittest.mock import patch

from agent.credits_tracker import AgentNotice
from run_agent import AIAgent


# ── A. Emitter behaviour ─────────────────────────────────────────────────────


def _bare_agent() -> AIAgent:
    """Build an AIAgent without running __init__ (no heavy init required).

    Only the two callback slots used by _emit_notice / _emit_notice_clear are
    installed — mirrors the pattern in test_steer.py.
    """
    agent = object.__new__(AIAgent)
    agent.notice_callback = None
    agent.notice_clear_callback = None
    return agent


class TestEmitNotice:
    def test_emit_notice_calls_callback_with_exact_notice(self):
        agent = _bare_agent()
        received = []
        notice = AgentNotice(
            text="credits 90% used",
            level="warn",
            kind="sticky",
            ttl_ms=None,
            key="credits.warn90",
            id="n1",
        )
        agent.notice_callback = received.append
        agent._emit_notice(notice)
        assert received == [notice]







# ── B. Constructor / init_agent signature threading ─────────────────────────


class TestSignatureThreading:
    def test_agent_init_exposes_notice_callback(self):
        sig = inspect.signature(AIAgent.__init__)
        assert "notice_callback" in sig.parameters
