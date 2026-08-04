"""Regression tests for AIAgent.commit_memory_session.

Context engines that accumulate
per-session state (LCM-style DAGs, summary stores) leaked that state from a
rotated-out session into whatever continued under the same compressor
instance.
"""

from __future__ import annotations

import pytest

from types import SimpleNamespace
from unittest.mock import MagicMock


def _make_minimal_agent(context_compressor, session_id="abc"):
    """Build an object with just enough surface for commit_memory_session to run.

    AIAgent.__init__ is too heavy for a focused unit test — bind the method
    to a SimpleNamespace-style object that has the attributes the method
    actually touches.
    """
    from run_agent import AIAgent

    obj = SimpleNamespace(
        context_compressor=context_compressor,
        session_id=session_id,
    )
    obj.commit_memory_session = AIAgent.commit_memory_session.__get__(obj)
    return obj


@pytest.mark.asyncio
async def test_commit_memory_session_notifies_context_engine():
    """The active built-in context engine receives the session boundary."""
    ctx = MagicMock()
    agent = _make_minimal_agent(ctx, session_id="sess-42")

    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    await agent.commit_memory_session(msgs)

    ctx.on_session_end.assert_called_once_with("sess-42", msgs)








