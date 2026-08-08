"""Test that AIAgent binds HERMES_SESSION_ID task-locally."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from run_agent import AIAgent
from gateway.session_context import get_session_env, reset_session_vars


@pytest.fixture(autouse=True)
def _cleanup_env():
    """Clear legacy process state and task-local state around each test."""
    os.environ.pop("HERMES_SESSION_ID", None)
    reset_session_vars()
    yield
    os.environ.pop("HERMES_SESSION_ID", None)
    reset_session_vars()


def test_session_id_context_set_on_init():
    """AIAgent.__init__ binds its generated ID without mutating os.environ."""
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    assert get_session_env("HERMES_SESSION_ID") == agent.session_id
    assert "HERMES_SESSION_ID" not in os.environ
    assert len(agent.session_id) > 0


def test_session_id_context_uses_provided_id():
    """An explicitly provided ID is bound to the current task context."""
    custom_id = "20260511_120000_abc12345"
    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        session_id=custom_id,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    assert get_session_env("HERMES_SESSION_ID") == custom_id
    assert "HERMES_SESSION_ID" not in os.environ
    assert agent.session_id == custom_id

