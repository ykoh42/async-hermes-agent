"""Retained upstream steering semantics for the native async delegate path.

The upstream file also covers the removed TUI/gateway RPC surface.  This
retained copy deliberately keeps the registry, lifecycle, and completion
contracts that exist in this library while leaving the removed transport
tests in upstream history.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools import delegate_tool


class _StubAgent:
    def __init__(self, accept: bool = True, boom: bool = False):
        self.accept = accept
        self.boom = boom
        self.steered: list[str] = []

    def steer(self, text: str) -> bool:
        if self.boom:
            raise RuntimeError("steer exploded")
        self.steered.append(text)
        return self.accept


def _with_registered(sid: str, agent, **extra) -> None:
    delegate_tool._register_subagent(
        {
            "subagent_id": sid,
            "parent_id": "root",
            "depth": 1,
            "goal": "test goal",
            "status": "running",
            "agent": agent,
            **extra,
        }
    )


@pytest.fixture(autouse=True)
def _clean_registry():
    delegate_tool._active_subagents.clear()
    yield
    delegate_tool._active_subagents.clear()


def test_steer_reaches_live_child():
    agent = _StubAgent()
    _with_registered("sid-steer-1", agent)
    assert delegate_tool.steer_subagent("sid-steer-1", "focus on pricing") is True
    assert agent.steered == ["focus on pricing"]


def test_unknown_or_empty_steer_is_false():
    agent = _StubAgent()
    _with_registered("sid-steer-2", agent)
    assert delegate_tool.steer_subagent("missing", "hello") is False
    assert delegate_tool.steer_subagent("sid-steer-2", "   ") is False
    assert agent.steered == []


def test_missing_rejected_and_raising_agents_degrade_to_false():
    rejected = _StubAgent(accept=False)
    raising = _StubAgent(boom=True)
    _with_registered("sid-rejected", rejected)
    _with_registered("sid-raising", raising)
    _with_registered("sid-missing-agent", None)
    assert delegate_tool.steer_subagent("sid-rejected", "hello") is False
    assert delegate_tool.steer_subagent("sid-raising", "hello") is False
    assert delegate_tool.steer_subagent("sid-missing-agent", "hello") is False


def test_recycled_id_cannot_be_unregistered_by_old_child():
    old_agent = _StubAgent()
    replacement = _StubAgent()
    _with_registered("sid-recycled", old_agent, owner_session_id="old")
    _with_registered("sid-recycled", replacement, owner_session_id="new")
    delegate_tool._unregister_subagent("sid-recycled", agent=old_agent)
    assert delegate_tool.steer_subagent("sid-recycled", "replacement") is True
    assert old_agent.steered == []
    assert replacement.steered == ["replacement"]


def test_registry_snapshot_does_not_expose_private_owner_state():
    transport = object()
    record = {"session_key": "private-owner"}
    _with_registered(
        "sid-private",
        _StubAgent(),
        owner_session_id="private-owner",
        owner_transport=transport,
        owner_session_record=record,
    )
    snapshot = delegate_tool.list_active_subagents()[0]
    assert snapshot["status"] == "running"
    for key in (
        "agent",
        "owner_session_id",
        "owner_transport",
        "owner_session_record",
        "accepting_steer",
    ):
        assert key not in snapshot
    assert "private-owner" not in repr(snapshot)


def test_owner_authority_requires_transport_and_live_session_identity():
    agent = _StubAgent()
    transport = object()
    session = object()
    _with_registered(
        "sid-authority",
        agent,
        owner_session_id="session-a",
        owner_transport=transport,
        owner_session_record=session,
    )
    assert (
        delegate_tool.steer_subagent(
            "sid-authority",
            "wrong transport",
            owner_session_id="session-a",
            owner_transport=object(),
            owner_session_record=session,
        )
        is False
    )
    assert (
        delegate_tool.steer_subagent(
            "sid-authority",
            "wrong generation",
            owner_session_id="session-a",
            owner_transport=transport,
            owner_session_record=object(),
        )
        is False
    )
    assert (
        delegate_tool.steer_subagent(
            "sid-authority",
            "accepted",
            owner_session_id="session-a",
            owner_transport=transport,
            owner_session_record=session,
        )
        is True
    )
    assert agent.steered == ["accepted"]


class _FinishedChild:
    _subagent_id = "sid-finished"
    _delegate_depth = 1
    _parent_subagent_id = None
    _credential_pool = None
    _delegate_output_schema = None
    model = "test-model"
    tool_progress_callback = None
    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_estimated_cost_usd = 0.0
    session_reasoning_tokens = 0

    def get_activity_summary(self):
        return {"api_call_count": 1, "max_iterations": 2, "current_tool": None}

    async def run_conversation(self, **_kwargs):
        return {
            "final_response": "done",
            "completed": True,
            "interrupted": False,
            "api_calls": 1,
            "messages": [],
            "pending_steer": "finish with pricing caveat",
        }

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_undelivered_steer_is_named_in_completion_entry():
    parent = SimpleNamespace(_current_task_id=None, _active_children=[])
    result = await delegate_tool._run_single_child(
        0, "finish the report", _FinishedChild(), parent
    )
    assert result["status"] == "completed"
    assert result["missed_steer"] == "finish with pricing caveat"
    assert "steer did not land" in result["summary"]
