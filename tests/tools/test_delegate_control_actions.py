"""TDD coverage for upstream delegate_task control actions."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import tools.delegate_tool as delegate_tool

pytestmark = pytest.mark.asyncio


@pytest.fixture
def clean_subagents():
    delegate_tool._active_subagents.clear()
    yield
    delegate_tool._active_subagents.clear()


async def test_delegate_schema_keeps_upstream_control_parameters():
    properties = delegate_tool.DELEGATE_TASK_SCHEMA["parameters"]["properties"]
    assert properties["action"]["enum"] == ["spawn", "list", "steer", "stop"]
    assert properties["subagent_id"]["type"] == "string"
    assert properties["message"]["type"] == "string"


async def test_delegate_action_list_is_scoped_to_parent_tree(clean_subagents):
    parent = SimpleNamespace()
    child = SimpleNamespace(_delegate_parent_ref=lambda: parent)
    delegate_tool._active_subagents["sa-1"] = {
        "subagent_id": "sa-1",
        "parent_id": None,
        "goal": "inspect",
        "model": "test-model",
        "status": "running",
        "started_at": 0.0,
        "agent": child,
    }

    payload = json.loads(await delegate_tool.delegate_task(action="list", parent_agent=parent))
    assert payload["count"] == 1
    assert payload["subagents"][0]["subagent_id"] == "sa-1"


async def test_delegate_action_steer_uses_existing_agent_steer(clean_subagents):
    parent = SimpleNamespace()
    received: list[str] = []
    child = SimpleNamespace(
        _delegate_parent_ref=lambda: parent,
        steer=lambda text: received.append(text) or True,
    )
    delegate_tool._active_subagents["sa-2"] = {
        "subagent_id": "sa-2",
        "agent": child,
        "accepting_steer": True,
    }

    payload = json.loads(
        await delegate_tool.delegate_task(
            action="steer",
            subagent_id="sa-2",
            message="focus on the failing test",
            parent_agent=parent,
        )
    )
    assert payload == {
        "action": "steer",
        "subagent_id": "sa-2",
        "status": "queued",
    }
    assert received == ["focus on the failing test"]
