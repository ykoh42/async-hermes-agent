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
    delegate_tool._recent_subagents.clear()
    yield
    delegate_tool._active_subagents.clear()
    delegate_tool._recent_subagents.clear()


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


async def test_rebuilt_parent_uses_durable_session_ownership(clean_subagents):
    old_parent = SimpleNamespace(session_id="sess-durable")
    child = SimpleNamespace(_delegate_parent_ref=lambda: old_parent, steer=lambda _: True)
    delegate_tool._register_subagent(
        {
            "subagent_id": "sa-durable",
            "agent": child,
            "goal": "inspect",
            "status": "running",
            "started_at": 0.0,
            "accepting_steer": True,
            "owner_agent_session_id": "sess-durable",
        }
    )
    rebuilt = SimpleNamespace(session_id="sess-durable")
    payload = json.loads(
        await delegate_tool.delegate_task(action="list", parent_agent=rebuilt)
    )
    assert payload["count"] == 1
    assert json.loads(
        await delegate_tool.delegate_task(
            action="steer",
            subagent_id="sa-durable",
            message="continue",
            parent_agent=rebuilt,
        )
    )["status"] == "queued"


async def test_durable_session_ownership_fails_closed_for_foreign_parent(
    clean_subagents,
):
    delegate_tool._register_subagent(
        {
            "subagent_id": "sa-foreign",
            "agent": SimpleNamespace(),
            "owner_agent_session_id": "sess-owner",
            "accepting_steer": True,
        }
    )
    result = await delegate_tool.delegate_task(
        action="list", parent_agent=SimpleNamespace(session_id="sess-other")
    )
    assert json.loads(result)["count"] == 0


async def test_recent_subagent_attribution_survives_unregister(clean_subagents):
    child = SimpleNamespace()
    delegate_tool._register_subagent(
        {
            "subagent_id": "sa-recent",
            "agent": child,
            "goal": "run a command",
            "delegation_id": "deleg-1",
        }
    )
    delegate_tool._unregister_subagent("sa-recent", agent=child)
    assert delegate_tool.get_subagent_attribution("sa-recent") == {
        "subagent_id": "sa-recent",
        "goal": "run a command",
        "delegation_id": "deleg-1",
    }


async def test_child_process_notification_includes_attribution(clean_subagents):
    child = SimpleNamespace()
    delegate_tool._register_subagent(
        {
            "subagent_id": "sa-notify",
            "agent": child,
            "goal": "run the build",
            "delegation_id": "deleg-2",
        }
    )
    from tools.process_registry import format_process_notification

    text = format_process_notification(
        {
            "type": "completion",
            "session_id": "proc_1",
            "task_id": "sa-notify",
            "command": "make build",
            "exit_code": 0,
            "output": "ok",
        }
    )
    assert text is not None
    assert "Started by subagent sa-notify" in text
    assert "deleg-2" in text
    assert "run the build" in text
