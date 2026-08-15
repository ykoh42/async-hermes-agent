"""Per-delegation cost fields in the native-async result entry."""

import json
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.delegate_tool import delegate_task


pytestmark = pytest.mark.asyncio


def _make_mock_parent(depth=0):
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "test-key"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent.session_estimated_cost_usd = 0.0
    parent.session_cost_source = "none"
    parent.session_cost_status = "unknown"
    parent._credential_pool = None
    return parent


def _make_mock_child(cost=0.1234567, cost_status="estimated"):
    child = MagicMock()
    child.run_conversation = AsyncMock(
        return_value={
            "final_response": "done",
            "completed": True,
            "api_calls": 2,
            "messages": [],
        }
    )
    child.session_prompt_tokens = 100
    child.session_completion_tokens = 50
    child.session_estimated_cost_usd = cost
    child.session_cost_status = cost_status
    child.model = "anthropic/claude-sonnet-4"
    child.session_id = "child-session"
    child._credential_pool = None
    child._subagent_id = None
    return child


async def _run(child):
    parent = _make_mock_parent()
    with patch("run_agent.AIAgent", return_value=child):
        result = json.loads(
            await delegate_task(goal="Test per-delegation cost", parent_agent=parent)
        )
    return parent, result


async def test_entry_carries_cost_usd_and_status():
    _, result = await _run(_make_mock_child())
    entry = result["results"][0]
    assert entry["cost_usd"] == pytest.approx(0.123457, abs=1e-6)
    assert entry["cost_status"] == "estimated"


async def test_reported_status_passes_through():
    _, result = await _run(_make_mock_child(cost=0.5, cost_status="reported"))
    assert result["results"][0]["cost_status"] == "reported"


async def test_zero_cost_child_has_zero_cost_entry():
    _, result = await _run(_make_mock_child(cost=0.0, cost_status="unknown"))
    assert result["results"][0]["cost_usd"] == 0.0


async def test_internal_child_cost_fields_are_stripped():
    _, result = await _run(_make_mock_child(cost=0.25))
    for entry in result["results"]:
        assert "_child_cost_usd" not in entry
        assert "_child_role" not in entry


async def test_parent_rollup_is_preserved():
    parent, _ = await _run(_make_mock_child(cost=0.25))
    assert parent.session_estimated_cost_usd == pytest.approx(0.25)
    assert parent.session_cost_source == "subagent"
    assert parent.session_cost_status == "estimated"


async def test_non_numeric_child_cost_degrades_to_zero():
    child = _make_mock_child(cost=0.0)
    child.session_estimated_cost_usd = "not-a-number"
    _, result = await _run(child)
    assert result["results"][0]["cost_usd"] == 0.0
