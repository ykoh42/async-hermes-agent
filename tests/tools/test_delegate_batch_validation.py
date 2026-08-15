"""Batch input validation for the native-async delegate_task API."""

import json
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.delegate_tool import delegate_task


pytestmark = pytest.mark.asyncio

GOOD_A = "Refactor the login handler to use the new session helper"
GOOD_B = "Write regression tests for the session expiry watcher"


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
    return parent


async def _call(tasks=None, *, goal=None):
    return json.loads(
        await delegate_task(
            tasks=tasks,
            goal=goal,
            parent_agent=_make_mock_parent(),
        )
    )


def _completed(index):
    return {
        "task_index": index,
        "status": "completed",
        "summary": "ok",
        "api_calls": 1,
        "duration_seconds": 1.0,
        "_child_role": None,
    }


async def test_duplicate_goals_are_allowed():
    with patch(
        "tools.delegate_tool._run_single_child",
        new=AsyncMock(side_effect=[_completed(0), _completed(1)]),
    ):
        result = await _call([{"goal": GOOD_A}, {"goal": GOOD_A}])
    assert "error" not in result
    assert len(result["results"]) == 2


async def test_case_whitespace_variant_duplicates_are_allowed():
    with patch(
        "tools.delegate_tool._run_single_child",
        new=AsyncMock(side_effect=[_completed(0), _completed(1)]),
    ):
        result = await _call([{"goal": GOOD_A}, {"goal": "  " + GOOD_A.upper() + "  "}])
    assert "error" not in result


@pytest.mark.parametrize("placeholder", ["TODO", "todo", "ToDo", "Task 123456789"])
async def test_placeholder_goals_are_rejected(placeholder):
    result = await _call([{"goal": GOOD_A}, {"goal": placeholder}])
    assert "error" in result
    assert "placeholder" in result["error"].lower()


@pytest.mark.parametrize(
    "goal",
    [
        "Implement <feature_name> end to end",
        "Summarize {file_path} for the report",
        "Deploy the service to <target environment> when ready",
        "Backfill rows for {customer id} in the billing table",
    ],
)
async def test_unexpanded_template_markers_are_rejected(goal):
    result = await _call([{"goal": GOOD_A}, {"goal": goal}])
    assert "error" in result
    assert "template" in result["error"].lower()


@pytest.mark.parametrize(
    "goal",
    [
        "Refactor the parser to return Vec<T> instead of raw pointers",
        "Fix the Result<String> error propagation in the config loader",
        "Render the sidebar inside a <div> wrapper with flex layout",
        'Update the fixture to emit {"key": 1} for the happy path',
        "Add a glob rule matching src/{a,b}/*.py to the lint config",
        "Rewrite the loop so {i} interpolates via f-strings correctly",
    ],
)
async def test_code_shaped_brackets_are_allowed(goal):
    with patch(
        "tools.delegate_tool._run_single_child",
        new=AsyncMock(side_effect=[_completed(0), _completed(1)]),
    ):
        result = await _call([{"goal": GOOD_A}, {"goal": goal}])
    assert "error" not in result


async def test_too_short_goal_is_rejected():
    result = await _call([{"goal": GOOD_A}, {"goal": "fix bug"}])
    assert "error" in result


async def test_one_task_batch_points_to_goal_form():
    result = await _call([{"goal": GOOD_A}])
    assert "error" in result
    assert "goal" in result["error"]
    assert "2" in result["error"]


async def test_valid_distinct_batch_still_runs():
    with patch(
        "tools.delegate_tool._run_single_child",
        new=AsyncMock(side_effect=[_completed(0), _completed(1)]),
    ):
        result = await _call([{"goal": GOOD_A}, {"goal": GOOD_B}])
    assert "error" not in result
    assert len(result["results"]) == 2


async def test_single_goal_form_is_exempt_from_batch_checks():
    with patch(
        "tools.delegate_tool._run_single_child",
        new=AsyncMock(return_value=_completed(0)),
    ):
        result = await _call(goal="test")
    assert "error" not in result
