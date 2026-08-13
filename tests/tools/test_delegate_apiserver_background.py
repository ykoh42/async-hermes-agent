"""API-server wake routing for background delegation."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from gateway.session_context import clear_session_vars, set_session_vars
from tools import async_delegation as ad
from tools.process_registry import process_registry

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture(autouse=True)
async def _clean_state(monkeypatch, tmp_path):
    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    await ad._reset_for_tests()
    process_registry.completion_queue = asyncio.Queue()
    yield
    await ad._reset_for_tests()


def _fake_parent():
    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "parent-session"
    parent._interrupt_requested = False
    parent._active_children = []
    return parent


def _patch_delegate(monkeypatch):
    import tools.delegate_tool as delegate_tool

    fake_child = MagicMock()
    fake_child._delegate_role = "leaf"
    fake_child.get_activity_summary.return_value = {
        "api_call_count": 0,
        "current_tool": None,
        "last_activity_ts": 0,
    }

    async def build_child(**_kwargs):
        from gateway.session_context import set_current_session_id

        set_current_session_id("child-internal-session")
        return fake_child

    async def run_child(task_index, goal, *_args, **_kwargs):
        return {
            "task_index": task_index,
            "status": "completed",
            "summary": f"done: {goal}",
            "api_calls": 1,
            "duration_seconds": 0.1,
            "model": "m",
            "exit_reason": "completed",
        }

    credentials = {
        "model": "m",
        "provider": None,
        "base_url": None,
        "api_key": None,
        "api_mode": None,
        "request_overrides": None,
        "max_output_tokens": None,
        "command": None,
        "args": None,
    }
    monkeypatch.setattr(delegate_tool, "_build_child_agent", build_child)
    monkeypatch.setattr(delegate_tool, "_run_single_child", run_child)
    monkeypatch.setattr(
        delegate_tool,
        "_resolve_delegation_credentials",
        AsyncMock(return_value=credentials),
    )
    monkeypatch.setattr(delegate_tool, "_finalize_child_results", AsyncMock())
    return delegate_tool


async def test_apiserver_session_with_id_dispatches_background(monkeypatch):
    delegate_tool = _patch_delegate(monkeypatch)
    tokens = set_session_vars(
        platform="api_server",
        chat_id="raw-sid-7",
        session_key="raw-sid-7",
        session_id="raw-sid-7",
        async_delivery=False,
    )
    try:
        output = await delegate_tool.delegate_task(
            goal="bg on api_server",
            context="ctx",
            background=True,
            parent_agent=_fake_parent(),
        )
        parsed = json.loads(output)
        assert parsed["status"] == "dispatched"
        event = await asyncio.wait_for(
            process_registry.completion_queue.get(), timeout=2
        )
    finally:
        clear_session_vars(tokens)
    assert event["origin_session_id"] == "raw-sid-7"


async def test_apiserver_session_without_id_stays_synchronous(monkeypatch):
    delegate_tool = _patch_delegate(monkeypatch)
    tokens = set_session_vars(
        platform="api_server",
        chat_id="",
        session_key="",
        session_id="",
        async_delivery=False,
    )
    try:
        output = await delegate_tool.delegate_task(
            goal="one-shot",
            context="ctx",
            background=True,
            parent_agent=_fake_parent(),
        )
    finally:
        clear_session_vars(tokens)
    parsed = json.loads(output)
    assert parsed.get("status") != "dispatched"
    assert "SYNCHRONOUSLY" in parsed["note"]
    assert process_registry.completion_queue.empty()
