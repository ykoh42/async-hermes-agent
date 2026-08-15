"""Regression tests for the native-async background review lifecycle."""

from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agent import background_review


def _parent_stub() -> SimpleNamespace:
    output: list[str] = []
    failures: list[tuple[str, Exception]] = []
    return SimpleNamespace(
        model="parent-model",
        platform="test",
        provider="openai",
        session_id="session-123",
        _memory_store=object(),
        _memory_enabled=True,
        _user_profile_enabled=False,
        _cached_system_prompt="stable prompt",
        session_start=datetime(2026, 1, 1, 12, 0, 0),
        enabled_toolsets=["memory", "skills"],
        disabled_toolsets=None,
        reasoning_config=None,
        prefill_messages=[],
        providers_allowed=None,
        providers_ignored=None,
        providers_order=None,
        provider_sort=None,
        provider_require_parameters=False,
        provider_data_collection=None,
        tools=[],
        valid_tool_names=set(),
        _tool_snapshot_generation=1,
        background_review_callback=None,
        memory_notifications="on",
        _MEMORY_REVIEW_PROMPT="review memory",
        _SKILL_REVIEW_PROMPT="review skills",
        _COMBINED_REVIEW_PROMPT="review both",
        _safe_print=output.append,
        _emit_auxiliary_failure=lambda task, exc: failures.append((task, exc)),
        _active_children=[],
        _background_review_agent=None,
    )


def _runtime() -> dict:
    return {
        "provider": "openai",
        "model": "parent-model",
        "api_key": "test-key",
        "base_url": "https://example.invalid/v1",
        "api_mode": None,
        "credential_pool": None,
        "request_overrides": {},
        "max_tokens": None,
        "command": None,
        "args": [],
        "routed": False,
    }


def _fake_review(events: list[tuple[str, object]], seen: dict[str, object]):
    class FakeReviewAgent:
        def __init__(self, **_kwargs):
            events.append(("init", None))
            self._session_messages = []
            self._end_session_on_close = True

        async def _ensure_provider_runtime(self):
            return True

        async def run_conversation(self, **_kwargs):
            events.append(("run_conversation", None))
            seen["end_session_on_close"] = self._end_session_on_close

        async def shutdown_memory_provider(self):
            events.append(("shutdown_memory_provider", None))

        async def close(self):
            events.append(("close", None))

        def interrupt(self, message=None):
            seen["interrupt"] = message

    return FakeReviewAgent


@pytest.mark.asyncio
async def test_background_review_shutdown_precedes_close():
    parent = _parent_stub()
    events: list[tuple[str, object]] = []
    seen: dict[str, object] = {}
    fake = _fake_review(events, seen)
    target, _ = background_review.spawn_background_review_thread(
        parent, [{"role": "user", "content": "hello"}], review_memory=True
    )

    with (
        patch("run_agent.AIAgent", fake),
        patch.object(background_review, "_resolve_review_runtime", AsyncMock(return_value=_runtime())),
        patch(
            "model_tools.get_tool_definitions",
            AsyncMock(return_value=[{"type": "function", "function": {"name": "memory"}}]),
        ),
    ):
        await target()

    assert [name for name, _ in events] == [
        "init",
        "run_conversation",
        "shutdown_memory_provider",
        "close",
    ]
    assert seen["end_session_on_close"] is False
    assert parent._background_review_agent is None
    assert parent._active_children == []


@pytest.mark.asyncio
async def test_background_review_registers_for_interrupt_then_unregisters():
    parent = _parent_stub()
    events: list[tuple[str, object]] = []
    seen: dict[str, object] = {}
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingFake(_fake_review(events, seen)):
        async def run_conversation(self, **_kwargs):
            started.set()
            await release.wait()

    target, _ = background_review.spawn_background_review_thread(
        parent, [], review_memory=True
    )
    with (
        patch("run_agent.AIAgent", BlockingFake),
        patch.object(background_review, "_resolve_review_runtime", AsyncMock(return_value=_runtime())),
        patch(
            "model_tools.get_tool_definitions",
            AsyncMock(return_value=[{"type": "function", "function": {"name": "memory"}}]),
        ),
    ):
        task = asyncio.create_task(target())
        await started.wait()
        fork = parent._background_review_agent
        assert fork is not None
        assert parent._active_children == [fork]
        fork.interrupt("superseded by a new live turn")
        assert seen["interrupt"] == "superseded by a new live turn"
        release.set()
        await task

    assert parent._background_review_agent is None
    assert parent._active_children == []


@pytest.mark.asyncio
async def test_new_turn_interrupts_pending_review_before_turn_setup():
    import agent.conversation_loop as conversation_loop

    calls: list[str | None] = []

    class PendingReview:
        def interrupt(self, message=None):
            calls.append(message)

    agent = SimpleNamespace(
        _background_review_agent=PendingReview(),
        _last_compaction_in_place=False,
        _last_compression_attempt_recorded=False,
        _last_compression_attempt_in_place=None,
    )
    with patch.object(
        conversation_loop,
        "build_turn_context",
        AsyncMock(side_effect=RuntimeError("setup stopped")),
    ):
        with pytest.raises(RuntimeError, match="setup stopped"):
            await conversation_loop.run_conversation(
                agent,
                "hello",
                moa_config={},
            )

    assert calls == ["superseded by a new live turn"]


def test_background_review_prompt_focus_is_appended_without_replacing_base():
    parent = _parent_stub()
    target, prompt = background_review.spawn_background_review_thread(
        parent,
        [],
        review_memory=True,
        focus="prioritize the memory boundary",
    )
    assert callable(target)
    assert prompt.startswith("review memory")
    assert "prioritize the memory boundary" in prompt
