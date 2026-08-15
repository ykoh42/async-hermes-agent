import asyncio
import datetime as dt
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agent import background_review
from run_agent import AIAgent


def _parent_stub():
    failures = []
    output = []
    agent = SimpleNamespace(
        model="parent-model",
        platform="test",
        provider="openai",
        session_id="session-123",
        _memory_store=object(),
        _memory_enabled=True,
        _user_profile_enabled=False,
        _cached_system_prompt="byte-stable parent prompt",
        ephemeral_system_prompt="gateway context",
        session_start=dt.datetime(2026, 1, 1, 12, 0, 0),
        enabled_toolsets=["memory", "skills", "terminal"],
        disabled_toolsets=["spotify"],
        reasoning_config={"enabled": True, "effort": "medium"},
        prefill_messages=[{"role": "user", "content": "prefill"}],
        providers_allowed=["anthropic"],
        providers_ignored=None,
        providers_order=None,
        provider_sort="throughput",
        provider_require_parameters=False,
        provider_data_collection=None,
        tools=[
            {"type": "function", "function": {"name": "memory"}},
            {"type": "function", "function": {"name": "terminal"}},
        ],
        valid_tool_names={"memory", "terminal"},
        _tool_snapshot_generation=7,
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
    return agent, failures, output


def _runtime(*, routed=False):
    return {
        "provider": "openrouter" if routed else "openai",
        "model": "aux-model" if routed else "parent-model",
        "api_key": "test-key",
        "base_url": "https://example.invalid/v1",
        "api_mode": None,
        "credential_pool": None,
        "request_overrides": {},
        "max_tokens": None,
        "command": None,
        "args": [],
        "routed": routed,
    }


def _recorder_class(captured, *, block_in_run=False, started=None, stopped=None):
    class Recorder:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs
            self._cached_system_prompt = None
            self._memory_store = None
            self._memory_enabled = None
            self._user_profile_enabled = None
            self._memory_nudge_interval = None
            self._skill_nudge_interval = None
            self._session_messages = []
            self.tools = []
            self.valid_tool_names = set()
            self.session_start = None
            self.session_id = None

        async def _ensure_provider_runtime(self):
            captured["runtime_initialized"] = True
            self._memory_store = "config-store"
            self._memory_nudge_interval = 10
            self._skill_nudge_interval = 10
            self._session_json_enabled = True
            self.compression_enabled = True
            return True

        async def run_conversation(self, *args, **kwargs):
            captured["run_kwargs"] = kwargs
            captured["cached_system_prompt"] = self._cached_system_prompt
            captured["session_start"] = self.session_start
            captured["session_id"] = self.session_id
            captured["tools"] = self.tools
            captured["valid_tool_names"] = self.valid_tool_names
            captured["tool_snapshot_initialized"] = (
                self._tool_snapshot_initialized
            )
            captured["tool_snapshot_generation"] = (
                self._tool_snapshot_generation
            )
            captured["memory_store"] = self._memory_store
            captured["memory_nudge_interval"] = self._memory_nudge_interval
            captured["skill_nudge_interval"] = self._skill_nudge_interval
            captured["session_json_enabled"] = self._session_json_enabled
            captured["compression_enabled"] = self.compression_enabled
            if started is not None:
                started.set()
            if block_in_run:
                try:
                    await asyncio.Event().wait()
                finally:
                    if stopped is not None:
                        stopped.set()
            from hermes_cli.plugins import resolve_pre_tool_block

            captured["terminal_block"] = await resolve_pre_tool_block(
                "terminal",
                {},
            )
            self._session_messages = []
            return {"final_response": "Nothing to save."}

        async def shutdown_memory_provider(self, *args, **kwargs):
            captured["shutdown_calls"] = captured.get("shutdown_calls", 0) + 1

        async def close(self):
            captured["close_calls"] = captured.get("close_calls", 0) + 1

    return Recorder


async def _run_target(parent, recorder, runtime, *, after_spawn=None):
    target, _ = background_review.spawn_background_review_thread(
        parent,
        [],
        review_memory=True,
    )
    if after_spawn is not None:
        after_spawn()
    definitions = [
        {"type": "function", "function": {"name": "memory"}},
        {"type": "function", "function": {"name": "skill_manage"}},
    ]
    with (
        patch("run_agent.AIAgent", recorder),
        patch.object(
            background_review,
            "_resolve_review_runtime",
            AsyncMock(return_value=runtime),
        ),
        patch(
            "model_tools.get_tool_definitions",
            AsyncMock(return_value=definitions),
        ),
    ):
        await target()


@pytest.mark.asyncio
async def test_same_model_review_preserves_cache_namespace_and_tool_boundary():
    parent, failures, _ = _parent_stub()
    captured = {}
    recorder = _recorder_class(captured)

    await _run_target(parent, recorder, _runtime())

    kwargs = captured["init_kwargs"]
    assert captured["cached_system_prompt"] == parent._cached_system_prompt
    assert captured["runtime_initialized"] is True
    assert captured["session_start"] == parent.session_start
    assert captured["session_id"] == parent.session_id
    assert kwargs["ephemeral_system_prompt"] == parent.ephemeral_system_prompt
    assert kwargs["reasoning_config"] == parent.reasoning_config
    assert kwargs["prefill_messages"] == parent.prefill_messages
    assert kwargs["prefill_messages"][0] is not parent.prefill_messages[0]
    assert kwargs["providers_allowed"] == parent.providers_allowed
    assert kwargs["provider_sort"] == parent.provider_sort
    assert captured["tools"] == parent.tools
    assert captured["tools"] is not parent.tools
    assert captured["tools"][0] is not parent.tools[0]
    assert captured["valid_tool_names"] == parent.valid_tool_names
    assert captured["tool_snapshot_initialized"] is True
    assert captured["memory_store"] is parent._memory_store
    assert captured["memory_nudge_interval"] == 0
    assert captured["skill_nudge_interval"] == 0
    assert captured["session_json_enabled"] is False
    assert captured["compression_enabled"] is False
    assert captured["terminal_block"] == (
        "Background review denied non-whitelisted tool: terminal. "
        "Only memory/skill tools are allowed."
    )
    assert captured["shutdown_calls"] >= 1
    assert captured["close_calls"] == 1
    assert failures == []

    from hermes_cli.plugins import _thread_tool_whitelist

    assert _thread_tool_whitelist.get()[0] is None


@pytest.mark.asyncio
async def test_routed_review_excludes_parent_cache_namespace_fields():
    parent, failures, _ = _parent_stub()
    captured = {}
    recorder = _recorder_class(captured)

    await _run_target(parent, recorder, _runtime(routed=True))

    kwargs = captured["init_kwargs"]
    for key in (
        "reasoning_config",
        "ephemeral_system_prompt",
        "prefill_messages",
        "providers_allowed",
        "provider_sort",
    ):
        assert key not in kwargs
    assert captured["cached_system_prompt"] is None
    assert captured["session_id"] == parent.session_id
    assert failures == []


@pytest.mark.asyncio
async def test_review_freezes_tool_snapshot_before_next_turn_can_publish():
    parent, failures, _ = _parent_stub()
    captured = {}
    recorder = _recorder_class(captured)
    scheduled_tools = [
        {"type": "function", "function": {"name": "memory"}},
        {"type": "function", "function": {"name": "terminal"}},
    ]

    def publish_next_turn_snapshot():
        parent.tools[0]["function"]["name"] = "late_mcp_tool"
        parent.valid_tool_names = {"late_mcp_tool"}
        parent._tool_snapshot_generation = 8

    await _run_target(
        parent,
        recorder,
        _runtime(),
        after_spawn=publish_next_turn_snapshot,
    )

    assert captured["tools"] == scheduled_tools
    assert captured["valid_tool_names"] == {"memory", "terminal"}
    assert captured["tool_snapshot_generation"] == 7
    assert failures == []


@pytest.mark.asyncio
async def test_review_cancellation_closes_fork_and_propagates():
    parent, failures, _ = _parent_stub()
    captured = {}
    started = asyncio.Event()
    stopped = asyncio.Event()
    recorder = _recorder_class(
        captured,
        block_in_run=True,
        started=started,
        stopped=stopped,
    )
    target, _ = background_review.spawn_background_review_thread(
        parent,
        [],
        review_memory=True,
    )
    definitions = [
        {"type": "function", "function": {"name": "memory"}},
    ]
    with (
        patch("run_agent.AIAgent", recorder),
        patch.object(
            background_review,
            "_resolve_review_runtime",
            AsyncMock(return_value=_runtime()),
        ),
        patch(
            "model_tools.get_tool_definitions",
            AsyncMock(return_value=definitions),
        ),
    ):
        task = asyncio.create_task(target())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert stopped.is_set()
    assert captured["shutdown_calls"] >= 1
    assert captured["close_calls"] == 1
    assert failures == []


@pytest.mark.asyncio
async def test_review_registers_and_unregisters_parent_interrupt_slots():
    """An in-flight review is visible to interrupt/next-turn cleanup."""
    parent, failures, _ = _parent_stub()
    captured = {}
    started = asyncio.Event()
    stopped = asyncio.Event()
    recorder = _recorder_class(
        captured,
        block_in_run=True,
        started=started,
        stopped=stopped,
    )
    target, _ = background_review.spawn_background_review_thread(
        parent,
        [],
        review_memory=True,
    )
    with (
        patch("run_agent.AIAgent", recorder),
        patch.object(
            background_review,
            "_resolve_review_runtime",
            AsyncMock(return_value=_runtime()),
        ),
        patch(
            "model_tools.get_tool_definitions",
            AsyncMock(
                return_value=[
                    {"type": "function", "function": {"name": "memory"}},
                ]
            ),
        ),
    ):
        task = asyncio.create_task(target())
        await started.wait()
        fork = parent._background_review_agent
        assert fork is not None
        assert parent._active_children == [fork]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert stopped.is_set()
    assert parent._background_review_agent is None
    assert parent._active_children == []
    assert failures == []


@pytest.mark.asyncio
async def test_agent_close_cancels_and_reaps_owned_review_task():
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def target():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    with patch.object(
        background_review,
        "spawn_background_review_thread",
        return_value=(target, "prompt"),
    ):
        agent._spawn_background_review([], review_memory=True)
        await started.wait()

    await agent.close()

    assert stopped.is_set()
    assert agent._background_review_tasks == set()
    assert not any(
        task.get_name() == "hermes-background-review"
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
    )


def test_review_prompts_preserve_class_first_and_unresolved_failure_policy():
    assert "FIRST-CLASS skill signals" in background_review._SKILL_REVIEW_PROMPT
    assert "UPDATE A CURRENTLY-LOADED SKILL" in background_review._SKILL_REVIEW_PROMPT
    assert "Unresolved failures" in background_review._SKILL_REVIEW_PROMPT
    assert "**Memory**: who the user is" in background_review._COMBINED_REVIEW_PROMPT
    assert "**Skills**: how to do this class of task" in background_review._COMBINED_REVIEW_PROMPT
    assert "Unresolved failures" in background_review._COMBINED_REVIEW_PROMPT
