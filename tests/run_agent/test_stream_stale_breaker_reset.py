"""Follow-up for the cross-turn stream-stale circuit breaker (#58962).

The breaker latches: once ``_consecutive_stale_streams`` reaches the give-up
threshold, ``interruptible_streaming_api_call`` raises BEFORE any stream is
attempted — so the "reset on successful stream" path can never run again on
its own. The breaker's error message tells the user to "switch models …
then retry", and the provider-fallback chain swaps providers on the same
agent object, so BOTH swap paths must clear the streak or a healthy new
provider would keep short-circuiting forever:

- ``switch_model()``   (user-initiated /model swap)
- ``try_activate_fallback()``  (automatic provider fallback)
- ``restore_primary_runtime()``  (turn-start restore back to the primary)

"""

import asyncio
from types import SimpleNamespace

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from run_agent import AIAgent


def _native_request_agent(execute, *, stale_timeout=0.05):
    return SimpleNamespace(
        _execute_model_request=execute,
        _compute_non_stream_stale_timeout=lambda _payload: stale_timeout,
        _provider_stale_timeout=stale_timeout,
        _consecutive_stale_streams=0,
        _current_streamed_assistant_text="",
        _touch_activity=MagicMock(),
        base_url="https://api.example.test/v1",
        model="test-model",
    )


@pytest.mark.asyncio
async def test_non_streaming_short_circuits_at_threshold(monkeypatch):
    from agent.chat_completion_helpers import interruptible_api_call

    monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "3")
    execute = AsyncMock()
    agent = _native_request_agent(execute)
    agent._consecutive_stale_streams = 3

    with pytest.raises(RuntimeError, match="unresponsive"):
        await interruptible_api_call(agent, {})

    execute.assert_not_awaited()
    assert agent._consecutive_stale_streams == 3


@pytest.mark.asyncio
async def test_non_streaming_timeout_increments_stale_streak():
    from agent.chat_completion_helpers import interruptible_api_call

    cancelled = asyncio.Event()

    async def execute(_payload):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    agent = _native_request_agent(execute, stale_timeout=0.01)

    with pytest.raises(TimeoutError, match="Non-streaming API call timed out"):
        await interruptible_api_call(agent, {})

    assert cancelled.is_set()
    assert agent._consecutive_stale_streams == 1


@pytest.mark.asyncio
async def test_provider_timeout_does_not_increment_stale_watchdog_streak():
    from agent.chat_completion_helpers import (
        interruptible_api_call,
        interruptible_streaming_api_call,
    )

    async def non_stream_timeout(_payload):
        raise TimeoutError("provider timeout")

    non_stream_agent = _native_request_agent(non_stream_timeout)
    with pytest.raises(TimeoutError, match="provider timeout"):
        await interruptible_api_call(non_stream_agent, {})
    assert non_stream_agent._consecutive_stale_streams == 0

    async def stream_timeout(_payload, **_kwargs):
        raise TimeoutError("provider stream timeout")

    stream_agent = _native_request_agent(stream_timeout)
    with pytest.raises(TimeoutError, match="provider stream timeout"):
        await interruptible_streaming_api_call(stream_agent, {})
    assert stream_agent._consecutive_stale_streams == 0


@pytest.mark.asyncio
async def test_non_streaming_wait_notice_heartbeat_is_owned(monkeypatch):
    from agent import chat_completion_helpers as helpers

    monkeypatch.setattr(helpers, "_STREAM_HEARTBEAT_INTERVAL", 0.01)
    response = object()

    async def execute(_payload):
        await asyncio.sleep(0.025)
        return response

    agent = _native_request_agent(execute, stale_timeout=1.0)
    agent._emit_wait_notice = MagicMock()

    assert await helpers.interruptible_api_call(
        agent, {"model": "heartbeat-model"}
    ) is response
    assert agent._emit_wait_notice.call_count >= 1
    assert "waiting on heartbeat-model" in (
        agent._emit_wait_notice.call_args_list[0].args[0]
    )
    assert not [
        task
        for task in asyncio.all_tasks()
        if task.get_name() == "provider-nonstream-heartbeat"
    ]


@pytest.mark.asyncio
async def test_non_streaming_wait_notice_failure_is_fail_open(monkeypatch):
    from agent import chat_completion_helpers as helpers

    monkeypatch.setattr(helpers, "_STREAM_HEARTBEAT_INTERVAL", 0.01)
    response = object()

    async def execute(_payload):
        await asyncio.sleep(0.025)
        return response

    agent = _native_request_agent(execute, stale_timeout=1.0)
    agent._emit_wait_notice = MagicMock(side_effect=ValueError("bad display state"))

    assert await helpers.interruptible_api_call(agent, {}) is response
    assert agent._emit_wait_notice.call_count >= 1


@pytest.mark.asyncio
async def test_stream_activity_reschedules_idle_timeout():
    from agent.chat_completion_helpers import interruptible_streaming_api_call

    response = object()

    async def execute(_payload, **kwargs):
        note_activity = kwargs["on_stream_activity"]
        await asyncio.sleep(0.03)
        note_activity()
        await asyncio.sleep(0.03)
        return response

    agent = _native_request_agent(execute, stale_timeout=0.05)
    agent._consecutive_stale_streams = 2

    assert await interruptible_streaming_api_call(agent, {}) is response
    assert agent._consecutive_stale_streams == 0


@pytest.mark.asyncio
async def test_stream_idle_timeout_retries_and_increments_streak_per_attempt():
    from agent.chat_completion_helpers import interruptible_streaming_api_call

    cancelled = asyncio.Event()

    async def execute(_payload, **_kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    agent = _native_request_agent(execute, stale_timeout=0.01)

    with pytest.raises(TimeoutError, match="produced no chunks"):
        await interruptible_streaming_api_call(agent, {})

    assert cancelled.is_set()
    assert agent._consecutive_stale_streams == 3


@pytest.mark.asyncio
async def test_chat_stream_stale_returns_partial_stub_and_closes_stream():
    from agent.chat_completion_helpers import interruptible_streaming_api_call
    from hermes_constants import PARTIAL_STREAM_STUB_ID

    stream_closed = asyncio.Event()

    class StalledStream:
        def __init__(self):
            self._sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._sent:
                self._sent = True
                return SimpleNamespace(
                    id="stream-id",
                    model="test-model",
                    usage=None,
                    choices=[
                        SimpleNamespace(
                            finish_reason=None,
                            delta=SimpleNamespace(
                                content="partial answer",
                                reasoning=None,
                                reasoning_content=None,
                                tool_calls=None,
                            ),
                        )
                    ],
                )
            await asyncio.Event().wait()
            raise StopAsyncIteration

        async def aclose(self):
            stream_closed.set()

    stream = StalledStream()

    class Completions:
        async def create(self, **_kwargs):
            return stream

    agent = AIAgent.__new__(AIAgent)
    agent.api_mode = "chat_completions"
    agent.provider = "test-provider"
    agent.model = "test-model"
    agent.base_url = "https://api.example.test/v1"
    agent.client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    agent._provider_stale_timeout = 0.01
    agent._consecutive_stale_streams = 0
    agent._current_streamed_assistant_text = ""
    agent._touch_activity = MagicMock()
    agent._capture_rate_limits = MagicMock()
    agent._capture_credits = MagicMock()
    agent._check_openrouter_cache_status = MagicMock()

    def record_delta(text):
        agent._current_streamed_assistant_text += text

    agent._fire_stream_delta = record_delta

    response = await interruptible_streaming_api_call(
        agent,
        {"model": "test-model", "messages": []},
    )

    assert response.id == PARTIAL_STREAM_STUB_ID
    assert response.choices[0].message.content == "partial answer"
    assert response.choices[0].finish_reason == "length"
    assert stream_closed.is_set()
    assert agent._consecutive_stale_streams == 0


@pytest.mark.asyncio
async def test_chat_stream_stale_surfaces_dropped_tool_after_visible_text():
    from agent.chat_completion_helpers import interruptible_streaming_api_call
    from hermes_constants import PARTIAL_STREAM_STUB_ID

    streams = []

    class StalledToolStream:
        def __init__(self):
            self._index = 0
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._index == 0:
                self._index += 1
                return SimpleNamespace(
                    id="stream-id",
                    model="test-model",
                    usage=None,
                    choices=[
                        SimpleNamespace(
                            finish_reason=None,
                            delta=SimpleNamespace(
                                content="partial answer",
                                reasoning=None,
                                reasoning_content=None,
                                tool_calls=None,
                            ),
                        )
                    ],
                )
            if self._index == 1:
                self._index += 1
                return SimpleNamespace(
                    id="stream-id",
                    model="test-model",
                    usage=None,
                    choices=[
                        SimpleNamespace(
                            finish_reason=None,
                            delta=SimpleNamespace(
                                content=None,
                                reasoning=None,
                                reasoning_content=None,
                                tool_calls=[
                                    SimpleNamespace(
                                        index=0,
                                        id="call-1",
                                        function=SimpleNamespace(
                                            name="terminal", arguments='{"cmd":'
                                        ),
                                        extra_content=None,
                                    )
                                ],
                            ),
                        )
                    ],
                )
            await asyncio.Event().wait()
            raise StopAsyncIteration

        async def aclose(self):
            self.closed = True

    class Completions:
        async def create(self, **_kwargs):
            stream = StalledToolStream()
            streams.append(stream)
            return stream

    agent = AIAgent.__new__(AIAgent)
    agent.api_mode = "chat_completions"
    agent.provider = "test-provider"
    agent.model = "test-model"
    agent.base_url = "https://api.example.test/v1"
    agent.client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    agent._provider_stale_timeout = 0.01
    agent._consecutive_stale_streams = 0
    agent._current_streamed_assistant_text = ""
    agent._current_stream_partial_tool_names = []
    agent._touch_activity = MagicMock()
    agent._capture_rate_limits = MagicMock()
    agent._capture_credits = MagicMock()
    agent._check_openrouter_cache_status = MagicMock()
    agent.stream_delta_callback = None
    agent._record_streamed_assistant_text = MagicMock()
    agent._fire_reasoning_delta = MagicMock()
    agent._fire_tool_gen_started = MagicMock()

    def record_delta(text):
        agent._current_streamed_assistant_text += text

    agent._fire_stream_delta = record_delta

    response = await interruptible_streaming_api_call(
        agent,
        {"model": "test-model", "messages": []},
    )

    warning = (
        "⚠ Stream stalled mid tool-call (terminal); the action was not "
        "executed. Ask me to retry if you want to continue."
    )
    assert response.id == PARTIAL_STREAM_STUB_ID
    assert response.choices[0].message.content == f"partial answer\n\n{warning}"
    assert response._dropped_tool_names == ["terminal"]
    assert warning in agent._current_streamed_assistant_text
    assert len(streams) == 3
    assert all(stream.closed for stream in streams)
    assert agent._consecutive_stale_streams == 0


def _make_agent_openrouter():
    """Minimal openrouter agent (skips __init__), mirroring
    tests/run_agent/test_switch_model_rollback.py."""
    agent = AIAgent.__new__(AIAgent)

    agent.provider = "openrouter"
    agent.model = "x-ai/grok-4"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "or-key-original"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock(name="OriginalClient")
    agent._client_kwargs = {
        "api_key": "or-key-original",
        "base_url": "https://openrouter.ai/api/v1",
    }
    agent.context_compressor = None
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = None
    agent._anthropic_client = None
    agent._is_anthropic_oauth = False
    agent._cached_system_prompt = "cached"
    agent._primary_runtime = {}
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = []
    agent._fallback_model = None
    agent._config_context_length = None

    return agent


def _make_fallback_agent(fallback_model):
    """Full-constructor agent for the fallback path, mirroring
    tests/run_agent/test_24996_fallback_exhaustion_cooldown.py."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _mock_client(base_url="https://openrouter.ai/api/v1", api_key="fb-key"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = api_key
    return mock


@pytest.mark.asyncio
async def test_switch_model_resets_stale_streak():
    """A user-initiated /model swap must clear the latched streak so the new
    provider gets a real stream attempt instead of an instant short-circuit."""
    agent = _make_agent_openrouter()
    agent._consecutive_stale_streams = 7  # past any reasonable threshold

    async def initialize_runtime():
        pending = agent._deferred_provider_runtime
        agent.provider = agent.requested_provider = pending["provider"]
        agent.model = pending["model"]
        agent.api_key = pending["api_key"]
        agent.base_url = pending["base_url"]
        agent.api_mode = pending["api_mode"]
        agent._deferred_provider_runtime = None

    agent._ensure_provider_runtime = AsyncMock(side_effect=initialize_runtime)
    agent._persist_pending_billing_route = AsyncMock()

    with patch("hermes_cli.config.load_config_readonly", new_callable=AsyncMock, return_value={}):
        await agent.switch_model(
            new_model="openai/gpt-5",
            new_provider="openrouter",
            api_key="or-key-new",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
        )

    assert agent._consecutive_stale_streams == 0


@pytest.mark.asyncio
async def test_switch_model_failure_does_not_reset_streak():
    """A failed swap rolls back — the agent is still on the wedged provider,
    so the breaker must stay latched (reset happens after the rebuild)."""
    agent = _make_agent_openrouter()
    agent._consecutive_stale_streams = 7

    agent._ensure_provider_runtime = AsyncMock(
        side_effect=RuntimeError("simulated client build failure")
    )

    with patch("hermes_cli.config.load_config_readonly", new_callable=AsyncMock, return_value={}):
        try:
            await agent.switch_model(
                new_model="openai/gpt-5",
                new_provider="openrouter",
                api_key="or-key-new",
                base_url="https://openrouter.ai/api/v1",
                api_mode="chat_completions",
            )
        except RuntimeError:
            pass

    assert agent._consecutive_stale_streams == 7


@pytest.mark.asyncio
async def test_fallback_activation_resets_stale_streak():
    """Automatic provider fallback swaps to a different backend; the streak
    measured the OLD provider and must not wedge the new one."""
    fbs = [{
        "provider": "openai",
        "model": "gpt-4o",
        "api_key": "fb-key",
        "base_url": "https://api.openai.com/v1",
    }]
    agent = _make_fallback_agent(fallback_model=fbs)
    agent._consecutive_stale_streams = 7

    with patch("openai.AsyncOpenAI", return_value=_mock_client()):
        assert await agent._try_activate_fallback() is True

    assert agent._consecutive_stale_streams == 0


@pytest.mark.asyncio
async def test_fallback_exhaustion_keeps_stale_streak():
    """When the chain is exhausted (no swap happened), the streak stays
    latched — the session is still wedged on the same provider."""
    agent = _make_fallback_agent(fallback_model=[])
    agent._consecutive_stale_streams = 7

    assert await agent._try_activate_fallback() is False
    assert agent._consecutive_stale_streams == 7
