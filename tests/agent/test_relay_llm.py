"""Tests for the native-async Relay physical LLM adapter."""

from __future__ import annotations

import asyncio
import contextvars
from types import SimpleNamespace

import pytest

pytest.importorskip("nemo_relay")

from agent import relay_llm, relay_runtime


@pytest.fixture()
def relay_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    relay_runtime._reset_for_tests()
    relay_runtime.get_host()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(),
        session_id="session-1",
        platform="cli",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
        lease,
        turn_id="turn-1",
        task_id="task-1",
    )
    lease.host.retain_managed_execution("test.relay_llm")
    try:
        yield lease.host.relay, turn
    finally:
        lease.host.release_managed_execution("test.relay_llm")
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()


def test_anthropic_stream_accumulator_merges_plain_provider_object():
    accumulator = relay_llm.AnthropicStreamAccumulator()
    accumulator.observe({
        "type": "message_start",
        "message": {
            "id": "message-1",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "usage": {"input_tokens": 10},
        },
    })
    accumulator.observe({
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": "hello"},
    })

    response = accumulator.response(
        SimpleNamespace(
            id="message-1",
            type="message",
            role="assistant",
            model="claude-test",
            content=[],
            stop_reason=None,
            usage={"input_tokens": 10},
        )
    )

    assert response.id == "message-1"
    assert response.content[0].text == "hello"
    assert response.usage.input_tokens == 10


def test_jsonable_does_not_probe_dynamic_attributes():
    class DynamicProviderObject:
        def __getattr__(self, name):
            raise AssertionError(f"unexpected dynamic attribute lookup: {name}")

        def __str__(self):
            return "opaque-provider-object"

    assert relay_llm._jsonable(DynamicProviderObject()) == "opaque-provider-object"


@pytest.mark.asyncio
async def test_async_provider_callback_preserves_caller_context(relay_turn):
    del relay_turn
    caller_value = contextvars.ContextVar("async_llm_caller_value", default="default")
    caller_value.set("caller")

    async def provider(_request):
        await asyncio.sleep(0)
        return {"caller_value": caller_value.get()}

    result = await relay_llm.execute(
        {"model": "test-model", "messages": []},
        provider,
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        metadata={"api_mode": "custom", "api_request_id": "request-async-context"},
    )

    assert result == {"caller_value": "caller"}


@pytest.mark.asyncio
async def test_async_non_stream_returns_namespaced_interceptor_result(relay_turn, monkeypatch):
    relay, _turn = relay_turn

    async def post_execute(_name, request, callback, **_kwargs):
        response = await callback(request)
        return {**response, "post_interceptor": True, "usage": {"input_tokens": 10}}

    monkeypatch.setattr(relay.llm, "execute", post_execute)

    async def provider(_request):
        return {"content": "raw"}

    result = await relay_llm.execute(
        {"model": "test-model", "messages": []},
        provider,
        session_id="session-1",
        name="test-provider",
        model_name="test-model",
        metadata={"api_mode": "custom", "api_request_id": "request-async-post"},
    )

    assert result.content == "raw"
    assert result.post_interceptor is True
    assert result.usage.input_tokens == 10


@pytest.mark.asyncio
async def test_async_logical_scope_is_completed_after_success(relay_turn):
    relay, turn = relay_turn
    outcomes = []
    original_pop = relay.scope.pop

    def record_pop(*args, **kwargs):
        outcomes.append((kwargs.get("output") or {}).get("outcome"))
        return original_pop(*args, **kwargs)

    relay.scope.pop = record_pop
    try:
        result = await relay_llm.execute(
            {"model": "test-model", "messages": []},
            lambda _request: _async_result({"content": "ok"}),
            session_id="session-1",
            name="test-provider",
            model_name="test-model",
            metadata={"api_mode": "custom", "api_request_id": "request-logical"},
        )
    finally:
        relay.scope.pop = original_pop

    assert result["content"] == "ok"
    assert outcomes == ["success"]
    assert turn.logical_llm_calls == {}


async def _async_result(value):
    await asyncio.sleep(0)
    return value


def test_codec_baseline_failure_is_explicit(relay_turn, monkeypatch, caplog):
    relay, _turn = relay_turn
    request_body = {"model": "test-model", "messages": []}
    request = relay.LLMRequest({}, request_body)

    class FailingCodec:
        def decode(self, _request):
            raise RuntimeError("simulated codec failure")

    monkeypatch.setattr(relay_llm, "_codec", lambda *_args, **_kwargs: FailingCodec())

    with caplog.at_level("WARNING", logger="agent.relay_llm"):
        baseline = relay_llm._codec_round_trip_request_body(
            relay,
            request,
            relay_request_body=request_body,
            metadata={"api_mode": "chat_completions"},
        )

    assert baseline is None
    assert "ignoring request rewrites" in caplog.text
