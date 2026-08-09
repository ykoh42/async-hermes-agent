"""Tests for the native Google AI Studio Gemini adapter."""

from __future__ import annotations

import asyncio
import inspect
import json
from types import SimpleNamespace

import pytest


class DummyResponse:
    def __init__(self, status_code=200, payload=None, headers=None, text=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}
        self.text = text if text is not None else json.dumps(self._payload)

    def json(self):
        return self._payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "headers", "body", "expected"),
    [
        (200, {}, "{}", "paid"),
        (429, {}, '{"error":{"message":"free_tier quota exhausted"}}', "free"),
        (429, {}, '{"error":{"message":"rate limited"}}', "paid"),
        (401, {}, "{}", "unknown"),
        (200, {"x-ratelimit-limit-requests-per-day": "1000"}, "{}", "free"),
        (429, {"x-ratelimit-limit-requests-per-day": "1501"}, "{}", "paid"),
    ],
)
async def test_probe_gemini_tier_preserves_upstream_classification(
    monkeypatch, status_code, headers, body, expected
):
    from agent.gemini_native_adapter import probe_gemini_tier

    recorded = {}

    class ProbeHTTP:
        def __init__(self, *, timeout):
            recorded["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, *, params, json, headers):
            recorded.update(
                url=url,
                params=params,
                json=json,
                headers=headers,
            )
            return DummyResponse(
                status_code=status_code,
                headers=headers_response,
                text=body,
            )

    headers_response = headers
    monkeypatch.setattr(
        "agent.gemini_native_adapter.httpx.AsyncClient", ProbeHTTP
    )

    result = await probe_gemini_tier(
        " key ",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        model="gemini-test",
        timeout=3.5,
    )

    assert result == expected
    assert recorded == {
        "timeout": 3.5,
        "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:generateContent",
        "params": {"key": "key"},
        "json": {
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "generationConfig": {"maxOutputTokens": 1},
        },
        "headers": {
            "Content-Type": "application/json",
            "X-Goog-Api-Client": recorded["headers"]["X-Goog-Api-Client"],
        },
    }


@pytest.mark.asyncio
async def test_probe_gemini_tier_returns_unknown_without_io(monkeypatch):
    from agent.gemini_native_adapter import probe_gemini_tier

    def forbidden_client(*args, **kwargs):
        raise AssertionError("empty keys must not create an HTTP client")

    monkeypatch.setattr(
        "agent.gemini_native_adapter.httpx.AsyncClient", forbidden_client
    )

    assert inspect.iscoroutinefunction(probe_gemini_tier)
    assert await probe_gemini_tier("  ") == "unknown"


@pytest.mark.asyncio
async def test_probe_gemini_tier_returns_unknown_on_transport_error(monkeypatch):
    from agent.gemini_native_adapter import probe_gemini_tier

    class BrokenHTTP:
        def __init__(self, *, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, *args, **kwargs):
            raise OSError("offline")

    monkeypatch.setattr(
        "agent.gemini_native_adapter.httpx.AsyncClient", BrokenHTTP
    )

    assert await probe_gemini_tier("key") == "unknown"











def test_parallel_tool_results_merge_into_one_user_content():
    """Gemini requires strict user/model alternation; two consecutive `user`
    contents are rejected with HTTP 400. Parallel tool calls produce two tool
    results in a row, so their functionResponses must be grouped into a single
    user content instead of two consecutive ones."""
    from agent.gemini_native_adapter import _build_gemini_contents

    messages = [
        {"role": "user", "content": "Read a.txt and b.txt"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}},
                {"id": "call_2", "type": "function",
                 "function": {"name": "read_file", "arguments": '{"path": "b.txt"}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "AAA"},
        {"role": "tool", "tool_call_id": "call_2", "content": "BBB"},
    ]

    contents, _ = _build_gemini_contents(messages)
    roles = [c["role"] for c in contents]

    # No two adjacent contents may share a role.
    assert all(roles[i] != roles[i - 1] for i in range(1, len(roles))), roles
    assert roles == ["user", "model", "user"]

    # Both parallel functionResponses land in the single trailing user content.
    response_parts = [
        p for p in contents[2]["parts"] if "functionResponse" in p
    ]
    outputs = [p["functionResponse"]["response"]["output"] for p in response_parts]
    assert outputs == ["AAA", "BBB"]


def test_consecutive_user_messages_merge_for_gemini_alternation():
    """Back-to-back user messages must also be merged, not sent as two
    consecutive user contents."""
    from agent.gemini_native_adapter import _build_gemini_contents

    messages = [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "ok"},
    ]
    contents, _ = _build_gemini_contents(messages)
    roles = [c["role"] for c in contents]
    assert roles == ["user", "model"], roles




def test_translate_native_response_surfaces_reasoning_and_tool_calls():
    from agent.gemini_native_adapter import translate_gemini_response

    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"thought": True, "text": "thinking..."},
                        {"functionCall": {"name": "search", "args": {"q": "hermes"}}},
                    ]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
        },
    }

    response = translate_gemini_response(payload, model="gemini-2.5-flash")
    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.reasoning == "thinking..."
    assert choice.message.tool_calls[0].function.name == "search"
    assert json.loads(choice.message.tool_calls[0].function.arguments) == {"q": "hermes"}


@pytest.mark.asyncio
async def test_native_client_uses_x_goog_api_key_and_native_models_endpoint(monkeypatch):
    from agent.gemini_native_adapter import GeminiNativeClient

    recorded = {}

    class DummyHTTP:
        async def post(self, url, json=None, headers=None, timeout=None):
            recorded["url"] = url
            recorded["json"] = json
            recorded["headers"] = headers
            return DummyResponse(
                payload={
                    "candidates": [
                        {
                            "content": {"parts": [{"text": "hello"}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 1,
                        "candidatesTokenCount": 1,
                        "totalTokenCount": 2,
                    },
                }
            )

        async def aclose(self):
            return None

    monkeypatch.setattr("agent.gemini_native_adapter.httpx.AsyncClient", lambda *a, **k: DummyHTTP())

    client = GeminiNativeClient(api_key="AIza-test", base_url="https://generativelanguage.googleapis.com/v1beta")
    response = await client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert recorded["url"] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    assert recorded["headers"]["x-goog-api-key"] == "AIza-test"
    assert "Authorization" not in recorded["headers"]
    assert response.choices[0].message.content == "hello"
    await client.close()
    assert client.is_closed is True


@pytest.mark.asyncio
async def test_native_client_close_finishes_through_repeated_cancellation():
    from agent.gemini_native_adapter import GeminiNativeClient

    close_started = asyncio.Event()
    allow_close = asyncio.Event()

    class BlockingHTTP:
        async def aclose(self):
            close_started.set()
            await allow_close.wait()

    client = GeminiNativeClient(api_key="AIza-test", http_client=BlockingHTTP())
    close_task = asyncio.create_task(client.close())
    await close_started.wait()
    close_task.cancel()
    await asyncio.sleep(0)
    close_task.cancel()

    await asyncio.sleep(0)
    assert close_task.done() is False

    allow_close.set()
    with pytest.raises(asyncio.CancelledError):
        await close_task

    assert client.is_closed is True








def test_native_client_accepts_injected_http_client():
    from agent.gemini_native_adapter import GeminiNativeClient

    injected = SimpleNamespace(aclose=lambda: None)
    client = GeminiNativeClient(api_key="AIza-test", http_client=injected)
    assert client._http is injected


def test_native_client_rejects_empty_api_key_with_actionable_message():
    """Empty/whitespace api_key must raise at construction, not produce a cryptic
    Google GFE 'Error 400 (Bad Request)!!1' HTML page on the first request."""
    from agent.gemini_native_adapter import GeminiNativeClient

    for bad in ("", "   ", None):
        with pytest.raises(RuntimeError) as excinfo:
            GeminiNativeClient(api_key=bad)  # type: ignore[arg-type]
        msg = str(excinfo.value)
        assert "GOOGLE_API_KEY" in msg and "GEMINI_API_KEY" in msg
        assert "aistudio.google.com" in msg


def test_native_client_exposes_coroutine_transport():
    import inspect
    from agent.gemini_native_adapter import GeminiNativeClient

    assert inspect.iscoroutinefunction(GeminiNativeClient._create_chat_completion)
    assert inspect.iscoroutinefunction(GeminiNativeClient.close)


def test_stream_event_translation_emits_tool_call_delta_with_stable_index():
    from agent.gemini_native_adapter import translate_stream_event

    tool_call_indices = {}
    event = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"functionCall": {"name": "search", "args": {"q": "abc"}}}
                    ]
                },
                "finishReason": "STOP",
            }
        ]
    }

    first = translate_stream_event(event, model="gemini-2.5-flash", tool_call_indices=tool_call_indices)
    second = translate_stream_event(event, model="gemini-2.5-flash", tool_call_indices=tool_call_indices)

    assert first[0].choices[0].delta.tool_calls[0].index == 0
    assert second[0].choices[0].delta.tool_calls[0].index == 0
    assert first[0].choices[0].delta.tool_calls[0].id == second[0].choices[0].delta.tool_calls[0].id
    assert first[0].choices[0].delta.tool_calls[0].function.arguments == '{"q": "abc"}'
    assert second[0].choices[0].delta.tool_calls[0].function.arguments == ""
    assert first[-1].choices[0].finish_reason == "tool_calls"










# ---------------------------------------------------------------------------
# X-Goog-Api-Client header tests
# ---------------------------------------------------------------------------


