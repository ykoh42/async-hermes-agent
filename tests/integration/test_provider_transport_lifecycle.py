"""Hermetic network E2E coverage for provider retry and cancellation."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

from hermes_state import SessionDB
from run_agent import AIAgent


pytestmark = pytest.mark.integration


def _chunk(
    *,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": "provider-e2e",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "transport-model",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


async def _write_sse(
    request: web.Request,
    chunks: list[dict[str, Any]],
) -> web.StreamResponse:
    response = web.StreamResponse(
        status=200,
        headers={"Content-Type": "text/event-stream"},
    )
    await response.prepare(request)
    for chunk in chunks:
        await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
    await response.write(b"data: [DONE]\n\n")
    await response.write_eof()
    return response


@asynccontextmanager
async def _server(handler) -> AsyncIterator[str]:
    application = web.Application()
    application.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(application)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = getattr(site, "_server").sockets
    port = sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        await runner.cleanup()


def _agent(base_url: str, *, session_db: SessionDB | None = None) -> AIAgent:
    agent = AIAgent(
        provider="custom",
        api_key="transport-key",
        base_url=base_url,
        model="transport-model",
        max_iterations=4,
        enabled_toolsets=["terminal"],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_db=session_db,
    )
    agent.compression_enabled = False
    return agent


@pytest.mark.asyncio
async def test_provider_retry_stream_tool_parse_and_client_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        "agent.conversation_loop.jittered_backoff",
        lambda *_args, **_kwargs: 0.0,
    )
    requests: list[dict[str, Any]] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        requests.append(payload)
        if len(requests) == 1:
            return web.json_response(
                {"error": {"message": "temporary provider failure"}},
                status=500,
            )
        if len(requests) == 2:
            return await _write_sse(
                request,
                [
                    _chunk(
                        delta={
                            "role": "assistant",
                            "reasoning_content": "retry recovered reasoning",
                        }
                    ),
                    _chunk(
                        delta={
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-retried-terminal",
                                    "type": "function",
                                    "function": {
                                        "name": "terminal",
                                        "arguments": json.dumps(
                                            {"command": "printf RETRIED_OBSERVATION"}
                                        ),
                                    },
                                }
                            ]
                        }
                    ),
                    _chunk(delta={}, finish_reason="tool_calls"),
                ],
            )
        return await _write_sse(
            request,
            [
                _chunk(
                    delta={
                        "role": "assistant",
                        "reasoning_content": "final reasoning",
                    }
                ),
                _chunk(delta={"content": "RETRY_COMPLETE"}),
                _chunk(delta={}, finish_reason="stop"),
            ],
        )

    async with _server(handler) as base_url:
        agent = _agent(base_url)
        agent._api_max_retries = 2
        owned_client = None
        async with (
            no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
            no_task_leaks(action=LeakAction.RAISE),
            agent,
        ):
            owned_client = agent.client
            result = await agent.run_conversation("Recover and run the tool.")

    assert owned_client is not None
    assert owned_client.is_closed() is True
    assert agent.client is None
    assert len(requests) == 3
    assert result["completed"] is True
    assert result["final_response"] == "RETRY_COMPLETE"
    assert [message["role"] for message in result["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert result["messages"][1]["reasoning"] == "retry recovered reasoning"
    assert "RETRIED_OBSERVATION" in result["messages"][2]["content"]
    assert result["messages"][3]["reasoning"] == "final reasoning"
    assert requests[2]["messages"][-1]["tool_call_id"] == "call-retried-terminal"


@pytest.mark.asyncio
async def test_provider_request_cancellation_propagates_persists_and_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    request_started = asyncio.Event()
    release_handler = asyncio.Event()
    handler_finished = asyncio.Event()

    async def handler(request: web.Request) -> web.StreamResponse:
        await request.json()
        request_started.set()
        try:
            await release_handler.wait()
            return await _write_sse(
                request,
                [
                    _chunk(delta={"content": "must not be delivered"}),
                    _chunk(delta={}, finish_reason="stop"),
                ],
            )
        except ConnectionResetError:
            return web.Response(status=499)
        finally:
            handler_finished.set()

    database = SessionDB(tmp_path / "state.db")
    try:
        async with _server(handler) as base_url:
            agent = _agent(base_url, session_db=database)
            owned_client = None
            async with (
                no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
                no_task_leaks(action=LeakAction.RAISE),
                agent,
            ):
                owned_client = agent.client
                turn = asyncio.create_task(
                    agent.run_conversation("Cancel this provider request."),
                    name="provider-cancellation-e2e",
                )
                await asyncio.wait_for(request_started.wait(), timeout=5)
                turn.cancel()
                release_handler.set()
                with pytest.raises(asyncio.CancelledError):
                    await turn
                await asyncio.wait_for(handler_finished.wait(), timeout=5)
                persisted = await database.get_messages(agent.session_id)

        assert owned_client is not None
        assert owned_client.is_closed() is True
        assert agent.client is None
        assert [message["role"] for message in persisted] == ["user"]
        assert persisted[0]["content"] == "Cancel this provider request."
    finally:
        release_handler.set()
        await database.close()
