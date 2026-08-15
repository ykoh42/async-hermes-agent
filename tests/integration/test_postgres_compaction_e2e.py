"""End-to-end compaction coverage against a real PostgreSQL SessionDB."""

from __future__ import annotations

import json
import os
import uuid

import pytest

from hermes_state_postgres import SessionDB
from run_agent import AIAgent


def _sse_response(content: str) -> tuple[dict, dict]:
    return (
        {
            "id": "postgres-compaction",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "postgres-compaction-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": content},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "postgres-compaction",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "postgres-compaction-model",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
        },
    )


@pytest.mark.asyncio
async def test_postgres_compaction_preserves_prompt_tail_and_resume(monkeypatch):
    dsn = os.environ.get("HERMES_POSTGRES_TEST_DSN")
    if not dsn:
        pytest.skip("set HERMES_POSTGRES_TEST_DSN for a real PostgreSQL run")

    from aiohttp import web
    from agent.context_compressor import is_compaction_summary_message

    requests: list[list[dict]] = []

    async def complete(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        messages = body.get("messages", [])
        requests.append(messages)
        serialized = json.dumps(messages, ensure_ascii=False).lower()
        if "resumed_after_compaction" in serialized:
            content = "RESUMED_AFTER_COMPACTION"
        elif "most recent preserved assistant turn" in serialized:
            content = "LIVE_COMPRESSION_TAIL_842"
        elif len(messages) > 5 or "summar" in serialized or "compress" in serialized:
            content = "COMPACTION_SUMMARY"
        else:
            content = "COMPACTION_INIT"
        if body.get("stream", True) is False:
            return web.json_response({
                "id": "postgres-compaction",
                "object": "chat.completion",
                "created": 1,
                "model": "postgres-compaction-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 4,
                    "total_tokens": 16,
                },
            })
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for chunk in _sse_response(content):
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    database = SessionDB(dsn)
    session_id = f"postgres-compaction-{uuid.uuid4()}"
    initial_agent = AIAgent(
        provider="custom",
        api_key="postgres-compaction-key",
        base_url=f"http://127.0.0.1:{port}/v1",
        model="postgres-compaction-model",
        max_iterations=2,
        disabled_toolsets=["*"],
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=True,
        session_db=database,
        session_id=session_id,
    )
    initial_agent.compression_enabled = False

    history = []
    for index in range(8):
        history.extend([
            {
                "role": "user",
                "content": f"Historical user turn {index}: " + "alpha beta gamma " * 30,
            },
            {
                "role": "assistant",
                "content": f"Historical assistant turn {index}: "
                + "delta epsilon zeta " * 30
                + (" LIVE_COMPRESSION_TAIL_842" if index == 7 else ""),
            },
        ])

    try:
        initial = await initial_agent.run_conversation(
            "Reply with exactly COMPACTION_INIT."
        )
        system_prompt = initial_agent._cached_system_prompt
        compressor = initial_agent.context_compressor
        compressor.protect_first_n = 1
        compressor.protect_last_n = 2
        compressor.threshold_tokens = 500
        compressor.tail_token_budget = 200
        compressor.max_summary_tokens = 400

        compressed, rebuilt_prompt = await initial_agent._compress_context(
            [dict(message) for message in history],
            system_prompt,
            approx_tokens=4_000,
            force=True,
        )
        summaries = [
            message for message in compressed if is_compaction_summary_message(message)
        ]
        followup = await initial_agent.run_conversation(
            "From the most recent preserved assistant turn, reply with only "
            "the token beginning LIVE_COMPRESSION_TAIL_.",
            conversation_history=compressed,
        )
        persisted_after_compaction = await database.get_messages(session_id)
    finally:
        await initial_agent.close()

    resumed_agent = AIAgent(
        provider="custom",
        api_key="postgres-compaction-key",
        base_url=f"http://127.0.0.1:{port}/v1",
        model="postgres-compaction-model",
        max_iterations=2,
        disabled_toolsets=["*"],
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=True,
        session_db=database,
        session_id=session_id,
    )
    resumed_agent.compression_enabled = False
    try:
        resumed_history = await database.get_messages_as_conversation(session_id)
        resumed = await resumed_agent.run_conversation(
            "Reply with exactly RESUMED_AFTER_COMPACTION.",
            conversation_history=resumed_history,
        )
        persisted_after_resume = await database.get_messages(session_id)
    finally:
        await resumed_agent.close()
        await database.delete_session(session_id)
        await database.close()
        await runner.cleanup()

    assert initial["completed"] is True
    assert initial["final_response"].strip() == "COMPACTION_INIT"
    assert compressor.compression_count == 1
    assert compressor._last_compress_aborted is False
    assert compressor._last_summary_fallback_used is False
    assert compressor._last_summary_error is None
    assert len(compressed) < len(history)
    assert len(summaries) == 1
    assert rebuilt_prompt == system_prompt
    assert followup["completed"] is True
    assert followup["final_response"].strip() == "LIVE_COMPRESSION_TAIL_842"
    assert resumed["completed"] is True
    assert resumed["final_response"].strip() == "RESUMED_AFTER_COMPACTION"
    assert len(persisted_after_compaction) >= 4
    assert len(persisted_after_resume) > len(persisted_after_compaction)
    assert requests
