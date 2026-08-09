"""Hermetic end-to-end coverage for the retained async agent chain."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

from hermes_state import SessionDB
from run_agent import AIAgent
from tools import env_probe


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_provider_terminal_session_and_trajectory_form_one_async_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use a real HTTP transport and terminal subprocess through ``AIAgent``."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    requests: list[dict[str, Any]] = []

    async def chat_completions(request: web.Request) -> web.Response:
        payload = await request.json()
        requests.append(payload)
        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream"},
        )
        await response.prepare(request)
        if len(requests) in {1, 3}:
            tool_name = "terminal" if len(requests) == 1 else "skills_list"
            tool_arguments = (
                {"command": "printf NATIVE_ASYNC_OBSERVATION"}
                if tool_name == "terminal"
                else {}
            )
            tool_call_id = (
                "call-terminal" if tool_name == "terminal" else "call-skills"
            )
            chunks = [
                {
                    "id": "tool-request",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "integration-model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": tool_call_id,
                                        "type": "function",
                                        "function": {
                                            "name": tool_name,
                                            "arguments": json.dumps(tool_arguments),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "tool-request",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "integration-model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
            ]
        else:
            chunks = [
                {
                    "id": "final-answer",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "integration-model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": (
                                    "Nothing to save."
                                    if len(requests) == 4
                                    else "NATIVE_ASYNC_FINAL"
                                ),
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "final-answer",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "integration-model",
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "stop"}
                    ],
                },
            ]
        chunks.append(
            {
                "id": chunks[-1]["id"],
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "integration-model",
                "choices": [],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 4,
                    "total_tokens": 16,
                },
            }
        )
        for chunk in chunks:
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    application = web.Application()
    application.router.add_post("/v1/chat/completions", chat_completions)
    runner = web.AppRunner(application)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = getattr(site, "_server").sockets
    port = sockets[0].getsockname()[1]

    database = SessionDB(tmp_path / "state.db")
    agent = AIAgent(
        provider="custom",
        api_key="integration-key",
        base_url=f"http://127.0.0.1:{port}/v1",
        model="integration-model",
        max_iterations=4,
        enabled_toolsets=["terminal", "skills"],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=True,
        session_db=database,
    )
    agent.compression_enabled = False

    try:
        # Warm accepted async file/SQLite helpers before the blocking detector
        # measures the provider → subprocess → persistence chain itself.
        await database.session_count()
        await env_probe.warm_environment_probe_async()
        await agent._ensure_provider_runtime()
        agent._skill_nudge_interval = 1
        async with (
            no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
            no_task_leaks(action=LeakAction.RAISE),
            agent,
        ):
            result = await agent.run_conversation(
                "Run the terminal command and use its observation."
            )
            review_tasks = tuple(agent._background_review_tasks)
            assert len(review_tasks) == 1
            await asyncio.gather(*review_tasks)
            persisted = await database.get_messages(agent.session_id)
    finally:
        await agent.close()
        await database.close()
        await runner.cleanup()

    assert result["completed"] is True
    assert result["final_response"] == "NATIVE_ASYNC_FINAL"
    assert [message["role"] for message in result["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert "NATIVE_ASYNC_OBSERVATION" in result["messages"][2]["content"]

    assert len(requests) == 4
    first_messages = requests[0]["messages"]
    second_messages = requests[1]["messages"]
    assert second_messages[: len(first_messages)] == first_messages
    assert [message["role"] for message in second_messages[-2:]] == [
        "assistant",
        "tool",
    ]
    assert second_messages[-1]["tool_call_id"] == "call-terminal"

    review_first = requests[2]
    review_second = requests[3]
    assert review_first["messages"][0] == requests[0]["messages"][0]
    assert review_first["tools"] == requests[0]["tools"]
    assert review_first["messages"][-1]["role"] == "user"
    assert review_first["messages"][-1]["content"].startswith(
        "Review the conversation above and update the skill library"
    )
    assert [message["role"] for message in review_second["messages"][-2:]] == [
        "assistant",
        "tool",
    ]
    assert review_second["messages"][-1]["tool_call_id"] == "call-skills"

    assert [message["role"] for message in persisted] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]

    rows = [
        json.loads(line)
        for line in (tmp_path / "trajectory_samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 1
    trajectory = rows[0]["conversations"]
    assert [turn["from"] for turn in trajectory] == [
        "system",
        "human",
        "gpt",
        "tool",
        "gpt",
    ]
    assert '"name": "terminal"' in trajectory[2]["value"]
    assert "NATIVE_ASYNC_OBSERVATION" in trajectory[3]["value"]
    assert trajectory[4]["value"].endswith("NATIVE_ASYNC_FINAL")
