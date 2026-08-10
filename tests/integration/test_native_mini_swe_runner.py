"""End-to-end native-async coverage for the retained single-task runner."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from blockbuster import BlockBuster
from pyleak import no_task_leaks
from pyleak.eventloop import LeakAction

import mini_swe_runner as runner_module
from mini_swe_runner import MiniSWERunner


pytestmark = pytest.mark.asyncio


def _tool_call(call_id: str, command: str, *, timeout: float | None = None):
    arguments = {"command": command}
    if timeout is not None:
        arguments["timeout"] = timeout
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name="terminal",
            arguments=json.dumps(arguments),
        ),
    )


def _assistant(*, content="", reasoning="", reasoning_details=None, tool_calls=None):
    return SimpleNamespace(
        content=content,
        reasoning_content=reasoning,
        reasoning_details=reasoning_details,
        tool_calls=list(tool_calls or []),
    )


class _SequenceCompletions:
    def __init__(self, *messages):
        self._messages = list(messages)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        await asyncio.sleep(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=self._messages.pop(0))]
        )


def _client(*messages):
    completions = _SequenceCompletions(*messages)
    return SimpleNamespace(
        base_url="https://provider.example/v1",
        chat=SimpleNamespace(completions=completions),
        close=AsyncMock(),
    )


@asynccontextmanager
async def _chat_completions_server(*assistant_messages):
    requests: list[dict] = []
    responses = list(assistant_messages)
    handlers: set[asyncio.Task[None]] = set()

    async def _handle(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            handlers.add(task)
        try:
            header_data = await reader.readuntil(b"\r\n\r\n")
            header_lines = header_data.decode("latin-1").split("\r\n")
            request_line = header_lines[0].split(" ", 2)
            assert request_line[0] == "POST"
            assert request_line[1] == "/v1/chat/completions"
            headers = {}
            for line in header_lines[1:]:
                if ":" in line:
                    key, value = line.split(":", 1)
                    headers[key.lower()] = value.strip()
            body = await reader.readexactly(int(headers.get("content-length", "0")))
            requests.append(json.loads(body))
            assistant_message = responses.pop(0)
            payload = {
                "id": f"chatcmpl-{len(requests)}",
                "object": "chat.completion",
                "created": 1,
                "model": "test/model",
                "choices": [
                    {
                        "index": 0,
                        "message": assistant_message,
                        "finish_reason": (
                            "tool_calls"
                            if assistant_message.get("tool_calls")
                            else "stop"
                        ),
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
            encoded = json.dumps(payload).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                + f"Content-Length: {len(encoded)}\r\n".encode()
                + b"Content-Type: application/json\r\n"
                + b"Connection: close\r\n\r\n"
                + encoded
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            if task is not None:
                handlers.discard(task)

    server = await asyncio.start_server(_handle, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    try:
        yield f"http://{host}:{port}/v1", requests
    finally:
        server.close()
        await server.wait_closed()
        if handlers:
            await asyncio.gather(*tuple(handlers), return_exceptions=True)


async def test_single_runner_real_asyncopenai_tool_round_trip(tmp_path):
    first_message = {
        "role": "assistant",
        "content": "Inspect.",
        "reasoning_content": "transport reasoning",
        "reasoning_details": [
            {"type": "reasoning.text", "text": "transport-state"}
        ],
        "tool_calls": [
            {
                "id": "real-call",
                "type": "function",
                "function": {
                    "name": "terminal",
                    "arguments": json.dumps({"command": "printf real-transport"}),
                },
            }
        ],
    }
    final_message = {
        "role": "assistant",
        "content": "Transport complete.",
    }

    async with no_task_leaks(action=LeakAction.RAISE):
        async with _chat_completions_server(
            first_message,
            final_message,
        ) as (base_url, requests):
            runner = MiniSWERunner(
                model="test/model",
                base_url=base_url,
                api_key="test-key",
                cwd=str(tmp_path),
            )
            blocker = BlockBuster()
            blocker.activate()
            try:
                result = await runner.run_task("Use the real transport")
            finally:
                blocker.deactivate()

    assert len(requests) == 2
    assert [message["role"] for message in requests[1]["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    replay = requests[1]["messages"][2]
    assert "reasoning" not in replay
    assert replay["reasoning_details"] == [
        {"type": "reasoning.text", "text": "transport-state"}
    ]
    assert "real-transport" in requests[1]["messages"][3]["content"]
    assert [item["from"] for item in result["conversations"]] == [
        "system",
        "human",
        "gpt",
        "tool",
        "gpt",
    ]
    assert result["conversations"][2]["value"].startswith(
        "<think>transport reasoning</think>"
    )
    assert result["conversations"][-1]["value"] == "Transport complete."
    assert runner.client is None


async def test_single_runner_provider_tool_observation_model_trajectory(tmp_path):
    client = _client(
        _assistant(
            content="Run both checks.",
            reasoning="inspect in order",
            reasoning_details=[{"type": "reasoning.text", "text": "signed-state"}],
            tool_calls=[
                _tool_call("call-1", "printf first"),
                _tool_call("call-2", "printf second"),
            ],
        ),
        _assistant(content="Finished.", reasoning="summarize observations"),
    )
    runner = MiniSWERunner(
        model="test/model",
        base_url="https://provider.example/v1",
        api_key="test-key",
        cwd=str(tmp_path),
        max_iterations=3,
    )
    runner.client = client
    runner._owns_client = False

    async with no_task_leaks(action=LeakAction.RAISE):
        blocker = BlockBuster()
        blocker.activate()
        try:
            result = await runner.run_task("Inspect the workspace")
        finally:
            blocker.deactivate()

    assert result["completed"] is True
    assert result["api_calls"] == 2
    assert [item["from"] for item in result["conversations"]] == [
        "system",
        "human",
        "gpt",
        "tool",
        "gpt",
    ]
    first_model = result["conversations"][2]["value"]
    observations = result["conversations"][3]["value"]
    final_model = result["conversations"][4]["value"]
    assert first_model.startswith("<think>inspect in order</think>Run both checks.")
    assert first_model.index('"name": "terminal"') < first_model.rindex(
        '"name": "terminal"'
    )
    assert observations.index('"tool_call_id": "call-1"') < observations.index(
        '"tool_call_id": "call-2"'
    )
    assert observations.index("first") < observations.index("second")
    assert final_model == "<think>summarize observations</think>Finished."

    second_request = client.chat.completions.calls[1]
    assert [message["role"] for message in second_request["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert [
        message["tool_call_id"]
        for message in second_request["messages"]
        if message["role"] == "tool"
    ] == ["call-1", "call-2"]
    replayed_assistant = second_request["messages"][2]
    assert "reasoning" not in replayed_assistant
    assert replayed_assistant["reasoning_details"] == [
        {"type": "reasoning.text", "text": "signed-state"}
    ]


async def test_single_runner_command_timeout_is_observed_and_reaped(tmp_path):
    client = _client(
        _assistant(
            tool_calls=[
                _tool_call(
                    "timeout-call",
                    'python -c "import time; time.sleep(5)"',
                    timeout=0.05,
                )
            ]
        ),
        _assistant(content="Recovered after timeout."),
    )
    runner = MiniSWERunner(
        model="test/model",
        cwd=str(tmp_path),
        max_iterations=3,
    )
    runner.client = client

    async with no_task_leaks(action=LeakAction.RAISE):
        result = await runner.run_task("Exercise timeout")

    observation = result["conversations"][3]["value"]
    assert '"exit_code": 124' in observation
    assert "Command timed out after 0.05s" in observation
    assert result["conversations"][-1]["value"] == "Recovered after timeout."


async def test_cancelled_single_runner_cleans_environment_and_owned_client(
    monkeypatch,
):
    provider_started = asyncio.Event()
    cleanup_finished = asyncio.Event()

    class _Environment:
        async def _ensure_initialized(self):
            return None

        async def cleanup(self):
            await asyncio.sleep(0)
            cleanup_finished.set()

    async def _stalled_create(**_kwargs):
        provider_started.set()
        await asyncio.Event().wait()

    environment = _Environment()
    monkeypatch.setattr(
        runner_module,
        "create_environment",
        lambda **_kwargs: environment,
    )
    client = SimpleNamespace(
        base_url="https://provider.example/v1",
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_stalled_create),
        ),
        close=AsyncMock(),
    )
    runner = MiniSWERunner(model="test/model")
    runner.client = client
    runner._owns_client = True

    async with no_task_leaks(action=LeakAction.RAISE):
        task = asyncio.create_task(runner.run_task("cancel me"))
        await asyncio.wait_for(provider_started.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert cleanup_finished.is_set()
    client.close.assert_awaited_once()
    assert runner.env is None
    assert runner.client is None


async def test_cancelled_single_runner_environment_startup_is_cleaned(
    monkeypatch,
):
    initialization_started = asyncio.Event()
    cleanup_finished = asyncio.Event()

    class _Environment:
        async def _ensure_initialized(self):
            initialization_started.set()
            await asyncio.Event().wait()

        async def cleanup(self):
            cleanup_finished.set()

    environment = _Environment()
    monkeypatch.setattr(
        runner_module,
        "create_environment",
        lambda **_kwargs: environment,
    )
    runner = MiniSWERunner(model="test/model")

    async with no_task_leaks(action=LeakAction.RAISE):
        task = asyncio.create_task(runner.run_task("cancel startup"))
        await asyncio.wait_for(initialization_started.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert cleanup_finished.is_set()
    assert runner.env is None


async def test_single_runner_closes_resolved_provider_client(
    tmp_path,
    monkeypatch,
):
    client = _client(_assistant(content="done"))
    resolver = AsyncMock(return_value=(client, "test/model"))
    monkeypatch.setattr(runner_module, "resolve_provider_client", resolver)
    runner = MiniSWERunner(model="test/model", cwd=str(tmp_path))

    result = await runner.run_task("resolve provider")

    assert result["completed"] is True
    resolver.assert_awaited_once_with("openrouter", model="test/model")
    client.close.assert_awaited_once()
    assert runner.client is None


async def test_separate_single_runners_execute_concurrently(tmp_path):
    both_started = asyncio.Event()
    release = asyncio.Event()
    started = 0

    async def _create(**_kwargs):
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        await release.wait()
        return SimpleNamespace(
            choices=[SimpleNamespace(message=_assistant(content="done"))]
        )

    runners = []
    for index in range(2):
        runner = MiniSWERunner(model="test/model", cwd=str(tmp_path / str(index)))
        runner.client = SimpleNamespace(
            base_url="https://provider.example/v1",
            chat=SimpleNamespace(completions=SimpleNamespace(create=_create)),
            close=AsyncMock(),
        )
        runners.append(runner)

    tasks = [
        asyncio.create_task(runner.run_task(f"task-{index}"))
        for index, runner in enumerate(runners)
    ]
    await asyncio.wait_for(both_started.wait(), timeout=1.0)
    release.set()
    results = await asyncio.gather(*tasks)

    assert started == 2
    assert [result["completed"] for result in results] == [True, True]


async def test_one_single_runner_serializes_concurrent_tasks(tmp_path):
    active = 0
    max_active = 0

    async def _create(**_kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=_assistant(content="done"))]
        )

    runner = MiniSWERunner(model="test/model", cwd=str(tmp_path))
    runner.client = SimpleNamespace(
        base_url="https://provider.example/v1",
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create)),
        close=AsyncMock(),
    )

    results = await asyncio.gather(
        runner.run_task("first"),
        runner.run_task("second"),
    )

    assert max_active == 1
    assert [result["completed"] for result in results] == [True, True]


async def test_single_runner_batch_writes_ordered_jsonl_natively(tmp_path, monkeypatch):
    class _Environment:
        async def _ensure_initialized(self):
            return None

        async def cleanup(self):
            return None

    monkeypatch.setattr(
        runner_module,
        "create_environment",
        lambda **_kwargs: _Environment(),
    )
    client = _client(
        _assistant(content="first answer"),
        _assistant(content="second answer"),
    )
    runner = MiniSWERunner(model="test/model")
    runner.client = client
    output = tmp_path / "single-runner.jsonl"

    results = await runner.run_batch(["first", "second"], str(output))

    lines = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert lines == results
    assert [line["conversations"][1]["value"] for line in lines] == [
        "first",
        "second",
    ]


async def test_cancelled_single_runner_batch_finishes_current_jsonl_record(
    monkeypatch,
):
    write_started = asyncio.Event()
    release_write = asyncio.Event()

    class _File:
        text = ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def write(self, value):
            write_started.set()
            await release_write.wait()
            self.text += value
            return len(value)

        async def flush(self):
            return None

    file_handle = _File()
    monkeypatch.setattr(runner_module.aiofiles, "open", lambda *_args, **_kwargs: file_handle)
    runner = MiniSWERunner(model="test/model")
    completed_result = {
        "conversations": [{"from": "human", "value": "durable"}],
        "completed": True,
        "api_calls": 1,
        "metadata": {"model": "test/model"},
    }
    runner.run_task = AsyncMock(return_value=completed_result)

    batch_task = asyncio.create_task(
        runner.run_batch(["durable"], "ignored.jsonl")
    )
    await asyncio.wait_for(write_started.wait(), timeout=1.0)
    batch_task.cancel()
    release_write.set()

    with pytest.raises(asyncio.CancelledError):
        await batch_task

    assert file_handle.text.endswith("\n")
    assert json.loads(file_handle.text) == completed_result


async def test_single_runner_constructor_is_state_only():
    with patch.object(runner_module, "AsyncOpenAI") as async_openai:
        runner = MiniSWERunner(
            base_url="https://provider.example/v1",
            api_key="test-key",
        )

    async_openai.assert_not_called()
    assert runner.client is None
    assert runner.env is None
