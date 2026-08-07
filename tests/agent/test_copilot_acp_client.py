"""Native-async Copilot ACP transport and safety regressions."""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agent.copilot_acp_client import CopilotACPClient, _build_subprocess_env


class _Writer:
    def __init__(self) -> None:
        self.payload = bytearray()

    def write(self, data: bytes) -> None:
        self.payload.extend(data)

    async def drain(self) -> None:
        return None


def _fake_process() -> SimpleNamespace:
    return SimpleNamespace(stdin=_Writer())


def test_acp_child_inherits_provider_key_but_not_tier_one_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "provider-key")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-secret")
    monkeypatch.setenv("GH_TOKEN", "github-secret")
    monkeypatch.setenv("AUXILIARY_VISION_API_KEY", "aux-secret")
    monkeypatch.setenv("VIRTUAL_ENV", "/tmp/foreign-venv")

    env = _build_subprocess_env()

    assert env["OPENAI_API_KEY"] == "provider-key"
    assert "TELEGRAM_BOT_TOKEN" not in env
    assert "GH_TOKEN" not in env
    assert "AUXILIARY_VISION_API_KEY" not in env
    assert "VIRTUAL_ENV" not in env


@pytest.mark.asyncio
async def test_stream_true_preserves_tool_call_deltas(tmp_path):
    client = CopilotACPClient(acp_cwd=str(tmp_path))
    tool_response = (
        "<tool_call>"
        '{"id":"call_read","type":"function",'
        '"function":{"name":"read_file",'
        '"arguments":"{\\"path\\":\\"README.md\\"}"}}'
        "</tool_call>"
    )
    with patch.object(
        client,
        "_run_prompt",
        new=AsyncMock(return_value=(tool_response, "")),
    ):
        stream = await client._create_chat_completion(
            model="copilot-acp",
            messages=[{"role": "user", "content": "read README.md"}],
            stream=True,
        )
        chunks = [chunk async for chunk in stream]

    delta = chunks[0].choices[0].delta
    assert delta.content is None
    assert chunks[0].choices[0].finish_reason == "tool_calls"
    assert delta.tool_calls[0].id == "call_read"
    assert delta.tool_calls[0].function.name == "read_file"
    assert json.loads(delta.tool_calls[0].function.arguments) == {
        "path": "README.md"
    }
    assert chunks[1].choices == []
    assert inspect.iscoroutinefunction(client.chat.completions.create)


@pytest.mark.asyncio
async def test_read_text_file_redacts_sensitive_content(tmp_path):
    client = CopilotACPClient(acp_cwd=str(tmp_path))
    target = tmp_path / "config.env"
    target.write_text("OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012")
    process = _fake_process()

    with patch("agent.redact._REDACT_ENABLED", True):
        handled = await client._handle_server_message(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "fs/read_text_file",
                "params": {"path": str(target)},
            },
            process=process,
            cwd=str(tmp_path),
            text_parts=[],
            reasoning_parts=[],
        )

    response = json.loads(process.stdin.payload.decode().strip())
    content = response["result"]["content"]
    assert handled is True
    assert "abc123def456" not in content
    assert "OPENAI_API_KEY=" in content


@pytest.mark.asyncio
async def test_read_text_file_decodes_utf8_content(tmp_path):
    client = CopilotACPClient(acp_cwd=str(tmp_path))
    target = tmp_path / "note.md"
    target.write_text("# 中文标题\nem dash — here\n", encoding="utf-8")
    process = _fake_process()

    await client._handle_server_message(
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "fs/read_text_file",
            "params": {"path": str(target)},
        },
        process=process,
        cwd=str(tmp_path),
        text_parts=[],
        reasoning_parts=[],
    )

    response = json.loads(process.stdin.payload.decode().strip())
    assert "error" not in response
    assert "中文标题" in response["result"]["content"]
    assert "em dash —" in response["result"]["content"]


@pytest.mark.asyncio
async def test_write_text_file_respects_safe_root(tmp_path, monkeypatch):
    client = CopilotACPClient(acp_cwd=str(tmp_path))
    safe_root = tmp_path / "workspace"
    safe_root.mkdir()
    outside = tmp_path / "outside.txt"
    process = _fake_process()
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))

    await client._handle_server_message(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "fs/write_text_file",
            "params": {"path": str(outside), "content": "blocked"},
        },
        process=process,
        cwd=str(tmp_path),
        text_parts=[],
        reasoning_parts=[],
    )

    response = json.loads(process.stdin.payload.decode().strip())
    assert "error" in response
    assert "HERMES_WRITE_SAFE_ROOT" in str(response["error"])
    assert not outside.exists()


_ACP_SERVER = r"""
import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    method = request["method"]
    request_id = request["id"]
    if method == "initialize":
        response = {"jsonrpc": "2.0", "id": request_id, "result": {}}
        print(json.dumps(response), flush=True)
    elif method == "session/new":
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"sessionId": "session-1"},
        }
        print(json.dumps(response), flush=True)
    elif method == "session/prompt":
        thought = {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "update": {
                    "sessionUpdate": "agent_thought_chunk",
                    "content": {"text": "reasoning"},
                }
            },
        }
        text = {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"text": "answer"},
                }
            },
        }
        print(json.dumps(thought), flush=True)
        print(json.dumps(text), flush=True)
        print(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": {}}), flush=True)
        break
"""


def _tool_server(target: Path) -> str:
    return f'''\
import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    method = request["method"]
    request_id = request["id"]
    if method == "initialize":
        result = {{}}
    elif method == "session/new":
        result = {{"sessionId": "session-1"}}
    elif method == "session/prompt":
        prompt = request["params"]["prompt"][0]["text"]
        if "Tool:\\n" in prompt:
            message = "final after tool"
        else:
            tool_call = {{
                "id": "call_read",
                "type": "function",
                "function": {{
                    "name": "read_file",
                    "arguments": json.dumps({{"path": {str(target)!r}}}),
                }},
            }}
            message = "<tool_call>" + json.dumps(tool_call) + "</tool_call>"
        update = {{
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {{
                "update": {{
                    "sessionUpdate": "agent_message_chunk",
                    "content": {{"text": message}},
                }}
            }},
        }}
        print(json.dumps(update), flush=True)
        result = {{}}
    print(json.dumps({{"jsonrpc": "2.0", "id": request_id, "result": result}}), flush=True)
    if method == "session/prompt":
        break
'''


@pytest.mark.asyncio
@pytest.mark.live_system_guard_bypass
async def test_real_async_subprocess_round_trip(tmp_path):
    client = CopilotACPClient(
        acp_command=sys.executable,
        acp_args=["-u", "-c", _ACP_SERVER],
        acp_cwd=str(tmp_path),
    )

    completion = await client.chat.completions.create(
        model="copilot-acp",
        messages=[{"role": "user", "content": "hello"}],
    )

    message = completion.choices[0].message
    assert message.content == "answer"
    assert message.reasoning == "reasoning"
    assert client.is_closed is True
    assert client._active_process is None


@pytest.mark.asyncio
async def test_run_prompt_preserves_real_home(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    (hermes_home / "home").mkdir(parents=True)
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    monkeypatch.setenv("HOME", str(real_home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
    monkeypatch.delenv("TERMINAL_HOME_MODE", raising=False)
    monkeypatch.setattr("hermes_constants.is_container", lambda: False)
    client = CopilotACPClient(acp_cwd=str(tmp_path))

    with patch(
        "agent.copilot_acp_client.asyncio.create_subprocess_exec",
        new=AsyncMock(side_effect=FileNotFoundError("copilot not found")),
    ) as spawn:
        with pytest.raises(RuntimeError, match="Could not start Copilot ACP command"):
            await client._run_prompt("hello", timeout_seconds=1)

    env = spawn.await_args.kwargs["env"]
    assert env["HOME"] == str(real_home)
    assert env["HERMES_REAL_HOME"] == str(real_home)


@pytest.mark.asyncio
async def test_run_prompt_passes_home_when_parent_env_is_clean(monkeypatch, tmp_path):
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    client = CopilotACPClient(acp_cwd=str(tmp_path))

    with patch(
        "agent.copilot_acp_client.asyncio.create_subprocess_exec",
        new=AsyncMock(side_effect=FileNotFoundError("copilot not found")),
    ) as spawn:
        with pytest.raises(RuntimeError, match="Could not start Copilot ACP command"):
            await client._run_prompt("hello", timeout_seconds=1)

    assert spawn.await_args.kwargs["env"]["HOME"]


@pytest.mark.asyncio
async def test_cancellation_terminates_and_reaps_process(tmp_path):
    script = "import sys, time\nfor _ in sys.stdin: time.sleep(60)\n"
    client = CopilotACPClient(
        acp_command=sys.executable,
        acp_args=["-u", "-c", script],
        acp_cwd=str(tmp_path),
    )
    task = asyncio.create_task(
        client.chat.completions.create(
            model="copilot-acp",
            messages=[{"role": "user", "content": "wait"}],
        )
    )
    for _ in range(100):
        if client._active_process is not None:
            break
        await asyncio.sleep(0.01)
    process = client._active_process
    assert process is not None

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.returncode is not None
    assert client._active_process is None


@pytest.mark.asyncio
async def test_close_is_idempotent(tmp_path):
    client = CopilotACPClient(acp_cwd=str(tmp_path))
    await client.close()
    await client.close()
    assert client.is_closed is True


@pytest.mark.asyncio
async def test_external_process_credentials_resolve_without_blocking(monkeypatch):
    from hermes_cli import auth

    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", "copilot-test")
    monkeypatch.setenv("HERMES_COPILOT_ACP_ARGS", "--acp --stdio --debug")
    monkeypatch.setattr(auth.shutil, "which", lambda command: f"/bin/{command}")

    credentials = await auth.resolve_external_process_provider_credentials(
        "copilot-acp"
    )

    assert credentials["command"] == "/bin/copilot-test"
    assert credentials["args"] == ["--acp", "--stdio", "--debug"]
    assert credentials["base_url"] == "acp://copilot"


@pytest.mark.asyncio
async def test_runtime_provider_preserves_acp_process_contract(monkeypatch):
    from hermes_cli import runtime_provider

    resolver = AsyncMock(
        return_value={
            "api_key": "copilot-acp",
            "base_url": "acp://copilot",
            "command": "/bin/copilot",
            "args": ["--acp", "--stdio"],
            "source": "process",
        }
    )
    monkeypatch.setattr(
        runtime_provider.auth_mod,
        "resolve_external_process_provider_credentials",
        resolver,
    )

    runtime = await runtime_provider.resolve_runtime_provider(
        requested="copilot-acp",
        target_model="gpt-4.1",
    )

    assert runtime["provider"] == "copilot-acp"
    assert runtime["api_mode"] == "chat_completions"
    assert runtime["command"] == "/bin/copilot"
    assert runtime["args"] == ["--acp", "--stdio"]
    resolver.assert_awaited_once_with("copilot-acp")


@pytest.mark.asyncio
async def test_direct_agent_lazy_initializes_native_acp_client():
    from run_agent import AIAgent

    agent = AIAgent(
        model="gpt-4.1",
        provider="copilot-acp",
        api_key="copilot-acp",
        base_url="acp://copilot",
        acp_command=sys.executable,
        acp_args=["-V"],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._runtime_config_loaded = True
    agent._runtime_config_snapshot = {}
    try:
        assert await agent._ensure_provider_runtime() is True
        assert isinstance(agent.client, CopilotACPClient)
        assert inspect.iscoroutinefunction(agent.client.chat.completions.create)
        assert agent._client_kwargs["command"] == sys.executable
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_agent_conversation_round_trip_preserves_acp_reasoning(
    tmp_path, monkeypatch
):
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    agent = AIAgent(
        model="gpt-4.1",
        provider="copilot-acp",
        api_key="copilot-acp",
        base_url="acp://copilot",
        acp_command=sys.executable,
        acp_args=["-u", "-c", _ACP_SERVER],
        quiet_mode=True,
        max_iterations=2,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._runtime_config_loaded = True
    agent._runtime_config_snapshot = {}
    try:
        result = await agent.run_conversation("hello")
    finally:
        await agent.close()

    assert result["completed"] is True
    assert result["final_response"] == "answer"
    assert result["api_calls"] == 1
    assistant = next(
        message for message in reversed(result["messages"])
        if message.get("role") == "assistant"
    )
    assert assistant["reasoning"] == "reasoning"


@pytest.mark.asyncio
@pytest.mark.live_system_guard_bypass
async def test_agent_tool_loop_preserves_call_observation_order(tmp_path, monkeypatch):
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    target = tmp_path / "note.txt"
    target.write_text("tool observation")
    agent = AIAgent(
        model="gpt-4.1",
        provider="copilot-acp",
        api_key="copilot-acp",
        base_url="acp://copilot",
        acp_command=sys.executable,
        acp_args=["-u", "-c", _tool_server(target)],
        quiet_mode=True,
        max_iterations=3,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._runtime_config_loaded = True
    agent._runtime_config_snapshot = {}
    try:
        result = await agent.run_conversation("read the note")
    finally:
        await agent.close()

    assert result["completed"] is True
    assert result["final_response"] == "final after tool"
    assert result["api_calls"] == 2
    relevant = [
        message["role"]
        for message in result["messages"]
        if message.get("role") in {"assistant", "tool"}
    ]
    assert relevant[-3:] == ["assistant", "tool", "assistant"]
    tool_message = next(
        message for message in result["messages"]
        if message.get("role") == "tool"
    )
    assert "tool observation" in tool_message["content"]


@pytest.mark.asyncio
async def test_auxiliary_provider_returns_native_acp_client(monkeypatch):
    from agent import auxiliary_client

    monkeypatch.setattr(
        "hermes_cli.auth.resolve_external_process_provider_credentials",
        AsyncMock(
            return_value={
                "api_key": "copilot-acp",
                "base_url": "acp://copilot",
                "command": sys.executable,
                "args": ["-V"],
            }
        ),
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        AsyncMock(return_value={}),
    )

    client, model = await auxiliary_client.resolve_provider_client(
        "copilot-acp", "gpt-4.1"
    )
    try:
        assert isinstance(client, CopilotACPClient)
        assert model == "gpt-4.1"
    finally:
        await client.close()
