"""Focused native-async tests for the restored text_to_speech tool."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import shlex
import subprocess
import sys

import aiofiles
import httpx
import pytest
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

from tools import tts_tool
from tools.registry import registry


def _shell_command(*args: str) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(args))
    return " ".join(shlex.quote(str(arg)) for arg in args)


def _copy_provider_config() -> dict:
    copy_script = (
        "import pathlib,sys;"
        "pathlib.Path(sys.argv[2]).write_bytes("
        "pathlib.Path(sys.argv[1]).read_bytes())"
    )
    return {
        "provider": "test-command",
        "providers": {
            "test-command": {
                "type": "command",
                "command": _shell_command(
                    sys.executable,
                    "-c",
                    copy_script,
                    "{input_path}",
                    "{output_path}",
                ),
                "output_format": "mp3",
            },
        },
    }


def test_public_tool_contract_matches_upstream_names():
    assert inspect.iscoroutinefunction(tts_tool.text_to_speech_tool)
    assert inspect.iscoroutinefunction(tts_tool.check_tts_requirements)
    assert tts_tool.TTS_SCHEMA["name"] == "text_to_speech"
    assert tts_tool.TTS_SCHEMA["parameters"]["required"] == ["text"]
    assert set(tts_tool.TTS_SCHEMA["parameters"]["properties"]) == {
        "text",
        "output_path",
        "speed",
        "instructions",
        "provider",
    }
    entry = registry.get_entry("text_to_speech")
    assert entry is not None
    assert entry.schema is tts_tool.TTS_SCHEMA
    assert inspect.iscoroutinefunction(entry.handler)


@pytest.mark.asyncio
async def test_empty_and_cleaned_empty_text_keep_json_error_contract(monkeypatch):
    async def no_config():
        return {}

    monkeypatch.setattr(tts_tool, "_load_tts_config", no_config)
    for value in ("", "  ", "<think>only reasoning</think>"):
        result = json.loads(await tts_tool.text_to_speech_tool(value))
        assert result["success"] is False
        assert "text" in result["error"].lower()


@pytest.mark.asyncio
async def test_output_path_traversal_is_rejected_before_provider_work(
    monkeypatch,
):
    async def no_config():
        return {}

    monkeypatch.setattr(tts_tool, "_load_tts_config", no_config)
    result = json.loads(
        await tts_tool.text_to_speech_tool(
            "hello",
            output_path="audio/../../etc/cron.d/malicious",
        )
    )
    assert result["success"] is False
    assert "traversal" in result["error"].lower()


@pytest.mark.asyncio
async def test_command_provider_runs_end_to_end_without_blocking_loop(
    tmp_path,
    monkeypatch,
):
    config = _copy_provider_config()

    async def load_config():
        return config

    monkeypatch.setattr(tts_tool, "_load_tts_config", load_config)
    monkeypatch.setattr(tts_tool, "_repair_ogg_container", _identity_async)
    output = tmp_path / "spoken.mp3"
    ticks = 0
    running = True

    async def ticker():
        nonlocal ticks
        while running:
            ticks += 1
            await asyncio.sleep(0)

    ticker_task = asyncio.create_task(ticker())
    try:
        async with (
            no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.25),
            no_task_leaks(action=LeakAction.RAISE),
        ):
            result = json.loads(
                await tts_tool.text_to_speech_tool(
                    "**Hello** from Hermes",
                    output_path=str(output),
                )
            )
    finally:
        running = False
        await ticker_task

    assert result == {
        "success": True,
        "file_path": str(output),
        "media_tag": f"MEDIA:{output}",
        "provider": "test-command",
        "voice_compatible": False,
    }
    async with aiofiles.open(output, encoding="utf-8") as output_file:
        assert await output_file.read() == "Hello from Hermes"
    assert ticks > 1


async def _identity_async(path: str) -> str:
    return path


@pytest.mark.asyncio
async def test_command_timeout_reaps_process_and_reader_tasks():
    command = _shell_command(
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
    )
    with pytest.raises(subprocess.TimeoutExpired):
        await tts_tool._run_command_tts(command, timeout=0.05)
    await asyncio.sleep(0)
    assert not [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and task.get_name().startswith("tts-command-")
    ]


@pytest.mark.asyncio
async def test_command_cancellation_reaps_process_and_reader_tasks():
    command = _shell_command(
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
    )
    task = asyncio.create_task(tts_tool._run_command_tts(command, timeout=60))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    await asyncio.sleep(0)
    assert not [
        child
        for child in asyncio.all_tasks()
        if child is not asyncio.current_task()
        and not child.done()
        and child.get_name().startswith("tts-command-")
    ]


@pytest.mark.asyncio
async def test_neutts_parent_launch_uses_async_subprocess(tmp_path, monkeypatch):
    captured = {}

    class Process:
        returncode = 0

        async def communicate(self):
            async with aiofiles.open(captured["output"], "wb") as output_file:
                await output_file.write(b"RIFF" + b"\0" * 32)
            return b"", b"OK"

    async def create(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        output_index = args.index("--out") + 1
        captured["output"] = args[output_index]
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    output = tmp_path / "neutts.wav"
    assert await tts_tool._generate_neutts(
        "hello",
        str(output),
        {"neutts": {"device": "cpu"}},
    ) == str(output)
    assert "neutts_synth.py" in captured["args"][1]
    assert captured["kwargs"]["stdin"] is asyncio.subprocess.DEVNULL
    async with aiofiles.open(output, "rb") as output_file:
        assert (await output_file.read()).startswith(b"RIFF")


@pytest.mark.asyncio
async def test_neutts_default_voice_assets_are_packaged():
    assert await aiofiles.os.path.exists(tts_tool._default_neutts_ref_audio())
    assert await aiofiles.os.path.exists(tts_tool._default_neutts_ref_text())


class _StreamingResponse:
    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False

    async def aiter_bytes(self, chunk_size):
        del chunk_size
        for chunk in self._chunks:
            await asyncio.sleep(0)
            yield chunk

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_async_response_reader_enforces_body_cap_and_closes():
    response = _StreamingResponse([b"12345", b"6789"])
    with pytest.raises(RuntimeError, match="xAI TTS response exceeds 8 bytes"):
        await tts_tool._read_tts_response_bytes(
            response,
            label="xAI TTS",
            limit=8,
        )
    assert response.closed is True


@pytest.mark.asyncio
async def test_xai_provider_uses_async_http_stream(tmp_path, monkeypatch):
    request_seen = None

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_seen
        request_seen = request
        await asyncio.sleep(0)
        return httpx.Response(
            200,
            headers={"Content-Type": "audio/mpeg"},
            content=b"native-async-audio",
        )

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def async_client(**kwargs):
        del kwargs
        return real_async_client(transport=transport)

    async def credentials():
        return {
            "api_key": "test-key",
            "provider": "xai-api",
            "base_url": "https://api.x.ai/v1",
        }

    monkeypatch.setattr(tts_tool.httpx, "AsyncClient", async_client)
    monkeypatch.setattr(
        "tools.xai_http.resolve_xai_http_credentials",
        credentials,
    )
    output = tmp_path / "xai.mp3"
    returned = await tts_tool._generate_xai_tts("hello", str(output), {})
    assert returned == str(output)
    async with aiofiles.open(output, "rb") as output_file:
        assert await output_file.read() == b"native-async-audio"
    assert request_seen is not None
    assert request_seen.url == "https://api.x.ai/v1/tts"
    assert json.loads(request_seen.content) == {
        "text": "hello",
        "voice_id": "eve",
        "language": "en",
    }


@pytest.mark.asyncio
async def test_openai_provider_awaits_stream_and_closes_client(
    tmp_path,
    monkeypatch,
):
    calls = {}

    class Response:
        async def stream_to_file(self, path):
            calls["streamed"] = True
            async with aiofiles.open(path, "wb") as output_file:
                await output_file.write(b"openai-audio")

    class ResponseContext:
        async def __aenter__(self):
            return Response()

        async def __aexit__(self, exc_type, exc, traceback):
            calls["response_closed"] = True

    class StreamingResponse:
        def create(self, **kwargs):
            calls["kwargs"] = kwargs
            return ResponseContext()

    class Speech:
        with_streaming_response = StreamingResponse()

    class Audio:
        speech = Speech()

    class Client:
        audio = Audio()

        def __init__(self, **kwargs):
            calls["client_kwargs"] = kwargs

        async def close(self):
            calls["client_closed"] = True

    async def resolve_config():
        return "test-openai-key", "https://api.openai.com/v1", False

    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: Client)
    monkeypatch.setattr(
        tts_tool,
        "_resolve_openai_audio_client_config",
        resolve_config,
    )
    output = tmp_path / "openai.mp3"
    returned = await tts_tool._generate_openai_tts(
        "hello",
        str(output),
        {},
    )
    assert returned == str(output)
    assert calls["streamed"] is True
    assert calls["response_closed"] is True
    assert calls["client_closed"] is True
    assert calls["kwargs"]["input"] == "hello"
    assert calls["kwargs"]["model"] == "gpt-4o-mini-tts"
    async with aiofiles.open(output, "rb") as output_file:
        assert await output_file.read() == b"openai-audio"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["kittentts", "piper"])
async def test_missing_local_providers_fail_clearly(provider, monkeypatch):
    async def config():
        return {"provider": provider}

    async def unavailable():
        return False

    monkeypatch.setattr(tts_tool, "_load_tts_config", config)
    monkeypatch.setattr(
        tts_tool,
        f"_check_{provider}_available",
        unavailable,
    )
    result = json.loads(await tts_tool.text_to_speech_tool("hello"))
    assert result["success"] is False
    assert "not installed" in result["error"]
