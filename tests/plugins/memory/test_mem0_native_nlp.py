"""Lifecycle tests for the native-async Mem0 NLP subprocess boundary."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from plugins.memory.mem0 import _native_nlp
from plugins.memory.mem0._native_nlp import NativeNLP


class _FakeStdin:
    def __init__(self, process):
        self.process = process
        self.closed = False

    def write(self, data):
        request = json.loads(data)
        self.process.requests.append(request)
        operation = request["operation"]
        if operation == "close":
            self.process.returncode = 0
            self.process.wait_event.set()
            return
        results = {
            "lemmatize": "run tea",
            "extract": [["PROPER", "Seoul"]],
            "extract_batch": [
                [["PROPER", "Seoul"]],
                [["TOPIC", "green tea"]],
            ],
        }
        self.process.stdout.put_nowait(
            (json.dumps({"result": results[operation]}) + "\n").encode()
        )

    async def drain(self):
        return None

    def close(self):
        self.closed = True

    async def wait_closed(self):
        return None


class _FakeStdout:
    def __init__(self):
        self.responses = asyncio.Queue()
        self.blocked = False
        self.release = asyncio.Event()

    def put_nowait(self, data):
        self.responses.put_nowait(data)

    async def readline(self):
        if self.blocked:
            await self.release.wait()
        return await self.responses.get()


class _FakeProcess:
    def __init__(self):
        self.stdout = _FakeStdout()
        self.stdin = _FakeStdin(self)
        self.requests = []
        self.returncode = None
        self.wait_event = asyncio.Event()
        self.terminated = False

    async def wait(self):
        await self.wait_event.wait()
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15
        self.wait_event.set()


@pytest.mark.asyncio
async def test_nlp_constructor_is_state_only_and_protocol_is_ordered(monkeypatch):
    process = _FakeProcess()
    spawn_calls = []

    async def locate(module_name):
        assert module_name == "spacy"
        return Path("/fake/spacy/__init__.py"), True

    async def spawn(*args, **kwargs):
        spawn_calls.append((args, kwargs))
        return process

    monkeypatch.setattr(_native_nlp, "locate_source_module", locate)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    nlp = NativeNLP()

    assert spawn_calls == []
    assert await nlp.lemmatize("Running tea") == "run tea"
    assert await nlp.extract("Visit Seoul") == [("PROPER", "Seoul")]
    assert await nlp.extract_batch(["Seoul", "green tea"]) == [
        [("PROPER", "Seoul")],
        [("TOPIC", "green tea")],
    ]
    await nlp.close()

    assert len(spawn_calls) == 1
    worker_path = Path(spawn_calls[0][0][1])
    assert worker_path.name == "_nlp_worker.py"
    assert worker_path.is_absolute()
    assert spawn_calls[0][0][2] == "--stdio"
    assert [request["operation"] for request in process.requests] == [
        "lemmatize",
        "extract",
        "extract_batch",
        "close",
    ]
    assert process.stdin.closed is True


@pytest.mark.asyncio
async def test_nlp_absence_preserves_upstream_fallback_without_process(monkeypatch):
    async def locate(module_name):
        assert module_name == "spacy"
        return None

    async def forbidden_spawn(*args, **kwargs):
        raise AssertionError("NLP subprocess must not start without spaCy")

    monkeypatch.setattr(_native_nlp, "locate_source_module", locate)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden_spawn)
    nlp = NativeNLP()

    assert await nlp.lemmatize("Original text") == "Original text"
    assert await nlp.extract("Original text") == []
    assert await nlp.extract_batch(["one", "two"]) == [[], []]
    await nlp.close()


@pytest.mark.asyncio
async def test_nlp_request_cancellation_terminates_owned_process(monkeypatch):
    process = _FakeProcess()

    async def locate(module_name):
        return Path("/fake/spacy/__init__.py"), True

    async def spawn(*args, **kwargs):
        return process

    monkeypatch.setattr(_native_nlp, "locate_source_module", locate)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    nlp = NativeNLP()
    process.stdout.blocked = True
    request = asyncio.create_task(nlp.lemmatize("blocked"))
    while not process.requests:
        await asyncio.sleep(0)
    request.cancel()

    with pytest.raises(asyncio.CancelledError):
        await request

    assert process.terminated is True
    await nlp.close()
