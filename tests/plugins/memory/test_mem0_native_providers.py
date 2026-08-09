"""Parity tests for native-async Mem0 OSS model providers."""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from plugins.memory.mem0._native_oss import (
    OllamaEmbedding,
    OllamaLLM,
    OpenAIEmbedding,
    OpenAILLM,
)


class _FakeEmbeddings:
    def __init__(self, owner):
        self.owner = owner

    async def create(self, **kwargs):
        self.owner.calls.append(kwargs)
        data = []
        for index, value in reversed(list(enumerate(kwargs["input"]))):
            try:
                encoded = float(value)
            except ValueError:
                encoded = float(index)
            data.append(SimpleNamespace(index=index, embedding=[encoded]))
        return SimpleNamespace(data=data)


class _FakeCompletions:
    def __init__(self, owner):
        self.owner = owner

    async def create(self, **kwargs):
        self.owner.completion_calls.append(kwargs)
        return self.owner.completion_response


class _FakeAsyncOpenAI:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.completion_calls = []
        self.completion_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="answer", tool_calls=None)
                )
            ]
        )
        self.close_calls = 0
        self.embeddings = _FakeEmbeddings(self)
        self.chat = SimpleNamespace(completions=_FakeCompletions(self))
        self.instances.append(self)

    async def close(self):
        self.close_calls += 1


class _FakeAsyncOllama:
    instances = []
    models = []
    embeddings = [[1.0, 2.0]]
    chat_response = {"message": {"content": "answer", "tool_calls": []}}
    list_exception = None
    list_entered = None
    list_release = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.close_calls = 0
        self.instances.append(self)

    async def list(self):
        self.calls.append(("list",))
        if self.list_entered is not None:
            self.list_entered.set()
        if self.list_release is not None:
            await self.list_release.wait()
        if self.list_exception is not None:
            raise self.list_exception
        return {"models": list(self.models)}

    async def pull(self, model):
        self.calls.append(("pull", model))
        return {"status": "success"}

    async def embed(self, **kwargs):
        self.calls.append(("embed", kwargs))
        return {"embeddings": list(self.embeddings)}

    async def chat(self, **kwargs):
        self.calls.append(("chat", kwargs))
        return self.chat_response

    async def close(self):
        self.close_calls += 1


@pytest.fixture(autouse=True)
def _reset_fake_openai(monkeypatch):
    _FakeAsyncOpenAI.instances.clear()
    monkeypatch.setattr("openai.AsyncOpenAI", _FakeAsyncOpenAI)


@pytest.fixture(autouse=True)
def _reset_fake_ollama(monkeypatch):
    _FakeAsyncOllama.instances.clear()
    _FakeAsyncOllama.models = []
    _FakeAsyncOllama.embeddings = [[1.0, 2.0]]
    _FakeAsyncOllama.chat_response = {
        "message": {"content": "answer", "tool_calls": []}
    }
    _FakeAsyncOllama.list_exception = None
    _FakeAsyncOllama.list_entered = None
    _FakeAsyncOllama.list_release = None
    module = ModuleType("ollama")
    module.AsyncClient = _FakeAsyncOllama
    monkeypatch.setitem(sys.modules, "ollama", module)


@pytest.mark.asyncio
async def test_openai_embedder_is_state_only_until_first_await(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    embedder = OpenAIEmbedding({"model": "text-embedding-3-small"})

    assert _FakeAsyncOpenAI.instances == []

    assert await embedder.embed("42") == [42.0]
    client = _FakeAsyncOpenAI.instances[0]
    assert client.kwargs == {
        "api_key": "environment-key",
        "base_url": "https://api.openai.com/v1",
    }
    assert client.calls == [
        {
            "input": ["42"],
            "model": "text-embedding-3-small",
            "encoding_format": "float",
        }
    ]
    await embedder.close()


@pytest.mark.asyncio
async def test_openai_embedder_preserves_base_url_and_dimension_contract(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://environment.test/v1")
    embedder = OpenAIEmbedding(
        {
            "api_key": "config-key",
            "model": "custom-embedding",
            "embedding_dims": 64,
            "openai_base_url": "https://config.test/v1",
        }
    )

    assert await embedder.embed("line one\nline two") == [0.0]

    client = _FakeAsyncOpenAI.instances[0]
    assert client.kwargs == {
        "api_key": "config-key",
        "base_url": "https://config.test/v1",
    }
    assert client.calls == [
        {
            "input": ["line one line two"],
            "model": "custom-embedding",
            "encoding_format": "float",
            "dimensions": 64,
        }
    ]
    await embedder.close()


@pytest.mark.asyncio
async def test_openai_embedder_keeps_deprecated_api_base_precedence(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_API_BASE", "https://legacy.test/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://new.test/v1")
    embedder = OpenAIEmbedding({})

    with pytest.warns(DeprecationWarning, match="OPENAI_API_BASE"):
        await embedder.embed("1")

    assert _FakeAsyncOpenAI.instances[0].kwargs["base_url"] == (
        "https://legacy.test/v1"
    )
    await embedder.close()


@pytest.mark.asyncio
async def test_openai_embedder_batches_one_hundred_and_restores_index_order(
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    embedder = OpenAIEmbedding({"model": "embedding-model"})
    texts = [str(index) for index in range(205)]

    embeddings = await embedder.embed_batch(texts)

    client = _FakeAsyncOpenAI.instances[0]
    assert [len(call["input"]) for call in client.calls] == [100, 100, 5]
    assert embeddings == [[float(index)] for index in range(205)]
    await embedder.close()


@pytest.mark.asyncio
async def test_openai_embedder_concurrent_first_calls_share_one_client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    embedder = OpenAIEmbedding({})

    assert await asyncio.gather(embedder.embed("1"), embedder.embed("2")) == [
        [1.0],
        [2.0],
    ]
    assert len(_FakeAsyncOpenAI.instances) == 1
    await embedder.close()


@pytest.mark.asyncio
async def test_openai_embedder_close_is_idempotent(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    embedder = OpenAIEmbedding({})
    await embedder.embed("1")
    client = _FakeAsyncOpenAI.instances[0]

    await embedder.close()
    await embedder.close()

    assert client.close_calls == 1


@pytest.mark.asyncio
async def test_openai_embedder_close_finishes_before_reraising_cancellation(
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    embedder = OpenAIEmbedding({})
    await embedder.embed("1")
    client = _FakeAsyncOpenAI.instances[0]
    entered = asyncio.Event()
    release = asyncio.Event()

    async def close():
        entered.set()
        await release.wait()
        client.close_calls += 1

    client.close = close
    task = asyncio.create_task(embedder.close())
    await entered.wait()

    task.cancel()
    await asyncio.sleep(0)
    assert task.done() is False
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.close_calls == 1


@pytest.mark.asyncio
async def test_ollama_embedder_checks_model_before_first_embedding():
    _FakeAsyncOllama.models = [{"name": "nomic-embed-text:latest"}]
    embedder = OllamaEmbedding(
        {
            "model": "nomic-embed-text",
            "ollama_base_url": "http://ollama.test:11434",
        }
    )

    assert _FakeAsyncOllama.instances == []
    assert await embedder.embed("fact") == [1.0, 2.0]

    client = _FakeAsyncOllama.instances[0]
    assert client.kwargs == {"host": "http://ollama.test:11434"}
    assert client.calls == [
        ("list",),
        ("embed", {"model": "nomic-embed-text", "input": "fact"}),
    ]
    await embedder.close()


@pytest.mark.asyncio
async def test_ollama_embedder_pulls_missing_model_once():
    embedder = OllamaEmbedding({"model": "custom:7b"})

    assert await embedder.embed("first") == [1.0, 2.0]
    assert await embedder.embed("second") == [1.0, 2.0]

    client = _FakeAsyncOllama.instances[0]
    assert client.calls == [
        ("list",),
        ("pull", "custom:7b"),
        ("embed", {"model": "custom:7b", "input": "first"}),
        ("embed", {"model": "custom:7b", "input": "second"}),
    ]
    await embedder.close()


@pytest.mark.asyncio
async def test_ollama_embedder_accepts_model_attribute_from_current_sdk():
    _FakeAsyncOllama.models = [SimpleNamespace(model="nomic-embed-text:latest")]
    embedder = OllamaEmbedding({})

    await embedder.embed("fact")

    assert not any(call[0] == "pull" for call in _FakeAsyncOllama.instances[0].calls)
    await embedder.close()


@pytest.mark.asyncio
async def test_ollama_embedder_rejects_empty_embedding_response():
    _FakeAsyncOllama.models = [{"model": "nomic-embed-text:latest"}]
    _FakeAsyncOllama.embeddings = []
    embedder = OllamaEmbedding({})

    with pytest.raises(ValueError, match="returned no embeddings"):
        await embedder.embed("fact")
    await embedder.close()


@pytest.mark.asyncio
async def test_ollama_embedder_batch_preserves_count_and_order():
    _FakeAsyncOllama.models = [{"model": "nomic-embed-text:latest"}]
    _FakeAsyncOllama.embeddings = [[1.0], [2.0]]
    embedder = OllamaEmbedding({})

    assert await embedder.embed_batch(["one", "two"]) == [[1.0], [2.0]]
    assert _FakeAsyncOllama.instances[0].calls[-1] == (
        "embed",
        {"model": "nomic-embed-text", "input": ["one", "two"]},
    )
    await embedder.close()


@pytest.mark.asyncio
async def test_ollama_embedder_batch_rejects_count_mismatch():
    _FakeAsyncOllama.models = [{"model": "nomic-embed-text:latest"}]
    _FakeAsyncOllama.embeddings = [[1.0]]
    embedder = OllamaEmbedding({})

    with pytest.raises(ValueError, match="returned 1 embeddings for 2 texts"):
        await embedder.embed_batch(["one", "two"])
    await embedder.close()


@pytest.mark.asyncio
async def test_ollama_embedder_concurrent_initialization_is_singleton():
    embedder = OllamaEmbedding({})

    await asyncio.gather(embedder.embed("one"), embedder.embed("two"))

    assert len(_FakeAsyncOllama.instances) == 1
    assert _FakeAsyncOllama.instances[0].calls.count(("list",)) == 1
    assert _FakeAsyncOllama.instances[0].calls.count(
        ("pull", "nomic-embed-text")
    ) == 1
    await embedder.close()


@pytest.mark.asyncio
async def test_ollama_embedder_closes_client_when_initialization_fails():
    _FakeAsyncOllama.list_exception = RuntimeError("list failed")
    embedder = OllamaEmbedding({})

    with pytest.raises(RuntimeError, match="list failed"):
        await embedder.embed("fact")

    assert _FakeAsyncOllama.instances[0].close_calls == 1
    await embedder.close()


@pytest.mark.asyncio
async def test_ollama_embedder_initialization_cancellation_cleans_up():
    entered = asyncio.Event()
    release = asyncio.Event()
    _FakeAsyncOllama.list_entered = entered
    _FakeAsyncOllama.list_release = release
    embedder = OllamaEmbedding({})
    task = asyncio.create_task(embedder.embed("fact"))
    await entered.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert _FakeAsyncOllama.instances[0].close_calls == 1
    await embedder.close()


@pytest.mark.asyncio
async def test_openai_llm_uses_gpt5_completion_token_contract(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    llm = OpenAILLM(
        {
            "model": "gpt-5-mini",
            "temperature": 0.2,
            "top_p": 0.3,
            "max_tokens": 321,
        }
    )
    messages = [{"role": "user", "content": "question"}]

    assert await llm.generate_response(messages) == "answer"

    client = _FakeAsyncOpenAI.instances[0]
    assert client.completion_calls == [
        {
            "model": "gpt-5-mini",
            "messages": messages,
            "temperature": 0.2,
            "top_p": 0.3,
            "max_completion_tokens": 321,
        }
    ]
    await llm.close()


@pytest.mark.asyncio
async def test_openai_llm_reasoning_model_drops_sampling_and_token_limit(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    llm = OpenAILLM(
        {
            "model": "o3",
            "temperature": 0.8,
            "top_p": 0.9,
            "max_tokens": 999,
            "reasoning_effort": "high",
        }
    )

    await llm.generate_response([{"role": "user", "content": "question"}])

    assert _FakeAsyncOpenAI.instances[0].completion_calls[0] == {
        "model": "o3",
        "messages": [{"role": "user", "content": "question"}],
        "reasoning_effort": "high",
    }
    await llm.close()


@pytest.mark.asyncio
async def test_openai_llm_forwards_json_tools_store_and_parses_tool_calls(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    llm = OpenAILLM({"model": "gpt-4.1-mini", "store": False})
    await llm._get_client()
    client = _FakeAsyncOpenAI.instances[0]
    client.completion_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="remember",
                                arguments='```json\n{"fact": "tea"}\n```',
                            )
                        )
                    ],
                )
            )
        ]
    )
    tools = [{"type": "function", "function": {"name": "remember"}}]
    response_format = {"type": "json_object"}

    result = await llm.generate_response(
        [{"role": "user", "content": "question"}],
        response_format=response_format,
        tools=tools,
        tool_choice="required",
    )

    assert result == {
        "content": None,
        "tool_calls": [{"name": "remember", "arguments": {"fact": "tea"}}],
    }
    request = client.completion_calls[0]
    assert request["response_format"] is response_format
    assert request["tools"] is tools
    assert request["tool_choice"] == "required"
    assert request["store"] is False
    await llm.close()


@pytest.mark.asyncio
async def test_openai_llm_preserves_openrouter_routing(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    llm = OpenAILLM(
        {
            "model": "primary",
            "models": ["fallback-one", "fallback-two"],
            "route": "fallback",
            "openrouter_base_url": "https://router.test/api/v1",
            "site_url": "https://app.test",
            "app_name": "Hermes",
        }
    )

    await llm.generate_response([{"role": "user", "content": "question"}])

    client = _FakeAsyncOpenAI.instances[0]
    assert client.kwargs == {
        "api_key": "router-key",
        "base_url": "https://router.test/api/v1",
    }
    request = client.completion_calls[0]
    assert "model" not in request
    assert request["models"] == ["fallback-one", "fallback-two"]
    assert request["route"] == "fallback"
    assert request["extra_headers"] == {
        "HTTP-Referer": "https://app.test",
        "X-Title": "Hermes",
    }
    await llm.close()


@pytest.mark.asyncio
async def test_openai_llm_awaits_native_response_callback(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    captured = []

    async def callback(llm, response, params):
        captured.append((llm, response, params))

    llm = OpenAILLM({"response_callback": callback})
    assert await llm.generate_response(
        [{"role": "user", "content": "question"}]
    ) == "answer"

    assert captured[0][0] is llm
    assert captured[0][1] is _FakeAsyncOpenAI.instances[0].completion_response
    assert captured[0][2] == _FakeAsyncOpenAI.instances[0].completion_calls[0]
    await llm.close()


@pytest.mark.asyncio
async def test_openai_llm_rejects_sync_response_callback_without_calling_it(
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    called = False

    def callback(*args):
        nonlocal called
        called = True

    llm = OpenAILLM({"response_callback": callback})

    with pytest.raises(RuntimeError, match="native-async response_callback"):
        await llm.generate_response([{"role": "user", "content": "question"}])
    assert called is False
    await llm.close()


@pytest.mark.asyncio
async def test_ollama_llm_preserves_default_request_contract():
    llm = OllamaLLM(
        {
            "model": "llama3.1:8b",
            "temperature": 0.2,
            "max_tokens": 321,
            "top_p": 0.3,
            "ollama_base_url": "http://ollama.test:11434",
        }
    )
    messages = [{"role": "user", "content": "question"}]

    assert _FakeAsyncOllama.instances == []
    assert await llm.generate_response(messages) == "answer"

    client = _FakeAsyncOllama.instances[0]
    assert client.kwargs == {"host": "http://ollama.test:11434"}
    assert client.calls == [
        (
            "chat",
            {
                "model": "llama3.1:8b",
                "messages": messages,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 321,
                    "top_p": 0.3,
                },
            },
        )
    ]
    await llm.close()


@pytest.mark.asyncio
async def test_ollama_llm_json_format_copies_and_extends_messages():
    llm = OllamaLLM({})
    messages = [{"role": "user", "content": "extract memories"}]

    await llm.generate_response(messages, response_format={"type": "json_object"})

    assert messages == [{"role": "user", "content": "extract memories"}]
    request = _FakeAsyncOllama.instances[0].calls[0][1]
    assert request["model"] == "llama3.1:70b"
    assert request["format"] == "json"
    assert request["messages"] == [
        {
            "role": "user",
            "content": "extract memories\n\nPlease respond with valid JSON only.",
        }
    ]
    await llm.close()


@pytest.mark.asyncio
async def test_ollama_llm_parses_string_and_mapping_tool_arguments():
    _FakeAsyncOllama.chat_response = {
        "message": {
            "content": None,
            "tool_calls": [
                {
                    "function": {
                        "name": "remember",
                        "arguments": 'prefix {"fact": "tea"} suffix',
                    }
                },
                {
                    "function": {
                        "name": "forget",
                        "arguments": {"id": "m1"},
                    }
                },
            ],
        }
    }
    llm = OllamaLLM({})
    tools = [{"type": "function", "function": {"name": "remember"}}]

    result = await llm.generate_response(
        [{"role": "user", "content": "question"}],
        tools=tools,
        tool_choice="required",
    )

    assert result == {
        "content": None,
        "tool_calls": [
            {"name": "remember", "arguments": {"fact": "tea"}},
            {"name": "forget", "arguments": {"id": "m1"}},
        ],
    }
    request = _FakeAsyncOllama.instances[0].calls[0][1]
    assert request["tools"] is tools
    assert "tool_choice" not in request
    await llm.close()
