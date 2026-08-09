"""Parity tests for native-async Mem0 OSS model providers."""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from plugins.memory.mem0._native_oss import OllamaEmbedding, OpenAIEmbedding


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


class _FakeAsyncOpenAI:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.close_calls = 0
        self.embeddings = _FakeEmbeddings(self)
        self.instances.append(self)

    async def close(self):
        self.close_calls += 1


class _FakeAsyncOllama:
    instances = []
    models = []
    embeddings = [[1.0, 2.0]]
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
