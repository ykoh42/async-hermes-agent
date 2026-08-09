"""Parity tests for native-async Mem0 OSS model providers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from plugins.memory.mem0._native_oss import OpenAIEmbedding


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


@pytest.fixture(autouse=True)
def _reset_fake_openai(monkeypatch):
    _FakeAsyncOpenAI.instances.clear()
    monkeypatch.setattr("openai.AsyncOpenAI", _FakeAsyncOpenAI)


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
