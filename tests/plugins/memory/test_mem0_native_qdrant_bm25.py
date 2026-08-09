"""BM25 parity tests for the native-async Mem0 Qdrant adapter."""

from __future__ import annotations

import asyncio
from types import ModuleType, SimpleNamespace
import sys

import pytest

from plugins.memory.mem0 import _native_vector
from plugins.memory.mem0._native_vector import Qdrant


class _Model:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeClient:
    instances = []
    existing = False
    sparse_vectors = None
    query_exception = None

    def __init__(self, **kwargs):
        self.calls = []
        self.close_calls = 0
        self.instances.append(self)

    async def get_collections(self):
        collections = [SimpleNamespace(name="mem0")] if self.existing else []
        return SimpleNamespace(collections=collections)

    async def get_collection(self, collection_name):
        self.calls.append(("get_collection", collection_name))
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(sparse_vectors=self.sparse_vectors)
            )
        )

    async def create_collection(self, **kwargs):
        self.calls.append(("create_collection", kwargs))

    async def create_payload_index(self, **kwargs):
        self.calls.append(("create_payload_index", kwargs))

    async def upsert(self, **kwargs):
        self.calls.append(("upsert", kwargs))

    async def query_points(self, **kwargs):
        self.calls.append(("query_points", kwargs))
        if self.query_exception is not None:
            raise self.query_exception
        return SimpleNamespace(points=[SimpleNamespace(id="m1", score=0.9)])

    async def set_payload(self, **kwargs):
        self.calls.append(("set_payload", kwargs))

    async def update_vectors(self, **kwargs):
        self.calls.append(("update_vectors", kwargs))

    async def retrieve(self, **kwargs):
        self.calls.append(("retrieve", kwargs))
        return [SimpleNamespace(id="m1", payload={"data": "fact"})]

    async def close(self):
        self.close_calls += 1


class _FilterBuilder:
    def _create_filter(self, filters):
        return ("filter", filters)


class _FakeSparseEncoder:
    instances = []
    outcomes = []

    def __init__(self):
        self.calls = []
        self.close_calls = 0
        self.instances.append(self)

    async def encode_batch(self, texts):
        self.calls.append(list(texts))
        outcome = self.outcomes.pop(0) if self.outcomes else None
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def close(self):
        self.close_calls += 1


@pytest.fixture(autouse=True)
def _fake_qdrant_modules(monkeypatch):
    _FakeClient.instances.clear()
    _FakeClient.existing = False
    _FakeClient.sparse_vectors = None
    _FakeClient.query_exception = None
    _FakeSparseEncoder.instances.clear()
    _FakeSparseEncoder.outcomes.clear()
    monkeypatch.setattr(_native_vector, "NativeSparseEncoder", _FakeSparseEncoder)

    models = ModuleType("qdrant_client.models")
    for name in (
        "PointStruct",
        "PointVectors",
        "SparseVector",
        "SparseVectorParams",
        "VectorParams",
    ):
        setattr(models, name, type(name, (_Model,), {}))
    models.Distance = SimpleNamespace(COSINE="cosine")
    models.Modifier = SimpleNamespace(IDF="idf")

    qdrant_client = ModuleType("qdrant_client")
    qdrant_client.AsyncQdrantClient = _FakeClient
    qdrant_client.models = models
    monkeypatch.setitem(sys.modules, "qdrant_client", qdrant_client)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", models)

    mem0 = ModuleType("mem0")
    mem0.__path__ = []
    vector_stores = ModuleType("mem0.vector_stores")
    vector_stores.__path__ = []
    mem0_vector = ModuleType("mem0.vector_stores.qdrant")
    mem0_vector.Qdrant = _FilterBuilder
    monkeypatch.setitem(sys.modules, "mem0", mem0)
    monkeypatch.setitem(sys.modules, "mem0.vector_stores", vector_stores)
    monkeypatch.setitem(sys.modules, "mem0.vector_stores.qdrant", mem0_vector)


def _store() -> Qdrant:
    return Qdrant(
        {
            "url": "https://qdrant.test",
            "collection_name": "mem0",
            "embedding_model_dims": 2,
        }
    )


def _upsert_points():
    request = next(
        call[1] for call in _FakeClient.instances[0].calls if call[0] == "upsert"
    )
    return request["points"]


@pytest.mark.asyncio
async def test_insert_batches_bm25_text_and_preserves_point_order():
    _FakeSparseEncoder.outcomes.append(
        [([1, 7], [0.25, 0.75]), ([3], [0.5])]
    )
    store = _store()

    await store.insert(
        vectors=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        ids=["m1", "m2", "m3"],
        payloads=[
            {"data": "raw one", "text_lemmatized": "lemma one"},
            {"data": "raw two"},
            {"data": ""},
        ],
    )

    assert _FakeSparseEncoder.instances[0].calls == [["lemma one", "raw two"]]
    points = _upsert_points()
    assert points[0].vector["bm25"].kwargs == {
        "indices": [1, 7],
        "values": [0.25, 0.75],
    }
    assert points[1].vector["bm25"].kwargs == {
        "indices": [3],
        "values": [0.5],
    }
    assert points[2].vector == {"": [0.5, 0.6]}
    await store.close()


@pytest.mark.asyncio
async def test_insert_count_mismatch_falls_back_per_row():
    _FakeSparseEncoder.outcomes.extend(
        [
            [([99], [0.99])],
            [([1], [0.1])],
            RuntimeError("bad row"),
        ]
    )
    store = _store()

    await store.insert(
        vectors=[[0.1, 0.2], [0.3, 0.4]],
        ids=["m1", "m2"],
        payloads=[{"data": "good"}, {"data": "bad"}],
    )

    assert _FakeSparseEncoder.instances[0].calls == [
        ["good", "bad"],
        ["good"],
        ["bad"],
    ]
    points = _upsert_points()
    assert points[0].vector["bm25"].indices == [1]
    assert points[1].vector == {"": [0.3, 0.4]}
    await store.close()


@pytest.mark.asyncio
async def test_keyword_search_uses_bm25_named_sparse_vector():
    _FakeSparseEncoder.outcomes.append([([4, 8], [0.4, 0.8])])
    store = _store()

    results = await store.keyword_search(
        "green tea",
        top_k=6,
        filters={"user_id": "u1"},
    )

    assert results == [SimpleNamespace(id="m1", score=0.9)]
    request = next(
        call[1] for call in _FakeClient.instances[0].calls if call[0] == "query_points"
    )
    assert request["using"] == "bm25"
    assert request["query"].kwargs == {
        "indices": [4, 8],
        "values": [0.4, 0.8],
    }
    assert request["query_filter"] == ("filter", {"user_id": "u1"})
    assert request["limit"] == 6
    await store.close()


@pytest.mark.asyncio
async def test_legacy_collection_disables_bm25_and_warns(caplog):
    _FakeClient.existing = True
    _FakeClient.sparse_vectors = None
    store = _store()

    assert await store.keyword_search("green tea") is None

    assert _FakeSparseEncoder.instances[0].calls == []
    assert not any(call[0] == "query_points" for call in _FakeClient.instances[0].calls)
    assert "predates v3 hybrid search" in caplog.text
    assert "use a fresh collection" in caplog.text
    await store.close()


@pytest.mark.asyncio
async def test_keyword_search_preserves_upstream_failure_fallback():
    _FakeSparseEncoder.outcomes.append([([4], [0.4])])
    _FakeClient.query_exception = RuntimeError("query failed")
    store = _store()

    assert await store.keyword_search("green tea") is None
    await store.close()


@pytest.mark.asyncio
async def test_full_update_refreshes_bm25_but_partial_update_does_not():
    _FakeSparseEncoder.outcomes.append([([5], [0.5])])
    store = _store()

    await store.update(
        "m1",
        vector=[0.2, 0.3],
        payload={"data": "raw", "text_lemmatized": "lemma"},
    )
    await store.update("m1", payload={"data": "payload only"})
    await store.update("m1", vector=[0.4, 0.5])

    assert _FakeSparseEncoder.instances[0].calls == [["lemma"]]
    assert _upsert_points()[0].vector["bm25"].kwargs == {
        "indices": [5],
        "values": [0.5],
    }
    client = _FakeClient.instances[0]
    assert sum(call[0] == "set_payload" for call in client.calls) == 1
    assert sum(call[0] == "update_vectors" for call in client.calls) == 1
    await store.close()


@pytest.mark.asyncio
async def test_close_is_idempotent_before_client_initialization():
    store = _store()

    await store.close()
    await store.close()

    assert _FakeClient.instances == []
    assert _FakeSparseEncoder.instances[0].close_calls == 1


@pytest.mark.asyncio
async def test_bm25_cancellation_propagates_and_encoder_closes(monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_encode(encoder, texts):
        encoder.calls.append(list(texts))
        entered.set()
        await release.wait()

    monkeypatch.setattr(_FakeSparseEncoder, "encode_batch", blocked_encode)
    store = _store()
    insert = asyncio.create_task(
        store.insert([[0.1, 0.2]], payloads=[{"data": "blocked"}])
    )
    await entered.wait()
    insert.cancel()

    with pytest.raises(asyncio.CancelledError):
        await insert

    await store.close()
    assert _FakeSparseEncoder.instances[0].close_calls == 1


@pytest.mark.asyncio
async def test_close_cancellation_finishes_all_owned_resources(monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_close(encoder):
        encoder.close_calls += 1
        entered.set()
        await release.wait()

    monkeypatch.setattr(_FakeSparseEncoder, "close", blocked_close)
    store = _store()
    await store.get("m1")
    client = _FakeClient.instances[0]
    close = asyncio.create_task(store.close())
    await entered.wait()
    close.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await close

    assert _FakeSparseEncoder.instances[0].close_calls == 1
    assert client.close_calls == 1


@pytest.mark.asyncio
async def test_owned_resource_cancellation_is_not_swallowed(monkeypatch):
    async def cancelled_close(encoder):
        encoder.close_calls += 1
        raise asyncio.CancelledError

    monkeypatch.setattr(_FakeSparseEncoder, "close", cancelled_close)
    store = _store()
    await store.get("m1")
    client = _FakeClient.instances[0]

    with pytest.raises(asyncio.CancelledError):
        await store.close()

    assert _FakeSparseEncoder.instances[0].close_calls == 1
    assert client.close_calls == 1
