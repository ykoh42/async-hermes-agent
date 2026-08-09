"""Parity tests for the native-async Mem0 OSS Qdrant adapter."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys

import pytest

from plugins.memory.mem0._native_vector import Qdrant


class _Model:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class _Modifier:
    IDF = "idf"


class _FakeAsyncQdrantClient:
    instances = []
    existing = False
    sparse_vectors = None
    collections_exception = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.close_calls = 0
        self.instances.append(self)

    async def get_collections(self):
        self.calls.append(("get_collections",))
        if self.collections_exception is not None:
            raise self.collections_exception
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
        return SimpleNamespace(points=[SimpleNamespace(id="m1", score=0.9)])

    async def query_batch_points(self, **kwargs):
        self.calls.append(("query_batch_points", kwargs))
        return [SimpleNamespace(points=[SimpleNamespace(id="batch", score=0.8)])]

    async def delete(self, **kwargs):
        self.calls.append(("delete", kwargs))

    async def set_payload(self, **kwargs):
        self.calls.append(("set_payload", kwargs))

    async def update_vectors(self, **kwargs):
        self.calls.append(("update_vectors", kwargs))

    async def retrieve(self, **kwargs):
        self.calls.append(("retrieve", kwargs))
        return [SimpleNamespace(id="m1", payload={"data": "fact"})]

    async def scroll(self, **kwargs):
        self.calls.append(("scroll", kwargs))
        return ([SimpleNamespace(id="m1")], "next")

    async def delete_collection(self, **kwargs):
        self.calls.append(("delete_collection", kwargs))

    async def close(self):
        self.close_calls += 1


@pytest.fixture(autouse=True)
def _fake_qdrant_modules(monkeypatch):
    _FakeAsyncQdrantClient.instances.clear()
    _FakeAsyncQdrantClient.existing = False
    _FakeAsyncQdrantClient.sparse_vectors = None
    _FakeAsyncQdrantClient.collections_exception = None

    models = ModuleType("qdrant_client.models")
    for name in (
        "PointIdsList",
        "PointStruct",
        "PointVectors",
        "QueryRequest",
        "SparseVectorParams",
        "VectorParams",
        "DatetimeRange",
        "FieldCondition",
        "Filter",
        "MatchAny",
        "MatchExcept",
        "MatchText",
        "MatchValue",
        "Range",
    ):
        setattr(models, name, type(name, (_Model,), {}))
    models.Distance = SimpleNamespace(COSINE="cosine")
    models.Modifier = _Modifier

    qdrant_client = ModuleType("qdrant_client")
    qdrant_client.AsyncQdrantClient = _FakeAsyncQdrantClient
    qdrant_client.models = models
    monkeypatch.setitem(sys.modules, "qdrant_client", qdrant_client)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", models)

@pytest.mark.asyncio
async def test_qdrant_is_state_only_and_disables_compatibility_thread():
    store = Qdrant(
        {
            "url": "https://qdrant.test",
            "api_key": "secret",
            "collection_name": "mem0",
            "embedding_model_dims": 3,
        }
    )

    assert _FakeAsyncQdrantClient.instances == []
    assert await store.search("query", [1.0, 0.0, 0.0], top_k=4) == [
        SimpleNamespace(id="m1", score=0.9)
    ]

    client = _FakeAsyncQdrantClient.instances[0]
    assert client.kwargs == {
        "url": "https://qdrant.test",
        "api_key": "secret",
        "check_compatibility": False,
    }
    create = next(call[1] for call in client.calls if call[0] == "create_collection")
    assert create["collection_name"] == "mem0"
    assert create["vectors_config"].kwargs == {
        "size": 3,
        "distance": "cosine",
        "on_disk": False,
    }
    assert create["sparse_vectors_config"]["bm25"].kwargs == {"modifier": "idf"}
    await store.close()


@pytest.mark.asyncio
async def test_qdrant_remote_options_take_precedence_over_embedded_path():
    store = Qdrant(
        {
            "path": "/ignored/local/path",
            "url": "https://qdrant.test",
            "host": "qdrant.internal",
            "port": 6333,
            "api_key": "secret",
            "https": False,
        }
    )

    await store.get("m1")

    assert _FakeAsyncQdrantClient.instances[0].kwargs == {
        "url": "https://qdrant.test",
        "host": "qdrant.internal",
        "port": 6333,
        "api_key": "secret",
        "https": False,
        "check_compatibility": False,
    }
    await store.close()


@pytest.mark.asyncio
async def test_qdrant_none_path_uses_native_async_default_transport():
    store = Qdrant({"path": None})

    await store.get("m1")

    client = _FakeAsyncQdrantClient.instances[0]
    assert client.kwargs == {"check_compatibility": False}
    assert store._is_local is True
    assert not any(call[0] == "create_payload_index" for call in client.calls)
    await store.close()


@pytest.mark.asyncio
async def test_qdrant_uses_configured_native_async_client():
    client = _FakeAsyncQdrantClient(source="configured")
    store = Qdrant({"client": client, "path": "/ignored/local/path"})

    await store.get("m1")

    assert store._client is client
    assert client.kwargs == {"source": "configured"}
    await store.close()
    assert client.close_calls == 1


@pytest.mark.asyncio
async def test_qdrant_rejects_configured_synchronous_client():
    class SyncClient:
        def get_collections(self):
            raise AssertionError("synchronous client must not be called")

    store = Qdrant({"client": SyncClient()})

    with pytest.raises(RuntimeError, match="native async configured client"):
        await store.get("m1")

    await store.close()


@pytest.mark.asyncio
async def test_qdrant_existing_collection_keeps_sparse_slot_state():
    _FakeAsyncQdrantClient.existing = True
    _FakeAsyncQdrantClient.sparse_vectors = {"bm25": object()}
    store = Qdrant(
        {"url": "https://qdrant.test", "collection_name": "mem0", "embedding_model_dims": 3}
    )

    await store.get("m1")

    client = _FakeAsyncQdrantClient.instances[0]
    assert not any(call[0] == "create_collection" for call in client.calls)
    assert store.has_bm25_slot is True
    await store.close()


@pytest.mark.asyncio
async def test_qdrant_insert_preserves_named_dense_vector_and_payload():
    store = Qdrant(
        {"url": "https://qdrant.test", "collection_name": "mem0", "embedding_model_dims": 2}
    )

    await store.insert(
        vectors=[[0.1, 0.2]],
        ids=["m1"],
        payloads=[{"data": "fact"}],
    )

    request = next(
        call[1] for call in _FakeAsyncQdrantClient.instances[0].calls if call[0] == "upsert"
    )
    assert request["collection_name"] == "mem0"
    assert request["points"][0].kwargs == {
        "id": "m1",
        "vector": {"": [0.1, 0.2]},
        "payload": {"data": "fact"},
    }
    await store.close()


@pytest.mark.asyncio
async def test_qdrant_search_builds_pinned_mem0_filter_without_runtime_import():
    store = Qdrant(
        {"url": "https://qdrant.test", "collection_name": "mem0", "embedding_model_dims": 2}
    )

    await store.search(
        "query",
        [0.1, 0.2],
        top_k=7,
        filters={"user_id": "u1"},
    )

    request = [
        call[1]
        for call in _FakeAsyncQdrantClient.instances[0].calls
        if call[0] == "query_points"
    ][0]
    assert request["collection_name"] == "mem0"
    assert request["query"] == [0.1, 0.2]
    assert request["limit"] == 7
    condition = request["query_filter"].kwargs["must"][0]
    assert request["query_filter"].kwargs == {
        "must": [condition],
        "should": None,
        "must_not": None,
    }
    assert condition.kwargs["key"] == "user_id"
    assert condition.kwargs["match"].kwargs == {"value": "u1"}
    await store.close()


@pytest.mark.asyncio
async def test_qdrant_crud_and_list_preserve_upstream_shapes():
    store = Qdrant(
        {"url": "https://qdrant.test", "collection_name": "mem0", "embedding_model_dims": 2}
    )

    memory = await store.get("m1")
    listed = await store.list(filters={"user_id": "u1"}, top_k=9)
    await store.update("m1", vector=[0.2, 0.3], payload={"data": "new"})
    await store.delete("m1")
    await store.delete_col()

    assert memory.payload == {"data": "fact"}
    assert listed[1] == "next"
    names = [call[0] for call in _FakeAsyncQdrantClient.instances[0].calls]
    assert "retrieve" in names
    assert "scroll" in names
    assert "upsert" in names
    assert "delete" in names
    assert "delete_collection" in names
    await store.close()


@pytest.mark.asyncio
async def test_qdrant_reset_deletes_and_recreates_hybrid_collection():
    store = Qdrant(
        {"url": "https://qdrant.test", "collection_name": "mem0", "embedding_model_dims": 2}
    )
    await store.get("m1")

    await store.reset()

    client = _FakeAsyncQdrantClient.instances[0]
    names = [call[0] for call in client.calls]
    assert names.count("create_collection") == 2
    assert names.count("delete_collection") == 1
    assert store.has_bm25_slot is True
    await store.close()


@pytest.mark.asyncio
async def test_qdrant_close_is_idempotent():
    store = Qdrant(
        {"url": "https://qdrant.test", "collection_name": "mem0", "embedding_model_dims": 2}
    )
    await store.get("m1")
    client = _FakeAsyncQdrantClient.instances[0]

    await store.close()
    await store.close()

    assert client.close_calls == 1


@pytest.mark.asyncio
async def test_qdrant_initialization_failure_closes_owned_client():
    _FakeAsyncQdrantClient.collections_exception = RuntimeError("probe failed")
    store = Qdrant(
        {"url": "https://qdrant.test", "collection_name": "mem0", "embedding_model_dims": 2}
    )

    with pytest.raises(RuntimeError, match="probe failed"):
        await store.get("m1")

    assert _FakeAsyncQdrantClient.instances[0].close_calls == 1
    await store.close()
