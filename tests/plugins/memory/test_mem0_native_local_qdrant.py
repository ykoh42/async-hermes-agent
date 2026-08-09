"""E2E tests for the native-async embedded Qdrant subprocess."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import uuid

import pytest
from blockbuster import BlockBuster
from qdrant_client import models

from plugins.memory.mem0 import _native_local_qdrant
from plugins.memory.mem0 import _native_memory
from plugins.memory.mem0 import _local_qdrant_worker
from plugins.memory.mem0._backend import OSSBackend
from plugins.memory.mem0._native_local_qdrant import NativeLocalQdrantClient
from plugins.memory.mem0._native_memory import Memory
from plugins.memory.mem0._native_vector import Qdrant

_FIRST_ID = "a98ba090-e2e5-4fe3-9ca8-bfa3fcb91df3"
_SECOND_ID = "9225f97d-a770-45fe-9e9a-65f027e75446"


class _Embedding:
    def __init__(self, config):
        self.closed = False

    async def embed(self, text, memory_action=None):
        return [float(len(text)), 1.0]

    async def close(self):
        self.closed = True


class _LLM:
    def __init__(self, config):
        self.closed = False

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_embedded_qdrant_crud_runs_through_owned_subprocess(tmp_path):
    path = tmp_path / "qdrant"
    store = Qdrant(
        {
            "path": str(path),
            "collection_name": "mem0",
            "embedding_model_dims": 2,
        }
    )
    ticks = 0
    stop = asyncio.Event()

    async def ticker():
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            await asyncio.sleep(0)

    ticker_task = asyncio.create_task(ticker())
    assert not path.exists()
    try:
        await store.insert(
            [[0.1, 0.2], [0.8, 0.2]],
            payloads=[
                {"data": "first", "user_id": "u1"},
                {"data": "second", "user_id": "u2"},
            ],
            ids=[_FIRST_ID, _SECOND_ID],
        )
    finally:
        stop.set()
        await ticker_task
    results = await store.search(
        "first",
        [0.1, 0.2],
        filters={"user_id": "u1"},
    )
    batches = await store.search_batch(
        ["first", "second"],
        [[0.1, 0.2], [0.8, 0.2]],
        filters={"user_id": "u1"},
    )
    record = await store.get(_FIRST_ID)
    listed = await store.list(filters={"user_id": "u1"})
    collections = await store.list_cols()
    info = await store.col_info()
    await store.update(
        _FIRST_ID,
        vector=[0.2, 0.3],
        payload={"data": "updated", "user_id": "u1"},
    )
    await store.delete(_SECOND_ID)

    assert path.exists()
    assert [result.id for result in results] == [_FIRST_ID]
    assert all(isinstance(result, models.ScoredPoint) for result in results)
    assert len(batches) == 2
    assert all(
        isinstance(result, models.ScoredPoint)
        for batch in batches
        for result in batch
    )
    assert isinstance(record, models.Record)
    assert record.payload["data"] == "first"
    assert isinstance(listed, tuple)
    assert isinstance(listed[0][0], models.Record)
    assert collections.collections[0].name == "mem0"
    assert info.points_count == 2

    client = store._client
    assert isinstance(client, NativeLocalQdrantClient)
    process = client._worker._process
    assert process is not None
    await store.close()

    assert ticks > 0
    assert process.returncode == 0


@pytest.mark.asyncio
async def test_embedded_qdrant_reset_recreates_empty_hybrid_collection(tmp_path):
    store = Qdrant(
        {
            "path": str(tmp_path / "qdrant"),
            "collection_name": "mem0",
            "embedding_model_dims": 2,
        }
    )
    await store.insert(
        [[0.1, 0.2]],
        payloads=[{"data": "first", "user_id": "u1"}],
        ids=[_FIRST_ID],
    )

    await store.reset()

    assert store.has_bm25_slot is True
    assert await store.get(_FIRST_ID) is None
    assert (await store.col_info()).points_count == 0
    await store.close()


@pytest.mark.asyncio
async def test_embedded_qdrant_normalizes_upstream_payload_value_types(tmp_path):
    vector_id = uuid.UUID(_FIRST_ID)
    store = Qdrant(
        {
            "path": tmp_path / "qdrant",
            "collection_name": "mem0",
            "embedding_model_dims": 2,
        }
    )
    await store.insert(
        [[0.1, 0.2]],
        payloads=[{"data": "first"}],
        ids=[vector_id],
    )

    await store.update(
        vector_id,
        payload={
            "created_at": datetime(2026, 8, 9, tzinfo=timezone.utc),
            "source_id": vector_id,
            "source_path": Path("/tmp/source"),
        },
    )
    record = await store.get(vector_id)

    assert record.payload == {
        "data": "first",
        "created_at": "2026-08-09T00:00:00Z",
        "source_id": _FIRST_ID,
        "source_path": "/tmp/source",
    }
    await store.delete(vector_id)
    assert await store.get(vector_id) is None
    await store.close()


@pytest.mark.asyncio
async def test_embedded_qdrant_preserves_builtin_client_error_type(tmp_path):
    store = Qdrant(
        {
            "path": tmp_path / "qdrant",
            "collection_name": "mem0",
            "embedding_model_dims": 2,
        }
    )
    await store.delete_col()

    with pytest.raises(ValueError, match="Collection mem0 not found"):
        await store.col_info()

    await store.close()


@pytest.mark.asyncio
async def test_embedded_qdrant_constructor_and_close_are_state_only(tmp_path):
    path = tmp_path / "qdrant"
    store = Qdrant(
        {
            "path": str(path),
            "collection_name": "mem0",
            "embedding_model_dims": 2,
        }
    )

    assert not path.exists()
    await store.close()

    assert not path.exists()


@pytest.mark.asyncio
async def test_embedded_dimension_probe_recreates_mismatched_collection(tmp_path):
    path = tmp_path / "qdrant"
    original = Qdrant(
        {
            "path": str(path),
            "collection_name": "mem0",
            "embedding_model_dims": 2,
        }
    )
    await original.insert(
        [[0.1, 0.2]],
        payloads=[{"data": "old"}],
        ids=[_FIRST_ID],
    )
    await original.close()

    await OSSBackend._recreate_collection_if_dims_changed(
        "qdrant",
        {
            "path": str(path),
            "collection_name": "mem0",
        },
        3,
    )

    replacement = Qdrant(
        {
            "path": str(path),
            "collection_name": "mem0",
            "embedding_model_dims": 3,
        }
    )
    await replacement._initialize()
    info = await replacement.col_info()

    assert info.config.params.vectors.size == 3
    assert await replacement.get(_FIRST_ID) is None
    await replacement.close()


@pytest.mark.asyncio
async def test_memory_add_search_delete_runs_on_default_embedded_store(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(_native_memory, "OpenAIEmbedding", _Embedding)
    monkeypatch.setattr(_native_memory, "OpenAILLM", _LLM)
    memory = Memory(
        {
            "embedder": {"provider": "openai", "config": {}},
            "llm": {"provider": "openai", "config": {}},
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": str(tmp_path / "qdrant"),
                    "collection_name": "mem0",
                    "embedding_model_dims": 2,
                },
            },
            "history_db_path": str(tmp_path / "history.db"),
            "version": "v1.1",
        }
    )

    added = await memory.add("green tea", user_id="u1", infer=False)
    found = await memory.search("green tea", filters={"user_id": "u1"})
    memory_id = added["results"][0]["id"]
    deleted = await memory.delete(memory_id)

    assert added["results"][0]["memory"] == "green tea"
    assert found["results"][0]["id"] == memory_id
    assert found["results"][0]["memory"] == "green tea"
    assert deleted == {"message": "Memory deleted successfully!"}
    await memory.close()


def test_embedded_qdrant_worker_path_is_installed_and_absolute():
    worker = Path(_native_local_qdrant.__file__).with_name(
        "_local_qdrant_worker.py"
    )

    assert worker.is_absolute()
    assert worker.is_file()


def test_embedded_worker_reinitializes_after_process_state_reset(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(_local_qdrant_worker, "_client", None)
    request = {
        "operation": "get_collections",
        "path": str(tmp_path / "qdrant"),
    }
    try:
        first = _local_qdrant_worker._execute(request)
        first_client = _local_qdrant_worker._client
        first_client.close()
        _local_qdrant_worker._client = None
        second = _local_qdrant_worker._execute(request)

        assert first == {"collections": []}
        assert second == {"collections": []}
        assert _local_qdrant_worker._client is not first_client
    finally:
        if _local_qdrant_worker._client is not None:
            _local_qdrant_worker._client.close()
            _local_qdrant_worker._client = None


@pytest.mark.asyncio
async def test_embedded_qdrant_does_not_block_the_parent_event_loop(tmp_path):
    store = Qdrant(
        {
            "path": str(tmp_path / "qdrant"),
            "collection_name": "mem0",
            "embedding_model_dims": 2,
        }
    )
    blocker = BlockBuster()
    blocker.activate()
    try:
        await store.insert(
            [[0.1, 0.2]],
            payloads=[{"data": "first"}],
            ids=[_FIRST_ID],
        )
        assert await store.get(_FIRST_ID) is not None
        await store.close()
    finally:
        blocker.deactivate()
        await store.close()
