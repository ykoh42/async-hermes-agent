"""Vertical parity tests for the native Mem0 OSS memory runtime."""

from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from plugins.memory.mem0 import _native_memory
from plugins.memory.mem0._native_memory import Memory


class _FakeEmbedder:
    instances = []

    def __init__(self, config):
        self.config = config
        self.calls = []
        self.closed = False
        self.instances.append(self)

    async def embed(self, text, memory_action=None):
        self.calls.append((text, memory_action))
        return [float(len(text)), 1.0]

    async def close(self):
        self.closed = True


class _FakeLLM:
    instances = []

    def __init__(self, config):
        self.config = config
        self.closed = False
        self.instances.append(self)

    async def close(self):
        self.closed = True


class _FakeVector:
    instances = []

    def __init__(self, config):
        self.config = config
        self.calls = []
        self.rows = {}
        self.closed = False
        self.instances.append(self)

    async def _initialize(self):
        self.calls.append(("initialize",))

    async def insert(self, vectors, payloads=None, ids=None):
        self.calls.append(("insert", vectors, payloads, ids))
        for memory_id, vector, payload in zip(ids, vectors, payloads, strict=True):
            self.rows[memory_id] = SimpleNamespace(
                id=memory_id,
                score=0.9,
                vector=vector,
                payload=payload,
            )

    async def search(self, query, vectors, top_k=5, filters=None):
        self.calls.append(("search", query, vectors, top_k, filters))
        return [
            row
            for row in self.rows.values()
            if all(row.payload.get(key) == value for key, value in (filters or {}).items())
        ][:top_k]

    async def keyword_search(self, query, top_k=5, filters=None):
        self.calls.append(("keyword_search", query, top_k, filters))
        return None

    async def get(self, vector_id):
        self.calls.append(("get", vector_id))
        return self.rows.get(vector_id)

    async def update(self, vector_id, vector=None, payload=None):
        self.calls.append(("update", vector_id, vector, payload))
        row = self.rows[vector_id]
        row.vector = vector
        row.payload = payload

    async def delete(self, vector_id):
        self.calls.append(("delete", vector_id))
        self.rows.pop(vector_id)

    async def close(self):
        self.closed = True


class _FakeDB:
    instances = []

    def __init__(self, path):
        self.path = path
        self.history = []
        self.closed = False
        self.instances.append(self)

    async def _initialize(self):
        self.history.append(("initialize",))

    async def add_history(self, memory_id, old_memory, new_memory, event, **kwargs):
        self.history.append(
            (memory_id, old_memory, new_memory, event, kwargs)
        )

    async def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _fake_components(monkeypatch):
    for component in (_FakeEmbedder, _FakeLLM, _FakeVector, _FakeDB):
        component.instances.clear()
    monkeypatch.setattr(_native_memory, "OpenAIEmbedding", _FakeEmbedder)
    monkeypatch.setattr(_native_memory, "OllamaEmbedding", _FakeEmbedder)
    monkeypatch.setattr(_native_memory, "OpenAILLM", _FakeLLM)
    monkeypatch.setattr(_native_memory, "OllamaLLM", _FakeLLM)
    monkeypatch.setattr(_native_memory, "Qdrant", _FakeVector)
    monkeypatch.setattr(_native_memory, "SQLiteManager", _FakeDB)


def _config(tmp_path):
    return {
        "embedder": {
            "provider": "openai",
            "config": {"model": "text-embedding-3-small"},
        },
        "llm": {"provider": "openai", "config": {"model": "gpt-5-mini"}},
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "url": "https://qdrant.test",
                "collection_name": "mem0",
                "embedding_model_dims": 2,
            },
        },
        "history_db_path": str(tmp_path / "history.db"),
        "version": "v1.1",
    }


@pytest.mark.asyncio
async def test_memory_constructor_is_state_only(tmp_path):
    memory = Memory(_config(tmp_path))

    assert _FakeEmbedder.instances == []
    assert _FakeLLM.instances == []
    assert _FakeVector.instances == []
    assert _FakeDB.instances == []

    await memory.initialize()

    assert len(_FakeEmbedder.instances) == 1
    assert len(_FakeLLM.instances) == 1
    assert len(_FakeVector.instances) == 1
    assert len(_FakeDB.instances) == 1
    assert _FakeVector.instances[0].calls == [("initialize",)]
    assert _FakeDB.instances[0].history == [("initialize",)]
    await memory.close()


@pytest.mark.asyncio
async def test_memory_initialization_failure_closes_partial_resources(
    monkeypatch,
    tmp_path,
):
    async def fail_database_initialization(self):
        self.history.append(("initialize",))
        raise RuntimeError("database initialization failed")

    monkeypatch.setattr(_FakeDB, "_initialize", fail_database_initialization)
    memory = Memory(_config(tmp_path))

    with pytest.raises(RuntimeError, match="database initialization failed"):
        await memory.initialize()

    assert memory.vector_store is None
    assert memory.db is None
    assert _FakeVector.instances[0].closed is True
    assert _FakeDB.instances[0].closed is True
    await memory.close()


@pytest.mark.asyncio
async def test_memory_infer_false_add_preserves_result_payload_and_history(tmp_path):
    memory = Memory(_config(tmp_path))
    messages = [
        {"role": "system", "content": "ignore"},
        {"role": "user", "content": "likes tea", "name": "alice"},
        {"role": "assistant", "content": "noted"},
    ]

    result = await memory.add(
        messages,
        user_id=" u1 ",
        agent_id="hermes",
        infer=False,
        metadata={"source": "manual"},
    )

    assert [item["memory"] for item in result["results"]] == ["likes tea", "noted"]
    assert [item["role"] for item in result["results"]] == ["user", "assistant"]
    assert result["results"][0]["actor_id"] == "alice"
    vector = _FakeVector.instances[0]
    payloads = [call[2][0] for call in vector.calls if call[0] == "insert"]
    assert payloads[0]["user_id"] == "u1"
    assert payloads[0]["agent_id"] == "hermes"
    assert payloads[0]["source"] == "manual"
    assert payloads[0]["role"] == "user"
    assert payloads[0]["actor_id"] == "alice"
    assert payloads[0]["data"] == "likes tea"
    assert payloads[0]["hash"]
    assert payloads[0]["created_at"] == payloads[0]["updated_at"]
    assert payloads[0]["text_lemmatized"] == "likes tea"
    assert len(_FakeDB.instances[0].history) == 3
    await memory.close()


@pytest.mark.asyncio
async def test_memory_infer_true_fails_clearly_without_partial_write(tmp_path):
    memory = Memory(_config(tmp_path))

    with pytest.raises(RuntimeError, match="infer=True pipeline is not native async"):
        await memory.add(
            [{"role": "user", "content": "likes tea"}],
            user_id="u1",
            agent_id="hermes",
            infer=True,
        )

    assert _FakeVector.instances[0].rows == {}
    assert _FakeDB.instances[0].history == [("initialize",)]
    await memory.close()


@pytest.mark.asyncio
async def test_memory_add_preserves_temporal_and_memory_type_validation(tmp_path):
    memory = Memory(_config(tmp_path))

    with pytest.raises(ValueError, match="timestamp parameter"):
        await memory.add("fact", user_id="u1", timestamp="2026-08-09")
    with pytest.raises(ValueError, match="expiration_date must be a valid date"):
        await memory.add(
            "fact",
            user_id="u1",
            expiration_date="not-a-date",
            infer=False,
        )
    with pytest.raises(ValueError, match="Invalid 'memory_type'"):
        await memory.add(
            "fact",
            user_id="u1",
            memory_type="semantic",
            infer=False,
        )
    with pytest.raises(RuntimeError, match="procedural memory pipeline"):
        await memory.add(
            "fact",
            agent_id="a1",
            memory_type="procedural_memory",
            infer=False,
        )

    result = await memory.add(
        "fact",
        user_id="u1",
        expiration_date=datetime(2026, 8, 10, 1, 2, tzinfo=timezone.utc),
        infer=False,
    )
    payload = _FakeVector.instances[0].rows[result["results"][0]["id"]].payload
    assert payload["expiration_date"] == "2026-08-10"
    await memory.close()


@pytest.mark.asyncio
async def test_memory_search_preserves_mem0_result_shape_and_overfetch(tmp_path):
    memory = Memory(_config(tmp_path))
    added = await memory.add(
        [{"role": "user", "content": "likes tea"}],
        user_id="u1",
        agent_id="hermes",
        infer=False,
        metadata={"source": "manual"},
    )

    result = await memory.search(" tea ", filters={"user_id": " u1 "}, top_k=4)

    memory_id = added["results"][0]["id"]
    assert result == {
        "results": [
            {
                "id": memory_id,
                "memory": "likes tea",
                "hash": _FakeVector.instances[0].rows[memory_id].payload["hash"],
                "metadata": {"source": "manual"},
                "score": 0.9,
                "created_at": _FakeVector.instances[0].rows[memory_id].payload["created_at"],
                "updated_at": _FakeVector.instances[0].rows[memory_id].payload["updated_at"],
                "user_id": "u1",
                "agent_id": "hermes",
                "role": "user",
            }
        ]
    }
    search_call = next(
        call for call in _FakeVector.instances[0].calls if call[0] == "search"
    )
    assert search_call[1] == "tea"
    assert search_call[3] == 60
    assert search_call[4] == {"user_id": "u1"}
    await memory.close()


@pytest.mark.asyncio
async def test_memory_search_preserves_upstream_validation_contract(tmp_path):
    memory = Memory(_config(tmp_path))

    with pytest.raises(ValueError, match="reference_date parameter"):
        await memory.search(
            "query",
            filters={"user_id": "u1"},
            reference_date="2026-08-09",
        )
    with pytest.raises(ValueError, match="Top-level entity parameters"):
        await memory.search(
            "query",
            filters={"user_id": "u1"},
            user_id="u1",
        )
    with pytest.raises(ValueError, match="top_k must be a valid integer"):
        await memory.search("query", filters={"user_id": "u1"}, top_k=True)

    assert await memory.search(
        "query",
        filters={"user_id": "u1"},
        threshold=None,
    ) == {"results": []}
    await memory.close()


@pytest.mark.asyncio
async def test_memory_search_hides_expired_memories_by_default(tmp_path):
    memory = Memory(_config(tmp_path))
    await memory.add(
        "expired fact",
        user_id="u1",
        expiration_date="2020-01-01",
        infer=False,
    )

    assert await memory.search(
        "fact",
        filters={"user_id": "u1"},
    ) == {"results": []}
    shown = await memory.search(
        "fact",
        filters={"user_id": "u1"},
        show_expired=True,
    )
    assert shown["results"][0]["memory"] == "expired fact"
    assert shown["results"][0]["expiration_date"] == "2020-01-01"
    await memory.close()


@pytest.mark.asyncio
async def test_memory_update_and_delete_preserve_history(tmp_path):
    memory = Memory(_config(tmp_path))
    added = await memory.add(
        [{"role": "user", "content": "likes tea", "name": "alice"}],
        user_id="u1",
        agent_id="hermes",
        infer=False,
    )
    memory_id = added["results"][0]["id"]
    created_at = _FakeVector.instances[0].rows[memory_id].payload["created_at"]

    assert await memory.update(memory_id, data="likes coffee") == {
        "message": "Memory updated successfully!"
    }
    updated_payload = _FakeVector.instances[0].rows[memory_id].payload
    assert updated_payload["data"] == "likes coffee"
    assert updated_payload["created_at"] == created_at
    assert updated_payload["actor_id"] == "alice"
    assert updated_payload["updated_at"] != created_at

    assert await memory.update(
        memory_id,
        expiration_date=date(2026, 8, 10),
    ) == {"message": "Memory updated successfully!"}
    assert _FakeVector.instances[0].rows[memory_id].payload["expiration_date"] == (
        "2026-08-10"
    )

    assert await memory.delete(memory_id) == {
        "message": "Memory deleted successfully!"
    }
    assert memory_id not in _FakeVector.instances[0].rows
    assert [row[3] for row in _FakeDB.instances[0].history[1:]] == [
        "ADD",
        "UPDATE",
        "UPDATE",
        "DELETE",
    ]
    assert _FakeDB.instances[0].history[-1][4]["is_deleted"] == 1
    await memory.close()


@pytest.mark.asyncio
async def test_memory_close_releases_all_owned_components(tmp_path):
    memory = Memory(_config(tmp_path))
    await memory.initialize()

    await memory.close()
    await memory.close()

    assert _FakeEmbedder.instances[0].closed is True
    assert _FakeLLM.instances[0].closed is True
    assert _FakeVector.instances[0].closed is True
    assert _FakeDB.instances[0].closed is True
