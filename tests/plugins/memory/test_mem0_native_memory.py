"""Vertical parity tests for the native Mem0 OSS memory runtime."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
import hashlib
import json
from types import SimpleNamespace

import pytest

from plugins.memory.mem0 import _native_memory
from plugins.memory.mem0._native_memory import Memory
from plugins.memory.mem0._native_oss import SQLiteManager as NativeSQLiteManager
from plugins.memory.mem0._native_prompts import (
    ADDITIVE_EXTRACTION_PROMPT,
    AGENT_CONTEXT_SUFFIX,
    PROCEDURAL_MEMORY_SYSTEM_PROMPT,
)


class _FakeEmbedder:
    instances = []
    batch_exception = None

    def __init__(self, config):
        self.config = config
        self.calls = []
        self.closed = False
        self.instances.append(self)

    async def embed(self, text, memory_action=None):
        self.calls.append((text, memory_action))
        return [float(len(text)), 1.0]

    async def embed_batch(self, texts, memory_action=None):
        self.calls.append(("batch", list(texts), memory_action))
        if self.batch_exception is not None:
            raise self.batch_exception
        return [[float(len(text)), 1.0] for text in texts]

    async def close(self):
        self.closed = True


class _FakeLLM:
    instances = []
    response = '{"memory": []}'
    exception = None

    def __init__(self, config):
        self.config = config
        self.calls = []
        self.closed = False
        self.instances.append(self)

    async def generate_response(self, **kwargs):
        self.calls.append(kwargs)
        if self.exception is not None:
            raise self.exception
        return self.response

    async def close(self):
        self.closed = True


class _FakeNLP:
    instances = []
    entities = {}
    lemmas = {}

    def __init__(self):
        self.calls = []
        self.closed = False
        self.instances.append(self)

    async def lemmatize(self, text):
        self.calls.append(("lemmatize", text))
        return self.lemmas.get(text, text)

    async def extract(self, text):
        self.calls.append(("extract", text))
        return list(self.entities.get(text, []))

    async def extract_batch(self, texts):
        self.calls.append(("extract_batch", list(texts)))
        return [list(self.entities.get(text, [])) for text in texts]

    async def close(self):
        self.closed = True


class _FakeVector:
    instances = []
    insert_failures = 0

    def __init__(self, config):
        self.config = config
        self.calls = []
        self.rows = {}
        self.keyword_rows = []
        self.closed = False
        self.instances.append(self)

    async def _initialize(self):
        self.calls.append(("initialize",))

    async def _get_client(self):
        return self

    async def insert(self, vectors, payloads=None, ids=None):
        self.calls.append(("insert", vectors, payloads, ids))
        if self.insert_failures:
            type(self).insert_failures -= 1
            raise RuntimeError("batch insert failed")
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
        return list(self.keyword_rows)[:top_k] if self.keyword_rows else None

    async def search_batch(self, queries, vectors_list, top_k=1, filters=None):
        self.calls.append(
            ("search_batch", queries, vectors_list, top_k, filters)
        )
        return [
            await self.search(query, vector, top_k=top_k, filters=filters)
            for query, vector in zip(queries, vectors_list, strict=True)
        ]

    async def list(self, filters=None, top_k=100):
        self.calls.append(("list", filters, top_k))
        rows = [
            row
            for row in self.rows.values()
            if all(
                row.payload.get(key) == value
                for key, value in (filters or {}).items()
            )
        ][:top_k]
        return [rows]

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

    async def reset(self):
        self.calls.append(("reset",))
        self.rows.clear()

    async def close(self):
        self.closed = True


class _FakeDB:
    instances = []
    batch_history_exception = None

    def __init__(self, path):
        self.path = path
        self.history = []
        self.last_messages = []
        self.saved_messages = []
        self.batch_history_calls = []
        self.closed = False
        self.instances.append(self)

    async def _initialize(self):
        self.history.append(("initialize",))

    async def add_history(self, memory_id, old_memory, new_memory, event, **kwargs):
        self.history.append(
            (memory_id, old_memory, new_memory, event, kwargs)
        )

    async def batch_add_history(self, records):
        self.batch_history_calls.append(records)
        if self.batch_history_exception is not None:
            raise self.batch_history_exception

    async def get_history(self, memory_id):
        return [row for row in self.history if row[0] == memory_id]

    async def get_last_messages(self, session_scope, limit=10):
        assert limit == 10
        return list(self.last_messages)

    async def save_messages(self, messages, session_scope):
        self.saved_messages.append((messages, session_scope))

    async def reset(self):
        self.history.clear()
        self.last_messages.clear()
        self.saved_messages.clear()

    async def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _fake_components(monkeypatch):
    for component in (_FakeEmbedder, _FakeLLM, _FakeNLP, _FakeVector, _FakeDB):
        component.instances.clear()
    _FakeLLM.response = '{"memory": []}'
    _FakeLLM.exception = None
    _FakeEmbedder.batch_exception = None
    _FakeNLP.entities = {}
    _FakeNLP.lemmas = {}
    _FakeVector.insert_failures = 0
    _FakeDB.batch_history_exception = None
    monkeypatch.setattr(_native_memory, "OpenAIEmbedding", _FakeEmbedder)
    monkeypatch.setattr(_native_memory, "OllamaEmbedding", _FakeEmbedder)
    monkeypatch.setattr(_native_memory, "OpenAILLM", _FakeLLM)
    monkeypatch.setattr(_native_memory, "OllamaLLM", _FakeLLM)
    monkeypatch.setattr(_native_memory, "NativeNLP", _FakeNLP)
    monkeypatch.setattr(_native_memory, "Qdrant", _FakeVector)
    monkeypatch.setattr(_native_memory, "PGVector", _FakeVector)
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
async def test_memory_initializes_native_pgvector_provider(tmp_path):
    config = _config(tmp_path)
    config["vector_store"] = {
        "provider": "pgvector",
        "config": {
            "connection_string": "postgresql://db.test/memory",
            "collection_name": "mem0",
            "embedding_model_dims": 2,
        },
    }
    memory = Memory(config)

    await memory.initialize()

    assert _FakeVector.instances[0].config == config["vector_store"]["config"]
    assert _FakeVector.instances[0].calls == [("initialize",)]
    await memory.close()


@pytest.mark.asyncio
async def test_memory_preserves_injected_pgvector_pool_without_deepcopy(tmp_path):
    class NonCopyablePool:
        def __deepcopy__(self, memo):
            raise TypeError("pool contains event-loop locks")

    pool = NonCopyablePool()
    config = _config(tmp_path)
    config["vector_store"] = {
        "provider": "pgvector",
        "config": {
            "connection_pool": pool,
            "collection_name": "mem0",
            "embedding_model_dims": 2,
        },
    }
    memory = Memory(config)

    await memory.initialize()

    assert _FakeVector.instances[0].config["connection_pool"] is pool
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
async def test_memory_add_preserves_upstream_vision_message_normalization(tmp_path):
    memory = Memory(_config(tmp_path))
    result = await memory.add(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "image_url", "image_url": {"url": "image"}},
                    {"type": "text", "text": "second"},
                ],
            }
        ],
        user_id="u1",
        infer=False,
    )

    assert result["results"][0]["memory"] == "first second"
    assert _FakeEmbedder.instances[0].calls[-1] == ("first second", "add")
    await memory.close()


@pytest.mark.asyncio
async def test_memory_add_awaits_native_llm_for_enabled_vision(tmp_path):
    config = _config(tmp_path)
    config["llm"]["config"]["enable_vision"] = True
    config["llm"]["config"]["vision_details"] = "high"
    memory = Memory(config)
    _FakeLLM.response = "image description"
    message = {
        "role": "user",
        "content": {
            "type": "image_url",
            "image_url": {"url": "https://image.test/example.png"},
        },
    }

    result = await memory.add([message], user_id="u1", infer=False)

    assert result["results"][0]["memory"] == "image description"
    llm_messages = _FakeLLM.instances[0].calls[0]["messages"]
    assert llm_messages[0]["content"][1] == {
        "type": "image_url",
        "image_url": {
            "url": "https://image.test/example.png",
            "detail": "high",
        },
    }
    await memory.close()


@pytest.mark.asyncio
async def test_memory_infer_true_preserves_phased_add_pipeline(tmp_path):
    memory = Memory(_config(tmp_path))
    await memory.initialize()
    vector = _FakeVector.instances[0]
    existing_text = "already stored"
    existing_id = "existing-id"
    vector.rows[existing_id] = SimpleNamespace(
        id=existing_id,
        score=0.92,
        vector=[1.0, 1.0],
        payload={
            "data": existing_text,
            "hash": hashlib.md5(existing_text.encode()).hexdigest(),
            "user_id": "u1",
            "agent_id": "hermes",
        },
    )
    database = _FakeDB.instances[0]
    database.last_messages = [
        {"role": "user", "content": "prior context", "name": None}
    ]
    _FakeLLM.response = json.dumps(
        {
            "memory": [
                {"text": existing_text, "attributed_to": "user"},
                {"text": "plans a Seoul trip", "attributed_to": "user"},
                {"text": "plans a Seoul trip", "attributed_to": "user"},
                {"text": "was recommended tea", "attributed_to": "assistant"},
            ]
        }
    )
    messages = [
        {"role": "user", "content": "I plan to visit Seoul"},
        {"role": "assistant", "content": "Try the tea houses"},
    ]

    result = await memory.add(
        messages,
        user_id="u1",
        agent_id="hermes",
        infer=True,
        metadata={"source": "conversation"},
        prompt="Keep travel detail",
    )

    assert [item["memory"] for item in result["results"]] == [
        "plans a Seoul trip",
        "was recommended tea",
    ]
    search_call = next(call for call in vector.calls if call[0] == "search")
    assert search_call[1] == (
        "user: I plan to visit Seoul\nassistant: Try the tea houses\n"
    )
    assert search_call[3:] == (10, {"user_id": "u1", "agent_id": "hermes"})
    llm_call = _FakeLLM.instances[0].calls[0]
    assert llm_call["response_format"] == {"type": "json_object"}
    assert llm_call["messages"][0]["content"] == ADDITIVE_EXTRACTION_PROMPT
    assert "Keep travel detail" in llm_call["messages"][1]["content"]
    assert '"id": "0", "text": "already stored"' in (
        llm_call["messages"][1]["content"]
    )
    embedder_calls = _FakeEmbedder.instances[0].calls
    assert embedder_calls[0] == (search_call[1], "search")
    assert embedder_calls[1] == (
        "batch",
        [
            existing_text,
            "plans a Seoul trip",
            "plans a Seoul trip",
            "was recommended tea",
        ],
        "add",
    )
    inserted_payloads = next(call[2] for call in vector.calls if call[0] == "insert")
    assert [payload["data"] for payload in inserted_payloads] == [
        "plans a Seoul trip",
        "was recommended tea",
    ]
    assert inserted_payloads[0]["attributed_to"] == "user"
    assert inserted_payloads[1]["attributed_to"] == "assistant"
    assert all(payload["source"] == "conversation" for payload in inserted_payloads)
    assert len(database.batch_history_calls) == 1
    assert database.saved_messages == [(messages, "agent_id=hermes&user_id=u1")]
    await memory.close()


@pytest.mark.asyncio
async def test_memory_infer_true_batch_links_entities_like_upstream(tmp_path):
    _FakeLLM.response = json.dumps(
        {
            "memory": [
                {"text": "Plans a Seoul trip"},
                {"text": "Will visit Seoul cafes"},
            ]
        }
    )
    _FakeNLP.entities = {
        "Plans a Seoul trip": [("PROPER", "Seoul")],
        "Will visit Seoul cafes": [
            ("PROPER", "Seoul"),
            ("TOPIC", "Seoul cafes"),
        ],
    }
    memory = Memory(_config(tmp_path))

    result = await memory.add("travel", user_id="u1", infer=True)

    main_store, entity_store = _FakeVector.instances
    assert entity_store.config["collection_name"] == "mem0_entities"
    assert len(entity_store.rows) == 2
    result_ids = {item["id"] for item in result["results"]}
    seoul = next(
        row for row in entity_store.rows.values() if row.payload["data"] == "Seoul"
    )
    assert set(seoul.payload["linked_memory_ids"]) == result_ids
    assert seoul.payload["entity_type"] == "PROPER"
    assert seoul.payload["user_id"] == "u1"
    assert len(main_store.rows) == 2
    await memory.close()


@pytest.mark.asyncio
async def test_memory_search_applies_upstream_bm25_and_entity_boosts(tmp_path):
    _FakeNLP.lemmas = {"tea Seoul": "tea seoul"}
    _FakeNLP.entities = {"tea Seoul": [("PROPER", "Seoul")]}
    memory = Memory(_config(tmp_path))
    await memory.initialize()
    main_store = _FakeVector.instances[0]
    main_store.rows["memory-1"] = SimpleNamespace(
        id="memory-1",
        score=0.4,
        vector=[1.0, 1.0],
        payload={"data": "Seoul tea", "user_id": "u1"},
    )
    main_store.keyword_rows = [
        SimpleNamespace(id="memory-1", score=5.0, payload={})
    ]
    entity_store = await memory._get_entity_store()
    entity_store.rows["entity-1"] = SimpleNamespace(
        id="entity-1",
        score=0.8,
        vector=[1.0, 1.0],
        payload={
            "data": "Seoul",
            "linked_memory_ids": ["memory-1"],
            "user_id": "u1",
        },
    )

    result = await memory.search(
        "tea Seoul",
        filters={"user_id": "u1"},
        top_k=5,
        explain=True,
    )

    assert result["results"][0]["score"] == pytest.approx(0.52)
    assert result["results"][0]["score_details"] == {
        "semantic_score": 0.4,
        "bm25_score": 0.5,
        "entity_boost": pytest.approx(0.4),
        "raw_score": pytest.approx(1.3),
        "max_possible_score": 2.5,
        "final_score": pytest.approx(0.52),
        "threshold": 0.1,
    }
    keyword_call = next(
        call for call in main_store.calls if call[0] == "keyword_search"
    )
    assert keyword_call[1] == "tea seoul"
    entity_search = next(
        call for call in entity_store.calls if call[0] == "search"
    )
    assert entity_search[3:] == (500, {"user_id": "u1"})
    await memory.close()


@pytest.mark.asyncio
async def test_memory_entity_boost_preserves_external_cancellation(tmp_path):
    _FakeNLP.entities = {"Seoul": [("PROPER", "Seoul")]}
    memory = Memory(_config(tmp_path))
    await memory.initialize()
    entity_store = await memory._get_entity_store()

    async def cancelled_search(*args, **kwargs):
        raise asyncio.CancelledError

    entity_store.search = cancelled_search

    with pytest.raises(asyncio.CancelledError):
        await memory.search("Seoul", filters={"user_id": "u1"})

    await memory.close()


@pytest.mark.asyncio
async def test_memory_update_and_delete_relink_entities(tmp_path):
    _FakeNLP.entities = {"new text": [("TOPIC", "new topic")]}
    memory = Memory(_config(tmp_path))
    added = await memory.add("old text", user_id="u1", infer=False)
    memory_id = added["results"][0]["id"]
    entity_store = await memory._get_entity_store()
    entity_store.rows["old-entity"] = SimpleNamespace(
        id="old-entity",
        score=1.0,
        vector=[1.0, 1.0],
        payload={
            "data": "old topic",
            "linked_memory_ids": [memory_id],
            "user_id": "u1",
        },
    )

    await memory.update(memory_id, data="new text")

    assert "old-entity" not in entity_store.rows
    new_entity = next(iter(entity_store.rows.values()))
    assert new_entity.payload["data"] == "new topic"
    assert new_entity.payload["linked_memory_ids"] == [memory_id]

    await memory.delete(memory_id)

    assert entity_store.rows == {}
    await memory.close()


@pytest.mark.asyncio
async def test_memory_infer_true_saves_messages_for_empty_extraction(tmp_path):
    memory = Memory(_config(tmp_path))
    messages = [{"role": "user", "content": "hello"}]

    assert await memory.add(messages, user_id="u1", infer=True) == {"results": []}

    assert _FakeDB.instances[0].saved_messages == [(messages, "user_id=u1")]
    assert _FakeVector.instances[0].rows == {}
    await memory.close()


@pytest.mark.asyncio
async def test_memory_infer_true_preserves_agent_only_prompt_context(tmp_path):
    memory = Memory(_config(tmp_path))

    assert await memory.add("fact", agent_id="agent-1", infer=True) == {
        "results": []
    }

    system_prompt = _FakeLLM.instances[0].calls[0]["messages"][0]["content"]
    assert system_prompt == ADDITIVE_EXTRACTION_PROMPT + AGENT_CONTEXT_SUFFIX
    await memory.close()


@pytest.mark.asyncio
async def test_memory_infer_true_preserves_external_cancellation(tmp_path):
    memory = Memory(_config(tmp_path))
    _FakeLLM.exception = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await memory.add("fact", user_id="u1", infer=True)

    assert _FakeDB.instances[0].saved_messages == []
    assert _FakeVector.instances[0].rows == {}
    await memory.close()


@pytest.mark.asyncio
async def test_memory_infer_true_preserves_upstream_batch_fallbacks(tmp_path):
    memory = Memory(_config(tmp_path))
    _FakeLLM.response = json.dumps(
        {"memory": [{"text": "first fact"}, {"text": "second fact"}]}
    )
    _FakeEmbedder.batch_exception = RuntimeError("batch embed failed")
    _FakeVector.insert_failures = 1
    _FakeDB.batch_history_exception = RuntimeError("batch history failed")

    result = await memory.add("source message", user_id="u1", infer=True)

    assert [item["memory"] for item in result["results"]] == [
        "first fact",
        "second fact",
    ]
    embedder_calls = _FakeEmbedder.instances[0].calls
    assert embedder_calls == [
        ("user: source message\n", "search"),
        ("batch", ["first fact", "second fact"], "add"),
        ("first fact", "add"),
        ("second fact", "add"),
    ]
    insert_calls = [
        call for call in _FakeVector.instances[0].calls if call[0] == "insert"
    ]
    assert [len(call[1]) for call in insert_calls] == [2, 1, 1]
    assert [row[3] for row in _FakeDB.instances[0].history[1:]] == ["ADD", "ADD"]
    assert _FakeDB.instances[0].saved_messages == [
        ([{"role": "user", "content": "source message"}], "user_id=u1")
    ]
    await memory.close()


@pytest.mark.asyncio
async def test_memory_infer_true_llm_failure_matches_upstream_save_order(tmp_path):
    memory = Memory(_config(tmp_path))
    _FakeLLM.exception = RuntimeError("provider failed")

    assert await memory.add("fact", user_id="u1", infer=True) == {"results": []}

    assert _FakeDB.instances[0].saved_messages == []
    assert _FakeVector.instances[0].rows == {}
    await memory.close()


@pytest.mark.asyncio
async def test_memory_infer_true_never_uses_asyncio_thread_fallback(
    monkeypatch,
    tmp_path,
):
    async def forbidden(*args, **kwargs):
        raise AssertionError("asyncio.to_thread must not be used")

    monkeypatch.setattr(asyncio, "to_thread", forbidden)
    _FakeLLM.response = json.dumps({"memory": [{"text": "native async fact"}]})
    memory = Memory(_config(tmp_path))

    result = await memory.add("source", user_id="u1", infer=True)

    assert result["results"][0]["memory"] == "native async fact"
    await memory.close()


@pytest.mark.asyncio
async def test_memory_infer_true_persists_history_and_restart_context(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(_native_memory, "SQLiteManager", NativeSQLiteManager)
    config = _config(tmp_path)
    first_messages = [{"role": "user", "content": "I prefer green tea"}]
    _FakeLLM.response = json.dumps(
        {"memory": [{"text": "User prefers green tea"}]}
    )
    first_memory = Memory(config)

    first_result = await first_memory.add(
        first_messages,
        user_id="u1",
        infer=True,
    )
    memory_id = first_result["results"][0]["id"]
    history = await first_memory.db.get_history(memory_id)
    assert history[0]["new_memory"] == "User prefers green tea"
    assert history[0]["event"] == "ADD"
    await first_memory.close()

    _FakeLLM.response = '{"memory": []}'
    restarted_memory = Memory(config)
    await restarted_memory.add(
        [{"role": "user", "content": "What do I drink?"}],
        user_id="u1",
        infer=True,
    )

    extraction_prompt = _FakeLLM.instances[-1].calls[0]["messages"][1]["content"]
    assert "## Last k Messages\nuser: I prefer green tea" in extraction_prompt
    await restarted_memory.close()


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
async def test_memory_add_creates_procedural_memory_with_upstream_contract(
    tmp_path,
):
    memory = Memory(_config(tmp_path))
    metadata = {"source": "trajectory"}
    messages = [
        {"role": "user", "content": "Open the project."},
        {"role": "assistant", "content": "Opened /workspace."},
    ]
    _FakeLLM.response = "```markdown\n## Summary\nOpened /workspace.\n```"

    result = await memory.add(
        messages,
        agent_id="a1",
        metadata=metadata,
        expiration_date="2026-08-10",
        infer=False,
        memory_type="procedural_memory",
    )

    memory_id = result["results"][0]["id"]
    assert result == {
        "results": [
            {
                "id": memory_id,
                "memory": "## Summary\nOpened /workspace.",
                "event": "ADD",
            }
        ]
    }
    assert _FakeLLM.instances[0].calls == [
        {
            "messages": [
                {
                    "role": "system",
                    "content": PROCEDURAL_MEMORY_SYSTEM_PROMPT,
                },
                *messages,
                {
                    "role": "user",
                    "content": (
                        "Create procedural memory of the above conversation."
                    ),
                },
            ]
        }
    ]
    assert _FakeEmbedder.instances[0].calls == [
        ("## Summary\nOpened /workspace.", "add")
    ]
    payload = _FakeVector.instances[0].rows[memory_id].payload
    assert payload["data"] == "## Summary\nOpened /workspace."
    assert payload["source"] == "trajectory"
    assert payload["agent_id"] == "a1"
    assert payload["expiration_date"] == "2026-08-10"
    assert payload["memory_type"] == "procedural_memory"
    assert payload["hash"] == hashlib.md5(
        b"## Summary\nOpened /workspace."
    ).hexdigest()
    assert payload["text_lemmatized"] == "## Summary\nOpened /workspace."
    history = _FakeDB.instances[0].history[-1]
    assert history[0:4] == (
        memory_id,
        None,
        "## Summary\nOpened /workspace.",
        "ADD",
    )
    assert _FakeDB.instances[0].saved_messages == []
    assert metadata == {"source": "trajectory"}
    await memory.close()


@pytest.mark.asyncio
async def test_procedural_memory_without_agent_id_follows_normal_add_path(
    tmp_path,
):
    memory = Memory(_config(tmp_path))

    result = await memory.add(
        "ordinary fact",
        user_id="u1",
        infer=False,
        memory_type="procedural_memory",
    )

    memory_id = result["results"][0]["id"]
    payload = _FakeVector.instances[0].rows[memory_id].payload
    assert result["results"][0]["memory"] == "ordinary fact"
    assert "memory_type" not in payload
    assert _FakeLLM.instances[0].calls == []
    await memory.close()


@pytest.mark.asyncio
async def test_procedural_memory_uses_custom_prompt(tmp_path):
    memory = Memory(_config(tmp_path))
    _FakeLLM.response = "custom summary"

    await memory.add(
        "step",
        agent_id="a1",
        memory_type="procedural_memory",
        prompt="Custom procedural prompt",
    )

    assert _FakeLLM.instances[0].calls[0]["messages"][0] == {
        "role": "system",
        "content": "Custom procedural prompt",
    }
    await memory.close()


@pytest.mark.asyncio
async def test_procedural_memory_preserves_cancellation_without_write(tmp_path):
    memory = Memory(_config(tmp_path))

    _FakeLLM.exception = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await memory.add(
            "cancelled step",
            agent_id="a1",
            memory_type="procedural_memory",
        )

    assert _FakeVector.instances[0].rows == {}
    assert _FakeDB.instances[0].history == [("initialize",)]
    await memory.close()


@pytest.mark.asyncio
async def test_memory_get_and_get_all_preserve_upstream_shapes(tmp_path):
    memory = Memory(_config(tmp_path))
    active = await memory.add(
        "active memory",
        user_id="u1",
        metadata={"source": "trajectory"},
        infer=False,
    )
    expired = await memory.add(
        "expired memory",
        user_id="u1",
        expiration_date="2020-01-01",
        infer=False,
    )
    active_id = active["results"][0]["id"]
    expired_id = expired["results"][0]["id"]
    payload = _FakeVector.instances[0].rows[active_id].payload

    result = await memory.get(active_id)
    missing = await memory.get("missing")
    visible = await memory.get_all(filters={"user_id": " u1 "})
    all_results = await memory.get_all(
        filters={"user_id": "u1"},
        show_expired=True,
    )

    assert result == {
        "id": active_id,
        "memory": "active memory",
        "hash": payload["hash"],
        "metadata": {"source": "trajectory"},
        "score": None,
        "created_at": payload["created_at"],
        "updated_at": payload["updated_at"],
        "user_id": "u1",
        "role": "user",
    }
    assert missing is None
    assert [item["id"] for item in visible["results"]] == [active_id]
    assert "score" not in visible["results"][0]
    assert {item["id"] for item in all_results["results"]} == {
        active_id,
        expired_id,
    }
    assert _FakeVector.instances[0].calls[-2:] == [
        ("list", {"user_id": "u1"}, 80),
        ("list", {"user_id": "u1"}, 20),
    ]
    await memory.close()


@pytest.mark.asyncio
async def test_memory_get_all_preserves_upstream_validation(tmp_path):
    memory = Memory(_config(tmp_path))

    with pytest.raises(ValueError, match="Top-level entity parameters"):
        await memory.get_all(filters={"user_id": "u1"}, agent_id="a1")
    with pytest.raises(ValueError, match="top_k must be a valid integer"):
        await memory.get_all(filters={"user_id": "u1"}, top_k=True)
    with pytest.raises(ValueError, match="Must be a non-negative integer"):
        await memory.get_all(filters={"user_id": "u1"}, top_k=-1)
    with pytest.raises(ValueError, match="filters must contain at least one"):
        await memory.get_all(filters={"source": "trajectory"})

    await memory.close()


@pytest.mark.asyncio
async def test_memory_delete_all_and_history_preserve_upstream_contract(tmp_path):
    memory = Memory(_config(tmp_path))
    first = await memory.add("first", user_id="u1", infer=False)
    second = await memory.add("second", user_id="u1", infer=False)
    retained = await memory.add("retained", user_id="u2", infer=False)
    first_id = first["results"][0]["id"]
    second_id = second["results"][0]["id"]
    retained_id = retained["results"][0]["id"]

    result = await memory.delete_all(user_id=" u1 ")

    assert result == {"message": "Memories deleted successfully!"}
    assert set(_FakeVector.instances[0].rows) == {retained_id}
    assert [entry[3] for entry in await memory.history(first_id)] == [
        "ADD",
        "DELETE",
    ]
    assert [entry[3] for entry in await memory.history(second_id)] == [
        "ADD",
        "DELETE",
    ]
    with pytest.raises(ValueError, match="At least one filter is required"):
        await memory.delete_all()

    await memory.close()


@pytest.mark.asyncio
async def test_memory_delete_all_preserves_external_cancellation(tmp_path):
    memory = Memory(_config(tmp_path))
    await memory.add("first", user_id="u1", infer=False)
    await memory.add("second", user_id="u1", infer=False)
    vector = _FakeVector.instances[0]
    started = asyncio.Event()

    async def cancelled_delete(vector_id):
        started.set()
        await asyncio.Future()

    vector.delete = cancelled_delete
    task = asyncio.create_task(memory.delete_all(user_id="u1"))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(vector.rows) == 2
    assert not any(
        len(row) > 3 and row[3] == "DELETE"
        for row in _FakeDB.instances[0].history
    )
    await memory.close()


@pytest.mark.asyncio
async def test_memory_reset_clears_all_stores_and_remains_usable(tmp_path):
    memory = Memory(_config(tmp_path))
    await memory.add("before reset", user_id="u1", infer=False)
    entity_store = await memory._get_entity_store()
    entity_store.rows["entity"] = SimpleNamespace(
        id="entity",
        payload={"data": "before", "user_id": "u1"},
    )
    original_db = _FakeDB.instances[0]

    result = await memory.reset()
    after = await memory.add("after reset", user_id="u1", infer=False)

    assert result is None
    assert original_db.closed is True
    assert memory.db is _FakeDB.instances[1]
    assert _FakeVector.instances[0].calls.count(("reset",)) == 1
    assert entity_store.calls.count(("reset",)) == 1
    assert entity_store.closed is True
    assert after["results"][0]["memory"] == "after reset"
    assert len(_FakeVector.instances[0].rows) == 1
    assert await memory.history(after["results"][0]["id"])
    await memory.close()


@pytest.mark.asyncio
async def test_memory_reset_cancellation_closes_partial_replacement(
    monkeypatch,
    tmp_path,
):
    memory = Memory(_config(tmp_path))
    await memory.initialize()
    original_db = _FakeDB.instances[0]
    replacement_started = asyncio.Event()

    async def cancelled_initialize(database):
        replacement_started.set()
        await asyncio.Future()

    monkeypatch.setattr(_FakeDB, "_initialize", cancelled_initialize)
    task = asyncio.create_task(memory.reset())
    await replacement_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert original_db.closed is True
    assert _FakeDB.instances[1].closed is True
    await memory.close()


@pytest.mark.asyncio
async def test_memory_project_chat_and_from_config_match_async_surface(tmp_path):
    memory = await Memory.from_config(_config(tmp_path))

    assert memory._initialized is True

    with pytest.raises(
        ValueError,
        match="Project updates are not supported by the OSS Memory SDK",
    ):
        await memory.project.update()
    with pytest.raises(
        ValueError,
        match="decay parameter is not supported by the OSS Memory SDK",
    ):
        await memory.project.update(decay=True)
    with pytest.raises(NotImplementedError, match="Chat function not implemented"):
        await memory.chat("hello")

    await memory.close()


@pytest.mark.asyncio
async def test_procedural_memory_accepts_only_native_async_llm_override(tmp_path):
    class _ExternalLLM:
        def __init__(self):
            self.calls = []

        async def ainvoke(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(content="```markdown\nexternal summary\n```")

    external = _ExternalLLM()
    memory = Memory(_config(tmp_path))

    result = await memory.add(
        "step",
        agent_id="a1",
        memory_type="procedural_memory",
        llm=external,
    )

    assert result["results"][0]["memory"] == "external summary"
    assert external.calls[0]["input"][-1] == {
        "role": "user",
        "content": "Create procedural memory of the above conversation.",
    }
    assert _FakeLLM.instances[0].calls == []

    with pytest.raises(TypeError, match="native async ainvoke"):
        await memory.add(
            "sync step",
            agent_id="a1",
            memory_type="procedural_memory",
            llm=SimpleNamespace(invoke=lambda **kwargs: None),
        )
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
    await memory._get_entity_store()

    await memory.close()
    await memory.close()

    assert _FakeEmbedder.instances[0].closed is True
    assert _FakeLLM.instances[0].closed is True
    assert all(vector.closed for vector in _FakeVector.instances)
    assert _FakeDB.instances[0].closed is True
    assert _FakeNLP.instances[0].closed is True
    with pytest.raises(RuntimeError, match="closed Mem0 Memory"):
        await memory.add("fact", user_id="u1", infer=False)
