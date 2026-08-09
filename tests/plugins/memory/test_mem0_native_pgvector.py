"""Parity tests for the native-async Mem0 PGVector adapter."""

from __future__ import annotations

import asyncio

import pytest

from plugins.memory.mem0._native_vector import (
    PGVector,
    _build_filter_conditions,
)


def _sql_text(query):
    return query.as_string(None) if hasattr(query, "as_string") else str(query)


class _CursorContext:
    def __init__(self, cursor):
        self.cursor = cursor

    async def __aenter__(self):
        return self.cursor

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _ConnectionContext:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _FakeCursor:
    def __init__(self, pool):
        self.pool = pool
        self.rows = []

    async def execute(self, query, params=None):
        statement = " ".join(_sql_text(query).split())
        self.pool.calls.append(("execute", statement, params))
        if "information_schema.tables" in statement and "COUNT" not in statement:
            self.rows = [(name,) for name in self.pool.tables]
        elif "SELECT id, vector <=>" in statement:
            self.rows = list(self.pool.semantic_rows)
        elif "ts_rank_cd" in statement:
            self.rows = list(self.pool.keyword_rows)
        elif "SELECT id, vector, payload" in statement:
            self.rows = list(self.pool.record_rows)
        elif "SELECT table_name," in statement:
            self.rows = list(self.pool.info_rows)
        elif "SELECT atttypmod FROM pg_attribute" in statement:
            self.rows = list(self.pool.dimension_rows)
        elif "pg_extension" in statement:
            self.rows = list(self.pool.extension_rows)
        else:
            self.rows = []

    async def executemany(self, query, params):
        statement = " ".join(_sql_text(query).split())
        self.pool.calls.append(("executemany", statement, list(params)))

    async def fetchall(self):
        return list(self.rows)

    async def fetchone(self):
        return self.rows[0] if self.rows else None


class _FakeConnection:
    def __init__(self, pool):
        self.pool = pool

    def cursor(self):
        return _CursorContext(_FakeCursor(self.pool))


class _FakePool:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.tables = []
        self.semantic_rows = []
        self.keyword_rows = []
        self.record_rows = []
        self.info_rows = []
        self.dimension_rows = [(3,)]
        self.extension_rows = []
        self.closed = False
        self.instances.append(self)

    async def open(self, *, wait):
        self.calls.append(("open", wait))

    def connection(self):
        return _ConnectionContext(_FakeConnection(self))

    async def close(self):
        self.closed = True
        self.calls.append(("close",))


@pytest.fixture(autouse=True)
def _fake_pool(monkeypatch):
    _FakePool.instances.clear()
    monkeypatch.setattr("psycopg_pool.AsyncConnectionPool", _FakePool)


def _config(**overrides):
    config = {
        "dbname": "postgres",
        "collection_name": "mem0",
        "embedding_model_dims": 3,
        "user": "user",
        "password": "password",
        "host": "db.test",
        "port": 5432,
        "diskann": False,
        "hnsw": True,
        "minconn": 1,
        "maxconn": 5,
    }
    config.update(overrides)
    return config


def test_pgvector_config_validation_matches_upstream():
    with pytest.raises(ValueError, match="Extra fields not allowed: invented"):
        PGVector(_config(invented=True))
    with pytest.raises(ValueError, match="'user' and 'password'"):
        PGVector({"host": "db.test", "port": 5432})
    with pytest.raises(ValueError, match="'host' and 'port'"):
        PGVector({"user": "user", "password": "password"})

    assert PGVector({"connection_string": "postgresql://db.test/mem0"})


@pytest.mark.asyncio
async def test_pgvector_constructor_is_state_only_and_initializes_schema():
    store = PGVector(_config(collection_name='mem0"; DROP TABLE users; --'))

    assert _FakePool.instances == []

    await store._initialize()

    pool = _FakePool.instances[0]
    assert pool.kwargs["min_size"] == 1
    assert pool.kwargs["max_size"] == 5
    assert "dbname=postgres" in pool.kwargs["conninfo"]
    assert pool.calls[0] == ("open", True)
    statements = [call[1] for call in pool.calls if call[0] == "execute"]
    assert "CREATE EXTENSION IF NOT EXISTS vector" in statements
    assert any(
        'CREATE TABLE IF NOT EXISTS "mem0""; DROP TABLE users; --"' in statement
        for statement in statements
    )
    assert any("USING hnsw" in statement for statement in statements)
    assert any("USING gin(to_tsvector" in statement for statement in statements)
    await store.close()


@pytest.mark.asyncio
async def test_pgvector_requires_native_libpq_cancellation(monkeypatch):
    monkeypatch.setattr(
        "psycopg.capabilities.has_cancel_safe",
        lambda *, check: False,
    )
    store = PGVector(_config())

    with pytest.raises(RuntimeError, match="libpq 17 or newer"):
        await store._initialize()

    assert _FakePool.instances == []


@pytest.mark.asyncio
async def test_pgvector_preserves_connection_string_sslmode():
    store = PGVector(
        _config(
            connection_string=(
                "postgresql://user:password@db.test/postgres?application_name=mem0"
            ),
            sslmode="require",
        )
    )

    await store._initialize()

    conninfo = _FakePool.instances[0].kwargs["conninfo"]
    assert "application_name=mem0" in conninfo
    assert "sslmode=require" in conninfo
    await store.close()


@pytest.mark.asyncio
async def test_pgvector_recreates_collection_when_dimensions_change():
    pool = _FakePool()
    pool.tables = ["mem0"]
    pool.dimension_rows = [(2,)]
    store = PGVector(_config(connection_pool=pool, embedding_model_dims=3))

    await store._initialize()

    statements = [call[1] for call in pool.calls if call[0] == "execute"]
    assert any("SELECT atttypmod FROM pg_attribute" in item for item in statements)
    assert any("DROP TABLE IF EXISTS \"mem0\"" in item for item in statements)
    assert any("CREATE TABLE IF NOT EXISTS \"mem0\"" in item for item in statements)
    await store.close()


@pytest.mark.asyncio
async def test_pgvector_does_not_close_injected_async_pool():
    pool = _FakePool()
    pool.tables = ["mem0"]
    store = PGVector(_config(connection_pool=pool))

    await asyncio.gather(store._initialize(), store._initialize())
    await store.close()

    assert len(_FakePool.instances) == 1
    assert ("open", True) not in pool.calls
    assert pool.closed is False


@pytest.mark.asyncio
async def test_pgvector_initialization_cancellation_closes_owned_pool(monkeypatch):
    entered = asyncio.Event()

    async def blocking_open(self, *, wait):
        assert wait is True
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(_FakePool, "open", blocking_open)
    store = PGVector(_config())
    task = asyncio.create_task(store._initialize())
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert _FakePool.instances[0].closed is True


def test_pgvector_filter_builder_matches_upstream_parameterization():
    conditions, params = _build_filter_conditions(
        {
            "user_id": "u1",
            "active": True,
            "score": {"gte": 2},
            "tag": {"in": ["a", "b"]},
            "title": {"icontains": "50%_off"},
            "present": "*",
            "$or": [{"agent_id": "a1"}, {"run_id": "r1"}],
            "$not": [{"state": "deleted"}],
        }
    )

    assert conditions == [
        "payload->>%s = %s",
        "payload->>%s = %s",
        "(payload->>%s)::numeric >= %s",
        "payload->>%s = ANY(%s)",
        "payload->>%s ILIKE %s ESCAPE '\\'",
        "payload ? %s",
        "((payload->>%s = %s) OR (payload->>%s = %s))",
        "NOT ((payload->>%s = %s))",
    ]
    assert params == [
        "user_id",
        "u1",
        "active",
        "true",
        "score",
        2.0,
        "tag",
        ["a", "b"],
        "title",
        "%50\\%\\_off%",
        "present",
        "agent_id",
        "a1",
        "run_id",
        "r1",
        "state",
        "deleted",
    ]


@pytest.mark.parametrize(
    "filters",
    [
        None,
        {},
        {"user_id": "u1", "active": False, "attempt": 3},
        {"score": {"eq": 1, "ne": 2, "gt": 3, "gte": 4}},
        {"score": {"lt": 5, "lte": 6}},
        {"tag": {"in": ["a", 2], "nin": ["b", 3]}},
        {"title": {"contains": r"50%_off\\today"}},
        {"title": {"icontains": r"50%_off\\today"}},
        {"tag": ["a", 2], "present": "*"},
        {"$or": [{"user_id": "u1", "run_id": "r1"}, {"agent_id": "a1"}]},
        {"$not": [{"state": "deleted"}, {"state": "expired"}]},
    ],
)
def test_pgvector_filter_builder_is_exactly_upstream(filters):
    from mem0.vector_stores.pgvector import (
        _build_filter_conditions as upstream_build_filter_conditions,
    )

    assert _build_filter_conditions(filters) == upstream_build_filter_conditions(
        filters
    )


@pytest.mark.asyncio
async def test_pgvector_crud_search_and_result_shapes():
    store = PGVector(_config())
    await store._initialize()
    pool = _FakePool.instances[0]
    pool.tables = ["mem0"]
    pool.semantic_rows = [("id-1", 0.25, {"data": "fact"})]
    pool.keyword_rows = [("id-1", 2.5, {"data": "fact"})]
    pool.record_rows = [("id-1", "[1,2,3]", {"data": "fact"})]
    pool.info_rows = [("mem0", 1, "16 kB")]

    await store.insert(
        vectors=[[1.0, 2.0, 3.0]],
        ids=["id-1"],
        payloads=[{"data": "fact"}],
    )
    semantic = await store.search(
        "fact",
        [1.0, 2.0, 3.0],
        top_k=4,
        filters={"user_id": "u1"},
    )
    keyword = await store.keyword_search(
        "fact",
        top_k=4,
        filters={"user_id": "u1"},
    )
    assert semantic[0].id == "id-1"
    assert semantic[0].score == 0.75
    assert semantic[0].payload == {"data": "fact"}
    assert keyword[0].score == 2.5

    assert (await store.get("id-1")).id == "id-1"
    listed = await store.list(filters={"user_id": "u1"}, top_k=10)
    assert listed[0][0].payload == {"data": "fact"}
    assert await store.col_info() == {"name": "mem0", "count": 1, "size": "16 kB"}
    await store.update("id-1", vector=[3.0, 2.0, 1.0], payload={"data": "new"})
    await store.delete("id-1")
    await store.delete_col()

    calls = pool.calls
    assert any(call[0] == "executemany" for call in calls)
    statements = [call[1] for call in calls if call[0] == "execute"]
    assert any("UPDATE \"mem0\" SET vector" in item for item in statements)
    assert any("UPDATE \"mem0\" SET payload" in item for item in statements)
    assert any("DELETE FROM \"mem0\"" in item for item in statements)
    assert any("DROP TABLE IF EXISTS \"mem0\"" in item for item in statements)
    await store.close()


@pytest.mark.asyncio
async def test_pgvector_close_is_idempotent_and_preserves_cancellation():
    store = PGVector(_config())
    await store._initialize()
    pool = _FakePool.instances[0]
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocking_close():
        entered.set()
        await release.wait()
        pool.closed = True

    pool.close = blocking_close
    task = asyncio.create_task(store.close())
    await entered.wait()
    task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert pool.closed is True
    await store.close()
