"""Native-async parity tests for Mem0 backends."""

import asyncio
import copy
import json
import sys
import types

import httpx
import pytest

from plugins.memory.mem0._backend import (
    MemoryError,
    MemoryNotFoundError,
    OSSBackend,
    PlatformBackend,
    SelfHostedBackend,
)


class _StubServer:
    def __init__(self):
        self.requests: list[httpx.Request] = []

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path, method = request.url.path, request.method
        if path == "/v1/ping/" and method == "GET":
            return httpx.Response(
                200,
                json={
                    "user_email": "user@example.com",
                    "org_id": "org-1",
                    "project_id": "project-1",
                },
            )
        if path.endswith("/search/") and method == "POST":
            return httpx.Response(
                200,
                json={"results": [{"id": "m1", "memory": "tea", "score": 0.9}]},
            )
        if path.endswith("/add/") and method == "POST":
            return httpx.Response(200, json={"status": "PENDING", "event_id": "evt-1"})
        if path.startswith("/v1/memories/") and method in {"PUT", "DELETE"}:
            return httpx.Response(200, json={"message": "ok"})
        if path == "/search" and method == "POST":
            return httpx.Response(
                200,
                json={"results": [{"id": "m1", "memory": "tea", "score": 0.9}]},
            )
        if path == "/memories" and method == "POST":
            return httpx.Response(200, json={"results": [{"id": "new"}]})
        if path.startswith("/memories/") and method in {"PUT", "DELETE"}:
            if path.endswith("/missing"):
                return httpx.Response(404, json={"detail": "Memory not found"})
            return httpx.Response(200, json={"message": "ok"})
        return httpx.Response(404, json={"detail": "not found"})


@pytest.mark.asyncio
async def test_platform_backend_preserves_cloud_request_contract(monkeypatch):
    server = _StubServer()
    async_client = httpx.AsyncClient

    def build_client(*args, **kwargs):
        return async_client(
            *args,
            **kwargs,
            transport=httpx.MockTransport(server.handler),
        )

    monkeypatch.setattr(httpx, "AsyncClient", build_client)
    backend = PlatformBackend("secret")
    try:
        results = await backend.search(
            "test query",
            filters={"user_id": "u1"},
            top_k=5,
            rerank=True,
        )
        added = await backend.add(
            [{"role": "user", "content": "hi"}],
            user_id="u1",
            agent_id="hermes",
            infer=False,
        )
        await backend.update("m1", "new text")
        await backend.delete("m1")
    finally:
        await backend.close()

    assert results == [{"id": "m1", "memory": "tea", "score": 0.9}]
    assert added["event_id"] == "evt-1"
    assert [request.url.path for request in server.requests] == [
        "/v1/ping/",
        "/v3/memories/search/",
        "/v3/memories/add/",
        "/v1/memories/m1/",
        "/v1/memories/m1/",
    ]
    search_payload = json.loads(server.requests[1].content)
    assert search_payload == {
        "query": "test query",
        "filters": {"user_id": "u1"},
        "top_k": 5,
        "rerank": True,
    }
    add_payload = json.loads(server.requests[2].content)
    assert add_payload["infer"] is False
    assert "metadata" not in add_payload
    assert server.requests[0].headers["authorization"] == "Token secret"
    assert server.requests[0].headers["mem0-user-id"]
    assert backend.org_id == "org-1"
    assert backend.project_id == "project-1"
    assert backend.user_email == "user@example.com"


@pytest.mark.asyncio
async def test_platform_search_trims_and_validates_like_memory_client(monkeypatch):
    server = _StubServer()
    async_client = httpx.AsyncClient

    def build_client(*args, **kwargs):
        return async_client(
            *args,
            **kwargs,
            transport=httpx.MockTransport(server.handler),
        )

    monkeypatch.setattr(httpx, "AsyncClient", build_client)
    backend = PlatformBackend("secret")
    try:
        await backend.search("  trimmed query  ", filters={"user_id": "u1"})
        with pytest.raises(ValueError, match="cannot be empty"):
            await backend.search("   ", filters={"user_id": "u1"})
    finally:
        await backend.close()

    payload = json.loads(server.requests[1].content)
    assert payload["query"] == "trimmed query"
    assert [request.url.path for request in server.requests].count("/v1/ping/") == 1


@pytest.mark.asyncio
async def test_platform_404_uses_mem0_client_exception_contract(monkeypatch):
    async def handler(request):
        if request.url.path == "/v1/ping/":
            return httpx.Response(200, json={})
        return httpx.Response(404, json={"detail": "Memory not found"})

    async_client = httpx.AsyncClient

    def build_client(*args, **kwargs):
        return async_client(
            *args,
            **kwargs,
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(httpx, "AsyncClient", build_client)
    backend = PlatformBackend("secret")
    try:
        with pytest.raises(MemoryNotFoundError) as raised:
            await backend.update("missing", "new")
    finally:
        await backend.close()

    assert str(raised.value) == "Memory not found"
    assert raised.value.error_code == "HTTP_404"


@pytest.mark.asyncio
async def test_platform_generic_http_error_matches_mem0_base_exception(monkeypatch):
    async def handler(request):
        if request.url.path == "/v1/ping/":
            return httpx.Response(200, json={})
        return httpx.Response(500, text="server failed")

    async_client = httpx.AsyncClient

    def build_client(*args, **kwargs):
        return async_client(
            *args,
            **kwargs,
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(httpx, "AsyncClient", build_client)
    backend = PlatformBackend("secret")
    try:
        with pytest.raises(MemoryError) as raised:
            await backend.delete("m1")
    finally:
        await backend.close()

    assert type(raised.value).__name__ == "MemoryError"
    assert repr(raised.value).startswith("MemoryError(message='server failed'")


@pytest.mark.asyncio
async def test_platform_ping_requires_complete_org_project_pair(monkeypatch):
    async def handler(request):
        return httpx.Response(
            200,
            json={"org_id": "org-only", "user_email": "user@example.com"},
        )

    async_client = httpx.AsyncClient

    def build_client(*args, **kwargs):
        return async_client(
            *args,
            **kwargs,
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(httpx, "AsyncClient", build_client)
    backend = PlatformBackend("secret")
    try:
        await backend._initialize()
    finally:
        await backend.close()

    assert backend.org_id is None
    assert backend.project_id is None
    assert backend.user_email == "user@example.com"


@pytest.mark.asyncio
async def test_self_hosted_backend_preserves_http_contract():
    server = _StubServer()
    backend = SelfHostedBackend(
        "admin-key",
        "http://mem0.test",
        transport=httpx.MockTransport(server.handler),
    )
    try:
        result = await backend.search(
            "where",
            filters={"user_id": "u1"},
            top_k=3,
        )
        await backend.add(
            [{"role": "user", "content": "hi"}],
            user_id="u1",
            agent_id="hermes",
        )
        await backend.update("m1", "new")
        await backend.delete("m1")
    finally:
        await backend.close()

    assert result[0]["memory"] == "tea"
    assert [request.url.path for request in server.requests] == [
        "/search",
        "/memories",
        "/memories/m1",
        "/memories/m1",
    ]
    assert server.requests[0].headers["x-api-key"] == "admin-key"
    assert "authorization" not in server.requests[0].headers


@pytest.mark.asyncio
async def test_self_hosted_http_errors_propagate():
    server = _StubServer()
    backend = SelfHostedBackend(
        "admin-key",
        "http://mem0.test",
        transport=httpx.MockTransport(server.handler),
    )
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await backend.delete("missing")
    finally:
        await backend.close()


def test_oss_legacy_base_urls_are_normalized_without_mutating_input():
    raw = {
        "llm": {
            "provider": "openai",
            "config": {"model": "gpt-5-mini", "api_base": "https://llm.test/v1"},
        },
        "embedder": {
            "provider": "ollama",
            "config": {"model": "nomic-embed-text", "api_base": "http://ollama:11434"},
        },
        "vector_store": {"provider": "qdrant", "config": {}},
    }
    before = copy.deepcopy(raw)

    backend = OSSBackend(raw)

    assert backend._config["llm"]["config"]["openai_base_url"] == "https://llm.test/v1"
    assert backend._config["embedder"]["config"]["ollama_base_url"] == "http://ollama:11434"
    assert raw == before


@pytest.mark.asyncio
async def test_oss_backend_initializes_embedded_qdrant_natively(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("MEM0_DIR", str(tmp_path / "mem0-home"))
    backend = OSSBackend(
        {
            "llm": {
                "provider": "openai",
                "config": {"model": "gpt-5-mini"},
            },
            "embedder": {
                "provider": "openai",
                "config": {"model": "text-embedding-3-small"},
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {"path": str(tmp_path / "qdrant")},
            },
        }
    )

    await backend._initialize()

    assert backend._memory is not None
    assert backend._memory.vector_store._is_local is True
    await backend.close()


@pytest.mark.asyncio
async def test_oss_backend_initializes_native_memory_once(monkeypatch):
    initialized = asyncio.Event()

    class FakeMemory:
        instances = []

        def __init__(self, config):
            self.config = config
            self.closed = False
            self.instances.append(self)

        async def initialize(self):
            initialized.set()
            await asyncio.sleep(0)

        async def close(self):
            self.closed = True

    monkeypatch.setattr(
        "plugins.memory.mem0._native_memory.Memory",
        FakeMemory,
    )
    backend = OSSBackend(
        {
            "llm": {
                "provider": "openai",
                "config": {"model": "gpt-5-mini"},
            },
            "embedder": {
                "provider": "openai",
                "config": {"model": "unknown-dim-model"},
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {"url": "https://qdrant.test"},
            },
        }
    )

    await asyncio.gather(backend._initialize(), backend._initialize())

    assert initialized.is_set()
    assert len(FakeMemory.instances) == 1
    assert backend._memory is FakeMemory.instances[0]
    assert FakeMemory.instances[0].config == backend._config
    await backend.close()


@pytest.mark.asyncio
async def test_oss_backend_closes_memory_when_initialization_fails(monkeypatch):
    class FakeMemory:
        instance = None

        def __init__(self, config):
            self.config = config
            self.closed = False
            type(self).instance = self

        async def initialize(self):
            raise RuntimeError("initialization failed")

        async def close(self):
            self.closed = True

    monkeypatch.setattr(
        "plugins.memory.mem0._native_memory.Memory",
        FakeMemory,
    )
    backend = OSSBackend(
        {
            "llm": {
                "provider": "openai",
                "config": {"model": "gpt-5-mini"},
            },
            "embedder": {
                "provider": "openai",
                "config": {"model": "unknown-dim-model"},
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {"url": "https://qdrant.test"},
            },
        }
    )

    with pytest.raises(RuntimeError, match="initialization failed"):
        await backend._initialize()

    assert backend._memory is None
    assert FakeMemory.instance.closed is True


@pytest.mark.asyncio
async def test_oss_backend_close_waits_for_concurrent_initialization(monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()

    class FakeMemory:
        instance = None

        def __init__(self, config):
            self.config = config
            self.closed = False
            type(self).instance = self

        async def initialize(self):
            entered.set()
            await release.wait()

        async def close(self):
            self.closed = True

    monkeypatch.setattr(
        "plugins.memory.mem0._native_memory.Memory",
        FakeMemory,
    )
    backend = OSSBackend(
        {
            "llm": {
                "provider": "openai",
                "config": {"model": "gpt-5-mini"},
            },
            "embedder": {
                "provider": "openai",
                "config": {"model": "unknown-dim-model"},
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {"url": "https://qdrant.test"},
            },
        }
    )
    initialize_task = asyncio.create_task(backend._initialize())
    await entered.wait()
    close_task = asyncio.create_task(backend.close())
    await asyncio.sleep(0)

    assert close_task.done() is False
    release.set()
    await asyncio.gather(initialize_task, close_task)

    assert backend._memory is None
    assert FakeMemory.instance.closed is True


@pytest.mark.asyncio
async def test_oss_qdrant_dimension_probe_disables_client_thread(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def collection_exists(self, collection_name):
            assert collection_name == "mem0"
            return False

        async def close(self):
            captured["closed"] = True

    qdrant_client = types.ModuleType("qdrant_client")
    qdrant_client.AsyncQdrantClient = FakeClient
    monkeypatch.setitem(sys.modules, "qdrant_client", qdrant_client)

    await OSSBackend._recreate_collection_if_dims_changed(
        "qdrant",
        {"url": "https://qdrant.test", "api_key": "secret"},
        1536,
    )

    assert captured == {
        "url": "https://qdrant.test",
        "api_key": "secret",
        "check_compatibility": False,
        "closed": True,
    }


@pytest.mark.asyncio
async def test_oss_qdrant_dimension_probe_uses_native_local_proxy(monkeypatch):
    captured = {}

    class ForbiddenClient:
        def __init__(self, **kwargs):
            raise AssertionError("local AsyncQdrantClient must not be constructed")

    class LocalClient:
        def __init__(self, path, models):
            captured["path"] = path
            captured["models"] = models

        async def collection_exists(self, collection_name):
            captured["collection_name"] = collection_name
            return False

        async def close(self):
            captured["closed"] = True

    models = object()
    qdrant_client = types.ModuleType("qdrant_client")
    qdrant_client.AsyncQdrantClient = ForbiddenClient
    qdrant_client.models = models
    monkeypatch.setitem(sys.modules, "qdrant_client", qdrant_client)
    monkeypatch.setattr(
        "plugins.memory.mem0._native_local_qdrant.NativeLocalQdrantClient",
        LocalClient,
    )

    await OSSBackend._recreate_collection_if_dims_changed(
        "qdrant",
        {"path": "/tmp/mem0-test"},
        1536,
    )

    assert captured == {
        "path": "/tmp/mem0-test",
        "models": models,
        "collection_name": "mem0",
        "closed": True,
    }


@pytest.mark.asyncio
async def test_oss_backend_uses_async_memory_v2_signatures():
    class FakeMemory:
        def __init__(self):
            self.calls = []

        async def search(self, query, **kwargs):
            self.calls.append(("search", query, kwargs))
            return {"results": [{"id": "m1", "memory": "fact"}]}

        async def add(self, messages, **kwargs):
            self.calls.append(("add", messages, kwargs))
            return {"results": []}

        async def update(self, memory_id, **kwargs):
            self.calls.append(("update", memory_id, kwargs))

        async def delete(self, memory_id):
            self.calls.append(("delete", memory_id))

    memory = FakeMemory()
    backend = OSSBackend.__new__(OSSBackend)
    backend._memory = memory
    backend._collection_check = None
    backend._initialize_lock = asyncio.Lock()

    assert await backend.search("query", filters={"user_id": "u1"}, top_k=4) == [
        {"id": "m1", "memory": "fact"}
    ]
    await backend.add(
        [{"role": "user", "content": "fact"}],
        user_id="u1",
        agent_id="hermes",
        infer=False,
    )
    await backend.update("m1", "new")
    await backend.delete("m1")

    assert memory.calls[0][2] == {"filters": {"user_id": "u1"}, "top_k": 4}
    assert memory.calls[1][2]["infer"] is False
    assert memory.calls[2] == ("update", "m1", {"data": "new"})
    assert memory.calls[3] == ("delete", "m1")


@pytest.mark.asyncio
async def test_oss_backend_close_delegates_to_native_memory_close():
    class FakeMemory:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    memory = FakeMemory()
    backend = OSSBackend.__new__(OSSBackend)
    backend._memory = memory
    backend._initialize_lock = asyncio.Lock()

    await backend.close()

    assert memory.closed is True
    assert backend._memory is None


@pytest.mark.asyncio
async def test_oss_backend_close_rejects_sync_memory_close_without_calling_it():
    class SyncMemory:
        def __init__(self):
            self.close_called = False

        def close(self):
            self.close_called = True

    memory = SyncMemory()
    backend = OSSBackend.__new__(OSSBackend)
    backend._memory = memory
    backend._initialize_lock = asyncio.Lock()

    with pytest.raises(RuntimeError, match="native-async close"):
        await backend.close()

    assert memory.close_called is False
    assert backend._memory is None


@pytest.mark.asyncio
async def test_oss_backend_close_preserves_cancellation():
    entered = asyncio.Event()

    class FakeMemory:
        async def close(self):
            entered.set()
            await asyncio.Event().wait()

    backend = OSSBackend.__new__(OSSBackend)
    backend._memory = FakeMemory()
    backend._initialize_lock = asyncio.Lock()
    task = asyncio.create_task(backend.close())
    await entered.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert backend._memory is None


@pytest.mark.asyncio
async def test_http_backends_never_call_asyncio_to_thread(monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError("asyncio.to_thread must not be used")

    monkeypatch.setattr(asyncio, "to_thread", forbidden)
    server = _StubServer()
    async_client = httpx.AsyncClient

    def build_client(*args, **kwargs):
        return async_client(
            *args,
            **kwargs,
            transport=httpx.MockTransport(server.handler),
        )

    monkeypatch.setattr(httpx, "AsyncClient", build_client)
    backend = PlatformBackend("secret")
    try:
        assert await backend.search("q", filters={"user_id": "u1"})
    finally:
        await backend.close()
