"""Provider-level parity tests for the native-async Mem0 plugin."""

import asyncio
import json

import httpx
import pytest

import plugins.memory.mem0 as mem0_plugin
from plugins.memory.mem0 import Mem0MemoryProvider
from plugins.memory.mem0._backend import MemoryNotFoundError, PlatformBackend


class FakeBackend:
    def __init__(self, search_results=None):
        self.search_results = search_results or []
        self.captured = []
        self.closed = False

    async def search(self, query, *, filters, top_k=10, rerank=True):
        self.captured.append(
            ("search", query, {"filters": filters, "top_k": top_k, "rerank": rerank})
        )
        return self.search_results

    async def add(self, messages, *, user_id, agent_id, infer=False, metadata=None):
        self.captured.append(
            (
                "add",
                messages,
                {
                    "user_id": user_id,
                    "agent_id": agent_id,
                    "infer": infer,
                    "metadata": metadata,
                },
            )
        )
        return {"status": "PENDING", "event_id": "evt-test-123"}

    async def update(self, memory_id, text):
        self.captured.append(("update", memory_id, text))
        return {"result": "Memory updated.", "memory_id": memory_id}

    async def delete(self, memory_id):
        self.captured.append(("delete", memory_id))
        return {"result": "Memory deleted.", "memory_id": memory_id}

    async def close(self):
        self.closed = True


async def _provider(monkeypatch, tmp_path, backend, **kwargs):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    provider = Mem0MemoryProvider()
    monkeypatch.setattr(provider, "_create_backend", lambda: backend)
    await provider.initialize("test-session", **kwargs)
    return provider


@pytest.mark.asyncio
async def test_mem0_tool_contract_and_result_shapes(monkeypatch, tmp_path):
    backend = FakeBackend(
        search_results=[{"id": "mem-1", "memory": "likes tea", "score": 0.9}]
    )
    provider = await _provider(monkeypatch, tmp_path, backend, user_id="u123")

    searched = json.loads(
        await provider.handle_tool_call(
            "mem0_search",
            {"query": "drink", "top_k": 7, "rerank": True},
        )
    )
    added = json.loads(
        await provider.handle_tool_call("mem0_add", {"content": "likes tea"})
    )
    updated = json.loads(
        await provider.handle_tool_call(
            "mem0_update",
            {"memory_id": "mem-1", "text": "likes coffee"},
        )
    )
    deleted = json.loads(
        await provider.handle_tool_call("mem0_delete", {"memory_id": "mem-1"})
    )

    assert searched["results"][0]["id"] == "mem-1"
    assert added == {"result": "Fact queued for storage.", "event_id": "evt-test-123"}
    assert updated == {"result": "Memory updated.", "memory_id": "mem-1"}
    assert deleted == {"result": "Memory deleted.", "memory_id": "mem-1"}
    assert backend.captured[0][2] == {
        "filters": {"user_id": "u123"},
        "top_k": 7,
        "rerank": True,
    }
    assert backend.captured[1][2]["infer"] is False
    assert backend.captured[1][2]["metadata"] == {"channel": "cli"}


@pytest.mark.asyncio
async def test_memory_not_found_does_not_increment_provider_breaker(
    monkeypatch, tmp_path
):
    class MissingBackend(FakeBackend):
        async def update(self, memory_id, text):
            raise MemoryNotFoundError(
                "Memory not found",
                "HTTP_404",
            )

    provider = await _provider(monkeypatch, tmp_path, MissingBackend())

    result = json.loads(
        await provider.handle_tool_call(
            "mem0_update",
            {"memory_id": "missing", "text": "new"},
        )
    )

    assert result == {"error": "Memory not found: missing"}
    assert provider._consecutive_failures == 0


@pytest.mark.asyncio
async def test_sync_turn_preserves_message_and_identity_contract(monkeypatch, tmp_path):
    backend = FakeBackend()
    provider = await _provider(
        monkeypatch,
        tmp_path,
        backend,
        user_id="u123",
        platform="telegram",
    )

    await provider.sync_turn("user said", "assistant replied", session_id="s1")

    assert backend.captured == [
        (
            "add",
            [
                {"role": "user", "content": "user said"},
                {"role": "assistant", "content": "assistant replied"},
            ],
            {
                "user_id": "u123",
                "agent_id": "hermes",
                "infer": True,
                "metadata": {"channel": "telegram"},
            },
        )
    ]


@pytest.mark.asyncio
async def test_prefetch_reuses_turn_start_request(monkeypatch, tmp_path):
    backend = FakeBackend(
        search_results=[{"id": "m1", "memory": "user prefers dark mode"}]
    )
    provider = await _provider(monkeypatch, tmp_path, backend, user_id="u123")

    await provider.on_turn_start(1, "what theme do I like?")
    result = await provider.prefetch("what theme do I like?")

    assert result == "## Mem0 Memory\n- user prefers dark mode"
    assert len([call for call in backend.captured if call[0] == "search"]) == 1


@pytest.mark.asyncio
async def test_slow_prefetch_returns_without_blocking_loop(monkeypatch, tmp_path):
    entered = asyncio.Event()
    release = asyncio.Event()

    class SlowBackend(FakeBackend):
        async def search(self, query, *, filters, top_k=10, rerank=True):
            entered.set()
            await release.wait()
            return await super().search(
                query,
                filters=filters,
                top_k=top_k,
                rerank=rerank,
            )

    backend = SlowBackend(search_results=[{"id": "m1", "memory": "lives in Berlin"}])
    provider = await _provider(monkeypatch, tmp_path, backend)
    monkeypatch.setattr(mem0_plugin, "_PREFETCH_WAIT_SECS", 0.01)
    heartbeat = 0

    async def beat():
        nonlocal heartbeat
        while True:
            heartbeat += 1
            await asyncio.sleep(0)

    heartbeat_task = asyncio.create_task(beat())
    try:
        assert await provider.prefetch("where do I live?") == ""
        assert entered.is_set()
        assert heartbeat > 1
        release.set()
        assert "lives in Berlin" in await provider.prefetch("where do I live?")
    finally:
        heartbeat_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await heartbeat_task
        await provider.shutdown()


@pytest.mark.asyncio
async def test_user_id_resolution_and_async_config_file(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    config_path = tmp_path / "mem0.json"
    config_path.write_text('{"user_id": "ryan@example.com"}', encoding="utf-8")
    provider = Mem0MemoryProvider()
    monkeypatch.setattr(provider, "_create_backend", lambda: FakeBackend())

    await provider.initialize("test", user_id="gateway-id", platform="telegram")
    await provider.save_config({"rerank": "true"}, str(tmp_path))

    assert provider._user_id == "ryan@example.com"
    assert json.loads(config_path.read_text(encoding="utf-8")) == {
        "user_id": "ryan@example.com",
        "rerank": "true",
    }
    assert config_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_availability_uses_async_profile_config(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("MEM0_API_KEY", raising=False)
    monkeypatch.delenv("MEM0_HOST", raising=False)
    provider = Mem0MemoryProvider()

    assert await provider.is_available() is False

    (tmp_path / "mem0.json").write_text(
        '{"mode": "platform", "host": "http://mem0.test"}',
        encoding="utf-8",
    )
    assert await provider.is_available() is True


@pytest.mark.asyncio
async def test_backend_initialization_failure_preserves_provider_contract(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "mem0.json").write_text(
        '{"mode": "oss", "oss": {}}',
        encoding="utf-8",
    )
    provider = Mem0MemoryProvider()

    await provider.initialize("test-session")

    assert provider._backend is None
    assert provider._init_error == "'vector_store'"
    result = json.loads(
        await provider.handle_tool_call("mem0_search", {"query": "anything"})
    )
    assert result == {
        "error": (
            "Mem0 backend not initialized: 'vector_store'. "
            "Check that vector store is running and reachable."
        )
    }


@pytest.mark.asyncio
async def test_oss_thread_fallback_is_rejected_during_provider_initialization(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "mem0.json").write_text(
        json.dumps(
            {
                "mode": "oss",
                "oss": {
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
                },
            }
        ),
        encoding="utf-8",
    )
    provider = Mem0MemoryProvider()

    await provider.initialize("test-session")

    assert provider._backend is None
    assert "has no native-async runtime" in provider._init_error


@pytest.mark.asyncio
async def test_platform_validation_failure_closes_backend_and_preserves_error(
    monkeypatch, tmp_path
):
    async_client = httpx.AsyncClient
    requests = []

    async def handler(request):
        requests.append(request)
        return httpx.Response(401, json={"detail": "Invalid API key"})

    def build_client(*args, **kwargs):
        return async_client(
            *args,
            **kwargs,
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(httpx, "AsyncClient", build_client)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MEM0_API_KEY", "invalid-key")
    provider = Mem0MemoryProvider()

    await provider.initialize("test-session")

    assert [request.url.path for request in requests] == ["/v1/ping/"]
    assert provider._backend is None
    assert provider._init_error == "Error: Invalid API key"


@pytest.mark.asyncio
async def test_platform_validation_cancellation_closes_backend_and_propagates(
    monkeypatch, tmp_path
):
    entered = asyncio.Event()
    closed = asyncio.Event()

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def get(self, path):
            entered.set()
            await asyncio.Event().wait()

        async def aclose(self):
            closed.set()

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MEM0_API_KEY", "test-key")
    provider = Mem0MemoryProvider()

    initialize = asyncio.create_task(provider.initialize("test-session"))
    await entered.wait()
    initialize.cancel()

    with pytest.raises(asyncio.CancelledError):
        await initialize
    assert closed.is_set()
    assert provider._backend is None


@pytest.mark.asyncio
async def test_platform_failure_cleanup_survives_repeated_cancellation(
    monkeypatch, tmp_path
):
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    close_finished = asyncio.Event()
    async_client = httpx.AsyncClient

    async def handler(request):
        return httpx.Response(401, json={"detail": "Invalid API key"})

    def build_client(*args, **kwargs):
        return async_client(
            *args,
            **kwargs,
            transport=httpx.MockTransport(handler),
        )

    async def slow_close(self):
        close_started.set()
        await release_close.wait()
        await self._client.aclose()
        close_finished.set()

    monkeypatch.setattr(httpx, "AsyncClient", build_client)
    monkeypatch.setattr(PlatformBackend, "close", slow_close)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MEM0_API_KEY", "invalid-key")
    provider = Mem0MemoryProvider()

    initialize = asyncio.create_task(provider.initialize("test-session"))
    await close_started.wait()
    initialize.cancel()
    await asyncio.sleep(0)
    initialize.cancel()
    await asyncio.sleep(0)

    assert initialize.done() is False
    release_close.set()
    with pytest.raises(asyncio.CancelledError):
        await initialize
    assert close_finished.is_set()
    assert provider._backend is None


@pytest.mark.asyncio
async def test_shutdown_cancels_prefetch_and_closes_backend(monkeypatch, tmp_path):
    release = asyncio.Event()

    class SlowBackend(FakeBackend):
        async def search(self, query, *, filters, top_k=10, rerank=True):
            await release.wait()
            return []

    backend = SlowBackend()
    provider = await _provider(monkeypatch, tmp_path, backend)
    await provider.on_turn_start(1, "pending")

    await provider.shutdown()

    assert backend.closed is True
    assert provider._prefetch_task is None


@pytest.mark.asyncio
async def test_shutdown_ignores_backend_close_failure(monkeypatch, tmp_path):
    class FailingCloseBackend(FakeBackend):
        async def close(self):
            raise RuntimeError("close failed")

    provider = await _provider(monkeypatch, tmp_path, FailingCloseBackend())

    await provider.shutdown()

    assert provider._backend is None


@pytest.mark.asyncio
async def test_shutdown_collects_prefetch_through_repeated_cancellation(
    monkeypatch, tmp_path
):
    cancellation_started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    class SlowBackend(FakeBackend):
        async def search(self, query, *, filters, top_k=10, rerank=True):
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancellation_started.set()
                await release.wait()
            finished.set()
            return []

    backend = SlowBackend()
    provider = await _provider(monkeypatch, tmp_path, backend)
    await provider.on_turn_start(1, "pending")

    shutdown = asyncio.create_task(provider.shutdown())
    await cancellation_started.wait()
    shutdown.cancel()
    await asyncio.sleep(0)
    shutdown.cancel()
    await asyncio.sleep(0)

    assert shutdown.done() is False
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await shutdown
    assert finished.is_set()
    assert backend.closed is True
    assert provider._prefetch_task is None


@pytest.mark.asyncio
async def test_prefetch_replacement_collects_previous_task_on_cancellation(
    monkeypatch, tmp_path
):
    first_cancelled = asyncio.Event()
    release_first = asyncio.Event()
    first_finished = asyncio.Event()
    second_started = asyncio.Event()

    class SlowBackend(FakeBackend):
        async def search(self, query, *, filters, top_k=10, rerank=True):
            if query == "first":
                try:
                    await release_first.wait()
                except asyncio.CancelledError:
                    first_cancelled.set()
                    await release_first.wait()
                first_finished.set()
            else:
                second_started.set()
            return []

    provider = await _provider(monkeypatch, tmp_path, SlowBackend())
    await provider.on_turn_start(1, "first")

    replacement = asyncio.create_task(provider.on_turn_start(2, "second"))
    await first_cancelled.wait()
    replacement.cancel()
    await asyncio.sleep(0)
    replacement.cancel()
    await asyncio.sleep(0)

    assert replacement.done() is False
    release_first.set()
    with pytest.raises(asyncio.CancelledError):
        await replacement
    assert first_finished.is_set()
    assert second_started.is_set() is False
    await provider.shutdown()


def test_schema_and_system_prompt_preserve_public_names():
    provider = Mem0MemoryProvider()
    assert [schema["name"] for schema in provider.get_tool_schemas()] == [
        "mem0_search",
        "mem0_add",
        "mem0_update",
        "mem0_delete",
    ]
    prompt = provider.system_prompt_block()
    for name in ("mem0_search", "mem0_add", "mem0_update", "mem0_delete"):
        assert name in prompt
