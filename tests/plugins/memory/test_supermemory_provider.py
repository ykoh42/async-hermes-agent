import asyncio
import json
import os
import stat
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

from agent.memory_manager import MemoryManager
from agent.secret_scope import reset_secret_scope, set_secret_scope
from plugins.memory import load_memory_provider
from plugins.memory.supermemory import (
    SupermemoryMemoryProvider,
    _clean_text_for_capture,
    _format_connection_summary,
    _format_prefetch_context,
    _load_supermemory_config,
    _probe_supermemory_connection,
    _save_supermemory_config,
)

pytestmark = pytest.mark.asyncio


class FakeClient:
    def __init__(self, api_key: str, timeout: float, container_tag: str, search_mode: str = "hybrid",
                 base_url: str = ""):
        self.api_key = api_key
        self.timeout = timeout
        self.container_tag = container_tag
        self.search_mode = search_mode
        self.base_url = base_url
        self.add_calls = []
        self.search_results = []
        self.profile_response = {"static": [], "dynamic": [], "search_results": []}
        self.ingest_calls = []
        self.forgotten_ids = []
        self.forget_by_query_response = {"success": True, "message": "Forgot"}
        self.closed = False

    async def initialize(self):
        return None

    async def add_memory(self, content, metadata=None, *, entity_context="",
                         container_tag=None, custom_id=None):
        self.add_calls.append({
            "content": content,
            "metadata": metadata,
            "entity_context": entity_context,
            "container_tag": container_tag,
            "custom_id": custom_id,
        })
        return {"id": "mem_123"}

    async def search_memories(self, query, *, limit=5, container_tag=None, search_mode=None):
        return self.search_results

    async def get_profile(self, query=None, *, container_tag=None):
        return self.profile_response

    async def forget_memory(self, memory_id, *, container_tag=None):
        self.forgotten_ids.append(memory_id)

    async def forget_by_query(self, query, *, container_tag=None):
        return self.forget_by_query_response

    async def ingest_conversation(self, session_id, messages, metadata=None):
        self.ingest_calls.append({"session_id": session_id, "messages": messages, "metadata": metadata})

    async def close(self):
        self.closed = True


@pytest_asyncio.fixture
async def provider(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    p = SupermemoryMemoryProvider()
    await p.initialize("session-1", hermes_home=str(tmp_path), platform="cli")
    try:
        yield p
    finally:
        await p.shutdown()


async def test_is_available_false_without_api_key(monkeypatch):
    monkeypatch.delenv("SUPERMEMORY_API_KEY", raising=False)
    p = SupermemoryMemoryProvider()
    assert await p.is_available() is False


async def test_load_and_save_config_round_trip(tmp_path):
    await _save_supermemory_config(
        {"container_tag": "demo-tag", "auto_capture": False},
        str(tmp_path),
    )
    cfg = await _load_supermemory_config(str(tmp_path))
    # container_tag is kept raw — sanitization happens in initialize() after template resolution
    assert cfg["container_tag"] == "demo-tag"
    assert cfg["auto_capture"] is False
    assert cfg["auto_recall"] is True


async def test_clean_text_for_capture_strips_injected_context():
    text = "hello\n<supermemory-context>ignore me</supermemory-context>\nworld"
    assert _clean_text_for_capture(text) == "hello\nworld"


async def test_format_prefetch_context_deduplicates_overlap():
    result = _format_prefetch_context(
        static_facts=["Jordan prefers short answers"],
        dynamic_facts=["Jordan prefers short answers", "Uses Hermes"],
        search_results=[{"memory": "Uses Hermes", "similarity": 0.9}],
        max_results=10,
    )
    assert result.count("Jordan prefers short answers") == 1
    assert result.count("Uses Hermes") == 1
    assert "<supermemory-context>" in result


async def test_prefetch_includes_profile_on_first_turn(provider):
    provider._client.profile_response = {
        "static": ["Jordan prefers short answers"],
        "dynamic": ["Current project is Supermemory provider"],
        "search_results": [{"memory": "Working on Hermes memory provider", "similarity": 0.88}],
    }
    await provider.on_turn_start(1, "start")
    result = await provider.prefetch("what am I working on?")
    assert "User Profile (Persistent)" in result
    assert "Recent Context" in result
    assert "Relevant Memories" in result


async def test_prefetch_propagates_cancellation(provider):
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked_profile(query=None, *, container_tag=None):
        started.set()
        await release.wait()

    provider._client.get_profile = blocked_profile
    task = asyncio.create_task(provider.prefetch("cancel this recall"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_sync_turn_buffers_short_messages(provider):
    # Trivial filtering is no longer applied at sync time — every non-empty turn
    # is buffered and only the full session is written at session boundaries.
    await provider.sync_turn("ok", "sure", session_id="session-1")
    assert provider._session_turns == [{"user": "ok", "assistant": "sure"}]
    assert provider._client.add_calls == []


async def test_on_session_end_ingests_clean_messages(provider):
    messages = [
        {"role": "system", "content": "skip"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    await provider.on_session_end(messages)
    assert len(provider._client.ingest_calls) == 1
    payload = provider._client.ingest_calls[0]
    assert payload["session_id"] == "session-1"
    assert payload["messages"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    assert payload["metadata"]["type"] == "full_session"
    assert payload["metadata"]["session_id"] == "session-1"
    assert payload["metadata"]["message_count"] == 2
    # Buffer is cleared after a normal session-end ingest.
    assert provider._session_turns == []


async def test_merge_metadata_stamps_sm_source():
    # sm_source routes Hermes writes into the "Hermes" Space in the Supermemory
    # app (functional routing, not telemetry) — must always be present.
    from plugins.memory.supermemory import _SupermemoryClient

    client = _SupermemoryClient.__new__(_SupermemoryClient)
    merged = client._merge_metadata({"type": "explicit_memory"})
    assert merged["sm_source"] == "hermes"
    assert merged["type"] == "explicit_memory"

    # Legacy "source" is migrated into "type" when type is absent.
    merged2 = client._merge_metadata({"source": "conversation_turn"})
    assert merged2["sm_source"] == "hermes"
    assert merged2["type"] == "conversation_turn"
    assert "source" not in merged2


async def test_shutdown_flushes_buffer_and_closes_client(provider):
    await provider.sync_turn(
        "Please remember this request in long-term memory",
        "Absolutely, I will keep that in long-term memory.",
        session_id="session-1",
    )
    assert len(provider._session_turns) == 1

    client = provider._client
    await provider.on_memory_write("add", "memory", "Jordan likes concise docs")
    await provider.shutdown()

    assert len(client.add_calls) == 1
    assert len(client.ingest_calls) == 1
    payload = client.ingest_calls[0]
    assert payload["session_id"] == "session-1"
    assert payload["metadata"]["partial"] is True
    assert payload["metadata"]["type"] == "full_session"
    assert client.closed is True
    assert provider._client is None


async def test_shutdown_closes_client_when_ingest_is_cancelled(provider):
    await provider.sync_turn(
        "Please remember this request in long-term memory",
        "Absolutely, I will keep that in long-term memory.",
        session_id="session-1",
    )
    client = provider._client
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked_ingest(session_id, messages, metadata=None):
        started.set()
        await release.wait()

    client.ingest_conversation = blocked_ingest
    task = asyncio.create_task(provider.shutdown())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert client.closed is True
    assert provider._client is None
    assert provider._active is False


async def test_initialize_closes_partial_client_when_cancelled(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    started = asyncio.Event()
    release = asyncio.Event()
    clients = []

    class BlockedClient(FakeClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            clients.append(self)

        async def initialize(self):
            started.set()
            await release.wait()

    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", BlockedClient)
    candidate = SupermemoryMemoryProvider()
    task = asyncio.create_task(
        candidate.initialize("session-1", hermes_home=str(tmp_path), platform="cli")
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert clients[0].closed is True
    assert candidate._client is None
    assert candidate._active is False


async def test_store_tool_returns_saved_payload(provider):
    result = json.loads(
        await provider.handle_tool_call(
            "supermemory_store",
            {"content": "Jordan likes concise docs"},
        )
    )
    assert result["saved"] is True
    assert result["id"] == "mem_123"


async def test_search_tool_formats_results(provider):
    provider._client.search_results = [
        {"id": "m1", "memory": "Jordan likes concise docs", "similarity": 0.92}
    ]
    result = json.loads(
        await provider.handle_tool_call(
            "supermemory_search",
            {"query": "concise docs"},
        )
    )
    assert result["count"] == 1
    assert result["results"][0]["similarity"] == 92


async def test_forget_tool_by_id(provider):
    result = json.loads(
        await provider.handle_tool_call("supermemory_forget", {"id": "m1"})
    )
    assert result == {"forgotten": True, "id": "m1"}
    assert provider._client.forgotten_ids == ["m1"]


async def test_profile_tool_formats_sections(provider):
    provider._client.profile_response = {
        "static": ["Jordan prefers concise docs"],
        "dynamic": ["Working on Supermemory provider"],
        "search_results": [],
    }
    result = json.loads(
        await provider.handle_tool_call("supermemory_profile", {})
    )
    assert result["static_count"] == 1
    assert result["dynamic_count"] == 1
    assert "User Profile (Persistent)" in result["profile"]


async def test_handle_tool_call_returns_error_when_unconfigured(monkeypatch):
    monkeypatch.delenv("SUPERMEMORY_API_KEY", raising=False)
    p = SupermemoryMemoryProvider()
    result = json.loads(
        await p.handle_tool_call("supermemory_search", {"query": "x"})
    )
    assert "error" in result


# -- Identity template tests --------------------------------------------------


async def test_identity_template_resolved_in_container_tag(monkeypatch, tmp_path):
    """container_tag with {identity} resolves to profile-scoped tag."""
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    await _save_supermemory_config(
        {"container_tag": "hermes-{identity}"},
        str(tmp_path),
    )
    p = SupermemoryMemoryProvider()
    await p.initialize(
        "s1",
        hermes_home=str(tmp_path),
        platform="cli",
        agent_identity="coder",
    )
    assert p._container_tag == "hermes_coder"
    await p.shutdown()


async def test_container_tag_env_var_override(monkeypatch, tmp_path):
    """SUPERMEMORY_CONTAINER_TAG env var overrides config."""
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setenv("SUPERMEMORY_CONTAINER_TAG", "env-override")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    p = SupermemoryMemoryProvider()
    await p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
    assert p._container_tag == "env_override"
    await p.shutdown()


# -- Search mode tests --------------------------------------------------------


async def test_invalid_search_mode_falls_back_to_default(monkeypatch, tmp_path):
    """Invalid search_mode falls back to 'hybrid'."""
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    await _save_supermemory_config({"search_mode": "invalid_mode"}, str(tmp_path))
    p = SupermemoryMemoryProvider()
    await p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
    assert p._search_mode == "hybrid"
    await p.shutdown()


# -- Base URL tests -------------------------------------------------------------


async def test_base_url_defaults_to_cloud(monkeypatch, tmp_path):
    """Without config or env override, the client targets api.supermemory.ai."""
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.delenv("SUPERMEMORY_BASE_URL", raising=False)
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    p = SupermemoryMemoryProvider()
    await p.initialize("s1", hermes_home=str(tmp_path), platform="cli")
    assert p._base_url == "https://api.supermemory.ai"
    assert p._client.base_url == "https://api.supermemory.ai"
    await p.shutdown()


async def test_client_passes_custom_base_url_to_sdk(monkeypatch):
    """SDK operations and raw conversation ingest share one normalized base URL."""
    import sys
    import types

    from plugins.memory.supermemory import _SupermemoryClient

    captured = {}

    class StubSupermemory:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def close(self):
            return None

    module = types.ModuleType("supermemory")
    module.AsyncSupermemory = StubSupermemory
    monkeypatch.setitem(sys.modules, "supermemory", module)
    http_client = AsyncMock()
    http_client.is_closed = False
    monkeypatch.setattr(
        "plugins.memory.supermemory._create_httpx_client",
        AsyncMock(return_value=http_client),
    )

    client = _SupermemoryClient(
        api_key="test-key",
        timeout=1.0,
        container_tag="hermes",
        base_url="http://localhost:6767/",
    )
    await client.initialize()

    assert client._base_url == "http://localhost:6767"
    assert captured["base_url"] == "http://localhost:6767"
    assert captured["http_client"] is http_client
    await client.close()
    http_client.aclose.assert_awaited_once()


async def test_real_async_sdk_transport_contract(monkeypatch):
    pytest.importorskip("supermemory")
    from plugins.memory.supermemory import _SupermemoryClient

    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(
            (
                request.method,
                request.url.path,
                json.loads(request.content or b"{}"),
            )
        )
        payloads = {
            "/v3/documents": {"id": "doc-real-sdk"},
            "/v4/search": {"results": []},
            "/v4/profile": {
                "profile": {"static": [], "dynamic": []},
                "searchResults": {"results": []},
            },
            "/v4/memories": {},
        }
        if request.url.path == "/v4/conversations":
            return httpx.Response(204, request=request)
        return httpx.Response(
            200,
            json=payloads[request.url.path],
            request=request,
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(
        "plugins.memory.supermemory._create_httpx_client",
        AsyncMock(return_value=http_client),
    )
    client = _SupermemoryClient(
        api_key="test-key",
        timeout=1.0,
        container_tag="hermes",
        base_url="https://mock.supermemory.test",
    )
    await client.initialize()
    try:
        assert await client.add_memory(
            "native async memory",
            {"type": "fact"},
        ) == {"id": "doc-real-sdk"}
        assert await client.search_memories("native async") == []
        assert await client.get_profile("native async") == {
            "static": [],
            "dynamic": [],
            "search_results": [],
        }
        await client.forget_memory("doc-real-sdk")
        await client.ingest_conversation(
            "session-1",
            [{"role": "user", "content": "hello"}],
        )
    finally:
        await client.close()

    assert [(method, path) for method, path, _ in requests] == [
        ("POST", "/v3/documents"),
        ("POST", "/v4/search"),
        ("POST", "/v4/profile"),
        ("DELETE", "/v4/memories"),
        ("POST", "/v4/conversations"),
    ]
    assert requests[0][2] == {
        "content": "native async memory",
        "containerTags": ["hermes"],
        "metadata": {"sm_source": "hermes", "type": "fact"},
    }
    assert http_client.is_closed


@pytest.mark.parametrize(
    ("base_url", "expected_url"),
    [
        ("https://api.supermemory.ai", "https://api.supermemory.ai/v4/conversations"),
        ("http://localhost:6767", "http://localhost:6767/v4/conversations"),
    ],
)
async def test_ingest_conversation_uses_client_base_url(base_url, expected_url):
    """Raw conversation ingest follows the same endpoint as SDK operations."""
    from plugins.memory.supermemory import _SupermemoryClient

    client = _SupermemoryClient.__new__(_SupermemoryClient)
    client._api_key = "test-key"
    client._container_tag = "hermes"
    client._timeout = 1.0
    client._base_url = base_url

    captured = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["Authorization"]
        captured["source"] = request.headers["x-sm-source"]
        captured["payload"] = json.loads(request.content)
        return httpx.Response(204, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client._http_client = http_client
        await client.ingest_conversation(
            "s1",
            [{"role": "user", "content": "hello there"}],
        )

    assert captured["url"] == expected_url
    assert captured["authorization"] == "Bearer test-key"
    assert captured["source"] == "hermes"
    assert captured["payload"] == {
        "conversationId": "s1",
        "messages": [{"role": "user", "content": "hello there"}],
        "containerTags": ["hermes"],
    }


# -- Multi-container tests ----------------------------------------------------


async def test_multi_container_disabled_by_default(provider):
    """Multi-container is off by default; schemas have no container_tag param."""
    assert provider._enable_custom_containers is False
    schemas = provider.get_tool_schemas()
    for s in schemas:
        assert "container_tag" not in s["parameters"]["properties"]


async def test_get_config_schema_minimal():
    """get_config_schema only returns the API key field."""
    p = SupermemoryMemoryProvider()
    schema = p.get_config_schema()
    assert len(schema) == 1
    assert schema[0]["key"] == "api_key"
    assert schema[0]["secret"] is True


async def test_probe_supermemory_connection_missing_key(tmp_path):
    status = await _probe_supermemory_connection("", str(tmp_path))
    assert status["ok"] is False
    assert status["error"] == "SUPERMEMORY_API_KEY not set"
    assert status["container_tag"] == "hermes"


async def test_provider_discovery_and_memory_manager_coroutine_contract():
    loaded = await load_memory_provider("supermemory")
    assert loaded is not None
    assert loaded.name == "supermemory"

    manager = MemoryManager()
    manager.add_provider(loaded)
    assert manager.has_tool("supermemory_store")
    assert manager.has_tool("supermemory-search")
    await manager.shutdown_all()


async def test_concurrent_profile_initialization_isolates_api_keys(
    monkeypatch,
    tmp_path,
):
    monkeypatch.delenv("SUPERMEMORY_API_KEY", raising=False)
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)

    async def initialize_profile(name: str, api_key: str):
        token = set_secret_scope({"SUPERMEMORY_API_KEY": api_key})
        try:
            candidate = SupermemoryMemoryProvider()
            await candidate.initialize(
                f"session-{name}",
                hermes_home=str(tmp_path / name),
                platform="cli",
                agent_identity=name,
            )
            await asyncio.sleep(0)
            return candidate
        finally:
            reset_secret_scope(token)

    first, second = await asyncio.gather(
        initialize_profile("alpha", "key-alpha"),
        initialize_profile("beta", "key-beta"),
    )
    try:
        assert first._client.api_key == "key-alpha"
        assert second._client.api_key == "key-beta"
        assert first._hermes_home != second._hermes_home
    finally:
        await asyncio.gather(first.shutdown(), second.shutdown())


async def test_native_provider_lifecycle_does_not_block_or_leak_tasks(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)

    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        candidate = SupermemoryMemoryProvider()
        await candidate.initialize(
            "session-1",
            hermes_home=str(tmp_path),
            platform="cli",
        )
        await candidate.on_turn_start(1, "hello")
        await candidate.prefetch("what do you remember?", session_id="session-1")
        await candidate.sync_turn("hello", "hi", session_id="session-1")
        await candidate.on_memory_write("add", "memory", "Remember this")
        await candidate.on_session_end(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ]
        )
        await candidate.shutdown()


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits not enforced on Windows")
async def test_save_config_sets_owner_only_permissions(tmp_path):
    """supermemory.json must be written with 0o600 so API key is not world-readable."""
    await _save_supermemory_config({"api_key": "sm-test-key"}, str(tmp_path))
    config_file = tmp_path / "supermemory.json"
    assert config_file.exists()
    mode = stat.S_IMODE(config_file.stat().st_mode)
    assert mode == 0o600, f"Expected 0o600 (owner-only), got {oct(mode)}"


@pytest.mark.skipif(os.name == "nt", reason="symlink behavior differs on Windows")
async def test_save_config_preserves_existing_symlink(tmp_path):
    target_dir = tmp_path / "managed"
    target_dir.mkdir()
    target = target_dir / "supermemory.json"
    target.write_text('{"auto_recall": false}', encoding="utf-8")
    link_home = tmp_path / "profile"
    link_home.mkdir()
    link = link_home / "supermemory.json"
    link.symlink_to(target)

    await _save_supermemory_config({"container_tag": "profile-tag"}, str(link_home))

    assert link.is_symlink()
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "auto_recall": False,
        "container_tag": "profile-tag",
    }
