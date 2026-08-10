"""Native-async parity, durability, and lifecycle coverage for RetainDB."""

from __future__ import annotations

import asyncio
import inspect
import json
import mimetypes

import aiofiles
import aiosqlite
import pytest
import pytest_asyncio
from aiohttp import web
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

from agent.memory_manager import MemoryManager
from plugins.memory import load_memory_provider
from plugins.memory.retaindb import (
    CONTEXT_SCHEMA,
    FILE_DELETE_SCHEMA,
    FILE_INGEST_SCHEMA,
    FILE_LIST_SCHEMA,
    FILE_READ_SCHEMA,
    FILE_UPLOAD_SCHEMA,
    FORGET_SCHEMA,
    PROFILE_SCHEMA,
    REMEMBER_SCHEMA,
    SEARCH_SCHEMA,
    RetainDBMemoryProvider,
    _build_overlay,
    _Client,
    _WriteQueue,
)

pytestmark = pytest.mark.asyncio


class FakeClient:
    def __init__(
        self,
        api_key: str = "test-key",
        base_url: str = "http://retaindb.test",
        project: str = "test-project",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.project = project
        self.calls: list[tuple[str, tuple, dict]] = []
        self.closed = False
        self.ingest_gate: asyncio.Event | None = None

    async def _record(self, name: str, *args, result=None, **kwargs):
        self.calls.append((name, args, kwargs))
        return result if result is not None else {"ok": True}

    async def query_context(self, *args, **kwargs):
        return await self._record(
            "query_context",
            *args,
            result={"results": [{"content": "Relevant memory"}]},
            **kwargs,
        )

    async def search(self, *args, **kwargs):
        return await self._record(
            "search",
            *args,
            result={"results": [{"id": "memory-1"}]},
            **kwargs,
        )

    async def get_profile(self, *args, **kwargs):
        return await self._record(
            "get_profile",
            *args,
            result={"memories": [{"content": "Profile memory"}]},
            **kwargs,
        )

    async def add_memory(self, *args, **kwargs):
        return await self._record(
            "add_memory",
            *args,
            result={"id": "memory-added"},
            **kwargs,
        )

    async def delete_memory(self, *args, **kwargs):
        return await self._record(
            "delete_memory",
            *args,
            result={"deleted": True},
            **kwargs,
        )

    async def ingest_session(self, *args, **kwargs):
        if self.ingest_gate is not None:
            await self.ingest_gate.wait()
        return await self._record("ingest_session", *args, **kwargs)

    async def ask_user(self, *args, **kwargs):
        return await self._record(
            "ask_user",
            *args,
            result={"answer": "Dialectic answer"},
            **kwargs,
        )

    async def get_agent_model(self, *args, **kwargs):
        return await self._record(
            "get_agent_model",
            *args,
            result={
                "memory_count": 2,
                "persona": "Careful",
                "persistent_instructions": ["Keep parity"],
                "working_style": "Async",
            },
            **kwargs,
        )

    async def seed_agent_identity(self, *args, **kwargs):
        return await self._record("seed_agent_identity", *args, **kwargs)

    async def upload_file(self, *args, **kwargs):
        return await self._record(
            "upload_file",
            *args,
            result={"file": {"id": "file-1", "name": args[1]}},
            **kwargs,
        )

    async def list_files(self, *args, **kwargs):
        return await self._record(
            "list_files",
            *args,
            result={"files": [{"id": "file-1"}]},
            **kwargs,
        )

    async def get_file(self, *args, **kwargs):
        return await self._record(
            "get_file",
            *args,
            result={
                "file": {
                    "id": args[0],
                    "name": "note.md",
                    "mime_type": "text/markdown",
                    "rdb_uri": "rdb://note.md",
                }
            },
            **kwargs,
        )

    async def read_file_content(self, *args, **kwargs):
        return await self._record(
            "read_file_content",
            *args,
            result=b"stored text",
            **kwargs,
        )

    async def ingest_file(self, *args, **kwargs):
        return await self._record(
            "ingest_file",
            *args,
            result={"ingested": True},
            **kwargs,
        )

    async def delete_file(self, *args, **kwargs):
        return await self._record(
            "delete_file",
            *args,
            result={"deleted": True},
            **kwargs,
        )

    async def close(self) -> None:
        self.closed = True


@pytest_asyncio.fixture(autouse=True)
async def isolate_profile(tmp_path, monkeypatch):
    for name in (
        "RETAINDB_API_KEY",
        "RETAINDB_BASE_URL",
        "RETAINDB_PROJECT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


async def _make_provider(tmp_path, monkeypatch, **config):
    config_path = tmp_path / "config.yaml"
    provider_config = {
        "base_url": "http://retaindb.test",
        "project": "test-project",
        **config,
    }
    async with aiofiles.open(config_path, "w", encoding="utf-8") as handle:
        await handle.write(
            "memory:\n"
            "  provider: retaindb\n"
            "  retaindb:\n"
            f"    base_url: {provider_config['base_url']}\n"
            f"    project: {provider_config['project']}\n"
        )
    fake = FakeClient()

    def factory(api_key, base_url, project):
        fake.api_key = api_key
        fake.base_url = base_url
        fake.project = project
        return fake

    monkeypatch.setenv("RETAINDB_API_KEY", "profile-key")
    monkeypatch.setattr("plugins.memory.retaindb._Client", factory)
    provider = RetainDBMemoryProvider()
    await provider.initialize(
        "session-1",
        hermes_home=str(tmp_path),
        user_id="user-1",
        agent_id="agent-1",
    )
    return provider, fake


async def test_public_api_and_schemas_preserve_upstream_contract():  # noqa: ASYNC124
    expected = {
        "is_available": ("self",),
        "initialize": ("self", "session_id"),
        "prefetch": ("self", "query", "session_id"),
        "queue_prefetch": ("self", "query", "session_id"),
        "sync_turn": ("self", "user_content", "assistant_content", "session_id"),
        "handle_tool_call": ("self", "tool_name", "args"),
        "on_memory_write": ("self", "action", "target", "content"),
        "shutdown": ("self",),
    }
    for name, parameters in expected.items():
        method = getattr(RetainDBMemoryProvider, name)
        assert inspect.iscoroutinefunction(method)
        assert (
            tuple(inspect.signature(method).parameters)[: len(parameters)] == parameters
        )

    assert [
        schema["name"] for schema in RetainDBMemoryProvider().get_tool_schemas()
    ] == [
        PROFILE_SCHEMA["name"],
        SEARCH_SCHEMA["name"],
        CONTEXT_SCHEMA["name"],
        REMEMBER_SCHEMA["name"],
        FORGET_SCHEMA["name"],
        FILE_UPLOAD_SCHEMA["name"],
        FILE_LIST_SCHEMA["name"],
        FILE_READ_SCHEMA["name"],
        FILE_INGEST_SCHEMA["name"],
        FILE_DELETE_SCHEMA["name"],
    ]


async def test_overlay_preserves_deduplication_limits_and_format():  # noqa: ASYNC124
    profile = {
        "memories": [
            {"content": "Already local"},
            *({"content": f"Profile {index}"} for index in range(7)),
        ]
    }
    query = {
        "results": [
            {"content": "Profile 0"},
            {"content": "Query result"},
        ]
    }
    result = _build_overlay(profile, query, local_entries=["Already local"])
    assert result.startswith("[RetainDB Context]\nProfile:\n")
    assert "Already local" not in result
    assert result.count("Profile 0") == 1
    assert "Profile 5" not in result
    assert "- Query result" in result
    assert _build_overlay({}, {}) == ""


async def test_client_fallbacks_preserve_routes_and_cancellation(monkeypatch):
    client = _Client("key", "https://api.test///", "project")
    assert client.base_url == "https://api.test"
    calls: list[tuple[str, str, dict]] = []

    async def request(method, path, **kwargs):
        calls.append((method, path, kwargs))
        if len(calls) in {1, 3, 5}:
            raise RuntimeError("legacy fallback")
        return {"ok": True}

    monkeypatch.setattr(client, "request", request)
    assert await client.get_profile("u/1") == {"ok": True}
    assert await client.add_memory("u", "s", "fact") == {"ok": True}
    assert await client.delete_memory("m/1") == {"ok": True}
    assert [path for _, path, _ in calls] == [
        "/v1/memory/profile/u%2F1",
        "/v1/memories",
        "/v1/memory",
        "/v1/memories",
        "/v1/memory/m%2F1",
        "/v1/memories/m%2F1",
    ]

    async def cancelled(*args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(client, "request", cancelled)
    with pytest.raises(asyncio.CancelledError):
        await client.add_memory("u", "s", "fact")


async def test_real_http_transport_routes_headers_files_and_close():
    requests: list[dict] = []

    async def handler(request):
        record = {
            "method": request.method,
            "path": request.path,
            "raw_path": request.raw_path,
            "authorization": request.headers.get("Authorization"),
            "api_key": request.headers.get("X-API-Key"),
            "runtime": request.headers.get("x-sdk-runtime"),
        }
        if request.path == "/v1/files" and request.method == "POST":
            reader = await request.multipart()
            fields = {}
            while part := await reader.next():
                fields[part.name] = await part.read()
            record["fields"] = fields
            requests.append(record)
            return web.json_response({"file": {"id": "file-real"}})
        if request.path.endswith("/content"):
            requests.append(record)
            return web.Response(body=b"real content")
        try:
            record["body"] = await request.json()
        except Exception:
            record["body"] = None
        requests.append(record)
        return web.json_response({"results": []})

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        client = _Client(
            "Bearer real-key",
            f"http://127.0.0.1:{port}",
            "real-project",
        )
        try:
            await client.query_context("user", "session", "query")
            uploaded = await client.upload_file(
                b"payload",
                "note.md",
                "/notes/note.md",
                "text/markdown",
                "project",
                "project-id",
            )
            content = await client.read_file_content("file/real")
        finally:  # noqa: ASYNC102 - both transport owners must close
            await client.close()
            await runner.cleanup()

    assert uploaded == {"file": {"id": "file-real"}}
    assert content == b"real content"
    context = requests[0]
    assert context == {
        "method": "POST",
        "path": "/v1/context/query",
        "raw_path": "/v1/context/query",
        "authorization": "Bearer real-key",
        "api_key": "real-key",
        "runtime": "hermes-plugin",
        "body": {
            "project": "real-project",
            "query": "query",
            "user_id": "user",
            "session_id": "session",
            "include_memories": True,
            "max_tokens": 1200,
        },
    }
    upload = requests[1]
    assert upload["api_key"] is None
    assert upload["fields"]["file"] == b"payload"
    assert upload["fields"]["path"] == b"/notes/note.md"
    assert requests[2]["raw_path"] == "/v1/files/file%2Freal/content"
    assert client._http_client is None


async def test_http_error_shape_matches_upstream():
    async def failure(request):  # noqa: ASYNC124
        return web.json_response({"message": "denied"}, status=403)

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", failure)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    client = _Client("key", f"http://127.0.0.1:{port}", "project")
    try:
        with pytest.raises(
            RuntimeError,
            match=r"RetainDB POST /v1/context/query failed \(403\): denied",
        ):
            await client.query_context("user", "session", "query")
    finally:
        await client.close()
        await runner.cleanup()


async def test_http_cancellation_and_timeout_propagate_without_task_leaks():
    entered = asyncio.Event()
    releases = [asyncio.Event(), asyncio.Event()]
    request_count = 0

    async def delayed(request):
        nonlocal request_count
        release = releases[request_count]
        request_count += 1
        entered.set()
        await release.wait()
        return web.json_response({"results": []})

    app = web.Application()
    app.router.add_post("/v1/context/query", delayed)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    client = _Client("key", f"http://127.0.0.1:{port}", "project")
    try:
        async with no_task_leaks(action=LeakAction.RAISE):
            request = asyncio.create_task(
                client.query_context("user", "session", "cancel"),
                name="retaindb-cancel-test",
            )
            await entered.wait()
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            releases[0].set()
            await asyncio.sleep(0)

            import httpx

            with pytest.raises(httpx.ReadTimeout):
                await client.request(
                    "POST",
                    "/v1/context/query",
                    json_body={},
                    timeout=0.01,
                )
            releases[1].set()
            await asyncio.sleep(0)
    finally:
        for release in releases:
            release.set()
        await client.close()
        await runner.cleanup()


async def test_request_preserves_requests_ok_semantics_for_3xx():
    class RedirectResponse:
        status_code = 304
        text = ""

        def json(self):
            raise ValueError("no body")

    class FakeHttpClient:
        closed = False

        async def request(self, *args, **kwargs):  # noqa: ASYNC124
            return RedirectResponse()

        async def aclose(self):
            self.closed = True

    client = _Client("key", "https://api.test", "project")
    fake = FakeHttpClient()
    client._http_client = fake
    assert await client.request("GET", "/not-modified") == ""
    await client.close()
    assert fake.closed is True


async def test_write_queue_durably_enqueues_drains_and_closes(tmp_path):
    client = FakeClient()
    path = tmp_path / "queue.db"
    queue = _WriteQueue(client, path)
    await queue.initialize()
    await queue.enqueue("user", "session", [{"role": "user", "content": "hi"}])
    await queue.shutdown()

    async with aiosqlite.connect(path) as conn:
        cursor = await conn.execute("SELECT COUNT(*) FROM pending")
        count = (await cursor.fetchone())[0]
        await cursor.close()
    assert count == 0
    assert client.calls[0][0] == "ingest_session"
    assert queue._conn is None
    assert queue._writer_task is None


async def test_write_queue_preserves_fifo_dispatch_order(tmp_path):
    client = FakeClient()
    queue = _WriteQueue(client, tmp_path / "fifo.db")
    await queue.initialize()
    for index in range(4):
        await queue.enqueue(
            "user",
            f"session-{index}",
            [{"role": "user", "content": str(index)}],
        )
    await queue._q.join()
    await queue.shutdown()
    assert [call[1][1] for call in client.calls if call[0] == "ingest_session"] == [
        f"session-{index}" for index in range(4)
    ]


async def test_write_queue_replays_checkpoint_after_restart(tmp_path):
    path = tmp_path / "recovery.db"
    async with aiosqlite.connect(path) as conn:
        await conn.execute(
            """CREATE TABLE pending (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT, session_id TEXT, messages_json TEXT,
                created_at TEXT, last_error TEXT
            )"""
        )
        await conn.execute(
            "INSERT INTO pending "
            "(user_id, session_id, messages_json, created_at) VALUES (?,?,?,?)",
            ("user", "session", '[{"role":"user","content":"replay"}]', "now"),
        )
        await conn.commit()

    client = FakeClient()
    queue = _WriteQueue(client, path)
    await queue.initialize()
    await queue._q.join()
    await queue.shutdown()
    assert client.calls == [
        (
            "ingest_session",
            ("user", "session", [{"role": "user", "content": "replay"}]),
            {},
        )
    ]


async def test_write_queue_closes_database_when_checkpoint_is_malformed(tmp_path):
    path = tmp_path / "malformed.db"
    async with aiosqlite.connect(path) as conn:
        await conn.execute(
            """CREATE TABLE pending (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT, session_id TEXT, messages_json TEXT,
                created_at TEXT, last_error TEXT
            )"""
        )
        await conn.execute(
            "INSERT INTO pending "
            "(user_id, session_id, messages_json, created_at) VALUES (?,?,?,?)",
            ("user", "session", "not-json", "now"),
        )
        await conn.commit()

    queue = _WriteQueue(FakeClient(), path)
    with pytest.raises(json.JSONDecodeError):
        await queue.initialize()
    assert queue._conn is None
    assert queue._writer_task is None


async def test_queue_shutdown_drains_inflight_write_and_rejects_late_enqueue(
    tmp_path,
):
    client = FakeClient()
    client.ingest_gate = asyncio.Event()
    queue = _WriteQueue(client, tmp_path / "queue.db")
    await queue.initialize()
    await queue.enqueue("user", "session", [{"role": "user"}])
    shutdown = asyncio.create_task(queue.shutdown())
    await asyncio.sleep(0)
    assert not shutdown.done()
    with pytest.raises(RuntimeError, match="shutting down"):
        await queue.enqueue("late", "session", [])
    client.ingest_gate.set()
    await shutdown


async def test_queue_shutdown_cannot_overtake_inflight_enqueue(
    tmp_path,
    monkeypatch,
):
    client = FakeClient()
    queue = _WriteQueue(client, tmp_path / "enqueue-race.db")
    await queue.initialize()
    insert_started = asyncio.Event()
    release_insert = asyncio.Event()
    original_execute = queue._conn.execute

    async def gated_execute(sql, parameters=None):
        if sql.startswith("INSERT INTO pending"):
            insert_started.set()
            await release_insert.wait()
        if parameters is None:
            return await original_execute(sql)
        return await original_execute(sql, parameters)

    monkeypatch.setattr(queue._conn, "execute", gated_execute)
    enqueue = asyncio.create_task(queue.enqueue("user", "session", [{"role": "user"}]))
    await insert_started.wait()
    shutdown = asyncio.create_task(queue.shutdown())
    await asyncio.sleep(0)
    release_insert.set()
    await enqueue
    await shutdown

    assert [call[0] for call in client.calls] == ["ingest_session"]


async def test_initialize_config_env_precedence_and_state_only_constructor(
    tmp_path,
    monkeypatch,
):
    provider = RetainDBMemoryProvider()
    assert provider._client is None
    assert provider._queue is None

    async def config():  # noqa: ASYNC124
        return {
            "base_url": "https://config.example/",
            "project": "config-project",
        }

    fake = FakeClient()

    def factory(api_key, base_url, project):
        fake.api_key = api_key
        fake.base_url = base_url
        fake.project = project
        return fake

    monkeypatch.setattr("plugins.memory.retaindb._load_retaindb_config", config)
    monkeypatch.setattr("plugins.memory.retaindb._Client", factory)
    monkeypatch.setenv("RETAINDB_API_KEY", "scoped-key")
    monkeypatch.setenv("RETAINDB_BASE_URL", "https://env.example/")
    await provider.initialize(
        "session",
        hermes_home=str(tmp_path / "profile-name"),
    )
    assert (fake.api_key, fake.base_url, fake.project) == (
        "scoped-key",
        "https://env.example",
        "config-project",
    )
    await provider.shutdown()
    assert fake.closed is True


async def test_prefetch_is_one_turn_delayed_and_preserves_exact_format(
    tmp_path,
    monkeypatch,
):
    provider, fake = await _make_provider(tmp_path, monkeypatch)
    assert await provider.prefetch("first") == ""
    await provider.queue_prefetch("remembered query")
    await asyncio.gather(*tuple(provider._owned_tasks))
    result = await provider.prefetch("next")
    assert result == (
        "[RetainDB Context]\n"
        "Profile:\n"
        "- Profile memory\n"
        "Relevant memories:\n"
        "- Relevant memory\n\n"
        "[RetainDB User Synthesis]\n"
        "Dialectic answer\n\n"
        "[RetainDB Agent Self-Model]\n"
        "Persona: Careful\n"
        "Instructions:\n"
        "- Keep parity\n"
        "Working style: Async"
    )
    assert await provider.prefetch("consumed") == ""
    ask = next(call for call in fake.calls if call[0] == "ask_user")
    assert ask[2] == {"reasoning_level": "low"}
    await provider.shutdown()


async def test_tools_preserve_results_arguments_and_file_security(
    tmp_path,
    monkeypatch,
):
    provider, fake = await _make_provider(tmp_path, monkeypatch)
    note = tmp_path / "note.md"
    async with aiofiles.open(note, "w", encoding="utf-8") as handle:
        await handle.write("# Note\n")

    search = json.loads(
        await provider.handle_tool_call(
            "retaindb_search",
            {"query": "topic", "top_k": 99},
        )
    )
    context = json.loads(
        await provider.handle_tool_call(
            "retaindb_context",
            {"query": "topic"},
        )
    )
    remembered = json.loads(
        await provider.handle_tool_call(
            "retaindb_remember",
            {
                "content": "Remember this",
                "memory_type": "preference",
                "importance": 0.9,
            },
        )
    )
    uploaded = json.loads(
        await provider.handle_tool_call(
            "retaindb_upload_file",
            {"local_path": str(note), "ingest": True},
        )
    )
    read = json.loads(
        await provider.handle_tool_call(
            "retaindb_read_file",
            {"file_id": "file-1"},
        )
    )
    assert search == {"results": [{"id": "memory-1"}]}
    assert context["context"].startswith("[RetainDB Context]")
    assert remembered == {"id": "memory-added"}
    assert uploaded["ingest"] == {"ingested": True}
    assert read["content"] == "stored text"
    search_call = next(call for call in fake.calls if call[0] == "search")
    assert search_call[2] == {"top_k": 20}
    upload_call = next(call for call in fake.calls if call[0] == "upload_file")
    assert upload_call[1][:5] == (
        b"# Note\n",
        "note.md",
        "/note.md",
        mimetypes.guess_type("note.md")[0] or "application/octet-stream",
        "PROJECT",
    )

    auth = tmp_path / "auth.json"
    async with aiofiles.open(auth, "w", encoding="utf-8") as handle:
        await handle.write('{"RETAINDB_API_KEY":"secret"}')
    import agent.file_safety as file_safety

    monkeypatch.setattr(file_safety, "_hermes_home_path", lambda: tmp_path)
    blocked = await provider._dispatch(
        "retaindb_upload_file",
        {"local_path": str(auth)},
    )
    assert "credential store" in blocked["error"]
    await provider.shutdown()


async def test_missing_tool_arguments_and_unknown_tool_preserve_shapes(
    tmp_path,
    monkeypatch,
):
    provider, _ = await _make_provider(tmp_path, monkeypatch)
    for tool_name, key in (
        ("retaindb_search", "query"),
        ("retaindb_context", "query"),
        ("retaindb_remember", "content"),
        ("retaindb_forget", "memory_id"),
        ("retaindb_upload_file", "local_path"),
        ("retaindb_read_file", "file_id"),
        ("retaindb_ingest_file", "file_id"),
        ("retaindb_delete_file", "file_id"),
    ):
        assert await provider._dispatch(tool_name, {}) == {
            "error": f"{key} is required"
        }
    assert await provider._dispatch("unknown", {}) == {"error": "Unknown tool: unknown"}
    await provider.shutdown()


async def test_sync_turn_memory_mirror_and_soul_seed(
    tmp_path,
    monkeypatch,
):
    async with aiofiles.open(tmp_path / "SOUL.md", "w", encoding="utf-8") as handle:
        await handle.write("Stay precise.")
    provider, fake = await _make_provider(tmp_path, monkeypatch)
    await asyncio.gather(*tuple(provider._owned_tasks))
    await provider.sync_turn("User turn", "Assistant turn")
    await provider._queue._q.join()
    await provider.on_memory_write("add", "user", "Prefers dark mode")
    await provider.on_memory_write("remove", "user", "ignored")
    names = [call[0] for call in fake.calls]
    assert "seed_agent_identity" in names
    ingest = next(call for call in fake.calls if call[0] == "ingest_session")
    assert ingest[1][0:2] == ("user-1", "session-1")
    assert [message["role"] for message in ingest[1][2]] == ["user", "assistant"]
    mirror = next(call for call in fake.calls if call[0] == "add_memory")
    assert mirror[2] == {"memory_type": "preference"}
    await provider.shutdown()


async def test_shutdown_finishes_cleanup_then_propagates_cancellation(
    tmp_path,
    monkeypatch,
):
    provider, fake = await _make_provider(tmp_path, monkeypatch)
    gate = asyncio.Event()

    async def background():
        await gate.wait()

    task = asyncio.create_task(background(), name="retaindb-test-background")
    provider._track_task(task)
    shutdown = asyncio.create_task(provider.shutdown())
    await asyncio.sleep(0)
    shutdown.cancel()
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await shutdown
    assert fake.closed is True
    assert provider._client is None
    assert provider._queue is None


async def test_shutdown_prevents_prefetch_spawn_after_waiting_for_prior_task(
    tmp_path,
    monkeypatch,
):
    provider, fake = await _make_provider(tmp_path, monkeypatch)
    gate = asyncio.Event()
    prior = asyncio.create_task(gate.wait(), name="retaindb-prior-prefetch")
    provider._prefetch_tasks = [prior]
    provider._track_task(prior)

    prefetch = asyncio.create_task(provider.queue_prefetch("new query"))
    await asyncio.sleep(0)
    shutdown = asyncio.create_task(provider.shutdown())
    await asyncio.sleep(0)
    gate.set()
    await prefetch
    await shutdown

    assert not any(
        call[0] in {"query_context", "ask_user", "get_agent_model"}
        for call in fake.calls
    )
    assert provider._owned_tasks == set()


async def test_discovery_and_native_lifecycle_do_not_block_or_leak(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("RETAINDB_API_KEY", "key")
    loaded = await load_memory_provider("retaindb")
    assert loaded is not None
    assert loaded.name == "retaindb"
    await loaded.shutdown()

    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        provider, _ = await _make_provider(tmp_path, monkeypatch)
        await provider.queue_prefetch("query")
        await provider.sync_turn("user", "assistant")
        await provider.shutdown()


async def test_memory_manager_end_to_end_routes_recall_tools_and_writes(
    tmp_path,
    monkeypatch,
):
    async with aiofiles.open(tmp_path / "config.yaml", "w", encoding="utf-8") as handle:
        await handle.write(
            "memory:\n"
            "  provider: retaindb\n"
            "  retaindb:\n"
            "    base_url: http://retaindb.test\n"
            "    project: manager-project\n"
        )
    fake = FakeClient()

    def factory(api_key, base_url, project):
        fake.api_key = api_key
        fake.base_url = base_url
        fake.project = project
        return fake

    monkeypatch.setenv("RETAINDB_API_KEY", "manager-key")
    monkeypatch.setattr("plugins.memory.retaindb._Client", factory)
    provider = RetainDBMemoryProvider()
    manager = MemoryManager()
    manager.add_provider(provider)
    await manager.initialize_all(
        "manager-session",
        hermes_home=str(tmp_path),
        user_id="manager-user",
        agent_id="manager-agent",
    )

    assert json.loads(await manager.handle_tool_call("retaindb_profile", {})) == {
        "memories": [{"content": "Profile memory"}]
    }
    await manager.queue_prefetch_all("What should the agent remember?")
    assert await manager.flush_pending(timeout=1.0)
    await asyncio.gather(*tuple(provider._owned_tasks))
    context = await manager.prefetch_all("Use the prior context")
    assert "[RetainDB Context]" in context
    assert "[RetainDB User Synthesis]" in context
    assert "[RetainDB Agent Self-Model]" in context

    await manager.sync_all(
        "Remember this turn",
        "I will remember it",
        session_id="manager-session",
    )
    assert await manager.flush_pending(timeout=1.0)
    await manager.shutdown_all()
    assert fake.closed is True
    ingest = next(call for call in fake.calls if call[0] == "ingest_session")
    assert ingest[1][:2] == ("manager-user", "manager-session")


async def test_uninitialized_and_closed_client_fail_clearly():
    provider = RetainDBMemoryProvider()
    result = json.loads(await provider.handle_tool_call("retaindb_profile", {}))
    assert result == {"error": "RetainDB not initialized"}

    client = _Client("key", "https://api.test", "project")
    await client.close()
    with pytest.raises(RuntimeError, match="client is closed"):
        await client.get_profile("user")
