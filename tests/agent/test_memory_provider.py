"""Tests for memory-provider orchestration and context helpers."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from blockbuster import BlockBuster

from agent.memory_manager import (
    MemoryManager,
    build_memory_context_block,
    normalize_tool_schema,
    sanitize_context,
)
from agent.memory_provider import MemoryProvider


class _Provider(MemoryProvider):
    def __init__(self, name="external", *, context="", delay=0.0):
        self._name = name
        self.context = context
        self.delay = delay
        self.events = []

    @property
    def name(self):
        return self._name

    async def is_available(self):
        return True

    async def initialize(self, session_id, **kwargs):
        self.events.append(("initialize", session_id, kwargs))

    async def prefetch(self, query, *, session_id=""):
        if self.delay:
            await asyncio.sleep(self.delay)
        self.events.append(("prefetch", query, session_id))
        return self.context

    async def queue_prefetch(self, query, *, session_id=""):
        self.events.append(("queue", query, session_id))

    async def sync_turn(
        self,
        user_content,
        assistant_content,
        *,
        session_id="",
        messages=None,
    ):
        self.events.append(
            ("sync", user_content, assistant_content, session_id, messages)
        )

    def get_tool_schemas(self):
        return [{
            "name": f"{self.name}_recall",
            "parameters": {"type": "object"},
        }]

    async def handle_tool_call(self, tool_name, args, **kwargs):
        self.events.append(("tool", tool_name, args))
        return json.dumps({"ok": True})

    async def on_session_end(self, messages):
        self.events.append(("end", messages))

    async def on_session_switch(self, new_session_id, **kwargs):
        self.events.append(("switch", new_session_id, kwargs))

    async def shutdown(self):
        self.events.append(("shutdown",))


def test_normalize_tool_schema_accepts_bare_and_wrapped_functions():
    bare = {"name": "recall", "parameters": {"type": "object"}}
    wrapped = {"type": "function", "function": bare}

    assert normalize_tool_schema(bare) is bare
    assert normalize_tool_schema(wrapped) is bare


def test_normalize_tool_schema_rejects_missing_names():
    assert normalize_tool_schema(None) is None
    assert normalize_tool_schema("recall") is None
    assert normalize_tool_schema({}) is None
    assert normalize_tool_schema({"type": "function", "function": {}}) is None


def test_sanitize_context_removes_internal_memory_fences():
    raw = "before<memory-context>private</memory-context>after"

    assert sanitize_context(raw) == "beforeafter"


def test_build_memory_context_block_strips_existing_fenced_payload():
    wrapped = build_memory_context_block(
        "<memory-context>remember tea</memory-context>"
    )

    assert wrapped.count("<memory-context>") == 1
    assert wrapped.count("</memory-context>") == 1
    assert "remember tea" not in wrapped


@pytest.mark.asyncio
async def test_memory_manager_awaits_prefetch_and_tool_dispatch():
    manager = MemoryManager()
    builtin = _Provider("builtin", context="built in")
    external = _Provider("external", context="external")
    manager.add_provider(builtin)
    manager.add_provider(external)

    assert await manager.prefetch_all("question") == "built in\n\nexternal"
    assert json.loads(
        await manager.handle_tool_call("external_recall", {"query": "q"})
    ) == {"ok": True}


@pytest.mark.asyncio
async def test_memory_manager_preserves_background_write_order():
    manager = MemoryManager()
    provider = _Provider()
    manager.add_provider(provider)

    await manager.sync_all("u1", "a1", session_id="s")
    await manager.queue_prefetch_all("u2", session_id="s")
    assert await manager.flush_pending(timeout=1.0)

    assert [event[0] for event in provider.events] == ["sync", "queue"]


@pytest.mark.asyncio
async def test_memory_manager_session_boundary_orders_end_before_switch():
    manager = MemoryManager()
    provider = _Provider()
    manager.add_provider(provider)

    await manager.commit_session_boundary_async(
        [{"role": "user", "content": "hello"}],
        new_session_id="new",
        parent_session_id="old",
    )
    assert await manager.flush_pending(timeout=1.0)

    assert [event[0] for event in provider.events] == ["end", "switch"]


@pytest.mark.asyncio
async def test_memory_manager_timeout_does_not_block_event_loop():
    manager = MemoryManager(external_prefetch_timeout=0.01)
    provider = _Provider(delay=0.2)
    manager.add_provider(provider)
    heartbeat = 0

    async def beat():
        nonlocal heartbeat
        while True:
            heartbeat += 1
            await asyncio.sleep(0)

    heartbeat_task = asyncio.create_task(beat())
    try:
        assert await manager.prefetch_all("question") == ""
    finally:
        heartbeat_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await heartbeat_task

    assert heartbeat > 1


@pytest.mark.asyncio
async def test_prefetch_timeout_does_not_overlap_uncancellable_provider():
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    class _UncooperativeProvider(_Provider):
        async def prefetch(self, query, *, session_id=""):
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                # A real provider may be in a cancellation boundary that
                # cannot finish immediately.  The manager must still return
                # at its timeout and suppress overlapping calls.
                await release.wait()
            self.events.append(("prefetch", query, session_id))
            finished.set()
            return "late"

    manager = MemoryManager(external_prefetch_timeout=0.01)
    provider = _UncooperativeProvider()
    manager.add_provider(provider)

    first = asyncio.create_task(manager.prefetch_all("first"))
    await started.wait()
    await asyncio.wait_for(first, timeout=0.2)
    assert first.result() == ""
    assert await manager.prefetch_all("second") == ""
    assert [event[1] for event in provider.events if event[0] == "prefetch"] == []

    release.set()
    await asyncio.wait_for(finished.wait(), timeout=0.2)
    assert await manager.prefetch_all("third") == "late"


@pytest.mark.asyncio
async def test_memory_manager_shutdown_drains_then_closes_provider():
    manager = MemoryManager()
    provider = _Provider()
    manager.add_provider(provider)
    await manager.sync_all("user", "assistant")

    await manager.shutdown_all()

    assert [event[0] for event in provider.events] == ["sync", "shutdown"]
    assert manager.shutdown_drain_state["status"] == "drained"


@pytest.mark.asyncio
async def test_memory_manager_shutdown_collects_tasks_after_drain_timeout(
    monkeypatch,
):
    from agent import memory_manager as memory_module

    started = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def uncooperative_write():
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        finished.set()

    monkeypatch.setattr(memory_module, "_SYNC_DRAIN_TIMEOUT_S", 0.01)
    manager = MemoryManager()
    provider = _Provider()
    manager.add_provider(provider)
    write_task = asyncio.create_task(uncooperative_write())
    manager._background_tasks[write_task] = "sync"
    await started.wait()

    shutdown = asyncio.create_task(manager.shutdown_all())
    await cancelled.wait()
    assert shutdown.done() is False
    release.set()
    await shutdown

    assert finished.is_set()
    assert manager.shutdown_drain_state == {
        "status": "timed_out",
        "abandoned_writes": 1,
        "abandoned_prefetches": 0,
        "active_tasks": 0,
    }
    assert provider.events == [("shutdown",)]


@pytest.mark.asyncio
async def test_memory_manager_shutdown_survives_repeated_cancellation(
    monkeypatch,
):
    from agent import memory_manager as memory_module

    started = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def uncooperative_prefetch():
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        finished.set()

    monkeypatch.setattr(memory_module, "_SYNC_DRAIN_TIMEOUT_S", 0.01)
    manager = MemoryManager()
    provider = _Provider()
    manager.add_provider(provider)
    prefetch_task = asyncio.create_task(uncooperative_prefetch())
    manager._external_prefetch_tasks[provider.name] = prefetch_task
    await started.wait()

    shutdown = asyncio.create_task(manager.shutdown_all())
    await cancelled.wait()
    shutdown.cancel()
    await asyncio.sleep(0)
    shutdown.cancel()
    await asyncio.sleep(0)

    assert shutdown.done() is False
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await shutdown
    assert finished.is_set()
    assert manager.shutdown_drain_state["active_tasks"] == 0
    assert provider.events == [("shutdown",)]


def test_memory_manager_rejects_sync_provider_contract():
    class _SyncProvider(_Provider):
        def prefetch(self, query, *, session_id=""):
            return "blocking"

    from hermes_cli.plugins import _PluginContractError

    with pytest.raises(_PluginContractError, match="prefetch"):
        MemoryManager().add_provider(_SyncProvider())


@pytest.mark.asyncio
async def test_memory_manager_propagates_provider_initialization_failure():
    class _BrokenProvider(_Provider):
        async def initialize(self, session_id, **kwargs):
            raise RuntimeError("provider unavailable")

    manager = MemoryManager()
    manager.add_provider(_BrokenProvider())

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await manager.initialize_all("session")


@pytest.mark.asyncio
async def test_user_memory_provider_loads_with_native_async_contract(
    tmp_path,
    monkeypatch,
):
    plugin_dir = tmp_path / "plugins" / "example_memory"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text(
        """
from agent.memory_provider import MemoryProvider

class ExampleMemoryProvider(MemoryProvider):
    @property
    def name(self):
        return "example_memory"

    async def is_available(self):
        return True

    async def initialize(self, session_id, **kwargs):
        self.session_id = session_id

    def get_tool_schemas(self):
        return [{"name": "example_recall", "parameters": {"type": "object"}}]

    async def handle_tool_call(self, tool_name, args, **kwargs):
        return '{"ok": true}'
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from plugins.memory import load_memory_provider

    blockbuster = BlockBuster()
    blockbuster.activate()
    try:
        provider = await load_memory_provider("example_memory")
    finally:
        blockbuster.deactivate()
    assert provider is not None
    manager = MemoryManager()
    manager.add_provider(provider)
    await manager.initialize_all("session-1", hermes_home=str(tmp_path))

    assert provider.session_id == "session-1"
    assert json.loads(
        await manager.handle_tool_call("example_recall", {})
    ) == {"ok": True}


@pytest.mark.asyncio
async def test_deferred_runtime_initializes_and_injects_user_memory_provider(
    tmp_path,
    monkeypatch,
):
    plugin_dir = tmp_path / "plugins" / "deferred_memory"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text(
        """
from agent.memory_provider import MemoryProvider

class DeferredMemoryProvider(MemoryProvider):
    @property
    def name(self):
        return "deferred_memory"

    async def is_available(self):
        return True

    async def initialize(self, session_id, **kwargs):
        self.session_id = session_id
        self.init_kwargs = kwargs

    def system_prompt_block(self):
        return "deferred memory guidance"

    def get_tool_schemas(self):
        return [{"name": "deferred_recall", "parameters": {"type": "object"}}]
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = SimpleNamespace(
        _memory_manager=None,
        _memory_manager_started=False,
        skip_memory=False,
        platform="feishu",
        session_id="session-2",
        _session_db=None,
        _user_id="open-id",
        _user_id_alt="union-id",
        _user_name=None,
        _chat_id=None,
        _chat_name=None,
        _chat_type=None,
        _thread_id=None,
        _gateway_session_key=None,
        tools=[],
        valid_tool_names=set(),
        enabled_toolsets=None,
        disabled_toolsets=None,
        _emit_warning=lambda _message: None,
        _emit_status=lambda _message: None,
    )

    from agent.agent_init import _initialize_memory_manager

    blockbuster = BlockBuster()
    blockbuster.activate()
    try:
        await _initialize_memory_manager(
            agent,
            {"memory": {"provider": "deferred_memory"}},
        )
    finally:
        blockbuster.deactivate()

    assert agent._memory_manager_started is True
    assert agent._memory_manager.build_system_prompt() == "deferred memory guidance"
    assert agent.valid_tool_names == {"deferred_recall"}
    provider = agent._memory_manager.providers[0]
    assert provider.session_id == "session-2"
    assert provider.init_kwargs["platform"] == "feishu"
    assert provider.init_kwargs["user_id"] == "open-id"
    assert provider.init_kwargs["user_id_alt"] == "union-id"
