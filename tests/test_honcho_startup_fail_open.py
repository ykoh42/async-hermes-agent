"""Regression tests for Honcho startup fail-open behavior."""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.memory.honcho import HonchoMemoryProvider


pytestmark = pytest.mark.asyncio


class _FakeHonchoConfig(SimpleNamespace):
    async def resolve_session_name(self, **kwargs):
        return "test-session"


def _configured_hybrid_config() -> _FakeHonchoConfig:
    return _FakeHonchoConfig(
        enabled=True,
        api_key=None,
        base_url="http://127.0.0.1:8000",
        recall_mode="hybrid",
        init_on_session_start=False,
        injection_frequency="every-turn",
        context_cadence=1,
        dialectic_cadence=1,
        query_rewrite=False,
        first_turn_base_wait=3.0,
        first_turn_dialectic_wait=2.0,
        dialectic_depth=1,
        dialectic_depth_levels=None,
        reasoning_heuristic=True,
        reasoning_level_cap="high",
        context_tokens=None,
        message_max_chars=25000,
        session_strategy="per-directory",
    )


def _configured_tools_config(*, init_on_session_start: bool = False) -> _FakeHonchoConfig:
    cfg = _configured_hybrid_config()
    cfg.recall_mode = "tools"
    cfg.init_on_session_start = init_on_session_start
    return cfg


async def test_stalled_init_only_delays_first_turn_prefetch(monkeypatch):
    """A stalled init may bound-wait on turn 1 only; later turns fail open."""
    provider = HonchoMemoryProvider()
    cfg = _configured_hybrid_config()
    release = asyncio.Event()

    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        AsyncMock(return_value=cfg),
    )

    async def stalled_session_init(self, cfg, session_id, **kwargs):
        await release.wait()

    monkeypatch.setattr(HonchoMemoryProvider, "_do_session_init", stalled_session_init)
    await provider.initialize("session-1", platform="cli")
    provider._FIRST_TURN_BASE_TIMEOUT = 0.1

    try:
        provider._turn_count = 1
        start = time.perf_counter()
        assert await provider.prefetch("first question") == ""
        assert time.perf_counter() - start >= 0.08

        for turn in (2, 3, 4):
            provider._turn_count = turn
            start = time.perf_counter()
            assert await provider.prefetch("follow-up question") == ""
            assert time.perf_counter() - start < 0.05
    finally:
        release.set()
        await provider.shutdown()


async def test_honcho_background_init_rechecks_state_after_lock_race():
    """Startup should not spawn/crash if init completes while waiting for lock."""
    provider = HonchoMemoryProvider()
    provider._config = _configured_hybrid_config()
    provider._lazy_init_kwargs = {"platform": "cli"}
    provider._lazy_init_session_id = "session-1"

    class RacingLock:
        async def __aenter__(self):
            provider._session_initialized = True
            provider._lazy_init_kwargs = None
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    provider._init_lock = RacingLock()

    await provider._start_session_init_background()

    assert provider._init_task is None
    assert provider._session_initialized is True
    await provider.shutdown()


async def test_first_turn_base_wait_is_shared_by_init_and_context_fetch():
    """Session init and base retrieval share one configured turn-1 deadline."""
    provider = HonchoMemoryProvider()
    cfg = _configured_hybrid_config()
    cfg.first_turn_base_wait = 0.2
    cfg.timeout = None
    release_context = asyncio.Event()

    class SlowManager:
        async def get_prefetch_context(self, session_key, user_message=None):
            await release_context.wait()
            return {"representation": "late"}

        async def set_context_result(self, session_key, result):
            return None

        async def pop_context_result(self, session_key):
            return {}

    async def finish_init():
        await asyncio.sleep(0.1)
        provider._manager = SlowManager()
        provider._session_initialized = True

    provider._config = cfg
    provider._session_key = "test-session"
    provider._recall_mode = "context"
    provider._turn_count = 1
    provider._last_dialectic_turn = 0
    provider._FIRST_TURN_BASE_TIMEOUT = cfg.first_turn_base_wait
    provider._init_task = provider._track_task(asyncio.create_task(finish_init()))

    try:
        started = time.perf_counter()
        assert await provider.prefetch("what do you know about me?") == ""
        elapsed = time.perf_counter() - started
        assert 0.08 <= elapsed < 1.0
    finally:
        release_context.set()
        await provider.shutdown()


async def test_honcho_sync_turn_waits_for_full_background_startup(monkeypatch):
    """Manager assignment alone is not readiness while background init continues."""
    provider = HonchoMemoryProvider()
    cfg = _configured_hybrid_config()
    session_created = asyncio.Event()
    migration_started = asyncio.Event()
    release_migration = asyncio.Event()
    get_calls = []

    class StartupManager:
        def __init__(self, *args, **kwargs):
            pass

        async def get_or_create(self, session_key):
            get_calls.append(session_key)
            session_created.set()
            return SimpleNamespace(messages=[])

        async def migrate_memory_files(self, session_key, mem_dir):
            migration_started.set()
            await release_migration.wait()

        async def prefetch_context(self, session_key, user_message=None):
            return None

        async def _flush_session(self, session):
            return None

        async def shutdown(self):
            return None

    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        AsyncMock(return_value=cfg),
    )
    monkeypatch.setattr(
        "plugins.memory.honcho.client.get_honcho_client",
        AsyncMock(return_value=object()),
    )
    monkeypatch.setattr("plugins.memory.honcho.session.HonchoSessionManager", StartupManager)

    await provider.initialize("session-1", platform="cli")
    try:
        await asyncio.wait_for(session_created.wait(), timeout=1)
        await asyncio.wait_for(migration_started.wait(), timeout=1)
        assert provider._manager is not None
        assert provider._session_initialized is False

        await provider.sync_turn("hello", "world")

        assert get_calls == ["test-session"]
    finally:
        release_migration.set()
        await provider.shutdown()

    assert provider._session_initialized is True


async def test_honcho_system_prompt_advertises_active_while_background_init_runs(monkeypatch):
    """Prompt metadata should not require a completed network session."""
    provider = HonchoMemoryProvider()
    cfg = _configured_hybrid_config()
    release = asyncio.Event()

    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        AsyncMock(return_value=cfg),
    )

    async def slow_session_init(self, cfg, session_id, **kwargs):
        await release.wait()
        self._session_initialized = True

    monkeypatch.setattr(HonchoMemoryProvider, "_do_session_init", slow_session_init)

    await provider.initialize("session-1", platform="cli")
    try:
        prompt = provider.system_prompt_block()
        assert "Honcho Memory" in prompt
        assert "hybrid mode" in prompt
    finally:
        release.set()
        await provider.shutdown()


async def test_honcho_tools_eager_init_failure_does_not_leave_ready_manager(monkeypatch):
    """Failed eager tools startup must not leave hooks seeing a ready session."""
    provider = HonchoMemoryProvider()
    cfg = _configured_tools_config(init_on_session_start=True)

    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        AsyncMock(return_value=cfg),
    )

    async def failing_session_init(self, cfg, session_id, **kwargs):
        self._manager = SimpleNamespace()
        self._session_key = "test-session"
        raise RuntimeError("boom")

    monkeypatch.setattr(HonchoMemoryProvider, "_do_session_init", failing_session_init)

    await provider.initialize("session-1", platform="cli")
    assert provider._session_initialized is False
    assert provider._manager is None

    background_started = asyncio.Event()

    async def mark_background_started(*args, **kwargs):
        background_started.set()

    provider._start_session_init_background = mark_background_started
    await provider.sync_turn("hello", "world")
    await provider.on_memory_write("add", "user", "prefers safe Honcho startup")

    assert not background_started.is_set()

    result = json.loads(await provider.handle_tool_call("honcho_profile", {"peer": "user"}))
    assert "could not be initialized" in result["error"]
    assert provider._manager is None
    await provider.shutdown()


async def test_honcho_tools_lazy_hooks_do_not_prestart_background_init(monkeypatch):
    """Tools lazy mode lets the first tool call own session initialization."""
    provider = HonchoMemoryProvider()
    cfg = _configured_tools_config(init_on_session_start=False)

    monkeypatch.setattr(
        "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
        AsyncMock(return_value=cfg),
    )

    await provider.initialize("session-1", platform="cli")
    background_started = asyncio.Event()

    async def mark_background_started(*args, **kwargs):
        background_started.set()

    provider._start_session_init_background = mark_background_started

    assert await provider.prefetch("what do you know?") == ""
    await provider.queue_prefetch("what do you know?")
    await provider.sync_turn("hello", "world")
    await provider.on_memory_write("add", "user", "prefers fail-open memory")

    assert not background_started.is_set()
    assert provider._session_initialized is False

    class ToolManager:
        async def get_peer_card(self, session_key, peer="user"):
            return ["ready"]

        async def shutdown(self):
            return None

    init_calls = []

    async def fake_session_init(self, cfg, session_id, **kwargs):
        init_calls.append(session_id)
        self._manager = ToolManager()
        self._session_key = "test-session"
        self._session_initialized = True

    monkeypatch.setattr(HonchoMemoryProvider, "_do_session_init", fake_session_init)

    result = json.loads(await provider.handle_tool_call("honcho_profile", {"peer": "user"}))

    assert result == {"result": ["ready"]}
    assert init_calls == ["session-1"]
    assert not background_started.is_set()
    await provider.shutdown()


async def test_cancelled_provider_shutdown_cleans_owned_tasks_and_manager():
    provider = HonchoMemoryProvider()
    release = asyncio.Event()

    async def blocked_work():
        await release.wait()

    owned_task = provider._track_task(asyncio.create_task(blocked_work()))
    provider._manager = SimpleNamespace(shutdown=AsyncMock())

    shutdown_task = asyncio.create_task(provider.shutdown())
    await asyncio.sleep(0)
    shutdown_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await shutdown_task

    assert owned_task.done()
    provider._manager.shutdown.assert_awaited_once()
