"""Native-async parity and lifecycle coverage for Hindsight memory."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiofiles
import pytest
import pytest_asyncio
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

from plugins.memory import load_memory_provider
from plugins.memory.hindsight import (
    HindsightMemoryProvider,
    RECALL_SCHEMA,
    REFLECT_SCHEMA,
    RETAIN_SCHEMA,
    _append_capability_cache,
    _append_capability_locks,
    _check_api_supports_update_mode_append,
    _embedded_profile_env_path,
    _load_simple_env,
    _materialize_embedded_profile_env,
    _normalize_observation_scopes,
    _normalize_retain_tags,
    _resolve_bank_id_template,
    _sanitize_bank_segment,
)

pytestmark = pytest.mark.asyncio


class FakeClient:
    def __init__(self) -> None:
        self.retain_calls: list[dict] = []
        self.recall_calls: list[dict] = []
        self.reflect_calls: list[dict] = []
        self.statuses: list[object] = []
        self.closed = False
        self.operations = self
        self.retain_gate: asyncio.Event | None = None

    async def aretain_batch(self, **kwargs):
        if self.retain_gate is not None:
            await self.retain_gate.wait()
        self.retain_calls.append(kwargs)
        return SimpleNamespace(operation_id="retain-op", operation_ids=None)

    async def arecall(self, **kwargs):
        self.recall_calls.append(kwargs)
        return SimpleNamespace(
            results=[
                SimpleNamespace(text="Memory 1"),
                SimpleNamespace(text="Memory 2"),
            ]
        )

    async def areflect(self, **kwargs):
        self.reflect_calls.append(kwargs)
        return SimpleNamespace(text="Synthesized answer")

    async def get_operation_status(self, **kwargs):
        if self.statuses:
            response = self.statuses.pop(0)
            if isinstance(response, BaseException):
                raise response
            return response
        return SimpleNamespace(status="completed")

    async def aclose(self) -> None:
        self.closed = True


async def _write_config(home: Path, **overrides) -> dict:
    config = {
        "mode": "cloud",
        "apiKey": "test-key",
        "api_url": "http://hindsight.test",
        "bank_id": "test-bank",
        "budget": "mid",
        "memory_mode": "hybrid",
    }
    config.update(overrides)
    path = home / "hindsight" / "config.json"
    await aiofiles.os.makedirs(path.parent, exist_ok=True)
    async with aiofiles.open(path, "w", encoding="utf-8") as handle:
        await handle.write(json.dumps(config))
    return config


async def _make_candidate(home: Path, monkeypatch, **overrides):
    await _write_config(home, **overrides)
    monkeypatch.setattr(
        "plugins.memory.hindsight.get_hermes_home",
        lambda: home,
    )
    candidate = HindsightMemoryProvider()
    await candidate.initialize(
        "test-session",
        hermes_home=str(home),
        platform="cli",
    )
    candidate._client = FakeClient()
    return candidate


@pytest_asyncio.fixture(autouse=True)
async def clean_state(tmp_path, monkeypatch):
    for key in (
        "HINDSIGHT_API_KEY",
        "HINDSIGHT_API_URL",
        "HINDSIGHT_BANK_ID",
        "HINDSIGHT_BUDGET",
        "HINDSIGHT_MODE",
        "HINDSIGHT_TIMEOUT",
        "HINDSIGHT_IDLE_TIMEOUT",
        "HINDSIGHT_LLM_API_KEY",
        "HINDSIGHT_RETAIN_TAGS",
        "HINDSIGHT_RETAIN_OBSERVATION_SCOPES",
        "HINDSIGHT_RETAIN_SOURCE",
        "HINDSIGHT_RETAIN_USER_PREFIX",
        "HINDSIGHT_RETAIN_ASSISTANT_PREFIX",
    ):
        monkeypatch.delenv(key, raising=False)
    isolated_home = tmp_path / "user-home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: isolated_home))
    _append_capability_cache.clear()
    _append_capability_locks.clear()
    yield
    _append_capability_cache.clear()
    _append_capability_locks.clear()


@pytest_asyncio.fixture
async def provider(tmp_path, monkeypatch):
    await _write_config(tmp_path)
    monkeypatch.setattr(
        "plugins.memory.hindsight.get_hermes_home",
        lambda: tmp_path,
    )
    candidate = HindsightMemoryProvider()
    await candidate.initialize(
        "test-session",
        hermes_home=str(tmp_path),
        platform="cli",
    )
    client = FakeClient()
    candidate._client = client
    try:
        yield candidate
    finally:
        await candidate.shutdown()


async def test_public_api_keeps_upstream_names_and_arguments():  # noqa: ASYNC124
    expected = {
        "backup_paths": ["self"],
        "is_available": ["self"],
        "save_config": ["self", "values", "hermes_home"],
        "get_config_schema": ["self"],
        "initialize": ["self", "session_id", "kwargs"],
        "system_prompt_block": ["self"],
        "prefetch": ["self", "query", "session_id"],
        "queue_prefetch": ["self", "query", "session_id"],
        "sync_turn": ["self", "user_content", "assistant_content", "session_id"],
        "get_tool_schemas": ["self"],
        "handle_tool_call": ["self", "tool_name", "args", "kwargs"],
        "on_session_switch": [
            "self",
            "new_session_id",
            "parent_session_id",
            "reset",
            "kwargs",
        ],
        "shutdown": ["self"],
    }
    synchronous = {
        "backup_paths",
        "get_config_schema",
        "system_prompt_block",
        "get_tool_schemas",
    }
    for name, parameters in expected.items():
        method = getattr(HindsightMemoryProvider, name)
        assert inspect.iscoroutinefunction(method) is (name not in synchronous)
        assert list(inspect.signature(method).parameters) == parameters


async def test_schemas_and_pure_normalizers_preserve_upstream_values(provider):  # noqa: ASYNC124
    assert RETAIN_SCHEMA["name"] == "hindsight_retain"
    assert RECALL_SCHEMA["name"] == "hindsight_recall"
    assert REFLECT_SCHEMA["name"] == "hindsight_reflect"
    assert provider.get_tool_schemas() == [
        RETAIN_SCHEMA,
        RECALL_SCHEMA,
        REFLECT_SCHEMA,
    ]
    assert _normalize_retain_tags("one,two,one") == ["one", "two"]
    assert _normalize_observation_scopes("per_tag") == "per_tag"
    assert _normalize_observation_scopes('[ ["one"], ["two"] ]') == [
        ["one"],
        ["two"],
    ]
    assert _sanitize_bank_segment("team / alpha") == "team-alpha"
    assert _resolve_bank_id_template(
        "hermes-{profile}-{user}",
        "fallback",
        profile="coder one",
        user="u/1",
    ) == "hermes-coder-one-u-1"


async def test_save_and_load_config_are_owner_only(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "plugins.memory.hindsight.get_hermes_home",
        lambda: tmp_path,
    )
    candidate = HindsightMemoryProvider()
    await candidate.save_config(
        {"mode": "cloud", "bank_id": "private-bank"},
        str(tmp_path),
    )
    path = tmp_path / "hindsight" / "config.json"
    async with aiofiles.open(path, encoding="utf-8") as handle:
        assert json.loads(await handle.read())["bank_id"] == "private-bank"
    if os.name == "posix":
        assert stat.S_IMODE((await aiofiles.os.stat(path)).st_mode) == 0o600


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs privileges")
async def test_save_config_preserves_symlink_and_target_owner_mode(
    tmp_path,
):
    target_dir = tmp_path / "managed"
    link_home = tmp_path / "profile"
    await aiofiles.os.makedirs(target_dir, exist_ok=True)
    await aiofiles.os.makedirs(link_home / "hindsight", exist_ok=True)
    target = target_dir / "hindsight.json"
    async with aiofiles.open(target, "w", encoding="utf-8") as handle:
        await handle.write('{"bank_id":"old"}')
    link = link_home / "hindsight" / "config.json"
    await aiofiles.os.symlink(target, link)

    candidate = HindsightMemoryProvider()
    await candidate.save_config({"bank_id": "new"}, str(link_home))

    assert await aiofiles.os.path.islink(link)
    async with aiofiles.open(target, encoding="utf-8") as handle:
        assert json.loads(await handle.read())["bank_id"] == "new"
    if os.name == "posix":
        assert stat.S_IMODE((await aiofiles.os.stat(target)).st_mode) == 0o600


async def test_profile_env_is_bom_safe_owner_only_and_round_trips(tmp_path):
    config = {
        "profile": "hermes",
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "idle_timeout": 91,
    }
    env_path = _embedded_profile_env_path(config)
    await aiofiles.os.makedirs(env_path.parent, exist_ok=True)
    async with aiofiles.open(env_path, "w", encoding="utf-8-sig") as handle:
        await handle.write("HINDSIGHT_API_PORT=9123\n")
    assert (await _load_simple_env(env_path))["HINDSIGHT_API_PORT"] == "9123"

    written = await _materialize_embedded_profile_env(
        config,
        llm_api_key="sk-secret",
    )
    values = await _load_simple_env(written)
    assert values == {
        "HINDSIGHT_API_LLM_PROVIDER": "openai",
        "HINDSIGHT_API_LLM_API_KEY": "sk-secret",
        "HINDSIGHT_API_LLM_MODEL": "gpt-4o-mini",
        "HINDSIGHT_API_LOG_LEVEL": "info",
        "HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT": "91",
    }
    if os.name == "posix":
        assert stat.S_IMODE((await aiofiles.os.stat(written)).st_mode) == 0o600


async def test_profile_env_secret_is_removed_when_validation_fails(monkeypatch):
    config = {
        "profile": "hermes",
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
    }

    async def fail_validation(path):  # noqa: ASYNC124
        raise PermissionError(f"not owner-only: {path}")

    monkeypatch.setattr(
        "plugins.memory.hindsight._validate_profile_env_permissions",
        fail_validation,
    )
    with pytest.raises(PermissionError):
        await _materialize_embedded_profile_env(
            config,
            llm_api_key="sk-doomed",
        )
    assert not await aiofiles.os.path.exists(_embedded_profile_env_path(config))


async def test_capability_probe_is_cached_and_serialized(monkeypatch):
    calls = 0

    async def fetch(*args, **kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return "0.6.1"

    monkeypatch.setattr(
        "plugins.memory.hindsight._fetch_hindsight_api_version",
        fetch,
    )
    results = await asyncio.gather(
        *(
            _check_api_supports_update_mode_append("http://same.test")
            for _ in range(8)
        )
    )
    assert results == [True] * 8
    assert calls == 1
    assert all(
        (lock := lock_ref()) is None or not lock.locked()
        for lock_ref in _append_capability_locks.values()
    )


async def test_tool_handlers_preserve_results_and_sdk_arguments(provider):
    client = provider._client
    retained = json.loads(
        await provider.handle_tool_call(
            "hindsight_retain",
            {"content": "Remember Seoul", "tags": ["city"]},
        )
    )
    recalled = json.loads(
        await provider.handle_tool_call(
            "hindsight_recall",
            {"query": "Where?"},
        )
    )
    reflected = json.loads(
        await provider.handle_tool_call(
            "hindsight_reflect",
            {"query": "Summarize"},
        )
    )
    assert retained == {"result": "Memory stored successfully."}
    assert recalled == {"result": "1. Memory 1\n2. Memory 2"}
    assert reflected == {"result": "Synthesized answer"}
    assert client.retain_calls[0]["bank_id"] == "test-bank"
    assert client.retain_calls[0]["items"][0]["tags"] == ["city"]
    assert client.recall_calls[0] == {
        "bank_id": "test-bank",
        "query": "Where?",
        "budget": "mid",
        "max_tokens": 4096,
        "types": ["observation"],
    }


async def test_missing_and_unknown_tool_arguments_preserve_error_shape(provider):
    missing_retain = json.loads(
        await provider.handle_tool_call("hindsight_retain", {})
    )
    missing_recall = json.loads(
        await provider.handle_tool_call("hindsight_recall", {})
    )
    unknown = json.loads(await provider.handle_tool_call("unknown", {}))
    assert "Missing required parameter: content" in missing_retain["error"]
    assert "Missing required parameter: query" in missing_recall["error"]
    assert "Unknown tool: unknown" in unknown["error"]


async def test_custom_config_controls_prompt_recall_and_metadata(
    tmp_path,
    monkeypatch,
):
    candidate = await _make_candidate(
        tmp_path,
        monkeypatch,
        memory_mode="tools",
        recall_prefetch_method="reflect",
        recall_budget="high",
        recall_types="observation,world",
        retain_tags="default,shared",
        observation_scopes="per_tag",
        retain_source="integration",
        retain_user_prefix="Human",
        retain_assistant_prefix="Agent",
        bank_id_template="bank-{profile}-{session}",
    )
    try:
        assert candidate._budget == "high"
        assert candidate._prefetch_method == "reflect"
        assert candidate._recall_types == ["observation", "world"]
        assert candidate._observation_scopes == "per_tag"
        assert candidate._bank_id == "bank-test-session"
        assert candidate.get_tool_schemas() == [
            RETAIN_SCHEMA,
            RECALL_SCHEMA,
            REFLECT_SCHEMA,
        ]
        assert "Active (tools mode)." in candidate.system_prompt_block()
        turn = candidate._build_turn_messages("hello", "hi")
        assert turn[0]["content"] == "Human: hello"
        assert turn[1]["content"] == "Agent: hi"
        metadata = candidate._build_metadata(message_count=2, turn_index=1)
        assert metadata["source"] == "integration"
        assert metadata["platform"] == "cli"
    finally:
        await candidate.shutdown()


async def test_context_mode_hides_tools_and_keeps_exact_prompt(
    tmp_path,
    monkeypatch,
):
    candidate = await _make_candidate(
        tmp_path,
        monkeypatch,
        memory_mode="context",
        bank_id="context-bank",
    )
    try:
        assert candidate.get_tool_schemas() == []
        assert candidate.system_prompt_block() == (
            "# Hindsight Memory\n"
            "Active (context mode). Bank: context-bank, budget: mid.\n"
            "Relevant memories are automatically injected into context."
        )
    finally:
        await candidate.shutdown()


async def test_prefetch_is_one_turn_delayed_and_keeps_upstream_format(provider):
    assert await provider.prefetch("first") == ""
    await provider.queue_prefetch("remember me")
    context = await provider.prefetch("next")
    assert context == (
        "# Hindsight Memory (persistent cross-session context)\n"
        "Use this to answer questions about the user and prior sessions. "
        "Do not call tools to look up information that is already present here."
        "\n\n- Memory 1\n- Memory 2"
    )


async def test_recall_sync_uses_current_query_and_skips_background_queue(
    tmp_path,
    monkeypatch,
):
    provider = await _make_candidate(
        tmp_path,
        monkeypatch,
        recall_sync=True,
    )
    try:
        assert provider._recall_sync is True
        await provider.queue_prefetch("queued query")
        assert provider._prefetch_task is None
        context = await provider.prefetch("current query")
        assert "- Memory 1" in context
        assert provider._client.recall_calls[-1]["query"] == "current query"
        assert provider._client.recall_calls[-1]["types"] == ["observation"]
    finally:
        await provider.shutdown()


async def test_recall_status_reports_count_and_clears_after_empty_turn(
    provider,
):
    await provider.queue_prefetch("first query")
    assert "- Memory 1" in await provider.prefetch("next query")
    status = provider.recall_status()
    assert status is not None
    assert (status.provider_label, status.count, status.glyph) == (
        "Hindsight",
        2,
        "👁️",
    )

    async def empty_recall(**kwargs):
        return SimpleNamespace(results=[])

    provider._client.arecall = empty_recall
    await provider.queue_prefetch("empty query")
    assert await provider.prefetch("following query") == ""
    assert provider.recall_status() is None


async def test_retain_indicator_emits_only_when_a_retain_is_dispatched(
    tmp_path,
    monkeypatch,
):
    await _write_config(tmp_path)
    monkeypatch.setattr(
        "plugins.memory.hindsight.get_hermes_home",
        lambda: tmp_path,
    )
    statuses: list[str] = []
    provider = HindsightMemoryProvider()
    await provider.initialize(
        "indicator-session",
        hermes_home=str(tmp_path),
        status_callback=statuses.append,
    )
    provider._client = FakeClient()
    provider._resolve_retain_target = AsyncMock(
        return_value=(provider._document_id, None),
    )
    try:
        await provider.sync_turn("hello", "answer")
        await provider._retain_queue.join()
        assert statuses == ["👁️ Hindsight — saving to memory…"]
    finally:
        await provider.shutdown()
    assert await provider.prefetch("consumed") == ""


async def test_prefetch_skip_truncation_preamble_and_reflect(
    tmp_path,
    monkeypatch,
):
    candidate = await _make_candidate(
        tmp_path,
        monkeypatch,
        recall_prefetch_method="reflect",
        recall_max_input_chars=4,
        recall_prompt_preamble="Custom memory header",
    )
    try:
        await candidate.queue_prefetch("abcdefgh")
        assert await candidate.prefetch("next") == (
            "Custom memory header\n\nSynthesized answer"
        )
        assert candidate._client.reflect_calls[0]["query"] == "abcd"
        candidate._auto_recall = False
        await candidate.queue_prefetch("ignored")
        assert len(candidate._client.reflect_calls) == 1
        candidate._auto_recall = True
        candidate._memory_mode = "tools"
        await candidate.queue_prefetch("ignored")
        assert len(candidate._client.reflect_calls) == 1
    finally:
        await candidate.shutdown()


async def test_sync_turn_preserves_fifo_and_append_delta(provider, monkeypatch):
    async def modern(*args, **kwargs):  # noqa: ASYNC124
        return "0.6.1"

    monkeypatch.setattr(
        "plugins.memory.hindsight._fetch_hindsight_api_version",
        modern,
    )
    await provider.sync_turn("one", "answer one", session_id="test-session")
    await provider.sync_turn("two", "answer two", session_id="test-session")
    await provider._retain_queue.join()

    calls = provider._client.retain_calls
    assert len(calls) == 2
    assert [call["document_id"] for call in calls] == [
        "test-session",
        "test-session",
    ]
    assert [call["items"][0]["update_mode"] for call in calls] == [
        "append",
        "append",
    ]
    first_content = json.loads(calls[0]["items"][0]["content"])
    second_content = json.loads(calls[1]["items"][0]["content"])
    assert len(first_content) == 1
    assert len(second_content) == 1
    assert first_content[0][0]["content"] == "User: one"
    assert second_content[0][0]["content"] == "User: two"
    assert calls[0]["items"][0]["metadata"]["turn_index"] == "1"
    assert calls[1]["items"][0]["metadata"]["turn_index"] == "2"


async def test_legacy_retain_resends_full_session(provider, monkeypatch):
    async def legacy(*args, **kwargs):  # noqa: ASYNC124
        return "0.4.9"

    monkeypatch.setattr(
        "plugins.memory.hindsight._fetch_hindsight_api_version",
        legacy,
    )
    await provider.sync_turn("one", "a")
    await provider.sync_turn("two", "b")
    await provider._retain_queue.join()
    calls = provider._client.retain_calls
    assert len(json.loads(calls[0]["items"][0]["content"])) == 1
    assert len(json.loads(calls[1]["items"][0]["content"])) == 2
    assert "update_mode" not in calls[1]["items"][0]
    assert calls[0]["document_id"] == provider._document_id


async def test_prefetch_waits_for_server_retain_visibility(provider, monkeypatch):
    async def modern(*args, **kwargs):  # noqa: ASYNC124
        return "0.6.1"

    monkeypatch.setattr(
        "plugins.memory.hindsight._fetch_hindsight_api_version",
        modern,
    )
    provider._client.statuses = [
        SimpleNamespace(status="pending"),
        SimpleNamespace(status="completed"),
    ]
    provider._RETAIN_OP_POLL_INTERVAL_S = 0
    await provider.sync_turn("one", "a")
    await provider.queue_prefetch("one")
    await provider.prefetch("next")
    assert provider._client.recall_calls
    assert not provider._pending_retain_ops


async def test_server_visibility_timeout_drops_unresolved_operations(provider):
    provider._RETAIN_OP_POLL_INTERVAL_S = 0.001
    provider._pending_retain_ops.add("never-finishes")
    provider._retain_ops_bank_id = provider._bank_id
    provider._client.statuses = [SimpleNamespace(status="pending")] * 100
    assert await provider._wait_for_retains_drained(0.01) is False
    assert provider._pending_retain_ops == set()


async def test_operation_not_found_is_treated_as_complete(provider):
    missing = RuntimeError("gone")
    missing.status = 404
    provider._client.statuses = [missing]
    assert await provider._is_retain_op_complete("test-bank", "gone") is True


async def test_session_switch_flushes_old_buffer_before_rotation(
    provider,
    monkeypatch,
):
    async def modern(*args, **kwargs):  # noqa: ASYNC124
        return "0.6.1"

    monkeypatch.setattr(
        "plugins.memory.hindsight._fetch_hindsight_api_version",
        modern,
    )
    provider._retain_every_n_turns = 3
    await provider.sync_turn("old", "answer")
    await provider.on_session_switch(
        "new-session",
        parent_session_id="test-session",
    )
    await provider._retain_queue.join()
    call = provider._client.retain_calls[0]
    assert call["document_id"] == "test-session"
    assert call["items"][0]["metadata"]["session_id"] == "test-session"
    assert "session:test-session" in call["items"][0]["tags"]
    assert provider._session_id == "new-session"
    assert provider._parent_session_id == "test-session"
    assert provider._session_turns == []
    assert provider._turn_counter == 0


async def test_session_switch_serializes_flush_behind_inflight_retain(
    provider,
    monkeypatch,
):
    monkeypatch.setattr(
        "plugins.memory.hindsight._check_api_supports_update_mode_append",
        AsyncMock(return_value=True),
    )
    gate = asyncio.Event()
    provider._client.retain_gate = gate
    await provider.sync_turn("first", "answer")
    switch = asyncio.create_task(provider.on_session_switch("new-session"))
    await asyncio.sleep(0)
    gate.set()
    await switch
    await provider._retain_queue.join()
    assert len(provider._client.retain_calls) == 2
    assert [
        call["document_id"] for call in provider._client.retain_calls
    ] == ["test-session", "test-session"]


async def test_session_switch_cancels_old_prefetch_and_drops_cached_text(provider):
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_recall(**kwargs):
        started.set()
        await release.wait()
        return SimpleNamespace(results=[SimpleNamespace(text="stale")])

    provider._client.arecall = slow_recall
    await provider.queue_prefetch("old query")
    await started.wait()
    await provider.on_session_switch("new-session")
    release.set()
    assert provider._prefetch_result == ""
    assert not provider._prefetch_tasks


async def test_availability_and_local_runtime_failure(tmp_path, monkeypatch):
    await _write_config(tmp_path, mode="cloud", apiKey="available")
    monkeypatch.setattr(
        "plugins.memory.hindsight.get_hermes_home",
        lambda: tmp_path,
    )
    assert await HindsightMemoryProvider().is_available() is True

    await _write_config(tmp_path, mode="local_embedded", apiKey="")

    async def unavailable():  # noqa: ASYNC124
        return False, "broken numpy runtime"

    monkeypatch.setattr(
        "plugins.memory.hindsight._check_local_runtime",
        unavailable,
    )
    assert await HindsightMemoryProvider().is_available() is False
    candidate = HindsightMemoryProvider()
    await candidate.initialize("session", hermes_home=str(tmp_path))
    assert candidate._mode == "disabled"
    await candidate.shutdown()


async def test_shutdown_drains_writer_closes_client_and_rejects_late_writes(
    tmp_path,
    monkeypatch,
):
    await _write_config(tmp_path)
    monkeypatch.setattr(
        "plugins.memory.hindsight.get_hermes_home",
        lambda: tmp_path,
    )
    candidate = HindsightMemoryProvider()
    await candidate.initialize("session", hermes_home=str(tmp_path))
    client = FakeClient()
    gate = asyncio.Event()
    client.retain_gate = gate
    candidate._client = client
    monkeypatch.setattr(
        "plugins.memory.hindsight._check_api_supports_update_mode_append",
        AsyncMock(return_value=True),
    )

    await candidate.sync_turn("one", "a")
    shutdown = asyncio.create_task(candidate.shutdown())
    await asyncio.sleep(0)
    assert not shutdown.done()
    gate.set()
    await shutdown
    assert len(client.retain_calls) == 1
    assert client.closed is True
    await candidate.sync_turn("late", "ignored")
    assert candidate._retain_queue.empty()


async def test_shutdown_finishes_cleanup_then_propagates_cancellation(
    tmp_path,
    monkeypatch,
):
    await _write_config(tmp_path)
    monkeypatch.setattr(
        "plugins.memory.hindsight.get_hermes_home",
        lambda: tmp_path,
    )
    candidate = HindsightMemoryProvider()
    await candidate.initialize("session", hermes_home=str(tmp_path))
    client = FakeClient()
    close_started = asyncio.Event()
    release_close = asyncio.Event()

    async def close():
        close_started.set()
        await release_close.wait()
        client.closed = True

    client.aclose = close
    candidate._client = client
    task = asyncio.create_task(candidate.shutdown())
    await close_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    release_close.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.closed is True
    assert candidate._client is None


async def test_embedded_daemon_uses_async_subprocess_boundary(
    tmp_path,
    monkeypatch,
):
    config = {
        "mode": "local_embedded",
        "profile": "hermes-test",
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
    }
    candidate = HindsightMemoryProvider()
    candidate._config = config
    candidate._timeout = 2
    commands: list[tuple[str, ...]] = []

    async def available():  # noqa: ASYNC124
        return True, None

    async def run(*args):
        commands.append(args)
        if args == ("daemon", "start"):
            path = _embedded_profile_env_path(config)
            values = await _load_simple_env(path)
            values["HINDSIGHT_API_PORT"] = "9234"
            async with aiofiles.open(path, "w", encoding="utf-8") as handle:
                await handle.write(
                    "".join(f"{key}={value}\n" for key, value in values.items())
                )
        return 0

    monkeypatch.setattr(
        "plugins.memory.hindsight._check_local_runtime",
        available,
    )
    monkeypatch.setattr(candidate, "_run_embedded_cli", run)
    await candidate._start_embedded_daemon()
    assert commands == [("daemon", "start")]
    assert candidate._api_url == "http://127.0.0.1:9234"


async def test_embedded_config_change_stops_before_restart(monkeypatch):
    config = {
        "mode": "local_embedded",
        "profile": "hermes-test",
        "llm_provider": "openai",
        "llm_model": "new-model",
    }
    path = _embedded_profile_env_path(config)
    await aiofiles.os.makedirs(path.parent, exist_ok=True)
    async with aiofiles.open(path, "w", encoding="utf-8") as handle:
        await handle.write(
            "HINDSIGHT_API_LLM_PROVIDER=openai\n"
            "HINDSIGHT_API_LLM_MODEL=old-model\n"
            "HINDSIGHT_API_PORT=9234\n"
        )
    candidate = HindsightMemoryProvider()
    candidate._config = config
    commands: list[tuple[str, ...]] = []

    async def available():  # noqa: ASYNC124
        return True, None

    async def run(*args):
        commands.append(args)
        if args == ("daemon", "start"):
            values = await _load_simple_env(path)
            values["HINDSIGHT_API_PORT"] = "9234"
            async with aiofiles.open(path, "w", encoding="utf-8") as handle:
                await handle.write(
                    "".join(f"{key}={value}\n" for key, value in values.items())
                )
        return 0

    monkeypatch.setattr(
        "plugins.memory.hindsight._check_local_runtime",
        available,
    )
    monkeypatch.setattr(candidate, "_run_embedded_cli", run)
    await candidate._start_embedded_daemon()
    assert commands == [("daemon", "stop"), ("daemon", "start")]


async def test_embedded_connection_retry_closes_stale_client(monkeypatch):
    candidate = HindsightMemoryProvider()
    candidate._mode = "local_embedded"
    candidate._config = {"profile": "hermes"}
    candidate._api_url = "http://127.0.0.1:9234"
    stale = FakeClient()

    async def fail(**kwargs):
        raise ConnectionRefusedError("connection refused")

    stale.arecall = fail
    candidate._client = stale
    replacement = FakeClient()

    class Factory:
        def __new__(cls, **kwargs):
            return replacement

    async def start():  # noqa: ASYNC124
        return None

    async def load():  # noqa: ASYNC124
        return Factory

    async def prepare(client):  # noqa: ASYNC124
        return None

    monkeypatch.setattr(candidate, "_start_embedded_daemon", start)
    monkeypatch.setattr(
        "plugins.memory.hindsight._load_hindsight_client_class",
        load,
    )
    monkeypatch.setattr(
        "plugins.memory.hindsight._prepare_hindsight_client_transport",
        prepare,
    )
    response = await candidate._run_hindsight_operation(
        lambda client: client.arecall(query="retry")
    )
    assert response.results[0].text == "Memory 1"
    assert stale.closed is True
    await candidate.shutdown()


async def test_concurrent_embedded_retries_do_not_close_recovered_client(monkeypatch):
    candidate = HindsightMemoryProvider()
    candidate._mode = "local_embedded"
    candidate._config = {"profile": "hermes"}
    candidate._api_url = "http://127.0.0.1:9234"
    stale = FakeClient()
    failed_calls = 0
    both_failed = asyncio.Event()

    async def fail(**kwargs):
        nonlocal failed_calls
        failed_calls += 1
        if failed_calls == 2:
            both_failed.set()
        await both_failed.wait()
        raise ConnectionRefusedError("connection refused")

    stale.arecall = fail
    candidate._client = stale
    replacement = FakeClient()
    factory_calls = 0

    class Factory:
        def __new__(cls, **kwargs):
            nonlocal factory_calls
            factory_calls += 1
            return replacement

    starts = 0

    async def start():
        nonlocal starts
        starts += 1

    async def load():  # noqa: ASYNC124
        return Factory

    async def prepare(client):  # noqa: ASYNC124
        return None

    monkeypatch.setattr(candidate, "_start_embedded_daemon", start)
    monkeypatch.setattr(
        "plugins.memory.hindsight._load_hindsight_client_class",
        load,
    )
    monkeypatch.setattr(
        "plugins.memory.hindsight._prepare_hindsight_client_transport",
        prepare,
    )
    responses = await asyncio.gather(
        *(
            candidate._run_hindsight_operation(
                lambda client: client.arecall(query="retry")
            )
            for _ in range(2)
        )
    )
    assert all(response.results[0].text == "Memory 1" for response in responses)
    assert stale.closed is True
    assert replacement.closed is False
    assert factory_calls == 1
    assert starts == 1
    await candidate.shutdown()


async def test_discovery_loads_hindsight_without_importing_optional_sdk(
    tmp_path,
    monkeypatch,
):
    await _write_config(tmp_path)
    monkeypatch.setattr(
        "plugins.memory.hindsight.get_hermes_home",
        lambda: tmp_path,
    )
    loaded = await load_memory_provider("hindsight")
    assert loaded is not None
    assert loaded.name == "hindsight"
    assert loaded._client is None
    await loaded.shutdown()


async def test_native_lifecycle_does_not_block_or_leak_tasks(
    tmp_path,
    monkeypatch,
):
    await _write_config(tmp_path)
    monkeypatch.setattr(
        "plugins.memory.hindsight.get_hermes_home",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "plugins.memory.hindsight._check_api_supports_update_mode_append",
        AsyncMock(return_value=True),
    )
    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        candidate = HindsightMemoryProvider()
        await candidate.initialize(
            "service-session",
            hermes_home=str(tmp_path),
            platform="service",
        )
        candidate._client = FakeClient()
        await candidate.sync_turn("hello", "hi")
        await candidate.queue_prefetch("hello")
        await candidate.prefetch("next")
        await candidate.handle_tool_call(
            "hindsight_recall",
            {"query": "hello"},
        )
        await candidate.on_session_switch(
            "next-session",
            parent_session_id="service-session",
        )
        await candidate.shutdown()
