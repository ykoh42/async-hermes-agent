"""Regression coverage for the async-first public core."""

import inspect
import asyncio
import json
import shlex
import sys
import time
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, patch

from agent.conversation_loop import run_conversation
from agent.conversation_compression import compress_context
from agent.chat_completion_helpers import handle_max_iterations
from agent.trajectory import save_trajectory
from agent.turn_context import build_turn_context
from agent.turn_finalizer import finalize_turn
from agent import tool_executor
from agent.tool_executor import execute_tool_calls_segmented
from hermes_state import AsyncSessionDB, SessionDB
from run_agent import AIAgent
from model_tools import handle_function_call
from tools.registry import registry
from tools.clarify_tool import clarify_tool
from tools.memory_tool import MemoryStore, memory_tool
from tools.skills_tool import skill_view, skills_list
from tools.file_tools import read_file_tool, write_file_tool
from tools.terminal_tool import terminal_tool


def test_conversation_and_chat_are_coroutines():
    from agent.agent_runtime_helpers import (
        invoke_tool,
        recover_with_credential_pool,
        try_recover_primary_transport,
    )
    from agent.credential_pool import load_pool
    from agent.auxiliary_client import _validate_llm_response
    from agent.auxiliary_client import (
        _aggregate_chat_stream,
        _create_with_progress,
        _create_with_stream,
    )
    from agent import relay_llm, relay_tools
    from tools.file_tools import (
        patch_tool,
        read_file_tool,
        search_tool,
        write_file_tool,
    )
    from batch_runner import BatchRunner
    from hermes_cli.middleware import run_llm_execution_middleware
    from hermes_cli.goals import (
        GoalManager,
        _get_session_db,
        draft_contract,
        judge_goal,
        load_goal,
        migrate_goal_to_session,
        save_goal,
    )

    assert inspect.iscoroutinefunction(AIAgent.run_conversation)
    assert inspect.iscoroutinefunction(AIAgent.chat)
    assert inspect.iscoroutinefunction(AIAgent.close)
    assert inspect.iscoroutinefunction(AIAgent.switch_model)
    assert inspect.iscoroutinefunction(AIAgent._ensure_db_session)
    assert inspect.iscoroutinefunction(AIAgent._persist_session)
    assert inspect.iscoroutinefunction(AIAgent._flush_messages_to_session_db)
    assert inspect.iscoroutinefunction(AIAgent._save_session_log)
    assert inspect.iscoroutinefunction(AIAgent.shutdown_memory_provider)
    assert inspect.iscoroutinefunction(AIAgent.commit_memory_session)
    assert inspect.iscoroutinefunction(AIAgent._handle_max_iterations)
    assert inspect.iscoroutinefunction(AIAgent._execute_tool_calls)
    assert inspect.iscoroutinefunction(AIAgent._conversation_root_id)
    assert inspect.iscoroutinefunction(handle_max_iterations)
    assert inspect.iscoroutinefunction(build_turn_context)
    assert inspect.iscoroutinefunction(finalize_turn)
    assert inspect.iscoroutinefunction(run_conversation)
    assert inspect.iscoroutinefunction(compress_context)
    assert inspect.iscoroutinefunction(execute_tool_calls_segmented)
    assert inspect.iscoroutinefunction(BatchRunner.run)
    assert inspect.iscoroutinefunction(terminal_tool)
    assert inspect.iscoroutinefunction(read_file_tool)
    assert inspect.iscoroutinefunction(write_file_tool)
    assert inspect.iscoroutinefunction(patch_tool)
    assert inspect.iscoroutinefunction(search_tool)
    assert inspect.iscoroutinefunction(clarify_tool)
    assert inspect.iscoroutinefunction(memory_tool)
    assert inspect.iscoroutinefunction(skills_list)
    assert inspect.iscoroutinefunction(skill_view)
    assert inspect.iscoroutinefunction(invoke_tool)
    assert inspect.iscoroutinefunction(_validate_llm_response)
    assert inspect.iscoroutinefunction(_aggregate_chat_stream)
    assert inspect.iscoroutinefunction(_create_with_progress)
    assert inspect.iscoroutinefunction(_create_with_stream)
    assert inspect.iscoroutinefunction(relay_llm.execute)
    assert inspect.iscoroutinefunction(relay_llm.execute_current)
    assert inspect.iscoroutinefunction(relay_llm.complete_logical_call)
    assert inspect.iscoroutinefunction(relay_tools.execute)
    assert inspect.iscoroutinefunction(run_llm_execution_middleware)
    assert inspect.iscoroutinefunction(_get_session_db)
    assert inspect.iscoroutinefunction(load_goal)
    assert inspect.iscoroutinefunction(save_goal)
    assert inspect.iscoroutinefunction(migrate_goal_to_session)
    assert inspect.iscoroutinefunction(judge_goal)
    assert inspect.iscoroutinefunction(draft_contract)
    for method_name in (
        "load", "set", "set_contract", "pause", "resume", "clear", "mark_done",
        "add_subgoal", "remove_subgoal", "clear_subgoals", "wait_on",
        "wait_on_session", "wait_for_seconds", "stop_waiting", "evaluate_after_turn",
    ):
        assert inspect.iscoroutinefunction(getattr(GoalManager, method_name))
    assert inspect.iscoroutinefunction(AIAgent._try_activate_fallback)
    assert inspect.iscoroutinefunction(AIAgent._try_recover_primary_transport)
    assert inspect.iscoroutinefunction(AIAgent._recover_with_credential_pool)
    assert inspect.iscoroutinefunction(AIAgent._swap_credential)
    assert inspect.iscoroutinefunction(recover_with_credential_pool)
    assert inspect.iscoroutinefunction(try_recover_primary_transport)
    assert inspect.iscoroutinefunction(load_pool)
    active_entries = list(registry._tools.values())
    assert active_entries
    assert all(entry.is_async for entry in active_entries)
    assert all(inspect.iscoroutinefunction(entry.handler) for entry in active_entries)


@pytest.mark.asyncio
async def test_async_session_billing_route_update_uses_native_connection(tmp_path):
    """A model switch must not fall back to the synchronous SessionDB writer."""
    database = SessionDB(db_path=tmp_path / "state.db")
    async_database = AsyncSessionDB(database)
    try:
        await async_database.create_session("route", "test", model="initial")
        await async_database.update_session_billing_route(
            "route",
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            billing_mode="chat_completions",
        )
        session = await async_database.get_session("route")
        assert session["billing_provider"] == "openrouter"
        assert session["billing_base_url"] == "https://openrouter.ai/api/v1"
        assert session["billing_mode"] == "chat_completions"
    finally:
        await async_database.close()
        database.close()


@pytest.mark.asyncio
async def test_async_session_db_can_bootstrap_from_a_path(tmp_path):
    """The active path does not need a synchronously opened SessionDB."""
    database = AsyncSessionDB(tmp_path / "state.db")
    try:
        await database.create_session("native", "test", model="test-model")
        await database.append_message("native", "user", "hello")
        await database.append_message("native", "assistant", "world")

        messages = await database.get_messages_as_conversation("native")

        assert [message["content"] for message in messages] == ["hello", "world"]
        assert (tmp_path / "state.db").exists()
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_async_session_db_meta_and_gateway_listing_match_native_shape(tmp_path):
    """Management readers can migrate without reopening the sync database."""
    database = AsyncSessionDB(tmp_path / "state.db")
    try:
        await database.set_meta("goal:session", '{"status":"active"}')
        assert await database.get_meta("goal:session") == '{"status":"active"}'
        assert await database.delete_meta("missing") is False
        assert await database.delete_meta("goal:session") is True
        assert await database.get_meta("goal:session") is None

        await database.create_session(
            "gateway-old",
            "telegram",
            session_key="chat:1",
            started_at=1.0,
        )
        await database.create_session(
            "gateway-new",
            "telegram",
            session_key="chat:1",
            started_at=2.0,
        )
        await database.create_session(
            "gateway-other",
            "discord",
            session_key="chat:2",
            started_at=3.0,
        )
        await database.end_session("gateway-old", "superseded")

        rows = await database.list_gateway_sessions(platform="telegram")
        assert [row["id"] for row in rows] == ["gateway-new"]
        assert rows[0]["session_key"] == "chat:1"
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_native_file_read_deduplicates_and_write_invalidates(tmp_path, monkeypatch):
    """File-loop guards survive the sync-backend removal on native I/O."""
    monkeypatch.setattr("tools.file_tools._check_sensitive_path", lambda *_args: None)
    path = tmp_path / "sample.txt"
    path.write_text("first\nsecond\n", encoding="utf-8")

    first = json.loads(await read_file_tool(str(path), task_id="native-file"))
    second = json.loads(await read_file_tool(str(path), task_id="native-file"))

    assert "content" in first
    assert second["dedup"] is True
    assert second["status"] == "unchanged"

    written = json.loads(
        await write_file_tool(str(path), "updated\n", task_id="native-file")
    )
    refreshed = json.loads(await read_file_tool(str(path), task_id="native-file"))

    assert written["files_modified"] == [str(path)]
    assert "updated" in refreshed["content"]
    assert refreshed.get("dedup") is not True


@pytest.mark.asyncio
async def test_model_switch_route_is_deferred_to_the_async_turn_boundary(tmp_path):
    """The sync state switch never writes SQLite directly."""
    database = SessionDB(db_path=tmp_path / "state.db")
    async_database = AsyncSessionDB(database)
    agent = AIAgent.__new__(AIAgent)
    agent._session_db = database
    agent._async_session_db = async_database
    agent._persist_disabled = False
    agent.session_id = "route"
    agent._pending_billing_route = {
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "billing_mode": "chat_completions",
    }
    try:
        await async_database.create_session("route", "test")
        await agent._persist_pending_billing_route()
        session = await async_database.get_session("route")
        assert session["billing_provider"] == "openrouter"
        assert agent._pending_billing_route is None
    finally:
        await async_database.close()
        database.close()


@pytest.mark.asyncio
async def test_model_switch_uses_deferred_native_provider_runtime(monkeypatch):
    """Switching providers must not rebuild a synchronous SDK client."""
    from agent.agent_runtime_helpers import switch_model

    monkeypatch.setattr(
        "agent.chat_completion_helpers._reset_stale_streak", lambda _agent: None
    )
    agent = SimpleNamespace(
        provider="openrouter",
        requested_provider="openrouter",
        model="old-model",
        api_key="old-key",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        _deferred_provider_runtime=None,
        _fallback_chain=[
            {"provider": "openrouter", "model": "old-fallback"},
            {"provider": "groq", "model": "keep"},
        ],
        _fallback_model=None,
        _fallback_activated=True,
        _fallback_index=1,
        _client_kwargs={},
        _use_prompt_caching=False,
        _use_native_cache_layout=False,
        reasoning_config={"enabled": True},
        context_compressor=None,
        _session_db=None,
        _pending_billing_route=None,
        _cached_system_prompt="stale",
    )

    async def ensure_runtime():
        pending = agent._deferred_provider_runtime
        agent.provider = pending["provider"]
        agent.requested_provider = pending["provider"]
        agent.model = pending["model"]
        agent.api_key = pending["api_key"]
        agent.base_url = pending["base_url"]
        agent.api_mode = pending["api_mode"]
        agent._async_provider_request_timeout = None
        agent._async_provider_stale_timeout = None
        agent._deferred_provider_runtime = None

    async def persist_route():
        agent.persisted_route = dict(agent._pending_billing_route)
        agent._pending_billing_route = None

    agent._ensure_provider_runtime = ensure_runtime
    agent._persist_pending_billing_route = persist_route
    agent._create_openai_client = lambda *_args, **_kwargs: pytest.fail(
        "switch_model must not construct a synchronous client"
    )

    await switch_model(
        agent,
        new_model="new-model",
        new_provider="groq",
        api_key="new-key",
        base_url="https://api.groq.com/openai/v1",
        api_mode="chat_completions",
    )

    assert agent.model == "new-model"
    assert agent.provider == "groq"
    assert agent._cached_system_prompt is None
    assert agent._fallback_chain == []
    assert agent.persisted_route["provider"] == "groq"


@pytest.mark.asyncio
async def test_async_plugin_lifecycle_requires_coroutine_callbacks():
    """Sync plugin hooks cannot quietly stall an async agent turn."""
    from hermes_cli.plugins import AsyncPluginCapabilityError, PluginManager

    manager = PluginManager()
    manager._hooks["turn"] = [lambda **_kwargs: None]
    with pytest.raises(AsyncPluginCapabilityError, match="coroutine lifecycle hooks"):
        await manager.invoke_hook_async("turn")

    async def callback(**_kwargs):
        return "native"

    manager._hooks["turn"] = [callback]
    assert await manager.invoke_hook_async("turn") == ["native"]


@pytest.mark.asyncio
async def test_external_memory_manager_fails_fast_without_running_sync_hooks():
    """An optional legacy memory provider cannot block an async turn."""
    from agent.agent_runtime_helpers import AsyncCapabilityError
    from agent.conversation_compression import compress_context

    class SyncManager:
        def sync_all(self, *_args, **_kwargs):
            pytest.fail("legacy memory sync must never run on the async loop")

        def queue_prefetch_all(self, *_args, **_kwargs):
            pytest.fail("legacy memory prefetch must never run on the async loop")

    agent = AIAgent.__new__(AIAgent)
    agent._memory_manager = SyncManager()

    with pytest.raises(AsyncCapabilityError, match="External MemoryManager"):
        await agent.shutdown_memory_provider([])

    with pytest.raises(AsyncCapabilityError, match="External MemoryManager"):
        await compress_context(agent, [], "system")


@pytest.mark.asyncio
async def test_deferred_runtime_rejects_sync_only_extension_surfaces_early():
    """Provider construction must stop before an external legacy extension runs."""
    from agent.agent_init import initialize_deferred_runtime
    from agent.agent_runtime_helpers import AsyncCapabilityError

    async def assert_rejected(**attributes):
        state = {
            "_deferred_provider_runtime": {"provider": "openrouter", "model": "test"},
            "_async_provider_init_lock": None,
            "_memory_manager": None,
            "_async_unsupported_context_engine": None,
        }
        state.update(attributes)
        agent = SimpleNamespace(**state)
        with pytest.raises(AsyncCapabilityError):
            await initialize_deferred_runtime(agent)

    await assert_rejected(_memory_manager=object())
    await assert_rejected(_async_unsupported_context_engine="third-party")




def test_api_timeout_resolution_uses_the_constructor_snapshot(monkeypatch):
    """Core request construction must not reread provider settings mid-turn."""
    import run_agent

    def fail_if_settings_are_read(*_args, **_kwargs):
        raise AssertionError("a native async turn must use its timeout snapshot")

    monkeypatch.setattr(run_agent, "get_provider_request_timeout", fail_if_settings_are_read)
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", fail_if_settings_are_read)
    agent = SimpleNamespace(
        _async_provider_request_timeout=42.0,
        _async_provider_stale_timeout=84.0,
        provider="custom",
        model="test-model",
    )

    assert AIAgent._resolved_api_call_timeout(agent) == 42.0
    assert AIAgent._resolved_api_call_stale_timeout_base(agent) == (84.0, False)


@pytest.mark.asyncio
async def test_api_key_pool_rotation_is_native_async_and_persistent(monkeypatch, tmp_path):
    """A billing failure rotates pool keys without a thread-bound I/O path."""
    from agent.agent_runtime_helpers import recover_with_credential_pool
    from agent.credential_pool import AUTH_TYPE_API_KEY, CredentialPool, PooledCredential
    from agent.error_classifier import FailoverReason

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    first = PooledCredential(
        provider="openrouter",
        id="first",
        label="first",
        auth_type=AUTH_TYPE_API_KEY,
        priority=0,
        source="manual",
        access_token="first-key",
    )
    second = PooledCredential(
        provider="openrouter",
        id="second",
        label="second",
        auth_type=AUTH_TYPE_API_KEY,
        priority=1,
        source="manual",
        access_token="second-key",
    )
    pool = CredentialPool("openrouter", [first, second])
    pool._current_id = first.id
    agent = SimpleNamespace(
        provider="openrouter",
        api_key="first-key",
        base_url="https://openrouter.ai/api/v1",
        _credential_pool=pool,
        _credential_pool_entry_id=first.id,
        log_prefix="",
    )

    async def swap(entry):
        agent.api_key = entry.runtime_api_key
        agent._credential_pool_entry_id = entry.id

    agent._swap_credential = swap
    monkeypatch.setattr(
        asyncio,
        "to_thread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("credential rotation must not call asyncio.to_thread")
        ),
    )

    recovered, retried_429 = await recover_with_credential_pool(
        agent,
        status_code=402,
        has_retried_429=False,
        classified_reason=FailoverReason.billing,
    )

    assert (recovered, retried_429) == (True, False)
    assert agent.api_key == "second-key"
    persisted = json.loads((hermes_home / "auth.json").read_text())
    by_id = {entry["id"]: entry for entry in persisted["credential_pool"]["openrouter"]}
    assert by_id["first"]["last_status"] == "exhausted"
    assert by_id["second"].get("last_status") is None


@pytest.mark.asyncio
async def test_native_pool_loader_reads_persisted_entries_without_sync_loader(monkeypatch, tmp_path):
    """A fallback/restore turn reads its pool through the async auth boundary."""
    from agent.credential_pool import load_pool

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {},
                "credential_pool": {
                    "openrouter": [
                        {
                            "id": "pool-key",
                            "label": "OpenRouter",
                            "auth_type": "api_key",
                            "priority": 0,
                            "source": "manual",
                            "access_token": "pool-token",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    def fail_if_threaded(*_args, **_kwargs):
        raise AssertionError("pool loader must not use asyncio.to_thread")

    monkeypatch.setattr(asyncio, "to_thread", fail_if_threaded)
    pool = await load_pool("openrouter")
    selected = await pool.select()

    assert selected is not None
    assert selected.runtime_api_key == "pool-token"


@pytest.mark.asyncio
async def test_deferred_runtime_initializes_from_async_pool_without_legacy_router(
    monkeypatch, tmp_path,
):
    """No-key construction defers auth.json I/O until a native async turn starts."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {},
                "credential_pool": {
                    "openrouter": [
                        {
                            "id": "pool-key",
                            "label": "OpenRouter",
                            "auth_type": "api_key",
                            "priority": 0,
                            "source": "manual",
                            "access_token": "pool-token",
                            "base_url": "https://openrouter.ai/api/v1",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            side_effect=AssertionError("legacy provider router must not run"),
        ),
    ):
        agent = AIAgent(
            provider="openrouter",
            model="openai/gpt-4.1-mini",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    def fail_if_threaded(*_args, **_kwargs):
        raise AssertionError("deferred provider initialization must not use asyncio.to_thread")

    monkeypatch.setattr(asyncio, "to_thread", fail_if_threaded)
    try:
        assert agent._deferred_provider_runtime is not None
        assert await agent._ensure_provider_runtime() is True
        assert agent._deferred_provider_runtime is None
        assert agent.api_key == "pool-token"
        assert agent.base_url == "https://openrouter.ai/api/v1"
        assert getattr(agent.client, "_sync", None) is None
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_explicit_credentials_defer_sdk_construction_until_await(monkeypatch):
    """``AIAgent.__init__`` remains state-only even with an explicit key."""
    native_client = SimpleNamespace(close=AsyncMock(), _platform=None)
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch(
            "run_agent.OpenAI",
            side_effect=AssertionError("__init__ must not create a sync client"),
        ),
        patch("openai.AsyncOpenAI", return_value=native_client) as async_openai,
    ):
        agent = AIAgent(
            provider="custom",
            model="test-model",
            api_key="explicit-key",
            base_url="https://example.invalid/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        assert agent.client is None
        assert agent._deferred_provider_runtime is not None
        async_openai.assert_not_called()

        monkeypatch.setattr(
            asyncio,
            "to_thread",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("deferred initialization must not thread-hop"),
            ),
        )
        await agent._ensure_provider_runtime()

    async_openai.assert_called_once()
    assert agent.client is native_client
    assert agent.client._platform == "Unknown"
    await agent.close()


@pytest.mark.asyncio
async def test_custom_env_key_preserves_constructor_route_and_tls_snapshot(
    monkeypatch, tmp_path,
):
    """Async initialization never rereads TLS config for an env-key custom route."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    tls_snapshot = object()
    transport = object()
    native_client = SimpleNamespace(aclose=AsyncMock(), _platform=None)

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch(
            "run_agent.OpenAI",
            side_effect=AssertionError("__init__ must not create a sync client"),
        ),
        patch("hermes_cli.config.load_config", return_value={
            "custom_providers": [{
                "name": "Local endpoint",
                "base_url": "https://custom.example/v1",
                "ssl_verify": False,
            }],
        }),
        patch("agent.agent_init.resolve_httpx_verify", return_value=tls_snapshot),
        patch(
            "agent.process_bootstrap.build_keepalive_http_client",
            return_value=transport,
        ) as build_http_client,
        patch("openai.AsyncOpenAI", return_value=native_client),
    ):
        agent = AIAgent(
            provider="custom",
            model="test-model",
            base_url="https://custom.example/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        monkeypatch.setattr(
            asyncio,
            "to_thread",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("native initialization must not thread-hop"),
            ),
        )

        await agent._ensure_provider_runtime()

    assert agent.api_key == "environment-key"
    assert agent.base_url == "https://custom.example/v1"
    build_http_client.assert_called_once_with(
        "https://custom.example/v1",
        verify=tls_snapshot,
    )
    await agent.close()


@pytest.mark.asyncio
async def test_turn_retries_with_rotated_api_key_pool_entry(monkeypatch, tmp_path):
    """The live conversation loop resumes on the next API-key pool entry."""
    from agent.credential_pool import AUTH_TYPE_API_KEY, CredentialPool, PooledCredential

    class BillingError(RuntimeError):
        status_code = 402

    first = PooledCredential(
        provider="openrouter",
        id="first",
        label="first",
        auth_type=AUTH_TYPE_API_KEY,
        priority=0,
        source="manual",
        access_token="first-key",
    )
    second = PooledCredential(
        provider="openrouter",
        id="second",
        label="second",
        auth_type=AUTH_TYPE_API_KEY,
        priority=1,
        source="manual",
        access_token="second-key",
    )
    pool = CredentialPool("openrouter", [first, second])
    pool._current_id = first.id
    database = SessionDB(tmp_path / "state.db")
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            provider="openrouter",
            api_key="first-key",
            base_url="https://openrouter.ai/api/v1",
            model="test-model",
            credential_pool=pool,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            save_trajectories=False,
        )
    agent._session_db = database
    agent._async_session_db = AsyncSessionDB(database)
    agent._session_db_created = False
    agent.compression_enabled = False
    attempts = 0

    async def model_response(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise BillingError("first account has no credits")
        return SimpleNamespace(
            id="complete",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        role="assistant",
                        content="rotated successfully",
                        tool_calls=None,
                        reasoning=None,
                        reasoning_content=None,
                    ),
                )
            ],
            usage=None,
        )

    agent._execute_model_request = model_response
    monkeypatch.setattr(
        asyncio,
        "to_thread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("live credential rotation must not use asyncio.to_thread")
        ),
    )
    try:
        result = await agent.run_conversation("continue with another account")
        assert result["final_response"] == "rotated successfully"
        assert attempts == 2
        assert agent.api_key == "second-key"
    finally:
        await agent.close()
        database.close()


@pytest.mark.asyncio
async def test_anthropic_pool_refresh_uses_native_async_transport(monkeypatch, tmp_path):
    """Anthropic OAuth refresh never reaches urllib or the sync pool method."""
    from agent.credential_pool import AUTH_TYPE_OAUTH, CredentialPool, PooledCredential

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    entry = PooledCredential(
        provider="anthropic",
        id="oauth",
        label="oauth",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="manual",
        access_token="expired-access",
        refresh_token="refresh-token",
        expires_at_ms=0,
    )
    pool = CredentialPool("anthropic", [entry])

    async def refresh(refresh_token, *, use_json):
        assert refresh_token == "refresh-token"
        assert use_json is False
        return {
            "access_token": "fresh-access",
            "refresh_token": "fresh-refresh",
            "expires_at_ms": 4_102_444_800_000,
        }

    monkeypatch.setattr(
        "agent.anthropic_adapter.refresh_anthropic_oauth_pure", refresh,
    )
    monkeypatch.setattr(
        asyncio,
        "to_thread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("OAuth refresh must not call asyncio.to_thread")
        ),
    )

    refreshed = await pool.try_refresh_matching(credential_id="oauth")

    assert refreshed is not None
    assert refreshed.access_token == "fresh-access"
    persisted = json.loads((hermes_home / "auth.json").read_text())
    assert persisted["credential_pool"]["anthropic"][0]["access_token"] == "fresh-access"


@pytest.mark.asyncio
async def test_unsupported_oauth_pool_refresh_fails_fast_without_a_thread(monkeypatch):
    """A provider without a native OAuth lifecycle is never silently bridged."""
    from agent.agent_runtime_helpers import AsyncCapabilityError
    from agent.credential_pool import AUTH_TYPE_OAUTH, CredentialPool, PooledCredential

    pool = CredentialPool(
        "openai-codex",
        [
            PooledCredential(
                provider="openai-codex",
                id="codex",
                label="Codex",
                auth_type=AUTH_TYPE_OAUTH,
                priority=0,
                source="manual",
                access_token="expired-access",
                refresh_token="refresh-token",
            )
        ],
    )

    def fail_if_threaded(*_args, **_kwargs):
        raise AssertionError("unsupported OAuth must not use asyncio.to_thread")

    monkeypatch.setattr(asyncio, "to_thread", fail_if_threaded)
    with pytest.raises(AsyncCapabilityError, match="openai-codex"):
        await pool.try_refresh_matching(credential_id="codex")


@pytest.mark.asyncio
async def test_cancelling_native_oauth_refresh_propagates_without_partial_write(
    monkeypatch, tmp_path,
):
    """Cancellation keeps the pool entry intact and is never converted to retry state."""
    from agent.credential_pool import AUTH_TYPE_OAUTH, CredentialPool, PooledCredential

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    pool = CredentialPool(
        "anthropic",
        [
            PooledCredential(
                provider="anthropic",
                id="anthropic",
                label="Anthropic",
                auth_type=AUTH_TYPE_OAUTH,
                priority=0,
                source="manual",
                access_token="expired-access",
                refresh_token="refresh-token",
                expires_at_ms=0,
            )
        ],
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked_refresh(*_args, **_kwargs):
        started.set()
        await release.wait()
        raise AssertionError("cancelled refresh must not resume")

    monkeypatch.setattr(
        "agent.anthropic_adapter.refresh_anthropic_oauth_pure",
        blocked_refresh,
    )
    task = asyncio.create_task(pool.try_refresh_matching(credential_id="anthropic"))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    entry = pool.entries()[0]
    assert entry.access_token == "expired-access"
    assert not (hermes_home / "auth.json").exists()


@pytest.mark.asyncio
async def test_turn_retries_after_native_anthropic_oauth_refresh(monkeypatch, tmp_path):
    """A 401 refreshes the active Anthropic pool entry and resumes the turn."""
    from agent.credential_pool import AUTH_TYPE_OAUTH, CredentialPool, PooledCredential

    class UnauthorizedError(RuntimeError):
        status_code = 401

    entry = PooledCredential(
        provider="anthropic",
        id="oauth",
        label="oauth",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="manual",
        access_token="expired-access",
        refresh_token="refresh-token",
        expires_at_ms=4_102_444_800_000,
    )
    pool = CredentialPool("anthropic", [entry])
    pool._current_id = entry.id
    database = SessionDB(tmp_path / "state.db")
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            provider="anthropic",
            api_mode="chat_completions",
            api_key="expired-access",
            base_url="https://example.invalid/v1",
            model="test-model",
            credential_pool=pool,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            save_trajectories=False,
        )
    agent._session_db = database
    agent._async_session_db = AsyncSessionDB(database)
    agent._session_db_created = False
    agent.compression_enabled = False
    agent._is_entitlement_failure = lambda *_args, **_kwargs: False
    attempts = 0

    async def refresh(*_args, **_kwargs):
        return {
            "access_token": "fresh-access",
            "refresh_token": "fresh-refresh",
            "expires_at_ms": 4_102_444_800_000,
        }

    async def model_response(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise UnauthorizedError("expired token")
        return SimpleNamespace(
            id="complete",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        role="assistant",
                        content="refreshed successfully",
                        tool_calls=None,
                        reasoning=None,
                        reasoning_content=None,
                    ),
                )
            ],
            usage=None,
        )

    class NativeAnthropicClient:
        async def close(self):
            return None

    monkeypatch.setattr(
        "agent.anthropic_adapter.refresh_anthropic_oauth_pure", refresh,
    )
    monkeypatch.setattr(
        "agent.anthropic_adapter.build_anthropic_client",
        lambda *_args, **_kwargs: NativeAnthropicClient(),
    )
    monkeypatch.setattr(
        asyncio,
        "to_thread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("live OAuth refresh must not call asyncio.to_thread")
        ),
    )
    agent._execute_model_request = model_response
    try:
        result = await agent.run_conversation("refresh and continue")
        assert result["final_response"] == "refreshed successfully"
        assert attempts == 2
        assert agent.api_key == "fresh-access"
    finally:
        await agent.close()
        database.close()


def test_native_file_tool_imports_expand_the_async_model_surface():
    """File tools join the model surface only after native migration."""
    from tools import file_tools  # noqa: F401

    assert registry.get_entry("read_file").is_async is True
    assert registry.get_entry("write_file").is_async is True


@pytest.mark.asyncio
async def test_native_file_tools_do_not_use_a_sync_dispatch_bridge(monkeypatch, tmp_path):
    """A model can write, patch, read, and search without ``to_thread``."""
    import importlib

    file_tools = importlib.import_module("tools.file_tools")
    importlib.reload(file_tools)
    from tools.registry import registry as active_registry

    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    # macOS test temp directories live under /private/var, a system prefix
    # the production write guard correctly protects from model edits.
    monkeypatch.setattr(file_tools, "_check_sensitive_path", lambda *_args: None)
    monkeypatch.setattr(
        asyncio,
        "to_thread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("native file tools must not call asyncio.to_thread")
        ),
    )

    written = json.loads(
        await active_registry.dispatch(
            "write_file",
            {"path": "notes.txt", "content": "first\nsecond\n"},
            task_id="native-file-test",
        )
    )
    assert written["bytes_written"] == len("first\nsecond\n".encode())

    patched = json.loads(
        await active_registry.dispatch(
            "patch",
            {
                "path": "notes.txt",
                "old_string": "second",
                "new_string": "third",
            },
            task_id="native-file-test",
        )
    )
    assert patched["replacements"] == 1

    read = json.loads(
        await active_registry.dispatch(
            "read_file", {"path": "notes.txt"}, task_id="native-file-test"
        )
    )
    assert read["content"] == "1|first\n2|third"

    found = json.loads(
        await active_registry.dispatch(
            "search_files",
            {"pattern": "third", "path": "."},
            task_id="native-file-test",
        )
    )
    assert found["total_count"] == 1


@pytest.mark.asyncio
async def test_native_v4a_patch_is_async_and_validates_before_mutating(monkeypatch, tmp_path):
    """Multi-file edits keep V4A semantics without a synchronous backend."""
    import importlib

    file_tools = importlib.import_module("tools.file_tools")
    importlib.reload(file_tools)
    from tools.registry import registry as active_registry

    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.setattr(file_tools, "_check_sensitive_path", lambda *_args: None)
    monkeypatch.setattr(
        asyncio,
        "to_thread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("native V4A patch must not call asyncio.to_thread")
        ),
    )
    (tmp_path / "target.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    (tmp_path / "obsolete.txt").write_text("remove me\n", encoding="utf-8")

    result = json.loads(
        await active_registry.dispatch(
            "patch",
            {
                "mode": "patch",
                "patch": (
                    "*** Begin Patch\n"
                    "*** Update File: target.txt\n"
                    "@@\n"
                    " alpha\n"
                    "-beta\n"
                    "+gamma\n"
                    "*** Add File: created.txt\n"
                    "+created\n"
                    "*** Delete File: obsolete.txt\n"
                    "*** End Patch"
                ),
            },
            task_id="native-v4a-test",
        )
    )
    assert result["success"] is True
    assert (tmp_path / "target.txt").read_text(encoding="utf-8") == "alpha\ngamma\n"
    assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "created"
    assert not (tmp_path / "obsolete.txt").exists()

    invalid = json.loads(
        await active_registry.dispatch(
            "patch",
            {
                "mode": "patch",
                "patch": (
                    "*** Begin Patch\n"
                    "*** Update File: target.txt\n"
                    "@@\n"
                    "-does-not-exist\n"
                    "+replacement\n"
                    "*** Add File: must-not-exist.txt\n"
                    "+nope\n"
                    "*** End Patch"
                ),
            },
            task_id="native-v4a-test",
        )
    )
    assert "error" in invalid
    assert not (tmp_path / "must-not-exist.txt").exists()


@pytest.mark.asyncio
async def test_clarify_tool_awaits_the_platform_callback():
    """HITL remains a scheduler barrier without blocking the event loop."""
    async def callback(question, choices, *, multi_select=False):
        assert question == "Choose a trajectory"
        assert choices == ["A", "B"]
        assert multi_select is False
        await asyncio.sleep(0)
        return "B"

    response = json.loads(
        await registry.dispatch(
            "clarify",
            {"question": "Choose a trajectory", "choices": ["A", "B"]},
            callback=callback,
        )
    )
    assert response["user_response"] == "B"


@pytest.mark.asyncio
async def test_fallback_swaps_to_a_native_async_client(monkeypatch):
    """A fallback client must be awaited directly, never thread-wrapped."""

    class NativeCompletions:
        async def create(self, **_kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="fallback answer"),
                    )
                ]
            )

    class NativeClient:
        _sync = None
        api_key = "fallback-key"
        base_url = "https://fallback.example/v1"

        def __init__(self):
            self.chat = SimpleNamespace(completions=NativeCompletions())

        async def close(self):
            return None

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch(
            "agent.credential_pool.load_pool",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(entries=lambda: []),
        ),
        patch("hermes_cli.config.load_config", return_value={}),
    ):
        agent = AIAgent(
            provider="custom",
            base_url="https://primary.example/v1",
            api_key="primary-key",
            model="primary-model",
            fallback_model=[
                {
                    "provider": "custom",
                    "model": "fallback-model",
                    "base_url": "https://fallback.example/v1",
                    "api_key": "fallback-key",
                }
            ],
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    native_client = NativeClient()
    def fail_if_threaded(*_args, **_kwargs):
        raise AssertionError("fallback must not use asyncio.to_thread")

    monkeypatch.setattr(asyncio, "to_thread", fail_if_threaded)
    try:
        with patch("openai.AsyncOpenAI", return_value=native_client) as async_openai:
            assert await agent._try_activate_fallback()
            response = await agent._execute_model_request(
                {"model": agent.model, "messages": []}
            )

        async_openai.assert_called_once()
        assert agent.client is native_client
        assert response.choices[0].message.content == "fallback answer"
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_agent_turn_lock_serializes_one_instance():
    agent = AIAgent.__new__(AIAgent)
    running = 0
    maximum = 0

    async def enter_turn():
        nonlocal running, maximum
        async with agent._get_async_turn_lock():
            running += 1
            maximum = max(maximum, running)
            await asyncio.sleep(0)
            running -= 1

    await asyncio.gather(enter_turn(), enter_turn())
    assert maximum == 1


@pytest.mark.asyncio
async def test_distinct_agents_can_run_turns_in_parallel():
    first_agent = AIAgent.__new__(AIAgent)
    second_agent = AIAgent.__new__(AIAgent)
    running = 0
    maximum = 0

    async def enter_turn(agent):
        nonlocal running, maximum
        async with agent._get_async_turn_lock():
            running += 1
            maximum = max(maximum, running)
            await asyncio.sleep(0)
            running -= 1

    await asyncio.gather(enter_turn(first_agent), enter_turn(second_agent))
    assert maximum == 2


@pytest.mark.asyncio
async def test_run_conversation_serializes_turns_for_one_agent(monkeypatch, tmp_path):
    """The public turn API, not only its raw lock, serializes one agent."""
    database = SessionDB(tmp_path / "state.db")
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            model="test-model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            save_trajectories=False,
        )
    agent._session_db = database
    agent._async_session_db = AsyncSessionDB(database)
    agent._session_db_created = False
    agent.compression_enabled = False
    agent._skip_mcp_refresh = True

    first_model_call = asyncio.Event()
    release_model = asyncio.Event()
    model_calls = 0
    model_started_at = None

    async def slow_model(*_args, **_kwargs):
        nonlocal model_calls, model_started_at
        model_calls += 1
        model_started_at = time.monotonic()
        first_model_call.set()
        await release_model.wait()
        return SimpleNamespace(
            id=f"response-{model_calls}",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        role="assistant",
                        content=f"answer-{model_calls}",
                        tool_calls=None,
                        reasoning=None,
                        reasoning_content=None,
                    ),
                )
            ],
            usage=None,
        )

    agent._execute_model_request = slow_model
    monkeypatch.setattr(
        "tools.env_probe.get_environment_probe_line",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("async prompt construction must not wait for env_probe")
        ),
    )
    monkeypatch.setattr(
        asyncio,
        "to_thread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("public async turns must not call asyncio.to_thread")
        ),
    )
    submitted_at = time.monotonic()
    first = asyncio.create_task(agent.run_conversation("first"))
    second = None
    try:
        await asyncio.wait_for(first_model_call.wait(), timeout=2)
        assert model_started_at is not None
        assert model_started_at - submitted_at < 0.25
        second = asyncio.create_task(agent.run_conversation("second"))
        await asyncio.sleep(0.05)
        assert model_calls == 1
        assert not second.done()

        release_model.set()
        first_result, second_result = await asyncio.gather(first, second)
        assert first_result["final_response"] == "answer-1"
        assert second_result["final_response"] == "answer-2"
    finally:
        release_model.set()
        pending = [task for task in (first, second) if task is not None and not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await agent.close()
        database.close()


@pytest.mark.asyncio
async def test_default_compression_prologue_uses_static_context_metadata(monkeypatch, tmp_path):
    """The default compression path must not synchronously discover models."""
    database = SessionDB(tmp_path / "state.db")
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            model="test-model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            save_trajectories=False,
        )
    agent._session_db = database
    agent._async_session_db = AsyncSessionDB(database)
    agent._session_db_created = False
    agent._skip_mcp_refresh = True

    async def model_response(*_args, **_kwargs):
        return SimpleNamespace(
            id="response-1",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        role="assistant",
                        content="answer",
                        tool_calls=None,
                        reasoning=None,
                        reasoning_content=None,
                    ),
                )
            ],
            usage=None,
        )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("async compression prologue attempted blocking I/O")

    agent._execute_model_request = model_response
    monkeypatch.setattr("agent.model_metadata._ensure_requests", fail_if_called)
    monkeypatch.setattr(asyncio, "to_thread", fail_if_called)
    try:
        result = await agent.run_conversation("hello")
        assert result["final_response"] == "answer"
    finally:
        await agent.close()
        database.close()


@pytest.mark.asyncio
async def test_run_conversation_allows_distinct_agents_to_overlap(monkeypatch, tmp_path):
    """Separate agent instances keep their model I/O concurrent."""
    databases = [SessionDB(tmp_path / f"state-{index}.db") for index in range(2)]
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agents = [
            AIAgent(
                api_key="test-key",
                base_url="https://example.invalid/v1",
                model="test-model",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                save_trajectories=False,
            )
            for _ in range(2)
        ]
    for agent, database in zip(agents, databases):
        agent._session_db = database
        agent._async_session_db = AsyncSessionDB(database)
        agent._session_db_created = False
        agent.compression_enabled = False

    both_models_started = asyncio.Event()
    release_models = asyncio.Event()
    running_models = 0
    maximum_models = 0

    def make_model(label):
        async def slow_model(*_args, **_kwargs):
            nonlocal running_models, maximum_models
            running_models += 1
            maximum_models = max(maximum_models, running_models)
            if maximum_models == 2:
                both_models_started.set()
            try:
                await release_models.wait()
                return SimpleNamespace(
                    id=f"response-{label}",
                    choices=[
                        SimpleNamespace(
                            finish_reason="stop",
                            message=SimpleNamespace(
                                role="assistant",
                                content=f"answer-{label}",
                                tool_calls=None,
                                reasoning=None,
                                reasoning_content=None,
                            ),
                        )
                    ],
                    usage=None,
                )
            finally:
                running_models -= 1

        return slow_model

    for index, agent in enumerate(agents):
        agent._execute_model_request = make_model(index)
    monkeypatch.setattr(
        asyncio,
        "to_thread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("public async turns must not call asyncio.to_thread")
        ),
    )
    tasks = [
        asyncio.create_task(agent.run_conversation(f"prompt-{index}"))
        for index, agent in enumerate(agents)
    ]
    try:
        await asyncio.wait_for(both_models_started.wait(), timeout=0.5)
        assert maximum_models == 2
        release_models.set()
        results = await asyncio.gather(*tasks)
        assert [result["final_response"] for result in results] == [
            "answer-0",
            "answer-1",
        ]
    finally:
        release_models.set()
        pending = [task for task in tasks if not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for agent in agents:
            await agent.close()
        for database in databases:
            database.close()


@pytest.mark.asyncio
async def test_mcp_work_stays_on_the_calling_event_loop(monkeypatch):
    """MCP transport work must not cross into a legacy background loop."""
    from tools.mcp_tool import _await_mcp_operation

    seen_loops = []

    def fail_if_called(*args, **kwargs):
        raise AssertionError("MCP async path must not call asyncio.to_thread")

    monkeypatch.setattr(asyncio, "to_thread", fail_if_called)

    async def operation():
        seen_loops.append(asyncio.get_running_loop())
        return "ok"

    caller_loop = asyncio.get_running_loop()
    assert await _await_mcp_operation(operation, timeout=1) == "ok"
    assert seen_loops == [caller_loop]


@pytest.mark.asyncio
async def test_close_is_awaitable_and_idempotent():
    agent = AIAgent.__new__(AIAgent)

    await agent.close()
    await agent.close()

    assert agent._closed is True


@pytest.mark.asyncio
async def test_close_never_calls_the_sync_client_close():
    """The async lifecycle owns only native async transports."""
    agent = AIAgent.__new__(AIAgent)
    closed_native = False

    class SyncSource:
        def close(self):
            raise AssertionError("async close must not call sync client.close()")

    class NativeClient:
        async def aclose(self):
            nonlocal closed_native
            closed_native = True

    agent.client = SyncSource()
    agent._async_client = NativeClient()
    await agent.close()

    assert closed_native is True
    assert agent.client is None


@pytest.mark.asyncio
async def test_trajectory_writer_is_awaitable(tmp_path):
    filename = tmp_path / "trajectory.jsonl"
    await save_trajectory(
        [{"from": "human", "value": "hello"}],
        "test-model",
        True,
        filename=str(filename),
    )

    assert '"model": "test-model"' in filename.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_async_session_db_writes_without_to_thread(tmp_path, monkeypatch):
    database = SessionDB(tmp_path / "state.db")
    session_db = AsyncSessionDB(database)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("core async persistence must not call asyncio.to_thread")

    monkeypatch.setattr(asyncio, "to_thread", fail_if_called)
    try:
        await session_db.create_session("async-session", "test", model="test-model")
        await session_db.append_message("async-session", "user", "hello")
        await session_db.end_session("async-session", "test_complete")

        stored = database.get_session("async-session")
        assert stored["model"] == "test-model"
        assert [message["content"] for message in database.get_messages("async-session")] == ["hello"]
        assert stored["end_reason"] == "test_complete"
    finally:
        await session_db.close()
        database.close()


@pytest.mark.asyncio
async def test_conversation_root_uses_async_session_db(tmp_path, monkeypatch):
    database = SessionDB(tmp_path / "state.db")
    database.create_session("root", "test")
    database.create_session("child", "test", parent_session_id="root")
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "child"
    agent._parent_session_id = None
    agent._session_db = database
    session_db = AsyncSessionDB(database)
    agent._async_session_db = session_db

    monkeypatch.setattr(
        database,
        "get_conversation_root",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("async turn must not read through SessionDB")
        ),
    )
    try:
        assert await agent._conversation_root_id() == "root"
    finally:
        await session_db.close()
        database.close()


@pytest.mark.asyncio
async def test_async_session_db_loads_compression_snapshot(tmp_path, monkeypatch):
    database = SessionDB(tmp_path / "state.db")
    session_db = AsyncSessionDB(database)
    monkeypatch.setattr(
        asyncio,
        "to_thread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("async compression reload must not call asyncio.to_thread")
        ),
    )
    try:
        await session_db.create_session("parent", "test")
        await session_db.create_session(
            "child", "test", parent_session_id="parent"
        )
        await session_db.append_message("parent", "user", "parent message")
        await session_db.append_message("child", "assistant", "child message")

        assert [message["content"] for message in await session_db.get_messages_as_conversation("child")] == [
            "child message"
        ]
        assert [
            message["content"]
            for message in await session_db.get_messages_as_conversation(
                "child", include_ancestors=True
            )
        ] == ["parent message", "child message"]
    finally:
        await session_db.close()
        database.close()


@pytest.mark.asyncio
async def test_async_session_db_persists_compression_guards(tmp_path, monkeypatch):
    database = SessionDB(tmp_path / "state.db")
    session_db = AsyncSessionDB(database)
    monkeypatch.setattr(
        asyncio,
        "to_thread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("compression guards must not call asyncio.to_thread")
        ),
    )
    try:
        await session_db.create_session("guarded", "test")
        await session_db.record_compression_failure_cooldown(
            "guarded", time.time() + 30, "temporary failure"
        )
        await session_db.set_compression_fallback_streak("guarded", 2)
        await session_db.set_compression_ineffective_count("guarded", 1)

        cooldown = await session_db.get_compression_failure_cooldown("guarded")
        assert cooldown is not None
        assert cooldown["error"] == "temporary failure"
        assert await session_db.get_compression_fallback_streak("guarded") == 2
        assert await session_db.get_compression_ineffective_count("guarded") == 1
    finally:
        await session_db.close()
        database.close()


@pytest.mark.asyncio
async def test_micro_compaction_persists_through_async_session_db(tmp_path, monkeypatch):
    """Micro-compaction must not fall back to SessionDB or a worker thread."""
    from agent.context_compressor import (
        COMPRESSED_SUMMARY_METADATA_KEY,
        ContextCompressor,
    )

    database = SessionDB(tmp_path / "state.db")
    session_db = AsyncSessionDB(database)
    messages = [{"role": "system", "content": "system"}]
    for index in range(6):
        messages.extend(
            [
                {"role": "user", "content": f"question {index}"},
                {
                    "role": "assistant",
                    "content": f"answer {index} " + "x" * 400,
                },
            ]
        )
    compressor = ContextCompressor(
        model="test-model",
        threshold_percent=0.75,
        protect_first_n=1,
        protect_last_n=2,
        quiet_mode=True,
        config_context_length=40960,
        provider="test",
    )
    compressor._micro_compact_enabled = True
    compressor._session_id = "micro-session"

    async def summarize(_text):
        return "native async micro summary"

    compressor._micro_summarize_one = summarize
    monkeypatch.setattr(
        asyncio,
        "to_thread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("micro-compaction must not call asyncio.to_thread")
        ),
    )
    monkeypatch.setattr(
        database,
        "archive_and_compact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("micro-compaction must not call SessionDB")
        ),
    )
    try:
        await session_db.create_session("micro-session", "test")
        for message in messages:
            await session_db.append_message(
                "micro-session", message["role"], message["content"]
            )

        compacted = await compressor._micro_compact(messages, session_db=session_db)
        assert any(
            message.get(COMPRESSED_SUMMARY_METADATA_KEY) for message in compacted
        )
        persisted = await session_db.get_messages_as_conversation("micro-session")
        assert any(
            "native async micro summary" in str(message.get("content"))
            for message in persisted
        )
    finally:
        await session_db.close()
        database.close()


@pytest.mark.asyncio
async def test_async_auxiliary_accounting_writes_without_to_thread(tmp_path, monkeypatch):
    """An auxiliary model response persists usage through AsyncSessionDB only."""
    from agent.aux_accounting import reset_accounting_context, set_accounting_context
    from agent.auxiliary_client import _validate_llm_response

    database = SessionDB(tmp_path / "state.db")
    session_db = AsyncSessionDB(database)
    monkeypatch.setattr(
        asyncio,
        "to_thread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("auxiliary accounting must not call asyncio.to_thread")
        ),
    )
    token = set_accounting_context(session_db, "aux-session")
    response = SimpleNamespace(
        model="aux-model",
        usage=SimpleNamespace(prompt_tokens=21, completion_tokens=8, total_tokens=29),
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
    )
    try:
        assert await _validate_llm_response(response, "compression") is response
        async with session_db._connection.execute(
            "SELECT task, model, input_tokens, output_tokens "
            "FROM session_model_usage WHERE session_id = ?",
            ("aux-session",),
        ) as cursor:
            row = await cursor.fetchone()
        assert tuple(row) == ("compression", "aux-model", 21, 8)
    finally:
        reset_accounting_context(token)
        await session_db.close()
        database.close()


@pytest.mark.asyncio
async def test_async_session_db_backfills_api_sidecar_without_to_thread(
    tmp_path, monkeypatch
):
    database = SessionDB(tmp_path / "state.db")
    session_db = AsyncSessionDB(database)

    monkeypatch.setattr(
        asyncio,
        "to_thread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("api sidecar persistence must not call asyncio.to_thread")
        ),
    )
    try:
        await session_db.create_session("sidecar-session", "test")
        await session_db.append_message("sidecar-session", "user", "clean prompt")

        assert await session_db.set_latest_user_api_content(
            "sidecar-session", "clean prompt", "clean prompt\n\n<context/>"
        ) == 1
        messages = database.get_messages_as_conversation("sidecar-session")
        assert messages[0]["api_content"] == "clean prompt\n\n<context/>"
    finally:
        await session_db.close()
        database.close()


@pytest.mark.asyncio
async def test_async_session_db_compacts_and_releases_lease_without_to_thread(
    tmp_path, monkeypatch
):
    """The async compaction primitives preserve active/archived transcript rows."""
    database = SessionDB(tmp_path / "state.db")
    session_db = AsyncSessionDB(database)

    monkeypatch.setattr(
        asyncio,
        "to_thread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("compaction persistence must not call asyncio.to_thread")
        ),
    )
    try:
        await session_db.create_session("compact-session", "test", model="test-model")
        await session_db.append_message("compact-session", "user", "original question")
        assert await session_db.try_acquire_compression_lock(
            "compact-session", "test-holder"
        )

        await session_db.archive_and_compact(
            "compact-session",
            [
                {"role": "user", "content": "original question"},
                {"role": "assistant", "content": "compressed answer"},
            ],
        )
        await session_db.update_system_prompt("compact-session", "stable system prompt")
        await session_db.release_compression_lock("compact-session", "test-holder")

        active = database.get_messages("compact-session")
        archived = database.get_messages("compact-session", include_inactive=True)
        assert [message["content"] for message in active] == [
            "original question",
            "compressed answer",
        ]
        assert len(archived) == 3
        assert database.get_session("compact-session")["system_prompt"] == "stable system prompt"
        assert await session_db.get_compression_lock_holder("compact-session") is None
    finally:
        await session_db.close()
        database.close()


@pytest.mark.asyncio
async def test_in_place_compression_uses_native_async_sqlite_path(tmp_path, monkeypatch):
    """A complete in-place compaction never re-enters the sync SessionDB API."""
    database = SessionDB(tmp_path / "state.db")
    session_db = AsyncSessionDB(database)

    class Compressor:
        compression_count = 1

        async def compress(self, messages, **_kwargs):
            self._last_compression_made_progress = True
            self._last_summary_fallback_used = False
            self._last_summary_error = None
            self._last_aux_model_failure_model = None
            self._last_aux_model_failure_error = None
            return [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "compressed answer"},
            ]

    compressor = Compressor()
    await session_db.create_session("compression-session", "test", model="test-model")
    await session_db.append_message("compression-session", "user", "question")
    await session_db.append_message("compression-session", "assistant", "long answer")

    async def commit_memory_session(_messages):
        return None

    agent = SimpleNamespace(
        _session_db=database,
        _get_async_session_db=lambda: session_db,
        session_id="compression-session",
        context_compressor=compressor,
        api_mode=None,
        compression_in_place=True,
        _cached_system_prompt="stable system prompt",
        _memory_manager=None,
        _todo_store=SimpleNamespace(format_for_injection=lambda: ""),
        _emit_status=lambda *_args: None,
        _emit_warning=lambda *_args: None,
        _touch_activity=lambda *_args: None,
        _invalidate_system_prompt=AsyncMock(),
        _build_system_prompt=lambda _message: "stable system prompt",
        commit_memory_session=commit_memory_session,
        platform="test",
        model="test-model",
        _session_init_model_config={},
        _flushed_db_message_ids=set(),
        _last_flushed_db_idx=0,
        _last_compaction_in_place=False,
        _last_compression_attempt_in_place=None,
        _compression_feasibility_checked=True,
        _compression_activity_heartbeat_interval=0.1,
        log_prefix="",
        tools=[],
        event_callback=None,
    )
    monkeypatch.setattr(
        asyncio,
        "to_thread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("compression must not call asyncio.to_thread")
        ),
    )
    try:
        compressed, prompt = await compress_context(
            agent,
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "long answer"},
            ],
            "system",
            force=True,
        )
        assert [message["content"] for message in compressed] == [
            "question",
            "compressed answer",
        ]
        assert prompt == "stable system prompt"
        assert agent._last_compaction_in_place is True
        assert [message["content"] for message in database.get_messages(agent.session_id)] == [
            "question",
            "compressed answer",
        ]
    finally:
        await session_db.close()
        database.close()


@pytest.mark.asyncio
async def test_rotating_compression_publishes_child_with_native_async_sqlite(tmp_path):
    """The optional rotation mode retains its atomic parent/child handoff."""
    database = SessionDB(tmp_path / "state.db")
    session_db = AsyncSessionDB(database)

    class Compressor:
        compression_count = 1

        async def compress(self, _messages, **_kwargs):
            self._last_compression_made_progress = True
            self._last_summary_fallback_used = False
            self._last_summary_error = None
            self._last_aux_model_failure_model = None
            self._last_aux_model_failure_error = None
            return [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "compressed answer"},
            ]

    await session_db.create_session("rotation-parent", "test", model="test-model")
    await session_db.append_message("rotation-parent", "user", "question")
    await session_db.append_message("rotation-parent", "assistant", "long answer")

    async def flush_messages(*_args, **_kwargs):
        return True

    async def commit_memory_session(_messages):
        return None

    agent = SimpleNamespace(
        _session_db=database,
        _get_async_session_db=lambda: session_db,
        session_id="rotation-parent",
        context_compressor=Compressor(),
        api_mode=None,
        compression_in_place=False,
        _cached_system_prompt="stable system prompt",
        _memory_manager=None,
        _todo_store=SimpleNamespace(format_for_injection=lambda: ""),
        _emit_status=lambda *_args: None,
        _emit_warning=lambda *_args: None,
        _touch_activity=lambda *_args: None,
        _invalidate_system_prompt=AsyncMock(),
        _build_system_prompt=lambda _message: "stable system prompt",
        commit_memory_session=commit_memory_session,
        _flush_messages_to_session_db=flush_messages,
        platform="test",
        model="test-model",
        _session_init_model_config={},
        _flushed_db_message_ids=set(),
        _last_flushed_db_idx=0,
        _last_compaction_in_place=False,
        _last_compression_attempt_in_place=None,
        _compression_feasibility_checked=True,
        _compression_activity_heartbeat_interval=0.1,
        log_prefix="",
        tools=[],
        event_callback=None,
    )
    try:
        compressed, _ = await compress_context(
            agent,
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "long answer"},
            ],
            "system",
            force=True,
        )
        assert agent.session_id != "rotation-parent"
        assert database.get_session("rotation-parent")["end_reason"] == "compression"
        assert [message["content"] for message in database.get_messages(agent.session_id)] == [
            message["content"] for message in compressed
        ]
    finally:
        await session_db.close()
        database.close()


@pytest.mark.asyncio
async def test_agent_session_lifecycle_uses_native_async_store(tmp_path, monkeypatch):
    database = SessionDB(tmp_path / "state.db")
    agent = AIAgent.__new__(AIAgent)
    agent._persist_disabled = False
    agent._session_db = database
    agent._async_session_db = AsyncSessionDB(database)
    agent._async_session_persist_lock = None
    agent._session_db_created = False
    agent.session_id = "agent-async-session"
    agent.platform = "cli"
    agent.model = "test-model"
    agent._session_init_model_config = {}
    agent._cached_system_prompt = "system"
    agent._parent_session_id = None
    agent._last_flushed_db_idx = 0
    agent._flushed_db_message_session_id = None
    agent._flushed_db_message_ids = set()
    agent._db_flush_scan_prefix = None
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = None
    agent._persist_user_message_timestamp = None
    agent._pending_cli_user_message = None
    agent._active_compression_lock_holder = None

    def fail_if_called(*args, **kwargs):
        raise AssertionError("agent session lifecycle must not call asyncio.to_thread")

    monkeypatch.setattr(asyncio, "to_thread", fail_if_called)
    try:
        await agent._ensure_db_session()
        assert await agent._flush_messages_to_session_db([
            {"role": "user", "content": "hello"}
        ]) is True
        assert [message["content"] for message in database.get_messages(agent.session_id)] == ["hello"]
    finally:
        await agent._async_session_db.close()
        database.close()


@pytest.mark.asyncio
async def test_persist_session_does_not_reenter_its_async_lock(tmp_path, monkeypatch):
    """The persist funnel owns one lock and calls its unlocked DB writer."""
    database = SessionDB(tmp_path / "state.db")
    agent = AIAgent.__new__(AIAgent)
    agent._persist_disabled = False
    agent._session_db = database
    agent._async_session_db = AsyncSessionDB(database)
    agent._async_session_persist_lock = None
    agent._session_db_created = False
    agent.session_id = "persist-session"
    agent.platform = "cli"
    agent.model = "test-model"
    agent._session_init_model_config = {}
    agent._cached_system_prompt = "system"
    agent._parent_session_id = None
    agent._last_flushed_db_idx = 0
    agent._flushed_db_message_session_id = None
    agent._flushed_db_message_ids = set()
    agent._db_flush_scan_prefix = None
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = None
    agent._persist_user_message_timestamp = None
    agent._pending_cli_user_message = None
    agent._active_compression_lock_holder = None
    agent._inflight_turn_id = None
    agent._inflight_turn_session_id = None

    async def save_log(_messages):
        return None

    agent._save_session_log = save_log
    monkeypatch.setattr(
        asyncio,
        "to_thread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("persistence must not call asyncio.to_thread")
        ),
    )
    try:
        await asyncio.wait_for(
            agent._persist_session([{"role": "user", "content": "hello"}]),
            timeout=0.5,
        )
        assert [message["content"] for message in database.get_messages(agent.session_id)] == [
            "hello"
        ]
    finally:
        await agent._async_session_db.close()
        database.close()


@pytest.mark.asyncio
async def test_native_async_terminal_does_not_use_to_thread(monkeypatch, tmp_path):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("terminal must use asyncio subprocesses directly")

    monkeypatch.setattr(asyncio, "to_thread", fail_if_called)
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))

    result = json.loads(
        await terminal_tool("printf async-terminal", task_id="async-test")
    )

    assert result == {
        "output": "async-terminal",
        "exit_code": 0,
        "error": None,
    }


@pytest.mark.asyncio
async def test_synthetic_turn_records_trajectory_without_to_thread(monkeypatch, tmp_path):
    """Exercise the public turn path with an async model and real session DB."""
    from run_agent import AIAgent

    database = SessionDB(tmp_path / "state.db")
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            model="test-model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            save_trajectories=False,
        )
    agent._session_db = database
    agent._async_session_db = AsyncSessionDB(database)
    agent._session_db_created = False
    agent.compression_enabled = False

    async def model_response(*_args, **_kwargs):
        return SimpleNamespace(
            id="synthetic-response",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        role="assistant",
                        content="async answer",
                        tool_calls=None,
                        reasoning=None,
                        reasoning_content=None,
                    ),
                )
            ],
            usage=None,
        )

    agent._execute_model_request = model_response

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("the public async turn must not call asyncio.to_thread")

    monkeypatch.setattr(asyncio, "to_thread", fail_if_called)
    try:
        result = await agent.run_conversation("hello async")
        assert result["completed"] is True
        assert result["final_response"] == "async answer"
        assert [message["role"] for message in result["messages"]] == [
            "user",
            "assistant",
        ]
        assert [message["content"] for message in database.get_messages(agent.session_id)] == [
            "hello async",
            "async answer",
        ]
    finally:
        await agent.close()
        database.close()


@pytest.mark.asyncio
async def test_oauth_renewal_fails_fast_without_sync_refresh(monkeypatch, tmp_path):
    """An async turn must not invoke the legacy OAuth-refresh helpers."""
    from agent.agent_runtime_helpers import AsyncCapabilityError

    class UnauthorizedError(Exception):
        status_code = 401

    database = SessionDB(tmp_path / "state.db")
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            provider="openai-codex",
            api_mode="codex_responses",
            model="test-model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            save_trajectories=False,
        )
    agent._session_db = database
    agent._async_session_db = AsyncSessionDB(database)
    agent._session_db_created = False
    agent.compression_enabled = False

    async def unauthorized_model(*_args, **_kwargs):
        raise UnauthorizedError("expired OAuth token")

    agent._execute_model_request = unauthorized_model
    agent._try_refresh_codex_client_credentials = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("async turn must not call the sync credential refresher")
    )
    try:
        with pytest.raises(AsyncCapabilityError, match="OAuth renewal"):
            await agent.run_conversation("hello async")
    finally:
        await agent.close()
        database.close()


@pytest.mark.asyncio
async def test_codex_responses_main_path_uses_native_async_client(monkeypatch):
    """Codex main turns use AsyncOpenAI Responses, never a chat shim thread."""
    agent = AIAgent.__new__(AIAgent)
    agent.api_mode = "codex_responses"
    agent.model = "gpt-test"

    created_requests = []

    class Responses:
        async def create(self, **kwargs):
            created_requests.append(kwargs)
            return "native-response"

    agent.client = native_client = SimpleNamespace(responses=Responses())

    result = await agent._execute_model_request(
        {"model": "gpt-test", "stream": False},
    )

    assert result == "native-response"
    assert agent._async_codex_client is native_client
    assert created_requests == [{"model": "gpt-test"}]


@pytest.mark.asyncio
async def test_synthetic_model_tool_observation_turn_preserves_order(monkeypatch, tmp_path):
    """The model → tool → observation → model training shape stays ordered."""
    database = SessionDB(tmp_path / "state.db")
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            model="test-model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            save_trajectories=False,
        )
    agent._session_db = database
    agent._async_session_db = AsyncSessionDB(database)
    agent._session_db_created = False
    agent.compression_enabled = False
    agent.valid_tool_names = {"terminal"}

    responses = iter(
        [
            SimpleNamespace(
                id="tool-request",
                choices=[
                    SimpleNamespace(
                        finish_reason="tool_calls",
                        message=SimpleNamespace(
                            role="assistant",
                            content="",
                            tool_calls=[
                                SimpleNamespace(
                                    id="call-1",
                                    type="function",
                                    function=SimpleNamespace(
                                        name="terminal",
                                        arguments='{"command": "printf tool-observation"}',
                                    ),
                                )
                            ],
                            reasoning="inspect the tool result",
                            reasoning_content=None,
                        ),
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                id="final-answer",
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(
                            role="assistant",
                            content="tool observation incorporated",
                            tool_calls=None,
                            reasoning="answer after observation",
                            reasoning_content=None,
                        ),
                    )
                ],
                usage=None,
            ),
        ]
    )

    async def model_response(*_args, **_kwargs):
        return next(responses)

    agent._execute_model_request = model_response
    monkeypatch.setattr(
        asyncio,
        "to_thread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("the synthetic async turn must not call asyncio.to_thread")
        ),
    )
    try:
        result = await agent.run_conversation("run a command")
        messages = result["messages"]
        assert [message["role"] for message in messages] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        assert messages[1]["tool_calls"][0]["id"] == "call-1"
        assert messages[2]["tool_call_id"] == "call-1"
        assert json.loads(messages[2]["content"]) == {
            "output": "tool-observation",
            "exit_code": 0,
            "error": None,
        }
        assert result["final_response"] == "tool observation incorporated"
        assert [message["role"] for message in database.get_messages(agent.session_id)] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
    finally:
        await agent.close()
        database.close()


@pytest.mark.asyncio
async def test_cancelled_turn_persists_partial_session_and_reraises(monkeypatch, tmp_path):
    """Cancellation keeps the crash-safe user row, then propagates cancellation."""
    database = SessionDB(tmp_path / "state.db")
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            model="test-model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            save_trajectories=False,
        )
    agent._session_db = database
    agent._async_session_db = AsyncSessionDB(database)
    agent._session_db_created = False
    agent.compression_enabled = False
    model_started = asyncio.Event()

    async def slow_model(*_args, **_kwargs):
        model_started.set()
        await asyncio.sleep(60)

    agent._execute_model_request = slow_model

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("cancelled async turn must not call asyncio.to_thread")

    monkeypatch.setattr(asyncio, "to_thread", fail_if_called)
    task = asyncio.create_task(agent.run_conversation("persist this before cancel"))
    try:
        await asyncio.wait_for(model_started.wait(), timeout=0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert [message["content"] for message in database.get_messages(agent.session_id)] == [
            "persist this before cancel"
        ]
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await agent.close()
        database.close()


@pytest.mark.asyncio
async def test_cancelled_tool_batch_persists_ordered_cancelled_observation(
    monkeypatch, tmp_path
):
    """A cancelled native tool task leaves a closed, ordered transcript."""
    import importlib

    active_registry = importlib.import_module("tools.registry").registry
    active_model_tools = importlib.import_module("model_tools")
    database = SessionDB(tmp_path / "state.db")
    tool_name = "__async_core_slow_tool__"
    tool_started = asyncio.Event()

    async def slow_tool(_args, **_kwargs):
        tool_started.set()
        await asyncio.sleep(60)
        return "unreachable"

    original_get_entry = active_registry.get_entry

    def get_entry(name):
        if name == tool_name:
            return SimpleNamespace(is_async=True)
        return original_get_entry(name)

    async def dispatch_native_handler(name, args, *_args, **_kwargs):
        assert name == tool_name
        return await slow_tool(args)

    monkeypatch.setattr(active_registry, "get_entry", get_entry)
    monkeypatch.setattr(active_model_tools, "handle_function_call", dispatch_native_handler)

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            model="test-model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            save_trajectories=False,
        )
    agent._session_db = database
    agent._async_session_db = AsyncSessionDB(database)
    agent._session_db_created = False
    agent.compression_enabled = False
    agent.valid_tool_names = {tool_name}

    async def model_response(*_args, **_kwargs):
        return SimpleNamespace(
            id="slow-tool-request",
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    message=SimpleNamespace(
                        role="assistant",
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id="slow-call",
                                type="function",
                                function=SimpleNamespace(
                                    name=tool_name,
                                    arguments="{}",
                                ),
                            )
                        ],
                        reasoning=None,
                        reasoning_content=None,
                    ),
                )
            ],
            usage=None,
        )

    agent._execute_model_request = model_response

    task = asyncio.create_task(agent.run_conversation("cancel this tool"))
    try:
        await asyncio.wait_for(tool_started.wait(), timeout=0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        stored = database.get_messages(agent.session_id)
        assert [message["role"] for message in stored] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        assert "cancelled" in stored[-2]["content"].lower()
        assert stored[-1]["content"] == "Operation interrupted."
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await agent.close()
        database.close()


@pytest.mark.asyncio
async def test_async_registry_handler_is_awaited(monkeypatch):
    name = "__async_core_test_tool__"

    async def handler(args, **kwargs):
        return '{"value": %d}' % (args["value"] + 1)

    registry.register(
        name=name,
        toolset="async-core-test",
        schema={
            "name": name,
            "description": "test-only async handler",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
        },
        handler=handler,
    )
    try:
        result = await handle_function_call(name, {"value": 4})
        assert result == '{"value": 5}'
    finally:
        monkeypatch.setattr(registry, "_tools", {
            key: value for key, value in registry._tools.items() if key != name
        })


@pytest.mark.asyncio
async def test_memory_tool_uses_native_async_file_path(monkeypatch, tmp_path):
    """Memory remains durable without exposing a blocking handler to the loop."""
    import importlib

    memory_tool_module = importlib.import_module("tools.memory_tool")
    active_registry = importlib.import_module("tools.registry").registry

    monkeypatch.setattr(memory_tool_module, "get_memory_dir", lambda: tmp_path / "memories")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("memory tool must not call asyncio.to_thread")

    monkeypatch.setattr(asyncio, "to_thread", fail_if_called)
    store = memory_tool_module.MemoryStore(memory_char_limit=200, user_char_limit=200)
    await store.load_from_disk()
    initial_snapshot = store.format_for_system_prompt("memory")
    response = await memory_tool_module.memory_tool(
        action="add",
        target="memory",
        content="Prefer native async file I/O.",
        store=store,
    )

    assert '"success": true' in response
    assert "Prefer native async file I/O." in (tmp_path / "memories" / "MEMORY.md").read_text()
    assert store.format_for_system_prompt("memory") == initial_snapshot
    assert await active_registry.dispatch(
        "memory",
        {"action": "add", "target": "memory", "content": "Registry awaits handlers."},
        store=store,
    )


@pytest.mark.asyncio
async def test_native_background_terminal_process_is_reaped_at_cleanup():
    """Async terminal children never outlive their owning task/session."""
    import importlib

    terminal_module = importlib.import_module("tools.terminal_tool")

    task_id = "async-background-reap"
    command = f"{shlex.quote(sys.executable)} -c 'import time; time.sleep(30)'"
    started = await terminal_module.terminal_tool(
        command, background=True, task_id=task_id
    )
    assert "Background process started" in started
    processes = list(terminal_module._background_processes[task_id])

    await terminal_module.cleanup_vm(task_id)

    assert task_id not in terminal_module._background_processes
    assert all(process.returncode is not None for process in processes)


@pytest.mark.asyncio
async def test_skills_tools_keep_the_public_name_and_use_async_file_reads(
    monkeypatch,
    tmp_path,
):
    """The registry exposes skills_list/skill_view, not async-suffixed tools."""
    import importlib

    from agent import skill_utils

    skills_tool = importlib.import_module("tools.skills_tool")
    active_registry = importlib.import_module("tools.registry").registry

    root = tmp_path / "skills"
    skill_file = root / "training" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        "---\nname: training\ndescription: Generate trajectories.\n---\n"
        "# Training\n\nUse tool observations in the trace.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(skills_tool, "_skills_dir", lambda: root)
    monkeypatch.setattr(skill_utils, "get_external_skills_dirs", lambda: [])

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("skills tools must not call asyncio.to_thread")

    monkeypatch.setattr(asyncio, "to_thread", fail_if_called)
    listed = await skills_tool.skills_list()
    viewed = await skills_tool.skill_view("training")

    assert '"name": "training"' in listed
    assert "Use tool observations in the trace." in viewed
    assert "training" in await active_registry.dispatch("skills_list", {})
    assert "Training" in await active_registry.dispatch(
        "skill_view", {"name": "training"}
    )


@pytest.mark.asyncio
async def test_async_tool_scheduler_preserves_barriers_and_result_order(monkeypatch):
    import importlib

    active_executor = importlib.import_module("agent.tool_executor")
    active_model_tools = importlib.import_module("model_tools")
    active_registry = importlib.import_module("tools.registry").registry
    monkeypatch.setattr(active_model_tools, "registry", active_registry)
    events = []
    originals = dict(active_registry._tools)
    monkeypatch.setattr(active_registry, "_tools", dict(originals))
    monkeypatch.setattr(
        active_executor, "_begin_tool_execution", lambda *_args, **_kwargs: None
    )

    async def parallel_handler(args, **kwargs):
        events.append(f"start:{args['id']}")
        await asyncio.sleep(0)
        events.append(f"end:{args['id']}")
        return args["id"]

    async def barrier_handler(args, **kwargs):
        events.append("barrier")
        return "barrier"

    for name, handler in (("web_search", parallel_handler), ("web_extract", parallel_handler), ("clarify", barrier_handler)):
        active_registry.register(
            name=name,
            toolset="test",
            schema={"name": name, "parameters": {"type": "object"}},
            handler=handler,
            override=True,
        )

    calls = [
        SimpleNamespace(id="one", function=SimpleNamespace(name="web_search", arguments='{"id": "one"}')),
        SimpleNamespace(id="two", function=SimpleNamespace(name="web_extract", arguments='{"id": "two"}')),
        SimpleNamespace(id="three", function=SimpleNamespace(name="clarify", arguments="{}")),
        SimpleNamespace(id="four", function=SimpleNamespace(name="web_search", arguments='{"id": "four"}')),
    ]
    agent = SimpleNamespace(
        session_id="",
        valid_tool_names=[],
        _current_user_task=None,
        _tool_guardrails=SimpleNamespace(
            before_call=lambda *_args: SimpleNamespace(allows_execution=True)
        ),
        _append_guardrail_observation=lambda _name, _args, result, **_kwargs: result,
        _apply_pending_steer_to_tool_results=lambda *_args: None,
        tool_progress_callback=None,
        tool_complete_callback=None,
    )
    messages = []

    await active_executor.execute_tool_calls_segmented(
        agent,
        SimpleNamespace(tool_calls=calls),
        messages,
        "test-task",
    )

    assert events.index("barrier") > events.index("end:one")
    assert events.index("barrier") > events.index("end:two")
    assert events.index("start:four") > events.index("barrier")
    assert [message["tool_call_id"] for message in messages] == ["one", "two", "three", "four"]
