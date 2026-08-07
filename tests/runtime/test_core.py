"""Regression coverage for the public runtime core."""

import inspect
import asyncio
import json
import shlex
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from blockbuster import BlockBuster
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

from agent.conversation_loop import run_conversation
from agent.conversation_compression import compress_context
from agent.chat_completion_helpers import handle_max_iterations
from agent.trajectory import save_trajectory
from agent.turn_context import build_turn_context
from agent.turn_finalizer import finalize_turn
from agent import tool_executor
from agent.tool_executor import execute_tool_calls_segmented
from hermes_state import SessionDB
from run_agent import AIAgent
from model_tools import get_tool_definitions, handle_function_call
from tools.registry import ToolRegistry, check_fn_cache_scope, registry
from tools.clarify_tool import clarify_tool
from tools.memory_tool import MemoryStore, memory_tool
from tools.skills_tool import skill_view, skills_list
from tools.file_tools import read_file_tool, write_file_tool
from tools.terminal_tool import terminal_tool


def test_conversation_and_chat_are_coroutines():
    from agent.context_engine import ContextEngine
    from agent.conversation_loop import (
        _apply_context_engine_selection,
        _notify_context_engine_turn_complete,
    )
    from run_agent import main as agent_main
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
    from tools.file_tools import (
        _resolve_path_for_task,
        patch_tool,
        read_file_tool,
        search_tool,
        write_file_tool,
    )
    from tools.terminal_tool import (
        get_session_cwd,
        record_session_cwd,
        register_task_env_overrides,
    )
    from tools.session_search_tool import check_session_search_requirements
    from batch_runner import BatchRunner, main as batch_main
    from trajectory_compressor import (
        CompressionConfig,
        TrajectoryCompressor,
        main as trajectory_main,
    )
    from hermes_cli.middleware import run_llm_execution_middleware
    from hermes_constants import get_hermes_dir
    from gateway.status import _pid_exists
    from agent.skill_commands import (
        build_skill_invocation_message,
        get_skill_commands,
        reload_skills,
        scan_skill_commands,
    )
    from agent.skill_utils import normalize_skill_lookup_name

    assert inspect.iscoroutinefunction(AIAgent.run_conversation)
    assert inspect.iscoroutinefunction(AIAgent.chat)
    assert inspect.iscoroutinefunction(AIAgent.close)
    assert inspect.iscoroutinefunction(agent_main)
    assert inspect.iscoroutinefunction(AIAgent.switch_model)
    assert inspect.iscoroutinefunction(AIAgent._ensure_db_session)
    assert inspect.iscoroutinefunction(check_session_search_requirements)
    assert inspect.iscoroutinefunction(AIAgent._persist_session)
    assert inspect.iscoroutinefunction(AIAgent._flush_messages_to_session_db)
    assert inspect.iscoroutinefunction(AIAgent._save_session_log)
    assert inspect.iscoroutinefunction(AIAgent._dump_api_request_debug)
    assert inspect.iscoroutinefunction(AIAgent.shutdown_memory_provider)
    assert inspect.iscoroutinefunction(AIAgent.commit_memory_session)
    assert inspect.iscoroutinefunction(AIAgent.reset_session_state)
    assert inspect.iscoroutinefunction(AIAgent._transition_context_engine_session)
    assert inspect.iscoroutinefunction(AIAgent._handle_max_iterations)
    assert inspect.iscoroutinefunction(AIAgent._execute_tool_calls)
    assert inspect.iscoroutinefunction(AIAgent._conversation_root_id)
    assert inspect.iscoroutinefunction(handle_max_iterations)
    assert inspect.iscoroutinefunction(build_turn_context)
    assert inspect.iscoroutinefunction(finalize_turn)
    assert inspect.iscoroutinefunction(run_conversation)
    assert inspect.iscoroutinefunction(compress_context)
    assert inspect.iscoroutinefunction(ContextEngine.compress)
    assert inspect.iscoroutinefunction(ContextEngine.select_context)
    assert inspect.iscoroutinefunction(ContextEngine.on_turn_complete)
    assert inspect.iscoroutinefunction(ContextEngine.on_session_start)
    assert inspect.iscoroutinefunction(ContextEngine.on_session_end)
    assert inspect.iscoroutinefunction(ContextEngine.on_session_reset)
    assert inspect.iscoroutinefunction(ContextEngine.handle_tool_call)
    assert inspect.iscoroutinefunction(_apply_context_engine_selection)
    assert inspect.iscoroutinefunction(_notify_context_engine_turn_complete)
    assert inspect.iscoroutinefunction(execute_tool_calls_segmented)
    assert inspect.iscoroutinefunction(get_tool_definitions)
    assert inspect.iscoroutinefunction(ToolRegistry.get_definitions)
    assert inspect.iscoroutinefunction(ToolRegistry.check_tool_availability)
    assert inspect.iscoroutinefunction(check_fn_cache_scope)
    assert inspect.iscoroutinefunction(BatchRunner.run)
    assert inspect.iscoroutinefunction(batch_main)
    assert inspect.iscoroutinefunction(CompressionConfig.from_yaml)
    assert inspect.iscoroutinefunction(TrajectoryCompressor.close)
    assert inspect.iscoroutinefunction(trajectory_main)
    assert inspect.iscoroutinefunction(SessionDB._parse_schema_columns)
    assert inspect.iscoroutinefunction(get_hermes_dir)
    assert inspect.iscoroutinefunction(_pid_exists)
    assert inspect.iscoroutinefunction(terminal_tool)
    assert inspect.iscoroutinefunction(get_session_cwd)
    assert inspect.iscoroutinefunction(record_session_cwd)
    assert inspect.iscoroutinefunction(register_task_env_overrides)
    assert inspect.iscoroutinefunction(_resolve_path_for_task)
    assert inspect.iscoroutinefunction(read_file_tool)
    assert inspect.iscoroutinefunction(write_file_tool)
    assert inspect.iscoroutinefunction(patch_tool)
    assert inspect.iscoroutinefunction(search_tool)
    assert inspect.iscoroutinefunction(clarify_tool)
    assert inspect.iscoroutinefunction(memory_tool)
    assert inspect.iscoroutinefunction(skills_list)
    assert inspect.iscoroutinefunction(skill_view)
    assert inspect.iscoroutinefunction(scan_skill_commands)
    assert inspect.iscoroutinefunction(get_skill_commands)
    assert inspect.iscoroutinefunction(reload_skills)
    assert inspect.iscoroutinefunction(build_skill_invocation_message)
    assert inspect.iscoroutinefunction(normalize_skill_lookup_name)
    assert inspect.iscoroutinefunction(invoke_tool)
    assert inspect.iscoroutinefunction(_validate_llm_response)
    assert inspect.iscoroutinefunction(_aggregate_chat_stream)
    assert inspect.iscoroutinefunction(_create_with_progress)
    assert inspect.iscoroutinefunction(_create_with_stream)
    assert inspect.iscoroutinefunction(run_llm_execution_middleware)
    assert inspect.iscoroutinefunction(AIAgent._try_activate_fallback)
    assert inspect.iscoroutinefunction(AIAgent._try_recover_primary_transport)
    assert inspect.iscoroutinefunction(AIAgent._recover_with_credential_pool)
    assert inspect.iscoroutinefunction(AIAgent._replace_primary_openai_client)
    assert inspect.iscoroutinefunction(AIAgent._swap_credential)
    assert inspect.iscoroutinefunction(
        AIAgent._describe_image_for_anthropic_fallback
    )
    assert inspect.iscoroutinefunction(AIAgent._materialize_data_url_for_vision)
    assert inspect.iscoroutinefunction(AIAgent._preprocess_anthropic_content)
    assert inspect.iscoroutinefunction(recover_with_credential_pool)
    assert inspect.iscoroutinefunction(try_recover_primary_transport)
    assert inspect.iscoroutinefunction(load_pool)
    active_entries = list(registry._tools.values())
    assert active_entries
    assert all(inspect.iscoroutinefunction(entry.handler) for entry in active_entries)


@pytest.mark.asyncio
async def test_invoke_tool_preserves_upstream_arguments(monkeypatch):
    from agent.agent_runtime_helpers import invoke_tool
    import model_tools

    dispatch = AsyncMock(return_value="ok")
    monkeypatch.setattr(model_tools, "handle_function_call", dispatch)
    monkeypatch.setattr(
        registry,
        "get_entry",
        lambda _name: SimpleNamespace(toolset="core"),
    )
    agent = SimpleNamespace(
        session_id="session",
        _current_turn_id="turn",
        _current_api_request_id="request",
        _current_user_task="question",
        valid_tool_names={"example"},
        enabled_toolsets=["core"],
        disabled_toolsets=["browser"],
    )

    result = await invoke_tool(
        agent,
        "example",
        {"value": 1},
        "task",
        "call",
        [],
        True,
        True,
        [{"middleware": "seen"}],
        True,
    )

    assert result == "ok"
    dispatch.assert_awaited_once_with(
        "example",
        {"value": 1},
        "task",
        tool_call_id="call",
        session_id="session",
        turn_id="turn",
        api_request_id="request",
        user_task="question",
        enabled_tools=["example"],
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
        tool_request_middleware_trace=[{"middleware": "seen"}],
        enabled_toolsets=["core"],
        disabled_toolsets=["browser"],
    )


@pytest.mark.asyncio
async def test_session_search_availability_does_not_block_event_loop(
    monkeypatch, tmp_path
):
    import hermes_state
    from tools.session_search_tool import check_session_search_requirements

    monkeypatch.setattr(
        hermes_state, "_default_db_path", lambda: tmp_path / "state.db"
    )

    blockbuster = BlockBuster()
    blockbuster.activate()
    try:
        assert await check_session_search_requirements() is True
    finally:
        blockbuster.deactivate()


@pytest.mark.asyncio
async def test_session_lifecycle_does_not_block_or_leak(tmp_path):
    """Exercise real SQLite and transcript cleanup under all runtime audits."""
    from hermes_state import SessionDB

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / "session.jsonl").write_text("", encoding="utf-8")
    database = SessionDB(tmp_path / "state.db")

    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        blockbuster = BlockBuster()
        blockbuster.activate()
        try:
            await database.ensure_session(
                "session", source="library", cwd="/workspace/repo"
            )
            message_id = await database.append_message(
                "session",
                role="user",
                content="hello",
                platform_message_id="platform-message",
            )
            assert await database.message_count("session") == 1
            assert not await database.has_archived_messages("session")
            assert await database.has_platform_message_id(
                "session", "platform-message"
            )
            assert await database.session_count(source="library") == 1
            assert await database.session_count_by_source() == {"library": 1}
            await database.queue_token_counts(
                "session", input_tokens=3, api_call_count=1
            )
            assert await database.flush_token_counts() is True
            await database.backfill_repo_roots(
                {"/workspace/repo": "/workspace/repo"}
            )
            assert (await database.get_session("session"))["git_repo_root"] == (
                "/workspace/repo"
            )
            assert [
                row["id"]
                for row in await database.search_sessions(source="library")
            ] == ["session"]
            assert await database.latest_user_message_row_id("session") == (
                message_id
            )
            assert await database.get_message_role(
                "session", message_id
            ) == "user"
            assert await database.resolve_session_id("sess") == "session"
            assert [
                row["session_id"]
                for row in await database.search_messages("hello")
            ] == ["session"]
            await database.append_message(
                "session",
                role="assistant",
                content="错误日志：数据库连接超时",
            )
            cjk_hits = await database.search_messages("数据库连接")
            assert [row["session_id"] for row in cjk_hits] == ["session"]
            assert [item["content"] for item in cjk_hits[0]["context"]] == [
                "hello",
                "错误日志：数据库连接超时",
            ]
            assert [
                row["preview"]
                for row in await database.list_recent_user_messages("session")
            ] == ["hello"]
            assert [
                row["id"]
                for row in await database.search_sessions_by_id("sess")
            ] == ["session"]
            assert await database.set_session_title(
                "session", "  Native\n Async  "
            )
            assert not await database.set_auto_title_if_empty(
                "session", "Ignored"
            )
            assert await database.set_session_pinned("session", True)
            assert await database.set_session_archived("session", True)
            assert await database.set_session_archived("session", False)
            await database.end_session("session", "done")
            await database.reopen_session("session")
            assert (await database.get_session("session"))["ended_at"] is None
            await database.end_session("session", "done")
            assert [
                row["id"]
                for row in await database.list_prune_candidates(
                    older_than_days=None, title_like="native"
                )
            ] == ["session"]
            assert await database.archive_sessions(
                older_than_days=None, title_like="native"
            ) == 1
            assert await database.set_session_archived("session", False)
            assert await database.archive_stale_sessions(-1) == 0
            assert await database.maybe_auto_archive(
                idle_days=-1, min_interval_hours=0
            ) == {"skipped": False, "archived": 0}
            assert await database.maybe_auto_prune_and_vacuum(
                retention_days=10_000,
                min_interval_hours=0,
                vacuum=False,
            ) == {"skipped": False, "pruned": 0, "vacuumed": False}
            assert (await database.logical_size_bytes()) > 0
            assert await database.rebuild_fts() >= 1
            assert await database.optimize_fts() >= 0
            assert await database.vacuum() >= 0

            await database.create_session("compression-parent", source="runtime")
            await database.end_session("compression-parent", "compression")
            await database.create_session(
                "compression-orphan",
                source="runtime",
                parent_session_id="compression-parent",
            )
            await database.append_message(
                "compression-orphan", role="user", content="unfinished"
            )
            connection = await database._get_connection()
            await connection.execute(
                "UPDATE sessions SET started_at = ? WHERE id = ?",
                (time.time() - 800_000, "compression-orphan"),
            )
            await connection.commit()
            assert await database.finalize_orphaned_compression_sessions() == 1
            assert [
                row["id"]
                for row in await database.list_sessions_rich(
                    search_query="session",
                    order_by_last_active=True,
                    include_pinned=True,
                )
            ] == ["session"]
            assert (
                await database.get_session_rich_row(
                    "session", compact_rows=True
                )
            )["preview"] == "hello"
            assert await database.delete_session(
                "session", sessions_dir=sessions_dir
            )
        finally:
            await database.close()
            blockbuster.deactivate()

    assert not (sessions_dir / "session.jsonl").exists()


@pytest.mark.asyncio
async def test_session_resume_lineage_does_not_block_or_leak(tmp_path):
    """Audit the real resume, compression-lineage, and rewind I/O paths."""
    from hermes_state import SessionDB

    database = SessionDB(tmp_path / "state.db")

    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        blockbuster = BlockBuster()
        blockbuster.activate()
        try:
            await database.create_session("root", source="library")
            await database.append_message(
                "root", role="user", content="root question"
            )
            await database.append_message(
                "root", role="assistant", content="root answer"
            )
            await database.end_session("root", "compression")
            await database.create_session(
                "tip", source="library", parent_session_id="root"
            )
            target_id = await database.append_message(
                "tip", role="user", content="tip question"
            )
            await database.append_message(
                "tip", role="assistant", content="tip answer"
            )

            assert await database.resolve_resume_session_id("root") == "tip"
            model_history, display_history = (
                await database.get_resume_conversations("tip")
            )
            assert [message["content"] for message in model_history] == [
                "tip question",
                "tip answer",
            ]
            assert [message["content"] for message in display_history] == [
                "root question",
                "root answer",
                "tip question",
                "tip answer",
            ]
            assert [
                message["content"]
                for message in await database.get_ancestor_display_prefix(
                    "tip"
                )
            ] == ["root question", "root answer"]
            assert await database.get_compression_lineage("tip") == [
                "root",
                "tip",
            ]
            assert (
                await database.rewind_to_message("tip", target_id)
            )["rewound_count"] == 2
            assert await database.restore_rewound("tip", target_id) == 2
        finally:
            await database.close()
            blockbuster.deactivate()


@pytest.mark.asyncio
async def test_skill_invocation_does_not_block_or_leak(tmp_path):
    """Audit real skill discovery, preprocessing, and message assembly."""
    import agent.skill_commands as skill_commands
    from agent import skill_utils
    from agent.skill_preprocessing import run_inline_shell

    skill_dir = tmp_path / "native-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: native-skill\n"
        "description: Native async audit skill.\n"
        "---\n\n"
        "session=${HERMES_SESSION_ID}\n"
        "cwd=!`pwd`\n",
        encoding="utf-8",
    )
    skill_commands._skill_commands = {}
    skill_commands._skill_commands_platform = None

    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        blockbuster = BlockBuster()
        blockbuster.activate()
        try:
            skill_utils._ENV_DETECT_CACHE.clear()
            assert isinstance(
                await skill_utils.skill_matches_environment(
                    {"environments": ["docker"]}
                ),
                bool,
            )
            with (
                patch("tools.skills_tool.SKILLS_DIR", tmp_path),
                patch(
                    "tools.skills_tool._external_skills_dirs",
                    AsyncMock(return_value=[]),
                ),
                patch(
                    "tools.skills_tool._get_disabled_skill_names",
                    AsyncMock(return_value=set()),
                ),
                patch(
                    "agent.skill_preprocessing.load_skills_config",
                    AsyncMock(
                        return_value={
                            "template_vars": True,
                            "inline_shell": True,
                            "inline_shell_timeout": 5,
                        }
                    ),
                ),
            ):
                await skill_commands.scan_skill_commands()
                message = await skill_commands.build_skill_invocation_message(
                    "/native-skill", task_id="audit-session"
                )
            inline_shell = asyncio.create_task(
                run_inline_shell("sleep 30", skill_dir, timeout=60)
            )
            await asyncio.sleep(0.05)
            inline_shell.cancel()
            with pytest.raises(asyncio.CancelledError):
                await inline_shell
        finally:
            blockbuster.deactivate()
            skill_utils._ENV_DETECT_CACHE.clear()
            skill_commands._skill_commands = {}
            skill_commands._skill_commands_platform = None

    assert message is not None
    assert "session=audit-session" in message
    assert f"cwd={skill_dir}" in message


@pytest.mark.asyncio
async def test_web_tool_availability_does_not_block_event_loop(
    monkeypatch, tmp_path
):
    from tools import web_tools

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for key in (
        "BRAVE_SEARCH_API_KEY",
        "EXA_API_KEY",
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
        "PARALLEL_API_KEY",
        "SEARXNG_URL",
        "TAVILY_API_KEY",
        "XAI_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(web_tools, "_ddgs_package_importable", lambda: False)

    blockbuster = BlockBuster()
    blockbuster.activate()
    try:
        assert await web_tools.check_web_api_key() is False
    finally:
        blockbuster.deactivate()


@pytest.mark.asyncio
async def test_builtin_tool_availability_checks_do_not_block_event_loop(
    monkeypatch, tmp_path
):
    from tools.registry import discover_builtin_tools

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    discover_builtin_tools()
    checks = {
        entry.check_fn
        for entry in registry._tools.values()
        if entry.check_fn is not None
    }

    blockbuster = BlockBuster()
    blockbuster.activate()
    try:
        for check in checks:
            result = check()
            if inspect.isawaitable(result):
                result = await result
            assert isinstance(result, bool)
    finally:
        blockbuster.deactivate()


def test_constructor_does_not_read_runtime_config_or_create_logs():
    """AIAgent construction is state-only; I/O starts at the first await."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch(
            "hermes_cli.config.load_config",
            side_effect=AssertionError("constructor read config.yaml"),
        ),
        patch(
            "hermes_logging.setup_logging",
            side_effect=AssertionError("constructor initialized file logging"),
        ),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            model="test-model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert agent._deferred_provider_runtime is not None
    assert agent.tools == []
    assert agent.valid_tool_names == set()


@pytest.mark.asyncio
async def test_openai_client_rebuild_reuses_native_async_runtime_recipe():
    agent = AIAgent.__new__(AIAgent)
    agent.provider = "openrouter"
    agent.model = "test-model"
    agent.api_mode = "chat_completions"
    agent.api_key = "old-key"
    agent.base_url = "https://old.example/v1"
    agent._provider_request_timeout = 12
    agent._provider_stale_timeout = 34
    agent._client_kwargs = {
        "api_key": "new-key",
        "base_url": "https://new.example/v1",
        "default_headers": {"X-Test": "preserved"},
    }
    agent._deferred_provider_runtime = None
    captured = None

    async def ensure_runtime():
        nonlocal captured
        captured = dict(agent._deferred_provider_runtime)
        agent._deferred_provider_runtime = None
        return True

    agent._ensure_provider_runtime = ensure_runtime

    assert await agent._replace_primary_openai_client(reason="test") is True
    assert captured == {
        "provider": "openrouter",
        "model": "test-model",
        "api_key": "new-key",
        "base_url": "https://new.example/v1",
        "api_mode": "chat_completions",
        "request_timeout": 12,
        "stale_timeout": 34,
        "client_kwargs": agent._client_kwargs,
        "update_primary": False,
    }


@pytest.mark.asyncio
async def test_async_context_manager_initializes_provider_mcp_and_tools():
    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        model="test-model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._ensure_provider_runtime = AsyncMock()
    agent.close = AsyncMock()

    with (
        patch("tools.mcp_tool.retain_mcp_lifecycle", new=AsyncMock()) as retain,
        patch("tools.mcp_tool.discover_mcp_tools", new=AsyncMock()) as discover,
        patch("tools.mcp_tool.release_mcp_lifecycle", new=AsyncMock()) as release,
        patch(
            "tools.mcp_tool.refresh_agent_mcp_tools", new=AsyncMock()
        ) as refresh,
    ):
        async with agent as entered:
            assert entered is agent
            assert agent._tool_snapshot_initialized is True

    agent._ensure_provider_runtime.assert_awaited_once()
    retain.assert_awaited_once_with(agent)
    discover.assert_awaited_once()
    refresh.assert_awaited_once_with(agent, quiet_mode=True)
    release.assert_not_awaited()
    agent.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_context_manager_rolls_back_failed_mcp_initialization():
    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        model="test-model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._ensure_provider_runtime = AsyncMock()

    with (
        patch("tools.mcp_tool.retain_mcp_lifecycle", new=AsyncMock()),
        patch("tools.mcp_tool.discover_mcp_tools", new=AsyncMock()),
        patch("tools.mcp_tool.release_mcp_lifecycle", new=AsyncMock()) as release,
        patch(
            "tools.mcp_tool.refresh_agent_mcp_tools",
            new=AsyncMock(side_effect=RuntimeError("snapshot failed")),
        ),
        pytest.raises(RuntimeError, match="snapshot failed"),
    ):
        await agent.__aenter__()

    release.assert_awaited_once_with(agent)
    assert agent._mcp_lifecycle_retained is False
    assert agent._mcp_discovery_started is False


@pytest.mark.asyncio
async def test_session_billing_route_update_uses_native_connection(tmp_path):
    """A model switch must not fall back to the synchronous SessionDB writer."""
    database = SessionDB(tmp_path / "state.db")
    database = database
    try:
        await database.create_session("route", "test", model="initial")
        await database.update_session_billing_route(
            "route",
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            billing_mode="chat_completions",
        )
        session = await database.get_session("route")
        assert session["billing_provider"] == "openrouter"
        assert session["billing_base_url"] == "https://openrouter.ai/api/v1"
        assert session["billing_mode"] == "chat_completions"
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_session_db_can_bootstrap_from_a_path(tmp_path):
    """The active path does not need a synchronously opened SessionDB."""
    database = SessionDB(tmp_path / "state.db")
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
async def test_session_db_meta_round_trip(tmp_path):
    """Session metadata uses the native async connection."""
    database = SessionDB(tmp_path / "state.db")
    try:
        await database.set_meta("goal:session", '{"status":"active"}')
        assert await database.get_meta("goal:session") == '{"status":"active"}'
        assert await database.delete_meta("missing") is False
        assert await database.delete_meta("goal:session") is True
        assert await database.get_meta("goal:session") is None

    finally:
        await database.close()


@pytest.mark.asyncio
async def test_native_file_read_deduplicates_and_write_invalidates(tmp_path, monkeypatch):
    """File-loop guards survive the sync-backend removal on native I/O."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "tools.file_tools._check_sensitive_path", AsyncMock(return_value=None)
    )
    path = tmp_path / "sample.txt"
    task_id = f"native-file-{tmp_path.name}"
    path.write_text("first\nsecond\n", encoding="utf-8")

    first = json.loads(await read_file_tool(str(path), task_id=task_id))
    second = json.loads(await read_file_tool(str(path), task_id=task_id))

    assert "content" in first
    assert second["dedup"] is True
    assert second["status"] == "unchanged"

    written = json.loads(
        await write_file_tool(str(path), "updated\n", task_id=task_id)
    )
    refreshed = json.loads(await read_file_tool(str(path), task_id=task_id))

    assert written["files_modified"] == [str(path)]
    assert "updated" in refreshed["content"]
    assert refreshed.get("dedup") is not True


@pytest.mark.asyncio
async def test_model_switch_route_is_deferred_to_the_async_turn_boundary(tmp_path):
    """The sync state switch never writes SQLite directly."""
    database = SessionDB(tmp_path / "state.db")
    database = database
    agent = AIAgent.__new__(AIAgent)
    agent._session_db = database
    agent._persist_disabled = False
    agent.session_id = "route"
    agent._pending_billing_route = {
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "billing_mode": "chat_completions",
    }
    try:
        await database.create_session("route", "test")
        await agent._persist_pending_billing_route()
        session = await database.get_session("route")
        assert session["billing_provider"] == "openrouter"
        assert agent._pending_billing_route is None
    finally:
        await database.close()


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
        agent._provider_request_timeout = None
        agent._provider_stale_timeout = None
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
async def test_plugin_lifecycle_requires_coroutine_callbacks():
    """Sync plugin hooks cannot quietly stall an async agent turn."""
    from hermes_cli.plugins import PluginContractError, PluginManager

    manager = PluginManager()
    manager._hooks["turn"] = [lambda **_kwargs: None]
    with pytest.raises(PluginContractError, match="coroutine lifecycle hooks"):
        await manager.invoke_hook("turn")

    async def callback(**_kwargs):
        return "native"

    manager._hooks["turn"] = [callback]
    assert await manager.invoke_hook("turn") == ["native"]


@pytest.mark.asyncio
async def test_deferred_runtime_rejects_sync_only_context_engine_early(monkeypatch):
    """Provider construction must stop before an external legacy extension runs."""
    from agent.agent_init import initialize_deferred_runtime
    from agent.context_engine import ContextEngine
    from hermes_cli.plugins import PluginContractError

    class SyncEngine(ContextEngine):
        @property
        def name(self):
            return "third-party"

        def update_from_response(self, usage):
            return None

        def should_compress(self, prompt_tokens=None):
            return False

        def compress(self, messages, **kwargs):
            return messages

    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_context_engine",
        AsyncMock(return_value=SyncEngine()),
    )

    async def assert_rejected(**attributes):
        state = {
            "_deferred_provider_runtime": {"provider": "openrouter", "model": "test"},
            "_provider_init_lock": None,
            "_dotenv_loaded": True,
            "_runtime_config_loaded": True,
            "_runtime_config_snapshot": {"context": {"engine": "compressor"}},
        }
        state.update(attributes)
        agent = SimpleNamespace(**state)
        with pytest.raises(PluginContractError, match="compress.*coroutine"):
            await initialize_deferred_runtime(agent)

    await assert_rejected(
        _runtime_config_snapshot={"context": {"engine": "third-party"}}
    )


@pytest.mark.asyncio
async def test_deferred_runtime_starts_native_context_engine_with_tools(monkeypatch):
    """A configured native engine keeps the upstream selection/tool contract."""
    from agent.context_engine import ContextEngine

    class NativeEngine(ContextEngine):
        def __init__(self):
            self.started = []

        @property
        def name(self):
            return "native-test"

        def update_from_response(self, usage):
            return None

        def should_compress(self, prompt_tokens=None):
            return False

        async def compress(self, messages, **kwargs):
            return messages

        async def on_session_start(self, session_id, **kwargs):
            self.started.append((session_id, kwargs))

        def get_tool_schemas(self):
            return [{
                "name": "context_lookup",
                "description": "Look up context",
                "parameters": {"type": "object", "properties": {}},
            }]

        async def handle_tool_call(self, name, args, **kwargs):
            return json.dumps({"name": name})

    source_engine = NativeEngine()
    config = {
        "context": {"engine": "native-test"},
        "compression": {"model_thresholds": {"test-model": 0.42}},
    }
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_context_engine",
        AsyncMock(return_value=source_engine),
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        AsyncMock(return_value=config),
    )
    context_length = AsyncMock(return_value=128_000)
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        context_length,
    )

    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        provider="openrouter",
        model="test-model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    await agent._ensure_provider_runtime()

    assert agent.context_compressor is not source_engine
    assert agent.context_compressor.name == "native-test"
    assert agent.context_compressor.context_length == 128_000
    assert agent.context_compressor.model_thresholds == {"test-model": 0.42}
    assert len(agent.context_compressor.started) == 1
    assert "context_lookup" in agent._context_engine_tool_names
    assert "context_lookup" in agent.valid_tool_names
    assert any(
        tool.get("function", {}).get("name") == "context_lookup"
        for tool in agent.tools
    )
    context_length.assert_awaited_once()
    await agent.close()




def test_api_timeout_resolution_uses_the_constructor_snapshot(monkeypatch):
    """Core request construction must not reread provider settings mid-turn."""
    import run_agent

    def fail_if_settings_are_read(*_args, **_kwargs):
        raise AssertionError("a native async turn must use its timeout snapshot")

    monkeypatch.setattr(run_agent, "get_provider_request_timeout", fail_if_settings_are_read)
    monkeypatch.setattr(run_agent, "get_provider_stale_timeout", fail_if_settings_are_read)
    agent = SimpleNamespace(
        _provider_request_timeout=42.0,
        _provider_stale_timeout=84.0,
        provider="custom",
        model="test-model",
    )

    assert AIAgent._resolved_api_call_timeout(agent) == 42.0
    assert AIAgent._resolved_api_call_stale_timeout_base(agent) == (84.0, False)


@pytest.mark.asyncio
async def test_api_key_pool_rotation_is_native_transport_and_persistent(monkeypatch, tmp_path):
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
@pytest.mark.parametrize("constructor_provider", ["qwen-oauth", None])
async def test_deferred_runtime_uses_native_qwen_oauth_resolver(
    monkeypatch,
    tmp_path,
    constructor_provider,
):
    """Explicit and config-selected Qwen routes keep the upstream OAuth path."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    config = {
        "context": {"engine": "compressor"},
        "model": {"provider": "qwen-oauth", "default": "qwen3-coder-plus"},
    }
    resolve_qwen = AsyncMock(
        return_value={
            "provider": "qwen-oauth",
            "base_url": "https://portal.qwen.ai/v1",
            "api_key": "qwen-access-token",
            "source": "qwen-cli",
            "expires_at_ms": 123456789,
        }
    )

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch(
            "hermes_cli.config.load_config_readonly",
            new=AsyncMock(return_value=config),
        ),
        patch(
            "hermes_cli.auth.resolve_qwen_runtime_credentials",
            new=resolve_qwen,
        ),
    ):
        agent = AIAgent(
            provider=constructor_provider,
            model="qwen3-coder-plus",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        monkeypatch.setattr(
            asyncio,
            "to_thread",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("Qwen OAuth resolution must not thread-hop"),
            ),
        )
        await agent._ensure_provider_runtime()

    assert agent.provider == "qwen-oauth"
    assert agent.api_mode == "chat_completions"
    assert agent.api_key == "qwen-access-token"
    assert agent.base_url == "https://portal.qwen.ai/v1"
    resolve_qwen.assert_awaited()
    await agent.close()


@pytest.mark.asyncio
async def test_explicit_credentials_defer_sdk_construction_until_await(monkeypatch):
    """``AIAgent.__init__`` remains state-only even with an explicit key."""
    native_client = SimpleNamespace(close=AsyncMock(), _platform=None)
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI", return_value=native_client) as openai_factory,
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
        openai_factory.assert_not_called()

        monkeypatch.setattr(
            asyncio,
            "to_thread",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("deferred initialization must not thread-hop"),
            ),
        )
        await agent._ensure_provider_runtime()

    openai_factory.assert_called_once()
    assert agent.client is native_client
    assert agent.client._platform == "Unknown"
    await agent.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "api_mode"),
    [
        ("global.anthropic.claude-sonnet-4-6", "anthropic_messages"),
        ("amazon.nova-pro-v1:0", "bedrock_converse"),
    ],
)
async def test_bedrock_runtime_is_built_only_at_async_boundary(
    monkeypatch,
    model,
    api_mode,
):
    """Both Bedrock transports preserve lazy construction and native async dispatch."""
    bedrock_client = SimpleNamespace(aclose=AsyncMock())
    build_bedrock = AsyncMock(return_value=bedrock_client)
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch(
            "agent.anthropic_adapter.build_anthropic_bedrock_client",
            new=build_bedrock,
        ),
    ):
        agent = AIAgent(
            provider="bedrock",
            model=model,
            api_key="aws-sdk",
            base_url="https://bedrock-runtime.eu-west-1.amazonaws.com",
            api_mode=api_mode,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        assert agent.client is None
        assert agent._deferred_provider_runtime is not None
        build_bedrock.assert_not_awaited()

        monkeypatch.setattr(
            asyncio,
            "to_thread",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("Bedrock initialization must not thread-hop"),
            ),
        )
        await agent._ensure_provider_runtime()

    assert agent.api_mode == api_mode
    assert agent._bedrock_region == "eu-west-1"
    if api_mode == "anthropic_messages":
        build_bedrock.assert_awaited_once_with("eu-west-1")
        assert agent._anthropic_client is bedrock_client
    else:
        build_bedrock.assert_not_awaited()
        assert agent._anthropic_client is None
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
        patch("run_agent.OpenAI", return_value=native_client) as openai_factory,
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
    ):
        agent = AIAgent(
            provider="custom",
            model="test-model",
            base_url="https://custom.example/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        openai_factory.assert_not_called()
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
        await database.close()


@pytest.mark.asyncio
async def test_anthropic_pool_refresh_uses_native_transport_transport(monkeypatch, tmp_path):
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
async def test_unsupported_oauth_pool_refresh_preserves_upstream_noop(monkeypatch):
    """An unknown OAuth provider keeps the upstream no-op refresh semantics."""
    from agent.credential_pool import AUTH_TYPE_OAUTH, CredentialPool, PooledCredential

    pool = CredentialPool(
        "unsupported-oauth",
        [
            PooledCredential(
                provider="unsupported-oauth",
                id="unsupported",
                label="Unsupported",
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
    refreshed = await pool.try_refresh_matching(credential_id="unsupported")

    assert refreshed is not None
    assert refreshed.id == "unsupported"
    assert refreshed.access_token == "expired-access"


@pytest.mark.asyncio
async def test_nonpersistent_terminal_cleanup_awaits_native_handler(monkeypatch):
    from agent import chat_completion_helpers

    cleanup = AsyncMock()
    monkeypatch.setattr(chat_completion_helpers, "is_persistent_env", lambda _task_id: False)
    monkeypatch.setattr(chat_completion_helpers, "cleanup_vm", cleanup)

    await chat_completion_helpers.cleanup_task_resources(
        SimpleNamespace(verbose_logging=False),
        "task-1",
    )

    cleanup.assert_awaited_once_with("task-1")


@pytest.mark.asyncio
async def test_persistent_terminal_cleanup_remains_deferred(monkeypatch):
    from agent import chat_completion_helpers

    cleanup = AsyncMock()
    monkeypatch.setattr(chat_completion_helpers, "is_persistent_env", lambda _task_id: True)
    monkeypatch.setattr(chat_completion_helpers, "cleanup_vm", cleanup)

    await chat_completion_helpers.cleanup_task_resources(
        SimpleNamespace(verbose_logging=False),
        "task-1",
    )

    cleanup.assert_not_awaited()


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
        await database.close()


def test_native_file_tool_imports_expand_the_async_model_surface():
    """File tools join the model surface only after native migration."""
    from tools import file_tools  # noqa: F401

    assert inspect.iscoroutinefunction(registry.get_entry("read_file").handler)
    assert inspect.iscoroutinefunction(registry.get_entry("write_file").handler)


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
    monkeypatch.setattr(
        file_tools, "_check_sensitive_path", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        asyncio,
        "to_thread",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("native file tools must not call asyncio.to_thread")
        ),
    )

    blockbuster = BlockBuster()
    blockbuster.activate()
    try:
        written = json.loads(
            await active_registry.dispatch(
                "write_file",
                {"path": "notes.txt", "content": "first\nsecond\n"},
                task_id="native-file-test",
            )
        )
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
        read = json.loads(
            await active_registry.dispatch(
                "read_file", {"path": "notes.txt"}, task_id="native-file-test"
            )
        )
        found = json.loads(
            await active_registry.dispatch(
                "search_files",
                {"pattern": "third", "path": "."},
                task_id="native-file-test",
            )
        )
    finally:
        blockbuster.deactivate()

    assert written["bytes_written"] == len("first\nsecond\n".encode())
    assert patched["replacements"] == 1
    assert read["content"] == "1|first\n2|third"
    assert found["total_count"] == 1


@pytest.mark.asyncio
async def test_native_v4a_patch_is_async_and_validates_before_mutating(monkeypatch, tmp_path):
    """Multi-file edits keep V4A semantics without a synchronous backend."""
    import importlib

    file_tools = importlib.import_module("tools.file_tools")
    importlib.reload(file_tools)
    from tools.registry import registry as active_registry

    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.setattr(
        file_tools, "_check_sensitive_path", AsyncMock(return_value=None)
    )
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
async def test_fallback_swaps_to_a_native_transport_client(monkeypatch):
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
        with patch("run_agent.OpenAI", return_value=native_client) as openai_factory:
            assert await agent._try_activate_fallback()
            response = await agent._execute_model_request(
                {"model": agent.model, "messages": []}
            )

        openai_factory.assert_called_once()
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
        async with agent._get_turn_lock():
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
        async with agent._get_turn_lock():
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
        await database.close()


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
    monkeypatch.setattr(asyncio, "to_thread", fail_if_called)
    try:
        result = await agent.run_conversation("hello")
        assert result["final_response"] == "answer"
    finally:
        await agent.close()
        await database.close()


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
        await asyncio.wait_for(both_models_started.wait(), timeout=5.0)
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
            await database.close()


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
async def test_close_cancels_and_awaits_active_turn_without_task_leak():
    agent = AIAgent.__new__(AIAgent)
    turn_started = asyncio.Event()

    async def active_turn() -> None:
        turn_started.set()
        await asyncio.Event().wait()

    async with no_task_leaks(action=LeakAction.RAISE):
        turn_task = asyncio.create_task(active_turn())
        await turn_started.wait()
        agent._active_turn_task = turn_task

        await agent.close()

        assert turn_task.cancelled()
        assert agent._closed is True


@pytest.mark.asyncio
async def test_close_releases_retained_mcp_lifecycle(monkeypatch):
    from tools import mcp_tool

    agent = AIAgent.__new__(AIAgent)
    agent._mcp_lifecycle_retained = True
    released = []

    async def release(owner):
        released.append(owner)

    monkeypatch.setattr(mcp_tool, "release_mcp_lifecycle", release)

    await agent.close()

    assert released == [agent]
    assert agent._mcp_lifecycle_retained is False


@pytest.mark.asyncio
async def test_close_awaits_the_native_primary_client():
    """The async lifecycle closes the primary transport directly."""
    agent = AIAgent.__new__(AIAgent)
    closed_native = False

    class NativeClient:
        async def aclose(self):
            nonlocal closed_native
            closed_native = True

    agent.client = NativeClient()
    agent._anthropic_client = None
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
async def test_session_db_writes_without_to_thread(tmp_path, monkeypatch):
    database = SessionDB(tmp_path / "state.db")
    session_db = database

    def fail_if_called(*args, **kwargs):
        raise AssertionError("core async persistence must not call asyncio.to_thread")

    monkeypatch.setattr(asyncio, "to_thread", fail_if_called)
    try:
        await session_db.create_session("async-session", "test", model="test-model")
        await session_db.append_message("async-session", "user", "hello")
        await session_db.end_session("async-session", "test_complete")

        stored = await database.get_session("async-session")
        assert stored["model"] == "test-model"
        assert [
            message["content"]
            for message in await database.get_messages("async-session")
        ] == ["hello"]
        assert stored["end_reason"] == "test_complete"
    finally:
        await session_db.close()
        await database.close()


@pytest.mark.asyncio
async def test_conversation_root_uses_async_session_db(tmp_path, monkeypatch):
    database = SessionDB(tmp_path / "state.db")
    await database.create_session("root", "test")
    await database.create_session("child", "test", parent_session_id="root")
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "child"
    agent._parent_session_id = None
    agent._session_db = database
    session_db = database

    try:
        assert await agent._conversation_root_id() == "root"
    finally:
        await session_db.close()
        await database.close()


@pytest.mark.asyncio
async def test_session_db_loads_compression_snapshot(tmp_path, monkeypatch):
    database = SessionDB(tmp_path / "state.db")
    session_db = database
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
        await database.close()


@pytest.mark.asyncio
async def test_session_db_persists_compression_guards(tmp_path, monkeypatch):
    database = SessionDB(tmp_path / "state.db")
    session_db = database
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
        await database.close()


@pytest.mark.asyncio
async def test_micro_compaction_persists_through_async_session_db(tmp_path, monkeypatch):
    """Micro-compaction must not fall back to SessionDB or a worker thread."""
    from agent.context_compressor import (
        COMPRESSED_SUMMARY_METADATA_KEY,
        ContextCompressor,
    )

    database = SessionDB(tmp_path / "state.db")
    session_db = database
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
        await database.close()


@pytest.mark.asyncio
async def test_auxiliary_accounting_writes_without_to_thread(tmp_path, monkeypatch):
    """An auxiliary model response persists usage through SessionDB only."""
    from agent.aux_accounting import reset_accounting_context, set_accounting_context
    from agent.auxiliary_client import _validate_llm_response

    database = SessionDB(tmp_path / "state.db")
    session_db = database
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
        await database.close()


@pytest.mark.asyncio
async def test_session_db_backfills_api_sidecar_without_to_thread(
    tmp_path, monkeypatch
):
    database = SessionDB(tmp_path / "state.db")
    session_db = database

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
        messages = await database.get_messages_as_conversation("sidecar-session")
        assert messages[0]["api_content"] == "clean prompt\n\n<context/>"
    finally:
        await session_db.close()
        await database.close()


@pytest.mark.asyncio
async def test_session_db_compacts_and_releases_lease_without_to_thread(
    tmp_path, monkeypatch
):
    """The async compaction primitives preserve active/archived transcript rows."""
    database = SessionDB(tmp_path / "state.db")
    session_db = database

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

        active = await database.get_messages("compact-session")
        archived = await database.get_messages(
            "compact-session", include_inactive=True
        )
        assert [message["content"] for message in active] == [
            "original question",
            "compressed answer",
        ]
        assert len(archived) == 3
        assert (await database.get_session("compact-session"))["system_prompt"] == (
            "stable system prompt"
        )
        assert await session_db.get_compression_lock_holder("compact-session") is None
    finally:
        await session_db.close()
        await database.close()


@pytest.mark.asyncio
async def test_in_place_compression_uses_native_transport_sqlite_path(tmp_path, monkeypatch):
    """A complete in-place compaction never re-enters the sync SessionDB API."""
    database = SessionDB(tmp_path / "state.db")
    session_db = database

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
                session_id="compression-session",
        context_compressor=compressor,
        api_mode=None,
        compression_in_place=True,
        _cached_system_prompt="stable system prompt",
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
        assert [
            message["content"]
            for message in await database.get_messages(agent.session_id)
        ] == [
            "question",
            "compressed answer",
        ]
    finally:
        await session_db.close()
        await database.close()


@pytest.mark.asyncio
async def test_rotating_compression_publishes_child_with_native_transport_sqlite(tmp_path):
    """The optional rotation mode retains its atomic parent/child handoff."""
    database = SessionDB(tmp_path / "state.db")
    session_db = database

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
                session_id="rotation-parent",
        context_compressor=Compressor(),
        api_mode=None,
        compression_in_place=False,
        _cached_system_prompt="stable system prompt",
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
        assert (await database.get_session("rotation-parent"))["end_reason"] == "compression"
        assert [
            message["content"]
            for message in await database.get_messages(agent.session_id)
        ] == [
            message["content"] for message in compressed
        ]
    finally:
        await session_db.close()
        await database.close()


@pytest.mark.asyncio
async def test_agent_session_lifecycle_uses_native_transport_store(tmp_path, monkeypatch):
    database = SessionDB(tmp_path / "state.db")
    agent = AIAgent.__new__(AIAgent)
    agent._persist_disabled = False
    agent._session_db = database
    agent._session_persist_lock = None
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
        assert [
            message["content"]
            for message in await database.get_messages(agent.session_id)
        ] == ["hello"]
    finally:
        await agent._session_db.close()
        await database.close()


@pytest.mark.asyncio
async def test_persist_session_does_not_reenter_its_async_lock(tmp_path, monkeypatch):
    """The persist funnel owns one lock and calls its unlocked DB writer."""
    database = SessionDB(tmp_path / "state.db")
    agent = AIAgent.__new__(AIAgent)
    agent._persist_disabled = False
    agent._session_db = database
    agent._session_persist_lock = None
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
        assert [
            message["content"]
            for message in await database.get_messages(agent.session_id)
        ] == [
            "hello"
        ]
    finally:
        await agent._session_db.close()
        await database.close()


@pytest.mark.asyncio
async def test_native_transport_terminal_does_not_use_to_thread(monkeypatch, tmp_path):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("terminal must use asyncio subprocesses directly")

    monkeypatch.setattr(asyncio, "to_thread", fail_if_called)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.chdir(tmp_path)

    blockbuster = BlockBuster()
    blockbuster.activate()
    try:
        result = json.loads(
            await terminal_tool("printf async-terminal", task_id="async-test")
        )
    finally:
        blockbuster.deactivate()

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
    agent._session_db_created = False
    agent.compression_enabled = False
    agent._skip_mcp_refresh = True

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
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=4,
                total_tokens=15,
            ),
        )

    agent._execute_model_request = model_response

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("the public async turn must not call asyncio.to_thread")

    monkeypatch.setattr(asyncio, "to_thread", fail_if_called)
    try:
        async with (
            no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
            no_task_leaks(action=LeakAction.RAISE),
        ):
            blockbuster = BlockBuster()
            blockbuster.activate()
            try:
                result = await agent.run_conversation("hello async")
            finally:
                blockbuster.deactivate()
        assert result["completed"] is True
        assert result["final_response"] == "async answer"
        assert [message["role"] for message in result["messages"]] == [
            "user",
            "assistant",
        ]
        assert [
            message["content"]
            for message in await database.get_messages(agent.session_id)
        ] == [
            "hello async",
            "async answer",
        ]
        session = await database.get_session(agent.session_id)
        assert session["input_tokens"] == 11
        assert session["output_tokens"] == 4
        assert session["api_call_count"] == 1
    finally:
        await agent.close()
        await database.close()


@pytest.mark.asyncio
async def test_oauth_renewal_uses_native_async_refresh(monkeypatch, tmp_path):
    """A reactive OAuth renewal is awaited and never replaced by a sync helper."""
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
    agent._session_db_created = False
    agent.compression_enabled = False

    async def unauthorized_model(*_args, **_kwargs):
        raise UnauthorizedError("expired OAuth token")

    agent._execute_model_request = unauthorized_model
    agent._try_refresh_codex_client_credentials = AsyncMock(return_value=False)
    try:
        result = await agent.run_conversation("hello async")
        assert result["completed"] is False
        assert result["failed"] is True
        assert "expired OAuth token" in result["error"]
        agent._try_refresh_codex_client_credentials.assert_awaited_once_with(force=True)
    finally:
        await agent.close()
        await database.close()


@pytest.mark.asyncio
async def test_codex_responses_main_path_uses_native_transport_client(monkeypatch):
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
    assert agent.client is native_client
    assert created_requests == [{"model": "gpt-test"}]


@pytest.mark.asyncio
async def test_synthetic_model_tool_observation_turn_preserves_order(monkeypatch, tmp_path):
    """The model → tool → observation → model training shape stays ordered."""
    monkeypatch.chdir(tmp_path)
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
            save_trajectories=True,
        )
    agent._session_db = database
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
        assert [
            message["role"]
            for message in await database.get_messages(agent.session_id)
        ] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]

        rows = [
            json.loads(line)
            for line in (tmp_path / "trajectory_samples.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(rows) == 1
        trajectory = rows[0]["conversations"]
        assert [turn["from"] for turn in trajectory] == [
            "system",
            "human",
            "gpt",
            "tool",
            "gpt",
        ]
        assert "<think>\ninspect the tool result\n</think>" in trajectory[2]["value"]
        assert '"name": "terminal"' in trajectory[2]["value"]
        assert '"tool_call_id": "call-1"' in trajectory[3]["value"]
        assert "tool-observation" in trajectory[3]["value"]
        assert "<think>\nanswer after observation\n</think>" in trajectory[4]["value"]
        assert trajectory[4]["value"].endswith("tool observation incorporated")
    finally:
        await agent.close()
        await database.close()


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
    task = None
    try:
        async with no_task_leaks(action=LeakAction.RAISE):
            task = asyncio.create_task(
                agent.run_conversation("persist this before cancel")
            )
            await asyncio.wait_for(model_started.wait(), timeout=0.5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert [
            message["content"]
            for message in await database.get_messages(agent.session_id)
        ] == [
            "persist this before cancel"
        ]
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await agent.close()
        await database.close()


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
    agent._session_db_created = False
    agent.compression_enabled = False
    agent.valid_tool_names = {tool_name}
    agent._tool_snapshot_initialized = True

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

    task = None
    try:
        async with no_task_leaks(action=LeakAction.RAISE):
            task = asyncio.create_task(agent.run_conversation("cancel this tool"))
            await asyncio.wait_for(tool_started.wait(), timeout=0.5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        stored = await database.get_messages(agent.session_id)
        assert [message["role"] for message in stored] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        assert "cancelled" in stored[-2]["content"].lower()
        assert stored[-1]["content"] == "Operation interrupted."
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await agent.close()
        await database.close()


@pytest.mark.asyncio
async def test_registry_handler_is_awaited(monkeypatch):
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
async def test_memory_tool_uses_native_transport_file_path(monkeypatch, tmp_path):
    """Memory remains durable without exposing a blocking handler to the loop."""
    import importlib

    memory_tool_module = importlib.import_module("tools.memory_tool")
    active_registry = importlib.import_module("tools.registry").registry

    monkeypatch.setattr(memory_tool_module, "get_memory_dir", lambda: tmp_path / "memories")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("memory tool must not call asyncio.to_thread")

    monkeypatch.setattr(asyncio, "to_thread", fail_if_called)
    store = memory_tool_module.MemoryStore(memory_char_limit=200, user_char_limit=200)
    blockbuster = BlockBuster()
    blockbuster.activate()
    try:
        await store.load_from_disk()
        initial_snapshot = store.format_for_system_prompt("memory")
        response = await memory_tool_module.memory_tool(
            action="add",
            target="memory",
            content="Prefer native async file I/O.",
            store=store,
        )
        registry_response = await active_registry.dispatch(
            "memory",
            {"action": "add", "target": "memory", "content": "Registry awaits handlers."},
            store=store,
        )
    finally:
        blockbuster.deactivate()

    assert '"success": true' in response
    assert "Prefer native async file I/O." in (tmp_path / "memories" / "MEMORY.md").read_text()
    assert store.format_for_system_prompt("memory") == initial_snapshot
    assert registry_response


@pytest.mark.asyncio
async def test_native_background_terminal_process_is_reaped_at_cleanup():
    """Async terminal children never outlive their owning task/session."""
    import importlib

    terminal_module = importlib.import_module("tools.terminal_tool")
    process_module = importlib.import_module("tools.process_registry")

    task_id = "async-background-reap"
    command = f"{shlex.quote(sys.executable)} -c 'import time; time.sleep(30)'"
    started = json.loads(
        await terminal_module.terminal_tool(command, background=True, task_id=task_id)
    )
    assert started["output"] == "Background process started"
    session = process_module.process_registry.get(started["session_id"])
    assert session is not None
    assert session.process is not None

    await terminal_module.cleanup_vm(task_id)

    assert started["session_id"] not in process_module.process_registry.snapshot_running_ids(
        task_id
    )
    assert session.process.returncode is not None


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
async def test_tool_scheduler_preserves_barriers_and_result_order(monkeypatch):
    import importlib

    active_executor = importlib.import_module("agent.tool_executor")
    active_model_tools = importlib.import_module("model_tools")
    active_registry = importlib.import_module("tools.registry").registry
    monkeypatch.setattr(active_model_tools, "registry", active_registry)
    events = []
    originals = dict(active_registry._tools)
    monkeypatch.setattr(active_registry, "_tools", dict(originals))
    monkeypatch.setattr(
        active_executor, "_begin_tool_execution", AsyncMock(return_value=None)
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
