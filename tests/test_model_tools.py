"""Tests for model_tools.py — function call dispatch and toolsets."""

import asyncio
import inspect
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from model_tools import (
    handle_function_call,
    get_all_tool_names,
    get_toolset_for_tool,
    _LEGACY_TOOLSET_MAP,
    TOOL_TO_TOOLSET_MAP,
)
from tools.todo_tool import TodoStore
from agent.agent_runtime_helpers import invoke_tool


# =========================================================================
# handle_function_call
# =========================================================================

class TestHandleFunctionCall:
    def test_public_signature_matches_upstream(self):
        assert list(inspect.signature(handle_function_call).parameters) == [
            "function_name",
            "function_args",
            "task_id",
            "tool_call_id",
            "session_id",
            "turn_id",
            "api_request_id",
            "user_task",
            "enabled_tools",
            "skip_pre_tool_call_hook",
            "skip_tool_request_middleware",
            "skip_tool_execution_middleware",
            "tool_request_middleware_trace",
            "enabled_toolsets",
            "disabled_toolsets",
        ]

    @pytest.mark.asyncio
    async def test_upstream_positional_dispatch_contract_is_preserved(self):
        with patch(
            "model_tools.registry.dispatch",
            new_callable=AsyncMock,
            return_value='{"ok":true}',
        ) as dispatch:
            result = await handle_function_call(
                "web_search",
                {"q": "test"},
                "task-1",
                "tool-1",
                "session-1",
                "turn-1",
                "request-1",
                "user task",
                ["web_search"],
                True,
                True,
                True,
                [],
                ["web"],
                [],
            )

        assert result == '{"ok":true}'
        assert dispatch.await_args.kwargs["session_id"] == "session-1"
        assert dispatch.await_args.kwargs["turn_id"] == "turn-1"
        assert dispatch.await_args.kwargs["api_request_id"] == "request-1"

    @pytest.mark.asyncio
    async def test_stateful_tool_dispatches_with_agent_context(self):
        store = TodoStore()
        agent = SimpleNamespace(
            _todo_store=store,
            valid_tool_names={"todo"},
            enabled_toolsets=None,
            disabled_toolsets=None,
            session_id="session-1",
        )
        result = json.loads(
            await invoke_tool(
                agent,
                "todo",
                {
                    "todos": [
                        {
                            "id": "live-check",
                            "content": "Native async dispatch",
                            "status": "completed",
                        }
                    ]
                },
                "task-1",
                skip_tool_request_middleware=True,
                skip_tool_execution_middleware=True,
            )
        )

        assert result["todos"] == [
            {
                "id": "live-check",
                "content": "Native async dispatch",
                "status": "completed",
            }
        ]
        assert result["summary"]["completed"] == 1

    @pytest.mark.asyncio
    async def test_concurrent_agent_tool_contexts_are_isolated(self):
        agents = [
            SimpleNamespace(
                _todo_store=TodoStore(),
                valid_tool_names={"todo"},
                enabled_toolsets=None,
                disabled_toolsets=None,
                session_id=f"session-{index}",
            )
            for index in range(2)
        ]

        results = await asyncio.gather(
            *(
                invoke_tool(
                    agent,
                    "todo",
                    {
                        "todos": [
                            {
                                "id": f"task-{index}",
                                "content": f"Agent {index}",
                                "status": "pending",
                            }
                        ]
                    },
                    f"task-{index}",
                    skip_tool_request_middleware=True,
                    skip_tool_execution_middleware=True,
                )
                for index, agent in enumerate(agents)
            )
        )

        assert [json.loads(result)["todos"][0]["id"] for result in results] == [
            "task-0",
            "task-1",
        ]
        assert [
            store.read()[0]["id"]
            for store in (agent._todo_store for agent in agents)
        ] == ["task-0", "task-1"]

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        result = json.loads(await handle_function_call("totally_fake_tool_xyz", {}))
        assert "error" in result
        assert "totally_fake_tool_xyz" in result["error"]

    @pytest.mark.asyncio
    async def test_tool_request_middleware_failure_is_fail_open(self):
        with (
            patch(
                "hermes_cli.middleware.apply_tool_request_middleware",
                new_callable=AsyncMock,
                side_effect=RuntimeError("request middleware failed"),
            ),
            patch(
                "model_tools.registry.dispatch",
                new_callable=AsyncMock,
                return_value='{"ok":true}',
            ) as dispatch,
        ):
            result = await handle_function_call(
                "web_search",
                {"q": "original"},
                skip_pre_tool_call_hook=True,
                skip_tool_execution_middleware=True,
            )

        assert result == '{"ok":true}'
        assert dispatch.await_args.args[1] == {"q": "original"}

    @pytest.mark.asyncio
    async def test_pre_tool_hook_failure_is_fail_open(self):
        with (
            patch(
                "hermes_cli.plugins.resolve_pre_tool_block",
                new_callable=AsyncMock,
                side_effect=RuntimeError("pre hook failed"),
            ),
            patch(
                "model_tools.registry.dispatch",
                new_callable=AsyncMock,
                return_value='{"ok":true}',
            ) as dispatch,
        ):
            result = await handle_function_call(
                "web_search",
                {"q": "test"},
                skip_tool_request_middleware=True,
                skip_tool_execution_middleware=True,
            )

        assert result == '{"ok":true}'
        dispatch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_dispatch_failure_returns_upstream_tool_error_shape(self):
        with patch(
            "model_tools.registry.dispatch",
            new_callable=AsyncMock,
            side_effect=RuntimeError("provider secret=abc"),
        ):
            result = json.loads(
                await handle_function_call(
                    "web_search",
                    {"q": "test"},
                    skip_pre_tool_call_hook=True,
                    skip_tool_request_middleware=True,
                    skip_tool_execution_middleware=True,
                )
            )

        assert result == {
            "error": "[TOOL_ERROR] Error executing web_search: provider secret=abc"
        }

    @pytest.mark.asyncio
    async def test_transform_hook_failure_is_fail_open(self):
        with (
            patch(
                "model_tools.registry.dispatch",
                new_callable=AsyncMock,
                return_value='{"ok":true}',
            ),
            patch(
                "hermes_cli.lifecycle.has_hook",
                side_effect=lambda name: name == "transform_tool_result",
            ),
            patch(
                "hermes_cli.lifecycle.invoke_hook",
                new_callable=AsyncMock,
                side_effect=RuntimeError("transform failed"),
            ),
        ):
            result = await handle_function_call(
                "web_search",
                {"q": "test"},
                skip_pre_tool_call_hook=True,
                skip_tool_request_middleware=True,
                skip_tool_execution_middleware=True,
            )

        assert result == '{"ok":true}'

    @pytest.mark.asyncio
    async def test_dispatch_cancellation_is_not_converted_to_tool_error(self):
        with patch(
            "model_tools.registry.dispatch",
            new_callable=AsyncMock,
            side_effect=asyncio.CancelledError,
        ):
            with pytest.raises(asyncio.CancelledError):
                await handle_function_call(
                    "web_search",
                    {"q": "test"},
                    skip_pre_tool_call_hook=True,
                    skip_tool_request_middleware=True,
                    skip_tool_execution_middleware=True,
                )



    @pytest.mark.asyncio
    async def test_post_tool_call_receives_non_negative_integer_duration_ms(self):
        """Regression: post_tool_call and transform_tool_result hooks must
        receive a non-negative integer ``duration_ms`` kwarg measuring
        dispatch latency.  Inspired by Claude Code 2.1.119, which added
        ``duration_ms`` to its PostToolUse hook inputs.
        """
        with (
            patch("model_tools.registry.dispatch", return_value='{"ok":true}'),
            patch("hermes_cli.plugins.has_hook", return_value=True),
            patch(
                "hermes_cli.plugins.invoke_hook",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_invoke_hook,
        ):
            await handle_function_call("web_search", {"q": "test"}, task_id="t1")

        kwargs_by_hook = {
            c.args[0]: c.kwargs for c in mock_invoke_hook.call_args_list
        }
        assert "duration_ms" in kwargs_by_hook["post_tool_call"]
        assert "duration_ms" in kwargs_by_hook["transform_tool_result"]

        post_duration = kwargs_by_hook["post_tool_call"]["duration_ms"]
        transform_duration = kwargs_by_hook["transform_tool_result"]["duration_ms"]
        assert isinstance(post_duration, int)
        assert post_duration >= 0
        # Both hooks should observe the same measured duration.
        assert post_duration == transform_duration
        # pre_tool_call does NOT get duration_ms (nothing has run yet).
        assert "duration_ms" not in kwargs_by_hook["pre_tool_call"]


    @pytest.mark.asyncio
    async def test_tool_request_and_execution_middleware_wrap_registry_dispatch(self, monkeypatch):
        seen = {}

        async def request_middleware(**kwargs):
            return {
                "args": {**kwargs["args"], "rewritten": True},
                "source": "test-middleware",
                "reason": "rewrite",
            }

        async def execution_middleware(**kwargs):
            seen["execution_args"] = kwargs["args"]
            return await kwargs["next_call"]({**kwargs["args"], "wrapped": True})

        async def fake_dispatch(tool_name, args, **kwargs):
            seen["dispatch"] = (tool_name, args, kwargs)
            return json.dumps({"ok": True, "args": args})

        manager = type(
            "Manager",
            (),
            {"_middleware": {"tool_request": [request_middleware], "tool_execution": [execution_middleware]}},
        )()
        async def invoke_middleware(kind, **kwargs):
            return [
                await callback(**kwargs)
                for callback in manager._middleware.get(kind, [])
            ]
        monkeypatch.setattr("hermes_cli.plugins.invoke_middleware", invoke_middleware)
        monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
        hook_calls = []
        async def invoke_hook(hook_name, **kwargs):
            hook_calls.append((hook_name, kwargs))
            return []
        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
        monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
        monkeypatch.setattr("model_tools.registry.dispatch", fake_dispatch)

        result = json.loads(
            await handle_function_call(
                "web_search",
                {"q": "test"},
                task_id="task-1",
                tool_call_id="tool-1",
                session_id="session-1",
            )
        )

        assert seen["execution_args"] == {"q": "test", "rewritten": True}
        assert seen["dispatch"][1] == {"q": "test", "rewritten": True, "wrapped": True}
        assert result["args"] == {"q": "test", "rewritten": True, "wrapped": True}
        expected_trace = [{"source": "test-middleware", "reason": "rewrite"}]
        pre_call = next(call for call in hook_calls if call[0] == "pre_tool_call")
        post_call = next(call for call in hook_calls if call[0] == "post_tool_call")
        assert pre_call[1]["middleware_trace"] == expected_trace
        assert post_call[1]["middleware_trace"] == expected_trace


# =========================================================================
# Pre-tool-call blocking via plugin hooks
# =========================================================================

class TestPreToolCallBlocking:
    """Verify that pre_tool_call hooks can block tool execution."""

    @pytest.mark.asyncio
    async def test_blocked_tool_returns_error_and_skips_dispatch(self, monkeypatch):
        hook_calls = []

        async def fake_invoke_hook(hook_name, **kwargs):
            hook_calls.append((hook_name, kwargs))
            if hook_name == "pre_tool_call":
                return [{"action": "block", "message": "Blocked by policy"}]
            return []

        dispatch_called = False
        _orig_dispatch = None

        def fake_dispatch(*args, **kwargs):
            nonlocal dispatch_called
            dispatch_called = True
            raise AssertionError("dispatch should not run when blocked")

        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_invoke_hook)
        monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: True)
        monkeypatch.setattr("model_tools.registry.dispatch", fake_dispatch)

        result = json.loads(
            await handle_function_call("read_file", {"path": "test.txt"}, task_id="t1")
        )
        assert result == {"error": "Blocked by policy"}
        assert not dispatch_called
        post_call = next(call for call in hook_calls if call[0] == "post_tool_call")
        assert post_call[1]["status"] == "blocked"
        assert post_call[1]["error_type"] == "plugin_block"
        assert post_call[1]["error_message"] == "Blocked by policy"
        assert post_call[1]["duration_ms"] == 0

    @pytest.mark.asyncio
    async def test_blocked_tool_skips_read_loop_notification(self, monkeypatch):
        notifications = []

        async def fake_invoke_hook(hook_name, **kwargs):
            if hook_name == "pre_tool_call":
                return [{"action": "block", "message": "Blocked"}]
            return []

        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_invoke_hook)
        monkeypatch.setattr("model_tools.registry.dispatch",
                            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not run")))
        monkeypatch.setattr("tools.file_tools.notify_other_tool_call",
                            lambda task_id: notifications.append(task_id))

        result = json.loads(
            await handle_function_call("web_search", {"q": "test"}, task_id="t1")
        )
        assert result == {"error": "Blocked"}
        assert notifications == []

    @pytest.mark.asyncio
    async def test_invalid_hook_returns_do_not_block(self, monkeypatch):
        """Malformed hook returns should be ignored — tool executes normally."""
        async def fake_invoke_hook(hook_name, **kwargs):
            if hook_name == "pre_tool_call":
                return [
                    "block",
                    {"action": "block"},           # missing message
                    {"action": "deny", "message": "nope"},
                ]
            return []

        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_invoke_hook)
        async def dispatch(*_args, **_kwargs):
            return json.dumps({"ok": True})

        monkeypatch.setattr("model_tools.registry.dispatch", dispatch)

        result = json.loads(
            await handle_function_call("read_file", {"path": "test.txt"}, task_id="t1")
        )
        assert result == {"ok": True}


# =========================================================================
# Legacy toolset map
# =========================================================================

class TestLegacyToolsetMap:
    def test_expected_legacy_names(self):
        expected = [
            "web_tools", "terminal_tools", "vision_tools",
            "image_tools", "skills_tools", "browser_tools",
            "file_tools",
        ]
        for name in expected:
            assert name in _LEGACY_TOOLSET_MAP, f"Missing legacy toolset: {name}"



# =========================================================================
# Backward-compat wrappers
# =========================================================================

class TestBackwardCompat:
    @pytest.mark.asyncio
    async def test_lazy_discovery_updates_public_maps_in_place(self):
        import model_tools
        from tools.registry import registry

        tool_map = model_tools.TOOL_TO_TOOLSET_MAP
        requirements = model_tools.TOOLSET_REQUIREMENTS

        await model_tools.get_all_tool_names()

        assert model_tools.TOOL_TO_TOOLSET_MAP is tool_map
        assert model_tools.TOOLSET_REQUIREMENTS is requirements
        assert tool_map == registry.get_tool_to_toolset_map()
        assert requirements == registry.get_toolset_requirements()

    @pytest.mark.asyncio
    async def test_get_all_tool_names_returns_list(self):
        names = await get_all_tool_names()
        assert isinstance(names, list)
        assert len(names) > 0
        # Should contain well-known tools
        assert "web_search" in names
        assert "terminal" in names

    @pytest.mark.asyncio
    async def test_get_toolset_for_tool(self):
        result = await get_toolset_for_tool("web_search")
        assert result is not None
        assert isinstance(result, str)




# =========================================================================
# _coerce_number — inf / nan must fall through to the original string
# (regression: fix: eliminate duplicate checkpoint entries and JSON-unsafe coercion)
# =========================================================================

class TestCoerceNumberInfNan:
    """_coerce_number must honor its documented contract ("Returns original
    string on failure") for inf/nan inputs, because float('inf') and
    float('nan') are not JSON-compliant under strict serialization."""

    def test_inf_returns_original_string(self):
        from model_tools import _coerce_number
        assert _coerce_number("inf") == "inf"


    def test_nan_returns_original_string(self):
        from model_tools import _coerce_number
        assert _coerce_number("nan") == "nan"



    def test_normal_numbers_still_coerce(self):
        """Guard against over-correction — real numbers still coerce."""
        from model_tools import _coerce_number
        assert _coerce_number("42") == 42
        assert _coerce_number("3.14") == 3.14
        assert _coerce_number("1e3") == 1000

class TestDisabledToolsetsPostureToolset:
    """Regression test for #57315: disabling a posture toolset (`coding`,
    posture: True) must preserve the shared core tools it re-lists but does
    not own -- same non-core-delta subtraction as hermes-* bundles (#33924) --
    while atomic toolsets stay fully removable."""

    @pytest.mark.asyncio
    async def test_disabling_coding_preserves_core_but_atomic_disables_still_remove(self):
        from model_tools import get_tool_definitions

        # web_search is check_fn-gated (needs an API key); probe only the core
        # tools actually present in baseline so gating cannot mask the fix.
        core_probe = {"terminal", "read_file", "write_file", "web_search"}

        baseline = {
            t["function"]["name"]
            for t in await get_tool_definitions(quiet_mode=True)
        }
        present_core = core_probe & baseline
        # Sanity: at least some probed core tools are available in this env.
        assert present_core, "no probed core tools present in baseline"

        no_coding = {
            t["function"]["name"]
            for t in await get_tool_definitions(
                disabled_toolsets=["coding"], quiet_mode=True
            )
        }
        # Previously the full resolve_toolset("coding") subtraction stripped
        # these shared core tools, collapsing the schema to a handful (#57315).
        assert present_core <= no_coding, (
            f"Core tools stripped by disabling 'coding': {present_core - no_coding}"
        )

        # Atomic (non-posture) toolsets must still be fully removable.
        no_terminal = {
            t["function"]["name"]
            for t in await get_tool_definitions(
                disabled_toolsets=["terminal"], quiet_mode=True
            )
        }
        assert "terminal" not in no_terminal

        no_file = {
            t["function"]["name"]
            for t in await get_tool_definitions(
                disabled_toolsets=["file"], quiet_mode=True
            )
        }
        assert "write_file" not in no_file
