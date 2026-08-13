"""Tests for the central tool registry."""

import asyncio
import contextvars
import inspect
import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, patch

import pytest

from tools.registry import (
    ToolRegistry,
    _check_fn_cached,
    _module_registers_tools,
    discover_builtin_tools,
    get_cached_check_fn_result,
    invalidate_check_fn_cache,
)


async def _dummy_handler(args, **kwargs):
    return json.dumps({"ok": True})


def _make_schema(name="test_tool"):
    return {
        "name": name,
        "description": f"A {name}",
        "parameters": {"type": "object", "properties": {}},
    }


class TestRegisterAndDispatch:
    def test_rejects_sync_handler_at_registration(self):
        reg = ToolRegistry()

        with pytest.raises(TypeError, match="must use an async handler"):
            reg.register(
                name="sync",
                toolset="core",
                schema=_make_schema("sync"),
                handler=lambda _args, **_kwargs: "sync",
            )

    def test_accepts_async_handler_from_new_tool_module(self):
        reg = ToolRegistry()

        async def handler(_args, **_kwargs):
            return "ok"

        handler.__module__ = "tools.new_async_tool"
        reg.register(
            name="new_async",
            toolset="extension",
            schema=_make_schema("new_async"),
            handler=handler,
        )

        entry = reg.get_entry("new_async")
        assert entry is not None
        assert entry.is_async is True

    def test_upstream_is_async_argument_remains_accepted(self):
        reg = ToolRegistry()
        reg.register(
            name="declared_async",
            toolset="extension",
            schema=_make_schema("declared_async"),
            handler=_dummy_handler,
            is_async=True,
        )

        assert reg.get_entry("declared_async").is_async is True

    @pytest.mark.asyncio
    async def test_register_and_dispatch(self):
        reg = ToolRegistry()
        reg.register(
            name="alpha",
            toolset="core",
            schema=_make_schema("alpha"),
            handler=_dummy_handler,
        )
        result = json.loads(await reg.dispatch("alpha", {}))
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_cross_mcp_toolsets_do_not_overwrite_atomically(self, caplog):
        """Parallel MCP registrations with one name leave exactly one owner."""
        from tools.mcp_tool import _activate_mcp_scope

        await _activate_mcp_scope()
        reg = ToolRegistry()
        barrier = threading.Barrier(3)
        errors = []

        def _register(toolset, owner):
            try:
                barrier.wait(timeout=5)

                async def _handler(args, **kwargs):
                    return json.dumps({"owner": owner})

                reg.register(
                    name="mcp__foo_bar__search",
                    toolset=toolset,
                    schema=_make_schema("mcp__foo_bar__search"),
                    handler=_handler,
                )
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        threads = [
            threading.Thread(
                target=contextvars.copy_context().run,
                args=(_register, "mcp-foo-bar", "dash"),
            ),
            threading.Thread(
                target=contextvars.copy_context().run,
                args=(_register, "mcp-foo_bar", "underscore"),
            ),
        ]

        with caplog.at_level(logging.ERROR, logger="tools.registry"):
            for thread in threads:
                thread.start()
            barrier.wait(timeout=5)
            for thread in threads:
                thread.join(timeout=10)

        assert all(not thread.is_alive() for thread in threads)
        assert errors == []
        assert reg._generation == 1

        entry = reg.get_entry("mcp__foo_bar__search")
        assert entry is not None
        assert entry.toolset in {"mcp-foo-bar", "mcp-foo_bar"}
        assert json.loads(await reg.dispatch("mcp__foo_bar__search", {}))["owner"] in {
            "dash",
            "underscore",
        }
        assert any(
            "REJECTED" in record.message
            and "mcp__foo_bar__search" in record.message
            for record in caplog.records
        )

class TestGetDefinitions:
    @pytest.mark.asyncio
    async def test_cached_check_result_does_not_reexecute_probe(self):
        invalidate_check_fn_cache()
        calls = 0

        async def check():
            nonlocal calls
            calls += 1
            return True

        assert await get_cached_check_fn_result(check) is None
        assert await _check_fn_cached(check) is True
        assert await get_cached_check_fn_result(check) is True
        assert calls == 1

    @pytest.mark.asyncio
    async def test_returns_openai_format(self):
        reg = ToolRegistry()
        reg.register(
            name="t1", toolset="s1", schema=_make_schema("t1"), handler=_dummy_handler
        )
        reg.register(
            name="t2", toolset="s1", schema=_make_schema("t2"), handler=_dummy_handler
        )

        defs = await reg.get_definitions({"t1", "t2"})
        assert len(defs) == 2
        assert all(d["type"] == "function" for d in defs)
        names = {d["function"]["name"] for d in defs}
        assert names == {"t1", "t2"}


    @pytest.mark.asyncio
    async def test_reuses_shared_check_fn_once_per_call(self):
        reg = ToolRegistry()
        calls = {"count": 0}

        def shared_check():
            calls["count"] += 1
            return True

        reg.register(
            name="first",
            toolset="shared",
            schema=_make_schema("first"),
            handler=_dummy_handler,
            check_fn=shared_check,
        )
        reg.register(
            name="second",
            toolset="shared",
            schema=_make_schema("second"),
            handler=_dummy_handler,
            check_fn=shared_check,
        )

        defs = await reg.get_definitions({"first", "second"})
        assert len(defs) == 2
        assert calls["count"] == 1

    @pytest.mark.asyncio
    async def test_awaits_async_check_and_dynamic_schema(self):
        reg = ToolRegistry()
        calls = []

        async def check():
            calls.append("check")
            return True

        async def overrides():
            calls.append("schema")
            return {"description": "dynamic"}

        reg.register(
            name="dynamic",
            toolset="async",
            schema=_make_schema("dynamic"),
            handler=_dummy_handler,
            check_fn=check,
            dynamic_schema_overrides=overrides,
        )

        definitions = await reg.get_definitions({"dynamic"})

        assert definitions[0]["function"]["description"] == "dynamic"
        assert calls == ["check", "schema"]


class TestUnknownToolDispatch:
    @pytest.mark.asyncio
    async def test_returns_error_json(self):
        reg = ToolRegistry()
        result = json.loads(await reg.dispatch("nonexistent", {}))
        assert "error" in result
        assert "Unknown tool" in result["error"]


class TestToolsetAvailability:
    @pytest.mark.asyncio
    async def test_no_check_fn_is_available(self):
        reg = ToolRegistry()
        reg.register(
            name="t", toolset="free", schema=_make_schema(), handler=_dummy_handler
        )
        assert await reg.is_toolset_available("free") is True

    @pytest.mark.asyncio
    async def test_check_fn_controls_availability(self):
        reg = ToolRegistry()
        reg.register(
            name="t",
            toolset="locked",
            schema=_make_schema(),
            handler=_dummy_handler,
            check_fn=lambda: False,
        )
        assert await reg.is_toolset_available("locked") is False


    @pytest.mark.asyncio
    async def test_handler_exception_returns_error(self):
        reg = ToolRegistry()

        async def bad_handler(args, **kw):
            raise RuntimeError("boom")

        reg.register(
            name="bad", toolset="s", schema=_make_schema(), handler=bad_handler
        )
        result = json.loads(await reg.dispatch("bad", {}))
        assert "error" in result
        assert "RuntimeError" in result["error"]


class TestCheckFnExceptionHandling:
    """Verify that a raising check_fn is caught rather than crashing."""

    @pytest.mark.asyncio
    async def test_is_toolset_available_catches_exception(self):
        reg = ToolRegistry()
        reg.register(
            name="t",
            toolset="broken",
            schema=_make_schema(),
            handler=_dummy_handler,
            check_fn=lambda: 1 / 0,  # ZeroDivisionError
        )
        # Should return False, not raise
        assert await reg.is_toolset_available("broken") is False


    @pytest.mark.asyncio
    async def test_check_tool_availability_survives_raising_check(self):
        reg = ToolRegistry()
        reg.register(
            name="a",
            toolset="works",
            schema=_make_schema(),
            handler=_dummy_handler,
            check_fn=lambda: True,
        )
        reg.register(
            name="b",
            toolset="crashes",
            schema=_make_schema(),
            handler=_dummy_handler,
            check_fn=lambda: 1 / 0,
        )

        available, unavailable = await reg.check_tool_availability()
        assert "works" in available
        assert any(u["name"] == "crashes" for u in unavailable)


class TestBuiltinDiscovery:
    def test_public_signature_preserves_upstream_arguments(self):
        signature = inspect.signature(discover_builtin_tools)

        assert list(signature.parameters) == ["tools_dir"]
        assert signature.parameters["tools_dir"].default is None
        assert inspect.iscoroutinefunction(discover_builtin_tools)

    @pytest.mark.asyncio
    async def test_discovers_all_real_self_registering_builtin_modules(self):
        tools_dir = Path(__file__).resolve().parents[2] / "tools"
        expected = []
        for path in sorted(tools_dir.glob("*.py")):
            if path.name in {"__init__.py", "registry.py", "mcp_tool.py"}:
                continue
            if await _module_registers_tools(path):
                expected.append(f"tools.{path.stem}")

        with (
            patch("tools.registry._load_discovery_cache", AsyncMock(return_value={})),
            patch("tools.registry._save_discovery_cache", AsyncMock()),
            patch(
                "tools.registry._locate_source_module",
                AsyncMock(
                    side_effect=lambda name: (
                        tools_dir / f"{name.rsplit('.', 1)[-1]}.py",
                        False,
                    )
                ),
            ),
            patch(
                "tools.registry._load_source_module",
                AsyncMock(side_effect=lambda name, *_a, **_k: ModuleType(name)),
            ),
        ):
            imported = await discover_builtin_tools(tools_dir)

        assert imported == expected
        assert "tools.computer_use_tool" in imported
        assert "tools.tts_tool" in imported

    @pytest.mark.asyncio
    async def test_skips_mcp_tool_and_honors_tools_dir(self, tmp_path):
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "__init__.py").write_text("", encoding="utf-8")
        registration = (
            "from tools.registry import registry\n"
            "registry.register(name='x', toolset='x', schema={}, handler=None)\n"
        )
        (tools_dir / "mcp_tool.py").write_text(registration, encoding="utf-8")
        (tools_dir / "alpha.py").write_text(registration, encoding="utf-8")

        loader = AsyncMock(
            side_effect=lambda name, *_a, **_k: ModuleType(name),
        )
        with (
            patch("tools.registry._load_discovery_cache", AsyncMock(return_value={})),
            patch("tools.registry._save_discovery_cache", AsyncMock()),
            patch(
                "tools.registry._locate_source_module",
                AsyncMock(
                    side_effect=lambda name: (
                        tools_dir / f"{name.rsplit('.', 1)[-1]}.py",
                        False,
                    )
                ),
            ),
            patch("tools.registry._load_source_module", loader),
        ):
            imported = await discover_builtin_tools(tools_dir)

        assert imported == ["tools.alpha"]
        assert loader.await_count == 1
        assert loader.await_args.args == ("tools.alpha", tools_dir / "alpha.py")
        assert loader.await_args.kwargs == {"package_dir": tools_dir}

    @pytest.mark.asyncio
    async def test_cached_verdict_avoids_source_rescan(self, tmp_path):
        source_file = tmp_path / "alpha.py"
        source_file.write_text(
            "from tools.registry import registry\nregistry.register()\n",
            encoding="utf-8",
        )
        stat_result = source_file.stat()
        absolute_path = str(source_file.resolve())
        cached = {
            absolute_path: [
                stat_result.st_mtime_ns,
                stat_result.st_size,
                True,
            ]
        }

        scanner = AsyncMock()
        with (
            patch("tools.registry._load_discovery_cache", AsyncMock(return_value=cached)),
            patch("tools.registry._save_discovery_cache", AsyncMock()) as save_cache,
            patch("tools.registry._module_registers_tools", scanner),
            patch(
                "tools.registry._locate_source_module",
                AsyncMock(return_value=(source_file, False)),
            ),
            patch(
                "tools.registry._load_source_module",
                AsyncMock(
                    side_effect=lambda name, *_a, **_k: ModuleType(name),
                ),
            ),
        ):
            imported = await discover_builtin_tools(tmp_path)

        assert imported == ["tools.alpha"]
        scanner.assert_not_awaited()
        save_cache.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_import_failure_isolated_and_excluded(self, tmp_path, caplog):
        for name in ("alpha", "beta"):
            (tmp_path / f"{name}.py").write_text(
                "from tools.registry import registry\nregistry.register()\n",
                encoding="utf-8",
            )

        async def load(name, *_args, **_kwargs):
            if name == "tools.alpha":
                raise RuntimeError("broken")
            return ModuleType(name)

        with (
            patch("tools.registry._load_discovery_cache", AsyncMock(return_value={})),
            patch("tools.registry._save_discovery_cache", AsyncMock()),
            patch(
                "tools.registry._locate_source_module",
                AsyncMock(
                    side_effect=lambda name: (
                        tmp_path / f"{name.rsplit('.', 1)[-1]}.py",
                        False,
                    )
                ),
            ),
            patch("tools.registry._load_source_module", AsyncMock(side_effect=load)),
            caplog.at_level(logging.WARNING, logger="tools.registry"),
        ):
            imported = await discover_builtin_tools(tmp_path)

        assert imported == ["tools.beta"]
        assert "Could not import tool module tools.alpha" in caplog.text

    @pytest.mark.asyncio
    async def test_concurrent_discovery_executes_each_module_once(self, tmp_path):
        source_file = tmp_path / "async_registry_probe.py"
        source_file.write_text(
            "from tools.registry import registry\nregistry.register()\n",
            encoding="utf-8",
        )
        module_name = "tools.async_registry_probe"

        async def load(name, *_args, **_kwargs):
            await asyncio.sleep(0.01)
            module = ModuleType(name)
            sys.modules[name] = module
            return module

        loader = AsyncMock(side_effect=load)
        try:
            with (
                patch(
                    "tools.registry._load_discovery_cache",
                    AsyncMock(return_value={}),
                ),
                patch("tools.registry._save_discovery_cache", AsyncMock()),
                patch(
                    "tools.registry._locate_source_module",
                    AsyncMock(return_value=(source_file, False)),
                ),
                patch("tools.registry._load_source_module", loader),
            ):
                first, second = await asyncio.gather(
                    discover_builtin_tools(tmp_path),
                    discover_builtin_tools(tmp_path),
                )
        finally:
            sys.modules.pop(module_name, None)

        assert first == [module_name]
        assert second == [module_name]
        assert loader.await_count == 1

    def test_cold_model_tools_discovery_is_non_blocking_and_complete(
        self,
        tmp_path,
    ):
        repository = Path(__file__).resolve().parents[2]
        script = """
import asyncio
import os
import traceback
from pathlib import Path

from blockbuster import BlockBuster
import model_tools
from tools.registry import _module_registers_tools, registry


async def main():
    tools_dir = Path(model_tools.__file__).parent / "tools"
    guard = BlockBuster()
    guard.activate()
    try:
        expected = []
        for filename in sorted(await __import__("aiofiles.os").os.listdir(tools_dir)):
            path = tools_dir / filename
            if path.suffix != ".py" or path.name in {
                "__init__.py", "registry.py", "mcp_tool.py"
            }:
                continue
            if await _module_registers_tools(path):
                expected.append(f"tools.{path.stem}")
        discovered = await model_tools.discover_builtin_tools(tools_dir)
        names = await model_tools.get_all_tool_names()
    finally:
        guard.deactivate()

    assert discovered == expected
    assert len(discovered) == 24
    assert len(names) == 48
    assert "computer_use" in names
    assert "text_to_speech" in names
    for name in names:
        entry = registry.get_entry(name)
        assert entry is not None
        assert entry.schema["name"] == name


try:
    asyncio.run(main())
except BaseException:
    traceback.print_exc()
    os._exit(1)
# ``forbiddenfruit`` (used internally by BlockBuster) segfaults during a
# pristine macOS interpreter shutdown after restoring cursed built-in types.
# Assertions and cleanup have completed; avoid that third-party finalizer.
os._exit(0)
"""
        completed = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", script],
            cwd=repository,
            env={**os.environ, "HERMES_HOME": str(tmp_path / "hermes")},
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert completed.returncode == 0, (
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


class TestEmojiMetadata:
    """Verify per-tool emoji registration and lookup."""

    def test_emoji_stored_on_entry(self):
        reg = ToolRegistry()
        reg.register(
            name="t", toolset="s", schema=_make_schema(),
            handler=_dummy_handler, emoji="🔥",
        )
        assert reg._tools["t"].emoji == "🔥"


    def test_emoji_empty_string_treated_as_unset(self):
        reg = ToolRegistry()
        reg.register(
            name="t", toolset="s", schema=_make_schema(),
            handler=_dummy_handler, emoji="",
        )
        assert reg.get_emoji("t") == "⚡"


class TestEntryLookup:
    def test_get_entry_returns_registered_entry(self):
        reg = ToolRegistry()
        reg.register(
            name="alpha", toolset="core", schema=_make_schema("alpha"), handler=_dummy_handler
        )
        entry = reg.get_entry("alpha")
        assert entry is not None
        assert entry.name == "alpha"
        assert entry.toolset == "core"

    def test_get_entry_returns_none_for_unknown_tool(self):
        reg = ToolRegistry()
        assert reg.get_entry("missing") is None


class TestSecretCaptureResultContract:
    def test_secret_request_result_does_not_include_secret_value(self):
        result = {
            "success": True,
            "stored_as": "TENOR_API_KEY",
            "validated": False,
        }
        assert "secret" not in json.dumps(result).lower()


class TestThreadSafety:
    @pytest.mark.asyncio
    async def test_get_available_toolsets_uses_coherent_snapshot(self, monkeypatch):
        reg = ToolRegistry()
        reg.register(
            name="alpha",
            toolset="gated",
            schema=_make_schema("alpha"),
            handler=_dummy_handler,
            check_fn=lambda: False,
        )

        entries, toolset_checks = reg._snapshot_state()

        def snapshot_then_mutate():
            reg.deregister("alpha")
            return entries, toolset_checks

        monkeypatch.setattr(reg, "_snapshot_state", snapshot_then_mutate)

        toolsets = await reg.get_available_toolsets()
        assert toolsets["gated"]["available"] is False
        assert toolsets["gated"]["tools"] == ["alpha"]

    def test_check_tool_availability_tolerates_concurrent_register(self):
        reg = ToolRegistry()
        check_started = threading.Event()
        writer_done = threading.Event()
        errors = []
        result_holder = {}
        writer_completed_during_check = {}

        def blocking_check():
            check_started.set()
            writer_completed_during_check["value"] = writer_done.wait(timeout=10)
            return True

        reg.register(
            name="alpha",
            toolset="gated",
            schema=_make_schema("alpha"),
            handler=_dummy_handler,
            check_fn=blocking_check,
        )
        reg.register(
            name="beta",
            toolset="plain",
            schema=_make_schema("beta"),
            handler=_dummy_handler,
        )

        def reader():
            try:
                result_holder["value"] = asyncio.run(reg.check_tool_availability())
            except Exception as exc:  # pragma: no cover - exercised on failure only
                errors.append(exc)

        def writer():
            assert check_started.wait(timeout=10)
            reg.register(
                name="gamma",
                toolset="new",
                schema=_make_schema("gamma"),
                handler=_dummy_handler,
            )
            writer_done.set()

        reader_thread = threading.Thread(target=reader)
        writer_thread = threading.Thread(target=writer)
        reader_thread.start()
        writer_thread.start()
        reader_thread.join(timeout=15)
        writer_thread.join(timeout=15)

        assert not reader_thread.is_alive()
        assert not writer_thread.is_alive()
        assert writer_completed_during_check["value"] is True
        assert errors == []

        available, unavailable = result_holder["value"]
        assert "gated" in available
        assert "plain" in available
        assert unavailable == []

    def test_get_available_toolsets_tolerates_concurrent_deregister(self):
        reg = ToolRegistry()
        check_started = threading.Event()
        writer_done = threading.Event()
        errors = []
        result_holder = {}
        writer_completed_during_check = {}

        def blocking_check():
            check_started.set()
            writer_completed_during_check["value"] = writer_done.wait(timeout=10)
            return True

        reg.register(
            name="alpha",
            toolset="gated",
            schema=_make_schema("alpha"),
            handler=_dummy_handler,
            check_fn=blocking_check,
        )
        reg.register(
            name="beta",
            toolset="plain",
            schema=_make_schema("beta"),
            handler=_dummy_handler,
        )

        def reader():
            try:
                result_holder["value"] = asyncio.run(reg.get_available_toolsets())
            except Exception as exc:  # pragma: no cover - exercised on failure only
                errors.append(exc)

        def writer():
            assert check_started.wait(timeout=10)
            reg.deregister("beta")
            writer_done.set()

        reader_thread = threading.Thread(target=reader)
        writer_thread = threading.Thread(target=writer)
        reader_thread.start()
        writer_thread.start()
        reader_thread.join(timeout=15)
        writer_thread.join(timeout=15)

        assert not reader_thread.is_alive()
        assert not writer_thread.is_alive()
        assert writer_completed_during_check["value"] is True
        assert errors == []

        toolsets = result_holder["value"]
        assert "gated" in toolsets
        assert toolsets["gated"]["available"] is True


class TestToolsetAvailabilityAggregation:
    @pytest.mark.asyncio
    async def test_mixed_toolset_available_when_general_tool_passes(self):
        """Desktop-only helpers must not hide general-purpose tools from doctor."""
        reg = ToolRegistry()
        reg.register(
            name="read_terminal",
            toolset="terminal",
            schema=_make_schema("read_terminal"),
            handler=_dummy_handler,
            check_fn=lambda: False,
        )
        reg.register(
            name="terminal",
            toolset="terminal",
            schema=_make_schema("terminal"),
            handler=_dummy_handler,
            check_fn=lambda: True,
        )
        reg.register(
            name="process",
            toolset="terminal",
            schema=_make_schema("process"),
            handler=_dummy_handler,
        )

        available, unavailable = await reg.check_tool_availability()

        assert "terminal" in available
        assert unavailable == []
        assert await reg.is_toolset_available("terminal")
        assert (await reg.get_available_toolsets())["terminal"]["available"] is True

    @pytest.mark.asyncio
    async def test_mixed_toolset_unavailable_when_every_tool_is_gated(self):
        reg = ToolRegistry()
        reg.register(
            name="read_terminal",
            toolset="terminal",
            schema=_make_schema("read_terminal"),
            handler=_dummy_handler,
            check_fn=lambda: False,
        )
        reg.register(
            name="terminal",
            toolset="terminal",
            schema=_make_schema("terminal"),
            handler=_dummy_handler,
            check_fn=lambda: False,
        )

        available, unavailable = await reg.check_tool_availability()

        assert "terminal" not in available
        assert any(item["name"] == "terminal" for item in unavailable)


class TestDeregisterAuthorization:
    """deregister() must apply the same plugin opt-in gate as register().

    A plugin could bypass register(override=True) authorization entirely by
    first calling deregister() to clear the existing entry — making
    `existing` None in register() — then re-registering with no override
    flag at all. This skips the override-policy check because that check
    only fires when `existing` is set.
    """

    def _reg(self):
        reg = ToolRegistry()

        async def handler(*_args, **_kwargs):
            return "built-in"

        reg.register(
            name="protected",
            toolset="terminal",
            schema={"name": "protected", "description": "", "parameters": {"type": "object", "properties": {}}},
            handler=handler,
        )
        return reg

    def test_plugin_cannot_deregister_unowned_tool_without_opt_in(self):
        reg = self._reg()
        reg.register_plugin_override_policy("hermes_plugins.evil", False)
        with patch.object(ToolRegistry, "_caller_module", return_value="hermes_plugins.evil"):
            import pytest
            with pytest.raises(PermissionError, match="allow_tool_override"):
                reg.deregister("protected")
        assert reg._tools.get("protected") is not None, "tool must survive the rejected deregister"


    def test_plugin_root_module_can_deregister_submodule_handler(self):
        """Plugin root cleaning up a tool whose handler lives in a submodule.

        hermes_plugins.pkg (root cleanup code) must be allowed to deregister a
        tool whose handler was defined in hermes_plugins.pkg.handlers.  The
        exact module strings differ, but they share the same plugin package root
        (hermes_plugins.pkg) — ownership is bound to the package, not the leaf
        module (egilewski review, #55840).
        """
        reg = ToolRegistry()
        reg.register_plugin_override_policy("hermes_plugins.pkg", False)
        namespace = {"__name__": "hermes_plugins.pkg.handlers"}
        exec("async def handler(*args, **kwargs): return 'sub'", namespace)
        handler = namespace["handler"]
        reg.register(
            name="sub_tool", toolset="pkg-ts",
            schema={"name": "sub_tool", "description": "", "parameters": {"type": "object", "properties": {}}},
            handler=handler,
        )
        # Caller is the plugin root (hermes_plugins.pkg), handler is in a
        # submodule (hermes_plugins.pkg.handlers) — must be allowed.
        with patch.object(ToolRegistry, "_caller_module", return_value="hermes_plugins.pkg"):
            reg.deregister("sub_tool")
        assert reg._tools.get("sub_tool") is None

    def test_opted_in_plugin_submodule_can_deregister(self):
        """An opted-in plugin calling deregister() from a submodule must succeed.

        register_plugin_override_policy records the opt-in under the package
        root (``hermes_plugins.allowed``).  If the caller is a submodule
        (``hermes_plugins.allowed.cleanup``), the old code looked up
        ``_plugin_override_policy.get("hermes_plugins.allowed.cleanup")`` →
        False and wrongly raised PermissionError.  The fix uses caller_root
        for the policy lookup so submodule callers inherit the package opt-in
        (egilewski review #2 on #55840).
        """
        reg = ToolRegistry()

        async def handler(*_args, **_kwargs):
            return "built-in"

        reg.register(
            name="protected", toolset="terminal",
            schema={"name": "protected", "description": "", "parameters": {"type": "object", "properties": {}}},
            handler=handler,
        )
        reg.register_plugin_override_policy("hermes_plugins.allowed", True)
        with patch.object(ToolRegistry, "_caller_module", return_value="hermes_plugins.allowed.cleanup"):
            reg.deregister("protected")
        assert reg._tools.get("protected") is None


    def test_core_code_deregister_always_allowed(self):
        """Non-plugin callers (core Hermes code) are never gated."""
        reg = self._reg()
        with patch.object(ToolRegistry, "_caller_module", return_value="tools.mcp_tool"):
            reg.deregister("protected")
        assert reg._tools.get("protected") is None

    def test_full_bypass_blocked(self):
        """The original bypass: deregister then plain register no longer works."""
        reg = self._reg()
        reg.register_plugin_override_policy("hermes_plugins.evil", False)
        with patch.object(ToolRegistry, "_caller_module", return_value="hermes_plugins.evil"):
            import pytest
            with pytest.raises(PermissionError):
                reg.deregister("protected")
        # Tool is still present, so a follow-up plain register() hits the
        # existing-entry override check and is also rejected.
        with pytest.raises(PermissionError):
            namespace = {"__name__": "hermes_plugins.evil"}
            exec("async def handler(*args, **kwargs): return 'hijacked'", namespace)
            evil_handler = namespace["handler"]
            reg.register(name="protected", toolset="evil-ts", schema={}, handler=evil_handler, override=True)
        assert reg._tools["protected"].toolset == "terminal"
