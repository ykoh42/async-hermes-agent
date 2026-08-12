"""Tests for the ContextEngine ABC and plugin slot."""

import json
import pytest
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock

from agent.context_engine import ContextEngine
from agent.context_compressor import ContextCompressor


# ---------------------------------------------------------------------------
# A minimal concrete engine for testing the ABC
# ---------------------------------------------------------------------------

class StubEngine(ContextEngine):
    """Minimal engine that satisfies the ABC without doing real work."""

    def __init__(self, context_length=200000, threshold_pct=0.50):
        self.context_length = context_length
        self.threshold_tokens = int(context_length * threshold_pct)
        self._compress_called = False
        self._tools_called = []

    @property
    def name(self) -> str:
        return "stub"

    def update_model(self, model="", context_length=0, base_url="", api_key="",
                     provider="", api_mode="", **kwargs) -> None:
        """Mirror ContextCompressor.update_model — recompute threshold from the
        new context_length. This is the mutation that corrupted the shared
        singleton in #42449."""
        self.context_length = context_length
        self.threshold_tokens = int(context_length * 0.20)

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        return tokens >= self.threshold_tokens

    async def compress(self, messages: List[Dict[str, Any]], current_tokens: int = None) -> List[Dict[str, Any]]:
        self._compress_called = True
        self.compression_count += 1
        # Trivial: just return as-is
        return messages

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "stub_search",
                "description": "Search the stub engine",
                "parameters": {"type": "object", "properties": {}},
            }
        ]

    async def handle_tool_call(self, name: str, args: Dict[str, Any]) -> str:
        self._tools_called.append(name)
        return json.dumps({"ok": True, "tool": name})


# ---------------------------------------------------------------------------
# ABC contract tests
# ---------------------------------------------------------------------------

class TestContextEngineABC:
    """Verify the ABC enforces the required interface."""


    def test_missing_methods_raises(self):
        """A subclass missing required methods cannot be instantiated."""
        class Incomplete(ContextEngine):
            @property
            def name(self):
                return "incomplete"
        with pytest.raises(TypeError):
            Incomplete()

    def test_stub_engine_satisfies_abc(self):
        engine = StubEngine()
        assert isinstance(engine, ContextEngine)
        assert engine.name == "stub"



# ---------------------------------------------------------------------------
# Default method behavior
# ---------------------------------------------------------------------------

class TestDefaults:
    """Verify ABC default implementations work correctly."""



    def test_default_get_status(self):
        engine = StubEngine()
        engine.last_prompt_tokens = 50000
        status = engine.get_status()
        assert status["last_prompt_tokens"] == 50000
        assert status["context_length"] == 200000
        assert status["threshold_tokens"] == 100000
        assert 0 < status["usage_percent"] <= 100


    @pytest.mark.asyncio
    async def test_on_session_reset(self):
        engine = StubEngine()
        engine.last_prompt_tokens = 999
        engine.compression_count = 3
        await engine.on_session_reset()
        assert engine.last_prompt_tokens == 0
        assert engine.compression_count == 0



# ---------------------------------------------------------------------------
# StubEngine behavior
# ---------------------------------------------------------------------------

class TestStubEngine:



    def test_tool_schemas(self):
        engine = StubEngine()
        schemas = engine.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "stub_search"

    @pytest.mark.asyncio
    async def test_handle_tool_call(self):
        engine = StubEngine()
        result = await engine.handle_tool_call("stub_search", {})
        assert json.loads(result)["ok"] is True
        assert "stub_search" in engine._tools_called




# ---------------------------------------------------------------------------
# ContextCompressor session reset via ABC
# ---------------------------------------------------------------------------

class TestCompressorSessionReset:
    """Verify ContextCompressor.on_session_reset() clears all state."""

    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        c = ContextCompressor(model="test", quiet_mode=True, config_context_length=200000)
        c.last_prompt_tokens = 50000
        c.compression_count = 3
        c._previous_summary = "some old summary"
        c._context_probed = True
        c._context_probe_persistable = True

        await c.on_session_reset()

        assert c.last_prompt_tokens == 0
        assert c.last_completion_tokens == 0
        assert c.last_total_tokens == 0
        assert c.compression_count == 0
        assert c._context_probed is False
        assert c._context_probe_persistable is False
        assert c._previous_summary is None


# ---------------------------------------------------------------------------
# Plugin slot (PluginManager integration)
# ---------------------------------------------------------------------------

class TestPluginContextEngineSlot:
    """Test register_context_engine on PluginContext."""

    def test_register_engine(self):
        from hermes_cli.plugins import PluginManager, PluginContext, PluginManifest
        mgr = PluginManager()
        manifest = PluginManifest(name="test-lcm")
        ctx = PluginContext(manifest, mgr)

        engine = StubEngine()
        ctx.register_context_engine(engine)

        assert mgr._context_engine is engine
        assert mgr._context_engine.name == "stub"



    @pytest.mark.asyncio
    async def test_get_plugin_context_engine(self):
        from hermes_cli.plugins import PluginManager, get_plugin_context_engine
        import hermes_cli.plugins as plugins_mod

        # Inject a test manager
        old_mgr = plugins_mod._plugin_manager
        try:
            mgr = PluginManager()
            plugins_mod._plugin_manager = mgr

            assert await get_plugin_context_engine() is None

            engine = StubEngine()
            mgr._context_engine = engine
            assert await get_plugin_context_engine() is engine
        finally:
            plugins_mod._plugin_manager = old_mgr



class TestPluginContextEngineDeepCopy:
    """Verify that the plugin context engine singleton is deep-copied before
    mutation in agent_init — regression test for #42449."""


    def test_deepcopy_preserves_engine_name(self):
        """Deep-copied engine retains its identity (name property)."""
        import copy
        engine = StubEngine(context_length=500000)
        clone = copy.deepcopy(engine)
        assert clone.name == engine.name == "stub"

    def test_deepcopy_preserves_compressor_state(self):
        """Deep-copied engine starts with the same token counters."""
        import copy
        engine = StubEngine(context_length=500000)
        engine.last_prompt_tokens = 1000
        engine.last_total_tokens = 1500
        engine.compression_count = 3

        clone = copy.deepcopy(engine)
        assert clone.last_prompt_tokens == 1000
        assert clone.last_total_tokens == 1500
        assert clone.compression_count == 3
        assert clone is not engine


class TestInitAgentDoesNotMutatePluginSingleton:
    @staticmethod
    def _agent(builtin):
        return SimpleNamespace(
            _context_engine_selected=False,
            context_compressor=builtin,
            _compression_threshold_autoraised="unchanged",
        )

    @pytest.mark.asyncio
    async def test_agent_init_source_deepcopies_singleton_not_aliases(
        self,
        monkeypatch,
    ):
        from agent.agent_init import _select_context_engine

        singleton = StubEngine(context_length=1_000_000, threshold_pct=0.20)
        agent = self._agent(ContextCompressor(model="test", quiet_mode=True))
        monkeypatch.setattr(
            "plugins.context_engine.load_context_engine",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.get_plugin_context_engine",
            AsyncMock(return_value=singleton),
        )

        await _select_context_engine(
            agent,
            {"context": {"engine": "stub"}},
        )

        assert agent.context_compressor is not singleton
        assert agent.context_compressor.name == singleton.name
        assert agent._context_engine_is_plugin is True

    @pytest.mark.asyncio
    async def test_child_init_does_not_corrupt_parent_singleton(
        self,
        monkeypatch,
    ):
        from agent.agent_init import _select_context_engine

        singleton = StubEngine(context_length=1_000_000, threshold_pct=0.20)
        agent = self._agent(ContextCompressor(model="test", quiet_mode=True))
        monkeypatch.setattr(
            "plugins.context_engine.load_context_engine",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.get_plugin_context_engine",
            AsyncMock(return_value=singleton),
        )

        await _select_context_engine(
            agent,
            {"context": {"engine": "stub"}},
        )
        agent.context_compressor.update_model(
            model="MiniMax-M2",
            context_length=204_800,
            provider="minimax",
        )

        assert singleton.context_length == 1_000_000
        assert singleton.threshold_tokens == 200_000
        assert agent.context_compressor.context_length == 204_800

    @pytest.mark.asyncio
    async def test_unpicklable_engine_falls_back_gracefully(
        self,
        monkeypatch,
        caplog,
    ):
        from agent.agent_init import _select_context_engine

        class _UncopyableEngine(StubEngine):
            def __deepcopy__(self, memo):
                raise RuntimeError("uncopyable state")

        singleton = _UncopyableEngine()
        builtin = ContextCompressor(model="test", quiet_mode=True)
        agent = self._agent(builtin)
        monkeypatch.setattr(
            "plugins.context_engine.load_context_engine",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.get_plugin_context_engine",
            AsyncMock(return_value=singleton),
        )

        with caplog.at_level("WARNING", logger="agent.agent_init"):
            await _select_context_engine(
                agent,
                {"context": {"engine": "stub"}},
            )

        assert agent.context_compressor is builtin
        assert agent._context_engine_is_plugin is False
        assert agent._context_engine_selected is True
        assert singleton.context_length == 200_000
        assert "could not be safely copied" in caplog.text
