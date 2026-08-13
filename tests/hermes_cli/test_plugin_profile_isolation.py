from __future__ import annotations

import asyncio
import gc
import sys
import weakref
from pathlib import Path

import pytest
import yaml

import hermes_cli.plugins as plugins
from agent.image_gen_provider import ImageGenProvider
from agent.image_gen_registry import (
    _plugin_providers as plugin_image_providers,
    _providers as shared_image_providers,
    get_provider as get_image_provider,
    register_provider as register_image_provider,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from agent.browser_provider import BrowserProvider
from agent import browser_registry, tts_registry, video_gen_registry, web_search_registry
from agent.tts_provider import TTSProvider
from agent.video_gen_provider import VideoGenProvider
from agent.web_search_provider import WebSearchProvider
from tools.registry import registry


class _SharedImageProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "profile-probe-provider"

    async def generate(self, prompt, aspect_ratio="landscape", **kwargs):
        return {"marker": "shared"}


class _VideoProvider(VideoGenProvider):
    def __init__(self, marker: str):
        self.marker = marker

    @property
    def name(self) -> str:
        return "profile-isolation-video"

    async def generate(self, prompt, **kwargs):
        return {"marker": self.marker}


class _WebProvider(WebSearchProvider):
    def __init__(self, marker: str):
        self.marker = marker

    @property
    def name(self) -> str:
        return "profile-isolation-web"

    async def is_available(self) -> bool:
        return True


class _BrowserProvider(BrowserProvider):
    def __init__(self, marker: str):
        self.marker = marker

    @property
    def name(self) -> str:
        return "profile-isolation-browser"

    async def is_available(self) -> bool:
        return True

    async def create_session(self, task_id: str):
        return {"marker": self.marker}

    async def close_session(self, session_id: str) -> bool:
        return True

    async def emergency_cleanup(self, session_id: str) -> None:
        return None


class _TTSProvider(TTSProvider):
    def __init__(self, marker: str):
        self.marker = marker

    @property
    def name(self) -> str:
        return "profile-isolation-tts"

    async def synthesize(self, text, output_path, **kwargs):
        return output_path


def _write_plugin(home: Path, *, marker: str, extra: bool = False) -> None:
    plugin_dir = home / "plugins" / "profile-probe"
    plugin_dir.mkdir(parents=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["profile-probe"]}})
    )
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "profile-probe", "kind": "standalone"})
    )
    extra_registration = ""
    if extra:
        extra_registration = (
            "    ctx.register_tool(name='profile_only_tool', "
            "toolset='profile_only', schema={'name': 'profile_only_tool', "
            "'description': 'only here', 'parameters': {'type': 'object', "
            "'properties': {}}}, handler=profile_only_tool)\n"
        )
    (plugin_dir / "__init__.py").write_text(
        "from agent.image_gen_provider import ImageGenProvider\n"
        "from agent.context_engine import ContextEngine\n"
        "from functools import partial\n"
        "from tools.registry import registry\n"
        f"MARKER = {marker!r}\n"
        "class ProbeProvider(ImageGenProvider):\n"
        "    @property\n"
        "    def name(self): return 'profile-probe-provider'\n"
        "    async def generate(self, prompt, aspect_ratio='landscape', "
        "**kwargs): return {'marker': MARKER}\n"
        "class LateProvider(ProbeProvider):\n"
        "    @property\n"
        "    def name(self): return 'late-profile-provider'\n"
        "class ProbeEngine(ContextEngine):\n"
        "    marker = MARKER\n"
        "    @property\n"
        "    def name(self): return 'profile-probe-engine'\n"
        "    def update_from_response(self, usage): pass\n"
        "    def should_compress(self, prompt_tokens=None): return False\n"
        "    async def compress(self, messages, **kwargs): return messages\n"
        "async def profile_probe_tool(args, **kwargs): return MARKER\n"
        "async def profile_only_tool(args, **kwargs): return MARKER\n"
        "async def profile_probe_hook(**kwargs): return MARKER\n"
        "async def profile_probe_middleware(**kwargs): return MARKER\n"
        "def late_register():\n"
        "    from agent.image_gen_registry import register_provider\n"
        "    register_provider(LateProvider())\n"
        "    registry.register_toolset_alias('late-profile-alias', "
        "'profile_probe')\n"
        "def register(ctx):\n"
        "    ctx.register_tool(name='profile_probe_tool', "
        "toolset='profile_probe', schema={'name': 'profile_probe_tool', "
        "'description': 'probe', 'parameters': {'type': 'object', "
        "'properties': {}}}, handler=partial(profile_probe_tool))\n"
        "    registry.register_toolset_alias('profile-probe-alias', "
        "'profile_probe')\n"
        "    ctx.register_hook('pre_llm_call', profile_probe_hook)\n"
        "    ctx.register_middleware('llm_request', profile_probe_middleware)\n"
        "    ctx.register_context_engine(ProbeEngine())\n"
        "    ctx.register_auxiliary_task(key='profile_probe_aux', "
        "display_name='Profile probe', description=MARKER)\n"
        "    ctx.register_image_gen_provider(ProbeProvider())\n"
        f"{extra_registration}"
    )


async def _read_profile(home: Path) -> dict[str, object]:
    token = set_hermes_home_override(home)
    try:
        await plugins.discover_plugins()
        manager = plugins.get_plugin_manager()
        provider = get_image_provider("profile-probe-provider")
        module = manager._plugins["profile-probe"].module
        module.late_register()
        late_provider = get_image_provider("late-profile-provider")
        return {
            "manager": manager,
            "module": module,
            "module_name": module.__name__,
            "tool": await registry.dispatch("profile_probe_tool", {}),
            "only_tool": registry.get_entry("profile_only_tool"),
            "hook": await plugins.invoke_hook("pre_llm_call"),
            "middleware": await plugins.invoke_middleware("llm_request"),
            "alias": registry.get_toolset_alias_target("profile-probe-alias"),
            "late_alias": registry.get_toolset_alias_target("late-profile-alias"),
            "provider_marker": getattr(provider, "__class__").__module__,
            "provider": provider,
            "late_provider": late_provider,
            "context_engine": manager._context_engine,
            "aux": manager._aux_tasks["profile_probe_aux"],
        }
    finally:
        reset_hermes_home_override(token)


@pytest.fixture(autouse=True)
def _clean_plugin_profiles(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    bundled.mkdir()

    async def get_bundled_plugins_dir() -> Path:  # noqa: ASYNC124
        return bundled

    monkeypatch.setattr(plugins, "get_bundled_plugins_dir", get_bundled_plugins_dir)
    shared_image_providers.pop("profile-probe-provider", None)
    plugins._reset_plugin_profiles_for_tests()
    yield
    plugins._reset_plugin_profiles_for_tests()
    shared_image_providers.pop("profile-probe-provider", None)


@pytest.mark.asyncio
async def test_user_plugins_are_isolated_sequentially_and_concurrently(tmp_path):
    home_a = tmp_path / "a"
    home_b = tmp_path / "b"
    _write_plugin(home_a, marker="A", extra=True)
    _write_plugin(home_b, marker="B")

    shared_provider = _SharedImageProvider()
    register_image_provider(shared_provider)
    sequential_a = await _read_profile(home_a)
    sequential_b = await _read_profile(home_b)

    assert sequential_a["tool"] == "A"
    assert sequential_b["tool"] == "B"
    assert sequential_a["only_tool"] is not None
    assert sequential_b["only_tool"] is None
    assert sequential_a["hook"] == ["A"]
    assert sequential_b["hook"] == ["B"]
    assert sequential_a["middleware"] == ["A"]
    assert sequential_b["middleware"] == ["B"]
    assert sequential_a["context_engine"].marker == "A"
    assert sequential_b["context_engine"].marker == "B"
    assert sequential_a["aux"]["description"] == "A"
    assert sequential_b["aux"]["description"] == "B"
    assert sequential_a["alias"] == sequential_b["alias"] == "profile_probe"
    assert sequential_a["late_alias"] == sequential_b["late_alias"] == "profile_probe"
    assert sequential_a["manager"] is not sequential_b["manager"]
    assert sequential_a["module_name"] != sequential_b["module_name"]
    assert sequential_a["provider"] is not sequential_b["provider"]
    assert sequential_a["late_provider"] is not sequential_b["late_provider"]
    assert sequential_a["provider"] is not shared_provider
    assert sequential_b["provider"] is not shared_provider
    assert sequential_a["provider_marker"] != sequential_b["provider_marker"]

    concurrent_a, concurrent_b = await asyncio.gather(
        _read_profile(home_a),
        _read_profile(home_b),
    )
    assert concurrent_a["tool"] == "A"
    assert concurrent_b["tool"] == "B"
    assert concurrent_a["hook"] == ["A"]
    assert concurrent_b["hook"] == ["B"]
    assert concurrent_a["middleware"] == ["A"]
    assert concurrent_b["middleware"] == ["B"]

    empty_home = tmp_path / "empty"
    empty_home.mkdir()
    token = set_hermes_home_override(empty_home)
    try:
        await plugins.discover_plugins()
        assert get_image_provider("profile-probe-provider") is shared_provider
        assert registry.get_entry("profile_probe_tool") is None
    finally:
        reset_hermes_home_override(token)


def test_all_plugin_provider_registries_use_active_profile_overlay():
    manager_a = plugins.PluginManager()
    manager_b = plugins.PluginManager()
    for manager, marker in ((manager_a, "A"), (manager_b, "B")):
        context = plugins.PluginContext(
            plugins.PluginManifest(
                name="delayed-context",
                source="user",
            ),
            manager,
        )
        context.register_video_gen_provider(_VideoProvider(marker))
        context.register_web_search_provider(_WebProvider(marker))
        context.register_browser_provider(_BrowserProvider(marker))
        context.register_tts_provider(_TTSProvider(marker))

    for manager, marker in ((manager_a, "A"), (manager_b, "B")):
        token = plugins._ACTIVE_PLUGIN_MANAGER.set(manager)
        try:
            assert video_gen_registry.get_provider(
                "profile-isolation-video"
            ).marker == marker
            assert web_search_registry.get_provider(
                "profile-isolation-web"
            ).marker == marker
            assert browser_registry.get_provider(
                "profile-isolation-browser"
            ).marker == marker
            assert tts_registry.get_provider(
                "profile-isolation-tts"
            ).marker == marker
        finally:
            plugins._ACTIVE_PLUGIN_MANAGER.reset(token)

    manager_a._clear_owned_state()
    manager_b._clear_owned_state()


def test_user_plugins_are_isolated_across_sequential_event_loops(tmp_path):
    home_a = tmp_path / "loop-a"
    home_b = tmp_path / "loop-b"
    _write_plugin(home_a, marker="loop-A")
    _write_plugin(home_b, marker="loop-B")

    result_a = asyncio.run(_read_profile(home_a))
    result_b = asyncio.run(_read_profile(home_b))

    assert result_a["tool"] == "loop-A"
    assert result_b["tool"] == "loop-B"
    assert result_a["manager"] is not result_b["manager"]
    assert result_a["module_name"] != result_b["module_name"]


def test_contended_discovery_does_not_retain_closed_event_loop(tmp_path):
    home = tmp_path / "collectable-loop"
    _write_plugin(home, marker="collectable")

    async def load_twice():
        return await asyncio.gather(_read_profile(home), _read_profile(home))

    loop = asyncio.new_event_loop()
    loop_ref = weakref.ref(loop)
    first, second = loop.run_until_complete(load_twice())
    manager = first["manager"]
    assert manager is second["manager"]
    module_name = first["module_name"]
    loop.close()
    del loop
    for _ in range(3):
        gc.collect()

    assert loop_ref() is None
    assert module_name not in sys.modules
    assert manager not in registry._plugin_tools


@pytest.mark.asyncio
async def test_cancelled_discovery_rolls_back_modules_and_registries(tmp_path):
    home = tmp_path / "cancelled"
    plugin_dir = home / "plugins" / "profile-probe"
    plugin_dir.mkdir(parents=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["profile-probe"]}})
    )
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "profile-probe", "kind": "standalone"})
    )
    (plugin_dir / "__init__.py").write_text(
        "import asyncio\n"
        "from agent.image_gen_provider import ImageGenProvider\n"
        "STARTED = asyncio.Event()\n"
        "class CancelledProvider(ImageGenProvider):\n"
        "    @property\n"
        "    def name(self): return 'cancelled-plugin-provider'\n"
        "    async def generate(self, prompt, aspect_ratio='landscape', "
        "**kwargs): return {}\n"
        "async def cancelled_tool(args, **kwargs): return 'leaked'\n"
        "async def register(ctx):\n"
        "    ctx.register_tool(name='cancelled_plugin_tool', "
        "toolset='cancelled_plugin', schema={'name': "
        "'cancelled_plugin_tool', 'description': 'probe', 'parameters': "
        "{'type': 'object', 'properties': {}}}, handler=cancelled_tool)\n"
        "    ctx.register_image_gen_provider(CancelledProvider())\n"
        "    STARTED.set()\n"
        "    await asyncio.Event().wait()\n"
    )

    async def discover() -> None:
        token = set_hermes_home_override(home)
        try:
            await plugins.discover_plugins()
        finally:
            reset_hermes_home_override(token)

    task = asyncio.create_task(discover())
    module = None
    async with asyncio.timeout(2):
        while module is None:
            await asyncio.sleep(0.01)
            matches = [
                value
                for name, value in sys.modules.items()
                if name.startswith("hermes_plugins.")
                and getattr(value, "__file__", None)
                == str(plugin_dir / "__init__.py")
            ]
            started = getattr(matches[0], "STARTED", None) if matches else None
            if started is not None and started.is_set():
                module = matches[0]
    assert module is not None

    task.cancel()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not any(registry._plugin_tools.values())
    assert not any(registry._plugin_toolset_aliases.values())
    assert not any(plugin_image_providers.values())
    assert module.__name__ not in sys.modules
    assert not any(
        name.startswith(f"{module.__name__}.") for name in sys.modules
    )
    with pytest.raises(RuntimeError, match="no longer attached"):
        register_image_provider(module.CancelledProvider())
    with pytest.raises(RuntimeError, match="no longer attached"):
        registry.register(
            name="cancelled_plugin_tool",
            toolset="cancelled_plugin",
            schema={
                "name": "cancelled_plugin_tool",
                "description": "stale",
                "parameters": {"type": "object", "properties": {}},
            },
            handler=module.cancelled_tool,
        )
