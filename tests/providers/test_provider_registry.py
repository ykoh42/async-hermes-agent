import asyncio
import sys

import pytest
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

from providers import ProviderProfile
import providers


@pytest.fixture(autouse=True)
def isolate_provider_registry():
    registry = providers._REGISTRY.copy()
    aliases = providers._ALIASES.copy()
    provider_list_cache = (
        None
        if providers._PROVIDER_LIST_CACHE is None
        else list(providers._PROVIDER_LIST_CACHE)
    )
    discovered = providers._discovered

    yield

    providers._REGISTRY.clear()
    providers._REGISTRY.update(registry)
    providers._ALIASES.clear()
    providers._ALIASES.update(aliases)
    providers._PROVIDER_LIST_CACHE = provider_list_cache
    providers._discovered = discovered


def _profile(name: str, *aliases: str) -> ProviderProfile:
    return ProviderProfile(name=name, aliases=aliases)


def _reset_registry() -> None:
    providers._REGISTRY.clear()
    providers._ALIASES.clear()
    providers._PROVIDER_LIST_CACHE = None
    providers._discovered = True


@pytest.mark.asyncio
async def test_list_providers_reuses_cached_snapshot_until_registration_changes():
    _reset_registry()
    first = _profile("alpha")
    providers.register_provider(first)

    listed = await providers.list_providers()
    listed.clear()

    assert await providers.list_providers() == [first]

    # Hit-path copy guard: mutating a CACHED return must not corrupt the
    # module-level snapshot for later callers (aliasing bug class).
    (await providers.list_providers()).clear()
    assert await providers.list_providers() == [first]

    second = _profile("beta")
    providers.register_provider(second)

    assert await providers.list_providers() == [first, second]


@pytest.mark.asyncio
async def test_list_providers_dedupes_aliases_in_cached_snapshot():
    _reset_registry()
    profile = _profile("kimi", "moonshot", "kimi-k2")
    providers.register_provider(profile)

    assert await providers.get_provider_profile("moonshot") is profile
    assert await providers.list_providers() == [profile]


@pytest.mark.asyncio
async def test_public_lookup_discovers_profiles_through_awaited_boundary():
    """The public getter performs first-use discovery as a coroutine."""
    providers._REGISTRY.clear()
    providers._ALIASES.clear()
    providers._PROVIDER_LIST_CACHE = None
    providers._discovered = False
    for module_name in tuple(sys.modules):
        if module_name.startswith(("plugins.model_providers", "_hermes_user_provider")):
            sys.modules.pop(module_name, None)

    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        assert await providers.get_provider_profile("openrouter") is not None


@pytest.mark.asyncio
async def test_cancelled_discovery_rolls_back_partial_registry(monkeypatch):
    _reset_registry()
    providers._discovered = False
    providers._ASYNC_DISCOVERY_LOCK = None
    providers._ASYNC_DISCOVERY_LOOP = None
    started = asyncio.Event()

    async def interrupted_discovery():
        providers.register_provider(_profile("partial"))
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(
        providers,
        "_discover_providers_async_impl",
        interrupted_discovery,
    )
    task = asyncio.create_task(providers.get_provider_profile("partial"))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert providers._REGISTRY == {}
    assert providers._ALIASES == {}
    assert providers._discovered is False


@pytest.mark.asyncio
async def test_concurrent_first_lookups_share_one_discovery(monkeypatch):
    _reset_registry()
    providers._discovered = False
    providers._ASYNC_DISCOVERY_LOCK = None
    providers._ASYNC_DISCOVERY_LOOP = None
    calls = 0

    async def discover_once():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        providers.register_provider(_profile("shared"))
        providers._discovered = True

    monkeypatch.setattr(providers, "_discover_providers_async", discover_once)
    first, second = await asyncio.gather(
        providers.get_provider_profile("shared"),
        providers.get_provider_profile("shared"),
    )

    assert first is second
    assert first is not None
    assert calls == 1


@pytest.mark.asyncio
async def test_legacy_provider_relative_imports_use_async_source_loader(
    tmp_path, monkeypatch
):
    """Legacy ``providers/foo.py`` files must not read relatives synchronously."""
    from importlib._bootstrap_external import SourceFileLoader

    (tmp_path / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "legacy_helper.py").write_text(
        "MODEL = 'legacy-model'\n", encoding="utf-8"
    )
    (tmp_path / "legacy.py").write_text(
        "from .legacy_helper import MODEL\n"
        "from providers import ProviderProfile, register_provider\n"
        "register_provider(ProviderProfile(name='legacy', default_aux_model=MODEL))\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(providers, "_BUNDLED_PLUGINS_DIR", tmp_path / "missing")
    monkeypatch.setattr(providers, "_PROVIDERS_DIR", tmp_path)
    providers._REGISTRY.clear()
    providers._ALIASES.clear()
    providers._PROVIDER_LIST_CACHE = None
    providers._discovered = False

    original_get_data = SourceFileLoader.get_data

    def reject_sync_reads(loader, path):
        if str(path).startswith(str(tmp_path)):
            raise AssertionError(f"synchronous provider read: {path}")
        return original_get_data(loader, path)

    monkeypatch.setattr(SourceFileLoader, "get_data", reject_sync_reads)
    try:
        await providers._ensure_provider_profiles_loaded()
        profile = await providers.get_provider_profile("legacy")
        assert profile is not None
        assert profile.default_aux_model == "legacy-model"
    finally:
        for module_name in tuple(sys.modules):
            if module_name.startswith("providers.legacy"):
                module = sys.modules.pop(module_name)
                finder = getattr(module, "__hermes_async_source_finder__", None)
                if finder is not None and finder in sys.meta_path:
                    sys.meta_path.remove(finder)
