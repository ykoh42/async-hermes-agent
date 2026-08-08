import sys

import pytest

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


def test_list_providers_reuses_cached_snapshot_until_registration_changes():
    _reset_registry()
    first = _profile("alpha")
    providers.register_provider(first)

    listed = providers.list_providers()
    listed.clear()

    assert providers.list_providers() == [first]

    # Hit-path copy guard: mutating a CACHED return must not corrupt the
    # module-level snapshot for later callers (aliasing bug class).
    providers.list_providers().clear()
    assert providers.list_providers() == [first]

    second = _profile("beta")
    providers.register_provider(second)

    assert providers.list_providers() == [first, second]


def test_list_providers_dedupes_aliases_in_cached_snapshot():
    _reset_registry()
    profile = _profile("kimi", "moonshot", "kimi-k2")
    providers.register_provider(profile)

    assert providers.get_provider_profile("moonshot") is profile
    assert providers.list_providers() == [profile]


@pytest.mark.asyncio
async def test_sync_lookup_does_not_scan_provider_files_inside_event_loop():
    """An uninitialised sync getter must fail instead of blocking the loop."""
    providers._REGISTRY.clear()
    providers._ALIASES.clear()
    providers._PROVIDER_LIST_CACHE = None
    providers._discovered = False
    for module_name in tuple(sys.modules):
        if module_name.startswith(("plugins.model_providers", "_hermes_user_provider")):
            sys.modules.pop(module_name, None)

    with pytest.raises(RuntimeError, match="await the agent runtime boundary"):
        providers.get_provider_profile("openrouter")

    await providers._ensure_provider_profiles_loaded()
    assert providers.get_provider_profile("openrouter") is not None


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
        profile = providers.get_provider_profile("legacy")
        assert profile is not None
        assert profile.default_aux_model == "legacy-model"
    finally:
        for module_name in tuple(sys.modules):
            if module_name.startswith("providers.legacy"):
                module = sys.modules.pop(module_name)
                finder = getattr(module, "__hermes_async_source_finder__", None)
                if finder is not None and finder in sys.meta_path:
                    sys.meta_path.remove(finder)
