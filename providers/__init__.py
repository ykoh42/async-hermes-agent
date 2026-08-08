"""Provider module registry.

Provider profiles can live in two places:

1. Bundled plugins: ``plugins/model-providers/<name>/`` (shipped with hermes-agent)
2. User plugins: ``$HERMES_HOME/plugins/model-providers/<name>/``

Each plugin directory contains:
  - ``__init__.py`` — calls ``register_provider(profile)`` at import
  - ``plugin.yaml`` — manifest (name, kind: model-provider, version, description)

Discovery is lazy: the first awaited call to ``get_provider_profile()`` or
``list_providers()`` scans both locations and imports every plugin through
native async file I/O. User
plugins override bundled plugins on name collision (last-writer-wins), so
third parties can monkey-patch or replace any built-in profile without
editing the repo.

For backward compatibility, ``providers/*.py`` files (other than ``base.py``
and ``__init__.py``) are still discovered via ``pkgutil.iter_modules``.
This lets out-of-tree users drop a single-file profile into an editable
install without the plugin dir structure. New profiles should prefer the
plugin layout.

Usage::

    from providers import get_provider_profile
    profile = await get_provider_profile("nvidia")   # ProviderProfile or None
    profile = await get_provider_profile("kimi")     # checks name + aliases
"""

from __future__ import annotations

import asyncio
import logging
import sys
import types
from pathlib import Path

import aiofiles.os

from providers.base import OMIT_TEMPERATURE, ProviderProfile  # noqa: F401
from hermes_cli.async_source_loader import (
    load_source_module,
    load_source_package,
    unload_source_finder,
)

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, ProviderProfile] = {}
_ALIASES: dict[str, str] = {}
_PROVIDER_LIST_CACHE: list[ProviderProfile] | None = None
_discovered = False
_ASYNC_DISCOVERY_LOCK: asyncio.Lock | None = None
_ASYNC_DISCOVERY_LOOP: asyncio.AbstractEventLoop | None = None

# Repo-root ``plugins/model-providers/`` — populated at discovery time.
_BUNDLED_PLUGINS_DIR = (
    Path(__file__).resolve().parent.parent / "plugins" / "model-providers"
)
_PROVIDERS_DIR = Path(__file__).resolve().parent


def register_provider(profile: ProviderProfile) -> None:
    """Register a provider profile by name and aliases.

    Later registrations with the same name replace earlier ones — so user
    plugins under ``$HERMES_HOME/plugins/model-providers/`` can override
    bundled profiles without editing repo code.
    """
    global _PROVIDER_LIST_CACHE
    _REGISTRY[profile.name] = profile
    for alias in profile.aliases:
        _ALIASES[alias] = profile.name
    _PROVIDER_LIST_CACHE = None


def _get_provider_profile_cached(name: str) -> ProviderProfile | None:
    """Look up one already-discovered profile without performing I/O."""
    canonical = _ALIASES.get(name, name)
    return _REGISTRY.get(canonical)


def _list_providers_cached() -> list[ProviderProfile]:
    """Return an in-memory snapshot of already-discovered profiles."""
    global _PROVIDER_LIST_CACHE
    if _PROVIDER_LIST_CACHE is not None:
        return list(_PROVIDER_LIST_CACHE)
    # Deduplicate: _REGISTRY has canonical names; _ALIASES points to same objects
    seen: set[int] = set()
    result: list[ProviderProfile] = []
    for profile in _REGISTRY.values():
        pid = id(profile)
        if pid not in seen:
            seen.add(pid)
            result.append(profile)
    _PROVIDER_LIST_CACHE = result
    return list(result)


async def _user_plugins_dir_async() -> Path | None:
    """Return the user provider directory without synchronous stat calls."""
    try:
        from hermes_constants import get_hermes_home

        directory = get_hermes_home() / "plugins" / "model-providers"
        return directory if await aiofiles.os.path.isdir(directory) else None
    except Exception:
        return None


async def _import_plugin_dir_async(plugin_dir: Path, source: str) -> None:
    """Load one directory provider without blocking on source-file I/O."""
    init_file = plugin_dir / "__init__.py"
    if not await aiofiles.os.path.exists(init_file):
        return

    safe_name = plugin_dir.name.replace("-", "_")
    if source == "bundled":
        module_name = f"plugins.model_providers.{safe_name}"
        # ``model-providers`` is intentionally hyphenated on disk, so expose
        # the same import namespace as the synchronous loader without asking
        # Python's importer to scan the filesystem.
        if "plugins" not in sys.modules:
            package = types.ModuleType("plugins")
            package.__path__ = [str(_BUNDLED_PLUGINS_DIR.parent)]  # type: ignore[attr-defined]
            package.__package__ = "plugins"
            sys.modules["plugins"] = package
        if "plugins.model_providers" not in sys.modules:
            namespace = types.ModuleType("plugins.model_providers")
            namespace.__path__ = [str(_BUNDLED_PLUGINS_DIR)]  # type: ignore[attr-defined]
            namespace.__package__ = "plugins.model_providers"
            sys.modules["plugins.model_providers"] = namespace
    else:
        module_name = f"_hermes_user_provider_{safe_name}"

    if module_name in sys.modules:
        return

    try:
        await load_source_package(module_name, init_file)
    except asyncio.CancelledError:
        sys.modules.pop(module_name, None)
        for loaded_name in tuple(sys.modules):
            if loaded_name.startswith(f"{module_name}."):
                sys.modules.pop(loaded_name, None)
        raise
    except Exception as exc:
        logger.warning(
            "Failed to load %s provider plugin %s: %s", source, plugin_dir.name, exc
        )
        sys.modules.pop(module_name, None)
        for loaded_name in tuple(sys.modules):
            if loaded_name.startswith(f"{module_name}."):
                sys.modules.pop(loaded_name, None)


def _restore_async_discovery_state(
    registry: dict[str, ProviderProfile],
    aliases: dict[str, str],
    provider_cache: list[ProviderProfile] | None,
    discovered: bool,
    modules_before: set[str],
) -> None:
    """Roll back registry/module state after an interrupted discovery."""
    global _PROVIDER_LIST_CACHE, _discovered
    _REGISTRY.clear()
    _REGISTRY.update(registry)
    _ALIASES.clear()
    _ALIASES.update(aliases)
    _PROVIDER_LIST_CACHE = provider_cache
    _discovered = discovered
    for module_name in tuple(sys.modules):
        if module_name in modules_before:
            continue
        if module_name.startswith(
            ("plugins.model_providers", "_hermes_user_provider", "providers.")
        ):
            unload_source_finder(sys.modules[module_name])
            sys.modules.pop(module_name, None)


async def _discover_providers_async() -> None:
    """Populate provider profiles with cancellation-safe rollback."""
    snapshot = (
        dict(_REGISTRY),
        dict(_ALIASES),
        None if _PROVIDER_LIST_CACHE is None else list(_PROVIDER_LIST_CACHE),
        _discovered,
        {
            name
            for name in sys.modules
            if name.startswith(
                ("plugins.model_providers", "_hermes_user_provider", "providers.")
            )
        },
    )
    try:
        await _discover_providers_async_impl()
    except BaseException:
        _restore_async_discovery_state(*snapshot)
        raise


async def _discover_providers_async_impl() -> None:
    """Populate provider profiles through native async discovery."""
    global _discovered
    if _discovered:
        return

    if await aiofiles.os.path.isdir(_BUNDLED_PLUGINS_DIR):
        for child_name in sorted(await aiofiles.os.listdir(_BUNDLED_PLUGINS_DIR)):
            child = _BUNDLED_PLUGINS_DIR / child_name
            if child_name.startswith(("_", ".")):
                continue
            if await aiofiles.os.path.isdir(child):
                await _import_plugin_dir_async(child, "bundled")

    user_dir = await _user_plugins_dir_async()
    if user_dir is not None:
        for child_name in sorted(await aiofiles.os.listdir(user_dir)):
            child = user_dir / child_name
            if child_name.startswith(("_", ".")):
                continue
            if await aiofiles.os.path.isdir(child):
                await _import_plugin_dir_async(child, "user")

    # Preserve the legacy single-file extension point. There are no bundled
    # files in the retained tree, but user editable installs may still have
    # one. Load each source file through the same async boundary.
    providers_dir = _PROVIDERS_DIR
    for child_name in sorted(await aiofiles.os.listdir(providers_dir)):
        if not child_name.endswith(".py"):
            continue
        modname = child_name[:-3]
        if modname.startswith("_") or modname in {"base", "__init__"}:
            continue
        module_name = f"providers.{modname}"
        if module_name in sys.modules:
            continue
        source_path = providers_dir / child_name
        try:
            # ``SourceFileLoader`` would read the module synchronously when a
            # legacy profile performs a relative import.  Pre-read the module
            # and its package siblings into the async source finder instead.
            await load_source_module(
                module_name,
                source_path,
                package_dir=providers_dir,
            )
        except asyncio.CancelledError:
            sys.modules.pop(module_name, None)
            raise
        except Exception as exc:
            logger.warning("Failed to import legacy provider module %s: %s", modname, exc)
            sys.modules.pop(module_name, None)

    _discovered = True


async def _ensure_provider_profiles_loaded() -> None:
    """Await provider discovery once for the native agent runtime."""
    global _ASYNC_DISCOVERY_LOCK, _ASYNC_DISCOVERY_LOOP
    if _discovered:
        return
    loop = asyncio.get_running_loop()
    if (
        _ASYNC_DISCOVERY_LOCK is None
        or _ASYNC_DISCOVERY_LOOP is not loop
    ):
        _ASYNC_DISCOVERY_LOCK = asyncio.Lock()
        _ASYNC_DISCOVERY_LOOP = loop
    async with _ASYNC_DISCOVERY_LOCK:
        if not _discovered:
            await _discover_providers_async()
    _refresh_loaded_profile_projections()


def _refresh_loaded_profile_projections() -> None:
    """Refresh upstream registries already imported before async discovery."""
    for module_name, function_name in (
        ("hermes_cli.auth", "_inject_profile_provider_registry"),
        ("hermes_cli.config", "_inject_profile_env_vars"),
        ("hermes_cli.models", "_inject_profile_canonical_providers"),
    ):
        module = sys.modules.get(module_name)
        refresh = getattr(module, function_name, None) if module is not None else None
        if callable(refresh):
            refresh()


async def get_provider_profile(name: str) -> ProviderProfile | None:
    """Look up a provider profile by name or alias.

    Returns None if the provider has no profile (falls back to generic).
    """
    await _ensure_provider_profiles_loaded()
    return _get_provider_profile_cached(name)


async def list_providers() -> list[ProviderProfile]:
    """Return all registered provider profiles (one per canonical name)."""
    await _ensure_provider_profiles_loaded()
    return _list_providers_cached()
