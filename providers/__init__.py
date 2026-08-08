"""Provider module registry.

Provider profiles can live in two places:

1. Bundled plugins: ``plugins/model-providers/<name>/`` (shipped with hermes-agent)
2. User plugins: ``$HERMES_HOME/plugins/model-providers/<name>/``

Each plugin directory contains:
  - ``__init__.py`` — calls ``register_provider(profile)`` at import
  - ``plugin.yaml`` — manifest (name, kind: model-provider, version, description)

Discovery is lazy: the first synchronous call to ``get_provider_profile()`` or
``list_providers()`` outside an event loop scans both locations and imports
every plugin. The retained async agent awaits the discovery boundary before
using these in-memory getters. User
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
    profile = get_provider_profile("nvidia")   # ProviderProfile or None
    profile = get_provider_profile("kimi")     # checks name + aliases
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import logging
import sys
import types
from pathlib import Path

import aiofiles
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


def _require_sync_discovery_boundary() -> None:
    """Reject synchronous discovery while an event loop is running.

    The legacy getters remain synchronous for callers that use the registry
    outside the agent runtime.  Starting a directory scan from an async
    request, however, would synchronously read and execute every provider
    module.  The retained async runtime calls ``_ensure_provider_profiles_loaded``
    before any getter, so this guard only affects an uninitialised direct call
    and gives it an actionable failure instead of blocking the loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(
        "Provider profiles are not loaded; await the agent runtime boundary "
        "before using the synchronous provider registry"
    )


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


def get_provider_profile(name: str) -> ProviderProfile | None:
    """Look up a provider profile by name or alias.

    Returns None if the provider has no profile (falls back to generic).
    """
    if not _discovered:
        _require_sync_discovery_boundary()
        _discover_providers()
    canonical = _ALIASES.get(name, name)
    return _REGISTRY.get(canonical)


def list_providers() -> list[ProviderProfile]:
    """Return all registered provider profiles (one per canonical name)."""
    global _PROVIDER_LIST_CACHE
    if not _discovered:
        _require_sync_discovery_boundary()
        _discover_providers()
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


def _user_plugins_dir() -> Path | None:
    """Return ``$HERMES_HOME/plugins/model-providers/`` if it exists."""
    try:
        from hermes_constants import get_hermes_home

        d = get_hermes_home() / "plugins" / "model-providers"
        return d if d.is_dir() else None
    except Exception:
        return None


def _import_plugin_dir(plugin_dir: Path, source: str) -> None:
    """Import a single plugin directory so it self-registers.

    ``source`` is "bundled" or "user", used only for log messages.
    """
    init_file = plugin_dir / "__init__.py"
    if not init_file.exists():
        return

    # Give bundled plugins a stable import path (``plugins.model_providers.<name>``)
    # so relative imports within the plugin work. User plugins load via
    # ``importlib.util.spec_from_file_location`` with a unique module name so
    # multiple HERMES_HOME profiles don't alias each other.
    safe_name = plugin_dir.name.replace("-", "_")
    if source == "bundled":
        module_name = f"plugins.model_providers.{safe_name}"
    else:
        module_name = f"_hermes_user_provider_{safe_name}"

    if module_name in sys.modules:
        return  # already imported

    try:
        spec = importlib.util.spec_from_file_location(
            module_name, init_file, submodule_search_locations=[str(plugin_dir)]
        )
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        # This branch is retained for synchronous setup/compatibility callers;
        # the agent runtime uses ``_import_plugin_dir_async`` above.  Keep the
        # same PEP 263 decoding as SourceFileLoader without invoking its
        # synchronous file-reading method in the native path.
        source = importlib.util.decode_source(init_file.read_bytes())
        exec(compile(source, str(init_file), "exec"), module.__dict__)
    except Exception as exc:
        logger.warning(
            "Failed to load %s provider plugin %s: %s", source, plugin_dir.name, exc
        )
        sys.modules.pop(module_name, None)


def _discover_providers() -> None:
    """Populate the registry by importing every provider plugin.

    Order:
      1. Bundled plugins at ``<repo>/plugins/model-providers/<name>/``
      2. User plugins at ``$HERMES_HOME/plugins/model-providers/<name>/``
      3. Legacy per-file modules at ``providers/<name>.py`` (back-compat)

    Each step imports its plugins, which call ``register_provider()`` at
    module-level. Later steps win on name collision.
    """
    global _discovered
    if _discovered:
        return
    _discovered = True

    # 1. Bundled plugins — shipped with hermes-agent.
    if _BUNDLED_PLUGINS_DIR.is_dir():
        for child in sorted(_BUNDLED_PLUGINS_DIR.iterdir()):
            if not child.is_dir() or child.name.startswith(("_", ".")):
                continue
            _import_plugin_dir(child, "bundled")

    # 2. User plugins — under $HERMES_HOME/plugins/model-providers/<name>/.
    #    These can override any bundled profile of the same name (last-writer-wins
    #    in register_provider()).
    user_dir = _user_plugins_dir()
    if user_dir is not None:
        for child in sorted(user_dir.iterdir()):
            if not child.is_dir() or child.name.startswith(("_", ".")):
                continue
            _import_plugin_dir(child, "user")

    # 3. Legacy single-file profiles at providers/<name>.py. Kept for
    #    back-compat — if someone drops a ``providers/foo.py`` into an
    #    editable install, it still works without the plugin layout.
    try:
        import pkgutil

        import providers as _pkg

        for _importer, modname, _ispkg in pkgutil.iter_modules(_pkg.__path__):
            if modname.startswith("_") or modname == "base":
                continue
            try:
                importlib.import_module(f"providers.{modname}")
            except ImportError as exc:
                logger.warning(
                    "Failed to import legacy provider module %s: %s", modname, exc
                )
    except Exception:
        pass


async def _exec_source_module(module: types.ModuleType, source_path: Path) -> None:
    """Execute a provider source module after asynchronously reading it."""
    async with aiofiles.open(source_path, mode="rb") as handle:
        source_bytes = await handle.read()
    source = importlib.util.decode_source(source_bytes)
    exec(compile(source, str(source_path), "exec"), module.__dict__)


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
    """Populate provider profiles through native async discovery.

    The synchronous getter remains available for setup/compatibility callers,
    while the retained agent runtime calls this awaited boundary before any
    profile lookup can trigger filesystem access.
    """
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
