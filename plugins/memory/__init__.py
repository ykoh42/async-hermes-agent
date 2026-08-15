"""Memory provider plugin discovery.

Scans two directories for memory provider plugins:

1. Bundled providers: ``plugins/memory/<name>/`` (shipped with hermes-agent)
2. User-installed providers: ``$HERMES_HOME/plugins/<name>/``

Each subdirectory must contain ``__init__.py`` with a class implementing
the MemoryProvider ABC.  On name collisions, bundled providers take
precedence.

Only ONE provider can be active at a time, selected via
``memory.provider`` in config.yaml.

Usage:
    from plugins.memory import discover_memory_providers, load_memory_provider

    available = await discover_memory_providers()  # [(name, desc, available), ...]
    provider = await load_memory_provider("mnemosyne")  # MemoryProvider instance
"""

from __future__ import annotations

import asyncio
import importlib.machinery
import importlib.metadata
import importlib.util
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import aiofiles
import aiofiles.os

from hermes_cli.config import cfg_get
from hermes_cli.async_source_loader import _load_source_package

if TYPE_CHECKING:
    from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

_MEMORY_PLUGINS_DIR = Path(__file__).parent
ENTRY_POINTS_GROUP = "hermes_agent.memory_providers"
_REGISTERED_MEMORY_PROVIDER_SKILLS: dict[str, Path] = {}

# Synthetic parent package for user-installed providers, so they don't
# collide with bundled providers in sys.modules.
_USER_NAMESPACE = "_hermes_user_memory"


async def _exec_source_module(module: object, source_path: Path) -> None:
    """Execute a source module after asynchronously reading its file."""
    async with aiofiles.open(source_path, mode="rb") as handle:
        source_bytes = await handle.read()
    source = importlib.util.decode_source(source_bytes)
    exec(compile(source, str(source_path), "exec"), module.__dict__)


def _register_synthetic_package(name: str, search_locations: list[str]) -> None:
    """Register an empty package shell in sys.modules.

    User-installed providers import as ``_hermes_user_memory.<name>``, a
    dotted name whose parents exist nowhere on disk.  Unless those parents
    are present in ``sys.modules``, any relative import inside the plugin
    (``from . import config``) fails with
    ``ModuleNotFoundError: No module named '_hermes_user_memory'`` — the
    same reason the loader already registers ``plugins`` and
    ``plugins.memory`` for bundled providers.
    """
    if name in sys.modules:
        return
    spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
    spec.submodule_search_locations = search_locations
    sys.modules[name] = importlib.util.module_from_spec(spec)


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

async def _get_user_plugins_dir() -> Path | None:
    """Return ``$HERMES_HOME/plugins/`` or None if unavailable."""
    try:
        from hermes_constants import get_hermes_home
        d = get_hermes_home() / "plugins"
        return d if await aiofiles.os.path.isdir(d) else None
    except Exception:
        return None


async def _get_project_plugins_dir() -> Path | None:
    """Return the opt-in ``./.hermes/plugins`` directory."""
    try:
        from hermes_cli.plugins import _env_enabled

        if not _env_enabled("HERMES_ENABLE_PROJECT_PLUGINS"):
            return None
        cwd = await aiofiles.os.getcwd()
        directory = Path(cwd) / ".hermes" / "plugins"
        return directory if await aiofiles.os.path.isdir(directory) else None
    except Exception:
        return None


async def _is_memory_provider_dir(path: Path) -> bool:
    """Heuristic: does *path* look like a memory provider plugin?

    Checks for ``register_memory_provider`` or ``MemoryProvider`` in the
    ``__init__.py`` source.  Cheap text scan — no import needed.
    """
    init_file = path / "__init__.py"
    if not await aiofiles.os.path.exists(init_file):
        return False
    try:
        async with aiofiles.open(
            init_file, errors="replace", encoding="utf-8"
        ) as handle:
            source = await handle.read(8192)
        return "register_memory_provider" in source or "MemoryProvider" in source
    except Exception:
        return False


async def _iter_provider_dirs() -> list[tuple[str, Path]]:
    """Yield ``(name, path)`` for all discovered provider directories.

    Scans bundled first, then user-installed.  Bundled takes precedence
    on name collisions (first-seen wins via ``seen`` set).
    """
    seen: set = set()
    dirs: list[tuple[str, Path]] = []

    # 1. Bundled providers (plugins/memory/<name>/)
    if await aiofiles.os.path.isdir(_MEMORY_PLUGINS_DIR):
        for child_name in sorted(await aiofiles.os.listdir(_MEMORY_PLUGINS_DIR)):
            child = _MEMORY_PLUGINS_DIR / child_name
            if not await aiofiles.os.path.isdir(child) or child.name.startswith(("_", ".")):
                continue
            if not await aiofiles.os.path.exists(child / "__init__.py"):
                continue
            seen.add(child.name)
            dirs.append((child.name, child))

    # 2. User-installed providers ($HERMES_HOME/plugins/<name>/)
    user_dir = await _get_user_plugins_dir()
    project_dir = await _get_project_plugins_dir()
    for source_dir in (user_dir, project_dir):
        if not source_dir:
            continue
        for child_name in sorted(await aiofiles.os.listdir(source_dir)):
            child = source_dir / child_name
            if not await aiofiles.os.path.isdir(child) or child.name.startswith(("_", ".")):
                continue
            if child.name in seen:
                continue  # earlier source wins
            if not await _is_memory_provider_dir(child):
                continue  # skip non-memory plugins
            seen.add(child.name)
            dirs.append((child.name, child))

    return dirs


def _iter_entry_points() -> list[object]:
    """Enumerate memory-provider entry points without importing providers."""
    try:
        eps = importlib.metadata.entry_points()
        if hasattr(eps, "select"):
            return list(eps.select(group=ENTRY_POINTS_GROUP))
        if isinstance(eps, dict):
            return list(eps.get(ENTRY_POINTS_GROUP, []))
        return [ep for ep in eps if ep.group == ENTRY_POINTS_GROUP]
    except Exception as exc:
        logger.debug("Memory provider entry-point scan failed: %s", exc)
        return []


async def find_provider_dir(name: str) -> Path | None:
    """Resolve a provider name to its directory.

    Checks bundled first, then user-installed.
    """
    # Bundled
    bundled = _MEMORY_PLUGINS_DIR / name
    if await aiofiles.os.path.isdir(bundled) and await aiofiles.os.path.exists(
        bundled / "__init__.py"
    ):
        return bundled
    # User-installed, then opt-in project-local.
    for source_dir in (
        await _get_user_plugins_dir(),
        await _get_project_plugins_dir(),
    ):
        if not source_dir:
            continue
        candidate = source_dir / name
        if await aiofiles.os.path.isdir(candidate) and await _is_memory_provider_dir(candidate):
            return candidate
    return await _entry_point_package_dir(await find_provider_entry_point(name))


async def _entry_point_package_dir(entry_point: object | None) -> Path | None:
    """Resolve an entry-point package directory without importing it."""
    if entry_point is None:
        return None
    try:
        module_name = (getattr(entry_point, "value", "") or "").split(":", 1)[0].strip()
        if not module_name:
            return None
        parts = module_name.split(".")
        for root in sys.path:
            if not root:
                continue
            package_dir = Path(root).joinpath(*parts)
            if await aiofiles.os.path.isfile(package_dir / "__init__.py"):
                return package_dir
            if await aiofiles.os.path.isfile(package_dir.with_suffix(".py")):
                return None
    except Exception as exc:
        logger.debug(
            "Could not resolve directory for entry point '%s': %s",
            getattr(entry_point, "name", "?"),
            exc,
        )
    return None


async def find_provider_entry_point(name: str) -> object | None:
    """Resolve a provider name to a pip entry point, if installed."""
    return next((ep for ep in _iter_entry_points() if ep.name == name), None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def list_memory_provider_names() -> list[str]:
    """Cheap name-only listing of discoverable memory providers.

    Unlike :func:`discover_memory_providers`, this does NOT import provider
    modules or run availability checks — it's a directory scan only, safe to
    call at module-import time (e.g. when building the dashboard config
    schema).
    """
    names = {name for name, _ in await _iter_provider_dirs()}
    names.update(ep.name for ep in _iter_entry_points())
    return sorted(names)


async def discover_memory_providers() -> list[tuple[str, str, bool]]:
    """Scan bundled and user-installed directories for available providers.

    Returns list of (name, description, is_available) tuples.
    Bundled providers take precedence on name collisions.
    """
    results = []

    seen: set[str] = set()
    for name, child in await _iter_provider_dirs():
        # Read description from plugin.yaml if available
        desc = ""
        yaml_file = child / "plugin.yaml"
        if await aiofiles.os.path.exists(yaml_file):
            try:
                import yaml
                async with aiofiles.open(yaml_file, encoding="utf-8-sig") as f:
                    meta = yaml.safe_load(await f.read()) or {}
                desc = meta.get("description", "")
            except Exception:
                pass

        # Quick availability check — try loading and calling is_available()
        available = True
        try:
            provider = await _load_provider_from_dir(
                child,
                register_skills=False,
            )
            if provider:
                available = await provider.is_available()
            else:
                available = False
        except Exception:
            available = False

        results.append((name, desc, available))
        seen.add(name)

    for entry_point in _iter_entry_points():
        name = entry_point.name
        if name in seen:
            continue
        available = True
        try:
            provider = await _load_provider_from_entry_point(
                entry_point,
                register_skills=False,
            )
            available = bool(provider and await provider.is_available())
        except Exception:
            available = False
        results.append((name, "", available))
        seen.add(name)

    return results


async def load_memory_provider(
    name: str,
    *,
    register_skills: bool | None = None,
) -> MemoryProvider | None:
    """Load and return a MemoryProvider instance by name.

    Checks both bundled (``plugins/memory/<name>/``) and user-installed
    (``$HERMES_HOME/plugins/<name>/``) directories.  Bundled takes
    precedence on name collisions.

    Returns None if the provider is not found or fails to load.
    """
    if register_skills is None:
        register_skills = name == await _get_active_memory_provider()

    provider_dir = await find_provider_dir(name)
    entry_point = None if provider_dir else await find_provider_entry_point(name)
    if not provider_dir and entry_point is None:
        logger.debug("Memory provider '%s' not found in bundled, user plugins, or entry points", name)
        return None

    try:
        provider = (
            await _load_provider_from_dir(
                provider_dir,
                register_skills=register_skills,
            )
            if provider_dir
            else await _load_provider_from_entry_point(
                entry_point,
                register_skills=register_skills,
            )
        )
        if provider:
            return provider
        logger.warning("Memory provider '%s' loaded but no provider instance found", name)
        return None
    except Exception as e:
        logger.warning("Failed to load memory provider '%s': %s", name, e)
        return None


async def _load_provider_from_entry_point(
    entry_point: object,
    *,
    register_skills: bool = True,
) -> MemoryProvider | None:
    """Load an entry-point provider without blocking the caller."""
    from agent.memory_provider import MemoryProvider

    loaded = entry_point.load()
    if inspect.isawaitable(loaded):
        loaded = await loaded
    if isinstance(loaded, MemoryProvider):
        return loaded
    if isinstance(loaded, type) and issubclass(loaded, MemoryProvider):
        return loaded()
    if hasattr(loaded, "register"):
        collector = _ProviderCollector(
            getattr(entry_point, "name", "memory"),
            register_skills=register_skills,
        )
        result = loaded.register(collector)
        if inspect.isawaitable(result):
            await result
        return collector.provider
    if callable(loaded):
        result = loaded()
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, MemoryProvider):
            return result
        collector = _ProviderCollector(
            getattr(entry_point, "name", "memory"),
            register_skills=register_skills,
        )
        try:
            result = loaded(collector)
            if inspect.isawaitable(result):
                await result
        except TypeError:
            return None
        return collector.provider
    return None


async def _load_provider_from_dir(
    provider_dir: Path,
    *,
    register_skills: bool = True,
) -> MemoryProvider | None:
    """Import a provider module and extract the MemoryProvider instance.

    The module must have either:
    - A register(ctx) function (plugin-style) — we simulate a ctx
    - A top-level class that extends MemoryProvider — we instantiate it
    """
    name = provider_dir.name
    # Use a separate namespace for user-installed plugins so they don't
    # collide with bundled providers in sys.modules.
    _is_bundled = _MEMORY_PLUGINS_DIR in provider_dir.parents or provider_dir.parent == _MEMORY_PLUGINS_DIR
    module_name = f"plugins.memory.{name}" if _is_bundled else f"{_USER_NAMESPACE}.{name}"
    init_file = provider_dir / "__init__.py"

    if not await aiofiles.os.path.exists(init_file):
        return None

    # Check if already loaded.  A synthetic package shell registered by
    # discover_plugin_cli_commands() for relative-import support has no
    # __file__; only reuse modules that were actually loaded from disk.
    cached = sys.modules.get(module_name)
    if cached is not None and getattr(cached, "__file__", None):
        mod = cached
    else:
        # Handle relative imports within the plugin
        # First ensure the parent packages are registered
        for parent in ("plugins", "plugins.memory"):
            if parent not in sys.modules:
                parent_path = Path(__file__).parent
                if parent == "plugins":
                    parent_path = parent_path.parent
                parent_init = parent_path / "__init__.py"
                if await aiofiles.os.path.exists(parent_init):
                    spec = importlib.util.spec_from_file_location(
                        parent, str(parent_init),
                        submodule_search_locations=[str(parent_path)]
                    )
                    if spec and spec.loader:
                        parent_mod = importlib.util.module_from_spec(spec)
                        sys.modules[parent] = parent_mod
                        try:
                            await _exec_source_module(parent_mod, parent_init)
                        except asyncio.CancelledError:
                            sys.modules.pop(parent, None)
                            raise
                        except Exception:
                            pass

        # User-installed plugins need their synthetic parent registered the
        # same way, or relative imports inside the plugin cannot resolve.
        if not _is_bundled:
            _register_synthetic_package(_USER_NAMESPACE, [])

        try:
            mod = await _load_source_package(module_name, init_file)
        except asyncio.CancelledError:
            sys.modules.pop(module_name, None)
            for loaded_name in tuple(sys.modules):
                if loaded_name.startswith(f"{module_name}."):
                    sys.modules.pop(loaded_name, None)
            raise
        except Exception as e:
            logger.debug("Failed to execute source module %s: %s", module_name, e)
            sys.modules.pop(module_name, None)
            for loaded_name in tuple(sys.modules):
                if loaded_name.startswith(f"{module_name}."):
                    sys.modules.pop(loaded_name, None)
            return None

    # Try register(ctx) pattern first (how our plugins are written)
    if hasattr(mod, "register"):
        collector = _ProviderCollector(name, register_skills=register_skills)
        try:
            registration = mod.register(collector)
            if inspect.isawaitable(registration):
                await registration
            if collector.provider:
                return collector.provider
        except Exception as e:
            logger.debug("register() failed for %s: %s", name, e)

    # Fallback: find a MemoryProvider subclass and instantiate it
    from agent.memory_provider import MemoryProvider
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name, None)
        if (isinstance(attr, type) and issubclass(attr, MemoryProvider)
                and attr is not MemoryProvider):
            try:
                return attr()
            except Exception:
                pass

    return None


class _ProviderCollector:
    """Small plugin context used while loading a memory provider."""

    def __init__(self, name: str = "memory", *, register_skills: bool = True):
        self.name = name
        self.provider = None
        self._register_skills = register_skills
        self._context = None

    def register_memory_provider(self, provider):
        self.provider = provider

    def register_skill(self, *args, **kwargs):
        """Register provider skills only for the active provider."""
        if not self._register_skills:
            return
        try:
            from hermes_cli.plugins import (
                PluginContext,
                PluginManifest,
                get_plugin_manager,
            )

            manager = get_plugin_manager()
            if self._context is None:
                manifest = PluginManifest(name=self.name, key=self.name)
                self._context = PluginContext(manifest, manager)
            self._context.register_skill(*args, **kwargs)
            skill_name = args[0] if args else kwargs.get("name")
            if skill_name:
                qualified = f"{self.name}:{skill_name}"
                path = manager.find_plugin_skill(qualified)
                if path is not None:
                    _REGISTERED_MEMORY_PROVIDER_SKILLS[qualified] = path
        except Exception as exc:
            logger.debug(
                "Memory provider '%s' failed to register skill: %s",
                self.name,
                exc,
            )

    # No-op for other registration methods
    def register_tool(self, *args, **kwargs):
        pass

    def register_hook(self, *args, **kwargs):
        pass

    def register_auxiliary_task(self, *args, **kwargs):
        """Ignore optional auxiliary registration during provider discovery."""
        pass

    def register_cli_command(self, *args, **kwargs):
        pass  # CLI registration happens via discover_plugin_cli_commands()


async def _get_active_memory_provider() -> str | None:
    """Read the active memory provider name from config.yaml.

    Returns the provider name (e.g. ``"honcho"``) or None if no
    external provider is configured.  Lightweight — only reads config,
    no plugin loading.
    """
    try:
        from hermes_cli.config import load_config_readonly

        config = await load_config_readonly()
        return cfg_get(config, "memory", "provider") or None
    except Exception:
        return None


async def discover_plugin_cli_commands() -> list[dict]:
    """Return CLI commands for the **active** memory plugin only.

    Only one memory provider can be active at a time (set via
    ``memory.provider`` in config.yaml).  This function reads that
    value and only loads CLI registration for the matching plugin.
    If no provider is active, no commands are registered.

    Looks for a ``register_cli(subparser)`` function in the active
    plugin's ``cli.py``.  Returns a list of at most one dict with
    keys: ``name``, ``help``, ``description``, ``setup_fn``,
    ``handler_fn``.

    This is a lightweight scan — it only imports ``cli.py``, not the
    full plugin module.  Safe to call during argparse setup before
    any provider is loaded.
    """
    results: list[dict] = []
    if not await aiofiles.os.path.isdir(_MEMORY_PLUGINS_DIR):
        return results

    active_provider = await _get_active_memory_provider()
    if not active_provider:
        return results

    # Only look at the active provider's directory
    plugin_dir = await find_provider_dir(active_provider)
    if not plugin_dir:
        return results

    cli_file = plugin_dir / "cli.py"
    if not await aiofiles.os.path.exists(cli_file):
        return results

    _is_bundled = _MEMORY_PLUGINS_DIR in plugin_dir.parents or plugin_dir.parent == _MEMORY_PLUGINS_DIR
    module_name = f"plugins.memory.{active_provider}.cli" if _is_bundled else f"{_USER_NAMESPACE}.{active_provider}.cli"
    try:
        # Import the CLI module (lightweight — no SDK needed)
        if module_name in sys.modules:
            cli_mod = sys.modules[module_name]
        else:
            if not _is_bundled:
                # cli.py imports as _hermes_user_memory.<name>.cli, usually
                # before the provider itself is loaded.  Register its parent
                # packages so relative imports inside cli.py
                # ("from . import config") resolve without executing the
                # plugin's __init__.py.  The package shell has no __file__,
                # so _load_provider_from_dir() will still load the real
                # module later instead of reusing the shell.
                _register_synthetic_package(_USER_NAMESPACE, [])
                _register_synthetic_package(
                    f"{_USER_NAMESPACE}.{active_provider}", [str(plugin_dir)]
                )
            spec = importlib.util.spec_from_file_location(
                module_name, str(cli_file)
            )
            if not spec or not spec.loader:
                return results
            cli_mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = cli_mod
            try:
                await _exec_source_module(cli_mod, cli_file)
            except asyncio.CancelledError:
                sys.modules.pop(module_name, None)
                raise

        register_cli = getattr(cli_mod, "register_cli", None)
        if not callable(register_cli):
            return results

        # Read metadata from plugin.yaml if available
        help_text = f"Manage {active_provider} memory plugin"
        description = ""
        yaml_file = plugin_dir / "plugin.yaml"
        if await aiofiles.os.path.exists(yaml_file):
            try:
                import yaml

                async with aiofiles.open(yaml_file, encoding="utf-8-sig") as f:
                    meta = yaml.safe_load(await f.read()) or {}
                desc = meta.get("description", "")
                if desc:
                    help_text = desc
                    description = desc
            except Exception:
                pass

        handler_fn = getattr(cli_mod, f"{active_provider}_command", None) or \
                     getattr(cli_mod, "honcho_command", None)

        results.append({
            "name": active_provider,
            "help": help_text,
            "description": description,
            "setup_fn": register_cli,
            "handler_fn": handler_fn,
            "plugin": active_provider,
        })
    except Exception as e:
        logger.debug("Failed to scan CLI for memory plugin '%s': %s", active_provider, e)

    return results
