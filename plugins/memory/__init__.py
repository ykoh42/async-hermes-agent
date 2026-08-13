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
import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

import aiofiles
import aiofiles.os

from hermes_cli.config import cfg_get
from hermes_cli.async_source_loader import _load_source_package

if TYPE_CHECKING:
    from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

_MEMORY_PLUGINS_DIR = Path(__file__).parent

# Synthetic parent package for user-installed providers, so they don't
# collide with bundled providers in sys.modules.
_USER_NAMESPACE = "_hermes_user_memory"


async def _exec_source_module(module: object, source_path: Path) -> None:
    """Execute a source module after asynchronously reading its file."""
    async with aiofiles.open(source_path, mode="rb") as handle:
        source_bytes = await handle.read()
    source = importlib.util.decode_source(source_bytes)
    exec(compile(source, str(source_path), "exec"), module.__dict__)


def _register_synthetic_package(name: str, search_locations: List[str]) -> None:
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

async def _get_user_plugins_dir() -> Optional[Path]:
    """Return ``$HERMES_HOME/plugins/`` or None if unavailable."""
    try:
        from hermes_constants import get_hermes_home
        d = get_hermes_home() / "plugins"
        return d if await aiofiles.os.path.isdir(d) else None
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


async def _iter_provider_dirs() -> List[Tuple[str, Path]]:
    """Yield ``(name, path)`` for all discovered provider directories.

    Scans bundled first, then user-installed.  Bundled takes precedence
    on name collisions (first-seen wins via ``seen`` set).
    """
    seen: set = set()
    dirs: List[Tuple[str, Path]] = []

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
    if user_dir:
        for child_name in sorted(await aiofiles.os.listdir(user_dir)):
            child = user_dir / child_name
            if not await aiofiles.os.path.isdir(child) or child.name.startswith(("_", ".")):
                continue
            if child.name in seen:
                continue  # bundled takes precedence
            if not await _is_memory_provider_dir(child):
                continue  # skip non-memory plugins
            dirs.append((child.name, child))

    return dirs


async def find_provider_dir(name: str) -> Optional[Path]:
    """Resolve a provider name to its directory.

    Checks bundled first, then user-installed.
    """
    # Bundled
    bundled = _MEMORY_PLUGINS_DIR / name
    if await aiofiles.os.path.isdir(bundled) and await aiofiles.os.path.exists(
        bundled / "__init__.py"
    ):
        return bundled
    # User-installed
    user_dir = await _get_user_plugins_dir()
    if user_dir:
        user = user_dir / name
        if await aiofiles.os.path.isdir(user) and await _is_memory_provider_dir(user):
            return user
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def list_memory_provider_names() -> List[str]:
    """Cheap name-only listing of discoverable memory providers.

    Unlike :func:`discover_memory_providers`, this does NOT import provider
    modules or run availability checks — it's a directory scan only, safe to
    call at module-import time (e.g. when building the dashboard config
    schema).
    """
    return sorted({name for name, _ in await _iter_provider_dirs()})


async def discover_memory_providers() -> List[Tuple[str, str, bool]]:
    """Scan bundled and user-installed directories for available providers.

    Returns list of (name, description, is_available) tuples.
    Bundled providers take precedence on name collisions.
    """
    results = []

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
            provider = await _load_provider_from_dir(child)
            if provider:
                available = await provider.is_available()
            else:
                available = False
        except Exception:
            available = False

        results.append((name, desc, available))

    return results


async def load_memory_provider(name: str) -> Optional["MemoryProvider"]:
    """Load and return a MemoryProvider instance by name.

    Checks both bundled (``plugins/memory/<name>/``) and user-installed
    (``$HERMES_HOME/plugins/<name>/``) directories.  Bundled takes
    precedence on name collisions.

    Returns None if the provider is not found or fails to load.
    """
    provider_dir = await find_provider_dir(name)
    if not provider_dir:
        logger.debug("Memory provider '%s' not found in bundled or user plugins", name)
        return None

    try:
        provider = await _load_provider_from_dir(provider_dir)
        if provider:
            return provider
        logger.warning("Memory provider '%s' loaded but no provider instance found", name)
        return None
    except Exception as e:
        logger.warning("Failed to load memory provider '%s': %s", name, e)
        return None


async def _load_provider_from_dir(provider_dir: Path) -> Optional["MemoryProvider"]:
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
        collector = _ProviderCollector()
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
    """Fake plugin context that captures register_memory_provider calls."""

    def __init__(self):
        self.provider = None

    def register_memory_provider(self, provider):
        self.provider = provider

    # No-op for other registration methods
    def register_tool(self, *args, **kwargs):
        pass

    def register_hook(self, *args, **kwargs):
        pass

    def register_cli_command(self, *args, **kwargs):
        pass  # CLI registration happens via discover_plugin_cli_commands()


async def _get_active_memory_provider() -> Optional[str]:
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


async def discover_plugin_cli_commands() -> List[dict]:
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
    results: List[dict] = []
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
