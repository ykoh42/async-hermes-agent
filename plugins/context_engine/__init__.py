"""Context engine plugin discovery.

Scans ``plugins/context_engine/<name>/`` directories for context engine
plugins.  Each subdirectory must contain ``__init__.py`` with a class
implementing the ContextEngine ABC.

Context engines are separate from the general plugin system — they live
in the repo and are always available without user installation.  Only ONE
can be active at a time, selected via ``context.engine`` in config.yaml.
The default engine is ``"compressor"`` (the built-in ContextCompressor).

Usage:
    from plugins.context_engine import discover_context_engines, load_context_engine

    available = await discover_context_engines()   # [(name, desc, available), ...]
    engine = await load_context_engine("lcm")      # ContextEngine instance
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

import aiofiles
import aiofiles.os
from hermes_cli.async_source_loader import load_source_package

if TYPE_CHECKING:
    from agent.context_engine import ContextEngine

logger = logging.getLogger(__name__)

_CONTEXT_ENGINE_PLUGINS_DIR = Path(__file__).parent


async def _exec_source_module(module: object, source_path: Path) -> None:
    """Execute a source module after asynchronously reading its file.

    ``SourceFileLoader.exec_module`` performs a synchronous file read.  The
    loader is still used to create the normal module spec (including package
    metadata), but source acquisition is kept on the async path so plugin
    discovery does not block on disk I/O.
    """
    async with aiofiles.open(source_path, mode="rb") as handle:
        source_bytes = await handle.read()
    source = importlib.util.decode_source(source_bytes)
    exec(compile(source, str(source_path), "exec"), module.__dict__)


async def discover_context_engines() -> List[Tuple[str, str, bool]]:
    """Scan plugins/context_engine/ for available engines.

    Returns list of (name, description, is_available) tuples.
    Does NOT import the engines — just reads plugin.yaml for metadata
    and does a lightweight availability check.
    """
    results = []
    if not await aiofiles.os.path.isdir(_CONTEXT_ENGINE_PLUGINS_DIR):
        return results

    for child_name in sorted(
        await aiofiles.os.listdir(_CONTEXT_ENGINE_PLUGINS_DIR)
    ):
        child = _CONTEXT_ENGINE_PLUGINS_DIR / child_name
        if not await aiofiles.os.path.isdir(child) or child.name.startswith(("_", ".")):
            continue
        init_file = child / "__init__.py"
        if not await aiofiles.os.path.exists(init_file):
            continue

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
            engine = await _load_engine_from_dir(child)
            if engine is None:
                available = False
            elif hasattr(engine, "is_available"):
                available_result = engine.is_available()
                if inspect.isawaitable(available_result):
                    available_result = await available_result
                available = bool(available_result)
        except Exception:
            available = False

        results.append((child.name, desc, available))

    return results


async def load_context_engine(name: str) -> Optional["ContextEngine"]:
    """Load and return a ContextEngine instance by name.

    Returns None if the engine is not found or fails to load.
    """
    engine_dir = _CONTEXT_ENGINE_PLUGINS_DIR / name
    if not await aiofiles.os.path.isdir(engine_dir):
        logger.debug("Context engine '%s' not found in %s", name, _CONTEXT_ENGINE_PLUGINS_DIR)
        return None

    try:
        engine = await _load_engine_from_dir(engine_dir)
        if engine:
            return engine
        logger.warning("Context engine '%s' loaded but no engine instance found", name)
        return None
    except Exception as e:
        logger.warning("Failed to load context engine '%s': %s", name, e)
        return None


async def _load_engine_from_dir(engine_dir: Path) -> Optional["ContextEngine"]:
    """Import an engine module and extract the ContextEngine instance.

    The module must have either:
    - A register(ctx) function (plugin-style) — we simulate a ctx
    - A top-level class that extends ContextEngine — we instantiate it
    """
    name = engine_dir.name
    module_name = f"plugins.context_engine.{name}"
    init_file = engine_dir / "__init__.py"

    if not await aiofiles.os.path.exists(init_file):
        return None

    # Check if already loaded
    if module_name in sys.modules:
        mod = sys.modules[module_name]
    else:
        # Handle relative imports within the plugin
        # First ensure the parent packages are registered
        for parent in ("plugins", "plugins.context_engine"):
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
                    if spec:
                        parent_mod = importlib.util.module_from_spec(spec)
                        sys.modules[parent] = parent_mod
                        try:
                            await _exec_source_module(parent_mod, parent_init)
                        except asyncio.CancelledError:
                            sys.modules.pop(parent, None)
                            raise
                        except Exception:
                            pass

        try:
            mod = await load_source_package(module_name, init_file)
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

    # Try register(ctx) pattern first (how plugins are written)
    if hasattr(mod, "register"):
        collector = _EngineCollector(engine_name=name)
        try:
            registration = mod.register(collector)
            if inspect.isawaitable(registration):
                await registration
            if collector.engine:
                return collector.engine
        except Exception as e:
            logger.debug("register() failed for %s: %s", name, e)

    # Fallback: find a ContextEngine subclass and instantiate it
    from agent.context_engine import ContextEngine
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name, None)
        if (isinstance(attr, type) and issubclass(attr, ContextEngine)
                and attr is not ContextEngine):
            try:
                return attr()
            except Exception:
                pass

    return None


class _EngineCollector:
    """Fake plugin context that captures register_context_engine calls.

    Plugin context engines using the standard ``register(ctx)`` pattern may
    also call ``ctx.register_command(...)`` to expose slash commands (e.g.
    ``/lcm``). Forward those to the global plugin command registry so they
    behave identically to commands registered by normal plugins.
    """

    def __init__(self, engine_name: str = ""):
        self.engine = None
        self._engine_name = engine_name or "context_engine"
        self._registered_commands: list[str] = []

    def register_context_engine(self, engine):
        self.engine = engine

    def register_command(
        self,
        name: str,
        handler,
        description: str = "",
        args_hint: str = "",
    ) -> None:
        """Forward to the global plugin command registry."""
        clean = (name or "").lower().strip().lstrip("/").replace(" ", "-")
        if not clean:
            logger.warning(
                "Context engine '%s' tried to register a command with an empty name.",
                self._engine_name,
            )
            return

        try:
            from hermes_cli.plugins import get_plugin_manager
            manager = get_plugin_manager()
            if clean in manager._plugin_commands:
                # Don't clobber a regular plugin's command — same conflict
                # policy the plugin system uses for plugin-vs-plugin collisions.
                logger.warning(
                    "Context engine '%s' tried to register command '/%s' which "
                    "is already registered by a plugin. Skipping.",
                    self._engine_name, clean,
                )
                return
            manager._plugin_commands[clean] = {
                "handler": handler,
                "description": description or "Context engine command",
                "plugin": f"context-engine:{self._engine_name}",
                "args_hint": (args_hint or "").strip(),
            }
            self._registered_commands.append(clean)
            logger.debug(
                "Context engine '%s' registered command: /%s",
                self._engine_name, clean,
            )
        except Exception as exc:
            logger.debug(
                "Context engine '%s' could not register /%s: %s",
                self._engine_name, clean, exc,
            )

    # No-op for other registration methods
    def register_tool(self, *args, **kwargs):
        pass

    def register_hook(self, *args, **kwargs):
        pass

    def register_cli_command(self, *args, **kwargs):
        pass

    def register_memory_provider(self, *args, **kwargs):
        pass
