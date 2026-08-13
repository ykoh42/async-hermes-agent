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
import contextvars
import hashlib
import logging
import os
import sys
import threading
import types
import weakref
from dataclasses import dataclass, field
from pathlib import Path

import aiofiles.os

from providers.base import OMIT_TEMPERATURE, ProviderProfile  # noqa: F401
from hermes_cli.async_source_loader import (
    _load_source_module,
    _load_source_package,
    _unload_source_finder,
)

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, ProviderProfile] = {}
_ALIASES: dict[str, str] = {}
_PROVIDER_LIST_CACHE: list[ProviderProfile] | None = None
_discovered = False
_SHARED_REGISTRY_GENERATION = 0

# Bundled/legacy discovery mutates one process-shared registry, so an
# asyncio.Lock owned by whichever event loop arrived last cannot serialize
# concurrent cold starts in different threads.  Keep only loop-neutral state
# under a short, non-blocking guard and wake each waiting loop through its own
# Future.
_SHARED_DISCOVERY_GUARD = threading.RLock()
_SHARED_DISCOVERY_RUNNING = False
_SHARED_DISCOVERY_WAITERS: set[asyncio.Future[None]] = set()


@dataclass
class _UserProviderState:
    """One event loop and canonical HERMES_HOME's provider overlay."""

    namespace: str
    registry: dict[str, ProviderProfile] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    provider_list_cache: list[ProviderProfile] | None = None
    provider_list_generation: int = -1
    discovered: bool = False
    module_names: set[str] = field(default_factory=set)


_USER_PROVIDER_STATES: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, _UserProviderState]
] = weakref.WeakKeyDictionary()
_USER_PROVIDER_LOCKS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, asyncio.Lock]
] = weakref.WeakKeyDictionary()
_USER_PROVIDER_STATE_GUARD = threading.RLock()
_USER_PROVIDER_NAMESPACE_COUNTER = 0
_ACTIVE_USER_PROVIDER_STATE: contextvars.ContextVar[
    _UserProviderState | None
] = contextvars.ContextVar("active_user_provider_state", default=None)
_REGISTRATION_TARGET: contextvars.ContextVar[
    _UserProviderState | None
] = contextvars.ContextVar("provider_registration_target", default=None)

# Repo-root ``plugins/model-providers/`` — populated at discovery time.
_BUNDLED_PLUGINS_DIR = (
    Path(__file__).parent.parent / "plugins" / "model-providers"
)
_PROVIDERS_DIR = Path(__file__).parent


def register_provider(profile: ProviderProfile) -> None:
    """Register a provider profile by name and aliases.

    Later registrations with the same name replace earlier ones — so user
    plugins under ``$HERMES_HOME/plugins/model-providers/`` can override
    bundled profiles without editing repo code.
    """
    global _PROVIDER_LIST_CACHE, _SHARED_REGISTRY_GENERATION
    target = _REGISTRATION_TARGET.get()
    if target is not None:
        target.registry[profile.name] = profile
        for alias in profile.aliases:
            target.aliases[alias] = profile.name
        target.provider_list_cache = None
        return

    _REGISTRY[profile.name] = profile
    for alias in profile.aliases:
        _ALIASES[alias] = profile.name
    _PROVIDER_LIST_CACHE = None
    _SHARED_REGISTRY_GENERATION += 1


def _effective_user_provider_state() -> _UserProviderState | None:
    """Return the registration overlay during import, else the active profile."""
    return _REGISTRATION_TARGET.get() or _ACTIVE_USER_PROVIDER_STATE.get()


def _get_provider_profile_cached(name: str) -> ProviderProfile | None:
    """Look up one already-discovered profile without performing I/O."""
    state = _effective_user_provider_state()
    if state is not None:
        canonical = state.aliases.get(name, _ALIASES.get(name, name))
        profile = state.registry.get(canonical)
        if profile is not None:
            return profile
    canonical = _ALIASES.get(name, name)
    return _REGISTRY.get(canonical)


def _list_providers_cached() -> list[ProviderProfile]:
    """Return the process-shared bundled/legacy provider snapshot.

    Historical projection helpers in auth/config/models call this private
    function and mutate process-global maps.  It must never expose a
    profile-local user overlay to those projections.
    """
    global _PROVIDER_LIST_CACHE

    if _PROVIDER_LIST_CACHE is not None:
        return list(_PROVIDER_LIST_CACHE)
    seen: set[int] = set()
    result: list[ProviderProfile] = []
    for profile in _REGISTRY.values():
        profile_id = id(profile)
        if profile_id not in seen:
            seen.add(profile_id)
            result.append(profile)
    _PROVIDER_LIST_CACHE = result
    return list(result)


def _list_effective_providers_cached() -> list[ProviderProfile]:
    """Return the active profile overlay merged over shared providers."""
    state = _effective_user_provider_state()
    if state is not None:
        if (
            state.provider_list_cache is not None
            and state.provider_list_generation == _SHARED_REGISTRY_GENERATION
        ):
            return list(state.provider_list_cache)
        merged = dict(_REGISTRY)
        merged.update(state.registry)
        seen: set[int] = set()
        result: list[ProviderProfile] = []
        for profile in merged.values():
            profile_id = id(profile)
            if profile_id in seen:
                continue
            seen.add(profile_id)
            result.append(profile)
        state.provider_list_cache = result
        state.provider_list_generation = _SHARED_REGISTRY_GENERATION
        return list(result)
    return _list_providers_cached()


async def _user_plugins_dir() -> Path | None:
    """Return the user provider directory without synchronous stat calls."""
    try:
        from hermes_constants import get_hermes_home

        directory = get_hermes_home() / "plugins" / "model-providers"
        return directory if await aiofiles.os.path.isdir(directory) else None
    except Exception:
        return None


async def _canonical_user_provider_home() -> tuple[str, Path]:
    """Resolve the active profile home without sharing process-global state."""
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    realpath = aiofiles.os.wrap(os.path.realpath)
    canonical = os.path.normcase(await realpath(str(home)))
    return canonical, Path(canonical)


def _new_user_provider_state(profile_key: str) -> _UserProviderState:
    global _USER_PROVIDER_NAMESPACE_COUNTER
    _USER_PROVIDER_NAMESPACE_COUNTER += 1
    digest = hashlib.sha256(
        f"{_USER_PROVIDER_NAMESPACE_COUNTER}\0{profile_key}".encode()
    ).hexdigest()[:16]
    return _UserProviderState(namespace=digest)


def _get_user_provider_state(
    loop: asyncio.AbstractEventLoop,
    profile_key: str,
) -> tuple[_UserProviderState, asyncio.Lock]:
    """Return the loop/profile state and its same-loop discovery lock."""
    with _USER_PROVIDER_STATE_GUARD:
        per_loop_states = _USER_PROVIDER_STATES.setdefault(loop, {})
        state = per_loop_states.get(profile_key)
        if state is None:
            state = _new_user_provider_state(profile_key)
            per_loop_states[profile_key] = state
        per_loop_locks = _USER_PROVIDER_LOCKS.setdefault(loop, {})
        lock = per_loop_locks.get(profile_key)
        if lock is None:
            lock = asyncio.Lock()
            per_loop_locks[profile_key] = lock
        return state, lock


def _unload_provider_modules(module_names: set[str]) -> None:
    """Remove source finders and modules owned by one user overlay."""
    for module_name in sorted(module_names, key=len, reverse=True):
        module = sys.modules.pop(module_name, None)
        if module is not None:
            _unload_source_finder(module)


def _clear_user_provider_states() -> None:
    """Unload every profile overlay (private lifecycle/test boundary)."""
    with _USER_PROVIDER_STATE_GUARD:
        states = [
            state
            for per_loop in _USER_PROVIDER_STATES.values()
            for state in per_loop.values()
        ]
        _USER_PROVIDER_STATES.clear()
        _USER_PROVIDER_LOCKS.clear()
    for state in states:
        _unload_provider_modules(state.module_names)
        state.module_names.clear()
        state.registry.clear()
        state.aliases.clear()
        state.provider_list_cache = None
        state.provider_list_generation = -1
        state.discovered = False
    _ACTIVE_USER_PROVIDER_STATE.set(None)


async def _import_plugin_dir(
    plugin_dir: Path,
    source: str,
    *,
    user_namespace: str = "",
) -> str | None:
    """Load one directory provider without blocking on source-file I/O."""
    init_file = plugin_dir / "__init__.py"
    if not await aiofiles.os.path.exists(init_file):
        return None

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
        module_name = (
            f"_hermes_user_provider_{user_namespace}_{safe_name}"
            if user_namespace
            else f"_hermes_user_provider_{safe_name}"
        )

    if module_name in sys.modules:
        return module_name

    try:
        await _load_source_package(module_name, init_file)
    except asyncio.CancelledError:
        module = sys.modules.pop(module_name, None)
        if module is not None:
            _unload_source_finder(module)
        for loaded_name in tuple(sys.modules):
            if loaded_name.startswith(f"{module_name}."):
                loaded = sys.modules.pop(loaded_name, None)
                if loaded is not None:
                    _unload_source_finder(loaded)
        raise
    except Exception as exc:
        logger.warning(
            "Failed to load %s provider plugin %s: %s", source, plugin_dir.name, exc
        )
        module = sys.modules.pop(module_name, None)
        if module is not None:
            _unload_source_finder(module)
        for loaded_name in tuple(sys.modules):
            if loaded_name.startswith(f"{module_name}."):
                loaded = sys.modules.pop(loaded_name, None)
                if loaded is not None:
                    _unload_source_finder(loaded)
        return None
    return module_name


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
            _unload_source_finder(sys.modules[module_name])
            sys.modules.pop(module_name, None)


async def _discover_providers() -> None:
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
        await _discover_providers_impl()
    except BaseException:
        _restore_async_discovery_state(*snapshot)
        raise


async def _discover_providers_impl() -> None:
    """Populate shared bundled and legacy providers through async discovery."""
    global _discovered
    if _discovered:
        return

    if await aiofiles.os.path.isdir(_BUNDLED_PLUGINS_DIR):
        for child_name in sorted(await aiofiles.os.listdir(_BUNDLED_PLUGINS_DIR)):
            child = _BUNDLED_PLUGINS_DIR / child_name
            if child_name.startswith(("_", ".")):
                continue
            if await aiofiles.os.path.isdir(child):
                await _import_plugin_dir(child, "bundled")

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
            await _load_source_module(
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


def _restore_user_provider_state(
    state: _UserProviderState,
    registry: dict[str, ProviderProfile],
    aliases: dict[str, str],
    provider_cache: list[ProviderProfile] | None,
    provider_generation: int,
    discovered: bool,
    modules_before: set[str],
) -> None:
    """Roll back one profile overlay without disturbing sibling profiles."""
    state.registry.clear()
    state.registry.update(registry)
    state.aliases.clear()
    state.aliases.update(aliases)
    state.provider_list_cache = provider_cache
    state.provider_list_generation = provider_generation
    state.discovered = discovered
    added_modules = state.module_names - modules_before
    _unload_provider_modules(added_modules)
    state.module_names.intersection_update(modules_before)


async def _discover_user_provider_profiles(
    state: _UserProviderState,
    profile_home: Path,
) -> None:
    """Discover only the active HERMES_HOME's isolated user overlay."""
    if state.discovered:
        return
    snapshot = (
        dict(state.registry),
        dict(state.aliases),
        (
            None
            if state.provider_list_cache is None
            else list(state.provider_list_cache)
        ),
        state.provider_list_generation,
        state.discovered,
        set(state.module_names),
    )
    registration_token = _REGISTRATION_TARGET.set(state)
    try:
        user_dir = profile_home / "plugins" / "model-providers"
        if await aiofiles.os.path.isdir(user_dir):
            for child_name in sorted(await aiofiles.os.listdir(user_dir)):
                child = user_dir / child_name
                if child_name.startswith(("_", ".")):
                    continue
                if not await aiofiles.os.path.isdir(child):
                    continue
                plugin_registry = dict(state.registry)
                plugin_aliases = dict(state.aliases)
                module_name = await _import_plugin_dir(
                    child,
                    "user",
                    user_namespace=state.namespace,
                )
                if module_name is None:
                    # A failed plugin must not leave registrations performed
                    # before its exception visible to this profile.
                    state.registry.clear()
                    state.registry.update(plugin_registry)
                    state.aliases.clear()
                    state.aliases.update(plugin_aliases)
                    state.provider_list_cache = None
                    continue
                state.module_names.update(
                    loaded_name
                    for loaded_name in sys.modules
                    if loaded_name == module_name
                    or loaded_name.startswith(f"{module_name}.")
                )
        state.discovered = True
    except BaseException:
        _restore_user_provider_state(state, *snapshot)
        raise
    finally:
        _REGISTRATION_TARGET.reset(registration_token)


async def _ensure_user_provider_profiles_loaded() -> _UserProviderState:
    """Load and activate the current loop/profile overlay exactly once."""
    profile_key, profile_home = await _canonical_user_provider_home()
    loop = asyncio.get_running_loop()
    state, lock = _get_user_provider_state(loop, profile_key)
    try:
        if not state.discovered:
            async with lock:
                if not state.discovered:
                    await _discover_user_provider_profiles(state, profile_home)
    finally:
        with _USER_PROVIDER_STATE_GUARD:
            per_loop_locks = _USER_PROVIDER_LOCKS.get(loop)
            waiters = getattr(lock, "_waiters", None)
            if (
                per_loop_locks is not None
                and per_loop_locks.get(profile_key) is lock
                and not lock.locked()
                and not waiters
            ):
                per_loop_locks.pop(profile_key, None)
                if not per_loop_locks:
                    _USER_PROVIDER_LOCKS.pop(loop, None)
    _ACTIVE_USER_PROVIDER_STATE.set(state)
    return state


def _settle_shared_discovery_waiter(waiter: asyncio.Future[None]) -> None:
    if not waiter.done():
        waiter.set_result(None)


def _wake_shared_discovery_waiters(
    waiters: tuple[asyncio.Future[None], ...],
) -> None:
    for waiter in waiters:
        try:
            waiter.get_loop().call_soon_threadsafe(
                _settle_shared_discovery_waiter,
                waiter,
            )
        except RuntimeError:
            # A loop that has already shut down cannot resume its waiter and
            # must not retain the remaining process-wide discovery lifecycle.
            continue


async def _ensure_shared_provider_profiles_loaded() -> None:
    """Serialize one shared discovery across concurrently running loops."""
    global _SHARED_DISCOVERY_RUNNING
    while True:
        leader = False
        waiter: asyncio.Future[None] | None = None
        with _SHARED_DISCOVERY_GUARD:
            if _discovered:
                return
            if not _SHARED_DISCOVERY_RUNNING:
                _SHARED_DISCOVERY_RUNNING = True
                leader = True
            else:
                waiter = asyncio.get_running_loop().create_future()
                _SHARED_DISCOVERY_WAITERS.add(waiter)
        if not leader:
            assert waiter is not None
            try:
                await waiter
            finally:
                with _SHARED_DISCOVERY_GUARD:
                    _SHARED_DISCOVERY_WAITERS.discard(waiter)
            continue

        try:
            await _discover_providers()
        finally:
            with _SHARED_DISCOVERY_GUARD:
                _SHARED_DISCOVERY_RUNNING = False
                waiters = tuple(_SHARED_DISCOVERY_WAITERS)
                _SHARED_DISCOVERY_WAITERS.clear()
            _wake_shared_discovery_waiters(waiters)
        return


async def _ensure_provider_profiles_loaded() -> None:
    """Await provider discovery once for the native agent runtime."""
    await _ensure_shared_provider_profiles_loaded()
    # Downstream projection registries are process-global. Project only the
    # bundled/legacy shared layer; an A-only user provider must never be
    # injected into auth/config/models where profile B could inherit it.
    active_token = _ACTIVE_USER_PROVIDER_STATE.set(None)
    try:
        _refresh_loaded_profile_projections()
    finally:
        _ACTIVE_USER_PROVIDER_STATE.reset(active_token)
    await _ensure_user_provider_profiles_loaded()


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
    return _list_effective_providers_cached()
