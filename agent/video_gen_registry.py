"""
Video Generation Provider Registry
==================================

Central map of registered providers. Populated by plugins at import-time via
``PluginContext.register_video_gen_provider()``; consumed by the
``video_generate`` tool to dispatch each call to the active backend.

Active selection
----------------
The active provider is chosen by ``video_gen.provider`` in ``config.yaml``.
If unset, :func:`get_active_provider` applies fallback logic:

1. If exactly one *available* provider is registered, use it.
2. Otherwise return ``None`` (the tool surfaces a helpful error pointing
   the user at ``hermes tools``).

Mirrors ``agent/image_gen_registry.py`` so the two surfaces behave the
same: the unconfigured fallback is filtered by ``is_available()`` so a box
that has credentials for only one backend (e.g. DeepInfra, while the
``fal``/``xai`` plugins also register unconditionally) auto-selects it
instead of returning ``None``.
"""

from __future__ import annotations

import logging
import inspect
import sys

from agent.video_gen_provider import VideoGenProvider

logger = logging.getLogger(__name__)


_providers: dict[str, VideoGenProvider] = {}
_plugin_providers: dict[object, dict[str, VideoGenProvider]] = {}


def _plugin_scope(
    *, registration: bool = False, module_name: str = ""
) -> object | None:
    module = sys.modules.get("hermes_cli.plugins")
    current = getattr(module, "_current_plugin_registry_scope", None)
    scope = current(registration=registration) if callable(current) else None
    if scope is None and registration and module_name:
        resolve = getattr(module, "_plugin_registry_scope_for_module", None)
        if callable(resolve):
            return resolve(module_name)
    return scope


def _provider_snapshot(scope: str | None = None) -> dict[str, VideoGenProvider]:
    providers = dict(_providers)
    active_scope = scope if scope is not None else _plugin_scope()
    if active_scope is not None:
        providers.update(_plugin_providers.get(active_scope, {}))
    return providers


def _clear_plugin_scope(scope: object) -> None:
    _plugin_providers.pop(scope, None)


def register_provider(
    provider: VideoGenProvider,
    *,
    scope: str | None = None,
) -> None:
    """Register a video generation provider.

    Re-registration (same ``name``) overwrites the previous entry and logs
    a debug message — this makes hot-reload scenarios (tests, dev loops)
    behave predictably.
    """
    if not isinstance(provider, VideoGenProvider):
        raise TypeError(
            f"register_provider() expects a VideoGenProvider instance, "
            f"got {type(provider).__name__}"
        )
    name = provider.name
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Video gen provider .name must be a non-empty string")
    if not inspect.iscoroutinefunction(provider.generate):
        raise TypeError("Video gen provider .generate must be async")
    registration_scope = (
        scope
        if scope is not None
        else _plugin_scope(
            registration=True,
            module_name=type(provider).__module__,
        )
    )
    target = (
        _plugin_providers.setdefault(registration_scope, {})
        if registration_scope is not None
        else _providers
    )
    existing = target.get(name)
    target[name] = provider
    if existing is not None:
        logger.debug("Video gen provider '%s' re-registered (was %r)", name, type(existing).__name__)
    else:
        logger.debug("Registered video gen provider '%s' (%s)", name, type(provider).__name__)


def list_providers(*, scope: str | None = None) -> list[VideoGenProvider]:
    """Return all registered providers, sorted by name."""
    items = list(_provider_snapshot(scope).values())
    return sorted(items, key=lambda p: p.name)


def get_provider(
    name: str,
    *,
    scope: str | None = None,
) -> VideoGenProvider | None:
    """Return the provider registered under *name*, or None."""
    if not isinstance(name, str):
        return None
    return _provider_snapshot(scope).get(name.strip())


def snapshot_registration(
    name: str,
    *,
    scope: str | None = None,
) -> VideoGenProvider | None:
    target = _providers if scope is None else _plugin_providers.get(scope, {})
    return target.get(name.strip())


def restore_registration(
    name: str,
    current: VideoGenProvider,
    previous: VideoGenProvider | None,
    *,
    scope: str | None = None,
) -> bool:
    target = _providers if scope is None else _plugin_providers.setdefault(scope, {})
    key = name.strip()
    if target.get(key) is not current:
        return False
    if previous is None:
        target.pop(key, None)
    else:
        target[key] = previous
    if scope is not None and not target:
        _plugin_providers.pop(scope, None)
    return True


async def get_active_provider() -> VideoGenProvider | None:
    """Resolve the currently-active provider.

    Reads ``video_gen.provider`` from config.yaml; falls back per the
    module docstring.
    """
    configured: str | None = None
    try:
        from hermes_cli.config import load_config_readonly

        cfg = await load_config_readonly()
        section = cfg.get("video_gen") if isinstance(cfg, dict) else None
        if isinstance(section, dict):
            raw = section.get("provider")
            if isinstance(raw, str) and raw.strip():
                configured = raw.strip()
    except Exception as exc:
        logger.debug("Could not read video_gen.provider from config: %s", exc)

    snapshot = _provider_snapshot()

    if configured:
        provider = snapshot.get(configured)
        if provider is not None:
            return provider
        logger.debug(
            "video_gen.provider='%s' configured but not registered; failing closed",
            configured,
        )
        return None

    async def _is_available_safe(p: VideoGenProvider) -> bool:
        """Wrap ``is_available()`` so a buggy provider doesn't kill resolution."""
        try:
            return bool(await p.is_available())
        except Exception as exc:  # noqa: BLE001
            logger.debug("video_gen provider %s.is_available() raised %s", p.name, exc)
            return False

    # Fallback: single *available* provider — filter by is_available() so a
    # box with credentials for only one backend auto-selects it even when
    # other providers (fal/xai) register unconditionally without keys.
    # Mirrors agent/image_gen_registry.get_active_provider().
    available = [p for p in snapshot.values() if await _is_available_safe(p)]
    if len(available) == 1:
        return available[0]

    return None


def _reset_for_tests() -> None:
    """Clear the registry. **Test-only.**"""
    _providers.clear()
    _plugin_providers.clear()
