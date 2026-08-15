"""
Web Search Provider Registry
============================

Central map of registered web providers. Populated by plugins at import-time
via :meth:`PluginContext.register_web_search_provider`; consumed by the
``web_search`` and ``web_extract`` tool wrappers in :mod:`tools.web_tools` to
dispatch each call to the active backend.

Active selection
----------------
The active provider is chosen by configuration with this precedence:

1. ``web.search_backend`` / ``web.extract_backend``
   (per-capability override).
2. ``web.backend`` (shared fallback).
3. If exactly one capability-eligible provider is registered AND available,
   use it.
4. Legacy preference order — ``firecrawl`` → ``parallel`` → ``tavily`` →
   ``exa`` → ``searxng`` → ``brave-free`` → ``ddgs`` — filtered by
   availability. Matches the historic ``tools.web_tools._get_backend()``
   candidate order so installs that never set a config key keep landing
   on the same provider they did before the plugin migration.
5. Otherwise ``None`` — the tool surfaces a helpful error pointing at
   ``hermes tools``.

The capability filter (``supports_search`` / ``supports_extract``) is
applied at every step so a search-only provider (``brave-free``)
configured as ``web.extract_backend`` correctly falls through to an
extract-capable backend.
"""

from __future__ import annotations

import logging
import sys
import threading

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


_providers: dict[str, WebSearchProvider] = {}
_plugin_providers: dict[object, dict[str, WebSearchProvider]] = {}
_lock = threading.Lock()


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


def _provider_snapshot(scope: str | None = None) -> dict[str, WebSearchProvider]:
    with _lock:
        providers = dict(_providers)
        active_scope = scope if scope is not None else _plugin_scope()
        if active_scope is not None:
            providers.update(_plugin_providers.get(active_scope, {}))
        return providers


def _clear_plugin_scope(scope: object) -> None:
    with _lock:
        _plugin_providers.pop(scope, None)


def register_provider(
    provider: WebSearchProvider,
    *,
    scope: str | None = None,
) -> None:
    """Register a web search/extract provider.

    Re-registration (same ``name``) overwrites the previous entry and logs
    a debug message — makes hot-reload scenarios (tests, dev loops) behave
    predictably.
    """
    if not isinstance(provider, WebSearchProvider):
        raise TypeError(
            f"register_provider() expects a WebSearchProvider instance, "
            f"got {type(provider).__name__}"
        )
    name = provider.name
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Web provider .name must be a non-empty string")
    with _lock:
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
        logger.debug(
            "Web provider '%s' re-registered (was %r)",
            name, type(existing).__name__,
        )
    else:
        logger.debug(
            "Registered web provider '%s' (%s)",
            name, type(provider).__name__,
        )


def list_providers(*, scope: str | None = None) -> list[WebSearchProvider]:
    """Return all registered providers, sorted by name."""
    items = list(_provider_snapshot(scope).values())
    return sorted(items, key=lambda p: p.name)


def get_provider(
    name: str,
    *,
    scope: str | None = None,
) -> WebSearchProvider | None:
    """Return the provider registered under *name*, or None."""
    if not isinstance(name, str):
        return None
    return _provider_snapshot(scope).get(name.strip())


def snapshot_registration(
    name: str,
    *,
    scope: str | None = None,
) -> WebSearchProvider | None:
    with _lock:
        target = _providers if scope is None else _plugin_providers.get(scope, {})
        return target.get(name.strip())


def restore_registration(
    name: str,
    current: WebSearchProvider,
    previous: WebSearchProvider | None,
    *,
    scope: str | None = None,
) -> bool:
    with _lock:
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


# ---------------------------------------------------------------------------
# Active-provider resolution
# ---------------------------------------------------------------------------


def _read_config_key(*path: str) -> str | None:
    """Registry fallbacks are config-free; the async tool resolves overrides."""
    return None


# Legacy preference order — preserves behaviour for users who set no
# ``web.backend`` / ``web.<capability>_backend`` config key at all. Matches
# the historic candidate order in :func:`tools.web_tools._get_backend`
# (paid providers first so existing paid setups don't get downgraded to
# a free tier on upgrade). Filtered by ``is_available()`` at walk time so
# we don't surface a provider the user has no credentials for.
_LEGACY_PREFERENCE = (
    "firecrawl",
    "parallel",
    "tavily",
    "exa",
    "searxng",
    "brave-free",
    "ddgs",
)


async def _resolve(
    configured: str | None, *, capability: str
) -> WebSearchProvider | None:
    """Resolve the active provider for a capability ("search" | "extract").

    Resolution rules (in order):

    1. **Explicit config wins, ignoring availability.** If
       ``web.{capability}_backend`` or ``web.backend`` names a registered
       provider that supports *capability*, return it even if its
       :meth:`is_available` returns False — the dispatcher will surface a
       precise "X_API_KEY is not set" error to the user instead of silently
       routing somewhere else. Matches legacy
       :func:`tools.web_tools._get_backend` behavior for configured names.

    2. **Single-provider shortcut.** When only one registered provider
       supports *capability* AND ``is_available()`` reports True, return it.

    3. **Legacy preference walk, filtered by availability.** Walk the
       :data:`_LEGACY_PREFERENCE` order (firecrawl → parallel → tavily →
       exa → searxng → brave-free → ddgs) looking for a provider whose
       ``supports_<capability>()`` is True AND whose ``is_available()`` is
       True. Matches the historic ``tools.web_tools._get_backend()``
       candidate order so users with credentials but no explicit config
       key keep landing on the same provider as pre-migration. This is
       the path that fires when no config key is set — pick the
       highest-priority backend the user actually has credentials for.

    Returns None when no provider is configured AND no available provider
    matches the legacy preference; the dispatcher then returns a "set up a
    provider" error to the user.
    """
    snapshot = _provider_snapshot()

    def _capable(p: WebSearchProvider) -> bool:
        if capability == "search":
            return bool(p.supports_search())
        if capability == "extract":
            return bool(p.supports_extract())
        return False

    async def _is_available_safe(p: WebSearchProvider) -> bool:
        """Wrap ``is_available()`` so a buggy provider doesn't kill resolution."""
        try:
            return bool(await p.is_available())
        except Exception as exc:  # noqa: BLE001
            logger.debug("provider %s.is_available() raised %s", p.name, exc)
            return False

    # 1. Explicit config wins — return regardless of is_available() so the
    #    user gets a precise downstream error message rather than a silent
    #    backend switch. Matches _get_backend() in web_tools.py.
    if configured:
        provider = snapshot.get(configured)
        if provider is not None and _capable(provider):
            return provider
        if provider is None:
            logger.debug(
                "web backend '%s' configured but not registered; falling back",
                configured,
            )
        else:
            logger.debug(
                "web backend '%s' configured but does not support '%s'; falling back",
                configured, capability,
            )

    # 2. + 3. Fallback path — filter by availability so we don't surface
    #    a provider the user has no credentials for. Without this filter,
    #    a registered-but-unconfigured provider could end up "active" on
    #    a fresh install with no API keys at all.
    eligible = []
    for provider in snapshot.values():
        if _capable(provider) and await _is_available_safe(provider):
            eligible.append(provider)
    if len(eligible) == 1:
        return eligible[0]

    for legacy in _LEGACY_PREFERENCE:
        provider = snapshot.get(legacy)
        if (
            provider is not None
            and _capable(provider)
            and await _is_available_safe(provider)
        ):
            return provider

    return None


def _disabled_web_plugin_for(configured: str | None = None, *, capability: str | None = None) -> str | None:
    """Return the plugin key of a *disabled* bundled web plugin that would
    have provided the configured backend, or None.

    When a user sets ``web.extract_backend: firecrawl`` (or the search
    equivalent) but also lists ``web-firecrawl`` in ``plugins.disabled``,
    the provider never registers and the dispatcher would otherwise emit a
    misleading "No web extract provider configured. Set web.extract_backend
    to ..." error — even though the backend IS configured correctly. The
    real fix is to re-enable the plugin. This helper detects that case so
    the dispatcher can point the user at the actual cause (issue #40190
    follow-up: pi314's disabled-plugin symptom).

    Pass ``capability`` ("search" | "extract") to resolve the configured
    name straight from ``config.yaml`` (``web.<capability>_backend`` →
    ``web.backend``). This is more reliable than the resolved backend the
    dispatcher fell back to, since a disabled provider fails the
    ``_is_backend_available`` gate and the dispatcher silently drops to
    the shared default. An explicit ``configured`` name still wins when
    given.

    Matching is by convention: bundled web plugins live under the
    ``web/<vendor>`` key with the provider ``name`` differing only in
    hyphen/underscore (``brave-free`` provider ⇄ ``web/brave_free`` key,
    ``firecrawl`` ⇄ ``web/firecrawl``). We normalize both sides before
    comparing so every bundled provider is covered without hardcoding a
    per-vendor table.
    """
    def _norm(s: str) -> str:
        return s.strip().lower().replace("-", "_")

    if not configured and capability in ("search", "extract"):
        configured = (
            _read_config_key("web", f"{capability}_backend")
            or _read_config_key("web", "backend")
        )
    if not configured:
        return None

    want = _norm(configured)
    try:
        from hermes_cli.plugins import get_plugin_manager

        pm = get_plugin_manager()
        for key, loaded in pm._plugins.items():
            if not isinstance(key, str) or not key.startswith("web/"):
                continue
            if loaded.enabled:
                continue
            if loaded.error != "disabled via config":
                continue
            vendor = key.split("/", 1)[1]
            if _norm(vendor) == want:
                return key
    except Exception as exc:  # noqa: BLE001 — diagnostics are best-effort
        logger.debug("disabled-web-plugin lookup failed: %s", exc)
    return None


async def get_active_search_provider() -> WebSearchProvider | None:
    """Resolve the currently-active web search provider.

    Reads ``web.search_backend`` (preferred) or ``web.backend`` (shared
    fallback) from config.yaml; falls back per the module docstring.
    """
    explicit = _read_config_key("web", "search_backend") or _read_config_key("web", "backend")
    return await _resolve(explicit, capability="search")


async def get_active_extract_provider() -> WebSearchProvider | None:
    """Resolve the currently-active web extract provider.

    Reads ``web.extract_backend`` (preferred) or ``web.backend`` (shared
    fallback) from config.yaml; falls back per the module docstring.
    """
    explicit = _read_config_key("web", "extract_backend") or _read_config_key("web", "backend")
    return await _resolve(explicit, capability="extract")


def _reset_for_tests() -> None:
    """Clear the registry. **Test-only.**"""
    with _lock:
        _providers.clear()
        _plugin_providers.clear()
