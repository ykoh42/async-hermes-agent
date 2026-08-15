"""
TTS Provider Registry
=====================

Central map of registered TTS providers. Populated by plugins at
import-time via :meth:`PluginContext.register_tts_provider`; consumed
by :mod:`tools.tts_tool` to dispatch ``text_to_speech`` tool calls to
the active plugin backend **when** the configured ``tts.provider``
name is neither a built-in nor a command-type provider.

Built-ins-always-win
--------------------
Plugin names that collide with a built-in TTS provider (``edge``,
``openai``, ``elevenlabs``, ``minimax``, ``gemini``, ``mistral``,
``xai``, ``piper``, ``kittentts``, ``neutts``) are rejected at
registration with a warning. This invariant is also re-checked at
dispatch time in :func:`tools.tts_tool._dispatch_to_plugin_provider`.

Command-providers-win-over-plugins
----------------------------------
This registry doesn't enforce the command-vs-plugin precedence — that
lives in the dispatcher, which checks for a same-name
``tts.providers.<name>: type: command`` entry before consulting the
registry. The rationale is locality: a name declared in the user's
``config.yaml`` is more specific to their setup than a plugin that
happens to be installed.
"""

from __future__ import annotations

import logging
import inspect
import sys

from agent.tts_provider import TTSProvider

logger = logging.getLogger(__name__)


# Names reserved for native built-in TTS handlers. Plugins cannot
# register a name in this set — the registration call is rejected with
# a warning. **Kept in sync with ``BUILTIN_TTS_PROVIDERS`` in
# :mod:`tools.tts_tool`** — a regression test in
# ``tests/agent/test_tts_registry.py::TestBuiltinSync`` fails if the
# two lists drift. Importing from ``tools.tts_tool`` directly would
# create a circular dependency (``tools.tts_tool`` imports
# ``agent.tts_registry`` for dispatch).
_BUILTIN_NAMES = frozenset({
    "edge",
    "elevenlabs",
    "openai",
    "minimax",
    "xai",
    "mistral",
    "gemini",
    "neutts",
    "kittentts",
    "piper",
    "deepinfra",
})


_providers: dict[str, TTSProvider] = {}
_plugin_providers: dict[object, dict[str, TTSProvider]] = {}


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


def _provider_snapshot(scope: str | None = None) -> dict[str, TTSProvider]:
    providers = dict(_providers)
    active_scope = scope if scope is not None else _plugin_scope()
    if active_scope is not None:
        providers.update(_plugin_providers.get(active_scope, {}))
    return providers


def _clear_plugin_scope(scope: object) -> None:
    _plugin_providers.pop(scope, None)


def register_provider(provider: TTSProvider, *, scope: str | None = None) -> None:
    """Register a TTS provider.

    Rejects:

    - Non-:class:`TTSProvider` instances (raises :class:`TypeError`).
    - Empty/whitespace ``.name`` (raises :class:`ValueError`).
    - Names colliding with a built-in (logs a warning, silently
      ignores — built-ins-always-win invariant).

    Re-registration (same ``name``) overwrites the previous entry and
    logs a debug message — makes hot-reload scenarios (tests, dev
    loops) behave predictably.
    """
    if not isinstance(provider, TTSProvider):
        raise TypeError(
            f"register_provider() expects a TTSProvider instance, "
            f"got {type(provider).__name__}"
        )
    name = provider.name
    if not isinstance(name, str) or not name.strip():
        raise ValueError("TTS provider .name must be a non-empty string")
    key = name.strip().lower()
    if key in _BUILTIN_NAMES:
        logger.warning(
            "TTS provider '%s' shadows a built-in name; registration ignored. "
            "Built-in TTS providers (%s) always win — pick a different name.",
            key, ", ".join(sorted(_BUILTIN_NAMES)),
        )
        return
    for method_name in (
        "is_available",
        "list_voices",
        "list_models",
        "get_setup_schema",
        "default_model",
        "default_voice",
        "synthesize",
    ):
        if not inspect.iscoroutinefunction(getattr(provider, method_name)):
            raise TypeError(f"TTS provider .{method_name} must be async")
    if not inspect.isasyncgenfunction(provider.stream):
        raise TypeError("TTS provider .stream must be an async generator")
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
    existing = target.get(key)
    target[key] = provider
    if existing is not None:
        logger.debug(
            "TTS provider '%s' re-registered (was %r)",
            key, type(existing).__name__,
        )
    else:
        logger.debug(
            "Registered TTS provider '%s' (%s)",
            key, type(provider).__name__,
        )


def list_providers(*, scope: str | None = None) -> list[TTSProvider]:
    """Return all registered providers, sorted by name."""
    items = list(_provider_snapshot(scope).values())
    return sorted(items, key=lambda p: p.name)


def get_provider(name: str, *, scope: str | None = None) -> TTSProvider | None:
    """Return the provider registered under *name*, or None.

    Name matching is case-insensitive and whitespace-tolerant — mirrors
    how ``tools.tts_tool._get_provider`` normalizes the configured
    ``tts.provider`` value.
    """
    if not isinstance(name, str):
        return None
    return _provider_snapshot(scope).get(name.strip().lower())


def snapshot_registration(
    name: str,
    *,
    scope: str | None = None,
) -> TTSProvider | None:
    target = _providers if scope is None else _plugin_providers.get(scope, {})
    return target.get(name.strip().lower())


def restore_registration(
    name: str,
    current: TTSProvider,
    previous: TTSProvider | None,
    *,
    scope: str | None = None,
) -> bool:
    target = _providers if scope is None else _plugin_providers.setdefault(scope, {})
    key = name.strip().lower()
    if target.get(key) is not current:
        return False
    if previous is None:
        target.pop(key, None)
    else:
        target[key] = previous
    if scope is not None and not target:
        _plugin_providers.pop(scope, None)
    return True


def _reset_for_tests() -> None:
    """Clear the registry. **Test-only.**"""
    _providers.clear()
    _plugin_providers.clear()
