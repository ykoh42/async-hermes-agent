"""
Hermes Plugin System
====================

Discovers, loads, and manages plugins from four sources:

1. **Bundled plugins** – ``<repo>/plugins/<name>/`` (shipped with hermes-agent;
   ``memory/`` and ``context_engine/`` subdirs are excluded — they have their
   own discovery paths)
2. **User plugins**   – ``~/.hermes/plugins/<name>/``
3. **Project plugins** – ``./.hermes/plugins/<name>/`` (opt-in via
   ``HERMES_ENABLE_PROJECT_PLUGINS``)
4. **Pip plugins**     – packages that expose the ``hermes_agent.plugins``
   entry-point group.

Later sources override earlier ones on name collision, so a user or project
plugin with the same name as a bundled plugin replaces it.

Each directory plugin must contain a ``plugin.yaml`` manifest **and** an
``__init__.py`` with a ``register(ctx)`` function.

Lifecycle hooks
---------------
Plugins may register callbacks for any of the hooks in ``VALID_HOOKS``.
The agent core calls ``invoke_hook(name, **kwargs)`` at the appropriate
points.

Tool registration
-----------------
``PluginContext.register_tool()`` delegates to ``tools.registry.register()``
so plugin-defined tools appear alongside the built-in tools.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import hashlib
import importlib.metadata
import inspect
import logging
import os
import re
import sys
import threading
import types
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Callable, Mapping

import aiofiles
import aiofiles.os

# Bundled backend modules execute from async-preloaded source during discovery,
# but their imports of stable core ABCs/registries and an installed optional
# SDK still use Python's synchronous import protocol.  Prime that fixed import
# graph when the plugin subsystem itself is imported, before its awaited
# discovery boundary, so registration remains CPU-only on the event loop.
from agent import browser_provider as _browser_provider_bootstrap  # noqa: F401
from agent import browser_registry as _browser_registry_bootstrap  # noqa: F401
from agent import image_gen_provider as _image_gen_provider_bootstrap  # noqa: F401
from agent import image_gen_registry as _image_gen_registry_bootstrap  # noqa: F401
from agent import secret_scope as _secret_scope_bootstrap  # noqa: F401
from agent import video_gen_provider as _video_gen_provider_bootstrap  # noqa: F401
from agent import video_gen_registry as _video_gen_registry_bootstrap  # noqa: F401
from agent import web_search_provider as _web_search_provider_bootstrap  # noqa: F401
from agent import web_search_registry as _web_search_registry_bootstrap  # noqa: F401
from hermes_cli import profiles as _profiles_bootstrap  # noqa: F401
from tools import url_safety as _url_safety_bootstrap  # noqa: F401
from tools import website_policy as _website_policy_bootstrap  # noqa: F401
from tools import xai_http as _xai_http_bootstrap  # noqa: F401

# Bundled browser/web plugin packages are executed from asynchronously loaded
# source during discovery, but their package-level ``from plugins...`` imports
# still use Python's synchronous import machinery.  Prime that fixed shipped
# graph when the plugin subsystem itself is imported, before applications
# enter their event loop.  User and project plugin source remains discovered
# at the awaited profile boundary below.
from plugins.browser.browser_use import provider as _browser_use_bootstrap  # noqa: F401
from plugins.browser.browserbase import provider as _browserbase_bootstrap  # noqa: F401
from plugins.browser.firecrawl import provider as _browser_firecrawl_bootstrap  # noqa: F401
from plugins.web.brave_free import provider as _brave_free_bootstrap  # noqa: F401
from plugins.web.ddgs import provider as _ddgs_bootstrap  # noqa: F401
from plugins.web.exa import provider as _exa_bootstrap  # noqa: F401
from plugins.web.firecrawl import provider as _web_firecrawl_bootstrap  # noqa: F401
from plugins.web.parallel import provider as _parallel_provider_bootstrap  # noqa: F401
from plugins.web.searxng import provider as _searxng_bootstrap  # noqa: F401
from plugins.web.tavily import provider as _tavily_bootstrap  # noqa: F401
from plugins.web.xai import provider as _xai_web_bootstrap  # noqa: F401

try:
    import parallel as _parallel_bootstrap  # noqa: F401
except Exception:
    _parallel_bootstrap = None

from hermes_constants import get_hermes_home
from utils import env_var_enabled, fast_safe_load
from hermes_cli.config import cfg_get
from hermes_cli.middleware import OBSERVER_SCHEMA_VERSION, VALID_MIDDLEWARE
from hermes_cli.async_source_loader import (
    _load_source_module,
    _load_source_package,
    _locate_source_module,
    _unload_source_finder,
)


async def get_bundled_plugins_dir() -> Path:
    """Locate the bundled ``plugins/`` directory.

    Honours ``HERMES_BUNDLED_PLUGINS`` (set by the Nix wrapper / packaged
    installs) so read-only store paths are consulted first.  Falls back to
    the in-repo path used during development.
    """
    env_override = os.getenv("HERMES_BUNDLED_PLUGINS")
    if env_override:
        return Path(env_override)
    resolved_file = await aiofiles.os.wrap(os.path.realpath)(__file__)
    return Path(resolved_file).parent.parent / "plugins"

try:
    import yaml
except ImportError:  # pragma: no cover – yaml is optional at import time
    yaml = None  # type: ignore[assignment]


class PluginToolOverrideError(PermissionError):
    """Raised when a plugin attempts to override a built-in tool without
    operator opt-in via ``plugins.entries.<plugin_id>.allow_tool_override``.
    """


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plugin developer debug logging
# ---------------------------------------------------------------------------
#
# Set ``HERMES_PLUGINS_DEBUG=1`` to surface verbose plugin-discovery logs to
# stderr in addition to ~/.hermes/logs/agent.log. Aimed at plugin authors
# trying to figure out why their plugin isn't showing up: which directories
# were scanned, which manifests parsed, which plugins were skipped (and why),
# what each ``register(ctx)`` call registered, and full tracebacks on load
# failure.
#
# The env var is read once at import time; tests that need to flip it
# mid-process can call ``_install_plugin_debug_handler(force=True)``.

_PLUGINS_DEBUG = os.getenv("HERMES_PLUGINS_DEBUG", "").strip().lower() in {
    "1", "true", "yes", "on",
}
_DEBUG_HANDLER_INSTALLED = False


def _install_plugin_debug_handler(force: bool = False) -> None:
    """When HERMES_PLUGINS_DEBUG is on, tee plugin logs to stderr at DEBUG.

    Idempotent: only attaches the handler once per process unless ``force``
    is passed. Does not touch the root logger or other Hermes loggers.
    """
    global _DEBUG_HANDLER_INSTALLED, _PLUGINS_DEBUG
    if force:
        _PLUGINS_DEBUG = os.getenv("HERMES_PLUGINS_DEBUG", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
    if not _PLUGINS_DEBUG or _DEBUG_HANDLER_INSTALLED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("[plugins] %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    # Don't double-emit through the root logger when the central logging
    # config also writes to stderr. agent.log still captures everything.
    logger.propagate = True
    _DEBUG_HANDLER_INSTALLED = True
    logger.debug(
        "HERMES_PLUGINS_DEBUG=1 — verbose plugin discovery logging enabled"
    )


_install_plugin_debug_handler()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_HOOKS: set[str] = {
    "pre_tool_call",
    "post_tool_call",
    "transform_terminal_output",
    "transform_tool_result",
    # Transform LLM output before it's returned to the user.
    # Plugins return a string to replace the response text, or None/empty to leave unchanged.
    # First non-None string wins. Useful for vocabulary/personality transformation.
    "transform_llm_output",
    "pre_llm_call",
    "post_llm_call",
    # Verification-loop gate. Fired once per turn when the agent has edited code
    # and is about to verify/finish (after the verify-on-stop guard). A callback
    # may keep the agent going — run a check, defer it, tidy the diff — instead
    # of stopping by returning:
    #   {"action": "continue", "message": "<follow-up instruction>"}
    # The Claude-Code Stop shape {"decision": "block", "reason": "..."} (block
    # the stop == keep going) is accepted too. Anything else lets the turn
    # finish. Hermes' shipped guidance lives in the evidence-based
    # verification-stop nudge; this hook is for user/plugin policy and is
    # bounded by agent.max_verify_nudges.
    "pre_verify",
    "pre_api_request",
    "post_api_request",
    "api_request_error",
    # Pure provider-error classification transform.  Unlike the awaited
    # lifecycle hooks, this hook is consumed by the synchronous classifier;
    # only already-loaded synchronous callbacks are eligible there.
    "transform_api_error_classification",
    # Fire-and-forget stream observers.  These are queued by
    # agent.plugin_stream_hooks and cannot transform model output.
    "on_stream_start",
    "on_stream_delta",
    "on_stream_end",
    "on_interim_message",
    "on_session_start",
    "on_session_end",
    "on_session_finalize",
    "on_session_reset",
    "subagent_start",
    "subagent_stop",
    "pre_approval_request",
    "post_approval_response",
}

ENTRY_POINTS_GROUP = "hermes_agent.plugins"

_NS_PARENT = "hermes_plugins"


_ACTIVE_PLUGIN_MANAGER: contextvars.ContextVar[PluginManager | None] = (
    contextvars.ContextVar("active_plugin_manager", default=None)
)
_PLUGIN_REGISTRATION_TARGET: contextvars.ContextVar[
    tuple[PluginManager, str] | None
] = contextvars.ContextVar("plugin_registration_target", default=None)
_PLUGIN_MANAGERS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, PluginManager]
] = weakref.WeakKeyDictionary()
_PLUGIN_MANAGER_INSTANCES: weakref.WeakSet[PluginManager] = weakref.WeakSet()
_PLUGIN_MANAGER_GUARD = threading.RLock()


def _current_plugin_registry_scope(*, registration: bool = False) -> object | None:
    """Return the active profile overlay without importing plugin consumers.

    During registration only non-bundled plugins receive an overlay. Bundled
    registrations intentionally remain process-shared, matching upstream's
    built-in registry behavior.
    """
    target = _PLUGIN_REGISTRATION_TARGET.get()
    if target is not None:
        manager, source = target
        if registration and source not in {"user", "project"}:
            return None
        return manager._registry_scope
    if registration:
        return None
    manager = _ACTIVE_PLUGIN_MANAGER.get()
    return manager._registry_scope if manager is not None else None


def _plugin_registry_scope_for_module(module_name: str) -> object | None:
    """Resolve a delayed registration by its defining plugin namespace."""
    registry_module = sys.modules.get("tools.registry")
    registry = getattr(registry_module, "registry", None)
    namespace_for = getattr(registry, "_plugin_namespace_for_module", None)
    if not callable(namespace_for):
        return None
    namespace = namespace_for(module_name)
    if namespace is None:
        if module_name.startswith(f"{_NS_PARENT}."):
            raise RuntimeError(
                f"Plugin module {module_name!r} is no longer attached to an "
                "active plugin manager"
            )
        return None
    return registry._plugin_module_scopes.get(namespace)


async def _canonical_plugin_home() -> tuple[str, Path]:
    """Resolve the current task's canonical plugin home natively async."""
    realpath = aiofiles.os.wrap(os.path.realpath)
    canonical = os.path.normcase(await realpath(str(get_hermes_home())))
    return canonical, Path(canonical)


def _plugin_namespace(loop: asyncio.AbstractEventLoop, profile_key: str) -> str:
    digest = hashlib.sha256(f"{id(loop)}\0{profile_key}".encode()).hexdigest()[:16]
    return f"{_NS_PARENT}._profile_{digest}"


def _clear_profile_registries(scope: object) -> None:
    """Drop state owned by one user-plugin profile from loaded registries."""
    registry_modules = (
        "tools.registry",
        "agent.image_gen_registry",
        "agent.video_gen_registry",
        "agent.web_search_registry",
        "agent.browser_registry",
        "agent.tts_registry",
        "agent.secret_sources.registry",
    )
    for module_name in registry_modules:
        module = sys.modules.get(module_name)
        clear = getattr(module, "_clear_plugin_scope", None)
        if not callable(clear):
            clear = getattr(getattr(module, "registry", None), "_clear_plugin_scope", None)
        if callable(clear):
            clear(scope)


def _env_enabled(name: str) -> bool:
    """Return True when an env var is set to a truthy opt-in value."""
    return env_var_enabled(name)


async def _get_disabled_plugins(config: dict | None = None) -> set:
    """Read the disabled plugins list from config.yaml.

    Kept for backward compat and explicit deny-list semantics. A plugin
    name in this set will never load, even if it appears in
    ``plugins.enabled``.
    """
    try:
        if config is None:
            from hermes_cli.config import load_config_readonly

            config = await load_config_readonly()
        disabled = cfg_get(config, "plugins", "disabled", default=[])
        return set(disabled) if isinstance(disabled, list) else set()
    except Exception:
        return set()


async def _get_enabled_plugins(config: dict | None = None) -> set | None:
    """Read the enabled-plugins allow-list from config.yaml.

    Plugins are opt-in by default — only plugins whose name appears in
    this set are loaded. Returns:

    * ``None`` — the key is missing or malformed. Callers should treat
      this as "nothing enabled yet" (the opt-in default); the first
      ``migrate_config`` run populates the key with a grandfathered set
      of currently-installed user plugins so existing setups don't
      break on upgrade.
    * ``set()`` — an empty list was explicitly set; nothing loads.
    * ``set(...)`` — the concrete allow-list.
    """
    try:
        if config is None:
            from hermes_cli.config import load_config_readonly

            config = await load_config_readonly()
        plugins_cfg = config.get("plugins")
        if not isinstance(plugins_cfg, dict):
            return None
        if "enabled" not in plugins_cfg:
            return None
        enabled = plugins_cfg.get("enabled")
        if not isinstance(enabled, list):
            return None
        return set(enabled)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

_VALID_PLUGIN_KINDS: set[str] = {"standalone", "backend", "exclusive", "platform", "model-provider"}

# System-prompt sections are intentionally bounded and rendered only once for
# a new session.  The complete prompt is persisted by the async state layer;
# resumed agents reconstruct the section bytes from that stored prompt rather
# than invoking plugin code again.
SYSTEM_PROMPT_SECTION_POSITIONS = frozenset({"after_memory"})
DEFAULT_SYSTEM_PROMPT_SECTION_MAX_CHARS = 4_000
MAX_SYSTEM_PROMPT_SECTION_CHARS = 4_000
MAX_SYSTEM_PROMPT_SECTIONS = 32
MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS = 8_000
_SYSTEM_PROMPT_SECTION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SYSTEM_PROMPT_SECTION_HEADING_PREFIX = "## Plugin Context: "
PLUGIN_SECTIONS_START = "<!-- hermes-plugin-sections:start -->"
PLUGIN_SECTIONS_END = "<!-- hermes-plugin-sections:end -->"


def is_valid_system_prompt_section_id(value: Any) -> bool:
    return isinstance(value, str) and bool(_SYSTEM_PROMPT_SECTION_ID_RE.fullmatch(value))


@dataclass(frozen=True)
class PluginSystemPromptSection:
    id: str
    content: str | Callable[[dict[str, Any]], str]
    position: str
    max_chars: int
    plugin: str


@dataclass(frozen=True)
class RenderedPluginSystemPromptSection:
    id: str
    content: str
    position: str
    plugin: str


def format_system_prompt_section(section_id: str, content: str) -> str:
    return (
        f"{_SYSTEM_PROMPT_SECTION_HEADING_PREFIX}{section_id}\n"
        f"<!-- hermes-plugin-section-chars:{len(content)} -->\n\n"
        f"{content}"
    )


def format_system_prompt_sections(sections: list[RenderedPluginSystemPromptSection]) -> str:
    if not sections:
        return ""
    blocks = [format_system_prompt_section(item.id, item.content) for item in sections]
    return f"{PLUGIN_SECTIONS_START}\n" + "\n\n".join(blocks) + f"\n{PLUGIN_SECTIONS_END}"


@dataclass
class PluginRegistration:
    """Small disposable handle for a plugin-owned registration."""

    kind: str
    key: str
    release: Callable[[], None]
    _disposed: bool = field(default=False, init=False, repr=False)

    @property
    def active(self) -> bool:
        return not self._disposed

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        self.release()


@dataclass
class PluginManifest:
    """Parsed representation of a plugin.yaml manifest."""

    name: str
    version: str = ""
    description: str = ""
    author: str = ""
    requires_env: list[str | dict[str, Any]] = field(default_factory=list)
    provides_tools: list[str] = field(default_factory=list)
    provides_hooks: list[str] = field(default_factory=list)
    source: str = ""        # "user", "project", or "entrypoint"
    path: str | None = None
    # Plugin kind — see plugins.py module docstring for semantics.
    # ``standalone`` (default): hooks/tools of its own; opt-in via
    #                           ``plugins.enabled``.
    # ``backend``: pluggable backend for an existing core tool (e.g.
    #              image_gen). Built-in (bundled) backends auto-load;
    #              user-installed still gated by ``plugins.enabled``.
    # ``exclusive``: category with exactly one active provider (memory).
    #              Selection via ``<category>.provider`` config key; the
    #              category's own discovery system handles loading and the
    #              general scanner skips these.
    kind: str = "standalone"
    # Registry key — path-derived, used by ``plugins.enabled``/``disabled``
    # lookups and by ``hermes plugins list``. For a flat plugin at
    # ``plugins/disk-cleanup/`` the key is ``disk-cleanup``; for a nested
    # category plugin at ``plugins/image_gen/openai/`` the key is
    # ``image_gen/openai``. When empty, falls back to ``name``.
    key: str = ""


@dataclass
class LoadedPlugin:
    """Runtime state for a single loaded plugin."""

    manifest: PluginManifest
    module: types.ModuleType | None = None
    tools_registered: list[str] = field(default_factory=list)
    hooks_registered: list[str] = field(default_factory=list)
    middleware_registered: list[str] = field(default_factory=list)
    commands_registered: list[str] = field(default_factory=list)
    enabled: bool = False
    error: str | None = None


# ---------------------------------------------------------------------------
# PluginContext  – handed to each plugin's ``register()`` function
# ---------------------------------------------------------------------------

class PluginContext:
    """Facade given to plugins so they can register tools and hooks."""

    def __init__(self, manifest: PluginManifest, manager: PluginManager):
        self.manifest = manifest
        self._manager = manager
        self._profile_context: tuple[Path | None, Path | None] | None = None
        # ``register_skill`` is intentionally kept synchronous for the public
        # plugin contract.  The manager marks its async discovery contexts so
        # validation can cross an awaited filesystem boundary after the
        # callback returns; direct compatibility callers retain the upstream
        # immediate validation behaviour.
        self._defer_skill_validation = False
        self._deferred_skill_paths: list[tuple[str, Path]] = []
        # Lazy-built host-owned LLM facade — see ctx.llm property below.
        self._llm: Any = None
        self._subagent_lifecycle: Any = None

    @contextlib.contextmanager
    def _registration_scope(self):
        """Attribute delayed context-facade registrations to this profile."""
        token = _PLUGIN_REGISTRATION_TARGET.set((self._manager, self.manifest.source))
        try:
            yield
        finally:
            _PLUGIN_REGISTRATION_TARGET.reset(token)

    # -- host-owned LLM access ----------------------------------------------

    @property
    def llm(self) -> Any:
        """Return the plugin's :class:`agent.plugin_llm.PluginLlm` facade.

        Lets trusted plugins run host-owned chat or structured completions
        against the user's active model and auth without bringing their
        own provider keys. Override capability (model, agent id, auth
        profile) is fail-closed by default and gated through
        ``plugins.entries.<plugin_id>.llm.*`` config keys.

        See :mod:`agent.plugin_llm` for the full surface."""
        if self._llm is None:
            from agent.plugin_llm import PluginLlm
            plugin_id = self.manifest.key or self.manifest.name
            self._llm = PluginLlm(plugin_id=plugin_id)
        return self._llm

    @property
    def subagent_lifecycle(self) -> Any:
        """Return the public, plugin-safe subagent lifecycle service.

        The service only resolves the active host-owned parent agent when a
        child is launched. Plugins receive serializable handles and immutable
        snapshots; they never receive a live agent or a private registry.
        """
        if self._subagent_lifecycle is None:
            from agent.subagent_lifecycle import (
                SubagentLifecycleService,
                get_active_subagent_parent,
            )
            self._subagent_lifecycle = SubagentLifecycleService(
                get_active_subagent_parent
            )
        return self._subagent_lifecycle

    # -- profile awareness --------------------------------------------------

    @property
    def profile_name(self) -> str:
        """Return the active Hermes profile name (e.g. ``"default"``).

        The async discovery boundary resolves the process root once; each
        access classifies the current context-local ``HERMES_HOME`` without
        filesystem I/O. This keeps the property synchronous and dynamic in
        interactive, gateway, and worker sessions.

        Returns ``"default"`` for the default profile, the profile id when
        running under ``~/.hermes/profiles/<name>``, or ``"custom"`` when
        ``HERMES_HOME`` points somewhere unrecognized.
        """
        if self._profile_context is None:
            raise RuntimeError("PluginContext profile resolution is not initialized")
        from hermes_cli.profiles import _profile_name_from_context

        default_root, cwd = self._profile_context
        try:
            return _profile_name_from_context(get_hermes_home(), default_root, cwd)
        except Exception:
            return "default"

    # -- tool registration --------------------------------------------------

    def register_tool(
        self,
        name: str,
        toolset: str,
        schema: dict,
        handler: Callable,
        check_fn: Callable | None = None,
        requires_env: list | None = None,
        is_async: bool = False,
        description: str = "",
        emoji: str = "",
        override: bool = False,
    ) -> None:
        """Register a tool in the active registry **and** track it as plugin-provided.

        Pass ``override=True`` to replace an existing built-in tool with the
        same name (e.g. swap the default ``browser_navigate`` for a custom
        CDP-backed implementation). Without it, attempting to register a name
        already claimed by a different toolset is rejected.

        ``override=True`` against a built-in tool requires the operator to
        opt in via ``plugins.entries.<plugin_id>.allow_tool_override: true``
        in config.yaml — mirrors the trust gate pattern used for
        ``ctx.llm`` provider/model overrides (#23194). Without that gate,
        any enabled plugin could silently replace a privileged built-in
        like ``shell_exec`` or ``write_file`` and exfiltrate everything
        the model invokes through it.
        """
        if override and not self._tool_override_allowed(name):
            plugin_id = self.manifest.key or self.manifest.name
            raise PluginToolOverrideError(
                f"Plugin {self.manifest.name!r} cannot override built-in tool "
                f"{name!r}. Set "
                f"plugins.entries.{plugin_id}.allow_tool_override: true "
                f"in config.yaml to allow this plugin to replace built-in tools."
            )

        from tools.registry import registry

        with self._registration_scope():
            registry.register(
                name=name,
                toolset=toolset,
                schema=schema,
                handler=handler,
                check_fn=check_fn,
                requires_env=requires_env,
                is_async=is_async,
                description=description,
                emoji=emoji,
                override=override,
            )
        self._manager._plugin_tool_names.add(name)
        logger.debug(
            "Plugin %s registered tool: %s%s",
            self.manifest.name, name, " (override)" if override else "",
        )

    # -- override trust gate ------------------------------------------------

    def _tool_override_allowed(self, tool_name: str) -> bool:
        """Return True if this plugin is configured to override built-in tools.

        Bundled plugins (shipped with Hermes core) are trusted by default —
        an override there is a deliberate maintainer choice, not a third-party
        plugin trying to elevate privilege. For every other source, require
        ``allow_tool_override: true`` under
        ``plugins.entries.<plugin_id>`` in config.yaml.
        """
        source = getattr(self.manifest, "source", "") or ""
        if source == "bundled":
            return True
        cfg = self._manager._config
        plugin_id = self.manifest.key or self.manifest.name
        entries = (cfg.get("plugins") or {}).get("entries") or {}
        entry = entries.get(plugin_id) or {}
        return bool(entry.get("allow_tool_override", False))

    # -- message injection --------------------------------------------------

    def inject_message(self, content: str, role: str = "user") -> bool:
        """Inject a message into the active conversation.

        If the agent is idle (waiting for user input), this starts a new turn.
        If the agent is running, this interrupts and injects the message.

        This enables plugins (e.g. remote control viewers, messaging bridges)
        to send messages into the conversation from external sources.

        Returns True if the message was queued successfully.
        """
        cli = self._manager._cli_ref
        if cli is None:
            logger.warning("inject_message: no CLI reference (not available in gateway mode)")
            return False

        msg = content if role == "user" else f"[{role}] {content}"

        if getattr(cli, "_agent_running", False):
            # Agent is mid-turn — interrupt with the message
            cli._interrupt_queue.put(msg)
        else:
            # Agent is idle — queue as next input
            cli._pending_input.put(msg)
        return True

    # -- slash command registration -------------------------------------------

    def register_command(
        self,
        name: str,
        handler: Callable,
        description: str = "",
        args_hint: str = "",
    ) -> None:
        """Register a slash command (e.g. ``/lcm``) available in CLI and gateway sessions.

        The handler signature is ``fn(raw_args: str) -> str | None``.
        It may also be an async callable — the gateway dispatch handles both.

        ``args_hint`` is an optional short string (e.g. ``"<file>"`` or
        ``"dias:7 formato:json"``) used by gateway adapters to surface the
        command with an argument field — for example Discord's native slash
        command picker. Plugin commands without ``args_hint`` register as
        parameterless in Discord and still accept trailing text when invoked
        as free-form chat.

        Names are unique within the plugin command registry; later
        registrations replace earlier entries with the same normalized name.
        """
        clean = name.lower().strip().lstrip("/").replace(" ", "-")
        if not clean:
            logger.warning(
                "Plugin '%s' tried to register a command with an empty name.",
                self.manifest.name,
            )
            return

        self._manager._plugin_commands[clean] = {
            "handler": handler,
            "description": description or "Plugin command",
            "plugin": self.manifest.name,
            "args_hint": (args_hint or "").strip(),
        }
        logger.debug("Plugin %s registered command: /%s", self.manifest.name, clean)

    # -- tool dispatch -------------------------------------------------------

    async def dispatch_tool(self, tool_name: str, args: dict, **kwargs) -> str:
        """Dispatch a tool call through the registry, with parent agent context.

        This is the public interface for plugin slash commands that need to call
        tools like ``delegate_task`` without reaching into framework internals.
        The parent agent (if available) is resolved automatically — plugins never
        need to access the agent directly.

        Args:
            tool_name: Registry name of the tool (e.g. ``"delegate_task"``).
            args: Tool arguments dict (same as what the model would pass).
            **kwargs: Extra keyword args forwarded to the registry dispatch.

        Returns:
            JSON string from the tool handler (same format as model tool calls).
        """
        from tools.registry import registry

        # Wire up parent agent context when available (CLI mode).
        # In gateway mode _cli_ref is None — tools degrade gracefully
        # (workspace hints fall back to TERMINAL_CWD, no spinner).
        if "parent_agent" not in kwargs:
            cli = self._manager._cli_ref
            agent = getattr(cli, "agent", None) if cli else None
            if agent is not None:
                kwargs["parent_agent"] = agent

        return await registry.dispatch(tool_name, args, **kwargs)

    # -- context engine registration -----------------------------------------

    def register_context_engine(self, engine) -> None:
        """Register a context engine to replace the built-in ContextCompressor.

        Only one context engine plugin is allowed. If a second plugin tries
        to register one, it is rejected with a warning.

        The engine must be an instance of ``agent.context_engine.ContextEngine``.
        """
        if self._manager._context_engine is not None:
            logger.warning(
                "Plugin '%s' tried to register a context engine, but one is "
                "already registered. Only one context engine plugin is allowed.",
                self.manifest.name,
            )
            return
        # Defer the import to avoid circular deps at module level
        from agent.context_engine import ContextEngine
        if not isinstance(engine, ContextEngine):
            logger.warning(
                "Plugin '%s' tried to register a context engine that does not "
                "inherit from ContextEngine. Ignoring.",
                self.manifest.name,
            )
            return
        self._manager._context_engine = engine
        logger.info(
            "Plugin '%s' registered context engine: %s",
            self.manifest.name, engine.name,
        )

    # -- image gen provider registration ------------------------------------

    def register_image_gen_provider(self, provider) -> None:
        """Register an image generation backend.

        ``provider`` must be an instance of
        :class:`agent.image_gen_provider.ImageGenProvider`. The
        ``provider.name`` attribute is what ``image_gen.provider`` in
        ``config.yaml`` matches against when routing ``image_generate``
        tool calls.
        """
        from agent.image_gen_provider import ImageGenProvider
        from agent.image_gen_registry import register_provider

        if not isinstance(provider, ImageGenProvider):
            logger.warning(
                "Plugin '%s' tried to register an image_gen provider that does "
                "not inherit from ImageGenProvider. Ignoring.",
                self.manifest.name,
            )
            return
        with self._registration_scope():
            register_provider(provider)
        logger.info(
            "Plugin '%s' registered image_gen provider: %s",
            self.manifest.name, provider.name,
        )

    # -- video gen provider registration -------------------------------------

    def register_video_gen_provider(self, provider) -> None:
        """Register a video generation backend.

        ``provider`` must be an instance of
        :class:`agent.video_gen_provider.VideoGenProvider`. The
        ``provider.name`` attribute is what ``video_gen.provider`` in
        ``config.yaml`` matches against when routing ``video_generate``
        tool calls.
        """
        from agent.video_gen_provider import VideoGenProvider
        from agent.video_gen_registry import register_provider as _register_video_provider

        if not isinstance(provider, VideoGenProvider):
            logger.warning(
                "Plugin '%s' tried to register a video_gen provider that does "
                "not inherit from VideoGenProvider. Ignoring.",
                self.manifest.name,
            )
            return
        with self._registration_scope():
            _register_video_provider(provider)
        logger.info(
            "Plugin '%s' registered video_gen provider: %s",
            self.manifest.name, provider.name,
        )

    # -- web search/extract provider registration ----------------------------

    def register_web_search_provider(self, provider) -> None:
        """Register a web search/extract backend.

        ``provider`` must be an instance of
        :class:`agent.web_search_provider.WebSearchProvider`. The
        ``provider.name`` attribute is what ``web.search_backend`` /
        ``web.extract_backend`` / ``web.backend`` in ``config.yaml``
        matches against when routing ``web_search`` / ``web_extract``
        tool calls.
        """
        from agent.web_search_provider import WebSearchProvider
        from agent.web_search_registry import register_provider as _register_web_provider

        if not isinstance(provider, WebSearchProvider):
            logger.warning(
                "Plugin '%s' tried to register a web provider that does "
                "not inherit from WebSearchProvider. Ignoring.",
                self.manifest.name,
            )
            return
        with self._registration_scope():
            _register_web_provider(provider)
        logger.info(
            "Plugin '%s' registered web provider: %s",
            self.manifest.name, provider.name,
        )

    # -- browser provider registration ---------------------------------------

    def register_browser_provider(self, provider) -> None:
        """Register a cloud browser backend.

        ``provider`` must be an instance of
        :class:`agent.browser_provider.BrowserProvider`. The
        ``provider.name`` attribute is what ``browser.cloud_provider`` in
        ``config.yaml`` matches against when routing cloud-mode
        ``browser_*`` tool calls.

        Mirrors :meth:`register_web_search_provider` exactly — same
        registration shape, same gating, same logging. The browser
        subsystem's dispatcher (:func:`tools.browser_tool._get_cloud_provider`)
        consults the registry built up by these calls.
        """
        from agent.browser_provider import BrowserProvider
        from agent.browser_registry import register_provider as _register_browser_provider

        if not isinstance(provider, BrowserProvider):
            logger.warning(
                "Plugin '%s' tried to register a browser provider that does "
                "not inherit from BrowserProvider. Ignoring.",
                self.manifest.name,
            )
            return
        with self._registration_scope():
            _register_browser_provider(provider)
        logger.info(
            "Plugin '%s' registered browser provider: %s",
            self.manifest.name, provider.name,
        )

    # -- TTS provider registration -------------------------------------------

    def register_tts_provider(self, provider) -> None:
        """Register a text-to-speech backend under the upstream plugin API."""
        from agent.tts_provider import TTSProvider
        from agent.tts_registry import register_provider

        if not isinstance(provider, TTSProvider):
            logger.warning(
                "Plugin '%s' tried to register a TTS provider that does not "
                "inherit from TTSProvider. Ignoring.",
                self.manifest.name,
            )
            return
        with self._registration_scope():
            register_provider(provider)
        logger.info(
            "Plugin '%s' registered TTS provider: %s",
            self.manifest.name,
            provider.name,
        )

    # -- secret source registration ---------------------------------------

    def register_secret_source(self, source) -> None:
        """Register a native-async external secret-manager backend."""
        from agent.secret_sources.base import SecretSource
        from agent.secret_sources.registry import register_source

        if not isinstance(source, SecretSource):
            logger.warning(
                "Plugin '%s' tried to register a secret source that does "
                "not inherit from SecretSource. Ignoring.",
                self.manifest.name,
            )
            return
        with self._registration_scope():
            registered = register_source(source)
        if registered:
            logger.info(
                "Plugin '%s' registered secret source: %s",
                self.manifest.name,
                source.name,
            )

    # -- hook registration --------------------------------------------------

    # -- auxiliary task registration ---------------------------------------

    def register_auxiliary_task(
        self,
        key: str,
        *,
        display_name: str,
        description: str,
        defaults: dict[str, Any] | None = None,
    ) -> None:
        """Register a plugin-defined auxiliary LLM task.

        Auxiliary tasks are LLM-backed side jobs (vision analysis, web extraction,
        compression, smart-approval, etc.) that route through ``auxiliary_client.py``.
        Each task has its own ``auxiliary.<key>`` config block where users can
        pin a provider/model independent of the main chat model.

        Plugins use this to declare their own auxiliary tasks without touching
        core files. After registration, the task:

          - Appears in the ``hermes model → Configure auxiliary models`` picker
          - Has its provider/model/base_url/api_key bridged from config.yaml to
            ``AUXILIARY_<KEY_UPPER>_*`` env vars at gateway startup
          - Gets default routing fields (provider="auto", model="", etc.) merged
            into loaded configs so ``cfg.get("auxiliary", {}).get(key)`` works

        Args:
            key: stable task key (snake_case). Used in config ``auxiliary.<key>``
                and env vars ``AUXILIARY_<KEY_UPPER>_*``. Must not shadow a
                built-in task key (vision, compression, web_extract, approval,
                mcp, title_generation, skills_hub, curator).
            display_name: human-readable name shown in the picker.
            description: short one-line description shown next to the name.
            defaults: optional dict of default routing fields. Recognized keys:
                ``provider`` (default "auto"), ``model`` (default ""),
                ``base_url`` (default ""), ``api_key`` (default ""),
                ``timeout`` (default 60), ``extra_body`` (default {}),
                plus any task-specific extras (e.g. ``download_timeout``).
                Unknown keys are preserved verbatim — the plugin owns the
                schema for its own task.

        Raises:
            ValueError: if *key* is empty, contains invalid characters, or
                shadows a built-in auxiliary task key.

        Example:
            ctx.register_auxiliary_task(
                key="memory_retain_filter",
                display_name="Memory retain filter",
                description="hindsight pre-retain dedup/extract",
                defaults={"provider": "auto", "timeout": 30},
            )
        """
        # Validate key shape
        if not key or not isinstance(key, str):
            raise ValueError(
                f"Plugin '{self.manifest.name}' tried to register auxiliary task "
                f"with invalid key {key!r}"
            )
        if not all(c.isalnum() or c == "_" for c in key):
            raise ValueError(
                f"Plugin '{self.manifest.name}' auxiliary task key {key!r} "
                f"must contain only alphanumeric characters and underscores"
            )

        # This library build has no CLI-owned auxiliary-task catalog. Core
        # tasks are registered by their runtime owners; plugin keys only need
        # collision checks against other plugin registrations here.
        builtin_keys: set[str] = set()
        if key in builtin_keys:
            raise ValueError(
                f"Plugin '{self.manifest.name}' cannot register auxiliary task "
                f"{key!r} — that key is reserved for a built-in task. "
                f"Pick a plugin-namespaced key (e.g. '{self.manifest.name}_{key}')."
            )

        # Reject duplicate registrations across plugins
        existing = self._manager._aux_tasks.get(key)
        if existing is not None and existing.get("plugin") != self.manifest.name:
            raise ValueError(
                f"Plugin '{self.manifest.name}' cannot register auxiliary task "
                f"{key!r} — already registered by plugin "
                f"'{existing.get('plugin')}'"
            )

        # Normalize defaults — plugin owns the schema, but we ensure routing
        # fields exist with sensible types so consumers don't crash.
        merged_defaults: dict[str, Any] = {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 60,
            "extra_body": {},
        }
        if defaults:
            for k, v in defaults.items():
                merged_defaults[k] = v

        self._manager._aux_tasks[key] = {
            "key": key,
            "display_name": display_name,
            "description": description,
            "defaults": merged_defaults,
            "plugin": self.manifest.name,
        }
        logger.debug(
            "Plugin %s registered auxiliary task: %s (%s)",
            self.manifest.name,
            key,
            display_name,
        )

    def register_system_prompt_section(
        self,
        id: str,
        content: str | Callable[[dict[str, Any]], str],
        *,
        position: str = "after_memory",
        max_chars: int = DEFAULT_SYSTEM_PROMPT_SECTION_MAX_CHARS,
    ) -> PluginRegistration:
        """Register bounded prompt context frozen into new session prompts."""
        if not is_valid_system_prompt_section_id(id):
            raise ValueError(
                "system prompt section id must be 1-128 lowercase characters "
                "using letters, numbers, '.', '_', or '-'"
            )
        if not isinstance(content, str) and not callable(content):
            raise TypeError("system prompt section content must be a string or callable")
        if position not in SYSTEM_PROMPT_SECTION_POSITIONS:
            raise ValueError(
                "system prompt section position must be one of: "
                + ", ".join(sorted(SYSTEM_PROMPT_SECTION_POSITIONS))
            )
        if (
            isinstance(max_chars, bool)
            or not isinstance(max_chars, int)
            or not 0 < max_chars <= MAX_SYSTEM_PROMPT_SECTION_CHARS
        ):
            raise ValueError(
                "system prompt section max_chars must be between 1 and "
                f"{MAX_SYSTEM_PROMPT_SECTION_CHARS}"
            )
        existing = self._manager._system_prompt_sections.get(id)
        if existing is not None:
            raise ValueError(
                f"system prompt section {id!r} is already registered by "
                f"plugin {existing.plugin!r}"
            )
        section = PluginSystemPromptSection(
            id=id,
            content=content,
            position=position,
            max_chars=max_chars,
            plugin=self.manifest.key or self.manifest.name,
        )
        self._manager._system_prompt_sections[id] = section

        def release() -> None:
            if self._manager._system_prompt_sections.get(id) is section:
                self._manager._system_prompt_sections.pop(id, None)

        return PluginRegistration("system_prompt_section", id, release)

    def register_hook(self, hook_name: str, callback: Callable) -> None:
        """Register a lifecycle hook callback.

        Unknown hook names produce a warning but are still stored so
        forward-compatible plugins don't break.
        """
        if hook_name not in VALID_HOOKS:
            logger.warning(
                "Plugin '%s' registered unknown hook '%s' "
                "(valid: %s)",
                self.manifest.name,
                hook_name,
                ", ".join(sorted(VALID_HOOKS)),
            )
        self._manager._hooks.setdefault(hook_name, []).append(callback)
        logger.debug("Plugin %s registered hook: %s", self.manifest.name, hook_name)

    # -- middleware registration -------------------------------------------

    def register_middleware(self, kind: str, callback: Callable) -> None:
        """Register a behavior-changing middleware callback.

        Middleware is separate from observer hooks: request middleware may
        rewrite the effective payload, and execution middleware may wrap the
        real callback. Unknown kinds are stored for forward compatibility but
        warned so plugin authors can catch typos.
        """
        if kind not in VALID_MIDDLEWARE:
            logger.warning(
                "Plugin '%s' registered unknown middleware '%s' "
                "(valid: %s)",
                self.manifest.name,
                kind,
                ", ".join(sorted(VALID_MIDDLEWARE)),
            )
        self._manager._middleware.setdefault(kind, []).append(callback)
        logger.debug("Plugin %s registered middleware: %s", self.manifest.name, kind)

    # -- skill registration -------------------------------------------------

    def register_skill(
        self,
        name: str,
        path: Path,
        description: str = "",
        frontmatter: Mapping[str, Any] | None = None,
    ) -> PluginRegistration:
        """Register a read-only skill provided by this plugin.

        The skill becomes resolvable as ``'<plugin_name>:<name>'`` via
        ``skill_view()``.  It does **not** enter the flat
        ``~/.hermes/skills/`` tree and is **not** listed in the system
        prompt's ``<available_skills>`` index — plugin skills are
        opt-in explicit loads only.

        Raises:
            ValueError: if *name* contains ``':'`` or invalid characters.
            FileNotFoundError: if *path* does not exist.
        """
        from agent.skill_utils import _NAMESPACE_RE

        if ":" in name:
            raise ValueError(
                f"Skill name '{name}' must not contain ':' "
                f"(the namespace is derived from the plugin name "
                f"'{self.manifest.name}' automatically)."
            )
        if not name or not _NAMESPACE_RE.match(name):
            raise ValueError(
                f"Invalid skill name '{name}'. Must match [a-zA-Z0-9_-]+."
            )
        qualified = f"{self.manifest.name}:{name}"
        if self._defer_skill_validation:
            self._deferred_skill_paths.append((qualified, path))
        elif not path.exists():
            raise FileNotFoundError(f"SKILL.md not found at {path}")

        self._manager._plugin_skills[qualified] = {
            "path": path,
            "plugin": self.manifest.name,
            "bare_name": name,
            "description": description,
            "frontmatter": dict(frontmatter or {}),
        }
        current = self._manager._plugin_skills[qualified]

        def _release() -> None:
            if self._manager._plugin_skills.get(qualified) is current:
                self._manager._plugin_skills.pop(qualified, None)

        logger.debug(
            "Plugin %s registered skill: %s",
            self.manifest.name, qualified,
        )
        return PluginRegistration("skill", qualified, _release)


# ---------------------------------------------------------------------------
# PluginManager
# ---------------------------------------------------------------------------

class PluginManager:
    """Central manager that discovers, loads, and invokes plugins."""

    def __init__(self, scope_key: str | None = None) -> None:
        self._registry_scope = self
        # Retain the upstream constructor's explicit scope marker.  The
        # native-async profile manager still uses the manager object itself as
        # the registry namespace, while callers that construct a manager for a
        # known profile can inspect the immutable key without changing that
        # lifecycle behavior.
        self.scope_key = scope_key
        self._loop_finalizer: weakref.finalize | None = None
        _PLUGIN_MANAGER_INSTANCES.add(self)
        self._module_namespace = _NS_PARENT
        self._owned_module_names: set[str] = set()
        self._bound_module_names: set[str] = set()
        self._plugins: dict[str, LoadedPlugin] = {}
        self._hooks: dict[str, list[Callable]] = {}
        self._middleware: dict[str, list[Callable]] = {}
        self._plugin_tool_names: set[str] = set()
        self._context_engine = None  # Set by a plugin via register_context_engine()
        self._plugin_commands: dict[str, dict] = {}  # Slash commands registered by plugins
        self._system_prompt_sections: dict[str, PluginSystemPromptSection] = {}
        self._discovered: bool = False
        self._cli_ref = None  # Set by CLI after plugin discovery
        # Plugin skill registry: qualified name → metadata dict.
        self._plugin_skills: dict[str, dict[str, Any]] = {}
        # Plugin-registered auxiliary tasks: key → {key, display_name,
        # description, defaults, plugin}. See PluginContext.register_auxiliary_task.
        self._aux_tasks: dict[str, dict[str, Any]] = {}
        self._config: dict[str, Any] = {}
        self._profile_context: tuple[Path | None, Path | None] | None = None
        self._discovery_lock = asyncio.Lock()

    def _unload_source_finders(self) -> None:
        """Release import finders retained by the previous discovery sweep."""
        for loaded in self._plugins.values():
            module = loaded.module
            if module is not None:
                _unload_source_finder(module)

    def _clear_owned_state(self) -> None:
        """Remove profile-local registrations and imported plugin modules."""
        self._unload_source_finders()
        _clear_profile_registries(self._registry_scope)
        registry_module = sys.modules.get("tools.registry")
        unbind = getattr(
            getattr(registry_module, "registry", None),
            "_unbind_plugin_namespaces",
            None,
        )
        if callable(unbind):
            unbind(set(self._bound_module_names))
        self._bound_module_names.clear()
        for module_name in sorted(self._owned_module_names, reverse=True):
            for loaded_name in tuple(sys.modules):
                if loaded_name == module_name or loaded_name.startswith(
                    f"{module_name}."
                ):
                    sys.modules.pop(loaded_name, None)
        self._owned_module_names.clear()
        self._plugins.clear()
        self._hooks.clear()
        self._middleware.clear()
        self._plugin_tool_names.clear()
        self._plugin_commands.clear()
        self._system_prompt_sections.clear()
        self._plugin_skills.clear()
        self._aux_tasks.clear()
        self._context_engine = None

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    async def discover_and_load(self, force: bool = False) -> None:
        """Scan all plugin sources and load each plugin found.

        When ``force`` is true, clear cached discovery state first so config
        changes or newly-added bundled backends become visible in long-lived
        sessions without requiring a full agent restart.
        """
        discovery_lock = self._discovery_lock
        try:
            async with discovery_lock:
                if self._discovered and not force:
                    return
                if env_var_enabled("HERMES_SAFE_MODE"):
                    logger.info("HERMES_SAFE_MODE=1 — plugin discovery skipped")
                    self._discovered = True
                    return
                if force:
                    self._clear_owned_state()
                try:
                    await self._discover_and_load_inner()
                except BaseException:
                    self._clear_owned_state()
                    self._discovered = False
                    raise
                self._discovered = True
                _ACTIVE_PLUGIN_MANAGER.set(self)
        finally:
            if (
                self._discovery_lock is discovery_lock
                and not discovery_lock.locked()
                and not getattr(discovery_lock, "_waiters", None)
            ):
                # A contended asyncio.Lock retains its event loop even after
                # all waiters finish. Replace the idle lock so the loop/profile
                # cache remains weakly owned and loop finalization can unload
                # this manager's modules and registry overlays.
                self._discovery_lock = asyncio.Lock()

    async def _discover_and_load_inner(self) -> None:
        """The actual discovery sweep — see :meth:`discover_and_load`."""
        from hermes_cli.config import load_config_readonly
        from hermes_cli.profiles import _resolve_profile_context

        try:
            self._profile_context = await _resolve_profile_context()
        except Exception:
            logger.debug("Plugin profile resolution failed", exc_info=True)
            self._profile_context = (None, None)
        self._config = await load_config_readonly()
        manifests: list[PluginManifest] = []

        # 1. Bundled plugins (<repo>/plugins/<name>/)
        #
        # Repo-shipped plugins live next to hermes_cli/. Two layouts are
        # supported (see ``_scan_directory`` for details):
        #
        #   - flat: ``plugins/disk-cleanup/plugin.yaml`` (standalone)
        #   - category: ``plugins/image_gen/openai/plugin.yaml`` (backend)
        #
        # ``memory/``, ``context_engine/``, and ``model-providers/`` are
        # skipped at the top level — they have their own discovery systems
        # (plugins/memory/__init__.py, providers/__init__.py).
        repo_plugins = await get_bundled_plugins_dir()
        logger.debug("Scanning bundled plugins: %s", repo_plugins)
        bundled = await self._scan_directory(
            repo_plugins,
            source="bundled",
            skip_names={"memory", "context_engine", "model-providers"},
        )
        logger.debug("  bundled (top-level): %d manifest(s)", len(bundled))
        manifests.extend(bundled)

        # 2. User plugins (~/.hermes/plugins/)
        user_dir = get_hermes_home() / "plugins"
        logger.debug("Scanning user plugins: %s", user_dir)
        user_manifests = await self._scan_directory(user_dir, source="user")
        logger.debug("  user: %d manifest(s)", len(user_manifests))
        manifests.extend(user_manifests)

        # 3. Project plugins (./.hermes/plugins/)
        if _env_enabled("HERMES_ENABLE_PROJECT_PLUGINS"):
            project_dir = Path(await aiofiles.os.getcwd()) / ".hermes" / "plugins"
            logger.debug("Scanning project plugins: %s", project_dir)
            project_manifests = await self._scan_directory(
                project_dir, source="project"
            )
            logger.debug("  project: %d manifest(s)", len(project_manifests))
            manifests.extend(project_manifests)
        else:
            logger.debug(
                "Project plugins disabled (set HERMES_ENABLE_PROJECT_PLUGINS=1 to enable)"
            )

        # 4. Pip / entry-point plugins
        ep_manifests = await self._scan_entry_points()
        logger.debug("  entrypoints: %d manifest(s)", len(ep_manifests))
        manifests.extend(ep_manifests)

        # Load each manifest (skip user-disabled plugins).
        # Later sources override earlier ones on key collision — user
        # plugins take precedence over bundled, project plugins take
        # precedence over user. Dedup here so we only load the final
        # winner. Keys are path-derived (``image_gen/openai``,
        # ``disk-cleanup``) so ``tts/openai`` and ``image_gen/openai``
        # don't collide even when both manifests say ``name: openai``.
        disabled = await _get_disabled_plugins(self._config)
        enabled = await _get_enabled_plugins(
            self._config
        )  # None = opt-in default (nothing enabled)
        winners: dict[str, PluginManifest] = {}
        for manifest in manifests:
            winners[manifest.key or manifest.name] = manifest
        for manifest in winners.values():
            lookup_key = manifest.key or manifest.name

            # Explicit disable always wins (matches on key or on legacy
            # bare name for back-compat with existing user configs).
            if lookup_key in disabled or manifest.name in disabled:
                loaded = LoadedPlugin(manifest=manifest, enabled=False)
                loaded.error = "disabled via config"
                self._plugins[lookup_key] = loaded
                logger.debug("Skipping disabled plugin '%s'", lookup_key)
                continue

            # Exclusive plugins (memory providers) have their own
            # discovery/activation path. The general loader records the
            # manifest for introspection but does not load the module.
            if manifest.kind == "exclusive":
                loaded = LoadedPlugin(manifest=manifest, enabled=False)
                loaded.error = (
                    "exclusive plugin — activate via <category>.provider config"
                )
                self._plugins[lookup_key] = loaded
                logger.debug(
                    "Skipping '%s' (exclusive, handled by category discovery)",
                    lookup_key,
                )
                continue

            # Model provider plugins are loaded by providers/__init__.py
            # (its own lazy discovery keyed off first get_provider_profile()
            # call). We record the manifest here for introspection but do
            # not import the module — a second import would create two
            # ProviderProfile instances and break the "last writer wins"
            # override semantics between bundled and user plugins.
            if manifest.kind == "model-provider":
                loaded = LoadedPlugin(manifest=manifest, enabled=True)
                self._plugins[lookup_key] = loaded
                logger.debug(
                    "Skipping '%s' (model-provider, handled by providers/ discovery)",
                    lookup_key,
                )
                continue

            # Built-in backends auto-load — they ship with hermes and must
            # just work. Selection among them (e.g. which image_gen backend
            # services calls) is driven by ``<category>.provider`` config,
            # enforced by the tool wrapper.
            if manifest.source == "bundled" and manifest.kind == "backend":
                await self._load_plugin(manifest)
                continue

            # Everything else (standalone, user-installed backends,
            # entry-point plugins) is opt-in via plugins.enabled.
            # Accept both the path-derived key and the legacy bare name
            # so existing configs keep working.
            is_enabled = (
                enabled is not None
                and (lookup_key in enabled or manifest.name in enabled)
            )
            if not is_enabled:
                loaded = LoadedPlugin(manifest=manifest, enabled=False)
                loaded.error = (
                    f"not enabled in config (run `hermes plugins enable {lookup_key}` to activate)"
                )
                self._plugins[lookup_key] = loaded
                logger.debug(
                    "Skipping '%s' (not in plugins.enabled)", lookup_key
                )
                continue
            await self._load_plugin(manifest)

        if manifests:
            logger.info(
                "Plugin discovery complete: %d found, %d enabled",
                len(self._plugins),
                sum(1 for p in self._plugins.values() if p.enabled),
            )

    # -----------------------------------------------------------------------
    # Directory scanning
    # -----------------------------------------------------------------------

    async def _scan_directory(
        self,
        path: Path,
        source: str,
        skip_names: set[str] | None = None,
    ) -> list[PluginManifest]:
        """Read ``plugin.yaml`` manifests from subdirectories of *path*.

        Supports two layouts, mixed freely:

        * **Flat** — ``<root>/<plugin-name>/plugin.yaml``. Key is
          ``<plugin-name>`` (e.g. ``disk-cleanup``).
        * **Category** — ``<root>/<category>/<plugin-name>/plugin.yaml``,
          where the ``<category>`` directory itself has no ``plugin.yaml``.
          Key is ``<category>/<plugin-name>`` (e.g. ``image_gen/openai``).
          Depth is capped at two segments.

        *skip_names* is an optional allow-list of names to ignore at the
        top level (kept for back-compat; the current call sites no longer
        pass it now that categories are first-class).
        """
        return await self._scan_directory_level(
            path, source, skip_names=skip_names, prefix="", depth=0
        )

    async def _scan_directory_level(
        self,
        path: Path,
        source: str,
        *,
        skip_names: set[str] | None,
        prefix: str,
        depth: int,
    ) -> list[PluginManifest]:
        """Recursive implementation of :meth:`_scan_directory`.

        ``prefix`` is the category path already accumulated ("" at root,
        "image_gen" one level in). ``depth`` is the recursion depth; we
        cap at 2 so ``<root>/a/b/c/`` is ignored.
        """
        manifests: list[PluginManifest] = []
        if not await aiofiles.os.path.isdir(path):
            return manifests

        for child_name in sorted(await aiofiles.os.listdir(path)):
            child = path / child_name
            if not await aiofiles.os.path.isdir(child):
                continue
            if depth == 0 and skip_names and child.name in skip_names:
                continue
            manifest_file = child / "plugin.yaml"
            if not await aiofiles.os.path.exists(manifest_file):
                manifest_file = child / "plugin.yml"

            if await aiofiles.os.path.exists(manifest_file):
                manifest = await self._parse_manifest(
                    manifest_file, child, source, prefix
                )
                if manifest is not None:
                    manifests.append(manifest)
                continue

            # No manifest at this level. If we're still within the depth
            # cap, treat this directory as a category namespace and recurse
            # one level in looking for children with manifests.
            if depth >= 1:
                logger.debug("Skipping %s (no plugin.yaml, depth cap reached)", child)
                continue

            sub_prefix = f"{prefix}/{child.name}" if prefix else child.name
            manifests.extend(
                await self._scan_directory_level(
                    child,
                    source,
                    skip_names=None,
                    prefix=sub_prefix,
                    depth=depth + 1,
                )
            )

        return manifests

    async def _parse_manifest(
        self,
        manifest_file: Path,
        plugin_dir: Path,
        source: str,
        prefix: str,
    ) -> PluginManifest | None:
        """Parse a single ``plugin.yaml`` into a :class:`PluginManifest`.

        Returns ``None`` on parse failure (logs a warning).
        """
        try:
            if yaml is None:
                logger.warning("PyYAML not installed – cannot load %s", manifest_file)
                return None
            async with aiofiles.open(manifest_file, encoding="utf-8") as handle:
                data = fast_safe_load(await handle.read()) or {}

            name = data.get("name", plugin_dir.name)
            key = f"{prefix}/{plugin_dir.name}" if prefix else name

            raw_kind = data.get("kind", "standalone")
            if not isinstance(raw_kind, str):
                raw_kind = "standalone"
            kind = raw_kind.strip().lower()
            if kind not in _VALID_PLUGIN_KINDS:
                logger.warning(
                    "Plugin %s: unknown kind '%s' (valid: %s); treating as 'standalone'",
                    key, raw_kind, ", ".join(sorted(_VALID_PLUGIN_KINDS)),
                )
                kind = "standalone"

            # Auto-coerce user-installed memory providers to kind="exclusive"
            # so they're routed to plugins/memory discovery instead of being
            # loaded by the general PluginManager (which has no
            # register_memory_provider on PluginContext). Mirrors the
            # heuristic in plugins/memory/__init__.py:_is_memory_provider_dir.
            # Bundled memory providers are already skipped via skip_names.
            if kind == "standalone" and "kind" not in data:
                init_file = plugin_dir / "__init__.py"
                if await aiofiles.os.path.exists(init_file):
                    try:
                        async with aiofiles.open(
                            init_file, errors="replace", encoding="utf-8"
                        ) as handle:
                            source_text = (await handle.read(8192))
                        if (
                            "register_memory_provider" in source_text
                            or "MemoryProvider" in source_text
                        ):
                            kind = "exclusive"
                            logger.debug(
                                "Plugin %s: detected memory provider, "
                                "treating as kind='exclusive'",
                                key,
                            )
                        elif (
                            "register_provider" in source_text
                            and "ProviderProfile" in source_text
                        ):
                            # Model provider plugin (calls register_provider()
                            # from ``providers`` with a ProviderProfile). Route
                            # to providers/__init__.py discovery.
                            kind = "model-provider"
                            logger.debug(
                                "Plugin %s: detected model provider, "
                                "treating as kind='model-provider'",
                                key,
                            )
                    except Exception:
                        pass

            logger.debug(
                "Parsed manifest: key=%s name=%s kind=%s source=%s path=%s",
                key, name, kind, source, plugin_dir,
            )
            return PluginManifest(
                name=name,
                version=str(data.get("version", "")),
                description=data.get("description", ""),
                author=data.get("author", ""),
                requires_env=data.get("requires_env", []),
                provides_tools=data.get("provides_tools", []),
                provides_hooks=data.get("provides_hooks", []),
                source=source,
                path=str(plugin_dir),
                kind=kind,
                key=key,
            )
        except Exception as exc:
            logger.warning(
                "Failed to parse %s: %s", manifest_file, exc, exc_info=_PLUGINS_DEBUG,
            )
            return None

    # -----------------------------------------------------------------------
    # Entry-point scanning
    # -----------------------------------------------------------------------

    async def _scan_entry_points(self) -> list[PluginManifest]:
        """Check ``importlib.metadata`` for pip-installed plugins."""
        manifests: list[PluginManifest] = []
        try:
            eps = await aiofiles.os.wrap(importlib.metadata.entry_points)()
            # Python 3.12+ returns a SelectableGroups; earlier returns dict
            if hasattr(eps, "select"):
                group_eps = eps.select(group=ENTRY_POINTS_GROUP)
            elif isinstance(eps, dict):
                group_eps = eps.get(ENTRY_POINTS_GROUP, [])
            else:
                group_eps = [ep for ep in eps if ep.group == ENTRY_POINTS_GROUP]

            for ep in group_eps:
                manifest = PluginManifest(
                    name=ep.name,
                    source="entrypoint",
                    path=ep.value,
                    key=ep.name,
                )
                manifests.append(manifest)
        except Exception as exc:
            logger.debug("Entry-point scan failed: %s", exc)

        return manifests

    # -----------------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------------

    async def _load_plugin(self, manifest: PluginManifest) -> None:
        """Import a plugin module and call its ``register(ctx)`` function."""
        loaded = LoadedPlugin(manifest=manifest)
        logger.debug(
            "Loading plugin '%s' (source=%s, kind=%s, path=%s)",
            manifest.key or manifest.name, manifest.source, manifest.kind, manifest.path,
        )

        from tools.registry import registry as _registry
        _plugin_id = manifest.key or manifest.name
        _slug = _plugin_id.replace("/", "__").replace("-", "_")
        _module_name = f"{self._module_namespace}.{_slug}"
        override_context = PluginContext(manifest, self)
        override_context._profile_context = self._profile_context
        _registry.register_plugin_override_policy(
            _module_name,
            override_context._tool_override_allowed(""),
        )
        _registry._bind_plugin_scope(
            _module_name,
            (
                self._registry_scope
                if manifest.source in {"user", "project"}
                else None
            ),
        )
        self._bound_module_names.add(_module_name)
        registration_token = _PLUGIN_REGISTRATION_TARGET.set(
            (self, manifest.source)
        )
        try:
            if manifest.source in {"user", "project", "bundled"}:
                module = await self._load_directory_module(manifest)
            else:
                module = await self._load_entrypoint_module(manifest)

            loaded.module = module
            actual_module_name = getattr(module, "__module__", None)
            if isinstance(module, types.ModuleType):
                actual_module_name = module.__name__
            if isinstance(actual_module_name, str) and actual_module_name:
                _registry.register_plugin_override_policy(
                    actual_module_name,
                    override_context._tool_override_allowed(""),
                )
                _registry._bind_plugin_scope(
                    actual_module_name,
                    (
                        self._registry_scope
                        if manifest.source in {"user", "project"}
                        else None
                    ),
                )
                if manifest.source in {"user", "project", "bundled"}:
                    self._bound_module_names.add(actual_module_name)

            # Call register()
            register_fn = module if callable(module) else getattr(module, "register", None)
            if register_fn is None:
                loaded.error = "no register() function"
                logger.warning("Plugin '%s' has no register() function", manifest.name)
            else:
                ctx = PluginContext(manifest, self)
                ctx._profile_context = self._profile_context
                ctx._defer_skill_validation = True
                # Snapshot registry state BEFORE register() so each registry's
                # attribution counts only what THIS plugin actually added.
                # The previous approach diffed names against all already-loaded
                # plugins, which mis-credited a plugin that registered a hook /
                # middleware / tool name an earlier plugin had already used:
                # the shared name was attributed to the first plugin only, so
                # later plugins under-reported in `hermes plugins list`.
                _tools_before = set(self._plugin_tool_names)
                _hook_counts_before = {
                    h: len(cbs) for h, cbs in self._hooks.items()
                }
                _mw_counts_before = {
                    kind: len(cbs) for kind, cbs in self._middleware.items()
                }
                registration = register_fn(ctx)
                if inspect.isawaitable(registration):
                    await registration
                for qualified, skill_path in ctx._deferred_skill_paths:
                    if not await aiofiles.os.path.exists(skill_path):
                        self._plugin_skills.pop(qualified, None)
                        raise FileNotFoundError(
                            f"SKILL.md not found at {skill_path}"
                        )
                loaded.tools_registered = [
                    t for t in self._plugin_tool_names
                    if t not in _tools_before
                ]
                loaded.hooks_registered = [
                    h
                    for h, cbs in self._hooks.items()
                    if len(cbs) > _hook_counts_before.get(h, 0)
                ]
                loaded.middleware_registered = [
                    kind
                    for kind, cbs in self._middleware.items()
                    if len(cbs) > _mw_counts_before.get(kind, 0)
                ]
                loaded.commands_registered = [
                    c for c in self._plugin_commands
                    if self._plugin_commands[c].get("plugin") == manifest.name
                ]
                loaded.enabled = True
                logger.debug(
                    "  registered: %d tool(s), %d hook(s), %d middleware, %d slash command(s)",
                    len(loaded.tools_registered),
                    len(loaded.hooks_registered),
                    len(loaded.middleware_registered),
                    len(loaded.commands_registered),
                )

        except Exception as exc:
            loaded.error = str(exc)
            logger.warning(
                "Failed to load plugin '%s': %s",
                manifest.name, exc, exc_info=_PLUGINS_DEBUG,
            )
        finally:
            _PLUGIN_REGISTRATION_TARGET.reset(registration_token)
        self._plugins[manifest.key or manifest.name] = loaded

    async def _load_directory_module(
        self,
        manifest: PluginManifest,
        *,
        module_name: str | None = None,
    ) -> types.ModuleType:
        """Import a directory-based plugin as ``hermes_plugins.<slug>``.

        The module slug is derived from ``manifest.key`` so category-namespaced
        plugins (``image_gen/openai``) import as
        ``hermes_plugins.image_gen__openai`` without colliding with any
        future ``tts/openai``.
        """
        plugin_dir = Path(manifest.path)  # type: ignore[arg-type]
        init_file = plugin_dir / "__init__.py"

        # Ensure both namespace parents exist. Profile-local package names
        # prevent same-slug plugins from separate HERMES_HOME values from
        # sharing ``sys.modules`` state.
        for namespace in (_NS_PARENT, self._module_namespace):
            if namespace not in sys.modules:
                ns_pkg = types.ModuleType(namespace)
                ns_pkg.__path__ = []  # type: ignore[attr-defined]
                ns_pkg.__package__ = namespace
                sys.modules[namespace] = ns_pkg

        key = manifest.key or manifest.name
        slug = key.replace("/", "__").replace("-", "_")
        module_name = module_name or f"{self._module_namespace}.{slug}"
        self._owned_module_names.add(module_name)
        source_alias = None
        if manifest.source == "bundled":
            key_parts = key.split("/")
            alias_parts = ("plugins", *plugin_dir.parts[-len(key_parts):])
            if all(part.isidentifier() for part in alias_parts):
                source_alias = ".".join(alias_parts)
        try:
            module = await _load_source_package(
                module_name,
                init_file,
                source_alias=source_alias,
            )
        except asyncio.CancelledError:
            for loaded_name in tuple(sys.modules):
                if loaded_name == module_name or loaded_name.startswith(f"{module_name}."):
                    sys.modules.pop(loaded_name, None)
            raise
        except FileNotFoundError as exc:
            if exc.filename != str(init_file):
                raise
            for loaded_name in tuple(sys.modules):
                if loaded_name == module_name or loaded_name.startswith(f"{module_name}."):
                    sys.modules.pop(loaded_name, None)
            raise FileNotFoundError(f"No __init__.py in {plugin_dir}") from exc
        except Exception:
            for loaded_name in tuple(sys.modules):
                if loaded_name == module_name or loaded_name.startswith(f"{module_name}."):
                    sys.modules.pop(loaded_name, None)
            raise
        return module

    async def _load_entrypoint_module(
        self,
        manifest: PluginManifest,
    ) -> types.ModuleType:
        """Load a pip-installed plugin via its entry-point reference.

        ``EntryPoint.load()`` delegates to Python's synchronous source
        importer.  Real metadata entry points are therefore resolved to their
        module source and executed only after the source bytes have crossed an
        awaited file boundary.  A small in-memory fallback is retained for
        test/host adapters that intentionally expose only ``load()`` and have
        no importable module metadata; it cannot perform source discovery.
        """
        eps = await aiofiles.os.wrap(importlib.metadata.entry_points)()
        if hasattr(eps, "select"):
            group_eps = eps.select(group=ENTRY_POINTS_GROUP)
        elif isinstance(eps, dict):
            group_eps = eps.get(ENTRY_POINTS_GROUP, [])
        else:
            group_eps = [ep for ep in eps if ep.group == ENTRY_POINTS_GROUP]

        for ep in group_eps:
            if ep.name == manifest.name:
                module_name = str(getattr(ep, "module", "") or "").strip()
                attr_path = str(getattr(ep, "attr", "") or "").strip()
                if not module_name:
                    value = str(getattr(ep, "value", "") or "")
                    module_name, separator, attr_path = value.partition(":")
                    if not separator:
                        attr_path = ""
                    module_name = module_name.strip()

                module = sys.modules.get(module_name) if module_name else None
                if module is None and module_name:
                    source_info = await _locate_source_module(
                        module_name,
                        distribution=getattr(ep, "dist", None),
                    )
                    if source_info is not None:
                        source_path, is_package = source_info
                        try:
                            if is_package:
                                module = await _load_source_package(module_name, source_path)
                            else:
                                parent_name, _, _ = module_name.rpartition(".")
                                parent_init = source_path.parent / "__init__.py"
                                if (
                                    parent_name
                                    and parent_name not in sys.modules
                                    and await aiofiles.os.path.isfile(parent_init)
                                ):
                                    await _load_source_package(parent_name, parent_init)
                                module = await _load_source_module(
                                    module_name,
                                    source_path,
                                    package_dir=source_path.parent,
                                )
                        except asyncio.CancelledError:
                            for loaded_name in tuple(sys.modules):
                                if loaded_name == module_name or loaded_name.startswith(
                                    f"{module_name}."
                                ):
                                    sys.modules.pop(loaded_name, None)
                            raise
                        except Exception:
                            for loaded_name in tuple(sys.modules):
                                if loaded_name == module_name or loaded_name.startswith(
                                    f"{module_name}."
                                ):
                                    sys.modules.pop(loaded_name, None)
                            raise

                # Non-standard in-memory entry-point adapters (used by host
                # applications and tests) may intentionally omit ``module`` /
                # ``value`` metadata.  Preserve that contract without using
                # it as a fallback for real package imports.
                if module is None:
                    load_fn = getattr(ep, "load", None)
                    if callable(load_fn) and type(ep).__module__ != "importlib.metadata":
                        loaded = load_fn()
                        if inspect.isawaitable(loaded):
                            loaded = await loaded
                        module = loaded

                if module is None:
                    raise ImportError(
                        f"Entrypoint '{manifest.name}' has no importable native "
                        "async source module"
                    )

                target: Any = module
                for attr_name in filter(None, attr_path.split(".")):
                    target = getattr(target, attr_name)
                return target

        raise ImportError(
            f"Entry point '{manifest.name}' not found in group '{ENTRY_POINTS_GROUP}'"
        )

    # -----------------------------------------------------------------------
    # Hook invocation
    # -----------------------------------------------------------------------

    async def invoke_hook(self, hook_name: str, **kwargs: Any) -> list[Any]:
        """Invoke lifecycle hooks through the native async plugin contract.

        An async agent cannot safely guess whether a third-party synchronous
        callback performs I/O.  Such callbacks are rejected explicitly rather
        than being hidden in a worker thread or allowed to stall every turn.
        """
        kwargs.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
        callbacks = self._hooks.get(hook_name, [])
        results: list[Any] = []
        for callback in callbacks:
            try:
                if not inspect.iscoroutinefunction(callback):
                    raise _PluginContractError(
                        "Async Hermes requires coroutine lifecycle hooks; "
                        f"{getattr(callback, '__name__', repr(callback))} is synchronous"
                    )
                result = await callback(**kwargs)
                if result is not None:
                    results.append(result)
            except _PluginContractError:
                raise
            except Exception as exc:
                logger.warning(
                    "Hook '%s' callback %s raised: %s",
                    hook_name,
                    getattr(callback, "__name__", repr(callback)),
                    exc,
                )
        return results

    def render_system_prompt_sections(
        self, session_info: dict[str, Any]
    ) -> list[RenderedPluginSystemPromptSection]:
        """Render registered sections deterministically and fail open."""
        frozen_info = types.MappingProxyType(dict(session_info))
        rendered: list[RenderedPluginSystemPromptSection] = []
        total_chars = len(PLUGIN_SECTIONS_START) + len(PLUGIN_SECTIONS_END) + 2
        for section_id in sorted(self._system_prompt_sections):
            section = self._system_prompt_sections[section_id]
            if len(rendered) >= MAX_SYSTEM_PROMPT_SECTIONS:
                logger.warning("Plugin system prompt section count budget exceeded; skipping %s", section.id)
                continue
            try:
                value = section.content(frozen_info) if callable(section.content) else section.content
            except Exception as exc:
                logger.warning("Plugin system prompt section %s raised and was skipped: %s", section.id, exc)
                continue
            if not isinstance(value, str):
                logger.warning("Plugin system prompt section %s returned non-string; skipped", section.id)
                continue
            text = value.strip()
            if not text or PLUGIN_SECTIONS_START in text or PLUGIN_SECTIONS_END in text:
                continue
            if len(text) > section.max_chars:
                logger.warning("Plugin system prompt section %s exceeded max_chars; skipped", section.id)
                continue
            rendered_chars = len(format_system_prompt_section(section.id, text))
            if rendered:
                rendered_chars += 2
            if total_chars + rendered_chars > MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS:
                logger.warning("Plugin system prompt aggregate budget exceeded; skipping %s", section.id)
                continue
            rendered.append(
                RenderedPluginSystemPromptSection(
                    id=section.id,
                    content=text,
                    position=section.position,
                    plugin=section.plugin,
                )
            )
            total_chars += rendered_chars
        return rendered

    def has_hook(self, hook_name: str) -> bool:
        """Return True when at least one callback is registered for a hook."""
        return bool(self._hooks.get(hook_name))

    def has_middleware(self, kind: str) -> bool:
        """Return True when at least one callback is registered for middleware."""
        return bool(self._middleware.get(kind))

    async def invoke_middleware(self, kind: str, **kwargs: Any) -> list[Any]:
        """Invoke coroutine middleware callbacks registered for *kind*."""
        callbacks = self._middleware.get(kind, [])
        results: list[Any] = []
        for callback in callbacks:
            try:
                if not inspect.iscoroutinefunction(callback):
                    raise _PluginContractError(
                        "Async Hermes requires coroutine middleware callbacks; "
                        f"{getattr(callback, '__name__', repr(callback))} is synchronous"
                    )
                result = await callback(**kwargs)
                if result is not None:
                    results.append(result)
            except _PluginContractError:
                raise
            except Exception as exc:
                logger.warning(
                    "Middleware '%s' callback %s raised: %s",
                    kind,
                    getattr(callback, "__name__", repr(callback)),
                    exc,
                )
        return results

    # -----------------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------------

    def list_plugins(self) -> list[dict[str, Any]]:
        """Return a list of info dicts for all discovered plugins."""
        result: list[dict[str, Any]] = []
        for key, loaded in sorted(self._plugins.items()):
            result.append(
                {
                    "name": loaded.manifest.name,
                    "key": loaded.manifest.key or loaded.manifest.name,
                    "kind": loaded.manifest.kind,
                    "version": loaded.manifest.version,
                    "description": loaded.manifest.description,
                    "source": loaded.manifest.source,
                    "enabled": loaded.enabled,
                    "tools": len(loaded.tools_registered),
                    "hooks": len(loaded.hooks_registered),
                    "middleware": len(loaded.middleware_registered),
                    "commands": len(loaded.commands_registered),
                    "error": loaded.error,
                }
            )
        return result

    # -----------------------------------------------------------------------
    # Plugin skill lookups
    # -----------------------------------------------------------------------

    def find_plugin_skill(self, qualified_name: str) -> Path | None:
        """Return the ``Path`` to a plugin skill's SKILL.md, or ``None``."""
        entry = self._plugin_skills.get(qualified_name)
        return entry["path"] if entry else None

    def list_plugin_skills(self, plugin_name: str) -> list[str]:
        """Return sorted bare names of all skills registered by *plugin_name*."""
        prefix = f"{plugin_name}:"
        return sorted(
            e["bare_name"]
            for qn, e in self._plugin_skills.items()
            if qn.startswith(prefix)
        )

    def remove_plugin_skill(self, qualified_name: str) -> None:
        """Remove a stale registry entry (silently ignores missing keys)."""
        self._plugin_skills.pop(qualified_name, None)


# ---------------------------------------------------------------------------
# Module-level singleton & convenience functions
# ---------------------------------------------------------------------------

_plugin_manager: PluginManager | None = None


def get_plugin_manager() -> PluginManager:
    """Return the active profile manager, with a no-loop legacy projection."""
    global _plugin_manager
    active = _ACTIVE_PLUGIN_MANAGER.get()
    if active is not None:
        return active
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


async def _get_profile_plugin_manager() -> PluginManager:
    """Return the loop + canonical-HERMES_HOME manager for this task."""
    global _plugin_manager
    loop = asyncio.get_running_loop()
    profile_key, _profile_home = await _canonical_plugin_home()
    with _PLUGIN_MANAGER_GUARD:
        per_loop = _PLUGIN_MANAGERS.setdefault(loop, {})
        manager = per_loop.get(profile_key)
        if manager is None:
            legacy = _plugin_manager
            if (
                legacy is not None
                and getattr(legacy, "_profile_key", None) is None
                and getattr(legacy, "_owner_loop_ref", None) is None
            ):
                manager = legacy
            else:
                manager = PluginManager()
            manager._profile_key = profile_key
            manager._owner_loop_ref = weakref.ref(loop)
            manager._module_namespace = _plugin_namespace(loop, profile_key)
            manager._loop_finalizer = weakref.finalize(
                loop,
                manager._clear_owned_state,
            )
            per_loop[profile_key] = manager
        _plugin_manager = manager
    _ACTIVE_PLUGIN_MANAGER.set(manager)
    return manager


def _reset_plugin_profiles_for_tests() -> None:
    """Clear all plugin profile state. Test-only."""
    global _plugin_manager
    with _PLUGIN_MANAGER_GUARD:
        managers = set(_PLUGIN_MANAGER_INSTANCES)
        if _plugin_manager is not None:
            managers.add(_plugin_manager)
        for manager in managers:
            if manager._loop_finalizer is not None:
                manager._loop_finalizer.detach()
                manager._loop_finalizer = None
            manager._clear_owned_state()
            manager._discovered = False
        _PLUGIN_MANAGERS.clear()
        _plugin_manager = None
    _ACTIVE_PLUGIN_MANAGER.set(None)


async def discover_plugins(force: bool = False) -> None:
    """Discover and load all plugins.

    Default behavior is idempotent. Pass ``force=True`` to rescan plugin
    manifests and reload state in the current process.
    """
    manager = await _get_profile_plugin_manager()
    await manager.discover_and_load(force=force)


class _PluginContractError(RuntimeError):
    """Raised when an unconverted plugin reaches the native async runtime."""


async def invoke_hook(hook_name: str, **kwargs: Any) -> list[Any]:
    """Await native lifecycle hooks without a sync compatibility bridge."""
    await discover_plugins()
    return await get_plugin_manager().invoke_hook(hook_name, **kwargs)


async def invoke_middleware(kind: str, **kwargs: Any) -> list[Any]:
    """Invoke registered middleware callbacks through the native async contract."""
    await discover_plugins()
    return await get_plugin_manager().invoke_middleware(kind, **kwargs)


def has_middleware(kind: str) -> bool:
    """Return True when middleware callbacks are registered for ``kind``."""
    manager = get_plugin_manager()
    method = getattr(manager, "has_middleware", None)
    if callable(method):
        return bool(method(kind))
    return bool(getattr(manager, "_middleware", {}).get(kind))


def has_hook(hook_name: str) -> bool:
    """Return True when a loaded plugin handles a hook."""
    return get_plugin_manager().has_hook(hook_name)


def render_system_prompt_sections(
    session_info: dict[str, Any],
) -> list[RenderedPluginSystemPromptSection]:
    """Render sections from the already-loaded active plugin manager.

    Discovery remains an awaited operation.  Prompt construction is a pure
    in-memory render and therefore does not introduce a synchronous I/O
    compatibility bridge.
    """
    return get_plugin_manager().render_system_prompt_sections(session_info)


def iter_hook_callbacks(hook_name: str) -> tuple[Callable[..., Any], ...]:
    """Return currently loaded callbacks for a fire-and-forget observer.

    Discovery remains an awaited operation owned by the normal plugin manager;
    this synchronous read is deliberately limited to the already-loaded hook
    table so token-path dispatch never performs I/O or imports.
    """
    callbacks = getattr(get_plugin_manager(), "_hooks", {}).get(hook_name, ())
    return tuple(callbacks)


def get_plugin_error_classification(
    *,
    provider: str = "",
    model: str = "",
    status_code: int | None = None,
    error_type: str = "",
    error_code: str = "",
    error_message: str = "",
    error_body: dict[str, Any] | None = None,
    error: BaseException | None = None,
    approx_tokens: int = 0,
    context_length: int = 0,
    num_messages: int = 0,
) -> dict[str, Any] | None:
    """Return the first valid synchronous API-error transform.

    ``classify_api_error`` is intentionally a pure synchronous operation:
    it is also used by compression and stream-recovery helpers that do not
    perform I/O.  The native-async plugin lifecycle therefore cannot be
    awaited from this boundary without violating the no-thread/no-nested-
    loop contract.  Discovery remains owned by the awaited plugin boundary;
    this helper only inspects callbacks already loaded in the active manager.
    Async callbacks are skipped (the normal lifecycle dispatcher will reject
    synchronous/async contract mismatches separately).

    The payload and sanitized first-valid result match upstream's
    ``transform_api_error_classification`` contract.
    """
    from agent.error_classifier import FailoverReason

    callbacks = iter_hook_callbacks("transform_api_error_classification")
    payload: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "status_code": status_code,
        "error_type": error_type,
        "error_code": error_code,
        "error_message": error_message,
        "error_body": error_body if isinstance(error_body, dict) else {},
        "error": error,
        "approx_tokens": approx_tokens,
        "context_length": context_length,
        "num_messages": num_messages,
    }
    winner: dict[str, Any] | None = None
    skipped_valid = 0
    for callback in callbacks:
        if inspect.iscoroutinefunction(callback):
            logger.debug(
                "Skipping async API-error transform callback %s in sync classifier",
                getattr(callback, "__name__", repr(callback)),
            )
            continue
        try:
            try:
                parameters = inspect.signature(callback).parameters
            except (TypeError, ValueError):
                parameters = None
            if parameters is None or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            ):
                result = callback(**payload)
            else:
                accepted = {
                    name: value
                    for name, value in payload.items()
                    if name in parameters
                    and parameters[name].kind
                    in {
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    }
                }
                result = callback(**accepted)
            if inspect.isawaitable(result):
                logger.debug(
                    "Skipping awaitable API-error transform callback %s in sync classifier",
                    getattr(callback, "__name__", repr(callback)),
                )
                continue
        except Exception as exc:
            logger.warning(
                "API-error transform callback %s raised: %s",
                getattr(callback, "__name__", repr(callback)),
                exc,
            )
            continue
        if not isinstance(result, dict):
            continue
        reason_raw = result.get("reason")
        if isinstance(reason_raw, FailoverReason):
            reason = reason_raw
        elif isinstance(reason_raw, str):
            try:
                reason = FailoverReason(reason_raw.strip().lower())
            except ValueError:
                continue
        else:
            continue
        if winner is not None:
            skipped_valid += 1
            continue
        winner = {"reason": reason}
        for key in (
            "retryable",
            "should_compress",
            "should_rotate_credential",
            "should_fallback",
        ):
            if key in result:
                winner[key] = bool(result[key])
        message = result.get("message")
        if isinstance(message, str) and message.strip():
            winner["message"] = message.strip()[:500]
        error_context = result.get("error_context")
        if isinstance(error_context, dict):
            winner["error_context"] = error_context
    if winner is not None and skipped_valid:
        logger.warning(
            "transform_api_error_classification: skipped %d valid "
            "classification(s) after the first result in registration order won",
            skipped_valid,
        )
    return winner


_thread_tool_whitelist: contextvars.ContextVar[
    tuple[set[str] | None, str]
] = contextvars.ContextVar(
    "thread_tool_whitelist",
    default=(None, "Tool '{tool_name}' denied"),
)


def set_thread_tool_whitelist(
    allowed: set[str] | None,
    deny_msg_fmt: str = (
        "Tool '{tool_name}' denied: not in this thread's tool whitelist"
    ),
) -> None:
    """Restrict tool dispatch in the current native async context."""
    _thread_tool_whitelist.set((allowed, deny_msg_fmt))


def clear_thread_tool_whitelist() -> None:
    """Clear the historical thread-scoped tool restriction."""
    _thread_tool_whitelist.set((None, "Tool '{tool_name}' denied"))


@dataclass(frozen=True)
class _PreToolCallDirective:
    action: str | None = None
    message: str | None = None
    rule_key: str | None = None


def _first_pre_tool_call_directive(hook_results: list[Any]) -> _PreToolCallDirective:
    """Return the first valid policy directive from hook results."""
    for result in hook_results:
        if not isinstance(result, dict):
            continue
        action = result.get("action")
        if action not in ("block", "approve"):
            continue
        message = result.get("message")
        message = message if isinstance(message, str) and message else None
        # A block directive requires a message (it becomes the tool result);
        # an approve directive can carry an optional reason.
        if action == "block" and not message:
            continue
        rule_key = result.get("rule_key") if action == "approve" else None
        rule_key = rule_key.strip() if isinstance(rule_key, str) else None
        if not rule_key:
            rule_key = None
        return _PreToolCallDirective(action=action, message=message, rule_key=rule_key)

    return _PreToolCallDirective()


async def _get_pre_tool_call_directive_details(
    tool_name: str,
    args: dict[str, Any] | None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    middleware_trace: list[dict[str, Any]] | None = None,
) -> _PreToolCallDirective:
    """Check async ``pre_tool_call`` hooks for a policy directive."""
    allowed, deny_msg_fmt = _thread_tool_whitelist.get()
    if allowed is not None and tool_name not in allowed:
        return _PreToolCallDirective(
            action="block",
            message=deny_msg_fmt.format(tool_name=tool_name),
        )

    hook_results = await invoke_hook(
        "pre_tool_call",
        tool_name=tool_name,
        args=args if isinstance(args, dict) else {},
        task_id=task_id,
        session_id=session_id,
        tool_call_id=tool_call_id,
        turn_id=turn_id,
        api_request_id=api_request_id,
        middleware_trace=list(middleware_trace or []),
    )
    return _first_pre_tool_call_directive(hook_results)


async def get_pre_tool_call_directive(
    tool_name: str,
    args: dict[str, Any] | None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    middleware_trace: list[dict[str, Any]] | None = None,
) -> tuple[str | None, str | None]:
    """Check ``pre_tool_call`` hooks for a blocking or approval directive.

    Backward-compatible public helper: returns ``(directive, message)`` where
    ``directive`` is ``"block"``, ``"approve"``, or ``None``. Internal callers
    that need approve-specific metadata use
    :func:`_get_pre_tool_call_directive_details`.
    """
    details = await _get_pre_tool_call_directive_details(
        tool_name, args, task_id=task_id, session_id=session_id,
        tool_call_id=tool_call_id, turn_id=turn_id,
        api_request_id=api_request_id, middleware_trace=middleware_trace,
    )
    return (details.action, details.message)


async def get_pre_tool_call_block_message(
    tool_name: str,
    args: dict[str, Any] | None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    middleware_trace: list[dict[str, Any]] | None = None,
) -> str | None:
    """Return a plugin's async ``pre_tool_call`` block message, if any."""
    directive, message = await get_pre_tool_call_directive(
        tool_name,
        args,
        task_id=task_id,
        session_id=session_id,
        tool_call_id=tool_call_id,
        turn_id=turn_id,
        api_request_id=api_request_id,
        middleware_trace=middleware_trace,
    )
    return message if directive == "block" else None


async def resolve_pre_tool_block(
    tool_name: str,
    args: dict[str, Any] | None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    middleware_trace: list[dict[str, Any]] | None = None,
) -> str | None:
    """Resolve a pre-tool policy directive to a final block message or ``None``."""
    details = await _get_pre_tool_call_directive_details(
        tool_name,
        args,
        task_id=task_id,
        session_id=session_id,
        tool_call_id=tool_call_id,
        turn_id=turn_id,
        api_request_id=api_request_id,
        middleware_trace=middleware_trace,
    )
    if details.action == "block":
        return details.message
    if details.action == "approve":
        try:
            from tools.approval import request_tool_approval

            result = await request_tool_approval(
                tool_name,
                details.message or "",
                rule_key=details.rule_key or tool_name,
            )
        except Exception:
            logger.exception("Plugin approval gate failed for %s", tool_name)
            return f"BLOCKED: plugin approval gate failed for {tool_name}"
        if not result.get("approved"):
            return str(
                result.get("message")
                or f"BLOCKED: plugin approval required for {tool_name}"
            )
    return None


async def get_pre_verify_continue_message(
    *,
    session_id: str = "",
    platform: str = "",
    model: str = "",
    coding: bool = False,
    attempt: int = 0,
    final_response: str = "",
    changed_paths: list[str] | None = None,
) -> str | None:
    """Check user ``pre_verify`` hooks for a directive to keep the agent going.

    Fired once per turn when the agent edited code and is about to verify/finish.
    A hook keeps the turn going (run a check, defer it, tidy the diff) by
    returning::

        {"action": "continue", "message": "<follow-up for the model>"}

    The Claude-Code Stop shape ``{"decision": "block", "reason": "..."}`` (block
    the stop == keep going) is accepted too. The first directive carrying a
    non-empty message wins; any other return lets the turn finish. Mirrors
    :func:`get_pre_tool_call_block_message` — the call site stays a one-liner.

    ``coding`` / ``attempt`` let a hook scope itself (``if not coding`` …) and
    self-throttle (``if attempt`` …), the same way a ``pre_tool_call`` hook
    scopes on ``tool_name``.
    """
    hook_results = await invoke_hook(
        "pre_verify",
        session_id=session_id,
        platform=platform,
        model=model,
        coding=coding,
        attempt=attempt,
        final_response=final_response,
        changed_paths=list(changed_paths or []),
    )

    for result in hook_results:
        if not isinstance(result, dict):
            continue
        action = str(result.get("action") or result.get("decision") or "").strip().lower()
        if action not in ("continue", "block"):
            continue
        message = result.get("message") or result.get("reason")
        if isinstance(message, str) and message.strip():
            return message.strip()

    return None


async def _ensure_plugins_discovered(force: bool = False) -> PluginManager:
    """Return the current profile manager after ensuring discovery has run.

    Pass ``force=True`` to rescan in the current process.
    """
    manager = await _get_profile_plugin_manager()
    await manager.discover_and_load(force=force)
    return manager


async def get_plugin_context_engine():
    """Return the plugin-registered context engine, or None."""
    return (await _ensure_plugins_discovered())._context_engine


async def get_plugin_command_handler(name: str) -> Callable | None:
    """Return the handler for a plugin-registered slash command, or ``None``."""
    entry = (await _ensure_plugins_discovered())._plugin_commands.get(name)
    return entry["handler"] if entry else None


async def get_plugin_commands() -> dict[str, dict]:
    """Return the full plugin commands dict (name → {handler, description, plugin}).

    Triggers idempotent plugin discovery so callers can use plugin commands
    before any explicit discover_plugins() call.
    """
    return (await _ensure_plugins_discovered())._plugin_commands


def get_plugin_auxiliary_tasks() -> list[dict[str, Any]]:
    """Return all plugin-registered auxiliary tasks as a stable-ordered list.

    Each entry is the registration dict from
    :meth:`PluginContext.register_auxiliary_task`:
    ``{key, display_name, description, defaults, plugin}``.

    Async runtime bootstrap performs discovery before auxiliary resolution;
    this accessor only reads the resulting in-memory registry. Sorted by
    ``key`` for deterministic ordering in pickers and tests.
    """
    manager = get_plugin_manager()
    return [manager._aux_tasks[k] for k in sorted(manager._aux_tasks)]


def get_plugin_toolsets() -> list[tuple]:
    """Return plugin toolsets as ``(key, label, description)`` tuples.

    Used by the ``hermes tools`` TUI so plugin-provided toolsets appear
    alongside the built-in ones and can be toggled on/off per platform.
    """
    manager = get_plugin_manager()
    if not manager._plugin_tool_names:
        return []

    try:
        from tools.registry import registry
    except Exception:
        return []

    # Group plugin tool names by their toolset
    toolset_tools: dict[str, list[str]] = {}
    toolset_plugin: dict[str, LoadedPlugin] = {}
    for tool_name in manager._plugin_tool_names:
        entry = registry.get_entry(tool_name)
        if not entry:
            continue
        ts = entry.toolset
        toolset_tools.setdefault(ts, []).append(entry.name)

    # Map toolsets back to the plugin that registered them
    for _name, loaded in manager._plugins.items():
        for tool_name in loaded.tools_registered:
            entry = registry.get_entry(tool_name)
            if entry and entry.toolset in toolset_tools:
                toolset_plugin.setdefault(entry.toolset, loaded)

    result = []
    for ts_key in sorted(toolset_tools):
        plugin = toolset_plugin.get(ts_key)
        label = f"🔌 {ts_key.replace('_', ' ').title()}"
        if plugin and plugin.manifest.description:
            desc = plugin.manifest.description
        else:
            desc = ", ".join(sorted(toolset_tools[ts_key]))
        result.append((ts_key, label, desc))

    return result
