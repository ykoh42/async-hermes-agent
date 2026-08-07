#!/usr/bin/env python3
"""
Browser Tool Module

This module provides browser automation tools using agent-browser CLI.  It
supports multiple backends — **Browser Use** (cloud, default for Nous
subscribers), **Browserbase** (cloud, direct credentials), and **local
Chromium** — with identical agent-facing behaviour.  The backend is
auto-detected from config and available credentials.

The tool uses agent-browser's accessibility tree (ariaSnapshot) for text-based
page representation, making it ideal for LLM agents without vision capabilities.

Features:
- **Local mode** (default): zero-cost headless Chromium via agent-browser.
  Works on Linux servers without a display.  One-time setup:
  ``agent-browser install`` (downloads Chromium) or
  ``agent-browser install --with-deps`` (also installs system libraries for
  Debian/Ubuntu/Docker).
- **Cloud mode**: Browserbase or Browser Use cloud execution when configured.
- Session isolation per task ID
- Text-based page snapshots using accessibility tree
- Element interaction via ref selectors (@e1, @e2, etc.)
- Task-aware content extraction using LLM summarization
- Automatic cleanup of browser sessions

Environment Variables:
- BROWSERBASE_API_KEY: API key for direct Browserbase cloud mode
- BROWSERBASE_PROJECT_ID: Project ID for direct Browserbase cloud mode
- BROWSER_USE_API_KEY: API key for direct Browser Use cloud mode
- BROWSERBASE_PROXIES: Enable/disable residential proxies (default: "true")
- BROWSERBASE_ADVANCED_STEALTH: Enable advanced stealth mode with custom Chromium,
  requires Scale Plan (default: "false")
- BROWSERBASE_KEEP_ALIVE: Enable keepAlive for session reconnection after disconnects,
  requires paid plan (default: "true")
- BROWSERBASE_SESSION_TIMEOUT: Custom session timeout in seconds (max 21600 = 6h).
  Set to extend beyond project default. Common values: 600 (10min), 1800 (30min) (default: none)

Usage:
    from tools.browser_tool import browser_navigate, browser_snapshot, browser_click

    # Navigate to a page
    result = browser_navigate("https://example.com", task_id="task_123")

    # Get page snapshot
    snapshot = browser_snapshot(task_id="task_123")

    # Click an element
    browser_click("@e5", task_id="task_123")
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import shutil
import stat
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path

import aiofiles
import aiofiles.os
import httpx

from agent.redact import redact_cdp_url
from hermes_constants import (
    agent_browser_runnable,
    get_hermes_home,
    get_hermes_home_override,
)
from utils import env_int, is_truthy_value
from hermes_cli.config import DEFAULT_CONFIG, cfg_get, load_config_readonly
from hermes_cli._subprocess_compat import windows_hide_flags


def __getattr__(name: str):
    """Lazy module attributes (PEP 562) — import diet for cold start.

    ``agent.auxiliary_client.call_llm`` is only needed on extraction and vision
    paths, so it loads on first use. The module-level name is preserved for the
    test-patch surface.
    """
    if name == "call_llm":
        from agent.auxiliary_client import call_llm as _call_llm

        globals()["call_llm"] = _call_llm
        return _call_llm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


async def _lazy_call_llm(*args, **kwargs):
    """Invoke ``call_llm`` through module globals so test patches of
    ``tools.browser_tool.call_llm`` are honored, importing lazily otherwise."""
    fn = globals().get("call_llm")
    if fn is None:
        fn = __getattr__("call_llm")
    return await fn(*args, **kwargs)


# Browser-specific tool keys passed through to the agent-browser subprocess
# AFTER credential stripping.  agent-browser is a Node process loading npm
# deps; handing it the full operator keyring (#29157 / GHSA-m4m8-xjp4-5rmm)
# means a compromised transitive dependency could read every Hermes secret
# straight out of process.env.  Strip by default, then re-add only the
# browser-backend keys the worker legitimately needs.
_BROWSER_PASSTHROUGH_KEYS: tuple[str, ...] = (
    "BROWSERBASE_API_KEY",
    "BROWSERBASE_PROJECT_ID",
    "BROWSER_USE_API_KEY",
    "FIRECRAWL_API_KEY",
    "FIRECRAWL_API_URL",
    "FIRECRAWL_BROWSER_TTL",
)


def _build_browser_env() -> dict:
    """Credential-scrubbed env for an agent-browser subprocess.

    Strips Hermes-managed secrets (provider keys, gateway tokens, GitHub auth,
    infra secrets) then re-adds only the browser-backend keys the worker needs.
    The ``hermes_subprocess_env`` import is deferred to keep ``browser_tool``
    importable under test harnesses that load it against a stubbed ``tools``
    package (tests/tools/test_managed_browserbase_and_modal.py).
    """
    from tools.environments.local import hermes_subprocess_env

    env = hermes_subprocess_env(inherit_credentials=False)
    for _key in _BROWSER_PASSTHROUGH_KEYS:
        if _key in os.environ:
            env[_key] = os.environ[_key]
    return env


from tools.website_policy import check_website_access
from tools.url_safety import (
    is_always_blocked_url as _is_always_blocked_url,
    is_safe_url as _is_safe_url,
    normalize_url_for_request as _normalize_url_for_request,
    sensitive_query_param_name as _sensitive_query_param_name,
)
# Browser-provider ABC + registry — PR #25214 moved the per-vendor providers
# (Browserbase / Browser Use / Firecrawl) out of ``tools/browser_providers/``
# and into ``plugins/browser/<vendor>/``. The dispatcher consults the
# registry; the legacy class names are re-exported below as backward-compat
# shims for callers that import them from this module.
from agent.browser_provider import BrowserProvider as CloudBrowserProvider  # noqa: F401  (legacy alias)
from agent.browser_registry import (  # noqa: F401  (test-patchable surface)
    get_provider as _registry_get_browser_provider,
)
from plugins.browser.browserbase.provider import (  # noqa: F401  (legacy import surface)
    BrowserbaseBrowserProvider as BrowserbaseProvider,
)
from plugins.browser.browser_use.provider import (  # noqa: F401
    BrowserUseBrowserProvider as BrowserUseProvider,
)
from plugins.browser.firecrawl.provider import (  # noqa: F401
    FirecrawlBrowserProvider as FirecrawlProvider,
)
from tools.tool_backend_helpers import normalize_browser_cloud_provider

# Camofox local anti-detection browser backend.
# When CAMOFOX_URL is set, all browser operations route through the
# camofox REST API instead of the agent-browser CLI.
from tools.browser_camofox import is_camofox_mode as _is_camofox_mode


logger = logging.getLogger(__name__)

# Standard PATH entries for environments with minimal PATH (e.g. systemd services).
# Includes Android/Termux and macOS Homebrew locations needed for agent-browser,
# npx, node, and Android's glibc runner (grun).
_SANE_PATH_DIRS = (
    "/data/data/com.termux/files/usr/bin",
    "/data/data/com.termux/files/usr/sbin",
    "/opt/homebrew/bin",
    "/opt/homebrew/sbin",
    "/usr/local/sbin",
    "/usr/local/bin",
    "/usr/sbin",
    "/usr/bin",
    "/sbin",
    "/bin",
)
_SANE_PATH = os.pathsep.join(_SANE_PATH_DIRS)


_cached_homebrew_node_dirs: Optional[tuple[str, ...]] = None


async def _discover_homebrew_node_dirs() -> tuple[str, ...]:
    """Find Homebrew versioned Node.js bin directories (e.g. node@20, node@24).

    When Node is installed via ``brew install node@24`` and NOT linked into
    /opt/homebrew/bin, agent-browser isn't discoverable on the default PATH.
    This function finds those directories so they can be prepended.
    """
    global _cached_homebrew_node_dirs
    if _cached_homebrew_node_dirs is not None:
        return _cached_homebrew_node_dirs

    dirs: list[str] = []
    homebrew_opt = "/opt/homebrew/opt"
    if not await aiofiles.os.path.isdir(homebrew_opt):
        _cached_homebrew_node_dirs = ()
        return _cached_homebrew_node_dirs
    try:
        for entry in await aiofiles.os.listdir(homebrew_opt):
            if entry.startswith("node") and entry != "node":
                bin_dir = os.path.join(homebrew_opt, entry, "bin")
                if await aiofiles.os.path.isdir(bin_dir):
                    dirs.append(bin_dir)
    except OSError:
        pass
    _cached_homebrew_node_dirs = tuple(dirs)
    return _cached_homebrew_node_dirs


async def _browser_candidate_path_dirs() -> list[str]:
    """Return ordered browser CLI PATH candidates shared by discovery and execution."""
    hermes_home = get_hermes_home()
    hermes_node_bin = str(hermes_home / "node" / "bin")
    hermes_node_root = str(hermes_home / "node")
    hermes_nm_bin = str(hermes_home / "node_modules" / ".bin")
    return [
        hermes_node_bin,
        hermes_node_root,
        hermes_nm_bin,
        *list(await _discover_homebrew_node_dirs()),
        *_SANE_PATH_DIRS,
    ]


async def _merge_browser_path(existing_path: str = "") -> str:
    """Prepend browser-specific PATH fallbacks without reordering existing entries."""
    path_parts = [p for p in (existing_path or "").split(os.pathsep) if p]
    existing_parts = set(path_parts)
    prefix_parts: list[str] = []

    for part in await _browser_candidate_path_dirs():
        if not part or part in existing_parts or part in prefix_parts:
            continue
        if await aiofiles.os.path.isdir(part):
            prefix_parts.append(part)

    return os.pathsep.join(prefix_parts + path_parts)


# Throttle screenshot cleanup to avoid repeated full directory scans.
_last_screenshot_cleanup_by_dir: dict[str, float] = {}

# ============================================================================
# Configuration
# ============================================================================

# Default timeout for browser commands (seconds)
DEFAULT_COMMAND_TIMEOUT = 30

# Floor for ``open`` (navigate) — cold daemon + first Chromium launch can exceed
# the generic command_timeout on slow or library-starved Linux hosts.
MIN_OPEN_TIMEOUT = 60
MIN_FIRST_OPEN_TIMEOUT = 120

# Max chars for snapshot content before truncation/summarization. Aligned
# with web_tools.DEFAULT_EXTRACT_CHAR_LIMIT (15000) — the snapshot and
# web_extract paths share the same truncate-and-store pattern, so the model
# gets the same per-page budget from both.
SNAPSHOT_SUMMARIZE_THRESHOLD = 15000

# Hard ceiling on the full-snapshot file written to cache/web when a snapshot
# is truncated or LLM-summarized. Mirrors web_tools.MAX_STORED_TEXT_CHARS —
# the model only ever sees the truncated view; the stored copy exists for
# read_file paging and must not write unbounded bytes to disk.
MAX_STORED_SNAPSHOT_CHARS = 2_000_000

# Commands that legitimately return empty stdout (e.g. close, record).
_EMPTY_OK_COMMANDS: frozenset = frozenset({"close", "record"})

_cached_command_timeout: Optional[int] = None
_command_timeout_resolved = False


def _sanitize_url_for_logs(value: object) -> str:
    """Mask secrets in logged browser endpoint URLs and URL-like errors.

    Thin wrapper over :func:`agent.redact.redact_cdp_url`, which is the single
    source of truth for CDP-URL log redaction. Kept as a local name because
    several browser-tool log sites reference it; the redaction policy itself
    lives once in ``redact.py`` so the browser tool and the CDP supervisor
    cannot drift apart.
    """
    return redact_cdp_url(value)


async def _get_command_timeout() -> int:
    """Return the configured browser command timeout from config.yaml.

    Reads ``config["browser"]["command_timeout"]`` and falls back to
    ``DEFAULT_COMMAND_TIMEOUT`` (30s) if unset or unreadable.  Result is
    cached after the first call and cleared by ``cleanup_all_browsers()``.
    """
    global _cached_command_timeout, _command_timeout_resolved
    if _command_timeout_resolved and _cached_command_timeout is not None:
        return _cached_command_timeout

    result = DEFAULT_COMMAND_TIMEOUT
    try:
        cfg = await load_config_readonly()
        val = cfg_get(cfg, "browser", "command_timeout")
        if val is not None:
            result = max(int(val), 5)  # Floor at 5s to avoid instant kills
    except Exception as e:
        logger.debug("Could not read command_timeout from config: %s", e)
    # Assign the cached value BEFORE flipping the resolved flag so a
    # concurrent reader cannot observe ``resolved=True`` while the cache
    # is still ``None`` (see issue #14331).
    _cached_command_timeout = result
    _command_timeout_resolved = True
    return result


async def _safe_command_timeout() -> int:
    """Like ``_get_command_timeout`` but guaranteed non-None.

    Defense in depth against the race fixed in ``_get_command_timeout``:
    if anything ever returns ``None`` (e.g. cache reset mid-flight), fall
    back to ``DEFAULT_COMMAND_TIMEOUT``. Uses ``is not None`` rather than
    ``or`` so a legitimately configured ``0`` is preserved.
    """
    val = await _get_command_timeout()
    return val if val is not None else DEFAULT_COMMAND_TIMEOUT


async def _get_open_command_timeout(*, first_open: bool = False) -> int:
    """Timeout for agent-browser ``open`` (navigation / daemon cold start)."""
    base = await _safe_command_timeout()
    floor = MIN_FIRST_OPEN_TIMEOUT if first_open else MIN_OPEN_TIMEOUT
    return max(base, floor)


async def _needs_chromium_sandbox_bypass() -> bool:
    """Return True when Chromium needs --no-sandbox to start reliably."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return True
    if await _running_in_docker():
        return True
    userns_restrict = "/proc/sys/kernel/apparmor_restrict_unprivileged_userns"
    try:
        async with aiofiles.open(userns_restrict, encoding="utf-8") as f:
            if (await f.read()).strip() == "1":
                return True
    except OSError:
        pass
    return False


async def _read_command_output_files(
    stdout_path: str, stderr_path: str
) -> tuple[str, str]:
    """Best-effort read of agent-browser stdout/stderr temp files."""
    stdout = stderr = ""
    for path, slot in ((stdout_path, "stdout"), (stderr_path, "stderr")):
        try:
            async with aiofiles.open(path, "r", encoding="utf-8") as f:
                text = (await f.read()).strip()
        except OSError:
            continue
        if slot == "stdout":
            stdout = text
        else:
            stderr = text
    return stdout, stderr


async def _unlink_command_output_files(*paths: str) -> None:
    for path in paths:
        try:
            await aiofiles.os.remove(path)
        except OSError:
            pass


async def _format_browser_timeout_error(
    command: str,
    timeout: int,
    stdout: str,
    stderr: str,
) -> str:
    """Build an actionable timeout message from captured daemon output."""
    parts = [f"Command timed out after {timeout} seconds"]
    detail = (stderr or stdout or "").strip()
    if detail:
        parts.append(detail[:1500])

    combined = f"{stderr}\n{stdout}".lower()
    hints: list[str] = []
    if "sandbox" in combined:
        hints.append(
            "Chromium sandbox launch failed. Set AGENT_BROWSER_ARGS="
            "'--no-sandbox,--disable-dev-shm-usage' in your environment, "
            "or run: npx agent-browser install --with-deps"
        )
    elif command == "open" and await _is_local_mode():
        if await _running_in_docker():
            hints.append(
                "The browser daemon may still be starting or Chromium may be "
                "missing. Pull the latest image: "
                "docker pull ghcr.io/nousresearch/hermes-agent:latest"
            )
        else:
            hints.append(
                "The browser daemon may still be starting, or Chromium may be "
                "missing system libraries. Install/repair with: "
                "npx agent-browser install --with-deps "
                "(or: npx playwright install --with-deps chromium)"
            )
    if hints:
        parts.extend(hints)
    return "\n".join(parts)


def _get_vision_model() -> Optional[str]:
    """Model for browser_vision (screenshot analysis — multimodal)."""
    return os.getenv("AUXILIARY_VISION_MODEL", "").strip() or None


def _get_extraction_model() -> Optional[str]:
    """Model for page snapshot text summarization — same as web_extract."""
    return os.getenv("AUXILIARY_WEB_EXTRACT_MODEL", "").strip() or None


async def _resolve_cdp_override(cdp_url: str) -> str:
    """Normalize a user-supplied CDP endpoint into a concrete connectable URL.

    Accepts:
    - full websocket endpoints: ws://host:port/devtools/browser/...
    - HTTP discovery endpoints: http://host:port or http://host:port/json/version
    - bare websocket host:port values like ws://host:port

    For discovery-style endpoints we fetch /json/version and return the
    webSocketDebuggerUrl so downstream tools always receive a concrete browser
    websocket instead of an ambiguous host:port URL.
    """
    raw = (cdp_url or "").strip()
    if not raw:
        return ""

    lowered = raw.lower()
    if "/devtools/browser/" in lowered:
        return raw

    discovery_url = raw
    if lowered.startswith(("ws://", "wss://")):
        if (
            raw.count(":") == 2
            and raw.rstrip("/").rsplit(":", 1)[-1].isdigit()
            and "/" not in raw.split(":", 2)[-1]
        ):
            discovery_url = (
                "http://" if lowered.startswith("ws://") else "https://"
            ) + raw.split("://", 1)[1]
        else:
            return raw

    if discovery_url.lower().endswith("/json/version"):
        version_url = discovery_url
    else:
        version_url = discovery_url.rstrip("/") + "/json/version"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(version_url)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.warning(
            "Failed to resolve CDP endpoint %s via %s: %s",
            _sanitize_url_for_logs(raw),
            _sanitize_url_for_logs(version_url),
            _sanitize_url_for_logs(exc),
        )
        return raw

    ws_url = str(payload.get("webSocketDebuggerUrl") or "").strip()
    if ws_url:
        logger.info(
            "Resolved CDP endpoint %s -> %s",
            _sanitize_url_for_logs(raw),
            _sanitize_url_for_logs(ws_url),
        )
        return ws_url

    logger.warning(
        "CDP discovery at %s did not return webSocketDebuggerUrl; using raw endpoint",
        _sanitize_url_for_logs(version_url),
    )
    return raw


async def _get_cdp_override_raw() -> str:
    """Return the *configured* CDP override without any network I/O.

    Precedence is:
    1. ``BROWSER_CDP_URL`` env var (live override from ``/browser connect``)
    2. ``browser.cdp_url`` in config.yaml (persistent config)

    This is the availability-check variant: callers that only need to know
    *whether* a CDP override is configured (tool ``check_fn`` gates,
    ``_is_local_mode`` / ``_is_local_backend`` routing decisions,
    ``hermes doctor``) MUST use this instead of :func:`_get_cdp_override`.

    Rationale: ``_get_cdp_override`` resolves the endpoint over HTTP
    (``/json/version`` discovery, 10s timeout). Tool-schema assembly runs at
    every CLI/Desktop startup and probes several browser-family check_fns;
    when a *stale* ``browser.cdp_url`` points at a dead endpoint (the debug
    Chrome it referenced is long gone), each check blocked on a failing
    socket connect and startup stalled for 10+ seconds before the banner —
    with no error, just mystery slowness. Same principle as the existing
    "do not execute ``agent-browser --version`` here" rule in
    ``check_browser_requirements``: no side effects during schema build.
    """
    env_override = os.environ.get("BROWSER_CDP_URL", "").strip()
    if env_override:
        return env_override

    try:
        cfg = await load_config_readonly()
        browser_cfg = cfg.get("browser", {})
        if isinstance(browser_cfg, dict):
            return str(browser_cfg.get("cdp_url", "") or "").strip()
    except Exception as e:
        logger.debug("Could not read browser.cdp_url from config: %s", e)

    return ""


async def _get_cdp_override() -> str:
    """Return a normalized CDP URL override, or empty string.

    Precedence is:
    1. ``BROWSER_CDP_URL`` env var (live override from ``/browser connect``)
    2. ``browser.cdp_url`` in config.yaml (persistent config)

    When either is set, we skip both Browserbase and the local headless
    launcher and connect directly to the supplied Chrome DevTools Protocol
    endpoint.

    NOTE: resolution may perform an HTTP ``/json/version`` discovery request.
    Only call this on paths that are about to *connect* (session creation,
    supervisor attach). Pure is-it-configured gates must use
    :func:`_get_cdp_override_raw`.
    """
    raw = await _get_cdp_override_raw()
    if not raw:
        return ""
    return await _resolve_cdp_override(raw)


async def _get_dialog_policy_config() -> Tuple[str, float]:
    """Read ``browser.dialog_policy`` + ``browser.dialog_timeout_s`` from config.

    Returns a ``(policy, timeout_s)`` tuple, falling back to the supervisor's
    defaults when keys are absent or invalid.
    """
    # Defer imports so browser_tool can be imported in minimal environments.
    from tools.browser_supervisor import (
        DEFAULT_DIALOG_POLICY,
        DEFAULT_DIALOG_TIMEOUT_S,
        _VALID_POLICIES,
    )

    try:
        cfg = await load_config_readonly()
        browser_cfg = cfg.get("browser", {}) if isinstance(cfg, dict) else {}
        if not isinstance(browser_cfg, dict):
            return DEFAULT_DIALOG_POLICY, DEFAULT_DIALOG_TIMEOUT_S
        policy = str(browser_cfg.get("dialog_policy") or DEFAULT_DIALOG_POLICY)
        if policy not in _VALID_POLICIES:
            logger.debug("Invalid browser.dialog_policy=%r; using default", policy)
            policy = DEFAULT_DIALOG_POLICY
        timeout_raw = browser_cfg.get("dialog_timeout_s")
        try:
            timeout_s = (
                float(timeout_raw)
                if timeout_raw is not None
                else DEFAULT_DIALOG_TIMEOUT_S
            )
            if timeout_s <= 0:
                timeout_s = DEFAULT_DIALOG_TIMEOUT_S
        except (TypeError, ValueError):
            timeout_s = DEFAULT_DIALOG_TIMEOUT_S
        return policy, timeout_s
    except Exception:
        return DEFAULT_DIALOG_POLICY, DEFAULT_DIALOG_TIMEOUT_S


async def _ensure_cdp_supervisor(task_id: str) -> None:
    """Start a CDP supervisor for ``task_id`` if an endpoint is reachable.

    Idempotent — delegates to ``SupervisorRegistry.get_or_start`` which skips
    when a supervisor for this ``(task_id, cdp_url)`` already exists and
    tears down + restarts on URL change. Safe to call on every
    ``browser_navigate`` / ``/browser connect`` without worrying about
    double-attach.

    Resolves the CDP URL in this order:
      1. ``BROWSER_CDP_URL`` / ``browser.cdp_url`` — covers ``/browser connect``
         and config-set overrides.
      2. ``_active_sessions[task_id]["cdp_url"]`` — covers Browserbase + any
         other cloud provider whose ``create_session`` returns a raw CDP URL.

    Swallows all errors — failing to attach the supervisor must not break
    the browser session itself.  The agent simply won't see
    ``pending_dialogs`` / ``frame_tree`` fields in snapshots.
    """
    cdp_url = await _get_cdp_override()
    if not cdp_url:
        # Fallback: active session may carry a per-session CDP URL from a
        # cloud provider (Browserbase sets this).
        async with _cleanup_lock:
            session_info = _active_sessions.get(task_id, {})
        maybe = str(session_info.get("cdp_url") or "")
        if maybe:
            cdp_url = await _resolve_cdp_override(maybe)
    if not cdp_url:
        return
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY

        policy, timeout_s = await _get_dialog_policy_config()
        await SUPERVISOR_REGISTRY.get_or_start(
            task_id=task_id,
            cdp_url=cdp_url,
            dialog_policy=policy,
            dialog_timeout_s=timeout_s,
        )
    except Exception as exc:
        logger.debug(
            "CDP supervisor attach for task=%s failed (non-fatal): %s",
            task_id,
            exc,
        )


async def _stop_cdp_supervisor(task_id: str) -> None:
    """Stop the CDP supervisor for ``task_id`` if one exists. No-op otherwise."""
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY

        await SUPERVISOR_REGISTRY.stop(task_id)
    except Exception as exc:
        logger.debug(
            "CDP supervisor stop for task=%s failed (non-fatal): %s", task_id, exc
        )


# ============================================================================
# Cloud Provider Registry
# ============================================================================
#
# Per-vendor browser providers (Browserbase / Browser Use / Firecrawl) live as
# plugins under ``plugins/browser/<vendor>/`` and self-register through
# :mod:`agent.browser_registry` at plugin-discovery time. The legacy
# class-name registry below is preserved as a backward-compat shim so test
# fixtures that ``monkeypatch.setattr(browser_tool, "_PROVIDER_REGISTRY", ...)``
# keep working — but ``_get_cloud_provider()`` now consults
# :mod:`agent.browser_registry` for the actual lookup.
#
# When the test patches ``_PROVIDER_REGISTRY``, we honour it (so the cache
# unit tests still drive the function); otherwise the registry-backed path
# wins. This keeps the test surface stable while letting third-party
# plugins drop in under ``~/.hermes/plugins/browser/<vendor>/``.

_PROVIDER_REGISTRY: Dict[str, type] = {
    "browserbase": BrowserbaseProvider,
    "browser-use": BrowserUseProvider,
    "firecrawl": FirecrawlProvider,
}
# Frozen copy of the import-time _PROVIDER_REGISTRY, used by
# ``_is_legacy_provider_registry_overridden`` to detect test-time
# monkeypatching. NEVER mutate this dict.
_DEFAULT_PROVIDER_REGISTRY: Dict[str, type] = dict(_PROVIDER_REGISTRY)

_cached_cloud_provider: Optional[CloudBrowserProvider] = None
_cloud_provider_resolved = False
_allow_private_urls_resolved = False
_cached_allow_private_urls: Optional[bool] = None
_cached_agent_browser: Optional[str] = None
_agent_browser_resolved = False

# Lightpanda engine support — cached like _get_cloud_provider().
# agent-browser v0.25.3+ supports ``--engine lightpanda`` natively.
_cached_browser_engine: Optional[str] = None
_browser_engine_resolved = False


def _is_legacy_provider_registry_overridden() -> bool:
    """Return True when a test has patched ``_PROVIDER_REGISTRY`` to a custom value.

    Detected by spotting any registered class that *isn't* the canonical
    plugin-backed class for that name. Tests that
    ``monkeypatch.setattr(browser_tool, "_PROVIDER_REGISTRY", ...)`` install
    custom factories (`exploding_factory`, `lambda: fake_provider`, etc.);
    those entries fail the canonical-class identity check below.

    Note: a future maintainer adding a 4th built-in provider only needs to
    extend ``_DEFAULT_PROVIDER_REGISTRY`` below — they do NOT need to update
    a hardcoded set of keys here. The detection just compares each registered
    value against the corresponding canonical class.
    """
    try:
        for key, default_cls in _DEFAULT_PROVIDER_REGISTRY.items():
            if _PROVIDER_REGISTRY.get(key) is not default_cls:
                return True
        # Extra keys not in the default registry → also an override.
        return len(_PROVIDER_REGISTRY) != len(_DEFAULT_PROVIDER_REGISTRY)
    except Exception:
        return False


async def _ensure_browser_plugins_loaded() -> None:
    """Idempotently trigger plugin discovery so the browser registry is populated.

    Normally `model_tools` is imported early in any session and that
    triggers `discover_plugins()` as a side effect. But `_get_cloud_provider`
    can be called from contexts that haven't gone through `model_tools` —
    standalone scripts, certain unit-test paths, the parity-sweep harness.
    Make discovery idempotent and side-effect-only here so users always
    see registered plugins regardless of import order. Cheap: subsequent
    calls early-return inside `_ensure_plugins_discovered`.
    """
    try:
        from hermes_cli.plugins import _ensure_plugins_discovered

        await _ensure_plugins_discovered()
    except Exception as exc:
        logger.debug("Browser plugin discovery failed (non-fatal): %s", exc)


async def _get_cloud_provider() -> Optional[CloudBrowserProvider]:
    """Return the configured cloud browser provider, or None for local mode.

    Reads ``config["browser"]["cloud_provider"]`` once and caches the result
    for the process lifetime. An explicit ``local`` provider disables cloud
    fallback. If unset, fall back to Browser Use (managed Nous gateway or
    direct API key) and then Browserbase (direct credentials only) — the
    historic auto-detect order, now expressed as the
    :data:`agent.browser_registry._LEGACY_PREFERENCE` walk.

    Selection routes through :mod:`agent.browser_registry` so third-party
    browser plugins (``~/.hermes/plugins/browser/<vendor>/``) participate
    in explicit-config resolution. Test fixtures that override
    ``_PROVIDER_REGISTRY`` or ``BrowserUseProvider`` / ``BrowserbaseProvider``
    on this module still drive the function — see
    ``_is_legacy_provider_registry_overridden``.
    """
    global _cached_cloud_provider, _cloud_provider_resolved
    if _cloud_provider_resolved:
        return _cached_cloud_provider

    resolved: Optional[CloudBrowserProvider] = None
    try:
        cfg = await load_config_readonly()
        browser_cfg = cfg.get("browser", {})
        provider_key = None
        if isinstance(browser_cfg, dict) and "cloud_provider" in browser_cfg:
            provider_key = normalize_browser_cloud_provider(
                browser_cfg.get("cloud_provider")
            )
            if provider_key == "local":
                _cached_cloud_provider = None
                _cloud_provider_resolved = True
                return None
        if provider_key:
            try:
                if _is_legacy_provider_registry_overridden():
                    # Test fixture path: honour the patched dict so the
                    # cache-policy unit tests keep working.
                    factory = _PROVIDER_REGISTRY.get(provider_key)
                    if factory is not None:
                        resolved = factory()
                else:
                    # Ensure plugins are discovered so the registry is
                    # populated. Idempotent — cheap on subsequent calls.
                    await _ensure_browser_plugins_loaded()
                    resolved = _registry_get_browser_provider(provider_key)
                    if resolved is None:
                        # Explicit config name unknown to the registry —
                        # might be a typo, an uninstalled plugin, or a
                        # registry-population failure. Warn the user
                        # (legacy code would have surfaced a typed
                        # credentials error via direct class instantiation;
                        # post-migration we surface this WARNING instead).
                        logger.warning(
                            "browser.cloud_provider=%r is not a registered "
                            "browser plugin; falling back to auto-detect "
                            "(install the corresponding plugin or fix the "
                            "config key spelling).",
                            provider_key,
                        )
            except Exception:
                logger.warning(
                    "Failed to instantiate explicit cloud_provider %r; will retry on next call",
                    provider_key,
                    exc_info=True,
                )
                return None
    except Exception as e:
        # Config file may be temporarily unreadable; still try auto-detect so
        # env-based / managed-gateway credentials can resolve. Don't pin cache.
        logger.debug("Could not read cloud_provider from config: %s", e)

    if resolved is None:
        # Auto-detect path: Browser Use first (managed Nous gateway or
        # direct API key), then Browserbase (direct credentials). Uses
        # the legacy class names imported at the top of this module so
        # tests that ``monkeypatch.setattr(browser_tool, "BrowserUseProvider", ...)``
        # keep driving this branch deterministically. Third-party browser
        # plugins are intentionally NOT reachable from auto-detect — they
        # participate only via explicit ``browser.cloud_provider: <name>``,
        # mirroring the firecrawl gate documented on
        # :data:`agent.browser_registry._LEGACY_PREFERENCE`.
        try:
            fallback_provider = BrowserUseProvider()
            if await fallback_provider.is_available():
                resolved = fallback_provider
            else:
                fallback_provider = BrowserbaseProvider()
                if await fallback_provider.is_available():
                    resolved = fallback_provider
        except Exception:  # pragma: no cover - defensive: never poison cache
            logger.debug("Cloud provider auto-detect failed", exc_info=True)
            return None

    if resolved is None:
        # Transient None — credentials may self-heal. Don't poison the cache.
        return None

    _cached_cloud_provider = resolved
    _cloud_provider_resolved = True
    return _cached_cloud_provider


from hermes_constants import is_termux as _is_termux_environment


def _browser_install_hint() -> str:
    if _is_termux_environment():
        return "npm install -g agent-browser && agent-browser install"
    return "npm install -g agent-browser && agent-browser install --with-deps"


async def _requires_real_termux_browser_install(browser_cmd: str) -> bool:
    return (
        _is_termux_environment()
        and await _is_local_mode()
        and browser_cmd.strip() == "npx agent-browser"
    )


def _termux_browser_install_error() -> str:
    return (
        "Local browser automation on Termux cannot rely on the bare npx fallback. "
        f"Install agent-browser explicitly first: {_browser_install_hint()}"
    )


async def _is_local_mode() -> bool:
    """Return True when the browser tool will use a local browser backend."""
    if await _get_cdp_override_raw():
        return False
    return await _get_cloud_provider() is None


async def _is_local_backend() -> bool:
    """Return True when the browser runs locally AND the terminal is also local.

    SSRF protection is only meaningful for cloud backends (Browserbase,
    BrowserUse) where the agent could reach internal resources on a remote
    machine.  For local backends — Camofox, or the built-in headless
    Chromium without a cloud provider — the user already has full terminal
    and network access on the same machine, so the check adds no security
    value.

    However, when the terminal runs in a container (docker, modal, daytona,
    ssh, singularity), the browser on the host can access internal networks
    that the terminal cannot.  In this case, SSRF protection should be
    enabled even though the browser is technically "local".
    """
    # A CDP override points the browser at a separate Chrome process whose
    # network position is not guaranteed to match the terminal (it may live
    # off-host). Don't treat it as a trusted local backend — otherwise a
    # model-driven navigate could reach internal/metadata services reachable
    # from the CDP host but not the terminal. This MUST be checked before the
    # camofox short-circuit below so a Camofox backend combined with a CDP
    # override still fails the local check instead of returning local and
    # skipping the private/internal SSRF gate. The override is honored from
    # either the BROWSER_CDP_URL env var or a persistent `browser.cdp_url`
    # config (both via _get_cdp_override(), and both now suppress camofox in
    # browser_camofox.py). _is_local_mode() already treats any CDP override as
    # non-local; keep the two helpers in agreement.
    if await _get_cdp_override_raw():
        return False
    if await _is_camofox_mode():
        return True
    if await _get_cloud_provider() is not None:
        return False
    # When terminal runs in a container, browser on host can access
    # internal networks the terminal can't → treat as non-local.
    terminal_backend = os.getenv("TERMINAL_ENV", "local").strip().lower()
    return terminal_backend in ("local", "")


_auto_local_for_private_urls_resolved = False
_cached_auto_local_for_private_urls: bool = True


async def _get_browser_engine() -> str:
    """Return the configured browser engine (``auto``, ``lightpanda``, or ``chrome``).

    Reads ``config["browser"]["engine"]`` once and caches the result.
    Falls back to the ``AGENT_BROWSER_ENGINE`` env var, then ``auto``.

    ``auto`` means: don't pass ``--engine`` at all (agent-browser defaults to
    Chrome).  ``lightpanda`` or ``chrome`` are forwarded as
    ``--engine <value>`` to agent-browser v0.25.3+.

    Lightpanda is 1.3-5.8x faster on navigation but has no graphical
    renderer (no screenshots).
    """
    global _cached_browser_engine, _browser_engine_resolved
    if _browser_engine_resolved and _cached_browser_engine is not None:
        return _cached_browser_engine

    _browser_engine_resolved = True
    _cached_browser_engine = "auto"  # safe default

    # Config file takes priority
    try:
        cfg = await load_config_readonly()
        val = cfg.get("browser", {}).get("engine")
        if val and str(val).strip():
            _cached_browser_engine = str(val).strip().lower()
    except Exception as e:
        logger.debug("Could not read browser.engine from config: %s", e)

    # Fall back to env var (only if config didn't set a value)
    if _cached_browser_engine == "auto":
        env_val = os.environ.get("AGENT_BROWSER_ENGINE", "").strip().lower()
        if env_val:
            _cached_browser_engine = env_val

    # Validate: agent-browser only accepts "chrome" and "lightpanda".
    _VALID_ENGINES = {"auto", "lightpanda", "chrome"}
    if _cached_browser_engine not in _VALID_ENGINES:
        logger.warning(
            "Unknown browser engine %r (valid: %s), falling back to 'auto'",
            _cached_browser_engine,
            ", ".join(sorted(_VALID_ENGINES)),
        )
        _cached_browser_engine = "auto"

    return _cached_browser_engine


_cached_headed_mode: Optional[bool] = None
_headed_mode_resolved = False


async def _is_headed_mode() -> bool:
    """Return True when the browser should launch in headed (visible) mode.

    Reads ``config["browser"]["headed"]`` with ``AGENT_BROWSER_HEADED`` env
    var as fallback.  Result is cached after the first call.
    """
    global _cached_headed_mode, _headed_mode_resolved
    if _headed_mode_resolved and _cached_headed_mode is not None:
        return _cached_headed_mode

    _headed_mode_resolved = True
    _cached_headed_mode = False

    try:
        cfg = await load_config_readonly()
        val = cfg.get("browser", {}).get("headed")
        if val is not None:
            _cached_headed_mode = str(val).strip().lower() in ("true", "1", "yes")
    except Exception as e:
        logger.debug("Could not read browser.headed from config: %s", e)

    if not _cached_headed_mode:
        env_val = os.environ.get("AGENT_BROWSER_HEADED", "").strip()
        if env_val and env_val.lower() in ("true", "1", "yes"):
            _cached_headed_mode = True

    return _cached_headed_mode


async def _should_inject_engine(engine: str) -> bool:
    """Return True when the engine flag should be added to agent-browser commands.

    Only inject ``--engine`` for non-cloud, non-camofox local sessions where
    the engine is explicitly set (not ``auto``).
    """
    if engine == "auto":
        return False
    if await _is_camofox_mode():
        return False
    return await _is_local_mode()


async def _using_lightpanda_engine() -> bool:
    """Return True when local browser commands are configured for Lightpanda."""
    return await _get_browser_engine() == "lightpanda"


def _lightpanda_fallback_reason(
    engine: str, command: str, result: Dict[str, Any]
) -> Optional[str]:
    """Return the user-visible reason a Lightpanda result needs Chrome fallback.

    ``None`` means no fallback should run.  The returned string is copied into
    the fallback result so CLI/TUI/gateway users can see when Hermes silently
    switched from Lightpanda to Chrome for completeness.
    """
    if engine != "lightpanda":
        return None

    # Only retry commands where Chrome can meaningfully produce a different
    # result. Session-management commands (close, record) are tied to the
    # engine's daemon and can't be retried on a different engine.
    _FALLBACK_ELIGIBLE = {
        "open",
        "snapshot",
        "screenshot",
        "eval",
        "click",
        "fill",
        "scroll",
        "back",
        "press",
        "console",
        "errors",
    }
    if command not in _FALLBACK_ELIGIBLE:
        return None

    # Explicit failure
    if not result.get("success"):
        error = str(result.get("error") or "command failed").strip()
        return f"Lightpanda {command!r} failed ({error}); retried with Chrome."

    data = result.get("data", {})

    if command == "snapshot":
        snap = data.get("snapshot", "")
        # Empty or near-empty snapshots indicate Lightpanda couldn't render
        if not snap or len(snap.strip()) < 20:
            return (
                "Lightpanda returned an empty/too-short snapshot; retried with Chrome."
            )

    if command == "screenshot":
        # Lightpanda returns a placeholder PNG with its panda logo.
        # Since LP PR #1766 resized it to 1920x1080, the placeholder is
        # ~17 KB.  Real Chromium screenshots are typically 100 KB+.
        path = data.get("path", "")
        if path:
            try:
                size = os.path.getsize(path)
                if size < 20480:
                    logger.debug(
                        "Lightpanda screenshot is suspiciously small (%d bytes), "
                        "triggering Chrome fallback",
                        size,
                    )
                    return (
                        f"Lightpanda screenshot was suspiciously small ({size} bytes); "
                        "retried with Chrome."
                    )
            except OSError:
                return "Lightpanda screenshot file was missing/unreadable; retried with Chrome."

    return None


def _needs_lightpanda_fallback(
    engine: str, command: str, result: Dict[str, Any]
) -> bool:
    """Check if a Lightpanda result should trigger an automatic Chrome fallback."""
    return _lightpanda_fallback_reason(engine, command, result) is not None


def _annotate_lightpanda_fallback(
    result: Dict[str, Any], reason: str
) -> Dict[str, Any]:
    """Add a user-visible Chrome fallback warning to a browser command result."""
    warning = (
        f"⚠ Lightpanda fallback: Chrome was used for this browser action. {reason}"
    )
    annotated = dict(result)
    annotated["fallback_warning"] = warning
    annotated["browser_engine"] = "chrome"
    annotated["browser_engine_fallback"] = {
        "from": "lightpanda",
        "to": "chrome",
        "reason": reason,
    }
    data = annotated.get("data")
    if isinstance(data, dict):
        data = dict(data)
        data.setdefault("fallback_warning", warning)
        data.setdefault("browser_engine", "chrome")
        data.setdefault(
            "browser_engine_fallback",
            {"from": "lightpanda", "to": "chrome", "reason": reason},
        )
        annotated["data"] = data
    return annotated


def _copy_fallback_warning(
    target: Dict[str, Any], result: Dict[str, Any]
) -> Dict[str, Any]:
    """Copy browser fallback metadata from an internal result into a tool response."""
    if result.get("fallback_warning"):
        target["fallback_warning"] = result["fallback_warning"]
        target["browser_engine"] = result.get("browser_engine")
        target["browser_engine_fallback"] = result.get("browser_engine_fallback")
    return target


async def _run_chrome_fallback_command(
    task_id: str,
    command: str,
    args: List[str],
    timeout: int,
) -> Dict[str, Any]:
    """Run a browser command in a temporary Chrome session at the current URL.

    agent-browser locks the engine when a named daemon starts. Passing
    ``--engine chrome`` to the same Lightpanda ``--session`` cannot change that
    running daemon. This helper always uses a fresh temporary Chrome session,
    navigates it to the current Lightpanda URL, runs ``command``, then tears it
    down.
    """
    import uuid

    # 1. Grab the current URL from the Lightpanda session. Use
    # ``_engine_override=\"auto\"`` so this helper does not recursively trigger
    # Lightpanda→Chrome fallback if the eval call itself fails.
    url_result = await _run_browser_command(
        task_id, "eval", ["window.location.href"], timeout=10, _engine_override="auto"
    )
    current_url = None
    if url_result.get("success"):
        current_url = (
            url_result.get("data", {}).get("result", "").strip().strip('"').strip("'")
        )
    if not current_url:
        logger.warning(
            "Chrome fallback: could not determine current URL from LP session"
        )
        return {
            "success": False,
            "error": "Chrome fallback failed: could not determine current URL",
        }

    # 2. Create a temporary Chrome session (bypasses _get_session_info's cache).
    tmp_session = f"h_cfb_{uuid.uuid4().hex[:8]}"
    try:
        browser_cmd = await _find_agent_browser()
    except FileNotFoundError as e:
        return {"success": False, "error": str(e)}

    if not await _chromium_installed():
        if await _running_in_docker():
            hint = (
                "Chrome fallback requires Chromium, but it is missing. "
                "You're running in Docker — pull the latest image: "
                "docker pull ghcr.io/nousresearch/hermes-agent:latest"
            )
        else:
            hint = (
                "Chrome fallback requires Chromium, but it is missing. Install it with: "
                "npx agent-browser install --with-deps "
                "(or: npx playwright install --with-deps chromium)"
            )
        return {"success": False, "error": hint}

    # On Windows npx is npx.cmd — use shutil.which so CreateProcessW can
    # execute the batch shim.  shutil.which honours PATHEXT on Windows and
    # returns the plain executable on POSIX.  If npx isn't on PATH (Termux,
    # bare container), fall back to the bare name and let Popen raise with
    # a readable "FileNotFoundError: 'npx'" rather than WinError 193.
    if browser_cmd == "npx agent-browser":
        _npx_bin = await aiofiles.os.wrap(shutil.which)("npx") or "npx"
        cmd_prefix = [_npx_bin, "agent-browser"]
    else:
        cmd_prefix = [browser_cmd]
    base_args = cmd_prefix + ["--engine", "chrome", "--session", tmp_session, "--json"]

    task_socket_dir = os.path.join(
        _socket_safe_tmpdir(), f"agent-browser-{tmp_session}"
    )
    await aiofiles.os.makedirs(task_socket_dir, mode=0o700, exist_ok=True)
    browser_env = _build_browser_env()
    browser_env["AGENT_BROWSER_SOCKET_DIR"] = task_socket_dir
    browser_env["PATH"] = await _merge_browser_path(browser_env.get("PATH", ""))

    if "AGENT_BROWSER_IDLE_TIMEOUT_MS" not in browser_env:
        browser_env["AGENT_BROWSER_IDLE_TIMEOUT_MS"] = str(
            BROWSER_SESSION_INACTIVITY_TIMEOUT * 1000
        )

    async def _run_tmp(cmd: str, cmd_args: List[str]) -> Dict[str, Any]:
        full = base_args + [cmd] + cmd_args
        # Use temp-file stdout/stderr pattern (same as _run_browser_command)
        # to avoid pipe hang from agent-browser daemon inheriting fds.
        stdout_path = os.path.join(task_socket_dir, f"_stdout_{cmd}")
        stderr_path = os.path.join(task_socket_dir, f"_stderr_{cmd}")
        open_file = aiofiles.os.wrap(os.open)
        close_file = aiofiles.os.wrap(os.close)
        stdout_fd = await open_file(
            stdout_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
        )
        stderr_fd = await open_file(
            stderr_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
        )
        try:
            # On Windows, launch the child in a new process group so parent
            # console Ctrl+C doesn't kill it with STATUS_CONTROL_C_EXIT
            # (0xC000013A = rc 3221225786), AND insulate its stdio + handle
            # inheritance from the parent.
            #
            # Additional Windows hardening beyond CREATE_NEW_PROCESS_GROUP:
            # * STARTF_USESTDHANDLES + explicit handles → CreateProcess hands
            #   the child ONLY our three chosen handles (DEVNULL stdin +
            #   temp-file stdout/stderr). Without this, some parents leak
            #   console handles that break downstream grandchild spawns — the
            #   agent-browser Rust binary spawns a detached daemon grandchild,
            #   and that grandchild's CreateProcess dies silently
            #   ("Daemon process exited during startup with no error output")
            #   when inherited parent handles are in a weird state. Observed
            #   in the Hermes CLI where sys.stdout and sys.stderr both report
            #   fileno=1 (stderr dup'd onto stdout at the OS level).
            # * close_fds=True → block inheritance of every other handle.
            #   (Default on POSIX; must be explicit on Windows for stdio.)
            _popen_extra: dict = {}
            if os.name == "nt":
                # CREATE_NO_WINDOW → don't attach a console (cmd.exe would
                # otherwise briefly allocate one for the .cmd shim).
                # Do NOT add CREATE_NEW_PROCESS_GROUP: on Python 3.11 Windows
                # it interacts with asyncio's ProactorEventLoop such that the
                # subprocess creation cancels the running loop task, which
                # surfaces as KeyboardInterrupt in app.run() and tears down
                # the CLI mid-turn. The agent thread's subprocess spawn
                # unwound MainThread's prompt_toolkit loop that way — see
                # diag log: "asyncio.CancelledError → KeyboardInterrupt".
                _popen_extra["creationflags"] = windows_hide_flags()
                _popen_extra["close_fds"] = True
                _si = subprocess.STARTUPINFO()
                _si.dwFlags |= subprocess.STARTF_USESTDHANDLES
                _popen_extra["startupinfo"] = _si
            proc = await asyncio.create_subprocess_exec(
                *full,
                stdout=stdout_fd,
                stderr=stderr_fd,
                stdin=subprocess.DEVNULL,
                env=browser_env,
                **_popen_extra,
            )
        finally:
            await close_file(stdout_fd)
            await close_file(stderr_fd)
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return {"success": False, "error": f"Chrome fallback '{cmd}' timed out"}
        try:
            async with aiofiles.open(stdout_path, "r", encoding="utf-8") as f:
                stdout = (await f.read()).strip()
            if stdout:
                return json.loads(stdout.split("\n")[-1])
        except Exception as exc:
            logger.debug("Chrome fallback tmp cmd '%s' error: %s", cmd, exc)
        finally:
            for pth in (stdout_path, stderr_path):
                try:
                    await aiofiles.os.remove(pth)
                except OSError:
                    pass
        return {"success": False, "error": f"Chrome fallback '{cmd}' failed"}

    try:
        # 3. Navigate Chrome to the same URL.
        nav = await _run_tmp("open", [current_url])
        if not nav.get("success"):
            logger.warning("Chrome fallback: navigate failed: %s", nav.get("error"))
            return {
                "success": False,
                "error": f"Chrome fallback navigate failed: {nav.get('error')}",
            }

        # 4. Run the requested command in Chrome.
        return await _run_tmp(command, args)

    finally:
        # 5. Tear down the temporary Chrome session.
        try:
            await _run_tmp("close", [])
        except Exception:
            pass
        # Clean up socket directory
        await _remove_tree(task_socket_dir)


async def _chrome_fallback_screenshot(
    task_id: str,
    args: List[str],
    timeout: int,
) -> Dict[str, Any]:
    """Take a screenshot using a temporary Chrome session."""
    return await _run_chrome_fallback_command(task_id, "screenshot", args, timeout)


async def _auto_local_for_private_urls() -> bool:
    """Return whether a cloud-configured install should auto-spawn a local
    Chromium for LAN/localhost URLs.

    Reads ``browser.auto_local_for_private_urls`` once (default ``True``) and
    caches it for the process lifetime.  When enabled, ``browser_navigate``
    routes URLs whose host resolves to a private/loopback/LAN address to a
    local headless Chromium sidecar even when a cloud provider (Browserbase
    / Browser-Use / Firecrawl) is configured globally.  Public URLs continue
    to use the cloud provider in the same conversation.
    """
    global _auto_local_for_private_urls_resolved, _cached_auto_local_for_private_urls
    if _auto_local_for_private_urls_resolved:
        return _cached_auto_local_for_private_urls

    _auto_local_for_private_urls_resolved = True
    try:
        cfg = await load_config_readonly()
        browser_cfg = cfg.get("browser", {})
        if (
            isinstance(browser_cfg, dict)
            and "auto_local_for_private_urls" in browser_cfg
        ):
            _cached_auto_local_for_private_urls = bool(
                browser_cfg.get("auto_local_for_private_urls")
            )
    except Exception as e:
        logger.debug("Could not read auto_local_for_private_urls from config: %s", e)
    return _cached_auto_local_for_private_urls


async def _url_is_private(url: str) -> bool:
    """Return True when the URL's host resolves to a private/LAN/loopback address.

    Reuses ``tools.url_safety.is_safe_url`` as the oracle — if the SSRF check
    would reject the URL, we treat it as "private" for routing purposes.  DNS
    resolution failures are treated as NOT private (fall through to whatever
    backend is configured, which will surface the DNS error naturally).
    """
    try:
        # is_safe_url returns False for private/loopback/link-local/CGNAT AND
        # for DNS failures.  We only want the private-network case here, so
        # we parse + check the host shape as a DNS-failure sieve first.
        from urllib.parse import urlparse
        import ipaddress
        import socket

        parsed = urlparse(url)
        hostname = (parsed.hostname or "").strip().lower().rstrip(".")
        if not hostname:
            return False
        # Literal IP → check directly
        try:
            ip = ipaddress.ip_address(hostname)
            return (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                # 172.16.0.0/12: only covered by ip.is_private on Python
                # ≥3.11 (bpo-40791).  Explicit check keeps 3.10 runtimes
                # routing these to the local sidecar correctly.
                or ip in ipaddress.ip_network("172.16.0.0/12")
                or ip in ipaddress.ip_network("100.64.0.0/10")
            )
        except ValueError:
            pass
        # Hostname — must resolve to confirm it's private (bare "localhost"
        # resolves to 127.0.0.1 via /etc/hosts).  Short-circuit on obvious
        # names to avoid a DNS hop.
        if hostname in {
            "localhost",
        } or hostname.endswith(".localhost"):
            return True
        if (
            hostname.endswith(".local")
            or hostname.endswith(".lan")
            or hostname.endswith(".internal")
        ):
            return True
        try:
            addr_info = await asyncio.get_running_loop().getaddrinfo(
                hostname, None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
            )
        except socket.gaierror:
            return False  # DNS fail → not private, let the normal path fail
        for _, _, _, _, sockaddr in addr_info:
            try:
                ip = ipaddress.ip_address(sockaddr[0])
            except ValueError:
                continue
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip in ipaddress.ip_network("100.64.0.0/10")
            ):
                return True
        return False
    except Exception as exc:
        logger.debug("URL-privacy check failed for %s: %s", url, exc)
        return False


async def _navigation_session_key(task_id: str, url: str) -> str:
    """Pick the session key that should handle ``url`` for ``task_id``.

    Returns the bare task_id unless ALL of these are true:
      1. A cloud provider is configured (``_get_cloud_provider()`` is not None).
      2. Auto-local routing is enabled (``browser.auto_local_for_private_urls``,
         default True).
      3. The URL resolves to a private/LAN/loopback address.
      4. A CDP override is not active (that path owns the whole session).
      5. Camofox mode is not active (Camofox is already local-only).

    When all are true, returns ``f"{task_id}::local"`` so the hybrid-routing
    path spawns a local Chromium sidecar while the cloud session (if any)
    continues to serve public URLs.
    """
    if task_id is None:
        task_id = "default"
    if await _get_cdp_override_raw():
        return task_id
    if await _is_camofox_mode():
        return task_id
    if await _get_cloud_provider() is None:
        return task_id
    if not await _auto_local_for_private_urls():
        return task_id
    if not await _url_is_private(url):
        return task_id
    return f"{task_id}{_LOCAL_SUFFIX}"


def _is_local_sidecar_key(session_key: str) -> bool:
    """Return True when ``session_key`` is a hybrid-routing local sidecar."""
    return session_key.endswith(_LOCAL_SUFFIX)


def _bare_task_id_for_session_key(session_key: str) -> str:
    """Return the owning bare task id for an opaque browser session key."""
    if _is_local_sidecar_key(session_key):
        return session_key[: -len(_LOCAL_SUFFIX)]
    return session_key


def _session_info_owned_by_task(
    session_info: Dict[str, Any], task_id: str, session_key: str
) -> bool:
    """Return whether ``session_info`` still belongs to ``task_id``/``session_key``.

    Sessions created by current code carry explicit ownership metadata. Treat
    older in-memory entries without those fields as valid for hot-reload/test
    compatibility, but reject any explicit mismatch before a non-navigation
    tool can act on the wrong tab/session.
    """
    owner = session_info.get("owner_task_id")
    key = session_info.get("session_key")
    if owner is not None and owner != task_id:
        return False
    if key is not None and key != session_key:
        return False
    return True


def _last_session_key(task_id: str) -> str:
    """Return the live session key to use for a non-nav browser tool call.

    ``browser_navigate`` records which concrete session key served a task's
    most recent successful navigation. Non-navigation tools must reuse that key
    so click/fill/snapshot land in the same browser. If the recorded owner was
    later cleaned up or ownership metadata no longer matches, fail closed by
    dropping the stale binding instead of silently recreating or mutating the
    wrong browser.
    """
    if task_id is None:
        task_id = "default"
    recorded_key = _last_active_session_key.get(task_id)
    if not recorded_key:
        return task_id
    session_info = _active_sessions.get(recorded_key)
    if session_info and _session_info_owned_by_task(
        session_info, task_id, recorded_key
    ):
        return recorded_key
    _last_active_session_key.pop(task_id, None)
    logger.debug(
        "browser session ownership: dropping stale/mismatched last-active binding %s -> %s",
        task_id,
        recorded_key,
    )
    return task_id


async def _allow_private_urls() -> bool:
    """Return whether the browser is allowed to navigate to private/internal addresses.

    Reads ``config["browser"]["allow_private_urls"]``. Single-profile calls
    cache the result for the process lifetime; multiplexed profile turns resolve
    their context-local config on each call. Defaults to ``False`` (SSRF
    protection active).
    """
    global _cached_allow_private_urls, _allow_private_urls_resolved

    # The profile multiplexer scopes config with a ContextVar while sharing
    # this module. Never reuse another profile's private-network opt-out.
    if get_hermes_home_override() is not None:
        return await _resolve_allow_private_urls()

    if _allow_private_urls_resolved and _cached_allow_private_urls is not None:
        return _cached_allow_private_urls

    _allow_private_urls_resolved = True
    _cached_allow_private_urls = await _resolve_allow_private_urls()
    return _cached_allow_private_urls


async def _resolve_allow_private_urls() -> bool:
    """Read the browser private-URL toggle from the active config scope."""
    try:
        cfg = await load_config_readonly()
        browser_cfg = cfg.get("browser", {})
        if isinstance(browser_cfg, dict):
            return is_truthy_value(browser_cfg.get("allow_private_urls"), default=False)
    except Exception as e:
        logger.debug("Could not read allow_private_urls from config: %s", e)
    return False


def _socket_safe_tmpdir() -> str:
    """Return a short temp directory path suitable for Unix domain sockets.

    macOS sets ``TMPDIR`` to ``/var/folders/xx/.../T/`` (~51 chars).  When we
    append ``agent-browser-hermes_…`` the resulting socket path exceeds the
    104-byte macOS limit for ``AF_UNIX`` addresses, causing agent-browser to
    fail with "Failed to create socket directory" or silent screenshot failures.

    Linux ``tempfile.gettempdir()`` already returns ``/tmp``, so this is a
    no-op there.  On macOS we bypass ``TMPDIR`` and use ``/tmp`` directly
    (symlink to ``/private/tmp``, sticky-bit protected, always available).
    """
    if sys.platform == "darwin":
        return "/tmp"
    return tempfile.gettempdir()


# Track active sessions per "session key".
#
# A "session key" is either the bare task_id (cloud/default path) OR a composite
# like f"{task_id}::local" when the hybrid-routing feature spawns a local sidecar
# browser for a LAN/localhost URL while a cloud provider is configured globally.
# Both forms flow through the same _active_sessions / _run_browser_command /
# cleanup_browser code paths — the key is opaque to those internals.
#
# Stores: session_name (always), bb_session_id + cdp_url (cloud mode only)
_active_sessions: Dict[str, Dict[str, Any]] = {}  # session_key -> {session_name, ...}
_recording_sessions: set = set()  # session_keys with active recordings

# Tracks the most recent session_key used per task_id. Set by browser_navigate()
# after it chooses a backend for a URL; read by every non-nav browser tool
# (snapshot/click/fill/eval/...) so they target the session that served the last
# navigation.  Without this, a task that navigated to localhost on the local
# sidecar would fall back to the cloud session on its next snapshot call.
_last_active_session_key: Dict[str, str] = {}  # task_id -> session_key
_LOCAL_SUFFIX = "::local"

# Flag to track if cleanup has been done
_cleanup_done = False

# =============================================================================
# Inactivity Timeout Configuration
# =============================================================================

# Session inactivity timeout (seconds) - cleanup if no activity for this long.
# config.yaml is authoritative; BROWSER_INACTIVITY_TIMEOUT remains a legacy
# fallback so old deployments keep working if they have not migrated yet.
DEFAULT_SESSION_INACTIVITY_TIMEOUT = int(
    cfg_get(DEFAULT_CONFIG, "browser", "inactivity_timeout", default=120) or 120
)


async def _get_session_inactivity_timeout() -> int:
    result = env_int("BROWSER_INACTIVITY_TIMEOUT", DEFAULT_SESSION_INACTIVITY_TIMEOUT)
    try:
        cfg = await load_config_readonly()
        val = cfg_get(cfg, "browser", "inactivity_timeout")
        if val is not None:
            result = max(int(val), 30)  # Floor at 30s to avoid instant reaping
    except Exception as e:
        logger.debug("Could not read inactivity_timeout from config: %s", e)
    return result


BROWSER_SESSION_INACTIVITY_TIMEOUT = DEFAULT_SESSION_INACTIVITY_TIMEOUT

# Track last activity time per session
_session_last_activity: Dict[str, float] = {}

# Background cleanup task state
_cleanup_task: Optional[asyncio.Task[None]] = None
_cleanup_running = False
# Protects browser session maps across coroutine suspension points. Recording
# transitions use a dedicated lock because starting/stopping invokes the
# browser command path, which itself may resolve session state.
_cleanup_lock = asyncio.Lock()
_recording_lock = asyncio.Lock()


def _session_expiry_timestamp(session_info: Dict[str, Any]) -> Optional[float]:
    """Return a provider-authoritative session expiry as epoch seconds.

    Cloud providers may omit ``expires_at``. Unknown or malformed values are
    therefore treated as having no known expiry, preserving the existing
    lifecycle for local browsers and providers without an expiry contract.
    """
    value = session_info.get("expires_at")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None

    normalized = value.strip()
    if normalized.endswith(("Z", "z")):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        logger.warning("Ignoring invalid cloud browser session expiry timestamp")
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _session_has_expired(
    session_info: Dict[str, Any], *, now: Optional[float] = None
) -> bool:
    """Return whether a cached browser session crossed its provider deadline."""
    expires_at = _session_expiry_timestamp(session_info)
    if expires_at is None:
        return False
    return (time.time() if now is None else now) >= expires_at


async def _emergency_cleanup_all_sessions() -> None:
    """
    Emergency cleanup of all active browser sessions.
    Called on process exit or interrupt to prevent orphaned sessions.

    Also runs the orphan reaper to clean up daemons left behind by previously
    crashed hermes processes — this way every clean hermes exit sweeps
    accumulated orphans, not just ones that actively used the browser tool.
    """
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True

    # Clean up this process's own sessions first, so their owner_pid files
    # are removed before the reaper scans.
    if _active_sessions:
        logger.info(
            "Emergency cleanup: closing %s active session(s)...", len(_active_sessions)
        )
        try:
            await cleanup_all_browsers()
        except Exception as e:
            logger.error("Emergency cleanup error: %s", e)
        finally:
            async with _cleanup_lock:
                _active_sessions.clear()
                _session_last_activity.clear()
                _recording_sessions.clear()

    # Sweep orphans from other crashed hermes processes.  Safe even if we
    # never used the browser — uses owner_pid liveness to avoid reaping
    # daemons owned by other live hermes processes.
    try:
        await _reap_orphaned_browser_sessions()
    except Exception as e:
        logger.debug("Orphan reap on exit failed: %s", e)


# =============================================================================
# Inactivity Cleanup Functions
# =============================================================================


async def _cleanup_inactive_browser_sessions() -> None:
    """
    Clean up browser sessions that have been inactive for longer than the timeout.

    This function is called periodically by the background cleanup thread to
    automatically close sessions that haven't been used recently, preventing
    orphaned sessions (local or Browserbase) from accumulating.
    """
    current_time = time.time()
    sessions_to_cleanup = []

    async with _cleanup_lock:
        activity_snapshot = list(_session_last_activity.items())
    for task_id, last_time in activity_snapshot:
        if current_time - last_time > BROWSER_SESSION_INACTIVITY_TIMEOUT:
            sessions_to_cleanup.append(task_id)

    for task_id in sessions_to_cleanup:
        try:
            async with _cleanup_lock:
                last_activity = _session_last_activity.get(task_id, current_time)
            elapsed = int(current_time - last_activity)
            logger.info(
                "Cleaning up inactive session for task: %s (inactive for %ss)",
                task_id,
                elapsed,
            )
            await cleanup_browser(task_id)
            async with _cleanup_lock:
                _session_last_activity.pop(task_id, None)
        except Exception as e:
            logger.warning("Error cleaning up inactive session %s: %s", task_id, e)


async def _write_owner_pid(socket_dir: str, session_name: str) -> None:
    """Record the current hermes PID as the owner of a browser socket dir.

    Written atomically to ``<socket_dir>/<session_name>.owner_pid`` so the
    orphan reaper can distinguish daemons owned by a live hermes process
    (don't reap) from daemons whose owner crashed (reap).  Best-effort —
    an OSError here just falls back to the legacy ``tracked_names``
    heuristic in the reaper.
    """
    try:
        path = os.path.join(socket_dir, f"{session_name}.owner_pid")
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(str(os.getpid()))
    except OSError as exc:
        logger.debug("Could not write owner_pid file for %s: %s", session_name, exc)


async def _remove_tree(root: str) -> None:
    """Best-effort recursive removal without blocking the event loop."""
    try:
        names = await aiofiles.os.listdir(root)
    except OSError:
        return
    for name in names:
        path = os.path.join(root, name)
        try:
            is_directory = stat.S_ISDIR(
                (await aiofiles.os.stat(path, follow_symlinks=False)).st_mode
            )
        except OSError:
            continue
        if is_directory:
            await _remove_tree(path)
        else:
            try:
                await aiofiles.os.remove(path)
            except OSError:
                pass
    try:
        await aiofiles.os.rmdir(root)
    except OSError:
        pass


async def _terminate_host_pid(pid: int) -> None:
    """Terminate a verified process tree without a blocking wait call."""
    import psutil

    try:
        parent = psutil.Process(pid)
        processes = [*parent.children(recursive=True), parent]
    except psutil.NoSuchProcess:
        return
    for process in reversed(processes):
        try:
            process.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    deadline = asyncio.get_running_loop().time() + 3.0
    remaining = processes
    while remaining and asyncio.get_running_loop().time() < deadline:
        remaining = [process for process in remaining if process.is_running()]
        if remaining:
            await asyncio.sleep(0.05)
    for process in remaining:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def _verify_reapable_browser_daemon(
    daemon_pid: int, socket_dir: str, session_name: str
) -> bool:
    """Confirm a live PID is genuinely *this* session's agent-browser daemon.

    The orphan reaper scans world-writable, predictably-named temp paths
    (``/tmp/agent-browser-h_*`` etc.) and reads a daemon PID from a ``.pid``
    file we do not write ourselves — the agent-browser daemon writes it.  A
    same-user actor can therefore plant a fake socket dir whose ``.pid`` points
    at an arbitrary victim process, or a recycled PID can land on an unrelated
    process after the real daemon exits.  Either way, terminating that PID
    (a *tree* kill via ``_terminate_host_pid``) is an arbitrary-process DoS.

    Before reaping we require, via ``psutil`` (a hard dependency, cross-platform
    for same-user processes — the only processes the reaper can signal):

      1. **Identity** — the process looks like agent-browser: ``agent-browser``
         appears in its name or command line.
      2. **Binding** — the process is bound to *this* session's socket dir: the
         socket dir path (or its basename) appears in the command line, or in
         ``AGENT_BROWSER_SOCKET_DIR`` in the process environment.

    Requirement (2) is the real spoof defense: a planted process pointing at a
    victim PID will not have the victim's cmdline/environ referencing our
    socket dir.  An attacker would need a process that genuinely embeds this
    exact session path — i.e. a real daemon they already own and could signal
    directly.  Fail-closed: any ambiguity (unreadable cmdline, no match) means
    we refuse to reap and leave the process and its socket dir alone.

    Returns ``True`` only when both checks pass.
    """
    try:
        import psutil
    except ImportError:  # psutil is a hard dep; defensive only
        logger.warning(
            "Refusing to reap browser daemon PID %d (session %s): "
            "psutil unavailable for identity verification",
            daemon_pid,
            session_name,
        )
        return False

    try:
        proc = psutil.Process(daemon_pid)
        name = (proc.name() or "").lower()
        cmdline = " ".join(proc.cmdline() or []).lower()
    except psutil.NoSuchProcess:
        # Vanished between the liveness check and now — nothing to reap.
        return False
    except (psutil.AccessDenied, OSError) as exc:
        logger.warning(
            "Refusing to reap browser daemon PID %d (session %s): "
            "could not read process identity (%s)",
            daemon_pid,
            session_name,
            exc,
        )
        return False

    looks_like_browser = "agent-browser" in name or "agent-browser" in cmdline
    if not looks_like_browser:
        logger.warning(
            "Refusing to reap PID %d (session %s): not an agent-browser "
            "process (name=%r)",
            daemon_pid,
            session_name,
            name,
        )
        return False

    # Binding check: the live process must reference *this* socket dir.
    socket_dir_l = socket_dir.lower()
    socket_base_l = os.path.basename(socket_dir).lower()
    bound = socket_dir_l in cmdline or (socket_base_l and socket_base_l in cmdline)
    if not bound:
        try:
            env_dir = (proc.environ() or {}).get("AGENT_BROWSER_SOCKET_DIR", "")
            bound = bool(env_dir) and os.path.normpath(env_dir) == os.path.normpath(
                socket_dir
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            # environ() can be denied even same-user on some platforms.
            # cmdline already failed to bind — fail closed.
            bound = False

    if not bound:
        logger.warning(
            "Refusing to reap agent-browser PID %d: not bound to session "
            "socket dir %s (possible recycled PID or planted pid file)",
            daemon_pid,
            socket_dir,
        )
        return False

    return True


async def _reap_orphaned_browser_sessions() -> None:
    """Scan for orphaned agent-browser daemon processes from previous runs.

    When the Python process that created a browser session exits uncleanly
    (SIGKILL, crash, gateway restart), the in-memory ``_active_sessions``
    tracking is lost but the node + Chromium processes keep running.

    This function scans the tmp directory for ``agent-browser-*`` socket dirs
    left behind by previous runs, reads the daemon PID files, and kills any
    daemons whose owning hermes process is no longer alive.

    Ownership detection priority:
      1. ``<session>.owner_pid`` file (written by current code) — if the
         referenced hermes PID is alive, leave the daemon alone regardless
         of whether it's in *this* process's ``_active_sessions``.  This is
         cross-process safe: two concurrent hermes instances won't reap each
         other's daemons.
      2. Fallback for daemons that predate owner_pid: check
         ``_active_sessions`` in the current process.  If not tracked here,
         treat as orphan (legacy behavior).

    Safe to call from any context — atexit, cleanup thread, or on demand.
    """
    tmpdir = _socket_safe_tmpdir()
    try:
        entries = await aiofiles.os.listdir(tmpdir)
    except OSError:
        return
    socket_dirs = [
        os.path.join(tmpdir, entry)
        for entry in entries
        if entry.startswith((
            "agent-browser-h_",
            "agent-browser-cdp_",
            "agent-browser-hermes_",
        ))
    ]

    if not socket_dirs:
        return

    # Build set of session_names currently tracked by this process (fallback path)
    async with _cleanup_lock:
        tracked_names = {
            info.get("session_name")
            for info in _active_sessions.values()
            if info.get("session_name")
        }

    reaped = 0
    for socket_dir in socket_dirs:
        dir_name = os.path.basename(socket_dir)
        # dir_name is "agent-browser-{session_name}"
        session_name = dir_name.removeprefix("agent-browser-")
        if not session_name:
            continue

        # Ownership check: prefer owner_pid file (cross-process safe).
        owner_pid_file = os.path.join(socket_dir, f"{session_name}.owner_pid")
        owner_alive: Optional[bool] = None  # None = owner_pid missing/unreadable
        if await aiofiles.os.path.isfile(owner_pid_file):
            try:
                async with aiofiles.open(owner_pid_file, encoding="utf-8") as handle:
                    owner_pid = int((await handle.read()).strip())
                # ``os.kill(pid, 0)`` is NOT a no-op on Windows (bpo-14484).
                # Use the cross-platform existence check.
                from gateway.status import _pid_exists

                owner_alive = await _pid_exists(owner_pid)
            except (ValueError, OSError):
                owner_alive = None  # corrupt file — fall through

        if owner_alive is True:
            # Owner is alive — this session belongs to a live hermes process.
            continue

        if owner_alive is None:
            # No owner_pid file (legacy daemon).  Fall back to in-process
            # tracking: if this process knows about the session, leave alone.
            if session_name in tracked_names:
                continue

        # owner_alive is False (dead owner) OR legacy daemon not tracked here.
        pid_file = os.path.join(socket_dir, f"{session_name}.pid")
        if not await aiofiles.os.path.isfile(pid_file):
            # No daemon PID file — just a stale dir, remove it
            await _remove_tree(socket_dir)
            continue

        try:
            async with aiofiles.open(pid_file, encoding="utf-8") as handle:
                daemon_pid = int((await handle.read()).strip())
        except (ValueError, OSError):
            await _remove_tree(socket_dir)
            continue

        # Check if the daemon is still alive. ``os.kill(pid, 0)`` on Windows
        # is NOT a no-op — use the handle-based existence check.
        from gateway.status import _pid_exists

        if not await _pid_exists(daemon_pid):
            await _remove_tree(socket_dir)
            continue

        # The PID is live — but the .pid file lives in a world-writable,
        # predictably-named temp dir we don't write ourselves, and PIDs get
        # recycled after the real daemon exits.  Verify the process really is
        # *this* session's agent-browser daemon before tree-killing it; refuse
        # otherwise (don't touch the process, leave the socket dir for a later
        # sweep once the imposter PID is gone).  Fixes the arbitrary same-user
        # process DoS in issue #14073.
        if not _verify_reapable_browser_daemon(daemon_pid, socket_dir, session_name):
            continue

        # Daemon is alive and its owner is dead (or legacy + untracked).  Reap.
        # Use the process-tree termination helper so Chromium children
        # (renderer, GPU, etc.) are cleaned up, not just the daemon parent.
        try:
            await _terminate_host_pid(daemon_pid)
            logger.info(
                "Reaped orphaned browser daemon PID %d (session %s)",
                daemon_pid,
                session_name,
            )
            reaped += 1
        except (ProcessLookupError, PermissionError, OSError):
            pass

        # Clean up the socket directory
        await _remove_tree(socket_dir)

    if reaped:
        logger.info(
            "Reaped %d orphaned browser session(s) from previous run(s)", reaped
        )


async def _browser_cleanup_thread_worker() -> None:
    """
    Background thread that periodically cleans up inactive browser sessions.

    Runs every 30 seconds and checks for sessions that haven't been used
    within the BROWSER_SESSION_INACTIVITY_TIMEOUT period.
    On first run, also reaps orphaned sessions from previous process lifetimes.
    """
    # One-time orphan reap on startup
    try:
        await _reap_orphaned_browser_sessions()
    except Exception as e:
        logger.warning("Orphan reap error: %s", e)

    while _cleanup_running:
        try:
            await _cleanup_inactive_browser_sessions()
        except Exception as e:
            logger.warning("Cleanup thread error: %s", e)

        await asyncio.sleep(30)


async def _start_browser_cleanup_thread() -> None:
    """Start the background cleanup task if it is not already running."""
    global _cleanup_task, _cleanup_running, BROWSER_SESSION_INACTIVITY_TIMEOUT

    async with _cleanup_lock:
        if _cleanup_task is not None and not _cleanup_task.done():
            return
        BROWSER_SESSION_INACTIVITY_TIMEOUT = await _get_session_inactivity_timeout()
        _cleanup_running = True
        _cleanup_task = asyncio.create_task(
            _browser_cleanup_thread_worker(), name="browser-cleanup"
        )
        logger.info(
            "Started inactivity cleanup task (timeout: %ss)",
            BROWSER_SESSION_INACTIVITY_TIMEOUT,
        )


async def _stop_browser_cleanup_thread() -> None:
    """Stop the background cleanup task."""
    global _cleanup_task, _cleanup_running
    async with _cleanup_lock:
        _cleanup_running = False
        task = _cleanup_task
        _cleanup_task = None
    if task is not None and task is not asyncio.current_task():
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def _update_session_activity(task_id: str) -> None:
    """Update the last activity timestamp for a session."""
    # This function contains no await and therefore runs atomically with
    # respect to other tasks on the event loop.
    _session_last_activity[task_id] = time.time()


# ============================================================================
# Tool Schemas
# ============================================================================

BROWSER_TOOL_SCHEMAS = [
    {
        "name": "browser_navigate",
        "description": "Navigate to a URL in the browser. Initializes the session and loads the page. Must be called before other browser tools. For simple information retrieval, prefer web_search or web_extract (faster, cheaper). For plain-text endpoints — URLs ending in .md, .txt, .json, .yaml, .yml, .csv, .xml, raw.githubusercontent.com, or any documented API endpoint — prefer curl via the terminal tool or web_extract; the browser stack is overkill and much slower for these. Use browser tools when you need to interact with a page (click, fill forms, dynamic content). Returns a compact page snapshot with interactive elements and ref IDs — no need to call browser_snapshot separately after navigating.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to navigate to (e.g., 'https://example.com')",
                }
            },
            "required": ["url"],
        },
    },
    {
        "name": "browser_snapshot",
        "description": "Get a text-based snapshot of the current page's accessibility tree. Returns interactive elements with ref IDs (like @e1, @e2) for browser_click and browser_type. full=false (default): compact view with interactive elements. full=true: complete page content. Snapshots over 15000 chars are truncated or LLM-summarized; when that happens the complete snapshot is saved to a file and the output includes its path so you can page through the rest with read_file. Requires browser_navigate first. Note: browser_navigate already returns a compact snapshot — use this to refresh after interactions that change the page, or with full=true for complete content.",
        "parameters": {
            "type": "object",
            "properties": {
                "full": {
                    "type": "boolean",
                    "description": "If true, returns complete page content. If false (default), returns compact view with interactive elements only.",
                    "default": False,
                }
            },
            "required": [],
        },
    },
    {
        "name": "browser_click",
        "description": "Click on an element identified by its ref ID from the snapshot (e.g., '@e5'). The ref IDs are shown in square brackets in the snapshot output. Requires browser_navigate and browser_snapshot to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "The element reference from the snapshot (e.g., '@e5', '@e12')",
                }
            },
            "required": ["ref"],
        },
    },
    {
        "name": "browser_type",
        "description": "Type text into an input field identified by its ref ID. Clears the field first, then types the new text. Requires browser_navigate and browser_snapshot to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "The element reference from the snapshot (e.g., '@e3')",
                },
                "text": {
                    "type": "string",
                    "description": "The text to type into the field",
                },
            },
            "required": ["ref", "text"],
        },
    },
    {
        "name": "browser_scroll",
        "description": "Scroll the page in a direction. Use this to reveal more content that may be below or above the current viewport. Requires browser_navigate to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down"],
                    "description": "Direction to scroll",
                }
            },
            "required": ["direction"],
        },
    },
    {
        "name": "browser_back",
        "description": "Navigate back to the previous page in browser history. Requires browser_navigate to be called first.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "browser_press",
        "description": "Press a keyboard key. Useful for submitting forms (Enter), navigating (Tab), or keyboard shortcuts. Requires browser_navigate to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Key to press (e.g., 'Enter', 'Tab', 'Escape', 'ArrowDown')",
                }
            },
            "required": ["key"],
        },
    },
    {
        "name": "browser_get_images",
        "description": "Get a list of all images on the current page with their URLs and alt text. Useful for finding images to analyze with the vision tool. Requires browser_navigate to be called first.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "browser_vision",
        "description": "Take a screenshot of the current page so you can inspect it visually. Use this when you need to understand what the page looks like - especially for CAPTCHAs, visual verification challenges, complex layouts, or cases where the text snapshot misses important visual information. When your active model has native vision, the screenshot is attached to your context directly and you inspect it on the next turn; otherwise Hermes falls back to an auxiliary vision model and returns a text analysis. Includes a screenshot_path that you can share with the user by including MEDIA:<screenshot_path> in your response. Requires browser_navigate to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "What you want to know about the page visually. Be specific about what you're looking for.",
                },
                "annotate": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, overlay numbered [N] labels on interactive elements. Each [N] maps to ref @eN for subsequent browser commands. Useful for QA and spatial reasoning about page layout.",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "browser_console",
        "description": "Get browser console output and JavaScript errors from the current page. Returns console.log/warn/error/info messages and uncaught JS exceptions. Use this to detect silent JavaScript errors, failed API calls, and application warnings. Requires browser_navigate to be called first. When 'expression' is provided, evaluates JavaScript in the page context and returns the result — use this for DOM inspection, reading page state, or extracting data programmatically.",
        "parameters": {
            "type": "object",
            "properties": {
                "clear": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, clear the message buffers after reading",
                },
                "expression": {
                    "type": "string",
                    "description": "JavaScript expression to evaluate in the page context. Runs in the browser like DevTools console — full access to DOM, window, document. Return values are serialized to JSON. Example: 'document.title' or 'document.querySelectorAll(\"a\").length'",
                },
            },
            "required": [],
        },
    },
]


# ============================================================================
# Utility Functions
# ============================================================================


def _create_local_session(task_id: str) -> Dict[str, Any]:
    import uuid

    session_name = f"h_{uuid.uuid4().hex[:10]}"
    logger.info("Created local browser session %s for task %s", session_name, task_id)
    return {
        "session_name": session_name,
        "bb_session_id": None,
        "cdp_url": None,
        "features": {"local": True},
    }


def _create_cdp_session(task_id: str, cdp_url: str) -> Dict[str, Any]:
    """Create a session that connects to a user-supplied CDP endpoint."""
    import uuid

    session_name = f"cdp_{uuid.uuid4().hex[:10]}"
    logger.info(
        "Created CDP browser session %s → %s for task %s",
        session_name,
        _sanitize_url_for_logs(cdp_url),
        task_id,
    )
    return {
        "session_name": session_name,
        "bb_session_id": None,
        "cdp_url": cdp_url,
        "features": {"cdp_override": True},
    }


async def _get_session_info(task_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get or create session info for the given session key.

    In cloud mode, creates a Browserbase session with proxies enabled.
    In local mode, generates a session name for agent-browser --session.
    Also starts the inactivity cleanup task and updates activity tracking.

    Args:
        task_id: Session key.  Normally the task_id as-is, but may carry the
            ``::local`` suffix for the hybrid-routing local sidecar — in that
            case the cloud provider is skipped even when one is configured,
            and a local Chromium session is created instead.

    Returns:
        Dict with session_name (always), bb_session_id + cdp_url (cloud only)
    """
    if task_id is None:
        task_id = "default"

    # Start the cleanup task if not running (handles inactivity timeouts)
    await _start_browser_cleanup_thread()

    # Update activity timestamp for this session
    _update_session_activity(task_id)

    async with _cleanup_lock:
        existing_session = _active_sessions.get(task_id)

    if existing_session is not None:
        if not _session_has_expired(existing_session):
            return existing_session

        logger.info(
            "Replacing expired cloud browser session for task %s",
            task_id,
        )
        await _cleanup_single_browser_session(task_id)
        # Cleanup removes the activity entry. The replacement session must be
        # tracked by the inactivity reaper just like an initial session.
        _update_session_activity(task_id)

        # Guard against a concurrent replacement: another task may have
        # already cleaned up the expired session and created a fresh one
        # while we were waiting.  If so, return the live replacement instead
        # of falling through to create yet another session.
        async with _cleanup_lock:
            replacement = _active_sessions.get(task_id)
        if replacement is not None and replacement is not existing_session:
            return replacement

    # Hybrid routing: session keys ending with ``::local`` force a local
    # Chromium regardless of the globally-configured cloud provider.  Public
    # URLs in the same conversation continue to use the cloud session under
    # the bare task_id key.
    force_local = _is_local_sidecar_key(task_id)

    # Create session outside the lock (network call in cloud mode)
    cdp_override = await _get_cdp_override()
    if cdp_override and not force_local:
        session_info: Dict[str, Any] = _create_cdp_session(task_id, cdp_override)
    elif force_local:
        session_info = _create_local_session(task_id)
    else:
        provider = await _get_cloud_provider()
        if provider is None:
            session_info = _create_local_session(task_id)
        else:
            try:
                session_info = await provider.create_session(task_id)
                # Validate cloud provider returned a usable session
                if not session_info or not isinstance(session_info, dict):
                    raise ValueError(
                        f"Cloud provider returned invalid session: {session_info!r}"
                    )
                if session_info.get("cdp_url"):
                    # Some cloud providers (including Browser-Use v3) return an HTTP
                    # CDP discovery URL instead of a raw websocket endpoint.
                    session_info = dict(session_info)
                    session_info["cdp_url"] = await _resolve_cdp_override(
                        str(session_info["cdp_url"])
                    )
            except Exception as e:
                provider_name = type(provider).__name__
                logger.warning(
                    "Cloud provider %s failed (%s); attempting fallback to local "
                    "Chromium for task %s",
                    provider_name,
                    e,
                    task_id,
                    exc_info=True,
                )
                try:
                    session_info = _create_local_session(task_id)
                except Exception as local_error:
                    raise RuntimeError(
                        f"Cloud provider {provider_name} failed ({e}) and local "
                        f"fallback also failed ({local_error})"
                    ) from e
                # Mark session as degraded for observability
                if isinstance(session_info, dict):
                    session_info = dict(session_info)
                    session_info["fallback_from_cloud"] = True
                    session_info["fallback_reason"] = str(e)
                    session_info["fallback_provider"] = provider_name

    # Double-check: another task may have created a session while we were doing
    # the network call. Use the existing one to preserve the upstream contract.
    async with _cleanup_lock:
        if task_id in _active_sessions:
            return _active_sessions[task_id]
        session_info = dict(session_info)
        session_info.setdefault("session_key", task_id)
        session_info.setdefault(
            "owner_task_id", _bare_task_id_for_session_key(task_id)
        )
        _active_sessions[task_id] = session_info

    # Lazy-start the CDP supervisor now that the session exists (if the
    # backend surfaces a CDP URL via override or session_info["cdp_url"]).
    # Idempotent; swallows errors. See _ensure_cdp_supervisor for details.
    # Skip for local sidecars — they have no CDP URL.
    if not force_local:
        await _ensure_cdp_supervisor(task_id)

    return session_info


async def _agent_browser_candidate_present(path: str | None) -> bool:
    if not path:
        return False
    if " " in path and path.split()[0].endswith("npx"):
        return True
    return await aiofiles.os.path.exists(path) and (
        os.name == "nt" or await aiofiles.os.access(path, os.X_OK)
    )


async def _find_agent_browser(*, validate: bool = True) -> str:
    """
    Find the agent-browser CLI executable.

    Checks in order: current PATH, Homebrew/common bin dirs, Hermes-managed
    node, local node_modules/.bin/, npx fallback.

    Returns:
        Path to agent-browser executable

    Raises:
        FileNotFoundError: If agent-browser is not installed
    """
    global _cached_agent_browser, _agent_browser_resolved
    if _agent_browser_resolved:
        if _cached_agent_browser is None:
            raise FileNotFoundError(
                "agent-browser CLI not found (cached). Install it with: "
                f"{_browser_install_hint()}\n"
                "Or run 'npm install' in the repo root to install locally.\n"
                "Or ensure npx is available in your PATH."
            )
        return _cached_agent_browser

    # Note: _agent_browser_resolved is set at each return site below
    # (not before the search) to prevent a race where a concurrent thread
    # sees resolved=True but _cached_agent_browser is still None.
    #
    # Every candidate below is validated with ``agent_browser_runnable`` before
    # it is cached. A bare ``shutil.which`` hit is NOT trusted: agent-browser's
    # npm postinstall re-points a global install symlink at our local
    # node_modules binary, which disappears on the next ``hermes update`` and
    # leaves a dangling link that ``which`` still reports but exec fails on with
    # exit 127 (issue #48521). Validating lets a dead candidate fall through to
    # the next working resolution (extended PATH → local .bin → npx) instead of
    # caching the broken one and silently killing every browser tool.

    which = aiofiles.os.wrap(shutil.which)

    # Check if it's in PATH (global install)
    which_result = await which("agent-browser")
    if which_result and (
        await agent_browser_runnable(which_result)
        if validate
        else await _agent_browser_candidate_present(which_result)
    ):
        if not validate:
            return which_result
        _cached_agent_browser = which_result
        _agent_browser_resolved = True
        return which_result

    # Build an extended search PATH including Hermes-managed Node, macOS
    # versioned Homebrew installs, and fallback system dirs like Termux.
    extended_path = await _merge_browser_path("")
    if extended_path:
        which_result = await which("agent-browser", path=extended_path)
        if which_result and (
            await agent_browser_runnable(which_result)
            if validate
            else await _agent_browser_candidate_present(which_result)
        ):
            if not validate:
                return which_result
            _cached_agent_browser = which_result
            _agent_browser_resolved = True
            return which_result

    # Check local node_modules/.bin/ (npm install in repo root).
    # On Windows, npm drops three shims in .bin: an extensionless POSIX shell
    # script (for Git Bash / WSL), `agent-browser.cmd` (for cmd/PowerShell),
    # and `agent-browser.ps1` (for PowerShell). CreateProcess (used by Python's
    # subprocess on Windows) cannot execute the extensionless shim — it raises
    # WinError 193 "%1 is not a valid Win32 application". We must resolve to the
    # `.cmd` shim instead. `shutil.which` consults PATHEXT, so we delegate to it
    # with an explicit path so POSIX hosts still pick the extensionless shim.
    repo_root = Path(__file__).parent.parent
    local_bin_dir = repo_root / "node_modules" / ".bin"
    if await aiofiles.os.path.isdir(local_bin_dir):
        local_which = await which("agent-browser", path=str(local_bin_dir))
        if local_which and (
            await agent_browser_runnable(local_which)
            if validate
            else await _agent_browser_candidate_present(local_which)
        ):
            if not validate:
                return local_which
            _cached_agent_browser = local_which
            _agent_browser_resolved = True
            return _cached_agent_browser

    # Check common npx locations (also search the extended fallback PATH)
    npx_path = await which("npx")
    if not npx_path and extended_path:
        npx_path = await which("npx", path=extended_path)
    if npx_path:
        if not validate:
            return "npx agent-browser"
        _cached_agent_browser = "npx agent-browser"
        _agent_browser_resolved = True
        return _cached_agent_browser

    if not validate:
        raise FileNotFoundError("agent-browser CLI not found")

    _agent_browser_resolved = True
    raise FileNotFoundError(
        "agent-browser CLI not found. Install it with: "
        f"{_browser_install_hint()}\n"
        "Or run 'npm install' in the repo root to install locally.\n"
        "Or ensure npx is available in your PATH."
    )


def _extract_screenshot_path_from_text(text: str) -> Optional[str]:
    """Extract a screenshot file path from agent-browser human-readable output."""
    if not text:
        return None

    patterns = [
        r"Screenshot saved to ['\"](?P<path>/[^'\"]+?\.png)['\"]",
        r"Screenshot saved to (?P<path>/\S+?\.png)(?:\s|$)",
        r"(?P<path>/\S+?\.png)(?:\s|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            path = match.group("path").strip().strip("'\"")
            if path:
                return path

    return None


async def _run_browser_command(
    task_id: str,
    command: str,
    args: Optional[List[str]] = None,
    timeout: Optional[int] = None,
    _engine_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run an agent-browser CLI command using our pre-created Browserbase session.

    Args:
        task_id: Task identifier to get the right session
        command: The command to run (e.g., "open", "click")
        args: Additional arguments for the command
        timeout: Command timeout in seconds.  ``None`` reads
                 ``browser.command_timeout`` from config (default 30s).
        _engine_override: Force a specific engine for this call only.  Used
                          internally by the Lightpanda fallback to retry with
                          Chrome without touching global state.

    Returns:
        Parsed JSON response from agent-browser
    """
    if timeout is None:
        timeout = await _safe_command_timeout()
    args = args or []

    # Build the command
    try:
        browser_cmd = await _find_agent_browser()
    except FileNotFoundError as e:
        logger.warning("agent-browser CLI not found: %s", e)
        return {"success": False, "error": str(e)}

    if await _requires_real_termux_browser_install(browser_cmd):
        error = _termux_browser_install_error()
        logger.warning("browser command blocked on Termux: %s", error)
        return {"success": False, "error": error}

    # Local mode with no Chromium on disk: fail fast with an actionable
    # message instead of hanging for _command_timeout seconds per call.
    # Skip when engine=lightpanda — LP doesn't need Chromium for navigation.
    if (
        await _is_local_mode()
        and not await _chromium_installed()
        and await _get_browser_engine() != "lightpanda"
        and not await _maybe_autoinstall_chromium()
    ):
        if await _running_in_docker():
            hint = (
                "Chromium browser is missing. You're running in Docker — pull "
                "the latest image to get the bundled Chromium: "
                "docker pull ghcr.io/nousresearch/hermes-agent:latest"
            )
        else:
            hint = (
                "Chromium browser is missing. Install it with: "
                "npx agent-browser install --with-deps "
                "(or: npx playwright install --with-deps chromium)"
            )
        logger.warning("browser command blocked: %s", hint)
        return {"success": False, "error": hint}

    from tools.interrupt import is_interrupted

    if is_interrupted():
        return {"success": False, "error": "Interrupted"}

    # Get session info (creates Browserbase session with proxies if needed)
    try:
        session_info = await _get_session_info(task_id)
    except Exception as e:
        logger.warning("Failed to create browser session for task=%s: %s", task_id, e)
        return {
            "success": False,
            "error": f"Failed to create browser session: {str(e)}",
        }

    # Build the command with the appropriate backend flag.
    # Cloud mode: --cdp <websocket_url> connects to Browserbase.
    # Local mode: --session <name> launches a local headless Chromium.
    # The rest of the command (--json, command, args) is identical.
    if session_info.get("cdp_url"):
        # Cloud mode — connect to remote Browserbase browser via CDP
        # IMPORTANT: Do NOT use --session with --cdp. In agent-browser >=0.13,
        # --session creates a local browser instance and silently ignores --cdp.
        backend_args = ["--cdp", session_info["cdp_url"]]
    else:
        # Local mode — launch Chromium (headless by default, headed when configured)
        backend_args = ["--session", session_info["session_name"]]
        if await _is_headed_mode():
            backend_args.append("--headed")

    # Lightpanda engine injection (local mode only, agent-browser v0.25.3+).
    # Use the resolved session backend rather than global cloud-provider state:
    # hybrid private-URL routing can create a local sidecar while a cloud
    # provider remains configured for public URLs.
    engine = _engine_override or await _get_browser_engine()
    if (
        engine != "auto"
        and not await _is_camofox_mode()
        and not session_info.get("cdp_url")
    ):
        backend_args += ["--engine", engine]

    # Keep concrete executable paths intact, even when they contain spaces.
    # Only the synthetic npx fallback needs to expand into multiple argv items.
    # shutil.which resolves npx → npx.cmd on Windows; bare "npx" stays on POSIX.
    if browser_cmd == "npx agent-browser":
        _npx_bin = await aiofiles.os.wrap(shutil.which)("npx") or "npx"
        cmd_prefix = [_npx_bin, "agent-browser"]
    else:
        cmd_prefix = [browser_cmd]

    cmd_parts = cmd_prefix + backend_args + ["--json", command] + args

    try:
        # Give each task its own socket directory to prevent concurrency conflicts.
        # Without this, parallel workers fight over the same default socket path,
        # causing "Failed to create socket directory: Permission denied" errors.
        task_socket_dir = os.path.join(
            _socket_safe_tmpdir(), f"agent-browser-{session_info['session_name']}"
        )
        await aiofiles.os.makedirs(task_socket_dir, mode=0o700, exist_ok=True)
        # Record this hermes PID as the session owner (cross-process safe
        # orphan detection — see _write_owner_pid).
        await _write_owner_pid(task_socket_dir, session_info["session_name"])
        logger.debug(
            "browser cmd=%s task=%s socket_dir=%s (%d chars)",
            command,
            task_id,
            task_socket_dir,
            len(task_socket_dir),
        )

        browser_env = _build_browser_env()

        # Ensure subprocesses inherit the same browser-specific PATH fallbacks
        # used during CLI discovery.
        browser_env["PATH"] = await _merge_browser_path(
            browser_env.get("PATH", "")
        )
        browser_env["AGENT_BROWSER_SOCKET_DIR"] = task_socket_dir

        # Tell the agent-browser daemon to self-terminate after being idle
        # for our configured inactivity timeout.  This is the daemon-side
        # counterpart to our Python-side _cleanup_inactive_browser_sessions
        # — the daemon kills itself and its Chrome children when no CLI
        # commands arrive within the window.  Added in agent-browser 0.24.
        if "AGENT_BROWSER_IDLE_TIMEOUT_MS" not in browser_env:
            idle_ms = str(BROWSER_SESSION_INACTIVITY_TIMEOUT * 1000)
            browser_env["AGENT_BROWSER_IDLE_TIMEOUT_MS"] = idle_ms

        # Inject --no-sandbox when needed (issue #15765):
        # - Running as root: Chromium always refuses to start without it
        # - Ubuntu 23.10+ / AppArmor systems: unprivileged user namespaces
        #   are restricted, causing Chromium to exit with "No usable sandbox"
        #   even for non-root users running under systemd or containers.
        # Honour either the legacy AGENT_BROWSER_CHROME_FLAGS (never consumed by
        # agent-browser itself, but documented in older notes) or the real
        # AGENT_BROWSER_ARGS — if the user pre-sets either, don't overwrite it.
        if (
            "AGENT_BROWSER_ARGS" not in browser_env
            and "AGENT_BROWSER_CHROME_FLAGS" not in browser_env
        ):
            if await _needs_chromium_sandbox_bypass():
                logger.debug(
                    "browser: sandbox bypass needed (root/docker/AppArmor userns) — "
                    "injecting --no-sandbox"
                )
                browser_env["AGENT_BROWSER_ARGS"] = (
                    "--no-sandbox,--disable-dev-shm-usage"
                )

        # Use temp files for stdout/stderr instead of pipes.
        # agent-browser starts a background daemon that inherits file
        # descriptors.  With capture_output=True (pipes), the daemon keeps
        # the pipe fds open after the CLI exits, so communicate() never
        # sees EOF and blocks until the timeout fires.
        stdout_path = os.path.join(task_socket_dir, f"_stdout_{command}")
        stderr_path = os.path.join(task_socket_dir, f"_stderr_{command}")
        open_file = aiofiles.os.wrap(os.open)
        close_file = aiofiles.os.wrap(os.close)
        stdout_fd = await open_file(
            stdout_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
        )
        stderr_fd = await open_file(
            stderr_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
        )
        try:
            # See matching comment at the other Popen site above — on
            # Windows we put agent-browser in its own process group, force
            # STARTF_USESTDHANDLES so CreateProcess hands the child ONLY our
            # three explicit handles (no leaked parent-console handles to
            # confuse the Rust binary's daemon-spawn), and close_fds=True to
            # block inheritance of everything else.
            _popen_extra: dict = {}
            if os.name == "nt":
                # See matching block at the other Popen site — CREATE_NO_WINDOW
                # only, NO CREATE_NEW_PROCESS_GROUP (cancels asyncio loop task
                # on Python 3.11 Windows → KeyboardInterrupt in CLI MainThread).
                _popen_extra["creationflags"] = windows_hide_flags()
                _popen_extra["close_fds"] = True
                _si = subprocess.STARTUPINFO()
                _si.dwFlags |= subprocess.STARTF_USESTDHANDLES
                _popen_extra["startupinfo"] = _si
            proc = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdout=stdout_fd,
                stderr=stderr_fd,
                stdin=subprocess.DEVNULL,
                env=browser_env,
                **_popen_extra,
            )
        finally:
            await close_file(stdout_fd)
            await close_file(stderr_fd)

        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            stdout, stderr = await _read_command_output_files(stdout_path, stderr_path)
            await _unlink_command_output_files(stdout_path, stderr_path)
            if stderr and stderr.strip():
                logger.warning(
                    "browser '%s' stderr after timeout: %s",
                    command,
                    stderr.strip()[:500],
                )
            logger.warning(
                "browser '%s' timed out after %ds (task=%s, socket_dir=%s)",
                command,
                timeout,
                task_id,
                task_socket_dir,
            )
            result = {
                "success": False,
                "error": await _format_browser_timeout_error(
                    command, timeout, stdout, stderr
                ),
            }
            # Fall through to fallback check below
        else:
            async with aiofiles.open(stdout_path, "r", encoding="utf-8") as f:
                stdout = await f.read()
            async with aiofiles.open(stderr_path, "r", encoding="utf-8") as f:
                stderr = await f.read()
            returncode = proc.returncode

            # Clean up temp files (best-effort)
            for p in (stdout_path, stderr_path):
                try:
                    await aiofiles.os.remove(p)
                except OSError:
                    pass

            # Log stderr for diagnostics — use warning level on failure so it's visible
            if stderr and stderr.strip():
                level = logging.WARNING if returncode != 0 else logging.DEBUG
                logger.log(
                    level, "browser '%s' stderr: %s", command, stderr.strip()[:500]
                )

            stdout_text = stdout.strip()

            # Empty output with rc=0 is a broken state — treat as failure rather
            # than silently returning {"success": True, "data": {}}.
            # Some commands (close, record) legitimately return no output.
            if (
                not stdout_text
                and returncode == 0
                and command not in _EMPTY_OK_COMMANDS
            ):
                logger.warning("browser '%s' returned empty output (rc=0)", command)
                result = {
                    "success": False,
                    "error": f"Browser command '{command}' returned no output",
                }
            elif stdout_text:
                try:
                    parsed = json.loads(stdout_text)
                    # Warn if snapshot came back empty (common sign of daemon/CDP issues)
                    if command == "snapshot" and parsed.get("success"):
                        snap_data = parsed.get("data", {})
                        if not snap_data.get("snapshot") and not snap_data.get("refs"):
                            logger.warning(
                                "snapshot returned empty content. "
                                "Possible stale daemon or CDP connection issue. "
                                "returncode=%s",
                                returncode,
                            )
                    result = parsed
                except json.JSONDecodeError:
                    raw = stdout_text[:2000]
                    logger.warning(
                        "browser '%s' returned non-JSON output (rc=%s): %s",
                        command,
                        returncode,
                        raw[:500],
                    )

                    if command == "screenshot":
                        stderr_text = (stderr or "").strip()
                        combined_text = "\n".join(
                            part for part in [stdout_text, stderr_text] if part
                        )
                        recovered_path = _extract_screenshot_path_from_text(
                            combined_text
                        )

                        if recovered_path and await aiofiles.os.path.exists(
                            recovered_path
                        ):
                            logger.info(
                                "browser 'screenshot' recovered file from non-JSON output: %s",
                                recovered_path,
                            )
                            result = {
                                "success": True,
                                "data": {
                                    "path": recovered_path,
                                    "raw": raw,
                                },
                            }
                        else:
                            result = {
                                "success": False,
                                "error": f"Non-JSON output from agent-browser for '{command}': {raw}",
                            }
                    else:
                        result = {
                            "success": False,
                            "error": f"Non-JSON output from agent-browser for '{command}': {raw}",
                        }
            elif returncode != 0:
                # Check for errors
                error_msg = (
                    stderr.strip()
                    if stderr
                    else f"Command failed with code {returncode}"
                )
                logger.warning(
                    "browser '%s' failed (rc=%s): %s",
                    command,
                    returncode,
                    error_msg[:300],
                )
                result = {"success": False, "error": error_msg}
            else:
                result = {"success": True, "data": {}}

    except Exception as e:
        logger.warning("browser '%s' exception: %s", command, e, exc_info=True)
        result = {"success": False, "error": str(e)}

    # --- Lightpanda automatic Chrome fallback ---
    # If engine is lightpanda and the result looks broken, retry with Chrome.
    # This runs for ALL exit paths (timeout, empty, non-JSON, nonzero rc, parsed).
    fallback_reason = _lightpanda_fallback_reason(engine, command, result)
    if fallback_reason:
        logger.info(
            "Lightpanda fallback: retrying '%s' with Chrome (task=%s): %s",
            command,
            task_id,
            fallback_reason,
        )
        # For screenshots, use the dedicated Chrome fallback helper
        # (spins up a separate Chrome session to the same URL).
        if command == "screenshot":
            fallback_result = await _chrome_fallback_screenshot(
                task_id, args or [], timeout
            )
        else:
            fallback_result = await _run_chrome_fallback_command(
                task_id, command, args, timeout
            )
        return _annotate_lightpanda_fallback(fallback_result, fallback_reason)

    return result


async def _store_full_snapshot(snapshot_text: str) -> Optional[str]:
    """Write a full page snapshot to cache/web and return its absolute path.

    Called whenever a snapshot exceeds SNAPSHOT_SUMMARIZE_THRESHOLD and the
    model is about to receive a truncated or LLM-summarized view. Mirrors
    ``web_tools._store_full_text``: the file lands in the same cache/web
    directory (mounted read-only into remote backends via
    credential_files._CACHE_DIRS) so the agent's read_file/terminal tools can
    page through the complete accessibility tree — including element refs that
    the truncated view dropped — on any backend.

    The stored copy is secret-redacted (same force-redaction boundary as
    ``_redact_browser_output``) since page-rendered API keys or tokens must
    not be written to disk unmasked. The filename is keyed on a content hash,
    so repeated snapshots of the same page state dedupe to one file. Returns
    None on failure (storage is best-effort; the truncated view is still
    returned to the model).
    """
    try:
        import hashlib
        from hermes_constants import get_hermes_dir
        from agent.redact import redact_sensitive_text

        content = redact_sensitive_text(snapshot_text, force=True)
        if len(content) > MAX_STORED_SNAPSHOT_CHARS:
            content = (
                content[:MAX_STORED_SNAPSHOT_CHARS]
                + f"\n\n[... stored copy truncated at {MAX_STORED_SNAPSHOT_CHARS:,} chars "
                f"of {len(content):,} ...]"
            )
        cache_dir = await get_hermes_dir("cache/web", "web_cache")
        await aiofiles.os.makedirs(cache_dir, exist_ok=True)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:10]
        path = cache_dir / f"browser-snapshot-{digest}.txt"
        async with aiofiles.open(path, "w", encoding="utf-8") as handle:
            await handle.write(content)
        return str(path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to store full browser snapshot: %s", exc)
        return None


async def _extract_relevant_content(
    snapshot_text: str, user_task: Optional[str] = None
) -> str:
    """Use LLM to extract relevant content from a snapshot based on the user's task.

    The full snapshot is stored to cache/web first (summarization is lossy —
    the pointer lets the agent read anything the summary dropped). Falls back
    to simple truncation when no auxiliary text model is configured.
    """
    stored_path = await _store_full_snapshot(snapshot_text)
    stored_note = (
        (
            f"\n\n[Summarized from a {len(snapshot_text):,}-char snapshot. Full snapshot "
            f"saved to: {stored_path} — read it with read_file if anything is missing.]"
        )
        if stored_path
        else ""
    )
    if user_task:
        extraction_prompt = (
            f"You are a content extractor for a browser automation agent.\n\n"
            f"The user's task is: {user_task}\n\n"
            f"Given the following page snapshot (accessibility tree representation), "
            f"extract and summarize the most relevant information for completing this task. Focus on:\n"
            f"1. Interactive elements (buttons, links, inputs) that might be needed\n"
            f"2. Text content relevant to the task (prices, descriptions, headings, important info)\n"
            f"3. Navigation structure if relevant\n\n"
            f"Keep ref IDs (like [ref=e5]) for interactive elements so the agent can use them.\n\n"
            f"Page Snapshot:\n{snapshot_text}\n\n"
            f"Provide a concise summary that preserves actionable information and relevant content."
        )
    else:
        extraction_prompt = (
            f"Summarize this page snapshot, preserving:\n"
            f"1. All interactive elements with their ref IDs (like [ref=e5])\n"
            f"2. Key text content and headings\n"
            f"3. Important information visible on the page\n\n"
            f"Page Snapshot:\n{snapshot_text}\n\n"
            f"Provide a concise summary focused on interactive elements and key content."
        )

    # Redact secrets from snapshot before sending to auxiliary LLM.
    # Without this, a page displaying env vars or API keys would leak
    # secrets to the extraction model before run_agent.py's general
    # redaction layer ever sees the tool result.
    from agent.redact import redact_sensitive_text

    extraction_prompt = redact_sensitive_text(extraction_prompt)

    try:
        call_kwargs = {
            "task": "web_extract",
            "messages": [{"role": "user", "content": extraction_prompt}],
            "max_tokens": 4000,
            "temperature": 0.1,
        }
        model = _get_extraction_model()
        if model:
            call_kwargs["model"] = model
        response = await _lazy_call_llm(**call_kwargs)
        extracted = (response.choices[0].message.content or "").strip()
        if not extracted:
            # _truncate_snapshot stores its own pointer (dedupes to the same
            # cache file by content hash), so return it without stored_note.
            return await _truncate_snapshot(snapshot_text)
        # Redact any secrets the auxiliary LLM may have echoed back.
        return redact_sensitive_text(extracted) + stored_note
    except Exception:
        return await _truncate_snapshot(snapshot_text)


async def _truncate_snapshot(
    snapshot_text: str, max_chars: int = SNAPSHOT_SUMMARIZE_THRESHOLD
) -> str:
    """Structure-aware truncation for snapshots.

    Cuts at line boundaries so that accessibility tree elements are never
    split mid-line. The full snapshot is saved to cache/web (same pattern as
    web_extract's truncate-and-store) and the appended note tells the agent
    exactly where the complete text lives and how to page through it with
    read_file — element refs beyond the cut are in the file, not lost.

    Args:
        snapshot_text: The snapshot text to truncate
        max_chars: Maximum characters to keep

    Returns:
        Truncated text with a stored-full-text pointer if truncated
    """
    if len(snapshot_text) <= max_chars:
        return snapshot_text

    stored_path = await _store_full_snapshot(snapshot_text)

    lines = snapshot_text.split("\n")
    result: list[str] = []
    chars = 0
    # Reserve space for the truncation note (the stored-path variant is the
    # longer of the two). Clamp so tiny max_chars values still keep content.
    reserve = min(110 + len(stored_path or ""), max_chars // 2)
    for line in lines:
        if chars + len(line) + 1 > max_chars - reserve:
            break
        result.append(line)
        chars += len(line) + 1
    remaining = len(lines) - len(result)
    if remaining > 0:
        if stored_path:
            next_line = len(result) + 1
            result.append(
                f"\n[... {remaining} more lines truncated — full snapshot: "
                f'read_file path="{stored_path}" offset={next_line} limit=200]'
            )
        else:
            result.append(
                f"\n[... {remaining} more lines truncated, use browser_snapshot for full content]"
            )
    return "\n".join(result)


def _redact_browser_output(value: Any) -> Any:
    """Redact secrets from browser-originated data before returning to the model.

    Browser snapshots, console messages, JS exceptions, and eval results can
    contain page-rendered API keys, cookies, bearer tokens, or pasted secrets.
    Tool output is a model boundary, so force redaction here even if global log
    redaction is disabled for debugging.
    """
    from agent.redact import redact_sensitive_text

    if isinstance(value, str):
        return redact_sensitive_text(value, force=True)
    if isinstance(value, list):
        return [_redact_browser_output(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_browser_output(item) for item in value)
    if isinstance(value, dict):
        return {key: _redact_browser_output(item) for key, item in value.items()}
    return value


# ============================================================================
# Browser Tool Functions
# ============================================================================


async def browser_navigate(url: str, task_id: Optional[str] = None) -> str:
    """
    Navigate to a URL in the browser.

    Args:
        url: The URL to navigate to
        task_id: Task identifier for session isolation

    Returns:
        JSON string with navigation result (includes stealth features info on first nav)
    """
    # Secret exfiltration protection — block URLs that embed API keys or
    # tokens in query parameters. A prompt injection could trick the agent
    # into navigating to https://evil.com/steal?key=sk-ant-... to exfil secrets.
    # Also check URL-decoded form to catch %2D encoding tricks (e.g. sk%2Dant%2D...).
    import urllib.parse
    from agent.redact import _PREFIX_RE

    url_decoded = urllib.parse.unquote(url)
    if _PREFIX_RE.search(url) or _PREFIX_RE.search(url_decoded):
        return json.dumps({
            "success": False,
            "error": "Blocked: URL contains what appears to be an API key or token. "
            "Secrets must not be sent in URLs.",
        })
    url = _normalize_url_for_request(url)
    normalized_decoded = urllib.parse.unquote(url)
    if _PREFIX_RE.search(url) or _PREFIX_RE.search(normalized_decoded):
        return json.dumps({
            "success": False,
            "error": "Blocked: URL contains what appears to be an API key or token. "
            "Secrets must not be sent in URLs.",
        })

    # SSRF protection — block private/internal addresses before navigating.
    # Skipped for local backends (Camofox, headless Chromium without a cloud
    # provider) because the agent already has full local network access via
    # the terminal tool.  Also skipped when hybrid routing will auto-spawn a
    # local Chromium sidecar for this URL (cloud provider configured +
    # private URL + ``browser.auto_local_for_private_urls`` enabled) — the
    # cloud provider never sees the URL in that case.  Can also be opted
    # out globally via ``browser.allow_private_urls`` in config.
    effective_task_id = task_id or "default"
    nav_session_key = await _navigation_session_key(effective_task_id, url)
    auto_local_this_nav = _is_local_sidecar_key(nav_session_key)

    sensitive_query_key = _sensitive_query_param_name(url)
    if (
        sensitive_query_key
        and not await _is_local_backend()
        and not auto_local_this_nav
    ):
        return json.dumps({
            "success": False,
            "error": (
                "Blocked: URL contains a credential-like query parameter "
                f"({sensitive_query_key}). Cloud browser backends are third-party "
                "readers; use a local browser/CDP session or remove the sensitive "
                "query parameter before navigating."
            ),
        })

    # Always-blocked floor: cloud metadata / IMDS endpoints are denied
    # regardless of backend, hybrid routing, or allow_private_urls.
    # There's no legitimate agent use case for navigating to
    # 169.254.169.254 / metadata.google.internal / ECS task metadata
    # via a browser, and routing those to a local Chromium sidecar
    # on an EC2/GCP/Azure host exfiltrates IAM credentials (#16234).
    # The floor is UNCONDITIONAL — it must fire for every backend,
    # including the pure-local headless Chromium and off-host CDP cases
    # (a local Chromium on a cloud VM still reaches the host IMDS).
    if await _is_always_blocked_url(url):
        return json.dumps({
            "success": False,
            "error": "Blocked: URL targets a cloud metadata endpoint",
        })

    if (
        not await _is_local_backend()
        and not auto_local_this_nav
        and not await _allow_private_urls()
        and not await _is_safe_url(url)
    ):
        return json.dumps({
            "success": False,
            "error": "Blocked: URL targets a private or internal address",
        })

    # Website policy check — block before navigating
    blocked = await check_website_access(url)
    if blocked:
        return json.dumps({
            "success": False,
            "error": blocked["message"],
            "blocked_by_policy": {
                "host": blocked["host"],
                "rule": blocked["rule"],
                "source": blocked["source"],
            },
        })

    # Camofox backend — delegate after safety checks pass
    if await _is_camofox_mode():
        from tools.browser_camofox import camofox_navigate

        return await camofox_navigate(url, task_id)

    if auto_local_this_nav:
        configured_provider = await _get_cloud_provider()
        logger.info(
            "browser_navigate: auto-routing %s to local Chromium sidecar "
            "(cloud provider %s stays on cloud for public URLs; "
            "set browser.auto_local_for_private_urls: false to disable)",
            url,
            type(configured_provider).__name__ if configured_provider else "none",
        )

    # Get session info to check if this is a new session
    # (will create one with features logged if not exists)
    session_info = await _get_session_info(nav_session_key)
    is_first_nav = session_info.get("_first_nav", True)

    # Auto-start recording if configured and this is first navigation
    if is_first_nav:
        session_info["_first_nav"] = False
        await _maybe_start_recording(nav_session_key)

    result = await _run_browser_command(
        nav_session_key,
        "open",
        [url],
        timeout=await _get_open_command_timeout(first_open=is_first_nav),
    )

    if result.get("success"):
        data = result.get("data", {})
        title = data.get("title", "")
        final_url = data.get("url", url)

        # Post-redirect SSRF check — if the browser followed a redirect to a
        # private/internal address, block the result so the model can't read
        # internal content via subsequent browser_snapshot calls.
        # Skipped for local backends (same rationale as the pre-nav check),
        # and for the hybrid local sidecar (we're already on a local browser
        # hitting a private URL by design).
        # Always-blocked floor (cloud metadata / IMDS) is enforced for every
        # backend and even when auto_local_this_nav is true — see pre-nav
        # check for rationale (#16234).
        if (
            final_url
            and final_url != url
            and await _is_always_blocked_url(final_url)
        ):
            await _run_browser_command(
                nav_session_key, "open", ["about:blank"], timeout=10
            )
            return json.dumps({
                "success": False,
                "error": "Blocked: redirect landed on a cloud metadata endpoint",
            })

        if (
            not await _is_local_backend()
            and not auto_local_this_nav
            and not await _allow_private_urls()
            and final_url
            and final_url != url
            and not await _is_safe_url(final_url)
        ):
            # Navigate away to a blank page to prevent snapshot leaks
            await _run_browser_command(
                nav_session_key, "open", ["about:blank"], timeout=10
            )
            return json.dumps({
                "success": False,
                "error": "Blocked: redirect landed on a private/internal address",
            })

        response = {"success": True, "url": final_url, "title": title}
        # Remember only a successful, non-blocked navigation as the task owner.
        # Failed opens and blocked redirects must not retarget follow-up clicks
        # or snapshots to a newly-created but irrelevant session.
        async with _cleanup_lock:
            _last_active_session_key[effective_task_id] = nav_session_key
        _copy_fallback_warning(response, result)

        # Detect common "blocked" page patterns from title/url
        blocked_patterns = [
            "access denied",
            "access to this page has been denied",
            "blocked",
            "bot detected",
            "verification required",
            "please verify",
            "are you a robot",
            "captcha",
            "cloudflare",
            "ddos protection",
            "checking your browser",
            "just a moment",
            "attention required",
        ]
        title_lower = title.lower()

        if any(pattern in title_lower for pattern in blocked_patterns):
            response["bot_detection_warning"] = (
                f"Page title '{title}' suggests bot detection. The site may have blocked this request. "
                "Options: 1) Try adding delays between actions, 2) Access different pages first, "
                "3) Enable advanced stealth (BROWSERBASE_ADVANCED_STEALTH=true, requires Scale plan), "
                "4) Some sites have very aggressive bot detection that may be unavoidable."
            )

        # Include feature info on first navigation so model knows what's active
        if is_first_nav and "features" in session_info:
            features = session_info["features"]
            active_features = [k for k, v in features.items() if v]
            if not features.get("proxies"):
                response["stealth_warning"] = (
                    "Running WITHOUT residential proxies. Bot detection may be more aggressive. "
                    "Consider upgrading Browserbase plan for proxy support."
                )
            response["stealth_features"] = active_features

        # Auto-take a compact snapshot so the model can act immediately
        # without a separate browser_snapshot call.
        try:
            snap_result = await _run_browser_command(
                nav_session_key, "snapshot", ["-c"]
            )
            if snap_result.get("success"):
                snap_data = snap_result.get("data", {})
                snapshot_text = snap_data.get("snapshot", "")
                refs = snap_data.get("refs", {})
                if len(snapshot_text) > SNAPSHOT_SUMMARIZE_THRESHOLD:
                    snapshot_text = await _truncate_snapshot(snapshot_text)
                response["snapshot"] = _redact_browser_output(snapshot_text)
                response["element_count"] = len(refs) if refs else 0
                if snap_result.get("fallback_warning") and not response.get(
                    "fallback_warning"
                ):
                    _copy_fallback_warning(response, snap_result)
        except Exception as e:
            logger.debug("Auto-snapshot after navigate failed: %s", e)

        return json.dumps(response, ensure_ascii=False)
    else:
        return json.dumps(
            {"success": False, "error": result.get("error", "Navigation failed")},
            ensure_ascii=False,
        )


async def browser_snapshot(
    full: bool = False, task_id: Optional[str] = None, user_task: Optional[str] = None
) -> str:
    """
    Get a text-based snapshot of the current page's accessibility tree.

    Args:
        full: If True, return complete snapshot. If False, return compact view.
        task_id: Task identifier for session isolation
        user_task: The user's current task (for task-aware extraction)

    Returns:
        JSON string with page snapshot
    """
    if await _is_camofox_mode():
        from tools.browser_camofox import camofox_snapshot

        return await camofox_snapshot(full, task_id, user_task)

    effective_task_id = _last_session_key(task_id or "default")

    # Build command args based on full flag
    args = []
    if not full:
        args.extend(["-c"])  # Compact mode

    result = await _run_browser_command(effective_task_id, "snapshot", args)

    if result.get("success"):
        data = result.get("data", {})
        snapshot_text = data.get("snapshot", "")
        refs = data.get("refs", {})

        # ── Private-network guard: block snapshots from eval-navigated private pages ──
        # After any eval (browser_console) that may have changed location.href to a
        # private/internal address, the snapshot would expose private page content.
        # Re-check the current URL before returning the snapshot.
        if (
            not await _is_local_backend()
            and not _is_local_sidecar_key(effective_task_id)
            and not await _allow_private_urls()
        ):
            try:
                _url_result = await _run_browser_command(
                    effective_task_id,
                    "eval",
                    ["window.location.href"],
                    timeout=5,
                    _engine_override="auto",
                )
                if _url_result.get("success"):
                    _current_url = (
                        _url_result
                        .get("data", {})
                        .get("result", "")
                        .strip()
                        .strip('"')
                        .strip("'")
                    )
                    if _current_url and not await _is_safe_url(_current_url):
                        return json.dumps(
                            {
                                "success": False,
                                "error": (
                                    "Blocked: page URL targets a private or internal address "
                                    f"({_current_url}). This may have been caused by a "
                                    "JavaScript navigation via browser_console."
                                ),
                            },
                            ensure_ascii=False,
                        )
            except Exception as _url_exc:
                logger.debug("browser_snapshot: URL safety check failed (%s)", _url_exc)

        # Check if snapshot needs summarization
        if len(snapshot_text) > SNAPSHOT_SUMMARIZE_THRESHOLD and user_task:
            snapshot_text = await _extract_relevant_content(snapshot_text, user_task)
        elif len(snapshot_text) > SNAPSHOT_SUMMARIZE_THRESHOLD:
            snapshot_text = await _truncate_snapshot(snapshot_text)

        response = {
            "success": True,
            "snapshot": _redact_browser_output(snapshot_text),
            "element_count": len(refs) if refs else 0,
        }
        _copy_fallback_warning(response, result)

        # Merge supervisor state (pending dialogs + frame tree) when a CDP
        # supervisor is attached to this task. No-op otherwise. See
        # website/docs/developer-guide/browser-supervisor.md.
        try:
            from tools.browser_supervisor import SUPERVISOR_REGISTRY

            _supervisor = SUPERVISOR_REGISTRY.get(effective_task_id)
            if _supervisor is not None:
                _sv_snap = _supervisor.snapshot()
                if _sv_snap.active:
                    response.update(_redact_browser_output(_sv_snap.to_dict()))
        except Exception as _sv_exc:
            logger.debug("supervisor snapshot merge failed: %s", _sv_exc)

        return json.dumps(response, ensure_ascii=False)
    else:
        response = {
            "success": False,
            "error": result.get("error", "Failed to get snapshot"),
        }
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)


async def browser_click(ref: str, task_id: Optional[str] = None) -> str:
    """
    Click on an element.

    Args:
        ref: Element reference (e.g., "@e5")
        task_id: Task identifier for session isolation

    Returns:
        JSON string with click result
    """
    if await _is_camofox_mode():
        from tools.browser_camofox import camofox_click

        return await camofox_click(ref, task_id)

    effective_task_id = _last_session_key(task_id or "default")
    blocked = await _blocked_private_page_action(effective_task_id, "click")
    if blocked is not None:
        return blocked

    # Ensure ref starts with @
    if not ref.startswith("@"):
        ref = f"@{ref}"

    result = await _run_browser_command(effective_task_id, "click", [ref])

    if result.get("success"):
        response = {"success": True, "clicked": ref}
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)
    else:
        response = {
            "success": False,
            "error": result.get("error", f"Failed to click {ref}"),
        }
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)


async def browser_type(ref: str, text: str, task_id: Optional[str] = None) -> str:
    """
    Type text into an input field.

    Args:
        ref: Element reference (e.g., "@e3")
        text: Text to type
        task_id: Task identifier for session isolation

    Returns:
        JSON string with type result
    """
    if await _is_camofox_mode():
        from tools.browser_camofox import camofox_type

        return await camofox_type(ref, text, task_id)

    effective_task_id = _last_session_key(task_id or "default")
    blocked = await _blocked_private_page_action(effective_task_id, "type")
    if blocked is not None:
        return blocked

    # Ensure ref starts with @
    if not ref.startswith("@"):
        ref = f"@{ref}"

    # Use fill command (clears then types)
    result = await _run_browser_command(effective_task_id, "fill", [ref, text])

    from agent.display import (
        redact_browser_typed_text_for_display,
        redact_tool_args_for_display,
    )

    display_text = (redact_tool_args_for_display("browser_type", {"text": text}) or {})[
        "text"
    ]

    if result.get("success"):
        response = {
            "success": True,
            # Run typed text through the secret-pattern redactor so API keys /
            # tokens don't leak into tool progress or chat history.  Normal
            # text passes through unchanged.  The raw value was already sent
            # to the browser command above.
            "typed": display_text,
            "element": ref,
        }
        response = _copy_fallback_warning(response, result)
        response = redact_browser_typed_text_for_display(response, text)
        return json.dumps(response, ensure_ascii=False)
    else:
        response = {
            "success": False,
            "error": result.get("error", f"Failed to type into {ref}"),
        }
        response = _copy_fallback_warning(response, result)
        response = redact_browser_typed_text_for_display(response, text)
        return json.dumps(response, ensure_ascii=False)


async def browser_scroll(direction: str, task_id: Optional[str] = None) -> str:
    """
    Scroll the page.

    Args:
        direction: "up" or "down"
        task_id: Task identifier for session isolation

    Returns:
        JSON string with scroll result
    """
    # Validate direction
    if direction not in {"up", "down"}:
        return json.dumps(
            {
                "success": False,
                "error": f"Invalid direction '{direction}'. Use 'up' or 'down'.",
            },
            ensure_ascii=False,
        )

    # Single scroll with pixel amount instead of 5x subprocess calls.
    # agent-browser supports: agent-browser scroll down 500
    # ~500px is roughly half a viewport of travel.
    _SCROLL_PIXELS = 500

    if await _is_camofox_mode():
        from tools.browser_camofox import camofox_scroll

        # Camofox REST API doesn't support pixel args; use repeated calls
        _SCROLL_REPEATS = 5
        result = ""
        for _ in range(_SCROLL_REPEATS):
            result = await camofox_scroll(direction, task_id)
        return result

    effective_task_id = _last_session_key(task_id or "default")

    result = await _run_browser_command(
        effective_task_id, "scroll", [direction, str(_SCROLL_PIXELS)]
    )
    if not result.get("success"):
        response = {
            "success": False,
            "error": result.get("error", f"Failed to scroll {direction}"),
        }
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)

    response = {"success": True, "scrolled": direction}
    return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)


async def browser_back(task_id: Optional[str] = None) -> str:
    """
    Navigate back in browser history.

    Args:
        task_id: Task identifier for session isolation

    Returns:
        JSON string with navigation result
    """
    if await _is_camofox_mode():
        from tools.browser_camofox import camofox_back

        return await camofox_back(task_id)

    effective_task_id = _last_session_key(task_id or "default")
    result = await _run_browser_command(effective_task_id, "back", [])

    if result.get("success"):
        # Browser history can land on a private/internal/cloud-metadata
        # address that the browser_navigate preflight never saw (e.g. a
        # redirect chain from an earlier legitimate navigation touched an
        # internal host, or client-side history was otherwise manipulated).
        # Re-check post-navigation, matching every other content-returning
        # entry point (browser_snapshot/vision/console/eval, and click/type/
        # press via _blocked_private_page_action) — the floor must fire for
        # every backend, not just the initial navigate.
        if await _eval_ssrf_guard_active(effective_task_id):
            _blocked_url = await _current_page_private_url(effective_task_id)
            if _blocked_url:
                return json.dumps(
                    {
                        "success": False,
                        "error": (
                            "Blocked: page URL targets a private or internal address "
                            f"({_blocked_url}). Browser history navigation (back) "
                            "landed on this address."
                        ),
                    },
                    ensure_ascii=False,
                )
        data = result.get("data", {})
        response = {"success": True, "url": data.get("url", "")}
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)
    else:
        response = {"success": False, "error": result.get("error", "Failed to go back")}
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)


async def browser_press(key: str, task_id: Optional[str] = None) -> str:
    """
    Press a keyboard key.

    Args:
        key: Key to press (e.g., "Enter", "Tab")
        task_id: Task identifier for session isolation

    Returns:
        JSON string with key press result
    """
    if await _is_camofox_mode():
        from tools.browser_camofox import camofox_press

        return await camofox_press(key, task_id)

    effective_task_id = _last_session_key(task_id or "default")
    blocked = await _blocked_private_page_action(effective_task_id, "press")
    if blocked is not None:
        return blocked
    result = await _run_browser_command(effective_task_id, "press", [key])

    if result.get("success"):
        response = {"success": True, "pressed": key}
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)
    else:
        response = {
            "success": False,
            "error": result.get("error", f"Failed to press {key}"),
        }
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)


async def _blocked_private_page_action(
    effective_task_id: str, action: str
) -> Optional[str]:
    """Return a blocked payload when an unsafe cloud page would receive input."""
    if not await _eval_ssrf_guard_active(effective_task_id):
        return None
    blocked_url = await _current_page_private_url(effective_task_id)
    if not blocked_url:
        return None
    return json.dumps(
        {
            "success": False,
            "error": (
                "Blocked: page URL targets a private or internal address "
                f"({blocked_url}). Refusing to {action} on this page in this "
                "browser mode."
            ),
        },
        ensure_ascii=False,
    )


async def browser_console(
    clear: bool = False,
    expression: Optional[str] = None,
    task_id: Optional[str] = None,
) -> str:
    """Get browser console messages and JavaScript errors, or evaluate JS in the page.

    When ``expression`` is provided, evaluates JavaScript in the page context
    (like the DevTools console) and returns the result.  Otherwise returns
    console output (log/warn/error/info) and uncaught exceptions.

    Args:
        clear: If True, clear the message/error buffers after reading
        expression: JavaScript expression to evaluate in the page context
        task_id: Task identifier for session isolation

    Returns:
        JSON string with console messages/errors, or eval result
    """
    # --- JS evaluation mode ---
    if expression is not None:
        policy_error = await _enforce_browser_eval_policy(expression)
        if policy_error:
            return json.dumps(
                {"success": False, "error": policy_error}, ensure_ascii=False
            )
        return await _browser_eval(expression, task_id)

    # --- Console output mode (original behaviour) ---
    if await _is_camofox_mode():
        from tools.browser_camofox import camofox_console

        return await camofox_console(clear, task_id)

    effective_task_id = _last_session_key(task_id or "default")

    if await _eval_ssrf_guard_active(effective_task_id):
        _blocked_url = await _current_page_private_url(effective_task_id)
        if _blocked_url:
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        "Blocked: page URL targets a private or internal address "
                        f"({_blocked_url}). This may have been caused by a "
                        "JavaScript navigation via browser_console."
                    ),
                },
                ensure_ascii=False,
            )

    console_args = ["--clear"] if clear else []
    error_args = ["--clear"] if clear else []

    console_task = asyncio.create_task(
        _run_browser_command(effective_task_id, "console", console_args)
    )
    errors_task = asyncio.create_task(
        _run_browser_command(effective_task_id, "errors", error_args)
    )
    console_result, errors_result = await asyncio.gather(console_task, errors_task)

    messages = []
    if console_result.get("success"):
        for msg in console_result.get("data", {}).get("messages", []):
            messages.append({
                "type": msg.get("type", "log"),
                "text": _redact_browser_output(msg.get("text", "")),
                "source": "console",
            })

    errors = []
    if errors_result.get("success"):
        for err in errors_result.get("data", {}).get("errors", []):
            errors.append({
                "message": _redact_browser_output(err.get("message", "")),
                "source": "exception",
            })

    response = {
        "success": True,
        "console_messages": messages,
        "js_errors": errors,
        "total_messages": len(messages),
        "total_errors": len(errors),
    }
    _copy_fallback_warning(response, console_result)
    if errors_result.get("fallback_warning") and not response.get("fallback_warning"):
        _copy_fallback_warning(response, errors_result)
    return json.dumps(response, ensure_ascii=False)


async def _eval_ssrf_guard_active(effective_task_id: str) -> bool:
    """Return True when eval-driven private-network access must be guarded.

    Matches the gating used by ``browser_navigate`` / ``browser_snapshot`` /
    ``browser_vision``: the SSRF guard is only meaningful for non-local
    backends (cloud browser, or a containerized terminal whose browser-on-host
    can reach internal networks the terminal can't), and is skipped for local
    sidecar sessions and when ``allow_private_urls`` is set.
    """
    return (
        not await _is_local_backend()
        and not _is_local_sidecar_key(effective_task_id)
        and not await _allow_private_urls()
    )


# URL-shaped literals embedded in a JS expression (http/https only).  Used to
# pre-screen ``browser_console(expression=...)`` calls that fetch/XHR/navigate
# to a private host directly — that path never updates ``location.href`` so the
# post-eval page-URL recheck below can't see it.
_JS_URL_LITERAL_RE = re.compile(r"""https?://[^\s'"`)\]<>]+""", re.IGNORECASE)


async def _expression_targets_private_url(expression: str) -> Optional[str]:
    """Return the first private/always-blocked URL literal in a JS expression.

    Best-effort: scans for ``http(s)://...`` literals (fetch/XHR/navigation
    targets the agent may have embedded) and returns the first one that targets
    a private/internal address or the always-blocked cloud-metadata floor.
    Returns ``None`` when no such literal is found.
    """
    if not isinstance(expression, str):
        return None
    for match in _JS_URL_LITERAL_RE.findall(expression):
        candidate = match.rstrip(".,;")
        if await _is_always_blocked_url(candidate) or not await _is_safe_url(
            candidate
        ):
            return candidate
    return None


async def _current_page_private_url(effective_task_id: str) -> Optional[str]:
    """Return the current page URL when it targets a private/internal address.

    Reads ``window.location.href`` via a low-cost eval and returns it when the
    page has been navigated (e.g. via ``location.href = '...'`` in a prior
    eval) to an address the SSRF guard would reject.  Returns ``None`` when the
    page is public, the URL can't be determined, or the check errors (fail-open
    on probe failure, matching the snapshot/vision guards).
    """
    try:
        url_result = await _run_browser_command(
            effective_task_id,
            "eval",
            ["window.location.href"],
            timeout=5,
            _engine_override="auto",
        )
        if url_result.get("success"):
            current_url = (
                url_result
                .get("data", {})
                .get("result", "")
                .strip()
                .strip('"')
                .strip("'")
            )
            if current_url and (
                await _is_always_blocked_url(current_url)
                or not await _is_safe_url(current_url)
            ):
                return current_url
    except Exception as exc:
        logger.debug("_current_page_private_url: probe failed (%s)", exc)
    return None


_RISKY_BROWSER_EVAL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bdocument\s*\.\s*cookie\b", re.I), "document.cookie"),
    (re.compile(r"\b(?:localStorage|sessionStorage)\b", re.I), "web storage"),
    (re.compile(r"\bindexedDB\b", re.I), "IndexedDB"),
    (re.compile(r"\bcaches\s*\.\s*(?:open|match|keys)\b", re.I), "Cache Storage"),
    (
        re.compile(
            r"\bnavigator\s*\.\s*(?:clipboard|credentials|serviceWorker)\b", re.I
        ),
        "navigator sensitive API",
    ),
    (
        re.compile(r"\b(?:fetch|XMLHttpRequest|WebSocket|EventSource)\s*\(", re.I),
        "network request",
    ),
    (re.compile(r"\bnavigator\s*\.\s*sendBeacon\s*\(", re.I), "network beacon"),
    (
        re.compile(r"\bdocument\s*\.\s*forms\b.*\bvalue\b", re.I | re.S),
        "form value extraction",
    ),
    (
        re.compile(
            r"\bquerySelector(?:All)?\s*\([^)]*(?:input|textarea|password)[^)]*\).*\bvalue\b",
            re.I | re.S,
        ),
        "form value extraction",
    ),
)
_JS_STRING_LITERAL_RE = re.compile(
    r"""'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|`(?:\\.|[^`\\])*`""",
    re.S,
)
_SENSITIVE_BROWSER_EVAL_TOKENS: tuple[tuple[str, str], ...] = (
    ("cookie", "document.cookie"),
    ("localStorage", "web storage"),
    ("sessionStorage", "web storage"),
    ("indexedDB", "IndexedDB"),
    ("caches", "Cache Storage"),
    ("clipboard", "navigator sensitive API"),
    ("credentials", "navigator sensitive API"),
    ("serviceWorker", "navigator sensitive API"),
    ("fetch", "network request"),
    ("XMLHttpRequest", "network request"),
    ("WebSocket", "network request"),
    ("EventSource", "network request"),
    ("sendBeacon", "network beacon"),
)


async def _allow_unsafe_browser_evaluate() -> bool:
    """Return whether sensitive browser JS evaluation is explicitly allowed.

    When true, ``browser_console(expression=...)`` runs without the
    sensitive-primitive denylist even if ``browser.restrict_evaluate`` is set.
    """
    try:
        cfg = await load_config_readonly()
        return is_truthy_value(
            cfg_get(cfg, "browser", "allow_unsafe_evaluate"), default=False
        )
    except Exception as e:
        logger.debug("Could not read browser.allow_unsafe_evaluate from config: %s", e)
        return False


async def _restrict_browser_evaluate() -> bool:
    """Return whether the sensitive-primitive eval denylist is enabled.

    Off by default. ``browser_console(expression=...)`` is the agent's only
    programmatic page-inspection path, and the denylist blocks the *names* of
    common primitives (``fetch``, ``cookie``, ``querySelector(...input...)``)
    rather than any actual exfiltration — which also blocks a large class of
    legitimate DOM extraction (any selector or page script text containing
    those words). Egress itself is still gated by the SSRF/private-URL guards
    in ``_browser_eval`` regardless of this setting. Users who want the
    strict vocabulary denylist (e.g. when browsing hostile pages with a
    logged-in profile) opt in with ``browser.restrict_evaluate: true``;
    ``browser.allow_unsafe_evaluate: true`` overrides it back off.
    """
    try:
        cfg = await load_config_readonly()
        return is_truthy_value(
            cfg_get(cfg, "browser", "restrict_evaluate"), default=False
        )
    except Exception as e:
        logger.debug("Could not read browser.restrict_evaluate from config: %s", e)
        return False


def _decode_js_string_literal(literal: str) -> str:
    """Best-effort decode of a JavaScript string literal for policy checks.

    This is not a JS parser.  It only normalizes common escaped property names
    such as ``document["co\\x6fkie"]`` before the fail-closed sensitive-token
    check below.
    """
    if len(literal) < 2:
        return literal
    body = literal[1:-1]
    try:
        return bytes(body, "utf-8").decode("unicode_escape")
    except Exception:
        return body


def _decoded_js_string_literals(expression: str) -> list[str]:
    return [
        _decode_js_string_literal(match.group(0))
        for match in _JS_STRING_LITERAL_RE.finditer(expression)
    ]


def _sensitive_browser_eval_token_reason(expression: str) -> Optional[str]:
    """Return a risk reason for direct or quoted sensitive browser primitives.

    ``browser_console(expression=...)`` executes in the page origin.  A denylist
    that only searches direct spellings like ``document.cookie`` and ``fetch(``
    misses equivalent JavaScript property access such as ``document["cookie"]``
    or ``globalThis["fetch"](...)``.  Treat sensitive primitive names as risky
    whether they appear as identifiers or decoded string-literal property names.
    Concatenating all string literals catches simple obfuscations like
    ``document["coo" + "kie"]`` while the config opt-in preserves the escape
    hatch for trusted pages.
    """
    string_literals = _decoded_js_string_literals(expression)
    concatenated_literals = "".join(string_literals).lower()
    for token, reason in _SENSITIVE_BROWSER_EVAL_TOKENS:
        if re.search(rf"\b{re.escape(token)}\b", expression, re.I):
            return reason
        token_lower = token.lower()
        if any(token_lower in literal.lower() for literal in string_literals):
            return reason
        if token_lower in concatenated_literals:
            return reason
    return None


def _risky_browser_eval_reason(expression: str) -> Optional[str]:
    """Return a human-readable reason if a JS expression uses risky primitives."""
    if not expression:
        return None
    for pattern, reason in _RISKY_BROWSER_EVAL_PATTERNS:
        if pattern.search(expression):
            return reason
    return _sensitive_browser_eval_token_reason(expression)


async def _enforce_browser_eval_policy(expression: str) -> Optional[str]:
    """Block sensitive browser JS evaluation when the opt-in denylist is on.

    The denylist is opt-in (``browser.restrict_evaluate: true``) because it
    gates on primitive *names*, which cripples legitimate DOM extraction —
    see ``_restrict_browser_evaluate``. Network egress to private/internal
    addresses is enforced separately in ``_browser_eval`` and does not depend
    on this policy.
    """
    if not await _restrict_browser_evaluate():
        return None
    if await _allow_unsafe_browser_evaluate():
        return None
    reason = _risky_browser_eval_reason(expression)
    if not reason:
        return None
    return (
        "Blocked: browser_console(expression=...) tried to use sensitive browser "
        f"JavaScript primitive ({reason}) while browser.restrict_evaluate is "
        "enabled. Use browser_snapshot/browser_get_images/browser_console "
        "without expression for normal inspection, or set "
        "browser.restrict_evaluate: false in config.yaml to allow "
        "programmatic evaluation."
    )


async def _browser_eval(expression: str, task_id: Optional[str] = None) -> str:
    """Evaluate a JavaScript expression in the page context and return the result."""
    effective_task_id = _last_session_key(task_id or "default")

    if await _eval_ssrf_guard_active(effective_task_id):
        blocked_literal = await _expression_targets_private_url(expression)
        if blocked_literal:
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        "Blocked: JavaScript expression targets a private or "
                        f"internal address ({blocked_literal}). Reading internal "
                        "endpoints via browser_console is not permitted in this "
                        "browser mode."
                    ),
                },
                ensure_ascii=False,
            )

    # Camofox keeps its own raw-``task_id``-keyed session map, so pass the raw
    # id (matching every other Camofox tool) rather than the resolved
    # agent-browser session key.  The literal pre-scan above already ran.
    if await _is_camofox_mode():
        return await _camofox_eval(expression, task_id)

    # ── Private-network guard (eval return-value path) ──────────────────────
    # The literal pre-scan above closes the direct-fetch sub-path
    # (`fetch('http://127.0.0.1/secret')`).  The post-eval page-URL recheck
    # below closes the navigate-then-read sub-path (`location.href = '...'`
    # then read the DOM) — eval returns arbitrary JS results directly, never
    # touching snapshot/vision, so both sub-paths gate on the same condition.

    # --- Fast path: route through the supervisor's persistent CDP WS ---------
    # When a CDPSupervisor is alive for this task_id, ``Runtime.evaluate`` runs
    # on the already-connected WebSocket — zero subprocess startup cost vs
    # spawning an ``agent-browser eval`` CLI process.  Falls through to the
    # subprocess path on any error so behaviour is unchanged when no
    # supervisor is running (e.g. plain agent-browser without a CDP backend).
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY

        supervisor = SUPERVISOR_REGISTRY.get(effective_task_id)
        if supervisor is not None:
            sup_result = await supervisor.evaluate_runtime(expression)
            if sup_result.get("ok"):
                raw_result = sup_result.get("result")
                # Match the agent-browser path: if the value is a JSON string,
                # parse it so the model gets structured data.
                parsed = raw_result
                if isinstance(raw_result, str):
                    try:
                        parsed = json.loads(raw_result)
                    except (json.JSONDecodeError, ValueError):
                        pass  # keep as string
                # Post-eval page-URL recheck: if this (or a prior) eval
                # navigated the page to a private address, withhold the result.
                if await _eval_ssrf_guard_active(effective_task_id):
                    _blocked_url = await _current_page_private_url(effective_task_id)
                    if _blocked_url:
                        return json.dumps(
                            {
                                "success": False,
                                "error": (
                                    "Blocked: page URL targets a private or internal "
                                    f"address ({_blocked_url}). This may have been "
                                    "caused by a JavaScript navigation via "
                                    "browser_console."
                                ),
                            },
                            ensure_ascii=False,
                        )
                response = {
                    "success": True,
                    "result": _redact_browser_output(parsed),
                    "result_type": type(parsed).__name__,
                    "method": "cdp_supervisor",
                }
                return json.dumps(response, ensure_ascii=False, default=str)
            # JS exception is a real failure — surface it instead of falling
            # through to the subprocess path (which would just re-run and
            # produce the same exception, but slower).
            err = sup_result.get("error") or "evaluate_runtime failed"
            if "supervisor" not in err.lower():
                # Real JS-side error — return it.
                return json.dumps({"success": False, "error": err}, ensure_ascii=False)
            # Supervisor-side failure (loop down, no session) — fall through.
            logger.debug(
                "browser_eval: supervisor path unavailable (%s), falling back to subprocess",
                err,
            )
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("browser_eval: supervisor path errored (%s), falling back", exc)

    # --- Fallback: agent-browser CLI subprocess (original path) -------------
    result = await _run_browser_command(effective_task_id, "eval", [expression])

    if not result.get("success"):
        err = result.get("error", "eval failed")
        # Detect backend capability gaps and give the model a clear signal
        if any(
            hint in err.lower()
            for hint in (
                "unknown command",
                "not supported",
                "not found",
                "no such command",
            )
        ):
            response = {
                "success": False,
                "error": f"JavaScript evaluation is not supported by this browser backend. {err}",
            }
            return json.dumps(_copy_fallback_warning(response, result))
        # A live DOM node / NodeList / Window can't be JSON-serialized by CDP
        # and fails the eval with "Object reference chain is too long".  The
        # supervisor fast path retries with returnByValue=false, but the CLI
        # subprocess can't, so turn the cryptic protocol error into actionable
        # guidance instead of surfacing it raw.
        if "reference chain is too long" in err.lower():
            response = {
                "success": False,
                "error": (
                    "Expression returned a live DOM node / NodeList / Window, "
                    "which can't be serialized. Extract a primitive value "
                    "(e.g. .innerText, .href, .src, .value) or use "
                    "JSON.stringify() / a snapshot tool instead."
                ),
            }
            return json.dumps(_copy_fallback_warning(response, result))
        response = {
            "success": False,
            "error": err,
        }
        return json.dumps(_copy_fallback_warning(response, result))

    data = result.get("data", {})
    raw_result = data.get("result")

    # The eval command returns the JS result as a string.  If the string
    # is valid JSON, parse it so the model gets structured data.
    parsed = raw_result
    if isinstance(raw_result, str):
        try:
            parsed = json.loads(raw_result)
        except (json.JSONDecodeError, ValueError):
            pass  # keep as string

    response = {
        "success": True,
        "result": _redact_browser_output(parsed),
        "result_type": type(parsed).__name__,
    }
    # Post-eval page-URL recheck: if this (or a prior) eval navigated the page
    # to a private address, withhold the result (mirrors the supervisor path).
    if await _eval_ssrf_guard_active(effective_task_id):
        _blocked_url = await _current_page_private_url(effective_task_id)
        if _blocked_url:
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        "Blocked: page URL targets a private or internal address "
                        f"({_blocked_url}). This may have been caused by a "
                        "JavaScript navigation via browser_console."
                    ),
                },
                ensure_ascii=False,
            )
    return json.dumps(
        _copy_fallback_warning(response, result), ensure_ascii=False, default=str
    )


async def _camofox_current_page_private_url(tab_id: str, user_id: str) -> Optional[str]:
    """Return the Camofox page URL when it targets a private/internal address.

    Camofox analogue of ``_current_page_private_url`` (evaluate endpoint instead
    of the agent-browser CLI).  Returns ``None`` when the page is public, the URL
    can't be determined, or the probe errors (fail-open on probe failure,
    matching the snapshot/vision guards — do not change to fail-closed without
    also changing the sibling).
    """
    try:
        from tools.browser_camofox import _post

        data = await _post(
            f"/tabs/{tab_id}/evaluate",
            body={"expression": "window.location.href", "userId": user_id},
        )
        current_url = str(data.get("result") if isinstance(data, dict) else data or "")
        current_url = current_url.strip().strip('"').strip("'")
        if current_url and (
            await _is_always_blocked_url(current_url)
            or not await _is_safe_url(current_url)
        ):
            return current_url
    except Exception as exc:
        logger.debug("_camofox_current_page_private_url: probe failed (%s)", exc)
    return None


async def _camofox_eval(expression: str, task_id: Optional[str] = None) -> str:
    """Evaluate JS via Camofox's /tabs/{tab_id}/evaluate endpoint (if available)."""
    from tools.browser_camofox import _ensure_tab, _post

    try:
        tab_info = await _ensure_tab(task_id or "default")
        tab_id = tab_info.get("tab_id") or tab_info.get("id")
        user_id = tab_info["user_id"]
        if not isinstance(tab_id, str) or not tab_id:
            return tool_error("Camofox did not return a tab ID", success=False)
        resp = await _post(
            f"/tabs/{tab_id}/evaluate",
            body={"expression": expression, "userId": user_id},
        )

        # Camofox returns the result in a JSON envelope
        raw_result = resp.get("result") if isinstance(resp, dict) else resp
        parsed = raw_result
        if isinstance(raw_result, str):
            try:
                parsed = json.loads(raw_result)
            except (json.JSONDecodeError, ValueError):
                pass

        if await _eval_ssrf_guard_active(task_id or "default"):
            _blocked_url = await _camofox_current_page_private_url(tab_id, user_id)
            if _blocked_url:
                return json.dumps(
                    {
                        "success": False,
                        "error": (
                            "Blocked: page URL targets a private or internal address "
                            f"({_blocked_url}). This may have been caused by a "
                            "JavaScript navigation via browser_console."
                        ),
                    },
                    ensure_ascii=False,
                )

        return json.dumps(
            {
                "success": True,
                "result": _redact_browser_output(parsed),
                "result_type": type(parsed).__name__,
            },
            ensure_ascii=False,
            default=str,
        )
    except Exception as e:
        error_msg = str(e)
        # Graceful degradation — server may not support eval
        if any(code in error_msg for code in ("404", "405", "501")):
            return json.dumps({
                "success": False,
                "error": "JavaScript evaluation is not supported by this Camofox server. "
                "Use browser_snapshot or browser_vision to inspect page state.",
            })
        return tool_error(error_msg, success=False)


async def _maybe_start_recording(task_id: str) -> None:
    """Start recording if browser.record_sessions is enabled in config."""
    async with _recording_lock:
        if task_id in _recording_sessions:
            return
        try:
            hermes_home = get_hermes_home()
            cfg = await load_config_readonly()
            record_enabled = cfg_get(
                cfg, "browser", "record_sessions", default=False
            )

            if not record_enabled:
                return

            recordings_dir = hermes_home / "browser_recordings"
            await aiofiles.os.makedirs(recordings_dir, exist_ok=True)
            await _cleanup_old_recordings(max_age_hours=72)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            recording_path = (
                recordings_dir / f"session_{timestamp}_{task_id[:16]}.webm"
            )

            result = await _run_browser_command(
                task_id, "record", ["start", str(recording_path)]
            )
            if result.get("success"):
                _recording_sessions.add(task_id)
                logger.info(
                    "Auto-recording browser session %s to %s",
                    task_id,
                    recording_path,
                )
            else:
                logger.debug(
                    "Could not start auto-recording: %s", result.get("error")
                )
        except Exception as e:
            logger.debug("Auto-recording setup failed: %s", e)


async def _maybe_stop_recording(task_id: str) -> None:
    """Stop recording if one is active for this session."""
    async with _recording_lock:
        if task_id not in _recording_sessions:
            return
        try:
            result = await _run_browser_command(task_id, "record", ["stop"])
            if result.get("success"):
                path = result.get("data", {}).get("path", "")
                logger.info(
                    "Saved browser recording for session %s: %s", task_id, path
                )
        except Exception as e:
            logger.debug("Could not stop recording for %s: %s", task_id, e)
        finally:
            _recording_sessions.discard(task_id)


async def browser_get_images(task_id: Optional[str] = None) -> str:
    """
    Get all images on the current page.

    Args:
        task_id: Task identifier for session isolation

    Returns:
        JSON string with list of images (src and alt)
    """
    if await _is_camofox_mode():
        from tools.browser_camofox import camofox_get_images

        return await camofox_get_images(task_id)

    effective_task_id = _last_session_key(task_id or "default")

    # Use eval to run JavaScript that extracts images
    js_code = """JSON.stringify(
        [...document.images].map(img => ({
            src: img.src,
            alt: img.alt || '',
            width: img.naturalWidth,
            height: img.naturalHeight
        })).filter(img => img.src && !img.src.startsWith('data:'))
    )"""

    result = await _run_browser_command(effective_task_id, "eval", [js_code])

    if result.get("success"):
        # ── Private-network guard (sibling of snapshot/vision/eval guards) ──
        if await _eval_ssrf_guard_active(effective_task_id):
            _blocked_url = await _current_page_private_url(effective_task_id)
            if _blocked_url:
                return json.dumps(
                    {
                        "success": False,
                        "error": (
                            "Blocked: page URL targets a private or internal address "
                            f"({_blocked_url}). This may have been caused by a "
                            "JavaScript navigation via browser_console."
                        ),
                    },
                    ensure_ascii=False,
                )

        data = result.get("data", {})
        raw_result = data.get("result", "[]")

        try:
            # Parse the JSON string returned by JavaScript
            if isinstance(raw_result, str):
                images = json.loads(raw_result)
            else:
                images = raw_result

            response = {
                "success": True,
                "images": _redact_browser_output(images),
                "count": len(images),
            }
            return json.dumps(
                _copy_fallback_warning(response, result), ensure_ascii=False
            )
        except json.JSONDecodeError:
            response = {
                "success": True,
                "images": [],
                "count": 0,
                "warning": "Could not parse image data",
            }
            return json.dumps(
                _copy_fallback_warning(response, result), ensure_ascii=False
            )
    else:
        response = {
            "success": False,
            "error": result.get("error", "Failed to get images"),
        }
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)


async def browser_vision(
    question: str,
    annotate: bool = False,
    task_id: Optional[str] = None,
) -> Union[str, Dict[str, Any]]:
    """
    Take a screenshot of the current page for visual inspection.

    Captures what's visually displayed in the browser. When the active model
    supports native vision, the screenshot is attached directly to the
    conversation so the model can inspect it on the next turn; otherwise Hermes
    falls back to the auxiliary vision model and returns a text analysis. Useful
    for visual content the text-based snapshot may not capture (CAPTCHAs,
    verification challenges, images, complex layouts, etc.).

    The screenshot is saved persistently and its file path is returned so it
    can be shared with users via MEDIA:<path> in the response.

    Args:
        question: What you want to know about the page visually
        annotate: If True, overlay numbered [N] labels on interactive elements
        task_id: Task identifier for session isolation

    Returns:
        A JSON string with vision analysis results and screenshot_path, or a
        multimodal tool-result envelope carrying the screenshot and metadata.
    """
    if await _is_camofox_mode():
        from tools.browser_camofox import camofox_vision

        return await camofox_vision(question, annotate, task_id)

    import base64
    import uuid as uuid_mod
    from hermes_constants import get_hermes_dir

    screenshots_dir = await get_hermes_dir("cache/screenshots", "browser_screenshots")
    screenshot_path = screenshots_dir / f"browser_screenshot_{uuid_mod.uuid4().hex}.png"
    effective_task_id = _last_session_key(task_id or "default")
    result: Dict[str, Any] = {}

    # ── Private-network guard: block vision from eval-navigated private pages ──
    # After any eval (browser_console) that may have changed location.href to a
    # private/internal address, the screenshot would expose private page content
    # to the vision model.  Re-check the current URL before capturing anything.
    if (
        not await _is_local_backend()
        and not _is_local_sidecar_key(effective_task_id)
        and not await _allow_private_urls()
    ):
        try:
            _url_result = await _run_browser_command(
                effective_task_id,
                "eval",
                ["window.location.href"],
                timeout=5,
                _engine_override="auto",
            )
            if _url_result.get("success"):
                _current_url = (
                    _url_result
                    .get("data", {})
                    .get("result", "")
                    .strip()
                    .strip('"')
                    .strip("'")
                )
                if _current_url and not await _is_safe_url(_current_url):
                    return json.dumps(
                        {
                            "success": False,
                            "error": (
                                "Blocked: page URL targets a private or internal address "
                                f"({_current_url}). This may have been caused by a "
                                "JavaScript navigation via browser_console."
                            ),
                        },
                        ensure_ascii=False,
                    )
        except Exception as _url_exc:
            logger.debug("browser_vision: URL safety check failed (%s)", _url_exc)

    # Lightpanda has no graphical renderer — pre-route screenshots to Chrome
    # via the fallback helper instead of letting the normal path fail with a
    # CDP error or return a placeholder PNG.  The normal analysis path below
    # still owns base64 encoding, provider routing, resizing retry, redaction,
    # and response shape.
    engine = await _get_browser_engine()
    _lp_prerouted = False
    _lp_fallback_warning = None
    if engine == "lightpanda" and await _should_inject_engine(engine):
        logger.debug(
            "browser_vision: pre-routing screenshot to Chrome (engine=lightpanda)"
        )
        screenshot_args = []
        if annotate:
            screenshot_args.append("--annotate")
        fb_result = await _chrome_fallback_screenshot(
            effective_task_id,
            screenshot_args,
            await _get_command_timeout(),
        )
        fb_reason = "Lightpanda has no graphical renderer for screenshots; used Chrome for vision capture."
        fb_result = _annotate_lightpanda_fallback(fb_result, fb_reason)
        if fb_result.get("success"):
            _lp_prerouted = True
            _lp_fallback_warning = fb_result.get("fallback_warning")
            fb_path = fb_result.get("data", {}).get("path", "")
            if fb_path and await aiofiles.os.path.exists(fb_path):
                from hermes_constants import get_hermes_dir

                screenshots_dir = await get_hermes_dir(
                    "cache/screenshots", "browser_screenshots"
                )
                await aiofiles.os.makedirs(screenshots_dir, exist_ok=True)
                persistent_path = (
                    screenshots_dir / f"browser_screenshot_{uuid_mod.uuid4().hex}.png"
                )
                async with aiofiles.open(fb_path, "rb") as source:
                    async with aiofiles.open(persistent_path, "wb") as target:
                        await target.write(await source.read())
                screenshot_path = persistent_path
        else:
            logger.warning(
                "Lightpanda Chrome fallback vision screenshot failed: %s",
                fb_result.get("error"),
            )
            # Fall through to the normal screenshot path so _run_browser_command
            # can still produce the standard fallback metadata/error.
            _lp_prerouted = False

    try:
        await aiofiles.os.makedirs(screenshots_dir, exist_ok=True)

        # Prune old screenshots (older than 24 hours) to prevent unbounded disk growth
        await _cleanup_old_screenshots(screenshots_dir, max_age_hours=24)

        if _lp_prerouted and await aiofiles.os.path.exists(screenshot_path):
            result = {
                "success": True,
                "data": {
                    "path": str(screenshot_path),
                    "fallback_warning": _lp_fallback_warning,
                    "browser_engine": "chrome",
                    "browser_engine_fallback": {
                        "from": "lightpanda",
                        "to": "chrome",
                        "reason": "Lightpanda has no graphical renderer for screenshots; used Chrome for vision capture.",
                    },
                },
                "fallback_warning": _lp_fallback_warning,
                "browser_engine": "chrome",
                "browser_engine_fallback": {
                    "from": "lightpanda",
                    "to": "chrome",
                    "reason": "Lightpanda has no graphical renderer for screenshots; used Chrome for vision capture.",
                },
            }
        else:
            # Take screenshot using agent-browser
            screenshot_args = []
            if annotate:
                screenshot_args.append("--annotate")
            screenshot_args.append("--full")
            screenshot_args.append(str(screenshot_path))
            result = await _run_browser_command(
                effective_task_id,
                "screenshot",
                screenshot_args,
                # If the Lightpanda pre-route already failed, force Chrome so
                # _run_browser_command doesn't trigger a redundant LP fallback.
                _engine_override="auto" if _lp_prerouted else None,
            )

        if not result.get("success"):
            error_detail = result.get("error", "Unknown error")
            _cp = await _get_cloud_provider()
            mode = "local" if _cp is None else f"cloud ({_cp.name})"
            error_response = {
                "success": False,
                "error": f"Failed to take screenshot ({mode} mode): {error_detail}",
            }
            return json.dumps(
                _copy_fallback_warning(error_response, result), ensure_ascii=False
            )

        actual_screenshot_path = result.get("data", {}).get("path")
        if actual_screenshot_path:
            screenshot_path = Path(actual_screenshot_path)

        # Check if screenshot file was created
        if not await aiofiles.os.path.exists(screenshot_path):
            _cp = await _get_cloud_provider()
            mode = "local" if _cp is None else f"cloud ({_cp.name})"
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        f"Screenshot file was not created at {screenshot_path} ({mode} mode). "
                        f"This may indicate a socket path issue (macOS /var/folders/), "
                        f"a missing Chromium install ('agent-browser install'), "
                        f"or a stale daemon process."
                    ),
                },
                ensure_ascii=False,
            )

        # Convert screenshot to base64 at full resolution.
        async with aiofiles.open(screenshot_path, "rb") as handle:
            _screenshot_bytes = await handle.read()
        _screenshot_b64 = base64.b64encode(_screenshot_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{_screenshot_b64}"

        # Fast path: when native image routing is in effect for the active main
        # model, attach the screenshot directly instead of describing it through
        # an auxiliary vision LLM. The model inspects the pixels on its next
        # turn — no aux call, no information loss. Consistent with vision_analyze.
        from tools.vision_tools import (
            _build_native_vision_tool_result,
            _should_use_native_vision_fast_path,
        )

        if await _should_use_native_vision_fast_path():
            native_result = _build_native_vision_tool_result(
                image_url=str(screenshot_path),
                question=question,
                image_data_url=data_url,
                image_size_bytes=len(_screenshot_bytes),
            )
            meta = native_result.setdefault("meta", {})
            meta["screenshot_path"] = str(screenshot_path)
            if _lp_fallback_warning:
                meta["fallback_warning"] = _lp_fallback_warning
            if annotate and result.get("data", {}).get("annotations"):
                meta["annotations"] = result["data"]["annotations"]
            native_result["text_summary"] = (
                f"{native_result.get('text_summary', '')} "
                f"Screenshot path: {screenshot_path}"
            ).strip()
            return native_result

        vision_prompt = (
            f"You are analyzing a screenshot of a web browser.\n\n"
            f"User's question: {question}\n\n"
            f"Provide a detailed and helpful answer based on what you see in the screenshot. "
            f"If there are interactive elements, describe them. If there are verification challenges "
            f"or CAPTCHAs, describe what type they are and what action might be needed. "
            f"Focus on answering the user's specific question."
        )

        # Use the centralized LLM router
        vision_model = _get_vision_model()
        logger.debug(
            "browser_vision: analysing screenshot (%d bytes)", len(_screenshot_bytes)
        )

        # Read vision timeout/temperature from config (auxiliary.vision.*).
        # Local vision models (llama.cpp, ollama) can take well over 30s for
        # screenshot analysis, so the default timeout must be generous.
        vision_timeout = 120.0
        vision_temperature = 0.1
        try:
            _cfg = await load_config_readonly()
            _vision_cfg = cfg_get(_cfg, "auxiliary", "vision", default={})
            _vt = _vision_cfg.get("timeout")
            if _vt is not None:
                vision_timeout = float(_vt)
            _vtemp = _vision_cfg.get("temperature")
            if _vtemp is not None:
                vision_temperature = float(_vtemp)
        except Exception:
            pass

        image_part: Dict[str, Any] = {
            "type": "image_url",
            "image_url": {"url": data_url},
        }
        call_kwargs: Dict[str, Any] = {
            "task": "vision",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_prompt},
                        image_part,
                    ],
                }
            ],
            "max_tokens": 2000,
            "temperature": vision_temperature,
            "timeout": vision_timeout,
        }
        if vision_model:
            call_kwargs["model"] = vision_model
        # Try full-size screenshot; on size-related rejection, downscale and retry.
        try:
            response = await _lazy_call_llm(**call_kwargs)
        except Exception as _api_err:
            from tools.vision_tools import (
                _is_image_size_error,
                _resize_image_for_vision,
                _RESIZE_TARGET_BYTES,
            )

            if _is_image_size_error(_api_err) and len(data_url) > _RESIZE_TARGET_BYTES:
                logger.info(
                    "Vision API rejected screenshot (%.1f MB); "
                    "auto-resizing to ~%.0f MB and retrying...",
                    len(data_url) / (1024 * 1024),
                    _RESIZE_TARGET_BYTES / (1024 * 1024),
                )
                # A new caller cancellation must supersede the rejected image error.
                data_url = await _resize_image_for_vision(  # noqa: ASYNC120
                    screenshot_path, mime_type="image/png"
                )
                image_part["image_url"]["url"] = data_url
                response = await _lazy_call_llm(**call_kwargs)  # noqa: ASYNC120
            else:
                raise

        analysis = (response.choices[0].message.content or "").strip()
        # Redact secrets the vision LLM may have read from the screenshot.
        from agent.redact import redact_sensitive_text

        analysis = redact_sensitive_text(analysis)
        response_data: Dict[str, Any] = {
            "success": True,
            "analysis": analysis or "Vision analysis returned no content.",
            "screenshot_path": str(screenshot_path),
        }
        _copy_fallback_warning(response_data, result)
        # Include annotation data if annotated screenshot was taken
        if annotate and result.get("data", {}).get("annotations"):
            response_data["annotations"] = result["data"]["annotations"]
        return json.dumps(response_data, ensure_ascii=False)

    except Exception as e:
        # Keep the screenshot if it was captured successfully — the failure is
        # in the LLM vision analysis, not the capture.  Deleting a valid
        # screenshot loses evidence the user might need.  The 24-hour cleanup
        # in _cleanup_old_screenshots prevents unbounded disk growth.
        logger.warning("browser_vision failed: %s", e, exc_info=True)
        error_info = {
            "success": False,
            "error": f"Error during vision analysis: {str(e)}",
        }
        if await aiofiles.os.path.exists(screenshot_path):
            error_info["screenshot_path"] = str(screenshot_path)
            error_info["note"] = (
                "Screenshot was captured but vision analysis failed. You can still share it via MEDIA:<path>."
            )
        _copy_fallback_warning(error_info, result)
        return json.dumps(error_info, ensure_ascii=False)


async def _cleanup_old_screenshots(screenshots_dir, max_age_hours=24) -> None:
    """Remove browser screenshots older than max_age_hours to prevent disk bloat.

    Throttled to run at most once per hour per directory to avoid repeated
    scans on screenshot-heavy workflows.
    """
    key = str(screenshots_dir)
    now = time.time()
    if now - _last_screenshot_cleanup_by_dir.get(key, 0.0) < 3600:
        return
    _last_screenshot_cleanup_by_dir[key] = now

    try:
        cutoff = time.time() - (max_age_hours * 3600)
        entries = await aiofiles.os.listdir(screenshots_dir)
        for name in entries:
            if not name.startswith("browser_screenshot_") or not name.endswith(".png"):
                continue
            f = screenshots_dir / name
            try:
                if (await aiofiles.os.stat(f)).st_mtime < cutoff:
                    await aiofiles.os.remove(f)
            except Exception as e:
                logger.debug("Failed to clean old screenshot %s: %s", f, e)
    except Exception as e:
        logger.debug("Screenshot cleanup error (non-critical): %s", e)


async def _cleanup_old_recordings(max_age_hours=72) -> None:
    """Remove browser recordings older than max_age_hours to prevent disk bloat."""
    try:
        hermes_home = get_hermes_home()
        recordings_dir = hermes_home / "browser_recordings"
        if not await aiofiles.os.path.exists(recordings_dir):
            return
        cutoff = time.time() - (max_age_hours * 3600)
        entries = await aiofiles.os.listdir(recordings_dir)
        for name in entries:
            if not name.startswith("session_") or not name.endswith(".webm"):
                continue
            f = recordings_dir / name
            try:
                if (await aiofiles.os.stat(f)).st_mtime < cutoff:
                    await aiofiles.os.remove(f)
            except Exception as e:
                logger.debug("Failed to clean old recording %s: %s", f, e)
    except Exception as e:
        logger.debug("Recording cleanup error (non-critical): %s", e)


# ============================================================================
# Cleanup and Management Functions
# ============================================================================


async def cleanup_browser(task_id: Optional[str] = None) -> None:
    """
    Clean up browser session(s) for a task.

    Called automatically when a task completes or when inactivity timeout is reached.
    Closes both the agent-browser/Browserbase session and Camofox sessions.

    When ``task_id`` is a bare task identifier (no ``::local`` suffix), reaps
    BOTH the cloud/primary session AND any hybrid-routing local sidecar that
    may have been spawned for LAN/localhost URLs in the same task.  When
    ``task_id`` already carries a ``::local`` suffix (called from the inactivity
    cleanup loop against a specific session key), reaps only that one.

    Args:
        task_id: Task identifier (or explicit session key)
    """
    if task_id is None:
        task_id = "default"

    # Expand to the full set of session keys to reap. For a bare task_id
    # that includes the cloud/primary key + the local sidecar if one exists.
    if _is_local_sidecar_key(task_id):
        session_keys = [task_id]
        bare_task_id = task_id[: -len(_LOCAL_SUFFIX)]
    else:
        session_keys = [task_id]
        sidecar_key = f"{task_id}{_LOCAL_SUFFIX}"
        async with _cleanup_lock:
            if sidecar_key in _active_sessions:
                session_keys.append(sidecar_key)
        bare_task_id = task_id

    for session_key in session_keys:
        await _cleanup_single_browser_session(session_key)

    # Drop stale last-active ownership. Cleaning a bare task drops its binding;
    # cleaning a sidecar drops the binding only if that sidecar was still the
    # recorded owner. This prevents a later click/snapshot from resurrecting a
    # cleaned sidecar on about:blank while preserving a primary-session binding.
    async with _cleanup_lock:
        if _is_local_sidecar_key(task_id):
            if _last_active_session_key.get(bare_task_id) == task_id:
                _last_active_session_key.pop(bare_task_id, None)
        else:
            _last_active_session_key.pop(bare_task_id, None)
        no_active_sessions = not _active_sessions

    if no_active_sessions:
        await _stop_browser_cleanup_thread()


async def _cleanup_single_browser_session(task_id: str) -> None:
    """Internal: reap a single browser session by its exact session key."""
    # Stop the CDP supervisor for this task FIRST so we close our WebSocket
    # before the backend tears down the underlying CDP endpoint.
    await _stop_cdp_supervisor(task_id)

    # Also clean up Camofox session if running in Camofox mode.
    # Skip full close when managed persistence is enabled — the browser
    # profile (and its session cookies) must survive across agent tasks.
    # The inactivity reaper still frees idle resources.
    if await _is_camofox_mode():
        try:
            from tools.browser_camofox import camofox_close, camofox_soft_cleanup

            if not await camofox_soft_cleanup(task_id):
                await camofox_close(task_id)
        except Exception as e:
            logger.debug("Camofox cleanup for task %s: %s", task_id, e)

    logger.debug("cleanup_browser called for task_id: %s", task_id)
    async with _cleanup_lock:
        logger.debug("Active sessions: %s", list(_active_sessions.keys()))
        # Check if session exists, but don't remove yet -
        # _run_browser_command needs it to build the close command.
        session_info = _active_sessions.get(task_id)

    if session_info:
        bb_session_id = session_info.get("bb_session_id", "unknown")
        logger.debug(
            "Found session for task %s: bb_session_id=%s", task_id, bb_session_id
        )

        # Stop auto-recording before closing (saves the file)
        await _maybe_stop_recording(task_id)

        # An expired cloud CDP URL cannot accept an agent-browser close command.
        # Avoid feeding it back through _get_session_info(), which would try to
        # renew the session recursively while cleanup is still in progress.
        if _session_has_expired(session_info):
            logger.debug(
                "Skipping agent-browser close for expired session %s",
                task_id,
            )
        else:
            try:
                await _run_browser_command(task_id, "close", [], timeout=10)
                logger.debug(
                    "agent-browser close command completed for task %s",
                    task_id,
                )
            except Exception as e:
                logger.warning("agent-browser close failed for task %s: %s", task_id, e)

        async with _cleanup_lock:
            _active_sessions.pop(task_id, None)
            _session_last_activity.pop(task_id, None)

        # Cloud mode: close the cloud browser session via provider API.
        # Local sidecars have bb_session_id=None so this no-ops for them.
        if bb_session_id:
            provider = await _get_cloud_provider()
            if provider is not None:
                try:
                    await provider.close_session(bb_session_id)
                except Exception as e:
                    logger.warning("Could not close cloud browser session: %s", e)

        # Kill the daemon process and clean up socket directory
        session_name = session_info.get("session_name", "")
        if session_name:
            socket_dir = os.path.join(
                _socket_safe_tmpdir(), f"agent-browser-{session_name}"
            )
            if await aiofiles.os.path.exists(socket_dir):
                # agent-browser writes {session}.pid in the socket dir
                pid_file = os.path.join(socket_dir, f"{session_name}.pid")
                if await aiofiles.os.path.isfile(pid_file):
                    try:
                        async with aiofiles.open(pid_file, encoding="utf-8") as handle:
                            daemon_pid = int((await handle.read()).strip())
                        await _terminate_host_pid(daemon_pid)
                        logger.debug(
                            "Killed daemon pid %s for %s", daemon_pid, session_name
                        )
                    except (ProcessLookupError, ValueError, PermissionError, OSError):
                        logger.debug(
                            "Could not kill daemon pid for %s (already dead or inaccessible)",
                            session_name,
                        )
                await _remove_tree(socket_dir)

        logger.debug("Removed task %s from active sessions", task_id)
    else:
        logger.debug("No active session found for task_id: %s", task_id)


async def cleanup_all_browsers() -> None:
    """
    Clean up all active browser sessions.

    Useful for cleanup on shutdown.
    """
    async with _cleanup_lock:
        task_ids = list(_active_sessions.keys())
    for task_id in task_ids:
        await cleanup_browser(task_id)

    # Tear down CDP supervisors for all tasks so background tasks exit.
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY

        await SUPERVISOR_REGISTRY.stop_all()
    except Exception:
        pass

    # Reset cached lookups so they are re-evaluated on next use.
    global _cached_agent_browser, _agent_browser_resolved
    global _cached_command_timeout, _command_timeout_resolved
    global _cached_chromium_installed
    global _cached_browser_engine, _browser_engine_resolved
    global _cached_homebrew_node_dirs
    _cached_agent_browser = None
    _agent_browser_resolved = False
    _cached_homebrew_node_dirs = None
    # Flip the resolved flag BEFORE nulling the cache so a concurrent
    # reader never sees ``resolved=True`` with ``cache=None`` (#14331).
    _command_timeout_resolved = False
    _cached_command_timeout = None
    _cached_chromium_installed = None
    global _chromium_autoinstall_attempted
    _chromium_autoinstall_attempted = False
    _cached_browser_engine = None
    _browser_engine_resolved = False
    await _stop_browser_cleanup_thread()


# ============================================================================
# Requirements Check
# ============================================================================


# Cache for Chromium discovery. Invalidated by _reset_browser_caches.
_cached_chromium_installed: Optional[bool] = None


def _chromium_search_roots() -> List[str]:
    """Directories to scan for a Chromium / headless-shell build.

    Order mirrors what agent-browser and Playwright actually probe:

    1. ``PLAYWRIGHT_BROWSERS_PATH`` when set (Docker image sets this to
       ``/opt/hermes/.playwright``).
    2. ``~/.cache/ms-playwright`` — Playwright's default on Linux/macOS.
    3. ``~/Library/Caches/ms-playwright`` — Playwright's default on macOS.
    4. ``%USERPROFILE%\\AppData\\Local\\ms-playwright`` — Playwright's default
       on Windows.
    """
    roots: List[str] = []
    env_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "").strip()
    if env_path and env_path != "0":
        roots.append(env_path)
    home = os.path.expanduser("~")
    roots.append(os.path.join(home, ".cache", "ms-playwright"))
    if sys.platform == "darwin":
        roots.append(os.path.join(home, "Library", "Caches", "ms-playwright"))
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
        roots.append(os.path.join(local, "ms-playwright"))
    return roots


async def _chromium_installed() -> bool:
    """Return True when a usable Chromium (or headless-shell) build is on disk.

    Checks, in order:

    1. ``AGENT_BROWSER_EXECUTABLE_PATH`` env var — the official way to point
       agent-browser at a pre-installed Chrome/Chromium.
    2. System Chrome/Chromium in PATH (``google-chrome``, ``chromium``,
       ``chromium-browser``, ``chrome``).
    3. Playwright's browser cache (current logic) — directories containing
       ``chromium-*`` or ``chromium_headless_shell-*``.

    agent-browser (0.26+) downloads Playwright's chromium / headless-shell
    builds into ``PLAYWRIGHT_BROWSERS_PATH`` and won't start without at least
    one of the three above being present.  Without a browser binary the CLI
    hangs on first use until the command timeout fires (often ~30s).  Guarding
    the tool behind this check prevents advertising a capability that will
    fail at runtime.
    """
    global _cached_chromium_installed
    if _cached_chromium_installed is not None:
        return _cached_chromium_installed

    # 1. AGENT_BROWSER_EXECUTABLE_PATH — explicit user-configured browser
    ab_path = os.environ.get("AGENT_BROWSER_EXECUTABLE_PATH", "").strip()
    which = aiofiles.os.wrap(shutil.which)
    if ab_path:
        if await aiofiles.os.path.isfile(ab_path) or await which(ab_path):
            _cached_chromium_installed = True
            return True

    # 2. System Chrome/Chromium in PATH (common names)
    system_chrome = (
        await which("google-chrome")
        or await which("chromium")
        or await which("chromium-browser")
        or await which("chrome")
    )
    if system_chrome:
        _cached_chromium_installed = True
        return True

    # 3. Playwright browser cache (legacy — chromium-* / chromium_headless_shell-* dirs)
    for root in _chromium_search_roots():
        if not root or not await aiofiles.os.path.isdir(root):
            continue
        try:
            entries = await aiofiles.os.listdir(root)
        except OSError:
            continue
        # Playwright names them ``chromium-<build>`` and
        # ``chromium_headless_shell-<build>``; agent-browser accepts either.
        for entry in entries:
            if entry.startswith("chromium-") or entry.startswith(
                "chromium_headless_shell-"
            ):
                _cached_chromium_installed = True
                return True

    _cached_chromium_installed = False
    return False


# One-shot per process: a 170MB download that fails (or is slow) must not be
# retried on every browser call. Reset by _reset_browser_caches() for tests.
_chromium_autoinstall_attempted = False


async def _maybe_autoinstall_chromium() -> bool:
    """Best-effort, gated download of the Chromium *binary* on local cold start.

    Closes the "the PR doesn't actually install the missing browser" gap for
    the common case — a Chromium binary that was simply never downloaded.
    Scope is deliberately narrow:

    - Binary only (``agent-browser install``), never ``--with-deps`` — that
      shells ``apt`` and needs root, so missing *system libraries* stay a user
      action (the timeout/blocked hints already point there).
    - Gated by ``security.allow_lazy_installs`` (same opt-out as every other
      lazy install) and skipped in Docker, where Chromium ships in the image.
    - Attempted once per process.

    Returns True only when Chromium is present afterwards.
    """
    global _chromium_autoinstall_attempted
    if _chromium_autoinstall_attempted:
        return await _chromium_installed()
    _chromium_autoinstall_attempted = True

    if await _running_in_docker():
        return False

    try:
        config = await load_config_readonly()
        allow_lazy_installs = bool(
            (config.get("security") or {}).get("allow_lazy_installs", True)
        )
    except Exception:
        allow_lazy_installs = True
    if not allow_lazy_installs or os.environ.get("HERMES_DISABLE_LAZY_INSTALLS") == "1":
        return False

    try:
        browser_cmd = await _find_agent_browser()
    except FileNotFoundError:
        return False

    if browser_cmd == "npx agent-browser":
        npx = await aiofiles.os.wrap(shutil.which)("npx") or "npx"
        install_cmd = [npx, "-y", "agent-browser", "install"]
    else:
        install_cmd = [browser_cmd, "install"]

    logger.info(
        "browser: Chromium missing — auto-installing the browser binary "
        "(one-time ~170MB; disable via security.allow_lazy_installs)"
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            *install_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_build_browser_env(),
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=600
        )
    except (OSError, TimeoutError) as e:
        if "proc" in locals() and proc.returncode is None:
            proc.kill()
            await proc.wait()
        logger.warning("browser: Chromium auto-install failed to start: %s", e)
        return False

    if proc.returncode != 0:
        tail = (
            (stderr_bytes or stdout_bytes or b"")
            .decode("utf-8", errors="replace")
            .strip()[-300:]
        )
        logger.warning(
            "browser: Chromium auto-install exited %s: %s", proc.returncode, tail
        )
        return False

    global _cached_chromium_installed
    _cached_chromium_installed = None
    return await _chromium_installed()


async def _running_in_docker() -> bool:
    """Best-effort detection of whether we're inside a Docker container."""
    if await aiofiles.os.path.exists("/.dockerenv"):
        return True
    try:
        async with aiofiles.open("/proc/1/cgroup", "rt", encoding="utf-8") as fp:
            return "docker" in await fp.read()
    except OSError:
        return False


async def check_browser_requirements() -> bool:
    """
    Check if browser tool requirements are met.

    In **local mode** (no cloud provider configured): the ``agent-browser``
    CLI must be findable. Chrome/Chromium is required for the default Chrome
    engine and for fallback/screenshot paths, but not for Lightpanda-only text
    navigation/snapshot workflows.

    In **cloud mode** (Browserbase, Browser Use, or Firecrawl): the CLI
    and the provider's required credentials must be present. The cloud
    provider hosts its own Chromium, so no local browser binary is needed.

    Returns:
        True if all requirements are met, False otherwise
    """
    # Camofox backend — only needs the server URL, no agent-browser CLI
    if await _is_camofox_mode():
        return True

    # CDP override mode can connect to an existing remote/local browser endpoint
    # without requiring the local agent-browser binary on PATH.
    # Raw (no-I/O) check: this runs during tool-schema assembly at startup,
    # where a stale endpoint must not cost a blocking HTTP probe.
    if await _get_cdp_override_raw():
        return True

    # The agent-browser CLI is required for local launch and cloud-provider flows.
    # Tool-schema assembly runs during Desktop startup; do not execute
    # ``agent-browser --version`` here, because Windows .cmd shims route through
    # cmd.exe and can flash a console before the user invokes any browser tool.
    # Actual browser execution paths still validate the candidate before use.
    try:
        browser_cmd = await _find_agent_browser(validate=False)
    except FileNotFoundError:
        return False

    # On Termux, the bare npx fallback is too fragile to treat as a satisfied
    # local browser dependency. Require a real install (global or local) so the
    # browser tool is not advertised as available when it will likely fail on
    # first use.
    if await _requires_real_termux_browser_install(browser_cmd):
        return False

    # In cloud mode, also require provider credentials. Cloud browsers
    # don't need a local Chromium binary.
    provider = await _get_cloud_provider()
    if provider is not None:
        return await provider.is_available()

    # Local mode with Lightpanda can provide text/navigation tools without a
    # local Chromium install. Chrome fallback, screenshots, and browser_vision
    # will still return actionable Chromium install errors if invoked.
    if await _using_lightpanda_engine():
        return True

    # Local Chrome mode: agent-browser needs a Chromium build on disk. Without
    # it the CLI hangs on first use until the command timeout fires.
    if not await _chromium_installed():
        return False

    return True


async def check_browser_vision_requirements() -> bool:
    """Whether ``browser_vision`` should be advertised to the model.

    Requires BOTH a working browser (``check_browser_requirements``) AND a
    resolvable vision backend. Without the vision check, the tool stays in
    the model's tool list even when no vision provider is configured, then
    fails at call time with a cryptic provider-side error like
    ``unknown variant `image_url`, expected `text``` (issue #31179).
    """
    if not await check_browser_requirements():
        return False
    try:
        from tools.vision_tools import check_vision_requirements
    except ImportError:
        return False
    return await check_vision_requirements()


# ============================================================================
# Module Test
# ============================================================================


async def _main() -> None:
    """Run the module's diagnostic demo."""
    print("🌐 Browser Tool Module")
    print("=" * 40)

    _cp = await _get_cloud_provider()
    mode = "local" if _cp is None else f"cloud ({_cp.name})"
    print(f"   Mode: {mode}")

    # Check requirements
    if await check_browser_requirements():
        print("✅ All requirements met")
    else:
        print("❌ Missing requirements:")
        try:
            browser_cmd = await _find_agent_browser()
            if await _requires_real_termux_browser_install(browser_cmd):
                print(
                    "   - bare npx fallback found (insufficient on Termux local mode)"
                )
                print(f"     Install: {_browser_install_hint()}")
            elif _cp is None and not await _chromium_installed():
                print("   - Chromium browser binary not found")
                searched = ", ".join(_chromium_search_roots()) or "(no candidate paths)"
                print(f"     Searched: {searched}")
                if await _running_in_docker():
                    print(
                        "     Docker: pull the latest image — the current one "
                        "predates the bundled Chromium install"
                    )
                    print("       docker pull ghcr.io/nousresearch/hermes-agent:latest")
                else:
                    print("     Install it with:")
                    print("       npx agent-browser install --with-deps")
                    print("     Or:  npx playwright install --with-deps chromium")
        except FileNotFoundError:
            print("   - agent-browser CLI not found")
            print(f"     Install: {_browser_install_hint()}")
        if _cp is not None and not await _cp.is_available():
            print(f"   - {_cp.name} credentials not configured")
            print(
                "   Tip: set browser.cloud_provider to 'local' to use free local mode instead"
            )

    print("\n📋 Available Browser Tools:")
    for schema in BROWSER_TOOL_SCHEMAS:
        print(f"  🔹 {schema['name']}: {str(schema['description'])[:60]}...")

    print("\n💡 Usage:")
    print("  from tools.browser_tool import browser_navigate, browser_snapshot")
    print("  result = browser_navigate('https://example.com', task_id='my_task')")
    print("  snapshot = browser_snapshot(task_id='my_task')")


if __name__ == "__main__":
    asyncio.run(_main())


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error

_BROWSER_SCHEMA_MAP = {s["name"]: s for s in BROWSER_TOOL_SCHEMAS}


async def _handle_browser_navigate(args: Dict[str, Any], **context: Any) -> str:
    return await browser_navigate(
        url=args.get("url", ""), task_id=context.get("task_id")
    )


async def _handle_browser_snapshot(args: Dict[str, Any], **context: Any) -> str:
    return await browser_snapshot(
        full=args.get("full", False),
        task_id=context.get("task_id"),
        user_task=context.get("user_task"),
    )


async def _handle_browser_click(args: Dict[str, Any], **context: Any) -> str:
    return await browser_click(ref=args.get("ref", ""), task_id=context.get("task_id"))


async def _handle_browser_type(args: Dict[str, Any], **context: Any) -> str:
    return await browser_type(
        ref=args.get("ref", ""),
        text=args.get("text", ""),
        task_id=context.get("task_id"),
    )


async def _handle_browser_scroll(args: Dict[str, Any], **context: Any) -> str:
    return await browser_scroll(
        direction=args.get("direction", "down"),
        task_id=context.get("task_id"),
    )


async def _handle_browser_back(args: Dict[str, Any], **context: Any) -> str:
    return await browser_back(task_id=context.get("task_id"))


async def _handle_browser_press(args: Dict[str, Any], **context: Any) -> str:
    return await browser_press(key=args.get("key", ""), task_id=context.get("task_id"))


async def _handle_browser_get_images(args: Dict[str, Any], **context: Any) -> str:
    return await browser_get_images(task_id=context.get("task_id"))


async def _handle_browser_vision(args: Dict[str, Any], **context: Any):
    return await browser_vision(
        question=args.get("question", ""),
        annotate=args.get("annotate", False),
        task_id=context.get("task_id"),
    )


async def _handle_browser_console(args: Dict[str, Any], **context: Any) -> str:
    return await browser_console(
        clear=args.get("clear", False),
        expression=args.get("expression"),
        task_id=context.get("task_id"),
    )


registry.register(
    name="browser_navigate",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_navigate"],
    handler=_handle_browser_navigate,
    check_fn=check_browser_requirements,
    emoji="🌐",
)
registry.register(
    name="browser_snapshot",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_snapshot"],
    handler=_handle_browser_snapshot,
    check_fn=check_browser_requirements,
    emoji="📸",
)
registry.register(
    name="browser_click",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_click"],
    handler=_handle_browser_click,
    check_fn=check_browser_requirements,
    emoji="👆",
)
registry.register(
    name="browser_type",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_type"],
    handler=_handle_browser_type,
    check_fn=check_browser_requirements,
    emoji="⌨️",
)
registry.register(
    name="browser_scroll",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_scroll"],
    handler=_handle_browser_scroll,
    check_fn=check_browser_requirements,
    emoji="📜",
)
registry.register(
    name="browser_back",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_back"],
    handler=_handle_browser_back,
    check_fn=check_browser_requirements,
    emoji="◀️",
)
registry.register(
    name="browser_press",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_press"],
    handler=_handle_browser_press,
    check_fn=check_browser_requirements,
    emoji="⌨️",
)

registry.register(
    name="browser_get_images",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_get_images"],
    handler=_handle_browser_get_images,
    check_fn=check_browser_requirements,
    emoji="🖼️",
)
registry.register(
    name="browser_vision",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_vision"],
    handler=_handle_browser_vision,
    check_fn=check_browser_vision_requirements,
    emoji="👁️",
)
registry.register(
    name="browser_console",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_console"],
    handler=_handle_browser_console,
    check_fn=check_browser_requirements,
    emoji="🖥️",
)
