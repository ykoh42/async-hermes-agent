"""Camofox browser backend — local anti-detection browser via REST API.

Camofox-browser is a self-hosted Node.js server wrapping Camoufox (Firefox
fork with C++ fingerprint spoofing).  It exposes a REST API that maps 1:1
to our browser tool interface: accessibility snapshots with element refs,
click/type/scroll by ref, screenshots, etc.

When ``CAMOFOX_URL`` is set (e.g. ``http://localhost:9377``), the browser
tools route through this module instead of the ``agent-browser`` CLI.

Setup::

    # Option 1: npm
    git clone https://github.com/jo-inc/camofox-browser && cd camofox-browser
    npm install && npm start   # downloads Camoufox (~300MB) on first run

    # Option 2: Docker
    docker run -p 9377:9377 -e CAMOFOX_PORT=9377 jo-inc/camofox-browser

Then set ``CAMOFOX_URL=http://localhost:9377`` in ``~/.hermes/.env``.
For Docker Camofox, optionally set ``CAMOFOX_REWRITE_LOOPBACK_URLS=true``
so page URLs like ``http://127.0.0.1:3000`` are opened inside the
container as ``http://host.docker.internal:3000``.
"""

from __future__ import annotations

import base64
import asyncio
import contextvars
import json
import logging
import os
import threading
import uuid
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING
from urllib.parse import SplitResult, urlsplit, urlunsplit

import aiofiles
import aiofiles.os

if TYPE_CHECKING:
    import httpx
from agent.secret_scope import get_secret
from agent.ssl_verify import _create_httpx_client
from hermes_constants import get_hermes_home, get_hermes_home_override
from hermes_cli.config import cfg_get, load_config_readonly
from tools.browser_camofox_state import get_camofox_identity
from tools.registry import tool_error

logger = logging.getLogger(__name__)


_CamofoxScopeKey = tuple[object, str]
_CAMOFOX_NO_LOOP = object()
_camofox_scope_context: contextvars.ContextVar[
    tuple[str, _CamofoxScopeKey] | None
] = contextvars.ContextVar("camofox_profile_scope", default=None)
_camofox_scope_aliases: dict[_CamofoxScopeKey, _CamofoxScopeKey] = {}
_camofox_scope_lock = threading.RLock()


def _lexical_camofox_profile_identity() -> str:
    return os.path.normcase(os.fspath(get_hermes_home()))


def _current_camofox_scope_key() -> _CamofoxScopeKey:
    lexical = _lexical_camofox_profile_identity()
    try:
        loop: object = asyncio.get_running_loop()
    except RuntimeError:
        loop = _CAMOFOX_NO_LOOP
    active = _camofox_scope_context.get()
    if active is not None and active[0] == lexical and active[1][0] is loop:
        return active[1]
    with _camofox_scope_lock:
        return _camofox_scope_aliases.get((loop, lexical), (loop, lexical))


@dataclass
class _CamofoxProfileState:
    profile_home: str
    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    sessions_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    vnc_url: str | None = None
    vnc_url_checked: bool = False
    cached_cmd_timeout: int | None = None
    cmd_timeout_resolved: bool = False


_camofox_states: dict[_CamofoxScopeKey, _CamofoxProfileState] = {}


def _camofox_state_for(scope: _CamofoxScopeKey) -> _CamofoxProfileState:
    with _camofox_scope_lock:
        return _camofox_states.setdefault(
            scope,
            _CamofoxProfileState(profile_home=scope[1]),
        )


def _camofox_state() -> _CamofoxProfileState:
    return _camofox_state_for(_current_camofox_scope_key())


def _merge_camofox_state(
    source: _CamofoxScopeKey,
    target: _CamofoxScopeKey,
) -> None:
    if source == target:
        return
    with _camofox_scope_lock:
        staged = _camofox_states.pop(source, None)
        if staged is None:
            return
        current = _camofox_states.get(target)
        if current is None:
            staged.profile_home = target[1]
            _camofox_states[target] = staged
            return
        current.sessions.update(staged.sessions)


async def _activate_camofox_scope() -> _CamofoxScopeKey:
    lexical = _lexical_camofox_profile_identity()
    loop = asyncio.get_running_loop()
    active = _camofox_scope_context.get()
    if active is not None and active[0] == lexical and active[1][0] is loop:
        return active[1]
    expanduser = aiofiles.os.wrap(os.path.expanduser)
    expanded = str(await expanduser(lexical))
    is_absolute = (
        expanded.startswith(("/", "\\\\"))
        or (len(expanded) >= 3 and expanded[1] == ":" and expanded[2] in "/\\")
    )
    if not is_absolute:
        expanded = str(await aiofiles.os.getcwd()) + os.sep + expanded
    realpath = aiofiles.os.wrap(os.path.realpath)
    canonical = os.path.normcase(str(await realpath(expanded)))
    scope: _CamofoxScopeKey = (loop, canonical)
    with _camofox_scope_lock:
        _camofox_scope_aliases[(loop, lexical)] = scope
    _camofox_scope_context.set((lexical, scope))
    _merge_camofox_state((loop, lexical), scope)
    _merge_camofox_state((_CAMOFOX_NO_LOOP, lexical), scope)
    _camofox_state_for(scope)
    return scope


class _ScopedCamofoxSessions(MutableMapping):
    """Dict-compatible current-profile view for the historical private map."""

    def _active(self) -> dict:
        return _camofox_state().sessions

    def __getitem__(self, key):
        return self._active()[key]

    def __setitem__(self, key, value) -> None:
        self._active()[key] = value

    def __delitem__(self, key) -> None:
        del self._active()[key]

    def __iter__(self):
        return iter(tuple(self._active()))

    def __len__(self) -> int:
        return len(self._active())

    def clear(self) -> None:
        self._active().clear()

    def copy(self) -> dict:
        return self._active().copy()


def __getattr__(name: str):
    """Resolve httpx lazily while preserving the module patch surface."""
    if name == "httpx":
        module = _get_httpx_module()
        globals()["httpx"] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _get_httpx_module():
    import httpx as httpx_module

    return httpx_module

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 30  # fallback when config is unreadable
_SNAPSHOT_MAX_CHARS = 80_000  # camofox paginates at this limit
_vnc_url: Optional[str] = None  # cached from /health response
_vnc_url_checked = False  # only probe once per process

# Cached command timeout from config (resolved lazily, like browser_tool)
_cached_cmd_timeout: Optional[int] = None
_cmd_timeout_resolved = False


async def _get_command_timeout() -> int:
    """Return ``browser.command_timeout`` from config, falling back to 30s.

    Mirrors :func:`tools.browser_tool._get_command_timeout` so both the
    local browser path and the Camofox path honour the same config knob.
    Result is cached after the first call.
    """
    global _cached_cmd_timeout, _cmd_timeout_resolved
    scoped_state: _CamofoxProfileState | None = None
    if get_hermes_home_override() is not None:
        await _activate_camofox_scope()
        scoped_state = _camofox_state()
        if (
            scoped_state.cmd_timeout_resolved
            and scoped_state.cached_cmd_timeout is not None
        ):
            return scoped_state.cached_cmd_timeout
    elif _cmd_timeout_resolved and _cached_cmd_timeout is not None:
        return _cached_cmd_timeout

    result = _DEFAULT_TIMEOUT
    try:
        cfg = await load_config_readonly()
        val = cfg_get(cfg, "browser", "command_timeout")
        if val is not None:
            result = max(int(val), 5)  # floor at 5s
    except Exception as exc:
        logger.debug("Could not read browser.command_timeout: %s", exc)
    if scoped_state is None:
        _cached_cmd_timeout = result
        _cmd_timeout_resolved = True
    else:
        scoped_state.cached_cmd_timeout = result
        scoped_state.cmd_timeout_resolved = True
    return result


def _auth_headers() -> Dict[str, str]:
    """Return Authorization header when CAMOFOX_API_KEY is set."""
    key = (get_secret("CAMOFOX_API_KEY", "") or "").strip()
    if key:
        return {"Authorization": f"Bearer {key}"}
    return {}


def get_camofox_url() -> str:
    """Return the configured Camofox server URL, or empty string."""
    return (get_secret("CAMOFOX_URL", "") or "").rstrip("/")


async def _config_cdp_url() -> str:
    """Persistent ``browser.cdp_url`` from config.yaml, or empty string.

    Read here (instead of importing ``browser_tool._get_cdp_override`` to avoid
    a circular import) so Camofox can yield to a config-based CDP override the
    same way it already yields to the ``BROWSER_CDP_URL`` env override.
    """
    try:
        browser_cfg = (await load_config_readonly()).get("browser", {})
        if isinstance(browser_cfg, dict):
            return str(browser_cfg.get("cdp_url", "") or "").strip()
    except Exception:
        pass
    return ""


async def is_camofox_mode() -> bool:
    """True when Camofox backend is configured and no CDP override is active.

    A CDP override takes priority over Camofox so the browser tools operate on
    the real CDP browser (and a CDP backend is treated as non-local for SSRF
    checks) instead of being silently routed to Camofox. The override may come
    from the ``BROWSER_CDP_URL`` env var (set by ``/browser connect``) OR a
    persistent ``browser.cdp_url`` in config.yaml — both are honored, matching
    ``browser_tool._get_cdp_override()``'s precedence. (Previously only the env
    var suppressed Camofox, so ``CAMOFOX_URL`` + a config CDP override still
    routed navigation through Camofox.)
    """
    if os.getenv("BROWSER_CDP_URL", "").strip():
        return False
    if await _config_cdp_url():
        return False
    return bool(get_camofox_url())


async def check_camofox_available() -> bool:
    """Verify the Camofox server is reachable."""
    global _vnc_url, _vnc_url_checked
    scoped_state: _CamofoxProfileState | None = None
    if get_hermes_home_override() is not None:
        await _activate_camofox_scope()
        scoped_state = _camofox_state()
    url = get_camofox_url()
    if not url:
        return False
    try:
        async with (await _create_httpx_client(timeout=5)) as client:
            resp = await client.get(f"{url}/health")
        already_checked = (
            _vnc_url_checked
            if scoped_state is None
            else scoped_state.vnc_url_checked
        )
        if resp.status_code == 200 and not already_checked:
            try:
                data = resp.json()
                vnc_port = data.get("vncPort")
                if isinstance(vnc_port, int) and 1 <= vnc_port <= 65535:
                    from urllib.parse import urlparse

                    parsed = urlparse(url)
                    host = parsed.hostname or "localhost"
                    resolved_vnc_url = f"http://{host}:{vnc_port}"
                    if scoped_state is None:
                        _vnc_url = resolved_vnc_url
                    else:
                        scoped_state.vnc_url = resolved_vnc_url
            except (ValueError, KeyError):
                pass
            if scoped_state is None:
                _vnc_url_checked = True
            else:
                scoped_state.vnc_url_checked = True
        return resp.status_code == 200
    except Exception:
        return False


async def get_vnc_url() -> Optional[str]:
    """Return the VNC URL if the Camofox server exposes one, or None."""
    if get_hermes_home_override() is not None:
        await _activate_camofox_scope()
        state = _camofox_state()
        if not state.vnc_url_checked:
            await check_camofox_available()
        return state.vnc_url
    if not _vnc_url_checked:
        await check_camofox_available()
    return _vnc_url


async def _get_camofox_config() -> Dict[str, Any]:
    """Return the ``browser.camofox`` config block, or an empty dict."""
    try:
        camofox_cfg = (
            (await load_config_readonly()).get("browser", {}).get("camofox", {})
        )
    except Exception as exc:
        logger.warning("camofox config check failed, defaulting to disabled: %s", exc)
        return {}
    return camofox_cfg if isinstance(camofox_cfg, dict) else {}


async def _managed_persistence_enabled() -> bool:
    """Return whether Hermes-managed persistence is enabled for Camofox.

    When enabled, sessions use a stable profile-scoped userId so the
    Camofox server can map it to a persistent browser profile directory.
    When disabled (default), each session gets a random userId (ephemeral).

    Controlled by ``browser.camofox.managed_persistence`` in config.yaml.
    """
    return bool((await _get_camofox_config()).get("managed_persistence"))


def _camofox_identity_override(
    task_id: Optional[str], camofox_cfg: Dict[str, Any]
) -> Optional[Dict[str, str]]:
    """Return an externally configured Camofox identity, if one is set.

    Integrations that own the visible Camofox browser can set a shared user ID
    so Hermes operates in the same browser profile instead of creating a
    separate private session.
    """
    user_id = (get_secret("CAMOFOX_USER_ID", "") or "").strip() or str(
        camofox_cfg.get("user_id") or ""
    ).strip()
    if not user_id:
        return None

    session_key = (
        (get_secret("CAMOFOX_SESSION_KEY", "") or "").strip()
        or str(camofox_cfg.get("session_key") or "").strip()
        or f"task_{(task_id or 'default')[:16]}"
    )
    return {"user_id": user_id, "session_key": session_key}


def _env_flag(name: str) -> Optional[bool]:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return None
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    logger.debug("Ignoring invalid boolean env %s=%r", name, raw)
    return None


def _adopt_existing_tab_enabled(camofox_cfg: Dict[str, Any]) -> bool:
    """Return whether Hermes should recover an existing Camofox tab ID."""
    env_value = _env_flag("CAMOFOX_ADOPT_EXISTING_TAB")
    if env_value is not None:
        return env_value
    return bool(camofox_cfg.get("adopt_existing_tab"))


def _loopback_rewrite_enabled(camofox_cfg: Dict[str, Any]) -> bool:
    """Return whether loopback navigation URLs should be rewritten for Docker.

    ``CAMOFOX_URL`` itself often points at a host-published Docker port such as
    ``http://127.0.0.1:9377``.  That is correct for Hermes talking to the
    Camofox control API, but a page URL like ``http://127.0.0.1:3000`` is opened
    by the browser *inside* the Docker container.  In that context loopback
    points at the container, not the host running the web app.

    The rewrite is opt-in because non-Docker Camofox installs run the browser on
    the host, where loopback URLs are already correct.
    """
    env_value = _env_flag("CAMOFOX_REWRITE_LOOPBACK_URLS")
    if env_value is not None:
        return env_value
    return bool(camofox_cfg.get("rewrite_loopback_urls"))


def _loopback_rewrite_host(camofox_cfg: Dict[str, Any]) -> str:
    """Return the host alias used when rewriting loopback page URLs."""
    return (
        os.getenv("CAMOFOX_LOOPBACK_HOST_ALIAS", "").strip()
        or str(camofox_cfg.get("loopback_host_alias") or "").strip()
        or "host.docker.internal"
    )


def _is_loopback_hostname(hostname: Optional[str]) -> bool:
    """Return True for localhost/127.0.0.0/8/::1-style hostnames."""
    if not hostname:
        return False
    host = hostname.strip().strip("[]").lower()
    if host in {"localhost", "localhost.localdomain"}:
        return True
    try:
        import ipaddress

        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


async def _rewrite_loopback_url_for_camofox(
    url: str,
) -> tuple[str, Optional[Dict[str, str]]]:
    """Rewrite loopback page URLs for Docker-hosted Camofox, if configured.

    Returns ``(rewritten_url, metadata)``.  ``metadata`` is present only when a
    rewrite happened so the tool result can disclose the change to the model.
    """
    camofox_cfg = await _get_camofox_config()
    if not _loopback_rewrite_enabled(camofox_cfg):
        return url, None

    try:
        parsed = urlsplit(url)
    except ValueError:
        return url, None

    if parsed.scheme not in {"http", "https"} or not _is_loopback_hostname(
        parsed.hostname
    ):
        return url, None

    alias = _loopback_rewrite_host(camofox_cfg)
    if not alias:
        return url, None

    userinfo = ""
    if parsed.username:
        userinfo = parsed.username
        if parsed.password:
            userinfo += f":{parsed.password}"
        userinfo += "@"
    host_part = f"[{alias}]" if ":" in alias and not alias.startswith("[") else alias
    port_part = f":{parsed.port}" if parsed.port else ""
    rewritten = urlunsplit(
        SplitResult(
            parsed.scheme,
            f"{userinfo}{host_part}{port_part}",
            parsed.path,
            parsed.query,
            parsed.fragment,
        )
    )
    return rewritten, {
        "from": parsed.hostname or "",
        "to": alias,
        "original_url": url,
        "rewritten_url": rewritten,
    }


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
# Maps task_id -> {"user_id": str, "tab_id": str|None}
_sessions: MutableMapping[str, Dict[str, Any]] = _ScopedCamofoxSessions()


def _sessions_lock_for_profile() -> asyncio.Lock:
    return _camofox_state().sessions_lock


async def _adopt_existing_tab(session: Dict[str, Any]) -> Dict[str, Any]:
    """Attach process-local state to an already-open managed Camofox tab.

    Some integrations own the visible Camofox tab outside Hermes. Gateway
    restarts can leave this module's in-memory session cache empty even though
    Camofox still has that tab, so rehydrate tab_id before creating a new tab.
    """
    if session.get("tab_id") or not session.get("adopt_existing_tab"):
        return session

    if not get_camofox_url():
        return session

    try:
        tabs = (
            await _get("/tabs", params={"userId": session["user_id"]}, timeout=5)
        ).get("tabs", [])
    except Exception as exc:
        logger.debug(
            "Camofox tab adoption failed for %s: %s", session.get("user_id"), exc
        )
        return session

    if not isinstance(tabs, list) or not tabs:
        return session

    session_key = session.get("session_key")
    matching_tabs = [
        tab
        for tab in tabs
        if isinstance(tab, dict) and tab.get("listItemId") == session_key
    ]
    candidates = matching_tabs or [tab for tab in tabs if isinstance(tab, dict)]
    latest = candidates[-1] if candidates else None
    tab_id = latest.get("tabId") if isinstance(latest, dict) else None
    if isinstance(tab_id, str) and tab_id:
        session["tab_id"] = tab_id
        logger.debug(
            "Adopted existing Camofox tab %s for %s", tab_id, session.get("user_id")
        )

    return session


async def _get_session(task_id: Optional[str]) -> Dict[str, Any]:
    """Get or create a camofox session for the given task.

    When managed persistence is enabled, uses a deterministic userId
    derived from the Hermes profile so the Camofox server can map it
    to the same persistent browser profile across restarts.
    """
    await _activate_camofox_scope()
    task_id = task_id or "default"
    async with _sessions_lock_for_profile():
        if task_id in _sessions:
            return await _adopt_existing_tab(_sessions[task_id])

        camofox_cfg = await _get_camofox_config()
        identity_override = _camofox_identity_override(task_id, camofox_cfg)
        if identity_override:
            session = {
                "user_id": identity_override["user_id"],
                "tab_id": None,
                "session_key": identity_override["session_key"],
                "managed": True,
                "adopt_existing_tab": _adopt_existing_tab_enabled(camofox_cfg),
            }
        elif bool(camofox_cfg.get("managed_persistence")):
            identity = get_camofox_identity(task_id)
            session = {
                "user_id": identity["user_id"],
                "tab_id": None,
                "session_key": identity["session_key"],
                "managed": True,
                "adopt_existing_tab": _adopt_existing_tab_enabled(camofox_cfg),
            }
        else:
            session = {
                "user_id": f"hermes_{uuid.uuid4().hex[:10]}",
                "tab_id": None,
                "session_key": f"task_{task_id[:16]}",
                "managed": False,
                "adopt_existing_tab": False,
            }
        _sessions[task_id] = session
        return await _adopt_existing_tab(session)


async def _ensure_tab(
    task_id: Optional[str], url: str = "about:blank"
) -> Dict[str, Any]:
    """Ensure a tab exists for the session, creating one if needed."""
    session = await _get_session(task_id)
    if session["tab_id"]:
        return session
    base = get_camofox_url()
    async with (
        await _create_httpx_client(timeout=await _get_command_timeout())
    ) as client:
        resp = await client.post(
            f"{base}/tabs",
            json={
                "userId": session["user_id"],
                "listItemId": session["session_key"],
                "url": url,
            },
            headers=_auth_headers(),
        )
    resp.raise_for_status()
    data = resp.json()
    session["tab_id"] = data.get("tabId")
    return session


async def _drop_session(task_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Remove and return session info."""
    await _activate_camofox_scope()
    task_id = task_id or "default"
    async with _sessions_lock_for_profile():
        return _sessions.pop(task_id, None)


async def camofox_soft_cleanup(task_id: Optional[str] = None) -> bool:
    """Release the in-memory session without destroying the server-side context.

    When managed persistence is enabled the browser profile (and its cookies)
    must survive across agent tasks.  This helper drops only the local tracking
    entry and returns ``True``.  When managed persistence is *not* enabled it
    does nothing and returns ``False`` so the caller can fall back to
    :func:`camofox_close`.
    """
    camofox_cfg = await _get_camofox_config()
    if bool(camofox_cfg.get("managed_persistence")) or _camofox_identity_override(
        task_id, camofox_cfg
    ):
        await _drop_session(task_id)
        logger.debug("Camofox soft cleanup for task %s (managed persistence)", task_id)
        return True
    return False


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


async def _post(path: str, body: dict, timeout: Optional[int] = None) -> dict:
    """POST JSON to camofox and return parsed response."""
    if timeout is None:
        timeout = await _get_command_timeout()
    url = f"{get_camofox_url()}{path}"
    async with (await _create_httpx_client(timeout=timeout)) as client:
        resp = await client.post(url, json=body, headers=_auth_headers())
    resp.raise_for_status()
    return resp.json()


async def _get(
    path: str, params: Optional[dict] = None, timeout: Optional[int] = None
) -> dict:
    """GET from camofox and return parsed response."""
    if timeout is None:
        timeout = await _get_command_timeout()
    url = f"{get_camofox_url()}{path}"
    async with (await _create_httpx_client(timeout=timeout)) as client:
        resp = await client.get(url, params=params, headers=_auth_headers())
    resp.raise_for_status()
    return resp.json()


async def _get_raw(
    path: str, params: Optional[dict] = None, timeout: Optional[int] = None
) -> httpx.Response:
    """GET from camofox and return raw response (for binary data)."""
    if timeout is None:
        timeout = await _get_command_timeout()
    url = f"{get_camofox_url()}{path}"
    async with (await _create_httpx_client(timeout=timeout)) as client:
        resp = await client.get(url, params=params, headers=_auth_headers())
    resp.raise_for_status()
    return resp


async def _delete(
    path: str, body: Optional[dict] = None, timeout: Optional[int] = None
) -> dict:
    """DELETE to camofox and return parsed response."""
    if timeout is None:
        timeout = await _get_command_timeout()
    url = f"{get_camofox_url()}{path}"
    async with (await _create_httpx_client(timeout=timeout)) as client:
        resp = await client.request("DELETE", url, json=body, headers=_auth_headers())
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


async def camofox_navigate(url: str, task_id: Optional[str] = None) -> str:
    """Navigate to a URL via Camofox."""
    httpx = _get_httpx_module()
    try:
        browser_url, rewrite_info = await _rewrite_loopback_url_for_camofox(url)
        session = await _get_session(task_id)
        if not session["tab_id"]:
            # Create tab with the target URL directly
            session = await _ensure_tab(task_id, browser_url)
            data = {"ok": True, "url": browser_url}
        else:
            # Navigate existing tab — recover from stale tab 404
            try:
                data = await _post(
                    f"/tabs/{session['tab_id']}/navigate",
                    {"userId": session["user_id"], "url": browser_url},
                    timeout=60,
                )
            except httpx.HTTPStatusError as e:
                if e.response is not None and e.response.status_code == 404:
                    logger.warning(
                        "Camofox tab %s returned 404 — tab was garbage collected. "
                        "Creating a fresh tab.",
                        session["tab_id"],
                    )
                    session["tab_id"] = None
                    # A new caller cancellation must supersede the stale-tab error.
                    session = await _ensure_tab(task_id, browser_url)  # noqa: ASYNC120
                    data = {"ok": True, "url": browser_url}
                else:
                    raise
        result = {
            "success": True,
            "url": data.get("url", browser_url),
            "title": data.get("title", ""),
        }
        if rewrite_info:
            result["requested_url"] = url
            result["url_rewrite"] = rewrite_info
            result["warning"] = (
                "Rewrote loopback URL for Docker-hosted Camofox: "
                f"{rewrite_info['from']} -> {rewrite_info['to']}"
            )
        vnc = await get_vnc_url()
        if vnc:
            result["vnc_url"] = vnc
            result["vnc_hint"] = (
                "Browser is visible via VNC. "
                "Share this link with the user so they can watch the browser live."
            )

        # Auto-take a compact snapshot so the model can act immediately
        try:
            snap_data = await _get(
                f"/tabs/{session['tab_id']}/snapshot",
                params={"userId": session["user_id"]},
            )
            snapshot_text = snap_data.get("snapshot", "")
            from tools.browser_tool import (
                SNAPSHOT_SUMMARIZE_THRESHOLD,
                _truncate_snapshot,
            )

            if len(snapshot_text) > SNAPSHOT_SUMMARIZE_THRESHOLD:
                snapshot_text = await _truncate_snapshot(snapshot_text)
            result["snapshot"] = snapshot_text
            result["element_count"] = snap_data.get("refsCount", 0)
        except Exception:
            pass  # Navigation succeeded; snapshot is a bonus

        return json.dumps(result)
    except httpx.HTTPStatusError as e:
        return tool_error(f"Navigation failed: {e}", success=False)
    except httpx.RequestError:
        return json.dumps({
            "success": False,
            "error": f"Cannot connect to Camofox at {get_camofox_url()}. "
            "Is the server running? Start with: npm start (in camofox-browser dir) "
            "or: docker run -p 9377:9377 -e CAMOFOX_PORT=9377 jo-inc/camofox-browser",
        })
    except Exception as e:
        return tool_error(str(e), success=False)


async def _camofox_private_page_block(
    session: Dict[str, Any], task_id: Optional[str], action: str
) -> Optional[str]:
    """Return a blocked payload when the current Camofox page is private/internal.

    Mirrors the eval-path guard added for ``_camofox_eval`` (browser_tool.py):
    Camofox snapshot / vision / image-extraction all read current page state, so
    on a non-local backend they can leak the content of an intranet/metadata
    page the terminal itself can't reach.  The gate matches ``browser_snapshot``
    / ``browser_vision`` — only active when the SSRF guard applies (non-local
    backend, not a local sidecar, ``allow_private_urls`` unset).  Fail-open on
    probe failure, matching the sibling guards.

    Imports are deferred to call time because ``browser_tool`` imports this
    module; importing it at module load would create a circular import.
    """
    from tools.browser_tool import (
        _camofox_current_page_private_url,
        _eval_ssrf_guard_active,
    )

    if not await _eval_ssrf_guard_active(task_id or "default"):
        return None
    blocked_url = await _camofox_current_page_private_url(
        session["tab_id"], session["user_id"]
    )
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


async def camofox_snapshot(
    full: bool = False,
    task_id: Optional[str] = None,
    user_task: Optional[str] = None,
) -> str:
    """Get accessibility tree snapshot from Camofox."""
    try:
        session = await _get_session(task_id)
        if not session["tab_id"]:
            return tool_error(
                "No browser session. Call browser_navigate first.", success=False
            )

        blocked = await _camofox_private_page_block(
            session, task_id, "read a page snapshot"
        )
        if blocked:
            return blocked

        data = await _get(
            f"/tabs/{session['tab_id']}/snapshot",
            params={"userId": session["user_id"]},
        )

        snapshot = data.get("snapshot", "")
        refs_count = data.get("refsCount", 0)

        # Apply same summarization logic as the main browser tool
        from tools.browser_tool import (
            SNAPSHOT_SUMMARIZE_THRESHOLD,
            _extract_relevant_content,
            _truncate_snapshot,
        )

        if len(snapshot) > SNAPSHOT_SUMMARIZE_THRESHOLD:
            if user_task:
                snapshot = await _extract_relevant_content(snapshot, user_task)
            else:
                snapshot = await _truncate_snapshot(snapshot)

        return json.dumps({
            "success": True,
            "snapshot": snapshot,
            "element_count": refs_count,
        })
    except Exception as e:
        return tool_error(str(e), success=False)


async def camofox_click(ref: str, task_id: Optional[str] = None) -> str:
    """Click an element by ref via Camofox."""
    try:
        session = await _get_session(task_id)
        if not session["tab_id"]:
            return tool_error(
                "No browser session. Call browser_navigate first.", success=False
            )

        blocked = await _camofox_private_page_block(session, task_id, "click")
        if blocked:
            return blocked

        # Strip @ prefix if present (our tool convention)
        clean_ref = ref.lstrip("@")

        data = await _post(
            f"/tabs/{session['tab_id']}/click",
            {"userId": session["user_id"], "ref": clean_ref},
        )
        return json.dumps({
            "success": True,
            "clicked": clean_ref,
            "url": data.get("url", ""),
        })
    except Exception as e:
        return tool_error(str(e), success=False)


async def camofox_type(ref: str, text: str, task_id: Optional[str] = None) -> str:
    """Type text into an element by ref via Camofox."""
    try:
        session = await _get_session(task_id)
        if not session["tab_id"]:
            return tool_error(
                "No browser session. Call browser_navigate first.", success=False
            )

        blocked = await _camofox_private_page_block(session, task_id, "type")
        if blocked:
            return blocked

        clean_ref = ref.lstrip("@")

        await _post(
            f"/tabs/{session['tab_id']}/type",
            {"userId": session["user_id"], "ref": clean_ref, "text": text},
        )
        from agent.display import (
            redact_browser_typed_text_for_display,
            redact_tool_args_for_display,
        )

        display_text = (
            redact_tool_args_for_display("browser_type", {"text": text}) or {}
        )["text"]

        response = {
            "success": True,
            # Match browser_tool.browser_type: run typed text through the
            # secret-pattern redactor so API keys / tokens don't leak into
            # tool progress or chat history.  The raw text is still typed into
            # the page; only the returned display value is redacted.
            "typed": display_text,
            "element": clean_ref,
        }
        response = redact_browser_typed_text_for_display(response, text)
        return json.dumps(response)
    except Exception as e:
        from agent.display import redact_browser_typed_text_for_display

        return tool_error(
            redact_browser_typed_text_for_display(str(e), text), success=False
        )


async def camofox_scroll(direction: str, task_id: Optional[str] = None) -> str:
    """Scroll the page via Camofox."""
    try:
        session = await _get_session(task_id)
        if not session["tab_id"]:
            return tool_error(
                "No browser session. Call browser_navigate first.", success=False
            )

        await _post(
            f"/tabs/{session['tab_id']}/scroll",
            {"userId": session["user_id"], "direction": direction},
        )
        return json.dumps({"success": True, "scrolled": direction})
    except Exception as e:
        return tool_error(str(e), success=False)


async def camofox_back(task_id: Optional[str] = None) -> str:
    """Navigate back via Camofox."""
    try:
        session = await _get_session(task_id)
        if not session["tab_id"]:
            return tool_error(
                "No browser session. Call browser_navigate first.", success=False
            )

        data = await _post(
            f"/tabs/{session['tab_id']}/back",
            {"userId": session["user_id"]},
        )
        return json.dumps({"success": True, "url": data.get("url", "")})
    except Exception as e:
        return tool_error(str(e), success=False)


async def camofox_press(key: str, task_id: Optional[str] = None) -> str:
    """Press a keyboard key via Camofox."""
    try:
        session = await _get_session(task_id)
        if not session["tab_id"]:
            return tool_error(
                "No browser session. Call browser_navigate first.", success=False
            )

        blocked = await _camofox_private_page_block(session, task_id, "press")
        if blocked:
            return blocked

        await _post(
            f"/tabs/{session['tab_id']}/press",
            {"userId": session["user_id"], "key": key},
        )
        return json.dumps({"success": True, "pressed": key})
    except Exception as e:
        return tool_error(str(e), success=False)


async def camofox_close(task_id: Optional[str] = None) -> str:
    """Close the browser session via Camofox."""
    try:
        session = await _drop_session(task_id)
        if not session:
            return json.dumps({"success": True, "closed": True})

        await _delete(
            f"/sessions/{session['user_id']}",
        )
        return json.dumps({"success": True, "closed": True})
    except Exception as e:
        return json.dumps({"success": True, "closed": True, "warning": str(e)})


async def camofox_get_images(task_id: Optional[str] = None) -> str:
    """Get images on the current page via Camofox.

    Extracts image information from the accessibility tree snapshot,
    since Camofox does not expose a dedicated /images endpoint.
    """
    try:
        session = await _get_session(task_id)
        if not session["tab_id"]:
            return tool_error(
                "No browser session. Call browser_navigate first.", success=False
            )

        blocked = await _camofox_private_page_block(
            session, task_id, "extract page images"
        )
        if blocked:
            return blocked

        import re

        data = await _get(
            f"/tabs/{session['tab_id']}/snapshot",
            params={"userId": session["user_id"]},
        )
        snapshot = data.get("snapshot", "")

        # Parse img elements from the accessibility tree.
        # Format: img "alt text" or img "alt text" [eN]
        # URLs appear on /url: lines following img entries
        images = []
        lines = snapshot.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("- img ", "img ")):
                alt_match = re.search(r'img\s+"([^"]*)"', stripped)
                alt = alt_match.group(1) if alt_match else ""
                # Look for URL on the next line
                src = ""
                if i + 1 < len(lines):
                    url_match = re.search(r"/url:\s*(\S+)", lines[i + 1].strip())
                    if url_match:
                        src = url_match.group(1)
                if alt or src:
                    images.append({"src": src, "alt": alt})

        return json.dumps({
            "success": True,
            "images": images,
            "count": len(images),
        })
    except Exception as e:
        return tool_error(str(e), success=False)


async def camofox_vision(
    question: str,
    annotate: bool = False,
    task_id: Optional[str] = None,
) -> str:
    """Take a screenshot and analyze it with vision AI via Camofox."""
    try:
        session = await _get_session(task_id)
        if not session["tab_id"]:
            return tool_error(
                "No browser session. Call browser_navigate first.", success=False
            )

        blocked = await _camofox_private_page_block(
            session, task_id, "capture a screenshot"
        )
        if blocked:
            return blocked

        # Get screenshot as binary PNG
        resp = await _get_raw(
            f"/tabs/{session['tab_id']}/screenshot",
            params={"userId": session["user_id"]},
        )

        # Save screenshot to cache
        from hermes_constants import get_hermes_home

        screenshots_dir = get_hermes_home() / "browser_screenshots"
        await aiofiles.os.makedirs(screenshots_dir, exist_ok=True)
        screenshot_path = str(
            screenshots_dir / f"browser_screenshot_{uuid.uuid4().hex[:8]}.png"
        )

        async with aiofiles.open(screenshot_path, "wb") as f:
            await f.write(resp.content)

        # Encode for vision LLM
        img_b64 = base64.b64encode(resp.content).decode("utf-8")

        # Also get annotated snapshot if requested
        annotation_context = ""
        if annotate:
            try:
                snap_data = await _get(
                    f"/tabs/{session['tab_id']}/snapshot",
                    params={"userId": session["user_id"]},
                )
                annotation_context = f"\n\nAccessibility tree (element refs for interaction):\n{snap_data.get('snapshot', '')[:3000]}"
            except Exception:
                pass

        # Redact secrets from annotation context before sending to vision LLM.
        # The screenshot image itself cannot be redacted, but at least the
        # text-based accessibility tree snippet won't leak secret values.
        from agent.redact import redact_sensitive_text

        annotation_context = redact_sensitive_text(annotation_context)

        # Send to vision LLM
        from agent.auxiliary_client import call_llm

        vision_prompt = (
            f"Analyze this browser screenshot and answer: {question}"
            f"{annotation_context}"
        )

        try:
            _cfg = await load_config_readonly()
            _vision_cfg = cfg_get(_cfg, "auxiliary", "vision", default={})
            _vision_timeout = float(_vision_cfg.get("timeout", 120))
            _vision_temperature = float(_vision_cfg.get("temperature", 0.1))
        except Exception:
            _vision_timeout = 120.0
            _vision_temperature = 0.1

        response = await call_llm(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}",
                            },
                        },
                    ],
                }
            ],
            task="vision",
            temperature=_vision_temperature,
            timeout=_vision_timeout,
        )
        analysis = (
            (response.choices[0].message.content or "").strip()
            if response.choices
            else ""
        )

        # Redact secrets the vision LLM may have read from the screenshot.
        from agent.redact import redact_sensitive_text

        analysis = redact_sensitive_text(analysis)

        return json.dumps({
            "success": True,
            "analysis": analysis,
            "screenshot_path": screenshot_path,
        })
    except Exception as e:
        return tool_error(str(e), success=False)


async def camofox_console(clear: bool = False, task_id: Optional[str] = None) -> str:
    """Get console output — limited support in Camofox.

    Camofox does not expose browser console logs via its REST API.
    Returns an empty result with a note.
    """
    return json.dumps({
        "success": True,
        "console_messages": [],
        "js_errors": [],
        "total_messages": 0,
        "total_errors": 0,
        "note": "Console log capture is not available with the Camofox backend. "
        "Use browser_snapshot or browser_vision to inspect page state.",
    })
