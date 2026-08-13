"""Entry point for the `computer_use` tool.

Universal (any-model) desktop control across macOS, Windows, and Linux via
cua-driver's background computer-use primitive. Replaces #4562's
Anthropic-native `computer_20251124` approach — the schema here is standard
OpenAI function-calling so every tool-capable model can drive it.

Linux is the most recent runtime (X11 + Wayland, via cua-driver-rs's
AT-SPI tree path); it is enabled here alongside macOS and Windows. When a
host's display server or accessibility stack isn't reachable, cua-driver's
`health_report` (surfaced by `hermes computer-use doctor`) reports the
exact blocked check rather than the toolset silently failing.

Return contract
---------------
For text-only results (wait, key, list_apps, focus_app, failures, etc.):
  JSON string.

For captures / actions with `capture_after=True`:
  A dict wrapped as the OpenAI-style multi-part tool-message content:

      {
        "_multimodal": True,
        "content": [
            {"type": "text", "text": "<human-readable summary + SOM index>"},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64,<b64>"}},
        ],
        "text_summary": "<text used for fallback string content>",
      }

  run_agent.py's tool-message builder inspects `_multimodal` and emits a
  list-shaped `content` for OpenAI-compatible providers. The Anthropic
  adapter splices the base64 image into a `tool_result` block (see
  `agent/anthropic_adapter.py`). Every provider that supports multi-part
  tool content gets the image; text-only providers see the summary only.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import base64
import json
import logging
import os
import re
import struct
import sys
import threading
import aiofiles
import aiofiles.os
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home
from tools.computer_use.backend import (
    ActionResult,
    CaptureResult,
    ComputerUseBackend,
    UIElement,
)

logger = logging.getLogger(__name__)


_ComputerUseScopeKey = tuple[object, str]
_COMPUTER_USE_NO_LOOP = object()
_computer_use_scope_context: contextvars.ContextVar[
    tuple[str, _ComputerUseScopeKey] | None
] = contextvars.ContextVar("computer_use_profile_scope", default=None)
_computer_use_scope_aliases: dict[
    _ComputerUseScopeKey, _ComputerUseScopeKey
] = {}
_computer_use_scope_lock = threading.RLock()


def _lexical_computer_use_profile_identity() -> str:
    return os.path.normcase(os.fspath(get_hermes_home()))


def _current_computer_use_scope_key() -> _ComputerUseScopeKey:
    lexical = _lexical_computer_use_profile_identity()
    try:
        loop: object = asyncio.get_running_loop()
    except RuntimeError:
        loop = _COMPUTER_USE_NO_LOOP
    active = _computer_use_scope_context.get()
    if active is not None and active[0] == lexical and active[1][0] is loop:
        return active[1]
    with _computer_use_scope_lock:
        return _computer_use_scope_aliases.get((loop, lexical), (loop, lexical))


@dataclass
class _ComputerUseProfileState:
    profile_home: str
    default_backend: ComputerUseBackend | None = None
    backends: dict[str, ComputerUseBackend] = field(default_factory=dict)
    backend_init_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    backend_call_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    backend_permission_modes: dict[str, str] = field(default_factory=dict)
    session_auto_approve: dict[str, bool] = field(default_factory=dict)
    always_allow: dict[str, set] = field(default_factory=dict)
    aux_vision_route_cache: dict[tuple[str, str], bool] = field(
        default_factory=dict
    )
    backend_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    approval_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    approval_callback: Any = None


_computer_use_states: dict[
    _ComputerUseScopeKey, _ComputerUseProfileState
] = {}


def _computer_use_state_for(
    scope: _ComputerUseScopeKey,
) -> _ComputerUseProfileState:
    with _computer_use_scope_lock:
        return _computer_use_states.setdefault(
            scope,
            _ComputerUseProfileState(profile_home=scope[1]),
        )


def _computer_use_state() -> _ComputerUseProfileState:
    return _computer_use_state_for(_current_computer_use_scope_key())


def _merge_computer_use_state(
    source: _ComputerUseScopeKey,
    target: _ComputerUseScopeKey,
) -> None:
    if source == target:
        return
    with _computer_use_scope_lock:
        staged = _computer_use_states.pop(source, None)
        if staged is None:
            return
        current = _computer_use_states.get(target)
        if current is None:
            staged.profile_home = target[1]
            _computer_use_states[target] = staged
            return
        current.backends.update(staged.backends)
        current.backend_init_locks.update(staged.backend_init_locks)
        current.backend_call_locks.update(staged.backend_call_locks)
        current.backend_permission_modes.update(staged.backend_permission_modes)
        current.session_auto_approve.update(staged.session_auto_approve)
        current.always_allow.update(staged.always_allow)
        current.aux_vision_route_cache.update(staged.aux_vision_route_cache)
        if current.default_backend is None:
            current.default_backend = staged.default_backend
        if current.approval_callback is None:
            current.approval_callback = staged.approval_callback


async def _activate_computer_use_scope() -> _ComputerUseScopeKey:
    lexical = _lexical_computer_use_profile_identity()
    loop = asyncio.get_running_loop()
    active = _computer_use_scope_context.get()
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
    scope: _ComputerUseScopeKey = (loop, canonical)
    with _computer_use_scope_lock:
        _computer_use_scope_aliases[(loop, lexical)] = scope
    _computer_use_scope_context.set((lexical, scope))
    _merge_computer_use_state((loop, lexical), scope)
    _merge_computer_use_state((_COMPUTER_USE_NO_LOOP, lexical), scope)
    _computer_use_state_for(scope)
    return scope


class _ScopedComputerUseDict(MutableMapping):
    """Dict-compatible active-profile view retained for private test hooks."""

    def __init__(self, field_name: str) -> None:
        self._field_name = field_name

    def _active(self) -> dict:
        return getattr(_computer_use_state(), self._field_name)

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


async def _finish_owned_backend_cleanup(
    cleanup: asyncio.Task[Any],
    *,
    error_message: str,
) -> None:
    """Finish owned teardown despite repeated cancellation of the caller."""
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            await asyncio.shield(cleanup)
            break
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised below
            if cleanup.cancelled():
                raise
            if cancellation is None:
                cancellation = exc
        except Exception:
            logger.debug(error_message, exc_info=True)
            break
    if cancellation is not None:
        raise cancellation


# ---------------------------------------------------------------------------
# Approval & safety
# ---------------------------------------------------------------------------

_approval_callback = None


def set_approval_callback(cb) -> None:
    """Register a callback for computer_use approval prompts (used by CLI).

    Matches the terminal_tool._approval_callback pattern. The callback
    receives (action, args, summary) and returns one of:
      "approve_once" | "approve_session" | "always_approve" | "deny".
    """
    global _approval_callback
    _approval_callback = cb
    _computer_use_state().approval_callback = cb


# Actions that read, not mutate. Always allowed.
_SAFE_ACTIONS = frozenset({
    "capture", "wait", "list_apps", "list_windows", "cua_browser_state",
})

# Actions that mutate user-visible state. Go through approval.
_DESTRUCTIVE_ACTIONS = frozenset({
    "click", "double_click", "right_click", "middle_click",
    "drag", "scroll", "type", "key", "set_value", "focus_app",
    "cua_browser_prepare", "cua_browser_navigate", "cua_browser_click",
    "cua_browser_type", "cua_browser_pointer", "cua_browser_dialog",
    "cua_browser_set_input_files", "cua_browser_download",
})

# Hard-blocked key combinations. Mirrored from #4562 — these are destructive
# regardless of approval level (e.g. logout kills the session Hermes runs in).
_BLOCKED_KEY_COMBOS = {
    frozenset({"cmd", "shift", "backspace"}),   # empty trash
    frozenset({"cmd", "option", "backspace"}),   # force delete
    frozenset({"cmd", "ctrl", "q"}),             # lock screen
    frozenset({"cmd", "shift", "q"}),            # log out
    frozenset({"cmd", "option", "shift", "q"}),  # force log out
    # Windows secure/session shortcuts. The Windows driver accepts Win-key
    # combos, and Alt is canonicalized to option below, so block the
    # destructive variants before any backend sees them.
    frozenset({"win", "l"}),
    frozenset({"ctrl", "option", "delete"}),
    frozenset({"ctrl", "option", "del"}),
    frozenset({"option", "f4"}),
}

_KEY_ALIASES = {
    "command": "cmd", "control": "ctrl", "alt": "option", "⌘": "cmd", "⌥": "option",
    "windows": "win", "super": "win", "meta": "win",
}


def _canon_key_combo(keys: str) -> frozenset:
    # Split on both "+" and "-": the cua-driver backend's _parse_key_combo
    # accepts hyphen-separated combos too, so "ctrl-alt-delete" executes as
    # the real destructive shortcut. Mirror its separators here, otherwise the
    # _BLOCKED_KEY_COMBOS gate is trivially bypassed with hyphen notation.
    parts = [p.strip().lower() for p in re.split(r"\s*[+\-]\s*", keys) if p.strip()]
    parts = [_KEY_ALIASES.get(p, p) for p in parts]
    return frozenset(parts)


# Dangerous text patterns for the `type` action. Same list as #4562.
_BLOCKED_TYPE_PATTERNS = [
    re.compile(r"curl\s+[^|]*\|\s*bash", re.IGNORECASE),
    re.compile(r"curl\s+[^|]*\|\s*sh", re.IGNORECASE),
    re.compile(r"wget\s+[^|]*\|\s*bash", re.IGNORECASE),
    re.compile(r"\bsudo\s+rm\s+-[rf]", re.IGNORECASE),
    re.compile(r"\brm\s+-rf\s+/\s*$", re.IGNORECASE),
    re.compile(r":\s*\(\)\s*\{\s*:\|:\s*&\s*\}", re.IGNORECASE),  # fork bomb
]


def _is_blocked_type(text: str) -> Optional[str]:
    for pat in _BLOCKED_TYPE_PATTERNS:
        if pat.search(text):
            return pat.pattern
    return None


# ---------------------------------------------------------------------------
# Backend selection — env-swappable for tests
# ---------------------------------------------------------------------------

# Per-Hermes-session cached backends. Each backend owns its own target,
# typed-browser route, grant namespace, and lifecycle. The dict-compatible
# views preserve upstream private test hooks while routing state by profile.
_AUX_VISION_ROUTE_CACHE: MutableMapping[Tuple[str, str], bool] = (
    _ScopedComputerUseDict("aux_vision_route_cache")
)
_backend: Optional[ComputerUseBackend] = None
_backends: MutableMapping[str, ComputerUseBackend] = _ScopedComputerUseDict(
    "backends"
)
_backend_init_locks: MutableMapping[str, asyncio.Lock] = _ScopedComputerUseDict(
    "backend_init_locks"
)
_backend_call_locks: MutableMapping[str, asyncio.Lock] = _ScopedComputerUseDict(
    "backend_call_locks"
)
_backend_permission_modes: MutableMapping[str, str] = _ScopedComputerUseDict(
    "backend_permission_modes"
)
_session_auto_approve: MutableMapping[str, bool] = _ScopedComputerUseDict(
    "session_auto_approve"
)
_always_allow: MutableMapping[str, set] = _ScopedComputerUseDict("always_allow")


def _backend_lock_for_profile() -> asyncio.Lock:
    return _computer_use_state().backend_lock


def _approval_lock_for_profile() -> asyncio.Lock:
    return _computer_use_state().approval_lock


def _cua_permission_mode(session_id: str) -> str:
    """Map Hermes's explicit approval bypass onto Cua's immutable mode."""
    try:
        from tools.approval import (
            get_current_session_key,
            is_approval_bypass_active_for_session,
        )

        if is_approval_bypass_active_for_session(session_id):
            return "unrestricted"
        current_key = get_current_session_key(default="")
        if current_key and is_approval_bypass_active_for_session(current_key):
            return "unrestricted"
    except Exception:
        pass
    return "standard"


async def _session_init_lock(session_id: str) -> asyncio.Lock:
    await _activate_computer_use_scope()
    async with _backend_lock_for_profile():
        return _backend_init_locks.setdefault(session_id, asyncio.Lock())


async def _get_backend(session_id: str = "") -> ComputerUseBackend:
    global _backend
    sid = str(session_id or "")
    init_lock = await _session_init_lock(sid)
    state = _computer_use_state()
    async with init_lock:
        while True:
            # Re-read the authoritative mode on every iteration. Stopping a
            # stale backend may race a policy edge, and upstream deliberately
            # resolves that edge before constructing the replacement.
            permission_mode = _cua_permission_mode(sid)
            stale_backend: Optional[ComputerUseBackend] = None
            stale_call_lock: Optional[asyncio.Lock] = None

            async with _backend_lock_for_profile():
                if (
                    sid == ""
                    and _backend is not None
                    and state.default_backend is None
                    and sid not in _backends
                ):
                    # Compatibility staging for callers/tests that assign the
                    # historical private singleton before the first await.
                    state.default_backend = _backend
                    _backend = None
                if (
                    sid == ""
                    and state.default_backend is not None
                    and sid not in _backends
                ):
                    _backends[sid] = state.default_backend
                    _backend_call_locks[sid] = asyncio.Lock()
                    _backend_permission_modes[sid] = permission_mode
                cached = _backends.get(sid)
                if cached is not None:
                    if _backend_permission_modes.get(sid, "standard") == permission_mode:
                        return cached
                    stale_backend = _backends.pop(sid)
                    stale_call_lock = _backend_call_locks.pop(sid, None)
                    _backend_permission_modes.pop(sid, None)
                    if sid == "":
                        state.default_backend = None

            if stale_backend is not None:
                try:
                    if stale_call_lock is not None:
                        async with stale_call_lock:
                            await stale_backend.stop()
                    else:
                        await stale_backend.stop()
                except Exception:
                    logger.debug(
                        "computer_use stale backend teardown failed for session %s",
                        sid,
                        exc_info=True,
                    )
                continue

            backend_name = os.environ.get(
                "HERMES_COMPUTER_USE_BACKEND", "cua"
            ).lower()
            if backend_name in {"cua", "cua-driver", ""}:
                from tools.computer_use.cua_backend import CuaDriverBackend

                backend: ComputerUseBackend = CuaDriverBackend(
                    permission_mode=permission_mode
                )
            elif backend_name == "noop":  # pragma: no cover
                backend = _NoopBackend()
            else:
                raise RuntimeError(
                    f"Unknown HERMES_COMPUTER_USE_BACKEND={backend_name!r}"
                )

            try:
                await backend.start()
                async with _backend_lock_for_profile():
                    _backends[sid] = backend
                    _backend_call_locks[sid] = asyncio.Lock()
                    _backend_permission_modes[sid] = permission_mode
                    if sid == "":
                        state.default_backend = backend
            except BaseException:
                cleanup = asyncio.create_task(
                    backend.stop(),
                    name=f"computer-use-init-cleanup-{sid or 'default'}",
                )
                await _finish_owned_backend_cleanup(
                    cleanup,
                    error_message=(
                        "computer_use backend initialization cleanup failed"
                    ),
                )
                raise
            return backend


async def release_computer_use_session(session_id: str) -> bool:
    """Release one session-owned computer-use backend."""
    global _backend
    sid = str(session_id or "")
    init_lock = await _session_init_lock(sid)
    state = _computer_use_state()
    async with init_lock:
        async with _backend_lock_for_profile():
            if sid == "" and _backend is not None and state.default_backend is None:
                state.default_backend = _backend
                _backend = None
            backend = _backends.pop(sid, None)
            call_lock = _backend_call_locks.pop(sid, None)
            _backend_permission_modes.pop(sid, None)
            if sid == "" and backend is None:
                backend = state.default_backend
            if sid == "" and state.default_backend is backend:
                state.default_backend = None

        async with _approval_lock_for_profile():
            _session_auto_approve.pop(sid, None)
            _always_allow.pop(sid, None)

        if backend is None:
            return False
        try:
            if call_lock is not None:
                async with call_lock:
                    await backend.stop()
            else:
                await backend.stop()
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(
                backend.stop(),
                name=f"computer-use-release-{sid or 'default'}",
            )
            await _finish_owned_backend_cleanup(
                cleanup,
                error_message="computer_use cancelled release cleanup failed",
            )
            raise
        except Exception:
            logger.debug(
                "computer_use backend release failed for session %s",
                sid,
                exc_info=True,
            )
        return True


async def _shutdown_all_backends() -> None:
    global _backend
    await _activate_computer_use_scope()
    loop = asyncio.get_running_loop()
    with _computer_use_scope_lock:
        states = tuple(
            state
            for scope, state in _computer_use_states.items()
            if scope[0] is loop
        )

    unique: Dict[int, Tuple[ComputerUseBackend, Optional[asyncio.Lock]]] = {}
    seen_states: set[int] = set()
    for state in states:
        if id(state) in seen_states:
            continue
        seen_states.add(id(state))
        async with state.backend_lock:
            unique.update(
                {
                    id(backend): (backend, state.backend_call_locks.get(sid))
                    for sid, backend in state.backends.items()
                }
            )
            if state.default_backend is not None:
                unique.setdefault(
                    id(state.default_backend),
                    (state.default_backend, state.backend_call_locks.get("")),
                )
            state.default_backend = None
            state.backends.clear()
            state.backend_call_locks.clear()
            state.backend_permission_modes.clear()
            state.backend_init_locks.clear()

        async with state.approval_lock:
            state.session_auto_approve.clear()
            state.always_allow.clear()

    # Historical singleton staging is not profile-owned until an awaited
    # backend boundary consumes it. Test/process-final teardown still owns it.
    if _backend is not None:
        unique.setdefault(id(_backend), (_backend, None))
        _backend = None

    async def _stop_one(
        backend: ComputerUseBackend,
        call_lock: Optional[asyncio.Lock],
    ) -> None:
        try:
            if call_lock is not None:
                async with call_lock:
                    await backend.stop()
            else:
                await backend.stop()
        except Exception as exc:
            logger.debug("cua-driver teardown failed: %s", exc)

    async def _close_all() -> None:
        await asyncio.gather(
            *(_stop_one(backend, lock) for backend, lock in unique.values())
        )

    cleanup = asyncio.create_task(_close_all(), name="computer-use-shutdown")
    try:
        await asyncio.shield(cleanup)
    except asyncio.CancelledError:
        await _finish_owned_backend_cleanup(
            cleanup,
            error_message="computer_use shutdown cleanup failed",
        )
        raise


async def reset_backend_for_tests() -> None:  # pragma: no cover
    """Test helper: tear down cached backends and per-session state."""
    await _shutdown_all_backends()
    loop = asyncio.get_running_loop()
    with _computer_use_scope_lock:
        states = tuple(
            state
            for scope, state in _computer_use_states.items()
            if scope[0] is loop
        )
    for state in states:
        state.aux_vision_route_cache.clear()


class _NoopBackend(ComputerUseBackend):  # pragma: no cover
    """Test/CI stub. Records calls; returns trivial results."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        self._started = False

    async def start(self) -> None: self._started = True
    async def stop(self) -> None: self._started = False
    async def is_available(self) -> bool: return True

    async def capture(
        self,
        mode: str = "som",
        app: Optional[str] = None,
        pid: Optional[int] = None,
        window_id: Optional[int] = None,
    ) -> CaptureResult:
        self.calls.append((
            "capture",
            {"mode": mode, "app": app, "pid": pid, "window_id": window_id},
        ))
        return CaptureResult(mode=mode, width=1024, height=768, png_b64=None,
                             elements=[], app=app or "", window_title="")

    async def click(self, **kw) -> ActionResult:
        self.calls.append(("click", kw))
        return ActionResult(ok=True, action="click")

    async def drag(self, **kw) -> ActionResult:
        self.calls.append(("drag", kw))
        return ActionResult(ok=True, action="drag")

    async def scroll(self, **kw) -> ActionResult:
        self.calls.append(("scroll", kw))
        return ActionResult(ok=True, action="scroll")

    async def type_text(self, text: str, **kw) -> ActionResult:
        self.calls.append(("type", {"text": text, **kw}))
        return ActionResult(ok=True, action="type")

    async def key(self, keys: str, **kw) -> ActionResult:
        self.calls.append(("key", {"keys": keys, **kw}))
        return ActionResult(ok=True, action="key")

    async def list_apps(self) -> List[Dict[str, Any]]:
        self.calls.append(("list_apps", {}))
        return []

    async def list_windows(self) -> List[Dict[str, Any]]:
        self.calls.append(("list_windows", {}))
        return []

    async def focus_app(self, app: str, raise_window: bool = False) -> ActionResult:
        self.calls.append(("focus_app", {"app": app, "raise": raise_window}))
        return ActionResult(ok=True, action="focus_app")

    async def set_value(self, value: str, element: Optional[int] = None) -> ActionResult:
        self.calls.append(("set_value", {"value": value, "element": element}))
        return ActionResult(ok=True, action="set_value")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

async def handle_computer_use(args: Dict[str, Any], **kwargs) -> Any:
    """Main entry point — dispatched by tools.registry.

    Returns either a JSON string (text-only) or a dict marked `_multimodal`
    (image + summary) which run_agent.py wraps into the tool message.
    """
    action = (args.get("action") or "").strip().lower()
    if not action:
        return json.dumps({"error": "missing `action`"})
    # Per-run key for approval-state and daemon-mode isolation across
    # concurrent sessions.
    session_id = str(kwargs.get("session_id") or "")
    await _activate_computer_use_scope()

    # Safety: validate actions before approval prompt.
    if action in {"type", "cua_browser_type"}:
        text = args.get("text", "")
        pat = _is_blocked_type(text)
        if pat:
            return json.dumps({
                "error": f"blocked pattern in type text: {pat!r}",
                "hint": "Dangerous shell patterns cannot be typed via computer_use.",
            })

    if action == "key":
        keys = args.get("keys", "")
        combo = _canon_key_combo(keys)
        for blocked in _BLOCKED_KEY_COMBOS:
            if blocked.issubset(combo) and len(blocked) <= len(combo):
                return json.dumps({
                    "error": f"blocked key combo: {sorted(blocked)}",
                    "hint": "Destructive system shortcuts are hard-blocked.",
                })

    if args.get("bring_to_front") and args.get("delivery_mode") != "foreground":
        return json.dumps({
            "error": "bring_to_front requires delivery_mode='foreground'",
            "code": "bring_to_front_requires_foreground",
        })

    # Approval gate (destructive actions only).
    if action in _DESTRUCTIVE_ACTIONS:
        err = await _request_approval(action, args, session_id)
        if err is not None:
            return err
    # Persistent focus is a separate, visible side effect from the input
    # itself. Keep its approval scope distinct even when the input rung has
    # already been approved for this session.
    if args.get("bring_to_front") or (
        action == "focus_app" and args.get("raise_window")
    ):
        err = await _request_approval("bring_to_front", args, session_id)
        if err is not None:
            return err

    # Dispatch to backend.
    try:
        backend = await _get_backend(session_id=session_id)
    except Exception as e:
        return json.dumps({
            "error": f"computer_use backend unavailable: {e}",
            "hint": "If the cua-driver binary is missing, run `hermes computer-use install`. "
                    "If a Python dependency is missing, the error above shows the exact install command.",
        })

    try:
        async with _backend_lock_for_profile():
            call_lock = _backend_call_locks.setdefault(session_id, asyncio.Lock())
        async with call_lock:
            return await _dispatch(backend, action, args)
    except Exception as e:
        logger.exception("computer_use %s failed", action)
        return json.dumps({"error": f"{action} failed: {e}"})


async def _request_approval(action: str, args: Dict[str, Any],
                      session_id: str = "") -> Optional[str]:
    """Return None if approved, or a JSON error string if denied.

    Approval is scoped by (action, delivery_mode) AND by session_id.
    Foreground delivery is a visible focus change, so a prior background
    approval — even ``approve_session`` on the same action — must NOT
    silently authorize it (NousResearch/hermes-agent#67052).
    ``always_approve`` (the blanket "auto-approve everything" unlock) still
    covers foreground, since the user explicitly opted into unattended
    operation. State is keyed on session_id so concurrent runs don't leak
    unlocks into one another.
    """
    await _activate_computer_use_scope()
    is_foreground = args.get("delivery_mode") == "foreground"
    scope_key = (action, "foreground" if is_foreground else "background")
    async with _approval_lock_for_profile():
        if _session_auto_approve.get(session_id):
            return None
        if scope_key in _always_allow.get(session_id, set()):
            return None
    cb = _computer_use_state().approval_callback
    if cb is None:
        # No CLI approval wired — default allow. Gateway approval is handled
        # one layer out via the normal tool-approval infra.
        return None
    summary = _summarize_action(action, args)
    try:
        verdict = cb(action, args, summary)
        if inspect.isawaitable(verdict):
            verdict = await verdict
    except Exception as e:
        logger.warning("approval callback failed: %s", e)
        verdict = "deny"
    if verdict == "approve_once":
        return None
    if verdict == "approve_session" or verdict == "always_approve":
        async with _approval_lock_for_profile():
            _always_allow.setdefault(session_id, set()).add(scope_key)
            if verdict == "always_approve":
                _session_auto_approve[session_id] = True
        return None
    if verdict == "timeout":
        return json.dumps({
            "error": (
                "approval prompt timed out — the user did not respond. "
                "Silence is not consent; do not retry without the user."
            ),
            "action": action,
        })
    return json.dumps({"error": "denied by user", "action": action})


def _summarize_action(action: str, args: Dict[str, Any]) -> str:
    fg = " [FOREGROUND — briefly raises the window / changes focus]" \
        if args.get("delivery_mode") == "foreground" else ""
    if action in {"click", "double_click", "right_click", "middle_click"}:
        if args.get("element") is not None:
            return f"{action} element #{args['element']}{fg}"
        coord = args.get("coordinate")
        if coord:
            return f"{action} at {tuple(coord)}{fg}"
        return action + fg
    if action == "drag":
        src = args.get("from_element") or args.get("from_coordinate")
        dst = args.get("to_element") or args.get("to_coordinate")
        return f"drag {src} → {dst}{fg}"
    if action == "scroll":
        return f"scroll {args.get('direction', '?')} x{args.get('amount', 3)}{fg}"
    if action == "type":
        text = args.get("text", "")
        return f"type {text[:60]!r}" + ("..." if len(text) > 60 else "") + fg
    if action == "key":
        return f"key {args.get('keys', '')!r}{fg}"
    if action == "focus_app":
        return f"focus {args.get('app', '')!r}" + (" (raise)" if args.get("raise_window") else "")
    return action + fg


async def _dispatch(backend: ComputerUseBackend, action: str, args: Dict[str, Any]) -> Any:
    capture_after = bool(args.get("capture_after"))

    if action == "capture":
        mode = str(args.get("mode", "som"))
        if mode not in {"som", "vision", "ax"}:
            return json.dumps({"error": f"bad mode {mode!r}; use som|vision|ax"})
        capture_kwargs: Dict[str, Any] = {"mode": mode, "app": args.get("app")}
        if args.get("pid") is not None or args.get("window_id") is not None:
            capture_kwargs.update({
                "pid": args.get("pid"),
                "window_id": args.get("window_id"),
            })
        cap = await backend.capture(**capture_kwargs)
        return await _capture_response(cap, max_elements=_coerce_max_elements(args.get("max_elements")))

    if action == "wait":
        seconds = float(args.get("seconds", 1.0))
        res = await backend.wait(seconds)
        return _text_response(res)

    if action == "list_apps":
        apps = await backend.list_apps()
        return json.dumps({"apps": apps, "count": len(apps)})

    if action == "list_windows":
        windows = await backend.list_windows()
        return json.dumps({"windows": windows, "count": len(windows)})

    if action == "focus_app":
        app = args.get("app")
        if not app:
            return json.dumps({"error": "focus_app requires `app`"})
        res = await backend.focus_app(app, raise_window=bool(args.get("raise_window")))
        return await _maybe_follow_capture(backend, res, capture_after)

    # cua-driver's typed browser surface is namespaced inside the existing
    # computer_use tool so it cannot collide with native browser/MCP tools.
    # The backend owns the opaque driver session, target, tab and ref state;
    # none of those capabilities can be supplied across Hermes sessions.
    if action == "cua_browser_state":
        state_args: Dict[str, Any] = {}
        for public, internal in (
            ("pid", "pid"),
            ("window_id", "window_id"),
            ("tab_id", "tab_id"),
            ("snapshot_format", "snapshot_format"),
            ("query", "query"),
            ("scope_ref", "scope_ref"),
            ("continuation", "continuation"),
        ):
            if args.get(public) is not None:
                state_args[internal] = args[public]
        return json.dumps(await backend.typed_browser_state(**state_args))

    if action == "cua_browser_prepare":
        return json.dumps(await backend.typed_browser_prepare(
            pid=args.get("pid"),
            window_id=args.get("window_id"),
            profile_mode=args.get("profile_mode", "isolated_new"),
            profile_name=args.get("profile_name"),
            allow_launch=bool(args.get("allow_launch")),
        ))

    browser_tools = {
        "cua_browser_navigate": "browser_navigate",
        "cua_browser_click": "browser_click",
        "cua_browser_type": "browser_type",
        "cua_browser_pointer": "browser_pointer",
        "cua_browser_dialog": "browser_dialog",
        "cua_browser_set_input_files": "browser_set_input_files",
        "cua_browser_download": "browser_download",
    }
    driver_tool = browser_tools.get(action)
    if driver_tool is not None:
        call_args: Dict[str, Any] = {}
        allowed_fields = {
            "browser_navigate": ("url",),
            "browser_click": ("ref", "input_route", "x", "y"),
            "browser_type": ("ref", "text"),
            "browser_pointer": (
                "ref", "destination_ref", "input_route", "x", "y",
                "to_x", "to_y", "delta_x", "delta_y",
            ),
            "browser_dialog": (
                "dialog_id", "prompt_text", "delivery_mode",
            ),
            "browser_set_input_files": ("ref", "files"),
            "browser_download": ("ref", "destination_root"),
        }
        for field in allowed_fields[driver_tool]:
            if args.get(field) is not None:
                call_args[field] = args[field]
        if (
            driver_tool in {"browser_click", "browser_pointer"}
            and args.get("coordinate") is not None
        ):
            coordinate = args["coordinate"]
            if isinstance(coordinate, (list, tuple)) and len(coordinate) == 2:
                call_args["x"], call_args["y"] = coordinate
        pointer_action = args.get("browser_pointer_action")
        dialog_action = args.get("browser_dialog_action")
        # Direct adapter callers may omit the public discriminator from args;
        # retain this narrow compatibility path without making it usable to
        # override the namespaced action selected by handle_computer_use.
        nested_action = args.get("action")
        if nested_action not in browser_tools:
            if driver_tool == "browser_pointer" and pointer_action is None:
                pointer_action = nested_action
            if driver_tool == "browser_dialog" and dialog_action is None:
                dialog_action = nested_action
        if pointer_action is not None:
            call_args["action"] = pointer_action
        if dialog_action is not None:
            call_args["action"] = dialog_action
        if args.get("browser_type_mode") is not None:
            call_args["mode"] = args["browser_type_mode"]
        return json.dumps(await backend.typed_browser_action(
            driver_tool,
            tab_id=args.get("tab_id"),
            args=call_args,
        ))

    # delivery_mode / bring_to_front thread through every input action so the
    # model can escalate background → foreground per cua-driver's ladder.
    delivery_mode = args.get("delivery_mode")
    bring_to_front = bool(args.get("bring_to_front"))

    if action in {"click", "double_click", "right_click", "middle_click"}:
        button = args.get("button")
        click_count = 1
        if action == "double_click":
            click_count = 2
        elif action == "right_click":
            button = "right"
        elif action == "middle_click":
            button = "middle"
        else:
            button = button or "left"
        element = args.get("element")
        coord = args.get("coordinate") or (None, None)
        x, y = (coord[0], coord[1]) if coord and coord[0] is not None else (None, None)
        res = await backend.click(
            element=element if element is not None else None,
            x=x, y=y, button=button or "left", click_count=click_count,
            modifiers=args.get("modifiers"),
            delivery_mode=delivery_mode, bring_to_front=bring_to_front,
        )
        return await _maybe_follow_capture(backend, res, capture_after)

    if action == "drag":
        has_elements = args.get("from_element") is not None and args.get("to_element") is not None
        has_coords = args.get("from_coordinate") and args.get("to_coordinate")
        if not has_elements and not has_coords:
            return json.dumps({
                "error": "drag requires from_coordinate/to_coordinate or from_element/to_element",
            })
        res = await backend.drag(
            from_element=args.get("from_element"),
            to_element=args.get("to_element"),
            from_xy=tuple(args["from_coordinate"]) if args.get("from_coordinate") else None,
            to_xy=tuple(args["to_coordinate"]) if args.get("to_coordinate") else None,
            button=args.get("button", "left"),
            modifiers=args.get("modifiers"),
            delivery_mode=delivery_mode, bring_to_front=bring_to_front,
        )
        return await _maybe_follow_capture(backend, res, capture_after)

    if action == "scroll":
        coord = args.get("coordinate") or (None, None)
        res = await backend.scroll(
            direction=args.get("direction", "down"),
            amount=int(args.get("amount", 3)),
            element=args.get("element"),
            x=coord[0] if coord and coord[0] is not None else None,
            y=coord[1] if coord and coord[1] is not None else None,
            modifiers=args.get("modifiers"),
            delivery_mode=delivery_mode, bring_to_front=bring_to_front,
        )
        return await _maybe_follow_capture(backend, res, capture_after)

    if action == "type":
        res = await backend.type_text(args.get("text", ""),
                                delivery_mode=delivery_mode, bring_to_front=bring_to_front)
        return await _maybe_follow_capture(backend, res, capture_after)

    if action == "key":
        res = await backend.key(args.get("keys", ""),
                          delivery_mode=delivery_mode, bring_to_front=bring_to_front)
        return await _maybe_follow_capture(backend, res, capture_after)

    if action == "set_value":
        value = args.get("value")
        if value is None:
            return json.dumps({"error": "set_value requires `value`"})
        res = await backend.set_value(value=str(value), element=args.get("element"))
        return await _maybe_follow_capture(backend, res, capture_after)

    return json.dumps({"error": f"unknown action {action!r}"})


# ---------------------------------------------------------------------------
# Response shaping
# ---------------------------------------------------------------------------

def _classify_action_result(res: ActionResult) -> Dict[str, Any]:
    """Choose the next ladder step from semantic evidence, in precedence order.

    An escalation recommendation is advisory. It never overrides a confirmed
    effect and it never turns an unverifiable action into permission to repeat
    input. The model must first obtain fresh evidence.
    """
    if res.effect == "confirmed" or res.verified is True:
        return {"decision": "done"}
    if res.effect == "unverifiable":
        return {"decision": "verify_fresh_state"}
    if res.effect == "suspected_noop" or not res.ok or res.code is not None:
        decision: Dict[str, Any] = {"decision": "escalate"}
        if isinstance(res.escalation, dict):
            decision["recommended"] = res.escalation.get("recommended")
        return decision
    # Transport success without semantic proof is not proof of effect.
    return {"decision": "verify_fresh_state"}


def _action_payload(res: ActionResult) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"ok": res.ok, "action": res.action}
    if res.message:
        payload["message"] = res.message
    # Surface cua-driver's structured verdict additively so the model can
    # follow the verify → escalate ladder. Only include fields the driver
    # actually returned (None = old driver / not carried). ok is transport
    # success; effect/escalation are the semantic verdict.
    if res.verified is not None:
        payload["verified"] = res.verified
    if res.effect is not None:
        payload["effect"] = res.effect
    if res.escalation is not None:
        payload["escalation"] = res.escalation
    if res.path is not None:
        payload["path"] = res.path
    if res.degraded is not None:
        payload["degraded"] = res.degraded
    if res.delivery_mode is not None:
        payload["delivery_mode"] = res.delivery_mode
    if res.code is not None:
        payload["code"] = res.code
    if res.meta:
        payload["meta"] = res.meta
    payload["verdict"] = _classify_action_result(res)
    return payload


def _text_response(res: ActionResult) -> str:
    return json.dumps(_action_payload(res))


# Default cap for the AX `elements` array returned by capture. Dense UIs
# (Electron apps, Obsidian, JetBrains IDEs) can publish 500+ AX nodes, which
# can exhaust session context after a single capture. The model-facing
# `max_elements` argument lets callers raise this when they need the full tree.
_DEFAULT_MAX_ELEMENTS = 100
# Hard upper bound on caller-supplied `max_elements`. Without this, a tool
# call passing a very large integer would silently disable the safeguard and
# reintroduce the original unbounded behavior.
_MAX_ALLOWED_MAX_ELEMENTS = 1000
_MIN_PROVIDER_IMAGE_DIMENSION = 8


def _image_dimensions_from_b64(image_b64: str) -> Optional[Tuple[int, int]]:
    """Return (width, height) for common inline screenshot formats.

    Some providers reject images below 8x8 before the model sees the tool
    result. Inspecting the encoded bytes here lets computer_use fall back to
    its AX/SOM text payload instead of sending an unusable placeholder.
    """
    if not image_b64:
        return None
    try:
        raw = base64.b64decode(image_b64, validate=False)
    except Exception:
        return None

    # PNG: signature + IHDR width/height.
    if raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) >= 24:
        try:
            width, height = struct.unpack(">II", raw[16:24])
            return int(width), int(height)
        except Exception:
            return None

    # JPEG: scan for SOF markers that carry dimensions.
    if raw.startswith(b"\xff\xd8") and len(raw) > 4:
        i = 2
        while i + 9 < len(raw):
            if raw[i] != 0xFF:
                i += 1
                continue
            marker = raw[i + 1]
            i += 2
            while marker == 0xFF and i < len(raw):
                marker = raw[i]
                i += 1
            if marker in {0xD8, 0xD9}:
                continue
            if marker == 0xDA:
                break
            if i + 2 > len(raw):
                break
            segment_len = int.from_bytes(raw[i:i + 2], "big")
            if segment_len < 2 or i + segment_len > len(raw):
                break
            if marker in {
                0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
            } and segment_len >= 7:
                height = int.from_bytes(raw[i + 3:i + 5], "big")
                width = int.from_bytes(raw[i + 5:i + 7], "big")
                return int(width), int(height)
            i += segment_len
    return None


def _coerce_max_elements(value: Any) -> int:
    """Validate the caller-supplied ``max_elements``.

    Falls back to :data:`_DEFAULT_MAX_ELEMENTS` for missing / non-integer /
    sub-1 inputs so the cap can never be silently disabled by a malformed
    tool-call argument. Clamps oversized values to
    :data:`_MAX_ALLOWED_MAX_ELEMENTS` so a caller cannot bypass the
    safeguard by passing a very large integer.
    """
    if value is None:
        return _DEFAULT_MAX_ELEMENTS
    try:
        n = int(value)
    except (TypeError, ValueError):
        return _DEFAULT_MAX_ELEMENTS
    if n < 1:
        return _DEFAULT_MAX_ELEMENTS
    if n > _MAX_ALLOWED_MAX_ELEMENTS:
        return _MAX_ALLOWED_MAX_ELEMENTS
    return n


async def _capture_response(cap: CaptureResult, max_elements: int = _DEFAULT_MAX_ELEMENTS) -> Any:
    total_elements = len(cap.elements)
    visible_elements = cap.elements[:max_elements]
    truncated_elements = max(0, total_elements - len(visible_elements))
    image_dimensions = _image_dimensions_from_b64(cap.png_b64 or "") if cap.png_b64 else None
    response_width = image_dimensions[0] if image_dimensions else cap.width
    response_height = image_dimensions[1] if image_dimensions else cap.height
    image_too_small = bool(
        image_dimensions
        and (
            image_dimensions[0] < _MIN_PROVIDER_IMAGE_DIMENSION
            or image_dimensions[1] < _MIN_PROVIDER_IMAGE_DIMENSION
        )
    )

    # Index only what's actually surfaced in the response — otherwise the
    # human-readable summary references element indices the model cannot
    # find in the JSON `elements` array (e.g. max_elements=10 vs the default
    # 40-line index window).
    element_index = _format_elements(visible_elements)
    summary_lines = [
        f"capture mode={cap.mode} {response_width}x{response_height}"
        + (f" app={cap.app}" if cap.app else "")
        + (f" window={cap.window_title!r}" if cap.window_title else ""),
        f"{total_elements} interactable element(s):",
    ]
    if element_index:
        summary_lines.extend(element_index)
    # Multimodal and AX paths both reference `summary`; build it once up-front
    # so the aux-vision routing branch (which fires before either path is
    # selected) has a valid value to hand to _route_capture_through_aux_vision.
    # The AX path appends the "truncated to N of M" note to summary_lines
    # below and rebuilds; the multimodal path keeps this version untouched.
    if image_too_small and image_dimensions is not None:
        summary_lines.append(
            f"  (screenshot omitted: {image_dimensions[0]}x{image_dimensions[1]} "
            f"is below the {_MIN_PROVIDER_IMAGE_DIMENSION}x{_MIN_PROVIDER_IMAGE_DIMENSION} "
            "provider minimum)"
        )
    summary = "\n".join(summary_lines)

    if cap.png_b64 and cap.mode != "ax" and not image_too_small:
        # Decide whether to hand the screenshot to the auxiliary.vision
        # pipeline (text-only result) or keep the multimodal envelope (main
        # model handles vision natively). Issue #24015: previously the
        # multimodal envelope was returned unconditionally, so non-vision
        # main models tripped HTTP 404 / 400 at the provider boundary even
        # when auxiliary.vision was explicitly configured to handle this.
        if await _should_route_through_aux_vision():
            routed = await _route_capture_through_aux_vision(cap, summary)
            if routed is not None:
                return routed
            # Aux routing was requested but failed (vision node down, aux call
            # raised, empty analysis, etc.). Routing being requested means the
            # main model may not be able to consume images; falling through to
            # the multimodal envelope can break the capture with a provider
            # error. Degrade to the AX/SOM text payload instead so element
            # indices remain usable while vision is unavailable.
            summary_lines.append(
                "  (vision unavailable: the auxiliary vision model could not "
                "be reached; screenshot omitted. Element-index actions still "
                "work — drive via the element list above.)"
            )
            if truncated_elements:
                summary_lines.append(
                    f"  (response truncated to {len(visible_elements)} of "
                    f"{total_elements} elements; raise max_elements or pass "
                    "app= to narrow)"
                )
            payload = {
                "mode": cap.mode,
                "width": response_width,
                "height": response_height,
                "app": cap.app,
                "window_title": cap.window_title,
                "elements": [_element_to_dict(e) for e in visible_elements],
                "total_elements": total_elements,
                "summary": "\n".join(summary_lines),
                "vision_unavailable": True,
            }
            if truncated_elements:
                payload["truncated_elements"] = truncated_elements
            return json.dumps(payload)

        # Prefer the explicit MIME type cua-driver attaches to its image
        # parts (Surface 7 of NousResearch/hermes-agent#47072 — trycua/cua#1961
        # made `mimeType` part of every MCP image-part response). Fall back
        # to base64-prefix sniffing for older cua-driver builds that didn't
        # carry the field. JPEG base64 starts with /9j/; PNG with iVBOR.
        _mime = cap.image_mime_type
        if not _mime:
            _b64_prefix = cap.png_b64[:8]
            _mime = "image/jpeg" if _b64_prefix.startswith("/9j/") else "image/png"
        # The multimodal response carries the screenshot, not the AX
        # elements array, so a "response truncated to N of M elements"
        # note would be inaccurate — skip it on this branch.
        return {
            "_multimodal": True,
            "content": [
                {"type": "text", "text": summary},
                {"type": "image_url",
                 "image_url": {"url": f"data:{_mime};base64,{cap.png_b64}"}},
            ],
            "text_summary": summary,
            "meta": {"mode": cap.mode, "width": response_width, "height": response_height,
                     "elements": total_elements, "png_bytes": cap.png_bytes_len},
        }
    # AX-only (or image-missing fallback): text path actually carries the
    # `elements` array, so the truncation note applies here.
    if truncated_elements:
        summary_lines.append(
            f"  (response truncated to {len(visible_elements)} of {total_elements} elements; "
            f"raise max_elements or pass app= to narrow)"
        )
    summary = "\n".join(summary_lines)
    payload: Dict[str, Any] = {
        "mode": cap.mode,
        "width": response_width,
        "height": response_height,
        "app": cap.app,
        "window_title": cap.window_title,
        "elements": [_element_to_dict(e) for e in visible_elements],
        "total_elements": total_elements,
        "summary": summary,
    }
    if truncated_elements:
        payload["truncated_elements"] = truncated_elements
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# auxiliary.vision routing for captured screenshots (#24015)
# ---------------------------------------------------------------------------

# Longest image side handed to the aux vision model. Full-resolution desktop
# captures tokenize heavily and can overflow small local-model context windows;
# ~1456px keeps SOM badges legible while cutting per-capture vision latency.
_MAX_VISION_DIM = 1456


def _shrink_capture_for_vision(raw: bytes, ext: str,
                               max_dim: int = _MAX_VISION_DIM) -> bytes:
    """Downscale encoded image bytes so the longest side is <= max_dim.

    Returns the original bytes unchanged when the image already fits or when
    Pillow is unavailable/fails — no worse than the pre-shrink behavior.
    """
    try:
        from io import BytesIO
        from PIL import Image
        img = Image.open(BytesIO(raw))
        if max(img.size) <= max_dim:
            return raw
        img.thumbnail((max_dim, max_dim))
        out = BytesIO()
        img.save(out, format="JPEG" if ext == ".jpg" else "PNG")
        return out.getvalue()
    except Exception as exc:
        logger.debug("computer_use: vision downscale skipped: %s", exc)
        return raw

async def _should_route_through_aux_vision() -> bool:
    """Return True when ``_capture_response`` should hand the PNG to aux vision.

    Reads the active main provider/model and the loaded config and asks the
    routing helper. Any failure (config import, runtime override missing,
    etc.) returns False so the existing multimodal envelope continues to be
    returned — fail open on the routing decision so a broken config can
    never silently drop the screenshot for vision-capable main models.
    """
    await _activate_computer_use_scope()
    try:
        from agent.auxiliary_client import _read_main_model, _read_main_provider
        from hermes_cli.config import load_config_readonly
        from tools.computer_use.vision_routing import (
            should_route_capture_to_aux_vision,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use: aux-vision routing import failed: %s", exc)
        return False
    try:
        provider = await _read_main_provider() or ""
        model = await _read_main_model() or ""
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use: aux-vision routing config read failed: %s", exc)
        return False
    cache_key = (str(provider), str(model))
    cached = _AUX_VISION_ROUTE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        cfg = await load_config_readonly()
        decision = bool(should_route_capture_to_aux_vision(provider, model, cfg))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use: aux-vision routing decision failed: %s", exc)
        return False
    _AUX_VISION_ROUTE_CACHE[cache_key] = decision
    return decision


async def _capture_after_mode() -> str:
    """Mode for ``capture_after`` follow-ups. Default ``som`` (screenshot)."""
    try:
        from hermes_cli.config import load_config_readonly

        raw = ((await load_config_readonly() or {}).get("computer_use") or {}).get(
            "capture_after_mode", "som"
        )
    except Exception:
        return "som"
    mode = str(raw or "som").strip().lower()
    return mode if mode in {"som", "vision", "ax"} else "som"


async def _route_capture_through_aux_vision(
    cap: CaptureResult,
    summary: str,
) -> Optional[str]:
    """Pre-analyse the captured PNG via ``vision_analyze`` and return a text result.

    The captured base64 PNG is materialised to ``$HERMES_HOME/cache/vision/``
    and handed to ``vision_analyze_tool`` with a generic describe prompt.
    The resulting text description is merged into the existing AX/SOM
    summary so the main model receives a single text payload that mentions
    every interactable element AND a description of what the screenshot
    looked like.

    Returns:
      A JSON-encoded text response on success.
      ``None`` on failure (caller falls back to the multimodal envelope).
    """
    if not cap.png_b64:
        return None
    try:
        import base64 as _base64
        import uuid as _uuid

        from hermes_constants import get_hermes_dir
        from tools.vision_tools import vision_analyze_tool
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use: aux-vision import failed: %s", exc)
        return None

    temp_image_path = None
    try:
        try:
            raw = _base64.b64decode(cap.png_b64, validate=False)
        except Exception as exc:
            logger.debug("computer_use: failed to decode capture base64: %s", exc)
            return None

        # Pick an extension that matches the on-disk bytes so vision_analyze's
        # MIME sniffing returns the right content-type.
        # Surface 7: prefer the explicit MIME type cua-driver supplied.
        _mime_for_ext = cap.image_mime_type or ""
        if _mime_for_ext == "image/jpeg" or (not _mime_for_ext and cap.png_b64[:8].startswith("/9j/")):
            ext = ".jpg"
        else:
            ext = ".png"
        cache_dir = await get_hermes_dir("cache/vision", "temp_vision_images")
        await aiofiles.os.makedirs(cache_dir, exist_ok=True)
        temp_image_path = cache_dir / f"computer_use_{_uuid.uuid4().hex}{ext}"
        raw = _shrink_capture_for_vision(raw, ext)
        async with aiofiles.open(temp_image_path, "wb") as stream:
            await stream.write(raw)

        prompt = (
            "Describe what is visible in this desktop application screenshot in "
            "concise but specific terms. Mention the app name and window "
            "title if visible, the overall layout, any labelled buttons, "
            "menus or text fields, and any prominent text content the user "
            "would need to know about. Do not invent details that are not "
            "actually visible.\n\n"
            f"AX/SOM index for cross-reference:\n{summary}"
        )

        result_json = await vision_analyze_tool(str(temp_image_path), prompt)
    except Exception as exc:
        logger.warning(
            "computer_use: auxiliary.vision pre-analysis failed (%s); "
            "returning to caller without aux analysis",
            exc,
        )
        return None
    finally:
        if temp_image_path is not None:
            try:
                await aiofiles.os.remove(temp_image_path)
            except Exception:
                pass

    analysis_text = ""
    if isinstance(result_json, str):
        try:
            parsed = json.loads(result_json)
            if isinstance(parsed, dict):
                analysis_text = str(parsed.get("analysis") or "").strip()
        except (TypeError, json.JSONDecodeError):
            analysis_text = result_json.strip()

    if not analysis_text:
        return None

    return json.dumps({
        "mode": cap.mode,
        "width": cap.width,
        "height": cap.height,
        "app": cap.app,
        "window_title": cap.window_title,
        "elements": [_element_to_dict(e) for e in cap.elements],
        "summary": summary,
        "vision_analysis": analysis_text,
        "vision_analysis_routed_via": "auxiliary.vision",
    })


async def _maybe_follow_capture(
    backend: ComputerUseBackend, res: ActionResult, do_capture: bool,
) -> Any:
    if not do_capture:
        return _text_response(res)
    # Skip the follow-up capture when the action itself failed: showing a
    # normal-looking screenshot after a failure misleads the model into thinking
    # the action succeeded. Return the error text instead.
    if not res.ok:
        return _text_response(res)
    try:
        # Preserve the exact selected window when possible. Linux may expose a
        # generic app name for several unrelated windows, so app-only recapture
        # can silently switch targets after a successful action.
        target = getattr(backend, "_last_target", None) or {}
        pid = target.get("pid")
        window_id = target.get("window_id")
        mode = await _capture_after_mode()
        if pid is not None and window_id is not None:
            cap = await backend.capture(mode=mode, pid=pid, window_id=window_id)
        else:
            cap = await backend.capture(mode=mode, app=getattr(backend, "_last_app", None))
    except Exception as e:
        logger.warning("follow-up capture failed: %s", e)
        return _text_response(res)
    # Combine action summary with the capture.
    resp = await _capture_response(cap)
    if isinstance(resp, dict) and resp.get("_multimodal"):
        # Keep the complete evidence/verdict contract visible when an image is
        # attached; otherwise capture_after would accidentally discard the
        # very signal that governs whether repeating input is allowed.
        prefix = json.dumps(_action_payload(res))
        resp["content"][0]["text"] = prefix + "\n\n" + resp["content"][0]["text"]
        resp["text_summary"] = prefix + "\n\n" + resp["text_summary"]
        resp["action_result"] = _action_payload(res)
        return resp
    # Fallback: action + text capture merged.
    try:
        data = json.loads(resp)
    except (TypeError, json.JSONDecodeError):
        data = {"capture": resp}
    data.update(_action_payload(res))
    return json.dumps(data)


def _format_elements(elements: List[UIElement], max_lines: int = 40) -> List[str]:
    out: List[str] = []
    for e in elements[:max_lines]:
        label = e.label.replace("\n", " ")[:60]
        out.append(f"  #{e.index} {e.role} {label!r} @ {e.bounds}"
                   + (f" [{e.app}]" if e.app else ""))
    if len(elements) > max_lines:
        out.append(f"  ... +{len(elements) - max_lines} more (call capture with app= to narrow)")
    return out


def _element_to_dict(e: UIElement) -> Dict[str, Any]:
    return {
        "index": e.index,
        "role": e.role,
        "label": e.label,
        "bounds": list(e.bounds),
        "app": e.app,
    }


# ---------------------------------------------------------------------------
# Availability check (used by the tool registry check_fn)
# ---------------------------------------------------------------------------

async def check_computer_use_requirements() -> bool:
    """Return True iff computer_use can run on this host.

    Conditions: macOS, Windows, or Linux + cua-driver binary installed (or
    override via env). cua-driver runs on all three; the Linux path is
    headed/X11 today (Wayland via XWayland), pure-Wayland progress tracked
    upstream. Linux users see specific blocked checks via
    `hermes computer-use doctor` if their session is incomplete (e.g. no
    DISPLAY set).
    """
    if sys.platform not in ("darwin", "win32", "linux"):
        return False
    from tools.computer_use.cua_backend import cua_driver_binary_available
    return await cua_driver_binary_available()


def get_computer_use_schema() -> Dict[str, Any]:
    from tools.computer_use.schema import COMPUTER_USE_SCHEMA
    return COMPUTER_USE_SCHEMA
