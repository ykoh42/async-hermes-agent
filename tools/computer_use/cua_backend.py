"""Native-async cua-driver backend for macOS, Windows, and Linux.

The backend speaks MCP over stdio directly on the caller's event loop. A
lifecycle task owns the MCP context managers so they are entered and exited by
the same task, while tool calls remain ordinary awaited operations.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import uuid
import weakref
from collections import deque
from pathlib import PureWindowsPath
from typing import Any

import aiofiles
import aiofiles.os
import aiofiles.tempfile

from hermes_cli._subprocess_compat import windows_hide_flags
from tools.computer_use.backend import (
    ActionResult,
    CaptureResult,
    ComputerUseBackend,
    UIElement,
)
from tools.computer_use.browser_route import CuaTypedBrowserRoute

logger = logging.getLogger(__name__)

_MISSING = object()


def _mcp_field(obj, snake: str, camel: str, default=None):
    """Read an MCP model field across SDK snake/camel spellings."""
    value = getattr(obj, snake, _MISSING)
    if value is not _MISSING:
        return value
    value = getattr(obj, camel, _MISSING)
    return default if value is _MISSING else value

_CUA_DRIVER_CMD_ENV = "HERMES_CUA_DRIVER_CMD"
_CUA_DRIVER_DEFAULT_CMD = "cua-driver"
_CUA_DRIVER_ARGS = ["mcp"]
_SCREEN_CAPTURE_SENTINELS = {"screen", "desktop", "fullscreen", "full screen", "all"}
_DESKTOP_WINDOW_NAMES = (
    "progman", "workerw", "program manager",
    "shell_traywnd", "taskbar",
    "finder", "desktop", "dock",
)
_NON_APP_WINDOW_TITLE_PREFIXES = (
    "@!",
    "Desktop",
    "gnome-shell",
    "GNOME Shell",
)
_CUA_TELEMETRY_ENV_VAR = "CUA_DRIVER_RS_TELEMETRY_ENABLED"


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        process.kill()
    await process.wait()


async def _run_command(  # noqa: ASYNC109 - explicit public parity timeout
    argv: list[str],
    *,
    timeout: float,  # noqa: ASYNC109
    env: dict[str, str] | None = None,
    stdin_data: bytes | None = None,
) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE if stdin_data is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        creationflags=windows_hide_flags(),
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(stdin_data),
            timeout=timeout,
        )
    except BaseException:
        await _terminate_process(process)
        raise
    return (
        int(process.returncode or 0),
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


def _action_result_from(
    name: str,
    ok: bool,
    message: str,
    meta: dict[str, Any],
    structured: dict[str, Any],
    *,
    requested_delivery: str | None = None,
) -> ActionResult:
    sc = structured if isinstance(structured, dict) else {}

    def _pick(key: str) -> Any:
        if key in sc:
            return sc.get(key)
        return meta.get(key)

    verified = _pick("verified")
    if not isinstance(verified, bool):
        verified = None
    effect = _pick("effect")
    if not isinstance(effect, str):
        effect = None
    escalation = _pick("escalation")
    if not isinstance(escalation, dict):
        escalation = None
    path = _pick("path")
    if not isinstance(path, str):
        path = None
    degraded = _pick("degraded")
    if not isinstance(degraded, bool):
        degraded = None
    code = _pick("code") or _pick("reason_code")
    if not isinstance(code, str):
        code = None
    delivery_mode = requested_delivery if isinstance(requested_delivery, str) else None
    return ActionResult(
        ok=ok,
        action=name,
        message=message,
        meta=meta,
        verified=verified,
        effect=effect,
        escalation=escalation,
        path=path,
        degraded=degraded,
        delivery_mode=delivery_mode,
        code=code,
    )


async def _computer_use_cfg() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        return (await load_config_readonly() or {}).get("computer_use") or {}
    except Exception:
        return {}


async def _cua_no_overlay() -> bool:
    val = (await _computer_use_cfg()).get("no_overlay")
    if val is not None:
        return bool(val)
    if sys.platform == "darwin":
        return True
    if sys.platform != "linux":
        return False
    if not os.environ.get("DISPLAY"):
        return True
    try:
        async with aiofiles.open("/proc/version", encoding="utf-8") as stream:
            if "microsoft" in (await stream.read()).lower():
                return True
    except Exception:
        pass
    return False


async def _cua_telemetry_disabled() -> bool:
    return not bool((await _computer_use_cfg()).get("cua_telemetry", False))


async def _computer_use_max_image_dimension() -> int | None:
    try:
        dim = int((await _computer_use_cfg()).get("max_image_dimension", 1456))
    except (TypeError, ValueError):
        return 1456
    return dim if dim > 0 else None


async def cua_driver_child_env(
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(base_env if base_env is not None else os.environ)
    if await _cua_telemetry_disabled():
        env[_CUA_TELEMETRY_ENV_VAR] = "0"
    return env


def _z_index_uninformative(windows: list[dict[str, Any]]) -> bool:
    if not windows:
        return True
    return len({w.get("z_index", 0) for w in windows}) <= 1


def _parse_xprop_net_active_window(stdout: str) -> int | None:
    text = stdout or ""
    match = re.search(r"window id # (0x[0-9a-fA-F]+)", text)
    if not match:
        match = re.search(r"(0x[0-9a-fA-F]+)", text)
    if not match:
        return None
    try:
        return int(match.group(1), 16)
    except ValueError:
        return None


async def _linux_x11_active_window_id() -> int | None:
    if sys.platform != "linux" or not os.environ.get("DISPLAY"):
        return None
    try:
        returncode, stdout, _ = await _run_command(
            ["xprop", "-root", "_NET_ACTIVE_WINDOW"],
            timeout=2.0,
        )
    except Exception:
        return None
    if returncode != 0:
        return None
    return _parse_xprop_net_active_window(stdout)


def _is_real_app_window(window: dict[str, Any]) -> bool:
    title = window.get("title", "")
    return not any(
        title.startswith(prefix) or title.lower().startswith(prefix.lower())
        for prefix in _NON_APP_WINDOW_TITLE_PREFIXES
    )


async def _select_capture_target(
    windows: list[dict[str, Any]],
    *,
    app_requested: bool,
    exact_target: bool = False,
) -> dict[str, Any]:
    candidates = [window for window in windows if not window["off_screen"]]
    pool = candidates
    if not exact_target and not app_requested and sys.platform == "linux":
        real_apps = [window for window in candidates if _is_real_app_window(window)]
        if real_apps:
            pool = real_apps
        if pool and _z_index_uninformative(pool):
            active_id = await _linux_x11_active_window_id()
            if active_id is not None:
                for window in pool:
                    if window.get("window_id") == active_id:
                        return window
    if pool:
        return pool[0]
    return windows[0]


def _wsl_windows_path_to_posix(path: str) -> str:
    if not re.match(r"^[A-Za-z]:[\\/]", path):
        return path
    try:
        from hermes_constants import is_wsl

        if not is_wsl():
            return path
    except Exception:
        return path
    windows_path = PureWindowsPath(path)
    drive = (windows_path.drive or "").rstrip(":").lower()
    if not drive:
        return path
    return os.path.join(
        "/mnt",
        drive,
        *(str(part) for part in windows_path.parts[1:]),
    )


def _is_macos() -> bool:
    return sys.platform == "darwin"


def _has_path_separator(value: str) -> bool:
    return os.sep in value or (os.altsep is not None and os.altsep in value)


def _candidate_cua_driver_commands(override: str | None = None) -> list[str]:
    configured = (
        override
        if override is not None
        else os.environ.get(_CUA_DRIVER_CMD_ENV, "")
    ).strip()
    if configured:
        return [configured]
    home = os.path.expanduser("~")
    candidates = [_CUA_DRIVER_DEFAULT_CMD]
    if sys.platform == "win32":
        candidates.extend([
            os.path.join(home, ".local", "bin", "cua-driver.exe"),
            os.path.join(home, ".local", "bin", "cua-driver"),
        ])
    else:
        candidates.extend([
            os.path.join(home, ".local", "bin", "cua-driver"),
            os.path.join(home, ".cargo", "bin", "cua-driver"),
            "/opt/homebrew/bin/cua-driver",
            "/usr/local/bin/cua-driver",
        ])
    return candidates


async def resolve_cua_driver_cmd(override: str | None = None) -> str | None:
    which = aiofiles.os.wrap(shutil.which)
    expanduser = aiofiles.os.wrap(os.path.expanduser)
    for candidate in _candidate_cua_driver_commands(override):
        expanded = await expanduser(candidate)
        if _has_path_separator(expanded):
            if await which(expanded):
                return expanded
        else:
            resolved = await which(expanded)
            if resolved:
                return resolved
    return None


async def cua_driver_binary_available() -> bool:
    return await resolve_cua_driver_cmd() is not None


_no_overlay_support: dict[str, bool] = {}
_no_overlay_locks: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    dict[str, weakref.ReferenceType[asyncio.Lock]],
] = weakref.WeakKeyDictionary()
_no_overlay_locks_guard = threading.RLock()


def _no_overlay_lock(driver_cmd: str) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    with _no_overlay_locks_guard:
        for candidate in tuple(_no_overlay_locks):
            if candidate.is_closed():
                _no_overlay_locks.pop(candidate, None)
        locks = _no_overlay_locks.setdefault(loop, {})
        lock_ref = locks.get(driver_cmd)
        lock = lock_ref() if lock_ref is not None else None
        if lock is None:
            lock = asyncio.Lock()
            locks[driver_cmd] = weakref.ref(lock)
        return lock


async def _cua_driver_supports_no_overlay(driver_cmd: str) -> bool:
    cached = _no_overlay_support.get(driver_cmd)
    if cached is not None:
        return cached
    async with _no_overlay_lock(driver_cmd):
        cached = _no_overlay_support.get(driver_cmd)
        if cached is not None:
            return cached
        try:
            from tools.environments.local import _sanitize_subprocess_env

            env = await _sanitize_subprocess_env(await cua_driver_child_env())
            _, stdout, stderr = await _run_command(
                [driver_cmd, "--help"],
                timeout=3.0,
                env=env,
            )
            supported = "--no-overlay" in stdout + stderr
        except Exception:
            supported = False
        _no_overlay_support[driver_cmd] = supported
        return supported


async def _mcp_args_with_overlay_flag(
    args: list[str],
    driver_cmd: str = _CUA_DRIVER_DEFAULT_CMD,
) -> list[str]:
    if await _cua_no_overlay() and await _cua_driver_supports_no_overlay(driver_cmd):
        return [*args, "--no-overlay"]
    return list(args)


async def _resolve_mcp_invocation(  # noqa: ASYNC109 - upstream signature
    driver_cmd: str,
    *,
    timeout: float = 6.0,  # noqa: ASYNC109
) -> tuple[str, list[str]]:
    fallback = await _mcp_args_with_overlay_flag(
        list(_CUA_DRIVER_ARGS),
        driver_cmd=driver_cmd,
    )
    try:
        from tools.environments.local import _sanitize_subprocess_env

        env = await _sanitize_subprocess_env(await cua_driver_child_env())
        returncode, stdout, _ = await _run_command(
            [driver_cmd, "manifest"],
            timeout=timeout,
            env=env,
        )
    except Exception:
        return driver_cmd, fallback
    output = stdout.strip()
    if returncode != 0 or not output:
        return driver_cmd, fallback
    try:
        manifest = json.loads(output)
    except (TypeError, ValueError):
        return driver_cmd, fallback
    if not isinstance(manifest, dict):
        return driver_cmd, fallback
    invocation = manifest.get("mcp_invocation")
    if not isinstance(invocation, dict):
        return driver_cmd, fallback
    args = invocation.get("args")
    command = invocation.get("command")
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        return driver_cmd, fallback
    if not isinstance(command, str) or not command:
        return driver_cmd, await _mcp_args_with_overlay_flag(args, driver_cmd=driver_cmd)
    command = _wsl_windows_path_to_posix(command)
    if not _has_path_separator(command):
        return driver_cmd, await _mcp_args_with_overlay_flag(args, driver_cmd=driver_cmd)
    return command, await _mcp_args_with_overlay_flag(args, driver_cmd=command)


async def cua_driver_update_check(  # noqa: ASYNC109 - upstream signature
    *,
    timeout: float | None = None,  # noqa: ASYNC109
) -> dict[str, Any] | None:
    if timeout is None:
        timeout = 25.0 if sys.platform == "win32" else 8.0
    driver_cmd = await resolve_cua_driver_cmd()
    if not driver_cmd:
        return None
    try:
        from tools.environments.local import _sanitize_subprocess_env

        env = await _sanitize_subprocess_env(await cua_driver_child_env())
        _, stdout, _ = await _run_command(
            [driver_cmd, "check-update", "--json"],
            timeout=timeout,
            env=env,
        )
        data = json.loads(stdout.strip())
    except Exception:
        return None
    if not isinstance(data, dict) or data.get("error"):
        return None
    return data


async def cua_driver_update_nudge() -> str | None:
    state = await cua_driver_update_check()
    if not state or not state.get("update_available"):
        return None
    latest = state.get("latest_version") or "?"
    current = state.get("current_version") or "?"
    return (
        f"cua-driver {latest} is available (you have {current}); "
        "update with `hermes computer-use install --upgrade`."
    )


_update_checked = False


def _maybe_nudge_update() -> asyncio.Task[None] | None:
    global _update_checked
    if _update_checked:
        return None
    _update_checked = True

    async def _check() -> None:
        try:
            message = await cua_driver_update_nudge()
        except Exception:
            return
        if message:
            logger.info("computer_use: %s", message)

    return asyncio.create_task(_check(), name="cua-driver-update-check")


def cua_driver_install_hint() -> str:
    if sys.platform == "win32":
        installer = (
            "  irm https://raw.githubusercontent.com/trycua/cua/main/"
            "libs/cua-driver/scripts/install.ps1 | iex"
        )
    else:
        installer = (
            "  /bin/bash -c \"$(curl -fsSL "
            "https://raw.githubusercontent.com/trycua/cua/main/"
            "libs/cua-driver/scripts/install.sh)\""
        )
    return (
        "cua-driver is not installed. Install with one of:\n"
        "  hermes computer-use install\n"
        "Or run the upstream installer directly:\n"
        f"{installer}\n"
        "Or run `hermes tools` and enable the Computer Use toolset to install it automatically."
    )


_ELEMENT_LINE_RE = re.compile(
    r'^\s*(?:-\s+)?\[(\d+)\]\s+(\w+)'
    r'(?:'
    r'\s*=\s*"([^"]*)"'
    r'|\s+"([^"]*)"'
    r'|\s+\((?!\d+\))([^)]*)\)'
    r')?'
    r'(?:\s+(?:\(\d+\)\s+)?id=([^\s\[\]]+))?',
    re.MULTILINE,
)


def _parse_elements_from_tree(markdown: str) -> list[UIElement]:
    """Parse UIElement list from get_window_state AX tree markdown.

    Last-resort fallback for cua-driver builds that don't carry the
    canonical ``structuredContent.elements`` array (see
    ``_parse_elements_from_structured`` — Surface 2 of #47072 prefers
    that path).

    Captures the label whichever form cua-driver used: ``= "value"``,
    ``"quoted"``, ``(parenthesised)``, or ``id=Label``. Bounds always
    come back ``(0, 0, 0, 0)`` because the markdown surface doesn't
    carry them — yet another reason to prefer the structured path;
    element-index clicks don't need them (the driver resolves the index
    to a frame internally).
    """
    elements = []
    for m in _ELEMENT_LINE_RE.finditer(markdown):
        # groups 3-6: value / quoted / paren / id= label (first non-None wins)
        label = m.group(3) or m.group(4) or m.group(5) or m.group(6) or ""
        elements.append(UIElement(
            index=int(m.group(1)),
            role=m.group(2),
            label=label,
            bounds=(0, 0, 0, 0),
        ))
    return elements


def _parse_elements_from_structured(raw_elements: list[dict[str, Any]]) -> list[UIElement]:
    """Surface 2 of NousResearch/hermes-agent#47072: read the canonical
    ``structuredContent.elements`` array cua-driver-rs emits on every
    ``get_window_state`` response (trycua/cua#1961).

    Each entry has at minimum ``element_index``, ``role``, ``label``;
    ``frame`` (``{x, y, w, h}``) is included whenever the AT-SPI /
    AXFrame call returned usable bounds. Older code parsed the same
    information out of the markdown tree via a regex (lossy: bounds
    were always ``(0, 0, 0, 0)``) — this path preserves the real
    frame so downstream consumers (e.g. ``UIElement.center()``) work
    against pixel coordinates instead of just the index lookup.

    Unknown / malformed entries are skipped rather than failing the
    whole walk — the wrapper degrades to "fewer elements" rather than
    "no elements" on a bad row.
    """
    elements: list[UIElement] = []
    for raw in raw_elements:
        if not isinstance(raw, dict):
            continue
        idx = raw.get("element_index")
        if not isinstance(idx, int):
            continue
        role = raw.get("role") if isinstance(raw.get("role"), str) else ""
        label = raw.get("label") if isinstance(raw.get("label"), str) else ""
        frame = raw.get("frame") if isinstance(raw.get("frame"), dict) else None
        bounds: tuple[int, int, int, int] = (0, 0, 0, 0)
        if frame:
            try:
                bounds = (
                    int(frame.get("x", 0)),
                    int(frame.get("y", 0)),
                    int(frame.get("w", 0)),
                    int(frame.get("h", 0)),
                )
            except (TypeError, ValueError):
                bounds = (0, 0, 0, 0)
        # Surface 6: opaque element_token. cua-driver-rs format is
        # `s{snapshot_hex}:{index}`. We treat it as a black-box string —
        # the driver owns the parse + LRU semantics.
        raw_token = raw.get("element_token")
        token = raw_token if isinstance(raw_token, str) and raw_token else None
        elements.append(UIElement(
            index=idx,
            role=role,
            label=label,
            bounds=bounds,
            element_token=token,
        ))
    return elements


def _image_dimensions_from_bytes(raw: bytes) -> tuple[int, int]:
    """Best-effort PNG/JPEG dimension sniffing without extra dependencies."""
    if raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) >= 24:
        width = int.from_bytes(raw[16:20], "big")
        height = int.from_bytes(raw[20:24], "big")
        if width > 0 and height > 0:
            return width, height

    if raw.startswith(b"\xff\xd8"):
        i = 2
        n = len(raw)
        while i + 9 < n:
            if raw[i] != 0xFF:
                i += 1
                continue
            marker = raw[i + 1]
            i += 2
            if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                continue
            if i + 2 > n:
                break
            segment_len = int.from_bytes(raw[i:i + 2], "big")
            if segment_len < 2 or i + segment_len > n:
                break
            if marker in {
                0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
            }:
                if segment_len >= 7:
                    height = int.from_bytes(raw[i + 3:i + 5], "big")
                    width = int.from_bytes(raw[i + 5:i + 7], "big")
                    if width > 0 and height > 0:
                        return width, height
                break
            i += segment_len

    return 0, 0


def _split_tree_text(full_text: str) -> tuple[str, str]:
    """Split get_window_state text into (summary_line, tree_markdown)."""
    lines = full_text.split("\n", 1)
    summary = lines[0]
    tree = lines[1] if len(lines) > 1 else ""
    return summary, tree


def _parse_key_combo(keys: str) -> tuple[str | None, list[str]]:
    """Parse a key string like 'cmd+s' into (key, modifiers).

    Returns (key, modifiers) where key is the non-modifier key and modifiers
    is a list of modifier names (cmd, shift, option, ctrl).
    """
    MODIFIER_NAMES = {"cmd", "command", "shift", "option", "alt", "ctrl", "control", "fn"}
    KEY_ALIASES = {"command": "cmd", "alt": "option", "control": "ctrl"}

    parts = [p.strip().lower() for p in re.split(r'[+\-]', keys) if p.strip()]
    modifiers = []
    key = None
    for part in parts:
        normalized = KEY_ALIASES.get(part, part)
        if normalized in MODIFIER_NAMES:
            modifiers.append(normalized)
        else:
            key = part  # last non-modifier wins
    return key, modifiers


class _EmbeddedCuaDaemon:
    _START_TIMEOUT_SECONDS = 15.0

    def __init__(self, driver_cmd: str, permission_mode: str) -> None:
        if permission_mode != "unrestricted":
            raise ValueError("embedded permission override supports unrestricted only")
        self.permission_mode = permission_mode
        self._driver_cmd = driver_cmd
        self._command = driver_cmd
        self._mcp_args = list(_CUA_DRIVER_ARGS)
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_tail: deque[str] = deque(maxlen=20)
        self._stderr_task: asyncio.Task[None] | None = None
        token = uuid.uuid4().hex[:12]
        if sys.platform == "win32":
            self.socket_path = rf"\\.\pipe\hermes-cua-{token}"
        else:
            self.socket_path = os.path.join(tempfile.gettempdir(), f"hc-{token}.sock")

    async def child_env(self) -> dict[str, str]:
        env = await cua_driver_child_env()
        env["CUA_DRIVER_PERMISSION_MODE"] = "unrestricted"
        env["CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS"] = "1"
        return env

    async def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        while True:
            line = await process.stderr.readline()
            if not line:
                return
            value = line.decode("utf-8", errors="replace").strip()
            if value:
                self._stderr_tail.append(value)
                logger.debug("embedded cua-driver: %s", value)

    async def start(self) -> None:
        if self._process is not None and self._process.returncode is None:
            return
        from tools.environments.local import _sanitize_subprocess_env

        if not self._driver_cmd:
            self._driver_cmd = await resolve_cua_driver_cmd() or ""
        if not self._driver_cmd:
            raise RuntimeError(cua_driver_install_hint())
        self._command, self._mcp_args = await _resolve_mcp_invocation(self._driver_cmd)
        env = await _sanitize_subprocess_env(await self.child_env())
        command = [
            self._command,
            "serve",
            "--embedded",
            "--socket",
            self.socket_path,
            "--no-permissions-gate",
            "--permission-mode",
            "unrestricted",
            "--dangerously-bypass-approvals",
        ]
        self._process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            creationflags=windows_hide_flags(),
        )
        self._stderr_task = asyncio.create_task(
            self._drain_stderr(),
            name=f"cua-driver-stderr-{id(self)}",
        )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._START_TIMEOUT_SECONDS
        try:
            while loop.time() < deadline:
                if self._process.returncode is not None:
                    detail = "; ".join(self._stderr_tail) or "no diagnostic output"
                    raise RuntimeError(
                        f"embedded cua-driver exited during startup: {detail}"
                    )
                try:
                    returncode, _, _ = await _run_command(
                        [self._command, "status", "--socket", self.socket_path],
                        timeout=2.0,
                        env=env,
                    )
                except (TimeoutError, OSError):
                    returncode = -1
                if returncode == 0:
                    return
                await asyncio.sleep(0.1)
        except BaseException:
            await self.stop()
            raise
        await self.stop()
        detail = "; ".join(self._stderr_tail) or "daemon did not become ready"
        raise RuntimeError(f"embedded cua-driver startup timed out: {detail}")

    def proxy_invocation(self) -> tuple[str, list[str]]:
        if self._process is None or self._process.returncode is not None:
            raise RuntimeError("embedded cua-driver daemon is not running")
        return self._command, [
            *self._mcp_args,
            "--embedded",
            "--socket",
            self.socket_path,
        ]

    async def stop(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.returncode is None:
            from tools.environments.local import _sanitize_subprocess_env

            with contextlib.suppress(OSError, TimeoutError):
                env = await _sanitize_subprocess_env(await self.child_env())
                await _run_command(
                    [self._command, "stop", "--socket", self.socket_path],
                    timeout=3.0,
                    env=env,
                )
            if process.returncode is None:
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except TimeoutError:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=2.0)
                    except TimeoutError:
                        await _terminate_process(process)
        task = self._stderr_task
        self._stderr_task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if sys.platform != "win32" and await aiofiles.os.path.exists(self.socket_path):
            with contextlib.suppress(OSError):
                await aiofiles.os.remove(self.socket_path)


class _CuaDriverSession:
    _LIFECYCLE_CALLS = frozenset({"start_session", "end_session"})

    def __init__(
        self,
        embedded_daemon: _EmbeddedCuaDaemon | None = None,
    ) -> None:
        self._embedded_daemon = embedded_daemon
        self._session: Any = None
        self._started = False
        self._start_lock = asyncio.Lock()
        self._call_lock = asyncio.Lock()
        self._lifecycle_task: asyncio.Task[None] | None = None
        self._ready: asyncio.Future[None] | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._capabilities: dict[str, set[str]] = {}
        self._tool_schemas: dict[str, dict[str, Any]] = {}
        self._capability_version = ""
        self._declared_session_id: str | None = None

    def _require_started(self) -> None:
        if not self._started or self._session is None:
            raise RuntimeError("cua-driver session not started")

    async def _populate_capabilities(self, session: Any) -> None:
        self._capabilities = {}
        self._tool_schemas = {}
        self._capability_version = ""
        try:
            tools_list = await session.list_tools()
            for tool in getattr(tools_list, "tools", []) or []:
                name = getattr(tool, "name", None)
                if not isinstance(name, str):
                    continue
                extra = getattr(tool, "model_extra", None) or {}
                capabilities = getattr(tool, "capabilities", None)
                if capabilities is None:
                    capabilities = extra.get("capabilities")
                self._capabilities[name] = {
                    value
                    for value in capabilities or []
                    if isinstance(value, str)
                }
                schema = getattr(tool, "inputSchema", None)
                if schema is None:
                    schema = extra.get("inputSchema")
                self._tool_schemas[name] = (
                    dict(schema) if isinstance(schema, dict) else {}
                )
            version = getattr(tools_list, "capability_version", None)
            if version is None:
                version = (getattr(tools_list, "model_extra", None) or {}).get(
                    "capability_version"
                )
            if isinstance(version, str):
                self._capability_version = version
        except Exception as exc:
            logger.debug(
                "cua-driver tools/list capability discovery failed: %s",
                exc,
            )

    async def _lifecycle(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from tools.environments.local import _sanitize_subprocess_env

        ready = self._ready
        shutdown_event = self._shutdown_event
        try:
            driver_cmd = await resolve_cua_driver_cmd()
            if not driver_cmd:
                raise RuntimeError(cua_driver_install_hint())
            if self._embedded_daemon is not None:
                command, args = self._embedded_daemon.proxy_invocation()
                child_env = await self._embedded_daemon.child_env()
            else:
                command, args = await _resolve_mcp_invocation(driver_cmd)
                child_env = await cua_driver_child_env()
            params = StdioServerParameters(
                command=command,
                args=args,
                env=await _sanitize_subprocess_env(child_env),
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    await self._populate_capabilities(session)
                    self._session = session
                    self._started = True
                    if ready is not None and not ready.done():
                        ready.set_result(None)
                    if shutdown_event is not None:
                        await shutdown_event.wait()
        except BaseException as exc:
            if ready is not None and not ready.done():
                ready.set_exception(exc)
            elif not isinstance(exc, asyncio.CancelledError):
                logger.warning("cua-driver lifecycle ended: %s", exc)
            raise
        finally:
            self._session = None
            self._started = False

    async def start(self) -> None:
        async with self._start_lock:
            if self._started:
                return
            loop = asyncio.get_running_loop()
            self._ready = loop.create_future()
            self._shutdown_event = asyncio.Event()
            self._lifecycle_task = asyncio.create_task(
                self._lifecycle(),
                name=f"cua-driver-lifecycle-{id(self)}",
            )
            try:
                await asyncio.wait_for(asyncio.shield(self._ready), timeout=30.0)
            except BaseException:
                await self._stop_locked()
                raise

    async def _stop_locked(self) -> None:
        event = self._shutdown_event
        task = self._lifecycle_task
        if event is not None:
            event.set()
        cancelled: asyncio.CancelledError | None = None
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except TimeoutError:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised below
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                cancelled = exc
            except Exception as exc:
                logger.warning("cua-driver shutdown error: %s", exc)
        self._lifecycle_task = None
        self._shutdown_event = None
        self._ready = None
        self._session = None
        self._started = False
        if cancelled is not None:
            raise cancelled

    async def stop(self) -> None:
        async with self._start_lock:
            await self._stop_locked()

    def supports_capability(
        self,
        capability: str,
        tool: str | None = None,
    ) -> bool:
        if tool is not None:
            return capability in self._capabilities.get(tool, set())
        return any(capability in values for values in self._capabilities.values())

    def _has_tool(self, name: str) -> bool:
        return name in self._capabilities

    def supports_input_property(self, tool: str, property_name: str) -> bool:
        schema = self._tool_schemas.get(tool, {})
        properties = schema.get("properties") if isinstance(schema, dict) else None
        return isinstance(properties, dict) and property_name in properties

    @property
    def capabilities_discovered(self) -> bool:
        return bool(self._capabilities)

    @property
    def capability_version(self) -> str:
        return self._capability_version

    @staticmethod
    def _logical_error_text(result: dict[str, Any]) -> str:
        chunks: list[str] = []
        for value in (result.get("data"), result.get("structuredContent")):
            if isinstance(value, str):
                chunks.append(value)
            elif value is not None:
                try:
                    chunks.append(json.dumps(value, sort_keys=True))
                except (TypeError, ValueError):
                    chunks.append(str(value))
        return "\n".join(chunks)

    @classmethod
    def _is_ended_session_result(cls, result: Any) -> bool:
        if not isinstance(result, dict) or result.get("isError") is not True:
            return False
        message = cls._logical_error_text(result).lower()
        return (
            "session" in message
            and ("has ended" in message or "session ended" in message)
            and "start_session" in message
        )

    @staticmethod
    def _is_closed_session_error(exc: Exception) -> bool:
        name = exc.__class__.__name__
        module = getattr(exc.__class__, "__module__", "")
        return (
            name in {"ClosedResourceError", "BrokenResourceError", "EndOfStream"}
            or (module.startswith("anyio") and "Resource" in name)
            or isinstance(exc, (BrokenPipeError, EOFError))
        )

    @staticmethod
    def _is_transient_daemon_error(exc: Exception) -> bool:
        message = str(exc)
        return (
            "Resource temporarily unavailable" in message
            or "os error 35" in message
            or "daemon transport error" in message
            or "daemon proxy" in message
        )

    async def _call_tool_direct(  # noqa: ASYNC109 - internal timeout budget
        self,
        name: str,
        args: dict[str, Any],
        timeout: float,  # noqa: ASYNC109
    ) -> dict[str, Any]:
        self._require_started()
        result = await asyncio.wait_for(
            self._session.call_tool(name, args),
            timeout=timeout,
        )
        return _extract_tool_result(result)

    async def _call_tool_via_cli(  # noqa: ASYNC109 - upstream signature
        self,
        name: str,
        args: dict[str, Any],
        timeout: float,  # noqa: ASYNC109
    ) -> dict[str, Any]:
        call_args = dict(args)
        shot_file: str | None = None
        if name == "get_window_state" and "screenshot_out_file" not in call_args:
            temp = aiofiles.tempfile.NamedTemporaryFile(
                prefix="cua_shot_",
                suffix=".png",
                delete=False,
            )
            async with temp as stream:
                shot_file = stream.name
            call_args["screenshot_out_file"] = shot_file

        driver_command = await resolve_cua_driver_cmd()
        if not driver_command:
            raise RuntimeError(cua_driver_install_hint())
        child_env = await cua_driver_child_env()
        socket_args: list[str] = []
        if self._embedded_daemon is not None:
            driver_command = self._embedded_daemon.proxy_invocation()[0]
            child_env = await self._embedded_daemon.child_env()
            socket_args = ["--socket", self._embedded_daemon.socket_path]
        from tools.environments.local import _sanitize_subprocess_env

        env = await _sanitize_subprocess_env(child_env)
        command = [
            driver_command,
            "call",
            name,
            json.dumps(call_args),
            *socket_args,
        ]
        parsed: Any = None
        last_error = ""
        backoff = 0.5
        try:
            for attempt in range(4):
                try:
                    _, stdout, stderr = await _run_command(
                        command,
                        timeout=max(15.0, timeout),
                        env=env,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"cua-driver CLI fallback for {name} failed to spawn: {exc}"
                    ) from exc
                output = stdout.strip()
                last_error = output[:200] or stderr[:200]
                starts = [index for index in (output.find("{"), output.find("[")) if index >= 0]
                if starts:
                    try:
                        parsed = json.loads(output[min(starts):])
                    except json.JSONDecodeError:
                        parsed = None
                if parsed is not None:
                    break
                if attempt < 3:
                    logger.warning(
                        "cua-driver CLI fallback for %s got no JSON "
                        "(attempt %d/4); retrying in %.1fs",
                        name,
                        attempt + 1,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 2
            if parsed is None:
                raise RuntimeError(
                    f"cua-driver CLI fallback for {name} returned no JSON "
                    f"after 4 attempts: {last_error}"
                )
            images: list[str] = []
            data: Any = None
            structured = parsed if isinstance(parsed, dict) else None
            is_error = False
            if isinstance(parsed, dict):
                is_error = (
                    parsed.get("isError") is True
                    or parsed.get("is_error") is True
                )
                screenshot = parsed.get("screenshot_png_b64")
                if not screenshot:
                    file_path = parsed.get("screenshot_file_path") or shot_file
                    if file_path and await aiofiles.os.path.exists(file_path):
                        try:
                            async with aiofiles.open(file_path, "rb") as stream:
                                screenshot = base64.b64encode(
                                    await stream.read()
                                ).decode("ascii")
                        except Exception as exc:
                            logger.debug(
                                "cua-driver CLI fallback: failed reading %s: %s",
                                file_path,
                                exc,
                            )
                if screenshot:
                    images.append(screenshot)
                tree = parsed.get("tree_markdown")
                if tree is not None:
                    count = parsed.get("element_count")
                    summary = f"{count} elements" if count is not None else ""
                    data = f"{summary}\n{tree}" if summary else tree
            return {
                "data": data,
                "images": images,
                "structuredContent": structured,
                "isError": is_error,
            }
        finally:
            if shot_file and await aiofiles.os.path.exists(shot_file):
                with contextlib.suppress(OSError):
                    await aiofiles.os.remove(shot_file)

    async def _restart(self) -> None:
        await self.stop()
        await self.start()

    async def call_tool(  # noqa: ASYNC109 - upstream public signature
        self,
        name: str,
        args: dict[str, Any],
        timeout: float = 30.0,  # noqa: ASYNC109
    ) -> dict[str, Any]:
        async with self._call_lock:
            if not self._started and name not in self._LIFECYCLE_CALLS:
                logger.warning(
                    "cua-driver session not active on %s; (re)starting before call",
                    name,
                )
                await self.start()
            self._require_started()
            call_error: Exception | None = None
            try:
                result = await self._call_tool_direct(name, args, timeout)
            except Exception as exc:
                call_error = exc
            if call_error is not None:
                if self._is_transient_daemon_error(call_error):
                    logger.warning(
                        "cua-driver MCP transport failed on %s; using CLI transport",
                        name,
                    )
                    return await self._call_tool_via_cli(name, args, timeout)
                if not self._is_closed_session_error(call_error):
                    raise call_error
                logger.warning(
                    "cua-driver MCP session closed during %s; reconnecting once",
                    name,
                )
                await self._restart()
                result = await self._call_tool_direct(name, args, timeout)

            if name == "start_session" and result.get("isError") is not True:
                declared = args.get("session")
                if isinstance(declared, str) and declared:
                    self._declared_session_id = declared

            if (
                self._is_ended_session_result(result)
                and self._declared_session_id
                and name not in self._LIFECYCLE_CALLS
            ):
                first_result = result
                revived = await self._call_tool_direct(
                    "start_session",
                    {"session": self._declared_session_id},
                    timeout,
                )
                if revived.get("isError") is not True:
                    result = await self._call_tool_direct(name, args, timeout)
                else:
                    result = first_result

            if (
                name == "end_session"
                and result.get("isError") is not True
                and args.get("session") == self._declared_session_id
            ):
                self._declared_session_id = None
            return result

def _extract_tool_result(mcp_result: Any) -> dict[str, Any]:
    """Convert an mcp CallToolResult into a plain dict.

    cua-driver returns a mix of text parts, image parts, and structuredContent.
    We flatten into:
      {
        "data": <text or parsed json>,
        "images": [b64, ...],
        "image_mime_types": [mime, ...],   # parallel to `images`, "" when absent
        "structuredContent": <dict|None>,
        "isError": bool,
      }
    structuredContent is populated from the MCP result's structuredContent field
    (MCP spec §2024-11-05+) and takes precedence for structured data like
    list_windows window arrays.

    `image_mime_types` is the explicit `mimeType` cua-driver emits on every
    image part as of trycua/cua#1961 (Surface 7 of
    NousResearch/hermes-agent#47072). Each entry corresponds index-for-index
    with `images`; an empty string entry signals the part carried no
    mimeType (older cua-driver build), and the caller should fall back to
    base64-prefix sniffing.
    """
    data: Any = None
    images: list[str] = []
    image_mime_types: list[str] = []
    # Use identity, not truthiness: unittest mocks and proxy objects commonly
    # synthesize truthy attributes that were never present in the real result.
    is_error = getattr(mcp_result, "isError", False) is True
    structured: dict | None = getattr(mcp_result, "structuredContent", None) or None
    text_chunks: list[str] = []
    for part in getattr(mcp_result, "content", []) or []:
        ptype = getattr(part, "type", None)
        if ptype == "text":
            text_chunks.append(getattr(part, "text", "") or "")
        elif ptype == "image":
            b64 = getattr(part, "data", None)
            if b64:
                images.append(b64)
                mime = _mcp_field(part, "mime_type", "mimeType") or ""
                image_mime_types.append(mime)
    if text_chunks:
        joined = "\n".join(t for t in text_chunks if t)
        try:
            data = json.loads(joined) if joined.strip().startswith(("{", "[")) else joined
        except json.JSONDecodeError:
            data = joined
    return {
        "data": data,
        "images": images,
        "image_mime_types": image_mime_types,
        "structuredContent": structured,
        "isError": is_error,
    }


def _image_from_tool_result(out: dict[str, Any]) -> tuple[str | None, str | None]:
    """Pull a (png_b64, mime_type) pair out of a flattened tool result.

    cua-driver delivers window screenshots in two shapes depending on tool +
    transport:

      * As an MCP ``image`` content part — surfaced by ``_extract_tool_result``
        in ``out["images"]`` with a parallel ``image_mime_types`` entry. This
        is what ``get_window_state`` emits over the stdio MCP transport.
      * As a base64 field inside ``structuredContent`` —
        ``screenshot_png_b64`` (+ ``screenshot_mime_type``). This is what
        ``get_window_state`` returns when its structured payload carries the
        image instead of a content part (newer driver builds; also the shape
        seen via the ``cua-driver call`` CLI surface).

    Checking both makes capture() robust to either delivery shape, so the
    image never silently drops just because the driver moved it between the
    content list and structuredContent. Returns ``(None, None)`` when neither
    location carries an image.
    """
    images = out.get("images") or []
    if images and images[0]:
        mimes = out.get("image_mime_types") or []
        mime = mimes[0] if mimes and mimes[0] else None
        return images[0], mime

    structured = out.get("structuredContent") or {}
    b64 = structured.get("screenshot_png_b64") or structured.get("png_b64")
    if b64:
        mime = (
            structured.get("screenshot_mime_type")
            or structured.get("mime_type")
            or None
        )
        return b64, mime

    return None, None


def _positive_int(value: Any) -> int | None:
    """Return a positive integer, rejecting booleans and malformed values."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _ingest_windows(raw_windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise cua-driver ``list_windows`` entries, dropping unusable ones.

    Every downstream operation needs both an integer ``pid`` (for
    get_window_state / action tools) and ``window_id`` (for screenshot /
    element clicks), so a window missing either is uncapturable.

    Crucially, on X11 a window's PID comes from the *optional*
    ``_NET_WM_PID`` property — the desktop root, panels, and
    override-redirect popups routinely omit it, so the driver reports
    ``pid: null`` for them. Coercing every entry unconditionally
    (``int(w["pid"])``) let one such window abort enumeration of the real,
    targetable windows. We skip the unusable entries instead so capture()
    and focus_app() still find the windows that matter.

    ``z_index`` follows CUA Driver semantics: higher = closer to front.
    Wayland may return ``z_index: null`` (undefined stacking order); we
    treat null as the lowest priority so real windows still sort above
    desktop/root windows, and the backmost never ends up selected as the
    capture target.
    """
    windows: list[dict[str, Any]] = []
    for w in raw_windows:
        # Compatibility envelopes are untrusted input: skip non-dict members
        # instead of raising AttributeError on one malformed record.
        if not isinstance(w, dict):
            continue
        pid_int = _positive_int(w.get("pid"))
        window_id_int = _positive_int(w.get("window_id"))
        if pid_int is None or window_id_int is None:
            continue
        z_raw = w.get("z_index")
        z_index = z_raw if isinstance(z_raw, (int, float)) and not isinstance(z_raw, bool) else 0
        app_name = w.get("app_name", "")
        title = w.get("title", "")
        windows.append({
            "app_name": app_name if isinstance(app_name, str) else "",
            "pid": pid_int,
            "window_id": window_id_int,
            # cua-driver 0.6.x on Linux may return JSON null here.
            # Only explicit False means off-screen; null means unknown.
            "off_screen": w.get("is_on_screen") is False,
            "title": title if isinstance(title, str) else "",
            "z_index": z_index,
        })
    return windows


def _windows_from_tool_result(out: dict[str, Any]) -> list[dict[str, Any]]:
    """Return list_windows payloads across cua-driver result shapes."""
    structured = out.get("structuredContent")
    if isinstance(structured, dict):
        windows = structured.get("windows")
        if isinstance(windows, list) and windows:
            return windows

    data = out.get("data")
    if isinstance(data, dict):
        windows = data.get("windows")
        if isinstance(windows, list) and windows:
            return windows
        legacy_windows = data.get("_legacy_windows")
        if isinstance(legacy_windows, list) and legacy_windows:
            return legacy_windows

    windows = out.get("windows")
    if isinstance(windows, list) and windows:
        return windows
    legacy_windows = out.get("_legacy_windows")
    if isinstance(legacy_windows, list) and legacy_windows:
        return legacy_windows
    return []


def _apps_from_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    apps: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for summary in _ingest_windows(windows):
        name = summary["app_name"]
        if not name:
            continue
        key = (name, summary["pid"])
        if key in seen:
            continue
        seen.add(key)
        apps.append({"name": name, "pid": summary["pid"]})
    return apps


# ---------------------------------------------------------------------------
# The backend itself
# ---------------------------------------------------------------------------

class CuaDriverBackend(ComputerUseBackend):
    """Default computer-use backend. Cross-platform via cua-driver MCP."""

    def __init__(self, permission_mode: str = "standard") -> None:
        if permission_mode not in {"standard", "unrestricted"}:
            raise ValueError(f"unsupported cua-driver permission mode: {permission_mode}")
        self.permission_mode = permission_mode
        self._embedded_daemon = (
            _EmbeddedCuaDaemon("", permission_mode)
            if permission_mode == "unrestricted"
            else None
        )
        self._session = _CuaDriverSession(self._embedded_daemon)
        # Sticky context — updated by capture(), used by action tools.
        self._active_pid: int | None = None
        self._active_window_id: int | None = None
        self._last_app: str | None = None  # last app name targeted via capture/focus_app
        # Exact identity for capture_after. App names may be generic on Linux
        # (for example, multiple unrelated Qt windows can say Qt6Application).
        self._last_target: dict[str, int | None] | None = None
        # Surface 6 of NousResearch/hermes-agent#47072: per-snapshot
        # `element_index -> element_token` map populated on capture().
        # Action tools (click/scroll/set_value/...) attach the matching
        # token alongside `element_index` so cua-driver detects "stale"
        # explicitly instead of silently re-resolving to a different
        # element. Cleared whenever a fresh capture overwrites the
        # snapshot context.
        self._snapshot_tokens: dict[int, str] = {}
        # Per-instance cua-driver session id. cua-driver's MCP server
        # instructions ask every consumer to declare a stable session
        # at the start of a run (start_session) and tear it down at
        # the end (end_session). Doing so:
        #   - Gets a distinct agent-cursor color per Hermes run, with
        #     overlay rendering visualising where actions land
        #     (without moving the real OS cursor).
        #   - Isolates per-session config + recording ownership so
        #     concurrent Hermes runs / subagents don't step on each
        #     other.
        # We mint a UUID4-based id once per CuaDriverBackend instance —
        # one Hermes run = one backend = one session — and pass it as
        # `session` on every cua-driver tool call. Sessions are an
        # additive feature on the cua-driver side: when our id is
        # unknown to the driver (older builds), the tool calls
        # degrade to the anonymous / unsynced path documented in the
        # MCP server instructions.
        self._session_id: str = f"hermes-{uuid.uuid4().hex[:12]}"
        self._typed_browser = CuaTypedBrowserRoute(
            session_id=self._session_id,
            call_tool=self._session.call_tool,
            has_tool=self._session._has_tool,
        )
        self._update_task: asyncio.Task[None] | None = None

    def _browser_route(self) -> CuaTypedBrowserRoute:
        """Return the per-backend typed route, including test-constructed instances."""
        route = getattr(self, "_typed_browser", None)
        if route is None:
            route = CuaTypedBrowserRoute(
                session_id=self._session_id,
                call_tool=self._session.call_tool,
                has_tool=self._session._has_tool,
            )
            self._typed_browser = route
        return route

    # ── Lifecycle ──────────────────────────────────────────────────
    async def start(self) -> None:
        self._update_task = _maybe_nudge_update()
        startup_error: BaseException | None = None
        try:
            if self._embedded_daemon is not None:
                await self._embedded_daemon.start()
            await self._session.start()
        except BaseException as exc:  # noqa: ASYNC103 - re-raised after cleanup
            startup_error = exc
        if startup_error is not None:
            try:
                if self._embedded_daemon is not None:
                    await self._embedded_daemon.stop()
            finally:
                update_task = self._update_task
                self._update_task = None
                if update_task is not None:
                    update_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await update_task
            raise startup_error

        # Declare the run's session identity to cua-driver. From the
        # cua-driver server instructions: "start_session(session) once
        # at the start of a run → declares THIS run's identity (a
        # stable id you choose). Pass that same `session` on every
        # action below. It owns your agent cursor (a distinct color
        # per id) and follows the run across apps/windows." Failure
        # to start the session is non-fatal — cua-driver's tools
        # accept anonymous calls (the cursor just won't render),
        # so we degrade rather than abort.
        try:
            await self._session.call_tool("start_session", {"session": self._session_id})
        except Exception as e:
            logger.debug("cua-driver start_session failed (continuing anonymous): %s", e)

        # Post-handshake session tuning. Both guard on `_started`: before the
        # handshake flips it, call_tool would re-enter session.start() (see
        # _LIFECYCLE_CALLS) and tests that stub start() would recurse.
        if self._session._started:
            # Cap screenshot size so every later get_window_state / SOM
            # capture pays less over the daemon socket and in the model turn.
            max_dim = await _computer_use_max_image_dimension()
            if max_dim:
                try:
                    await self.set_config(max_image_dimension=max_dim)
                except Exception as e:
                    logger.debug("cua-driver set_config(max_image_dimension) failed: %s", e)
            # Belt-and-suspenders when --no-overlay is unsupported or ignored:
            # hide the agent cursor overlay via the session API so macOS idle
            # redraw loops cannot keep burning CPU after the first action.
            if await _cua_no_overlay():
                try:
                    await self.set_agent_cursor_enabled(False, cursor_id=self._session_id)
                except Exception as e:
                    logger.debug("cua-driver set_agent_cursor_enabled failed: %s", e)

    async def stop(self) -> None:
        # Tear the cua-driver session down before disconnecting so the
        # driver can clean up per-session state (cursor overlay, recording
        # ownership, config overrides). Best-effort — even if it fails,
        # the connection drop below releases the daemon-side state via
        # the session_end hook cua-driver registers internally.
        async def _close() -> None:
            if self._session._started:
                try:
                    await self._session.call_tool(
                        "end_session", {"session": self._session_id}
                    )
                except Exception as exc:
                    logger.debug(
                        "cua-driver end_session failed (continuing teardown): %s",
                        exc,
                    )
            try:
                await self._session.stop()
            finally:
                try:
                    if self._embedded_daemon is not None:
                        await self._embedded_daemon.stop()
                finally:
                    update_task = self._update_task
                    self._update_task = None
                    if update_task is not None:
                        update_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await update_task

        cleanup = asyncio.create_task(
            _close(), name=f"cua-driver-close-{id(self)}"
        )
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await cleanup
            raise

    async def is_available(self) -> bool:
        # cua-driver runs on macOS, Windows, and Linux. The Linux path is
        # the most recent addition (X11 + Wayland both supported upstream
        # as of mid-2026). Override the platform check at your own risk:
        # other Unix-likes haven't been exercised end-to-end.
        if sys.platform not in ("darwin", "win32", "linux"):
            return False
        return await cua_driver_binary_available()

    def _clear_active_target(self) -> None:
        """Forget a capture/focus target so a failed lookup cannot misroute input."""
        self._active_pid = None
        self._active_window_id = None
        self._last_app = None
        self._last_target = None
        self._snapshot_tokens = {}

    def _failed_capture(self, mode: str, message: str = "") -> CaptureResult:
        """Return an empty capture after disarming any prior target context."""
        self._clear_active_target()
        return CaptureResult(
            mode=mode,
            width=0,
            height=0,
            png_b64=None,
            elements=[],
            app="",
            window_title=message,
            png_bytes_len=0,
        )

    async def _call_capture_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Call a capture-stage tool and disarm state on transport or logical failure."""
        try:
            out = await self._session.call_tool(name, args)
        except Exception:
            self._clear_active_target()
            raise
        if out.get("isError") is True:
            message = out.get("data")
            self._clear_active_target()
            raise RuntimeError(
                f"cua-driver {name} failed"
                + (f": {message}" if isinstance(message, str) and message else "")
            )
        return out

    async def _load_windows(self) -> list[dict[str, Any]]:
        """Load normalized visible windows, with the shared CLI recovery path.

        Windows are sorted by ``z_index`` **descending**: CUA Driver
        defines higher values as closer to the front, so the frontmost
        window ends up at index 0 — which is what ``capture()`` and
        ``focus_app()`` pick as the default target.  ``_ingest_windows``
        already normalised null ``z_index`` (Wayland) to 0, so those
        windows sort to the back.
        """
        out = await self._call_capture_tool(
            "list_windows",
            {"on_screen_only": True, "session": self._session_id},
        )
        windows = _ingest_windows(_windows_from_tool_result(out))
        windows.sort(key=lambda w: w["z_index"], reverse=True)
        if windows:
            return windows

        logger.warning(
            "cua-driver list_windows returned no windows over MCP; "
            "re-fetching via CLI transport",
        )
        try:
            cli_out = await self._session._call_tool_via_cli(
                "list_windows",
                {"on_screen_only": True, "session": self._session_id},
                20.0,
            )
        except Exception as exc:
            logger.error("cua-driver CLI re-fetch for list_windows failed: %s", exc)
            return []
        if cli_out.get("isError") is True:
            logger.error("cua-driver CLI re-fetch for list_windows returned an error")
            self._clear_active_target()
            return []
        windows = _ingest_windows(_windows_from_tool_result(cli_out))
        windows.sort(key=lambda w: w["z_index"], reverse=True)
        return windows

    async def _match_windows_for_app(
        self, windows: list[dict[str, Any]], app: str
    ) -> list[dict[str, Any]]:
        """Resolve ``app=`` through exact names before convenience substrings.

        Linux ``list_windows`` can omit an app name while ``list_apps`` retains
        name/bundle-ID metadata. Exact direct names and exact metadata aliases
        must win over substring matches: querying ``Code`` must not silently
        select ``Visual Studio Code`` merely because it is frontmost.
        """
        app_lower = app.strip().lower()
        if not app_lower:
            return []

        direct_exact = [
            w for w in windows
            if app_lower == str(w.get("app_name", "")).strip().lower()
        ]
        if direct_exact:
            return direct_exact

        try:
            running_apps = await self.list_apps()
        except Exception as exc:
            # A title can still be the only usable identity on X11 when app
            # enumeration is unavailable, so retain the constrained title
            # fallback below instead of treating this as a hard no-match.
            logger.debug("computer_use list_apps fallback failed for %r: %s", app, exc)
            running_apps = []

        exact_pids: set[int] = set()
        partial_pids: set[int] = set()
        for raw_app in running_apps:
            if not isinstance(raw_app, dict) or raw_app.get("running") is False:
                continue
            raw_pid = raw_app.get("pid")
            if isinstance(raw_pid, bool) or not isinstance(raw_pid, (int, str)):
                continue
            try:
                pid = int(raw_pid)
            except ValueError:
                continue
            if pid <= 0:
                continue

            aliases = {
                value.strip().lower()
                for key in ("bundle_id", "bundleId", "name", "app_name", "display_name")
                if isinstance((value := raw_app.get(key)), str) and value.strip()
            }
            if app_lower in aliases:
                exact_pids.add(pid)
            elif any(app_lower in alias for alias in aliases):
                partial_pids.add(pid)

        metadata_exact = [w for w in windows if w.get("pid") in exact_pids]
        if metadata_exact:
            return metadata_exact

        direct_partial = [
            w for w in windows
            if app_lower in str(w.get("app_name", "")).lower()
        ]
        if direct_partial:
            return direct_partial

        metadata_partial = [w for w in windows if w.get("pid") in partial_pids]
        if metadata_partial:
            return metadata_partial

        # Some X11 backends expose a title but no app name. Restrict this final
        # fallback to nameless rows so a localized app name is not overridden
        # merely because its title happens to be in the caller's language.
        return [
            w for w in windows
            if not str(w.get("app_name", "")).strip()
            and app_lower in str(w.get("title", "")).lower()
        ]

    # ── Capture ────────────────────────────────────────────────────
    async def capture(
        self,
        mode: str = "som",
        app: str | None = None,
        pid: int | None = None,
        window_id: int | None = None,
    ) -> CaptureResult:
        """Capture the frontmost on-screen window or an exact known target.

        Maps hermes `capture(mode, app)` → cua-driver `list_windows` +
        `get_window_state` (ax/som) or `screenshot` (vision).
        """
        # Step 1: enumerate on-screen windows to find target pid/window_id.
        # Surface 3 of NousResearch/hermes-agent#47072: read the canonical
        # `structuredContent.windows` array directly. Pre-fix the wrapper
        # also kept a text-line regex (`_WINDOW_LINE_RE`) as a fallback for
        # cua-driver builds that predated structuredContent; the supersede
        # PR's effective minimum (trycua/cua#1961 + #1908) is well past
        # that, so the fallback is gone — the wrapper now treats the
        # structured shape as the only contract.
        # An exact pid/window pair is both the stable capture_after target and
        # the escape hatch when app/window discovery is unavailable on X11.
        if pid is not None or window_id is not None:
            if pid is None or window_id is None:
                return self._failed_capture(
                    mode, "<capture targeting requires both pid and window_id>",
                )
            target_pid = _positive_int(pid)
            target_window_id = _positive_int(window_id)
            if target_pid is None or target_window_id is None:
                return self._failed_capture(
                    mode, "<capture targeting requires positive integer pid and window_id>",
                )
            windows = [{
                "app_name": app or "",
                "pid": target_pid,
                "window_id": target_window_id,
                "off_screen": False,
                "title": "",
                "z_index": 0,
            }]
        else:
            try:
                windows = await self._load_windows()
            except Exception:
                self._clear_active_target()
                raise
            if not windows:
                return self._failed_capture(mode)

        # Filter by app name (case-insensitive substring) if requested.
        # When the filter matches nothing, surface that explicitly instead of
        # silently capturing the frontmost window — on macOS the `app_name`
        # returned by list_windows is the localized name (e.g. "計算機"), so
        # `app="Calculator"` legitimately matches no windows on a non-English
        # system and the caller needs to retry with the localized name.
        if pid is None and window_id is None and app and app.strip().lower() in _SCREEN_CAPTURE_SENTINELS:
            # Whole-screen / desktop request. cua-driver has no virtual-desktop
            # capture tool, so resolve to the OS shell/desktop window (the
            # desktop backdrop or the taskbar/menu-bar), which list_windows
            # does surface. This makes "show me my screen" and "click the
            # taskbar" work; a single image still can't span multiple monitors
            # — that's a driver limitation, not a wrapper one.
            def _is_desktop_window(w: dict[str, Any]) -> bool:
                haystack = f"{w.get('app_name', '')} {w.get('title', '')}".lower()
                return any(name in haystack for name in _DESKTOP_WINDOW_NAMES)

            desktop = [w for w in windows if _is_desktop_window(w)]
            if not desktop:
                return self._failed_capture(
                    mode,
                    (
                        f"<no desktop/shell window found for app={app!r}; "
                        f"cua-driver captures one window at a time and exposes "
                        f"no whole-virtual-desktop or per-monitor capture. "
                        f"Call list_apps / capture(app='<AppName>') to target a "
                        f"specific window instead. On Windows the taskbar is "
                        f"'Shell_TrayWnd' and the desktop is 'Progman'.>"
                    ),
                )
            # Prefer the desktop backdrop (Progman/WorkerW/Finder) over the
            # taskbar when both are present, so a bare "screen" capture shows
            # the full desktop rather than just the task strip.
            windows = sorted(
                desktop,
                key=lambda w: 0 if any(
                    n in f"{w.get('app_name', '')} {w.get('title', '')}".lower()
                    for n in ("progman", "workerw", "program manager", "finder", "desktop")
                ) else 1,
            )
        elif pid is None and window_id is None and app:
            filtered = await self._match_windows_for_app(windows, app)
            if not filtered:
                return self._failed_capture(
                    mode,
                    (
                        f"<no on-screen window matched app={app!r}; "
                        f"call list_apps to see available app names or bundle IDs "
                        f"(macOS reports localized names, e.g. '計算機' "
                        f"instead of 'Calculator'; some Linux/Qt apps only "
                        f"resolve via list_apps metadata)>"
                    ),
                )
            windows = filtered

        # Pick first on-screen window (sorted by z_index / z-order above).
        # On Linux, unqualified default captures skip desktop/shell helper
        # windows and, with tied/unknown z_index, may additionally consult
        # _NET_ACTIVE_WINDOW (#58026).
        target = await _select_capture_target(
            windows,
            app_requested=bool(app),
            exact_target=pid is not None or window_id is not None,
        )
        self._active_pid = target["pid"]
        self._active_window_id = target["window_id"]
        # Tokens belong to the prior window snapshot. Disarm them before any
        # capture call so an exception cannot pair old tokens with this target.
        self._snapshot_tokens = {}
        app_name = target["app_name"]
        # Record the resolved app name so capture_after= follow-ups can re-target
        # the same app rather than falling back to the frontmost window.
        if app or not self._last_app:
            self._last_app = app_name or app or ""
        self._last_target = {
            "pid": self._active_pid,
            "window_id": self._active_window_id,
        }

        # Step 2: capture.
        png_b64: str | None = None
        image_mime_type: str | None = None
        elements: list[UIElement] = []
        width = height = 0
        window_title = ""

        if mode == "vision":
            # Plain screenshot, no AX walk. cua-driver dropped the standalone
            # `screenshot` tool (≥0.5.x) and folded full-window PNG capture
            # into `get_window_state`. Route accordingly:
            #   * Driver advertises `screenshot` (older builds) → use it; it's
            #     the cheapest path (no AX tree walked server-side).
            #   * Otherwise (current drivers) → call `get_window_state` but
            #     DISCARD the AX tree/elements, returning only the PNG. Vision
            #     mode's whole contract is "just the pixels, no element noise",
            #     so we drop everything but the image.
            # When capability discovery hasn't run (empty map), we don't trust
            # a negative `_has_tool` answer — we still try `screenshot` first
            # and fall back if the driver rejects it, so the path self-heals on
            # any driver version.
            use_screenshot = (
                self._session._has_tool("screenshot")
                or not self._session.capabilities_discovered
            )
            sc_out: dict[str, Any] | None = None
            if use_screenshot:
                sc_out = await self._call_capture_tool(
                    "screenshot",
                    {
                        "window_id": self._active_window_id,
                        "format": "jpeg",
                        "quality": 85,
                        "session": self._session_id,
                    },
                )
                png_b64, image_mime_type = _image_from_tool_result(sc_out)
                if not png_b64:
                    # Driver had no usable `screenshot` (e.g. "Unknown tool:
                    # screenshot" on ≥0.5.x, or an empty image part). Fall
                    # through to the get_window_state path below.
                    sc_out = None

            if sc_out is None:
                gws_out = await self._call_capture_tool(
                    "get_window_state",
                    {
                        "pid": self._active_pid,
                        "window_id": self._active_window_id,
                        "session": self._session_id,
                    },
                )
                png_b64, image_mime_type = _image_from_tool_result(gws_out)
                # Still grab the window title — it's cheap and useful in the
                # vision response — but deliberately leave `elements` empty so
                # vision stays free of AX-tree noise.
                text = gws_out["data"] if isinstance(gws_out["data"], str) else ""
                _, tree = _split_tree_text(text)
                wt = re.search(r'AXWindow\s+"([^"]+)"', tree)
                if wt:
                    window_title = wt.group(1)

            if not png_b64:
                # Both MCP attempts came back imageless without raising (flaky
                # bridge dropping the heavy payload) — re-fetch the window
                # state over the CLI transport, which embeds a screenshot.
                logger.warning(
                    "cua-driver vision capture returned no image over MCP "
                    "(window_id=%s); re-fetching via CLI transport",
                    self._active_window_id,
                )
                try:
                    cli_out = await self._session._call_tool_via_cli(
                        "get_window_state",
                        {
                            "pid": self._active_pid,
                            "window_id": self._active_window_id,
                            "session": self._session_id,
                        },
                        30.0,
                    )
                    if cli_out.get("isError") is True:
                        self._clear_active_target()
                    elif cli_out.get("images"):
                        png_b64 = cli_out["images"][0]
                        image_mime_type = "image/png"
                except Exception as cli_exc:
                    logger.error(
                        "cua-driver CLI re-fetch for vision screenshot failed: %s", cli_exc,
                    )
        else:
            # get_window_state: AX tree + screenshot.
            gws_out = await self._call_capture_tool(
                "get_window_state",
                {
                    "pid": self._active_pid,
                    "window_id": self._active_window_id,
                    "session": self._session_id,
                },
            )
            # The persistent MCP session can return a degenerate result —
            # empty/partial data with NO exception — when the bridge is flaky
            # (e.g. it reconnected mid-call and dropped the heavy
            # get_window_state payload). That surfaces to the model as a silent
            # 0x0 capture. Detect "no screenshot AND no parseable tree" and
            # force a one-shot CLI-transport re-fetch, which talks to the daemon
            # over a different socket and returns the full result. This is
            # distinct from the EAGAIN McpError path (handled in call_tool);
            # here the MCP call "succeeded" but gave us nothing usable.
            def _gws_is_empty(out: dict[str, Any]) -> bool:
                if out.get("images"):
                    return False
                sc_ = out.get("structuredContent") or {}
                # Modern drivers carry the payload in structuredContent
                # (elements array / embedded screenshot) with no markdown
                # tree — that is NOT an empty result.
                if sc_.get("elements") or sc_.get("screenshot_png_b64"):
                    return False
                txt = out.get("data") if isinstance(out.get("data"), str) else ""
                _, tr = _split_tree_text(txt or "")
                return not (tr and tr.strip())

            if _gws_is_empty(gws_out):
                logger.warning(
                    "cua-driver get_window_state returned an empty result over MCP "
                    "(pid=%s window_id=%s); re-fetching via CLI transport",
                    self._active_pid, self._active_window_id,
                )
                try:
                    cli_out = await self._session._call_tool_via_cli(
                        "get_window_state",
                        {
                            "pid": self._active_pid,
                            "window_id": self._active_window_id,
                            "session": self._session_id,
                        },
                        30.0,
                    )
                    if cli_out.get("isError") is True:
                        self._clear_active_target()
                    elif not _gws_is_empty(cli_out):
                        gws_out = cli_out
                except Exception as cli_exc:
                    logger.error(
                        "cua-driver CLI re-fetch for get_window_state failed: %s", cli_exc,
                    )

            text = gws_out["data"] if isinstance(gws_out["data"], str) else ""
            summary, tree = _split_tree_text(text)

            # Surface 2 of NousResearch/hermes-agent#47072: prefer the
            # canonical structuredContent.elements array (trycua/cua#1961).
            # Falls back to markdown regex parsing for cua-driver builds
            # that didn't carry the structured shape — those bounds come
            # back (0,0,0,0); the structured path preserves real frames.
            sc_elements = (gws_out.get("structuredContent") or {}).get("elements")
            if isinstance(sc_elements, list) and sc_elements:
                elements = _parse_elements_from_structured(sc_elements)
            else:
                elements = _parse_elements_from_tree(tree) if tree else []

            # Surface 6: refresh the snapshot-token cache from this
            # capture. Tokens are tied to a specific cua-driver snapshot
            # — when a fresh capture lands, the prior snapshot's tokens
            # are stale, so we overwrite the whole map (and clear it
            # entirely when the new capture carries none).
            self._snapshot_tokens = {
                e.index: e.element_token
                for e in elements
                if e.element_token
            }

            # Image may arrive as an MCP image part or inside
            # structuredContent (screenshot_png_b64) depending on the driver
            # build — _image_from_tool_result handles both.
            png_b64, image_mime_type = _image_from_tool_result(gws_out)

            # Extract window title from the AX tree first AXWindow line.
            wt = re.search(r'AXWindow\s+"([^"]+)"', tree)
            if wt:
                window_title = wt.group(1)

        png_bytes_len = 0
        if png_b64:
            try:
                raw = base64.b64decode(png_b64, validate=False)
                png_bytes_len = len(raw)
                detected_width, detected_height = _image_dimensions_from_bytes(raw)
                if detected_width and detected_height:
                    width = detected_width
                    height = detected_height
            except Exception:
                png_bytes_len = len(png_b64) * 3 // 4

        return CaptureResult(
            mode=mode,
            width=width,
            height=height,
            png_b64=png_b64,
            elements=elements,
            app=app_name,
            window_title=window_title,
            png_bytes_len=png_bytes_len,
            image_mime_type=image_mime_type,
        )

    # ── Pointer ────────────────────────────────────────────────────
    def _apply_delivery(
        self,
        action: str,
        args: dict[str, Any],
        delivery_mode: str | None,
    ) -> ActionResult | None:
        """Attach delivery_mode to an input-action args dict.

        Background is the default and never needs a flag. Foreground is only
        sent when the live action schema accepts it; on an older driver that
        lacks the property we refuse with a structured
        ``foreground_unsupported`` result instead of silently downgrading to
        background (which would land the input somewhere the model didn't
        expect). Returns an ActionResult to short-circuit on refusal, or None
        to proceed. See NousResearch/hermes-agent#67052 phase B.
        """
        if not delivery_mode or delivery_mode == "background":
            return None
        if delivery_mode != "foreground":
            return ActionResult(
                ok=False, action=action, code="bad_delivery_mode",
                message=f"unknown delivery_mode {delivery_mode!r} — use background|foreground.",
            )
        # Foreground requested. Only send it if the driver understands it.
        if not self._session.supports_input_property(action, "delivery_mode"):
            return ActionResult(
                ok=False, action=action, code="foreground_unsupported",
                delivery_mode="foreground",
                message=(
                    "The connected cua-driver action schema does not accept "
                    "delivery_mode, so foreground delivery is unavailable. "
                    "Use another verified rung without assuming the reported "
                    "package version describes the live schema."
                ),
            )
        args["delivery_mode"] = "foreground"
        return None

    async def _run_input_action(
        self,
        action: str,
        args: dict[str, Any],
        delivery_mode: str | None,
        bring_to_front: bool,
    ) -> ActionResult:
        """Apply one delivery rung, optionally focusing via its own tool.

        ``bring_to_front`` is never an input-action property.  When explicitly
        requested, the separately approved standalone focus action runs first,
        then the original foreground input runs unchanged.
        """
        refusal = self._apply_delivery(action, args, delivery_mode)
        if refusal is not None:
            return refusal
        if bring_to_front:
            if delivery_mode != "foreground":
                return ActionResult(
                    ok=False,
                    action=action,
                    code="bring_to_front_requires_foreground",
                    message="bring_to_front requires delivery_mode='foreground'.",
                )
            if not self._session._has_tool("bring_to_front"):
                return ActionResult(
                    ok=False,
                    action=action,
                    code="bring_to_front_unsupported",
                    delivery_mode="foreground",
                    message="The connected cua-driver does not advertise the standalone bring_to_front tool.",
                )
            if self._active_pid is None or self._active_window_id is None:
                return ActionResult(
                    ok=False,
                    action=action,
                    code="bring_to_front_target_required",
                    delivery_mode="foreground",
                    message="Capture an exact target before requesting persistent foreground focus.",
                )
            focused = await self.bring_to_front(
                pid=self._active_pid,
                window_id=self._active_window_id,
            )
            if not focused.ok:
                return focused
        result = await self._action(action, args)
        if bring_to_front:
            result.meta["foreground_focus"] = {
                "invoked": True,
                "tool": "bring_to_front",
            }
        return result

    async def click(
        self,
        *,
        element: int | None = None,
        x: int | None = None,
        y: int | None = None,
        button: str = "left",
        click_count: int = 1,
        modifiers: list[str] | None = None,
        delivery_mode: str | None = None,
        bring_to_front: bool = False,
    ) -> ActionResult:
        pid = self._active_pid
        if pid is None:
            return ActionResult(ok=False, action="click",
                                message="No active window — call capture() first.")

        # Choose tool by click_count only — single-vs-double — and pass the
        # button through to `click`'s `button` enum (Surface 5 of
        # NousResearch/hermes-agent#47072). cua-driver-rs gained an explicit
        # `button: "left"|"right"|"middle"` arg on `click` in trycua/cua#1961
        # which rejects unknown buttons; before that, `middle` was silently
        # mapped to a left-click via name-routing through `right_click`.
        # `right_click`/`middle_click` MCP tools are deprecated aliases —
        # kept around but no longer invoked from here.
        button_norm = (button or "left").lower()
        if button_norm not in {"left", "right", "middle"}:
            return ActionResult(ok=False, action="click",
                                message=f"unknown button {button!r} — expected left, right, middle.")
        tool = "double_click" if click_count == 2 else "click"

        args: dict[str, Any] = {"pid": pid, "button": button_norm}
        if element is not None:
            if self._active_window_id is None:
                return ActionResult(ok=False, action=tool,
                                    message="No active window_id for element_index click.")
            args["element_index"] = element
            args["window_id"] = self._active_window_id
        elif x is not None and y is not None:
            if self._active_window_id is None:
                return ActionResult(ok=False, action=tool,
                                    message="No active window_id for coordinate click.")
            args["x"] = x
            args["y"] = y
            args["window_id"] = self._active_window_id
        else:
            return ActionResult(ok=False, action=tool,
                                message="click requires element= or x/y.")
        if modifiers:
            args["modifier"] = modifiers

        return await self._run_input_action(tool, args, delivery_mode, bring_to_front)

    async def drag(
        self,
        *,
        from_element: int | None = None,
        to_element: int | None = None,
        from_xy: tuple[int, int] | None = None,
        to_xy: tuple[int, int] | None = None,
        button: str = "left",
        modifiers: list[str] | None = None,
        delivery_mode: str | None = None,
        bring_to_front: bool = False,
    ) -> ActionResult:
        pid = self._active_pid
        if pid is None:
            return ActionResult(ok=False, action="drag",
                                message="No active window — call capture() first.")
        args: dict[str, Any] = {"pid": pid}
        if from_element is not None and to_element is not None:
            if self._active_window_id is None:
                return ActionResult(ok=False, action="drag",
                                    message="No active window_id for element-based drag.")
            args["from_element"] = from_element
            args["to_element"] = to_element
            args["window_id"] = self._active_window_id
        elif from_xy is not None and to_xy is not None:
            if self._active_window_id is None:
                return ActionResult(ok=False, action="drag",
                                    message="No active window_id for coordinate drag.")
            args["from_x"], args["from_y"] = int(from_xy[0]), int(from_xy[1])
            args["to_x"], args["to_y"] = int(to_xy[0]), int(to_xy[1])
            args["window_id"] = self._active_window_id
        else:
            return ActionResult(ok=False, action="drag",
                                message="drag requires from_element/to_element or from_coordinate/to_coordinate.")
        return await self._run_input_action("drag", args, delivery_mode, bring_to_front)

    async def scroll(
        self,
        *,
        direction: str,
        amount: int = 3,
        element: int | None = None,
        x: int | None = None,
        y: int | None = None,
        modifiers: list[str] | None = None,
        delivery_mode: str | None = None,
        bring_to_front: bool = False,
    ) -> ActionResult:
        pid = self._active_pid
        if pid is None:
            return ActionResult(ok=False, action="scroll",
                                message="No active window — call capture() first.")
        args: dict[str, Any] = {
            "pid": pid,
            "direction": direction,
            "amount": max(1, min(50, amount)),
        }
        if element is not None and self._active_window_id is not None:
            args["element_index"] = element
            args["window_id"] = self._active_window_id
        elif x is not None and y is not None:
            if self._active_window_id is None:
                return ActionResult(ok=False, action="scroll",
                                    message="No active window_id for coordinate scroll.")
            # CUA Driver 0.7.1 Linux schema rejects x/y on scroll. Only
            # include them when the driver explicitly advertises support
            # for coordinate scrolling; otherwise omit and let the driver
            # scroll the targeted window (window_id is still sent for
            # routing).  This is the safe default when capabilities
            # haven't been discovered yet (older drivers).
            if self._session.supports_capability(
                "input.scroll.coordinates", tool="scroll"
            ):
                args["x"] = x
                args["y"] = y
            args["window_id"] = self._active_window_id
        return await self._run_input_action("scroll", args, delivery_mode, bring_to_front)

    # ── Keyboard ───────────────────────────────────────────────────
    async def type_text(self, text: str, *, delivery_mode: str | None = None,
                  bring_to_front: bool = False) -> ActionResult:
        pid = self._active_pid
        window_id = self._active_window_id
        if pid is None or window_id is None:
            return ActionResult(ok=False, action="type_text",
                                message="No active window — call capture() first.")
        args: dict[str, Any] = {"pid": pid, "window_id": window_id, "text": text}
        return await self._run_input_action("type_text", args, delivery_mode, bring_to_front)

    async def key(self, keys: str, *, delivery_mode: str | None = None,
            bring_to_front: bool = False) -> ActionResult:
        pid = self._active_pid
        window_id = self._active_window_id
        if pid is None or window_id is None:
            return ActionResult(ok=False, action="key",
                                message="No active window — call capture() first.")

        key_name, modifiers = _parse_key_combo(keys)
        if not key_name:
            return ActionResult(ok=False, action="key",
                                message=f"Could not parse key from '{keys}'.")

        if modifiers:
            # hotkey requires at least one modifier + one key.
            args: dict[str, Any] = {"pid": pid, "window_id": window_id,
                                    "keys": modifiers + [key_name]}
            return await self._run_input_action("hotkey", args, delivery_mode, bring_to_front)
        else:
            args = {"pid": pid, "window_id": window_id, "key": key_name}
            return await self._run_input_action("press_key", args, delivery_mode, bring_to_front)

    # ── Value setter ────────────────────────────────────────────────
    async def set_value(self, value: str, element: int | None = None) -> ActionResult:
        """Set a value on an element. Handles AXPopUpButton selects natively."""
        pid = self._active_pid
        window_id = self._active_window_id
        if pid is None or window_id is None:
            return ActionResult(ok=False, action="set_value",
                                message="No active window — call capture() first.")
        if element is None:
            return ActionResult(ok=False, action="set_value",
                                message="set_value requires element= (element index).")
        args: dict[str, Any] = {
            "pid": pid,
            "window_id": window_id,
            "element_index": element,
            "value": value,
        }
        return await self._action("set_value", args)

    # ── Introspection ──────────────────────────────────────────────
    async def list_apps(self) -> list[dict[str, Any]]:
        out = await self._session.call_tool("list_apps", {"session": self._session_id})
        structured = out.get("structuredContent")
        data = out.get("data")

        # structuredContent is the canonical MCP payload. Empty lists fall
        # through so a populated compatibility envelope can still recover.
        if isinstance(structured, dict):
            apps = structured.get("apps")
            if isinstance(apps, list) and apps:
                return apps
        # Older drivers and direct CLI fallbacks may put apps in data instead.
        if isinstance(data, list) and data:
            return data
        if isinstance(data, dict):
            apps = data.get("apps")
            if isinstance(apps, list) and apps:
                return apps
        apps = out.get("apps")
        if isinstance(apps, list) and apps:
            return apps

        derived = _apps_from_windows(_windows_from_tool_result(out))
        if derived:
            return derived

        # Old text-only drivers retain a small, name/PID-only fallback.
        if isinstance(data, str):
            parsed_apps = []
            for line in data.splitlines():
                m = re.search(r'(.+?)\s+\(pid\s+(\d+)\)', line)
                if m:
                    parsed_apps.append({"name": m.group(1).strip(), "pid": int(m.group(2))})
            return parsed_apps
        return []

    async def list_windows(self) -> list[dict[str, Any]]:
        return await self._load_windows()

    async def focus_app(self, app: str, raise_window: bool = False) -> ActionResult:
        """Target an app, optionally invoking standalone foreground focus.

        cua-driver background-automation never needs to bring a window to the
        front: capture(app=...) already selects the right window via
        list_windows. We implement focus_app as a pure window-selector —
        enumerate on-screen windows, find the best match for *app*, and store
        its pid/window_id so that subsequent click/type calls hit the right
        process.

        The default remains non-disruptive. ``raise_window=True`` is explicit,
        separately approved by the Hermes adapter, and uses cua-driver's
        standalone ``bring_to_front`` tool rather than an action property.
        """
        try:
            windows = await self._load_windows()
        except Exception:
            self._clear_active_target()
            raise

        matched = await self._match_windows_for_app(windows, app)
        # Don't silently fall back to the frontmost window when the filter
        # matches nothing — that hides the real failure (often a localized
        # macOS app name mismatch, e.g. caller passed "Calculator" but
        # list_windows returns "計算機").
        target = matched[0] if matched else None
        if target:
            self._active_pid = target["pid"]
            self._active_window_id = target["window_id"]
            self._snapshot_tokens = {}
            self._last_app = target["app_name"] or app  # retained for back-compat diagnostics
            self._last_target = {
                "pid": self._active_pid,
                "window_id": self._active_window_id,
            }
            if raise_window:
                if not self._session._has_tool("bring_to_front"):
                    return ActionResult(
                        ok=False,
                        action="focus_app",
                        code="bring_to_front_unsupported",
                        message="The connected cua-driver does not advertise the standalone bring_to_front tool.",
                    )
                focused = await self.bring_to_front(
                    pid=self._active_pid,
                    window_id=self._active_window_id,
                )
                if not focused.ok:
                    return focused
                focused.action = "focus_app"
                focused.meta["target_selected"] = True
                return focused
            return ActionResult(
                ok=True, action="focus_app",
                message=f"Targeted {target['app_name']} (pid {self._active_pid}, "
                        f"window {self._active_window_id}) without raising window.",
            )
        self._clear_active_target()
        return ActionResult(ok=False, action="focus_app",
                            message=f"No on-screen window found for app '{app}'.")

    # ── App lifecycle ────────────────────────────────────────────────
    #
    # cua-driver exposes launch_app / kill_app / bring_to_front as a
    # complete set. focus_app() above is a *window-selector* (no
    # process state change); these methods drive the process layer.

    async def launch_app(
        self,
        *,
        bundle_id: str | None = None,
        name: str | None = None,
        urls: list[str] | None = None,
        additional_arguments: list[str] | None = None,
        creates_new_application_instance: bool = False,
    ) -> dict[str, Any]:
        """Idempotent launch. Returns ``{pid, bundle_id, name, windows[]}``
        so callers can skip an extra ``list_windows`` round-trip before
        ``get_window_state``.

        ``creates_new_application_instance=True`` forces a new instance
        even if the app is already running — use it when concurrent
        runs may touch the same app so each session gets its own
        isolated window."""
        if not bundle_id and not name:
            raise ValueError("launch_app requires either bundle_id or name")
        args: dict[str, Any] = {"session": self._session_id}
        if bundle_id:
            args["bundle_id"] = bundle_id
        if name:
            args["name"] = name
        if urls:
            args["urls"] = list(urls)
        if additional_arguments:
            args["additional_arguments"] = list(additional_arguments)
        if creates_new_application_instance:
            args["creates_new_application_instance"] = True
        out = await self._session.call_tool("launch_app", args)
        return out["structuredContent"] or {"data": out["data"]}

    async def kill_app(self, *, pid: int) -> ActionResult:
        """Terminate by pid. Equivalent to ``kill -9`` on POSIX,
        ``taskkill /F`` on Windows."""
        return await self._action("kill_app", {"pid": int(pid)})

    async def bring_to_front(self, *, pid: int,
                       window_id: int | None = None) -> ActionResult:
        """Activate a window so subsequent foreground-dispatched input
        lands on it. cua-driver's docstring notes this is the cheaper
        path than per-call SetForegroundWindow flashes."""
        args: dict[str, Any] = {"pid": int(pid)}
        if window_id is not None:
            args["window_id"] = int(window_id)
        # The live 0.9-era schema is strict and deliberately has no session
        # property. It is a standalone native focus operation, not a
        # session-scoped input action.
        return await self._action("bring_to_front", args, inject_session=False)

    # ── Typed browser (cua-driver 0.9 contract) ───────────────────
    async def typed_browser_state(self, **kwargs: Any) -> dict[str, Any]:
        """Exact-bind a native browser window or read fresh semantic state."""
        return await self._browser_route().observe(**kwargs)

    async def typed_browser_prepare(self, **kwargs: Any) -> dict[str, Any]:
        """Prepare an explicitly approved driver-owned browser profile."""
        return await self._browser_route().prepare(**kwargs)

    async def typed_browser_action(
        self,
        driver_tool: str,
        *,
        tab_id: str | None = None,
        args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one namespaced typed-browser mutation in this exact route."""
        return await self._browser_route().mutate(driver_tool, tab_id=tab_id, args=args)

    # ── Pointer + display introspection ─────────────────────────────

    async def move_cursor(self, x: int, y: int) -> ActionResult:
        """Move the agent-cursor *overlay* to a screen point. This is a
        visual hint — it does NOT move the real OS pointer (cua-driver
        explicitly avoids stealing pointer focus). The overlay glides
        smoothly to the target, so consumers use it before a click to
        give a visible "where the agent is going" cue."""
        return await self._action("move_cursor", {"x": int(x), "y": int(y)})

    async def get_cursor_position(self) -> tuple[int, int]:
        """Return the *real* OS cursor position in screen points
        (origin top-left)."""
        out = await self._session.call_tool(
            "get_cursor_position", {"session": self._session_id}
        )
        sc = out.get("structuredContent") or {}
        return int(sc.get("x", 0)), int(sc.get("y", 0))

    async def get_screen_size(self) -> dict[str, Any]:
        """Return the logical size of the main display in points plus
        its backing scale factor. Shape:
        ``{width, height, backing_scale_factor}``."""
        out = await self._session.call_tool(
            "get_screen_size", {"session": self._session_id}
        )
        return out.get("structuredContent") or {}

    async def zoom(self, *, window_id: int, x: float, y: float, w: float, h: float,
             factor: float = 1.0, format: str = "jpeg",
             quality: int = 85) -> dict[str, Any]:
        """Return a JPEG / PNG of a sub-region of a window, optionally
        scaled. cua-driver supports zoom-to-rect for callers that need
        a higher-resolution view of a specific element."""
        return await self._session.call_tool("zoom", {
            "window_id": int(window_id),
            "x": float(x), "y": float(y), "w": float(w), "h": float(h),
            "factor": float(factor),
            "format": format, "quality": int(quality),
            "session": self._session_id,
        })

    # ── Agent cursor (overlay) ──────────────────────────────────────
    #
    # Sessions (start_session/end_session, wired in start/stop) own the
    # cursor. These knobs tune its appearance + behavior per-session.
    # All accept an optional `cursor_id` to address a specific cursor
    # when the run drives multiple (rare); the default is this run's
    # session id.

    async def set_agent_cursor_enabled(self, enabled: bool, *,
                                 cursor_id: str | None = None) -> ActionResult:
        """Toggle the agent cursor overlay's visibility for this run."""
        args: dict[str, Any] = {"enabled": bool(enabled)}
        if cursor_id:
            args["cursor_id"] = cursor_id
        return await self._action("set_agent_cursor_enabled", args)

    async def set_agent_cursor_motion(self, *,
                                glide_ms: float | None = None,
                                dwell_ms: float | None = None,
                                idle_hide_ms: float | None = None,
                                cursor_id: str | None = None) -> ActionResult:
        """Tune the overlay's motion timings — glide duration, post-click
        dwell, idle-hide delay. Each None means "leave at current value"."""
        args: dict[str, Any] = {}
        if glide_ms is not None:
            args["glide_ms"] = float(glide_ms)
        if dwell_ms is not None:
            args["dwell_ms"] = float(dwell_ms)
        if idle_hide_ms is not None:
            args["idle_hide_ms"] = float(idle_hide_ms)
        if cursor_id:
            args["cursor_id"] = cursor_id
        return await self._action("set_agent_cursor_motion", args)

    async def set_agent_cursor_style(self, *,
                               gradient_colors: list[str] | None = None,
                               bloom_color: str | None = None,
                               image_path: str | None = None,
                               cursor_id: str | None = None) -> ActionResult:
        """Customise the cursor body. ``gradient_colors`` are CSS hex
        strings tip→tail; ``bloom_color`` is the radial halo; an
        ``image_path`` (.svg/.png/.ico) replaces the silhouette
        entirely. Empty values revert to the palette default."""
        args: dict[str, Any] = {}
        if gradient_colors is not None:
            args["gradient_colors"] = list(gradient_colors)
        if bloom_color is not None:
            args["bloom_color"] = bloom_color
        if image_path is not None:
            args["image_path"] = image_path
        if cursor_id:
            args["cursor_id"] = cursor_id
        return await self._action("set_agent_cursor_style", args)

    async def get_agent_cursor_state(self, *,
                               cursor_id: str | None = None) -> dict[str, Any]:
        """Return ``{x, y, config: {cursor_color, cursor_icon, ...},
        enabled}`` for this run's cursor (or the named ``cursor_id``)."""
        args: dict[str, Any] = {"session": self._session_id}
        if cursor_id:
            args["cursor_id"] = cursor_id
        out = await self._session.call_tool("get_agent_cursor_state", args)
        return out.get("structuredContent") or {}

    # ── Recording / replay ──────────────────────────────────────────

    async def start_recording(self, *, output_dir: str,
                        record_video: bool = False) -> dict[str, Any]:
        """Enable trajectory recording (per-turn screenshots + action
        JSON) to ``output_dir``. ``record_video=True`` ALSO captures
        the main display to ``<output_dir>/recording.mp4`` (H.264).
        Recording ownership is keyed by this run's session id so
        concurrent runs don't fight over the recorder."""
        out = await self._session.call_tool("start_recording", {
            "output_dir": output_dir,
            "record_video": bool(record_video),
            "session": self._session_id,
        })
        return out.get("structuredContent") or {}

    async def stop_recording(self) -> dict[str, Any]:
        """Disable recording and finalise the mp4 (if video was on).
        Returns the recorder's final state including ``last_video_path``."""
        out = await self._session.call_tool("stop_recording", {
            "session": self._session_id,
        })
        return out.get("structuredContent") or {}

    async def get_recording_state(self) -> dict[str, Any]:
        """Return the current recorder state without changing it.
        Shape: ``{recording, enabled, output_dir, next_turn,
        last_video_path, last_error, owner, video_active}``."""
        out = await self._session.call_tool(
            "get_recording_state", {"session": self._session_id}
        )
        return out.get("structuredContent") or {}

    async def replay_trajectory(self, *, trajectory_dir: str,
                          dry_run: bool = False,
                          speed_factor: float = 1.0) -> dict[str, Any]:
        """Replay a prior recording's turn stream by re-invoking each
        turn's tool call in lexical order. ``dry_run=True`` logs without
        actually firing the tools."""
        return await self._session.call_tool("replay_trajectory", {
            "trajectory_dir": trajectory_dir,
            "dry_run": bool(dry_run),
            "speed_factor": float(speed_factor),
            "session": self._session_id,
        })

    async def install_ffmpeg(self) -> dict[str, Any]:
        """Bootstrap ffmpeg for ``start_recording(record_video=True)``
        on Linux / Windows. macOS records natively via ScreenCaptureKit
        and doesn't need ffmpeg."""
        return await self._session.call_tool(
            "install_ffmpeg", {"session": self._session_id}
        )

    # ── Config ──────────────────────────────────────────────────────

    async def get_config(self) -> dict[str, Any]:
        """Return the current cua-driver runtime config."""
        out = await self._session.call_tool(
            "get_config", {"session": self._session_id}
        )
        return out.get("structuredContent") or {}

    async def set_config(self, **config) -> ActionResult:
        """Set cua-driver config keys. Common keys include
        ``max_image_dimension`` (image-output resizing), recording
        flags, etc. Unknown keys are passed through verbatim — cua-driver
        validates against its own schema."""
        return await self._action("set_config", dict(config))

    # ── Lower-level introspection ───────────────────────────────────

    async def get_accessibility_tree(self) -> dict[str, Any]:
        """Return a lightweight snapshot of running regular apps +
        on-screen visible windows with bounds, z-order, owner pid.
        Roughly the data ``list_windows`` exposes, in one call. Most
        callers should prefer ``capture()`` / ``focus_app()`` which
        already use this shape internally."""
        out = await self._session.call_tool(
            "get_accessibility_tree", {"session": self._session_id}
        )
        return out.get("structuredContent") or {"data": out["data"]}

    # ── Browser page tool ───────────────────────────────────────────

    async def page(self, *, pid: int, action: str,
             **page_args: Any) -> dict[str, Any]:
        """Interact with a browser page loaded in a running app (Chrome,
        Safari, Edge, ...). cua-driver routes through CDP / Apple Events
        / AX tree depending on the target. ``action`` + ``page_args``
        shape depends on the requested operation (e.g. ``action="eval"``
        takes ``js: str``); see cua-driver's ``page`` tool description
        for the full grammar."""
        args: dict[str, Any] = {
            "pid": int(pid),
            "action": action,
            "session": self._session_id,
        }
        args.update(page_args)
        return await self._session.call_tool("page", args)

    # ── Generic escape hatch ────────────────────────────────────────

    async def call_tool(  # noqa: ASYNC109 - upstream public signature
                  self, name: str, args: dict[str, Any] | None = None,
                  *, timeout: float = 30.0) -> dict[str, Any]:  # noqa: ASYNC109
        """Call any cua-driver MCP tool by name with arbitrary args.
        ``session`` is injected (preserves the caller's explicit one
        via setdefault). For tools the wrapper doesn't already type-
        wrap, this is the supported escape hatch — preferred over
        reaching for ``self._session.call_tool`` directly because it
        keeps the session-id contract consistent with everything else."""
        payload = dict(args) if args else {}
        payload.setdefault("session", self._session_id)
        return await self._session.call_tool(name, payload, timeout=timeout)

    # ── Internal ───────────────────────────────────────────────────
    def _maybe_attach_element_token(self, tool: str, args: dict[str, Any]) -> None:
        """Surface 6: when the wrapper is about to call a token-capable
        tool with `element_index`, look up the matching `element_token`
        from the last snapshot and attach it. cua-driver-rs's contract
        for combined args is documented in trycua/cua#1961:

          "element_token takes precedence over element_index when both
           supplied. Returns an explicit 'stale' error if the snapshot
           has been superseded."

        Gated on the per-tool capability claim so we don't send the
        field to drivers that predate the surface (which would reject
        the schema with `additionalProperties: false`).
        """
        idx = args.get("element_index")
        if not isinstance(idx, int):
            return
        token = self._snapshot_tokens.get(idx)
        if not token:
            return
        if not self._session.supports_capability(
            "accessibility.element_tokens", tool=tool
        ):
            return
        args["element_token"] = token

    async def _action(
        self,
        name: str,
        args: dict[str, Any],
        *,
        inject_session: bool = True,
    ) -> ActionResult:
        # Attach the snapshot's element_token whenever the call carries
        # an element_index and the target tool advertises support.
        self._maybe_attach_element_token(name, args)
        # Carry this run's session id so the cua-driver agent cursor
        # and per-session state (config overrides, recording ownership)
        # stay tied to this run. setdefault preserves any explicit
        # session a caller already supplied.
        if inject_session:
            args.setdefault("session", self._session_id)
        try:
            out = await self._session.call_tool(name, args)
        except Exception as e:
            logger.exception("cua-driver %s call failed", name)
            return ActionResult(ok=False, action=name, message=f"cua-driver error: {e}")
        ok = not out["isError"]
        data = out["data"]
        structured = out.get("structuredContent") or {}
        message = ""
        if isinstance(data, dict):
            message = str(data.get("message", ""))
        elif isinstance(data, str):
            message = data
        if not message and isinstance(structured, dict):
            message = str(structured.get("message", ""))
        # Merge data + structuredContent into meta for debugging, structured
        # winning on key overlap (it is the canonical verdict surface).
        meta: dict[str, Any] = {}
        if isinstance(data, dict):
            meta.update(data)
        if isinstance(structured, dict):
            meta.update(structured)
        return _action_result_from(name, ok, message, meta, structured,
                                   requested_delivery=args.get("delivery_mode"))
