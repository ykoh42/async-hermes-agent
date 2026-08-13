#!/usr/bin/env python3
"""
Code Execution Tool -- Programmatic Tool Calling (PTC)

Lets the LLM write a Python script that calls Hermes tools via RPC,
collapsing multi-step tool chains into a single inference turn.

Architecture (two transports):

  **Local backend (UDS):**
  1. Parent generates a `hermes_tools.py` stub module with UDS RPC functions
  2. Parent opens a Unix domain socket and starts an RPC listener task
  3. Parent spawns a child process that runs the LLM's script
  4. Tool calls travel over the UDS back to the parent for dispatch

  **Remote backends (file-based RPC):**
  1. Parent generates `hermes_tools.py` with file-based RPC stubs
  2. Parent ships both files to the remote environment
  3. Script runs inside the terminal backend (Docker/SSH/Modal/Daytona/etc.)
  4. Tool calls are written as request files; a polling task on the parent
     reads them via env.execute(), dispatches, and writes response files
  5. The script polls for response files and continues

In both cases, only the script's stdout is returned to the LLM; intermediate
tool results never enter the context window.

Local execution uses Unix domain sockets on POSIX and loopback TCP on Windows.
Remote execution additionally requires Python 3 in the terminal backend.
"""

import base64
import asyncio
import json
import logging
import os
import platform
import re
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid

_IS_WINDOWS = platform.system() == "Windows"
from typing import Any, Dict, List, Optional, Tuple

import aiofiles
import aiofiles.os
import aiofiles.tempfile

# Availability gate.  On Windows we fall back to loopback TCP for the
# sandbox RPC transport (AF_UNIX is unreliable on Windows Python) — see
# ``_use_tcp_rpc`` in ``_execute_local`` below.  That makes execute_code
# available on every platform Hermes itself runs on.
logger = logging.getLogger(__name__)

SANDBOX_AVAILABLE = True

# The 7 tools allowed inside the sandbox. The intersection of this list
# and the session's enabled tools determines which stubs are generated.
SANDBOX_ALLOWED_TOOLS = frozenset([
    "web_search",
    "web_extract",
    "read_file",
    "write_file",
    "search_files",
    "patch",
    "terminal",
])

# Resource limit defaults (overridable via config.yaml → code_execution.*)
DEFAULT_TIMEOUT = 300        # 5 minutes
DEFAULT_MAX_TOOL_CALLS = 50
MAX_STDOUT_BYTES = 50_000    # 50 KB
MAX_STDERR_BYTES = 10_000    # 10 KB


def _assemble_stdout_result(
    head: bytes,
    tail: bytes = b"",
    *,
    total_bytes: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Build display stdout plus explicit truncation metadata.

    The agent receives execute_code results as JSON. A textual truncation
    marker can be missed or later re-truncated by a client layer, so keep the
    marker for humans and also expose byte counts for deterministic handling.
    """
    captured = head + tail
    total = len(captured) if total_bytes is None else max(total_bytes, len(captured))
    truncated = total > len(captured)
    omitted = max(0, total - len(captured))

    if truncated:
        stdout_text = (
            head.decode("utf-8", errors="replace")
            + f"\n\n... [OUTPUT TRUNCATED - {omitted:,} bytes omitted "
            f"out of {total:,} total] ...\n\n"
            + tail.decode("utf-8", errors="replace")
        )
    else:
        stdout_text = captured.decode("utf-8", errors="replace")

    metadata: Dict[str, Any] = {
        "stdout_truncated": truncated,
        "stdout_bytes_captured": len(captured),
        "stdout_bytes_total": total,
        "stdout_bytes_omitted": omitted,
    }
    if truncated:
        metadata["warning"] = (
            "execute_code stdout was truncated; the script did run, but only "
            "the captured head/tail output is included. Re-run only with "
            "narrower output if the omitted data is required."
        )
    return stdout_text, metadata


def _truncate_stdout_text(stdout_text: str) -> Tuple[str, Dict[str, Any]]:
    """Cap a complete stdout string by bytes using the same head/tail policy."""
    stdout_bytes = stdout_text.encode("utf-8", errors="replace")
    if len(stdout_bytes) <= MAX_STDOUT_BYTES:
        return _assemble_stdout_result(stdout_bytes)

    head_bytes = int(MAX_STDOUT_BYTES * 0.4)
    tail_bytes = MAX_STDOUT_BYTES - head_bytes
    return _assemble_stdout_result(
        stdout_bytes[:head_bytes],
        stdout_bytes[-tail_bytes:],
        total_bytes=len(stdout_bytes),
    )

# Environment variable scrubbing rules (shared between the local + remote
# backends).  Secret-substring block is applied first; anything left must
# match a safe prefix, the operational HERMES_ allowlist, or (on Windows) an
# OS-essential name.  Delegate-task child context is also an exact-name
# operational marker: without it, a sandbox script that spawns/imports Hermes
# code can lose the DB-layer Kanban mutation guard while still inheriting
# HERMES_HOME.
#
# NB: the broad "HERMES_" prefix was deliberately removed (#27303) — it leaked
# HERMES_*-named config that lacks a secret substring (e.g. HERMES_BASE_URL,
# HERMES_KANBAN_DB, HERMES_*_WEBHOOK).  The child only needs the few
# location/profile vars in _HERMES_CHILD_ALLOWED below; HERMES_RPC_SOCKET /
# HERMES_RPC_DIR / TZ / HOME are injected explicitly after scrubbing.
_SAFE_ENV_PREFIXES = ("PATH", "HOME", "USER", "LANG", "LC_", "TERM",
                      "TMPDIR", "TMP", "TEMP", "SHELL", "LOGNAME",
                      "XDG_", "PYTHONPATH", "VIRTUAL_ENV", "CONDA")
_SECRET_SUBSTRINGS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL",
                      "PASSWD", "AUTH", "DSN", "WEBHOOK",
                      # Abbreviations that appear in real-world credential
                      # variable names but were previously undetected:
                      # CREDS (CREDENTIALS abbreviated), BEARER
                      # (Authorization: Bearer tokens), APIKEY (written
                      # without an underscore). "PASS" is intentionally NOT
                      # added — it false-positives on legitimate non-secret
                      # vars (BYPASS_CACHE, COMPASS_DIR, PASSENGER_HOST) while
                      # PASSWORD/PASSWD already cover the credential cases.
                      "CREDS", "BEARER", "APIKEY")

# Operational HERMES_* vars the child legitimately needs by exact name — these
# are non-secret runtime-location flags (the same set hermes_cli treats as the
# runtime location) that repo-root modules a sandbox script imports may read at
# import time.  None match _SECRET_SUBSTRINGS.
_HERMES_CHILD_ALLOWED = frozenset({
    "HERMES_HOME",
    "HERMES_PROFILE",
    "HERMES_CONFIG",
    "HERMES_ENV",
    "HERMES_DELEGATED_CHILD_CONTEXT",
})

# Windows-only: a handful of variables are required by the OS/CRT itself.
# Without them, even stdlib calls like ``socket.socket()`` fail with
# WinError 10106 (Winsock can't locate mswsock.dll) and ``subprocess``
# can't resolve cmd.exe.  These are well-known OS paths, not secrets, so
# we allow them through by exact name.  The _SECRET_SUBSTRINGS block
# still runs as a safety net (none of these names match those substrings).
_WINDOWS_ESSENTIAL_ENV_VARS = frozenset({
    "SYSTEMROOT",       # %SYSTEMROOT%\System32 — Winsock needs this
    "SYSTEMDRIVE",      # C: (or wherever Windows lives)
    "WINDIR",           # usually same as SYSTEMROOT
    "COMSPEC",          # cmd.exe path — subprocess shell=True needs it
    "PATHEXT",          # .COM;.EXE;.BAT;... — shell lookup
    "OS",               # "Windows_NT" — some tools gate on this
    "PROCESSOR_ARCHITECTURE",
    "NUMBER_OF_PROCESSORS",
    "PUBLIC",           # C:\Users\Public
    "ALLUSERSPROFILE",  # C:\ProgramData — some stdlib paths use it
    "PROGRAMDATA",      # C:\ProgramData
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "PROGRAMW6432",
    "APPDATA",          # %USERPROFILE%\AppData\Roaming — Python uses it
    "LOCALAPPDATA",     # %USERPROFILE%\AppData\Local
    "USERPROFILE",      # C:\Users\<name> — Python's expanduser uses it
    "USERDOMAIN",
    "USERNAME",
    "HOMEDRIVE",        # C:
    "HOMEPATH",         # \Users\<name>
    "COMPUTERNAME",
})


async def _scrub_child_env(source_env, is_passthrough=None, is_windows=None):
    """Produce the scrubbed child-process env for execute_code.

    Rules (order matters):
      1. Passthrough vars (skill- or config-declared) pass through the active
         profile secret scope; an absent scoped value is omitted and an
         unscoped multiplex read fails closed.
      2. Secret-substring names (KEY/TOKEN/DSN/WEBHOOK/etc.) are blocked.
      3. Names matching a safe prefix pass.
      4. Operational HERMES_* vars (_HERMES_CHILD_ALLOWED) pass by exact name.
      5. On Windows, a small OS-essential allowlist passes by exact name
         — without these the child can't even create a socket or spawn a
         subprocess.

    Extracted into a helper so tests can exercise the logic without
    spawning a subprocess.
    """
    resolve_passthrough_value = None
    if is_passthrough is None:
        try:
            from tools.env_passthrough import (
                is_env_passthrough as _ep,
                resolve_passthrough_value,
            )
        except Exception:
            _ep = lambda _: False  # noqa: E731
            resolve_passthrough_value = lambda _name, _fallback: None  # noqa: E731
        is_passthrough = _ep
    else:
        try:
            from tools.env_passthrough import resolve_passthrough_value
        except Exception:
            resolve_passthrough_value = lambda _name, _fallback: None  # noqa: E731
    if is_windows is None:
        is_windows = _IS_WINDOWS

    scrubbed = {}
    # Non-secret HERMES_* vars dropped by the tightened allowlist (#27303). The
    # broad "HERMES_" prefix used to pass these through; now only the
    # operational set does. The drop is intentional (those vars can carry
    # config like HERMES_KANBAN_DB / HERMES_BASE_URL), but a sandbox script
    # that imports a repo module reading one at import time would otherwise see
    # it silently unset. Surface the drop once so the behavior change is
    # diagnosable and points at the env_passthrough opt-in escape hatch.
    _dropped_hermes = []
    for k, v in source_env.items():
        passthrough = is_passthrough(k)
        if hasattr(passthrough, "__await__"):
            passthrough = await passthrough
        if passthrough:
            resolved = resolve_passthrough_value(k, v)
            if resolved is not None:
                scrubbed[k] = resolved
            continue
        if any(s in k.upper() for s in _SECRET_SUBSTRINGS):
            continue
        if any(k.startswith(p) for p in _SAFE_ENV_PREFIXES):
            scrubbed[k] = v
            continue
        if k in _HERMES_CHILD_ALLOWED:
            scrubbed[k] = v
            continue
        if is_windows and k.upper() in _WINDOWS_ESSENTIAL_ENV_VARS:
            scrubbed[k] = v
            continue
        if k.startswith("HERMES_"):
            # Non-secret (secrets were already dropped above) and not in any
            # allowlist — a deliberately-dropped HERMES_* var.
            _dropped_hermes.append(k)
    if _dropped_hermes:
        logger.debug(
            "execute_code: dropped %d non-allowlisted HERMES_* var(s) from the "
            "sandbox child env (%s). This is intentional hardening (#27303); if "
            "a sandbox script legitimately needs one, declare it via "
            "env_passthrough in the skill/config so it passes by explicit opt-in.",
            len(_dropped_hermes),
            ", ".join(sorted(_dropped_hermes)),
        )

    # delegate_task children are marked with a ContextVar, not os.environ, while
    # the execute_code sandbox crosses a process boundary. Bridge that context
    # into the child env and strip dispatcher-owned Kanban variables after the
    # normal secret/passthrough scrub so an explicit passthrough cannot re-grant
    # a delegated child the parent's board mutation capability.
    try:
        from agent.delegation_context import (
            is_delegated_child_process_context,
            scrub_kanban_env,
        )

        if is_delegated_child_process_context():
            scrubbed = scrub_kanban_env(scrubbed)
    except Exception:
        pass
    return scrubbed


async def check_sandbox_requirements() -> bool:
    """Check execute_code availability for the configured terminal backend."""
    if not SANDBOX_AVAILABLE:
        return False

    try:
        from tools.terminal_tool import (
            _get_env_config,
        )

        config = await _get_env_config()
    except Exception:
        logger.debug("Could not resolve terminal config for execute_code availability", exc_info=True)
        return False

    if config.get("env_type") == "vercel_sandbox":
        try:
            from tools.terminal_tool import _check_vercel_sandbox_requirements

            return bool(await _check_vercel_sandbox_requirements(config))
        except Exception:
            return False

    return True


# ---------------------------------------------------------------------------
# hermes_tools.py code generator
# ---------------------------------------------------------------------------

# Per-tool stub templates: (function_name, signature, docstring, args_dict_expr)
# The args_dict_expr builds the JSON payload sent over the RPC socket.
_TOOL_STUBS = {
    "web_search": (
        "web_search",
        "query: str, limit: int = 5",
        '"""Search the web. Returns dict with data.web list of {url, title, description}."""',
        '{"query": query, "limit": limit}',
    ),
    "web_extract": (
        "web_extract",
        "urls: list, char_limit: int = None",
        '"""Extract content from URLs (no LLM summarization). Returns dict with results list of {url, title, content, error}. Pages over char_limit (default 15000) are head+tail truncated with the full text stored on disk; the content footer gives the path. content is markdown."""',
        '{"urls": urls, "char_limit": char_limit}',
    ),
    "read_file": (
        "read_file",
        "path: str, offset: int = 1, limit: int = 2000",
        '"""Read a file (1-indexed lines). Returns dict with "content" and "total_lines"."""',
        '{"path": path, "offset": offset, "limit": limit}',
    ),
    "write_file": (
        "write_file",
        "path: str, content: str, cross_profile: bool = False",
        '"""Write content to a file (always overwrites). Returns dict with status. cross_profile=True opts out of the cross-Hermes-profile soft guard."""',
        '{"path": path, "content": content, "cross_profile": cross_profile}',
    ),
    "search_files": (
        "search_files",
        'pattern: str, target: str = "content", path: str = ".", file_glob: str = None, limit: int = 50, offset: int = 0, output_mode: str = "content", context: int = 0',
        '"""Search file contents (target="content") or find files by name (target="files"). Returns dict with "matches"."""',
        '{"pattern": pattern, "target": target, "path": path, "file_glob": file_glob, "limit": limit, "offset": offset, "output_mode": output_mode, "context": context}',
    ),
    "patch": (
        "patch",
        'path: str = None, old_string: str = None, new_string: str = None, replace_all: bool = False, mode: str = "replace", patch: str = None, cross_profile: bool = False',
        '"""Targeted find-and-replace (mode="replace") or V4A multi-file patches (mode="patch"). Returns dict with status. cross_profile=True opts out of the cross-Hermes-profile soft guard."""',
        '{"path": path, "old_string": old_string, "new_string": new_string, "replace_all": replace_all, "mode": mode, "patch": patch, "cross_profile": cross_profile}',
    ),
    "terminal": (
        "terminal",
        "command: str, timeout: int = None, workdir: str = None",
        '"""Run a shell command (foreground only). Returns dict with "output" and "exit_code"."""',
        '{"command": command, "timeout": timeout, "workdir": workdir}',
    ),
}


def _sandbox_failure_hint(stderr_text: str, enabled_tools=None) -> Optional[str]:
    """Map well-known sandbox script failures to one actionable recovery hint.

    Production mining (state.db): the top execute_code failure classes are
    hermes_tools import misuse (importing tools that aren't in the sandbox,
    23x in one window), calling the built-in helpers via import, treating
    tool results as strings instead of dicts, and importing third-party
    packages that don't exist in the sandbox interpreter. Bounded scan,
    first match wins, never raises.
    """
    if not stderr_text:
        return None
    window = stderr_text[:4000]
    try:
        m = re.search(
            r"cannot import name '(\w+)' from 'hermes_tools'", window
        )
        if m:
            missing = m.group(1)
            available = sorted(SANDBOX_ALLOWED_TOOLS & set(enabled_tools or SANDBOX_ALLOWED_TOOLS))
            builtin = {"json_parse", "shell_quote", "retry"}
            if missing in builtin:
                return (
                    f"{missing} is a BUILT-IN helper in the sandbox — no import "
                    f"needed. Remove it from the import line and call {missing}(...) directly."
                )
            return (
                f"'{missing}' is not available inside the execute_code sandbox. "
                f"Importable tools here: {', '.join(available)}. For anything "
                "else, use the normal tool call instead of execute_code."
            )
        m = re.search(r"NameError: name '(json_parse|shell_quote|retry)' is not defined", window)
        if m:
            return (
                f"{m.group(1)} is built into the generated sandbox module — "
                "call it directly at module scope without importing it."
            )
        m = re.search(r"ModuleNotFoundError: No module named '([\w.]+)'", window)
        if m:
            return (
                f"'{m.group(1)}' is not installed in the sandbox interpreter. "
                "Use Python stdlib inside execute_code, or run the code via "
                "terminal() with the project venv's python instead."
            )
        if re.search(r"TypeError: string indices must be integers|AttributeError: 'str' object has no attribute 'get'", window):
            return (
                "Tool functions in the sandbox return DICTS (already parsed) — "
                "do not json.loads() them or index them like strings. "
                "Example: read_file(path)['content']."
            )
    except Exception:
        return None
    return None


def generate_hermes_tools_module(enabled_tools: List[str],
                                 transport: str = "uds") -> str:
    """
    Build the source code for the hermes_tools.py stub module.

    Only tools in both SANDBOX_ALLOWED_TOOLS and enabled_tools get stubs.

    Args:
        enabled_tools: Tool names enabled in the current session.
        transport: ``"uds"`` for Unix domain socket (local backend) or
                   ``"file"`` for file-based RPC (remote backends).
    """
    tools_to_generate = sorted(SANDBOX_ALLOWED_TOOLS & set(enabled_tools))

    stub_functions = []
    export_names = []
    for tool_name in tools_to_generate:
        if tool_name not in _TOOL_STUBS:
            continue
        func_name, sig, doc, args_expr = _TOOL_STUBS[tool_name]
        stub_functions.append(
            f"def {func_name}({sig}):\n"
            f"    {doc}\n"
            f"    return _call({func_name!r}, {args_expr})\n"
        )
        export_names.append(func_name)

    if transport == "file":
        header = _FILE_TRANSPORT_HEADER
    else:
        header = _UDS_TRANSPORT_HEADER

    return header + "\n".join(stub_functions)


# ---- Shared helpers section (embedded in both transport headers) ----------

_COMMON_HELPERS = '''\

# ---------------------------------------------------------------------------
# Convenience helpers (avoid common scripting pitfalls)
# ---------------------------------------------------------------------------

def json_parse(text: str):
    """Parse JSON tolerant of control characters (strict=False).
    Use this instead of json.loads() when parsing output from terminal()
    or web_extract() that may contain raw tabs/newlines in strings."""
    return json.loads(text, strict=False)


def shell_quote(s: str) -> str:
    """Shell-escape a string for safe interpolation into commands.
    Use this when inserting dynamic content into terminal() commands:
        terminal(f"echo {shell_quote(user_input)}")
    """
    return shlex.quote(s)


def retry(fn, max_attempts=3, delay=2):
    """Retry a function up to max_attempts times with exponential backoff.
    Use for transient failures (network errors, API rate limits):
        result = retry(lambda: terminal("gh issue list ..."))
    """
    last_err = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt < max_attempts - 1:
                time.sleep(delay * (2 ** attempt))
    raise last_err

'''

# ---- UDS transport (local backend) ---------------------------------------

_UDS_TRANSPORT_HEADER = '''\
"""Auto-generated Hermes tools RPC stubs."""
import json, os, socket, shlex, threading, time

_sock = None
# The RPC server handles a single client connection serially and has no
# request-id in the protocol, so concurrent _call() invocations from multiple
# threads (e.g. ThreadPoolExecutor) would race on the shared socket and get
# each other's responses. Serialize the entire send+recv round-trip.
_call_lock = threading.Lock()
''' + _COMMON_HELPERS + '''\

def _connect():
    """Connect to the parent's RPC server via the transport it picked.

    HERMES_RPC_SOCKET can be either:
      - a filesystem path (POSIX Unix domain socket — the default on
        Linux and macOS)
      - a string of the form ``tcp://127.0.0.1:<port>`` (Windows, where
        AF_UNIX is unreliable — the parent falls back to loopback TCP)
    """
    global _sock
    if _sock is None:
        endpoint = os.environ["HERMES_RPC_SOCKET"]
        if endpoint.startswith("tcp://"):
            # tcp://host:port  (host is always 127.0.0.1 in practice — we
            # only bind loopback server-side)
            _host_port = endpoint[len("tcp://"):]
            _host, _, _port = _host_port.rpartition(":")
            _sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            _sock.connect((_host or "127.0.0.1", int(_port)))
        else:
            _sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            _sock.connect(endpoint)
        _sock.settimeout(300)
    return _sock

def _call(tool_name, args):
    """Send a tool call to the parent process and return the parsed result."""
    request = json.dumps({
        "tool": tool_name,
        "args": args,
        "token": os.environ.get("HERMES_RPC_TOKEN", ""),
    }) + "\\n"
    with _call_lock:
        conn = _connect()
        conn.sendall(request.encode())
        buf = b""
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                raise RuntimeError("Agent process disconnected")
            buf += chunk
            if buf.endswith(b"\\n"):
                break
    raw = buf.decode().strip()
    result = json.loads(raw)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return result
    return result

'''

# ---- File-based transport (remote backends) -------------------------------

_FILE_TRANSPORT_HEADER = '''\
"""Auto-generated Hermes tools RPC stubs (file-based transport)."""
import json, os, shlex, tempfile, threading, time

_RPC_DIR = os.environ.get("HERMES_RPC_DIR") or os.path.join(tempfile.gettempdir(), "hermes_rpc")
_seq = 0
# `_seq += 1` is not atomic (read-modify-write), so concurrent _call()
# invocations from multiple threads could allocate the same sequence number
# and clobber each other's request files. Guard seq allocation with a lock.
_seq_lock = threading.Lock()
''' + _COMMON_HELPERS + '''\

def _call(tool_name, args):
    """Send a tool call request via file-based RPC and wait for response."""
    global _seq
    with _seq_lock:
        _seq += 1
        seq = _seq
    seq_str = f"{seq:06d}"
    req_file = os.path.join(_RPC_DIR, f"req_{seq_str}")
    res_file = os.path.join(_RPC_DIR, f"res_{seq_str}")

    # Write request atomically (write to .tmp, then rename).
    # encoding="utf-8" is critical: on Windows-hosted remote backends
    # (or any non-UTF-8 locale) the default open() mode would mangle
    # non-ASCII chars in tool args when encoding them as JSON.
    tmp = req_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({
            "tool": tool_name,
            "args": args,
            "seq": seq,
            "token": os.environ.get("HERMES_RPC_TOKEN", ""),
        }, f)
    os.rename(tmp, req_file)

    # Wait for response with adaptive polling
    deadline = time.monotonic() + 300  # 5-minute timeout per tool call
    poll_interval = 0.05  # Start at 50ms
    while not os.path.exists(res_file):
        if time.monotonic() > deadline:
            raise RuntimeError(f"RPC timeout: no response for {tool_name} after 300s")
        time.sleep(poll_interval)
        poll_interval = min(poll_interval * 1.2, 0.25)  # Back off to 250ms

    with open(res_file, encoding="utf-8") as f:
        raw = f.read()

    # Clean up response file
    try:
        os.unlink(res_file)
    except OSError:
        pass

    result = json.loads(raw)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return result
    return result

'''


# ---------------------------------------------------------------------------
# RPC server (runs as an owned task in the parent event loop)
# ---------------------------------------------------------------------------

# Terminal parameters that must not be used from ephemeral sandbox scripts
_TERMINAL_BLOCKED_PARAMS = {"background", "pty", "notify_on_complete", "watch_patterns"}


async def _dispatch_rpc_request(
    request: dict,
    *,
    task_id: str,
    tool_call_log: list,
    tool_call_counter: list,
    max_tool_calls: int,
    allowed_tools: frozenset,
    rpc_token: str,
) -> str:
    """Validate and dispatch one sandbox RPC request."""
    if not rpc_token or not secrets.compare_digest(
        str(request.get("token") or "").encode(), rpc_token.encode()
    ):
        return tool_error("Unauthorized RPC request")

    tool_name = request.get("tool", "")
    tool_args = request.get("args", {})
    if tool_name not in allowed_tools:
        available = ", ".join(sorted(allowed_tools))
        return tool_error(
            f"Tool '{tool_name}' is not available in execute_code. "
            f"Available: {available}"
        )
    if tool_call_counter[0] >= max_tool_calls:
        return tool_error(
            f"Tool call limit reached ({max_tool_calls}). "
            "No more tool calls allowed in this execution."
        )
    if tool_name == "terminal" and isinstance(tool_args, dict):
        for param in _TERMINAL_BLOCKED_PARAMS:
            tool_args.pop(param, None)

    call_start = time.monotonic()
    try:
        from model_tools import handle_function_call

        result = await handle_function_call(
            tool_name,
            tool_args if isinstance(tool_args, dict) else {},
            task_id=task_id,
        )
        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("Tool call failed in sandbox: %s", exc, exc_info=True)
        result = tool_error(str(exc))

    tool_call_counter[0] += 1
    tool_call_log.append({
        "tool": tool_name,
        "args_preview": str(tool_args)[:80],
        "duration": round(time.monotonic() - call_start, 2),
    })
    return result


async def _rpc_server_loop(
    server_sock: socket.socket,
    task_id: str,
    tool_call_log: list,
    tool_call_counter: list,
    max_tool_calls: int,
    allowed_tools: frozenset,
    stop_event: asyncio.Event,
    rpc_token: str,
):
    """Serve the local sandbox RPC socket without blocking the event loop."""
    loop = asyncio.get_running_loop()
    server_sock.setblocking(False)
    conn = None
    try:
        while not stop_event.is_set():
            try:
                conn, _ = await asyncio.wait_for(
                    loop.sock_accept(server_sock),
                    timeout=0.1,
                )
                break
            except TimeoutError:
                continue
            except OSError:
                if stop_event.is_set():
                    return
                raise
        if conn is None:
            return
        conn.setblocking(False)

        buffer = b""
        while not stop_event.is_set():
            try:
                chunk = await asyncio.wait_for(
                    loop.sock_recv(conn, 65536),
                    timeout=0.25,
                )
            except TimeoutError:
                continue
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    request = json.loads(line.decode())
                    if not isinstance(request, dict):
                        raise ValueError("request must be an object")
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                    response = tool_error(f"Invalid RPC request: {exc}")
                else:
                    response = await _dispatch_rpc_request(
                        request,
                        task_id=task_id,
                        tool_call_log=tool_call_log,
                        tool_call_counter=tool_call_counter,
                        max_tool_calls=max_tool_calls,
                        allowed_tools=allowed_tools,
                        rpc_token=rpc_token,
                    )
                await loop.sock_sendall(conn, (response + "\n").encode())
    except asyncio.CancelledError:
        raise
    except OSError as exc:
        if not stop_event.is_set():
            logger.debug("RPC listener socket error: %s", exc, exc_info=True)
    finally:
        if conn is not None:
            conn.close()


# ---------------------------------------------------------------------------
# Remote execution support (file-based RPC via terminal backend)
# ---------------------------------------------------------------------------

async def _get_or_create_env(task_id: str):
    """Get or create the terminal environment for task_id."""
    from tools.terminal_tool import _get_env_config, _get_or_create_environment

    env = await _get_or_create_environment(task_id)
    config = await _get_env_config()
    return env, config["env_type"]


async def _ship_file_to_remote(env, remote_path: str, content: str) -> None:
    """Write content to a remote path through the async environment."""
    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
    quoted_remote_path = shlex.quote(remote_path)
    await env.execute(
        f"echo '{encoded}' | base64 -d > {quoted_remote_path}",
        cwd="/",
        timeout=30,
    )


async def _env_temp_dir(env: Any) -> str:
    """Return a writable temp dir for env-backed execute_code sandboxes."""
    get_temp_dir = getattr(env, "get_temp_dir", None)
    if callable(get_temp_dir):
        try:
            temp_dir = get_temp_dir()
            if hasattr(temp_dir, "__await__"):
                temp_dir = await temp_dir
            if isinstance(temp_dir, str) and temp_dir.startswith("/"):
                return temp_dir.rstrip("/") or "/"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Could not resolve execute_code env temp dir: %s", exc)
    return "/tmp"


async def _rpc_poll_loop(
    env,
    rpc_dir: str,
    task_id: str,
    tool_call_log: list,
    tool_call_counter: list,
    max_tool_calls: int,
    allowed_tools: frozenset,
    stop_event: asyncio.Event,
    rpc_token: str,
):
    """Poll remote request files and dispatch without a worker thread."""
    poll_interval = 0.1
    quoted_rpc_dir = shlex.quote(rpc_dir)
    while not stop_event.is_set():
        try:
            ls_result = await env.execute(
                f"ls -1 {quoted_rpc_dir}/req_* 2>/dev/null || true",
                cwd="/",
                timeout=10,
            )
            output = ls_result.get("output", "").strip()
            req_files = sorted(
                value.strip()
                for value in output.splitlines()
                if value.strip()
                and not value.strip().endswith(".tmp")
                and "/req_" in value.strip()
            )
            for req_file in req_files:
                if stop_event.is_set():
                    break
                quoted_req_file = shlex.quote(req_file)
                read_result = await env.execute(
                    f"cat {quoted_req_file}",
                    cwd="/",
                    timeout=10,
                )
                try:
                    request = json.loads(read_result.get("output", ""))
                    if not isinstance(request, dict):
                        raise ValueError("request must be an object")
                except (json.JSONDecodeError, ValueError):
                    logger.debug("Malformed RPC request in %s", req_file)
                    await env.execute(
                        f"rm -f {quoted_req_file}",
                        cwd="/",
                        timeout=5,
                    )
                    continue

                if not rpc_token or not secrets.compare_digest(
                    str(request.get("token") or "").encode(),
                    rpc_token.encode(),
                ):
                    logger.debug("Unauthorized RPC request in %s", req_file)
                    await env.execute(
                        f"rm -f {quoted_req_file}",
                        cwd="/",
                        timeout=5,
                    )
                    continue

                result = await _dispatch_rpc_request(
                    request,
                    task_id=task_id,
                    tool_call_log=tool_call_log,
                    tool_call_counter=tool_call_counter,
                    max_tool_calls=max_tool_calls,
                    allowed_tools=allowed_tools,
                    rpc_token=rpc_token,
                )
                seq = request.get("seq", 0)
                try:
                    seq_str = f"{int(seq):06d}"
                except (TypeError, ValueError):
                    seq_str = "000000"
                res_file = f"{rpc_dir}/res_{seq_str}"
                quoted_res_file = shlex.quote(res_file)
                encoded_result = base64.b64encode(
                    result.encode("utf-8")
                ).decode("ascii")
                await env.execute(
                    f"echo '{encoded_result}' | base64 -d > {quoted_res_file}.tmp"
                    f" && mv {quoted_res_file}.tmp {quoted_res_file}",
                    cwd="/",
                    timeout=60,
                )
                await env.execute(
                    f"rm -f {quoted_req_file}",
                    cwd="/",
                    timeout=5,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not stop_event.is_set():
                logger.debug("RPC poll error: %s", exc, exc_info=True)

        if not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
            except TimeoutError:
                pass


async def _execute_remote(
    code: str,
    task_id: Optional[str],
    enabled_tools: Optional[List[str]],
) -> str:
    """Run a script on a remote terminal backend via file-based RPC."""
    cfg = await _load_config()
    timeout = cfg.get("timeout", DEFAULT_TIMEOUT)
    max_tool_calls = cfg.get("max_tool_calls", DEFAULT_MAX_TOOL_CALLS)
    session_tools = set(enabled_tools) if enabled_tools else set()
    sandbox_tools = frozenset(SANDBOX_ALLOWED_TOOLS & session_tools)
    if not sandbox_tools:
        sandbox_tools = SANDBOX_ALLOWED_TOOLS

    effective_task_id = task_id or "default"
    env, env_type = await _get_or_create_env(effective_task_id)
    sandbox_id = uuid.uuid4().hex[:12]
    temp_dir = await _env_temp_dir(env)
    sandbox_dir = f"{temp_dir}/hermes_exec_{sandbox_id}"
    quoted_sandbox_dir = shlex.quote(sandbox_dir)
    quoted_rpc_dir = shlex.quote(f"{sandbox_dir}/rpc")
    tool_call_log: list = []
    tool_call_counter = [0]
    exec_start = time.monotonic()
    stop_event = asyncio.Event()
    rpc_task = None
    stdout_text = ""
    exit_code = -1
    status = "success"

    try:
        py_check = await env.execute(
            "command -v python3 >/dev/null 2>&1 && echo OK",
            cwd="/",
            timeout=15,
        )
        if "OK" not in py_check.get("output", ""):
            return json.dumps({
                "status": "error",
                "error": (
                    f"Python 3 is not available in the {env_type} terminal "
                    "environment. Install Python to use execute_code with "
                    "remote backends."
                ),
                "tool_calls_made": 0,
                "duration_seconds": 0,
            })

        await env.execute(f"mkdir -p {quoted_rpc_dir}", cwd="/", timeout=10)
        rpc_token = secrets.token_urlsafe(32)
        tools_src = generate_hermes_tools_module(
            list(sandbox_tools),
            transport="file",
        )
        await _ship_file_to_remote(
            env,
            f"{sandbox_dir}/hermes_tools.py",
            tools_src,
        )
        await _ship_file_to_remote(env, f"{sandbox_dir}/script.py", code)

        rpc_task = asyncio.create_task(
            _rpc_poll_loop(
                env,
                f"{sandbox_dir}/rpc",
                effective_task_id,
                tool_call_log,
                tool_call_counter,
                max_tool_calls,
                sandbox_tools,
                stop_event,
                rpc_token,
            ),
            name=f"execute-code-rpc-{sandbox_id}",
        )

        env_prefix = (
            f"HERMES_RPC_DIR={shlex.quote(f'{sandbox_dir}/rpc')} "
            f"HERMES_RPC_TOKEN={shlex.quote(rpc_token)} "
            "PYTHONDONTWRITEBYTECODE=1"
        )
        timezone = os.getenv("HERMES_TIMEZONE", "").strip()
        if timezone:
            env_prefix += f" TZ={shlex.quote(timezone)}"

        logger.info(
            "Executing code on %s backend (task %s)...",
            env_type,
            effective_task_id[:8],
        )
        script_result = await env.execute(
            f"cd {quoted_sandbox_dir} && {env_prefix} python3 script.py",
            timeout=timeout,
        )
        stdout_text = script_result.get("output", "") or ""
        exit_code = script_result.get("returncode", -1)
        if exit_code == 124:
            status = "timeout"
        elif exit_code == 130:
            status = "interrupted"
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        duration = round(time.monotonic() - exec_start, 2)
        logger.error(
            "execute_code remote failed after %ss with %d tool calls: %s: %s",
            duration,
            tool_call_counter[0],
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return json.dumps({
            "status": "error",
            "error": str(exc),
            "tool_calls_made": tool_call_counter[0],
            "duration_seconds": duration,
        }, ensure_ascii=False)
    finally:
        cleanup_task = asyncio.create_task(
            _cleanup_remote_execution(
                env,
                quoted_sandbox_dir,
                stop_event,
                rpc_task,
                sandbox_dir,
            ),
            name=f"execute-code-cleanup-{sandbox_id}",
        )
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            await asyncio.shield(cleanup_task)
            raise

    duration = round(time.monotonic() - exec_start, 2)
    stdout_text, stdout_metadata = _truncate_stdout_text(stdout_text)
    from tools.ansi_strip import strip_ansi
    stdout_text = strip_ansi(stdout_text)
    from agent.redact import redact_sensitive_text
    stdout_text = redact_sensitive_text(stdout_text, code_file=True)

    result: Dict[str, Any] = {
        "status": status,
        "output": stdout_text,
        "exit_code": exit_code,
        "tool_calls_made": tool_call_counter[0],
        "duration_seconds": duration,
    }
    result.update(stdout_metadata)
    if status == "timeout":
        timeout_msg = f"Script timed out after {timeout}s and was killed."
        result["error"] = timeout_msg
        result["output"] = (
            stdout_text + f"\n\n⏰ {timeout_msg}"
            if stdout_text
            else f"⏰ {timeout_msg}"
        )
    elif status == "interrupted":
        result["output"] = (
            stdout_text
            + "\n[execution interrupted — user sent a new message]"
        )
    elif exit_code != 0:
        result["status"] = "error"
        result["error"] = f"Script exited with code {exit_code}"
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def _read_stdout_head_tail(
    stream: asyncio.StreamReader,
) -> tuple[bytes, bytes, int]:
    head_limit = int(MAX_STDOUT_BYTES * 0.4)
    tail_limit = MAX_STDOUT_BYTES - head_limit
    head = bytearray()
    tail = bytearray()
    total = 0
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            break
        total += len(chunk)
        if len(head) < head_limit:
            take = min(head_limit - len(head), len(chunk))
            head.extend(chunk[:take])
            chunk = chunk[take:]
        if chunk:
            tail.extend(chunk)
            if len(tail) > tail_limit:
                del tail[: len(tail) - tail_limit]
    return bytes(head), bytes(tail), total


async def _read_stderr_head(stream: asyncio.StreamReader) -> bytes:
    captured = bytearray()
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            break
        if len(captured) < MAX_STDERR_BYTES:
            captured.extend(chunk[: MAX_STDERR_BYTES - len(captured)])
    return bytes(captured)


async def _cleanup_remote_execution(
    env: Any,
    quoted_sandbox_dir: str,
    stop_event: asyncio.Event,
    rpc_task: asyncio.Task | None,
    sandbox_dir: str,
) -> None:
    """Stop owned remote work and remove the per-run sandbox directory."""
    stop_event.set()
    if rpc_task is not None:
        done, pending = await asyncio.wait({rpc_task}, timeout=5)
        if pending:
            rpc_task.cancel()
        await asyncio.gather(*done, *pending, return_exceptions=True)
    try:
        await env.execute(
            f"rm -rf {quoted_sandbox_dir}",
            cwd="/",
            timeout=15,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug(
            "Failed to clean up remote sandbox %s",
            sandbox_dir,
        )


async def _finish_rpc_task(
    task: asyncio.Task | None,
    stop_event: asyncio.Event,
    server_sock: socket.socket | None = None,
) -> None:
    stop_event.set()
    if server_sock is not None:
        server_sock.close()
    if task is None:
        return
    done, pending = await asyncio.wait({task}, timeout=3)
    if pending:
        task.cancel()
    await asyncio.gather(*done, *pending, return_exceptions=True)


async def _cleanup_local_execution(
    process: asyncio.subprocess.Process | None,
    process_tasks: tuple[asyncio.Task | None, ...],
    rpc_task: asyncio.Task | None,
    stop_event: asyncio.Event,
    server_sock: socket.socket | None,
    sock_path: str | None,
) -> None:
    """Reap the child and close all per-run local resources."""
    if process is not None and process.returncode is None:
        await _kill_process_group(process, escalate=True)
    tasks = tuple(task for task in process_tasks if task is not None)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    await _finish_rpc_task(rpc_task, stop_event, server_sock)
    if sock_path:
        try:
            await aiofiles.os.remove(sock_path)
        except FileNotFoundError:
            pass


async def execute_code(
    code: str,
    task_id: Optional[str] = None,
    enabled_tools: Optional[List[str]] = None,
) -> str:
    """Run a Python script with RPC access to the retained Hermes tools."""
    if not SANDBOX_AVAILABLE:
        return tool_error(
            "execute_code sandbox is unavailable in this environment. "
            "Use normal tool calls (terminal, read_file, write_file, ...) instead."
        )
    if not code or not code.strip():
        return tool_error("No code provided.")

    from tools.terminal_tool import (
        _docker_has_host_access,
        _get_env_config,
    )

    env_config = await _get_env_config()
    env_type = env_config["env_type"]

    from tools.approval import check_execute_code_guard

    guard = await check_execute_code_guard(
        code,
        env_type,
        has_host_access=_docker_has_host_access(env_config),
    )
    if not guard.get("approved", False):
        return json.dumps({
            "status": "error",
            "error": (
                guard.get("message")
                or "execute_code blocked by approval guard."
            ),
            "tool_calls_made": 0,
            "duration_seconds": 0,
        }, ensure_ascii=False)
    if guard.get("user_approved"):
        from tools.interrupt import clear_current_thread_interrupt
        clear_current_thread_interrupt()

    if env_type != "local":
        return await _execute_remote(code, task_id, enabled_tools)

    from tools.interrupt import is_interrupted as _is_interrupted

    cfg = await _load_config()
    timeout = cfg.get("timeout", DEFAULT_TIMEOUT)
    max_tool_calls = cfg.get("max_tool_calls", DEFAULT_MAX_TOOL_CALLS)
    session_tools = set(enabled_tools) if enabled_tools else set()
    sandbox_tools = frozenset(SANDBOX_ALLOWED_TOOLS & session_tools)
    if not sandbox_tools:
        sandbox_tools = SANDBOX_ALLOWED_TOOLS

    tool_call_log: list = []
    tool_call_counter = [0]
    exec_start = time.monotonic()
    server_sock: socket.socket | None = None
    rpc_task: asyncio.Task | None = None
    process: asyncio.subprocess.Process | None = None
    stdout_task: asyncio.Task | None = None
    stderr_task: asyncio.Task | None = None
    wait_task: asyncio.Task | None = None
    stop_event = asyncio.Event()
    sock_path: str | None = None

    try:
        async with aiofiles.tempfile.TemporaryDirectory(
            prefix="hermes_sandbox_"
        ) as tmpdir:
            tools_path = os.path.join(tmpdir, "hermes_tools.py")
            script_path = os.path.join(tmpdir, "script.py")
            tools_src = generate_hermes_tools_module(list(sandbox_tools))
            async with aiofiles.open(tools_path, "w", encoding="utf-8") as handle:
                await handle.write(tools_src)
            async with aiofiles.open(script_path, "w", encoding="utf-8") as handle:
                await handle.write(code)

            rpc_token = secrets.token_urlsafe(32)
            use_tcp_rpc = _IS_WINDOWS
            if use_tcp_rpc:
                server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_sock.bind(("127.0.0.1", 0))
                host, port = server_sock.getsockname()[:2]
                rpc_endpoint = f"tcp://{host}:{port}"
            else:
                sock_path = f"/tmp/hermes_rpc_{uuid.uuid4().hex}.sock"
                server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                server_sock.bind(sock_path)
                await aiofiles.os.wrap(os.chmod)(sock_path, 0o600)
                rpc_endpoint = sock_path
            server_sock.listen(1)
            server_sock.setblocking(False)
            rpc_task = asyncio.create_task(
                _rpc_server_loop(
                    server_sock,
                    task_id or "default",
                    tool_call_log,
                    tool_call_counter,
                    max_tool_calls,
                    sandbox_tools,
                    stop_event,
                    rpc_token,
                ),
                name="execute-code-rpc-server",
            )

            child_env = await _scrub_child_env(os.environ)
            child_env["HERMES_RPC_SOCKET"] = rpc_endpoint
            child_env["HERMES_RPC_TOKEN"] = rpc_token
            child_env["PYTHONDONTWRITEBYTECODE"] = "1"
            child_env["PYTHONIOENCODING"] = "utf-8"
            child_env["PYTHONUTF8"] = "1"
            hermes_root = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
            python_path = child_env.get("PYTHONPATH", "")
            parts = [tmpdir, hermes_root]
            if python_path:
                parts.append(python_path)
            child_env["PYTHONPATH"] = os.pathsep.join(parts)
            timezone = os.getenv("HERMES_TIMEZONE", "").strip()
            if timezone:
                child_env["TZ"] = timezone
            child_env.pop("HERMES_TIMEZONE", None)

            from hermes_constants import apply_subprocess_home_env
            await apply_subprocess_home_env(child_env)

            mode = await _get_execution_mode()
            child_python = await _resolve_child_python(mode)
            child_cwd = await _resolve_child_cwd(
                mode,
                tmpdir,
                task_id=task_id or "",
            )
            creationflags = (
                subprocess.CREATE_NO_WINDOW
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                if _IS_WINDOWS
                else 0
            )
            process = await asyncio.create_subprocess_exec(
                child_python,
                script_path,
                cwd=child_cwd,
                env=child_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                start_new_session=not _IS_WINDOWS,
                creationflags=creationflags,
            )
            assert process.stdout is not None
            assert process.stderr is not None
            stdout_task = asyncio.create_task(
                _read_stdout_head_tail(process.stdout),
                name="execute-code-stdout",
            )
            stderr_task = asyncio.create_task(
                _read_stderr_head(process.stderr),
                name="execute-code-stderr",
            )
            wait_task = asyncio.create_task(
                process.wait(),
                name="execute-code-process",
            )
            deadline = time.monotonic() + timeout
            status = "success"
            poll_interval = 0.005
            activity_state = {
                "last_touch": time.monotonic(),
                "start": exec_start,
            }
            try:
                from tools.environments.base import touch_activity_if_due
            except Exception:
                touch_activity_if_due = None

            while not wait_task.done():
                if _is_interrupted():
                    await _kill_process_group(process)
                    status = "interrupted"
                    break
                now = time.monotonic()
                if now > deadline:
                    await _kill_process_group(process, escalate=True)
                    status = "timeout"
                    break
                if touch_activity_if_due is not None:
                    try:
                        touch_activity_if_due(
                            activity_state,
                            "execute_code running",
                        )
                    except Exception:
                        pass
                await asyncio.wait(
                    {wait_task},
                    timeout=min(
                        poll_interval,
                        max(0.0, deadline - now),
                    ),
                )
                poll_interval = min(0.2, poll_interval * 1.5)

            await wait_task
            head, tail, total_bytes = await stdout_task
            stderr_bytes = await stderr_task
            stdout_text, stdout_metadata = _assemble_stdout_result(
                head,
                tail,
                total_bytes=total_bytes,
            )
            stderr_text = stderr_bytes.decode("utf-8", errors="replace")
            exit_code = (
                process.returncode
                if process.returncode is not None
                else -1
            )
            duration = round(time.monotonic() - exec_start, 2)

            await _finish_rpc_task(rpc_task, stop_event, server_sock)
            rpc_task = None
            server_sock = None

            from tools.ansi_strip import strip_ansi
            stdout_text = strip_ansi(stdout_text)
            stderr_text = strip_ansi(stderr_text)
            from agent.redact import redact_sensitive_text
            stdout_text = redact_sensitive_text(stdout_text, code_file=True)
            stderr_text = redact_sensitive_text(stderr_text, code_file=True)

            result: Dict[str, Any] = {
                "status": status,
                "output": stdout_text,
                "exit_code": exit_code,
                "tool_calls_made": tool_call_counter[0],
                "duration_seconds": duration,
            }
            result.update(stdout_metadata)
            if status == "timeout":
                timeout_msg = (
                    f"Script timed out after {timeout}s and was killed."
                )
                result["error"] = timeout_msg
                result["output"] = (
                    stdout_text + f"\n\n⏰ {timeout_msg}"
                    if stdout_text
                    else f"⏰ {timeout_msg}"
                )
            elif status == "interrupted":
                result["output"] = (
                    stdout_text
                    + "\n[execution interrupted — user sent a new message]"
                )
            elif exit_code != 0:
                result["status"] = "error"
                result["error"] = (
                    stderr_text
                    or f"Script exited with code {exit_code}"
                )
                if stderr_text:
                    result["output"] = (
                        stdout_text
                        + "\n--- stderr ---\n"
                        + stderr_text
                    )
                hint = _sandbox_failure_hint(
                    stderr_text,
                    enabled_tools=sandbox_tools,
                )
                if hint:
                    result["hint"] = hint
            return json.dumps(result, ensure_ascii=False)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        duration = round(time.monotonic() - exec_start, 2)
        logger.error(
            "execute_code failed after %ss with %d tool calls: %s: %s",
            duration,
            tool_call_counter[0],
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return json.dumps({
            "status": "error",
            "error": str(exc),
            "tool_calls_made": tool_call_counter[0],
            "duration_seconds": duration,
        }, ensure_ascii=False)
    finally:
        cleanup_task = asyncio.create_task(
            _cleanup_local_execution(
                process,
                (stdout_task, stderr_task, wait_task),
                rpc_task,
                stop_event,
                server_sock,
                sock_path,
            ),
            name="execute-code-cleanup",
        )
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            await asyncio.shield(cleanup_task)
            raise


async def _kill_process_group(
    proc: asyncio.subprocess.Process,
    escalate: bool = False,
):
    """Terminate the child process group and reap the owned child."""
    if proc.returncode is not None:
        return
    try:
        if _IS_WINDOWS:
            proc.terminate()
        else:
            os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except (PermissionError, OSError):
        proc.terminate()

    grace = 5.0 if escalate else 1.0
    try:
        await asyncio.wait_for(proc.wait(), timeout=grace)
        return
    except TimeoutError:
        pass

    try:
        if _IS_WINDOWS:
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except (PermissionError, OSError):
        proc.kill()
    await proc.wait()


async def _load_config() -> dict:
    """Load code_execution config through native async config I/O."""
    try:
        from hermes_cli.config import read_user_config_raw

        raw = await read_user_config_raw()
        cfg = raw.get("code_execution", {}) if isinstance(raw, dict) else {}
        return cfg if isinstance(cfg, dict) else {}
    except asyncio.CancelledError:
        raise
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Execution mode resolution (strict vs project)
# ---------------------------------------------------------------------------

# Valid values for code_execution.mode. Kept as a module constant so tests
# and the config layer can reference the canonical set.
EXECUTION_MODES = ("project", "strict")
DEFAULT_EXECUTION_MODE = "project"


async def _get_execution_mode() -> str:
    """Return the active execute_code mode — 'project' or 'strict'.

    Reads ``code_execution.mode`` from config.yaml; invalid values fall back
    to ``DEFAULT_EXECUTION_MODE`` ('project') with a log warning.

    Mode semantics:
      - ``project`` (default): scripts run in the session's working directory
        with the active virtual environment's python, so project dependencies
        (pandas, torch, project packages) and files resolve naturally.
      - ``strict``: scripts run in an isolated temp directory with
        ``sys.executable`` (hermes-agent's python). Reproducible and the
        interpreter is guaranteed to work, but project deps and relative paths
        won't resolve.

    Env scrubbing and tool whitelist apply identically in both modes.
    """
    cfg_value = str((await _load_config()).get("mode", DEFAULT_EXECUTION_MODE)).strip().lower()
    if cfg_value in EXECUTION_MODES:
        return cfg_value
    logger.warning(
        "Ignoring code_execution.mode=%r (expected one of %s), falling back to %r",
        cfg_value, EXECUTION_MODES, DEFAULT_EXECUTION_MODE,
    )
    return DEFAULT_EXECUTION_MODE


_USABLE_PYTHON_CACHE: dict[str, bool] = {}


async def _is_usable_python(python_path: str) -> bool:
    """Check whether an interpreter is Python 3.8+ without blocking."""
    cached = _USABLE_PYTHON_CACHE.get(python_path)
    if cached is not None:
        return cached
    try:
        from agent.delegation_context import delegated_child_subprocess_env

        proc = await asyncio.create_subprocess_exec(
            python_path,
            "-c",
            "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            stdin=asyncio.subprocess.DEVNULL,
            env=delegated_child_subprocess_env(),
            creationflags=(
                subprocess.CREATE_NO_WINDOW if _IS_WINDOWS else 0
            ),
        )
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            result = False
        else:
            result = proc.returncode == 0
    except asyncio.CancelledError:
        raise
    except (OSError, subprocess.SubprocessError):
        result = False
    _USABLE_PYTHON_CACHE[python_path] = result
    return result


async def _resolve_child_python(mode: str) -> str:
    """Pick the Python interpreter for the execute_code subprocess.

    In ``strict`` mode, always ``sys.executable`` — guaranteed to work and
    keeps behavior fully reproducible across sessions.

    In ``project`` mode, prefer the user's active virtualenv/conda env's
    python so ``import pandas`` etc. work. Falls back to ``sys.executable``
    if no venv is detected, the candidate binary is missing/not executable,
    or it fails a Python 3.8+ version check.
    """
    if mode != "project":
        return sys.executable

    if _IS_WINDOWS:
        exe_names = ("python.exe", "python3.exe")
        subdirs = ("Scripts",)
    else:
        exe_names = ("python", "python3")
        subdirs = ("bin",)

    for var in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        root = os.environ.get(var, "").strip()
        if not root:
            continue
        for subdir in subdirs:
            for exe in exe_names:
                candidate = os.path.join(root, subdir, exe)
                if not (await aiofiles.os.path.isfile(candidate) and await aiofiles.os.access(candidate, os.X_OK)):
                    continue
                if await _is_usable_python(candidate):
                    return candidate
                # Found the interpreter but it failed the version check —
                # log once and fall through to sys.executable.
                logger.info(
                    "execute_code: skipping %s=%s (Python version < 3.8 or broken). "
                    "Using sys.executable instead.", var, candidate,
                )
                return sys.executable

    return sys.executable


async def _resolve_child_cwd(mode: str, staging_dir: str, task_id: str = "") -> str:
    """Resolve the working directory for the execute_code subprocess.

    - ``strict``: the staging tmpdir (today's behavior).
    - ``project``: the session's own cwd — its per-session cwd record
      (written after every completed terminal command), then the raw
      per-session cwd override registered via ``session.cwd.set`` /
      ``register_task_env_overrides``, then the session's TERMINAL_CWD
      (same as the terminal tool), or ``os.getcwd()`` if none points at a
      real dir. Falls back to the staging tmpdir as a last resort so we
      never invoke Popen with a nonexistent cwd.

    This mirrors the resolution ladder file tools and the terminal use
    (record → registered override → TERMINAL_CWD), so all file-writing
    paths within a session agree on the working directory. (#56047)
    """
    if mode != "project":
        return staging_dir
    if task_id:
        # 1. The session's cwd record — IS the session's `cd` state.
        try:
            from tools.terminal_tool import get_session_cwd

            recorded = get_session_cwd(task_id)
        except Exception:
            recorded = None
        if recorded and await aiofiles.os.path.isdir(recorded):
            return recorded
        # 2. Registered workspace override (session.cwd.set → gateway/TUI/ACP).
        try:
            from tools.file_tools import _registered_task_cwd_override

            session_cwd = await _registered_task_cwd_override(task_id)
        except Exception:
            session_cwd = None
        if session_cwd and await aiofiles.os.path.isdir(session_cwd):
            return session_cwd
    from agent.runtime_cwd import resolve_agent_cwd

    here = await resolve_agent_cwd()
    if await aiofiles.os.path.isdir(here):
        return str(here)
    return staging_dir


# ---------------------------------------------------------------------------
# OpenAI Function-Calling Schema
# ---------------------------------------------------------------------------

# Per-tool documentation lines for the execute_code description.
# Ordered to match the canonical display order.
_TOOL_DOC_LINES = [
    ("web_search",
     "  web_search(query: str, limit: int = 5) -> dict\n"
     "    Returns {\"data\": {\"web\": [{\"url\", \"title\", \"description\"}, ...]}}"),
    ("web_extract",
     "  web_extract(urls: list[str], char_limit: int = None) -> dict\n"
     "    Returns {\"results\": [{\"url\", \"title\", \"content\", \"error\"}, ...]} where content is markdown.\n"
     "    No LLM summarization. Pages over char_limit (default 15000) are head+tail truncated; full text stored on disk (path in the content footer)."),
    ("read_file",
     "  read_file(path: str, offset: int = 1, limit: int = 2000) -> dict\n"
     "    Lines are 1-indexed. Returns {\"content\": \"...\", \"total_lines\": N}"),
    ("write_file",
     "  write_file(path: str, content: str) -> dict\n"
     "    Always overwrites the entire file."),
    ("search_files",
     "  search_files(pattern: str, target=\"content\", path=\".\", file_glob=None, limit=50) -> dict\n"
     "    target: \"content\" (search inside files) or \"files\" (find files by name). Returns {\"matches\": [...]}"),
    ("patch",
     "  patch(path: str, old_string: str, new_string: str, replace_all: bool = False) -> dict\n"
     "    Replaces old_string with new_string in the file."),
    ("terminal",
     "  terminal(command: str, timeout=None, workdir=None) -> dict\n"
     "    Foreground only (no background/pty). Returns {\"output\": \"...\", \"exit_code\": N}"),
]


def _build_execute_code_schema(enabled_sandbox_tools: set, mode: str) -> dict:
    """Build the execute_code schema with description listing only enabled tools.

    When tools are disabled via ``hermes tools`` (e.g. web is turned off),
    the schema description should NOT mention web_search / web_extract —
    otherwise the model thinks they are available and keeps trying to use them.

    ``mode`` controls the working-directory sentence in the description:
      - ``'strict'``: scripts run in a temp dir (not the session's CWD)
      - ``'project'`` (default): scripts run in the session's CWD with the
        active venv's python
    """
    # Build tool documentation lines for only the enabled tools
    tool_lines = "\n".join(
        doc for name, doc in _TOOL_DOC_LINES if name in enabled_sandbox_tools
    )

    # Build example import list from enabled tools
    import_examples = [n for n in ("web_search", "terminal") if n in enabled_sandbox_tools]
    if not import_examples:
        import_examples = sorted(enabled_sandbox_tools)[:2]
    if import_examples:
        import_str = ", ".join(import_examples) + ", ..."
    else:
        import_str = "..."

    # Mode-specific CWD guidance. Project mode is the default and matches
    # terminal()'s filesystem/interpreter; strict mode retains the isolated
    # temp-dir staging and hermes-agent's own python.
    if mode == "strict":
        cwd_note = (
            "Scripts run in their own temp dir, not the session's CWD — use absolute paths "
            "(os.path.expanduser('~/.hermes/.env')) or terminal()/read_file() for user files."
        )
    else:
        cwd_note = (
            "Scripts run in the session's working directory with the active venv's python, "
            "so project deps (pandas, etc.) and relative paths work like in terminal()."
        )

    description = (
        "Run a Python script that calls Hermes tools programmatically. "
        "Use when you need 3+ tool calls with logic between them: "
        "filtering/reducing large outputs before they enter context, "
        "conditional branching, or loops (N pages/files, retry on failure). "
        "Use normal tool calls for single calls, results you must reason "
        "over in full, or anything needing user interaction.\n\n"
        f"Available via `from hermes_tools import ...`:\n\n"
        f"{tool_lines}\n\n"
        "Limits: 5-minute timeout, 50KB stdout cap, max 50 tool calls per script. "
        "terminal() is foreground-only (no background or pty).\n\n"
        f"{cwd_note}\n\n"
        "Print your final result to stdout; stdlib (json, re, csv, datetime, ...) "
        "is available for processing.\n\n"
        "Built-in helpers (no import): json_parse(text) — tolerant json.loads for "
        "terminal() output; shell_quote(s) — shlex.quote for dynamic shell args; "
        "retry(fn, max_attempts=3, delay=2) — exponential backoff for transient failures."
    )

    return {
        "name": "execute_code",
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python code to execute. Import tools with "
                        f"`from hermes_tools import {import_str}` "
                        "and print your final result to stdout."
                    ),
                },
            },
            "required": ["code"],
        },
    }


async def build_execute_code_schema(enabled_sandbox_tools: set = None,
                                    mode: str = None) -> dict:
    """Build the upstream schema after awaiting configured mode resolution."""
    if enabled_sandbox_tools is None:
        enabled_sandbox_tools = SANDBOX_ALLOWED_TOOLS
    if mode is None:
        mode = await _get_execution_mode()
    return _build_execute_code_schema(enabled_sandbox_tools, mode)


# Default schema used at registration time (all sandbox tools listed,
# default mode). Per-session callers rebuild after awaiting config.
EXECUTE_CODE_SCHEMA = _build_execute_code_schema(
    SANDBOX_ALLOWED_TOOLS,
    DEFAULT_EXECUTION_MODE,
)


# --- Registry ---
from tools.registry import registry, tool_error


async def _handle_execute_code(args, **kw):
    return await execute_code(
        code=args.get("code", ""),
        task_id=kw.get("task_id"),
        enabled_tools=kw.get("enabled_tools"),
    )

registry.register(
    name="execute_code",
    toolset="code_execution",
    schema=EXECUTE_CODE_SCHEMA,
    handler=_handle_execute_code,
    check_fn=check_sandbox_requirements,
    emoji="🐍",
    max_result_size_chars=100_000,
)
