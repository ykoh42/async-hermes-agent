"""Codex app-server JSON-RPC client.

Speaks the protocol documented in codex-rs/app-server/README.md (codex 0.125+).
Transport is newline-delimited JSON-RPC 2.0 over stdio: spawn `codex app-server`,
do an `initialize` handshake, then drive `thread/start` + `turn/start` and
consume streaming `item/*` notifications until `turn/completed`.

This module is the wire-level speaker only. Higher-level concerns (event
projection into Hermes' display, approval bridging, transcript projection into
AIAgent.messages, plugin migration) live in sibling modules.

Status: optional opt-in runtime gated behind `model.openai_runtime ==
"codex_app_server"`. Hermes' default tool dispatch is unchanged when this
runtime is not selected.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from tools.environments.local import hermes_subprocess_env

# Default minimum codex version we test against. The PR sets this from the
# `codex --version` parsed at install time; bumping is a one-line change here.
MIN_CODEX_VERSION = (0, 125, 0)


@dataclass
class CodexAppServerError(RuntimeError):
    """Raised on JSON-RPC errors from the app-server."""

    code: int
    message: str
    data: Any | None = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"codex app-server error {self.code}: {self.message}"


@dataclass
class _Pending:
    future: asyncio.Future[dict]
    method: str
    sent_at: float = field(default_factory=time.time)


async def _finish_owned_task(task: asyncio.Task[Any]) -> Any:
    """Finish one owned cleanup task before propagating cancellation."""
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised below
            if task.cancelled():
                raise
            if cancellation is None:
                cancellation = exc
        except Exception as exc:
            if cancellation is not None:
                raise cancellation from exc
            raise
    if cancellation is not None:
        raise cancellation
    return result


async def _finish_process_communicate(
    process: asyncio.subprocess.Process,
    communicate_task: asyncio.Task[tuple[bytes | None, bytes | None]],
) -> tuple[bytes | None, bytes | None]:
    """Drain and reap one owned Codex helper process."""
    async def drain_or_wait() -> tuple[bytes | None, bytes | None]:
        try:
            return await communicate_task
        except BaseException:
            await process.wait()
            raise

    return await _finish_owned_task(asyncio.create_task(drain_or_wait()))


class CodexAppServerClient:
    """Minimal JSON-RPC 2.0 client for `codex app-server` over stdio.

    The subprocess, stdio readers, request futures, and notification queues all
    live on the caller's event loop. No worker thread participates in the wire
    protocol.
    """

    def __init__(
        self,
        codex_bin: str = "codex",
        codex_home: str | None = None,
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._codex_bin = codex_bin
        # codex app-server is a model-driving CLI executor: it runs a
        # model-chosen agentic loop that executes shell commands, so it
        # legitimately needs LLM provider credentials (inherit_credentials=True)
        # to authenticate against the model endpoint. But the previous
        # `os.environ.copy()` also handed it every Tier-1 Hermes secret — gateway
        # bot tokens, GitHub auth, Modal/Daytona infra tokens, the dashboard
        # session token, AUXILIARY_* side-LLM keys, GATEWAY_RELAY_* auth — none
        # of which a coding subprocess has any use for. Route through the
        # centralized helper so Tier-1 + dynamic-internal secrets are always
        # stripped while provider creds still flow, matching copilot_acp_client
        # (#29157 sibling spawn-site gap).
        self._env_overrides = dict(env or {})
        self._codex_home = codex_home

        app_server_args = list(extra_args or [])
        cmd = [codex_bin, "app-server"] + app_server_args
        # Codex emits tracing to stderr; default WARN keeps it quiet for users.
        # Hide the console the codex child would otherwise flash on Windows
        # (#56747). Hide-only — stdio pipes stay intact for the app-server wire.
        from hermes_cli._subprocess_compat import windows_hide_flags

        self._cmd = cmd
        self._spawn_env: dict[str, str] | None = None
        self._creationflags = windows_hide_flags()
        self._proc: asyncio.subprocess.Process | None = None
        self._next_id = 1
        self._pending: dict[int, _Pending] = {}
        self._notifications: asyncio.Queue[dict] = asyncio.Queue()
        self._server_requests: asyncio.Queue[dict] = asyncio.Queue()
        self._stderr_lines: list[str] = []
        self._write_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._closed = False
        self._initialized = False
        self._reader: asyncio.Task[None] | None = None
        self._stderr_reader: asyncio.Task[None] | None = None

    # ---------- lifecycle ----------

    async def initialize(
        self,
        client_name: str = "hermes",
        client_title: str = "Hermes Agent",
        client_version: str = "0.1",
        capabilities: dict | None = None,
        timeout: float = 10.0,
    ) -> dict:
        """Send `initialize` + `initialized` handshake. Returns the server's
        InitializeResponse (userAgent, codexHome, platformFamily, platformOs)."""
        if self._initialized:
            raise RuntimeError("already initialized")
        params = {
            "clientInfo": {
                "name": client_name,
                "title": client_title,
                "version": client_version,
            },
            "capabilities": capabilities or {},
        }
        result = await self.request("initialize", params, timeout=timeout)
        await self.notify("initialized")
        self._initialized = True
        return result

    async def close(self, timeout: float = 3.0) -> None:
        """Close stdin and wait for the subprocess to exit, escalating to kill."""
        if self._closed:
            return
        self._closed = True
        cleanup_task = asyncio.create_task(self._close_owned_resources(timeout))
        await _finish_owned_task(cleanup_task)

    async def _close_owned_resources(self, timeout: float) -> None:
        error = RuntimeError("codex app-server client is closed")
        for pending in self._pending.values():
            if not pending.future.done():
                pending.future.set_exception(error)
        self._pending.clear()

        proc = self._proc
        if proc is not None:
            if proc.stdin is not None:
                proc.stdin.close()
                try:
                    await proc.stdin.wait_closed()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            if proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=timeout)
                except TimeoutError:
                    proc.kill()
                    await proc.wait()

        tasks = [task for task in (self._reader, self._stderr_reader) if task]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._reader = None
        self._stderr_reader = None
        self._proc = None

    async def __aenter__(self) -> CodexAppServerClient:
        await self._start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ---------- send/receive ----------

    async def request(
        self,
        method: str,
        params: dict | None = None,
        timeout: float = 30.0,
    ) -> dict:
        """Send a JSON-RPC request and await the response. Returns `result`,
        raises CodexAppServerError on `error`."""
        await self._start()
        rid = self._take_id()
        future = asyncio.get_running_loop().create_future()
        self._pending[rid] = _Pending(future=future, method=method)
        try:
            await self._send({"id": rid, "method": method, "params": params or {}})
            msg = await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            self._pending.pop(rid, None)
            raise TimeoutError(
                f"codex app-server method {method!r} timed out after {timeout}s"
            ) from None
        except BaseException:
            self._pending.pop(rid, None)
            raise
        if "error" in msg:
            err = msg["error"]
            raise CodexAppServerError(
                code=err.get("code", -1),
                message=err.get("message", ""),
                data=err.get("data"),
            )
        return msg.get("result", {})

    async def notify(self, method: str, params: dict | None = None) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        await self._start()
        await self._send({"method": method, "params": params or {}})

    async def respond(self, request_id: Any, result: dict) -> None:
        """Reply to a server-initiated request (e.g. approval prompts)."""
        await self._send({"id": request_id, "result": result})

    async def respond_error(
        self, request_id: Any, code: int, message: str, data: Any | None = None
    ) -> None:
        """Reply to a server-initiated request with an error."""
        err: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        await self._send({"id": request_id, "error": err})

    async def take_notification(self, timeout: float = 0.0) -> dict | None:
        """Pop the next streaming notification, or return None on timeout.

        timeout=0.0 means non-blocking. Use small positive timeouts inside the
        AIAgent turn loop to interleave reads with interrupt checks."""
        if timeout <= 0:
            try:
                return self._notifications.get_nowait()
            except asyncio.QueueEmpty:
                return None
        try:
            return await asyncio.wait_for(self._notifications.get(), timeout)
        except TimeoutError:
            return None

    async def take_server_request(self, timeout: float = 0.0) -> dict | None:
        """Pop the next server-initiated request (e.g. exec/applyPatch approval)."""
        if timeout <= 0:
            try:
                return self._server_requests.get_nowait()
            except asyncio.QueueEmpty:
                return None
        try:
            return await asyncio.wait_for(self._server_requests.get(), timeout)
        except TimeoutError:
            return None

    # ---------- diagnostics ----------

    def stderr_tail(self, n: int = 20) -> list[str]:
        """Return last n lines of codex's stderr (for error reports)."""
        return list(self._stderr_lines[-n:])

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    # ---------- internals ----------

    def _take_id(self) -> int:
        # JSON-RPC ids only need to be unique per-connection. A simple
        # monotonically increasing int is the common choice and matches what
        # codex's own clients use.
        rid = self._next_id
        self._next_id += 1
        return rid

    async def _build_spawn_env(self) -> dict[str, str]:
        spawn_env = await hermes_subprocess_env(inherit_credentials=True)
        spawn_env.update(self._env_overrides)
        from agent.secret_scope import get_secret, is_multiplex_active

        codex_home = self._codex_home
        if is_multiplex_active():
            spawn_env.pop("CODEX_HOME", None)
            codex_home = codex_home or str(get_secret("CODEX_HOME", "") or "").strip()
            if not codex_home:
                raise RuntimeError(
                    "Codex app-server requires a non-empty profile-scoped "
                    "CODEX_HOME while profile multiplexing is active."
                )
        if codex_home:
            spawn_env["CODEX_HOME"] = codex_home
        spawn_env.setdefault("RUST_LOG", "warn")
        self._spawn_env = spawn_env
        return spawn_env

    async def _start(self) -> None:
        if self._proc is not None:
            return
        if self._closed:
            raise RuntimeError("codex app-server client is closed")
        async with self._start_lock:
            if self._proc is not None:
                return
            spawn_env = await self._build_spawn_env()
            self._proc = await asyncio.create_subprocess_exec(
                *self._cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=spawn_env,
                creationflags=self._creationflags,
            )
            self._reader = asyncio.create_task(self._read_stdout())
            self._stderr_reader = asyncio.create_task(self._read_stderr())

    async def _send(self, obj: dict) -> None:
        if self._closed:
            raise RuntimeError("codex app-server client is closed")
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("codex app-server stdin not available")
        try:
            async with self._write_lock:
                self._proc.stdin.write((json.dumps(obj) + "\n").encode("utf-8"))
                await self._proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, RuntimeError) as exc:
            raise RuntimeError(
                f"codex app-server stdin closed unexpectedly: {exc}"
            ) from exc

    async def _read_stdout(self) -> None:
        if self._proc is None or self._proc.stdout is None:
            return
        try:
            while line := await self._proc.stdout.readline():
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    # Non-JSON output is unexpected on stdout; tracing belongs
                    # on stderr. Surface it via stderr buffer for diagnostics.
                    self._stderr_lines.append(
                        f"<non-json on stdout> {line[:200]!r}"
                    )
                    continue
                self._dispatch(msg)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._stderr_lines.append(f"<stdout reader error> {exc}")

    def _dispatch(self, msg: dict) -> None:
        # Reply (has id + result/error, no method)
        if "id" in msg and ("result" in msg or "error" in msg):
            pending = self._pending.pop(msg["id"], None)
            if pending is not None and not pending.future.done():
                pending.future.set_result(msg)
            return
        # Server-initiated request (has id + method)
        if "id" in msg and "method" in msg:
            self._server_requests.put_nowait(msg)
            return
        # Notification (no id)
        if "method" in msg:
            self._notifications.put_nowait(msg)

    async def _read_stderr(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            return
        try:
            while line := await self._proc.stderr.readline():
                if not line:
                    break
                self._stderr_lines.append(
                    line.decode("utf-8", "replace").rstrip()
                )
                # Bound memory: keep last 500 lines.
                if len(self._stderr_lines) > 500:
                    self._stderr_lines = self._stderr_lines[-500:]
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover
            pass


def parse_codex_version(output: str) -> tuple[int, int, int] | None:
    """Parse `codex --version` output. Returns (major, minor, patch) or None."""
    # Output format: "codex-cli 0.130.0" possibly followed by metadata.
    import re

    match = re.search(r"(\d+)\.(\d+)\.(\d+)", output or "")
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


async def check_codex_binary(
    codex_bin: str = "codex", min_version: tuple[int, int, int] = MIN_CODEX_VERSION
) -> tuple[bool, str]:
    """Verify codex CLI is installed and meets minimum version.

    Returns (ok, message). Used by setup wizard and runtime startup."""
    proc: asyncio.subprocess.Process | None = None
    communicate_task: asyncio.Task[tuple[bytes, bytes]] | None = None
    communication_error: Exception | None = None
    try:
        spawn_env = await hermes_subprocess_env(inherit_credentials=False)
        proc = await asyncio.create_subprocess_exec(
            codex_bin,
            "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
            env=spawn_env,
        )
        communicate_task = asyncio.create_task(proc.communicate())
        stdout, stderr = await asyncio.wait_for(
            asyncio.shield(communicate_task), timeout=10
        )
    except FileNotFoundError:
        return False, (
            f"codex CLI not found at {codex_bin!r}. Install with: "
            f"npm i -g @openai/codex"
        )
    except asyncio.CancelledError:
        if proc is not None and proc.returncode is None:
            proc.kill()
        if communicate_task is not None:
            await _finish_process_communicate(proc, communicate_task)
        raise
    except TimeoutError:
        if proc is not None and proc.returncode is None:
            proc.kill()
        if communicate_task is not None:
            await _finish_process_communicate(proc, communicate_task)
        return False, "codex --version timed out"
    except Exception as exc:
        communication_error = exc
    if communication_error is not None:
        if proc is not None and proc.returncode is None:
            proc.kill()
        if proc is not None and communicate_task is not None:
            try:
                await _finish_process_communicate(proc, communicate_task)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
        raise communication_error
    assert proc is not None
    if proc.returncode != 0:
        stderr_text = stderr.decode("utf-8", "replace").strip()
        return False, f"codex --version exited {proc.returncode}: {stderr_text}"
    stdout_text = stdout.decode("utf-8", "replace")
    version = parse_codex_version(stdout_text)
    if version is None:
        return False, f"could not parse codex version from: {stdout_text!r}"
    if version < min_version:
        return False, (
            f"codex {'.'.join(map(str, version))} is older than required "
            f"{'.'.join(map(str, min_version))}. Run: npm i -g @openai/codex"
        )
    return True, ".".join(map(str, version))
