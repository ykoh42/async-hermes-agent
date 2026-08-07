"""Native-async background process management for the terminal tool.

The public tool name, schema, result fields, and source location mirror Hermes
Agent.  Local subprocess I/O is owned by the caller's event loop; no reader
threads, blocking ``Popen`` handles, or thread bridges are used.
"""

from __future__ import annotations

import asyncio
import codecs
import errno
import json
import os
import signal
import struct
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from tools.environments.local import build_subprocess_env
from tools.registry import registry, tool_error

MAX_OUTPUT_CHARS = 200_000
FINISHED_TTL_SECONDS = 1800
MAX_PROCESSES = 64
MAX_ACTIVE_PROCESS_AGE = 86400


@dataclass
class ProcessSession:
    """One background process and the event-loop resources that own it."""

    id: str
    command: str
    task_id: str = ""
    session_key: str = ""
    cwd: str = ""
    started_at: float = field(default_factory=time.time)
    exited: bool = False
    exit_code: int | None = None
    output_buffer: str = ""
    pid: int | None = None
    completion_reason: str = ""
    termination_source: str = ""
    max_output_chars: int = MAX_OUTPUT_CHARS
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    _monitor_task: asyncio.Task[None] | None = field(default=None, repr=False)
    _pty_master_fd: int | None = field(default=None, repr=False)
    _completion_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)


class ProcessRegistry:
    """Track local background processes without leaving the event loop."""

    def __init__(self) -> None:
        self._running: dict[str, ProcessSession] = {}
        self._finished: dict[str, ProcessSession] = {}
        self._completion_consumed: set[str] = set()

    async def spawn_local(
        self,
        command: str,
        cwd: str | None = None,
        task_id: str = "",
        session_key: str = "",
        env_vars: dict[str, str] | None = None,
        use_pty: bool = False,
    ) -> ProcessSession:
        """Spawn a local command and start native async output collection."""
        from tools.terminal_tool import _rewrite_compound_background

        safe_command = _rewrite_compound_background(command)
        workdir = os.path.abspath(os.path.expanduser(cwd or os.getcwd()))
        shell = os.environ.get("SHELL") or "/bin/sh"
        env = build_subprocess_env(scrub_secrets=True, extra=env_vars)
        env["PYTHONUNBUFFERED"] = "1"
        session = ProcessSession(
            id=f"proc_{uuid.uuid4().hex[:12]}",
            command=command,
            task_id=str(task_id or ""),
            session_key=str(session_key or ""),
            cwd=workdir,
        )

        if use_pty and os.name == "posix":
            process = await self._spawn_posix_pty(
                session,
                [shell, "-lic", f"set +m; {safe_command}"],
                env,
            )
        else:
            process = await asyncio.create_subprocess_exec(
                shell,
                "-lc",
                f"set +m; {safe_command}",
                cwd=workdir,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )

        session.process = process
        session.pid = process.pid
        self._prune_finished()
        self._running[session.id] = session
        session._monitor_task = asyncio.create_task(
            self._monitor(session),
            name=f"process-monitor-{session.id}",
        )
        return session

    async def _spawn_posix_pty(
        self,
        session: ProcessSession,
        argv: list[str],
        env: dict[str, str],
    ) -> asyncio.subprocess.Process:
        import fcntl
        import pty
        import termios

        master_fd, slave_fd = pty.openpty()
        try:
            fcntl.ioctl(
                slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 120, 0, 0)
            )
            os.set_blocking(master_fd, False)
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=session.cwd,
                env=env,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                start_new_session=True,
            )
        except BaseException:
            os.close(master_fd)
            raise
        finally:
            os.close(slave_fd)
        session._pty_master_fd = master_fd
        return process

    async def _monitor(self, session: ProcessSession) -> None:
        process = session.process
        if process is None:
            return
        reader = asyncio.create_task(
            self._read_output(session),
            name=f"process-output-{session.id}",
        )
        try:
            return_code = await process.wait()
            _, pending = await asyncio.wait({reader}, timeout=0.25)
            if pending:
                reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            if session.completion_reason == "killed":
                session.exit_code = -signal.SIGTERM
            else:
                session.exit_code = return_code
                session.completion_reason = "exited"
            session.exited = True
            self._move_to_finished(session)
        finally:
            if not reader.done():
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
            self._close_pty(session)

    async def _read_output(self, session: ProcessSession) -> None:
        if session._pty_master_fd is not None:
            await self._read_pty(session)
            return
        process = session.process
        if process is None or process.stdout is None:
            return
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        while chunk := await process.stdout.read(64 * 1024):
            self._append_output(session, decoder.decode(chunk))
        self._append_output(session, decoder.decode(b"", final=True))

    async def _read_pty(self, session: ProcessSession) -> None:
        fd = session._pty_master_fd
        if fd is None:
            return
        loop = asyncio.get_running_loop()
        completed: asyncio.Future[None] = loop.create_future()
        decoder = codecs.getincrementaldecoder("utf-8")("replace")

        def readable() -> None:
            try:
                chunk = os.read(fd, 64 * 1024)
            except BlockingIOError:
                return
            except OSError as exc:
                if exc.errno != errno.EIO and not completed.done():
                    completed.set_exception(exc)
                elif not completed.done():
                    completed.set_result(None)
                return
            if chunk:
                self._append_output(session, decoder.decode(chunk))
            elif not completed.done():
                completed.set_result(None)

        loop.add_reader(fd, readable)
        try:
            await completed
        finally:
            loop.remove_reader(fd)
            self._append_output(session, decoder.decode(b"", final=True))

    @staticmethod
    def _append_output(session: ProcessSession, text: str) -> None:
        if not text:
            return
        session.output_buffer += text
        if len(session.output_buffer) > session.max_output_chars:
            session.output_buffer = session.output_buffer[-session.max_output_chars :]

    def _move_to_finished(self, session: ProcessSession) -> None:
        self._running.pop(session.id, None)
        self._finished[session.id] = session
        session._completion_event.set()

    @staticmethod
    def _close_pty(session: ProcessSession) -> None:
        fd = session._pty_master_fd
        session._pty_master_fd = None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    def _prune_finished(self) -> None:
        cutoff = time.time() - FINISHED_TTL_SECONDS
        self._finished = {
            session_id: session
            for session_id, session in self._finished.items()
            if session.started_at >= cutoff
        }
        overflow = len(self._running) + len(self._finished) - MAX_PROCESSES + 1
        if overflow > 0:
            oldest = sorted(self._finished.values(), key=lambda item: item.started_at)
            for session in oldest[:overflow]:
                self._finished.pop(session.id, None)

    def get(self, session_id: str) -> ProcessSession | None:
        return self._running.get(session_id) or self._finished.get(session_id)

    def poll(self, session_id: str) -> dict[str, Any]:
        """Return status plus the latest output without consuming completion."""
        from tools.ansi_strip import strip_ansi

        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}
        result: dict[str, Any] = {
            "session_id": session.id,
            "command": session.command,
            "status": "exited" if session.exited else "running",
            "pid": session.pid,
            "uptime_seconds": int(time.time() - session.started_at),
            "output_preview": strip_ansi(session.output_buffer[-1000:]),
        }
        if session.exited:
            result.update(
                exit_code=session.exit_code,
                completion_reason=session.completion_reason,
                termination_source=session.termination_source,
            )
        return result

    def is_completion_consumed(self, session_id: str) -> bool:
        return session_id in self._completion_consumed

    def read_log(
        self, session_id: str, offset: int = 0, limit: int = 200
    ) -> dict[str, Any]:
        """Return the process output using Hermes' line pagination contract."""
        from tools.ansi_strip import strip_ansi

        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}
        lines = strip_ansi(session.output_buffer).splitlines()
        if offset == 0 and limit > 0:
            selected = lines[-limit:]
            consumed = bool(selected) or not lines
        else:
            selected = lines[offset : offset + limit]
            stop = slice(offset, offset + limit).indices(len(lines))[1]
            consumed = not lines or (bool(selected) and stop == len(lines))
        if session.exited and consumed:
            self._completion_consumed.add(session_id)
        return {
            "session_id": session.id,
            "command": session.command,
            "status": "exited" if session.exited else "running",
            "output": "\n".join(selected),
            "total_lines": len(lines),
            "showing": f"{len(selected)} lines",
        }

    async def wait(self, session_id: str, timeout: int | None = None) -> dict[str, Any]:
        """Wait without blocking the event loop or stopping the process on timeout."""
        from tools.ansi_strip import strip_ansi
        from tools.interrupt import is_interrupted

        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}
        try:
            configured_timeout = int(os.getenv("TERMINAL_TIMEOUT", "180"))
        except (TypeError, ValueError):
            configured_timeout = 180
        requested_timeout = timeout
        effective_timeout = (
            min(timeout, configured_timeout) if timeout else configured_timeout
        )
        timeout_note = None
        if requested_timeout and requested_timeout > configured_timeout:
            timeout_note = (
                f"Requested wait of {requested_timeout}s was clamped "
                f"to configured limit of {configured_timeout}s"
            )

        deadline = asyncio.get_running_loop().time() + effective_timeout
        while not session.exited:
            if is_interrupted():
                result = {
                    "status": "interrupted",
                    "command": session.command,
                    "output": strip_ansi(session.output_buffer[-1000:]),
                    "note": "User sent a new message -- wait interrupted",
                }
                if timeout_note:
                    result["timeout_note"] = timeout_note
                return result
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                await asyncio.wait_for(
                    session._completion_event.wait(),
                    timeout=min(1.0, remaining),
                )
            except TimeoutError:
                continue

        if session.exited:
            monitor = session._monitor_task
            if monitor is not None:
                await asyncio.shield(monitor)
            self._completion_consumed.add(session_id)
            result = {
                "status": "exited",
                "command": session.command,
                "exit_code": session.exit_code,
                "completion_reason": session.completion_reason,
                "termination_source": session.termination_source,
                "output": strip_ansi(session.output_buffer[-2000:]),
            }
            if timeout_note:
                result["timeout_note"] = timeout_note
            return result

        note = (
            f"Wait window of {effective_timeout}s elapsed — the process is still running. "
            f"This is not an error. Uptime: {int(time.time() - session.started_at)}s. "
            "Poll again later."
        )
        result = {
            "status": "timeout",
            "command": session.command,
            "output": strip_ansi(session.output_buffer[-1000:]),
            "process_running": True,
            "timeout_note": f"{timeout_note}. {note}" if timeout_note else note,
        }
        return result

    async def kill_process(
        self,
        session_id: str,
        *,
        source: str = "process.kill",
        consume_output: bool = True,
    ) -> dict[str, Any]:
        """Terminate a process group, await its monitor, and return captured output."""
        from tools.ansi_strip import strip_ansi
        from tools.terminal_tool import _terminate_process

        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}
        if session.exited:
            monitor = session._monitor_task
            if monitor is not None:
                await asyncio.shield(monitor)
            if consume_output:
                self._completion_consumed.add(session_id)
            return {
                "status": "already_exited",
                "command": session.command,
                "exit_code": session.exit_code,
                "completion_reason": session.completion_reason,
                "termination_source": session.termination_source,
                "output": strip_ansi(session.output_buffer[-2000:]),
            }

        session.completion_reason = "killed"
        session.termination_source = source
        process = session.process
        try:
            if process is not None:
                await _terminate_process(process)
            monitor = session._monitor_task
            if monitor is not None:
                await asyncio.shield(monitor)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            session.completion_reason = ""
            session.termination_source = ""
            return {"status": "error", "error": str(exc)}
        if consume_output:
            self._completion_consumed.add(session_id)
        return {
            "status": "killed",
            "session_id": session.id,
            "completion_reason": session.completion_reason,
            "termination_source": session.termination_source,
            "output": strip_ansi(session.output_buffer[-2000:]),
        }

    async def write_stdin(self, session_id: str, data: str) -> dict[str, Any]:
        """Send raw input to a running process without appending a newline."""
        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}
        if session.exited:
            return {"status": "already_exited", "error": "Process has already finished"}
        payload = str(data).encode()
        try:
            if session._pty_master_fd is not None:
                await self._write_pty(session._pty_master_fd, payload)
            elif session.process is not None and session.process.stdin is not None:
                session.process.stdin.write(payload)
                await session.process.stdin.drain()
            else:
                return {"status": "error", "error": "Process stdin not available"}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
        return {"status": "ok", "bytes_written": len(data)}

    @staticmethod
    async def _write_pty(fd: int, payload: bytes) -> None:
        loop = asyncio.get_running_loop()
        view = memoryview(payload)
        while view:
            try:
                written = os.write(fd, view)
                view = view[written:]
            except BlockingIOError:
                ready: asyncio.Future[None] = loop.create_future()

                def mark_ready() -> None:
                    if not ready.done():
                        ready.set_result(None)

                loop.add_writer(fd, mark_ready)
                try:
                    await ready
                finally:
                    loop.remove_writer(fd)

    async def submit_stdin(self, session_id: str, data: str = "") -> dict[str, Any]:
        return await self.write_stdin(session_id, f"{data}\n")

    async def close_stdin(self, session_id: str) -> dict[str, Any]:
        """Close pipe stdin or send a POSIX terminal EOF character."""
        session = self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}
        if session.exited:
            return {"status": "already_exited", "error": "Process has already finished"}
        if session._pty_master_fd is not None:
            result = await self.write_stdin(session_id, "\x04")
            if result.get("status") == "ok":
                return {"status": "ok", "message": "EOF sent"}
            return result
        process = session.process
        if process is None or process.stdin is None:
            return {"status": "error", "error": "Process stdin not available"}
        try:
            process.stdin.close()
            await process.stdin.wait_closed()
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
        return {"status": "ok", "message": "stdin closed"}

    def list_sessions(
        self,
        task_id: str | None = None,
        session_key: str | None = None,
    ) -> list[dict[str, Any]]:
        sessions = [*self._running.values(), *self._finished.values()]
        if task_id or session_key:
            sessions = [
                session
                for session in sessions
                if (task_id and session.task_id == task_id)
                or (session_key and session.session_key == session_key)
            ]
        result = []
        for session in sessions:
            entry: dict[str, Any] = {
                "session_id": session.id,
                "command": session.command[:200],
                "cwd": session.cwd,
                "pid": session.pid,
                "started_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%S", time.localtime(session.started_at)
                ),
                "uptime_seconds": int(time.time() - session.started_at),
                "status": "exited" if session.exited else "running",
                "output_preview": session.output_buffer[-200:],
            }
            if session.exited:
                entry["exit_code"] = session.exit_code
            result.append(entry)
        return result

    def count_running(self) -> int:
        return len(self._running)

    def has_active_processes(self, task_id: str) -> bool:
        return any(session.task_id == task_id for session in self._running.values())

    def has_active_for_session(
        self,
        session_key: str,
        max_active_age: float | None = None,
    ) -> bool:
        now = time.time()
        return any(
            session.session_key == session_key
            and (max_active_age is None or now - session.started_at < max_active_age)
            for session in self._running.values()
        )

    def has_any_active(self) -> bool:
        return bool(self._running)

    def snapshot_running_ids(self, task_id: str) -> frozenset[str]:
        return frozenset(
            session.id
            for session in self._running.values()
            if session.task_id == task_id
        )

    async def kill_started_since(
        self,
        task_id: str,
        baseline_ids,
        *,
        source: str,
    ) -> int:
        return await self.kill_all(
            task_id,
            exclude_ids=frozenset(baseline_ids or ()),
            source=source,
            consume_output=True,
        )

    async def kill_all(
        self,
        task_id: str | None = None,
        *,
        exclude_ids: frozenset[str] = frozenset(),
        source: str = "kill_all",
        consume_output: bool = False,
    ) -> int:
        targets = [
            session.id
            for session in self._running.values()
            if (task_id is None or session.task_id == task_id)
            and session.id not in exclude_ids
        ]
        results = await asyncio.gather(
            *(
                self.kill_process(
                    session_id,
                    source=source,
                    consume_output=consume_output,
                )
                for session_id in targets
            )
        )
        return sum(
            result.get("status") in {"killed", "already_exited"} for result in results
        )


process_registry = ProcessRegistry()


PROCESS_SCHEMA = {
    "name": "process",
    "description": (
        "Manage background processes started with terminal(background=true). "
        "Actions: 'list' (show all), 'poll' (check status + new output), "
        "'log' (full output with pagination), 'wait' (block until done or timeout), "
        "'kill' (terminate), 'write' (send raw stdin data without newline), "
        "'submit' (send data + Enter, for answering prompts), 'close' "
        "(close stdin/send EOF)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list",
                    "poll",
                    "log",
                    "wait",
                    "kill",
                    "write",
                    "submit",
                    "close",
                ],
                "description": "Action to perform on background processes",
            },
            "session_id": {
                "type": "string",
                "description": (
                    "Process session ID (from terminal background output). "
                    "Required for all actions except 'list'."
                ),
            },
            "data": {
                "type": "string",
                "description": "Text to send to process stdin (for 'write' and 'submit' actions)",
            },
            "timeout": {
                "type": "integer",
                "description": (
                    "Max seconds to block for 'wait' action. Returns partial "
                    "output on timeout."
                ),
                "minimum": 1,
            },
            "offset": {
                "type": "integer",
                "description": "Line offset for 'log' action (default: last 200 lines)",
            },
            "limit": {
                "type": "integer",
                "description": "Max lines to return for 'log' action",
                "minimum": 1,
            },
        },
        "required": ["action"],
    },
}


def _redact_process_result(result: dict[str, Any]) -> dict[str, Any]:
    from agent.redact import redact_sensitive_text, redact_terminal_output

    command = result.get("command") or ""
    for field_name in ("output", "output_preview"):
        value = result.get(field_name)
        if isinstance(value, str) and value:
            result[field_name] = redact_terminal_output(value, command)
    if isinstance(result.get("command"), str) and result["command"]:
        result["command"] = redact_sensitive_text(result["command"], code_file=True)
    return result


async def _handle_process(args: dict[str, Any], **kwargs: Any) -> str:
    task_id = kwargs.get("task_id")
    action = args.get("action", "")
    raw_session_id = args.get("session_id")
    session_id = str(raw_session_id) if raw_session_id is not None else ""
    if action == "list":
        return json.dumps(
            {"processes": process_registry.list_sessions(task_id=task_id)},
            ensure_ascii=False,
        )
    if action not in {"poll", "log", "wait", "kill", "write", "submit", "close"}:
        return tool_error(
            f"Unknown process action: {action}. Use: list, poll, log, wait, "
            "kill, write, submit, close"
        )
    if not session_id:
        return tool_error(f"session_id is required for {action}")
    if action == "poll":
        result = process_registry.poll(session_id)
    elif action == "log":
        result = process_registry.read_log(
            session_id,
            offset=args.get("offset", 0),
            limit=args.get("limit", 200),
        )
    elif action == "wait":
        result = await process_registry.wait(session_id, timeout=args.get("timeout"))
    elif action == "kill":
        result = await process_registry.kill_process(session_id)
    elif action == "write":
        result = await process_registry.write_stdin(
            session_id, str(args.get("data", ""))
        )
    elif action == "submit":
        result = await process_registry.submit_stdin(
            session_id, str(args.get("data", ""))
        )
    else:
        result = await process_registry.close_stdin(session_id)
    return json.dumps(_redact_process_result(result), ensure_ascii=False)


registry.register(
    name="process",
    toolset="terminal",
    schema=PROCESS_SCHEMA,
    handler=_handle_process,
    emoji="⚙️",
)
