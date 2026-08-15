"""Native-async background process management for the terminal tool.

The public tool name, schema, result fields, and source location mirror Hermes
Agent.  Local subprocess I/O is owned by the caller's event loop; no reader
threads, blocking ``Popen`` handles, or thread bridges are used.
"""

from __future__ import annotations

import asyncio
import codecs
import contextvars
import errno
import inspect
import json
import logging
import os
import shlex
import signal
import struct
import sys
import time
import threading
import uuid
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os

from hermes_constants import get_hermes_home
# Preserve local imports below without first-use importlib I/O in the event loop.
from tools import ansi_strip as _ansi_strip_bootstrap  # noqa: F401
from tools.environments.local import _find_bash, build_subprocess_env
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

CHECKPOINT_PATH = get_hermes_home() / "processes.json"
_IMPORTED_CHECKPOINT_PATH = Path(CHECKPOINT_PATH)
MAX_OUTPUT_CHARS = 200_000
FINISHED_TTL_SECONDS = 1800
MAX_PROCESSES = 64
MAX_ACTIVE_PROCESS_AGE = 86400
WATCH_MIN_INTERVAL_SECONDS = 15
WATCH_STRIKE_LIMIT = 3
WATCH_GLOBAL_MAX_PER_WINDOW = 15
WATCH_GLOBAL_WINDOW_SECONDS = 10
WATCH_GLOBAL_COOLDOWN_SECONDS = 30
_REMOTE_POLL_INTERVAL_SECONDS = 2

_ProcessScopeKey = tuple[object, str]
_PROCESS_NO_LOOP = object()
_process_scope_context: contextvars.ContextVar[
    tuple[str, str] | None
] = contextvars.ContextVar("process_registry_profile_scope", default=None)
_process_scope_aliases: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, str]
] = weakref.WeakKeyDictionary()
_process_scope_aliases_lock = threading.RLock()


def _lexical_process_profile_identity() -> str:
    """Return the environment-only profile marker without filesystem I/O."""
    return os.path.normcase(os.fspath(get_hermes_home()))


def _current_process_scope_key() -> _ProcessScopeKey:
    lexical = _lexical_process_profile_identity()
    active = _process_scope_context.get()
    try:
        loop: object = asyncio.get_running_loop()
    except RuntimeError:
        loop = _PROCESS_NO_LOOP
    if active is not None and active[0] == lexical and loop is not _PROCESS_NO_LOOP:
        return loop, active[1]
    if loop is _PROCESS_NO_LOOP:
        return loop, lexical
    with _process_scope_aliases_lock:
        aliases = _process_scope_aliases.get(loop)
        canonical = aliases.get(lexical, lexical) if aliases is not None else lexical
    return loop, canonical


async def _activate_process_scope() -> tuple[_ProcessScopeKey, _ProcessScopeKey]:
    """Activate the current loop's canonical profile and return old/new keys."""
    lexical = _lexical_process_profile_identity()
    loop = asyncio.get_running_loop()
    lexical_key: _ProcessScopeKey = (loop, lexical)
    active = _process_scope_context.get()
    if active is not None and active[0] == lexical:
        canonical = active[1]
    else:
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
    canonical_key: _ProcessScopeKey = (loop, canonical)
    with _process_scope_aliases_lock:
        _process_scope_aliases.setdefault(loop, {})[lexical] = canonical
    _process_scope_context.set((lexical, canonical))
    return lexical_key, canonical_key


@dataclass
class _ProcessRegistryState:
    running: dict[str, ProcessSession] = field(default_factory=dict)
    finished: dict[str, ProcessSession] = field(default_factory=dict)
    completion_consumed: set[str] = field(default_factory=set)
    poll_observed: set[str] = field(default_factory=set)
    checkpoint_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending_watchers: list[dict[str, Any]] = field(default_factory=list)
    completion_queue: asyncio.Queue[dict[str, Any]] = field(
        default_factory=asyncio.Queue
    )
    global_watch_window_start: float = 0.0
    global_watch_window_hits: int = 0
    global_watch_tripped_until: float = 0.0
    global_watch_suppressed_during_trip: int = 0


async def _finish_cancelled_cleanup(
    cleanup: asyncio.Task[Any],
    *,
    error_message: str,
) -> None:
    """Wait through repeated caller cancellation for an owned cleanup task."""
    while True:
        try:
            await asyncio.shield(cleanup)
            return
        except asyncio.CancelledError:  # noqa: ASYNC103 - retry unless child cancelled
            if cleanup.cancelled():
                raise
        except Exception:
            logger.debug(error_message, exc_info=True)
            return


def format_uptime_short(seconds: int) -> str:
    """Format process uptime using the upstream compact representation."""
    normalized = max(0, int(seconds))
    if normalized < 60:
        return f"{normalized}s"
    minutes, remaining_seconds = divmod(normalized, 60)
    if minutes < 60:
        return f"{minutes}m {remaining_seconds}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h {remaining_minutes}m"


@dataclass
class ProcessSession:
    """One background process and the event-loop resources that own it."""

    id: str
    command: str
    task_id: str = ""
    session_key: str = ""
    pid: int | None = None
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    env_ref: Any = None
    cwd: str | None = None
    started_at: float = 0.0
    host_start_time: int | None = None
    exited: bool = False
    exit_code: int | None = None
    completion_reason: str = "exited"
    termination_source: str = ""
    output_buffer: str = ""
    max_output_chars: int = MAX_OUTPUT_CHARS
    detached: bool = False
    pid_scope: str = "host"
    watcher_platform: str = ""
    watcher_chat_id: str = ""
    watcher_user_id: str = ""
    watcher_user_name: str = ""
    watcher_thread_id: str = ""
    watcher_message_id: str = ""
    watcher_interval: int = 0
    notify_on_complete: bool = False
    watch_patterns: list[str] = field(default_factory=list)
    _watch_hits: int = field(default=0, repr=False)
    _watch_suppressed: int = field(default=0, repr=False)
    _watch_disabled: bool = field(default=False, repr=False)
    _watch_last_emit_at: float = field(default=0.0, repr=False)
    _watch_cooldown_until: float = field(default=0.0, repr=False)
    _watch_strike_candidate: bool = field(default=False, repr=False)
    _watch_consecutive_strikes: int = field(default=0, repr=False)
    _notification_enqueued: bool = field(default=False, repr=False)
    _monitor_task: asyncio.Task[None] | None = field(default=None, repr=False)
    _pty_master_fd: int | None = field(default=None, repr=False)
    _completion_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)


class ProcessRegistry:
    """Track local background processes without leaving the event loop."""

    def __init__(self) -> None:
        self._loop_profile_states: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, dict[str, _ProcessRegistryState]
        ] = weakref.WeakKeyDictionary()
        self._staged_profile_states: dict[str, _ProcessRegistryState] = {}

    def _prune_closed_loops(self) -> None:
        loop_states = getattr(self, "_loop_profile_states", None)
        if loop_states is not None:
            for loop in tuple(loop_states):
                if loop.is_closed():
                    loop_states.pop(loop, None)
        with _process_scope_aliases_lock:
            for loop in tuple(_process_scope_aliases):
                if loop.is_closed():
                    _process_scope_aliases.pop(loop, None)

    def _states_for_scope(
        self,
        scope: _ProcessScopeKey,
    ) -> tuple[dict[str, _ProcessRegistryState], str]:
        loop, profile = scope
        if loop is _PROCESS_NO_LOOP:
            states = getattr(self, "_staged_profile_states", None)
            if states is None:
                states = {}
                self._staged_profile_states = states
            return states, profile
        loop_states = getattr(self, "_loop_profile_states", None)
        if loop_states is None:
            loop_states = weakref.WeakKeyDictionary()
            self._loop_profile_states = loop_states
        return loop_states.setdefault(loop, {}), profile

    def _profile_state(self) -> _ProcessRegistryState:
        self._prune_closed_loops()
        states, profile = self._states_for_scope(_current_process_scope_key())
        return states.setdefault(profile, _ProcessRegistryState())

    async def _activate_profile_state(self) -> _ProcessRegistryState:
        self._prune_closed_loops()
        lexical_key, canonical_key = await _activate_process_scope()
        states, profile = self._states_for_scope(canonical_key)
        for source_key in (lexical_key, (_PROCESS_NO_LOOP, lexical_key[1])):
            source_states, source_profile = self._states_for_scope(source_key)
            if source_key == canonical_key or source_profile not in source_states:
                continue
            staged = source_states.pop(source_profile)
            active = states.get(profile)
            if active is None:
                states[profile] = staged
                continue
            active.running.update(staged.running)
            active.finished.update(staged.finished)
            active.completion_consumed.update(staged.completion_consumed)
            active.poll_observed.update(staged.poll_observed)
            active.pending_watchers.extend(staged.pending_watchers)
            while not staged.completion_queue.empty():
                active.completion_queue.put_nowait(
                    staged.completion_queue.get_nowait()
                )
        return states.setdefault(profile, _ProcessRegistryState())

    @property
    def _running(self) -> dict[str, ProcessSession]:
        return self._profile_state().running

    @_running.setter
    def _running(self, value: dict[str, ProcessSession]) -> None:
        self._profile_state().running = value

    @property
    def _finished(self) -> dict[str, ProcessSession]:
        return self._profile_state().finished

    @_finished.setter
    def _finished(self, value: dict[str, ProcessSession]) -> None:
        self._profile_state().finished = value

    @property
    def _completion_consumed(self) -> set[str]:
        return self._profile_state().completion_consumed

    @_completion_consumed.setter
    def _completion_consumed(self, value: set[str]) -> None:
        self._profile_state().completion_consumed = value

    @property
    def _poll_observed(self) -> set[str]:
        return self._profile_state().poll_observed

    @_poll_observed.setter
    def _poll_observed(self, value: set[str]) -> None:
        self._profile_state().poll_observed = value

    @property
    def _checkpoint_lock(self) -> asyncio.Lock:
        return self._profile_state().checkpoint_lock

    @_checkpoint_lock.setter
    def _checkpoint_lock(self, value: asyncio.Lock) -> None:
        self._profile_state().checkpoint_lock = value

    @property
    def pending_watchers(self) -> list[dict[str, Any]]:
        return self._profile_state().pending_watchers

    @pending_watchers.setter
    def pending_watchers(self, value: list[dict[str, Any]]) -> None:
        self._profile_state().pending_watchers = value

    @property
    def completion_queue(self) -> asyncio.Queue[dict[str, Any]]:
        return self._profile_state().completion_queue

    @completion_queue.setter
    def completion_queue(self, value: asyncio.Queue[dict[str, Any]]) -> None:
        self._profile_state().completion_queue = value

    @property
    def _global_watch_window_start(self) -> float:
        return self._profile_state().global_watch_window_start

    @_global_watch_window_start.setter
    def _global_watch_window_start(self, value: float) -> None:
        self._profile_state().global_watch_window_start = value

    @property
    def _global_watch_window_hits(self) -> int:
        return self._profile_state().global_watch_window_hits

    @_global_watch_window_hits.setter
    def _global_watch_window_hits(self, value: int) -> None:
        self._profile_state().global_watch_window_hits = value

    @property
    def _global_watch_tripped_until(self) -> float:
        return self._profile_state().global_watch_tripped_until

    @_global_watch_tripped_until.setter
    def _global_watch_tripped_until(self, value: float) -> None:
        self._profile_state().global_watch_tripped_until = value

    @property
    def _global_watch_suppressed_during_trip(self) -> int:
        return self._profile_state().global_watch_suppressed_during_trip

    @_global_watch_suppressed_during_trip.setter
    def _global_watch_suppressed_during_trip(self, value: int) -> None:
        self._profile_state().global_watch_suppressed_during_trip = value

    def _checkpoint_path(self) -> Path:
        configured = Path(CHECKPOINT_PATH)
        if configured != _IMPORTED_CHECKPOINT_PATH:
            return configured
        return Path(_current_process_scope_key()[1]) / "processes.json"

    async def spawn_local(
        self,
        command: str,
        cwd: str | None = None,
        task_id: str = "",
        session_key: str = "",
        env_vars: dict | None = None,
        use_pty: bool = False,
    ) -> ProcessSession:
        """Spawn a local command and start native async output collection."""
        await self._activate_profile_state()
        from tools.terminal_tool import _rewrite_compound_background

        safe_command = _rewrite_compound_background(command)
        child_env_vars = dict(env_vars or {})
        snapshot_path = child_env_vars.pop(
            "_HERMES_SESSION_SNAPSHOT_PATH",
            None,
        )
        if snapshot_path:
            safe_command = (
                f"source {shlex.quote(str(snapshot_path))} >/dev/null 2>&1 || true\n"
                + safe_command
            )
        raw_workdir = cwd or await aiofiles.os.getcwd()
        expanduser = aiofiles.os.wrap(os.path.expanduser)
        workdir = await aiofiles.os.path.abspath(await expanduser(raw_workdir))
        # Session snapshots are written by ``LocalEnvironment`` in Bash
        # syntax (``declare -x``, aliases, and ``shopt``).  ``$SHELL`` is not
        # guaranteed to be Bash and is commonly unset on Linux CI, where
        # falling back to dash made ``source`` a no-op and silently dropped
        # every exported session variable from background commands.  Preserve
        # the historical shell choice for direct registry callers that do not
        # consume a session snapshot.
        shell = (
            await _find_bash()
            if snapshot_path
            else os.environ.get("SHELL") or "/bin/sh"
        )
        env = await build_subprocess_env(scrub_secrets=True, extra=child_env_vars)
        env["PYTHONUNBUFFERED"] = "1"
        session = ProcessSession(
            id=f"proc_{uuid.uuid4().hex[:12]}",
            command=command,
            task_id=str(task_id or ""),
            session_key=str(session_key or ""),
            cwd=workdir,
            started_at=time.time(),
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
        session.host_start_time = await self._safe_host_start_time(process.pid)
        self._prune_finished()
        self._running[session.id] = session
        session._monitor_task = asyncio.create_task(
            self._monitor(session),
            name=f"process-monitor-{session.id}",
        )
        try:
            await self._write_checkpoint()
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(
                self.kill_process(
                    session.id,
                    source="failed_start",
                    consume_output=False,
                )
            )
            await _finish_cancelled_cleanup(
                cleanup,
                error_message="Process cleanup after cancelled spawn failed",
            )
            raise
        return session

    @staticmethod
    async def _env_temp_dir(env: Any) -> str:
        """Return the writable sandbox temp dir for env-backed background tasks."""
        get_temp_dir = getattr(env, "get_temp_dir", None)
        if callable(get_temp_dir):
            try:
                temp_dir = get_temp_dir()
                if inspect.isawaitable(temp_dir):
                    temp_dir = await temp_dir
                if isinstance(temp_dir, str) and temp_dir.startswith("/"):
                    return temp_dir.rstrip("/") or "/"
            except Exception as exc:
                logger.debug("Could not resolve environment temp dir: %s", exc)
        return "/tmp"

    async def spawn_via_env(
        self,
        env: Any,
        command: str,
        cwd: str | None = None,
        task_id: str = "",
        session_key: str = "",
        timeout: int = 10,
    ) -> ProcessSession:
        """Spawn and monitor a background process inside a remote backend."""
        await self._activate_profile_state()
        session = ProcessSession(
            id=f"proc_{uuid.uuid4().hex[:12]}",
            command=command,
            task_id=task_id,
            session_key=session_key,
            cwd=cwd,
            started_at=time.time(),
            env_ref=env,
            pid_scope="sandbox",
        )
        temp_dir = await self._env_temp_dir(env)
        log_path = f"{temp_dir}/hermes_bg_{session.id}.log"
        pid_path = f"{temp_dir}/hermes_bg_{session.id}.pid"
        exit_path = f"{temp_dir}/hermes_bg_{session.id}.exit"
        quoted_command = shlex.quote(command)
        quoted_temp_dir = shlex.quote(temp_dir)
        quoted_log_path = shlex.quote(log_path)
        quoted_pid_path = shlex.quote(pid_path)
        quoted_exit_path = shlex.quote(exit_path)
        background_command = (
            f"mkdir -p {quoted_temp_dir} && "
            f"( nohup bash -lc {quoted_command} > {quoted_log_path} 2>&1; "
            f"rc=$?; printf '%s\\n' \"$rc\" > {quoted_exit_path} ) & "
            f"echo $! > {quoted_pid_path} && cat {quoted_pid_path}"
        )

        async def cleanup_uncertain_start() -> None:
            await env.execute(
                f"test ! -f {quoted_pid_path} || "
                f"kill \"$(cat {quoted_pid_path})\" 2>/dev/null || true",
                timeout=5,
                rewrite_compound_background=False,
            )

        try:
            result = await env.execute(
                background_command,
                timeout=timeout,
                rewrite_compound_background=False,
            )
        except asyncio.CancelledError as cancellation:
            cleanup = asyncio.create_task(cleanup_uncertain_start())
            await _finish_cancelled_cleanup(
                cleanup,
                error_message="Remote process cleanup after cancelled spawn failed",
            )
            raise cancellation
        except Exception as exc:
            session.exited = True
            session.exit_code = -1
            session.completion_reason = "failed_start"
            session.termination_source = "failed_start"
            session.output_buffer = f"Failed to start: {exc}"
            return session

        output = str(result.get("output", "")).strip()
        for line in output.splitlines():
            if line.strip().isdigit():
                session.pid = int(line.strip())
                break
        if session.pid is None:
            session.exited = True
            session.exit_code = int(result.get("returncode", -1))
            if session.exit_code == 0:
                session.exit_code = -1
            session.completion_reason = "failed_start"
            session.termination_source = "failed_start"
            session.output_buffer = output
            return session

        self._prune_finished()
        self._running[session.id] = session
        session._monitor_task = asyncio.create_task(
            self._monitor_env(session, env, log_path, pid_path, exit_path),
            name=f"process-poller-{session.id}",
        )
        try:
            await self._write_checkpoint()
        except asyncio.CancelledError as cancellation:
            cleanup = asyncio.create_task(
                self.kill_process(
                    session.id,
                    source="failed_start",
                    consume_output=False,
                )
            )
            await _finish_cancelled_cleanup(
                cleanup,
                error_message="Remote process cleanup after cancelled checkpoint failed",
            )
            raise cancellation
        return session

    async def _monitor_env(
        self,
        session: ProcessSession,
        env: Any,
        log_path: str,
        pid_path: str,
        exit_path: str,
    ) -> None:
        """Poll one remote process without a reader thread."""
        quoted_log_path = shlex.quote(log_path)
        quoted_pid_path = shlex.quote(pid_path)
        quoted_exit_path = shlex.quote(exit_path)
        previous_output_length = 0
        while not session.exited:
            await asyncio.sleep(_REMOTE_POLL_INTERVAL_SECONDS)
            if session.exited:
                return
            try:
                result = await env.execute(
                    f"cat {quoted_log_path} 2>/dev/null",
                    timeout=10,
                )
                new_output = str(result.get("output", ""))
                if new_output:
                    delta = (
                        new_output[previous_output_length:]
                        if len(new_output) > previous_output_length
                        else ""
                    )
                    previous_output_length = len(new_output)
                    session.output_buffer = new_output[-session.max_output_chars :]
                    if delta:
                        self._check_watch_patterns(session, delta)

                check = await env.execute(
                    f"kill -0 \"$(cat {quoted_pid_path} 2>/dev/null)\" "
                    "2>/dev/null; echo $?",
                    timeout=5,
                )
                check_output = str(check.get("output", "")).strip()
                if check_output and check_output.splitlines()[-1].strip() != "0":
                    exit_result = await env.execute(
                        f"cat {quoted_exit_path} 2>/dev/null",
                        timeout=5,
                    )
                    exit_output = str(exit_result.get("output", "")).strip()
                    try:
                        session.exit_code = int(
                            exit_output.splitlines()[-1].strip()
                        )
                    except (ValueError, IndexError):
                        session.exit_code = -1
                    session.exited = True
                    if session.completion_reason != "killed":
                        session.completion_reason = "exited"
                    await self._move_to_finished(session)
                    return
            except asyncio.CancelledError:
                raise
            except Exception:
                session.exited = True
                session.exit_code = -1
                session.completion_reason = "lost"
                session.termination_source = "backend_lost"
                await self._move_to_finished(session)
                return

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
            await self._move_to_finished(session)
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

    def _append_output(self, session: ProcessSession, text: str) -> None:
        if not text:
            return
        session.output_buffer += text
        if len(session.output_buffer) > session.max_output_chars:
            session.output_buffer = session.output_buffer[-session.max_output_chars :]
        self._check_watch_patterns(session, text)

    def _check_watch_patterns(self, session: ProcessSession, new_text: str) -> None:
        """Queue rate-limited upstream watch-pattern events."""
        if not session.watch_patterns or session._watch_disabled or session.exited:
            return
        matched_lines: list[str] = []
        matched_pattern: str | None = None
        for line in new_text.splitlines():
            for pattern in session.watch_patterns:
                if pattern in line:
                    matched_lines.append(line.rstrip())
                    if matched_pattern is None:
                        matched_pattern = pattern
                    break
        if not matched_lines:
            return

        now = time.time()
        if session._watch_cooldown_until and now < session._watch_cooldown_until:
            session._watch_suppressed += len(matched_lines)
            if not session._watch_strike_candidate:
                session._watch_strike_candidate = True
                session._watch_consecutive_strikes += 1
                if session._watch_consecutive_strikes >= WATCH_STRIKE_LIMIT:
                    session._watch_disabled = True
                    session.notify_on_complete = True
                    self.completion_queue.put_nowait(
                        {
                            "session_id": session.id,
                            "session_key": session.session_key,
                            "command": session.command,
                            "type": "watch_disabled",
                            "suppressed": session._watch_suppressed,
                            "message": (
                                f"Watch patterns disabled for process {session.id} — "
                                f"{WATCH_STRIKE_LIMIT} consecutive rate-limit "
                                "windows triggered. Falling back to "
                                "notify_on_complete semantics."
                            ),
                        }
                    )
            return

        if session._watch_cooldown_until and not session._watch_strike_candidate:
            session._watch_consecutive_strikes = 0
        session._watch_strike_candidate = False
        session._watch_last_emit_at = now
        session._watch_cooldown_until = now + WATCH_MIN_INTERVAL_SECONDS
        session._watch_hits += 1
        suppressed = session._watch_suppressed
        session._watch_suppressed = 0

        if not self._global_watch_admit(now):
            return
        output = "\n".join(matched_lines[:20])
        if len(output) > 2000:
            output = output[:2000] + "\n...(truncated)"
        self.completion_queue.put_nowait(
            {
                "session_id": session.id,
                "session_key": session.session_key,
                "command": session.command,
                "type": "watch_match",
                "pattern": matched_pattern,
                "output": output,
                "suppressed": suppressed,
            }
        )

    def _global_watch_admit(self, now: float) -> bool:
        """Apply the upstream process-wide watch notification circuit breaker."""
        if self._global_watch_tripped_until and now >= self._global_watch_tripped_until:
            suppressed = self._global_watch_suppressed_during_trip
            self._global_watch_tripped_until = 0.0
            self._global_watch_suppressed_during_trip = 0
            self._global_watch_window_start = now
            self._global_watch_window_hits = 0
            if suppressed:
                self.completion_queue.put_nowait(
                    {
                        "type": "watch_overflow_released",
                        "session_id": "",
                        "session_key": "",
                        "command": "",
                        "suppressed": suppressed,
                        "message": (
                            "Watch-pattern notifications resumed. "
                            f"{suppressed} match event(s) were suppressed."
                        ),
                    }
                )
        if self._global_watch_tripped_until and now < self._global_watch_tripped_until:
            self._global_watch_suppressed_during_trip += 1
            return False
        if now - self._global_watch_window_start >= WATCH_GLOBAL_WINDOW_SECONDS:
            self._global_watch_window_start = now
            self._global_watch_window_hits = 0
        if self._global_watch_window_hits >= WATCH_GLOBAL_MAX_PER_WINDOW:
            self._global_watch_tripped_until = now + WATCH_GLOBAL_COOLDOWN_SECONDS
            self._global_watch_suppressed_during_trip += 1
            self.completion_queue.put_nowait(
                {
                    "type": "watch_overflow_tripped",
                    "session_id": "",
                    "session_key": "",
                    "command": "",
                    "message": (
                        f"Watch-pattern overflow: >{WATCH_GLOBAL_MAX_PER_WINDOW} "
                        f"notifications in {WATCH_GLOBAL_WINDOW_SECONDS}s. "
                        f"Suppressing events for {WATCH_GLOBAL_COOLDOWN_SECONDS}s."
                    ),
                }
            )
            return False
        self._global_watch_window_hits += 1
        return True

    def _enqueue_completion(self, session: ProcessSession) -> None:
        if session._notification_enqueued:
            return
        from tools.ansi_strip import strip_ansi

        session._notification_enqueued = True
        self.completion_queue.put_nowait(
            {
                "type": "completion",
                "session_id": session.id,
                "session_key": session.session_key,
                "command": session.command,
                "exit_code": session.exit_code,
                "completion_reason": session.completion_reason,
                "termination_source": session.termination_source,
                "output": strip_ansi(session.output_buffer[-2000:]),
                "started_at": session.started_at,
            }
        )

    async def _move_to_finished(self, session: ProcessSession) -> None:
        was_running = self._running.pop(session.id, None) is not None
        self._finished[session.id] = session
        await self._write_checkpoint()
        if was_running and session.notify_on_complete:
            self._enqueue_completion(session)
        # Publication is the final lifecycle edge: waiters may immediately
        # tear their event loop down after this signal, so every checkpoint
        # file handle and atomic replace must already be complete.
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

    async def get(self, session_id: str) -> ProcessSession | None:
        await self._activate_profile_state()
        session = self._running.get(session_id) or self._finished.get(session_id)
        return await self._refresh_detached_session(session)

    async def poll(self, session_id: str) -> dict[str, Any]:
        """Return status plus the latest output without consuming completion."""
        from tools.ansi_strip import strip_ansi

        session = await self.get(session_id)
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
            self._poll_observed.add(session_id)
        if session.detached:
            result["detached"] = True
            result["note"] = (
                "Process recovered after restart -- output history unavailable"
            )
        return result

    def is_completion_consumed(self, session_id: str) -> bool:
        return session_id in self._completion_consumed

    async def is_session_waiting(self, session_id: str) -> bool:
        """Return whether a goal loop should remain parked on this process."""
        if not session_id:
            return False
        session = await self.get(session_id)
        if session is None or session.exited:
            return False
        return not (
            session.watch_patterns
            and not session._watch_disabled
            and session._watch_hits > 0
        )

    def _drain_should_skip(
        self,
        session_id: str,
        *,
        skip_poll_observed: bool = True,
    ) -> bool:
        return session_id in self._completion_consumed or (
            skip_poll_observed and session_id in self._poll_observed
        )

    def drain_notifications(
        self,
        session_key: str = "",
        owns_event=None,
        *,
        skip_poll_observed: bool = True,
    ) -> list[tuple[dict[str, Any], str]]:
        """Drain routed process notifications without crossing session ownership."""
        results: list[tuple[dict[str, Any], str]] = []
        requeue: list[dict[str, Any]] = []
        for _ in range(self.completion_queue.qsize()):
            try:
                event = self.completion_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            is_async_delegation = event.get("type") == "async_delegation"
            event_session_key = str(event.get("session_key") or "")
            origin_session_id = str(event.get("origin_ui_session_id") or "")
            requires_positive_proof = is_async_delegation or bool(
                event_session_key or origin_session_id
            )
            if owns_event is not None and requires_positive_proof:
                try:
                    owned = bool(owns_event(event))
                except Exception:
                    owned = False
                if not owned:
                    requeue.append(event)
                    continue
            elif session_key and requires_positive_proof:
                if event_session_key != session_key:
                    requeue.append(event)
                    continue
            elif is_async_delegation and event.get("restored"):
                requeue.append(event)
                continue
            event_session_id = str(event.get("session_id") or "")
            if event.get("type") == "completion" and self._drain_should_skip(
                event_session_id,
                skip_poll_observed=skip_poll_observed,
            ):
                continue
            formatted = format_process_notification(event)
            if formatted:
                results.append((event, formatted))
        for event in requeue:
            self.completion_queue.put_nowait(event)
        return results

    async def read_log(
        self, session_id: str, offset: int | None = None, limit: int = 200
    ) -> dict[str, Any]:
        """Return the process output using Hermes' line pagination contract."""
        from tools.ansi_strip import strip_ansi

        session = await self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}
        lines = strip_ansi(session.output_buffer).splitlines()
        if offset is None and limit > 0:
            selected = lines[-limit:]
            consumed = bool(selected) or not lines
        else:
            offset = offset or 0
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

    async def wait(
        self,
        session_id: str,
        timeout: int | None = None,  # noqa: ASYNC109 - upstream public API
    ) -> dict:
        """Wait without blocking the event loop or stopping the process on timeout."""
        from tools.ansi_strip import strip_ansi
        from tools.interrupt import is_interrupted

        session = await self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}
        from agent.secret_scope import get_secret

        try:
            configured_timeout = int(get_secret("TERMINAL_TIMEOUT", "180"))
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
            session = await self._refresh_detached_session(session)
            if session is None or session.exited:
                break
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
        from tools.environments.local import _terminate_process

        session = await self.get(session_id)
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
            elif session.env_ref is not None and session.pid:
                async def terminate_remote() -> None:
                    await session.env_ref.execute(
                        f"kill {session.pid} 2>/dev/null",
                        timeout=5,
                    )
                    session.exited = True
                    session.exit_code = -signal.SIGTERM
                    await self._move_to_finished(session)
                    monitor = session._monitor_task
                    if monitor is not None:
                        await monitor

                remote_cleanup = asyncio.create_task(terminate_remote())
                try:
                    await asyncio.shield(remote_cleanup)
                except asyncio.CancelledError as cancellation:
                    await _finish_cancelled_cleanup(
                        remote_cleanup,
                        error_message="Remote process termination failed",
                    )
                    raise cancellation
            elif session.detached and session.pid_scope == "host" and session.pid:
                terminated = await self._terminate_host_pid(
                    session.pid,
                    session.host_start_time,
                )
                if not terminated:
                    session.exited = True
                    session.exit_code = None
                    session.completion_reason = "exited"
                    await self._move_to_finished(session)
                    if consume_output:
                        self._completion_consumed.add(session_id)
                    return {
                        "status": "already_exited",
                        "exit_code": None,
                        "output": strip_ansi(session.output_buffer[-2000:]),
                    }
                session.exited = True
                session.exit_code = None
                await self._move_to_finished(session)
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
        session = await self.get(session_id)
        if session is None:
            return {"status": "not_found", "error": f"No process with ID {session_id}"}
        if session.exited:
            return {"status": "already_exited", "error": "Process has already finished"}
        payload = str(data).encode("utf-8", "surrogateescape")
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
        session = await self.get(session_id)
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

    async def list_sessions(
        self,
        task_id: str | None = None,
        session_key: str | None = None,
    ) -> list:
        await self._activate_profile_state()
        sessions: list[ProcessSession] = []
        for session in [*self._running.values(), *self._finished.values()]:
            refreshed = await self._refresh_detached_session(session)
            if refreshed is not None:
                sessions.append(refreshed)
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
            if (
                task_id
                and session_key
                and session.task_id != task_id
                and session.session_key == session_key
            ):
                entry["session_scoped"] = True
            if session.watch_patterns and not session._watch_disabled:
                entry["watch_patterns"] = list(session.watch_patterns)
                entry["watch_hit"] = session._watch_hits > 0
            if session.notify_on_complete:
                entry["notify_on_complete"] = True
            if session.detached:
                entry["detached"] = True
            result.append(entry)
        return result

    def count_running(self) -> int:
        return len(self._running)

    async def has_active_processes(self, task_id: str) -> bool:
        await self._activate_profile_state()
        for session in tuple(self._running.values()):
            await self._refresh_detached_session(session)
        return any(session.task_id == task_id for session in self._running.values())

    async def has_active_for_session(
        self,
        session_key: str,
        max_active_age: float | None = None,
    ) -> bool:
        await self._activate_profile_state()
        for session in tuple(self._running.values()):
            await self._refresh_detached_session(session)
        now = time.time()
        return any(
            session.session_key == session_key
            and (max_active_age is None or now - session.started_at < max_active_age)
            for session in self._running.values()
        )

    async def has_any_active(self) -> bool:
        await self._activate_profile_state()
        for session in tuple(self._running.values()):
            await self._refresh_detached_session(session)
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
        exclude_ids: frozenset = frozenset(),
        source: str = "kill_all",
        consume_output: bool = False,
    ) -> int:
        await self._activate_profile_state()
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

    @staticmethod
    async def _safe_host_start_time(pid: int | None) -> int | None:
        """Return Linux kernel start ticks for PID-reuse validation."""
        if not pid or not sys.platform.startswith("linux"):
            return None
        try:
            async with aiofiles.open(f"/proc/{int(pid)}/stat", "rb") as handle:
                stat = await handle.read()
        except (FileNotFoundError, PermissionError, OSError, ValueError):
            return None
        _prefix, separator, fields = stat.rpartition(b") ")
        if not separator:
            return None
        parts = fields.split()
        try:
            return int(parts[19])
        except (IndexError, TypeError, ValueError):
            return None

    @classmethod
    async def _host_pid_is_ours(
        cls,
        pid: int | None,
        expected_start: int | None,
    ) -> bool:
        if not pid:
            return False
        from gateway.status import _pid_exists

        if not await _pid_exists(pid):
            return False
        if expected_start is None:
            return True
        return await cls._safe_host_start_time(pid) == expected_start

    async def _refresh_detached_session(
        self,
        session: ProcessSession | None,
    ) -> ProcessSession | None:
        """Move a recovered session to finished when its original PID is gone."""
        if (
            session is None
            or session.exited
            or not session.detached
            or session.pid_scope != "host"
        ):
            return session
        if await self._host_pid_is_ours(session.pid, session.host_start_time):
            return session
        session.exited = True
        session.exit_code = None
        session.completion_reason = "exited"
        await self._move_to_finished(session)
        return session

    @classmethod
    async def _terminate_host_pid(
        cls,
        pid: int,
        expected_start: int | None = None,
    ) -> bool:
        """Terminate a recovered host process without touching a recycled PID."""
        if not await cls._host_pid_is_ours(pid, expected_start):
            logger.warning(
                "Refusing to terminate host pid %d: process is gone or PID was recycled",
                pid,
            )
            return False

        if os.name == "nt":
            process = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(pid),
                "/T",
                "/F",
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                await process.wait()
            except asyncio.CancelledError:
                if process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                cleanup = asyncio.create_task(process.wait())
                await _finish_cancelled_cleanup(
                    cleanup,
                    error_message="Recovered host process cleanup failed",
                )
                raise
        else:
            pgid: int | None
            try:
                pgid = os.getpgid(pid)
            except (ProcessLookupError, PermissionError, OSError):
                pgid = None

            def send(signal_number: int) -> None:
                try:
                    if pgid is not None and pgid != os.getpgrp():
                        os.killpg(pgid, signal_number)
                    else:
                        os.kill(pid, signal_number)
                except (ProcessLookupError, PermissionError, OSError):
                    pass

            send(signal.SIGTERM)
            try:
                async with asyncio.timeout(2.0):
                    while await cls._host_pid_is_ours(pid, expected_start):
                        await asyncio.sleep(0.05)
            except TimeoutError:
                if await cls._host_pid_is_ours(pid, expected_start):
                    send(signal.SIGKILL)
        return True

    async def _write_checkpoint(
        self, extra_entries: list[dict[str, Any]] | None = None
    ) -> None:
        """Persist running host-process metadata with an atomic async replace."""
        await self._activate_profile_state()
        async with self._checkpoint_lock:
            entries: list[dict[str, Any]] = []
            for session in self._running.values():
                if session.exited:
                    continue
                if (
                    session.host_start_time is None
                    and session.pid_scope == "host"
                    and session.pid
                ):
                    session.host_start_time = await self._safe_host_start_time(
                        session.pid
                    )
                entries.append(
                    {
                        "session_id": session.id,
                        "command": session.command,
                        "pid": session.pid,
                        "pid_scope": session.pid_scope,
                        "host_start_time": session.host_start_time,
                        "cwd": session.cwd,
                        "started_at": session.started_at,
                        "task_id": session.task_id,
                        "session_key": session.session_key,
                        "watcher_platform": session.watcher_platform,
                        "watcher_chat_id": session.watcher_chat_id,
                        "watcher_user_id": session.watcher_user_id,
                        "watcher_user_name": session.watcher_user_name,
                        "watcher_thread_id": session.watcher_thread_id,
                        "watcher_message_id": session.watcher_message_id,
                        "watcher_interval": session.watcher_interval,
                        "notify_on_complete": session.notify_on_complete,
                        "watch_patterns": session.watch_patterns,
                    }
                )
            if extra_entries:
                tracked_ids = {entry.get("session_id") for entry in entries}
                entries.extend(
                    entry
                    for entry in extra_entries
                    if isinstance(entry, dict)
                    and entry.get("session_id") not in tracked_ids
                )
            checkpoint = self._checkpoint_path()
            temporary = checkpoint.with_name(
                f".{checkpoint.name}.{uuid.uuid4().hex}.tmp"
            )
            try:
                await aiofiles.os.makedirs(checkpoint.parent, exist_ok=True)
                async with aiofiles.open(temporary, "w", encoding="utf-8") as handle:
                    await handle.write(json.dumps(entries, ensure_ascii=False))
                    await handle.flush()
                await aiofiles.os.replace(temporary, checkpoint)
            except asyncio.CancelledError:
                try:
                    await aiofiles.os.remove(temporary)
                except OSError:
                    pass
                raise
            except Exception:
                logger.debug("Failed to write process checkpoint", exc_info=True)
                try:
                    await aiofiles.os.remove(temporary)
                except OSError:
                    pass

    async def recover_from_checkpoint(self) -> int:
        """Recover live host PIDs from the upstream processes.json checkpoint."""
        await self._activate_profile_state()
        checkpoint = self._checkpoint_path()
        try:
            async with aiofiles.open(checkpoint, encoding="utf-8") as handle:
                entries = json.loads(await handle.read())
        except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
            return 0
        if not isinstance(entries, list):
            return 0

        recovered = 0
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            pid = entry.get("pid")
            if not isinstance(pid, int) or not pid:
                continue
            pid_scope = entry.get("pid_scope", "host")
            if pid_scope != "host":
                continue
            recorded_start = entry.get("host_start_time")
            if not await self._host_pid_is_ours(pid, recorded_start):
                continue
            session_id = entry.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                continue
            self._running[session_id] = ProcessSession(
                id=session_id,
                command=str(entry.get("command", "unknown")),
                task_id=str(entry.get("task_id", "")),
                session_key=str(entry.get("session_key", "")),
                pid=pid,
                host_start_time=(
                    recorded_start if isinstance(recorded_start, int) else None
                ),
                pid_scope="host",
                cwd=str(entry.get("cwd", "")),
                started_at=float(entry.get("started_at", time.time())),
                detached=True,
                watcher_platform=str(entry.get("watcher_platform", "")),
                watcher_chat_id=str(entry.get("watcher_chat_id", "")),
                watcher_user_id=str(entry.get("watcher_user_id", "")),
                watcher_user_name=str(entry.get("watcher_user_name", "")),
                watcher_thread_id=str(entry.get("watcher_thread_id", "")),
                watcher_message_id=str(entry.get("watcher_message_id", "")),
                watcher_interval=int(entry.get("watcher_interval", 0) or 0),
                notify_on_complete=bool(entry.get("notify_on_complete", False)),
                watch_patterns=list(entry.get("watch_patterns", []) or []),
            )
            if self._running[session_id].watcher_interval > 0:
                session = self._running[session_id]
                self.pending_watchers.append(
                    {
                        "session_id": session.id,
                        "check_interval": session.watcher_interval,
                        "session_key": session.session_key,
                        "platform": session.watcher_platform,
                        "chat_id": session.watcher_chat_id,
                        "user_id": session.watcher_user_id,
                        "user_name": session.watcher_user_name,
                        "thread_id": session.watcher_thread_id,
                        "message_id": session.watcher_message_id,
                        "notify_on_complete": session.notify_on_complete,
                    }
                )
            recovered += 1

        await self._write_checkpoint()
        return recovered


process_registry = ProcessRegistry()


def _format_age(seconds: float) -> str:
    """Human-friendly elapsed string ('18m', '2h3m', '45s')."""
    try:
        normalized = int(max(0, seconds))
    except (TypeError, ValueError):
        return "?"
    if normalized < 60:
        return f"{normalized}s"
    minutes, normalized = divmod(normalized, 60)
    if minutes < 60:
        return f"{minutes}m" if normalized == 0 else f"{minutes}m{normalized}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h" if minutes == 0 else f"{hours}h{minutes}m"


def _format_async_delegation(evt: dict[str, Any]) -> str:
    """Format an async-delegation completion into a self-contained re-injection.

    Carries the FULL original task source (goal, the context the parent
    supplied, toolsets, role, model) plus dispatch time, status, and the
    complete result summary. When this re-enters the conversation the agent
    may be deep in unrelated context and won't remember why the subagent
    existed, so the block is written to stand entirely on its own — enough to
    use the result OR re-dispatch if the world has moved on.
    """
    deleg_id = evt.get("delegation_id", "unknown")
    goal = evt.get("goal", "") or ""
    context = evt.get("context")
    toolsets = evt.get("toolsets")
    role = evt.get("role") or "leaf"
    model = evt.get("model") or "?"
    status = evt.get("status") or "completed"
    summary = evt.get("summary")
    error = evt.get("error")
    api_calls = evt.get("api_calls", 0)
    duration = evt.get("duration_seconds", "?")
    dispatched_at = evt.get("dispatched_at")
    completed_at = evt.get("completed_at") or time.time()

    batch_results = evt.get("results")
    if evt.get("is_batch") or isinstance(batch_results, list):
        results = batch_results or []
        goals = evt.get("goals") or []
        count = len(results) if results else len(goals)
        total_duration = evt.get("total_duration_seconds", duration)
        lines = [
            f"[ASYNC DELEGATION BATCH COMPLETE — {deleg_id}]",
            f"A background fan-out of {count} subagent(s) you dispatched earlier "
            "has finished. All ran in parallel and waited on each other; their "
            "consolidated results are below. You may have moved on since "
            "dispatching — act on these or re-dispatch if things have changed.",
            "",
        ]
        if isinstance(dispatched_at, (int, float)):
            timestamp = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(dispatched_at)
            )
            age = f" ({_format_age(completed_at - dispatched_at)} ago)"
            lines.append(f"Dispatched: {timestamp}{age}")
        if context:
            lines.append(f"Context you provided: {context}")
        if toolsets:
            lines.append(f"Toolsets: {', '.join(toolsets)}")
        lines.append(
            f"Role: {role}   Model: {model}   Total duration: {total_duration}s"
        )
        if error and not results:
            lines.append("--- ERROR ---")
            lines.append(f"The batch did not complete successfully: {error}")
            return "\n".join(lines)
        for result in sorted(results, key=lambda item: item.get("task_index", 0)):
            index = result.get("task_index", 0)
            result_status = result.get("status", "?")
            result_summary = result.get("summary")
            result_error = result.get("error")
            result_goal = (
                goals[index] if index < len(goals) else result.get("goal", "")
            )
            icon = "✓" if result_status in ("completed", "success") else "✗"
            lines.append("")
            header = f"--- {icon} TASK {index + 1}/{count}"
            if result_goal:
                header += f": {result_goal}"
            header += f"  (status={result_status}"
            if result.get("api_calls"):
                header += f", api_calls={result['api_calls']}"
            if result.get("duration_seconds") is not None:
                header += f", {result['duration_seconds']}s"
            header += ") ---"
            lines.append(header)
            if result_status in ("completed", "success") and result_summary:
                lines.append(result_summary)
            elif result_summary:
                if result_error:
                    lines.append(f"({result_status}: {result_error})")
                lines.append("Partial output:")
                lines.append(result_summary)
            else:
                lines.append(
                    f"(no summary — status={result_status}"
                    + (f": {result_error}" if result_error else "")
                    + ")"
                )
            live_transcript = result.get("live_transcript")
            if live_transcript:
                lines.append(
                    "Full live transcript (complete tool/assistant trace): "
                    f"{live_transcript}"
                )
        return "\n".join(lines)

    age = ""
    if isinstance(dispatched_at, (int, float)):
        age = f" ({_format_age(completed_at - dispatched_at)} ago)"

    lines = [
        f"[ASYNC DELEGATION COMPLETE — {deleg_id}]",
        "A background subagent you dispatched earlier has finished. You may "
        "have moved on since dispatching it; the full task source is below so "
        "you can act on the result or re-dispatch if things have changed.",
        "",
    ]
    if isinstance(dispatched_at, (int, float)):
        timestamp = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(dispatched_at)
        )
        lines.append(f"Dispatched: {timestamp}{age}")
    lines.append(f"Original goal: {goal}")
    if context:
        lines.append(f"Context you provided: {context}")
    if toolsets:
        lines.append(f"Toolsets: {', '.join(toolsets)}")
    lines.append(f"Role: {role}   Model: {model}")
    lines.append(f"Status: {status}   API calls: {api_calls}   Duration: {duration}s")
    lines.append("--- RESULT ---")
    if status in ("completed", "success") and summary:
        lines.append(summary)
    elif status == "interrupted":
        lines.append(
            "The subagent was interrupted before completing"
            + (f": {error}" if error else ".")
        )
        if summary:
            lines.append("Partial output:")
            lines.append(summary)
    else:
        lines.append(
            f"The subagent did not complete successfully (status={status})."
            + (f"\n{error}" if error else "")
        )
        if summary:
            lines.append("Partial output:")
            lines.append(summary)
    return "\n".join(lines)


def format_process_notification(evt: dict) -> str | None:
    """Format a queued process event using the upstream reinjection shape."""
    event_type = evt.get("type", "completion")
    session_id = evt.get("session_id", "unknown")
    command = evt.get("command", "unknown")
    if event_type == "watch_disabled":
        return f"[IMPORTANT: {evt.get('message', '')}]"
    if event_type == "watch_match":
        text = (
            f"[IMPORTANT: Background process {session_id} matched watch pattern "
            f"\"{evt.get('pattern', '?')}\".\n"
            f"Command: {command}\n"
            f"Matched output:\n{evt.get('output', '')}"
        )
        suppressed = evt.get("suppressed", 0)
        if suppressed:
            text += f"\n({suppressed} earlier matches were suppressed by rate limit)"
        return text + "]"
    if event_type == "async_delegation":
        return _format_async_delegation(evt)

    exit_code = evt.get("exit_code", "?")
    reason = evt.get("completion_reason") or "exited"
    source = evt.get("termination_source") or ""
    signal_note = ", SIGTERM" if exit_code in {-15, 143, "-15", "143"} else ""
    if reason == "killed":
        status = f"terminated by {source or 'Hermes'}"
    elif reason == "lost":
        status = "marked lost because the process backend disappeared"
    elif reason == "failed_start":
        status = "failed to start"
    elif exit_code == 0:
        status = "completed normally"
    else:
        status = "exited"
    return (
        f"[IMPORTANT: Background process {session_id} {status} "
        f"(exit code {exit_code}{signal_note}).\n"
        f"Command: {command}\n"
        f"Output:\n{evt.get('output', '')}]"
    )


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
    session_key = str(kwargs.get("session_id") or "")
    action = args.get("action", "")
    raw_session_id = args.get("session_id")
    session_id = str(raw_session_id) if raw_session_id is not None else ""
    if action == "list":
        return json.dumps(
            {
                "processes": await process_registry.list_sessions(
                    task_id=task_id,
                    session_key=session_key,
                )
            },
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
        result = await process_registry.poll(session_id)
    elif action == "log":
        result = await process_registry.read_log(
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
