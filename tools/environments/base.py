"""Native-async base contract for Hermes execution environments.

The upstream module owns the common environment lifecycle and public import
surface.  Backend transports implement their I/O with native coroutine APIs;
state-only construction and pure command shaping stay synchronous.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import re
import shlex
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import IO, Any, Protocol

import aiofiles
import aiofiles.os

from hermes_constants import get_hermes_home


logger = logging.getLogger(__name__)
_SNAPSHOT_EXCLUDED_ENV_REGEX = (
    "^declare -x (HERMES_SESSION_|HERMES_UI_SESSION_ID|"
    "HERMES_CRON_AUTO_DELIVER_|HERMES_CRON_SESSION)"
)
_SHELL_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


_UNBOUNDED_CAPTURE_CHARS = 2**63 - 1
_activity_callback: contextvars.ContextVar[Callable[[str], None] | None] = (
    contextvars.ContextVar("environment_activity_callback", default=None)
)


class ProcessHandle(Protocol):
    """Native-async form of the upstream backend process protocol.

    Concrete backends may use an asyncio subprocess or an SDK-native remote
    execution handle. Awaiting the transport methods keeps the original
    public protocol name without retaining its thread-backed adapter.
    """

    async def poll(self) -> int | None: ...

    async def kill(self) -> None: ...

    async def wait(self, timeout: float | None = None) -> int: ...

    @property
    def stdout(self) -> IO[str] | asyncio.StreamReader | None: ...

    @property
    def returncode(self) -> int | None: ...


class _BoundedOutputCollector:
    """Retain the upstream 40/60 head-tail window of streamed text.

    ``append`` remains a CPU-only operation.  If a spill is requested, the
    full stream is retained up to the upstream five-million-character cap and
    persisted by the awaited :meth:`close_spill` lifecycle boundary.  This
    keeps filesystem I/O off the event loop without scheduling background
    writer tasks that could outlive the command.
    """

    _SPILL_CAP_CHARS = 5_000_000
    _SPILL_CAP_MARKER = "\n... [spill capped at 5,000,000 chars] ...\n"

    def __init__(self, max_chars: int, spill_path: Path | None = None):
        self.max_chars = max(1, int(max_chars))
        self._head_limit = int(self.max_chars * 0.4)
        self._tail_limit = self.max_chars - self._head_limit
        self._head: list[str] = []
        self._tail: deque[str] = deque()
        self._head_chars = 0
        self._tail_chars = 0
        self._total_chars = 0
        self._spill_path = spill_path
        self._spill_parts: list[str] = []
        self._spill_chars = 0
        self._spill_used = False
        self._spill_capped = False
        self._spill_closed = False
        self._persisted_spill_path: str | None = None
        self._spill_temporary_path: Path | None = None
        self._spill_lock = asyncio.Lock()

    def _maybe_spill(self, text: str) -> None:
        if self._spill_path is None or self._spill_capped:
            return
        if not self._spill_used:
            backlog = "".join(self._head) + "".join(self._tail)
            if backlog:
                self._spill_parts.append(backlog)
                self._spill_chars = len(backlog)
            self._spill_used = True
        budget = self._SPILL_CAP_CHARS - self._spill_chars
        if budget <= 0 or len(text) > budget:
            if budget > 0:
                self._spill_parts.append(text[:budget])
            self._spill_parts.append(self._SPILL_CAP_MARKER)
            self._spill_chars += len(text)
            self._spill_capped = True
            return
        self._spill_parts.append(text)
        self._spill_chars += len(text)

    @property
    def buffered_chars(self) -> int:
        return self._head_chars + self._tail_chars

    @property
    def total_chars(self) -> int:
        return self._total_chars

    def append(self, text: str) -> None:
        if not text:
            return
        text_len = len(text)
        if self._spill_path is not None and (
            self._spill_used or self._total_chars + text_len > self.max_chars
        ):
            self._maybe_spill(text)
        self._total_chars += text_len
        start = 0
        if self._head_chars < self._head_limit:
            take = min(self._head_limit - self._head_chars, text_len)
            if take:
                self._head.append(text[:take])
                self._head_chars += take
                start = take
        remaining = text_len - start
        if remaining <= 0 or self._tail_limit <= 0:
            return
        if remaining >= self._tail_limit:
            self._tail.clear()
            self._tail.append(text[-self._tail_limit :])
            self._tail_chars = self._tail_limit
            return
        chunk = text[start:]
        self._tail.append(chunk)
        self._tail_chars += len(chunk)
        while self._tail_chars > self._tail_limit:
            excess = self._tail_chars - self._tail_limit
            first = self._tail[0]
            if len(first) <= excess:
                self._tail.popleft()
                self._tail_chars -= len(first)
            else:
                self._tail[0] = first[excess:]
                self._tail_chars -= excess

    def _account_discarded(self, char_count: int) -> None:
        """Count already-drained middle text retained by neither head nor tail."""
        self._total_chars += max(0, int(char_count))

    def render(self, *, suffix: str = "") -> str:
        if len(suffix) >= self.max_chars:
            return suffix[-self.max_chars :]
        head = "".join(self._head)
        tail = "".join(self._tail)
        available = self.max_chars - len(suffix)
        if self._total_chars <= available:
            return head + tail + suffix
        notice = ""
        for _ in range(4):
            content_budget = max(0, available - len(notice))
            head_chars = int(content_budget * 0.4)
            tail_chars = content_budget - head_chars
            omitted = max(0, self._total_chars - head_chars - tail_chars)
            updated = (
                f"\n\n... [OUTPUT TRUNCATED - {omitted:,} chars omitted "
                f"out of {self._total_chars:,} total] ...\n\n"
            )
            if updated == notice:
                break
            notice = updated
        content_budget = max(0, available - len(notice))
        head_chars = int(content_budget * 0.4)
        tail_chars = content_budget - head_chars
        rendered_tail = tail[-tail_chars:] if tail_chars else ""
        return head[:head_chars] + notice[:available] + rendered_tail + suffix

    async def close_spill(self) -> str | None:
        """Persist and close the spill, returning its path when it was used."""
        async with self._spill_lock:
            from tools.environments.file_sync import _await_owned

            close_task = asyncio.create_task(
                self._close_spill(),
                name="terminal-spill-close",
            )
            try:
                return await _await_owned(close_task)
            except asyncio.CancelledError as cancellation:
                # The owned close is shielded and may have published the raw
                # stream before caller cancellation is observed.  Terminal
                # redaction happens only after this boundary, so remove every
                # unpublished artifact before propagating cancellation.
                discard_task = asyncio.create_task(
                    self._discard_spill(),
                    name="terminal-spill-discard",
                )
                try:
                    await _await_owned(discard_task)
                except asyncio.CancelledError:  # noqa: ASYNC103 - first re-raised
                    # Repeated caller cancellation cannot interrupt owned
                    # removal; preserve the first cancellation below.
                    pass
                raise cancellation

    async def _discard_spill(self) -> None:
        for path in (self._spill_temporary_path, self._spill_path):
            if path is None:
                continue
            try:
                await aiofiles.os.remove(path)
            except OSError:
                pass
        self._spill_parts.clear()
        self._spill_closed = True
        self._persisted_spill_path = None
        self._spill_temporary_path = None

    async def _close_spill(self) -> str | None:
        if self._spill_closed:
            return self._persisted_spill_path
        if not self._spill_used or self._spill_path is None:
            self._spill_closed = True
            return None

        temporary = self._spill_path.with_name(
            f".{self._spill_path.name}.{uuid.uuid4().hex}.tmp"
        )
        self._spill_temporary_path = temporary
        try:
            await aiofiles.os.makedirs(self._spill_path.parent, exist_ok=True)
            async with aiofiles.open(
                temporary,
                "w",
                encoding="utf-8",
                errors="replace",
            ) as handle:
                for part in self._spill_parts:
                    await handle.write(part)
                await handle.flush()
                await aiofiles.os.wrap(os.fsync)(handle.fileno())
            await aiofiles.os.wrap(os.chmod)(temporary, 0o600)
            await aiofiles.os.replace(temporary, self._spill_path)
        except asyncio.CancelledError:
            raise
        except OSError:
            self._spill_parts.clear()
            self._spill_closed = True
            return None
        finally:
            try:
                await aiofiles.os.remove(temporary)
            except OSError:
                pass
            self._spill_temporary_path = None

        self._spill_parts.clear()
        self._spill_closed = True
        self._persisted_spill_path = str(self._spill_path)
        return self._persisted_spill_path


def set_activity_callback(cb: Callable[[str], None] | None) -> None:
    _activity_callback.set(cb)


def get_activity_callback() -> Callable[[str], None] | None:
    return _activity_callback.get()


def touch_activity_if_due(state: dict, label: str) -> None:
    now = time.monotonic()
    interval = state.get("interval", 10.0)
    if now - state["last_touch"] < interval:
        return
    state["last_touch"] = now
    try:
        callback = get_activity_callback()
        if callback:
            elapsed = int(now - state["start"])
            callback(f"{label} ({elapsed}s elapsed)")
    except Exception:
        pass


async def get_sandbox_dir() -> Path:
    from agent.secret_scope import get_secret

    custom = get_secret("TERMINAL_SANDBOX_DIR")
    path = Path(custom) if custom else get_hermes_home() / "sandboxes"
    await aiofiles.os.makedirs(path, exist_ok=True)
    return path


async def _file_mtime_key(host_path: str) -> tuple[float, int] | None:
    try:
        stat_result = await aiofiles.os.stat(host_path)
    except OSError:
        return None
    return stat_result.st_mtime, stat_result.st_size


async def _load_json_store(path: Path) -> dict:
    """Load a JSON object without blocking the event loop."""
    try:
        async with aiofiles.open(path, encoding="utf-8") as handle:
            data = json.loads(await handle.read())
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


async def _save_json_store(path: Path, data: dict) -> None:
    """Atomically persist a JSON object without blocking file I/O."""
    await aiofiles.os.makedirs(path.parent, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        async with aiofiles.open(temporary, "w", encoding="utf-8") as handle:
            await handle.write(json.dumps(data, indent=2))
            await handle.flush()
            await aiofiles.os.wrap(os.fsync)(handle.fileno())
        await aiofiles.os.replace(temporary, path)
    finally:
        try:
            await aiofiles.os.remove(temporary)
        except OSError:
            pass


def _cwd_marker(session_id: str) -> str:
    return f"__HERMES_CWD_{session_id}__"


def _export_dump_excluding_session_vars(
    tmp_path: str,
    excluded_names: Iterable[str] = (),
) -> str:
    """Dump exported variables without persisting per-session identities."""
    safe_names = {
        name for name in excluded_names if isinstance(name, str) and name
    }
    extra_unset = " ".join(shlex.quote(name) for name in sorted(safe_names))
    if extra_unset:
        extra_unset = f" {extra_unset}"
    return (
        "{ ( "
        "unset ${!HERMES_SESSION_*} ${!HERMES_CRON_AUTO_DELIVER_*} "
        f"HERMES_UI_SESSION_ID{extra_unset} 2>/dev/null; "
        "export -p; "
        ") || true; } "
        f"> {tmp_path}"
    )


class BaseEnvironment(ABC):
    """Common native-async interface for retained Hermes backends."""

    _stdin_mode = "pipe"
    _snapshot_timeout = 30
    _profile_scoped_passthrough = False

    def __init__(self, cwd: str, timeout: int, env: dict | None = None):
        self.cwd = cwd
        self.timeout = timeout
        self.env = env or {}
        self._session_id = uuid.uuid4().hex[:12]
        temp_dir = "/tmp"
        self._snapshot_path = f"{temp_dir}/hermes-snap-{self._session_id}.sh"
        self._cwd_file = f"{temp_dir}/hermes-cwd-{self._session_id}.txt"
        self._cwd_marker = _cwd_marker(self._session_id)
        self._snapshot_ready = False
        self._snapshot_passthrough_names: set[str] = set()
        self._prefer_nonlogin = False
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def get_temp_dir(self) -> str:  # noqa: ASYNC124 - public await API
        return "/tmp"

    def _additional_profile_scoped_passthrough_names(self) -> Iterable[str]:
        """Return backend-specific names that must not persist in snapshots."""
        return ()

    async def _snapshot_excluded_passthrough_names(self) -> tuple[str, ...]:
        if not self._profile_scoped_passthrough:
            return ()
        try:
            from agent.secret_scope import is_multiplex_active

            if is_multiplex_active():
                from tools.env_passthrough import get_all_passthrough

                names = (
                    *(await get_all_passthrough()),
                    *self._additional_profile_scoped_passthrough_names(),
                )
                self._snapshot_passthrough_names.update(
                    name
                    for name in names
                    if isinstance(name, str) and _SHELL_ENV_NAME_RE.fullmatch(name)
                )
        except Exception:
            logger.debug(
                "Could not refresh profile-scoped snapshot exclusions",
                exc_info=True,
            )
        return tuple(sorted(self._snapshot_passthrough_names))

    async def init_session(self) -> None:
        """Capture the backend login-shell state at an awaited lazy boundary."""
        temp_dir = (await self.get_temp_dir()).rstrip("/") or "/"
        self._snapshot_path = f"{temp_dir}/hermes-snap-{self._session_id}.sh"
        self._cwd_file = f"{temp_dir}/hermes-cwd-{self._session_id}.txt"
        passthrough_names = await self._snapshot_excluded_passthrough_names()
        quoted_cwd = self._quote_cwd_for_cd(self.cwd)
        quoted_snapshot = self._quote_shell_path(self._snapshot_path)
        snapshot_tmp = self._quote_shell_path(
            self._snapshot_path + ".tmp.XXXXXXXXXX"
        )
        snapshot_tmp_var = '"$__hermes_snap_tmp"'
        bootstrap = (
            "umask 077\n"
            f"__hermes_snap_tmp=$(mktemp {snapshot_tmp}) || exit 1\n"
            f"{_export_dump_excluding_session_vars(snapshot_tmp_var, passthrough_names)}\n"
            "__hermes_fns=$(declare -F | awk '{print $3}' | "
            "grep -vE '^_[^_]') || true\n"
            '[ -n "$__hermes_fns" ] && declare -f $__hermes_fns '
            f">> {snapshot_tmp_var} 2>/dev/null || true\n"
            f"alias -p >> {snapshot_tmp_var}\n"
            f"echo 'shopt -s expand_aliases' >> {snapshot_tmp_var}\n"
            f"echo 'set +e' >> {snapshot_tmp_var}\n"
            f"echo 'set +u' >> {snapshot_tmp_var}\n"
            f"mv -f {snapshot_tmp_var} {quoted_snapshot} || "
            f"rm -f {snapshot_tmp_var}\n"
            f"builtin cd -- {quoted_cwd} 2>/dev/null || true\n"
            f"printf '\\n{self._cwd_marker}%s{self._cwd_marker}\\n' "
            '"$(pwd -P)"\n'
        )
        try:
            result = await self._run_bash(
                bootstrap,
                login=True,
                timeout=self._snapshot_timeout,
            )
            if int(result.get("returncode") or 0) != 0:
                raise RuntimeError(
                    "snapshot bootstrap failed with exit code "
                    f"{result.get('returncode')}"
                )
            self._snapshot_ready = True
            self._update_cwd(result)
            logger.info(
                "Session snapshot created (session=%s, cwd=%s)",
                self._session_id,
                self.cwd,
            )
        except Exception as exc:
            self._snapshot_ready = False
            detail = str(exc)
            prefer_nonlogin = False
            try:
                probe_result = await self._run_bash(
                    "true",
                    login=False,
                    timeout=min(15, self._snapshot_timeout),
                )
                prefer_nonlogin = int(probe_result.get("returncode") or 0) == 0
                if not prefer_nonlogin:
                    detail = (
                        probe_result.get("stdout") or detail
                    ).strip() or detail
            except asyncio.CancelledError:
                raise
            except Exception as probe_exc:
                detail = f"{detail}; non-login probe: {probe_exc}"
            self._prefer_nonlogin = prefer_nonlogin
            if prefer_nonlogin:
                logger.warning(
                    "init_session failed (session=%s): %s — login bash "
                    "unusable; falling back to non-login bash -c",
                    self._session_id,
                    exc,
                )
            else:
                logger.warning(
                    "init_session failed (session=%s): %s — falling back to "
                    "bash -l per command",
                    self._session_id,
                    detail,
                )
        self._initialized = True

    async def _ensure_initialized(self) -> None:
        """Create transport resources and session state on first use."""
        if self._initialized:
            return
        async with self._init_lock:
            if not self._initialized:
                await self.init_session()

    async def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int | float = 120,  # noqa: ASYNC109 - upstream API
        stdin_data: str | None = None,
        bounded_capture: bool = False,
    ) -> dict[str, Any]:
        """Run one backend shell and return output/returncode."""
        raise NotImplementedError(f"{type(self).__name__} must implement _run_bash()")

    async def _before_execute(self) -> None:  # noqa: ASYNC124 - hook
        return None

    @staticmethod
    def _quote_cwd_for_cd(cwd: str) -> str:
        if cwd == "~":
            return cwd
        if cwd == "~/":
            return "$HOME"
        if cwd.startswith("~/"):
            return f"$HOME/{shlex.quote(cwd[2:])}"
        return shlex.quote(cwd)

    def _quote_shell_path(self, path: str) -> str:
        return shlex.quote(path)

    def _wrap_command(self, command: str, cwd: str) -> str:
        escaped = command.replace("'", "'\\''")
        snapshot = self._quote_shell_path(self._snapshot_path)
        snapshot_tmp = self._quote_shell_path(
            self._snapshot_path + ".tmp.XXXXXXXXXX"
        )
        snapshot_tmp_var = '"$__hermes_snap_tmp"'
        parts: list[str] = []

        passthrough_names = tuple(sorted(self._snapshot_passthrough_names))
        saved_names: list[tuple[str, str, str]] = []
        for name in passthrough_names:
            marker = f"_HERMES_RUNTIME_PASSTHROUGH_{name}"
            present = f"{marker}_PRESENT"
            value = f"{marker}_VALUE"
            saved_names.append((name, present, value))
            parts.append(f"{present}=${{{name}+x}}")
            parts.append(f"{value}=${{{name}-}}")

        if self._snapshot_ready:
            parts.append(f"source {snapshot} >/dev/null 2>&1 || true")

        for name, present, value in saved_names:
            parts.append(
                f'if [ "${present}" = x ]; then export {name}="${value}"; '
                f"else unset {name}; fi"
            )
            parts.append(f"unset {present} {value}")

        parts.append(f"builtin cd -- {self._quote_cwd_for_cd(cwd)} || exit 126")
        parts.append(f"eval '{escaped}'")
        parts.append("__hermes_ec=$?")
        parts.append("umask 077")
        if self._snapshot_ready:
            parts.append(
                f"__hermes_snap_tmp=$(mktemp {snapshot_tmp}) && "
                f"{{ {_export_dump_excluding_session_vars(snapshot_tmp_var, passthrough_names)} "
                "&& "
                f"mv -f {snapshot_tmp_var} {snapshot}; }} "
                f"2>/dev/null || rm -f {snapshot_tmp_var} "
                "2>/dev/null || true"
            )
        parts.append(
            f"printf '\\n{self._cwd_marker}%s{self._cwd_marker}\\n' "
            '"$(pwd -P)"'
        )
        parts.append("exit $__hermes_ec")
        return "\n".join(parts)

    @staticmethod
    def _embed_stdin_heredoc(command: str, stdin_data: str) -> str:
        delimiter = f"HERMES_STDIN_{uuid.uuid4().hex[:12]}"
        return f"{command} << '{delimiter}'\n{stdin_data}\n{delimiter}"

    @staticmethod
    async def _finalize_wait_result(
        collector: _BoundedOutputCollector,
        rendered: str,
        returncode: int | None,
    ) -> dict[str, Any]:
        """Assemble a command result and close any owned spill file."""
        result: dict[str, Any] = {
            "output": rendered,
            "returncode": returncode,
        }
        spill = await collector.close_spill()
        if spill:
            result["output_total_chars"] = collector.total_chars
            result["full_output_path"] = spill
        return result

    @staticmethod
    async def _bounded_output_collector() -> _BoundedOutputCollector:
        try:
            from tools.tool_output_limits import get_max_bytes

            capture_limit = get_max_bytes()
        except Exception:
            capture_limit = 50_000

        spill_path: Path | None = None
        try:
            spill_dir = get_hermes_home() / "cache" / "terminal-output"
            cutoff = time.time() - 7 * 86400
            if await aiofiles.os.path.isdir(spill_dir):
                for name in await aiofiles.os.listdir(spill_dir):
                    if not name.startswith("out-") or not name.endswith(".log"):
                        continue
                    old = spill_dir / name
                    try:
                        if (await aiofiles.os.stat(old)).st_mtime < cutoff:
                            await aiofiles.os.remove(old)
                    except OSError:
                        pass
            spill_path = spill_dir / (
                f"out-{time.time_ns()}-{os.getpid()}-"
                f"{id(asyncio.current_task()) & 0xffff:x}.log"
            )
        except Exception:
            spill_path = None
        return _BoundedOutputCollector(capture_limit, spill_path=spill_path)

    async def _apply_bounded_capture(
        self,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        collector = await self._bounded_output_collector()
        collector.append(str(result.get("output", "")))
        return await self._finalize_wait_result(
            collector,
            collector.render(),
            result.get("returncode"),
        )

    def _update_cwd(self, result: dict[str, Any]) -> None:
        self._extract_cwd_from_output(result)

    def _extract_cwd_from_output(self, result: dict[str, Any]) -> None:
        output = str(result.get("output", ""))
        marker = self._cwd_marker
        last = output.rfind(marker)
        if last == -1:
            return
        first = output.rfind(marker, max(0, last - 4096), last)
        if first == -1 or first == last:
            return
        cwd_path = output[first + len(marker) : last].strip()
        if cwd_path:
            self.cwd = cwd_path
        line_start = output.rfind("\n", 0, first)
        if line_start == -1:
            line_start = first
        line_end = output.find("\n", last + len(marker))
        line_end = line_end + 1 if line_end != -1 else len(output)
        result["output"] = output[:line_start] + output[line_end:]

    async def execute(
        self,
        command: str,
        cwd: str = "",
        *,
        timeout: int | None = None,  # noqa: ASYNC109 - upstream API
        stdin_data: str | None = None,
        rewrite_compound_background: bool = True,
        bounded_capture: bool = False,
    ) -> dict:
        """Execute one command using the selected backend."""
        await self._ensure_initialized()
        await self._before_execute()
        await self._snapshot_excluded_passthrough_names()
        from tools.terminal_tool import (
            _rewrite_compound_background,
        )

        prepared, sudo_stdin = await self._prepare_command(command)
        if prepared is None:
            return {"output": "Command must be a string", "returncode": 1}
        if rewrite_compound_background:
            prepared = _rewrite_compound_background(prepared)
        if sudo_stdin is not None and stdin_data is not None:
            effective_stdin = sudo_stdin + stdin_data
        elif sudo_stdin is not None:
            effective_stdin = sudo_stdin
        else:
            effective_stdin = stdin_data
        if effective_stdin and self._stdin_mode == "heredoc":
            prepared = self._embed_stdin_heredoc(prepared, effective_stdin)
            effective_stdin = None
        wrapped = self._wrap_command(prepared, cwd or self.cwd)
        result = await self._run_bash(
            wrapped,
            login=not self._snapshot_ready and not self._prefer_nonlogin,
            timeout=timeout or self.timeout,
            stdin_data=effective_stdin,
            # Foreground terminal output must be bounded while its transport
            # stream is drained.  Internal execute() callers keep the default
            # False and therefore retain full-fidelity output.
            bounded_capture=bounded_capture,
        )
        self._update_cwd(result)
        return result

    async def _prepare_command(
        self,
        command: str,
    ) -> tuple[str | None, str | None]:
        """Transform sudo commands while preserving the upstream hook name."""
        from tools.terminal_tool import _transform_sudo_command

        return await _transform_sudo_command(command)

    @abstractmethod
    async def cleanup(self) -> None:
        """Release every resource owned by this environment."""

    async def stop(self) -> None:
        """Retained cleanup alias under its upstream name."""
        await self.cleanup()
