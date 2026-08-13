"""Shared native-async Hermes execution flow for Modal transports."""

from __future__ import annotations

import asyncio
import logging
import shlex
import time
import uuid
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from tools.environments.base import BaseEnvironment, touch_activity_if_due
from tools.interrupt import is_interrupted

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedModalExec:
    """Normalized command data passed to a transport-specific exec runner."""

    command: str
    cwd: str
    timeout: int
    stdin_data: str | None = None


@dataclass(frozen=True)
class ModalExecStart:
    """Transport response after starting an exec."""

    handle: Any | None = None
    immediate_result: dict | None = None


def wrap_modal_stdin_heredoc(command: str, stdin_data: str) -> str:
    """Append stdin as a shell heredoc for transports without stdin piping."""
    marker = f"HERMES_EOF_{uuid.uuid4().hex[:8]}"
    while marker in stdin_data:
        marker = f"HERMES_EOF_{uuid.uuid4().hex[:8]}"
    return f"{command} << '{marker}'\n{stdin_data}\n{marker}"


def wrap_modal_sudo_pipe(command: str, sudo_stdin: str) -> str:
    """Feed sudo via a shell pipe for transports without direct stdin piping."""
    return f"printf '%s\\n' {shlex.quote(sudo_stdin.rstrip())} | {command}"


async def _await_owned(task: asyncio.Task[Any]) -> Any:
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised
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


class BaseModalExecutionEnvironment(BaseEnvironment):
    """Execution flow for the managed Modal transport."""

    _stdin_mode = "payload"
    _poll_interval_seconds = 0.25
    _client_timeout_grace_seconds: float | None = None
    _interrupt_output = "[Command interrupted]"
    _unexpected_error_prefix = "Modal execution error"

    async def init_session(self) -> None:
        await self._before_execute()
        self._initialized = True

    async def execute(
        self,
        command: str,
        cwd: str = "",
        *,
        timeout: int | None = None,
        stdin_data: str | None = None,
        rewrite_compound_background: bool = True,
        bounded_capture: bool = False,
    ) -> dict:
        _ = rewrite_compound_background
        _ = bounded_capture
        await self._before_execute()
        prepared = await self._prepare_modal_exec(
            command,
            cwd=cwd,
            timeout=timeout,
            stdin_data=stdin_data,
        )

        try:
            start = await self._start_modal_exec(prepared)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return self._error_result(f"{self._unexpected_error_prefix}: {exc}")

        if start.immediate_result is not None:
            return start.immediate_result
        if start.handle is None:
            return self._error_result(
                f"{self._unexpected_error_prefix}: transport did not return an exec handle"
            )

        deadline = None
        if self._client_timeout_grace_seconds is not None:
            deadline = (
                time.monotonic()
                + prepared.timeout
                + self._client_timeout_grace_seconds
            )
        now = time.monotonic()
        activity_state = {"last_touch": now, "start": now}

        try:
            while True:
                if is_interrupted():
                    try:
                        await self._cancel_modal_exec(start.handle)
                    except Exception:
                        pass
                    return self._result(self._interrupt_output, 130)

                try:
                    result = await self._poll_modal_exec(start.handle)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    return self._error_result(
                        f"{self._unexpected_error_prefix}: {exc}"
                    )
                if result is not None:
                    return result

                if deadline is not None and time.monotonic() >= deadline:
                    try:
                        await self._cancel_modal_exec(start.handle)
                    except Exception:
                        pass
                    return self._timeout_result_for_modal(prepared.timeout)

                touch_activity_if_due(activity_state, "modal command running")
                await asyncio.sleep(self._poll_interval_seconds)
        except asyncio.CancelledError as cancellation:
            cancel_task = asyncio.create_task(
                self._cancel_modal_exec(start.handle),
                name="modal-cancel-exec",
            )
            try:
                await _await_owned(cancel_task)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "Modal exec cancel failed after cancellation",
                    exc_info=True,
                )
            raise cancellation

    async def _before_execute(self) -> None:
        """Hook for transports that require lazy setup or pre-exec sync."""

    async def _prepare_modal_exec(
        self,
        command: str,
        *,
        cwd: str = "",
        timeout: int | None = None,
        stdin_data: str | None = None,
    ) -> PreparedModalExec:
        effective_cwd = cwd or self.cwd
        effective_timeout = timeout or self.timeout
        exec_command = command
        exec_stdin = stdin_data if self._stdin_mode == "payload" else None
        if stdin_data is not None and self._stdin_mode == "heredoc":
            exec_command = wrap_modal_stdin_heredoc(exec_command, stdin_data)

        from tools.environments.local import _transform_sudo_command

        exec_command, sudo_stdin = await _transform_sudo_command(exec_command)
        if exec_command is None:
            exec_command = ""
        if sudo_stdin is not None:
            exec_command = wrap_modal_sudo_pipe(exec_command, sudo_stdin)
        return PreparedModalExec(
            command=exec_command,
            cwd=effective_cwd,
            timeout=effective_timeout,
            stdin_data=exec_stdin,
        )

    @staticmethod
    def _result(output: str, returncode: int) -> dict:
        return {"output": output, "returncode": returncode}

    def _error_result(self, output: str) -> dict:
        return self._result(output, 1)

    def _timeout_result_for_modal(self, timeout: int) -> dict:
        return self._result(f"Command timed out after {timeout}s", 124)

    @abstractmethod
    async def _start_modal_exec(self, prepared: PreparedModalExec) -> ModalExecStart:
        """Begin a transport-specific exec."""

    @abstractmethod
    async def _poll_modal_exec(self, handle: Any) -> dict | None:
        """Return a final result dict when complete, else ``None``."""

    @abstractmethod
    async def _cancel_modal_exec(self, handle: Any) -> None:
        """Cancel or terminate the active transport exec."""
