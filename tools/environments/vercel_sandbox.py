"""Vercel Sandbox execution environment."""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
import shlex
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import aiofiles
import httpx  # noqa: ASYNC127 - retained project/SDK transport dependency

from hermes_constants import get_hermes_home
from tools.environments.base import (
    BaseEnvironment,
    _BoundedOutputCollector,
    _UNBOUNDED_CAPTURE_CHARS,
    _load_json_store,
    _save_json_store,
    touch_activity_if_due,
)
from tools.environments.file_sync import (
    FileSyncManager,
    iter_sync_files,
    quoted_rm_command,
)


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vercel.sandbox import AsyncSandbox, Resources, SandboxStatus, WriteFile


DEFAULT_VERCEL_CWD = "/vercel/sandbox"
_DEFAULT_CONTAINER_DISK_MB = 51200
_CREATE_RETRY_ATTEMPTS = 3
_WRITE_RETRY_ATTEMPTS = 3
_TRANSIENT_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
_RETRY_BACKOFF_STEP = timedelta(milliseconds=100)
_MIN_SANDBOX_TIMEOUT = timedelta(minutes=5)
_MIN_RUNNING_WAIT = timedelta(seconds=1)
_RUNNING_WAIT_TIMEOUT = timedelta(seconds=30)
_RUNNING_WAIT_POLL_INTERVAL = timedelta(milliseconds=250)
_STOP_TIMEOUT = timedelta(seconds=15)
_STOP_POLL_INTERVAL = timedelta(milliseconds=500)
_CLIENT_CLOSE_TIMEOUT_SECONDS = 15.0
_SNAPSHOT_STORE_NAME = "vercel_sandbox_snapshots.json"

_T = TypeVar("_T")


def _ensure_vercel_sdk() -> None:
    """Require the pinned SDK without performing an implicit runtime install."""
    os.environ.setdefault("VERCEL_TELEMETRY_DISABLED", "1")
    try:
        from vercel.sandbox import AsyncSandbox as _AsyncSandbox  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Vercel Sandbox requires the 'vercel==0.7.2' package"
        ) from exc


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return chain


def _extract_status_code(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    for value in (
        getattr(exc, "status_code", None),
        getattr(response, "status_code", None),
    ):
        if isinstance(value, int):
            return value
    return None


def _is_transient_vercel_error(exc: BaseException) -> bool:
    for error in _exception_chain(exc):
        status_code = _extract_status_code(error)
        if status_code in _TRANSIENT_STATUS_CODES:
            return True
        if isinstance(
            error,
            (httpx.NetworkError, httpx.ProtocolError, httpx.ReadError),
        ):
            return True
        error_name = type(error).__name__.lower()
        if "ratelimit" in error_name or "servererror" in error_name:
            return True
    return False


async def _retry_vercel_call(
    label: str,
    callback: Callable[[], Awaitable[_T]],
    *,
    attempts: int,
) -> _T:
    backoff_seconds = _RETRY_BACKOFF_STEP.total_seconds()
    for attempt in range(1, attempts + 1):
        try:
            return await callback()
        except Exception as exc:
            if attempt >= attempts or not _is_transient_vercel_error(exc):
                raise
            logger.warning(
                "Vercel: %s failed (%s); retrying %d/%d",
                label,
                exc,
                attempt,
                attempts,
            )
            await asyncio.sleep(backoff_seconds * attempt)
    raise AssertionError("unreachable")


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


async def _extract_result_output(result: Any) -> str:
    try:
        output = result.output()
    except (AttributeError, TypeError):
        return _coerce_text(result)  # noqa: ASYNC910 - pure fallback
    if not inspect.isawaitable(output):
        raise RuntimeError("Vercel AsyncSandbox returned a synchronous command result")
    return _coerce_text(await output)


def _extract_result_returncode(result: Any) -> int:
    try:
        exit_code = result.exit_code
    except AttributeError:
        try:
            exit_code = result.returncode
        except AttributeError:
            return 1
    return exit_code if isinstance(exit_code, int) else 1


def _snapshot_store_path() -> Path:
    return get_hermes_home() / _SNAPSHOT_STORE_NAME


async def _load_snapshots() -> dict:
    return await _load_json_store(_snapshot_store_path())


async def _save_snapshots(data: dict) -> None:
    await _save_json_store(_snapshot_store_path(), data)


async def _get_snapshot_id(task_id: str) -> str | None:
    if not task_id:
        return None  # noqa: ASYNC910 - empty task id is a state-only no-op
    snapshot_id = (await _load_snapshots()).get(task_id)
    return snapshot_id if isinstance(snapshot_id, str) and snapshot_id else None


async def _store_snapshot(task_id: str, snapshot_id: str) -> None:
    if not task_id or not snapshot_id:
        return  # noqa: ASYNC910 - invalid metadata is a state-only no-op
    snapshots = await _load_snapshots()
    snapshots[task_id] = snapshot_id
    await _save_snapshots(snapshots)


async def _delete_snapshot(
    task_id: str,
    snapshot_id: str | None = None,
) -> None:
    if not task_id:
        return  # noqa: ASYNC910 - empty task id is a state-only no-op
    snapshots = await _load_snapshots()
    existing = snapshots.get(task_id)
    if existing is None:
        return
    if snapshot_id is not None and existing != snapshot_id:
        return
    snapshots.pop(task_id, None)
    await _save_snapshots(snapshots)


def _extract_snapshot_id(snapshot: Any) -> str | None:
    for attr in ("snapshot_id", "snapshotId", "id"):
        value = getattr(snapshot, attr, None)
        if isinstance(value, str) and value:
            return value
    if isinstance(snapshot, dict):
        for key in ("snapshot_id", "snapshotId", "id"):
            value = snapshot.get(key)
            if isinstance(value, str) and value:
                return value
    return None


@cache
def _sandbox_status_type() -> type[SandboxStatus]:
    _ensure_vercel_sdk()
    from vercel.sandbox import SandboxStatus

    return SandboxStatus


@cache
def _terminal_sandbox_states() -> frozenset[SandboxStatus]:
    SandboxStatus = _sandbox_status_type()
    return frozenset(
        {
            SandboxStatus.ABORTED,
            SandboxStatus.FAILED,
            SandboxStatus.STOPPED,
        }
    )


@dataclass(frozen=True, slots=True)
class _SandboxCreateParams:
    timeout: timedelta
    runtime: str | None = None
    resources: Resources | None = None


async def _await_owned(task: asyncio.Task[_T]) -> _T:
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
    return result  # noqa: ASYNC910 - shield above is the checkpoint


class VercelSandboxEnvironment(BaseEnvironment):
    """Vercel cloud sandbox backend."""

    _stdin_mode = "heredoc"

    def __init__(
        self,
        runtime: str | None = None,
        cwd: str = DEFAULT_VERCEL_CWD,
        timeout: int = 60,
        cpu: float = 1,
        memory: int = 5120,
        disk: int = _DEFAULT_CONTAINER_DISK_MB,
        persistent_filesystem: bool = True,
        task_id: str = "default",
    ):
        if disk not in {0, _DEFAULT_CONTAINER_DISK_MB}:
            raise ValueError(
                "Vercel Sandbox does not support configurable container_disk. "
                "Use the default shared setting."
            )
        super().__init__(cwd=cwd, timeout=timeout)
        self._runtime = runtime or None
        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._requested_cwd = cwd
        self._cpu = cpu
        self._memory = memory
        self._disk = disk
        self._lock = asyncio.Lock()
        self._sandbox: AsyncSandbox | None = None
        self._workspace_root = DEFAULT_VERCEL_CWD
        self._remote_home = DEFAULT_VERCEL_CWD
        self._sync_manager: FileSyncManager | None = None
        self._create_params: _SandboxCreateParams | None = None

    def _build_create_params(
        self,
        *,
        cpu: float,
        memory: int,
        disk: int,
    ) -> _SandboxCreateParams:
        if disk not in {0, _DEFAULT_CONTAINER_DISK_MB}:
            raise ValueError(
                "Vercel Sandbox does not support configurable container_disk. "
                "Use the default shared setting."
            )
        _ensure_vercel_sdk()
        from vercel.sandbox import Resources

        sandbox_timeout = max(
            timedelta(seconds=max(self.timeout, 0)),
            _MIN_SANDBOX_TIMEOUT,
        )
        vcpus = math.floor(cpu) if cpu > 0 else None
        memory_mb = memory if memory > 0 else None
        resources = (
            Resources(vcpus=vcpus, memory=memory_mb)
            if vcpus is not None or memory_mb is not None
            else None
        )
        return _SandboxCreateParams(
            timeout=sandbox_timeout,
            runtime=self._runtime,
            resources=resources,
        )

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return  # noqa: ASYNC910 - initialized fast path is state-only
        async with self._lock:
            if self._initialized:
                return  # noqa: ASYNC910 - lock peer initialized transport
            _ensure_vercel_sdk()
            self._create_params = self._build_create_params(
                cpu=self._cpu,
                memory=self._memory,
                disk=self._disk,
            )
            try:
                self._sandbox = await self._create_sandbox()
                await self._configure_attached_sandbox(
                    requested_cwd=self._requested_cwd
                )
                assert self._sync_manager is not None
                await self._sync_manager.sync(force=True)
                await self.init_session()
            except BaseException:
                sandbox = self._sandbox
                self._sandbox = None
                self._sync_manager = None
                if sandbox is not None:
                    cleanup = asyncio.create_task(
                        self._dispose_sandbox(sandbox, snapshot=False)
                    )
                    await _await_owned(  # noqa: ASYNC120 - cancellation wins
                        cleanup
                    )
                raise

    async def _create_sandbox(self) -> AsyncSandbox:
        _ensure_vercel_sdk()
        from vercel.sandbox import AsyncSandbox

        params = self._create_params
        if params is None:
            raise RuntimeError("Vercel sandbox create parameters are not initialized")
        snapshot_id = (
            await _get_snapshot_id(self._task_id) if self._persistent else None
        )
        if snapshot_id:
            try:
                return await _retry_vercel_call(
                    "sandbox restore",
                    lambda: AsyncSandbox.create(
                        timeout=params.timeout,
                        runtime=params.runtime,
                        resources=params.resources,
                        source={"type": "snapshot", "snapshot_id": snapshot_id},
                    ),
                    attempts=_CREATE_RETRY_ATTEMPTS,
                )
            except Exception as exc:
                logger.warning(
                    "Vercel: failed to restore snapshot %s for task %s; "
                    "falling back to a fresh sandbox: %s",
                    snapshot_id,
                    self._task_id,
                    exc,
                )
                await _delete_snapshot(self._task_id, snapshot_id)

        return await _retry_vercel_call(
            "sandbox create",
            lambda: AsyncSandbox.create(
                timeout=params.timeout,
                runtime=params.runtime,
                resources=params.resources,
            ),
            attempts=_CREATE_RETRY_ATTEMPTS,
        )

    async def _configure_attached_sandbox(self, *, requested_cwd: str) -> None:
        await self._wait_for_running()
        self._workspace_root = self._detect_workspace_root()
        self._remote_home = await self._detect_remote_home()
        container_base = (
            "/.hermes"
            if self._remote_home == "/"
            else f"{self._remote_home.rstrip('/')}/.hermes"
        )
        self._sync_manager = FileSyncManager(
            get_files_fn=lambda: iter_sync_files(container_base),
            upload_fn=self._vercel_upload,
            delete_fn=self._vercel_delete,
            bulk_upload_fn=self._vercel_bulk_upload,
            bulk_download_fn=self._vercel_bulk_download,
        )
        if requested_cwd == "~":
            self.cwd = self._remote_home
        elif requested_cwd in {"", DEFAULT_VERCEL_CWD}:
            self.cwd = self._workspace_root
        else:
            self.cwd = requested_cwd

    def _detect_workspace_root(self) -> str:
        sandbox = self._sandbox
        if sandbox is None:
            raise RuntimeError("Vercel sandbox is not attached")
        cwd = sandbox.sandbox.cwd
        return cwd if cwd.startswith("/") else DEFAULT_VERCEL_CWD

    async def _detect_remote_home(self) -> str:
        sandbox = self._sandbox
        if sandbox is None:
            raise RuntimeError("Vercel sandbox is not attached")
        try:
            result = await sandbox.run_command(
                "sh",
                ["-lc", 'printf %s "$HOME"'],
                cwd=self._workspace_root,
            )
            home = (await _extract_result_output(result)).strip()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(
                "Vercel: home detection failed for task %s: %s",
                self._task_id,
                exc,
            )
            return self._workspace_root  # noqa: ASYNC910 - command was awaited
        return home if home.startswith("/") else self._workspace_root

    async def _wait_for_running(
        self,
        timeout: timedelta = _RUNNING_WAIT_TIMEOUT,
    ) -> None:
        sandbox = self._sandbox
        if sandbox is None:
            raise RuntimeError("Vercel sandbox is not attached")
        SandboxStatus = _sandbox_status_type()
        status = sandbox.status
        if status is None or status == SandboxStatus.RUNNING:
            return  # noqa: ASYNC910 - status is already terminal for this wait
        if status in _terminal_sandbox_states():
            raise RuntimeError(f"Sandbox entered terminal state: {status}")
        try:
            await sandbox.wait_for_status(
                SandboxStatus.RUNNING,
                timeout=max(timeout, _MIN_RUNNING_WAIT),
                poll_interval=_RUNNING_WAIT_POLL_INTERVAL,
            )
        except TimeoutError as exc:
            status = sandbox.status
            if status in _terminal_sandbox_states():
                raise RuntimeError(
                    f"Sandbox entered terminal state: {status}"
                ) from exc
            raise RuntimeError(
                f"Sandbox did not reach running state (last status: {status})"
            ) from exc

    async def _close_sandbox_client(  # noqa: ASYNC910 - aclose is the checkpoint
        self,
        sandbox: AsyncSandbox | None,
    ) -> None:
        if sandbox is None:
            return  # noqa: ASYNC910 - empty transport is a true no-op
        try:
            await asyncio.wait_for(
                sandbox.client.aclose(),
                timeout=_CLIENT_CLOSE_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    async def _stop_sandbox(  # noqa: ASYNC910 - stop is the checkpoint
        self,
        sandbox: AsyncSandbox | None,
    ) -> None:
        if sandbox is None:
            return  # noqa: ASYNC910 - empty transport is a true no-op
        try:
            await asyncio.wait_for(
                sandbox.stop(
                    blocking=True,
                    timeout=_STOP_TIMEOUT,
                    poll_interval=_STOP_POLL_INTERVAL,
                ),
                timeout=_STOP_TIMEOUT.total_seconds() + 1,
            )
        except TypeError:
            try:
                await asyncio.wait_for(
                    sandbox.stop(),
                    timeout=_STOP_TIMEOUT.total_seconds() + 1,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    async def _snapshot_sandbox(self, sandbox: AsyncSandbox) -> str | None:
        if not self._persistent or not self._task_id:
            return None  # noqa: ASYNC910 - snapshot policy is state-only
        try:
            snapshot = await sandbox.snapshot()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Vercel: filesystem snapshot failed for task %s: %s",
                self._task_id,
                exc,
            )
            return None  # noqa: ASYNC910 - snapshot call was awaited
        snapshot_id = _extract_snapshot_id(snapshot)
        if not snapshot_id:
            logger.warning(
                "Vercel: filesystem snapshot for task %s did not return a snapshot id",
                self._task_id,
            )
            return None
        await _store_snapshot(self._task_id, snapshot_id)
        logger.info(
            "Vercel: saved filesystem snapshot %s for task %s",
            snapshot_id,
            self._task_id,
        )
        return snapshot_id

    async def _ensure_sandbox_ready(self) -> None:
        sandbox = self._sandbox
        requested_cwd = self.cwd or self._requested_cwd or DEFAULT_VERCEL_CWD
        if sandbox is None:
            self._sandbox = await self._create_sandbox()
            await self._configure_attached_sandbox(requested_cwd=requested_cwd)
            return
        try:
            await sandbox.refresh()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Vercel: sandbox refresh failed for task %s: %s; recreating",
                self._task_id,
                exc,
            )
            await self._close_sandbox_client(sandbox)
            self._sandbox = await self._create_sandbox()
            await self._configure_attached_sandbox(requested_cwd=requested_cwd)
            return
        status = sandbox.status
        if status in _terminal_sandbox_states():
            logger.warning(
                "Vercel: sandbox entered state %s for task %s; recreating",
                status,
                self._task_id,
            )
            await self._close_sandbox_client(sandbox)
            self._sandbox = await self._create_sandbox()
            await self._configure_attached_sandbox(requested_cwd=requested_cwd)
            return
        await self._wait_for_running()

    async def _vercel_upload(self, host_path: str, remote_path: str) -> None:
        await self._vercel_bulk_upload([(host_path, remote_path)])

    async def _vercel_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        if not files:
            return  # noqa: ASYNC910 - retained empty-input no-op
        payload: list[WriteFile] = []
        for host_path, remote_path in files:
            async with aiofiles.open(host_path, "rb") as handle:
                payload.append(
                    {"path": remote_path, "content": await handle.read()}
                )
        sandbox = self._sandbox
        if sandbox is None:
            raise RuntimeError("Vercel sandbox is not attached")
        await _retry_vercel_call(
            "write_files",
            lambda: sandbox.write_files(payload),
            attempts=_WRITE_RETRY_ATTEMPTS,
        )

    async def _vercel_delete(self, remote_paths: list[str]) -> None:
        if not remote_paths:
            return  # noqa: ASYNC910 - retained empty-input no-op
        sandbox = self._sandbox
        if sandbox is None:
            raise RuntimeError("Vercel sandbox is not attached")
        result = await sandbox.run_command(
            "bash",
            ["-lc", quoted_rm_command(remote_paths)],
            cwd=self._workspace_root,
        )
        if _extract_result_returncode(result) != 0:
            raise RuntimeError(
                "Vercel delete failed: "
                f"{(await _extract_result_output(result)).strip()}"
            )

    async def _vercel_bulk_download(self, dest_tar_path: Path) -> None:
        remote_hermes = (
            "/.hermes"
            if self._remote_home == "/"
            else f"{self._remote_home.rstrip('/')}/.hermes"
        )
        archive_member = remote_hermes.lstrip("/")
        remote_tar = f"/tmp/.hermes_sync.{os.getpid()}.tar"
        sandbox = self._sandbox
        if sandbox is None:
            raise RuntimeError("Vercel sandbox is not attached")
        try:
            result = await sandbox.run_command(
                "bash",
                [
                    "-lc",
                    f"tar cf {shlex.quote(remote_tar)} -C / "
                    f"{shlex.quote(archive_member)}",
                ],
                cwd=self._workspace_root,
            )
            if _extract_result_returncode(result) != 0:
                raise RuntimeError(
                    "Vercel bulk download failed: "
                    f"{(await _extract_result_output(result)).strip()}"
                )
            await sandbox.download_file(remote_tar, dest_tar_path)
        finally:
            cleanup = asyncio.create_task(
                self._remove_remote_tar(sandbox, remote_tar)
            )
            await _await_owned(  # noqa: ASYNC120 - cleanup precedes cancellation
                cleanup
            )

    async def _remove_remote_tar(  # noqa: ASYNC910 - run_command is checkpoint
        self,
        sandbox: AsyncSandbox,
        remote_tar: str,
    ) -> None:
        try:
            await sandbox.run_command(
                "bash",
                ["-lc", f"rm -f {shlex.quote(remote_tar)}"],
                cwd=self._workspace_root,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    async def _before_execute(self) -> None:
        async with self._lock:
            await self._ensure_sandbox_ready()
            if self._sync_manager is not None:
                await self._sync_manager.sync()

    async def _cancel_command(
        self,
        command_task: asyncio.Task[Any],
        sandbox: AsyncSandbox,
    ) -> None:
        command_task.cancel()
        await asyncio.gather(command_task, return_exceptions=True)
        await self._stop_sandbox(sandbox)

    async def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int | float = 120,
        stdin_data: str | None = None,
        bounded_capture: bool = False,
    ) -> dict[str, Any]:
        del stdin_data
        sandbox = self._sandbox
        if sandbox is None:
            raise RuntimeError("Vercel sandbox is not attached")
        command_task = asyncio.create_task(
            sandbox.run_command(
                "bash",
                ["-lc" if login else "-c", cmd_string],
                cwd=self._workspace_root,
            )
        )
        started = time.monotonic()
        activity_state = {"last_touch": started, "start": started}
        try:
            while not command_task.done():
                from tools.interrupt import is_interrupted

                if is_interrupted():
                    cancel = asyncio.create_task(
                        self._cancel_command(command_task, sandbox)
                    )
                    await _await_owned(cancel)
                    return {"output": "[Command interrupted]", "returncode": 130}
                remaining = float(timeout) - (time.monotonic() - started)
                if remaining <= 0:
                    cancel = asyncio.create_task(
                        self._cancel_command(command_task, sandbox)
                    )
                    await _await_owned(cancel)
                    return {
                        "output": f"[Command timed out after {timeout}s]",
                        "returncode": 124,
                    }
                await asyncio.wait(
                    {command_task},
                    timeout=min(0.25, remaining),
                )
                touch_activity_if_due(activity_state, "terminal command running")
            try:
                result = await command_task
                rendered = await _extract_result_output(result)
            except asyncio.CancelledError:
                raise
            except Exception:
                return {  # noqa: ASYNC910 - command/result was awaited
                    "output": "",
                    "returncode": 1,
                }
        except asyncio.CancelledError as cancellation:
            cancel = asyncio.create_task(
                self._cancel_command(command_task, sandbox)
            )
            await _await_owned(cancel)
            raise cancellation

        collector = (
            await self._bounded_output_collector()
            if bounded_capture
            else _BoundedOutputCollector(_UNBOUNDED_CAPTURE_CHARS)
        )
        collector.append(rendered)
        return await self._finalize_wait_result(
            collector,
            collector.render(),
            _extract_result_returncode(result),
        )

    async def cleanup(self) -> None:
        cleanup_task = asyncio.create_task(self._cleanup_owned())
        await _await_owned(cleanup_task)

    async def _cleanup_owned(self) -> None:
        async with self._lock:
            sandbox = self._sandbox
            sync_manager = self._sync_manager
            if sandbox is not None and sync_manager is not None:
                try:
                    await sync_manager.sync_back()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "Vercel: sync_back failed for task %s: %s",
                        self._task_id,
                        exc,
                    )
            self._sandbox = None
            self._sync_manager = None
        if sandbox is None:
            return
        await self._dispose_sandbox(sandbox, snapshot=True)

    async def _dispose_sandbox(
        self,
        sandbox: AsyncSandbox,
        *,
        snapshot: bool,
    ) -> None:
        try:
            if snapshot:
                await self._snapshot_sandbox(sandbox)
        finally:
            try:
                await self._stop_sandbox(sandbox)
            finally:
                await self._close_sandbox_client(sandbox)
