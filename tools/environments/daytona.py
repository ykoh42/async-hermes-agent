"""Daytona cloud execution environment.

Uses the Daytona Python SDK to run commands in cloud sandboxes.
Supports persistent sandboxes: when enabled, sandboxes are stopped on cleanup
and resumed on next creation, preserving the filesystem across sessions.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import shlex
from pathlib import Path
from typing import Any

import aiofiles

from tools.environments.base import BaseEnvironment, touch_activity_if_due
from tools.environments.file_sync import (
    FileSyncManager,
    _await_owned,
    iter_sync_files,
    quoted_mkdir_command,
    quoted_rm_command,
    unique_parent_dirs,
)

logger = logging.getLogger(__name__)

_DEFAULT_DAYTONA_API_URL = "https://app.daytona.io/api"
# Daytona 0.155.0 reads local .env files synchronously unless config.target is
# truthy. Pass a temporary truthy value, then restore None before any request,
# preserving the SDK's default-region behavior without blocking the loop.
_DEFAULT_TARGET_SENTINEL = "__hermes_daytona_default_target__"


async def _read_daytona_setting(name: str) -> str:
    """Read a Daytona setting through Hermes' native-async dotenv boundary."""
    try:
        from hermes_cli.config import get_env_value_prefer_dotenv
    except ImportError:
        return str(os.environ.get(name) or "").strip()
    value = await get_env_value_prefer_dotenv(name)
    return str(value or "").strip()


async def _new_daytona_client():
    """Construct AsyncDaytona without invoking its synchronous .env reader."""
    from daytona import (  # type: ignore[import-not-found]
        AsyncDaytona,
        DaytonaConfig,
        DaytonaError,
    )

    api_key = await _read_daytona_setting("DAYTONA_API_KEY")
    jwt_token = await _read_daytona_setting("DAYTONA_JWT_TOKEN")
    organization_id = await _read_daytona_setting("DAYTONA_ORGANIZATION_ID")
    api_url = (
        await _read_daytona_setting("DAYTONA_API_URL")
        or await _read_daytona_setting("DAYTONA_SERVER_URL")
        or _DEFAULT_DAYTONA_API_URL
    )
    target = await _read_daytona_setting("DAYTONA_TARGET")

    if not api_key and not jwt_token:
        raise DaytonaError("API key or JWT token is required")
    if jwt_token and not api_key and not organization_id:
        raise DaytonaError("Organization ID is required when using JWT token")

    client = AsyncDaytona(
        DaytonaConfig(
            api_key=api_key or None,
            jwt_token=jwt_token or None,
            organization_id=organization_id or None,
            api_url=api_url,
            target=target or _DEFAULT_TARGET_SENTINEL,
        )
    )
    if not target:
        # Reset the constructor-only sentinel before create() serializes a
        # request. Snapshot service stores the same target for image-context
        # uploads, so keep it aligned as well.
        client._target = None
        snapshot = getattr(client, "snapshot", None)
        if snapshot is not None and hasattr(snapshot, "_target"):
            snapshot._target = None
    return client


class DaytonaEnvironment(BaseEnvironment):
    """Daytona cloud sandbox execution backend."""

    _stdin_mode = "heredoc"

    def __init__(
        self,
        image: str,
        cwd: str = "/home/daytona",
        timeout: int = 60,
        cpu: int = 1,
        memory: int = 5120,
        disk: int = 10240,
        persistent_filesystem: bool = True,
        task_id: str = "default",
    ):
        super().__init__(cwd=cwd, timeout=timeout)
        self._requested_cwd = cwd
        self._image = image
        self._cpu = cpu
        self._memory = memory
        self._disk = disk
        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._daytona: Any = None
        self._sandbox: Any = None
        self._SandboxState: Any = None
        self._FileUpload: Any = None
        self._sync_manager: FileSyncManager | None = None
        self._remote_home = "/root"
        self._initialize_lock = asyncio.Lock()
        self._operation_lock = asyncio.Lock()
        self._sandbox_created = False

    async def init_session(self) -> None:
        """Lazily create/resume the sandbox and capture its shell snapshot."""
        await self._ensure_initialized()

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._initialize_lock:
            if self._initialized:
                return
            initialization_error: Exception | None = None
            try:
                await self._initialize_sandbox()
                await BaseEnvironment.init_session(self)
            except asyncio.CancelledError:
                await self._cleanup_failed_initialization()  # noqa: ASYNC120
                raise
            except Exception as exc:
                initialization_error = exc
            if initialization_error is not None:
                await self._cleanup_failed_initialization()
                raise initialization_error

    async def _cleanup_failed_initialization(self) -> None:
        """Release SDK resources acquired before lazy initialization failed."""

        async def _cleanup() -> None:
            sandbox = self._sandbox
            client = self._daytona
            try:
                if sandbox is not None:
                    try:
                        if self._sandbox_created:
                            await client.delete(sandbox)
                        else:
                            await sandbox.stop()
                    except Exception as exc:
                        logger.warning(
                            "Daytona: failed to clean up incomplete sandbox: %s",
                            exc,
                        )
            finally:
                self._sandbox = None
                if client is not None:
                    try:
                        await client.close()
                    except Exception as exc:
                        logger.warning("Daytona: client close failed: %s", exc)
                self._daytona = None

        await _await_owned(asyncio.create_task(_cleanup()))

    async def _initialize_sandbox(self) -> None:
        try:
            from daytona import (  # type: ignore[import-not-found]
                CreateSandboxFromImageParams,
                DaytonaError,
                Resources,
                SandboxState,
            )
            from daytona.common.filesystem import (  # type: ignore[import-not-found]
                FileUpload,
            )
        except ImportError as exc:
            raise ImportError(
                "Daytona backend requires the 'daytona' package; install "
                "async-hermes-agent[daytona]."
            ) from exc

        self._SandboxState = SandboxState
        self._FileUpload = FileUpload
        self._daytona = await _new_daytona_client()

        memory_gib = max(1, math.ceil(self._memory / 1024))
        disk_gib = max(1, math.ceil(self._disk / 1024))
        if disk_gib > 10:
            logger.warning(
                "Daytona: requested disk (%dGB) exceeds platform limit (10GB). "
                "Capping to 10GB.",
                disk_gib,
            )
            disk_gib = 10
        resources = Resources(cpu=self._cpu, memory=memory_gib, disk=disk_gib)
        labels = {"hermes_task_id": self._task_id}
        sandbox_name = f"hermes-{self._task_id}"

        if self._persistent:
            try:
                self._sandbox = await self._daytona.get(sandbox_name)
                await self._sandbox.start()
                logger.info(
                    "Daytona: resumed sandbox %s for task %s",
                    self._sandbox.id,
                    self._task_id,
                )
            except DaytonaError:
                self._sandbox = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Daytona: failed to resume sandbox for task %s: %s",
                    self._task_id,
                    exc,
                )
                self._sandbox = None

            if self._sandbox is None:
                try:
                    results = await self._daytona.list(labels=labels, limit=1)
                    items = list(getattr(results, "items", ()) or ())
                    legacy = items[0] if items else None
                    if legacy is not None:
                        self._sandbox = legacy
                        await self._sandbox.start()
                        logger.info(
                            "Daytona: resumed legacy sandbox %s for task %s",
                            self._sandbox.id,
                            self._task_id,
                        )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.debug(
                        "Daytona: no legacy sandbox found for task %s: %s",
                        self._task_id,
                        exc,
                    )
                    self._sandbox = None

        if self._sandbox is None:
            self._sandbox = await self._daytona.create(
                CreateSandboxFromImageParams(
                    image=self._image,
                    name=sandbox_name,
                    labels=labels,
                    auto_stop_interval=0,
                    resources=resources,
                )
            )
            self._sandbox_created = True
            logger.info(
                "Daytona: created sandbox %s for task %s",
                self._sandbox.id,
                self._task_id,
            )

        try:
            home_response = await self._sandbox.process.exec("echo $HOME")
            home = str(home_response.result or "").strip()
            if home:
                self._remote_home = home
                if self._requested_cwd in {"~", "/home/daytona"}:
                    self.cwd = home
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        logger.info(
            "Daytona: resolved home to %s, cwd to %s",
            self._remote_home,
            self.cwd,
        )

        self._sync_manager = FileSyncManager(
            get_files_fn=lambda: iter_sync_files(f"{self._remote_home}/.hermes"),
            upload_fn=self._daytona_upload,
            delete_fn=self._daytona_delete,
            bulk_upload_fn=self._daytona_bulk_upload,
            bulk_download_fn=self._daytona_bulk_download,
        )
        await self._sync_manager.sync(force=True)

    def _require_sandbox(self):
        if self._sandbox is None:
            raise RuntimeError("Daytona sandbox is not initialized")
        return self._sandbox

    async def _daytona_upload(self, host_path: str, remote_path: str) -> None:
        """Upload a single file via Daytona's native async bytes API."""
        sandbox = self._require_sandbox()
        parent = str(Path(remote_path).parent)
        await sandbox.process.exec(quoted_mkdir_command([parent]))
        async with aiofiles.open(host_path, "rb") as handle:
            content = await handle.read()
        await sandbox.fs.upload_file(content, remote_path)

    async def _daytona_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        """Upload many files in one native-async Daytona SDK call."""
        if not files:
            return
        sandbox = self._require_sandbox()
        parents = unique_parent_dirs(files)
        if parents:
            await sandbox.process.exec(quoted_mkdir_command(parents))

        uploads = []
        for host_path, remote_path in files:
            async with aiofiles.open(host_path, "rb") as handle:
                content = await handle.read()
            uploads.append(
                self._FileUpload(source=content, destination=remote_path)
            )
        await sandbox.fs.upload_files(uploads)

    async def _daytona_bulk_download(self, dest: Path) -> None:
        """Download remote .hermes/ as a tar archive without sync file I/O."""
        sandbox = self._require_sandbox()
        rel_base = f"{self._remote_home}/.hermes".lstrip("/")
        remote_tar = f"/tmp/.hermes_sync.{os.getpid()}.tar"
        await sandbox.process.exec(
            f"tar cf {shlex.quote(remote_tar)} -C / {shlex.quote(rel_base)}"
        )
        try:
            content = await sandbox.fs.download_file(remote_tar)
            async with aiofiles.open(dest, "wb") as handle:
                await handle.write(bytes(content or b""))
        finally:
            async def _remove_remote_tar() -> None:
                try:
                    await sandbox.process.exec(
                        f"rm -f {shlex.quote(remote_tar)}"
                    )
                except Exception:
                    pass

            await _await_owned(asyncio.create_task(_remove_remote_tar()))

    async def _daytona_delete(self, remote_paths: list[str]) -> None:
        """Batch-delete remote files via SDK exec."""
        await self._require_sandbox().process.exec(
            quoted_rm_command(remote_paths)
        )

    async def _ensure_sandbox_ready(self) -> None:
        """Restart sandbox if it was stopped by a previous interrupt."""
        sandbox = self._require_sandbox()
        await sandbox.refresh_data()
        if sandbox.state in {
            self._SandboxState.STOPPED,
            self._SandboxState.ARCHIVED,
        }:
            await sandbox.start()
            logger.info("Daytona: restarted sandbox %s", sandbox.id)

    async def _before_execute(self) -> None:
        """Ensure sandbox readiness and synchronize changed files."""
        async with self._operation_lock:
            await self._ensure_sandbox_ready()
        if self._sync_manager is not None:
            await self._sync_manager.sync()

    async def _stop_after_interruption(self, sandbox: Any) -> None:
        async def _stop() -> None:
            try:
                await sandbox.stop()
            except Exception:
                pass

        await _await_owned(asyncio.create_task(_stop()))

    async def _run_bash(
        self,
        command: str,
        *,
        login: bool = False,
        timeout: int | float = 120,
        stdin_data: str | None = None,
        bounded_capture: bool = False,
    ) -> dict[str, Any]:
        """Execute one shell command through Daytona's native async process API."""
        sandbox = self._require_sandbox()
        if login:
            shell_cmd = f"bash -l -c {shlex.quote(command)}"
        else:
            shell_cmd = f"bash -c {shlex.quote(command)}"

        command_task = asyncio.create_task(
            asyncio.wait_for(
                sandbox.process.exec(shell_cmd, timeout=timeout),
                timeout=float(timeout),
            )
        )

        async def cancel_command() -> None:
            command_task.cancel()
            await asyncio.gather(command_task, return_exceptions=True)

        async def cancel_command_and_stop() -> None:
            try:
                await cancel_command()
            finally:
                await self._stop_after_interruption(sandbox)

        started = asyncio.get_running_loop().time()
        activity_state = {"last_touch": started, "start": started}
        try:
            await asyncio.sleep(0)
            while not command_task.done():
                from tools.interrupt import is_interrupted

                if is_interrupted():
                    cleanup = asyncio.create_task(cancel_command_and_stop())
                    await _await_owned(cleanup)
                    return {
                        "output": "\n[Command interrupted]",
                        "returncode": 130,
                    }
                await asyncio.wait({command_task}, timeout=0.25)
                if command_task.done():
                    break
                touch_activity_if_due(
                    activity_state,
                    "terminal command running",
                )
            response = await command_task
        except TimeoutError:
            await self._stop_after_interruption(sandbox)
            return {
                "output": f"[Command timed out after {timeout}s]",
                "returncode": 124,
            }
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(cancel_command_and_stop())
            await _await_owned(cleanup)  # noqa: ASYNC120
            raise
        except Exception:
            # _ThreadedProcessHandle used by upstream surfaces SDK failures as
            # an empty command result with return code 1.
            return {"output": "", "returncode": 1}

        result = {
            "output": str(response.result or ""),
            "returncode": response.exit_code,
        }
        if bounded_capture:
            return await self._apply_bounded_capture(result)
        return result

    async def cleanup(self) -> None:
        """Sync back, stop/delete the sandbox, and close the owned SDK client."""

        async def _cleanup() -> None:
            async with self._operation_lock:
                sandbox = self._sandbox
                client = self._daytona
                if sandbox is None and client is None:
                    return
                try:
                    if sandbox is not None and self._sync_manager is not None:
                        logger.info("Daytona: syncing files from sandbox...")
                        try:
                            await self._sync_manager.sync_back()
                        except Exception as exc:
                            logger.warning("Daytona: sync_back failed: %s", exc)

                    if sandbox is not None:
                        try:
                            if self._persistent:
                                await sandbox.stop()
                                logger.info(
                                    "Daytona: stopped sandbox %s "
                                    "(filesystem preserved)",
                                    sandbox.id,
                                )
                            else:
                                await client.delete(sandbox)
                                logger.info(
                                    "Daytona: deleted sandbox %s", sandbox.id
                                )
                        except Exception as exc:
                            logger.warning("Daytona: cleanup failed: %s", exc)
                finally:
                    self._sandbox = None
                    if client is not None:
                        try:
                            await client.close()
                        except Exception as exc:
                            logger.warning("Daytona: client close failed: %s", exc)
                    self._daytona = None

        await _await_owned(asyncio.create_task(_cleanup()))
