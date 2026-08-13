"""SSH remote execution environment with ControlMaster connection persistence."""

from __future__ import annotations

import asyncio
import codecs
import hashlib
import logging
import os
import posixpath
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os
import aiofiles.tempfile

from tools.environments.base import (
    BaseEnvironment,
    _BoundedOutputCollector,
    _UNBOUNDED_CAPTURE_CHARS,
    touch_activity_if_due,
)
from tools.environments.file_sync import (
    FileSyncManager,
    iter_sync_files,
    quoted_mkdir_command,
    quoted_rm_command,
    unique_parent_dirs,
)


logger = logging.getLogger(__name__)


async def _which(executable: str) -> str | None:
    """Find an executable without performing filesystem I/O on the event loop."""
    if os.path.dirname(executable):
        candidates = [executable]
    else:
        candidates = [
            os.path.join(directory or os.curdir, executable)
            for directory in os.environ.get("PATH", os.defpath).split(os.pathsep)
        ]
    if os.name == "nt":
        extensions = os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(
            os.pathsep
        )
        expanded: list[str] = []
        for candidate in candidates:
            expanded.append(candidate)
            if not any(candidate.lower().endswith(ext.lower()) for ext in extensions):
                expanded.extend(candidate + ext for ext in extensions)
        candidates = expanded
    for candidate in candidates:
        if await aiofiles.os.path.isfile(candidate) and await aiofiles.os.access(
            candidate, os.X_OK
        ):
            return candidate
    return None  # noqa: ASYNC910 - every non-empty PATH candidate is awaited


async def _ensure_ssh_available() -> None:
    """Fail fast with a clear error when the SSH client is unavailable."""
    if not await _which("ssh"):
        raise RuntimeError(
            "SSH is not installed or not in PATH. Install OpenSSH client: "
            "apt install openssh-client"
        )
    if not await _which("scp"):
        raise RuntimeError(
            "SCP is not installed or not in PATH. Install OpenSSH client: "
            "apt install openssh-client"
        )


async def _terminate_and_reap(  # noqa: ASYNC910 - process.wait is the checkpoint
    process: asyncio.subprocess.Process,
) -> None:
    """Kill and reap one owned subprocess."""
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    try:
        await process.wait()
    except ProcessLookupError:
        pass


async def _await_owned(task: asyncio.Task[Any]) -> Any:
    """Finish an owned cleanup task before propagating caller cancellation."""
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


async def _finish_tasks(  # noqa: ASYNC910 - an empty task set is a true no-op
    tasks: list[asyncio.Task[Any]],
) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


class SSHEnvironment(BaseEnvironment):
    """Run commands on a remote machine over SSH."""

    def __init__(
        self,
        host: str,
        user: str,
        cwd: str = "~",
        timeout: int = 60,
        port: int = 22,
        key_path: str = "",
    ):
        super().__init__(cwd=cwd, timeout=timeout)
        self.host = host
        self.user = user
        self.port = port
        self.key_path = key_path

        # Keep construction state-only. Reading environment variables and
        # shaping a Path are pure; directory creation is deferred to the first
        # awaited use.
        temporary_root = (
            os.environ.get("TMPDIR")
            or os.environ.get("TEMP")
            or os.environ.get("TMP")
            or "/tmp"
        )
        self.control_dir = Path(temporary_root) / "hermes-ssh"
        socket_id = hashlib.sha256(
            f"{user}@{host}:{port}".encode()
        ).hexdigest()[:16]
        self._socket_id = socket_id
        self.control_socket = self.control_dir / f"{self._socket_id}.sock"

        self._remote_home = "/root" if user == "root" else f"/home/{user}"
        self._sync_manager: FileSyncManager | None = None
        self._initialization_lock = asyncio.Lock()
        self._cleanup_lock = asyncio.Lock()
        self._processes: set[asyncio.subprocess.Process] = set()

    def _build_ssh_command(self, extra_args: list | None = None) -> list:
        cmd = ["ssh"]
        cmd.extend(["-o", f"ControlPath={self.control_socket}"])
        cmd.extend(["-o", "ControlMaster=auto"])
        cmd.extend(["-o", "ControlPersist=300"])
        cmd.extend(["-o", "BatchMode=yes"])
        cmd.extend(["-o", "StrictHostKeyChecking=accept-new"])
        cmd.extend(["-o", "ConnectTimeout=10"])
        if self.port != 22:
            cmd.extend(["-p", str(self.port)])
        if self.key_path:
            cmd.extend(["-i", self.key_path])
        if extra_args:
            cmd.extend(extra_args)
        cmd.append(f"{self.user}@{self.host}")
        return cmd

    async def _spawn(
        self,
        argv: list[str],
        **kwargs: Any,
    ) -> asyncio.subprocess.Process:
        process = await asyncio.create_subprocess_exec(
            *argv,
            start_new_session=os.name == "posix",
            **kwargs,
        )
        self._processes.add(process)
        return process

    async def _terminate(self, process: asyncio.subprocess.Process) -> None:
        task = asyncio.create_task(_terminate_and_reap(process))
        try:
            await _await_owned(task)
        finally:
            self._processes.discard(process)

    async def _run_captured(
        self,
        argv: list[str],
        *,
        timeout: int | float,
        stdin_data: bytes | None = None,
        merge_stderr: bool = False,
    ) -> tuple[int, bytes, bytes]:
        process = await self._spawn(
            argv,
            stdin=(
                asyncio.subprocess.PIPE
                if stdin_data is not None
                else asyncio.subprocess.DEVNULL
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=(
                asyncio.subprocess.STDOUT
                if merge_stderr
                else asyncio.subprocess.PIPE
            ),
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(stdin_data),
                timeout=float(timeout),
            )
        except TimeoutError as exc:
            await self._terminate(  # noqa: ASYNC120 - caller cancellation wins
                process
            )
            raise subprocess.TimeoutExpired(argv, timeout) from exc
        except asyncio.CancelledError:
            await self._terminate(  # noqa: ASYNC120 - caller cancellation wins
                process
            )
            raise
        except BaseException:
            await self._terminate(  # noqa: ASYNC120 - caller cancellation wins
                process
            )
            raise
        finally:
            if process.returncode is not None:
                self._processes.discard(process)
        return int(process.returncode or 0), stdout or b"", stderr or b""

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return  # noqa: ASYNC910 - initialized fast path is state-only
        async with self._initialization_lock:
            if self._initialized:
                return  # noqa: ASYNC910 - lock peer completed initialization
            try:
                await _ensure_ssh_available()
                gettempdir = aiofiles.os.wrap(tempfile.gettempdir)
                self.control_dir = Path(await gettempdir()) / "hermes-ssh"
                self.control_socket = (
                    self.control_dir / f"{self._socket_id}.sock"
                )
                await aiofiles.os.makedirs(self.control_dir, exist_ok=True)
                await self._establish_connection()
                self._remote_home = await self._detect_remote_home()
                await self._ensure_remote_dirs()
                self._sync_manager = FileSyncManager(
                    get_files_fn=lambda: iter_sync_files(
                        f"{self._remote_home}/.hermes"
                    ),
                    upload_fn=self._scp_upload,
                    delete_fn=self._ssh_delete,
                    bulk_upload_fn=self._ssh_bulk_upload,
                    bulk_download_fn=self._ssh_bulk_download,
                )
                await self._sync_manager.sync(force=True)
                await self.init_session()
            except BaseException:
                cleanup_task = asyncio.create_task(self._cleanup_impl())
                try:
                    await _await_owned(cleanup_task)
                except Exception:
                    logger.debug(
                        "SSH: lazy initialization cleanup failed",
                        exc_info=True,
                    )
                raise

    async def _establish_connection(self) -> None:
        cmd = self._build_ssh_command()
        cmd.append("echo 'SSH connection established'")
        try:
            returncode, stdout, stderr = await self._run_captured(cmd, timeout=15)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"SSH connection to {self.user}@{self.host} timed out"
            ) from exc
        if returncode != 0:
            error_msg = stderr.decode(errors="replace").strip() or stdout.decode(
                errors="replace"
            ).strip()
            raise RuntimeError(f"SSH connection failed: {error_msg}")

    async def _detect_remote_home(self) -> str:
        """Detect the remote user's home directory."""
        try:
            cmd = self._build_ssh_command()
            cmd.append("echo $HOME")
            returncode, stdout, _stderr = await self._run_captured(cmd, timeout=10)
            home = stdout.decode(errors="replace").strip()
            if home and returncode == 0:
                logger.debug("SSH: remote home = %s", home)
                return home
        except Exception:
            pass
        if self.user == "root":
            return "/root"  # noqa: ASYNC910 - remote probe was awaited above
        return f"/home/{self.user}"  # noqa: ASYNC910 - probe was awaited above

    async def _ensure_remote_dirs(self) -> None:
        """Create base ~/.hermes directory tree on remote in one SSH call."""
        base = f"{self._remote_home}/.hermes"
        dirs = [base, f"{base}/skills", f"{base}/credentials", f"{base}/cache"]
        cmd = self._build_ssh_command()
        cmd.append(quoted_mkdir_command(dirs))
        await self._run_captured(cmd, timeout=10)

    async def _scp_upload(self, host_path: str, remote_path: str) -> None:
        """Upload a single file via scp over ControlMaster."""
        parent = str(Path(remote_path).parent)
        mkdir_cmd = self._build_ssh_command()
        mkdir_cmd.append(f"mkdir -p {shlex.quote(parent)}")
        await self._run_captured(mkdir_cmd, timeout=10)

        scp_cmd = ["scp", "-o", f"ControlPath={self.control_socket}"]
        if self.port != 22:
            scp_cmd.extend(["-P", str(self.port)])
        if self.key_path:
            scp_cmd.extend(["-i", self.key_path])
        scp_cmd.extend([host_path, f"{self.user}@{self.host}:{remote_path}"])
        returncode, _stdout, stderr = await self._run_captured(
            scp_cmd,
            timeout=30,
        )
        if returncode != 0:
            raise RuntimeError(
                f"scp failed: {stderr.decode(errors='replace').strip()}"
            )

    async def _copy_for_staging(self, source: str, destination: Path) -> None:
        stat_result = await aiofiles.os.stat(source)
        async with aiofiles.open(source, "rb") as source_handle:
            async with aiofiles.open(destination, "wb") as destination_handle:
                while chunk := await source_handle.read(64 * 1024):
                    await destination_handle.write(chunk)
        await aiofiles.os.wrap(os.chmod)(destination, stat_result.st_mode)
        await aiofiles.os.wrap(os.utime)(
            destination,
            ns=(stat_result.st_atime_ns, stat_result.st_mtime_ns),
        )

    async def _cleanup_bulk_processes(
        self,
        processes: list[asyncio.subprocess.Process],
        tasks: list[asyncio.Task[Any]],
    ) -> None:
        async def cleanup() -> None:
            await _finish_tasks(tasks)
            await asyncio.gather(
                *(self._terminate(process) for process in processes),
                return_exceptions=True,
            )

        await _await_owned(asyncio.create_task(cleanup()))

    async def _ssh_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        """Upload many files in a single tar-over-SSH stream."""
        if not files:
            return  # noqa: ASYNC910 - retained empty-input no-op

        base = f"{self._remote_home}/.hermes"
        parents = unique_parent_dirs(files)
        if parents:
            cmd = self._build_ssh_command()
            cmd.append(quoted_mkdir_command(parents))
            returncode, _stdout, stderr = await self._run_captured(
                cmd,
                timeout=30,
            )
            if returncode != 0:
                raise RuntimeError(
                    "remote mkdir failed: "
                    f"{stderr.decode(errors='replace').strip()}"
                )

        async with aiofiles.tempfile.TemporaryDirectory(
            prefix="hermes-ssh-bulk-"
        ) as staging_value:
            staging = Path(staging_value)
            for host_path, remote_path in files:
                try:
                    rel_remote = posixpath.relpath(remote_path, base)
                except ValueError as exc:
                    raise RuntimeError(
                        f"remote path {remote_path!r} is not under sync base "
                        f"{base!r}"
                    ) from exc
                if rel_remote == "." or rel_remote.startswith("../"):
                    raise RuntimeError(
                        f"remote path {remote_path!r} escapes sync base {base!r}"
                    )

                staged = staging / rel_remote
                await aiofiles.os.makedirs(staged.parent, exist_ok=True)
                try:
                    absolute_host = await aiofiles.os.path.abspath(host_path)
                    await aiofiles.os.symlink(absolute_host, staged)
                except OSError as exc:
                    if getattr(exc, "winerror", None) == 1314:
                        await self._copy_for_staging(  # noqa: ASYNC120 - cancellation wins
                            host_path,
                            staged,
                        )
                    else:
                        raise

            tar_cmd = ["tar", "-chf", "-", "-C", str(staging), "."]
            ssh_cmd = self._build_ssh_command()
            ssh_cmd.append(
                f"tar xf - --no-overwrite-dir -C {shlex.quote(base)}"
            )

            tar_process = await self._spawn(
                tar_cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                ssh_process = await self._spawn(
                    ssh_cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except BaseException:
                await self._terminate(  # noqa: ASYNC120 - cancellation wins
                    tar_process
                )
                raise

            async def pump() -> None:  # noqa: ASYNC910 - stream read is checkpoint
                assert tar_process.stdout is not None
                assert ssh_process.stdin is not None
                try:
                    while chunk := await tar_process.stdout.read(64 * 1024):
                        ssh_process.stdin.write(chunk)
                        await ssh_process.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    ssh_process.stdin.close()

            assert tar_process.stderr is not None
            assert ssh_process.stdout is not None
            assert ssh_process.stderr is not None
            pump_task = asyncio.create_task(pump())
            tar_stderr_task = asyncio.create_task(tar_process.stderr.read())
            ssh_stdout_task = asyncio.create_task(ssh_process.stdout.read())
            ssh_stderr_task = asyncio.create_task(ssh_process.stderr.read())
            tasks: list[asyncio.Task[Any]] = [
                pump_task,
                tar_stderr_task,
                ssh_stdout_task,
                ssh_stderr_task,
            ]
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(
                        pump_task,
                        tar_stderr_task,
                        ssh_stdout_task,
                        ssh_stderr_task,
                        tar_process.wait(),
                        ssh_process.wait(),
                    ),
                    timeout=120,
                )
            except TimeoutError as exc:
                await self._cleanup_bulk_processes(  # noqa: ASYNC120 - cancellation wins
                    [tar_process, ssh_process],
                    tasks,
                )
                raise RuntimeError("SSH bulk upload timed out") from exc
            except asyncio.CancelledError:
                await self._cleanup_bulk_processes(
                    [tar_process, ssh_process],
                    tasks,
                )
                raise
            except BaseException:
                await self._cleanup_bulk_processes(  # noqa: ASYNC120 - cancellation wins
                    [tar_process, ssh_process],
                    tasks,
                )
                raise
            finally:
                if tar_process.returncode is not None:
                    self._processes.discard(tar_process)
                if ssh_process.returncode is not None:
                    self._processes.discard(ssh_process)

            tar_stderr = results[1]
            ssh_stderr = results[3]
            if tar_process.returncode != 0:
                raise RuntimeError(
                    f"tar create failed (rc={tar_process.returncode}): "
                    f"{tar_stderr.decode(errors='replace').strip()}"
                )
            if ssh_process.returncode != 0:
                raise RuntimeError(
                    f"tar extract over SSH failed (rc={ssh_process.returncode}): "
                    f"{ssh_stderr.decode(errors='replace').strip()}"
                )

        logger.debug("SSH: bulk-uploaded %d file(s) via tar pipe", len(files))

    async def _ssh_bulk_download(self, dest: Path) -> None:
        """Download remote .hermes/ as a tar archive."""
        rel_base = f"{self._remote_home}/.hermes".lstrip("/")
        ssh_cmd = self._build_ssh_command()
        ssh_cmd.append(f"tar cf - -C / {shlex.quote(rel_base)}")
        process = await self._spawn(
            ssh_cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert process.stdout is not None
        assert process.stderr is not None

        async def write_archive() -> None:
            async with aiofiles.open(dest, "wb") as destination:
                while chunk := await process.stdout.read(64 * 1024):
                    await destination.write(chunk)

        write_task = asyncio.create_task(write_archive())
        stderr_task = asyncio.create_task(process.stderr.read())
        tasks: list[asyncio.Task[Any]] = [write_task, stderr_task]
        try:
            results = await asyncio.wait_for(
                asyncio.gather(write_task, stderr_task, process.wait()),
                timeout=120,
            )
        except TimeoutError as exc:
            await self._cleanup_bulk_processes(  # noqa: ASYNC120 - cancellation wins
                [process], tasks
            )
            raise subprocess.TimeoutExpired(ssh_cmd, 120) from exc
        except asyncio.CancelledError:
            await self._cleanup_bulk_processes([process], tasks)
            raise
        except BaseException:
            await self._cleanup_bulk_processes(  # noqa: ASYNC120 - cancellation wins
                [process], tasks
            )
            raise
        finally:
            if process.returncode is not None:
                self._processes.discard(process)
        if process.returncode != 0:
            raise RuntimeError(
                "SSH bulk download failed: "
                f"{results[1].decode(errors='replace').strip()}"
            )

    async def _ssh_delete(self, remote_paths: list[str]) -> None:
        """Batch-delete remote files in one SSH call."""
        cmd = self._build_ssh_command()
        cmd.append(quoted_rm_command(remote_paths))
        returncode, _stdout, stderr = await self._run_captured(cmd, timeout=10)
        if returncode != 0:
            raise RuntimeError(
                f"remote rm failed: {stderr.decode(errors='replace').strip()}"
            )

    async def _before_execute(self) -> None:
        """Sync files to remote via FileSyncManager."""
        assert self._sync_manager is not None
        await self._sync_manager.sync()

    async def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int | float = 120,
        stdin_data: str | None = None,
        bounded_capture: bool = False,
    ) -> dict[str, Any]:
        """Run one SSH process that invokes bash on the remote host."""
        cmd = self._build_ssh_command()
        if login:
            cmd.extend(["bash", "-l", "-c", shlex.quote(cmd_string)])
        else:
            cmd.extend(["bash", "-c", shlex.quote(cmd_string)])
        output = (
            await self._bounded_output_collector()
            if bounded_capture
            else _BoundedOutputCollector(_UNBOUNDED_CAPTURE_CHARS)
        )
        process: asyncio.subprocess.Process | None = None
        tasks: list[asyncio.Task[Any]] = []
        try:
            process = await self._spawn(
                cmd,
                stdin=(
                    asyncio.subprocess.PIPE
                    if stdin_data is not None
                    else asyncio.subprocess.DEVNULL
                ),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            assert process.stdout is not None

            async def read_output() -> None:
                decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                while chunk := await process.stdout.read(64 * 1024):
                    output.append(decoder.decode(chunk))
                tail = decoder.decode(b"", final=True)
                if tail:
                    output.append(tail)

            async def write_input() -> None:  # noqa: ASYNC910 - drain is checkpoint
                assert process is not None
                assert process.stdin is not None
                assert stdin_data is not None
                try:
                    process.stdin.write(stdin_data.encode())
                    await process.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    process.stdin.close()

            tasks.append(asyncio.create_task(read_output()))
            if stdin_data is not None:
                tasks.append(asyncio.create_task(write_input()))
            waiter = asyncio.create_task(process.wait())
            started = time.monotonic()
            activity_state = {
                "last_touch": started,
                "start": started,
                "interval": 10.0,
            }
            while not waiter.done():
                from tools.interrupt import is_interrupted

                if is_interrupted():
                    await self._cleanup_bulk_processes(
                        [process],
                        [waiter, *tasks],
                    )
                    suffix = "\n[Command interrupted]"
                    rendered = output.render(suffix=suffix)
                    if output.total_chars == 0:
                        rendered = rendered.lstrip()
                    return await self._finalize_wait_result(output, rendered, 130)
                remaining = float(timeout) - (time.monotonic() - started)
                if remaining <= 0:
                    break
                await asyncio.wait({waiter}, timeout=min(0.2, remaining))
                touch_activity_if_due(activity_state, "terminal command running")
            if not waiter.done():
                await self._cleanup_bulk_processes(
                    [process],
                    [waiter, *tasks],
                )
                suffix = f"\n[Command timed out after {timeout}s]"
                rendered = output.render(suffix=suffix)
                if output.total_chars == 0:
                    rendered = rendered.lstrip()
                return await self._finalize_wait_result(output, rendered, 124)
            await waiter
            self._processes.discard(process)
            if tasks:
                await asyncio.wait(tasks, timeout=0.25)
                await _finish_tasks(tasks)
        except asyncio.CancelledError:
            if process is not None:
                await self._cleanup_bulk_processes([process], tasks)
            raise
        except OSError:
            if process is not None:
                await self._cleanup_bulk_processes(  # noqa: ASYNC120 - cancellation wins
                    [process], tasks
                )
            raise
        except BaseException:
            if process is not None:
                await self._cleanup_bulk_processes(  # noqa: ASYNC120 - cancellation wins
                    [process], tasks
                )
            raise
        return await self._finalize_wait_result(
            output,
            output.render(),
            process.returncode,
        )

    async def cleanup(self) -> None:
        """Sync back files and close the ControlMaster connection."""
        task = asyncio.create_task(self._cleanup_after_initialization())
        await _await_owned(task)

    async def _cleanup_after_initialization(self) -> None:
        async with self._initialization_lock:
            await self._cleanup_impl()

    async def _cleanup_impl(self) -> None:
        async with self._cleanup_lock:
            try:
                if self._sync_manager is not None:
                    logger.info("SSH: syncing files from sandbox...")
                    await self._sync_manager.sync_back()
            finally:
                active = list(self._processes)
                if active:
                    await asyncio.gather(
                        *(self._terminate(process) for process in active),
                        return_exceptions=True,
                    )
                if await aiofiles.os.path.exists(self.control_socket):
                    try:
                        cmd = [
                            "ssh",
                            "-o",
                            f"ControlPath={self.control_socket}",
                            "-O",
                            "exit",
                            f"{self.user}@{self.host}",
                        ]
                        await self._run_captured(cmd, timeout=5)
                    except (OSError, subprocess.TimeoutExpired):
                        pass
                    try:
                        await aiofiles.os.remove(self.control_socket)
                    except OSError:
                        pass
