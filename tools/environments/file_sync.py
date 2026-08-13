"""Native-async file synchronization for remote execution backends."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import posixpath
import shlex
import signal
import tarfile
import time
from collections.abc import Awaitable, Callable
from pathlib import Path, PurePosixPath
from typing import Any

import aiofiles
import aiofiles.os
import aiofiles.tempfile

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

from hermes_constants import get_hermes_home
from tools.environments.base import _file_mtime_key


logger = logging.getLogger(__name__)
_monotonic = time.monotonic
_SYNC_INTERVAL_SECONDS = 5.0
_FORCE_SYNC_ENV = "HERMES_FORCE_FILE_SYNC"
_SYNC_BACK_MAX_RETRIES = 3
_SYNC_BACK_BACKOFF = (2, 4, 8)
_SYNC_BACK_MAX_BYTES = 2 * 1024 * 1024 * 1024
_LOOP_SYNC_BACK_LOCK_ATTRIBUTE = "_async_hermes_file_sync_back_lock"

GetFilesFn = Callable[[], Awaitable[list[tuple[str, str]]]]
UploadFn = Callable[[str, str], Awaitable[None]]
BulkUploadFn = Callable[[list[tuple[str, str]]], Awaitable[None]]
BulkDownloadFn = Callable[[Path], Awaitable[None]]
DeleteFn = Callable[[list[str]], Awaitable[None]]


async def iter_sync_files(
    container_base: str = "/root/.hermes",
) -> list[tuple[str, str]]:
    """Enumerate credential, skill, and cache files for one remote backend."""
    from tools.credential_files import (
        get_credential_file_mounts,
        iter_cache_files,
        iter_skills_files,
    )

    files: list[tuple[str, str]] = []
    for entry in await get_credential_file_mounts():
        remote = entry["container_path"].replace(
            "/root/.hermes", container_base, 1
        )
        files.append((entry["host_path"], remote))
    for entry in await iter_skills_files(container_base=container_base):
        files.append((entry["host_path"], entry["container_path"]))
    for entry in await iter_cache_files(container_base=container_base):
        files.append((entry["host_path"], entry["container_path"]))
    return files


async def _credential_host_paths() -> set[str]:
    try:
        from tools.credential_files import get_credential_file_mounts

        mounts = await get_credential_file_mounts()
    except Exception:
        return set()
    paths: set[str] = set()
    realpath = aiofiles.os.wrap(os.path.realpath)
    expanduser = aiofiles.os.wrap(os.path.expanduser)
    for entry in mounts:
        host_path = entry.get("host_path") if isinstance(entry, dict) else None
        if not host_path:
            continue
        try:
            paths.add(await realpath(await expanduser(str(host_path))))
        except OSError:
            paths.add(await expanduser(str(host_path)))
    return paths


def quoted_rm_command(remote_paths: list[str]) -> str:
    return "rm -f " + " ".join(shlex.quote(path) for path in remote_paths)


def quoted_mkdir_command(dirs: list[str]) -> str:
    return "mkdir -p " + " ".join(shlex.quote(path) for path in dirs)


def unique_parent_dirs(files: list[tuple[str, str]]) -> list[str]:
    return sorted({posixpath.dirname(remote) for _, remote in files})


async def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    async with aiofiles.open(path, "rb") as handle:
        while chunk := await handle.read(64 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def _get_loop_sync_back_lock() -> asyncio.Lock:
    """Return one sync-back lock for the current event loop.

    This is used where ``fcntl`` is unavailable.  Storing it on the loop keeps
    independent application loops isolated and avoids binding a module-global
    ``asyncio.Lock`` to the first loop that happens to contend on it.
    """
    loop = asyncio.get_running_loop()
    lock = getattr(loop, _LOOP_SYNC_BACK_LOCK_ATTRIBUTE, None)
    if lock is None:
        lock = asyncio.Lock()
        setattr(loop, _LOOP_SYNC_BACK_LOCK_ATTRIBUTE, lock)
    return lock


def _safe_tar_name(name: str) -> str:
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or not name
        or "\\" in name
        or any(part == ".." for part in path.parts)
    ):
        raise ValueError(f"unsafe tar member path: {name!r}")
    normalized = str(path)
    if normalized in {"", "."}:
        raise ValueError(f"unsafe tar member path: {name!r}")
    return normalized


def _parse_pax_path(payload: bytes) -> str | None:
    offset = 0
    result: str | None = None
    while offset < len(payload):
        space = payload.find(b" ", offset)
        if space < 0:
            break
        try:
            size = int(payload[offset:space])
        except ValueError:
            break
        if size <= 0 or offset + size > len(payload):
            break
        record = payload[space + 1 : offset + size].rstrip(b"\n")
        key, separator, value = record.partition(b"=")
        if separator and key == b"path":
            result = value.decode("utf-8", "surrogateescape")
        offset += size
    return result


async def _extract_safe_tar(
    archive_path: Path,
    staging: Path,
) -> list[tuple[Path, str, int, float]]:
    """Extract regular files from an uncompressed tar without blocking I/O."""
    extracted: dict[str, tuple[Path, str, int, float]] = {}
    pending_name: str | None = None
    pending_pax_path: str | None = None
    async with aiofiles.open(archive_path, "rb") as archive:
        while True:
            header = await archive.read(tarfile.BLOCKSIZE)
            if not header or header == b"\0" * tarfile.BLOCKSIZE:
                break
            if len(header) != tarfile.BLOCKSIZE:
                raise ValueError("truncated tar header")
            info = tarfile.TarInfo.frombuf(
                header,
                encoding="utf-8",
                errors="surrogateescape",
            )
            size = int(info.size)
            blocks = (size + tarfile.BLOCKSIZE - 1) // tarfile.BLOCKSIZE
            if info.type in {tarfile.GNUTYPE_LONGNAME, tarfile.XHDTYPE}:
                payload = await archive.read(blocks * tarfile.BLOCKSIZE)
                if len(payload) != blocks * tarfile.BLOCKSIZE:
                    raise ValueError("truncated tar extension record")
                raw = payload[:size]
                if info.type == tarfile.GNUTYPE_LONGNAME:
                    pending_name = raw.rstrip(b"\0\n").decode(
                        "utf-8", "surrogateescape"
                    )
                else:
                    pending_pax_path = _parse_pax_path(raw)
                continue

            name = pending_pax_path or pending_name or info.name
            pending_name = None
            pending_pax_path = None
            safe_name = _safe_tar_name(name)
            target = staging.joinpath(*PurePosixPath(safe_name).parts)
            if info.isdir():
                await aiofiles.os.makedirs(target, exist_ok=True)
                if blocks:
                    await archive.seek(blocks * tarfile.BLOCKSIZE, os.SEEK_CUR)
                continue
            if not info.isfile():
                # Symlinks, hardlinks, devices, and FIFOs must never cross the
                # sandbox boundary during sync-back.
                if blocks:
                    await archive.seek(blocks * tarfile.BLOCKSIZE, os.SEEK_CUR)
                continue
            await aiofiles.os.makedirs(target.parent, exist_ok=True)
            remaining = size
            async with aiofiles.open(target, "wb") as output:
                while remaining:
                    chunk = await archive.read(min(64 * 1024, remaining))
                    if not chunk:
                        raise ValueError("truncated tar member")
                    await output.write(chunk)
                    remaining -= len(chunk)
            padding = blocks * tarfile.BLOCKSIZE - size
            if padding:
                await archive.seek(padding, os.SEEK_CUR)
            mode = info.mode & 0o755
            if not mode & 0o100:
                mode &= ~0o111
            mode |= 0o600
            mode &= ~0o022
            await aiofiles.os.wrap(os.chmod)(target, mode)
            remote_path = "/" + safe_name
            extracted[remote_path] = (
                target,
                remote_path,
                mode,
                float(info.mtime),
            )
    return list(extracted.values())


async def _atomic_copy(
    source: Path,
    destination: Path,
    mode: int,
    mtime: float,
) -> None:
    await aiofiles.os.makedirs(destination.parent, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.hermes-sync-{os.getpid()}-{id(asyncio.current_task())}"
    )
    try:
        async with aiofiles.open(source, "rb") as source_handle:
            async with aiofiles.open(temporary, "wb") as destination_handle:
                while chunk := await source_handle.read(64 * 1024):
                    await destination_handle.write(chunk)
                await destination_handle.flush()
                await aiofiles.os.wrap(os.fsync)(destination_handle.fileno())
        await aiofiles.os.wrap(os.chmod)(temporary, mode)
        await aiofiles.os.wrap(os.utime)(temporary, (mtime, mtime))
        await aiofiles.os.replace(temporary, destination)
    finally:
        try:
            await aiofiles.os.remove(temporary)
        except FileNotFoundError:
            pass


class FileSyncManager:
    """Track and synchronize remote files through native async transports."""

    def __init__(
        self,
        get_files_fn: GetFilesFn,
        upload_fn: UploadFn,
        delete_fn: DeleteFn,
        sync_interval: float = _SYNC_INTERVAL_SECONDS,
        bulk_upload_fn: BulkUploadFn | None = None,
        bulk_download_fn: BulkDownloadFn | None = None,
    ):
        self._get_files_fn = get_files_fn
        self._upload_fn = upload_fn
        self._bulk_upload_fn = bulk_upload_fn
        self._bulk_download_fn = bulk_download_fn
        self._delete_fn = delete_fn
        self._synced_files: dict[str, tuple[float, int]] = {}
        self._pushed_hashes: dict[str, str] = {}
        self._upload_only_host_paths: set[str] = set()
        self._last_sync_time = 0.0
        self._sync_interval = sync_interval
        self._sync_lock = asyncio.Lock()

    async def sync(self, *, force: bool = False) -> None:
        async with self._sync_lock:
            if not force and not os.environ.get(_FORCE_SYNC_ENV):
                now = _monotonic()
                if now - self._last_sync_time < self._sync_interval:
                    return

            current_files = await self._get_files_fn()
            self._upload_only_host_paths.update(await _credential_host_paths())
            current_remote_paths = {remote for _, remote in current_files}
            to_upload: list[tuple[str, str]] = []
            new_files = dict(self._synced_files)
            for host_path, remote_path in current_files:
                file_key = await _file_mtime_key(host_path)
                if file_key is None or self._synced_files.get(remote_path) == file_key:
                    continue
                to_upload.append((host_path, remote_path))
                new_files[remote_path] = file_key
            to_delete = [
                path
                for path in self._synced_files
                if path not in current_remote_paths
            ]
            if not to_upload and not to_delete:
                self._last_sync_time = _monotonic()
                return

            previous_files = dict(self._synced_files)
            previous_hashes = dict(self._pushed_hashes)
            try:
                if to_upload and self._bulk_upload_fn is not None:
                    await self._bulk_upload_fn(to_upload)
                else:
                    for host_path, remote_path in to_upload:
                        await self._upload_fn(host_path, remote_path)
                if to_delete:
                    await self._delete_fn(to_delete)
                for host_path, remote_path in to_upload:
                    current_key = await _file_mtime_key(host_path)
                    if current_key != new_files.get(remote_path):
                        raise RuntimeError(
                            f"file changed during sync: {host_path}"
                        )
                    self._pushed_hashes[remote_path] = await _sha256_file(host_path)
                    if await _file_mtime_key(host_path) != current_key:
                        raise RuntimeError(
                            f"file changed while hashing: {host_path}"
                        )
                for path in to_delete:
                    new_files.pop(path, None)
                    self._pushed_hashes.pop(path, None)
                self._synced_files = new_files
                self._last_sync_time = _monotonic()
            except asyncio.CancelledError:
                self._synced_files = previous_files
                self._pushed_hashes = previous_hashes
                raise
            except Exception as exc:
                self._synced_files = previous_files
                self._pushed_hashes = previous_hashes
                logger.warning("file_sync: sync failed, rolled back state: %s", exc)

    async def sync_back(self, hermes_home: Path | None = None) -> None:
        if self._bulk_download_fn is None:
            return
        if not self._pushed_hashes and not self._synced_files:
            logger.debug("sync_back: no prior push state — skipping")
            return
        task = asyncio.create_task(
            self._sync_back_with_retries(hermes_home or get_hermes_home())
        )
        await _await_owned(task)

    async def _sync_back_with_retries(self, hermes_home: Path) -> None:
        lock_path = hermes_home / ".sync.lock"
        await aiofiles.os.makedirs(lock_path.parent, exist_ok=True)
        last_exception: Exception | None = None
        for attempt in range(_SYNC_BACK_MAX_RETRIES):
            try:
                await self._sync_back_once(lock_path)
                return
            except Exception as exc:
                last_exception = exc
                if attempt < _SYNC_BACK_MAX_RETRIES - 1:
                    delay = _SYNC_BACK_BACKOFF[attempt]
                    logger.warning(
                        "sync_back: attempt %d failed (%s), retrying in %ds",
                        attempt + 1,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
        logger.warning(
            "sync_back: all %d attempts failed: %s",
            _SYNC_BACK_MAX_RETRIES,
            last_exception,
        )

    async def _sync_back_once(self, lock_path: Path) -> None:
        await self._sync_back_locked(lock_path)

    async def _sync_back_with_deferred_sigint(self) -> None:
        deferred_sigint = False
        original_handler: Any = None
        try:
            original_handler = signal.getsignal(signal.SIGINT)

            def defer_sigint(_signum, _frame):
                nonlocal deferred_sigint
                deferred_sigint = True

            signal.signal(signal.SIGINT, defer_sigint)
        except ValueError:
            original_handler = None
        try:
            await self._sync_back_impl()
        finally:
            if original_handler is not None:
                signal.signal(signal.SIGINT, original_handler)
                if deferred_sigint:
                    signal.raise_signal(signal.SIGINT)

    async def _sync_back_locked(self, lock_path: Path) -> None:
        async with aiofiles.open(lock_path, "a+b") as lock_handle:
            if fcntl is None:
                async with _get_loop_sync_back_lock():
                    await self._sync_back_with_deferred_sigint()
                return
            while True:
                try:
                    fcntl.flock(  # noqa: ASYNC240 - LOCK_NB never blocks the loop
                        lock_handle.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                    break
                except BlockingIOError:
                    await asyncio.sleep(0.05)
            try:
                await self._sync_back_with_deferred_sigint()
            finally:
                try:
                    fcntl.flock(  # noqa: ASYNC240 - unlock cannot wait
                        lock_handle.fileno(),
                        fcntl.LOCK_UN,
                    )
                except OSError:
                    pass

    async def _sync_back_impl(self) -> None:
        if self._bulk_download_fn is None:
            raise RuntimeError("_sync_back_impl called without bulk_download_fn")
        try:
            file_mapping = list(await self._get_files_fn())
        except Exception:
            file_mapping = []

        async with aiofiles.tempfile.TemporaryDirectory(
            prefix="hermes-sync-back-"
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            archive_path = temporary_root / "remote.tar"
            staging = temporary_root / "staging"
            await aiofiles.os.makedirs(staging, exist_ok=True)
            await self._bulk_download_fn(archive_path)
            try:
                archive_size = (await aiofiles.os.stat(archive_path)).st_size
            except OSError:
                archive_size = 0
            if archive_size > _SYNC_BACK_MAX_BYTES:
                logger.warning(
                    "sync_back: remote tar is %d bytes (cap %d) — skipping extraction",
                    archive_size,
                    _SYNC_BACK_MAX_BYTES,
                )
                return
            extracted = await _extract_safe_tar(archive_path, staging)
            upload_only = self._upload_only_host_paths | (
                await _credential_host_paths()
            )
            applied = 0
            for staged_file, remote_path, mode, mtime in extracted:
                pushed_hash = self._pushed_hashes.get(remote_path)
                if pushed_hash is not None:
                    remote_hash = await _sha256_file(staged_file)
                    if remote_hash == pushed_hash:
                        continue
                host_path = self._resolve_host_path(remote_path, file_mapping)
                if host_path is None:
                    host_path = await self._infer_host_path(
                        remote_path,
                        file_mapping,
                        upload_only_host_paths=upload_only,
                    )
                if host_path is None or await self._is_upload_only_host_path(
                    host_path, upload_only
                ):
                    continue
                if pushed_hash is not None and await aiofiles.os.path.exists(host_path):
                    if await _sha256_file(host_path) != pushed_hash:
                        logger.warning(
                            "sync_back: conflict on %s — host modified since push, "
                            "remote also changed. Applying remote version "
                            "(last-write-wins).",
                            remote_path,
                        )
                await _atomic_copy(
                    staged_file,
                    Path(host_path),
                    mode,
                    mtime,
                )
                applied += 1
            if applied:
                logger.info("sync_back: applied %d changed file(s)", applied)
            else:
                logger.debug("sync_back: no remote changes detected")

    @staticmethod
    def _resolve_host_path(
        remote_path: str,
        file_mapping: list[tuple[str, str]] | None = None,
    ) -> str | None:
        for host, remote in file_mapping or []:
            if remote == remote_path:
                return host
        return None

    async def _infer_host_path(
        self,
        remote_path: str,
        file_mapping: list[tuple[str, str]] | None = None,
        *,
        upload_only_host_paths: set[str] | None = None,
    ) -> str | None:
        upload_only_host_paths = upload_only_host_paths or set()
        for host, remote in file_mapping or []:
            if await self._is_upload_only_host_path(host, upload_only_host_paths):
                continue
            remote_dir = str(PurePosixPath(remote).parent)
            if remote_path.startswith(remote_dir + "/"):
                host_dir = str(Path(host).parent)
                return host_dir + remote_path[len(remote_dir) :]
        return None

    @staticmethod
    async def _is_upload_only_host_path(
        host_path: str,
        upload_only_host_paths: set[str],
    ) -> bool:
        realpath = aiofiles.os.wrap(os.path.realpath)
        expanduser = aiofiles.os.wrap(os.path.expanduser)
        resolved = await realpath(await expanduser(host_path))
        return resolved in upload_only_host_paths
