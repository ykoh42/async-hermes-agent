"""Behavioral tests for native-async remote file synchronization."""

from __future__ import annotations

import asyncio
import io
import stat
import tarfile
from pathlib import Path

import aiofiles
import pytest

from tools.environments.file_sync import FileSyncManager, _safe_tar_name


def _callbacks(files, *, fail_upload=False):
    uploads: list[tuple[str, str]] = []
    deletes: list[list[str]] = []

    async def get_files():
        return list(files)

    async def upload(host_path, remote_path):
        if fail_upload:
            raise RuntimeError("upload failed")
        uploads.append((host_path, remote_path))

    async def delete(remote_paths):
        deletes.append(list(remote_paths))

    return get_files, upload, delete, uploads, deletes


@pytest.mark.asyncio
async def test_unchanged_files_are_not_reuploaded(tmp_path, monkeypatch):
    host = tmp_path / "a.txt"
    host.write_text("hello")
    files = [(str(host), "/root/.hermes/a.txt")]
    get_files, upload, delete, uploads, _ = _callbacks(files)
    manager = FileSyncManager(get_files, upload, delete, sync_interval=0)
    monkeypatch.setattr(
        "tools.environments.file_sync._credential_host_paths",
        _empty_credentials,
    )

    await manager.sync()
    await manager.sync()

    assert uploads == files


async def _empty_credentials():
    return set()


@pytest.mark.asyncio
async def test_removed_file_triggers_remote_delete(tmp_path, monkeypatch):
    host = tmp_path / "a.txt"
    host.write_text("hello")
    files = [(str(host), "/root/.hermes/a.txt")]
    get_files, upload, delete, _, deletes = _callbacks(files)
    manager = FileSyncManager(get_files, upload, delete, sync_interval=0)
    monkeypatch.setattr(
        "tools.environments.file_sync._credential_host_paths",
        _empty_credentials,
    )

    await manager.sync()
    files.clear()
    await manager.sync()

    assert deletes == [["/root/.hermes/a.txt"]]


@pytest.mark.asyncio
async def test_failed_upload_rolls_back_for_immediate_retry(tmp_path, monkeypatch):
    host = tmp_path / "a.txt"
    host.write_text("hello")
    files = [(str(host), "/root/.hermes/a.txt")]
    attempts = 0

    async def get_files():
        return files

    async def upload(_host, _remote):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient")

    async def delete(_paths):
        return None

    monkeypatch.setattr(
        "tools.environments.file_sync._credential_host_paths",
        _empty_credentials,
    )
    manager = FileSyncManager(get_files, upload, delete)

    await manager.sync()
    await manager.sync()

    assert attempts == 2
    assert list(manager._synced_files) == ["/root/.hermes/a.txt"]


@pytest.mark.asyncio
async def test_file_changed_during_upload_rolls_back_for_retry(
    tmp_path,
    monkeypatch,
):
    host = tmp_path / "changing.txt"
    async with aiofiles.open(host, "w", encoding="utf-8") as handle:
        await handle.write("first")
    remote = "/root/.hermes/changing.txt"
    attempts = 0

    async def get_files():
        return [(str(host), remote)]

    async def upload(_host, _remote):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            async with aiofiles.open(host, "w", encoding="utf-8") as handle:
                await handle.write("changed-after-upload")

    async def no_delete(_paths):
        return None

    monkeypatch.setattr(
        "tools.environments.file_sync._credential_host_paths",
        _empty_credentials,
    )
    manager = FileSyncManager(get_files, upload, no_delete, sync_interval=0)

    await manager.sync()
    assert manager._synced_files == {}
    assert manager._pushed_hashes == {}
    await manager.sync()

    assert attempts == 2
    assert manager._synced_files[remote] == (
        (await aiofiles.os.stat(host)).st_mtime,
        len("changed-after-upload"),
    )


@pytest.mark.asyncio
async def test_sync_back_waits_for_active_sync_transaction(tmp_path, monkeypatch):
    host = tmp_path / "new.png"
    host.write_bytes(b"new")
    remote = "/root/.hermes/cache/images/new.png"
    upload_started = asyncio.Event()
    release_upload = asyncio.Event()
    sync_back_started = asyncio.Event()

    async def get_files():
        return [(str(host), remote)]

    async def upload(_host, _remote):
        upload_started.set()
        await release_upload.wait()

    async def no_delete(_paths):
        return None

    async def bulk_download(destination: Path):
        sync_back_started.set()
        async with aiofiles.open(destination, "wb") as handle:
            await handle.write(_tar_bytes({}))

    monkeypatch.setattr(
        "tools.environments.file_sync._credential_host_paths",
        _empty_credentials,
    )
    manager = FileSyncManager(
        get_files,
        upload,
        no_delete,
        sync_interval=0,
        bulk_download_fn=bulk_download,
    )
    # Ensure sync_back has work to do once it obtains the shared transaction
    # lock, even though the upload is still in progress.
    manager._pushed_hashes["/_sentinel"] = "0" * 64

    sync_task = asyncio.create_task(manager.sync(force=True))
    await upload_started.wait()
    sync_back_task = asyncio.create_task(manager.sync_back(tmp_path))
    await asyncio.sleep(0)
    assert not sync_back_started.is_set()

    release_upload.set()
    await sync_task
    await sync_back_task
    assert sync_back_started.is_set()


def _tar_bytes(entries: dict[str, bytes], *, symlink: tuple[str, str] | None = None):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, content in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(content))
        if symlink is not None:
            info = tarfile.TarInfo(symlink[0])
            info.type = tarfile.SYMTYPE
            info.linkname = symlink[1]
            archive.addfile(info)
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_sync_back_applies_changed_file_atomically(tmp_path, monkeypatch):
    host = tmp_path / "skills" / "a.md"
    host.parent.mkdir()
    host.write_text("original")
    remote = "/root/.hermes/skills/a.md"

    async def get_files():
        return [(str(host), remote)]

    async def no_upload(_host, _remote):
        return None

    async def no_delete(_paths):
        return None

    async def download(destination: Path):
        async with aiofiles.open(destination, "wb") as handle:
            await handle.write(
                _tar_bytes({"root/.hermes/skills/a.md": b"changed"})
            )

    monkeypatch.setattr(
        "tools.environments.file_sync._credential_host_paths",
        _empty_credentials,
    )
    manager = FileSyncManager(
        get_files,
        no_upload,
        no_delete,
        bulk_download_fn=download,
    )
    manager._synced_files[remote] = (0, 0)
    manager._pushed_hashes[remote] = "not-the-current-hash"

    await manager.sync_back(tmp_path)

    assert host.read_text() == "changed"


@pytest.mark.asyncio
async def test_sync_back_never_materializes_symlinks(tmp_path, monkeypatch):
    host = tmp_path / "skills" / "a.md"
    host.parent.mkdir()
    host.write_text("original")
    remote = "/root/.hermes/skills/a.md"

    async def get_files():
        return [(str(host), remote)]

    async def no_op(*_args):
        return None

    async def download(destination: Path):
        async with aiofiles.open(destination, "wb") as handle:
            await handle.write(
                _tar_bytes(
                    {},
                    symlink=("root/.hermes/skills/link", "/etc/passwd"),
                )
            )

    monkeypatch.setattr(
        "tools.environments.file_sync._credential_host_paths",
        _empty_credentials,
    )
    manager = FileSyncManager(
        get_files,
        no_op,
        no_op,
        bulk_download_fn=download,
    )
    manager._synced_files[remote] = (0, 0)

    await manager.sync_back(tmp_path)

    assert not (host.parent / "link").exists()
    assert host.read_text() == "original"


@pytest.mark.asyncio
async def test_infer_host_path_skips_upload_only_prefix(tmp_path):
    credential = tmp_path / "credentials" / "token.json"
    skill = tmp_path / "skills" / "SKILL.md"
    mapping = [
        (str(credential), "/root/.hermes/credentials/token.json"),
        (str(skill), "/root/.hermes/skills/SKILL.md"),
    ]
    manager = FileSyncManager(*_callbacks(mapping)[:3])

    inferred = await manager._infer_host_path(
        "/root/.hermes/skills/reference.md",
        mapping,
        upload_only_host_paths={str(credential.resolve())},
    )

    assert inferred == str(skill.parent / "reference.md")


@pytest.mark.asyncio
async def test_cancelled_sync_rolls_back_and_releases_lock(tmp_path, monkeypatch):
    host = tmp_path / "cancel.txt"
    async with aiofiles.open(host, "w", encoding="utf-8") as handle:
        await handle.write("data")
    remote = "/root/.hermes/cancel.txt"
    started = asyncio.Event()

    async def get_files():
        return [(str(host), remote)]

    async def upload(_host, _remote):
        started.set()
        await asyncio.Event().wait()

    async def no_delete(_paths):
        return None

    monkeypatch.setattr(
        "tools.environments.file_sync._credential_host_paths",
        _empty_credentials,
    )
    manager = FileSyncManager(get_files, upload, no_delete, sync_interval=0)
    task = asyncio.create_task(manager.sync())
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert manager._synced_files == {}
    assert manager._pushed_hashes == {}

    uploads: list[tuple[str, str]] = []

    async def succeeding_upload(host_path, remote_path):
        uploads.append((host_path, remote_path))

    manager._upload_fn = succeeding_upload
    await manager.sync()
    assert uploads == [(str(host), remote)]


@pytest.mark.asyncio
async def test_cancelled_sync_back_finishes_owned_cleanup_then_reraises(
    tmp_path,
    monkeypatch,
):
    started = asyncio.Event()
    release = asyncio.Event()
    archive_paths: list[Path] = []

    async def get_files():
        return []

    async def no_op(*_args):
        return None

    async def download(destination: Path):
        archive_paths.append(destination)
        started.set()
        await release.wait()
        async with aiofiles.open(destination, "wb") as handle:
            await handle.write(_tar_bytes({}))

    monkeypatch.setattr(
        "tools.environments.file_sync._credential_host_paths",
        _empty_credentials,
    )
    manager = FileSyncManager(
        get_files,
        no_op,
        no_op,
        bulk_download_fn=download,
    )
    manager._pushed_hashes["/_sentinel"] = "0" * 64
    task = asyncio.create_task(manager.sync_back(tmp_path))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert archive_paths
    assert not archive_paths[0].parent.exists()


@pytest.mark.asyncio
async def test_sync_back_retries_with_async_backoff(tmp_path, monkeypatch):
    attempts = 0
    delays: list[int] = []

    async def get_files():
        return []

    async def no_op(*_args):
        return None

    async def download(destination: Path):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError(f"network error {attempts}")
        async with aiofiles.open(destination, "wb") as handle:
            await handle.write(_tar_bytes({}))

    async def record_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr("tools.environments.file_sync.asyncio.sleep", record_sleep)
    manager = FileSyncManager(
        get_files,
        no_op,
        no_op,
        bulk_download_fn=download,
    )
    manager._pushed_hashes["/_sentinel"] = "0" * 64

    await manager.sync_back(tmp_path)

    assert attempts == 3
    assert delays == [2, 4]


@pytest.mark.asyncio
async def test_concurrent_sync_back_restores_process_signal_handler(
    tmp_path,
    monkeypatch,
):
    first_started = asyncio.Event()
    first_release = asyncio.Event()
    second_started = asyncio.Event()
    second_release = asyncio.Event()
    original_handler = object()
    current_handler = original_handler
    installed: list[object] = []

    def getsignal(_signal):
        return current_handler

    def set_signal(_signal, handler):
        nonlocal current_handler
        installed.append(handler)
        current_handler = handler

    async def get_files():
        return []

    async def no_op(*_args):
        return None

    def manager_for(started, release):
        async def download(destination: Path):
            started.set()
            await release.wait()
            async with aiofiles.open(destination, "wb") as handle:
                await handle.write(_tar_bytes({}))

        manager = FileSyncManager(
            get_files,
            no_op,
            no_op,
            bulk_download_fn=download,
        )
        manager._pushed_hashes["/_sentinel"] = "0" * 64
        return manager

    monkeypatch.setattr("tools.environments.file_sync.fcntl", None)
    monkeypatch.setattr("tools.environments.file_sync.signal.getsignal", getsignal)
    monkeypatch.setattr("tools.environments.file_sync.signal.signal", set_signal)
    first = asyncio.create_task(
        manager_for(first_started, first_release).sync_back(tmp_path)
    )
    await first_started.wait()
    second = asyncio.create_task(
        manager_for(second_started, second_release).sync_back(tmp_path)
    )
    await asyncio.sleep(0)
    assert not second_started.is_set()

    first_release.set()
    await first
    await second_started.wait()
    second_release.set()
    await second

    assert current_handler is original_handler
    assert len(installed) == 4


@pytest.mark.asyncio
async def test_sync_back_preserves_filtered_mode_and_mtime(tmp_path, monkeypatch):
    host = tmp_path / "skills" / "mode.sh"
    await aiofiles.os.makedirs(host.parent)
    async with aiofiles.open(host, "wb") as handle:
        await handle.write(b"original")
    remote = "/root/.hermes/skills/mode.sh"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        payload = b"#!/bin/sh\necho remote\n"
        info = tarfile.TarInfo("root/.hermes/skills/mode.sh")
        info.size = len(payload)
        info.mode = 0o777
        info.mtime = 123_456
        archive.addfile(info, io.BytesIO(payload))

    async def get_files():
        return [(str(host), remote)]

    async def no_op(*_args):
        return None

    async def download(destination: Path):
        async with aiofiles.open(destination, "wb") as handle:
            await handle.write(buffer.getvalue())

    monkeypatch.setattr(
        "tools.environments.file_sync._credential_host_paths",
        _empty_credentials,
    )
    manager = FileSyncManager(
        get_files,
        no_op,
        no_op,
        bulk_download_fn=download,
    )
    manager._pushed_hashes[remote] = "not-the-current-hash"

    await manager.sync_back(tmp_path)

    result = await aiofiles.os.stat(host)
    assert stat.S_IMODE(result.st_mode) == 0o755
    assert result.st_mtime == pytest.approx(123_456, abs=1)


def test_tar_names_reject_windows_and_posix_traversal():
    with pytest.raises(ValueError, match="unsafe tar member"):
        _safe_tar_name("../escape")
    with pytest.raises(ValueError, match="unsafe tar member"):
        _safe_tar_name(r"..\escape")
    assert _safe_tar_name("root/.hermes/skills/a.md") == (
        "root/.hermes/skills/a.md"
    )
