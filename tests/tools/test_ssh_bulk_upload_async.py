"""Native-async tests for SSH tar streaming and bulk file operations."""

from __future__ import annotations

import asyncio
import io
import sys
import tarfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tools.environments import ssh as ssh_env
from tools.environments.file_sync import quoted_mkdir_command, unique_parent_dirs
from tools.environments.ssh import SSHEnvironment


@pytest.fixture
def environment():
    result = SSHEnvironment(host="example.com", user="testuser")
    result._remote_home = "/home/testuser"
    return result


@pytest.mark.asyncio
async def test_empty_bulk_upload_is_noop(environment, monkeypatch):
    spawn = AsyncMock()
    run = AsyncMock()
    monkeypatch.setattr(environment, "_spawn", spawn)
    monkeypatch.setattr(environment, "_run_captured", run)

    await environment._ssh_bulk_upload([])

    spawn.assert_not_awaited()
    run.assert_not_awaited()


@pytest.mark.asyncio
async def test_bulk_upload_batches_mkdir_and_streams_base_relative_tar(
    environment,
    monkeypatch,
    tmp_path,
):
    first = tmp_path / "local-a.txt"
    second = tmp_path / "local-b.txt"
    first.write_text("aaa")
    second.write_text("bbb")
    archive = tmp_path / "uploaded.tar"
    files = [
        (str(first), "/home/testuser/.hermes/skills/a.txt"),
        (str(second), "/home/testuser/.hermes/credentials/b.txt"),
    ]
    run = AsyncMock(return_value=(0, b"", b""))
    monkeypatch.setattr(environment, "_run_captured", run)
    script = (
        "import pathlib,sys; "
        "pathlib.Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read())"
    )
    monkeypatch.setattr(
        environment,
        "_build_ssh_command",
        lambda extra_args=None: [sys.executable, "-c", script, str(archive)],
    )

    await environment._ssh_bulk_upload(files)

    mkdir_argv = run.await_args.args[0]
    assert mkdir_argv[-1] == quoted_mkdir_command(
        [
            "/home/testuser/.hermes/credentials",
            "/home/testuser/.hermes/skills",
        ]
    )
    with tarfile.open(archive) as uploaded:
        names = {name.removeprefix("./") for name in uploaded.getnames()}
        assert "skills/a.txt" in names
        assert "credentials/b.txt" in names
        assert all(not name.startswith("home/") for name in names)
        assert uploaded.extractfile("./skills/a.txt").read() == b"aaa"
        assert uploaded.extractfile("./credentials/b.txt").read() == b"bbb"
    assert environment._processes == set()


@pytest.mark.asyncio
async def test_bulk_upload_rejects_paths_escaping_sync_base(
    environment,
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "source.txt"
    source.write_text("data")
    monkeypatch.setattr(
        environment,
        "_run_captured",
        AsyncMock(return_value=(0, b"", b"")),
    )

    with pytest.raises(RuntimeError, match="escapes sync base"):
        await environment._ssh_bulk_upload(
            [(str(source), "/home/testuser/outside.txt")]
        )


@pytest.mark.asyncio
async def test_bulk_upload_reports_local_tar_and_remote_tar_errors(
    environment,
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "source.txt"
    source.write_text("data")
    files = [(str(source), "/home/testuser/.hermes/skills/source.txt")]
    monkeypatch.setattr(
        environment,
        "_run_captured",
        AsyncMock(return_value=(0, b"", b"")),
    )

    spawn = AsyncMock()
    monkeypatch.setattr(environment, "_spawn", spawn)
    tar_process = _FakeProcess(returncode=3, stderr=b"tar broke")
    ssh_process = _FakeProcess(returncode=0)
    spawn.side_effect = [tar_process, ssh_process]
    with pytest.raises(RuntimeError, match=r"tar create failed \(rc=3\): tar broke"):
        await environment._ssh_bulk_upload(files)

    tar_process = _FakeProcess(returncode=0)
    ssh_process = _FakeProcess(returncode=4, stderr=b"remote broke")
    spawn.side_effect = [tar_process, ssh_process]
    with pytest.raises(
        RuntimeError,
        match=r"tar extract over SSH failed \(rc=4\): remote broke",
    ):
        await environment._ssh_bulk_upload(files)


@pytest.mark.asyncio
async def test_bulk_upload_timeout_kills_and_reaps_both_processes(
    environment,
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "source.txt"
    source.write_text("data")
    files = [(str(source), "/home/testuser/.hermes/skills/source.txt")]
    monkeypatch.setattr(
        environment,
        "_run_captured",
        AsyncMock(return_value=(0, b"", b"")),
    )
    tar_process = _FakeProcess(returncode=None, never_finishes=True)
    ssh_process = _FakeProcess(returncode=None, never_finishes=True)
    monkeypatch.setattr(
        environment,
        "_spawn",
        AsyncMock(side_effect=[tar_process, ssh_process]),
    )
    real_wait_for = asyncio.wait_for

    async def immediate_timeout(awaitable, timeout):
        if timeout == 120:
            awaitable.cancel()
            await asyncio.gather(awaitable, return_exceptions=True)
            raise TimeoutError
        return await real_wait_for(awaitable, timeout)

    monkeypatch.setattr(ssh_env.asyncio, "wait_for", immediate_timeout)

    with pytest.raises(RuntimeError, match="SSH bulk upload timed out"):
        await environment._ssh_bulk_upload(files)

    assert tar_process.killed and tar_process.waited
    assert ssh_process.killed and ssh_process.waited


@pytest.mark.asyncio
async def test_bulk_upload_cancellation_kills_and_reaps_both_processes(
    environment,
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "source.txt"
    source.write_text("data")
    files = [(str(source), "/home/testuser/.hermes/skills/source.txt")]
    monkeypatch.setattr(
        environment,
        "_run_captured",
        AsyncMock(return_value=(0, b"", b"")),
    )
    tar_process = _FakeProcess(returncode=None, never_finishes=True)
    ssh_process = _FakeProcess(returncode=None, never_finishes=True)
    monkeypatch.setattr(
        environment,
        "_spawn",
        AsyncMock(side_effect=[tar_process, ssh_process]),
    )

    upload = asyncio.create_task(environment._ssh_bulk_upload(files))
    await asyncio.wait_for(tar_process.wait_started.wait(), timeout=1)
    await asyncio.wait_for(ssh_process.wait_started.wait(), timeout=1)
    upload.cancel()
    with pytest.raises(asyncio.CancelledError):
        await upload

    assert tar_process.killed and tar_process.waited
    assert ssh_process.killed and ssh_process.waited


@pytest.mark.asyncio
async def test_ssh_spawn_failure_kills_and_reaps_local_tar(
    environment,
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "source.txt"
    source.write_text("data")
    files = [(str(source), "/home/testuser/.hermes/skills/source.txt")]
    monkeypatch.setattr(
        environment,
        "_run_captured",
        AsyncMock(return_value=(0, b"", b"")),
    )
    tar_process = _FakeProcess(returncode=None, never_finishes=True)
    monkeypatch.setattr(
        environment,
        "_spawn",
        AsyncMock(side_effect=[tar_process, OSError("SSH binary not found")]),
    )

    with pytest.raises(OSError, match="SSH binary not found"):
        await environment._ssh_bulk_upload(files)

    assert tar_process.killed and tar_process.waited


@pytest.mark.asyncio
async def test_bulk_download_streams_archive_and_preserves_error_shape(
    environment,
    monkeypatch,
    tmp_path,
):
    destination = tmp_path / "remote.tar"
    process = _FakeProcess(returncode=0, stdout=b"archive bytes")
    monkeypatch.setattr(environment, "_spawn", AsyncMock(return_value=process))

    await environment._ssh_bulk_download(destination)

    assert destination.read_bytes() == b"archive bytes"
    assert environment._spawn.await_args.args[0][-1] == (
        "tar cf - -C / home/testuser/.hermes"
    )

    process = _FakeProcess(returncode=9, stderr=b"download broke")
    monkeypatch.setattr(environment, "_spawn", AsyncMock(return_value=process))
    with pytest.raises(RuntimeError, match="SSH bulk download failed: download broke"):
        await environment._ssh_bulk_download(destination)


def test_retained_bulk_helpers_preserve_shapes():
    assert quoted_mkdir_command(["/a", "/b/c"]) == "mkdir -p /a /b/c"
    assert unique_parent_dirs([]) == []
    assert unique_parent_dirs(
        [("a", "/z/b"), ("b", "/a/c"), ("c", "/z/d")]
    ) == ["/a", "/z"]


class _FakeWriter:
    def __init__(self):
        self.closed = False

    def write(self, _data):
        return None

    async def drain(self):
        return None

    def close(self):
        self.closed = True


class _FakeProcess:
    _next_pid = 1000

    def __init__(
        self,
        *,
        returncode: int | None,
        stdout: bytes = b"",
        stderr: bytes = b"",
        never_finishes: bool = False,
    ):
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.returncode = returncode
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_data(stdout)
        self.stdout.feed_eof()
        self.stderr = asyncio.StreamReader()
        self.stderr.feed_data(stderr)
        self.stderr.feed_eof()
        self.stdin = _FakeWriter()
        self._never_finishes = never_finishes
        self.killed = False
        self.waited = False
        self.wait_started = asyncio.Event()

    async def wait(self):
        self.waited = True
        self.wait_started.set()
        if self._never_finishes and not self.killed:
            await asyncio.Future()
        if self.returncode is None:
            self.returncode = -9
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9
