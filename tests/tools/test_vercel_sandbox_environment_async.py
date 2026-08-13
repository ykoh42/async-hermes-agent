"""Native-async parity tests for the Vercel Sandbox backend."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import re
import sys
import types
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace

import aiofiles
import httpx
import pytest
import pytest_asyncio


class _FakeRunResult:
    def __init__(self, output: str | bytes = "", exit_code: int = 0):
        self._output = output
        self.exit_code = exit_code

    async def output(self) -> str | bytes:
        return self._output


class _FakeSandboxStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    ABORTED = "aborted"
    SNAPSHOTTING = "snapshotting"


@dataclass(frozen=True)
class _FakeSnapshot:
    snapshot_id: str


class _FakeClient:
    def __init__(self):
        self.closed = 0

    async def aclose(self):
        self.closed += 1


class _FakeSandbox:
    def __init__(
        self,
        *,
        cwd: str = "/vercel/sandbox",
        home: str = "/home/vercel",
        status: _FakeSandboxStatus = _FakeSandboxStatus.RUNNING,
    ):
        self.sandbox = SimpleNamespace(cwd=cwd, id="sb-123")
        self.status = status
        self.home = home
        self.client = _FakeClient()
        self.run_command_calls: list[tuple[str, list[str], dict]] = []
        self.run_command_side_effects: list[object] = []
        self.write_files_calls: list[list[dict[str, object]]] = []
        self.write_files_side_effects: list[object] = []
        self.download_file_calls: list[tuple[str, Path]] = []
        self.download_file_content = b""
        self.stop_calls: list[dict] = []
        self.snapshot_calls = 0
        self.snapshot_side_effects: list[object] = []
        self.snapshot_id = "snap_default"
        self.refresh_calls = 0
        self.refresh_side_effects: list[object] = []
        self.wait_for_status_calls: list[tuple[object, object, object]] = []
        self.wait_for_status_side_effects: list[object] = []

    async def refresh(self) -> None:
        self.refresh_calls += 1
        if self.refresh_side_effects:
            effect = self.refresh_side_effects.pop(0)
            if isinstance(effect, BaseException):
                raise effect

    async def wait_for_status(self, status, *, timeout, poll_interval) -> None:
        self.wait_for_status_calls.append((status, timeout, poll_interval))
        if self.wait_for_status_side_effects:
            effect = self.wait_for_status_side_effects.pop(0)
            if isinstance(effect, BaseException):
                raise effect
        self.status = _FakeSandboxStatus(status)

    async def run_command(self, cmd: str, args=None, **kwargs):
        args = list(args or [])
        self.run_command_calls.append((cmd, args, kwargs))
        if self.run_command_side_effects:
            effect = self.run_command_side_effects.pop(0)
            if isinstance(effect, BaseException):
                raise effect
            if callable(effect):
                value = effect(cmd, args, kwargs)
                if inspect.isawaitable(value):
                    return await value
                return value
            return effect
        script = args[1] if len(args) > 1 else ""
        if 'printf %s "$HOME"' in script:
            return _FakeRunResult(self.home)
        return _FakeRunResult("")

    async def write_files(self, files):
        self.write_files_calls.append(files)
        if self.write_files_side_effects:
            effect = self.write_files_side_effects.pop(0)
            if isinstance(effect, BaseException):
                raise effect

    async def download_file(self, remote_path: str, local_path) -> str:
        destination = Path(local_path)
        self.download_file_calls.append((remote_path, destination))
        async with aiofiles.open(destination, "wb") as handle:
            await handle.write(self.download_file_content)
        return str(destination)

    async def stop(self, **kwargs) -> None:
        self.stop_calls.append(kwargs)
        self.status = _FakeSandboxStatus.STOPPED

    async def snapshot(self):
        self.snapshot_calls += 1
        if self.snapshot_side_effects:
            effect = self.snapshot_side_effects.pop(0)
            if isinstance(effect, BaseException):
                raise effect
            if callable(effect):
                value = effect()
                if inspect.isawaitable(value):
                    return await value
                return value
            return effect
        return _FakeSnapshot(self.snapshot_id)


@dataclass(frozen=True)
class _FakeResources:
    vcpus: int | None = None
    memory: int | None = None


class _FakeSDK:
    def __init__(self):
        self.create_kwargs: list[dict] = []
        self.create_side_effects: list[object] = []
        self.sandboxes: list[_FakeSandbox] = []

    @property
    def current(self):
        return self.sandboxes[-1]

    async def create(self, **kwargs):
        self.create_kwargs.append(kwargs)
        if self.create_side_effects:
            effect = self.create_side_effects.pop(0)
            if isinstance(effect, BaseException):
                raise effect
            sandbox = effect
        else:
            sandbox = _FakeSandbox()
        self.sandboxes.append(sandbox)
        return sandbox


def _cwd_result(
    body: str = "",
    *,
    cwd: str = "/vercel/sandbox",
    exit_code: int = 0,
):
    def result(_cmd: str, args: list[str], _kwargs: dict):
        script = args[1] if len(args) > 1 else ""
        match = re.search(r"__HERMES_CWD_[A-Za-z0-9]+__", script)
        marker = match.group(0) if match else "__HERMES_CWD_MISSING__"
        prefix = f"{body}\n\n" if body else "\n"
        return _FakeRunResult(
            f"{prefix}{marker}{cwd}{marker}\n",
            exit_code,
        )

    return result


@pytest.fixture
def vercel_sdk(monkeypatch):
    sdk = _FakeSDK()
    sandbox_module = types.ModuleType("vercel.sandbox")
    sandbox_module.AsyncSandbox = SimpleNamespace(create=sdk.create)
    sandbox_module.Resources = _FakeResources
    sandbox_module.WriteFile = dict
    sandbox_module.SandboxStatus = _FakeSandboxStatus
    vercel_module = types.ModuleType("vercel")
    vercel_module.sandbox = sandbox_module
    monkeypatch.setitem(sys.modules, "vercel", vercel_module)
    monkeypatch.setitem(sys.modules, "vercel.sandbox", sandbox_module)
    return sdk


@pytest.fixture
def vercel_module(vercel_sdk, monkeypatch):
    module = importlib.import_module("tools.environments.vercel_sandbox")
    module = importlib.reload(module)
    monkeypatch.setattr(module, "_ensure_vercel_sdk", lambda: None)

    async def no_files(_container_base="/root/.hermes", **_kwargs):
        return []

    async def no_credentials():
        return set()

    monkeypatch.setattr(module, "iter_sync_files", no_files)
    monkeypatch.setattr(
        "tools.environments.file_sync._credential_host_paths",
        no_credentials,
    )
    return module


@pytest_asyncio.fixture
async def make_env(vercel_module):
    environments = []

    async def factory(**kwargs):
        kwargs.setdefault("runtime", "node22")
        kwargs.setdefault("cwd", vercel_module.DEFAULT_VERCEL_CWD)
        kwargs.setdefault("timeout", 30)
        kwargs.setdefault("task_id", "task-123")
        environment = vercel_module.VercelSandboxEnvironment(**kwargs)
        environments.append(environment)
        await environment._ensure_initialized()
        return environment

    yield factory

    for environment in environments:
        environment._sync_manager = None
        await environment.cleanup()


def test_constructor_is_state_only_and_preserves_signature(vercel_module, vercel_sdk):
    signature = inspect.signature(vercel_module.VercelSandboxEnvironment)
    assert list(signature.parameters) == [
        "runtime",
        "cwd",
        "timeout",
        "cpu",
        "memory",
        "disk",
        "persistent_filesystem",
        "task_id",
    ]
    environment = vercel_module.VercelSandboxEnvironment(runtime="node22")
    assert environment._sandbox is None
    assert environment._sync_manager is None
    assert environment._create_params is None
    assert vercel_sdk.create_kwargs == []
    with pytest.raises(ValueError, match="does not support configurable"):
        vercel_module.VercelSandboxEnvironment(disk=1024)


@pytest.mark.asyncio
async def test_startup_cwd_resources_and_pending_error(
    make_env,
    vercel_sdk,
):
    vercel_sdk.create_side_effects.append(_FakeSandbox(cwd="/workspace"))
    environment = await make_env(cpu=1.9, memory=2048)
    assert environment.cwd == "/workspace"
    assert vercel_sdk.create_kwargs[-1]["resources"] == _FakeResources(
        vcpus=1,
        memory=2048,
    )

    vercel_sdk.create_side_effects.append(_FakeSandbox(home="/home/custom"))
    tilde = await make_env(cwd="~")
    assert tilde.cwd == "/home/custom"

    pending = _FakeSandbox(status=_FakeSandboxStatus.PENDING)
    pending.wait_for_status_side_effects.append(TimeoutError("pending"))
    vercel_sdk.create_side_effects.append(pending)
    with pytest.raises(RuntimeError, match="Sandbox did not reach running state"):
        await make_env(task_id="pending")


@pytest.mark.asyncio
async def test_initial_and_pre_execute_file_sync(
    make_env,
    vercel_module,
    vercel_sdk,
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "token.txt"
    source.write_text("secret-token")

    async def files(_container_base="/root/.hermes", **_kwargs):
        return [(str(source), "/home/vercel/.hermes/credentials/token.txt")]

    monkeypatch.setattr(vercel_module, "iter_sync_files", files)
    environment = await make_env()
    assert vercel_sdk.current.write_files_calls[0] == [
        {
            "path": "/home/vercel/.hermes/credentials/token.txt",
            "content": b"secret-token",
        }
    ]

    source.write_text("updated-token")
    monkeypatch.setenv("HERMES_FORCE_FILE_SYNC", "1")
    vercel_sdk.current.run_command_side_effects.append(_cwd_result("hello"))
    result = await environment.execute("echo hello")
    assert result == {"output": "hello\n", "returncode": 0}
    assert vercel_sdk.current.write_files_calls[-1][0]["content"] == b"updated-token"


@pytest.mark.asyncio
async def test_transient_create_and_write_retry(
    make_env,
    vercel_sdk,
    monkeypatch,
    tmp_path,
):
    transient = httpx.ReadError("retry")
    vercel_sdk.create_side_effects.extend([transient, _FakeSandbox()])
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    environment = await make_env(task_id="retry-create")
    assert len(vercel_sdk.create_kwargs) == 2

    sandbox = environment._sandbox
    assert sandbox is not None
    sandbox.write_files_side_effects.extend([httpx.ReadError("retry"), None])
    source = tmp_path / "source.txt"
    source.write_text("payload")
    await environment._vercel_bulk_upload([(str(source), "/remote/source.txt")])
    assert len(sandbox.write_files_calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["terminal", "refresh"])
async def test_execute_recreates_unhealthy_sandbox(
    make_env,
    vercel_sdk,
    failure,
):
    environment = await make_env()
    original = vercel_sdk.current
    if failure == "terminal":
        original.status = _FakeSandboxStatus.STOPPED
    else:
        original.refresh_side_effects.append(RuntimeError("refresh failed"))
    replacement = _FakeSandbox()
    replacement.run_command_side_effects.extend(
        [_FakeRunResult(replacement.home), _cwd_result("hello")]
    )
    vercel_sdk.create_side_effects.append(replacement)

    result = await environment.execute("echo hello")

    assert result == {"output": "hello\n", "returncode": 0}
    assert original.client.closed == 1
    assert environment._sandbox is replacement


@pytest.mark.asyncio
async def test_execute_argv_timeout_and_cancellation(make_env, vercel_sdk):
    environment = await make_env()
    sandbox = vercel_sdk.current
    sandbox.run_command_side_effects.append(_cwd_result("done"))
    result = await environment.execute("echo done")
    assert result == {"output": "done\n", "returncode": 0}
    command, args, kwargs = sandbox.run_command_calls[-1]
    assert command == "bash"
    assert args[0] == "-c"
    assert kwargs["cwd"] == "/vercel/sandbox"

    started = asyncio.Event()

    async def blocked(*_args):
        started.set()
        await asyncio.Future()

    sandbox.run_command_side_effects.append(blocked)
    timed_out = await environment.execute("sleep 30", timeout=0.01)
    assert timed_out == {
        "output": "[Command timed out after 0.01s]",
        "returncode": 124,
    }
    assert sandbox.stop_calls

    replacement = _FakeSandbox()
    replacement.run_command_side_effects.extend(
        [_FakeRunResult(replacement.home), _cwd_result()]
    )
    vercel_sdk.create_side_effects.append(replacement)
    await environment.execute("true")
    started = asyncio.Event()
    replacement.run_command_side_effects.append(blocked)
    running = asyncio.create_task(environment.execute("sleep 30", timeout=30))
    await asyncio.wait_for(started.wait(), timeout=1)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert replacement.stop_calls


@pytest.mark.asyncio
async def test_execute_bounded_capture_spills_full_sdk_result(
    make_env,
    vercel_sdk,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr("tools.tool_output_limits.get_max_bytes", lambda: 100)
    environment = await make_env()
    sandbox = vercel_sdk.current
    raw = "HEAD-" + ("x" * 240) + "-TAIL"
    sandbox.run_command_side_effects.append(_cwd_result(raw))

    result = await environment.execute(
        "generate output",
        bounded_capture=True,
    )

    assert result["returncode"] == 0
    assert result["output_total_chars"] > len(raw)
    assert len(result["output"]) <= 100
    assert "[OUTPUT TRUNCATED" in result["output"]
    spill = Path(result["full_output_path"])
    assert spill.read_text().startswith(raw)


@pytest.mark.asyncio
async def test_snapshot_restore_fallback_and_cleanup(
    make_env,
    vercel_module,
    vercel_sdk,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    await vercel_module._store_snapshot("task-123", "snap_saved")
    restored = _FakeSandbox(cwd="/restored")
    vercel_sdk.create_side_effects.append(restored)
    environment = await make_env()
    assert environment.cwd == "/restored"
    assert vercel_sdk.create_kwargs[-1]["source"] == {
        "type": "snapshot",
        "snapshot_id": "snap_saved",
    }
    restored.snapshot_id = "snap_cleanup"
    environment._sync_manager = None
    await environment.cleanup()
    await environment.cleanup()
    assert restored.snapshot_calls == 1
    assert len(restored.stop_calls) == 1
    assert restored.client.closed == 1
    assert await vercel_module._load_snapshots() == {
        "task-123": "snap_cleanup"
    }

    await vercel_module._store_snapshot("stale", "snap_stale")
    vercel_sdk.create_side_effects.extend(
        [RuntimeError("snapshot missing"), _FakeSandbox(cwd="/fresh")]
    )
    fresh = await make_env(task_id="stale")
    assert fresh.cwd == "/fresh"
    assert "source" in vercel_sdk.create_kwargs[-2]
    assert "source" not in vercel_sdk.create_kwargs[-1]
    assert "stale" not in await vercel_module._load_snapshots()


@pytest.mark.asyncio
async def test_nonpersistent_cleanup_and_snapshot_failure_still_dispose(
    make_env,
    vercel_sdk,
):
    nonpersistent = await make_env(
        task_id="nonpersistent",
        persistent_filesystem=False,
    )
    sandbox = vercel_sdk.current
    nonpersistent._sync_manager = None
    await nonpersistent.cleanup()
    assert sandbox.snapshot_calls == 0
    assert len(sandbox.stop_calls) == 1
    assert sandbox.client.closed == 1

    failing = await make_env(task_id="snapshot-fails")
    sandbox = vercel_sdk.current
    sandbox.snapshot_side_effects.append(RuntimeError("snapshot failed"))
    failing._sync_manager = None
    await failing.cleanup()
    assert sandbox.snapshot_calls == 1
    assert len(sandbox.stop_calls) == 1
    assert sandbox.client.closed == 1


@pytest.mark.asyncio
async def test_bulk_file_operations_preserve_commands_and_errors(
    make_env,
    vercel_sdk,
    tmp_path,
):
    environment = await make_env()
    sandbox = vercel_sdk.current
    source = tmp_path / "source.txt"
    source.write_bytes(b"payload")
    await environment._vercel_bulk_upload(
        [(str(source), "/home/vercel/.hermes/skills/source.txt")]
    )
    assert sandbox.write_files_calls[-1] == [
        {
            "path": "/home/vercel/.hermes/skills/source.txt",
            "content": b"payload",
        }
    ]

    await environment._vercel_delete(["/a b", "/c"])
    assert sandbox.run_command_calls[-1][1] == ["-lc", "rm -f '/a b' /c"]
    sandbox.run_command_side_effects.append(_FakeRunResult("denied", 2))
    with pytest.raises(RuntimeError, match="Vercel delete failed: denied"):
        await environment._vercel_delete(["/a"])

    destination = tmp_path / "remote.tar"
    sandbox.download_file_content = b"tar bytes"
    await environment._vercel_bulk_download(destination)
    assert destination.read_bytes() == b"tar bytes"
    assert sandbox.download_file_calls[-1][0].startswith("/tmp/.hermes_sync.")
    create_tar = sandbox.run_command_calls[-2]
    cleanup_tar = sandbox.run_command_calls[-1]
    assert "tar cf" in create_tar[1][1]
    assert "home/vercel/.hermes" in create_tar[1][1]
    assert cleanup_tar[1][1].startswith("rm -f /tmp/.hermes_sync.")


@pytest.mark.asyncio
async def test_cleanup_sync_back_failure_does_not_skip_disposal(
    make_env,
    vercel_sdk,
):
    environment = await make_env()
    sandbox = vercel_sdk.current

    class FailingSyncManager:
        async def sync_back(self):
            raise RuntimeError("download failed")

    environment._sync_manager = FailingSyncManager()
    await environment.cleanup()
    assert sandbox.snapshot_calls == 1
    assert len(sandbox.stop_calls) == 1
    assert sandbox.client.closed == 1


@pytest.mark.asyncio
async def test_cleanup_cancellation_finishes_snapshot_stop_and_client_close(
    make_env,
    vercel_sdk,
):
    environment = await make_env(task_id="cancel-cleanup")
    sandbox = vercel_sdk.current
    environment._sync_manager = None
    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed_snapshot():
        started.set()
        await release.wait()
        return _FakeSnapshot("snap-after-cancel")

    sandbox.snapshot_side_effects.append(delayed_snapshot)
    cleanup = asyncio.create_task(environment.cleanup())
    await asyncio.wait_for(started.wait(), timeout=1)
    cleanup.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await cleanup

    assert sandbox.snapshot_calls == 1
    assert len(sandbox.stop_calls) == 1
    assert sandbox.client.closed == 1


async def _no_sleep(_delay):
    return None
