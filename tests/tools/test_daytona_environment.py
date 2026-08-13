"""Focused parity and lifecycle tests for the native-async Daytona backend."""

from __future__ import annotations

import asyncio
import enum
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiofiles
import pytest


class _SandboxState(str, enum.Enum):
    STARTED = "started"
    STOPPED = "stopped"
    ARCHIVED = "archived"
    ERROR = "error"


class _DaytonaError(Exception):
    pass


class _Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Params:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Resources:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FileUpload:
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination


class _Process:
    def __init__(self, home="/root"):
        self.home = home
        self.calls: list[tuple[str, int | float | None]] = []
        self.response_for = None

    async def exec(self, command, timeout=None):
        self.calls.append((command, timeout))
        if self.response_for is not None:
            return await self.response_for(command, timeout)
        if command == "echo $HOME":
            return SimpleNamespace(result=self.home, exit_code=0)
        return SimpleNamespace(result="", exit_code=0)


class _FileSystem:
    def __init__(self):
        self.upload_file = AsyncMock()
        self.upload_files = AsyncMock()
        self.download_file = AsyncMock(return_value=b"archive-bytes")


class _Sandbox:
    def __init__(self, sandbox_id="sb-123", home="/root"):
        self.id = sandbox_id
        self.state = _SandboxState.STARTED
        self.process = _Process(home)
        self.fs = _FileSystem()
        self.start = AsyncMock()
        self.stop = AsyncMock()
        self.refresh_data = AsyncMock()


class _Paginated:
    def __init__(self, items=()):
        self.items = list(items)


class _Snapshot:
    def __init__(self):
        self._target = "uninitialized"


class _Client:
    instances: list["_Client"] = []
    next_instance: "_Client | None" = None

    def __init__(self, config):
        self.config = config
        self._target = config.target
        self.snapshot = _Snapshot()
        self.get = AsyncMock(side_effect=_DaytonaError("not found"))
        self.list = AsyncMock(return_value=_Paginated())
        self.create = AsyncMock(return_value=_Sandbox())
        self.delete = AsyncMock()
        self.close = AsyncMock()
        _Client.instances.append(self)
        if _Client.next_instance is not None:
            replacement = _Client.next_instance
            self.__dict__.update(replacement.__dict__)
            _Client.next_instance = None


def _install_sdk(monkeypatch):
    daytona = types.ModuleType("daytona")
    daytona.AsyncDaytona = _Client
    daytona.DaytonaConfig = _Config
    daytona.DaytonaError = _DaytonaError
    daytona.CreateSandboxFromImageParams = _Params
    daytona.Resources = _Resources
    daytona.SandboxState = _SandboxState

    common = types.ModuleType("daytona.common")
    filesystem = types.ModuleType("daytona.common.filesystem")
    filesystem.FileUpload = _FileUpload
    monkeypatch.setitem(sys.modules, "daytona", daytona)
    monkeypatch.setitem(sys.modules, "daytona.common", common)
    monkeypatch.setitem(sys.modules, "daytona.common.filesystem", filesystem)
    _Client.instances.clear()
    _Client.next_instance = None
    return daytona


@pytest.fixture
def daytona_sdk(monkeypatch):
    return _install_sdk(monkeypatch)


@pytest.fixture
def daytona_module(daytona_sdk, monkeypatch):
    from tools.environments import daytona as module

    values = {
        "DAYTONA_API_KEY": "test-key",
        "DAYTONA_API_URL": "https://daytona.example/api",
    }

    async def _setting(name):
        return values.get(name, "")

    monkeypatch.setattr(module, "_read_daytona_setting", _setting)
    monkeypatch.setattr(module, "iter_sync_files", AsyncMock(return_value=[]))
    return module


async def _initialized_environment(
    module,
    *,
    sandbox=None,
    persistent=True,
    **kwargs,
):
    sandbox = sandbox or _Sandbox()
    client = _Client(_Config(target="placeholder"))
    client.create = AsyncMock(return_value=sandbox)
    client.get = AsyncMock(side_effect=_DaytonaError("not found"))
    client.list = AsyncMock(return_value=_Paginated())
    _Client.next_instance = client
    environment = module.DaytonaEnvironment(
        image="test-image:latest",
        persistent_filesystem=persistent,
        **kwargs,
    )
    await environment.init_session()
    return environment, client, sandbox


@pytest.mark.asyncio
async def test_constructor_is_state_only_and_sdk_constructor_skips_sync_env(
    daytona_module,
):
    environment = daytona_module.DaytonaEnvironment(image="test-image:latest")

    assert _Client.instances == []
    assert environment._daytona is None
    await environment.init_session()

    constructed = _Client.instances[-1]
    assert constructed.config.api_key == "test-key"
    assert constructed.config.api_url == "https://daytona.example/api"
    assert constructed.config.target == "__hermes_daytona_default_target__"
    assert constructed._target is None
    assert constructed.snapshot._target is None


@pytest.mark.asyncio
async def test_setting_resolution_does_not_escape_profile_scope(
    daytona_sdk, monkeypatch
):
    from agent.secret_scope import UnscopedSecretError
    from hermes_cli import config
    from tools.environments import daytona as module

    monkeypatch.setenv("DAYTONA_API_KEY", "foreign-key")
    monkeypatch.setattr(
        config,
        "get_env_value_prefer_dotenv",
        AsyncMock(side_effect=UnscopedSecretError("scope required")),
    )

    with pytest.raises(UnscopedSecretError):
        await module._read_daytona_setting("DAYTONA_API_KEY")


@pytest.mark.asyncio
async def test_default_cwd_resolves_remote_home(daytona_module):
    sandbox = _Sandbox(home="/home/testuser")
    environment, _client, _sandbox = await _initialized_environment(
        daytona_module, sandbox=sandbox
    )

    assert environment.cwd == "/home/testuser"


@pytest.mark.asyncio
async def test_persistent_resumes_by_name_without_create(daytona_module):
    existing = _Sandbox("sb-existing")
    client = _Client(_Config(target="placeholder"))
    client.get = AsyncMock(return_value=existing)
    client.list = AsyncMock(return_value=_Paginated())
    client.create = AsyncMock()
    _Client.next_instance = client
    environment = daytona_module.DaytonaEnvironment(
        image="test-image:latest",
        persistent_filesystem=True,
        task_id="mytask",
    )

    await environment.init_session()

    client.get.assert_awaited_once_with("hermes-mytask")
    existing.start.assert_awaited_once()
    client.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_paginated_result_is_resumed(daytona_module):
    legacy = _Sandbox("sb-legacy")
    client = _Client(_Config(target="placeholder"))
    client.get = AsyncMock(side_effect=_DaytonaError("not found"))
    client.list = AsyncMock(return_value=_Paginated([legacy]))
    client.create = AsyncMock()
    _Client.next_instance = client
    environment = daytona_module.DaytonaEnvironment(
        image="test-image:latest", task_id="legacy-task"
    )

    await environment.init_session()

    client.list.assert_awaited_once_with(
        labels={"hermes_task_id": "legacy-task"}, limit=1
    )
    legacy.start.assert_awaited_once()
    client.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_nonpersistent_skips_lookup_and_preserves_resource_conversion(
    daytona_module,
):
    environment, client, _sandbox = await _initialized_environment(
        daytona_module,
        persistent=False,
        cpu=2,
        memory=5120,
        disk=20000,
    )

    client.get.assert_not_awaited()
    client.list.assert_not_awaited()
    params = client.create.await_args.args[0]
    assert params.image == "test-image:latest"
    assert params.auto_stop_interval == 0
    assert params.resources.cpu == 2
    assert params.resources.memory == 5
    assert params.resources.disk == 10
    assert environment._sandbox is not None


@pytest.mark.asyncio
async def test_execute_preserves_shell_command_and_result_contract(daytona_module):
    sandbox = _Sandbox()

    async def _response(command, timeout):
        if command == "echo $HOME":
            return SimpleNamespace(result="/root", exit_code=0)
        if "echo hello" in command:
            return SimpleNamespace(result="hello", exit_code=0)
        return SimpleNamespace(result="", exit_code=0)

    sandbox.process.response_for = _response
    environment, _client, _sandbox = await _initialized_environment(
        daytona_module, sandbox=sandbox
    )

    result = await environment.execute("echo hello")

    assert result["returncode"] == 0
    assert result["output"] == "hello"
    command, timeout = sandbox.process.calls[-1]
    assert command.startswith("bash -c ")
    assert "echo hello" in command
    assert timeout == 60


@pytest.mark.asyncio
async def test_sdk_error_preserves_upstream_empty_rc1_result(daytona_module):
    environment, _client, sandbox = await _initialized_environment(daytona_module)

    async def _fail(_command, _timeout):
        raise _DaytonaError("transient")

    sandbox.process.response_for = _fail

    result = await environment._run_bash("echo x")

    assert result == {"output": "", "returncode": 1}


@pytest.mark.asyncio
async def test_timeout_stops_sandbox_and_returns_124(daytona_module):
    environment, _client, sandbox = await _initialized_environment(daytona_module)

    async def _block(_command, _timeout):
        await asyncio.Event().wait()

    sandbox.process.response_for = _block

    result = await environment._run_bash("sleep 10", timeout=0.02)

    assert result == {
        "output": "[Command timed out after 0.02s]",
        "returncode": 124,
    }
    sandbox.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancellation_stops_sandbox_then_reraises(daytona_module):
    environment, _client, sandbox = await _initialized_environment(daytona_module)
    started = asyncio.Event()

    async def _block(_command, _timeout):
        started.set()
        await asyncio.Event().wait()

    sandbox.process.response_for = _block
    command = asyncio.create_task(environment._run_bash("sleep 10"))
    await asyncio.wait_for(started.wait(), timeout=1)

    command.cancel()
    with pytest.raises(asyncio.CancelledError):
        await command

    sandbox.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_interrupt_stops_sandbox_and_returns_upstream_130(daytona_module):
    from tools.interrupt import _bind_interrupt_event, _reset_interrupt_event

    environment, _client, sandbox = await _initialized_environment(daytona_module)
    started = asyncio.Event()

    async def _block(_command, _timeout):
        started.set()
        await asyncio.Future()

    sandbox.process.response_for = _block
    interrupt = asyncio.Event()
    token = _bind_interrupt_event(interrupt)
    try:
        command = asyncio.create_task(environment._run_bash("sleep 10"))
        await asyncio.wait_for(started.wait(), timeout=1)
        interrupt.set()
        result = await asyncio.wait_for(command, timeout=1)
    finally:
        _reset_interrupt_event(token)

    assert result == {"output": "\n[Command interrupted]", "returncode": 130}
    sandbox.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_natural_daytona_rc130_has_no_interrupt_marker(daytona_module):
    environment, _client, sandbox = await _initialized_environment(daytona_module)
    sandbox.process.response_for = AsyncMock(
        return_value=SimpleNamespace(result="natural", exit_code=130)
    )

    result = await environment._run_bash("exit 130")

    assert result == {"output": "natural", "returncode": 130}
    sandbox.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_stopped_sandbox_restarts_before_file_sync(daytona_module):
    environment, _client, sandbox = await _initialized_environment(daytona_module)
    sandbox.state = _SandboxState.STOPPED
    environment._sync_manager.sync = AsyncMock()

    await environment._before_execute()

    sandbox.refresh_data.assert_awaited_once()
    sandbox.start.assert_awaited_once()
    environment._sync_manager.sync.assert_awaited_once()


@pytest.mark.asyncio
async def test_single_upload_uses_quoted_parent_and_async_bytes(
    daytona_module, tmp_path
):
    environment, _client, sandbox = await _initialized_environment(daytona_module)
    host_file = tmp_path / "token.txt"
    async with aiofiles.open(host_file, "wb") as handle:
        await handle.write(b"secret")
    sandbox.process.calls.clear()
    remote_path = "/root/.hermes/skills/evil; touch /tmp/owned/file.txt"

    await environment._daytona_upload(str(host_file), remote_path)

    assert sandbox.process.calls[0][0] == (
        "mkdir -p '/root/.hermes/skills/evil; touch /tmp/owned'"
    )
    sandbox.fs.upload_file.assert_awaited_once_with(b"secret", remote_path)


@pytest.mark.asyncio
async def test_bulk_upload_reads_bytes_and_preserves_order(daytona_module, tmp_path):
    environment, _client, sandbox = await _initialized_environment(daytona_module)
    first = tmp_path / "first"
    second = tmp_path / "second"
    for path, content in ((first, b"one"), (second, b"two")):
        async with aiofiles.open(path, "wb") as handle:
            await handle.write(content)

    await environment._daytona_bulk_upload(
        [(str(first), "/remote/a"), (str(second), "/remote/b")]
    )

    uploads = sandbox.fs.upload_files.await_args.args[0]
    assert [(item.source, item.destination) for item in uploads] == [
        (b"one", "/remote/a"),
        (b"two", "/remote/b"),
    ]


@pytest.mark.asyncio
async def test_bulk_download_writes_bytes_and_removes_remote_tar(
    daytona_module, tmp_path
):
    environment, _client, sandbox = await _initialized_environment(daytona_module)
    destination = tmp_path / "remote.tar"
    sandbox.process.calls.clear()

    await environment._daytona_bulk_download(destination)

    async with aiofiles.open(destination, "rb") as handle:
        assert await handle.read() == b"archive-bytes"
    remote_path = sandbox.fs.download_file.await_args.args[0]
    assert sandbox.process.calls[0][0].startswith("tar cf ")
    assert sandbox.process.calls[-1][0] == f"rm -f {remote_path}"


@pytest.mark.asyncio
async def test_persistent_cleanup_syncs_stops_and_closes(daytona_module):
    environment, client, sandbox = await _initialized_environment(daytona_module)
    environment._sync_manager.sync_back = AsyncMock()

    await environment.cleanup()

    environment._sync_manager.sync_back.assert_awaited_once()
    sandbox.stop.assert_awaited_once()
    client.delete.assert_not_awaited()
    client.close.assert_awaited_once()
    assert environment._sandbox is None
    assert environment._daytona is None


@pytest.mark.asyncio
async def test_nonpersistent_cleanup_deletes_and_closes(daytona_module):
    environment, client, sandbox = await _initialized_environment(
        daytona_module, persistent=False
    )
    environment._sync_manager.sync_back = AsyncMock()

    await environment.cleanup()

    client.delete.assert_awaited_once_with(sandbox)
    sandbox.stop.assert_not_awaited()
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_cleanup_cancellation_finishes_owned_resources(daytona_module):
    environment, client, sandbox = await _initialized_environment(daytona_module)
    environment._sync_manager.sync_back = AsyncMock()
    stop_started = asyncio.Event()
    release_stop = asyncio.Event()

    async def _stop():
        stop_started.set()
        await release_stop.wait()

    sandbox.stop = AsyncMock(side_effect=_stop)
    cleanup = asyncio.create_task(environment.cleanup())
    await asyncio.wait_for(stop_started.wait(), timeout=1)

    cleanup.cancel()
    release_stop.set()
    with pytest.raises(asyncio.CancelledError):
        await cleanup

    client.close.assert_awaited_once()
    assert environment._sandbox is None


@pytest.mark.asyncio
async def test_cancelled_lazy_init_deletes_fresh_sandbox_and_closes_client(
    daytona_module,
):
    sandbox = _Sandbox()
    home_started = asyncio.Event()

    async def _block_home(command, timeout):
        if command == "echo $HOME":
            home_started.set()
            await asyncio.Event().wait()
        return SimpleNamespace(result="", exit_code=0)

    sandbox.process.response_for = _block_home
    client = _Client(_Config(target="placeholder"))
    client.create = AsyncMock(return_value=sandbox)
    client.get = AsyncMock(side_effect=_DaytonaError("not found"))
    client.list = AsyncMock(return_value=_Paginated())
    _Client.next_instance = client
    environment = daytona_module.DaytonaEnvironment(image="test-image:latest")
    initialization = asyncio.create_task(environment.init_session())
    await asyncio.wait_for(home_started.wait(), timeout=1)

    initialization.cancel()
    with pytest.raises(asyncio.CancelledError):
        await initialization

    client.delete.assert_awaited_once_with(sandbox)
    client.close.assert_awaited_once()
    assert environment._sandbox is None
