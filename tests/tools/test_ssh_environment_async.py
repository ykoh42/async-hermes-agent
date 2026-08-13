"""Native-async parity tests for the SSH execution environment."""

from __future__ import annotations

import asyncio
import inspect
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tools.environments import ssh as ssh_env
from tools.environments.ssh import SSHEnvironment


_SSH_HOST = os.getenv("TERMINAL_SSH_HOST", "")
_SSH_USER = os.getenv("TERMINAL_SSH_USER", "")
_SSH_PORT = int(os.getenv("TERMINAL_SSH_PORT", "22"))
_SSH_KEY = os.getenv("TERMINAL_SSH_KEY", "")
requires_ssh = pytest.mark.skipif(
    not (_SSH_HOST and _SSH_USER),
    reason="TERMINAL_SSH_HOST / TERMINAL_SSH_USER not set",
)


def test_constructor_and_command_contract_are_preserved(monkeypatch, tmp_path):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    signature = inspect.signature(SSHEnvironment)
    assert list(signature.parameters) == [
        "host",
        "user",
        "cwd",
        "timeout",
        "port",
        "key_path",
    ]

    environment = SSHEnvironment(
        host="example.com",
        user="alice",
        cwd="/work",
        timeout=17,
        port=2222,
        key_path="/keys/id_ed25519",
    )

    assert not environment.control_dir.exists()
    assert environment._sync_manager is None
    assert environment._initialized is False
    assert environment._build_ssh_command() == [
        "ssh",
        "-o",
        f"ControlPath={environment.control_socket}",
        "-o",
        "ControlMaster=auto",
        "-o",
        "ControlPersist=300",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=10",
        "-p",
        "2222",
        "-i",
        "/keys/id_ed25519",
        "alice@example.com",
    ]


def test_control_socket_is_short_deterministic_and_target_scoped(
    monkeypatch,
):
    monkeypatch.setenv(
        "TMPDIR",
        "/var/folders/2t/wbkw5yb158jc3zhswgl7tz9c0000gn/T",
    )
    first = SSHEnvironment(
        host="9373:9b91:4480:558d:708e:e601:24e8:d8d0",
        user="hermes",
    )
    second = SSHEnvironment(host=first.host, user="hermes")

    assert first.control_socket == second.control_socket
    assert len(str(first.control_socket)) + 17 <= 103
    assert SSHEnvironment(host="other", user="hermes").control_socket != (
        first.control_socket
    )
    assert SSHEnvironment(host=first.host, user="root").control_socket != (
        first.control_socket
    )
    assert SSHEnvironment(host=first.host, user="hermes", port=23).control_socket != (
        first.control_socket
    )


@pytest.mark.asyncio
async def test_preflight_preserves_clear_missing_binary_errors(monkeypatch):
    async def missing(_name):
        return None

    monkeypatch.setattr(ssh_env, "_which", missing)
    with pytest.raises(RuntimeError, match="SSH is not installed or not in PATH"):
        await ssh_env._ensure_ssh_available()

    async def only_ssh(name):
        return "/usr/bin/ssh" if name == "ssh" else None

    monkeypatch.setattr(ssh_env, "_which", only_ssh)
    with pytest.raises(RuntimeError, match="SCP is not installed or not in PATH"):
        await ssh_env._ensure_ssh_available()


@pytest.mark.asyncio
async def test_installed_openssh_client_is_executable():
    if shutil.which("ssh") is None or shutil.which("scp") is None:
        pytest.skip("OpenSSH client is not installed")
    if os.name != "nt":
        import pwd

        try:
            pwd.getpwuid(os.getuid())
        except KeyError:
            pytest.skip("The host user database cannot resolve the current uid")

    await ssh_env._ensure_ssh_available()
    process = await asyncio.create_subprocess_exec(
        "ssh",
        "-V",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5)
    assert process.returncode == 0
    assert b"OpenSSH" in stdout + stderr


@pytest.mark.asyncio
async def test_lazy_initialization_keeps_upstream_order_and_wiring(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    events: list[str] = []

    async def available():
        events.append("available")

    async def establish(self):
        events.append("establish")

    async def detect(self):
        events.append("detect")
        return "/home/alice"

    async def directories(self):
        events.append("directories")

    async def init_session(self):
        events.append("session")
        self._initialized = True

    captured: dict = {}

    class SyncManager:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def sync(self, *, force=False):
            events.append(f"sync:{force}")

    monkeypatch.setattr(ssh_env, "_ensure_ssh_available", available)
    monkeypatch.setattr(SSHEnvironment, "_establish_connection", establish)
    monkeypatch.setattr(SSHEnvironment, "_detect_remote_home", detect)
    monkeypatch.setattr(SSHEnvironment, "_ensure_remote_dirs", directories)
    monkeypatch.setattr(SSHEnvironment, "init_session", init_session)
    monkeypatch.setattr(ssh_env, "FileSyncManager", SyncManager)
    environment = SSHEnvironment(host="example.com", user="alice")

    assert events == []
    await environment._ensure_initialized()

    assert events == [
        "available",
        "establish",
        "detect",
        "directories",
        "sync:True",
        "session",
    ]
    assert environment._remote_home == "/home/alice"
    assert captured["upload_fn"] == environment._scp_upload
    assert captured["delete_fn"] == environment._ssh_delete
    assert captured["bulk_upload_fn"] == environment._ssh_bulk_upload
    assert captured["bulk_download_fn"] == environment._ssh_bulk_download
    assert await captured["get_files_fn"]() == await ssh_env.iter_sync_files(
        "/home/alice/.hermes"
    )


@pytest.mark.asyncio
async def test_lazy_initialization_failure_closes_partial_transport(monkeypatch):
    environment = SSHEnvironment(host="example.com", user="alice")
    monkeypatch.setattr(ssh_env, "_ensure_ssh_available", AsyncMock())
    monkeypatch.setattr(environment, "_establish_connection", AsyncMock())
    monkeypatch.setattr(
        environment,
        "_detect_remote_home",
        AsyncMock(side_effect=RuntimeError("detect failed")),
    )
    cleanup = AsyncMock()
    monkeypatch.setattr(environment, "_cleanup_impl", cleanup)

    with pytest.raises(RuntimeError, match="detect failed"):
        await environment._ensure_initialized()

    cleanup.assert_awaited_once_with()
    assert environment._initialized is False


@pytest.mark.asyncio
async def test_connection_and_remote_home_error_contracts(monkeypatch):
    environment = SSHEnvironment(host="host", user="user")
    monkeypatch.setattr(
        environment,
        "_run_captured",
        AsyncMock(return_value=(1, b"stdout detail", b"stderr detail")),
    )
    with pytest.raises(RuntimeError, match="SSH connection failed: stderr detail"):
        await environment._establish_connection()

    environment._run_captured = AsyncMock(
        side_effect=subprocess.TimeoutExpired(["ssh"], 15)
    )
    with pytest.raises(RuntimeError, match="SSH connection to user@host timed out"):
        await environment._establish_connection()

    environment._run_captured = AsyncMock(return_value=(0, b"/srv/user\n", b""))
    assert await environment._detect_remote_home() == "/srv/user"
    environment._run_captured = AsyncMock(side_effect=OSError("offline"))
    assert await environment._detect_remote_home() == "/home/user"


@pytest.mark.asyncio
async def test_scp_upload_preserves_mkdir_argv_options_and_error(monkeypatch):
    environment = SSHEnvironment(
        host="host",
        user="user",
        port=2222,
        key_path="/keys/id_ed25519",
    )
    run = AsyncMock(side_effect=[(0, b"", b""), (0, b"", b"")])
    monkeypatch.setattr(environment, "_run_captured", run)

    await environment._scp_upload(
        "/host/config.json",
        "/home/user/.hermes/credentials/config.json",
    )

    mkdir_argv = run.await_args_list[0].args[0]
    assert mkdir_argv[-1] == "mkdir -p /home/user/.hermes/credentials"
    assert run.await_args_list[0].kwargs == {"timeout": 10}
    assert run.await_args_list[1].args[0] == [
        "scp",
        "-o",
        f"ControlPath={environment.control_socket}",
        "-P",
        "2222",
        "-i",
        "/keys/id_ed25519",
        "/host/config.json",
        "user@host:/home/user/.hermes/credentials/config.json",
    ]
    assert run.await_args_list[1].kwargs == {"timeout": 30}

    run.reset_mock()
    run.side_effect = [(0, b"", b""), (1, b"", b"permission denied\n")]
    with pytest.raises(RuntimeError, match="scp failed: permission denied"):
        await environment._scp_upload("/host/file", "/remote/file")


@pytest.mark.asyncio
async def test_remote_directory_and_delete_commands_preserve_quoting(monkeypatch):
    environment = SSHEnvironment(host="host", user="user")
    environment._remote_home = "/home/user name"
    run = AsyncMock(return_value=(0, b"", b""))
    monkeypatch.setattr(environment, "_run_captured", run)

    await environment._ensure_remote_dirs()
    mkdir = run.await_args.args[0][-1]
    assert mkdir == (
        "mkdir -p '/home/user name/.hermes' "
        "'/home/user name/.hermes/skills' "
        "'/home/user name/.hermes/credentials' "
        "'/home/user name/.hermes/cache'"
    )

    await environment._ssh_delete(["/remote/a b", "/remote/c"])
    assert run.await_args.args[0][-1] == "rm -f '/remote/a b' /remote/c"
    run.return_value = (3, b"", b"delete failed")
    with pytest.raises(RuntimeError, match="remote rm failed: delete failed"):
        await environment._ssh_delete(["/remote/file"])


@pytest.mark.asyncio
async def test_run_bash_preserves_remote_argv_stdin_and_return_shape(monkeypatch):
    environment = SSHEnvironment(host="host", user="user")
    process = _CompletedProcess(7, b"combined output")
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(environment, "_spawn", spawn)

    result = await environment._run_bash(
        "printf hello",
        login=True,
        timeout=9,
        stdin_data="input",
    )

    argv = spawn.await_args.args[0]
    assert argv[-4:] == ["bash", "-l", "-c", "'printf hello'"]
    assert process.stdin.data == b"input"
    assert spawn.await_args.kwargs["stderr"] == asyncio.subprocess.STDOUT
    assert result == {"output": "combined output", "returncode": 7}


@pytest.mark.asyncio
async def test_run_bash_preserves_utf8_split_across_transport_chunks(monkeypatch):
    environment = SSHEnvironment(host="host", user="user")
    process = _CompletedProcess(0, b"")
    chunks = iter([b"\xed", b"\x95", b"\x9c", b""])

    async def read_chunk(_size):
        await asyncio.sleep(0)
        return next(chunks)

    process.stdout.read = read_chunk
    monkeypatch.setattr(environment, "_spawn", AsyncMock(return_value=process))

    result = await environment._run_bash("printf '\ud55c'")

    assert result == {"output": "\ud55c", "returncode": 0}


@pytest.mark.asyncio
async def test_run_bash_bounded_capture_spills_full_remote_stream(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr("tools.tool_output_limits.get_max_bytes", lambda: 100)
    environment = SSHEnvironment(host="host", user="user")
    raw = b"HEAD-" + (b"x" * 240) + b"-TAIL"
    process = _CompletedProcess(0, raw)
    monkeypatch.setattr(environment, "_spawn", AsyncMock(return_value=process))

    result = await environment._run_bash(
        "generate output",
        bounded_capture=True,
    )

    assert result["returncode"] == 0
    assert result["output_total_chars"] == len(raw)
    assert len(result["output"]) <= 100
    assert "[OUTPUT TRUNCATED" in result["output"]
    spill = Path(result["full_output_path"])
    assert spill.read_bytes() == raw


@pytest.mark.asyncio
async def test_run_bash_timeout_preserves_partial_output_and_reaps(monkeypatch):
    environment = SSHEnvironment(host="host", user="user")
    process = _PendingProcess(b"partial output")
    monkeypatch.setattr(environment, "_spawn", AsyncMock(return_value=process))

    result = await environment._run_bash("sleep 10", timeout=0.01)

    assert result == {
        "output": "partial output\n[Command timed out after 0.01s]",
        "returncode": 124,
    }
    assert process.killed is True
    assert process.waited is True


@pytest.mark.asyncio
async def test_run_bash_interrupt_reaps_real_transport_and_marks_result(
    monkeypatch,
) -> None:
    from tools.interrupt import _bind_interrupt_event, _reset_interrupt_event

    environment = SSHEnvironment(host="host", user="user")
    spawned = []

    async def spawn(_argv, **kwargs):
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import signal,time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
            **kwargs,
        )
        spawned.append(process)
        environment._processes.add(process)
        return process

    monkeypatch.setattr(environment, "_spawn", spawn)
    interrupt = asyncio.Event()
    token = _bind_interrupt_event(interrupt)
    try:
        run = asyncio.create_task(environment._run_bash("sleep 60", timeout=60))
        while not spawned:
            await asyncio.sleep(0)
        interrupt.set()

        result = await asyncio.wait_for(run, timeout=5)

        assert result == {
            "output": "[Command interrupted]",
            "returncode": 130,
        }
        assert spawned[0].returncode is not None
        assert spawned[0] not in environment._processes
    finally:
        _reset_interrupt_event(token)
        if "run" in locals() and not run.done():
            run.cancel()
            await asyncio.gather(run, return_exceptions=True)
        if spawned and spawned[0].returncode is None:
            spawned[0].kill()
            await spawned[0].wait()


@pytest.mark.asyncio
async def test_run_bash_natural_exit_130_has_no_interrupt_marker(monkeypatch):
    environment = SSHEnvironment(host="host", user="user")
    process = _CompletedProcess(130, b"natural output")
    monkeypatch.setattr(environment, "_spawn", AsyncMock(return_value=process))

    result = await environment._run_bash("exit 130", timeout=5)

    assert result == {"output": "natural output", "returncode": 130}


@pytest.mark.asyncio
async def test_run_bash_cancellation_kills_and_reaps(monkeypatch):
    environment = SSHEnvironment(host="host", user="user")
    process = _PendingProcess()
    monkeypatch.setattr(environment, "_spawn", AsyncMock(return_value=process))

    run = asyncio.create_task(environment._run_bash("sleep 10", timeout=30))
    await asyncio.wait_for(process.wait_started.wait(), timeout=1)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert process.killed is True
    assert process.waited is True


@pytest.mark.asyncio
async def test_captured_operation_cancellation_kills_and_reaps(monkeypatch):
    environment = SSHEnvironment(host="host", user="user")
    process = _CommunicatingProcess()
    monkeypatch.setattr(environment, "_spawn", AsyncMock(return_value=process))

    run = asyncio.create_task(
        environment._run_captured(["ssh", "user@host"], timeout=30)
    )
    await asyncio.wait_for(process.communicate_started.wait(), timeout=1)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert process.killed is True
    assert process.waited is True


@pytest.mark.asyncio
async def test_cleanup_syncs_back_exits_controlmaster_and_removes_socket(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    environment = SSHEnvironment(host="host", user="user")
    environment.control_dir.mkdir(parents=True)
    environment.control_socket.write_bytes(b"socket placeholder")

    manager = type("Manager", (), {})()
    manager.sync_back = AsyncMock()
    environment._sync_manager = manager
    run = AsyncMock(return_value=(0, b"", b""))
    monkeypatch.setattr(environment, "_run_captured", run)

    await environment.cleanup()

    manager.sync_back.assert_awaited_once_with()
    assert run.await_args.args[0] == [
        "ssh",
        "-o",
        f"ControlPath={environment.control_socket}",
        "-O",
        "exit",
        "user@host",
    ]
    assert not environment.control_socket.exists()


@requires_ssh
@pytest.mark.asyncio
async def test_real_ssh_echo_state_and_large_output():
    environment = SSHEnvironment(
        host=_SSH_HOST,
        user=_SSH_USER,
        cwd="/tmp",
        timeout=30,
        port=_SSH_PORT,
        key_path=_SSH_KEY,
    )
    try:
        echo = await environment.execute("echo hello-async-ssh")
        assert echo["returncode"] == 0
        assert "hello-async-ssh" in echo["output"]

        exported = await environment.execute("export HERMES_ASYNC_SSH_TEST=works")
        assert exported["returncode"] == 0
        persisted = await environment.execute("echo $HERMES_ASYNC_SSH_TEST")
        assert persisted["returncode"] == 0
        assert persisted["output"].strip() == "works"

        large = await environment.execute("seq 1 1000")
        assert large["returncode"] == 0
        lines = large["output"].strip().splitlines()
        assert lines[0] == "1"
        assert lines[-1] == "1000"
        assert len(lines) == 1000
    finally:
        await environment.cleanup()


class _InputWriter:
    def __init__(self):
        self.data = b""

    def write(self, data):
        self.data += data

    async def drain(self):
        return None

    def close(self):
        return None


class _CompletedProcess:
    def __init__(self, returncode: int, output: bytes):
        self.returncode = returncode
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_data(output)
        self.stdout.feed_eof()
        self.stdin = _InputWriter()

    async def wait(self):
        return self.returncode


class _PendingProcess:
    _next_pid = 4000

    def __init__(self, output: bytes = b""):
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.returncode = None
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_data(output)
        self.stdin = _InputWriter()
        self.killed = False
        self.waited = False
        self.wait_started = asyncio.Event()
        self._done = asyncio.Event()

    async def wait(self):
        self.waited = True
        self.wait_started.set()
        await self._done.wait()
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9
        self.stdout.feed_eof()
        self._done.set()


class _CommunicatingProcess(_PendingProcess):
    def __init__(self):
        super().__init__()
        self.communicate_started = asyncio.Event()

    async def communicate(self, _data):
        self.communicate_started.set()
        await self._done.wait()
        return b"", b""
