"""Native-async direct Modal environment behavior."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import aiofiles
import pytest

from tools.environments import modal


class _AioMethod:
    def __init__(self, callback):
        self.aio = callback


class _Stream:
    def __init__(self, value="", *, chunk_size=17):
        self.value = value
        self.chunk_size = chunk_size
        self.read_called = False
        self.started = asyncio.Event()
        self.release = None
        self.read = _AioMethod(self._read)

    async def _read(self):
        self.read_called = True
        return self.value

    async def __aiter__(self):
        self.started.set()
        if self.release is not None:
            await self.release.wait()
        for offset in range(0, len(self.value), self.chunk_size):
            yield self.value[offset : offset + self.chunk_size]


class _Process:
    def __init__(self, stdout="", stderr="", exit_code=0):
        self.stdout = _Stream(stdout)
        self.stderr = _Stream(stderr)
        self.stdin = SimpleNamespace(
            write=lambda _data: None,
            write_eof=lambda: None,
            drain=_AioMethod(self._drain),
        )
        self.wait = _AioMethod(self._wait)
        self._exit_code = exit_code

    async def _drain(self):
        return None

    async def _wait(self):
        return self._exit_code


class _Sandbox:
    def __init__(self, process=None):
        self.exec_calls = []
        self.terminated = False
        self.process = process or _Process()
        self.exec = _AioMethod(self._exec)
        self.terminate = _AioMethod(self._terminate)
        self.snapshot_filesystem = _AioMethod(self._snapshot)

    async def _exec(self, *args, **kwargs):
        self.exec_calls.append((args, kwargs))
        return self.process

    async def _terminate(self):
        self.terminated = True

    async def _snapshot(self):
        return SimpleNamespace(object_id="im-snapshot")


class _ModalClient:
    def __init__(
        self,
        token_id,
        token_secret,
        *,
        close_gate=None,
        enter_callback=None,
    ):
        self.credentials = (token_id, token_secret)
        self.enter_callback = enter_callback
        self.close_calls = 0
        self.close_started = asyncio.Event()
        self.close_gate = close_gate

    async def __aenter__(self):
        if self.enter_callback is not None:
            await self.enter_callback(*self.credentials)
        return self

    async def __aexit__(self, _exc_type, _exc, _tb):
        await self._close_client()

    async def _close_client(self):
        self.close_calls += 1
        self.close_started.set()
        if self.close_gate is not None:
            await self.close_gate.wait()


class _ModalConfig:
    def __init__(self, token_id=None, token_secret=None):
        self.values = {
            "server_url": "https://api.modal.test",
            "token_id": token_id,
            "token_secret": token_secret,
        }
        self.calls = []

    def get(self, key, *, use_env=True):
        self.calls.append((key, use_env))
        return self.values[key]

    def __getitem__(self, key):
        return self.values[key]


def _fake_modal_module(
    *,
    config_id=None,
    config_secret=None,
    client_factory=None,
    client_enter_callback=None,
    lookup_callback=None,
):
    clients = []
    lookup_calls = []
    sandbox_calls = []
    config = _ModalConfig(config_id, config_secret)

    def client_constructor(_server_url, _client_type, credentials):
        token_id, token_secret = credentials
        if client_factory is not None:
            client = client_factory(token_id, token_secret)
        else:
            client = _ModalClient(
                token_id,
                token_secret,
                enter_callback=client_enter_callback,
            )
        clients.append(client)
        return client

    async def lookup(name, **kwargs):
        lookup_calls.append((name, kwargs))
        if lookup_callback is not None:
            return await lookup_callback(name, **kwargs)
        return SimpleNamespace(client=kwargs.get("client"))

    async def create(*args, **kwargs):  # noqa: ASYNC124 - SDK test double
        sandbox = _Sandbox()
        sandbox.modal_client = kwargs.get("client")
        sandbox_calls.append((args, kwargs, sandbox))
        return sandbox

    module = SimpleNamespace(
        Client=client_constructor,
        App=SimpleNamespace(lookup=_AioMethod(lookup)),
        Sandbox=SimpleNamespace(create=_AioMethod(create)),
        config=SimpleNamespace(config=config),
    )
    module.clients = clients
    module.lookup_calls = lookup_calls
    module.sandbox_calls = sandbox_calls
    return module


def _install_fake_modal(monkeypatch, fake_modal):
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setitem(
        sys.modules,
        "modal_proto",
        SimpleNamespace(api_pb2=SimpleNamespace(CLIENT_TYPE_CLIENT=1)),
    )


def test_constructor_is_state_only():
    environment = modal.ModalEnvironment("python:3.11")
    assert environment._sandbox is None
    assert environment._app is None
    assert environment._modal_client is None
    assert environment._sync_manager is None


@pytest.mark.asyncio
async def test_direct_modal_clients_are_profile_scoped_and_owned(
    monkeypatch,
):
    from agent import secret_scope

    previous_multiplex = secret_scope.is_multiplex_active()
    base_token = secret_scope.set_secret_scope(None)
    secret_scope.set_multiplex_active(True)
    monkeypatch.setenv("MODAL_TOKEN_ID", "foreign-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "foreign-secret")
    both_creating = asyncio.Event()
    arrival_count = 0

    async def enter_client(_token_id, _token_secret):
        nonlocal arrival_count
        arrival_count += 1
        if arrival_count == 2:
            both_creating.set()
        await both_creating.wait()

    fake_modal = _fake_modal_module(client_enter_callback=enter_client)
    _install_fake_modal(monkeypatch, fake_modal)
    environments = {
        label: modal.ModalEnvironment(object(), persistent_filesystem=False)
        for label in ("a", "b")
    }

    async def initialize(label):
        token = secret_scope.set_secret_scope(
            {
                "MODAL_TOKEN_ID": f"{label}-id",
                "MODAL_TOKEN_SECRET": f"{label}-secret",
            }
        )
        try:
            await environments[label]._create_sandbox()
        finally:
            secret_scope.reset_secret_scope(token)

    try:
        await asyncio.gather(*(initialize(label) for label in environments))
        assert {client.credentials for client in fake_modal.clients} == {
            ("a-id", "a-secret"),
            ("b-id", "b-secret"),
        }
        assert all(
            kwargs["client"].credentials == sandbox.modal_client.credentials
            for _args, kwargs, sandbox in fake_modal.sandbox_calls
        )
        assert all(
            kwargs["client"].credentials != ("foreign-id", "foreign-secret")
            for _name, kwargs in fake_modal.lookup_calls
        )

        await asyncio.gather(
            *(environment._terminate_sandbox() for environment in environments.values())
        )
        assert all(client.close_calls == 1 for client in fake_modal.clients)
    finally:
        secret_scope.set_multiplex_active(previous_multiplex)
        secret_scope.reset_secret_scope(base_token)


@pytest.mark.asyncio
async def test_direct_modal_environment_fails_closed_without_profile_scope(
    monkeypatch,
):
    from agent import secret_scope

    previous_multiplex = secret_scope.is_multiplex_active()
    base_token = secret_scope.set_secret_scope(None)
    secret_scope.set_multiplex_active(True)
    monkeypatch.setenv("MODAL_TOKEN_ID", "foreign-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "foreign-secret")
    fake_modal = _fake_modal_module()
    _install_fake_modal(monkeypatch, fake_modal)
    environment = modal.ModalEnvironment(object(), persistent_filesystem=False)
    try:
        with pytest.raises(secret_scope.UnscopedSecretError, match="MODAL_TOKEN_ID"):
            await environment._create_sandbox()
    finally:
        secret_scope.set_multiplex_active(previous_multiplex)
        secret_scope.reset_secret_scope(base_token)

    assert fake_modal.clients == []
    assert fake_modal.lookup_calls == []


@pytest.mark.asyncio
async def test_direct_modal_environment_rejects_global_config_file_fallback(
    monkeypatch,
):
    from agent import secret_scope

    previous_multiplex = secret_scope.is_multiplex_active()
    base_token = secret_scope.set_secret_scope(None)
    secret_scope.set_multiplex_active(True)
    monkeypatch.setenv("MODAL_TOKEN_ID", "foreign-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "foreign-secret")
    fake_modal = _fake_modal_module(
        config_id="config-id",
        config_secret="config-secret",
    )
    _install_fake_modal(monkeypatch, fake_modal)
    environment = modal.ModalEnvironment(object(), persistent_filesystem=False)
    token = secret_scope.set_secret_scope({})
    try:
        with pytest.raises(RuntimeError, match="profile-scoped MODAL_TOKEN_ID"):
            await environment._create_sandbox()
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(previous_multiplex)
        secret_scope.reset_secret_scope(base_token)

    assert fake_modal.config.config.calls == []
    assert fake_modal.clients == []
    assert fake_modal.lookup_calls == []
    assert fake_modal.sandbox_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope",
    [
        {"MODAL_TOKEN_ID": "", "MODAL_TOKEN_SECRET": "profile-secret"},
        {"MODAL_TOKEN_ID": "profile-id", "MODAL_TOKEN_SECRET": ""},
    ],
)
async def test_direct_modal_environment_rejects_incomplete_profile_pair(
    monkeypatch,
    scope,
):
    from agent import secret_scope

    previous_multiplex = secret_scope.is_multiplex_active()
    base_token = secret_scope.set_secret_scope(None)
    secret_scope.set_multiplex_active(True)
    monkeypatch.setenv("MODAL_TOKEN_ID", "foreign-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "foreign-secret")
    fake_modal = _fake_modal_module(
        config_id="config-id",
        config_secret="config-secret",
    )
    _install_fake_modal(monkeypatch, fake_modal)
    environment = modal.ModalEnvironment(object(), persistent_filesystem=False)
    token = secret_scope.set_secret_scope(scope)
    try:
        with pytest.raises(RuntimeError, match="profile-scoped MODAL_TOKEN_ID"):
            await environment._create_sandbox()
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(previous_multiplex)
        secret_scope.reset_secret_scope(base_token)

    assert fake_modal.config.config.calls == []
    assert fake_modal.clients == []
    assert fake_modal.lookup_calls == []
    assert fake_modal.sandbox_calls == []


@pytest.mark.asyncio
async def test_direct_modal_environment_keeps_single_profile_config_fallback(
    monkeypatch,
):
    from agent import secret_scope

    previous_multiplex = secret_scope.is_multiplex_active()
    base_token = secret_scope.set_secret_scope(None)
    secret_scope.set_multiplex_active(False)
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    fake_modal = _fake_modal_module(
        config_id="config-id",
        config_secret="config-secret",
    )
    _install_fake_modal(monkeypatch, fake_modal)
    environment = modal.ModalEnvironment(object(), persistent_filesystem=False)
    try:
        await environment._create_sandbox()
    finally:
        secret_scope.set_multiplex_active(previous_multiplex)
        secret_scope.reset_secret_scope(base_token)

    # Returning None lets the Modal SDK preserve its ordinary ~/.modal.toml
    # account resolution in the single-profile process.
    assert fake_modal.config.config.calls == []
    assert fake_modal.clients == []
    assert fake_modal.lookup_calls == [("hermes-agent", {"create_if_missing": True})]
    assert fake_modal.sandbox_calls[0][1].get("client") is None
    await environment._terminate_sandbox()


@pytest.mark.asyncio
async def test_direct_modal_environment_keeps_explicit_caller_client_in_multiplex(
    monkeypatch,
):
    from agent import secret_scope

    previous_multiplex = secret_scope.is_multiplex_active()
    base_token = secret_scope.set_secret_scope(None)
    secret_scope.set_multiplex_active(True)
    monkeypatch.setenv("MODAL_TOKEN_ID", "foreign-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "foreign-secret")
    fake_modal = _fake_modal_module(
        config_id="config-id",
        config_secret="config-secret",
    )
    _install_fake_modal(monkeypatch, fake_modal)
    explicit_client = _ModalClient("caller-id", "caller-secret")
    environment = modal.ModalEnvironment(
        object(),
        persistent_filesystem=False,
        modal_sandbox_kwargs={"client": explicit_client},
    )
    token = secret_scope.set_secret_scope({})
    try:
        await environment._create_sandbox()
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(previous_multiplex)
        secret_scope.reset_secret_scope(base_token)

    assert fake_modal.config.config.calls == []
    assert fake_modal.clients == []
    assert fake_modal.lookup_calls == [
        ("hermes-agent", {"create_if_missing": True, "client": explicit_client})
    ]
    assert fake_modal.sandbox_calls[0][1]["client"] is explicit_client
    await environment._terminate_sandbox()
    assert explicit_client.close_calls == 0


@pytest.mark.asyncio
async def test_cancelled_direct_modal_lookup_closes_profile_client(
    monkeypatch,
):
    from agent import secret_scope

    previous_multiplex = secret_scope.is_multiplex_active()
    base_token = secret_scope.set_secret_scope(None)
    secret_scope.set_multiplex_active(True)
    lookup_started = asyncio.Event()
    never_lookup = asyncio.Event()
    close_gate = asyncio.Event()

    def create_client(token_id, token_secret):
        return _ModalClient(token_id, token_secret, close_gate=close_gate)

    async def blocked_lookup(_name, **_kwargs):
        lookup_started.set()
        await never_lookup.wait()

    fake_modal = _fake_modal_module(
        client_factory=create_client,
        lookup_callback=blocked_lookup,
    )
    _install_fake_modal(monkeypatch, fake_modal)
    environment = modal.ModalEnvironment(object(), persistent_filesystem=False)
    token = secret_scope.set_secret_scope(
        {
            "MODAL_TOKEN_ID": "profile-id",
            "MODAL_TOKEN_SECRET": "profile-secret",
        }
    )
    try:
        task = asyncio.create_task(environment._ensure_initialized())
        await lookup_started.wait()
        task.cancel()
        client = fake_modal.clients[0]
        await client.close_started.wait()
        task.cancel()
        close_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(previous_multiplex)
        secret_scope.reset_secret_scope(base_token)

    assert client.close_calls == 1
    assert environment._modal_client is None


@pytest.mark.asyncio
async def test_cancelled_profile_client_enter_is_closed_before_reraise(
    monkeypatch,
):
    from agent import secret_scope

    previous_multiplex = secret_scope.is_multiplex_active()
    base_token = secret_scope.set_secret_scope(None)
    secret_scope.set_multiplex_active(True)
    enter_started = asyncio.Event()
    never_enter = asyncio.Event()
    close_gate = asyncio.Event()

    async def blocked_enter(_token_id, _token_secret):
        enter_started.set()
        await never_enter.wait()

    def create_client(token_id, token_secret):
        return _ModalClient(
            token_id,
            token_secret,
            close_gate=close_gate,
            enter_callback=blocked_enter,
        )

    fake_modal = _fake_modal_module(client_factory=create_client)
    _install_fake_modal(monkeypatch, fake_modal)
    environment = modal.ModalEnvironment(object(), persistent_filesystem=False)
    token = secret_scope.set_secret_scope(
        {
            "MODAL_TOKEN_ID": "profile-id",
            "MODAL_TOKEN_SECRET": "profile-secret",
        }
    )
    try:
        task = asyncio.create_task(environment._create_sandbox())
        await enter_started.wait()
        task.cancel()
        client = fake_modal.clients[0]
        await client.close_started.wait()
        task.cancel()
        close_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(previous_multiplex)
        secret_scope.reset_secret_scope(base_token)

    assert client.close_calls == 1
    assert environment._modal_client is None
    assert fake_modal.lookup_calls == []


@pytest.mark.asyncio
async def test_run_bash_uses_modal_aio_transport():
    environment = modal.ModalEnvironment("python:3.11")
    sandbox = _Sandbox()
    environment._sandbox = sandbox

    result = await environment._run_bash("printf hello", timeout=3)

    assert result == {"output": "", "returncode": 0}
    assert sandbox.exec_calls == [(('bash', '-c', 'printf hello'), {'timeout': 3})]


@pytest.mark.asyncio
async def test_modal_sdk_error_preserves_upstream_empty_rc1_result():
    class _FailingExec:
        async def aio(self, *_args, **_kwargs):
            raise RuntimeError("sdk failed")

    environment = modal.ModalEnvironment("python:3.11")
    environment._sandbox = SimpleNamespace(exec=_FailingExec())

    result = await environment._run_bash("false")

    assert result == {"output": "", "returncode": 1}


@pytest.mark.asyncio
async def test_modal_stream_error_preserves_upstream_rc1_without_cancel_hook():
    class _FailingStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise RuntimeError("stream failed")

    process = _Process()
    process.stdout = _FailingStream()
    sandbox = _Sandbox(process)
    environment = modal.ModalEnvironment("python:3.11")
    environment._sandbox = sandbox

    result = await environment._run_bash("false")

    assert result == {"output": "", "returncode": 1}
    assert sandbox.terminated is False
    assert environment._sandbox is sandbox


@pytest.mark.asyncio
async def test_bounded_run_bash_streams_and_spills_full_ordered_output(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr("tools.tool_output_limits.get_max_bytes", lambda: 100)
    stdout = "stdout-" + ("a" * 500)
    stderr = "stderr-" + ("b" * 500)
    process = _Process(stdout=stdout, stderr=stderr, exit_code=9)
    environment = modal.ModalEnvironment("python:3.11")
    environment._sandbox = _Sandbox(process)

    result = await environment._run_bash("emit", bounded_capture=True)

    full_output = f"{stdout}\n{stderr}"
    assert result["returncode"] == 9
    assert len(result["output"]) <= 100
    assert result["output_total_chars"] == len(full_output)
    async with aiofiles.open(
        result["full_output_path"],
        encoding="utf-8",
    ) as handle:
        assert await handle.read() == full_output
    assert process.stdout.read_called is False
    assert process.stderr.read_called is False


@pytest.mark.asyncio
async def test_bounded_run_bash_decodes_split_utf8_stream_chunks():
    process = _Process(stdout="한글".encode())
    process.stdout.chunk_size = 1
    environment = modal.ModalEnvironment("python:3.11")
    environment._sandbox = _Sandbox(process)

    result = await environment._run_bash("emit", bounded_capture=True)

    assert result == {"output": "한글", "returncode": 0}


@pytest.mark.asyncio
async def test_bounded_run_bash_drains_both_streams_without_changing_order():
    stdout_started = asyncio.Event()
    stderr_started = asyncio.Event()

    class _CoordinatedStream(_Stream):
        def __init__(self, value, started, peer_started):
            super().__init__(value)
            self._started = started
            self._peer_started = peer_started

        async def __aiter__(self):
            self._started.set()
            await self._peer_started.wait()
            yield self.value

    process = _Process(exit_code=3)
    process.stdout = _CoordinatedStream(
        "stdout",
        stdout_started,
        stderr_started,
    )
    process.stderr = _CoordinatedStream(
        "stderr",
        stderr_started,
        stdout_started,
    )
    environment = modal.ModalEnvironment("python:3.11")
    environment._sandbox = _Sandbox(process)

    result = await asyncio.wait_for(
        environment._run_bash("emit", bounded_capture=True),
        timeout=1,
    )

    assert result == {"output": "stdout\nstderr", "returncode": 3}


@pytest.mark.asyncio
async def test_bounded_run_bash_cancellation_terminates_sandbox():
    process = _Process(stdout="blocked")
    process.stdout.release = asyncio.Event()
    sandbox = _Sandbox(process)
    environment = modal.ModalEnvironment("python:3.11")
    environment._sandbox = sandbox
    task = asyncio.create_task(
        environment._run_bash("sleep", bounded_capture=True)
    )
    await asyncio.wait_for(process.stdout.started.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert sandbox.terminated is True
    assert environment._sandbox is None


@pytest.mark.asyncio
async def test_bounded_collector_cancellation_does_not_spawn_process(monkeypatch):
    collector_started = asyncio.Event()

    async def blocked_collector():
        collector_started.set()
        await asyncio.Future()

    sandbox = _Sandbox()
    environment = modal.ModalEnvironment("python:3.11")
    environment._sandbox = sandbox
    monkeypatch.setattr(environment, "_bounded_output_collector", blocked_collector)
    task = asyncio.create_task(
        environment._run_bash("sleep", bounded_capture=True)
    )
    await asyncio.wait_for(collector_started.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert sandbox.exec_calls == []


@pytest.mark.asyncio
async def test_bounded_run_bash_transport_timeout_terminates_sandbox():
    class _TimeoutStream(_Stream):
        async def __aiter__(self):
            raise TimeoutError
            yield  # pragma: no cover

    process = _Process()
    process.stdout = _TimeoutStream()
    sandbox = _Sandbox(process)
    environment = modal.ModalEnvironment("python:3.11")
    environment._sandbox = sandbox

    result = await environment._run_bash(
        "sleep",
        timeout=7,
        bounded_capture=True,
    )

    assert result == {
        "output": "[Command timed out after 7s]",
        "returncode": 124,
    }
    assert sandbox.terminated is True
    assert environment._sandbox is None


@pytest.mark.asyncio
async def test_run_bash_interrupt_terminates_sandbox_with_partial_output():
    from tools.interrupt import _bind_interrupt_event, _reset_interrupt_event

    class _PartialBlockingStream(_Stream):
        async def __aiter__(self):
            yield self.value
            self.started.set()
            await asyncio.Future()

    process = _Process()
    process.stdout = _PartialBlockingStream("partial")
    sandbox = _Sandbox(process)
    environment = modal.ModalEnvironment("python:3.11")
    environment._sandbox = sandbox
    interrupt = asyncio.Event()
    token = _bind_interrupt_event(interrupt)
    try:
        command = asyncio.create_task(environment._run_bash("sleep"))
        await asyncio.wait_for(process.stdout.started.wait(), timeout=1)
        interrupt.set()
        result = await asyncio.wait_for(command, timeout=1)
    finally:
        _reset_interrupt_event(token)

    assert result == {
        "output": "partial\n[Command interrupted]",
        "returncode": 130,
    }
    assert sandbox.terminated is True
    assert environment._sandbox is None


@pytest.mark.asyncio
async def test_bounded_timeout_preserves_partial_output_and_spill(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr("tools.tool_output_limits.get_max_bytes", lambda: 100)

    class _PartialTimeoutStream(_Stream):
        async def __aiter__(self):
            yield self.value
            raise TimeoutError

    partial = "partial-" + ("x" * 500)
    process = _Process()
    process.stdout = _PartialTimeoutStream(partial)
    sandbox = _Sandbox(process)
    environment = modal.ModalEnvironment("python:3.11")
    environment._sandbox = sandbox

    result = await environment._run_bash(
        "sleep",
        timeout=7,
        bounded_capture=True,
    )

    assert result["returncode"] == 124
    assert result["output"].endswith("[Command timed out after 7s]")
    assert len(result["output"]) <= 100
    assert result["output_total_chars"] == len(partial)
    async with aiofiles.open(
        result["full_output_path"],
        encoding="utf-8",
    ) as handle:
        assert await handle.read() == partial
    assert sandbox.terminated is True
    assert environment._sandbox is None


@pytest.mark.asyncio
async def test_cleanup_snapshots_then_terminates(monkeypatch, tmp_path):
    environment = modal.ModalEnvironment(
        "python:3.11",
        task_id="task-a",
        persistent_filesystem=True,
    )
    sandbox = _Sandbox()
    environment._sandbox = sandbox
    environment._initialized = True
    stored = []

    async def store(  # noqa: ASYNC124 - coroutine-shaped test double
        task_id, snapshot_id
    ):
        stored.append((task_id, snapshot_id))

    monkeypatch.setattr(modal, "_store_direct_snapshot", store)
    await environment.cleanup()

    assert stored == [("task-a", "im-snapshot")]
    assert sandbox.terminated is True
    assert environment._sandbox is None
