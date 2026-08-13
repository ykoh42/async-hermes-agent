"""Native-async parity tests for the managed Modal environment."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from tools.environments import managed_modal, modal_utils


class _FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakeClient:
    def __init__(self, route):
        self.route = route
        self.calls: list[dict] = []
        self.closed = False

    async def request(self, method, url, **kwargs):
        call = {"method": method, "url": url, **kwargs}
        self.calls.append(call)
        return await self.route(call)

    async def aclose(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _native_dependencies(monkeypatch):
    async def resolve_gateway(vendor):
        return SimpleNamespace(
            vendor=vendor,
            gateway_origin="https://modal-gateway.example.com",
            nous_user_token="user-token",
            managed_mode=True,
        )

    async def no_credential_mounts():
        return []

    async def transform(command):
        return command, None

    monkeypatch.setattr(
        managed_modal,
        "resolve_managed_tool_gateway",
        resolve_gateway,
    )
    monkeypatch.setattr(
        "tools.credential_files.get_credential_file_mounts",
        no_credential_mounts,
    )
    monkeypatch.setattr(
        "tools.environments.local._transform_sudo_command",
        transform,
    )
    monkeypatch.setattr(modal_utils, "is_interrupted", lambda: False)


@pytest.mark.asyncio
async def test_managed_modal_execute_polls_until_completed(monkeypatch):
    poll_count = 0

    async def route(call):
        nonlocal poll_count
        method = call["method"]
        url = call["url"]
        payload = call.get("json")
        if method == "POST" and url.endswith("/v1/sandboxes"):
            return _FakeResponse(200, {"id": "sandbox-1"})
        if method == "POST" and url.endswith("/execs"):
            return _FakeResponse(
                202,
                {"execId": payload["execId"], "status": "running"},
            )
        if method == "GET" and "/execs/" in url:
            poll_count += 1
            if poll_count == 1:
                return _FakeResponse(200, {"status": "running"})
            return _FakeResponse(
                200,
                {"status": "completed", "output": "hello", "returncode": 0},
            )
        if method == "POST" and url.endswith("/terminate"):
            return _FakeResponse(200, {"status": "terminated"})
        raise AssertionError(f"Unexpected request: {method} {url}")

    client = _FakeClient(route)
    monkeypatch.setattr(managed_modal._httpx, "AsyncClient", lambda: client)
    monkeypatch.setattr(modal_utils.asyncio, "sleep", _no_sleep)

    env = managed_modal.ManagedModalEnvironment(
        image="python:3.11",
        modal_sandbox_kwargs={"cpu": "2", "memory": "4096"},
        task_id="task-7",
    )
    assert client.calls == [], "construction must perform no network I/O"

    result = await env.execute("echo hello")
    await env.cleanup()

    assert result == {"output": "hello", "returncode": 0}
    create = next(call for call in client.calls if call["url"].endswith("/v1/sandboxes"))
    assert create["json"] == {
        "image": "python:3.11",
        "cwd": "/root",
        "cpu": 2.0,
        "memoryMiB": 4096.0,
        "timeoutMs": 3_600_000,
        "idleTimeoutMs": 300_000,
        "persistentFilesystem": True,
        "logicalKey": "task-7",
    }
    assert create["headers"]["Authorization"] == "Bearer user-token"
    assert "x-idempotency-key" in create["headers"]
    terminate = next(call for call in client.calls if call["url"].endswith("/terminate"))
    assert terminate["json"] == {"snapshotBeforeTerminate": True}
    assert client.closed is True
    assert env._sandbox_id is None


@pytest.mark.asyncio
async def test_init_session_is_lazy_idempotent_and_cleanup_is_idempotent(monkeypatch):
    creates = 0

    async def route(call):
        nonlocal creates
        if call["url"].endswith("/v1/sandboxes"):
            creates += 1
            return _FakeResponse(200, {"id": "sandbox-1"})
        if call["url"].endswith("/terminate"):
            return _FakeResponse(200, {})
        raise AssertionError(call)

    client = _FakeClient(route)
    monkeypatch.setattr(managed_modal._httpx, "AsyncClient", lambda: client)
    env = managed_modal.ManagedModalEnvironment(image="python:3.11")

    await asyncio.gather(env.init_session(), env.init_session())
    assert creates == 1
    await env.cleanup()
    await env.cleanup()
    assert client.closed is True


@pytest.mark.asyncio
async def test_managed_modal_rejects_host_credential_passthrough(monkeypatch):
    async def credential_mounts():
        return [{
            "host_path": "/tmp/token.json",
            "container_path": "/root/.hermes/token.json",
        }]

    async def unexpected_gateway(_vendor):
        pytest.fail("gateway resolution must follow the credential-file guard")

    monkeypatch.setattr(
        "tools.credential_files.get_credential_file_mounts",
        credential_mounts,
    )
    monkeypatch.setattr(
        managed_modal,
        "resolve_managed_tool_gateway",
        unexpected_gateway,
    )
    env = managed_modal.ManagedModalEnvironment(image="python:3.11")

    with pytest.raises(ValueError, match="credential-file passthrough"):
        await env.init_session()


@pytest.mark.asyncio
async def test_managed_modal_execute_times_out_and_cancels(monkeypatch):
    monotonic_values = iter([0.0, 0.0, 12.5])

    async def route(call):
        method = call["method"]
        url = call["url"]
        payload = call.get("json")
        if method == "POST" and url.endswith("/v1/sandboxes"):
            return _FakeResponse(200, {"id": "sandbox-1"})
        if method == "POST" and url.endswith("/execs"):
            return _FakeResponse(202, {"execId": payload["execId"], "status": "running"})
        if method == "GET" and "/execs/" in url:
            return _FakeResponse(200, {"status": "running"})
        if method == "POST" and url.endswith("/cancel"):
            return _FakeResponse(202, {"status": "cancelling"})
        if method == "POST" and url.endswith("/terminate"):
            return _FakeResponse(200, {"status": "terminated"})
        raise AssertionError(call)

    client = _FakeClient(route)
    monkeypatch.setattr(managed_modal._httpx, "AsyncClient", lambda: client)
    monkeypatch.setattr(
        modal_utils,
        "time",
        SimpleNamespace(monotonic=lambda: next(monotonic_values)),
    )
    monkeypatch.setattr(modal_utils.asyncio, "sleep", _no_sleep)

    env = managed_modal.ManagedModalEnvironment(image="python:3.11")
    result = await env.execute("sleep 30", timeout=2)
    await env.cleanup()

    assert result == {
        "output": "Managed Modal exec timed out after 2s",
        "returncode": 124,
    }
    cancel = next(call for call in client.calls if call["url"].endswith("/cancel"))
    assert cancel["timeout"].connect == 1.0
    assert cancel["timeout"].read == 5.0


@pytest.mark.asyncio
async def test_external_cancellation_cancels_remote_exec_and_closes_on_cleanup(
    monkeypatch,
):
    poll_started = asyncio.Event()
    never = asyncio.Event()

    async def route(call):
        method = call["method"]
        url = call["url"]
        payload = call.get("json")
        if method == "POST" and url.endswith("/v1/sandboxes"):
            return _FakeResponse(200, {"id": "sandbox-1"})
        if method == "POST" and url.endswith("/execs"):
            return _FakeResponse(202, {"execId": payload["execId"], "status": "running"})
        if method == "GET" and "/execs/" in url:
            poll_started.set()
            await never.wait()
        if method == "POST" and url.endswith("/cancel"):
            return _FakeResponse(202, {})
        if method == "POST" and url.endswith("/terminate"):
            return _FakeResponse(200, {})
        raise AssertionError(call)

    client = _FakeClient(route)
    monkeypatch.setattr(managed_modal._httpx, "AsyncClient", lambda: client)
    env = managed_modal.ManagedModalEnvironment(image="python:3.11")

    execution = asyncio.create_task(env.execute("sleep 30"))
    await asyncio.wait_for(poll_started.wait(), timeout=1)
    execution.cancel()
    with pytest.raises(asyncio.CancelledError):
        await execution

    assert any(call["url"].endswith("/cancel") for call in client.calls)
    assert client.closed is False
    await env.cleanup()
    assert client.closed is True


@pytest.mark.asyncio
async def test_cleanup_finishes_client_close_before_propagating_cancellation(
    monkeypatch,
):
    terminate_started = asyncio.Event()
    release_terminate = asyncio.Event()

    async def route(call):
        if call["url"].endswith("/v1/sandboxes"):
            return _FakeResponse(200, {"id": "sandbox-1"})
        if call["url"].endswith("/terminate"):
            terminate_started.set()
            await release_terminate.wait()
            return _FakeResponse(200, {})
        raise AssertionError(call)

    client = _FakeClient(route)
    monkeypatch.setattr(managed_modal._httpx, "AsyncClient", lambda: client)
    env = managed_modal.ManagedModalEnvironment(image="python:3.11")
    await env.init_session()

    cleanup = asyncio.create_task(env.cleanup())
    await asyncio.wait_for(terminate_started.wait(), timeout=1)
    cleanup.cancel()
    await asyncio.sleep(0)
    assert cleanup.done() is False
    release_terminate.set()
    with pytest.raises(asyncio.CancelledError):
        await cleanup
    assert client.closed is True


@pytest.mark.asyncio
async def test_managed_modal_error_json_contract(monkeypatch):
    async def route(call):
        if call["url"].endswith("/v1/sandboxes"):
            return _FakeResponse(200, {"id": "sandbox-1"})
        if call["url"].endswith("/execs"):
            return _FakeResponse(403, {"error": "denied"})
        if call["url"].endswith("/terminate"):
            return _FakeResponse(200, {})
        raise AssertionError(call)

    client = _FakeClient(route)
    monkeypatch.setattr(managed_modal._httpx, "AsyncClient", lambda: client)
    env = managed_modal.ManagedModalEnvironment(image="python:3.11")

    result = await env.execute("echo hello")
    await env.cleanup()

    assert result == {
        "output": "Managed Modal exec failed: denied",
        "returncode": 1,
    }


@pytest.mark.asyncio
async def test_gateway_unavailable_error_is_preserved_at_awaited_boundary(monkeypatch):
    async def unavailable(_vendor):
        return None

    monkeypatch.setattr(
        managed_modal,
        "resolve_managed_tool_gateway",
        unavailable,
    )
    env = managed_modal.ManagedModalEnvironment(image="python:3.11")

    with pytest.raises(
        ValueError,
        match="configured tool gateway and Nous user token",
    ):
        await env.execute("true")


@pytest.mark.asyncio
async def test_create_failure_closes_owned_client(monkeypatch):
    async def route(call):
        assert call["url"].endswith("/v1/sandboxes")
        return _FakeResponse(503, {"message": "capacity unavailable"})

    client = _FakeClient(route)
    monkeypatch.setattr(managed_modal._httpx, "AsyncClient", lambda: client)
    env = managed_modal.ManagedModalEnvironment(image="python:3.11")

    with pytest.raises(
        RuntimeError,
        match="Managed Modal create failed: capacity unavailable",
    ):
        await env.init_session()

    assert client.closed is True
    assert env._client is None
    assert env._sandbox_id is None


async def _no_sleep(_seconds):  # noqa: ASYNC124 - coroutine-shaped test double
    return None
