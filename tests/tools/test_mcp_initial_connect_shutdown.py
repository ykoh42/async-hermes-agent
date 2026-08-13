"""Regression tests for initial MCP failure ownership and teardown."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from pyleak import no_task_leaks
from pyleak.eventloop import LeakAction


async def _reset_mcp_state(mcp_tool) -> None:
    await mcp_tool.shutdown_mcp_servers()
    with mcp_tool._lock:
        mcp_tool._servers.clear()
        mcp_tool._server_connecting.clear()
        mcp_tool._server_connect_errors.clear()


async def _cleanup_mcp_state(mcp_tool, extra_servers=()) -> None:
    for server in extra_servers:
        task = getattr(server, "_task", None)
        if task is not None and not task.done():
            await server.shutdown()
    await mcp_tool.shutdown_mcp_servers()
    with mcp_tool._lock:
        mcp_tool._servers.clear()
        mcp_tool._server_connecting.clear()
        mcp_tool._server_connect_errors.clear()


@pytest.mark.asyncio
async def test_initial_connect_failure_is_registry_owned_and_reaped(monkeypatch, tmp_path):
    """Normal discovery must retain the parked task for clean shutdown."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    await _reset_mcp_state(mcp_tool)
    created = []

    class _FailingServerTask(mcp_tool.MCPServerTask):
        def __init__(self, name):
            super().__init__(name)
            created.append(self)

        async def _run_stdio(self, config):
            raise ConnectionError("deterministic initial failure")

    monkeypatch.setattr(mcp_tool, "MCPServerTask", _FailingServerTask)
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "_MAX_INITIAL_CONNECT_RETRIES", 0)
    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 3600)

    try:
        assert await mcp_tool.register_mcp_servers({
            "initial-failure": {"command": "unused", "connect_timeout": 5}
        }) == []

        assert len(created) == 1
        server = created[0]
        with mcp_tool._lock:
            assert mcp_tool._servers["initial-failure"] is server
            assert "deterministic initial failure" in (
                mcp_tool._server_connect_errors["initial-failure"]
            )
        assert server._task is not None
        assert not server._task.done(), "recoverable initial failure was not parked"

        await mcp_tool.shutdown_mcp_servers()
        assert server._task.done()
        with mcp_tool._lock:
            assert not mcp_tool._servers
    finally:
        await _cleanup_mcp_state(mcp_tool, created)


@pytest.mark.asyncio
async def test_initial_connect_failure_revives_same_registered_server(monkeypatch, tmp_path):
    """A cached parked failure must revive through register_mcp_servers()."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.registry import ToolRegistry
    import tools.registry as registry_module

    await _reset_mcp_state(mcp_tool)
    created = []
    backend_up = asyncio.Event()
    revived = asyncio.Event()
    state = {"transport_calls": 0, "tool_calls": 0}
    mock_registry = ToolRegistry()

    class _Session:
        async def call_tool(self, name, arguments):
            state["tool_calls"] += 1
            return SimpleNamespace(
                isError=False,
                content=[SimpleNamespace(text=f"revived:{arguments['value']}")],
                structuredContent=None,
            )

    class _RecoveringServerTask(mcp_tool.MCPServerTask):
        def __init__(self, name):
            super().__init__(name)
            created.append(self)

        async def _run_stdio(self, config):
            assert mcp_tool._connect_server_claim.get() is None
            state["transport_calls"] += 1
            if not backend_up.is_set():
                raise ConnectionError("backend still booting")

            self.session = _Session()
            self._tools = [SimpleNamespace(
                name="ping",
                description="Return a deterministic revival result",
                inputSchema={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            )]
            # Match the real transports: discovery runs before _ready is set.
            await self._register_discovered_tools_if_needed()
            self._ready.set()
            revived.set()
            return await self._wait_for_lifecycle_event()

    monkeypatch.setattr(mcp_tool, "MCPServerTask", _RecoveringServerTask)
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "_MAX_INITIAL_CONNECT_RETRIES", 0)
    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 3600)
    monkeypatch.setattr(registry_module, "registry", mock_registry)

    config = {
        "recovering": {"command": "unused", "connect_timeout": 5}
    }

    try:
        assert await mcp_tool.register_mcp_servers(config) == []
        assert len(created) == 1
        server = created[0]
        with mcp_tool._lock:
            assert mcp_tool._servers["recovering"] is server
            assert "backend still booting" in (
                mcp_tool._server_connect_errors["recovering"]
            )
        assert not server._task.done()

        backend_up.set()
        await mcp_tool.register_mcp_servers(config)

        await asyncio.wait_for(revived.wait(), timeout=5)
        assert len(created) == 1, "revival created a duplicate server task"
        with mcp_tool._lock:
            assert mcp_tool._servers["recovering"] is server
            assert "recovering" not in mcp_tool._server_connect_errors
        assert state["transport_calls"] == 2
        assert server.session is not None
        assert server._error is None

        entry = mock_registry.get_entry("mcp__recovering__ping")
        assert entry is not None
        assert entry.check_fn() is True
        assert json.loads(await entry.handler({"value": "ok"})) == {
            "result": "revived:ok"
        }
        assert state["tool_calls"] == 1
    finally:
        await _cleanup_mcp_state(mcp_tool, created)


@pytest.mark.asyncio
async def test_terminal_initial_failure_is_not_retained(monkeypatch, tmp_path):
    """A non-recoverable startup error must not leave a dead cache entry."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    await _reset_mcp_state(mcp_tool)
    created = []

    class _AuthFailingServerTask(mcp_tool.MCPServerTask):
        def __init__(self, name):
            super().__init__(name)
            created.append(self)

        async def _run_stdio(self, config):
            raise PermissionError("terminal authentication failure")

    monkeypatch.setattr(mcp_tool, "MCPServerTask", _AuthFailingServerTask)
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "_is_auth_error", lambda exc: True)

    try:
        assert await mcp_tool.register_mcp_servers({
            "auth-failure": {"command": "unused", "connect_timeout": 5}
        }) == []
        assert len(created) == 1
        assert created[0]._task.done()
        with mcp_tool._lock:
            assert "auth-failure" not in mcp_tool._servers
            assert "terminal authentication failure" in (
                mcp_tool._server_connect_errors["auth-failure"]
            )
    finally:
        await _cleanup_mcp_state(mcp_tool, created)


@pytest.mark.asyncio
async def test_standalone_failed_connect_is_reaped_without_global_owner(monkeypatch, tmp_path):
    """Probe-only _connect_server failures must not publish parked servers."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    await _reset_mcp_state(mcp_tool)
    created = []

    class _ProbeServerTask(mcp_tool.MCPServerTask):
        def __init__(self, name):
            super().__init__(name)
            created.append(self)

        async def _run_stdio(self, config):
            raise ConnectionError("probe target unavailable")

    monkeypatch.setattr(mcp_tool, "MCPServerTask", _ProbeServerTask)
    monkeypatch.setattr(mcp_tool, "_MAX_INITIAL_CONNECT_RETRIES", 0)
    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 3600)
    try:
        with pytest.raises(ConnectionError, match="probe target unavailable"):
            await mcp_tool._connect_server("probe-only", {"command": "unused"})

        assert len(created) == 1
        assert created[0]._task.done()
        with mcp_tool._lock:
            assert "probe-only" not in mcp_tool._servers
            assert "probe-only" not in mcp_tool._server_connect_errors
    finally:
        await _cleanup_mcp_state(mcp_tool, created)


@pytest.mark.asyncio
async def test_standalone_failed_connect_cleanup_preserves_new_cancellation(monkeypatch):
    """Cancellation during orphan cleanup supersedes the earlier connect error."""
    from tools import mcp_tool

    shutdown_started = asyncio.Event()
    shutdown_cancelled = asyncio.Event()

    class _StalledCleanupServer:
        def __init__(self, name):
            self.name = name

        async def start(self, config):
            raise ConnectionError("probe target unavailable")

        async def shutdown(self):
            shutdown_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                shutdown_cancelled.set()
                raise

    monkeypatch.setattr(mcp_tool, "MCPServerTask", _StalledCleanupServer)

    async with no_task_leaks(action=LeakAction.RAISE):
        connect = asyncio.create_task(
            mcp_tool._connect_server("probe-only", {"command": "unused"})
        )
        await asyncio.wait_for(shutdown_started.wait(), timeout=1)
        connect.cancel()
        with pytest.raises(asyncio.CancelledError):
            await connect

    assert shutdown_cancelled.is_set()

@pytest.mark.asyncio
async def test_global_shutdown_closes_owned_stderr_log():
    """The shared subprocess stderr descriptor ends with the MCP lifecycle."""
    from unittest.mock import AsyncMock

    from tools import mcp_tool

    stderr_log = AsyncMock()
    scope = await mcp_tool._activate_mcp_scope()
    mcp_tool._mcp_stderr_log_fhs[scope] = stderr_log

    await mcp_tool.shutdown_mcp_servers()

    stderr_log.flush.assert_awaited_once_with()
    stderr_log.close.assert_awaited_once_with()
    assert scope not in mcp_tool._mcp_stderr_log_fhs


@pytest.mark.asyncio
async def test_global_shutdown_closes_unconsumed_oauth_port_reservations():
    """A provider built without starting auth releases its reserved socket."""
    from unittest.mock import MagicMock

    from tools import mcp_oauth, mcp_tool

    reserved_socket = MagicMock()
    mcp_oauth._reserved_sockets[49399] = reserved_socket
    mcp_oauth._oauth_port = 49399

    await mcp_tool.shutdown_mcp_servers()

    reserved_socket.close.assert_called_once_with()
    assert mcp_oauth._reserved_sockets == {}
    assert mcp_oauth._oauth_port is None


@pytest.mark.asyncio
async def test_global_shutdown_cancels_owned_oauth_manager_work():
    """The manager cache and deduplicated recovery tasks end with MCP."""
    from tools import mcp_oauth_manager, mcp_tool

    mcp_oauth_manager.reset_manager_for_tests()
    manager = mcp_oauth_manager.get_manager()
    entry = mcp_oauth_manager._ProviderEntry(
        server_url="https://mcp.example.invalid",
        oauth_config=None,
    )
    pending = asyncio.get_running_loop().create_future()
    entry.pending_401["token"] = pending
    manager._entries[manager._key("server")] = entry
    recovery_task = asyncio.create_task(asyncio.Event().wait())
    manager._inflight_tasks.add(recovery_task)

    await mcp_tool.shutdown_mcp_servers()

    assert pending.cancelled()
    assert recovery_task.cancelled()
    assert manager._entries == {}
    assert manager._inflight_tasks == set()
    assert mcp_oauth_manager._MANAGER is manager


def test_stderr_log_is_owned_and_closed_by_each_runtime_loop(tmp_path, monkeypatch):
    """Sequential asyncio runtimes never reuse an obsolete aiofiles wrapper."""
    from tools import mcp_tool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    async def exercise_runtime(name):
        await mcp_tool._write_stderr_log_header(name)
        scope = await mcp_tool._activate_mcp_scope()
        stderr_log = mcp_tool._mcp_stderr_log_fhs[scope]
        await mcp_tool.shutdown_mcp_servers()
        assert scope not in mcp_tool._mcp_stderr_log_fhs
        return stderr_log

    first_log = asyncio.run(exercise_runtime("first-loop"))
    second_log = asyncio.run(exercise_runtime("second-loop"))

    assert first_log.closed is True
    assert second_log.closed is True
    assert first_log is not second_log
