"""Native-async contracts for cua-driver 0.10 permission modes."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio


pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture(autouse=True)
async def _reset_computer_use_state():
    from tools.computer_use.tool import reset_backend_for_tests

    await reset_backend_for_tests()
    yield
    await reset_backend_for_tests()


async def test_normal_hermes_session_maps_to_standard_mode():
    from tools.computer_use import tool as computer_use

    with patch(
        "tools.approval.is_approval_bypass_active_for_session",
        return_value=False,
    ):
        assert computer_use._cua_permission_mode("session-a") == "standard"


async def test_any_explicit_hermes_bypass_maps_to_unrestricted_mode():
    from tools.computer_use import tool as computer_use

    with patch(
        "tools.approval.is_approval_bypass_active_for_session",
        return_value=True,
    ):
        assert computer_use._cua_permission_mode("session-a") == "unrestricted"


async def test_gateway_session_key_yolo_maps_to_unrestricted_mode():
    from tools import approval
    from tools.computer_use import tool as computer_use

    gateway_key = "agent:main:telegram:private:12345"
    token = approval.set_current_session_key(gateway_key)
    try:
        await approval.enable_session_yolo(gateway_key)
        assert computer_use._cua_permission_mode("db-sid-xyz") == "unrestricted"
        await approval.disable_session_yolo(gateway_key)
        assert computer_use._cua_permission_mode("db-sid-xyz") == "standard"
    finally:
        await approval.disable_session_yolo(gateway_key)
        try:
            approval.reset_current_session_key(token)
        except Exception:
            approval.set_current_session_key("")


async def test_mode_change_replaces_only_that_sessions_backend():
    from tools.computer_use import tool as computer_use

    created = []

    class _Backend:
        def __init__(self, permission_mode="standard"):
            self.permission_mode = permission_mode
            self.stopped = False
            created.append(self)

        async def start(self):
            pass

        async def stop(self):
            self.stopped = True

    yolo = False
    with patch(
        "tools.approval.is_approval_bypass_active_for_session",
        side_effect=lambda sid: yolo,
    ), patch("tools.computer_use.cua_backend.CuaDriverBackend", _Backend):
        standard = await computer_use._get_backend("session-a")
        other = await computer_use._get_backend("session-b")
        yolo = True
        unrestricted = await computer_use._get_backend("session-a")

    assert standard.permission_mode == "standard"
    assert standard.stopped is True
    assert unrestricted.permission_mode == "unrestricted"
    assert unrestricted is not standard
    assert other.permission_mode == "standard"
    assert other.stopped is False


async def test_mode_change_is_rechecked_after_stale_backend_stops():
    from tools.computer_use import tool as computer_use

    yolo = False
    created = []

    class _Backend:
        def __init__(self, permission_mode="standard"):
            self.permission_mode = permission_mode
            created.append(self)

        async def start(self):
            pass

        async def stop(self):
            nonlocal yolo
            yolo = False

    with patch(
        "tools.approval.is_approval_bypass_active_for_session",
        side_effect=lambda sid: yolo,
    ), patch("tools.computer_use.cua_backend.CuaDriverBackend", _Backend):
        original = await computer_use._get_backend("session-a")
        yolo = True
        replacement = await computer_use._get_backend("session-a")

    assert original.permission_mode == "standard"
    assert replacement.permission_mode == "standard"
    assert replacement is not original
    assert [backend.permission_mode for backend in created] == [
        "standard",
        "standard",
    ]


async def test_release_seam_stops_backend_and_clears_session_state():
    from tools.computer_use import tool as computer_use

    backend = AsyncMock()
    computer_use._backends["session-a"] = backend
    computer_use._backend_call_locks["session-a"] = asyncio.Lock()
    computer_use._backend_permission_modes["session-a"] = "unrestricted"
    computer_use._session_auto_approve["session-a"] = True
    computer_use._always_allow["session-a"] = {("click", "background")}

    assert await computer_use.release_computer_use_session("session-a") is True
    assert await computer_use.release_computer_use_session("session-a") is False
    backend.stop.assert_awaited_once_with()
    assert "session-a" not in computer_use._backend_permission_modes
    assert "session-a" not in computer_use._session_auto_approve
    assert "session-a" not in computer_use._always_allow


async def test_yolo_toggle_immediately_releases_mode_dependent_backend(
    monkeypatch,
):
    from tools import approval

    release = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "tools.computer_use.release_computer_use_session", release
    )
    await approval.enable_session_yolo("session-a")
    await approval.disable_session_yolo("session-a")

    assert [call.args for call in release.await_args_list] == [
        ("session-a",),
        ("session-a",),
    ]


async def test_unrestricted_embedded_daemon_uses_private_socket_and_two_part_ack(
    monkeypatch,
):
    from tools.computer_use import cua_backend

    class _Stderr:
        async def readline(self):
            return b""

    class _Process:
        def __init__(self):
            self.returncode = None
            self.stderr = _Stderr()
            self.wait = AsyncMock(side_effect=lambda: self._mark_stopped())
            self.terminate = lambda: setattr(self, "returncode", -15)

        def _mark_stopped(self):
            self.returncode = 0
            return 0

    process = _Process()
    create = AsyncMock(return_value=process)
    run = AsyncMock(return_value=(0, "running", ""))
    monkeypatch.setattr(cua_backend, "_resolve_mcp_invocation", AsyncMock(return_value=("/opt/cua-driver", ["mcp"])))
    monkeypatch.setattr(cua_backend.asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(cua_backend, "_run_command", run)
    monkeypatch.setattr(
        "tools.environments.local._sanitize_subprocess_env",
        AsyncMock(side_effect=lambda env: env),
    )
    monkeypatch.setattr(cua_backend.aiofiles.os.path, "exists", AsyncMock(return_value=False))

    daemon = cua_backend._EmbeddedCuaDaemon("cua-driver", "unrestricted")
    await daemon.start()
    proxy_command, proxy_args = daemon.proxy_invocation()
    await daemon.stop()

    command = create.await_args.args
    env = create.await_args.kwargs["env"]
    assert command[:2] == ("/opt/cua-driver", "serve")
    assert "--embedded" in command
    assert command[command.index("--permission-mode") + 1] == "unrestricted"
    assert "--dangerously-bypass-approvals" in command
    assert env["CUA_DRIVER_PERMISSION_MODE"] == "unrestricted"
    assert env["CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS"] == "1"
    assert proxy_command == "/opt/cua-driver"
    assert proxy_args == ["mcp", "--embedded", "--socket", daemon.socket_path]


async def test_standard_backend_does_not_spawn_an_embedded_daemon():
    from tools.computer_use.cua_backend import CuaDriverBackend

    standard = CuaDriverBackend(permission_mode="standard")
    unrestricted = CuaDriverBackend(permission_mode="unrestricted")
    assert standard._embedded_daemon is None
    assert unrestricted._embedded_daemon is not None
