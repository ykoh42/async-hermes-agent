"""Persistence tests for the Camofox browser backend."""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from tools import browser_camofox as camofox
from tools.browser_camofox import (
    _get_session,
    _managed_persistence_enabled,
    camofox_close,
    camofox_navigate,
    camofox_soft_cleanup,
    check_camofox_available,
    get_vnc_url,
)
from tools.browser_camofox_state import get_camofox_identity


def _enable_persistence():
    config = {"browser": {"camofox": {"managed_persistence": True}}}
    return patch(
        "tools.browser_camofox.load_config_readonly",
        new=AsyncMock(return_value=config),
    )


@pytest.fixture(autouse=True)
def _clear_session_state():
    camofox._sessions.clear()
    camofox._vnc_url = None
    camofox._vnc_url_checked = False
    yield
    camofox._sessions.clear()
    camofox._vnc_url = None
    camofox._vnc_url_checked = False


@pytest.fixture
def camofox_requests(monkeypatch) -> list[httpx.Request]:
    requests: list[httpx.Request] = []
    real_async_client = httpx.AsyncClient

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/health":
            payload = {"ok": True, "vncPort": 6080}
        elif request.method == "POST" and request.url.path == "/tabs":
            payload = {
                "tabId": f"tab-{len([r for r in requests if r.url.path == '/tabs'])}",
                "url": json.loads(request.content)["url"],
            }
        elif request.method == "GET" and request.url.path.endswith("/snapshot"):
            payload = {"snapshot": "", "refsCount": 0}
        else:
            payload = {"ok": True}
        return httpx.Response(200, json=payload, request=request)

    def client_factory(**kwargs):
        kwargs.pop("transport", None)
        kwargs.pop("mounts", None)
        return real_async_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(camofox.httpx, "AsyncClient", client_factory)
    return requests


class TestManagedPersistenceToggle:
    pytestmark = pytest.mark.asyncio

    async def test_disabled_by_default(self):
        config = {"browser": {"camofox": {"managed_persistence": False}}}
        with patch(
            "tools.browser_camofox.load_config_readonly",
            new=AsyncMock(return_value=config),
        ):
            assert await _managed_persistence_enabled() is False

    async def test_disabled_on_config_load_error(self):
        with patch(
            "tools.browser_camofox.load_config_readonly",
            new=AsyncMock(side_effect=Exception("fail")),
        ):
            assert await _managed_persistence_enabled() is False


class TestEphemeralMode:
    pytestmark = pytest.mark.asyncio

    async def test_session_gets_random_user_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        session = await _get_session("task-1")
        assert session["user_id"].startswith("hermes_")
        assert session["managed"] is False

    async def test_session_reuse_within_same_task(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        first = await _get_session("task-1")
        second = await _get_session("task-1")
        assert first is second


class TestManagedPersistenceMode:
    pytestmark = pytest.mark.asyncio

    async def test_session_gets_stable_user_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        with _enable_persistence():
            session = await _get_session("task-1")
        expected = get_camofox_identity("task-1")
        assert session["user_id"] == expected["user_id"]
        assert session["session_key"] == expected["session_key"]
        assert session["managed"] is True

    async def test_navigate_reuses_identity_after_close(
        self, tmp_path, monkeypatch, camofox_requests
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        with _enable_persistence():
            first = json.loads(
                await camofox_navigate("https://example.com", task_id="task-1")
            )
            await camofox_close("task-1")
            second = json.loads(
                await camofox_navigate("https://example.com", task_id="task-1")
            )

        assert first["success"] is True
        assert second["success"] is True
        tab_requests = [
            json.loads(request.content)
            for request in camofox_requests
            if request.method == "POST" and request.url.path == "/tabs"
        ]
        assert len(tab_requests) == 2
        assert tab_requests[0]["userId"] == tab_requests[1]["userId"]


class TestConfiguredCamofoxIdentity:
    pytestmark = pytest.mark.asyncio

    async def test_multiplex_scope_identity_wins_over_process_env_and_config(
        self, tmp_path, monkeypatch, camofox_requests
    ):
        from agent import secret_scope

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_URL", "https://default.example")
        monkeypatch.setenv("CAMOFOX_USER_ID", "default-profile-user")
        monkeypatch.setenv("CAMOFOX_SESSION_KEY", "default-profile-session")
        config = {
            "browser": {
                "camofox": {
                    "user_id": "secondary-config-user",
                    "session_key": "secondary-config-session",
                }
            }
        }
        secret_scope.set_multiplex_active(True)
        token = secret_scope.set_secret_scope({
            "CAMOFOX_URL": "https://secondary.example",
            "CAMOFOX_USER_ID": "secondary-scope-user",
            "CAMOFOX_SESSION_KEY": "secondary-scope-session",
        })
        try:
            with patch(
                "tools.browser_camofox.load_config_readonly",
                new=AsyncMock(return_value=config),
            ):
                result = json.loads(
                    await camofox_navigate(
                        "https://example.com", task_id="scoped-precedence"
                    )
                )
        finally:
            secret_scope.reset_secret_scope(token)
            secret_scope.set_multiplex_active(False)

        tab_request = next(
            request
            for request in camofox_requests
            if request.method == "POST" and request.url.path == "/tabs"
        )
        body = json.loads(tab_request.content)
        assert result["success"] is True
        assert str(tab_request.url).startswith("https://secondary.example/tabs")
        assert body["userId"] == "secondary-scope-user"
        assert body["listItemId"] == "secondary-scope-session"

    async def test_multiplex_scope_miss_uses_profile_config_not_process_env(
        self, tmp_path, monkeypatch
    ):
        from agent import secret_scope

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_USER_ID", "default-profile-user")
        monkeypatch.setenv("CAMOFOX_SESSION_KEY", "default-profile-session")
        config = {
            "browser": {
                "camofox": {
                    "user_id": "secondary-config-user",
                    "session_key": "secondary-config-session",
                }
            }
        }
        secret_scope.set_multiplex_active(True)
        token = secret_scope.set_secret_scope({})
        try:
            with patch(
                "tools.browser_camofox.load_config_readonly",
                new=AsyncMock(return_value=config),
            ):
                session = await _get_session("config-fallback")
        finally:
            secret_scope.reset_secret_scope(token)
            secret_scope.set_multiplex_active(False)

        assert session["user_id"] == "secondary-config-user"
        assert session["session_key"] == "secondary-config-session"

    async def test_multiplex_scope_miss_without_config_ignores_process_identity(
        self, tmp_path, monkeypatch
    ):
        from agent import secret_scope

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_USER_ID", "default-profile-user")
        monkeypatch.setenv("CAMOFOX_SESSION_KEY", "default-profile-session")
        secret_scope.set_multiplex_active(True)
        token = secret_scope.set_secret_scope({})
        try:
            with patch(
                "tools.browser_camofox.load_config_readonly",
                new=AsyncMock(return_value={}),
            ):
                session = await _get_session("fail-closed")
        finally:
            secret_scope.reset_secret_scope(token)
            secret_scope.set_multiplex_active(False)

        assert session["user_id"].startswith("hermes_")
        assert session["user_id"] != "default-profile-user"
        assert session["session_key"] == "task_fail-closed"
        assert session["managed"] is False

    async def test_env_identity_overrides_default_identity(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.setenv("CAMOFOX_USER_ID", "shared-camofox")
        monkeypatch.setenv("CAMOFOX_SESSION_KEY", "visible-tab")
        monkeypatch.setenv("CAMOFOX_ADOPT_EXISTING_TAB", "true")
        with patch(
            "tools.browser_camofox._get", new=AsyncMock(return_value={"tabs": []})
        ) as mock_get:
            session = await _get_session("task-1")

        assert session["user_id"] == "shared-camofox"
        assert session["session_key"] == "visible-tab"
        assert session["managed"] is True
        assert session["adopt_existing_tab"] is True
        mock_get.assert_awaited_once_with(
            "/tabs", params={"userId": "shared-camofox"}, timeout=5
        )

    async def test_soft_cleanup_preserves_externally_managed_session(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.setenv("CAMOFOX_USER_ID", "shared-camofox")
        with patch(
            "tools.browser_camofox._get", new=AsyncMock(return_value={"tabs": []})
        ):
            await _get_session("task-1")
        result = await camofox_soft_cleanup("task-1")
        assert result is True
        assert "task-1" not in camofox._sessions


class TestVncUrlDiscovery:
    pytestmark = pytest.mark.asyncio

    async def test_vnc_url_from_health_port(
        self, monkeypatch, camofox_requests
    ):
        monkeypatch.setenv("CAMOFOX_URL", "http://myhost:9377")
        assert await check_camofox_available() is True
        assert await get_vnc_url() == "http://myhost:6080"

    async def test_navigate_includes_vnc_hint(
        self, tmp_path, monkeypatch, camofox_requests
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        camofox._vnc_url = "http://localhost:6080"
        camofox._vnc_url_checked = True
        result = json.loads(
            await camofox_navigate("https://example.com", task_id="vnc-test")
        )
        assert result["vnc_url"] == "http://localhost:6080"
        assert "vnc_hint" in result


class TestCamofoxSoftCleanup:
    pytestmark = pytest.mark.asyncio

    async def test_returns_true_and_drops_session_when_enabled(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        with _enable_persistence():
            await _get_session("task-1")
            result = await camofox_soft_cleanup("task-1")
        assert result is True
        assert "task-1" not in camofox._sessions

    async def test_does_not_call_server_delete(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        with (
            _enable_persistence(),
            patch("tools.browser_camofox._delete", new=AsyncMock()) as mock_delete,
        ):
            await _get_session("task-1")
            await camofox_soft_cleanup("task-1")
        mock_delete.assert_not_awaited()
