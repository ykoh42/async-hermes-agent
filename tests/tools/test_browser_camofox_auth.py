"""Tests that Camofox browser sends Authorization header when CAMOFOX_API_KEY is set.

Regression test for https://github.com/NousResearch/hermes-agent/issues/20476
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.browser_camofox import (
    _auth_headers,
    camofox_back,
    camofox_click,
    camofox_close,
    camofox_navigate,
    camofox_press,
    camofox_scroll,
    camofox_snapshot,
    camofox_type,
    get_camofox_url,
)


def _mock_response(status=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_data or {}
    resp.content = b"\x89PNG\r\n\x1a\nfake"
    resp.raise_for_status = MagicMock()
    return resp


class TestAuthHeaders:
    """Unit tests for _auth_headers() helper."""

    def test_empty_when_no_key(self, monkeypatch):
        monkeypatch.delenv("CAMOFOX_API_KEY", raising=False)
        assert _auth_headers() == {}


    def test_empty_when_key_blank(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_API_KEY", "   ")
        assert _auth_headers() == {}

    def test_multiplex_scope_key_wins_over_process_environment(self, monkeypatch):
        from agent import secret_scope

        monkeypatch.setenv("CAMOFOX_API_KEY", "default-profile-key")
        secret_scope.set_multiplex_active(True)
        token = secret_scope.set_secret_scope({"CAMOFOX_API_KEY": "secondary-profile-key"})
        try:
            assert _auth_headers() == {"Authorization": "Bearer secondary-profile-key"}
        finally:
            secret_scope.reset_secret_scope(token)
            secret_scope.set_multiplex_active(False)

    def test_multiplex_scope_missing_key_fails_closed(self, monkeypatch):
        from agent import secret_scope

        monkeypatch.setenv("CAMOFOX_API_KEY", "default-profile-key")
        secret_scope.set_multiplex_active(True)
        token = secret_scope.set_secret_scope({})
        try:
            assert _auth_headers() == {}
        finally:
            secret_scope.reset_secret_scope(token)
            secret_scope.set_multiplex_active(False)

    def test_multiplex_scope_keeps_endpoint_and_key_in_same_profile(self, monkeypatch):
        from agent import secret_scope

        monkeypatch.setenv("CAMOFOX_URL", "https://default.example")
        monkeypatch.setenv("CAMOFOX_API_KEY", "default-profile-key")
        secret_scope.set_multiplex_active(True)
        token = secret_scope.set_secret_scope(
            {
                "CAMOFOX_URL": "https://secondary.example/",
                "CAMOFOX_API_KEY": "secondary-profile-key",
            }
        )
        try:
            assert get_camofox_url() == "https://secondary.example"
            assert _auth_headers() == {
                "Authorization": "Bearer secondary-profile-key"
            }
        finally:
            secret_scope.reset_secret_scope(token)
            secret_scope.set_multiplex_active(False)


class TestAuthHeadersSent:
    """Verify all HTTP call sites include auth headers when CAMOFOX_API_KEY is set."""

    pytestmark = pytest.mark.asyncio

    @pytest.fixture(autouse=True)
    def _set_key(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.setenv("CAMOFOX_API_KEY", "my-api-key")

    @patch("tools.browser_camofox._get", new_callable=AsyncMock)
    @patch("tools.browser_camofox.httpx.AsyncClient")
    async def test_ensure_tab_sends_auth(self, client_cls, mock_get):
        response = _mock_response(json_data={"tabId": "t1"})
        client = client_cls.return_value.__aenter__.return_value
        client.post = AsyncMock(return_value=response)
        mock_get.return_value = {"snapshot": "", "refsCount": 0}

        await camofox_navigate("https://example.com", task_id="auth_test_1")

        _, kwargs = client.post.call_args
        assert kwargs["headers"] == {"Authorization": "Bearer my-api-key"}

    @patch("tools.browser_camofox.httpx.AsyncClient")
    async def test_delete_sends_auth(self, client_cls):
        from tools.browser_camofox import _get_session

        session = await _get_session("auth_test_4")
        session["tab_id"] = "t4"
        response = _mock_response(json_data={"ok": True})
        client = client_cls.return_value.__aenter__.return_value
        client.request = AsyncMock(return_value=response)

        await camofox_close(task_id="auth_test_4")

        _, kwargs = client.request.call_args
        assert kwargs["headers"] == {"Authorization": "Bearer my-api-key"}


class TestNoAuthHeadersWhenKeyUnset:
    """Verify HTTP calls send empty headers when CAMOFOX_API_KEY is not set."""

    pytestmark = pytest.mark.asyncio

    @pytest.fixture(autouse=True)
    def _unset_key(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.delenv("CAMOFOX_API_KEY", raising=False)

    @patch("tools.browser_camofox._get", new_callable=AsyncMock)
    @patch("tools.browser_camofox.httpx.AsyncClient")
    async def test_no_auth_on_tab_creation(self, client_cls, mock_get):
        response = _mock_response(json_data={"tabId": "t5"})
        client = client_cls.return_value.__aenter__.return_value
        client.post = AsyncMock(return_value=response)
        mock_get.return_value = {"snapshot": "", "refsCount": 0}

        await camofox_navigate("https://example.com", task_id="noauth_test_1")

        _, kwargs = client.post.call_args
        assert kwargs.get("headers") == {}
