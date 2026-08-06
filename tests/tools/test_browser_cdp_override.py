import asyncio
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest


HOST = "example-host"
PORT = 9223
WS_URL = f"ws://{HOST}:{PORT}/devtools/browser/abc123"
HTTP_URL = f"http://{HOST}:{PORT}"
VERSION_URL = f"{HTTP_URL}/json/version"


def _client_factory(handler):
    real_async_client = httpx.AsyncClient

    def factory(**kwargs):
        return real_async_client(transport=httpx.MockTransport(handler), **kwargs)

    return factory


class TestResolveCdpOverride:
    pytestmark = pytest.mark.asyncio

    async def test_keeps_full_devtools_websocket_url(self):
        from tools.browser_tool import _resolve_cdp_override

        assert await _resolve_cdp_override(WS_URL) == WS_URL

    async def test_redacts_secret_query_params_in_success_log(self):
        from tools.browser_tool import _resolve_cdp_override

        raw = "https://cdp.example/json/version?access_token=super-secret-token-123456"
        resolved_ws = (
            "wss://cdp.example/devtools/browser/abc?token=super-secret-token-123456"
        )
        requests = []

        async def handler(request):
            requests.append(request)
            return httpx.Response(
                200,
                json={"webSocketDebuggerUrl": resolved_ws},
                request=request,
            )

        with (
            patch(
                "tools.browser_tool.httpx.AsyncClient",
                side_effect=_client_factory(handler),
            ),
            patch("tools.browser_tool.logger.info") as mock_info,
        ):
            resolved = await _resolve_cdp_override(raw)

        assert resolved == resolved_ws
        assert len(requests) == 1
        mock_info.assert_called_once()
        _, logged_raw, logged_ws = mock_info.call_args.args
        assert "super-secret-token-123456" not in logged_raw
        assert "super-secret-token-123456" not in logged_ws
        assert "access_token=***" in logged_raw
        assert "token=***" in logged_ws

    async def test_redacts_secret_query_params_in_failure_log(self):
        from tools.browser_tool import _resolve_cdp_override

        raw = "https://cdp.example?access_token=super-secret-token-123456"

        async def handler(request):
            raise RuntimeError(
                "upstream rejected https://cdp.example/json/version?"
                "access_token=super-secret-token-123456"
            )

        with (
            patch(
                "tools.browser_tool.httpx.AsyncClient",
                side_effect=_client_factory(handler),
            ),
            patch("tools.browser_tool.logger.warning") as mock_warning,
        ):
            resolved = await _resolve_cdp_override(raw)

        assert resolved == raw
        mock_warning.assert_called_once()
        _, logged_raw, logged_version_url, logged_error = mock_warning.call_args.args
        assert "super-secret-token-123456" not in logged_raw
        assert "super-secret-token-123456" not in logged_version_url
        assert "super-secret-token-123456" not in logged_error
        assert "access_token=***" in logged_raw
        assert "access_token=***" in logged_version_url
        assert "access_token=***" in logged_error

    async def test_normalizes_provider_returned_http_cdp_url_when_creating_session(
        self, monkeypatch
    ):
        import tools.browser_tool as browser_tool

        provider = Mock()
        provider.create_session = AsyncMock(return_value={
            "session_name": "cloud-session",
            "bb_session_id": "bu_123",
            "cdp_url": "https://cdp.browser-use.example/session",
            "features": {"browser_use": True},
        })
        requests = []

        async def handler(request):
            requests.append(request)
            return httpx.Response(
                200, json={"webSocketDebuggerUrl": WS_URL}, request=request
            )

        monkeypatch.setattr(browser_tool, "_active_sessions", {})
        monkeypatch.setattr(browser_tool, "_session_last_activity", {})
        monkeypatch.setattr(
            browser_tool,
            "_start_browser_cleanup_thread",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(browser_tool, "_update_session_activity", lambda _task: None)
        monkeypatch.setattr(
            browser_tool, "_get_cdp_override", AsyncMock(return_value="")
        )
        monkeypatch.setattr(
            browser_tool, "_get_cloud_provider", AsyncMock(return_value=provider)
        )

        with patch(
            "tools.browser_tool.httpx.AsyncClient",
            side_effect=_client_factory(handler),
        ):
            session_info = await browser_tool._get_session_info("task-browser-use")

        assert session_info["cdp_url"] == WS_URL
        provider.create_session.assert_awaited_once_with("task-browser-use")
        assert str(requests[0].url) == (
            "https://cdp.browser-use.example/session/json/version"
        )


class TestGetCdpOverride:
    pytestmark = pytest.mark.asyncio

    async def test_prefers_env_var_over_config(self, monkeypatch):
        import tools.browser_tool as browser_tool

        monkeypatch.setenv("BROWSER_CDP_URL", HTTP_URL)
        config_mock = AsyncMock(
            return_value={"browser": {"cdp_url": "http://config-host:9222"}}
        )
        monkeypatch.setattr(browser_tool, "load_config_readonly", config_mock)
        requests = []

        async def handler(request):
            requests.append(request)
            return httpx.Response(
                200, json={"webSocketDebuggerUrl": WS_URL}, request=request
            )

        with patch(
            "tools.browser_tool.httpx.AsyncClient",
            side_effect=_client_factory(handler),
        ):
            resolved = await browser_tool._get_cdp_override()

        assert resolved == WS_URL
        assert str(requests[0].url) == VERSION_URL
        config_mock.assert_not_awaited()

    async def test_uses_config_browser_cdp_url_when_env_missing(self, monkeypatch):
        import tools.browser_tool as browser_tool

        monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
        monkeypatch.setattr(
            browser_tool,
            "load_config_readonly",
            AsyncMock(return_value={"browser": {"cdp_url": HTTP_URL}}),
        )
        requests = []

        async def handler(request):
            requests.append(request)
            return httpx.Response(
                200, json={"webSocketDebuggerUrl": WS_URL}, request=request
            )

        with patch(
            "tools.browser_tool.httpx.AsyncClient",
            side_effect=_client_factory(handler),
        ):
            resolved = await browser_tool._get_cdp_override()

        assert resolved == WS_URL
        assert str(requests[0].url) == VERSION_URL

    async def test_camofox_yields_to_config_cdp_override(self, monkeypatch):
        import tools.browser_camofox as camofox

        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
        with patch(
            "tools.browser_camofox.load_config_readonly",
            new=AsyncMock(return_value={}),
        ):
            assert await camofox.is_camofox_mode() is True
        with patch(
            "tools.browser_camofox.load_config_readonly",
            new=AsyncMock(return_value={"browser": {"cdp_url": HTTP_URL}}),
        ):
            assert await camofox.is_camofox_mode() is False
        monkeypatch.setenv("BROWSER_CDP_URL", HTTP_URL)
        assert await camofox.is_camofox_mode() is False


class TestCreateCdpSession:
    def test_redacts_token_in_session_creation_log(self):
        from tools.browser_tool import _create_cdp_session

        cdp_url = "wss://cdp.example/devtools/browser/abc?token=super-secret-token-999"
        with patch("tools.browser_tool.logger.info") as mock_info:
            result = _create_cdp_session("task-1", cdp_url)
        assert result["cdp_url"] == cdp_url
        logged_args = " ".join(str(arg) for arg in mock_info.call_args.args)
        assert "super-secret-token-999" not in logged_args
        assert "token=***" in logged_args

    def test_plain_url_without_secrets_passes_through(self):
        from tools.browser_tool import _create_cdp_session

        plain_url = "ws://localhost:9222/devtools/browser/abc123"
        with patch("tools.browser_tool.logger.info") as mock_info:
            _create_cdp_session("task-2", plain_url)
        assert "localhost:9222" in " ".join(
            str(arg) for arg in mock_info.call_args.args
        )


class TestCDPSupervisorStartRedaction:
    pytestmark = pytest.mark.asyncio

    async def test_timeout_error_redacts_query_token(self):
        from tools.browser_supervisor import CDPSupervisor

        raw = "wss://cdp.example/devtools/browser/abc?token=super-secret-999"
        supervisor = CDPSupervisor("test-task", raw)

        async def never_ready():
            await asyncio.Event().wait()

        supervisor._run = never_ready
        with pytest.raises(TimeoutError) as caught:
            await supervisor.start(timeout=0.001)
        assert "super-secret-999" not in str(caught.value)
        assert "cdp_url=" in str(caught.value)

    async def test_timeout_error_preserves_plain_url(self):
        from tools.browser_supervisor import CDPSupervisor

        raw = "ws://127.0.0.1:9222/devtools/browser/abc"
        supervisor = CDPSupervisor("test-task", raw)

        async def never_ready():
            await asyncio.Event().wait()

        supervisor._run = never_ready
        with pytest.raises(TimeoutError) as caught:
            await supervisor.start(timeout=0.001)
        assert "127.0.0.1:9222" in str(caught.value)

    @pytest.mark.parametrize(
        "raw",
        [
            "wss://cdp.example/devtools/browser/abc?token=super-secret-999",
            "wss://user:p4ssw0rd@cdp.example/devtools/browser/x",
        ],
    )
    async def test_start_error_redacts_credentials(self, raw):
        from tools.browser_supervisor import CDPSupervisor

        error = ValueError(f"{raw} isn't a valid URI: hostname isn't provided")
        supervisor = CDPSupervisor("test-task", raw)

        async def fail_start():
            supervisor._start_error = error
            supervisor._ready_event.set()

        supervisor._run = fail_start
        with pytest.raises(RuntimeError) as caught:
            await supervisor.start(timeout=1)
        message = str(caught.value)
        assert "super-secret-999" not in message
        assert "p4ssw0rd" not in message
        assert caught.value.__cause__ is None
        assert caught.value.__suppress_context__ is True


class TestRedactCdpErrorText:
    def test_masks_query_token_in_exception(self):
        from tools.browser_supervisor import _redact_cdp_error_text

        output = _redact_cdp_error_text(
            ConnectionError("connect wss://h/x?token=leak-me failed")
        )
        assert "leak-me" not in output

    def test_preserves_non_secret_context(self):
        from tools.browser_supervisor import _redact_cdp_error_text

        output = _redact_cdp_error_text(
            ConnectionError("connect ws://127.0.0.1:9222/x failed: refused")
        )
        assert "127.0.0.1:9222" in output
        assert "refused" in output
