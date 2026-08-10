"""Tests for the Camofox browser backend."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tools import browser_camofox as camofox
from tools.browser_camofox import (
    _rewrite_loopback_url_for_camofox,
    camofox_click,
    camofox_close,
    camofox_console,
    camofox_get_images,
    camofox_navigate,
    camofox_press,
    camofox_snapshot,
    camofox_type,
    camofox_vision,
    check_camofox_available,
    is_camofox_mode,
)


class _CamofoxHTTP:
    """Stateful async transport for exercising the real Camofox HTTP helpers."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.snapshot = ""
        self.health: dict = {"ok": True}
        self.fail_paths: set[str] = set()
        self.connect_error = False

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.connect_error:
            raise httpx.ConnectError("unreachable", request=request)
        if request.url.path in self.fail_paths:
            raise RuntimeError("camofox failed while typing sk-proj-ABCD1234567890EFGH")
        if request.url.path == "/health":
            payload = self.health
        elif request.method == "POST" and request.url.path == "/tabs":
            body = json.loads(request.content)
            payload = {"tabId": "tab-created", "url": body["url"]}
        elif request.method == "GET" and request.url.path.endswith("/snapshot"):
            payload = {"snapshot": self.snapshot, "refsCount": 2 if self.snapshot else 0}
        else:
            payload = {"ok": True, "url": "https://x.com"}
        return httpx.Response(200, json=payload, request=request)


@pytest.fixture(autouse=True)
def _reset_camofox_state():
    camofox._sessions.clear()
    camofox._vnc_url = None
    camofox._vnc_url_checked = False
    camofox._cmd_timeout_resolved = False
    camofox._cached_cmd_timeout = None
    yield
    camofox._sessions.clear()


@pytest.fixture
def camofox_http(monkeypatch) -> _CamofoxHTTP:
    backend = _CamofoxHTTP()
    real_async_client = httpx.AsyncClient

    def client_factory(**kwargs):
        forwarded = {
            key: value
            for key, value in kwargs.items()
            if key not in {"transport", "mounts"}
        }
        return real_async_client(
            transport=httpx.MockTransport(backend.handle),
            **forwarded,
        )

    monkeypatch.setattr(camofox.httpx, "AsyncClient", client_factory)
    return backend


class TestCamofoxMode:
    pytestmark = pytest.mark.asyncio

    async def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("CAMOFOX_URL", raising=False)
        assert await is_camofox_mode() is False

    async def test_health_check_unreachable(self, monkeypatch, camofox_http):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:19999")
        camofox_http.connect_error = True
        assert await check_camofox_available() is False


def _config_with_camofox(**camofox_config):
    return {"browser": {"camofox": camofox_config}}


class TestCamofoxLoopbackRewrite:
    pytestmark = pytest.mark.asyncio

    async def test_rewrites_localhost_when_enabled(self, monkeypatch):
        monkeypatch.delenv("CAMOFOX_REWRITE_LOOPBACK_URLS", raising=False)
        monkeypatch.delenv("CAMOFOX_LOOPBACK_HOST_ALIAS", raising=False)
        with patch(
            "tools.browser_camofox.load_config_readonly",
            new=AsyncMock(
                return_value=_config_with_camofox(rewrite_loopback_urls=True)
            ),
        ):
            rewritten, metadata = await _rewrite_loopback_url_for_camofox(
                "http://127.0.0.1:8766/#settings"
            )

        assert rewritten == "http://host.docker.internal:8766/#settings"
        assert metadata == {
            "from": "127.0.0.1",
            "to": "host.docker.internal",
            "original_url": "http://127.0.0.1:8766/#settings",
            "rewritten_url": "http://host.docker.internal:8766/#settings",
        }

    async def test_env_alias_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_REWRITE_LOOPBACK_URLS", "true")
        monkeypatch.setenv("CAMOFOX_LOOPBACK_HOST_ALIAS", "192.168.1.10")
        with patch(
            "tools.browser_camofox.load_config_readonly",
            new=AsyncMock(
                return_value=_config_with_camofox(
                    rewrite_loopback_urls=False,
                    loopback_host_alias="host.docker.internal",
                )
            ),
        ):
            rewritten, metadata = await _rewrite_loopback_url_for_camofox(
                "http://[::1]:8080/path"
            )

        assert rewritten == "http://192.168.1.10:8080/path"
        assert metadata is not None
        assert metadata["from"] == "::1"
        assert metadata["to"] == "192.168.1.10"


class TestCamofoxNavigate:
    pytestmark = pytest.mark.asyncio

    async def test_creates_tab_on_first_navigate(
        self, monkeypatch, camofox_http
    ):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        result = json.loads(
            await camofox_navigate("https://example.com", task_id="t1")
        )
        assert result["success"] is True
        assert result["url"] == "https://example.com"
        assert camofox_http.requests[0].url.path == "/tabs"

    async def test_connection_error_returns_helpful_message(
        self, monkeypatch, camofox_http
    ):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:19999")
        camofox_http.connect_error = True
        result = json.loads(
            await camofox_navigate("https://example.com", task_id="t_err")
        )
        assert result["success"] is False
        assert "Cannot connect" in result["error"]


class TestCamofoxSnapshot:
    pytestmark = pytest.mark.asyncio

    async def test_no_session_returns_error(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        result = json.loads(await camofox_snapshot(task_id="no_such_task"))
        assert result["success"] is False
        assert "browser_navigate" in result["error"]

    async def test_returns_snapshot(self, monkeypatch, camofox_http):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        await camofox_navigate("https://x.com", task_id="t3")
        camofox_http.snapshot = '- heading "Test" [e1]\n- button "Submit" [e2]'

        result = json.loads(await camofox_snapshot(task_id="t3"))
        assert result["success"] is True
        assert "[e1]" in result["snapshot"]
        assert result["element_count"] == 2


class TestCamofoxInteractions:
    pytestmark = pytest.mark.asyncio

    async def test_click(self, monkeypatch, camofox_http):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        await camofox_navigate("https://x.com", task_id="t4")
        result = json.loads(await camofox_click("@e5", task_id="t4"))
        assert result["success"] is True
        assert result["clicked"] == "e5"

    async def test_type_redacts_api_key(self, monkeypatch, camofox_http):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.setenv("HERMES_REDACT_SECRETS", "true")
        await camofox_navigate("https://x.com", task_id="t5b")
        secret = "sk-proj-ABCD1234567890EFGH"

        result = json.loads(await camofox_type("@apikey", secret, task_id="t5b"))
        assert result["success"] is True
        assert secret not in json.dumps(result)
        assert result["typed"].startswith("sk-pro")

    async def test_type_failure_redacts_api_key(self, monkeypatch, camofox_http):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.setenv("HERMES_REDACT_SECRETS", "true")
        await camofox_navigate("https://x.com", task_id="t5c")
        secret = "sk-proj-ABCD1234567890EFGH"
        camofox_http.fail_paths.add("/tabs/tab-created/type")

        raw_result = await camofox_type("@apikey", secret, task_id="t5c")
        result = json.loads(raw_result)
        assert result["success"] is False
        assert secret not in raw_result
        assert "sk-pro" in raw_result

    async def test_press(self, monkeypatch, camofox_http):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        await camofox_navigate("https://x.com", task_id="t8")
        result = json.loads(await camofox_press("Enter", task_id="t8"))
        assert result["success"] is True
        assert result["pressed"] == "Enter"


class TestCamofoxClose:
    pytestmark = pytest.mark.asyncio

    async def test_close_session(self, monkeypatch, camofox_http):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        await camofox_navigate("https://x.com", task_id="t9")
        result = json.loads(await camofox_close(task_id="t9"))
        assert result["success"] is True
        assert result["closed"] is True
        assert camofox_http.requests[-1].method == "DELETE"

    async def test_close_nonexistent_session(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        result = json.loads(await camofox_close(task_id="nonexistent"))
        assert result["success"] is True


class TestCamofoxConsole:
    pytestmark = pytest.mark.asyncio

    async def test_console_returns_empty_with_note(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        result = json.loads(await camofox_console(task_id="t_console"))
        assert result["success"] is True
        assert result["total_messages"] == 0
        assert "not available" in result["note"]


class TestCamofoxGetImages:
    pytestmark = pytest.mark.asyncio

    async def test_get_images(self, monkeypatch, camofox_http):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        await camofox_navigate("https://x.com", task_id="t10")
        camofox_http.snapshot = '- img "Logo"\n  /url: https://x.com/img.png\n'

        result = json.loads(await camofox_get_images(task_id="t10"))
        assert result["success"] is True
        assert result["count"] == 1
        assert result["images"][0]["src"] == "https://x.com/img.png"


def _vision_response(content: str):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


class TestCamofoxVisionConfig:
    pytestmark = pytest.mark.asyncio

    @pytest.mark.parametrize(
        ("config", "expected_temperature", "expected_timeout", "analysis"),
        [
            (
                {"auxiliary": {"vision": {"temperature": 1, "timeout": 45}}},
                1.0,
                45.0,
                "Camofox screenshot analysis",
            ),
            (
                {"auxiliary": {"vision": {}}},
                0.1,
                120.0,
                "Default camofox screenshot analysis",
            ),
        ],
    )
    async def test_camofox_vision_config(
        self,
        tmp_path,
        monkeypatch,
        config,
        expected_temperature,
        expected_timeout,
        analysis,
    ):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        session = await camofox._get_session("vision")
        session["tab_id"] = "tab-vision"
        raw_response = httpx.Response(200, content=b"fakepng")

        with (
            patch(
                "tools.browser_camofox._get_raw",
                new=AsyncMock(return_value=raw_response),
            ),
            patch(
                "tools.browser_camofox._get",
                new=AsyncMock(return_value={"snapshot": '- button "Submit"'}),
            ),
            patch(
                "tools.browser_camofox.load_config_readonly",
                new=AsyncMock(return_value=config),
            ),
            patch(
                "agent.auxiliary_client.call_llm",
                new=AsyncMock(return_value=_vision_response(analysis)),
            ) as mock_llm,
        ):
            result = json.loads(
                await camofox_vision(
                    "what is on the page?", annotate=True, task_id="vision"
                )
            )

        assert result["success"] is True
        assert result["analysis"] == analysis
        assert mock_llm.call_args.kwargs["temperature"] == expected_temperature
        assert mock_llm.call_args.kwargs["timeout"] == expected_timeout


class TestBrowserToolRouting:
    pytestmark = pytest.mark.asyncio

    async def test_browser_navigate_routes_to_camofox(
        self, monkeypatch, camofox_http
    ):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        from tools.browser_tool import browser_navigate

        with patch(
            "tools.browser_tool._is_safe_url", new=AsyncMock(return_value=True)
        ):
            result = json.loads(
                await browser_navigate("https://example.com", task_id="t_route")
            )
        assert result["success"] is True

    async def test_check_requirements_passes_with_camofox(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        from tools.browser_tool import check_browser_requirements

        assert await check_browser_requirements() is True
