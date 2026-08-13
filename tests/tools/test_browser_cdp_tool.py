"""Native-async tests for the raw browser CDP tool."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
import websockets
from websockets.asyncio.server import serve

from tools import browser_cdp_tool


class _CDPServer:
    """Tiny same-event-loop CDP-over-WebSocket server."""

    def __init__(self) -> None:
        self._handlers: dict[str, Any] = {}
        self._responses: list[dict[str, Any]] = []
        self._server: Any = None
        self._host = "127.0.0.1"
        self._port = 0

    def on(self, method: str, handler) -> None:
        self._handlers[method] = handler

    async def start(self) -> str:
        async def handler(ws):
            try:
                async for raw in ws:
                    message = json.loads(raw)
                    call_id = message.get("id")
                    method = message.get("method", "")
                    params = message.get("params", {}) or {}
                    session_id = message.get("sessionId")
                    self._responses.append(message)
                    fn = self._handlers.get(method)
                    if fn is None:
                        reply = {
                            "id": call_id,
                            "error": {
                                "code": -32601,
                                "message": f"No handler for {method}",
                            },
                        }
                    else:
                        try:
                            result = fn(params, session_id)
                            if isinstance(result, Exception):
                                raise result
                            reply = {"id": call_id, "result": result}
                        except Exception as exc:
                            reply = {
                                "id": call_id,
                                "error": {"code": -1, "message": str(exc)},
                            }
                    if session_id:
                        reply["sessionId"] = session_id
                    await ws.send(json.dumps(reply))
            except websockets.exceptions.ConnectionClosed:
                pass

        self._server = await serve(handler, self._host, 0)
        socket = next(iter(self._server.sockets))
        self._port = socket.getsockname()[1]
        return f"ws://{self._host}:{self._port}/devtools/browser/mock"

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    def received(self) -> list[dict[str, Any]]:
        return list(self._responses)


@pytest_asyncio.fixture
async def cdp_server(monkeypatch):
    server = _CDPServer()
    ws_url = await server.start()
    monkeypatch.setattr(
        browser_cdp_tool,
        "_resolve_cdp_endpoint",
        AsyncMock(return_value=ws_url),
    )
    try:
        yield server
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_missing_method_returns_error():
    result = json.loads(await browser_cdp_tool.browser_cdp(method=""))
    assert "error" in result
    assert "method" in result["error"].lower()
    assert result.get("cdp_docs") == browser_cdp_tool.CDP_DOCS_URL


@pytest.mark.asyncio
async def test_non_string_method_returns_error():
    result = json.loads(await browser_cdp_tool.browser_cdp(method=123))
    assert "error" in result
    assert "method" in result["error"].lower()


@pytest.mark.asyncio
async def test_no_endpoint_returns_helpful_error(monkeypatch):
    monkeypatch.setattr(
        browser_cdp_tool, "_resolve_cdp_endpoint", AsyncMock(return_value="")
    )
    result = json.loads(
        await browser_cdp_tool.browser_cdp(method="Target.getTargets")
    )
    assert "error" in result
    assert "/browser connect" in result["error"]
    assert result.get("cdp_docs") == browser_cdp_tool.CDP_DOCS_URL


@pytest.mark.asyncio
async def test_browser_level_redacts_secret_result(cdp_server):
    fake_key = "sk-" + "CDPSECRETRESULT1234567890"
    cdp_server.on(
        "Runtime.evaluate",
        lambda params, sid: {"result": {"type": "string", "value": fake_key}},
    )
    result = json.loads(
        await browser_cdp_tool.browser_cdp(method="Runtime.evaluate")
    )
    assert result["success"] is True
    assert "CDPSECRETRESULT" not in json.dumps(result)
    assert result["result"]["result"]["value"].startswith("sk-")


PRIVATE_URL = "http://169.254.169.254/latest/meta-data/"


async def _true(*_args, **_kwargs):
    return True


@pytest.mark.asyncio
async def test_runtime_evaluate_blocked_when_current_page_is_private(monkeypatch):
    calls = []
    monkeypatch.setattr(
        browser_cdp_tool,
        "_resolve_cdp_endpoint",
        AsyncMock(return_value="ws://127.0.0.1:9222/devtools/browser/mock"),
    )
    import tools.browser_tool as browser_tool

    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", _true)
    monkeypatch.setattr(
        browser_tool,
        "_current_page_private_url",
        AsyncMock(return_value=PRIVATE_URL),
    )

    async def fake_call(*args, **kwargs):
        calls.append((args, kwargs))
        return {"result": {"value": "private data"}}

    monkeypatch.setattr(browser_cdp_tool, "_cdp_call", fake_call)
    result = json.loads(
        await browser_cdp_tool.browser_cdp(
            method="Runtime.evaluate",
            params={"expression": "document.body.innerText"},
            task_id="task-1",
        )
    )
    assert "error" in result
    assert PRIVATE_URL in result["error"]
    assert "private or internal address" in result["error"]
    assert calls == []


@pytest.mark.asyncio
async def test_frame_id_route_blocked_when_current_page_is_private(monkeypatch):
    supervisor_calls = []
    import tools.browser_tool as browser_tool

    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", _true)
    monkeypatch.setattr(
        browser_tool,
        "_current_page_private_url",
        AsyncMock(return_value=PRIVATE_URL),
    )

    async def fake_supervisor_route(**kwargs):
        supervisor_calls.append(kwargs)
        return json.dumps({"success": True, "result": {"value": "private data"}})

    monkeypatch.setattr(
        browser_cdp_tool, "_browser_cdp_via_supervisor", fake_supervisor_route
    )
    result = json.loads(
        await browser_cdp_tool.browser_cdp(
            method="Runtime.evaluate",
            params={"expression": "document.body.innerText"},
            frame_id="frame-1",
            task_id="task-1",
        )
    )
    assert "error" in result
    assert PRIVATE_URL in result["error"]
    assert supervisor_calls == []


@pytest.mark.asyncio
async def test_frame_id_route_allowed_when_page_is_not_private(monkeypatch):
    supervisor_calls = []
    import tools.browser_tool as browser_tool

    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", _true)
    monkeypatch.setattr(
        browser_tool, "_current_page_private_url", AsyncMock(return_value=None)
    )

    async def fake_supervisor_route(**kwargs):
        supervisor_calls.append(kwargs)
        return json.dumps({"success": True, "result": {"value": "ok"}})

    monkeypatch.setattr(
        browser_cdp_tool, "_browser_cdp_via_supervisor", fake_supervisor_route
    )
    result = json.loads(
        await browser_cdp_tool.browser_cdp(
            method="Runtime.evaluate",
            params={"expression": "document.title"},
            frame_id="frame-1",
            task_id="task-1",
        )
    )
    assert result.get("success") is True
    assert len(supervisor_calls) == 1


@pytest.mark.asyncio
async def test_page_navigate_to_private_url_blocked_before_cdp(monkeypatch):
    calls = []
    monkeypatch.setattr(
        browser_cdp_tool,
        "_resolve_cdp_endpoint",
        AsyncMock(return_value="ws://127.0.0.1:9222/devtools/browser/mock"),
    )
    import tools.browser_tool as browser_tool

    monkeypatch.setattr(browser_tool, "_eval_ssrf_guard_active", _true)

    async def fake_call(*args, **kwargs):
        calls.append((args, kwargs))
        return {"frameId": "f"}

    monkeypatch.setattr(browser_cdp_tool, "_cdp_call", fake_call)
    result = json.loads(
        await browser_cdp_tool.browser_cdp(
            method="Page.navigate",
            params={"url": PRIVATE_URL},
            task_id="task-1",
        )
    )
    assert "error" in result
    assert PRIVATE_URL in result["error"]
    assert calls == []


@pytest.mark.asyncio
async def test_private_guard_inactive_does_not_probe(monkeypatch, cdp_server):
    cdp_server.on("Runtime.evaluate", lambda params, sid: {"result": {"value": "ok"}})
    import tools.browser_tool as browser_tool

    monkeypatch.setattr(
        browser_tool, "_eval_ssrf_guard_active", AsyncMock(return_value=False)
    )
    fail_probe = AsyncMock(side_effect=AssertionError("must not probe page URL"))
    monkeypatch.setattr(browser_tool, "_current_page_private_url", fail_probe)
    result = json.loads(
        await browser_cdp_tool.browser_cdp(
            method="Runtime.evaluate",
            params={"expression": "document.title"},
            task_id="task-1",
        )
    )
    assert result["success"] is True
    assert result["result"]["result"]["value"] == "ok"
    fail_probe.assert_not_awaited()


@pytest.mark.asyncio
async def test_check_fn_does_not_probe_network(monkeypatch):
    import tools.browser_tool as browser_tool

    def boom(*_args, **_kwargs):
        raise AssertionError("check_fn must not perform network I/O")

    monkeypatch.setattr(browser_tool, "check_browser_requirements", _true)
    monkeypatch.setattr(browser_tool.httpx, "AsyncClient", boom)
    monkeypatch.setenv("BROWSER_CDP_URL", "http://127.0.0.1:9222")
    assert await browser_cdp_tool._browser_cdp_check() is True


@pytest.mark.asyncio
async def test_check_fn_false_when_browser_requirements_fail(monkeypatch):
    import tools.browser_tool as browser_tool

    monkeypatch.setattr(
        browser_tool, "check_browser_requirements", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(
        browser_tool,
        "_get_cdp_override_raw",
        AsyncMock(return_value="ws://localhost:9222/devtools/browser/x"),
    )
    assert await browser_cdp_tool._browser_cdp_check() is False
