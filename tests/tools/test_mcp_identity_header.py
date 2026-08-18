"""Upstream MCP identity-header tests, adapted to the native async boundary."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_identity_header_static_and_profile_modes():
    from tools.mcp_tool import _resolve_identity_header

    assert await _resolve_identity_header("srv", {}) is None
    assert await _resolve_identity_header(
        "srv",
        {
            "identity_header": {
                "name": "X-User-Id",
                "value_from": "static",
                "value": "alice",
            }
        },
    ) == ("X-User-Id", "alice")
    with patch(
        "hermes_cli.profiles.get_active_profile_name",
        new=AsyncMock(return_value="workbot"),
    ):
        assert await _resolve_identity_header(
            "srv",
            {
                "identity_header": {
                    "name": "X-Hermes-Profile",
                    "value_from": "profile",
                }
            },
        ) == ("X-Hermes-Profile", "workbot")


@pytest.mark.asyncio
async def test_identity_header_invalid_config_warns_and_ignores(caplog):
    from tools.mcp_tool import _resolve_identity_header

    cases = [
        {"identity_header": "not-a-mapping"},
        {"identity_header": {"value": "alice"}},
        {"identity_header": {"name": "X-User-Id"}},
        {
            "identity_header": {
                "name": "X-User-Id",
                "value_from": "per_call",
                "value": "alice",
            }
        },
    ]
    with caplog.at_level(logging.WARNING):
        for config in cases:
            assert await _resolve_identity_header("srv", config) is None
    assert sum("identity_header" in record.message for record in caplog.records) == len(cases)


@pytest.mark.asyncio
async def test_identity_header_merges_without_overriding_explicit_header():
    from tools.mcp_tool import _apply_identity_header

    headers = {"x-user-id": "explicit-wins"}
    result = await _apply_identity_header(
        "srv",
        {
            "identity_header": {"name": "X-User-Id", "value": "alice"},
        },
        headers,
    )
    assert result is headers
    assert headers == {"x-user-id": "explicit-wins"}


@pytest.mark.asyncio
async def test_http_identity_header_is_attached():
    from tools.mcp_tool import MCPServerTask

    captured: dict = {}

    class DummyClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class DummyTransport:
        async def __aenter__(self):
            return MagicMock(), MagicMock(), lambda: None

        async def __aexit__(self, *args):
            return False

    class DummySession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def initialize(self):
            return None

    from tools import mcp_tool

    real_httpx = mcp_tool.sdk_httpx()
    dummy_httpx = SimpleNamespace(
        URL=real_httpx.URL,
        Timeout=real_httpx.Timeout,
        AsyncClient=DummyClient,
    )

    server = MCPServerTask("remote")

    async def stop_discovery(self):
        self._shutdown_event.set()

    with (
        patch("tools.mcp_tool._MCP_HTTP_AVAILABLE", True),
        patch("tools.mcp_tool._MCP_NEW_HTTP", True),
        patch("tools.mcp_tool.streamable_http_client", return_value=DummyTransport()),
        patch("tools.mcp_tool.ClientSession", DummySession),
        patch("tools.mcp_tool.sdk_httpx", return_value=dummy_httpx),
        patch.object(MCPServerTask, "_discover_tools", stop_discovery),
    ):
        await server._run_http(
            {
                "url": "https://example.com/mcp",
                "identity_header": {"name": "X-User-Id", "value": "alice"},
            }
        )

    assert captured["headers"]["X-User-Id"] == "alice"


@pytest.mark.asyncio
async def test_stdio_identity_header_warns_and_does_not_break_import_error(caplog):
    from tools.mcp_tool import MCPServerTask

    server = MCPServerTask("local")
    with (
        patch("tools.mcp_tool._MCP_AVAILABLE", False),
        caplog.at_level(logging.WARNING),
        pytest.raises(ImportError),
    ):
        await server._run_stdio(
            {
                "command": "echo",
                "identity_header": {"name": "X-User-Id", "value": "alice"},
            }
        )
    assert any("identity_header" in record.message and "stdio" in record.message for record in caplog.records)
