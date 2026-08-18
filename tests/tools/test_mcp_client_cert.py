"""Tests for mTLS client certificate config on MCP HTTP/SSE transports.

Covers:

1. ``_resolve_client_cert`` helper — string, tuple, encrypted-key, validation
   errors, missing-file errors.

2. HTTP (new SDK ``streamable_http_client``) path forwards ``cert=`` into the
   user-owned ``httpx.AsyncClient``.

3. SSE path forwards ``cert`` and ``ssl_verify`` via an ``httpx_client_factory``
   without breaking the OAuth/headers/timeout passthrough.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _patch_sdk_async_client(dummy):
    """Patch ``AsyncClient`` on whichever httpx module the MCP SDK uses.

    mcp 2.0 moved the SDK's HTTP stack to ``httpx2``, so patching
    ``httpx.AsyncClient`` no longer intercepts the client Hermes builds for
    the SDK. Resolve the module the same way production does, via
    ``tools.mcp_tool.sdk_httpx``, so these tests follow the SDK rather than
    hardcoding a distribution name.
    """
    from tools.mcp_tool import sdk_httpx

    return patch.object(sdk_httpx(), "AsyncClient", dummy)


# ---------------------------------------------------------------------------
# _resolve_client_cert helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestResolveClientCert:
    async def test_returns_none_when_unset(self):
        from tools.mcp_tool import _resolve_client_cert

        assert await _resolve_client_cert("srv", {}) is None
        assert await _resolve_client_cert("srv", {"url": "https://x"}) is None

    async def test_string_form_single_pem(self, tmp_path):
        from tools.mcp_tool import _resolve_client_cert

        pem = tmp_path / "combined.pem"
        pem.write_text("dummy")

        result = await _resolve_client_cert("srv", {"client_cert": str(pem)})
        assert result == str(pem)


    async def test_list_form_two_elements(self, tmp_path):
        from tools.mcp_tool import _resolve_client_cert

        cert = tmp_path / "client.crt"
        key = tmp_path / "client.key"
        cert.write_text("cert")
        key.write_text("key")

        result = await _resolve_client_cert("srv", {
            "client_cert": [str(cert), str(key)],
        })
        assert result == (str(cert), str(key))


    async def test_password_must_be_string(self, tmp_path):
        from tools.mcp_tool import _resolve_client_cert

        cert = tmp_path / "client.crt"
        key = tmp_path / "client.key"
        cert.write_text("cert")
        key.write_text("key")

        with pytest.raises(ValueError, match=r"key passphrase.*must be a string"):
            await _resolve_client_cert("srv", {
                "client_cert": [str(cert), str(key), 42],
            })


# ---------------------------------------------------------------------------
# HTTP transport — cert forwarded into httpx.AsyncClient
# ---------------------------------------------------------------------------


class TestHTTPClientCert:
    def test_cert_forwarded_to_async_client(self, tmp_path):
        """The new-SDK HTTP path forwards ``client_cert`` to TLS setup."""
        from tools.mcp_tool import MCPServerTask

        cert = tmp_path / "client.pem"
        cert.write_text("dummy")

        server = MCPServerTask("remote")
        captured: dict = {}

        async def materialize_verify(verify, *, cert=None, trust_env=True):
            captured["cert"] = cert
            return False

        class DummyAsyncClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        class DummyTransportCtx:
            async def __aenter__(self):
                return MagicMock(), MagicMock(), (lambda: None)

            async def __aexit__(self, *a):
                return False

        class DummySession:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def initialize(self):
                return None

        async def _discover_tools(self):
            self._shutdown_event.set()

        async def _drive():
            with patch("tools.mcp_tool._MCP_HTTP_AVAILABLE", True), \
                 patch("tools.mcp_tool._MCP_NEW_HTTP", True), \
                 patch(
                     "agent.ssl_verify._materialize_httpx_verify",
                     materialize_verify,
                 ), \
                 _patch_sdk_async_client(DummyAsyncClient), \
                 patch("tools.mcp_tool.streamable_http_client",
                       return_value=DummyTransportCtx()), \
                 patch("tools.mcp_tool.ClientSession", DummySession), \
                 patch.object(MCPServerTask, "_discover_tools", _discover_tools):
                await server._run_http({
                    "url": "https://example.com/mcp",
                    "client_cert": str(cert),
                })

        asyncio.run(_drive())
        assert captured["cert"] == str(cert)


    def test_missing_cert_file_surfaces_clear_error(self, tmp_path):
        """A missing cert file fails fast with a server-scoped error message."""
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("remote")

        async def _drive():
            with patch("tools.mcp_tool._MCP_HTTP_AVAILABLE", True), \
                 patch("tools.mcp_tool._MCP_NEW_HTTP", True):
                await server._run_http({
                    "url": "https://example.com/mcp",
                    "client_cert": str(tmp_path / "nope.pem"),
                })

        with pytest.raises(FileNotFoundError, match=r"remote.*client_cert.*not found"):
            asyncio.run(_drive())


# ---------------------------------------------------------------------------
# SSE transport — cert + verify routed via httpx_client_factory
# ---------------------------------------------------------------------------


@pytest.fixture
def patch_sse_client():
    """Replace ``sse_client`` with a MagicMock that records its kwargs.

    Returns the captured kwargs dict so tests can assert how ``_run_http``
    called it.
    """
    captured_kwargs: dict = {}

    class _FakeStream:
        def __init__(self):
            self._read = AsyncMock()
            self._write = AsyncMock()

        async def __aenter__(self):
            return (self._read, self._write)

        async def __aexit__(self, *a):
            return False

    def fake_sse_client(**kwargs):
        captured_kwargs.clear()
        captured_kwargs.update(kwargs)
        return _FakeStream()

    class _FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            mock_session = MagicMock()
            mock_session.initialize = AsyncMock()
            return mock_session

        async def __aexit__(self, *a):
            return False

    with patch("tools.mcp_tool.sse_client", new=fake_sse_client), \
         patch("tools.mcp_tool.ClientSession", new=_FakeSession):
        yield captured_kwargs


class TestSSEClientCert:
    def test_no_factory_when_defaults(self, patch_sse_client):
        """With no cert and ssl_verify=True (default), the SDK's own factory is
        used — we don't inject one."""
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("sse-test")
        server._auth_type = ""
        server._sampling = None

        async def drive():
            with patch.object(MCPServerTask, "_wait_for_lifecycle_event",
                              new=AsyncMock(return_value="shutdown")), \
                 patch.object(MCPServerTask, "_discover_tools", new=AsyncMock()):
                try:
                    await asyncio.wait_for(
                        server._run_http({
                            "url": "https://example.com/mcp/sse",
                            "transport": "sse",
                        }),
                        timeout=2.0,
                    )
                except (TimeoutError, StopAsyncIteration, Exception):
                    pass

        asyncio.run(drive())
        assert "httpx_client_factory" not in patch_sse_client

    def test_factory_injected_when_cert_set(self, patch_sse_client, tmp_path):
        """With client_cert set, an httpx_client_factory is injected that
        applies the cert (and follow_redirects=True to match the SDK)."""
        from tools.mcp_tool import MCPServerTask

        cert = tmp_path / "client.pem"
        cert.write_text("dummy")

        server = MCPServerTask("sse-test")
        server._auth_type = ""
        server._sampling = None

        async def drive():
            with patch.object(MCPServerTask, "_wait_for_lifecycle_event",
                              new=AsyncMock(return_value="shutdown")), \
                 patch.object(MCPServerTask, "_discover_tools", new=AsyncMock()):
                try:
                    await asyncio.wait_for(
                        server._run_http({
                            "url": "https://example.com/mcp/sse",
                            "transport": "sse",
                            "client_cert": str(cert),
                        }),
                        timeout=2.0,
                    )
                except (TimeoutError, StopAsyncIteration, Exception):
                    pass

        asyncio.run(drive())

        factory = patch_sse_client.get("httpx_client_factory")
        assert factory is not None, "expected httpx_client_factory to be injected"

        # Enter the factory result the way the SDK does; capture the awaited
        # httpx client-builder kwargs.
        captured_client_kwargs: dict = {}

        class DummyAsyncClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        import httpx

        async def enter_factory():
            async def create_client(**kwargs):
                captured_client_kwargs.update(kwargs)
                return DummyAsyncClient()

            with patch(
                "agent.ssl_verify._create_httpx_client",
                side_effect=create_client,
            ):
                async with factory(
                    headers={"x": "y"},
                    timeout=httpx.Timeout(30.0),
                    auth=None,
                ):
                    pass

        asyncio.run(enter_factory())

        assert captured_client_kwargs["cert"] == str(cert)
        assert captured_client_kwargs["verify"] is True
        assert captured_client_kwargs["follow_redirects"] is True
        assert captured_client_kwargs["headers"] == {"x": "y"}

    def test_factory_forwards_custom_ca_bundle(self, patch_sse_client, tmp_path):
        """ssl_verify as a path is forwarded to the factory's httpx client."""
        from tools.mcp_tool import MCPServerTask

        ca_bundle = tmp_path / "ca.pem"
        ca_bundle.write_text("dummy")

        server = MCPServerTask("sse-test")
        server._auth_type = ""
        server._sampling = None

        async def drive():
            with patch.object(MCPServerTask, "_wait_for_lifecycle_event",
                              new=AsyncMock(return_value="shutdown")), \
                 patch.object(MCPServerTask, "_discover_tools", new=AsyncMock()):
                try:
                    await asyncio.wait_for(
                        server._run_http({
                            "url": "https://example.com/mcp/sse",
                            "transport": "sse",
                            "ssl_verify": str(ca_bundle),
                        }),
                        timeout=2.0,
                    )
                except (TimeoutError, StopAsyncIteration, Exception):
                    pass

        asyncio.run(drive())

        factory = patch_sse_client.get("httpx_client_factory")
        assert factory is not None

        captured_client_kwargs: dict = {}

        class DummyAsyncClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        async def enter_factory():
            async def create_client(**kwargs):
                captured_client_kwargs.update(kwargs)
                return DummyAsyncClient()

            with patch(
                "agent.ssl_verify._create_httpx_client",
                side_effect=create_client,
            ):
                async with factory(headers=None, timeout=None, auth=None):
                    pass

        asyncio.run(enter_factory())

        assert captured_client_kwargs["verify"] == str(ca_bundle)
        assert "cert" not in captured_client_kwargs

    def test_factory_enters_http_client_without_blocking(self, patch_sse_client):
        from blockbuster import BlockBuster
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("sse-test")
        server._auth_type = ""
        server._sampling = None

        async def drive():
            with patch.object(
                MCPServerTask,
                "_wait_for_lifecycle_event",
                new=AsyncMock(return_value="shutdown"),
            ), patch.object(
                MCPServerTask,
                "_discover_tools",
                new=AsyncMock(),
            ):
                await server._run_http(
                    {
                        "url": "https://example.com/mcp/sse",
                        "transport": "sse",
                        "ssl_verify": False,
                    }
                )

        asyncio.run(drive())
        factory = patch_sse_client.get("httpx_client_factory")
        assert factory is not None

        async def enter_factory():
            import httpx

            blocker = BlockBuster()
            blocker.activate()
            try:
                async with factory(
                    headers={"x-test": "value"},
                    timeout=httpx.Timeout(30.0),
                    auth=None,
                ) as client:
                    assert client.headers["x-test"] == "value"
                    assert client.follow_redirects is True
            finally:
                blocker.deactivate()

        asyncio.run(enter_factory())
