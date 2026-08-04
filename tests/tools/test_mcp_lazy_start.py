"""Behavior-contract tests for lazy MCP server startup (#56832).

A server configured with ``lazy: true`` whose config fingerprint matches an
on-disk schema-cache entry registers its tools WITHOUT spawning/connecting;
the first real call (raw tool OR resource/prompt utility) routes through the
existing connect path.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import tools.mcp_tool as mcp


@pytest.fixture(autouse=True)
def _reset_mcp_state():
    old_servers = dict(mcp._servers)
    old_lazy = dict(mcp._lazy_server_configs)
    old_fps = dict(mcp._lazy_server_fingerprints)
    old_names = dict(mcp._lazy_server_tool_names)
    old_locks = dict(mcp._lazy_server_connect_locks)
    old_connecting = set(mcp._server_connecting)
    yield
    mcp._servers.clear()
    mcp._servers.update(old_servers)
    mcp._lazy_server_configs.clear()
    mcp._lazy_server_configs.update(old_lazy)
    mcp._lazy_server_fingerprints.clear()
    mcp._lazy_server_fingerprints.update(old_fps)
    mcp._lazy_server_tool_names.clear()
    mcp._lazy_server_tool_names.update(old_names)
    mcp._lazy_server_connect_locks.clear()
    mcp._lazy_server_connect_locks.update(old_locks)
    mcp._server_connecting.clear()
    mcp._server_connecting.update(old_connecting)


def _fake_cache_entry():
    return {
        "fingerprint": "abc",
        "tools": [
            {
                "name": "browser_navigate",
                "description": "Navigate",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ],
        "utility_tools": [],
    }


def _lazy_config():
    return {
        "playwright": {
            "command": "npx",
            "args": ["-y", "@playwright/mcp"],
            "lazy": True,
        }
    }


@pytest.mark.asyncio
class TestLazyMcpRegistration:
    async def test_registers_from_cache_without_connect(self):
        config = _lazy_config()
        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_schema_cache.config_fingerprint", return_value="abc"), \
             patch("tools.mcp_schema_cache.get_cached_entry", new_callable=AsyncMock,
                   return_value=_fake_cache_entry()), \
             patch(
                 "tools.mcp_tool._register_from_cache",
                 return_value=["mcp_playwright_browser_navigate"],
             ) as mock_register, \
             patch("tools.mcp_tool._discover_and_register_server", new_callable=AsyncMock) as mock_discover:

            await mcp.register_mcp_servers(config)

        mock_register.assert_called_once()
        mock_discover.assert_not_called()

    async def test_cache_miss_falls_back_to_eager_connect(self):
        config = _lazy_config()
        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_schema_cache.config_fingerprint", return_value="abc"), \
             patch("tools.mcp_schema_cache.get_cached_entry", new_callable=AsyncMock,
                   return_value=None), \
             patch("tools.mcp_tool._discover_and_register_server", new_callable=AsyncMock,
                   return_value=[]) as mock_discover:

            await mcp.register_mcp_servers(config)

        mock_discover.assert_awaited_once()

    async def test_non_lazy_server_never_touches_cache(self):
        config = {"playwright": {"command": "npx", "args": []}}
        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_schema_cache.get_cached_entry", new_callable=AsyncMock) as mock_get, \
             patch("tools.mcp_tool._discover_and_register_server", new_callable=AsyncMock,
                   return_value=[]) as mock_discover:

            await mcp.register_mcp_servers(config)

        mock_get.assert_not_called()
        mock_discover.assert_awaited_once()

    async def test_lazy_server_not_reregistered_on_second_pass(self):
        config = _lazy_config()
        mcp._lazy_server_configs["playwright"] = dict(config["playwright"])
        mcp._lazy_server_tool_names["playwright"] = ["mcp_playwright_browser_navigate"]
        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._register_from_cache") as mock_register:

            names = await mcp.register_mcp_servers(config)

        mock_register.assert_not_called()
        assert "mcp_playwright_browser_navigate" in names


@pytest.mark.asyncio
class TestLazyFirstUseConnect:
    def _connected_server(self):
        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(
            return_value=SimpleNamespace(isError=False, content=[], structuredContent=None)
        )
        connected = SimpleNamespace(
            session=mock_session,
            _rpc_lock=MagicMock(),
            _pending_call_context=None,
        )
        connected._rpc_lock.__aenter__ = AsyncMock(return_value=None)
        connected._rpc_lock.__aexit__ = AsyncMock(return_value=None)
        return connected

    async def test_tool_handler_lazy_connects_on_first_call(self):
        config = {"command": "npx", "args": [], "lazy": True, "timeout": 5}
        mcp._lazy_server_configs["playwright"] = dict(config)
        mcp._lazy_server_fingerprints["playwright"] = "abc"

        connected = self._connected_server()

        async def _connect(name):
            mcp._servers["playwright"] = connected
            return True

        with patch.object(mcp, "_ensure_lazy_server_connected", new_callable=AsyncMock,
                          side_effect=_connect) as mock_connect:
            handler = mcp._make_tool_handler("playwright", "browser_navigate", 5)
            out = await handler({}, task_id="t1")

        mock_connect.assert_awaited_once_with("playwright")
        payload = json.loads(out)
        assert "error" not in payload
        assert payload.get("result") == ""

    async def test_list_resources_handler_lazy_connects_on_first_call(self):
        # Regression for the resource/prompt gap: utility handlers must also
        # route through the first-use connect path, or the first
        # list_resources/get_prompt on a lazy server fails.
        config = {"command": "npx", "args": [], "lazy": True, "timeout": 5}
        mcp._lazy_server_configs["playwright"] = dict(config)

        connected = self._connected_server()
        connected.session.list_resources = AsyncMock()

        async def _connect(name):
            mcp._servers["playwright"] = connected
            return True

        async def _fake_paginate(list_method, items_attr, server_name):
            return [SimpleNamespace(uri="file:///a", name="a", description="", mimeType="")]

        with patch.object(mcp, "_ensure_lazy_server_connected", new_callable=AsyncMock,
                          side_effect=_connect) as mock_connect, \
             patch.object(mcp, "_paginate_full_list", side_effect=_fake_paginate):
            handler = mcp._make_list_resources_handler("playwright", 5)
            out = await handler({})

        mock_connect.assert_awaited_once_with("playwright")
        payload = json.loads(out)
        assert "error" not in payload
        assert payload["resources"][0]["uri"] == "file:///a"

    async def test_get_prompt_handler_lazy_connects_on_first_call(self):
        config = {"command": "npx", "args": [], "lazy": True, "timeout": 5}
        mcp._lazy_server_configs["playwright"] = dict(config)

        connected = self._connected_server()
        connected.session.get_prompt = AsyncMock(
            return_value=SimpleNamespace(messages=[])
        )

        async def _connect(name):
            mcp._servers["playwright"] = connected
            return True

        with patch.object(mcp, "_ensure_lazy_server_connected", new_callable=AsyncMock,
                          side_effect=_connect) as mock_connect:
            handler = mcp._make_get_prompt_handler("playwright", 5)
            out = await handler({"name": "greeting"})

        mock_connect.assert_awaited_once_with("playwright")
        payload = json.loads(out)
        assert "error" not in payload

    async def test_check_fn_passes_for_lazy_registered_server(self):
        mcp._lazy_server_configs["playwright"] = {"lazy": True}
        mcp._lazy_server_fingerprints["playwright"] = "abc"
        assert mcp._make_check_fn("playwright")() is True

    async def test_check_fn_fails_for_unknown_server(self):
        assert mcp._make_check_fn("nope")() is False

    async def test_lazy_connect_respects_connect_cooldown(self):
        mcp._lazy_server_configs["playwright"] = {"command": "npx", "lazy": True}
        with patch.object(mcp, "_connect_cooldown_active", return_value=True), \
             patch.object(mcp, "_discover_and_register_server", new_callable=AsyncMock) as discover:
            assert await mcp._ensure_lazy_server_connected("playwright") is False
        discover.assert_not_awaited()

    async def test_lazy_connect_success_clears_lazy_state(self):
        config = {"command": "npx", "lazy": True}
        mcp._lazy_server_configs["playwright"] = dict(config)
        mcp._lazy_server_fingerprints["playwright"] = "abc"
        mcp._lazy_server_tool_names["playwright"] = ["mcp_playwright_browser_navigate"]

        connected = SimpleNamespace(
            session=MagicMock(),
            _registered_tool_names=["mcp_playwright_browser_navigate"],
        )

        async def _discover(name, config):
            mcp._servers["playwright"] = connected
            return ["mcp_playwright_browser_navigate"]

        with patch.object(mcp, "_discover_and_register_server", side_effect=_discover):
            assert await mcp._ensure_lazy_server_connected("playwright") is True

        assert "playwright" not in mcp._lazy_server_configs
        assert "playwright" not in mcp._lazy_server_fingerprints
        assert "playwright" not in mcp._lazy_server_tool_names

    async def test_concurrent_first_use_shares_one_connection(self):
        import asyncio

        mcp._lazy_server_configs["playwright"] = {
            "command": "npx",
            "lazy": True,
        }
        connected = SimpleNamespace(
            session=MagicMock(),
            _registered_tool_names=[],
        )
        calls = 0

        async def _discover(name, config):
            nonlocal calls
            calls += 1
            await asyncio.sleep(0)
            mcp._servers[name] = connected
            return []

        with patch.object(mcp, "_discover_and_register_server", side_effect=_discover):
            results = await asyncio.gather(
                mcp._ensure_lazy_server_connected("playwright"),
                mcp._ensure_lazy_server_connected("playwright"),
            )

        assert results == [True, True]
        assert calls == 1

    async def test_lazy_connect_deregisters_phantom_cached_tools(self):
        # Stale-cache reconciliation: the cached manifest advertised tool X,
        # but the live server only registers tool Y → X must be deregistered
        # after the first-use connect so the model stops seeing a phantom.
        from tools.registry import registry

        mcp._lazy_server_configs["playwright"] = {"command": "npx", "lazy": True}
        mcp._lazy_server_fingerprints["playwright"] = "stale-fp"
        mcp._lazy_server_tool_names["playwright"] = [
            "mcp_playwright_tool_x",
            "mcp_playwright_tool_y",
        ]

        connected = SimpleNamespace(
            session=MagicMock(),
            _registered_tool_names=["mcp_playwright_tool_y"],
        )

        async def _discover(name, config):
            mcp._servers["playwright"] = connected
            return ["mcp_playwright_tool_y"]

        with patch.object(mcp, "_discover_and_register_server", side_effect=_discover), \
             patch.object(registry, "deregister") as mock_dereg:
            assert await mcp._ensure_lazy_server_connected("playwright") is True

        mock_dereg.assert_called_once_with("mcp_playwright_tool_x")

    async def test_lazy_connect_failure_records_cooldown(self):
        mcp._lazy_server_configs["playwright"] = {"command": "npx", "lazy": True}

        async def _fail(name, config):
            raise RuntimeError("spawn failed")

        with patch.object(mcp, "_discover_and_register_server", side_effect=_fail), \
             patch.object(mcp, "_record_connect_failure") as mock_record:
            assert await mcp._ensure_lazy_server_connected("playwright") is False

        mock_record.assert_called_once_with("playwright")
        # Config retained so a later call can retry after cooldown.
        assert "playwright" in mcp._lazy_server_configs


class TestCacheLoadDescriptionScan:
    def test_scan_runs_on_cache_load_path(self):
        # Defense-in-depth: the cache file is user-writable JSON, so the
        # cache-load registration path must run the same injection scan as
        # eager discovery.
        entry = _fake_cache_entry()
        config = {"command": "npx", "args": [], "lazy": True}
        with patch.object(mcp, "_scan_mcp_description", return_value=[]) as mock_scan, \
             patch.object(mcp, "_convert_mcp_schema", side_effect=RuntimeError("stop")), \
             pytest.raises(RuntimeError):
            mcp._register_from_cache("playwright", config, entry)

        mock_scan.assert_called_once_with("playwright", "browser_navigate", "Navigate")


class TestResolveServerLazy:
    def test_default_off(self):
        assert mcp._resolve_server_lazy("s", {"command": "npx"}) is False

    def test_explicit_true(self):
        assert mcp._resolve_server_lazy("s", {"command": "npx", "lazy": True}) is True

    def test_explicit_false(self):
        assert mcp._resolve_server_lazy("s", {"command": "npx", "lazy": False}) is False
