import asyncio

import aiofiles
import pytest

from hermes_constants import (
    reset_hermes_home_override,
    set_hermes_home_override,
)


@pytest.mark.asyncio
async def test_sequential_profiles_keep_stderr_sinks_separate(tmp_path):
    from tools import mcp_tool

    homes = [tmp_path / "alpha", tmp_path / "beta"]
    handles = []
    scopes = []
    for name, home in zip(("alpha", "beta"), homes):
        token = set_hermes_home_override(home)
        try:
            await mcp_tool._write_stderr_log_header(name)
            scope = await mcp_tool._activate_mcp_scope()
            scopes.append(scope)
            handles.append(mcp_tool._mcp_stderr_log_fhs[scope])
        finally:
            reset_hermes_home_override(token)

    assert handles[0] is not handles[1]
    assert scopes[0] != scopes[1]
    async with aiofiles.open(homes[0] / "logs" / "mcp-stderr.log") as handle:
        assert "alpha" in await handle.read()
    async with aiofiles.open(homes[1] / "logs" / "mcp-stderr.log") as handle:
        assert "beta" in await handle.read()

    for home in homes:
        token = set_hermes_home_override(home)
        try:
            await mcp_tool.shutdown_mcp_servers()
        finally:
            reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_concurrent_profiles_isolate_same_named_server_and_registry(
    monkeypatch,
    tmp_path,
):
    from tools import mcp_tool
    from tools.registry import ToolRegistry
    import tools.registry as registry_module

    scoped_registry = ToolRegistry()
    monkeypatch.setattr(registry_module, "registry", scoped_registry)
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)

    tool_name = "mcp__shared__ping"
    observed_configs = {}

    class FakeServer:
        def __init__(self, marker):
            self.marker = marker
            self.name = "shared"
            self.session = object()
            self._registered_tool_names = [tool_name]

        async def shutdown(self):
            scoped_registry.deregister(tool_name)
            self.session = None

    async def fake_discover(name, config):
        scope = await mcp_tool._activate_mcp_scope()
        marker = config["env"]["PROFILE_TOKEN"]
        observed_configs[scope[1]] = dict(config)

        async def handler(args, **kwargs):
            return marker

        scoped_registry.register(
            name=tool_name,
            toolset=f"mcp-{name}",
            schema={
                "name": tool_name,
                "description": marker,
                "parameters": {"type": "object", "properties": {}},
            },
            handler=handler,
        )
        scoped_registry.register_toolset_alias(name, f"mcp-{name}")
        with mcp_tool._lock:
            mcp_tool._server_connecting.discard(name)
            mcp_tool._server_connect_errors.pop(name, None)
            mcp_tool._servers[name] = FakeServer(marker)
        mcp_tool._track_mcp_tool_server(tool_name, name)
        return [tool_name]

    monkeypatch.setattr(mcp_tool, "_discover_and_register_server", fake_discover)

    barrier = asyncio.Barrier(2)
    alpha_closed = asyncio.Event()

    async def exercise(marker, *, first):
        home = tmp_path / marker
        token = set_hermes_home_override(home)
        try:
            config = {
                "shared": {
                    "command": f"server-{marker}",
                    "headers": {"Authorization": f"Bearer {marker}"},
                    "env": {"PROFILE_TOKEN": marker},
                }
            }
            assert await mcp_tool.register_mcp_servers(config) == [tool_name]
            await mcp_tool._write_stderr_log_header(marker)
            entry = scoped_registry.get_entry(tool_name)
            assert entry is not None
            assert entry.schema["description"] == marker
            assert await entry.handler({}) == marker
            assert mcp_tool._servers["shared"].marker == marker
            await asyncio.wait_for(barrier.wait(), timeout=5)

            if first:
                await mcp_tool.shutdown_mcp_servers()
                alpha_closed.set()
            else:
                await alpha_closed.wait()
                # The other profile's shutdown must not deregister or stop us.
                entry = scoped_registry.get_entry(tool_name)
                assert entry is not None
                assert await entry.handler({}) == marker
                assert mcp_tool._servers["shared"].marker == marker
                await mcp_tool.shutdown_mcp_servers()
            return home
        finally:
            reset_hermes_home_override(token)

    alpha, beta = await asyncio.gather(
        exercise("alpha-secret", first=True),
        exercise("beta-secret", first=False),
    )

    assert len(observed_configs) == 2
    assert {cfg["command"] for cfg in observed_configs.values()} == {
        "server-alpha-secret",
        "server-beta-secret",
    }
    async with aiofiles.open(alpha / "logs" / "mcp-stderr.log") as handle:
        alpha_log = await handle.read()
    async with aiofiles.open(beta / "logs" / "mcp-stderr.log") as handle:
        beta_log = await handle.read()
    assert "alpha-secret" in alpha_log and "beta-secret" not in alpha_log
    assert "beta-secret" in beta_log and "alpha-secret" not in beta_log
