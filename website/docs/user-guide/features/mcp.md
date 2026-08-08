---
title: MCP
description: Extend the tool surface through native-async Model Context Protocol clients.
sidebar_position: 4
---

# MCP

Model Context Protocol servers provide external tools without permanently
growing the Hermes core schema. The retained client supports stdio,
Streamable HTTP, and SSE transports.

## Lifecycle

The first awaited agent boundary reads `mcp_servers` from
`$HERMES_HOME/config.yaml`, connects to configured servers, discovers their
catalogs, and refreshes the agent's tool snapshot. `await agent.close()` releases
the associated subprocesses, HTTP sessions, keepalives, and registrations.

```python
async with AIAgent(
    ...,
    enabled_toolsets=["mcp-database"],
) as agent:
    await agent.chat("Query the configured database server.")
```

A server named `database` receives the canonical toolset `mcp-database` and a
raw-name alias. Tool names are normalized so collisions between servers cannot
silently overwrite one another.

## Runtime behavior

- MCP calls are awaited on the owning event loop.
- Reconnection and timeout handling remain asynchronous.
- Tool observations preserve model call order in history and trajectories.
- A server can opt into parallel-safe calls; otherwise its operations remain
  serialized.
- Include/exclude patterns can narrow a server catalog.
- Elicitation is routed through the host clarification callback.

Large MCP catalogs can participate in progressive tool search, avoiding a large
schema prefix on every model request.

## Security boundary

An stdio MCP definition executes a local command with the Python process's
authority. A remote MCP server receives requests and can return content that the
model will consume. Use pinned packages, restricted working directories,
least-privilege tokens, TLS, catalog filters, and application approval for
sensitive operations.

Never embed tokens directly in a checked-in YAML file. Reference environment
variables such as `${env:TEAM_MCP_TOKEN}`.

Configuration examples and lifecycle usage are in
[Use MCP with Hermes](../../guides/use-mcp-with-hermes.md).
