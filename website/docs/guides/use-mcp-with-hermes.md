---
title: Use MCP with Hermes
description: Connect stdio or HTTP MCP servers to the native-async agent lifecycle.
sidebar_position: 3
---

# Use MCP with Hermes

The Model Context Protocol (MCP) connects external tool servers to the agent
without adding their schemas to the core. MCP discovery, calls, reconnection,
and shutdown are awaited on the agent event loop.

## Configure a stdio server

Add a server under `mcp_servers` in `$HERMES_HOME/config.yaml`. `HERMES_HOME`
defaults to `~/.hermes`.

```yaml
mcp_servers:
  filesystem:
    command: npx
    args:
      - -y
      - "@modelcontextprotocol/server-filesystem"
      - /workspace
    connect_timeout: 60
    timeout: 300
```

The configured command is started as a local subprocess. Install its runtime
and package manager separately; this Python library does not install Node.js or
third-party MCP servers.

## Configure an HTTP server

```yaml
mcp_servers:
  knowledge:
    url: "https://mcp.example.com/mcp"
    headers:
      Authorization: "Bearer ${env:KNOWLEDGE_MCP_TOKEN}"
    connect_timeout: 30
    timeout: 180
```

Keep the token in the process environment or `$HERMES_HOME/.env`:

```dotenv
KNOWLEDGE_MCP_TOKEN=...
```

Streamable HTTP is selected for `url` entries by default. Set `transport: sse`
only when the server implements the older SSE transport.

## Expose the server tools to an agent

At the first awaited lifecycle boundary, Hermes discovers configured servers and
registers each catalog under `mcp-<server-name>` (with the raw server name as an
alias). Use that toolset in the agent allowlist:

```python
from run_agent import AIAgent

async with AIAgent(
    ...,
    enabled_toolsets=["file", "mcp-filesystem"],
) as agent:
    result = await agent.run_conversation(
        "List the files exposed by the filesystem server."
    )
```

Large optional catalogs may be exposed through progressive tool search, while
core tools remain directly available. Tool observations appear in the normal
message sequence and in saved trajectories.

## Limit a server catalog

Use include or exclude patterns when a server exposes more tools than a task
needs:

```yaml
mcp_servers:
  github:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "${env:GITHUB_TOKEN}"
    tools:
      include: ["search_*", "get_*"]
      exclude: ["delete_*"]
```

Filtering reduces the visible capability surface; it is not a substitute for
least-privilege credentials or server-side authorization.

## Elicitation and shutdown

MCP servers can request user input through elicitation. Supply an async-capable
host callback through `clarify_callback` when the selected servers use it. The
agent converts approval-style elicitation into the same clarification boundary
used by the built-in `clarify` tool.

Always use `async with AIAgent(...)` or call `await agent.close()`. Shutdown
releases stdio subprocesses, HTTP sessions, keepalive tasks, and registered MCP
lifecycle state. See [MCP concepts](../user-guide/features/mcp.md) and
[Security](../user-guide/security.md) before connecting an untrusted server.
