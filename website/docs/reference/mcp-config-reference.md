---
sidebar_position: 2
title: "MCP Configuration Reference"
description: "Configure native-async MCP stdio, Streamable HTTP, and SSE clients"
---

# MCP Configuration Reference

Define MCP servers under `mcp_servers` in `$HERMES_HOME/config.yaml`. Discovery,
tool calls, reconnection, and shutdown run as tasks on the agent's event loop.

## Stdio server

```yaml
mcp_servers:
  filesystem:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
    env: {}
    timeout: 300
    connect_timeout: 60
```

`command` selects stdio transport. `args` is optional. The subprocess receives
a filtered environment plus the explicit `env` map; it does not inherit every
host secret automatically.

## Streamable HTTP server

```yaml
mcp_servers:
  remote:
    url: https://example.test/mcp
    headers:
      Authorization: "Bearer ${REMOTE_MCP_TOKEN}"
    timeout: 180
    connect_timeout: 30
```

An entry with `url` uses Streamable HTTP by default. Remote URL validation is
applied before connection.

## SSE server

```yaml
mcp_servers:
  legacy_sse:
    url: https://example.test/sse
    transport: sse
    timeout: 180
```

Use `transport: sse` only for a server implementing the older MCP SSE
transport.

## Common fields

| Field | Meaning |
| --- | --- |
| `enabled` | Enable the server; defaults to `true` |
| `command` | Stdio executable |
| `args` | Stdio argument list |
| `env` | Explicit stdio environment additions; values may reference secrets |
| `url` | Remote MCP endpoint |
| `transport` | `http` by default for URLs, or `sse` |
| `headers` | Remote request headers |
| `timeout` | Per-tool-call timeout in seconds; default `300` |
| `connect_timeout` | Initial connection timeout; default `60` |
| `protocol` | `auto` (default), `stateless`, or `legacy` protocol negotiation mode |
| `lazy` | Register from a valid schema cache and connect on first call when possible |
| `supports_parallel_tool_calls` | Opt this server's tools into parallel-safe scheduling; default `false` |
| `keepalive_interval` | Liveness-ping interval; default `180`, minimum `5` seconds |
| `idle_timeout_seconds` | Recycle an idle stdio server; `0` disables |
| `max_lifetime_seconds` | Recycle an aged stdio server; `0` disables |
| `skip_preflight` | Skip the remote content-type probe for a known valid endpoint |

Lifecycle limits may also be nested below `lifecycle`.

## Tool filtering

```yaml
mcp_servers:
  github:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
    tools:
      include: ["get_*", "search_repositories"]
      exclude: []
      resources: false
      prompts: true
```

- `tools.include` is a whitelist and takes precedence.
- `tools.exclude` is a blacklist used only when no include list is present.
- Exact names and case-sensitive `fnmatch` patterns are accepted.
- `resources` and `prompts` control generated utility tools, subject to the
  capabilities actually advertised by the server.

## Registered names

An MCP tool is exposed as:

```text
mcp__<sanitized_server>__<sanitized_tool>
```

For example, server `github` tool `search-repositories` becomes
`mcp__github__search_repositories`. The server also contributes the dynamic
toolset `mcp-github`; its raw server name is accepted as an alias by toolset
resolution.

Name collisions caused by sanitization fail closed rather than selecting an
arbitrary handler.

## Parallel calls

MCP tools are sequential by default. Set
`supports_parallel_tool_calls: true` only when the server and every relevant
operation are safe to overlap. Results still enter the model transcript in the
original tool-call order.

## Sampling

Servers that request MCP sampling can be configured under `sampling`:

```yaml
sampling:
  enabled: true
  model: example/model
  max_tokens_cap: 4096
  timeout: 30
  max_rpm: 10
  allowed_models: []
  max_tool_rounds: 5
```

Sampling uses host-owned model configuration and remains bounded by these
limits. Disable it for servers that should never invoke an LLM.

## Lifecycle

The first awaited agent boundary discovers configured servers. MCP ownership
is reference-counted across active agents, and `await agent.close()` releases
the agent's ownership. Always close agents so stdio subprocesses and remote
sessions terminate cleanly.

See [MCP](/user-guide/features/mcp) and
[Use MCP with Hermes](/guides/use-mcp-with-hermes).
