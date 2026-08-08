---
title: Security
description: Operate the async agent harness with least-privilege tools, credentials, and integrations.
sidebar_position: 10
---

# Security

An agent can execute commands, modify files, browse sites, and call external
services. Treat every enabled tool as real authority granted to model output.

## Use least-privilege toolsets

Give each workload only the toolsets it needs:

```python
agent = AIAgent(
    ...,
    enabled_toolsets=["web", "file"],
    disabled_toolsets=["terminal", "browser"],
)
```

The local terminal backend runs processes with the permissions of the Python
process. This distribution does not ship Docker, SSH, Modal, Daytona, or
Singularity isolation backends. Use operating-system, container, or service
isolation outside the library when commands must not reach the host.

## Protect credentials

- Store API keys in the process environment or `$HERMES_HOME/.env`.
- Keep non-secret behavior in `config.yaml`.
- Use scoped, revocable tokens for MCP and cloud providers.
- Never place secrets in prompts, skills, trajectories, logs, or source control.
- Redact tool observations before returning them to untrusted clients.

## Review skills and MCP servers

Skills are executable instructions in the practical sense: they influence a
model that can use tools. Review third-party skill text and supporting scripts
before exposing them.

An stdio MCP entry starts the configured command locally. An HTTP MCP server can
return untrusted content and request tool actions or elicitation. Pin server
packages, restrict catalog tools with include/exclude patterns, use
least-privilege credentials, and require host authorization for sensitive
operations. See [MCP](./features/mcp.md) and [Skills](./features/skills.md).

## Network boundaries

Browser and web tools guard private, loopback, link-local, and cloud-metadata
targets. Keep the default:

```yaml
security:
  allow_private_urls: false
```

Setting it to `true` expands SSRF reach for every caller using that Hermes home.
If access to a private application is intentional, prefer a separately isolated
agent with narrow credentials and network policy. Browser-specific behavior is
documented in [Browser automation](./features/browser.md).

## Service responsibilities

The library does not provide an authenticated HTTP API. A FastAPI or other host
must implement authentication, tenant isolation, rate limits, request size
limits, timeouts, audit policy, and mapping between users and ordered agent
instances.

Do not share one memory directory, session ID, or mutable agent instance across
untrusted tenants. Use a separate `HERMES_HOME`, database, and lifecycle where
the trust boundary requires it.

## Shutdown and cancellation

Use the async context manager or `await agent.close()` so subprocesses, MCP
sessions, provider clients, and child tasks are released. On external
cancellation, the runtime performs partial persistence and then re-raises
`CancelledError`; callers must still enforce their own timeout and retry policy.
