---
title: Integrations
description: Extend the native-async harness through providers, plugins, MCP, and skills.
sidebar_position: 1
---

# Integrations

The agent core is intentionally narrow. Capabilities are attached at its async
edges so the conversation loop, prompt-cache prefix, message alternation, and
trajectory order remain stable.

## Integration surfaces

| Surface | Best for | Lifecycle |
| --- | --- | --- |
| Model provider profiles | Selecting an inference endpoint and wire protocol | Initialized and closed by `AIAgent` |
| Provider plugins | Web search, browser, image/video, and external memory backends | Discovered lazily at an awaited boundary |
| MCP | External structured tools owned by another process or service | Connected, called, and closed asynchronously |
| Skills | Reusable instructions and supporting files | Discovered and read from disk on demand |
| Built-in tools | Fundamental file, terminal, memory, browser, and planning operations | Scheduled by the core tool executor |

Start with [Model providers](./providers.md). For external tools, see
[MCP](../user-guide/features/mcp.md); for instructional extensions, see
[Skills](../user-guide/features/skills.md).

## Plugin discovery

Bundled provider definitions live under the installed `plugins/` package.
Application-owned plugins can live under `$HERMES_HOME/plugins/`. Discovery is
lazy, and a user provider profile with the same name can override its bundled
counterpart.

A plugin must implement the native-async contract for its category. A
synchronous handler is rejected rather than hidden behind a thread bridge.
Provider-specific dependencies should remain optional and fail with a clear
installation hint when absent.

## What is not an integration surface here

This distribution does not ship the upstream CLI/TUI, desktop or dashboard,
messaging platforms, cron scheduler, FastAPI server, or editor adapter. Build
those applications around the library API rather than depending on residual
helper package names.

The framework-neutral embedding contract is documented in the
[Python library guide](../guides/python-library.md).
