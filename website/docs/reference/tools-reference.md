---
sidebar_position: 3
title: "Tools Reference"
description: "Retained built-in tools, required arguments, and availability behavior"
---

# Tools Reference

Tool schemas are registered from their implementation modules and filtered by
toolset, configuration, and each tool's availability check. The schema returned
by `await model_tools.get_tool_definitions(...)` is the canonical runtime
contract.

## Core retained tools

| Tool | Required input | Purpose |
| --- | --- | --- |
| `terminal` | `command` | Run a command with the retained local terminal backend |
| `process` | `action` | Inspect, poll, write to, or stop background processes |
| `read_file` | `path` | Read text or supported file content |
| `write_file` | `path`, `content` | Write a file subject to safety checks |
| `patch` | `mode` | Apply structured file patches |
| `search_files` | `pattern` | Search file names or contents |
| `web_search` | `query` | Search through the configured web provider |
| `web_extract` | `urls` | Extract content from one or more URLs |
| `vision_analyze` | `image_url`, `question` | Analyze an image |
| `image_generate` | `prompt` | Generate an image through the selected provider plugin |
| `skills_list` | none | Discover available skills |
| `skill_view` | `name` | Load one skill document |
| `skill_manage` | `action`, `name` | Create, edit, patch, or remove a skill |
| `memory` | `target` | Read or modify persistent memory/user-profile content |
| `session_search` | none | Search persisted sessions and messages |
| `todo` | none | Maintain turn planning state |
| `clarify` | `question` | Ask the host user for clarification through a callback |
| `delegate_task` | none | Launch one or more child-agent tasks; each task requires a goal |

Optional retained media tools are `video_analyze` (`video_url`, `question`) and
`video_generate` (`prompt`). They are not part of the default core list.

## Browser tools

When a browser backend is configured, the `browser` toolset can expose:

- `browser_navigate`
- `browser_snapshot`
- `browser_click`
- `browser_type`
- `browser_scroll`
- `browser_back`
- `browser_press`
- `browser_get_images`
- `browser_vision`
- `browser_console`
- `browser_cdp`
- `browser_dialog`

Browser schemas vary by operation and provider capability. Read the live schema
instead of hard-coding optional parameters in a host.

## MCP tools

Each discovered MCP tool is registered dynamically as:

```text
mcp__<server>__<tool>
```

Its parameter schema comes from the MCP server and is normalized for provider
compatibility. MCP resource and prompt utility tools are added only when the
server advertises those capabilities and configuration allows them.

See [MCP Configuration](./mcp-config-reference.md).

## Deferred tool search

When the configured tool schema would be large, the model-facing list may use
the bridge tools `tool_search`, `tool_describe`, and `tool_call`. They do not
create new capabilities: they defer schemas for tools already available to the
agent. Calls are validated against the underlying schema before dispatch.

## Availability checks

Registration does not guarantee that a tool is callable. `check_fn` gates
provider credentials, browser readiness, callbacks, and other prerequisites.
Unavailable tools are omitted from the model-facing list.

Hosts should enable only the toolsets needed by a conversation. A smaller list
reduces prompt footprint and limits accidental capability exposure.

## Async execution

Tool handlers on the retained active path are awaited directly. Parallel-safe
calls may overlap, but result messages retain model-call order. Interactive,
stateful, or safety-sensitive calls remain sequential. There is no thread-pool
fallback for a synchronous-only tool.

See [Tools and Toolsets](/user-guide/features/tools) and
[Toolsets Reference](./toolsets-reference.md).
