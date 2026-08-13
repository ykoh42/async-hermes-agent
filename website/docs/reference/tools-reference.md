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
| `x_search` | `query` | Search public X content through xAI; opt-in |
| `vision_analyze` | `image_url`, `question` | Analyze an image |
| `image_generate` | `prompt` | Generate an image through the selected provider plugin |
| `execute_code` | `code` | Run bounded Python orchestration over enabled Hermes tools |
| `skills_list` | none | Discover available skills |
| `skill_view` | `name` | Load one skill document |
| `skill_manage` | `action`, `name` | Create, edit, patch, or remove a skill |
| `memory` | `target` | Read or modify persistent memory/user-profile content |
| `session_search` | none | Search persisted sessions and messages |
| `todo` | none | Maintain turn planning state |
| `clarify` | `question` | Ask the host user for clarification through a callback |
| `delegate_task` | none | Launch one or more child-agent tasks; each task requires a goal |
| `text_to_speech` | `text` | Synthesize an audio file through the configured TTS provider |
| `computer_use` | `action` | Drive a cua-driver desktop session, subject to approval |

## Media and service tools

| Tool | Required input | Availability |
| --- | --- | --- |
| `video_analyze` | `video_url`, `question` | Opt-in `video` toolset |
| `video_generate` | `prompt` | Selected video-generation provider |
| `xai_video_edit` | `prompt`, `video_url` | xAI video-generation provider |
| `xai_video_extend` | `prompt`, `video_url` | xAI video-generation provider |
| `bfl_flux3_text_to_video` | `prompt` | BFL/Nous managed gateway |
| `bfl_flux3_image_to_video` | `prompt`, `input_image` | BFL/Nous managed gateway |
| `bfl_flux3_keyframes_to_video` | `prompt`, `input_images`, `keyframe_indices` | BFL/Nous managed gateway |
| `bfl_flux3_video_continuation` | `prompt`, `input_video` | BFL/Nous managed gateway |
| `bfl_flux3_get_result` | `id` | Existing BFL job |
| `bfl_flux3_prompting_guide` | none | BFL toolset enabled |
| `ha_list_entities` | none | `HASS_TOKEN` configured |
| `ha_get_state` | `entity_id` | `HASS_TOKEN` configured |
| `ha_list_services` | none | `HASS_TOKEN` configured |
| `ha_call_service` | `domain`, `service` | `HASS_TOKEN` configured |

`text_to_speech` returns synthesized media; speaker playback belongs to the
removed CLI/gateway UI and is not a separate library API. `computer_use`
requires the external `cua-driver` runtime even though its compatibility extra
does not add another Python package.

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
stateful, or safety-sensitive calls remain sequential. A synchronous-only tool
transport fails explicitly. File-backed tools still inherit the documented
`aiofiles` executor limitation; they are event-loop nonblocking but must not be
described as zero-thread or OS-native regular-file I/O.

See [Tools and Toolsets](/user-guide/features/tools) and
[Toolsets Reference](./toolsets-reference.md).
