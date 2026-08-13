---
sidebar_position: 4
title: "Toolsets Reference"
description: "Static tool groups and dynamically registered MCP/plugin toolsets"
---

# Toolsets Reference

Toolsets group related tool names. Pass a list through the existing
`enabled_toolsets` and `disabled_toolsets` constructor arguments; runtime
availability checks still apply after resolution.

```python
agent = AIAgent(
    ...,
    enabled_toolsets=["file", "terminal", "skills"],
)
```

## Static toolsets

| Toolset | Contents |
| --- | --- |
| `web` | `web_search`, `web_extract` |
| `search` | `web_search` |
| `x_search` | `x_search` |
| `vision` | `vision_analyze` |
| `video` | `video_analyze` |
| `image_gen` | `image_generate` |
| `video_gen` | `video_generate`, `xai_video_edit`, `xai_video_extend` |
| `bfl` | Six FLUX 3 submit, poll, and prompting-guide tools |
| `computer_use` | `computer_use` |
| `terminal` | `terminal`, `process` |
| `file` | `read_file`, `write_file`, `patch`, `search_files` |
| `skills` | `skills_list`, `skill_view`, `skill_manage` |
| `browser` | Browser operations plus `web_search` |
| `tts` | `text_to_speech` |
| `homeassistant` | Four Home Assistant entity/state/service tools |
| `todo` | `todo` |
| `memory` | `memory` |
| `session_search` | `session_search` |
| `clarify` | `clarify` |
| `delegation` | `delegate_task` |
| `context_engine` | Tools supplied by the selected context-engine plugin |

## Composite toolsets

| Toolset | Behavior |
| --- | --- |
| `debugging` | Terminal/process plus the `web` and `file` toolsets |
| `safe` | Web, vision, and image generation without terminal access |
| `coding` | Coding-oriented files, terminal, web, skills, browser, planning, memory, clarification, and delegation |
| `hermes-cli` | Historical upstream name for the full retained library tool list; no CLI application is included |

The `safe` name means “without terminal access”; it is not a complete security
sandbox. It includes external web calls and image generation.

## Dynamic toolsets

Built-in discovery also registers `code_execution` with `execute_code`. It is
not authored in the static `TOOLSETS` mapping, but it resolves through the same
registry-backed public helpers after tool discovery.

### MCP

A configured server named `github` contributes `mcp-github`. The raw server
name is also recognized as an alias. Contents follow live discovery and the
server's include/exclude filters.

### Plugins

Retained plugins can register tools into an existing or plugin-defined
toolset. Toolset resolution merges registry contributions without mutating the
static `TOOLSETS` table.

## Resolution rules

- Included toolsets are expanded recursively according to their definitions.
- Disabled tools/toolsets are filtered after expansion.
- Registry aliases resolve MCP and plugin names.
- Availability checks can remove tools whose backend or credential is absent.
- `all` or `*` expands registered toolsets, but it does not bypass an
  availability or safety check.

Use `get_toolset()`, `resolve_toolset()`, and `get_all_toolsets()` from
`toolsets.py` when a host needs to inspect the resolved configuration.

See [Tools Reference](./tools-reference.md).
