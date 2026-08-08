---
title: Tools and Toolsets
description: Select and execute the retained native-async tool surface.
sidebar_position: 2
---

# Tools and Toolsets

Tools are structured functions the model can call. Toolsets group related tools
so each agent exposes only the capability surface required by its task.

## Select toolsets per agent

```python
async with AIAgent(
    ...,
    enabled_toolsets=["file", "terminal", "web", "skills"],
    disabled_toolsets=["browser"],
) as agent:
    result = await agent.run_conversation("Investigate and patch the issue.")
```

An explicit allowlist is recommended for services and data generation. The
historical full-toolset name `hermes-cli` is retained for compatibility, but it
does not mean this distribution ships a CLI.

## Retained built-in groups

| Toolset | Main tools |
| --- | --- |
| `file` | `read_file`, `write_file`, `patch`, `search_files` |
| `terminal` | `terminal`, `process` using the local async subprocess backend |
| `web` / `search` | `web_search`, optionally `web_extract` |
| `browser` | navigate, snapshot, click, type, scroll, keyboard, CDP, console, dialogs, and browser vision |
| `skills` | `skills_list`, `skill_view`, `skill_manage` |
| `memory` | bounded persistent memory operations |
| `session_search` | search, browse, and read an attached session store |
| `vision` / `image_gen` | image analysis and provider-backed image generation |
| `todo` | durable planning state across context compression |
| `clarify` | request host/user clarification |
| `delegation` | run isolated child-agent subtasks |

Provider plugins can add web, browser, image, video, memory, or other optional
tools. MCP servers receive their own dynamic toolsets.

## Scheduling semantics

Native async handlers are awaited directly. Independent parallel-safe tool
calls may execute concurrently, but results are appended to model history in
the order the model emitted the calls. Sequential tools, interactive approval
boundaries, budget checks, guardrails, and steering barriers retain their
ordered semantics.

There is no fallback that invokes a synchronous handler in a worker thread. A
sync-only tool is rejected during async initialization or dispatch.

## Progressive discovery

Large optional plugin and MCP catalogs can be exposed through tool search so
every schema is not sent with every request. Fundamental tools in the core
surface are never deferred. This keeps the prompt-cache prefix and core schema
stable while still allowing edge capabilities to be discovered.

## Host interaction

Some tools require application decisions. For example, MCP elicitation and the
`clarify` tool use `clarify_callback`; terminal and other sensitive operations
may require host authorization policy. A headless service should fail clearly
when it has no valid interaction boundary rather than trying to read terminal
input.

Tool access is security authority. Review [Security](../security.md) before
enabling terminal, file-write, browser, third-party skill, or MCP capabilities.
