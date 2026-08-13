---
title: Feature Overview
description: What the native-async Hermes library retains and what applications must provide.
sidebar_position: 1
---

# Feature Overview

Async Hermes Agent is a library-focused conversion of the Hermes agent harness.
It preserves the core behavior needed for tool-using inference and trajectory
generation while moving the retained I/O path to native async APIs.

## Retained surfaces

| Surface | What it provides |
| --- | --- |
| Agent loop | Interleaved reasoning, model calls, tool calls, observations, compression, and finalization |
| Providers | Awaitable model transports and lazily discovered profiles, with documented SDK boundaries |
| Tools | File, local terminal, web, browser, vision/media, planning, clarify, delegation, memory, and session search |
| Skills | On-demand procedural instructions from local or shared directories |
| MCP | Stdio and HTTP external tool servers with async lifecycle ownership |
| Memory | Bounded file-backed memory plus optional external providers |
| Sessions | Explicit awaitable SQLite persistence and resume helpers |
| Training data | Ordered trajectories, batch generation, checkpoints, statistics, and compression utilities |

Learn more in [Tools](./tools.md), [Skills](./skills.md), [MCP](./mcp.md),
[Memory](./memory.md), [Browser automation](./browser.md), and
[Batch processing](./batch-processing.md).

## Native-async contract

The existing public names are coroutines:

```python
result = await agent.run_conversation("Question")
answer = await agent.chat("Follow-up")
await agent.close()
```

The retained runtime directly awaits model, tool, MCP, subprocess, and database
entry points. Unsupported synchronous provider and tool transports fail
explicitly. Regular-file operations currently use executor-backed `aiofiles`,
and the awaitable SQLite facade uses `aiosqlite`'s connection worker thread.
Supported CPython versions expose no portable asyncio regular-file or embedded
SQLite API. The package therefore guarantees directly awaitable,
event-loop-nonblocking entry points, not zero-thread or OS-native persistence.

Native async allows independent I/O-bound work to overlap; it is not a promise
that every workload will use less CPU or memory. Provider latency, model cost,
tool behavior, and host limits still dominate many deployments.

## Behavioral invariants

The conversion preserves the Hermes narrow-waist contracts:

- the system-prompt prefix remains stable during a conversation;
- strict model-message alternation is maintained;
- model tool-call order determines observation order, including parallel-safe
  execution;
- saved trajectories preserve reasoning, calls, observations, and the final
  answer;
- one agent's turns are serialized while independent agents may overlap;
- cancellation cleans up child tasks and attempts partial persistence.

## Intentionally not included

This package does not ship the upstream classic CLI, TUI, desktop app,
dashboard, messaging platforms, scheduler, FastAPI application, or editor
adapter. It also does not train a model. It provides the inference and
training-data harness that a service or fine-tuning pipeline can embed.
