---
sidebar_position: 2
title: "Upstream Differences"
description: "The deliberate differences between Hermes Agent v2026.8.13 and Async Hermes Agent"
---

# Upstream differences

Async Hermes Agent is based on upstream Hermes Agent `v2026.8.13` (Python
package version `0.20.1`). The table below records the deliberate differences
in the retained library surface. It is a migration guide, not a claim that the
removed upstream applications are still shipped.

| Area | Upstream `v2026.8.13` | Async Hermes Agent `0.20.1.2` | Integration impact |
| --- | --- | --- | --- |
| Public entry points | Retained names and module paths | The same retained names, arguments, defaults, and return shapes; I/O-bearing calls are coroutines | Existing library callers normally add `await` at the call site |
| Agent construction | Synchronous upstream lifecycle | `AIAgent.__init__()` is state-only; provider, session, MCP, and plugin setup starts at an awaited boundary | Use `async with AIAgent(...)` or `await agent.close()` |
| Conversation lifecycle | Synchronous turn execution | `await agent.run_conversation(...)` and `await agent.chat(...)`; turns on one agent remain serialized | Keep one agent per ordered conversation; separate agents can overlap |
| I/O model | Synchronous provider, MCP, subprocess, file, and SQLite boundaries | Native coroutine transports and awaited public I/O; regular files use `aiofiles`, SQLite uses `aiosqlite` | The host event loop is not blocked by the public I/O paths; zero-thread file/SQLite I/O is not promised |
| Cancellation and cleanup | Synchronous cleanup semantics | Partial state is persisted before external `CancelledError` is re-raised; owned clients, processes, and tasks are closed deterministically | Host cancellation can safely propagate through a request or job |
| Concurrency | Upstream application scheduling | Same-agent turn lock, bounded batch workers, profile-scoped caches and clients | Unrelated conversations and batch items can run concurrently without sharing mutable state |
| Sessions and memory | Upstream persistence behavior | Async `SessionDB`, FTS search, memory, checkpoint, export/import, and cold-process resume | `await` the existing session methods; use a stable `HERMES_HOME`/session policy |
| PostgreSQL SessionDB | Upstream ships its SQLite session store | Additive `hermes_state_postgres.SessionDB` uses SQLAlchemy Core + asyncpg; `hermes_state.SessionDB` remains unchanged | Install `postgres`, inject one worker-owned store explicitly, and close it from the host lifespan; PostgreSQL ranking can differ from SQLite BM25 |
| Trajectories | Upstream reasoning/tool/observation format | Same ordering and retained JSON shape, with async persistence and compression | Existing trajectory consumers can read the same retained fields |
| Training-data runner | Synchronous runner boundaries | `MiniSWERunner` and `BatchRunner` keep their upstream names while becoming coroutines; checkpoint, resume, shards, merged JSONL, and statistics remain | `await runner.run_task(...)` or `await runner.run(...)` |
| Profile isolation | Process-oriented environment and cache assumptions | Task-local secrets plus canonical `HERMES_HOME` state isolate concurrent profiles and symlink aliases | A/B profiles can run in one process without borrowing each other's credentials or files |
| Provider policy | Synchronous adapters, including SDK-specific bootstrap behavior | Native async adapters are used where available; unsupported or unsafe synchronous paths fail explicitly rather than moving to a hidden worker thread | Install the relevant extra and follow provider-specific limitations |
| FastAPI/service boundary | Upstream product applications may own service surfaces | No FastAPI server, CLI/TUI, messaging bridge, scheduler, dashboard, or desktop application is bundled | The host application owns HTTP lifecycle, auth, routing, quotas, and shutdown |
| MCP and skills | Upstream product-managed discovery | Retained stdio, Streamable HTTP, and SSE MCP clients plus filesystem/external skill discovery with async lifecycle cleanup | Configure them from the host's Hermes home and close the agent at shutdown |
| Optional providers and tools | Upstream distribution layout | Provider-specific extras remain opt-in (`anthropic`, `vertex`, `azure-identity`, `bedrock`, memory, web, media, and execution backends) | Install only the extras used by the selected configuration |
| Python and package version | Upstream baseline `0.20.1` | Python `>=3.11,<3.14`; package `0.20.1.2` (`async_revision=2`) | The first three version segments track upstream; the fourth tracks this async distribution |

## Intentional public-surface exception

The retained upstream callables preserve their public names and argument
shapes. `TrajectoryCompressor.close()` is the one explicit lifecycle addition:
the async port owns an async model client and needs a public cleanup boundary
for it. It is not an `aclose()` alias or a synchronous compatibility wrapper.

## What is not changed

The conversion does not redesign the model-tool schema, rename upstream
modules, add `arun_*` aliases, or silently run synchronous provider code in a
thread. Pure CPU transformations remain synchronous. Provider output,
message-role alternation, tool-call ordering, trajectory ordering, checkpoint
semantics, and retained return dictionaries are preserved and covered by
behavior-level parity tests.

For the supported feature set and installation commands, see
[Installation](../getting-started/installation.md). For host lifecycle
examples, see [Programmatic Integration](./programmatic-integration.md).
