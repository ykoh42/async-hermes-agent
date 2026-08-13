---
sidebar_position: 2
title: "Agent Loop Internals"
description: "Awaited turn setup, model/tool iteration, cancellation, persistence, and finalization"
---

# Agent Loop Internals

The public loop is `AIAgent.run_conversation()` in `run_agent.py`. Its detailed
implementation remains in `agent/conversation_loop.py`, with turn setup and
finalization split into focused modules at the same layer.

## Public interfaces

```python
result = await agent.run_conversation(
    "Investigate the failure",
    system_message=None,
    conversation_history=None,
    task_id=None,
)

answer = await agent.chat("Summarize the result")
await agent.close()
```

`run_conversation()` returns the full result dictionary used by upstream
library integrations. Important fields include `final_response`, `messages`,
`completed`, and `api_calls`; recovery paths may also report partial or cleanup
metadata. `chat()` returns only `result["final_response"]`.

The implementation preserves these names rather than adding `arun_*` aliases
or synchronous wrappers.

## Lifecycle

### Construction

`AIAgent.__init__()` stores state and configuration. It does not open network,
database, or MCP connections.

### Lazy initialization

`async with AIAgent(...)` initializes provider, plugin, and MCP state before
returning the agent. Calling `run_conversation()` directly also reaches the
same awaited initialization path.

### Close

`await agent.close()` is idempotent. It stops owned child tasks, waits for
their termination, releases MCP ownership, and closes owned clients and
session storage. Prefer the async context manager so exceptions cannot skip
cleanup.

## One turn

### 1. Serialize the instance

`run_conversation()` acquires a per-agent `asyncio.Lock`. Two callers sharing
one agent cannot interleave mutations to its prompt cache or transcript. They
wait in arrival order determined by the event loop. Separate agents have
independent locks.

### 2. Build turn context

`await build_turn_context()` in `agent/turn_context.py` performs the once-per-
turn prologue:

- normalize the user message and restore or build the system prompt;
- resume or create the durable session when a `SessionDB` is attached;
- hydrate turn-scoped state such as todos and interrupt tracking;
- prefetch configured memory and refresh the MCP tool snapshot;
- run the pre-model plugin hook;
- estimate context pressure and compress when required;
- persist the initial user turn before the first provider request.

Pure message transformations remain synchronous helpers inside this awaited
workflow.

### 3. Call the provider

The selected transport is awaited directly. Provider adapters translate the
OpenAI-shaped internal conversation into the provider's wire format and back
without changing the stored message model. Streaming callbacks may receive
text deltas, but the host still awaits one complete turn result.

### 4. Execute tools

When the assistant returns tool calls, `agent/tool_executor.py` divides the
batch at ordering and interaction barriers.

- Async handlers are awaited directly.
- Independent, parallel-safe calls may execute in a `TaskGroup`.
- Sequential calls preserve their barriers.
- Results are appended in the model's original call order.
- Safety checks, budgets, checkpointing, middleware, and callbacks remain in
  the dispatch path.

The completed assistant/tool sequence is persisted before the next provider
iteration.

### 5. Continue or finish

The loop repeats until it receives a final text response, an internal
interrupt, an error path, or an iteration-budget exit. If the normal budget is
exhausted without a final answer, the existing one-call, tool-free summary
path is used when eligible.

### 6. Finalize

`await finalize_turn()` in `agent/turn_finalizer.py`:

- closes incomplete tool sequences when necessary;
- saves a trajectory when enabled;
- cleans up task-scoped browser and process resources;
- persists the final transcript and session metadata;
- returns the result dictionary even when a non-critical cleanup surface
  fails, recording cleanup errors instead of discarding a valid answer.

## Message invariants

Messages use the familiar `system`, `user`, `assistant`, and `tool` roles.
Reasoning is retained on assistant messages separately from visible content.
The loop maintains provider-valid assistant tool calls followed by matching
tool observations and avoids orphaned tool results.

The cached system prompt is not rebuilt mid-conversation. Compression is the
intentional mechanism that can replace older context while maintaining a
stable resumed conversation.

## Cancellation and interrupts

External task cancellation and Hermes' internal interrupt have different
contracts:

- On `asyncio.CancelledError`, the loop shields the short finalization needed
  to persist partial session and trajectory state, then re-raises cancellation
  to the host.
- An internal interrupt stops at a safe boundary and returns the established
  partial result shape.

Provider, tool, compression, and persistence awaits remain cancellation
points. Cleanup tasks are awaited so they do not leak into the host loop.

## Synchronous fallback audit

The retained provider and tool transports do not call `asyncio.to_thread()`,
`run_in_executor()`, `run_until_complete()`, or blocking future `.result()` to
hide a synchronous SDK. A retained provider or tool without a supported
native-async path fails explicitly.

The persistence layer has a documented implementation boundary:
`aiofiles` implements regular-file operations through an executor, and
`aiosqlite` queues embedded SQLite work to a connection thread. CPython has no
portable asyncio API for regular files, metadata, durability operations, or
embedded SQLite. These operations remain directly awaitable and do not block
the host event loop, but zero-thread persistence is not a package guarantee.

This rule applies to I/O. Keeping a short deterministic calculation as a
normal `def` is expected and avoids coroutine overhead with no concurrency
benefit.

## Related documentation

- [Architecture](./architecture.md)
- [Session Storage](./session-storage.md)
- [Trajectory Format](./trajectory-format.md)
- [Tools and Toolsets](/user-guide/features/tools)
