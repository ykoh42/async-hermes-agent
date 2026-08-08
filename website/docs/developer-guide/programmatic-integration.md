---
sidebar_position: 7
title: "Programmatic Integration"
description: "Embed AIAgent in an existing asyncio application and manage its lifecycle safely"
---

# Programmatic Integration

Async Hermes Agent is consumed as a Python library. It does not bundle an ACP
server, JSON-RPC gateway, FastAPI application, or OpenAI-compatible HTTP server.
An application can add any of those boundaries while keeping ownership of
authentication, request validation, rate limits, and deployment.

## Smallest integration

```python
from run_agent import AIAgent


async def answer(message: str) -> str:
    async with AIAgent(...) as agent:
        return await agent.chat(message)
```

For full messages and metadata, await `run_conversation()` instead:

```python
result = await agent.run_conversation(message)
text = result["final_response"]
history = result["messages"]
```

Do not call these methods from `asyncio.to_thread()`, and do not add a
`run_until_complete()` wrapper inside an already-running event loop.

## Long-lived host lifecycle

Create long-lived agents during the host's async startup and close them during
shutdown:

```python
from contextlib import AsyncExitStack

from run_agent import AIAgent


stack = AsyncExitStack()
agent = await stack.enter_async_context(
    AIAgent(...)
)

try:
    result = await agent.run_conversation("Hello")
finally:
    await stack.aclose()
```

`close()` is idempotent, so explicit fallback cleanup is safe.

## FastAPI example

FastAPI is not a dependency of this package. If the host already uses it, wire
the agent into FastAPI's lifespan rather than adding framework code to the
library:

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI

from run_agent import AIAgent


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AIAgent(...) as agent:
        app.state.agent = agent
        yield


app = FastAPI(lifespan=lifespan)


@app.post("/chat")
async def chat(message: str):
    return await app.state.agent.run_conversation(message)
```

This example demonstrates lifecycle placement only. A production endpoint must
add its own request schema, identity and session mapping, authorization,
timeouts, quotas, and error policy.

## Choosing the agent concurrency model

A single `AIAgent` represents one mutable conversation. Its turn lock
serializes concurrent calls, which is correct for ordered conversation state.
Do not use one global agent when requests represent unrelated users.

Common host designs are:

- one agent per active conversation, cached by an application-owned session
  identifier;
- a short-lived agent per independent job;
- a bounded collection of independent agents for batch-style work.

Different agents can overlap provider, database, MCP, and tool I/O. Bound the
number of active agents according to provider limits and the external
resources each tool can consume.

## Session identity

Pass the established `session_id` constructor argument when a host needs to
resume a durable conversation. Use one `SessionDB` policy consistently and
avoid mapping multiple unrelated users to the same session identifier.

See [Session Storage](./session-storage.md) and
[Sessions](/user-guide/sessions).

## Streaming callbacks

`run_conversation()` and `chat()` accept the retained `stream_callback`
argument. The callback is invoked with visible text deltas while the coroutine
continues toward a complete result. Keep callback work short and non-blocking;
if it must perform I/O, enqueue the delta to host-owned async infrastructure
instead of blocking the callback.

## Cancellation and timeouts

Host cancellation propagates through provider and tool awaits. The agent first
finishes its short partial-state finalizer, then re-raises `CancelledError`.
Use ordinary asyncio timeout mechanisms around a turn and retain normal
`finally: await agent.close()` cleanup.

Do not suppress cancellation unless the host deliberately converts it into an
application-level result.

## Interactive tools

The retained `clarify` and approval paths require host callbacks when they need
human input. A headless service should register appropriate callbacks or
exclude interactive capabilities from its enabled toolsets. The library does
not read from a hidden terminal as a service fallback.

## Training-data workloads

For many independent prompts, use `BatchRunner` instead of constructing an
unbounded set of tasks. It supplies worker limits, checkpoints, JSONL-safe
writes, resume, statistics, and merged trajectories. It remains a data
generation harness, not a model-training implementation.

See [Python Library](/guides/python-library),
[Batch Processing](/user-guide/features/batch-processing), and
[Trajectory Format](./trajectory-format.md).
