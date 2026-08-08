---
title: Python Library
description: Embed the native-async Hermes agent loop in an asyncio application.
sidebar_position: 1
---

# Python Library

Async Hermes Agent is designed to be embedded. It ships the agent harness, not
an application server or user interface.

## Core interface

```python
import asyncio

from run_agent import AIAgent

agent = AIAgent(...)
result = await agent.run_conversation("Question")
answer = await agent.chat("Follow-up question")
await agent.close()
```

The method names and argument shapes remain at their upstream locations. The
methods are coroutines; calls without `await` only create coroutine objects.

Prefer the context manager:

```python
async with AIAgent(...) as agent:
    result = await agent.run_conversation("Complete this task")
```

Entering initializes the selected provider, discovers plugins, and establishes
configured MCP lifecycles. Exiting closes model clients, MCP sessions, child
tasks, memory providers, and an attached session database. `close()` is
idempotent.

## Conversation results

The stable result surface includes:

```python
result["final_response"]
result["messages"]
result["completed"]
```

Normal completed turns can also include `session_id`, provider/model routing,
token and cost fields, API-call counts, reasoning, and turn-exit metadata. Early
terminal/error results and providers that do not report a usage field may omit
them, so consumers should use `.get()` for optional metadata and accounting
data.

`chat()` is a convenience interface that returns only `final_response`.

## Conversation ownership

One agent represents one ordered conversation. Its turn lock serializes
concurrent calls:

```python
# Safe, but these two turns run in submission order rather than in parallel.
first, second = await asyncio.gather(
    agent.chat("First turn"),
    agent.chat("Second turn"),
)
```

For independent work, allocate independent agents:

```python
async with AIAgent(...) as a, AIAgent(...) as b:
    result_a, result_b = await asyncio.gather(
        a.chat("Independent task A"),
        b.chat("Independent task B"),
    )
```

## Cancellation

Cancelling an active turn cancels its child work, persists a partial session
when a `SessionDB` is attached, and re-raises `asyncio.CancelledError`:

```python
task = asyncio.create_task(agent.run_conversation("Long task"))
task.cancel()
try:
    await task
except asyncio.CancelledError:
    pass
```

Hermes-internal interrupts are different: they return a partial result dict so
the caller can display or inspect the work completed so far.

## Explicit durable sessions

Durable session storage is opt-in. Construct and pass `SessionDB`; an agent does
not create one automatically:

```python
from hermes_state import SessionDB
from run_agent import AIAgent

db = SessionDB("./state.db")
async with AIAgent(..., session_db=db, session_id="customer-42") as agent:
    await agent.chat("Remember this conversation")
```

The database connection and schema initialize on the first awaited operation.
The agent owns and closes the supplied database at shutdown. See
[Sessions](../user-guide/sessions.md) for explicit resume handling.

## Service integration

No FastAPI application is bundled. A host framework should own startup,
shutdown, authentication, quotas, and request routing while awaiting the library
directly:

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

FastAPI is intentionally a downstream example, not a dependency. A production
service should usually map each ordered conversation to its own agent instance;
sharing one instance serializes every request.
