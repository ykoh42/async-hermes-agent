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
from run_agent import AIAgent


async def chat_once():
    agent = AIAgent(...)
    try:
        result = await agent.run_conversation("Question")
        answer = await agent.chat("Follow-up question")
        return result, answer
    finally:
        await agent.close()
```

The method names and argument shapes remain at their upstream locations. The
methods are coroutines; calls without `await` only create coroutine objects.

Prefer the context manager:

```python
async def complete_task():
    async with AIAgent(...) as agent:
        return await agent.run_conversation("Complete this task")
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
import asyncio


async def ordered_turns(agent):
    # Safe, but these turns run in submission order rather than in parallel.
    return await asyncio.gather(
        agent.chat("First turn"),
        agent.chat("Second turn"),
    )
```

For independent work, allocate independent agents:

```python
async def independent_turns():
    async with AIAgent(...) as a, AIAgent(...) as b:
        return await asyncio.gather(
            a.chat("Independent task A"),
            b.chat("Independent task B"),
        )
```

## Cancellation

Cancelling an active turn cancels its child work, persists a partial session
when a `SessionDB` is attached, and re-raises `asyncio.CancelledError`:

```python
async def cancel_turn(agent):
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

For a chosen database path and persistence beginning at the turn prologue,
construct and pass `SessionDB` explicitly:

```python
from hermes_state import SessionDB
from run_agent import AIAgent


async def durable_turn():
    db = SessionDB("./state.db")
    async with AIAgent(..., session_db=db, session_id="customer-42") as agent:
        return await agent.chat("Remember this conversation")
```

The database connection and schema initialize on the first awaited operation.
Ordinary turns without an injected store do not persist a transcript. A recall
tool that needs session storage can lazily open the default
`$HERMES_HOME/state.db`; the agent then owns and closes that handle. The agent
also owns and closes an explicitly supplied database at shutdown. See
[Sessions](../user-guide/sessions.md) for explicit resume handling.

## Tools, skills, MCP, and memory

Select only the tool groups a conversation needs. Availability checks still
remove a tool when its credential, executable, callback, or backend is absent:

```python
async def work_with_extensions():
    async with AIAgent(
        ...,
        enabled_toolsets=["file", "skills", "memory", "mcp-project"],
    ) as agent:
        return await agent.chat("Read the project skill and continue the task.")
```

This example assumes a configured MCP server named `project`; its canonical
toolset is `mcp-project`. Put skills below
`$HERMES_HOME/skills/<name>/SKILL.md` or configure
`skills.external_dirs`. Configure MCP servers under `mcp_servers` and memory
under `memory` in `$HERMES_HOME/config.yaml`. The first awaited agent boundary
loads skills/plugins and discovers MCP tools. Agent shutdown closes the MCP
lease and the selected memory provider.

The built-in file-backed memory exposes the `memory` tool. An external memory
provider is selected with `memory.provider`; install that provider's optional
extra when its package metadata requires one. Skills and memory are mutable
application state, so isolate `HERMES_HOME` between untrusted tenants.

See [Skills](../user-guide/features/skills.md),
[Memory](../user-guide/features/memory.md), and
[MCP Configuration](../reference/mcp-config-reference.md).

## Service integration

No FastAPI application is bundled. A host framework should own startup,
shutdown, authentication, quotas, and request routing while awaiting the library
directly:

```python
from fastapi import FastAPI
from run_agent import AIAgent

app = FastAPI()


@app.post("/chat")
async def chat(message: str):
    # Isolated-job example. A production chat service should keep one agent
    # per conversation ID and close it when that conversation expires.
    async with AIAgent(...) as agent:
        return await agent.run_conversation(message)
```

FastAPI is intentionally a downstream example, not a dependency. Sharing one
global instance would both mix unrelated mutable conversation state and
serialize every request.
