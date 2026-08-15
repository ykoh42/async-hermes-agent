---
title: Sessions
description: Preserve and resume ordered conversations with awaitable SQLite or PostgreSQL SessionDB.
sidebar_position: 4
---

# Sessions

An `AIAgent` keeps its live conversation history across sequential calls. To
persist that history across agent instances or processes, attach a `SessionDB`
explicitly.

## In-memory lifecycle

```python
from run_agent import AIAgent


async def in_memory_conversation():
    async with AIAgent(...) as agent:
        await agent.chat("My project uses Python 3.12.")
        return await agent.chat("Which Python version did I mention?")
```

One instance represents one ordered conversation. Concurrent turns on that
instance are serialized; separate instances can run concurrently.

## Enable durable storage

```python
from hermes_state import SessionDB
from run_agent import AIAgent

async def stored_conversation():
    db = SessionDB("./state.db")
    async with AIAgent(
        ...,
        session_db=db,
        session_id="project-review",
    ) as agent:
        return await agent.run_conversation("Review this project.")
```

`SessionDB` uses `aiosqlite`. Its constructor is state-only; the connection and
schema initialize on the first awaited operation. Passing it explicitly starts
durable transcript persistence at the turn prologue and selects the database
path. A recall tool may otherwise open the default `$HERMES_HOME/state.db`
lazily. In either case the agent closes the database it owns during shutdown.
`SessionDB.close()` is idempotent.

## Use PostgreSQL for a service

SQLite remains the default and is a good fit for a single-process application.
For a service whose workers share a durable store, install the optional
PostgreSQL backend and inject it through the same existing `session_db=`
argument:

```bash
uv sync --extra postgres
```

```python
from hermes_state_postgres import SessionDB
from run_agent import AIAgent


async def postgres_conversation():
    db = SessionDB(
        "postgresql+asyncpg://user:password@db.example:5432/hermes",
    )
    try:
        async with AIAgent(
            ...,
            session_db=db,
            session_id="project-review",
        ) as agent:
            return await agent.run_conversation("Review this project.")
    finally:
        await db.close()
```

The PostgreSQL `SessionDB` keeps the SQLite method names and awaited calling
style; only the import and explicit DSN change. A store reads the active
profile's `database.postgres` pool and driver settings once when its first
database operation initializes the engine. Create a new store after changing
those settings. See [SessionDB storage and PostgreSQL settings](../developer-guide/session-storage.md)
for the supported options.

For a read-only endpoint, pass `read_only=True`:

```python
readonly_db = SessionDB(read_replica_url, read_only=True)
```

This blocks SessionDB writes and enables PostgreSQL transaction read-only mode;
it does not choose a replica automatically. In a multi-worker service, create
and close one store per worker lifespan and share it with that worker's agents.
Plan for a possible connection count of
`workers * (pool_size + max_overflow)`. Other retained stores, such as memory
plugin databases, remain separate from the core PostgreSQL SessionDB.

## Resume in a new agent

Passing only the old `session_id` does not automatically load its messages.
Resolve the current compression descendant, load model history, and provide it
to `run_conversation()`:

```python
from hermes_state import SessionDB
from run_agent import AIAgent

async def resume_conversation():
    db = SessionDB("./state.db")
    tip = await db.resolve_resume_session_id("project-review")
    model_history, display_history = await db.get_resume_conversations(tip)

    async with AIAgent(
        ...,
        session_db=db,
        session_id=tip,
    ) as agent:
        result = await agent.run_conversation(
            "Continue from the saved review.",
            conversation_history=model_history,
        )
    return result, display_history
```

`model_history` is the alternation-repaired history fed to the model.
`display_history` includes the full ancestor-to-tip lineage for applications
that render a transcript.

## Search and compression

The store retains message order, structured tool calls, reasoning, display
metadata, and compression lineage. Its FTS-backed search powers the optional
`session_search` toolset:

```python
async def search_prior_sessions():
    async with AIAgent(
        ...,
        session_db=SessionDB("./state.db"),
        enabled_toolsets=["session_search"],
    ) as agent:
        return await agent.chat("Find the earlier deployment discussion.")
```

Context compression can end one database session and continue in a linked child
session. Always call `resolve_resume_session_id()` before a cross-process
resume so post-compression messages are not missed.

Cancellation of an active conversation attempts a crash-safe partial persist
before propagating `CancelledError`. This protects recoverability but does not
replace application-level backups or SQLite filesystem durability planning.
