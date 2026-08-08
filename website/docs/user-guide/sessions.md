---
title: Sessions
description: Preserve and resume ordered conversations with the native-async SQLite SessionDB.
sidebar_position: 4
---

# Sessions

An `AIAgent` keeps its live conversation history across sequential calls. To
persist that history across agent instances or processes, attach a `SessionDB`
explicitly.

## In-memory lifecycle

```python
async with AIAgent(...) as agent:
    await agent.chat("My project uses Python 3.12.")
    answer = await agent.chat("Which Python version did I mention?")
```

One instance represents one ordered conversation. Concurrent turns on that
instance are serialized; separate instances can run concurrently.

## Enable durable storage

```python
from hermes_state import SessionDB
from run_agent import AIAgent

db = SessionDB("./state.db")

async with AIAgent(
    ...,
    session_db=db,
    session_id="project-review",
) as agent:
    result = await agent.run_conversation("Review this project.")
```

`SessionDB` uses `aiosqlite`. Its constructor is state-only; the connection and
schema initialize on the first awaited operation. `AIAgent` does not create a
database automatically and closes a supplied database during agent shutdown.
`SessionDB.close()` is idempotent.

## Resume in a new agent

Passing only the old `session_id` does not automatically load its messages.
Resolve the current compression descendant, load model history, and provide it
to `run_conversation()`:

```python
from hermes_state import SessionDB
from run_agent import AIAgent

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
```

`model_history` is the alternation-repaired history fed to the model.
`display_history` includes the full ancestor-to-tip lineage for applications
that render a transcript.

## Search and compression

The store retains message order, structured tool calls, reasoning, display
metadata, and compression lineage. Its FTS-backed search powers the optional
`session_search` toolset:

```python
agent = AIAgent(
    ...,
    session_db=SessionDB("./state.db"),
    enabled_toolsets=["session_search"],
)
```

Context compression can end one database session and continue in a linked child
session. Always call `resolve_resume_session_id()` before a cross-process
resume so post-compression messages are not missed.

Cancellation of an active conversation attempts a crash-safe partial persist
before propagating `CancelledError`. This protects recoverability but does not
replace application-level backups or SQLite filesystem durability planning.
