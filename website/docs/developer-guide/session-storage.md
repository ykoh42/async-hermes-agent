---
sidebar_position: 5
title: "Session Storage"
description: "Awaitable SQLite session persistence, search, lineage, and ownership"
---

# Session Storage

`SessionDB` in `hermes_state.py` stores conversations in SQLite. Its public I/O
methods are coroutines backed by `aiosqlite`, so callers do not need to add
their own thread wrapper. `aiosqlite` itself serializes SQLite calls on a
connection worker thread; this is an awaitable facade, not a zero-thread native
SQLite transport.

The default database is:

```text
$HERMES_HOME/state.db
```

`HERMES_HOME` defaults to `~/.hermes`. Pass an explicit path when an embedding
application should own storage placement.

## Direct use

```python
from hermes_state import SessionDB


db = SessionDB("/srv/my-agent/state.db")


async def inspect_session():
    try:
        await db.create_session(
            "conversation-1",
            source="api",
            model="example-model",
        )
        await db.append_message("conversation-1", "user", "Hello")
        await db.append_message("conversation-1", "assistant", "Hi")

        session = await db.get_session("conversation-1")
        messages = await db.get_messages("conversation-1")
        return session, messages
    finally:
        await db.close()
```

Construction records the path only. The connection and schema are initialized
lazily on the first awaited operation.

## Using storage with `AIAgent`

`AIAgent` receives optional storage through the existing `session_db=`
constructor argument. Without an injected `SessionDB`, an ordinary turn does
not persist a transcript. A recall tool that requires session storage may
instead open the default `$HERMES_HOME/state.db` lazily; from that point the
agent owns the handle. With a store attached at construction, a turn creates or
enriches the session row before its first provider request and incrementally
persists messages during the tool loop.

If you inject a database, lifecycle ownership transfers to that agent:
`await agent.close()` closes the attached `SessionDB`. Do not share one
instance among independently owned agents; give each agent its own instance or
put coordination behind a host abstraction that respects this close contract.

## Stored data

The schema preserves the upstream session model, including:

- session identity, source, provider/model metadata, timestamps, and working
  directory information;
- message roles, content, tool calls, tool-call identifiers, and tool names;
- reasoning and provider-specific replay metadata;
- token and auxiliary-model usage accounting;
- compression lineage, titles, archive state, and other session metadata;
- FTS indexes used by session and message search.

The system prompt is de-duplicated by hash. This supports stable prompt reuse
without storing an identical large prompt on every session row.

## Core operations

All operations below are awaited:

| Operation | Representative methods |
| --- | --- |
| Create/resume | `create_session()`, `ensure_session()`, `reopen_session()` |
| Append/replace | `append_message()`, `append_messages_batch()`, `replace_messages()` |
| Read | `get_session()`, `get_messages()`, `get_messages_as_conversation()` |
| Search | `search_messages()`, `search_sessions()`, `search_sessions_by_id()` |
| Compression | `try_acquire_compression_lock()`, `archive_and_compact()`, `release_compression_lock()` |
| Metadata | `update_session_meta()`, `update_session_model()`, `update_token_counts()` |
| Lifecycle | `end_session()`, `delete_session()`, `close()` |

Consult the method signatures in `hermes_state.py` for optional filters and
return fields; that file is the canonical API reference.

## Write serialization and WAL

One `SessionDB` instance lazily owns an `asyncio.Lock` for connection setup and
another for writes. SQLite WAL is enabled when the filesystem supports it,
with the existing journal fallback retained for filesystems where WAL is not
safe or available. SQLite's busy handling and bounded retry policy remain in
the database layer.

Separate `SessionDB` instances can operate concurrently, subject to SQLite's
normal file-level locking. A single instance still serializes mutations so
transcript order remains deterministic.

## FTS search

`search_messages()` uses the retained FTS5 routing and falls back according to
the database's available extensions and query shape. Search and index repair
remain awaited operations. Optional CJK indexing depends on the
`cjk_unicode61` loadable SQLite extension being built and available for the
current platform; when it is absent, the retained search fallback remains in
use.

```python
async def find_messages(db):
    return await db.search_messages(
        "deployment failure",
        role_filter=["user", "assistant"],
        limit=10,
    )
```

`session_search` exposes this storage to the model when its toolset is enabled.

## Crash and cancellation behavior

The turn prologue persists the user message before the first provider request.
Tool-loop progress is persisted incrementally, and finalization closes or
repairs incomplete protocol tails before its final write. On host task
cancellation, the agent shields the short finalization needed to leave durable
state consistent, then propagates cancellation.

JSONL trajectories are separate from `state.db`; see
[Trajectory Format](./trajectory-format.md).

## Shutdown

Always await `close()`. It is safe to call once in a `finally` block:

```python
async def list_recent_sessions():
    db = SessionDB()
    try:
        return await db.search_sessions(limit=20)
    finally:
        await db.close()
```

Reusing a closed `SessionDB` raises an error rather than silently opening a new
connection.
