---
sidebar_position: 5
title: "Session Storage"
description: "Native-async SQLite session persistence, search, lineage, and ownership"
---

# Session Storage

`SessionDB` in `hermes_state.py` stores conversations in SQLite. Its public
I/O methods are coroutines backed by `aiosqlite`; callers do not need a thread
wrapper around database work.

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
try:
    await db.create_session("conversation-1", source="api", model="example-model")
    await db.append_message("conversation-1", "user", "Hello")
    await db.append_message("conversation-1", "assistant", "Hi")

    session = await db.get_session("conversation-1")
    messages = await db.get_messages("conversation-1")
finally:
    await db.close()
```

Construction records the path only. The connection and schema are initialized
lazily on the first awaited operation.

## Using storage with `AIAgent`

`AIAgent` receives optional storage through the existing `session_db=`
constructor argument. Without an injected `SessionDB`, the library turn does
not silently create a global database. With one attached, a turn creates or
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
remain awaited operations. Optional CJK indexing depends on the bundled native
extension being available for the current platform.

```python
matches = await db.search_messages(
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
db = SessionDB()
try:
    rows = await db.search_sessions(limit=20)
finally:
    await db.close()
```

Reusing a closed `SessionDB` raises an error rather than silently opening a new
connection.
