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

An injected database is borrowed by the agent: `await agent.close()` ends the
agent's session work but does not close the attached `SessionDB`. The host that
created the store owns its lifecycle and should close it once during application
shutdown. In a service, create one store per worker lifespan and share that
store among the worker's agents; do not let an individual agent close the
shared store.

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

## PostgreSQL settings and read-only connections

See the [PostgreSQL production-readiness report](./postgres-production-readiness.md)
for the reproducible multi-worker, pool-recovery, and failure tests.

The optional PostgreSQL backend keeps the `SessionDB(db_path, read_only=False)`
constructor shape. Pass an explicit `postgresql+asyncpg://` DSN; do not rely on
an implicit `DATABASE_URL` lookup inside the library:

```python
from hermes_state_postgres import SessionDB

db = SessionDB("postgresql+asyncpg://user:password@db.example/hermes")
```

Pool and asyncpg options belong under the active profile's `config.yaml` and
use the driver names directly:

```yaml
database:
  postgres:
    pool_size: 5
    max_overflow: 10
    pool_timeout: 30
    pool_recycle: -1
    pool_pre_ping: true
    pool_use_lifo: false
    connect_args:
      timeout: 60
      command_timeout: null
      statement_cache_size: 100
      max_cached_statement_lifetime: 300
      max_cacheable_statement_size: 15360
      server_settings:
        application_name: async-hermes-agent
        statement_timeout: "60000"
        lock_timeout: "5000"
        idle_in_transaction_session_timeout: "600000"
```

The profile selected when the store is constructed owns these settings. The
async connection is still initialized at the first awaited operation, and the
validated options remain fixed for that store's lifetime. Changing the config
requires a newly created store. TLS and endpoint selection remain DSN concerns.

On a writable store, first initialization creates or additively reconciles the
retained tables, foreign keys, and query indexes under a PostgreSQL advisory
transaction lock, then records the retained schema version. It does not run
Alembic or perform destructive rewrites. A read-only store only validates the
existing version and never creates or migrates schema. When a future upstream
release changes the SQLite schema, the corresponding PostgreSQL column/data
migration must be ported and tested before that release is advertised for
existing PostgreSQL databases.

For a read replica or a search/diagnostic connection, use:

```python
readonly_db = SessionDB(read_replica_url, read_only=True)
```

This refuses SessionDB writes and forces PostgreSQL transactions into
read-only mode. It does not select a replica automatically, and it requires an
already initialized schema. A separate PostgreSQL read-only role or replica
endpoint provides an additional operational permission boundary. When sharing
one store in a service worker, estimate the possible connection count as
`workers * (pool_size + max_overflow)`.

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
