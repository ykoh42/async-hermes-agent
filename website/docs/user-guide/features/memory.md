---
title: Memory
description: Use bounded file-backed memory, user profiles, and optional async memory providers.
sidebar_position: 6
---

# Memory

Memory stores curated facts across conversations. It is separate from session
history: a session preserves a transcript, while memory keeps a small set of
facts intended to remain useful beyond one transcript.

## Built-in stores

The built-in provider uses two files:

```text
$HERMES_HOME/memories/
├── MEMORY.md    # Agent notes, environment facts, and conventions
└── USER.md      # User preferences and profile information
```

Enable the prompt surfaces in `$HERMES_HOME/config.yaml`:

```yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200
  user_char_limit: 1375
```

Then expose the write/read tool to the model:

```python
agent = AIAgent(..., enabled_toolsets=["memory"])
```

`skip_memory=True` disables memory loading for an agent regardless of the shared
configuration. The batch runner sets this deliberately so one dataset item
cannot contaminate another.

## Cache-stable snapshots

Memory and user-profile text are bounded and frozen into the conversation's
system-prompt snapshot. A memory tool call can update the files, but it does not
mutate the already cached prompt prefix halfway through a conversation. A later
conversation sees the updated snapshot.

## External providers

The retained plugin surface includes Mem0 and ByteRover integrations. The
built-in provider remains available, and at most one external memory provider
can be active at a time. Provider initialization, prefetch, turn sync, tool
calls, session transitions, and shutdown use async contracts.

Mem0's packaged SDK is optional (`mem0` install extra); self-hosted HTTP modes
can use the base async HTTP client. Availability still depends on external
credentials and services.

## Privacy and tenancy

Memory is application state, not a model-provider privacy boundary. Do not
store credentials or sensitive material that the model should not receive.
Isolate `HERMES_HOME`, memory provider namespaces, and session databases between
untrusted tenants. See [Security](../security.md).
