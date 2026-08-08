---
sidebar_position: 1
title: Quickstart
description: Run a first awaited conversation with the native-async Hermes agent library.
---

# Quickstart

The public API keeps the upstream module and method names. Library users add
`await`; there is no separate `arun_*` API and no synchronous wrapper.

## 1. Install

Follow the [Installation guide](./installation.md), then provide credentials in
the process environment or `$HERMES_HOME/.env`. The following example uses an
OpenRouter-compatible endpoint, but deliberately leaves model selection to you:

```bash
export OPENROUTER_API_KEY="..."
export OPENROUTER_MODEL="<a-current-tool-capable-model>"
```

Provider catalogs and free-model availability change over time, so examples do
not pin a free model name.

## 2. Run one conversation

```python
import asyncio
import os

from run_agent import AIAgent


async def main() -> None:
    async with AIAgent(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=os.environ["OPENROUTER_MODEL"],
        enabled_toolsets=["file", "terminal"],
    ) as agent:
        result = await agent.run_conversation(
            "Inspect the current project and identify its main modules."
        )
        print(result["final_response"])


asyncio.run(main())
```

`AIAgent.__init__()` performs state-only construction. The async context manager
initializes provider, plugin, and MCP resources and always awaits `close()`.

## 3. Choose the return shape

`run_conversation()` returns the full turn result. Its stable core fields are the
final response, message history, and completion state. Normal completed turns
also carry session and routing metadata, while early terminal/error results may
omit those optional fields:

```python
result = await agent.run_conversation("Use the available tools if needed.")
print(result["final_response"])
print(result["messages"])
print(result.get("session_id"))
```

For a string-only result, use `chat()`:

```python
answer = await agent.chat("Summarize your findings in three bullets.")
```

When not using `async with`, close explicitly:

```python
agent = AIAgent(...)
try:
    answer = await agent.chat("Hello")
finally:
    await agent.close()
```

## Concurrency behavior

Turns submitted concurrently to one `AIAgent` instance are serialized so they
cannot corrupt a shared conversation. Separate instances can overlap I/O-bound
work. Use one instance for an ordered conversation and separate instances for
independent requests.

Next, see the [Python library guide](../guides/python-library.md),
[tools](../user-guide/features/tools.md), and
[sessions](../user-guide/sessions.md).
