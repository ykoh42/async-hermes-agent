---
sidebar_position: 1
title: Quickstart
description: Choose any supported provider and run your first awaited Hermes Agent conversation.
---

# Quickstart

This guide starts after installation, lets you choose any supported provider,
and runs one complete conversation. The public API keeps the upstream module
and method names: add `await`; there is no separate `arun_*` API.

## 1. Install and verify

Follow the [Installation guide](./installation.md) and run its async API check
before configuring a provider.

## 2. Choose a provider and model

Pick the route that matches your deployment. No bundled provider profile is
the default or the assumed path.

| Route | Use it when | What the host supplies |
| --- | --- | --- |
| Bundled provider profile | You use one of the provider profiles shipped with the library | Provider name, model ID, and that provider's credential or identity |
| Custom OpenAI-compatible endpoint | You operate vLLM, SGLang, Ollama, LM Studio, or another compatible server | `custom`, base URL, model ID, and any required key |
| File-backed defaults | Several agent instances share one route | Non-secret model settings in `$HERMES_HOME/config.yaml`; credentials in `.env` |

See the complete [Provider catalog](../integrations/providers.md) before picking
a credential or optional dependency.

For a provider-neutral runnable example, let the host application expose four
ordinary application variables:

```bash
export MODEL_PROVIDER="<provider-profile>"
export MODEL_ID="<tool-capable-model-id>"
export MODEL_API_KEY="<credential-if-required>"
# Only custom or overridden routes need this:
export MODEL_BASE_URL="<optional-base-url>"
```

These are example variables owned by your host application, not additional
Hermes configuration keys. Provider catalogs, aliases, availability, and prices
change over time, so this guide does not pin a vendor or model name.

## 3. Run one conversation

```python
import asyncio
import os

from run_agent import AIAgent


async def main() -> None:
    async with AIAgent(
        provider=os.environ["MODEL_PROVIDER"],
        base_url=os.getenv("MODEL_BASE_URL") or None,
        api_key=os.getenv("MODEL_API_KEY") or None,
        model=os.environ["MODEL_ID"],
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

## 4. Choose the return shape

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

Next, see [Configuration](../user-guide/configuration.md), the
[Python library guide](../guides/python-library.md),
[Tools](../user-guide/features/tools.md), and
[Sessions](../user-guide/sessions.md).
