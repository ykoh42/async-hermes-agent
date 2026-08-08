---
title: Model Providers
description: Select a native-async model transport and install only its required dependencies.
sidebar_position: 2
---

# Model Providers

`AIAgent` separates a provider profile from the model identifier. A profile
resolves credentials, base URL, API mode, and transport behavior; `model` names
the model or deployment exposed by that provider.

## Prefer explicit construction

Explicit arguments are easiest to audit in services and tests:

```python
import os

from run_agent import AIAgent

agent = AIAgent(
    provider="openrouter",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    model=os.environ["OPENROUTER_MODEL"],
)
```

Do not assume that a provider's current free models, aliases, context windows,
or prices are stable. Supply a model that supports the tool-calling and
reasoning behavior required by your application.

## Retained transport families

The retained runtime has native async paths for these families:

| Family | Typical profiles | Dependency |
| --- | --- | --- |
| OpenAI-compatible chat/responses | OpenRouter, custom/local endpoints, DeepSeek, xAI and other compatible gateways | Base install |
| Native Anthropic | `anthropic` | `anthropic` extra |
| Google Gemini HTTP | `gemini` | Base install |
| Google Vertex | `vertex` | `vertex` extra for credentials |
| Microsoft Foundry/Azure | `azure-foundry` | `azure-identity` extra for Entra ID |
| AWS Bedrock | `bedrock` | `bedrock` extra |
| Codex Responses and Copilot ACP | Corresponding bundled profiles | Profile-specific credentials/runtime |

Bundled profile discovery includes additional OpenAI-compatible services. A
profile in the source tree means Hermes knows how to resolve that service; it
does not guarantee that an external account, endpoint, model, or optional SDK is
currently available.

## Install an optional transport

From a source checkout:

```bash
uv sync --extra anthropic
uv sync --extra vertex
uv sync --extra azure-identity
uv sync --extra bedrock
```

See [Installation](../getting-started/installation.md) for the complete extras
list.

## Custom OpenAI-compatible endpoint

```python
agent = AIAgent(
    provider="custom",
    base_url="http://127.0.0.1:8000/v1",
    api_key="local-or-required-key",
    model="your-served-model",
)
```

The endpoint must implement the selected OpenAI-compatible API and support the
message/tool schema used by your workload. Running a local model server is
outside this package.

## Configuration-based selection

For applications that prefer file configuration, use non-secret settings in
`$HERMES_HOME/config.yaml`:

```yaml
model:
  provider: openrouter
  default: "your/model-id"
  base_url: "https://openrouter.ai/api/v1"
```

Keep the credential in `$HERMES_HOME/.env` or the process environment:

```dotenv
OPENROUTER_API_KEY=...
```

Explicit constructor arguments take priority for that agent instance. Details
are in [Configuring models](../user-guide/configuring-models.md).

## Failure behavior

Provider setup and requests are awaited. If a selected API mode has no native
async transport, initialization fails explicitly; the runtime does not call a
synchronous SDK through `asyncio.to_thread()`.

Always close an initialized provider with `await agent.close()` or an async
context manager.
