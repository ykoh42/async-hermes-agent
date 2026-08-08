---
title: Configuring Models
description: Select provider, model, credentials, and reasoning settings for an async agent.
sidebar_position: 3
---

# Configuring Models

Every agent needs a provider route and a model identifier. Model names are
controlled by external providers and are not fixed in these docs.

## Explicit per-agent configuration

```python
import os

from run_agent import AIAgent

agent = AIAgent(
    provider="openrouter",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    model=os.environ["OPENROUTER_MODEL"],
    max_iterations=30,
)
```

This form is recommended for services, evaluations, and batch jobs because the
route is visible in code and can be injected by the host application.

## File-based defaults

For shared defaults, write `$HERMES_HOME/config.yaml`:

```yaml
model:
  provider: openrouter
  default: "your/model-id"
  base_url: "https://openrouter.ai/api/v1"
```

Put the credential in `$HERMES_HOME/.env`:

```dotenv
OPENROUTER_API_KEY=...
```

Explicit constructor values override defaults for that instance. Keep API keys
out of `config.yaml` even when a provider accepts them there.

## Native providers and extras

OpenAI-compatible endpoints use the base installation. Native Anthropic,
Vertex, Entra ID, and Bedrock routes need their corresponding optional extras.
See [Model providers](../integrations/providers.md) for the transport matrix and
[Installation](../getting-started/installation.md) for install commands.

## Reasoning and output limits

Pass provider-supported reasoning controls explicitly:

```python
agent = AIAgent(
    ...,
    reasoning_config={"enabled": True, "effort": "low"},
    max_tokens=4096,
)
```

Provider support varies. `max_tokens` limits one generated response; it is not
the model's total context window. Do not assume that every provider accepts the
same reasoning effort names or returns reasoning content.

For trajectory generation, choose a route that actually returns reasoning.
`BatchRunner` deliberately excludes samples with no recorded assistant
reasoning from its merged training trajectories.

## Custom or local endpoint

```python
agent = AIAgent(
    provider="custom",
    base_url="http://127.0.0.1:8000/v1",
    api_key="local",
    model="served-model-name",
)
```

The server must implement a compatible message and tool-call schema. The
library does not install or supervise the model server.

## Validate behavior, not only connectivity

A successful text reply does not prove a model can operate this harness. Before
production use, exercise:

- a model → tool call → tool observation → final response turn;
- reasoning capture if trajectories require it;
- the intended context length and compression path;
- cancellation and timeout behavior;
- provider usage fields if billing reports depend on them.

Unsupported synchronous transports fail at initialization rather than running
inside a hidden thread.
