---
sidebar_position: 2
title: Installation
description: Install the native-async Hermes agent library from Git or a source checkout.
---

# Installation

Async Hermes Agent is a Python library. It does not install the upstream Hermes
CLI, TUI, desktop application, messaging gateway, scheduler, or web service.

## Requirements

- Python 3.11, 3.12, or 3.13
- Git
- A model provider credential, unless you use a local endpoint

## Install the current repository

Install directly from Git with `uv`:

```bash
uv pip install "git+https://github.com/ykoh42/async-hermes-agent.git"
```

The equivalent `pip` command is:

```bash
python -m pip install "git+https://github.com/ykoh42/async-hermes-agent.git"
```

This documentation does not assume that a PyPI release exists. Pin a commit or
tag in production so deployments remain reproducible:

```bash
uv pip install "git+https://github.com/ykoh42/async-hermes-agent.git@<tag-or-commit>"
```

## Install from a checkout

```bash
git clone https://github.com/ykoh42/async-hermes-agent.git
cd async-hermes-agent
uv sync
```

For development and tests:

```bash
uv sync --extra dev
uv run pytest -q
```

## Provider-specific extras

The base installation includes the OpenAI-compatible async transport, MCP,
SQLite session storage, and the core tool runtime. Install only the extras
needed by the providers you select:

| Extra | Use |
| --- | --- |
| `anthropic` | Native Anthropic API |
| `vertex` | Google Vertex credentials and async transport support |
| `azure-identity` | Microsoft Entra ID authentication |
| `bedrock` | AWS Bedrock native async transport |
| `parallel-web` | Parallel web search provider |
| `fal` | FAL media generation provider |
| `mem0` | Mem0 memory provider |

From a checkout, for example:

```bash
uv sync --extra anthropic
uv sync --extra bedrock
```

Missing provider dependencies fail with an installation hint; the library does
not silently move synchronous provider code into a worker thread.

## Verify the API

```bash
python - <<'PY'
import inspect
from run_agent import AIAgent

assert inspect.iscoroutinefunction(AIAgent.run_conversation)
assert inspect.iscoroutinefunction(AIAgent.chat)
assert inspect.iscoroutinefunction(AIAgent.close)
print("native-async API available")
PY
```

This verifies the installed interface without making a paid model request.
Continue with the [Quickstart](./quickstart.md), then review
[configuration](../user-guide/configuration.md) and
[provider selection](../integrations/providers.md).
