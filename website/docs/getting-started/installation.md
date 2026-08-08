---
sidebar_position: 2
title: Installation
description: Install Async Hermes Agent, verify the async API, and add only the provider dependencies you need.
---

# Installation

Start here. Async Hermes Agent is distributed as a Python library; your
application supplies its own service or UI boundary.

## Requirements

- Python 3.11, 3.12, or 3.13
- Git
- A model provider credential, unless you use a local endpoint

## 1. Install the current verified release

The v0.20.4 GitHub release has a verified source tag. Install it directly
until the PyPI Trusted Publisher is configured for this repository:

```bash
uv pip install "git+https://github.com/ykoh42/async-hermes-agent.git@v0.20.4"
```

The package metadata for this release is `async-hermes-agent==0.20.4`.
PyPI publication is intentionally not presented as complete until the package
is visible at `pypi.org/project/async-hermes-agent/` and a clean install has
been verified there.

## 2. Or install from a checkout

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

## 3. Add provider-specific extras

The base installation includes the OpenAI-compatible async transport, MCP,
SQLite session storage, and the core tool runtime. Provider selection is a
separate decision: install only the extras required by the provider family you
choose.

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

## 4. Verify the async API

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
Continue with the [Quickstart](./quickstart.md). It begins with a neutral
provider-selection step, then runs the same awaited agent API for every route.
