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

## 1. Install the current release

Install the exact reviewed package version from PyPI:

```bash
uv pip install "async-hermes-agent==0.20.1.1"
```

The release workflow publishes through OIDC Trusted Publishing. The same
verified wheel, source distribution, and `SHA256SUMS` file are attached to the
[`v0.20.1.1` GitHub Release](https://github.com/ykoh42/async-hermes-agent/releases/tag/v0.20.1.1).
To install the reviewed source snapshot instead, pin the immutable tag:

```bash
uv pip install "git+https://github.com/ykoh42/async-hermes-agent.git@v0.20.1.1"
```

## Version policy

The four numeric segments keep the fork tied to its upstream baseline.
`0.20.1.1` means upstream Python package version `0.20.1` (tag `v2026.8.13`)
 plus async-distribution revision `1`. Fork-only releases increment the final
segment; a completed upstream port changes the first three segments and resets
the final segment to `1`.

The earlier `0.20.4` GitHub release used an independent scheme. Python version
tag should be explicitly reinstalled at the new upstream-aligned version:

```bash
uv pip install --reinstall "async-hermes-agent==0.20.1.1"
```

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
| `azure-identity` | Restricted Microsoft Entra ID support; see provider limitations |
| `bedrock` | AWS Bedrock transport; the pinned SDK still has blocking bootstrap boundaries |
| `parallel-web` | Parallel web search provider |
| `fal` | FAL image/video generation |
| `edge-tts` | Edge text-to-speech backend |
| `tts-premium` | ElevenLabs text-to-speech backend |
| `mistral` | Mistral text-to-speech backend |
| `piper-tts` | Local Piper backend, isolated in a profile-scoped subprocess broker |
| `modal` | Modal execution backend |
| `daytona` | Daytona execution backend |
| `vercel` | Vercel Sandbox execution backend |
| `mem0` | Mem0 memory provider, including optional OSS dependencies |
| `supermemory` | Supermemory memory provider |
| `hindsight` | Hindsight memory provider client; embedded-local mode also needs `hindsight-all` |
| `honcho` | Honcho memory provider |
| `postgres` | Opt-in PostgreSQL SessionDB backend (SQLAlchemy Core + asyncpg; run the real integration matrix before production use) |
| `dev` | Test, lint, leak-check, and type-check dependencies |

KittenTTS 0.8.1 is published by KittenML as a GitHub release wheel rather
than an official PyPI distribution. PyPI rejects direct URL dependencies in
uploaded package metadata, so this project cannot offer it as a normal extra.
The public-index package named `kittentts` is not the compatible KittenML 0.8.1
artifact. Install the hash-verified official wheel when selecting
`tts.provider: kittentts`; the wheel declares its own `soundfile` dependency:

```bash
python -m pip install \
  'https://github.com/KittenML/KittenTTS/releases/download/0.8.1/kittentts-0.8.1-py3-none-any.whl#sha256=482a436c4f1f3192153710376e459ff3689517ebcda7c2b051e2fd4187b41851'
```

Piper is available through the package extra:

```bash
python -m pip install 'async-hermes-agent[piper-tts]'
```

The compatibility extras `exa`, `firecrawl`, `homeassistant`,
`computer-use`, `vision`, and `mcp` currently add no Python distributions:
their retained Python dependencies are already in the base install. They keep
the upstream install names valid. External runtimes are still separate; for
example, `computer-use` requires a working `cua-driver` installation.

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
