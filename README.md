# Async Hermes Agent

Native-async, library-focused distribution of
[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent), based
on the upstream `v2026.7.30` (`v0.19.1`) release.

This repository keeps the Hermes agent loop, model providers, tool execution,
MCP, skills, persistent memory and sessions, trajectory generation, runner, and
batch runner. The CLI/TUI, messaging bridges, scheduler, dashboard, and FastAPI
application are intentionally outside this package.

The public core API keeps the upstream names and module locations. Existing
library integrations normally only need to add `await`:

```python
from run_agent import AIAgent

async with AIAgent(
    provider="openrouter",
    model="openrouter/auto",
    api_key="...",
) as agent:
    result = await agent.run_conversation("Investigate this repository")
    print(result["final_response"])
```

The compact string-returning interface is async as well:

```python
answer = await agent.chat("Summarize the result")
await agent.close()
```

`AIAgent.__init__()` performs state-only construction. Configuration, provider
clients, session storage, and MCP connections initialize lazily at the first
awaited boundary. Turns on one `AIAgent` instance are serialized; separate
instances can run concurrently.

## Install

Python 3.11 through 3.13 is supported.

```bash
uv pip install "git+https://github.com/ykoh42/async-hermes-agent.git"
```

For development:

```bash
git clone https://github.com/ykoh42/async-hermes-agent.git
cd async-hermes-agent
uv sync --extra dev
```

Provider-specific dependencies remain opt-in, for example:

```bash
uv sync --extra anthropic
uv sync --extra bedrock
uv sync --extra vertex
```

## Skills, MCP, and memory

User skills use the existing Hermes layout. Put each skill at:

```text
~/.hermes/skills/<skill-name>/SKILL.md
```

MCP servers are configured under `mcp_servers` in `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  filesystem:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
```

The file-backed memory and user profile surfaces also retain the normal Hermes
home under `~/.hermes`. Enable the `memory` toolset and the corresponding
`memory` settings in `config.yaml` when constructing a memory-enabled agent.

## Training and trajectories

Set `save_trajectories=True` on `AIAgent` for individual conversations. The
saved sequence preserves reasoning, tool calls, observations, and the final
answer for interleaved-thinking fine-tuning.

For datasets, use `BatchRunner` from the unchanged `batch_runner.py` module and
await its existing `run()` method. It retains bounded concurrency, checkpoints,
resume support, and JSONL output. `trajectory_compressor.py` remains available
for post-processing generated trajectories.

## Service integration

No web framework is bundled. A service should own its HTTP lifecycle and await
the library directly:

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from run_agent import AIAgent

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AIAgent(provider="openrouter", model="openrouter/auto") as agent:
        app.state.agent = agent
        yield

app = FastAPI(lifespan=lifespan)

@app.post("/chat")
async def chat(message: str):
    return await app.state.agent.run_conversation(message)
```

The package does not use `asyncio.to_thread()`, `run_in_executor()`,
`run_until_complete()`, or blocking `.result()` in the retained active agent
path. Optional providers without a native async transport fail fast instead of
silently running synchronous work in a thread.

## Verification

```bash
uv run pytest -q
uv run ruff check agent tools hermes_cli plugins providers \
  run_agent.py model_tools.py batch_runner.py hermes_state.py \
  trajectory_compressor.py
uv build
```

## Upstream relationship

The repository preserves original core file and function names to keep future
upstream imports reviewable. It is a divergent async distribution, not a claim
that these changes are drop-in mergeable to the synchronous upstream product.

Hermes Agent is built by [Nous Research](https://nousresearch.com). This
distribution retains the upstream MIT license; see [LICENSE](LICENSE).
