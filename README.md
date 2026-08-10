# Async Hermes Agent

Native-async, library-focused distribution of
[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent), based
on the upstream `v2026.8.3` (`v0.20.0`) release.

This repository keeps the Hermes agent loop, model providers, tool execution,
MCP, skills, persistent memory and sessions, trajectory generation, runner, and
batch runner. The CLI/TUI, messaging bridges, scheduler, dashboard, and FastAPI
application are intentionally outside this package.

The public core API keeps the upstream names and module locations. Existing
library integrations normally only need to add `await`:

```python
import asyncio
import os

from run_agent import AIAgent


async def main():
    async with AIAgent(
        provider="openrouter",
        model="openrouter/auto",
        api_key=os.environ["OPENROUTER_API_KEY"],
    ) as agent:
        result = await agent.run_conversation("Investigate this repository")
        print(result["final_response"])


asyncio.run(main())
```

Inside an async function, the compact string-returning interface and explicit
lifecycle are:

```python
agent = AIAgent(provider="openrouter", model="openrouter/auto")
try:
    answer = await agent.chat("Summarize the result")
finally:
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
uv sync --extra vertex
uv sync --extra azure-identity
uv sync --extra supermemory
uv sync --extra hindsight
uv sync --extra honcho
```

The Hindsight extra covers cloud and local-external modes. Its
`local_embedded` mode additionally requires the upstream `hindsight-all`
runtime.

The Honcho extra pins the native-async SDK version validated by this package.
Select `memory.provider: honcho` in `config.yaml`; connection, identity,
cadence, and session settings are documented in the
[Honcho provider guide](plugins/memory/honcho/README.md).

## Skills, MCP, and memory

Skills follow the existing Hermes layout. `HERMES_HOME` defaults to
`~/.hermes`; put each active skill at:

```text
$HERMES_HOME/skills/<skill-name>/SKILL.md
```

Each `SKILL.md` is a normal Hermes skill document with YAML frontmatter:

```markdown
---
name: code-review
description: Review a code change before it is merged.
---

# Code review

Read the change, run its tests, and report correctness issues first.
```

Upstream Hermes seeds its source-bundled skills through the product installer.
This library does not include that installer, so Git/wheel users add skill
directories explicitly or point at shared directories in `config.yaml`:

```yaml
skills:
  external_dirs:
    - ~/.agents/skills
    - /shared/team-skills
```

The `skills_list` and `skill_view` tools discover both the local and configured
external directories. Skill content remains outside the model-tool schema until
the model selects and reads it.

MCP servers are configured under `mcp_servers` in
`$HERMES_HOME/config.yaml`:

```yaml
mcp_servers:
  filesystem:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
```

The first awaited agent boundary discovers configured servers and registers
their tools under the server's toolset. MCP subprocesses and client sessions
are closed by `await agent.close()` or the async context manager.

The file-backed memory and user profile surfaces also retain the normal Hermes
home under `~/.hermes`. Enable the `memory` toolset and the corresponding
`memory` settings in `config.yaml` when constructing a memory-enabled agent.

## Training and trajectories

Set `save_trajectories=True` on `AIAgent` for individual conversations. The
saved sequence preserves reasoning, tool calls, observations, and the final
answer for interleaved-thinking fine-tuning. Completed samples append to
`trajectory_samples.jsonl` in the process working directory.

For datasets, use `BatchRunner` from the unchanged `batch_runner.py` module and
await its existing `run()` method. It retains bounded concurrency, checkpoints,
resume support, and JSONL output. `trajectory_compressor.py` remains available
for post-processing generated trajectories.

```python
import asyncio
import os

from batch_runner import BatchRunner


async def main():
    runner = BatchRunner(
        dataset_file="prompts.jsonl",
        batch_size=8,
        run_name="tool-training",
        distribution="terminal_only",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        model="openai/gpt-oss-20b:free",
        num_workers=4,
        reasoning_config={"enabled": True, "effort": "low"},
    )
    await runner.run(resume=True)


asyncio.run(main())
```

Each input line must be JSON with a `prompt` field. Outputs are written under
`data/<run_name>/`: per-batch JSONL shards, merged `trajectories.jsonl`,
`checkpoint.json`, and `statistics.json`.

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

## Contributing and security

Read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting changes and
[SECURITY.md](SECURITY.md) for private vulnerability reporting.

## Upstream relationship

The repository preserves original core file and function names to keep future
upstream imports reviewable. It is a divergent async distribution, not a claim
that these changes are drop-in mergeable to the synchronous upstream product.

Hermes Agent is built by [Nous Research](https://nousresearch.com). This
distribution retains the upstream MIT license; see [LICENSE](LICENSE).
