# Async Hermes Agent

Native-async, library-focused distribution of
[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent), based
on upstream tag `v2026.8.16` (Python package version `0.20.2`).

This repository keeps the Hermes agent loop, model providers, tool execution,
MCP, skills, persistent memory and sessions, trajectory generation, runner, and
batch runner. The interactive Hermes CLI/TUI, messaging bridges, scheduler,
dashboard, and FastAPI application are intentionally outside this package. The
upstream-compatible `batch_runner.py` script entrypoint remains available for
dataset generation.

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
async def chat_once():
    agent = AIAgent(provider="openrouter", model="openrouter/auto")
    try:
        return await agent.chat("Summarize the result")
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
uv pip install "async-hermes-agent==0.20.2.1"
```

Versioned packages are published to PyPI through GitHub OIDC Trusted
Publishing. The same verified wheel, source distribution, and checksums are
attached to the corresponding GitHub Release.

The package version has four numeric segments: `0.20.2.1` means upstream
Python version `0.20.2` plus async-distribution revision `1`. Fork-only releases
increment the fourth segment. When a new upstream version is ported, the first
three segments change to match it and the async revision restarts at `1`.

The earlier `0.20.4` GitHub release used the old independent version scheme.
If it was installed from that Git tag, migrate explicitly once:

```bash
uv pip install --reinstall "async-hermes-agent==0.20.2.1"
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

The [installation guide](https://ykoh42.github.io/async-hermes-agent/getting-started/installation)
lists
every current extra, including retained media, execution-backend, and memory
providers.

The Hindsight extra covers cloud and local-external modes. Its
`local_embedded` mode additionally requires the upstream `hindsight-all`
runtime.

The Honcho extra pins the native-async SDK version validated by this package.
Select `memory.provider: honcho` in `config.yaml`; connection, identity,
cadence, and session settings are documented in the
[Honcho provider guide](https://github.com/ykoh42/async-hermes-agent/blob/v0.20.2.1/plugins/memory/honcho/README.md).

OpenViking uses the core native-async HTTP transport and needs no Python
extra. Server setup, provider configuration, async lifecycle, recall, and tool
behavior are documented in the
[OpenViking provider guide](https://github.com/ykoh42/async-hermes-agent/blob/v0.20.2.1/plugins/memory/openviking/README.md).

## Sessions

`SessionDB` keeps the upstream export and import names under the original
`hermes_state.py` path. SQLite reads, writes, lineage reconstruction, and
resource cleanup are awaited directly:

```python
import asyncio

from hermes_state import SessionDB


async def copy_sessions():
    source = SessionDB("state.db")
    restored = SessionDB("restored-state.db")
    try:
        exported = await source.export_all()
        return await restored.import_sessions(exported)
    finally:
        await source.close()
        await restored.close()


asyncio.run(copy_sessions())
```

`export_all()`, `export_session()`, and `import_sessions()` preserve the
upstream dictionaries and validation limits. Import restores conversation
history but deliberately clears stale live-activity fields.

For an explicit PostgreSQL backend, install the opt-in extra and inject one
worker-owned store into each agent. CI exercises this backend against real
PostgreSQL services; the local test suite skips those integration tests when
no `HERMES_POSTGRES_TEST_DSN` is configured. The import and method names stay
the same; only the backend module and DSN change:

```bash
uv sync --extra postgres
```

```python
from hermes_state_postgres import SessionDB
from run_agent import AIAgent

db = SessionDB("postgresql+psycopg://user:password@db.example/hermes")
try:
    async with AIAgent(provider="openrouter", session_db=db) as agent:
        answer = await agent.run_conversation("Question")
finally:
    await db.close()
```

The DSN selects the PostgreSQL endpoint and credentials. Pool and psycopg
runtime settings are optional and use the same names as SQLAlchemy and
psycopg in the active profile's `config.yaml`:

```yaml
database:
  postgres:
    pool_size: 5
    max_overflow: 10
    pool_timeout: 30
    pool_recycle: -1
    pool_pre_ping: true
    pool_use_lifo: false
    connect_args:
      connect_timeout: 60
      prepare_threshold: 5
      application_name: async-hermes-agent
      options: >-
        -c statement_timeout=60000
        -c lock_timeout=5000
        -c idle_in_transaction_session_timeout=600000
```

These settings are captured when the store is first initialized and are not
hot-reloaded; create a new store after changing them. A read-only store keeps
the same public constructor and enforces both the SessionDB write guard and
PostgreSQL transaction-level read-only mode:

```python
readonly_db = SessionDB(
    "postgresql+psycopg://user:password@db.example/hermes",
    read_only=True,
)
```

`read_only=True` does not choose a replica automatically; the DSN still
selects the endpoint. For a service, size the database for the possible
connection count across workers: approximately
`workers * (pool_size + max_overflow)`.

The PostgreSQL settings use psycopg/libpq names. `connect_timeout` controls
connection establishment, `prepare_threshold: null` disables prepared
statements, and server timeouts are passed through libpq's `options` string.
The old asyncpg-only names (`timeout`, `command_timeout`, `server_settings`,
and asyncpg statement-cache options) are not accepted.

The injected store is borrowed by the agent, so a FastAPI or ASGI lifespan
should create and close one store per worker and share it among that worker's
agents. SQLite remains the default. PostgreSQL uses native ranking rather than
SQLite FTS5 BM25 scores. When an explicit PostgreSQL store is injected into
`AIAgent(session_db=...)`, durable async-delegation records and
`session_search(db=...)` use that same store; the default SQLite delegation
path and independent memory/plugin databases remain unchanged.

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

The upstream single-task training runner is retained at the same
`mini_swe_runner.py` import path. Its provider, terminal execution, cleanup,
and JSONL batch methods are native coroutines; the trajectory conversion and
return shapes remain unchanged:

```python
import asyncio

from mini_swe_runner import MiniSWERunner


async def run_one_task():
    runner = MiniSWERunner(
        model="openai/gpt-oss-20b:free",
        env_type="local",
        cwd="/workspace",
    )
    return await runner.run_task("Inspect and repair the project")


result = asyncio.run(run_one_task())
```

For datasets, the upstream-compatible command-line entrypoint remains
available from a source checkout:

```bash
python batch_runner.py \
    --dataset_file=data/prompts.jsonl \
    --batch_size=10 \
    --run_name=my_first_run \
    --model=anthropic/claude-sonnet-4.6 \
    --num_workers=4

# Resume an interrupted run.
python batch_runner.py \
    --dataset_file=data/prompts.jsonl \
    --batch_size=10 \
    --run_name=my_first_run \
    --resume
```

An installed package also supports `python -m batch_runner ...`. The CLI uses
Fire only at that process boundary; an async web host does not use Fire or
create a nested event loop. For library and FastAPI use, import `BatchRunner`
and await its existing `run()` method directly. It retains bounded concurrency,
checkpoints, resume support, and JSONL output. `trajectory_compressor.py`
remains available for post-processing generated trajectories.

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
from fastapi import FastAPI
from run_agent import AIAgent

app = FastAPI()

@app.post("/chat")
async def chat(message: str):
    # One AIAgent is one mutable conversation. A real host should keep one
    # instance per conversation ID; this short-lived example isolates calls.
    async with AIAgent(provider="openrouter", model="openrouter/auto") as agent:
        return await agent.run_conversation(message)
```

Provider, network, MCP, and subprocess paths use coroutine transports, and
optional providers without one fail explicitly. The filesystem layer uses
`aiofiles`, whose regular-file operations delegate to an executor, while
`aiosqlite` serializes SQLite calls on a connection worker thread. Eliminating
those portable Python limitations is outside the package's native-async
contract: public I/O remains directly awaitable and does not block the host
event loop, but the project does not claim zero-thread, OS-native regular-file
or embedded-SQLite I/O.

## Verification

```bash
uv run pytest -q
uv run ruff check agent tools hermes_cli plugins providers \
  run_agent.py model_tools.py mini_swe_runner.py batch_runner.py hermes_state.py \
  hermes_state_portability.py \
  hermes_state_schema.py \
  trajectory_compressor.py
uv build
```

## Contributing and security

Read [CONTRIBUTING.md](https://github.com/ykoh42/async-hermes-agent/blob/v0.20.2.1/CONTRIBUTING.md)
before submitting changes and
[SECURITY.md](https://github.com/ykoh42/async-hermes-agent/blob/v0.20.2.1/SECURITY.md)
for private vulnerability reporting.

## Upstream relationship

The repository preserves original core file and function names to keep future
upstream imports reviewable. It is a divergent async distribution, not a claim
that these changes are drop-in mergeable to the synchronous upstream product.

The deliberate differences from upstream `v2026.8.16` are documented in the
[upstream differences table](https://ykoh42.github.io/async-hermes-agent/developer-guide/upstream-differences).

Hermes Agent is built by [Nous Research](https://nousresearch.com). This
distribution retains the upstream MIT license; see
[LICENSE](https://github.com/ykoh42/async-hermes-agent/blob/v0.20.2.1/LICENSE).
