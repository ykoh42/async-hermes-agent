# Async Hermes Agent — Development Guide

This repository is a native-async, library-focused distribution of
[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent). It
started from upstream `v2026.8.3` (`v0.20.0`). Keep the upstream agent's
behavior while exposing its retained runtime through native coroutine APIs.

## Scope

The retained product surface is:

- the `AIAgent` conversation loop and model-provider transports;
- tool execution, including terminal, files, skills, memory, and MCP;
- sessions, context compression, checkpoints, and trajectories;
- the single-runner and `BatchRunner` data-generation paths; and
- reusable provider, plugin, and configuration support required by those
  paths.

This package does not ship the interactive Hermes CLI/TUI, messaging bridge,
scheduler, dashboard, desktop application, or FastAPI application. It does
retain the upstream `batch_runner.py` script/module entrypoint for dataset
generation. A downstream application owns its user interface, authentication,
routing, and service lifecycle.

Do not document or implement a removed upstream surface before its executable
code is deliberately restored. When restoring one, bring back its tests and
documentation in the same change.

## Public API

Keep existing module paths, public names, arguments, and return shapes wherever
the async conversion permits it. Callers should normally migrate by adding
`await`:

```python
from run_agent import AIAgent

async with AIAgent(...) as agent:
    result = await agent.run_conversation("Question")
    answer = await agent.chat("Follow-up")
```

`AIAgent.__init__()` performs state-only initialization. Network, database, and
MCP setup belongs at an awaited lazy boundary. `run_conversation()`, `chat()`,
`close()`, and `BatchRunner.run()` remain async under their upstream names; do
not add `arun_*` aliases or synchronous wrappers.

## Async invariants

- Do not call `asyncio.to_thread()`, `run_in_executor()`,
  `run_until_complete()`, or blocking `.result()` in the retained runtime.
- Do not keep duplicate synchronous implementations for compatibility.
- Await native async provider, MCP, subprocess, database, and file APIs.
- Keep CPU-only pure transformations synchronous.
- Serialize concurrent turns on one `AIAgent`; allow separate instances to run
  concurrently.
- Preserve cancellation. Safely persist partial session and trajectory state,
  then re-raise external `CancelledError`.
- Close owned clients, subprocesses, and child tasks deterministically.
- A provider or tool without a native async implementation must fail clearly;
  it must not silently fall back to a worker thread.

## Behavioral invariants

- The system-prompt prefix must remain byte-stable within a conversation.
- Preserve strict message-role alternation and tool-call/tool-result ordering.
- Preserve reasoning, tool call, observation, and final-answer ordering in
  trajectories.
- Parallel tool execution may change scheduling, never the order in which
  results are appended to model context.
- Preserve session resume, checkpoint, budget, guardrail, interrupt, and
  compression behavior.
- Capability belongs at the edges. Prefer an existing tool, skill, plugin, or
  MCP server over expanding the core model-tool schema.

## Repository map

```text
run_agent.py                 AIAgent public API and orchestration
agent/                       conversation, transport, memory, and trajectory internals
model_tools.py               tool dispatch
tools/                       retained tool implementations and MCP client
hermes_state.py              async SQLite-backed session state
hermes_state_portability.py  async session export and import
hermes_state_schema.py       async schema migrations and FTS DDL
mini_swe_runner.py           single-task trajectory runner
batch_runner.py              bounded concurrent dataset runner
trajectory_compressor.py     trajectory post-processing
providers/                   provider profiles and registry
plugins/                     retained provider/tool/memory plugins
hermes_cli/                  reusable config/auth/provider helpers, not a CLI entry point
gateway/                     shared session/platform types, not a messaging gateway
tests/                       unit, integration, E2E, async, and parity coverage
```

The reduced `hermes_cli/` and `gateway/` packages retain helpers imported by
the library. Their names are historical; their presence does not mean the CLI
or messaging product is shipped.

## Development

Python 3.11 through 3.13 is supported.

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check .
uv build
```

CI additionally runs `flake8-async` against the retained runtime. Add focused
tests for changes to cancellation, concurrency, provider dispatch, sessions,
MCP, skills, memory, or trajectory serialization. Prefer real temporary files
and SQLite databases over mocks for I/O integration paths.

## Change discipline

- Make surgical changes in the existing file and function whenever practical.
- Do not move files or rename upstream symbols solely for style.
- Do not add pass-through wrappers, speculative hooks, compatibility shims, or
  an application framework.
- Add a dependency only when the standard library and current dependencies
  cannot provide a native async implementation. Keep provider-specific
  dependencies in extras.
- Preserve unrelated user changes in a dirty worktree.
- Use `rg`/`rg --files` for discovery and `apply_patch` for manual edits.

## Upstream migrations

Treat upstream releases as behavior changes to review, not files to copy over
blindly:

1. Diff the new upstream release against `v2026.8.3` and classify each change
   by retained or removed surface.
2. Port retained behavior into the same paths while preserving the async
   invariants above.
3. Add or update behavior-level parity tests before accepting the port.
4. Restore a previously removed surface only through an explicit scoped
   change, together with its dependencies, tests, and current documentation.
5. Keep upstream attribution and record the new upstream baseline in the
   README when the migration is complete.

## Release versioning

The Python package uses four numeric release segments:
`<upstream-major>.<upstream-minor>.<upstream-patch>.<async-revision>`. Keep the
first three segments equal to the upstream Python package version recorded in
`[tool.async-hermes]` in `pyproject.toml`. Increment only the fourth segment
for fork-only releases; after porting a new upstream version, update the first
three segments and reset the async revision to `1`. The private root npm
metadata represents the same release as `<upstream-version>-async.<revision>`.

Release tags exactly match the Python version with a `v` prefix. Never move or
reuse an existing tag.

Git history and the upstream tag are the source of removed implementation and
documentation. Stale descriptions must not be kept in the live documentation
as a substitute for history.
