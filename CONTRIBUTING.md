# Contributing to Async Hermes Agent

Thank you for contributing. This project preserves the retained behavior of
[Hermes Agent](https://github.com/NousResearch/hermes-agent) while providing a
native-async, library-first API.

## Project scope

Changes should improve one of the retained surfaces: the agent loop, providers,
tools, MCP, skills, memory, sessions, trajectories, or batch execution. The
repository intentionally does not ship a CLI/TUI, messaging gateway, cron
scheduler, dashboard, desktop app, or web-service framework.

Do not add a removed product surface as incidental infrastructure. A deliberate
restoration must include the complete runtime path, dependencies, tests, and
documentation in one scoped proposal.

## Development setup

Python 3.11 through 3.13 is supported. Install the project and development
dependencies with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/ykoh42/async-hermes-agent.git
cd async-hermes-agent
uv sync --extra dev
```

Run the standard checks:

```bash
uv run pytest -q
uv run ruff check .
uv build
```

Provider-specific tests may require the corresponding optional dependency.
Do not commit API keys, `.env` files, generated trajectories, or test output.

## Native-async rules

- Preserve public module paths, function names, arguments, and return shapes.
- Convert the implementation in place; do not add `arun_*` aliases,
  synchronous wrappers, or duplicate sync/async implementations.
- Do not use `asyncio.to_thread()`, `run_in_executor()`,
  `run_until_complete()`, or blocking `.result()` in retained runtime code.
- Use native async transports and I/O libraries. Fail clearly when a selected
  provider or tool has no native async implementation.
- Keep CPU-only pure functions synchronous.
- Preserve cancellation and deterministically close owned resources.

## Behavior-preservation rules

- Keep the per-conversation system-prompt prefix stable.
- Preserve message-role alternation and model-generated tool-call order.
- Preserve trajectory ordering: reasoning, tool call, observation, final
  answer.
- Preserve sessions, checkpoints, budgets, interrupts, guardrails, context
  compression, and batch resume behavior.
- Tests should assert behavior and invariants rather than freeze incidental
  values.

## Code changes

Prefer a surgical edit in the existing upstream file and function. Avoid file
moves, pass-through wrappers, speculative abstractions, compatibility shims,
and unrelated cleanup. This keeps future upstream diffs reviewable.

New model-facing capability should normally be a skill, plugin, or MCP server,
not another permanent core tool. Add dependencies only when existing packages
and the standard library cannot support the required native-async behavior;
provider-specific dependencies belong in optional extras.

## Tests

Add focused coverage for every behavior change. Changes involving I/O,
configuration, sessions, MCP, skills, memory, or trajectories should exercise
the real path against temporary files or SQLite databases where practical.

Async-sensitive changes should cover, as applicable:

- event-loop responsiveness;
- cancellation and timeout cleanup;
- same-agent turn serialization and cross-agent concurrency;
- task and subprocess leaks;
- provider/tool exception propagation; and
- stable prompt, message, and trajectory ordering.

Live-provider tests are optional and must be explicitly gated by credentials.
The normal test suite must not consume paid APIs.

## Pull requests

A pull request should explain:

1. the retained behavior or bug being addressed;
2. why the change is compatible with the native-async invariants;
3. how parity with upstream behavior was verified; and
4. the exact tests and builds that were run.

Keep refactors separate from behavior changes when possible. When porting an
upstream release, identify the upstream commits or diff and include parity
tests for the ported intent.

By contributing, you agree that your contribution is licensed under the
[MIT License](LICENSE).
