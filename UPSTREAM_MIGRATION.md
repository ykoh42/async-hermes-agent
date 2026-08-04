# Upstream migration ledger

This branch starts from Hermes Agent `v2026.8.3` (`3c27eb623`) and preserves
the public module, function, argument, and return-value layout of upstream.
The product-only surfaces removed by the lean distribution are tracked
separately from native-async changes so future upgrades can review the async
delta without deletion noise.

## Runtime scope

Retained runtime behavior:

- core conversation, compression, provider, and tool loops;
- session, memory, trajectory, checkpoint, runner, and batch runner storage;
- terminal, file, memory, skill, MCP, clarify, todo, and delegation tools;
- provider discovery and the model-provider and web-provider plugin edges.

Excluded product surfaces:

- classic CLI, TUI, desktop, dashboard, and web UI;
- messaging gateways, cron, ACP, A2A, voice, and outbound webhooks;
- bundled optional skills/MCPs and unrelated product documentation.

Tests for retained code remain in-tree. Tests whose sole production target was
removed are removed with that target.

## Porting rules

1. Preserve upstream file and public symbol names.
2. Convert the upstream implementation in place; do not add sync wrappers or
   parallel `*_async` APIs.
3. Keep CPU-only helpers synchronous.
4. Do not use `asyncio.to_thread()`, `run_until_complete()`, or blocking
   `.result()` in active library paths.
5. Preserve system-prompt bytes, message alternation, tool ordering, and
   trajectory ordering.
6. For each upstream behavior, retain or adapt its upstream test and add an
   async liveness/cancellation assertion when I/O is involved.

## Change contracts

| Upstream change | Decision | Async translation | Evidence |
| --- | --- | --- | --- |
| `a1ff62a13` batch durability | ported | await flush and `fsync` before checkpoint publication | durability, cancellation, and resume tests |
| `3b9cf56af` reasoning-details tail budget | retained | CPU-only accounting stays synchronous | upstream tail-budget tests |
| `c2088efe9` compression timeout | ported in scope | await core timeout/cancellation; omit removed gateway watchdog | compression timeout, heartbeat, and cancellation tests |
| `33a2f29a6` provider transport fallback | retained | resolution stays synchronous; selected transports are awaited | provider resolution and failover tests |
| `7d066c3c5` system-prompt dedup | ported | content-addressed prompts use native `aiosqlite` transactions | dedup, compression-child, and real SQLite resume tests |
| `48e825456` approval read-only config | ported | await the async config boundary | approval read-only and deny-rule tests |
| `b1711c6f2` terminal recovery guidance | ported | annotations remain CPU-only; command execution and spill I/O are awaited | terminal hint, CWD, timeout, and spill tests |
| `89f920901` MCP whitespace warning | retained | validation remains CPU-only before native async connect | whitespace and MCP lifecycle tests |
| `2a3a7e6f5` skill-view dedup | ported | task-local async-safe cache | skill dedup, traversal, profile, and concurrent-agent tests |
| `#56832` lazy MCP startup | ported | async schema-cache I/O and serialized first-use connect; no background loop bridge | cache, concurrent first-use, stale-tool, and shutdown tests |

The table is updated in the same commit that completes each behavior contract.

## Verification baseline

The completed `v2026.8.3` migration was checked with:

- the retained test suite: 6,815 passed across 700 test files;
- full Ruff and async-blocking-call lint passes;
- a clean lockfile, sdist/wheel build, fresh-wheel import, and coroutine API checks;
- a live OpenRouter `openai/gpt-oss-20b:free` turn covering model response,
  terminal tool dispatch, reasoning capture, trajectory persistence, and an
  event-loop heartbeat during both network calls.
