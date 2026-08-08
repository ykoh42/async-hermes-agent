---
sidebar_position: 5
title: "FAQ & Troubleshooting"
description: "Common integration, async lifecycle, provider, MCP, session, and trajectory questions"
---

# FAQ & Troubleshooting

## Is this the full Hermes Agent product?

No. It is a native-async library distribution derived from upstream Hermes
Agent `v2026.8.3`. It keeps the agent loop and retained provider/tool/MCP/
skill/memory/session/trajectory surfaces. CLI/TUI, desktop/dashboard,
messaging, cron, and a bundled web service are intentionally outside scope.

## Is it API-compatible with upstream?

Public names and file locations are preserved where practical, but I/O-bearing
methods are coroutines. Existing library code normally changes from
`agent.chat(...)` to `await agent.chat(...)`, and from `agent.close()` to
`await agent.close()`. This is a divergent async distribution, not a drop-in
replacement for upstream product applications.

## Why is `AIAgent()` not awaited?

Construction performs state-only initialization. Provider, database, plugin,
and MCP work begins at `__aenter__()` or the first awaited turn.

## Do I need `arun_conversation()` or `aclose()`?

No. The existing names are the async API:

```python
result = await agent.run_conversation("Hello")
answer = await agent.chat("Hello")
await agent.close()
```

## Can I call it from FastAPI?

Yes. Create and close agents in FastAPI's lifespan and await them from route
handlers. FastAPI is not bundled, and the host must supply authentication,
session mapping, request limits, and error handling. See
[Programmatic Integration](/developer-guide/programmatic-integration).

## Why are requests on one agent serialized?

One `AIAgent` is one mutable conversation. Its turn lock protects transcript,
prompt-cache, and persistence ordering. Use different agent instances for
independent conversations.

## Does async automatically make every request faster?

No. It allows unrelated I/O-bound work to make progress while another request
waits. Provider latency, model generation, CPU-heavy work, external quotas, and
tool limits still determine performance.

## Which providers work?

The retained provider registry includes OpenRouter, OpenAI-compatible custom
endpoints, Anthropic, Gemini, Vertex, Bedrock, Azure Foundry, and other upstream
provider plugins. Some require an optional dependency extra. See
[Providers](/integrations/providers) and the provider's plugin manifest.

## A provider says the key is missing

Check, in order:

1. the explicit `api_key=` value passed by the host;
2. the provider-specific variable in the process environment;
3. `$HERMES_HOME/.env`;
4. that `provider`, `base_url`, and model ID refer to the same provider.

Do not send `OPENAI_API_KEY` to an unrelated custom endpoint. See
[Environment Variables](./environment-variables.md).

## I received a rate-limit error

Rate limits belong to the selected provider/account. The library retains
provider retry and credential-pool behavior, but it does not invent a local
OpenRouter-specific quota. Reduce concurrency, wait for the provider window,
or use an account/model with suitable limits.

## MCP tools do not appear

Verify that:

- the entry is under `mcp_servers` in the active `HERMES_HOME/config.yaml`;
- `enabled` is not false;
- the stdio command exists in the subprocess `PATH`, or the URL is reachable;
- include/exclude filters admit the tool;
- the agent reached an awaited initialization boundary;
- the agent remains open.

See [MCP Configuration](./mcp-config-reference.md).

## Why does an interactive tool fail in a service?

`clarify` and approval paths need host callbacks. Register a callback or omit
interactive capabilities from a headless agent. The library intentionally does
not fall back to an invisible terminal prompt.

## Where are sessions stored?

By default, `$HERMES_HOME/state.db`, normally `~/.hermes/state.db`. For tests or
multi-tenant services, use a separate `HERMES_HOME` or injected `SessionDB`
policy per isolation boundary. See
[Session Storage](/developer-guide/session-storage).

## Why was a batch sample discarded?

The batch path discards samples with no reasoning across all assistant turns.
During merge it also filters invalid JSON and records containing unknown tool
names. A retained record still needs consumer-defined quality and safety
review before training.

## Does this package train a model?

No. It generates trajectories and batch metadata. Fine-tuning, RL, dataset
curation, evaluation, and checkpoint management for a model trainer are
external responsibilities.

## The event loop appears blocked

Reproduce with the runtime blocking tests and identify the exact active path.
Do not hide a synchronous SDK behind `asyncio.to_thread()`. Convert the I/O
boundary to a native async client or fail explicitly when that capability is
selected.

## What should I include in a bug report?

Include the package version, Python version, platform, provider and API mode,
minimal awaited example, exception traceback, and whether the issue reproduces
with a temporary `HERMES_HOME`. Remove tokens, prompts, memory, and user data.
