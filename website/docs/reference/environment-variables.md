---
sidebar_position: 1
title: "Environment Variables"
description: "State location and credential environment variables used by the library"
---

# Environment Variables

Async Hermes Agent uses environment variables primarily for credentials and
secret-adjacent paths. Behavioral settings belong in
`$HERMES_HOME/config.yaml` or explicit `AIAgent` constructor arguments.

The library also loads secrets from `$HERMES_HOME/.env`. Do not commit that
file.

## Runtime location

| Variable | Purpose | Default |
| --- | --- | --- |
| `HERMES_HOME` | Root for `config.yaml`, `.env`, `state.db`, memory, skills, MCP tokens, logs, and caches | `~/.hermes` |

For tests and isolated services, set `HERMES_HOME` to a dedicated directory
before constructing an agent. Do not repurpose the operating system's `HOME`
variable as Hermes state.

## Common model-provider credentials

Pass `api_key=` explicitly when that is clearer for your host. The retained
provider registry also recognizes these common environment variables:

| Provider | Variables |
| --- | --- |
| OpenRouter | `OPENROUTER_API_KEY` |
| OpenAI-compatible OpenAI endpoint | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY`, `ANTHROPIC_TOKEN`, or `CLAUDE_CODE_OAUTH_TOKEN` according to the selected auth path |
| Google AI Studio | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| Nous | `NOUS_API_KEY` |
| Fireworks | `FIREWORKS_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| DeepInfra | `DEEPINFRA_API_KEY` |
| NVIDIA | `NVIDIA_API_KEY` |
| xAI | `XAI_API_KEY` |
| Hugging Face | `HF_TOKEN` |
| Azure Foundry | `AZURE_FOUNDRY_API_KEY`, with `AZURE_FOUNDRY_BASE_URL` |
| Gemini on Vertex | Service-account/ADC configuration rather than a static provider key |
| AWS Bedrock | Standard AWS SDK credential chain |

Additional retained provider plugins declare their accepted variables in
`plugins/model-providers/<provider>/__init__.py`. That declaration is the
canonical source when adding or auditing a provider.

## Tool and capability credentials

Only configure credentials for capabilities you enable. Common examples are:

| Capability | Variables |
| --- | --- |
| Web providers | `EXA_API_KEY`, `PARALLEL_API_KEY`, `FIRECRAWL_API_KEY`, `TAVILY_API_KEY`, or the selected provider's key |
| Image/video via FAL | `FAL_KEY` |
| Image generation via Krea | `KREA_API_KEY` |
| OpenAI image generation | `OPENAI_API_KEY` |
| OpenRouter media | `OPENROUTER_API_KEY` |
| Mem0 Platform | `MEM0_API_KEY` |
| ByteRover | `BRV_API_KEY` |

Capability plugin manifests under `plugins/` declare required variables. A
missing optional credential should disable or fail the selected capability; it
should not be replaced by an unrelated provider key.

## MCP secrets

MCP configuration can interpolate process or `.env` secrets:

```yaml
mcp_servers:
  github:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "${GITHUB_TOKEN}"
```

`${env:GITHUB_TOKEN}` is also accepted. Prefer references over committing a
literal token to `config.yaml`. See [MCP Configuration](./mcp-config-reference.md).

## Base URLs and other behavior

Some provider plugins retain a provider-specific `*_BASE_URL` for established
compatibility. For new host code, prefer the explicit `base_url=` argument or
the documented `config.yaml` provider section. Timeouts, tool selection,
reasoning policy, browser behavior, and concurrency are not secrets and should
not be introduced as new environment-variable-only configuration.

## Secret handling

- Keep `.env` permissions restricted and out of version control.
- Never place secrets in prompts, trajectories, or checked-in datasets.
- Do not forward the whole host environment to tools or MCP subprocesses.
- Use an isolated `HERMES_HOME` per tenant or security boundary.
- Rotate a credential if it appears in logs or an exported trajectory.

See [Configuration](/user-guide/configuration) and
[Providers](/integrations/providers) for the non-secret side of setup.
