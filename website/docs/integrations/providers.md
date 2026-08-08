---
title: Model Providers
description: Select a native-async model transport and install only its required dependencies.
sidebar_position: 2
---

# Model Providers

`AIAgent` separates a provider profile from the model identifier. A profile
resolves credentials, base URL, API mode, and transport behavior; `model` names
the model or deployment exposed by that provider.

## Prefer explicit construction

Explicit arguments are easiest to audit in services and tests:

```python
import os

from run_agent import AIAgent

agent = AIAgent(
    provider=os.environ["MODEL_PROVIDER"],
    base_url=os.getenv("MODEL_BASE_URL") or None,
    api_key=os.getenv("MODEL_API_KEY") or None,
    model=os.environ["MODEL_ID"],
)
```

`MODEL_PROVIDER`, `MODEL_ID`, `MODEL_API_KEY`, and `MODEL_BASE_URL` in this
example belong to the host application. They let one example cover every
provider without presenting one vendor as the default.

Do not assume that a provider's current free models, aliases, context windows,
or prices are stable. Supply a model that supports the tool-calling and
reasoning behavior required by your application.

## Bundled provider profiles

The provider registry ships the following profiles. Credential names come
from the profile source; OAuth and cloud-identity routes may not use a static
API-key variable.

| Profile | Credential or identity | Transport family |
| --- | --- | --- |
| `anthropic` | `ANTHROPIC_API_KEY`, `ANTHROPIC_TOKEN`, or Claude OAuth | Native Anthropic |
| `gemini` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Native Gemini HTTP |
| `vertex` | Google Application Default Credentials | Google Vertex |
| `bedrock` | AWS SDK credential chain | AWS Bedrock Converse |
| `azure-foundry` | `AZURE_FOUNDRY_API_KEY` and `AZURE_FOUNDRY_BASE_URL`, or Entra ID | OpenAI-compatible / Azure identity |
| `openai-codex` | ChatGPT/Codex OAuth state | Codex Responses |
| `copilot` | `COPILOT_GITHUB_TOKEN`, `GH_TOKEN`, or `GITHUB_TOKEN` | GitHub Copilot |
| `nous` | `NOUS_API_KEY` or Nous OAuth state | OpenAI-compatible |
| `openrouter` | `OPENROUTER_API_KEY` | OpenAI-compatible |
| `deepseek` | `DEEPSEEK_API_KEY` | OpenAI-compatible |
| `xai` | `XAI_API_KEY` or xAI OAuth state | OpenAI-compatible / Responses |
| `zai` | `GLM_API_KEY`, `ZAI_API_KEY`, or `Z_AI_API_KEY` | OpenAI-compatible |
| `kimi-coding` | `KIMI_API_KEY` or `KIMI_CODING_API_KEY` | OpenAI-compatible |
| `minimax` | `MINIMAX_API_KEY`; OAuth uses `minimax-oauth` | Anthropic-compatible |
| `alibaba` | `DASHSCOPE_API_KEY` | OpenAI-compatible |
| `huggingface` | `HF_TOKEN` | OpenAI-compatible |
| `fireworks` | `FIREWORKS_API_KEY` | OpenAI-compatible |
| `nvidia` | `NVIDIA_API_KEY` | OpenAI-compatible |
| `custom` | Host-defined key and base URL | OpenAI-compatible custom/local |

Additional bundled profiles include AI Gateway, Arcee, DeepInfra, GMI,
KiloCode, Novita, Ollama Cloud, OpenCode, Qwen OAuth, StepFun, Upstage, and
Xiaomi. The source of truth is `plugins/model-providers/`; this page groups
routes by contract instead of ranking vendors.

## Retained transport families

The retained runtime has native async paths for these families:

| Family | Typical profiles | Dependency |
| --- | --- | --- |
| OpenAI-compatible chat/responses | OpenRouter, custom/local endpoints, DeepSeek, xAI and other compatible gateways | Base install |
| Native Anthropic | `anthropic` | `anthropic` extra |
| Google Gemini HTTP | `gemini` | Base install |
| Google Vertex | `vertex` | `vertex` extra for credentials |
| Microsoft Foundry/Azure | `azure-foundry` | `azure-identity` extra for Entra ID |
| AWS Bedrock | `bedrock` | `bedrock` extra |
| Codex Responses and Copilot ACP | Corresponding bundled profiles | Profile-specific credentials/runtime |

Bundled profile discovery includes additional OpenAI-compatible services. A
profile in the source tree means Hermes knows how to resolve that service; it
does not guarantee that an external account, endpoint, model, or optional SDK is
currently available.

## Install an optional transport

From a source checkout:

```bash
uv sync --extra anthropic
uv sync --extra vertex
uv sync --extra azure-identity
uv sync --extra bedrock
```

See [Installation](../getting-started/installation.md) for the complete extras
list.

## Custom OpenAI-compatible endpoint

```python
agent = AIAgent(
    provider="custom",
    base_url="http://127.0.0.1:8000/v1",
    api_key="local-or-required-key",
    model="your-served-model",
)
```

The endpoint must implement the selected OpenAI-compatible API and support the
message/tool schema used by your workload. Running a local model server is
outside this package.

## Configuration-based selection

For applications that prefer file configuration, use non-secret settings in
`$HERMES_HOME/config.yaml`:

```yaml
model:
  provider: "<provider-profile>"
  default: "<model-id>"
  # base_url: "<custom-or-overridden-endpoint>"
```

Keep the credential in `$HERMES_HOME/.env` or the process environment:

```dotenv
<PROVIDER_CREDENTIAL_VARIABLE>=...
```

Explicit constructor arguments take priority for that agent instance. Details
are in [Configuring models](../user-guide/configuring-models.md).

## Failure behavior

Provider setup and requests are awaited. If a selected API mode has no native
async transport, initialization fails explicitly; the runtime does not call a
synchronous SDK through `asyncio.to_thread()`.

Always close an initialized provider with `await agent.close()` or an async
context manager.
