# Mem0 Memory Provider

Mem0-backed fact extraction and semantic retrieval for the retained async
memory-provider interface. It supports Mem0 Platform, a self-hosted Mem0 HTTP
server, and the in-process Mem0 OSS SDK.

## Select the provider

Enable the `memory` toolset when constructing `AIAgent` and select Mem0 in
`$HERMES_HOME/config.yaml`:

```yaml
memory:
  provider: mem0
```

`HERMES_HOME` defaults to `~/.hermes`.

## Platform mode

Put the secret in `$HERMES_HOME/.env` or the process environment:

```dotenv
MEM0_API_KEY=your-key
```

Optional behavioral settings belong in `$HERMES_HOME/mem0.json`:

```json
{
  "mode": "platform",
  "user_id": "user-123",
  "agent_id": "hermes",
  "rerank": false
}
```

## Self-hosted HTTP mode

Point the plugin at a running Mem0 HTTP server:

```json
{
  "host": "http://localhost:8888",
  "user_id": "user-123",
  "agent_id": "hermes"
}
```

Set `MEM0_API_KEY` to the server's API key when authentication is enabled.
Do not combine `host` with `mode: "oss"`; OSS mode takes precedence.

## OSS mode

The in-process SDK requires the optional dependency:

```bash
uv sync --extra mem0
```

Configure its LLM, embedder, and vector store in
`$HERMES_HOME/mem0.json`. For example:

```json
{
  "mode": "oss",
  "user_id": "user-123",
  "agent_id": "hermes",
  "oss": {
    "llm": {
      "provider": "openai",
      "config": {"model": "gpt-5-mini"}
    },
    "embedder": {
      "provider": "openai",
      "config": {"model": "text-embedding-3-small"}
    },
    "vector_store": {
      "provider": "qdrant",
      "config": {"path": "~/.hermes/mem0_qdrant"}
    }
  }
}
```

Provider credentials referenced by the OSS configuration remain secrets and
belong in the environment or `$HERMES_HOME/.env`.

## Configuration fields

| Key | Default | Description |
|-----|---------|-------------|
| `mode` | `platform` | `platform` or `oss` |
| `host` | empty | Self-hosted Mem0 HTTP base URL |
| `user_id` | session identity | Stable user identifier |
| `agent_id` | `hermes` | Stable agent identifier |
| `rerank` | `false` | Enable reranking in supported backends |
| `oss` | `{}` | Mem0 OSS backend configuration |

The plugin exposes `mem0_search`, `mem0_add`, `mem0_update`, and
`mem0_delete` through the memory-provider tool surface. Configuration is loaded
asynchronously when the provider initializes.
