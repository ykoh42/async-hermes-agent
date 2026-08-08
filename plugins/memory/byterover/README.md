# ByteRover Memory Provider

Persistent memory through the `brv` CLI, using a hierarchical knowledge tree
with tiered retrieval.

## Requirements

Install the ByteRover CLI:

```bash
curl -fsSL https://byterover.dev/install.sh | sh
# or
npm install -g byterover-cli
```

## Configuration

Enable the `memory` toolset when constructing `AIAgent` and select ByteRover in
`$HERMES_HOME/config.yaml`:

```yaml
memory:
  provider: byterover
  byterover:
    auto_extract: false
```

`HERMES_HOME` defaults to `~/.hermes`. The provider stores its profile-scoped
working tree under `$HERMES_HOME/byterover/`.

For optional cloud synchronization, put the secret in
`$HERMES_HOME/.env` or the process environment:

```dotenv
BRV_API_KEY=your-key
```

## Tools

| Tool | Description |
|------|-------------|
| `brv_query` | Search the knowledge tree |
| `brv_curate` | Store facts, decisions, and patterns |
| `brv_status` | Report CLI version, tree statistics, and sync state |

The provider resolves and invokes the `brv` subprocess asynchronously.
