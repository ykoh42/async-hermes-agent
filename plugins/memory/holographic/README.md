# Holographic Memory Provider

Local SQLite fact store with FTS5 search, trust scoring, entity resolution, and HRR-based compositional retrieval.

## Requirements

None — uses SQLite (always available). NumPy optional for HRR algebra.

## Configuration

Enable the `memory` toolset when constructing `AIAgent` and select
Holographic memory in `$HERMES_HOME/config.yaml`:

```yaml
memory:
  provider: holographic

plugins:
  hermes-memory-store:
    auto_extract: false
```

`HERMES_HOME` defaults to `~/.hermes`. Database setup and access run at the
awaited provider lifecycle boundary; no synchronous wrapper is required.

## Config

Config in `$HERMES_HOME/config.yaml` under `plugins.hermes-memory-store`:

| Key | Default | Description |
|-----|---------|-------------|
| `db_path` | `$HERMES_HOME/memory_store.db` | SQLite database path |
| `auto_extract` | `false` | Auto-extract facts at session end |
| `default_trust` | `0.5` | Default trust score for new facts |
| `hrr_dim` | `1024` | HRR vector dimensions |

## Tools

| Tool | Description |
|------|-------------|
| `fact_store` | 9 actions: add, search, probe, related, reason, contradict, update, remove, list |
| `fact_feedback` | Rate facts as helpful/unhelpful (trains trust scores) |
