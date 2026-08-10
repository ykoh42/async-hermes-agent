# RetainDB Memory Provider

Cloud memory API with hybrid search (Vector + BM25 + Reranking) and 7 memory
types. The provider uses the package's native async HTTP and SQLite runtime; no
provider-specific dependency is required.

## Setup

Select `retaindb` as `memory.provider` in the application configuration, expose
`RETAINDB_API_KEY` to the active profile, and optionally set the non-secret
provider configuration under `memory.retaindb`:

```yaml
memory:
  provider: retaindb
  retaindb:
    base_url: https://api.retaindb.com
    project: default
```

## Config

| Variable | Default | Description |
|----------|---------|-------------|
| `RETAINDB_API_KEY` | required | API key |
| `RETAINDB_BASE_URL` | `https://api.retaindb.com` | API endpoint; overrides config |
| `RETAINDB_PROJECT` | profile-scoped default | Project identifier; overrides config |

## Tools

| Tool | Description |
|------|-------------|
| `retaindb_profile` | User's stable profile |
| `retaindb_search` | Semantic search |
| `retaindb_context` | Task-relevant context |
| `retaindb_remember` | Store a fact with type and importance |
| `retaindb_forget` | Delete a memory by ID |
| `retaindb_upload_file` | Upload a file to the shared store |
| `retaindb_list_files` | List shared files |
| `retaindb_read_file` | Read stored text content |
| `retaindb_ingest_file` | Extract a stored file into memory |
| `retaindb_delete_file` | Delete a stored file |
