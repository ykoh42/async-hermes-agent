---
title: Configuration
description: Configure the native-async library without relying on a setup CLI.
sidebar_position: 2
---

# Configuration

Async Hermes Agent has no setup wizard. Configure an agent with explicit
constructor arguments and use files for shared capability settings.

## Configuration locations

`HERMES_HOME` defaults to `~/.hermes`:

```text
~/.hermes/
├── config.yaml          # Non-secret behavior and capability settings
├── .env                 # API keys and tokens
├── skills/              # Application-installed skills
├── memories/            # File-backed memory and user profile
└── plugins/             # Application-owned plugins
```

Select another isolated home before constructing an agent:

```bash
export HERMES_HOME=/srv/my-agent-state
```

The files are read lazily at an awaited lifecycle boundary. For tests, point
`HERMES_HOME` at a temporary directory before importing or constructing runtime
objects.

## Prefer explicit model arguments

Service code is easiest to reason about when model routing is explicit:

```python
agent = AIAgent(
    provider=os.environ["MODEL_PROVIDER"],
    base_url=os.getenv("MODEL_BASE_URL") or None,
    api_key=os.getenv("MODEL_API_KEY") or None,
    model=os.environ["MODEL_ID"],
    enabled_toolsets=["file", "terminal", "skills"],
)
```

`enabled_toolsets` and `disabled_toolsets` are per-agent controls. They are the
recommended replacement for upstream interface-specific toolset configuration.

## Keep secrets separate

Use the process environment or `$HERMES_HOME/.env` for credentials:

```dotenv
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
DEEPSEEK_API_KEY=...
TEAM_MCP_TOKEN=...
```

Use `config.yaml` for non-secret settings. Environment references are supported
where a nested configuration needs a secret:

```yaml
mcp_servers:
  team:
    url: "https://mcp.example.com/mcp"
    headers:
      Authorization: "Bearer ${env:TEAM_MCP_TOKEN}"
```

Do not commit `.env` or place raw credentials in examples, logs, trajectories,
or model prompts.

## Minimal capability configuration

Only include sections your application uses:

```yaml
model:
  provider: "<provider-profile>"
  default: "<model-id>"
  # base_url: "<custom-or-overridden-endpoint>"

memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200
  user_char_limit: 1375

skills:
  external_dirs:
    - /srv/team-skills

browser:
  cloud_provider: local

security:
  allow_private_urls: false
```

MCP server definitions are documented in
[Use MCP with Hermes](../guides/use-mcp-with-hermes.md). Model settings are
covered in [Configuring models](./configuring-models.md).

## Configuration lifetime

The system-prompt cached prefix and base tool selection are intentionally stable
for a conversation to preserve prompt caching. MCP- and plugin-derived tool
snapshots may refresh at a turn boundary, but remain fixed during that turn. Do
not mutate other configuration and expect an active conversation to rebuild its
past context. Apply those changes by creating a new agent lifecycle unless a
documented capability explicitly supports refresh.

The upstream `cli-config.yaml.example` contains settings for product surfaces
that are not shipped by this library. It is not a canonical library
configuration reference.
