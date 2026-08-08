---
title: Browser Automation
description: Use local Chromium, an existing CDP endpoint, or retained cloud browser plugins.
sidebar_position: 5
---

# Browser Automation

The browser toolset exposes accessibility-tree navigation and native CDP-backed
operations to the model. Browser work is awaited and its session is closed with
the agent lifecycle.

## Tool surface

The retained tools include navigation, accessibility snapshots, click, type,
scroll, back, key press, image collection, screenshot vision, console access,
CDP operations, and dialog handling. Enable them with:

```python
agent = AIAgent(..., enabled_toolsets=["browser"])
```

Accessibility snapshots assign stable element references for later click and
type calls. Screenshot analysis requires a configured vision-capable model.

## Local mode

Local mode uses the external `agent-browser` command and a Chromium-family
browser. Those Node/browser components are not installed by this Python
package:

```bash
npm install -g agent-browser
agent-browser install
```

Select local mode explicitly:

```yaml
browser:
  cloud_provider: local
  headed: false
  engine: auto
```

`headed: true` opens a visible browser. Supported engine values are `auto`,
`chrome`, and `lightpanda`, subject to the installed `agent-browser` version.

## Existing CDP browser

Attach to a Chrome/Chromium-compatible debugging endpoint without a UI command:

```yaml
browser:
  cloud_provider: local
  cdp_url: "http://127.0.0.1:9222"
```

The runtime resolves the browser WebSocket endpoint and uses the retained CDP
supervisor. Protect remote CDP endpoints as credentials: control URLs can grant
complete access to browser tabs, cookies, and authenticated sessions.

## Cloud providers

The retained browser plugins are:

| Provider key | Credentials |
| --- | --- |
| `browser-use` | `BROWSER_USE_API_KEY` |
| `browserbase` | `BROWSERBASE_API_KEY`, `BROWSERBASE_PROJECT_ID` |
| `firecrawl` | `FIRECRAWL_API_KEY` |

For example:

```yaml
browser:
  cloud_provider: browserbase
```

Set credentials in the environment or `$HERMES_HOME/.env`. External account
availability, billing, anti-bot behavior, proxies, and session limits are owned
by the selected service.

## Private-network protection

Navigation guards block private, loopback, link-local, and cloud-metadata
targets by default. Keep the preferred global setting disabled:

```yaml
security:
  allow_private_urls: false
```

With a cloud provider selected, `browser.auto_local_for_private_urls` defaults
to `true`: an explicitly requested private URL can be routed to a local browser
sidecar instead of sending it to the cloud provider, while redirect-based
private-network access remains guarded. Disable hybrid routing with:

```yaml
browser:
  cloud_provider: browserbase
  auto_local_for_private_urls: false
```

Enabling private URLs widens SSRF reach and should be confined to an isolated
agent and network. See [Security](../security.md).
