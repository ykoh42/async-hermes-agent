"""Firecrawl web search + extract plugin — bundled, auto-loaded.

Largest single plugin in this PR. Captures everything the previous
inline implementation in tools/web_tools.py did:

  - Direct cloud and self-hosted HTTP paths configured through
    FIRECRAWL_API_KEY / FIRECRAWL_API_URL.
  - Per-URL scrape loop with 60s timeout, SSRF re-check after redirect,
    website-policy gating, and format-aware content selection.
  - Robust response shape normalization for Firecrawl API variants.
"""

from __future__ import annotations

from plugins.web.firecrawl.provider import FirecrawlWebSearchProvider


def register(ctx) -> None:
    """Register the Firecrawl provider with the plugin context."""
    ctx.register_web_search_provider(FirecrawlWebSearchProvider())
