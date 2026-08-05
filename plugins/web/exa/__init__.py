"""Exa web search + extract plugin — bundled, auto-loaded.

Search and extraction call Exa's HTTP API through a native async client.
"""

from __future__ import annotations

from plugins.web.exa.provider import ExaWebSearchProvider


def register(ctx) -> None:
    """Register the Exa provider with the plugin context."""
    ctx.register_web_search_provider(ExaWebSearchProvider())
