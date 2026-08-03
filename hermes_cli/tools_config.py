"""Small config helpers shared by the async core and batch runner.

Interactive tool installation and platform-picker code is deliberately not
part of this library.  The retained functions are pure and keep their upstream
module path so core imports remain stable across Hermes releases.
"""

from __future__ import annotations

from typing import Any


def _parse_enabled_flag(value: Any, default: bool = True) -> bool:
    """Parse bool-like config values used by MCP settings."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return default


def enabled_mcp_server_names(config: dict[str, Any]) -> set[str]:
    """Return MCP server names not explicitly disabled in *config*."""
    servers = (config or {}).get("mcp_servers") or {}
    return {
        str(name)
        for name, server_config in servers.items()
        if isinstance(server_config, dict)
        and _parse_enabled_flag(server_config.get("enabled", True))
    }
