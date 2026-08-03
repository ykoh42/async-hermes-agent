"""Hermes lifecycle dispatch for plugins."""

from __future__ import annotations

import logging
from typing import Any, List

logger = logging.getLogger(__name__)


async def invoke_hook(hook_name: str, **kwargs: Any) -> List[Any]:
    """Invoke native-async plugin hooks from the agent runtime."""
    from hermes_cli import plugins

    return await plugins.invoke_hook(hook_name, **kwargs)


def has_hook(hook_name: str) -> bool:
    """Return whether a plugin consumes a hook."""
    from hermes_cli import plugins

    return plugins.has_hook(hook_name)
