"""Shared native-async FAL.ai SDK plumbing."""

from __future__ import annotations

import asyncio
from typing import Any


def import_fal_client() -> Any:
    """Import and return the optional ``fal_client`` SDK."""
    import fal_client  # type: ignore[import-not-found]

    return fal_client


async def _create_fal_client(fal_client: Any) -> Any:
    """Create FAL's async SDK client after awaited HTTP transport setup."""
    client = fal_client.AsyncClient()
    state = getattr(client, "__dict__", None)
    sdk_module = getattr(fal_client, "client", None)
    user_agent = getattr(sdk_module, "USER_AGENT", None)
    if not isinstance(state, dict) or not isinstance(user_agent, str):
        raise RuntimeError(
            "The installed fal-client runtime does not expose its async HTTP "
            "client defaults. Reinstall async-hermes-agent with the fal extra."
        )
    if "_client" in state:
        raise RuntimeError(
            "The installed fal-client runtime eagerly created an HTTP client. "
            "Reinstall async-hermes-agent with the fal extra."
        )

    auth = client._get_auth()
    from agent.ssl_verify import _create_httpx_client

    state["_client"] = await _create_httpx_client(
        headers={
            "Authorization": auth.header_value,
            "User-Agent": user_agent,
        },
        timeout=client.default_timeout,
    )
    return client


async def _close_fal_client(client: Any) -> None:
    """Close the FAL HTTP client fully before preserving caller cancellation."""
    close_task = asyncio.create_task(
        client._client.aclose(),
        name="fal-http-client-close",
    )
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            await asyncio.shield(close_task)
            break
        except asyncio.CancelledError as exc:  # noqa: ASYNC103
            if close_task.cancelled():
                if cancellation is None:
                    raise
                break
            if cancellation is None:
                cancellation = exc
            continue  # noqa: ASYNC104 - finish closing the owned client
        except Exception as exc:
            if cancellation is None:
                raise
            raise cancellation from exc  # noqa: ASYNC104
    if cancellation is not None:
        raise cancellation  # noqa: ASYNC104
