"""Shared native-async FAL.ai SDK plumbing."""

from __future__ import annotations

import asyncio
import sys
from typing import Any
from urllib.parse import urlencode


def import_fal_client() -> Any:
    """Return the preloaded SDK without cold-importing it on an event loop."""
    loaded = sys.modules.get("fal_client")
    if loaded is not None:
        return loaded
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        import fal_client  # type: ignore[import-not-found]

        return fal_client
    raise ImportError(
        "fal-client must be installed before the async runtime starts. "
        "Install async-hermes-agent[fal] and import run_agent before entering "
        "the application event loop."
    )


async def _create_fal_client(fal_client: Any, *, key: str | None = None) -> Any:
    """Create FAL's async SDK client after awaited HTTP transport setup."""
    client = (
        fal_client.AsyncClient(key=key)
        if key is not None
        else fal_client.AsyncClient()
    )
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


def _normalize_fal_queue_url_format(queue_run_origin: str) -> str:
    normalized_origin = str(queue_run_origin or "").strip().rstrip("/")
    if not normalized_origin:
        raise ValueError("Managed FAL queue origin is required")
    return f"{normalized_origin}/"


def _extract_http_status(exc: BaseException) -> int | None:
    """Return an HTTP status code from httpx/FAL exceptions, when present."""
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if isinstance(status, int):
            return status
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    return None


class _ManagedFalClient:
    """Drive a Nous-managed queue origin through FAL's async primitives."""

    def __init__(self, fal_client: Any, *, key: str, queue_run_origin: str):
        async_client_class = getattr(fal_client, "AsyncClient", None)
        if async_client_class is None:
            raise RuntimeError(
                "fal_client.AsyncClient is required for managed FAL gateway mode"
            )

        client_module = getattr(fal_client, "client", None)
        if client_module is None:
            raise RuntimeError(
                "fal_client.client is required for managed FAL gateway mode"
            )

        self._fal_client_module = fal_client
        self._key = key
        self._queue_url_format = _normalize_fal_queue_url_format(
            queue_run_origin
        )
        self._client: Any = None
        self._setup_lock = asyncio.Lock()
        self._maybe_retry_request = getattr(
            client_module,
            "_async_maybe_retry_request",
            None,
        )
        self._raise_for_status = getattr(client_module, "_raise_for_status", None)
        self._request_handle_class = getattr(
            client_module,
            "AsyncRequestHandle",
            None,
        )
        self._add_hint_header = getattr(client_module, "add_hint_header", None)
        self._add_priority_header = getattr(
            client_module,
            "add_priority_header",
            None,
        )
        self._add_timeout_header = getattr(
            client_module,
            "add_timeout_header",
            None,
        )

        if self._maybe_retry_request is None or self._raise_for_status is None:
            raise RuntimeError(
                "fal_client.client async request helpers are required for "
                "managed FAL gateway mode"
            )
        if self._request_handle_class is None:
            raise RuntimeError(
                "fal_client.client.AsyncRequestHandle is required for managed "
                "FAL gateway mode"
            )

    async def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        async with self._setup_lock:
            if self._client is None:
                self._client = await _create_fal_client(
                    self._fal_client_module,
                    key=self._key,
                )
        return self._client

    async def submit(
        self,
        application: str,
        arguments: dict[str, Any],
        *,
        path: str = "",
        hint: str | None = None,
        webhook_url: str | None = None,
        priority: Any = None,
        headers: dict[str, str] | None = None,
        start_timeout: int | float | None = None,
    ) -> Any:
        client = await self._ensure_client()
        url = self._queue_url_format + application
        if path:
            url += "/" + path.lstrip("/")
        if webhook_url is not None:
            url += "?" + urlencode({"fal_webhook": webhook_url})

        request_headers = dict(headers or {})
        if hint is not None and self._add_hint_header is not None:
            self._add_hint_header(hint, request_headers)
        if priority is not None:
            if self._add_priority_header is None:
                raise RuntimeError(
                    "fal_client.client.add_priority_header is required for "
                    "priority requests"
                )
            self._add_priority_header(priority, request_headers)
        if start_timeout is not None:
            if self._add_timeout_header is None:
                raise RuntimeError(
                    "fal_client.client.add_timeout_header is required for "
                    "timeout requests"
                )
            self._add_timeout_header(start_timeout, request_headers)

        response = await self._maybe_retry_request(
            client._client,
            "POST",
            url,
            json=arguments,
            timeout=getattr(client, "default_timeout", 120.0),
            headers=request_headers,
        )
        self._raise_for_status(response)

        data = response.json()
        return self._request_handle_class(
            request_id=data["request_id"],
            response_url=data["response_url"],
            status_url=data["status_url"],
            cancel_url=data["cancel_url"],
            client=client._client,
        )

    async def close(self) -> None:
        """Close the owned FAL SDK transport."""
        client = self._client
        self._client = None
        if client is not None:
            await _close_fal_client(client)


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
                break  # noqa: ASYNC104 - prior caller cancellation is preserved
            if cancellation is None:
                cancellation = exc
            continue  # noqa: ASYNC104 - finish closing the owned client
        except Exception as exc:
            if cancellation is None:
                raise
            raise cancellation from exc  # noqa: ASYNC104
    if cancellation is not None:
        raise cancellation  # noqa: ASYNC104
