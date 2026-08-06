"""Shared native-async FAL.ai SDK plumbing."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
from urllib.parse import urlencode


def import_fal_client() -> Any:
    """Import and return the optional ``fal_client`` SDK."""
    import fal_client  # type: ignore[import-not-found]

    return fal_client


def _normalize_fal_queue_url_format(queue_run_origin: str) -> str:
    normalized_origin = str(queue_run_origin or "").strip().rstrip("/")
    if not normalized_origin:
        raise ValueError("Managed FAL queue origin is required")
    return f"{normalized_origin}/"


def _extract_http_status(exc: BaseException) -> Optional[int]:
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if isinstance(status, int):
            return status
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None


class _ManagedFalClient:
    """Drive a Nous-managed FAL queue through ``fal_client`` async primitives."""

    def __init__(self, fal_client: Any, *, key: str, queue_run_origin: str):
        client_class = getattr(fal_client, "AsyncClient", None)
        client_module = getattr(fal_client, "client", None)
        if client_class is None or client_module is None:
            raise RuntimeError("fal_client.AsyncClient is required for managed FAL gateway mode")

        self._queue_url_format = _normalize_fal_queue_url_format(queue_run_origin)
        self._client = client_class(key=key)
        http_client = getattr(self._client, "_client", None)
        request = getattr(client_module, "_async_maybe_retry_request", None)
        raise_for_status = getattr(client_module, "_raise_for_status", None)
        request_handle_class = getattr(client_module, "AsyncRequestHandle", None)
        self._add_hint_header = getattr(client_module, "add_hint_header", None)
        self._add_priority_header = getattr(client_module, "add_priority_header", None)
        self._add_timeout_header = getattr(client_module, "add_timeout_header", None)
        if http_client is None or request is None or raise_for_status is None:
            raise RuntimeError("fal_client async HTTP primitives are required for managed gateway mode")
        if request_handle_class is None:
            raise RuntimeError("fal_client.AsyncRequestHandle is required for managed gateway mode")
        self._http_client: Any = http_client
        self._request: Any = request
        self._raise_for_status: Any = raise_for_status
        self._request_handle_class: Any = request_handle_class

    async def submit(
        self,
        application: str,
        arguments: Dict[str, Any],
        *,
        path: str = "",
        hint: Optional[str] = None,
        webhook_url: Optional[str] = None,
        priority: Any = None,
        headers: Optional[Dict[str, str]] = None,
        start_timeout: Optional[Union[int, float]] = None,
    ):
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
                raise RuntimeError("fal_client priority header helper is unavailable")
            self._add_priority_header(priority, request_headers)
        if start_timeout is not None:
            if self._add_timeout_header is None:
                raise RuntimeError("fal_client timeout header helper is unavailable")
            self._add_timeout_header(start_timeout, request_headers)

        response = await self._request(
            self._http_client,
            "POST",
            url,
            json=arguments,
            timeout=getattr(self._client, "default_timeout", 120.0),
            headers=request_headers,
        )
        self._raise_for_status(response)
        data = response.json()
        return self._request_handle_class(
            request_id=data["request_id"],
            response_url=data["response_url"],
            status_url=data["status_url"],
            cancel_url=data["cancel_url"],
            client=self._http_client,
        )

    async def close(self) -> None:
        await self._http_client.aclose()
