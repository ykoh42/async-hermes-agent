"""Generic managed-tool gateway helpers for Nous-hosted vendor passthroughs."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

import aiofiles
import aiofiles.os

from hermes_constants import get_hermes_home
from tools.tool_backend_helpers import managed_nous_tools_enabled

logger = logging.getLogger(__name__)

_DEFAULT_TOOL_GATEWAY_DOMAIN = "nousresearch.com"
_DEFAULT_TOOL_GATEWAY_SCHEME = "https"
_NOUS_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120

_TokenReader = Callable[[], str | None | Awaitable[str | None]]


@dataclass(frozen=True)
class ManagedToolGatewayConfig:
    vendor: str
    gateway_origin: str
    nous_user_token: str
    managed_mode: bool


def auth_json_path():
    """Return the Hermes auth store path, respecting HERMES_HOME overrides."""
    return get_hermes_home() / "auth.json"


async def _read_nous_provider_state() -> dict | None:
    try:
        path = auth_json_path()
        if not await aiofiles.os.path.isfile(path):
            return None
        async with aiofiles.open(path, encoding="utf-8") as handle:
            data = json.loads(await handle.read())
        providers = data.get("providers", {})
        if not isinstance(providers, dict):
            return None
        nous_provider = providers.get("nous", {})
        if isinstance(nous_provider, dict):
            return nous_provider
    except Exception:
        pass
    return None


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _access_token_is_expiring(expires_at: object, skew_seconds: int) -> bool:
    expires = _parse_timestamp(expires_at)
    if expires is None:
        return True
    remaining = (expires - datetime.now(UTC)).total_seconds()
    return remaining <= max(0, int(skew_seconds))


def _read_user_token_override() -> str | None:
    """Read the gateway token from the active secret scope or process env."""
    try:
        from agent.secret_scope import get_secret
    except ImportError:  # pragma: no cover - secret_scope ships in this package
        explicit = os.getenv("TOOL_GATEWAY_USER_TOKEN")
    else:
        # ``get_secret`` retains the legacy process-env lookup outside
        # multiplex mode and deliberately raises when multiplexing lacks an
        # active scope.  Never turn that isolation failure into a foreign
        # process-token fallback.
        explicit = get_secret("TOOL_GATEWAY_USER_TOKEN")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    return None


async def peek_nous_access_token() -> str | None:
    """Probe for a cached Nous token without triggering OAuth refresh."""
    explicit = _read_user_token_override()
    if explicit:
        return explicit

    nous_provider = await _read_nous_provider_state() or {}
    access_token = nous_provider.get("access_token")
    if isinstance(access_token, str) and access_token.strip():
        return access_token.strip()
    return None


async def read_nous_access_token() -> str | None:
    """Read or refresh a Nous Subscriber OAuth access token."""
    explicit = _read_user_token_override()
    if explicit:
        return explicit
    nous_provider = await _read_nous_provider_state() or {}
    cached_token = await peek_nous_access_token()

    if cached_token and not _access_token_is_expiring(
        nous_provider.get("expires_at"),
        _NOUS_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    ):
        return cached_token

    try:
        from hermes_cli.auth import resolve_nous_access_token

        refreshed_token = await resolve_nous_access_token(
            refresh_skew_seconds=_NOUS_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
        )
        if isinstance(refreshed_token, str) and refreshed_token.strip():
            return refreshed_token.strip()
    except Exception as exc:
        logger.debug("Nous access token refresh failed: %s", exc)

    return cached_token


def get_tool_gateway_scheme() -> str:
    """Return configured shared gateway URL scheme."""
    scheme = os.getenv("TOOL_GATEWAY_SCHEME", "").strip().lower()
    if not scheme:
        return _DEFAULT_TOOL_GATEWAY_SCHEME
    if scheme in {"http", "https"}:
        return scheme
    raise ValueError("TOOL_GATEWAY_SCHEME must be 'http' or 'https'")


def build_vendor_gateway_url(vendor: str) -> str:
    """Return the gateway origin for a specific vendor."""
    vendor_key = f"{vendor.upper().replace('-', '_')}_GATEWAY_URL"
    explicit_vendor_url = os.getenv(vendor_key, "").strip().rstrip("/")
    if explicit_vendor_url:
        return explicit_vendor_url

    shared_scheme = get_tool_gateway_scheme()
    shared_domain = os.getenv("TOOL_GATEWAY_DOMAIN", "").strip().strip("/")
    if shared_domain:
        return f"{shared_scheme}://{vendor}-gateway.{shared_domain}"
    return f"{shared_scheme}://{vendor}-gateway.{_DEFAULT_TOOL_GATEWAY_DOMAIN}"


async def _read_token(token_reader: _TokenReader) -> str | None:
    token = token_reader()
    if inspect.isawaitable(token):
        token = await token
    return token


async def resolve_managed_tool_gateway(
    vendor: str,
    gateway_builder: Callable[[str], str] | None = None,
    token_reader: Callable[[], str | None] | None = None,
) -> ManagedToolGatewayConfig | None:
    """Resolve shared managed-tool gateway config for a vendor."""
    if not await managed_nous_tools_enabled():
        return None

    resolved_gateway_builder = gateway_builder or build_vendor_gateway_url
    resolved_token_reader = token_reader or read_nous_access_token
    gateway_origin = resolved_gateway_builder(vendor)
    nous_user_token = await _read_token(resolved_token_reader)
    if not gateway_origin or not nous_user_token:
        return None

    return ManagedToolGatewayConfig(
        vendor=vendor,
        gateway_origin=gateway_origin,
        nous_user_token=nous_user_token,
        managed_mode=True,
    )


async def is_managed_tool_gateway_ready(
    vendor: str,
    gateway_builder: Callable[[str], str] | None = None,
    token_reader: Callable[[], str | None] | None = None,
) -> bool:
    """Return whether the gateway URL and a cached token are available."""
    return await resolve_managed_tool_gateway(
        vendor,
        gateway_builder=gateway_builder,
        token_reader=token_reader or peek_nous_access_token,
    ) is not None


_MANAGED_GATEWAY_VENDOR = "tool"


def managed_vendor_base_path(vendor: str) -> str:
    """Base path for a managed vendor's REST routes on the gateway host."""
    return f"/api/{vendor}"


def managed_vendor_upload_path(vendor: str) -> str:
    """Media upload endpoint for a managed vendor, on the same host."""
    return f"/api/uploads/{vendor}"


def managed_vendor_endpoints(
    vendor: str,
    gateway_builder: Callable[[str], str] | None = None,
) -> dict | None:
    """Return absolute managed-vendor URLs, or ``None`` when unresolved."""
    builder = gateway_builder or build_vendor_gateway_url
    try:
        origin = builder(_MANAGED_GATEWAY_VENDOR).rstrip("/")
    except ValueError:
        return None
    if not origin:
        return None
    return {
        "origin": origin,
        "base_url": f"{origin}{managed_vendor_base_path(vendor)}",
        "upload_path": managed_vendor_upload_path(vendor),
    }


def is_managed_nous_gateway_url(
    url: object,
    gateway_builder: Callable[[str], str] | None = None,
) -> bool:
    """Return whether a URL is on the configured Nous gateway origin."""
    if not isinstance(url, str) or not url.strip():
        return False
    builder = gateway_builder or build_vendor_gateway_url
    try:
        expected = urlsplit(builder(_MANAGED_GATEWAY_VENDOR))
        actual = urlsplit(url.strip())
    except ValueError:
        return False
    return bool(actual.scheme) and (
        actual.scheme,
        actual.netloc,
    ) == (expected.scheme, expected.netloc)


async def managed_gateway_auth_headers(
    url: object,
    gateway_builder: Callable[[str], str] | None = None,
    token_reader: Callable[[], str | None] | None = None,
) -> dict:
    """Return fresh auth headers for a managed gateway URL."""
    if not is_managed_nous_gateway_url(url, gateway_builder):
        return {}

    resolved_token_reader = token_reader or read_nous_access_token
    try:
        token = await _read_token(resolved_token_reader)
    except Exception as exc:
        logger.debug("Managed gateway token read failed for %s: %s", url, exc)
        return {}
    if not isinstance(token, str) or not token.strip():
        return {}
    return {"Authorization": f"Bearer {token.strip()}"}


_MEDIA_UPLOAD_PRESIGN_TIMEOUT_SECONDS = 15.0
_MEDIA_UPLOAD_PUT_READ_TIMEOUT_SECONDS = 60.0
_MEDIA_UPLOAD_PUT_WRITE_TIMEOUT_SECONDS = 300.0


def _describe_media_upload_refusal(response: Any) -> str:
    try:
        payload = response.json()
        message = payload.get("error", {}).get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    except Exception:
        pass
    return f"the gateway refused the upload (HTTP {response.status_code})"


async def _close_httpx_client(client: Any, *, task_name: str) -> None:
    close_task = asyncio.create_task(client.aclose(), name=task_name)
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
            continue  # noqa: ASYNC104 - owned client must reach terminal cleanup
        except Exception as exc:
            if cancellation is None:
                raise
            raise cancellation from exc  # noqa: ASYNC104
    if cancellation is not None:
        raise cancellation  # noqa: ASYNC104


def build_managed_media_uploader(
    server_url: object,
    upload_path: object,
    gateway_builder: Callable[[str], str] | None = None,
    token_reader: Callable[[], str | None] | None = None,
) -> Callable | None:
    """Build the upstream managed-media presign/upload coroutine."""
    if not is_managed_nous_gateway_url(server_url, gateway_builder):
        return None
    if not isinstance(upload_path, str) or not upload_path.startswith("/"):
        return None

    parts = urlsplit(str(server_url).strip())
    origin = f"{parts.scheme}://{parts.netloc}"
    presign_url = f"{origin}{upload_path}"

    async def upload(data: bytes, mime: str) -> str:
        import httpx

        from agent.ssl_verify import _create_httpx_client
        from tools.url_safety import create_ssrf_safe_client

        headers = await managed_gateway_auth_headers(
            server_url,
            gateway_builder,
            token_reader,
        )
        if not headers:
            raise RuntimeError("no Nous credential is available for the upload")

        presign_timeout = httpx.Timeout(_MEDIA_UPLOAD_PRESIGN_TIMEOUT_SECONDS)
        client = await _create_httpx_client(timeout=presign_timeout)
        try:
            presign = await client.post(
                presign_url,
                headers=headers,
                json={"contentType": mime, "contentLength": len(data)},
            )
        finally:
            await _close_httpx_client(
                client,
                task_name="managed-media-presign-client-close",
            )
        if presign.status_code != 200:
            raise RuntimeError(_describe_media_upload_refusal(presign))

        try:
            payload = presign.json()
        except Exception:
            payload = None
        upload_url = payload.get("uploadUrl") if isinstance(payload, dict) else None
        token = payload.get("token") if isinstance(payload, dict) else None
        if not (
            isinstance(upload_url, str)
            and upload_url
            and isinstance(token, str)
            and token
        ):
            raise RuntimeError("the gateway's upload response was malformed")

        put_timeout = httpx.Timeout(
            _MEDIA_UPLOAD_PRESIGN_TIMEOUT_SECONDS,
            read=_MEDIA_UPLOAD_PUT_READ_TIMEOUT_SECONDS,
            write=_MEDIA_UPLOAD_PUT_WRITE_TIMEOUT_SECONDS,
        )
        client = await create_ssrf_safe_client(timeout=put_timeout)
        try:
            put = await client.put(
                upload_url,
                content=data,
                headers={"Content-Type": mime},
            )
        finally:
            await _close_httpx_client(
                client,
                task_name="managed-media-put-client-close",
            )
        if put.status_code != 200:
            raise RuntimeError(
                f"storage refused the upload (HTTP {put.status_code})"
            )
        return f"nous-upload:{token}"

    return upload
