"""TLS verify resolution for httpx/OpenAI provider clients."""

from __future__ import annotations

import logging
import os
import ssl
from pathlib import Path
from typing import Any, Optional

import aiofiles.os
import certifi

logger = logging.getLogger(__name__)


def _coerce_insecure(ssl_verify: Any) -> bool:
    if ssl_verify is False:
        return True
    if isinstance(ssl_verify, str) and ssl_verify.strip().lower() in {"false", "0", "no", "off"}:
        return True
    return False


async def resolve_httpx_verify(
    *,
    ca_bundle: Optional[str] = None,
    ssl_verify: Any = None,
    base_url: str = "",
) -> bool | ssl.SSLContext:
    """Resolve httpx ``verify`` for provider HTTP clients.

    Priority:
    1. ``ssl_verify: false`` — disable verification (local dev only)
    2. explicit ``ca_bundle`` (per-provider ``ssl_ca_cert`` config field)
    3. ``HERMES_CA_BUNDLE``, ``SSL_CERT_FILE``, ``REQUESTS_CA_BUNDLE``,
       ``CURL_CA_BUNDLE`` env vars
    4. ``True`` (httpx/certifi default)

    ``base_url`` is used only for the insecure-mode warning message.
    """
    if _coerce_insecure(ssl_verify):
        logger.warning(
            "TLS certificate verification DISABLED (ssl_verify: false) for %s — "
            "this is intended for local development only and is unsafe on any "
            "network you do not fully control.",
            base_url or "a custom provider endpoint",
        )
        return False

    effective_ca = (
        (ca_bundle or "").strip()
        or os.getenv("HERMES_CA_BUNDLE", "").strip()
        or os.getenv("SSL_CERT_FILE", "").strip()
        or os.getenv("REQUESTS_CA_BUNDLE", "").strip()
        or os.getenv("CURL_CA_BUNDLE", "").strip()
    )
    if effective_ca:
        ca_path = await aiofiles.os.wrap(Path.expanduser)(Path(effective_ca))
        if await aiofiles.os.path.isfile(ca_path):
            return await aiofiles.os.wrap(ssl.create_default_context)(
                cafile=str(ca_path)
            )
        logger.warning(
            "CA bundle path does not exist: %s — falling back to default certificates",
            effective_ca,
        )
    return True


async def _resolve_httpx_client_verify(
    *,
    ca_bundle: Optional[str] = None,
    ssl_verify: Any = None,
    base_url: str = "",
) -> bool | ssl.SSLContext:
    """Materialize the default TLS context before constructing an HTTP client.

    The public resolver retains upstream's exact ``True`` default.  httpx turns
    that sentinel into an SSL context synchronously in its constructor, so the
    retained async runtime uses this private boundary to perform that work away
    from the event-loop thread without changing the public return shape.
    """
    verify = await resolve_httpx_verify(
        ca_bundle=ca_bundle,
        ssl_verify=ssl_verify,
        base_url=base_url,
    )
    if verify is not True:
        return verify
    ssl_cert_file = os.getenv("SSL_CERT_FILE")
    if ssl_cert_file:
        return await aiofiles.os.wrap(ssl.create_default_context)(
            cafile=ssl_cert_file
        )
    ssl_cert_dir = os.getenv("SSL_CERT_DIR")
    if ssl_cert_dir:
        return await aiofiles.os.wrap(ssl.create_default_context)(
            capath=ssl_cert_dir
        )
    default_ca = await aiofiles.os.wrap(certifi.where)()
    return await aiofiles.os.wrap(ssl.create_default_context)(cafile=default_ca)
