"""Preventive SSL CA certificate checks for Hermes Agent."""

from __future__ import annotations

import logging
import os
import ssl
from pathlib import Path

import aiofiles.os

from agent.errors import SSLConfigurationError

logger = logging.getLogger(__name__)

_CA_BUNDLE_ENV_VARS = (
    "HERMES_CA_BUNDLE",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
)
_SKIP_VALUES = {"1", "true", "yes", "on"}


def _skip_ssl_guard_enabled() -> bool:
    return os.getenv("HERMES_SKIP_SSL_GUARD", "").strip().lower() in _SKIP_VALUES


def _repair_hint() -> str:
    return (
        "Repair certifi/openai/httpx, or fix or unset the broken custom "
        "CA bundle environment variable."
    )


def _ssl_err(message: str) -> SSLConfigurationError:
    return SSLConfigurationError(f"{message}\n{_repair_hint()}")


async def _validate_bundle_path(
    label: str,
    value: str,
    *,
    require_substantial: bool = False,
) -> None:
    path = await aiofiles.os.wrap(Path.expanduser)(Path(value))
    if not await aiofiles.os.path.exists(path):
        raise _ssl_err(f"{label} points to a missing CA bundle: {value}")
    if not await aiofiles.os.path.isfile(path):
        raise _ssl_err(f"{label} does not point to a CA bundle file: {value}")
    if require_substantial and (await aiofiles.os.stat(path)).st_size < 1024:
        raise _ssl_err(f"{label} at {value} appears corrupted (too small)")
    try:
        context = await aiofiles.os.wrap(ssl.create_default_context)(cafile=str(path))
    except Exception as exc:
        raise _ssl_err(
            f"{label} CA bundle at {value} cannot be loaded: {exc}"
        ) from exc
    if not context.get_ca_certs():
        raise _ssl_err(f"{label} CA bundle at {value} did not load certificates")


async def verify_ca_bundle() -> None:
    """Verify explicit and certifi CA bundles before provider startup."""
    if _skip_ssl_guard_enabled():
        logger.debug("SSL CA bundle guard skipped via HERMES_SKIP_SSL_GUARD")
        return

    for env_var in _CA_BUNDLE_ENV_VARS:
        value = os.getenv(env_var)
        if value:
            await _validate_bundle_path(env_var, value)

    try:
        import certifi
    except Exception as exc:
        raise _ssl_err(f"certifi is not importable: {exc}") from exc
    certifi_path = await aiofiles.os.wrap(certifi.where)()
    await _validate_bundle_path(
        "certifi",
        str(certifi_path),
        require_substantial=True,
    )


async def verify_ca_bundle_with_fallback() -> None:
    """Retain the upstream entry point while enforcing the same check."""
    await verify_ca_bundle()
