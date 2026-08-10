"""TLS verify resolution for httpx/OpenAI provider clients."""

from __future__ import annotations

import asyncio
import logging
import os
import ssl
import warnings
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
    trust_env: bool = True,
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
    if trust_env:
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


async def _materialize_httpx_verify(
    verify: Any = True,
    *,
    cert: Any = None,
    trust_env: bool = True,
) -> Any:
    """Materialize httpx's native TLS inputs without changing its semantics."""
    if verify is True:
        ssl_cert_file = os.getenv("SSL_CERT_FILE") if trust_env else None
        ssl_cert_dir = os.getenv("SSL_CERT_DIR") if trust_env else None
        if ssl_cert_file:
            context = await aiofiles.os.wrap(ssl.create_default_context)(
                cafile=ssl_cert_file
            )
        elif ssl_cert_dir:
            context = await aiofiles.os.wrap(ssl.create_default_context)(
                capath=ssl_cert_dir
            )
        else:
            default_ca = await aiofiles.os.wrap(certifi.where)()
            context = await aiofiles.os.wrap(ssl.create_default_context)(
                cafile=default_ca
            )
    elif verify is False:
        if cert is None:
            return False
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    elif isinstance(verify, str):
        warnings.warn(
            "`verify=<str>` is deprecated. Use "
            "`verify=ssl.create_default_context(cafile=...)` or "
            "`verify=ssl.create_default_context(capath=...)` instead.",
            DeprecationWarning,
        )
        if await aiofiles.os.path.isdir(verify):
            context = await aiofiles.os.wrap(ssl.create_default_context)(
                capath=verify
            )
        else:
            context = await aiofiles.os.wrap(ssl.create_default_context)(
                cafile=verify
            )
    else:
        context = verify

    if cert is not None:
        warnings.warn(
            "`cert=...` is deprecated. Use `verify=<ssl_context>` instead,"
            "with `.load_cert_chain()` to configure the certificate chain.",
            DeprecationWarning,
        )
        if isinstance(cert, str):
            await aiofiles.os.wrap(context.load_cert_chain)(cert)
        else:
            await aiofiles.os.wrap(context.load_cert_chain)(*cert)
    return context


async def _materialize_httpx_proxy(proxy: Any) -> Any:
    import httpcore
    import httpx

    resolved = (
        httpx.Proxy(url=proxy)
        if isinstance(proxy, (str, httpx.URL))
        else proxy
    )
    if resolved.url.scheme != "https" or resolved.ssl_context is not None:
        return resolved
    proxy_context = await aiofiles.os.wrap(httpcore.default_ssl_context)()
    return httpx.Proxy(
        url=resolved.url,
        ssl_context=proxy_context,
        auth=resolved.auth,
        headers=resolved.headers,
    )


async def _create_httpx_client(**kwargs: Any) -> Any:
    """Construct an AsyncClient after native-async TLS and proxy setup."""
    import httpx
    from httpx._config import DEFAULT_LIMITS
    from httpx._utils import get_environment_proxies

    client_kwargs = dict(kwargs)
    custom_transport = client_kwargs.get("transport")
    if custom_transport is not None and client_kwargs.get("proxy") is None:
        return httpx.AsyncClient(**client_kwargs)

    trust_env = bool(client_kwargs.get("trust_env", True))
    cert = client_kwargs.pop("cert", None)
    verify = await _materialize_httpx_verify(
        client_kwargs.get("verify", True),
        cert=cert,
        trust_env=trust_env,
    )
    client_kwargs["verify"] = verify

    http1 = bool(client_kwargs.get("http1", True))
    http2 = bool(client_kwargs.get("http2", False))
    limits = client_kwargs.get("limits", DEFAULT_LIMITS)
    owned_transports = []
    if custom_transport is None:
        default_transport = httpx.AsyncHTTPTransport(
            verify=verify,
            trust_env=trust_env,
            http1=http1,
            http2=http2,
            limits=limits,
        )
        owned_transports.append(default_transport)
        client_kwargs["transport"] = default_transport

    try:
        proxy = client_kwargs.get("proxy")
        if proxy is not None:
            proxy_config = await _materialize_httpx_proxy(proxy)
            transport = httpx.AsyncHTTPTransport(
                verify=verify,
                trust_env=trust_env,
                http1=http1,
                http2=http2,
                limits=limits,
                proxy=proxy_config,
            )
            owned_transports.append(transport)
            proxy_mounts = {"all://": transport}
            mounts = client_kwargs.get("mounts")
            if mounts is not None:
                proxy_mounts.update(mounts)
            client_kwargs["mounts"] = proxy_mounts
            client_kwargs.pop("proxy")
        elif trust_env:
            proxy_map = await aiofiles.os.wrap(get_environment_proxies)()
            env_mounts = {}
            for pattern, proxy_url in proxy_map.items():
                if proxy_url is None:
                    env_mounts[pattern] = None
                    continue
                proxy_config = await _materialize_httpx_proxy(proxy_url)
                transport = httpx.AsyncHTTPTransport(
                    verify=verify,
                    trust_env=trust_env,
                    http1=http1,
                    http2=http2,
                    limits=limits,
                    proxy=proxy_config,
                )
                owned_transports.append(transport)
                env_mounts[pattern] = transport
            mounts = client_kwargs.get("mounts")
            if mounts is not None:
                env_mounts.update(mounts)
            client_kwargs["mounts"] = env_mounts

        return httpx.AsyncClient(**client_kwargs)
    except BaseException as construction_error:
        async def close_transports() -> None:
            await asyncio.gather(
                *(transport.aclose() for transport in reversed(owned_transports)),
                return_exceptions=True,
            )

        cleanup = asyncio.create_task(
            close_transports(),
            name="httpx-transport-construction-cleanup",
        )
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                await asyncio.shield(cleanup)
                break
            except asyncio.CancelledError as exc:  # noqa: ASYNC103
                if cleanup.cancelled():
                    break  # noqa: ASYNC104 - owned cleanup reached a terminal state
                if cancellation is None:
                    cancellation = exc
                continue  # noqa: ASYNC104 - finish closing owned transports
        if cancellation is not None:
            raise cancellation from construction_error  # noqa: ASYNC104
        raise
