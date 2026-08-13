"""TLS verify resolution for httpx/OpenAI provider clients."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import ssl
import sys
import warnings
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os
import certifi

from agent.secret_scope import get_secret

logger = logging.getLogger(__name__)


def _coerce_insecure(ssl_verify: Any) -> bool:
    if ssl_verify is False:
        return True
    if isinstance(ssl_verify, str) and ssl_verify.strip().lower() in {"false", "0", "no", "off"}:
        return True
    return False


def _ca_data(raw: bytes) -> str | bytes:
    """Return the in-memory form accepted by ``SSLContext``.

    PEM bundles must be text while a single ASN.1 certificate must remain
    bytes. Decoding after the awaited read keeps certificate parsing as the
    only synchronous work at this boundary.
    """
    if b"-----BEGIN CERTIFICATE-----" in raw:
        return raw.decode("ascii")
    return raw


async def _read_file_bytes(path: str | Path) -> bytes:
    async with aiofiles.open(path, "rb") as handle:
        return await handle.read()


async def _context_from_ca_file(path: str | Path) -> ssl.SSLContext:
    raw = await _read_file_bytes(path)
    return ssl.create_default_context(cadata=_ca_data(raw))


def _empty_verified_context() -> ssl.SSLContext:
    """Match ``create_default_context(capath=<empty>)`` without disk I/O."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.verify_mode = ssl.CERT_REQUIRED
    context.check_hostname = True
    if sys.version_info >= (3, 13):
        context.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN
        context.verify_flags |= ssl.VERIFY_X509_STRICT
    keylog_file = os.getenv("SSLKEYLOGFILE")
    if (
        keylog_file
        and not sys.flags.ignore_environment
        and hasattr(context, "keylog_filename")
    ):
        context.keylog_filename = keylog_file
    return context


async def _context_from_ca_directories(raw_paths: str) -> ssl.SSLContext:
    """Load OpenSSL-style CA directories through awaited file operations."""
    payloads: list[str | bytes] = []
    for raw_path in raw_paths.split(os.pathsep):
        if not raw_path:
            continue
        directory = Path(raw_path).expanduser()
        if not await aiofiles.os.path.isdir(directory):
            continue
        try:
            names = sorted(await aiofiles.os.listdir(directory))
        except OSError:
            continue
        for name in names:
            # OpenSSL's ``capath`` contract ignores entries that have not been
            # prepared with ``openssl rehash``. Loading arbitrary PEM files
            # here would silently broaden the caller's trust store.
            if re.fullmatch(r"[0-9A-Fa-f]{8}\.[0-9]+", name) is None:
                continue
            candidate = directory / name
            if not await aiofiles.os.path.isfile(candidate):
                continue
            try:
                payloads.append(_ca_data(await _read_file_bytes(candidate)))
            except (OSError, UnicodeDecodeError):
                # OpenSSL capath semantics ignore unrelated/unreadable entries.
                continue

    context = _empty_verified_context()
    for payload in payloads:
        try:
            context.load_verify_locations(cadata=payload)
        except ssl.SSLError:
            # Hashed certificate directories commonly contain auxiliary files;
            # OpenSSL ignores entries that are not usable trust anchors.
            continue
    return context


def _raw_default_verify_paths() -> tuple[str | None, str | None]:
    """Resolve OpenSSL's configured paths without ``os.stat`` calls."""
    cafile_env, openssl_cafile, capath_env, openssl_capath = (
        ssl._ssl.get_default_verify_paths()  # type: ignore[attr-defined]
    )
    return (
        os.getenv(cafile_env, openssl_cafile),
        os.getenv(capath_env, openssl_capath),
    )


async def _default_proxy_context() -> ssl.SSLContext:
    """Reproduce httpcore's system-plus-certifi proxy trust store."""
    context = await _context_from_ca_file(certifi.where())
    default_cafile, default_capath = _raw_default_verify_paths()
    if default_cafile and await aiofiles.os.path.isfile(default_cafile):
        try:
            system_data = _ca_data(await _read_file_bytes(default_cafile))
            context.load_verify_locations(cadata=system_data)
        except (OSError, UnicodeDecodeError, ssl.SSLError):
            pass
    if default_capath:
        system_context = await _context_from_ca_directories(default_capath)
        # Python does not expose one context's certificate objects as cadata.
        # Read the same directory into this context directly instead.
        for der_cert in system_context.get_ca_certs(binary_form=True):
            try:
                context.load_verify_locations(cadata=der_cert)
            except ssl.SSLError:
                continue
    return context


async def _preload_client_cert_files(cert: Any) -> None:
    """Await client-certificate reads before stdlib attaches the parsed chain."""
    paths = (cert,) if isinstance(cert, str) else tuple(cert[:2])
    for path in paths:
        if path is not None:
            await _read_file_bytes(path)


async def resolve_httpx_verify(
    *,
    ca_bundle: str | None = None,
    ssl_verify: Any = None,
    base_url: str = "",
) -> bool | ssl.SSLContext:
    """Resolve httpx ``verify`` for provider HTTP clients.

    Priority:
    1. ``ssl_verify: false`` — disable verification (local dev only)
    2. explicit ``ca_bundle`` (per-provider ``ssl_ca_cert`` config field)
    3. ``HERMES_CA_BUNDLE``, ``REQUESTS_CA_BUNDLE``, ``SSL_CERT_FILE``,
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

    candidates = (
        (ca_bundle or "").strip(),
        os.getenv("HERMES_CA_BUNDLE", "").strip(),
        os.getenv("REQUESTS_CA_BUNDLE", "").strip(),
        os.getenv("SSL_CERT_FILE", "").strip(),
        os.getenv("CURL_CA_BUNDLE", "").strip(),
    )
    for effective_ca in candidates:
        if not effective_ca:
            continue
        ca_path = Path(effective_ca).expanduser()
        if await aiofiles.os.path.isfile(ca_path):
            return await _context_from_ca_file(ca_path)
        logger.warning(
            "CA bundle path does not exist: %s — falling back to default certificates",
            effective_ca,
        )
    return True


async def _resolve_httpx_client_verify(
    *,
    ca_bundle: str | None = None,
    ssl_verify: Any = None,
    base_url: str = "",
    trust_env: bool = True,
) -> bool | ssl.SSLContext:
    """Materialize the default TLS context before constructing an HTTP client.

    The public resolver retains upstream's exact ``True`` default.  httpx turns
    that sentinel into an SSL context synchronously in its constructor, so this
    boundary awaits the certificate file read before parsing the in-memory
    certificate data without changing the public return shape.
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
            return await _context_from_ca_file(ssl_cert_file)
        ssl_cert_dir = os.getenv("SSL_CERT_DIR")
        if ssl_cert_dir:
            return await _context_from_ca_directories(ssl_cert_dir)
    return await _context_from_ca_file(certifi.where())


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
            context = await _context_from_ca_file(ssl_cert_file)
        elif ssl_cert_dir:
            context = await _context_from_ca_directories(ssl_cert_dir)
        else:
            context = await _context_from_ca_file(certifi.where())
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
            context = await _context_from_ca_directories(verify)
        else:
            context = await _context_from_ca_file(verify)
    else:
        context = verify

    if cert is not None:
        warnings.warn(
            "`cert=...` is deprecated. Use `verify=<ssl_context>` instead,"
            "with `.load_cert_chain()` to configure the certificate chain.",
            DeprecationWarning,
        )
        await _preload_client_cert_files(cert)
        if isinstance(cert, str):
            context.load_cert_chain(cert)
        else:
            context.load_cert_chain(*cert)
    return context


async def _materialize_httpx_proxy(proxy: Any) -> Any:
    import httpx

    resolved = (
        httpx.Proxy(url=proxy)
        if isinstance(proxy, (str, httpx.URL))
        else proxy
    )
    if resolved.url.scheme != "https" or resolved.ssl_context is not None:
        return resolved
    proxy_context = await _default_proxy_context()
    return httpx.Proxy(
        url=resolved.url,
        ssl_context=proxy_context,
        auth=resolved.auth,
        headers=resolved.headers,
    )


async def _create_httpx_client(
    *,
    _client_factory: Any = None,
    _transport_options: Any = None,
    **kwargs: Any,
) -> Any:
    """Construct an AsyncClient after native-async TLS and proxy setup."""
    import httpx
    from httpx._config import DEFAULT_LIMITS
    from httpx._utils import get_environment_proxies

    client_kwargs = dict(kwargs)
    client_factory = _client_factory or httpx.AsyncClient
    transport_options = dict(_transport_options or {})
    custom_transport = client_kwargs.get("transport")
    if custom_transport is not None and client_kwargs.get("proxy") is None:
        return client_factory(**client_kwargs)

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
            **transport_options,
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
                **transport_options,
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
            proxy_map = get_environment_proxies()
            env_mounts = {}
            for pattern, proxy_url in proxy_map.items():
                if proxy_url is None:
                    env_mounts[pattern] = None
                    continue
                proxy_config = await _materialize_httpx_proxy(proxy_url)
                transport = httpx.AsyncHTTPTransport(
                    **transport_options,
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

        return client_factory(**client_kwargs)
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


async def _create_openai_sdk_client(
    client_class: Any,
    /,
    **kwargs: Any,
) -> Any:
    """Construct an OpenAI async client with its exact default HTTP settings."""
    import httpx
    from openai import _base_client

    client_factory = getattr(_base_client, "AsyncHttpxClientWrapper", None)
    limits = getattr(_base_client, "DEFAULT_CONNECTION_LIMITS", None)
    timeout = getattr(_base_client, "DEFAULT_TIMEOUT", None)
    if client_factory is None or not isinstance(limits, httpx.Limits):
        raise RuntimeError(
            "The installed OpenAI runtime does not expose its async HTTP "
            "client defaults. Reinstall async-hermes-agent."
        )
    if not isinstance(timeout, httpx.Timeout):
        raise RuntimeError(
            "The installed OpenAI runtime does not expose its default timeout. "
            "Reinstall async-hermes-agent."
        )

    base_url = kwargs.get("base_url")
    if base_url is None:
        base_url = get_secret("OPENAI_BASE_URL")
    if base_url is None:
        base_url = "https://api.openai.com/v1"

    http_client = await _create_httpx_client(
        _client_factory=client_factory,
        base_url=base_url,
        timeout=timeout,
        limits=limits,
        follow_redirects=True,
    )
    try:
        return client_class(http_client=http_client, **kwargs)
    except BaseException as construction_error:
        close_task = asyncio.create_task(
            http_client.aclose(),
            name="openai-http-client-construction-cleanup",
        )
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                await asyncio.shield(close_task)
                break
            except asyncio.CancelledError as exc:  # noqa: ASYNC103
                if close_task.cancelled():
                    break  # noqa: ASYNC104 - owned cleanup reached a terminal state
                if cancellation is None:
                    cancellation = exc
                continue  # noqa: ASYNC104 - finish closing the owned client
        if cancellation is not None:
            raise cancellation from construction_error  # noqa: ASYNC104
        raise
