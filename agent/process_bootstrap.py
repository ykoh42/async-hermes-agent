"""Process-level bootstrap helpers for ``run_agent``.

Three concerns, all tied to ``AIAgent`` boot-time / runtime IO setup:

1. **OpenAI SDK bootstrap** — import ``AsyncOpenAI`` before an event loop starts
   while preserving Hermes' stable ``OpenAI`` name and patch location.

2. **Crash-resistant stdio** — ``_SafeWriter`` wraps stdout/stderr so
   ``OSError: Input/output error`` from broken pipes (systemd, Docker,
   thread teardown races) cannot crash the agent.  ``_install_safe_stdio``
   applies the wrapper.

3. **HTTP proxy resolution** — ``_get_proxy_from_env`` reads
   ``HTTPS_PROXY`` / ``HTTP_PROXY`` / ``ALL_PROXY``;
   ``_get_proxy_for_base_url`` respects ``NO_PROXY`` for the given base URL.

``run_agent`` re-exports every name so existing
``from run_agent import _get_proxy_from_env`` imports keep working
unchanged.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from typing import Any, Optional

# ``AsyncOpenAI.chat`` lazily imports the complete OpenAI resources package on
# first access.  If that import is deferred until the first model request, the
# importlib filesystem walk runs inside the event loop and can pause every
# concurrent request (the SDK's lazy property is synchronous).  Load the
# resource module alongside the stable client alias, before any turn starts;
# this keeps the actual provider→model chain free of first-use import stalls.
from openai import AsyncOpenAI as OpenAI
from openai.resources.chat import AsyncChat as _AsyncChatBootstrap  # noqa: F401

# ``httpx.AsyncHTTPTransport`` imports httpcore synchronously from its
# constructor.  The transport is created at the first awaited provider
# boundary, so preload the mandatory httpx transport dependency beside the
# other SDK bootstraps rather than letting importlib pause that first turn.
import httpcore as _HttpcoreBootstrap  # noqa: F401

# Anthropic's SDK also performs a sizeable lazy import graph.  The retained
# runtime reaches ``build_anthropic_client`` from an awaited turn, so loading
# the optional SDK here keeps that first-use import out of the event loop.
# Missing optional dependencies remain a normal provider-level ImportError.
try:
    import anthropic as _AnthropicBootstrap  # noqa: F401
except ImportError:
    _AnthropicBootstrap = None

# MCP OAuth is optional, but its SDK auth modules are imported lazily by the
# OAuth storage/manager.  Resolve those imports beside the other provider SDK
# bootstraps so the first OAuth-backed MCP connection cannot perform an
# importlib filesystem walk from inside an active request loop.  The helper
# keeps the optional dependency fail-open exactly as before.
try:
    from tools import mcp_oauth as _McpOAuthBootstrap

    _McpOAuthBootstrap._ensure_sdk_loaded()
except (ImportError, AttributeError):
    _McpOAuthBootstrap = None

from utils import base_url_hostname, normalize_proxy_url


class _SafeWriter:
    """Transparent stdio wrapper that catches OSError/ValueError from broken pipes.

    When hermes-agent runs as a systemd service, Docker container, or headless
    daemon, the stdout/stderr pipe can become unavailable (idle timeout, buffer
    exhaustion, socket reset). Any print() call then raises
    ``OSError: [Errno 5] Input/output error``, which can crash agent setup or
    run_conversation() — especially via double-fault when an except handler
    also tries to print.

    The shared stdout handle can also close during process teardown, raising
    ``ValueError: I/O operation on closed file`` instead of OSError.

    This wrapper delegates all writes to the underlying stream and silently
    catches both OSError and ValueError. It is transparent when the wrapped
    stream is healthy.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def write(self, data):
        try:
            return self._inner.write(data)
        except (OSError, ValueError):
            return len(data) if isinstance(data, str) else 0

    def flush(self):
        try:
            self._inner.flush()
        except (OSError, ValueError):
            pass

    def fileno(self):
        return self._inner.fileno()

    def isatty(self):
        try:
            return self._inner.isatty()
        except (OSError, ValueError):
            return False

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _get_proxy_from_env() -> Optional[str]:
    """Read proxy URL from environment variables.

    Checks HTTPS_PROXY, HTTP_PROXY, ALL_PROXY (and lowercase variants) in order.
    Returns the first valid proxy URL found, or None if no proxy is configured.
    """
    for key in (
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "https_proxy",
        "http_proxy",
        "all_proxy",
    ):
        value = os.environ.get(key, "").strip()
        if value:
            return normalize_proxy_url(value)
    return None


def _get_proxy_for_base_url(base_url: Optional[str]) -> Optional[str]:
    """Return an env-configured proxy unless NO_PROXY excludes this base URL."""
    proxy = _get_proxy_from_env()
    if not proxy or not base_url:
        return proxy

    host = base_url_hostname(base_url)
    if not host:
        return proxy

    try:
        if urllib.request.proxy_bypass_environment(host):
            return None
    except Exception:
        pass

    return proxy


async def build_keepalive_http_client(
    base_url: str = "",
    *,
    async_mode: bool = False,
    verify: Any = True,
) -> Any:
    """Build the native async httpx client for OpenAI SDK calls.

    The Hermes runtime always returns the native async transport. The
    upstream ``async_mode`` argument is retained so existing callers only need
    to add ``await`` at their async boundary; it no longer selects a sync
    client.

    Uses explicit ``HTTPS_PROXY`` / ``NO_PROXY`` env vars via
    ``_get_proxy_for_base_url``. Plain no-proxy mounts disable httpx's default
    ``trust_env`` proxy path, so macOS system proxy settings from
    ``urllib.request.getproxies()`` (which omit the ExceptionsList) are not
    applied. Mirrors ``AIAgent._build_keepalive_http_client``.

    Connection lifecycle is managed at the HTTP pool layer
    (``keepalive_expiry=20.0`` reaps idle connections before reverse proxies'
    typical 30-60 s timeouts) instead of the former custom
    ``socket_options`` transport, which broke streaming behind reverse
    proxies (#54049, #12952) and stalled TLS handshakes by stripping
    ``TCP_NODELAY``.

    ``verify`` is forwarded to httpx so auxiliary-client calls (compression,
    vision, web_extract, title generation, etc.) honor the same per-provider
    ``ssl_ca_cert`` / ``ssl_verify`` and ``HERMES_CA_BUNDLE`` settings the main
    client uses. HTTPS proxies receive their own default context; the
    provider-specific context is used only for the target connection.
    """
    import ssl

    import httpx

    from agent.ssl_verify import _resolve_httpx_client_verify

    if verify is True:
        target_verify = await _resolve_httpx_client_verify()
    elif isinstance(verify, str):
        target_verify = await _resolve_httpx_client_verify(ca_bundle=verify)
    elif verify is False or isinstance(verify, ssl.SSLContext):
        target_verify = verify
    else:
        raise TypeError("verify must be a bool, path, or ssl.SSLContext")

    proxy_url = _get_proxy_for_base_url(base_url)
    proxy: Any = proxy_url
    if proxy_url and httpx.URL(proxy_url).scheme == "https":
        proxy = httpx.Proxy(
            proxy_url,
            ssl_context=await _resolve_httpx_client_verify(),
        )

    limits = httpx.Limits(
        max_keepalive_connections=20,
        max_connections=100,
        keepalive_expiry=20.0,
    )
    # Generous read=None for SSE streaming endpoints.
    timeout = httpx.Timeout(connect=15.0, read=None, write=15.0, pool=10.0)
    transport = httpx.AsyncHTTPTransport(
        verify=target_verify,
        limits=limits,
        proxy=proxy,
    )
    try:
        return httpx.AsyncClient(
            transport=transport,
            timeout=timeout,
            trust_env=False,
        )
    except BaseException:
        await transport.aclose()
        raise


def _install_safe_stdio() -> None:
    """Wrap stdout/stderr so best-effort console output cannot crash the agent."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and not isinstance(stream, _SafeWriter):
            setattr(sys, stream_name, _SafeWriter(stream))


__all__ = [
    "OpenAI",
    "_SafeWriter",
    "_install_safe_stdio",
    "_get_proxy_from_env",
    "_get_proxy_for_base_url",
    "build_keepalive_http_client",
]
