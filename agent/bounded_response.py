"""Bounded reads of HTTP error response bodies.

When a provider returns a non-OK status on a *streaming* request, Hermes reads
the response body to build a useful diagnostic error. A bare ``response.read()``
on a streaming httpx response is unbounded in two dangerous ways:

1. A server can declare (or stream) an arbitrarily large body, so the read can
   balloon memory.
2. A server can open the body and then stall forever (no ``Content-Length``,
   no further bytes), so the read hangs the agent indefinitely.

Both are realistic against a misbehaving proxy, a hijacked endpoint, or a
provider having a bad day. The diagnostic body is only ever shown to the user
truncated to a few hundred characters, so reading megabytes — or blocking
forever — buys nothing.

``read_streaming_error_body`` bounds the read to a byte cap and enforces a
hard wall-clock deadline, returning the decoded text snippet. Callers pass the
returned text into their existing error builders instead of touching
``response.text`` (which would be unbounded / would raise after a partial
stream read).

A subtlety the implementation must respect: a wall-clock check placed only
between yielded chunks cannot interrupt a server that stalls mid-chunk. The
async response iterator therefore runs under an asyncio deadline; cancellation
closes the response and returns whatever partial bytes were collected.

Ported and adapted from openclaw/openclaw#95108 ("bound Anthropic error
streams"), generalized to cover Hermes's three streaming error-body sites
(native Gemini, Gemini Cloud Code, Antigravity Cloud Code).
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)

# Defaults chosen to comfortably hold any real provider error envelope (Google
# RPC error JSON, Anthropic error JSON) while rejecting pathological bodies.
DEFAULT_ERROR_BODY_MAX_BYTES = 64 * 1024
# Hard wall-clock deadline for the whole bounded read. A streaming error body
# that does not finish within this window is abandoned and the connection is
# closed; we keep whatever partial bytes arrived.
DEFAULT_ERROR_BODY_TIMEOUT_S = 10.0


async def read_streaming_error_body(
    response: httpx.Response,
    *,
    max_bytes: int = DEFAULT_ERROR_BODY_MAX_BYTES,
    timeout_s: float = DEFAULT_ERROR_BODY_TIMEOUT_S,
) -> str:
    """Native-async bounded error-body read for async HTTP responses.

    The active agent transport must not hand an ``httpx.AsyncByteStream`` to
    the synchronous worker-thread helper above.  Cancellation of the async
    iterator is sufficient to release the response stream; the provider's
    own read timeout remains the lower-level socket guard.
    """
    chunks: List[bytes] = []
    total = 0

    async def _drain() -> None:
        nonlocal total
        async for chunk in response.aiter_bytes():
            if not chunk:
                continue
            remaining = max_bytes - total
            if remaining <= 0:
                break
            chunks.append(chunk[:remaining])
            total += min(len(chunk), remaining)
            if len(chunk) > remaining:
                break

    try:
        await asyncio.wait_for(_drain(), timeout=timeout_s)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        if isinstance(asyncio.current_task(), asyncio.Task) and asyncio.current_task().cancelling():
            raise
        logger.debug(
            "bounded async error-body read: hard timeout after %.1fs (%d bytes so far)",
            timeout_s,
            total,
        )
    except Exception as exc:  # noqa: BLE001 - error path must not mask HTTP failure
        logger.debug("bounded async error-body read failed: %s", exc)
    finally:
        try:
            await response.aclose()
        except Exception:
            pass
    return b"".join(chunks).decode("utf-8", errors="replace")


async def read_error_body_or_default(
    response: httpx.Response,
    *,
    max_bytes: int = DEFAULT_ERROR_BODY_MAX_BYTES,
    timeout_s: float = DEFAULT_ERROR_BODY_TIMEOUT_S,
) -> Optional[str]:
    """Like ``read_streaming_error_body`` but returns ``None`` on empty body.

    Convenience for callers that distinguish "no body" from "empty string".
    """
    text = await read_streaming_error_body(
        response, max_bytes=max_bytes, timeout_s=timeout_s
    )
    return text or None
