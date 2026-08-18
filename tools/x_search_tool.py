#!/usr/bin/env python3
"""X Search tool backed by xAI's built-in ``x_search`` Responses API tool.

Authentication
--------------
The tool registers when **either** xAI credential path is available:

* ``XAI_API_KEY`` is set in ``~/.hermes/.env`` or the process environment
  (paid xAI API key), OR
* The user is signed in via xAI Grok OAuth — SuperGrok subscription —
  i.e. ``hermes auth add xai-oauth`` has been run and the stored refresh
  token still works.

Credential preference at call time matches
:func:`tools.xai_http.resolve_xai_http_credentials`: SuperGrok OAuth first,
direct OAuth resolver second, ``XAI_API_KEY`` last. That helper also
auto-refreshes the OAuth access token when it's within the refresh skew
window, so a ``True`` from :func:`check_x_search_requirements` means the
bearer is fetchable AND non-empty.

Defensive output
----------------
The tool surfaces two additional signals beyond xAI's raw response so callers
can tell a real citation-backed answer from an unsourced one:

* ``from_date`` / ``to_date`` are validated client-side before the HTTP call.
  Malformed (non ``YYYY-MM-DD``), inverted (``from_date > to_date``), and
  pure-future ranges (``from_date`` later than today UTC) fail fast with a
  clear error instead of burning an API call. ``to_date`` in the future is
  still allowed so callers can legitimately request "from yesterday to
  tomorrow".
* Successful responses carry ``degraded`` and ``degraded_reason`` fields.
  ``degraded`` is ``True`` when any narrowing filter (handles or dates) was
  active AND xAI returned no citations in either the top-level ``citations``
  array or the inline ``url_citation`` annotations. In that case the
  ``answer`` came from the model's own knowledge rather than the X index,
  and the caller should treat the result as unsourced.

Salvaged from PR #10786 (originally by @Jaaneek); credential resolution
reworked to honor both auth modes per Teknium's design.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, date, datetime
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

from tools.registry import registry, tool_error
from tools.xai_http import hermes_xai_user_agent, resolve_xai_http_credentials

logger = logging.getLogger(__name__)


def __getattr__(name: str):
    """Resolve httpx lazily while preserving the module patch surface."""
    if name == "httpx":
        module = _get_httpx_module()
        globals()["httpx"] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _get_httpx_module():
    import httpx as httpx_module

    return httpx_module

DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_X_SEARCH_MODEL = "grok-4.5"
DEFAULT_X_SEARCH_TIMEOUT_SECONDS = 180
DEFAULT_X_SEARCH_RETRIES = 2
X_SEARCH_REASONING_EFFORTS = ("low", "medium", "high", "xhigh")
MAX_HANDLES = 10


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


async def _load_x_search_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        return (await load_config_readonly()).get("x_search", {}) or {}
    except Exception:
        return {}


async def _get_x_search_model() -> str:
    cfg = await _load_x_search_config()
    return str(cfg.get("model") or "").strip() or DEFAULT_X_SEARCH_MODEL


async def _get_x_search_reasoning_effort() -> str | None:
    cfg = await _load_x_search_config()
    raw_value = cfg.get("reasoning_effort")
    if raw_value is None or not str(raw_value).strip():
        return None

    effort = str(raw_value).strip().lower()
    if effort not in X_SEARCH_REASONING_EFFORTS:
        allowed = ", ".join(X_SEARCH_REASONING_EFFORTS)
        raise ValueError(
            f"x_search.reasoning_effort must be one of: {allowed} "
            f"(got {raw_value!r})"
        )
    return effort


async def _get_x_search_timeout_seconds() -> int:
    cfg = await _load_x_search_config()
    raw_value = cfg.get("timeout_seconds", DEFAULT_X_SEARCH_TIMEOUT_SECONDS)
    try:
        return max(30, int(raw_value))
    except Exception:
        return DEFAULT_X_SEARCH_TIMEOUT_SECONDS


async def _get_x_search_retries() -> int:
    cfg = await _load_x_search_config()
    raw_value = cfg.get("retries", DEFAULT_X_SEARCH_RETRIES)
    try:
        return max(0, int(raw_value))
    except Exception:
        return DEFAULT_X_SEARCH_RETRIES


# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------


async def _resolve_xai_bearer() -> tuple[str, str, str]:
    """Return ``(api_key, base_url, source)``.

    ``source`` is one of ``"xai-oauth"`` or ``"xai"`` so callers (and tests)
    can tell which credential path won. Raises ``RuntimeError`` if no usable
    credential is available — the registered :func:`check_x_search_requirements`
    gate makes that case unreachable in normal operation, but the runtime
    check exists so a credential that expires between registration and
    invocation produces a clean tool error instead of a 401.
    """
    from hermes_cli.config import get_env_value_prefer_dotenv

    explicit_key = str((await get_env_value_prefer_dotenv("XAI_API_KEY")) or "").strip()
    if explicit_key:
        base_url = str(
            (await get_env_value_prefer_dotenv("XAI_BASE_URL")) or DEFAULT_XAI_BASE_URL
        ).strip().rstrip("/")
        return explicit_key, base_url, "xai"

    creds = await resolve_xai_http_credentials()
    api_key = str(creds.get("api_key") or "").strip()
    if not api_key:
        raise RuntimeError(
            "No xAI credentials available. Run `hermes auth add xai-oauth` "
            "to sign in with your SuperGrok subscription, or set XAI_API_KEY."
        )
    base_url = str(creds.get("base_url") or DEFAULT_XAI_BASE_URL).strip().rstrip("/")
    source = str(creds.get("provider") or "xai")
    return api_key, base_url, source


async def check_x_search_requirements() -> bool:
    """Return True when xAI credentials are available AND valid.

    ``resolve_xai_http_credentials`` calls the coroutine-native xAI credential
    pool, which auto-refreshes an expiring OAuth access token; a successful
    return therefore implies a usable bearer.
    """
    try:
        creds = await resolve_xai_http_credentials()
        return bool(str(creds.get("api_key") or "").strip())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_handles(handles: list[str] | None, field_name: str) -> list[str]:
    cleaned: list[str] = []
    for handle in handles or []:
        normalized = str(handle or "").strip().lstrip("@")
        if normalized:
            cleaned.append(normalized)
    if len(cleaned) > MAX_HANDLES:
        raise ValueError(f"{field_name} supports at most {MAX_HANDLES} handles")
    return cleaned


def _parse_iso_date(value: str, field_name: str) -> date:
    """Parse a strict YYYY-MM-DD string into a ``date``.

    xAI accepts any string in the ``from_date``/``to_date`` slots and silently
    returns an answer with no citations when the value is malformed or refers
    to a window where no posts can exist. That behavior burns a billable API
    call and produces a confident-sounding fluff answer that's hard for callers
    to distinguish from a real result. Validating client-side fails fast and
    gives the agent a clear error to act on.
    """
    raw = value.strip()
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{field_name} must be YYYY-MM-DD (got {raw!r})") from exc


def _validate_date_range(from_date: str, to_date: str) -> None:
    """Validate ``from_date`` / ``to_date`` before they reach xAI.

    Rules:
      * Either field, if non-empty, must parse as ``YYYY-MM-DD``.
      * When both are set, ``from_date <= to_date``.
      * ``from_date`` must not be later than today UTC — no posts can exist
        in a window that hasn't started yet, so the call would be guaranteed
        to return zero citations. ``to_date`` in the future is allowed
        (callers may legitimately set "from yesterday to tomorrow").
    """
    parsed_from: date | None = None
    parsed_to: date | None = None
    if from_date.strip():
        parsed_from = _parse_iso_date(from_date, "from_date")
    if to_date.strip():
        parsed_to = _parse_iso_date(to_date, "to_date")
    if parsed_from and parsed_to and parsed_from > parsed_to:
        raise ValueError(
            f"from_date ({parsed_from.isoformat()}) must be on or before "
            f"to_date ({parsed_to.isoformat()})"
        )
    if parsed_from is not None:
        today_utc = datetime.now(UTC).date()
        if parsed_from > today_utc:
            raise ValueError(
                f"from_date ({parsed_from.isoformat()}) is in the future; "
                f"X Search only indexes past posts (today UTC is "
                f"{today_utc.isoformat()})"
            )


def _extract_response_text(payload: dict[str, Any]) -> str:
    output_text = str(payload.get("output_text") or "").strip()
    if output_text:
        return output_text

    parts: list[str] = []
    for item in payload.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            ctype = content.get("type")
            if ctype in {"output_text", "text"}:
                text = str(content.get("text") or "").strip()
                if text:
                    parts.append(text)
    return "\n\n".join(parts).strip()


def _extract_inline_citations(payload: dict[str, Any]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for item in payload.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            for annotation in content.get("annotations", []) or []:
                if annotation.get("type") != "url_citation":
                    continue
                citations.append(
                    {
                        "url": annotation.get("url", ""),
                        "title": annotation.get("title", ""),
                        "start_index": annotation.get("start_index"),
                        "end_index": annotation.get("end_index"),
                    }
                )
    return citations


def _http_error_message(exc: httpx.HTTPStatusError) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)

    try:
        payload = response.json()
    except Exception:
        payload = None

    if isinstance(payload, dict):
        code = str(payload.get("code") or "").strip()
        error = str(payload.get("error") or "").strip()
        message = error or str(payload)
        if code and code not in message:
            message = f"{code}: {message}"
        return message or str(exc)

    text = str(getattr(response, "text", "") or "").strip()
    if text:
        return text[:500]
    return str(exc)


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------


async def x_search_tool(
    query: str,
    allowed_x_handles: list[str] | None = None,
    excluded_x_handles: list[str] | None = None,
    from_date: str = "",
    to_date: str = "",
    enable_image_understanding: bool = False,
    enable_video_understanding: bool = False,
) -> str:
    httpx = _get_httpx_module()
    if not query or not query.strip():
        return tool_error("query is required for x_search")

    try:
        api_key, base_url, source = await _resolve_xai_bearer()
    except RuntimeError as exc:
        return tool_error(str(exc))

    try:
        allowed = _normalize_handles(allowed_x_handles, "allowed_x_handles")
        excluded = _normalize_handles(excluded_x_handles, "excluded_x_handles")
        if allowed and excluded:
            return tool_error(
                "allowed_x_handles and excluded_x_handles cannot be used together"
            )

        try:
            _validate_date_range(from_date, to_date)
        except ValueError as exc:
            return tool_error(str(exc))

        try:
            reasoning_effort = await _get_x_search_reasoning_effort()
        except ValueError as exc:
            return tool_error(str(exc))

        tool_def: dict[str, Any] = {"type": "x_search"}
        if allowed:
            tool_def["allowed_x_handles"] = allowed
        if excluded:
            tool_def["excluded_x_handles"] = excluded
        if from_date.strip():
            tool_def["from_date"] = from_date.strip()
        if to_date.strip():
            tool_def["to_date"] = to_date.strip()
        if enable_image_understanding:
            tool_def["enable_image_understanding"] = True
        if enable_video_understanding:
            tool_def["enable_video_understanding"] = True

        payload: dict[str, Any] = {
            "model": await _get_x_search_model(),
            "input": [
                {
                    "role": "user",
                    "content": query.strip(),
                }
            ],
            "tools": [tool_def],
            "store": False,
        }
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}

        timeout_seconds = await _get_x_search_timeout_seconds()
        max_retries = await _get_x_search_retries()
        response: httpx.Response | None = None
        # ``requests.post`` followed redirects in upstream. Keep that behavior
        # explicit because httpx defaults to not following them.
        async with httpx.AsyncClient(follow_redirects=True) as client:
            for attempt in range(max_retries + 1):
                try:
                    response = await client.post(
                        f"{base_url}/responses",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                            "User-Agent": hermes_xai_user_agent(),
                        },
                        json=payload,
                        timeout=timeout_seconds,
                    )
                    response.raise_for_status()
                    break
                except httpx.HTTPStatusError as exc:
                    status_code = getattr(
                        getattr(exc, "response", None), "status_code", None
                    )
                    if (
                        status_code is None
                        or status_code < 500
                        or attempt >= max_retries
                    ):
                        raise
                    logger.warning(
                        "x_search upstream failure on attempt %s/%s: %s",
                        attempt + 1,
                        max_retries + 1,
                        _http_error_message(exc),
                    )
                    await asyncio.sleep(min(5.0, 1.5 * (attempt + 1)))
                except (
                    httpx.ReadTimeout,
                    httpx.ConnectTimeout,
                    httpx.NetworkError,
                    httpx.ProxyError,
                ) as exc:
                    if attempt >= max_retries:
                        raise
                    logger.warning(
                        "x_search transient failure on attempt %s/%s: %s",
                        attempt + 1,
                        max_retries + 1,
                        exc,
                    )
                    await asyncio.sleep(min(5.0, 1.5 * (attempt + 1)))

        if response is None:
            raise RuntimeError("x_search request did not return a response")

        data = response.json()

        answer = _extract_response_text(data)
        citations = list(data.get("citations") or [])
        inline_citations = _extract_inline_citations(data)

        # Degraded-result detection.
        #
        # xAI returns 200 OK with a synthesized answer even when its X index
        # has no posts matching the caller's narrowing filters. The answer
        # then comes from the model's training data, which is misleading
        # because it looks identical to a real, citation-backed result. When
        # any narrowing filter is active AND both citation channels came back
        # empty, mark the response as degraded so callers can decide to
        # broaden filters, retry, or fall back to a different source.
        active_filters: list[str] = []
        if allowed:
            active_filters.append("allowed_x_handles")
        if excluded:
            active_filters.append("excluded_x_handles")
        if from_date.strip():
            active_filters.append("from_date")
        if to_date.strip():
            active_filters.append("to_date")
        degraded = bool(active_filters) and not citations and not inline_citations
        degraded_reason = (
            f"no citations returned despite filters: {', '.join(active_filters)}"
            if degraded
            else None
        )

        return json.dumps(
            {
                "success": True,
                "provider": "xai",
                "credential_source": source,
                "tool": "x_search",
                "model": payload["model"],
                "query": query.strip(),
                "answer": answer,
                "citations": citations,
                "inline_citations": inline_citations,
                "degraded": degraded,
                "degraded_reason": degraded_reason,
            },
            ensure_ascii=False,
        )
    except httpx.HTTPStatusError as exc:
        logger.error("x_search failed: %s", exc, exc_info=True)
        return json.dumps(
            {
                "success": False,
                "provider": "xai",
                "tool": "x_search",
                "error": _http_error_message(exc),
                # Preserve the upstream requests exception name in the public
                # result while using httpx for native async transport.
                "error_type": "HTTPError",
            },
            ensure_ascii=False,
        )
    except httpx.ReadTimeout as exc:
        logger.error("x_search timed out: %s", exc, exc_info=True)
        return json.dumps(
            {
                "success": False,
                "provider": "xai",
                "tool": "x_search",
                "error": (
                    "xAI x_search timed out after "
                    f"{await _get_x_search_timeout_seconds()} seconds"
                ),
                "error_type": type(exc).__name__,
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        logger.error("x_search failed: %s", exc, exc_info=True)
        return json.dumps(
            {
                "success": False,
                "provider": "xai",
                "tool": "x_search",
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
            ensure_ascii=False,
        )


X_SEARCH_SCHEMA = {
    "name": "x_search",
    "description": (
        "Search X (Twitter) posts, profiles, and threads using xAI's built-in "
        "X Search tool. Read-only discovery only: use this for current "
        "discussion, reactions, or claims on public X rather than general web "
        "pages. Do not use it to post, reply, like, DM, upload media, delete, "
        "or inspect the user's authenticated X account — those require a "
        "separate authenticated X API surface outside this tool. Available "
        "when xAI credentials are configured (SuperGrok OAuth or XAI_API_KEY)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look up on X.",
            },
            "allowed_x_handles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of X handles to include exclusively (max 10).",
            },
            "excluded_x_handles": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of X handles to exclude (max 10).",
            },
            "from_date": {
                "type": "string",
                "description": "Optional start date in YYYY-MM-DD format.",
            },
            "to_date": {
                "type": "string",
                "description": "Optional end date in YYYY-MM-DD format.",
            },
            "enable_image_understanding": {
                "type": "boolean",
                "description": "Whether xAI should analyze images attached to matching X posts.",
                "default": False,
            },
            "enable_video_understanding": {
                "type": "boolean",
                "description": "Whether xAI should analyze videos attached to matching X posts.",
                "default": False,
            },
        },
        "required": ["query"],
    },
}


async def _handle_x_search(args, **kw):
    return await x_search_tool(
        query=args.get("query", ""),
        allowed_x_handles=args.get("allowed_x_handles"),
        excluded_x_handles=args.get("excluded_x_handles"),
        from_date=args.get("from_date", ""),
        to_date=args.get("to_date", ""),
        enable_image_understanding=bool(args.get("enable_image_understanding", False)),
        enable_video_understanding=bool(args.get("enable_video_understanding", False)),
    )


registry.register(
    name="x_search",
    toolset="x_search",
    schema=X_SEARCH_SCHEMA,
    handler=_handle_x_search,
    check_fn=check_x_search_requirements,
    requires_env=["XAI_API_KEY"],
    emoji="🐦",
    max_result_size_chars=100_000,
)
