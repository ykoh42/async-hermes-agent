"""Helper functions for the chat-completions code path.

Extracted from :class:`AIAgent` for cleanliness — bodies of the
non-streaming API call, request kwargs builder, assistant-message
materializer, provider-fallback activator, max-iterations handler,
and per-turn resource cleanup.

Each function takes the parent ``AIAgent`` as its first argument
(``agent``).  :class:`AIAgent` keeps thin forwarder methods so call
sites remain unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
import uuid
from types import SimpleNamespace
from typing import Any, Dict, Optional

from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_context import substitute_api_content
from agent.message_content import flatten_message_text
from agent.message_sanitization import (
    _repair_tool_call_arguments,
    _sanitize_surrogates,
)
from agent.stream_single_writer import (  # noqa: F401 - upstream re-export
    claim_stream_writer,
    stream_writer_is_current,
)
from hermes_constants import FINISH_REASON_LENGTH, PARTIAL_STREAM_STUB_ID
from tools.terminal_tool import cleanup_vm, is_persistent_env
from utils import base_url_host_matches, base_url_hostname, env_float, env_int

logger = logging.getLogger(__name__)
_OPENROUTER_PROVIDER_SORT_VALUES = {"throughput", "latency", "price"}
_STREAM_HEARTBEAT_INTERVAL = 30.0

# When the fallback chain is fully exhausted on a non-rate-limit failure
# (e.g. every provider returns a non-retryable client error like HTTP 400),
# arm a short cooldown so the NEXT turn's restore_primary_runtime stays gated
# and does not reset _fallback_index=0 to replay the entire chain again.
# Without this, a client/gateway that re-submits immediately would re-marshal
# the full (potentially 80k-token) context once per provider every turn and
# can drive a constrained host into memory/swap exhaustion.  Rate-limit /
# billing reasons keep their own 60s cooldown (set above); this is the
# narrower non-rate-limit case.  See issue #24996.
_FALLBACK_EXHAUSTED_COOLDOWN_S = 5.0


def estimate_request_context_tokens(api_payload: Any) -> int:
    """Estimate context/load tokens from an API payload, dict or messages list.

    The stale-call detectors historically assumed a Chat Completions request:
    they pulled ``api_kwargs["messages"]`` and ran a cheap char/4 estimate.
    Codex / Responses API requests carry the conversational payload in
    ``input`` (with additional load in ``instructions`` and ``tools``), so the
    legacy estimator reported ~0 tokens for every Codex turn and the
    context-tier scaling never fired.

    This helper handles both shapes:
      - bare list -> treat as Chat Completions ``messages``
      - dict with ``messages`` -> Chat Completions (+ ``tools`` if present)
      - dict with ``input`` -> Responses API (+ ``instructions``/``tools``)
      - any other dict -> fall back to summing string values
    """

    def _chars(value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, str):
            return len(value)
        return len(str(value))

    def _message_chars(messages: Any) -> int:
        if not isinstance(messages, list):
            return _chars(messages)
        return sum(_chars(item) for item in messages)

    if isinstance(api_payload, list):
        return _message_chars(api_payload) // 4

    if isinstance(api_payload, dict):
        messages = api_payload.get("messages")
        if isinstance(messages, list):
            total_chars = _message_chars(messages)
            if "tools" in api_payload:
                total_chars += _chars(api_payload.get("tools"))
            return total_chars // 4

        if "input" in api_payload:
            total_chars = (
                _chars(api_payload.get("input"))
                + _chars(api_payload.get("instructions"))
                + _chars(api_payload.get("tools"))
            )
            return total_chars // 4

        return sum(_chars(value) for value in api_payload.values()) // 4

    return _chars(api_payload) // 4


def _is_openai_codex_backend(agent) -> bool:
    base_url_lower = str(getattr(agent, "_base_url_lower", "") or "")
    base_url_hostname = str(getattr(agent, "_base_url_hostname", "") or "")
    return (
        getattr(agent, "provider", None) == "openai-codex"
        or (
            base_url_hostname == "chatgpt.com"
            and "/backend-api/codex" in base_url_lower
        )
    )


def openai_codex_stale_timeout_floor(est_tokens: int) -> float:
    """Minimum wall-clock stale timeout for openai-codex by estimated context.

    Gateway/Telegram sessions routinely ship ~15–25k tokens of tools +
    instructions before the first user message. Subscription-backed Codex can
    legitimately spend several minutes in backend admission/prefill at that
    size; the generic 90s non-stream stale default aborts healthy calls. The
    floor engages above 10k estimated tokens so those gateway-scale payloads
    are covered; smaller requests keep the generic default.
    """
    if est_tokens > 100_000:
        return 1200.0
    if est_tokens > 50_000:
        return 900.0
    if est_tokens > 10_000:
        return 600.0
    return 0.0


def _validated_openrouter_provider_sort(raw_sort: Any) -> Optional[str]:
    """Return a normalized OpenRouter provider.sort value or None."""
    if not isinstance(raw_sort, str):
        return None
    sort_value = raw_sort.strip().lower()
    if not sort_value:
        return None
    if sort_value in _OPENROUTER_PROVIDER_SORT_VALUES:
        return sort_value
    logger.warning(
        "Ignoring invalid OpenRouter provider.sort value %r (allowed: %s)",
        raw_sort,
        ", ".join(sorted(_OPENROUTER_PROVIDER_SORT_VALUES)),
    )
    return None


def _provider_preferences_for_agent(agent) -> Dict[str, Any]:
    """Build the validated provider-routing object shared by request paths."""
    preferences: Dict[str, Any] = {}
    if agent.providers_allowed:
        preferences["only"] = agent.providers_allowed
    if agent.providers_ignored:
        preferences["ignore"] = agent.providers_ignored
    if agent.providers_order:
        preferences["order"] = agent.providers_order
    provider_sort = _validated_openrouter_provider_sort(agent.provider_sort)
    if provider_sort:
        preferences["sort"] = provider_sort
    if agent.provider_require_parameters:
        preferences["require_parameters"] = True
    if agent.provider_data_collection:
        preferences["data_collection"] = agent.provider_data_collection
    return preferences


def _merge_nous_portal_messages_extra_body(agent, anthropic_kwargs: dict) -> dict:
    """Merge Portal ``tags`` / ``session_id`` onto an Anthropic Messages kwargs dict.

    The Nous provider profile is only consulted by the OpenAI-wire transport;
    anthropic_messages callers must merge it themselves. Passes ``session_id``
    only — not ``provider_preferences`` (those become a top-level ``provider``
    routing object on the OpenAI wire). Never blocks a turn on tagging.
    """
    if getattr(agent, "provider", None) not in {"nous", "nous-portal", "nousresearch"}:
        return anthropic_kwargs
    try:
        from providers import _get_provider_profile_cached

        nous_profile = _get_provider_profile_cached("nous")
        if nous_profile is not None:
            anthropic_kwargs.setdefault("extra_body", {}).update(
                nous_profile.build_extra_body(
                    session_id=getattr(agent, "session_id", None)
                )
            )
    except Exception as exc:  # noqa: BLE001 — never block a turn on tagging
        logger.debug("Nous Portal extra_body merge failed: %s", exc)
    return anthropic_kwargs


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _estimate_chunk_bytes(chunk: Any) -> int:
    """Cheap per-chunk size estimate for the stream diagnostic counters.

    The previous implementation used ``len(repr(chunk))`` — a full recursive
    repr of a pydantic model on EVERY streaming chunk (5.5-8.8 µs each,
    ~20-30 ms of pure CPU on a 3,000-chunk response, in the hottest loop in
    the agent). The counter only feeds a retry-diagnostic log line, so an
    estimate based on the delta payload lengths is plenty (2.1-2.4 µs, ~3x
    cheaper, and independent of model/pydantic field count). Chat Completions
    chunks are sized from their delta content/reasoning/tool-argument strings
    plus a small framing constant; anything shape-unknown (Anthropic events,
    stub providers) falls back to a flat constant so `bytes` stays monotonic
    and roughly proportional to traffic.
    """
    size = 40  # SSE/JSON framing floor per chunk
    try:
        choices = getattr(chunk, "choices", None)
        if choices:
            delta = getattr(choices[0], "delta", None)
            if delta is not None:
                for attr in ("content", "reasoning_content", "reasoning"):
                    v = getattr(delta, attr, None)
                    if isinstance(v, str):
                        size += len(v)
                tool_calls = getattr(delta, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        fn = getattr(tc, "function", None)
                        if fn is not None:
                            args = getattr(fn, "arguments", None)
                            if isinstance(args, str):
                                size += len(args)
                            name = getattr(fn, "name", None)
                            if isinstance(name, str):
                                size += len(name)
        else:
            # Non-chat-completions shapes (Anthropic events etc.): try the
            # common text fields, else keep the framing floor.
            for attr in ("text", "partial_json"):
                v = getattr(getattr(chunk, "delta", None), attr, None)
                if isinstance(v, str):
                    size += len(v)
    except Exception:
        pass
    return size


def _codex_wait_notice_recovery(
    *,
    stale_timeout: float,
    ttfb_enabled: bool,
    ttfb_timeout: float,
    last_event_ts: Optional[float],
    call_start: float,
    idle_enabled: bool,
    idle_timeout: float,
    elapsed: float,
) -> str:
    """Describe the earliest enabled Codex watchdog on the call timeline."""
    deadlines: list[float] = []
    if math.isfinite(stale_timeout):
        deadlines.append(stale_timeout)
    if last_event_ts is None:
        if ttfb_enabled and math.isfinite(ttfb_timeout):
            deadlines.append(ttfb_timeout)
    elif idle_enabled and math.isfinite(idle_timeout):
        deadlines.append(max(0.0, last_event_ts - call_start) + idle_timeout)
    if not deadlines or min(deadlines) <= elapsed:
        return ""
    return f"; auto-reconnect at {int(min(deadlines))}s"


# ── Cross-turn stale-call circuit breaker (#58962) ─────────────────────
# A session wedged against an unresponsive provider hits the stale detector
# on every call and loops forever (observed: 494 consecutive failures over
# 3+ days, each burning the full stale timeout × retries with no response).
# The agent carries ``_consecutive_stale_streams``: incremented on every
# stale kill, reset only when a call actually completes (or when the
# provider is swapped — switch_model / try_activate_fallback /
# restore_primary_runtime — since the streak measured the OLD provider).
# Past the give-up threshold, calls abort immediately with an actionable
# error instead of re-waiting out the stale timeout.

def _stale_streak(agent) -> int:
    try:
        return int(getattr(agent, "_consecutive_stale_streams", 0) or 0)
    except Exception:
        return 0


def _bump_stale_streak(agent) -> None:
    try:
        agent._consecutive_stale_streams = _stale_streak(agent) + 1
    except Exception:
        pass


def _reset_stale_streak(agent) -> None:
    try:
        agent._consecutive_stale_streams = 0
    except Exception:
        pass


def _check_stale_giveup(agent) -> None:
    """Raise immediately when the consecutive-stale streak is past the
    give-up threshold — no network attempt, no stale-timeout wait."""
    _giveup = env_int("HERMES_STREAM_STALE_GIVEUP", 5)
    _streak = _stale_streak(agent)
    if _giveup > 0 and _streak >= _giveup:
        raise RuntimeError(
            "Provider has been unresponsive (no response received) for "
            f"{_streak} consecutive stale attempts — aborting this call to "
            "avoid an indefinite stall. Switch models or start a new "
            "session, then retry."
        )


def should_use_direct_api_call(agent: Any) -> bool:
    """Return True because the native async runtime never spawns API workers."""
    del agent
    return True


async def direct_api_call(agent: Any, api_kwargs: dict) -> Any:
    """Execute a non-streaming provider request directly on the event loop."""
    return await interruptible_api_call(agent, api_kwargs)


async def _interruptible_codex_api_call(agent: Any, api_kwargs: dict) -> Any:
    """Run the upstream Codex internal stream with its three watchdogs."""
    compute_timeout = getattr(agent, "_compute_non_stream_stale_timeout", None)
    stale_timeout = (
        await compute_timeout(api_kwargs)
        if callable(compute_timeout)
        else float("inf")
    )
    estimated_tokens = estimate_request_context_tokens(api_kwargs)
    openai_codex_backend = _is_openai_codex_backend(agent)
    if openai_codex_backend:
        stale_timeout = max(
            stale_timeout,
            openai_codex_stale_timeout_floor(estimated_tokens),
        )
        hard_timeout = env_float("HERMES_CODEX_HARD_TIMEOUT_SECONDS", 1500.0)
        if hard_timeout > 0:
            stale_timeout = min(stale_timeout, hard_timeout)

    if estimated_tokens > 100_000:
        idle_timeout_default = 180.0
    elif estimated_tokens > 50_000:
        idle_timeout_default = 120.0
    elif estimated_tokens > 10_000:
        idle_timeout_default = 60.0
    else:
        idle_timeout_default = 12.0

    ttfb_timeout = env_float("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", 120.0)
    ttfb_enabled = ttfb_timeout > 0
    if ttfb_enabled and openai_codex_backend:
        disable_above = env_float(
            "HERMES_CODEX_TTFB_DISABLE_ABOVE_TOKENS", 10_000.0
        )
        strict = os.environ.get("HERMES_CODEX_TTFB_STRICT", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if (
            not strict
            and disable_above > 0
            and estimated_tokens >= disable_above
            and ttfb_timeout < idle_timeout_default
        ):
            logger.info(
                "Scaling openai-codex no-byte TTFB watchdog from %.0fs to %.0fs "
                "for large request (context=~%s tokens >= %.0f). Set "
                "HERMES_CODEX_TTFB_STRICT=1 to keep the smaller cutoff.",
                ttfb_timeout,
                idle_timeout_default,
                f"{estimated_tokens:,}",
                disable_above,
            )
            ttfb_timeout = idle_timeout_default
        ttfb_cap = env_float("HERMES_CODEX_TTFB_MAX_SECONDS", 120.0)
        if ttfb_cap > 0 and ttfb_timeout > ttfb_cap:
            logger.info(
                "Capping openai-codex no-byte TTFB timeout from %.0fs to %.0fs "
                "(context=~%s tokens). Set HERMES_CODEX_TTFB_MAX_SECONDS to tune.",
                ttfb_timeout,
                ttfb_cap,
                f"{estimated_tokens:,}",
            )
            ttfb_timeout = ttfb_cap

    idle_timeout = env_float(
        "HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS",
        idle_timeout_default,
    )
    idle_enabled = idle_timeout > 0

    agent._codex_stream_last_event_ts = None
    agent._codex_stream_last_progress_ts = None
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    stale_deadline = (
        started_at + stale_timeout if math.isfinite(stale_timeout) else math.inf
    )
    deadline_kind = "stale"
    initial_deadline = stale_deadline
    if ttfb_enabled and started_at + ttfb_timeout <= initial_deadline:
        initial_deadline = started_at + ttfb_timeout
        deadline_kind = "ttfb"
    timeout_scope = asyncio.timeout_at(
        initial_deadline if math.isfinite(initial_deadline) else None
    )

    def _note_codex_event() -> None:
        nonlocal deadline_kind
        now_wall = time.time()
        agent._codex_stream_last_event_ts = now_wall
        agent._codex_stream_last_progress_ts = now_wall
        next_deadline = stale_deadline
        deadline_kind = "stale"
        if idle_enabled and loop.time() + idle_timeout < next_deadline:
            next_deadline = loop.time() + idle_timeout
            deadline_kind = "idle"
        timeout_scope.reschedule(
            next_deadline if math.isfinite(next_deadline) else None
        )

    touch_activity = getattr(agent, "_touch_activity", None)
    if callable(touch_activity):
        touch_activity("waiting for non-streaming API response")

    heartbeat_stop = asyncio.Event()

    async def _heartbeat() -> None:
        while not heartbeat_stop.is_set():
            try:
                async with asyncio.timeout(_STREAM_HEARTBEAT_INTERVAL):
                    await heartbeat_stop.wait()
                return
            except TimeoutError:
                elapsed = loop.time() - started_at
                try:
                    recovery = _codex_wait_notice_recovery(
                        stale_timeout=stale_timeout,
                        ttfb_enabled=ttfb_enabled,
                        ttfb_timeout=ttfb_timeout,
                        last_event_ts=getattr(
                            agent, "_codex_stream_last_event_ts", None
                        ),
                        call_start=time.time() - elapsed,
                        idle_enabled=idle_enabled,
                        idle_timeout=idle_timeout,
                        elapsed=elapsed,
                    )
                    emit_wait = getattr(agent, "_emit_wait_notice", None)
                    if callable(emit_wait):
                        emit_wait(
                            "⏳ waiting on "
                            f"{api_kwargs.get('model', 'the provider')} — "
                            f"{int(elapsed)}s with no response yet (provider may "
                            f"be slow or overloaded{recovery})"
                        )
                except Exception:
                    logger.debug(
                        "wait-notice construction failed", exc_info=True
                    )

    heartbeat_task = asyncio.create_task(
        _heartbeat(), name="codex-nonstream-heartbeat"
    )
    try:
        try:
            async with timeout_scope:
                response = await agent._execute_model_request(
                    api_kwargs,
                    use_streaming=True,
                    on_first_delta=getattr(agent, "_codex_on_first_delta", None),
                    on_stream_activity=_note_codex_event,
                )
        finally:
            heartbeat_stop.set()
            await _finish_stream_heartbeat(heartbeat_task)
    except TimeoutError:
        if not timeout_scope.expired():
            raise
        elapsed = loop.time() - started_at
        model = api_kwargs.get("model", "unknown")
        buffer_status = getattr(agent, "_buffer_status", None)
        emit_wait = getattr(agent, "_emit_wait_notice", None)
        silent_hint = None
        hint = getattr(agent, "_codex_silent_hang_hint", None)
        if callable(hint):
            try:
                silent_hint = hint(model=api_kwargs.get("model"))
            except Exception:
                pass

        if deadline_kind == "ttfb":
            logger.warning(
                "Codex stream produced no bytes within TTFB cutoff "
                "(%.0fs > %.0fs, model=%s). Backend accepted the connection "
                "but sent no stream events. Killing connection so the retry "
                "loop can reconnect.",
                elapsed,
                ttfb_timeout,
                model,
            )
            message = (
                f"⚠️ No first byte from provider in {int(elapsed)}s "
                f"(codex stream, model: {model}). Reconnecting."
            )
            if silent_hint:
                message += f" {silent_hint}"
            if callable(buffer_status):
                buffer_status(message)
            if callable(emit_wait):
                emit_wait(
                    f"⚠ no response from provider in {int(elapsed)}s — "
                    "reconnecting..."
                )
            if callable(touch_activity):
                touch_activity(
                    f"codex stream killed after {int(elapsed)}s with no first byte"
                )
            error = (
                f"Codex stream produced no bytes within {int(elapsed)}s "
                f"(TTFB threshold: {int(ttfb_timeout)}s)"
            )
            if silent_hint:
                error += f". {silent_hint}"
            raise TimeoutError(error) from None

        if deadline_kind == "idle":
            last_event = getattr(agent, "_codex_stream_last_event_ts", None)
            idle_elapsed = (
                max(0.0, time.time() - last_event)
                if isinstance(last_event, (int, float))
                else idle_timeout
            )
            logger.warning(
                "Codex stream produced no SSE events for %.0fs after first byte "
                "(threshold %.0fs, model=%s, context=~%s tokens). Killing "
                "connection so the retry loop can reconnect.",
                idle_elapsed,
                idle_timeout,
                model,
                f"{estimated_tokens:,}",
            )
            if callable(buffer_status):
                buffer_status(
                    f"⚠️ Codex stream sent no events for {int(idle_elapsed)}s "
                    f"after first byte (model: {model}). Reconnecting."
                )
            if callable(touch_activity):
                touch_activity(
                    "codex stream killed after "
                    f"{int(idle_elapsed)}s with no SSE events"
                )
            raise TimeoutError(
                "Codex stream produced no SSE events for "
                f"{int(idle_elapsed)}s after first byte "
                f"(threshold: {int(idle_timeout)}s)"
            ) from None

        _bump_stale_streak(agent)
        logger.warning(
            "Non-streaming API call stale for %.0fs (threshold %.0fs). "
            "model=%s context=~%s tokens. Killing connection.",
            elapsed,
            stale_timeout,
            model,
            f"{estimated_tokens:,}",
        )
        message = (
            f"⚠️ No response from provider for {int(elapsed)}s "
            f"(non-streaming, model: {model})."
        )
        message += f" {silent_hint}" if silent_hint else " Aborting call."
        if callable(buffer_status):
            buffer_status(message)
        if callable(touch_activity):
            touch_activity(
                f"stale non-streaming call killed after {int(elapsed)}s"
            )
        error = (
            f"Non-streaming API call timed out after {int(elapsed)}s "
            f"with no response (threshold: {int(stale_timeout)}s)"
        )
        if silent_hint:
            error += f". {silent_hint}"
        raise TimeoutError(error) from None

    _reset_stale_streak(agent)
    return response


async def interruptible_api_call(agent: Any, api_kwargs: dict) -> Any:
    """Execute a bounded native-async non-streaming provider request."""
    _check_stale_giveup(agent)
    if getattr(agent, "api_mode", None) == "codex_responses":
        return await _interruptible_codex_api_call(agent, api_kwargs)
    touch_activity = getattr(agent, "_touch_activity", None)
    if callable(touch_activity):
        touch_activity("waiting for non-streaming API response")
    compute_timeout = getattr(agent, "_compute_non_stream_stale_timeout", None)
    timeout = await compute_timeout(api_kwargs) if callable(compute_timeout) else None
    timeout_scope = asyncio.timeout(
        timeout if timeout is not None and math.isfinite(timeout) else None
    )
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    heartbeat_stop = asyncio.Event()

    async def _heartbeat() -> None:
        while not heartbeat_stop.is_set():
            try:
                async with asyncio.timeout(_STREAM_HEARTBEAT_INTERVAL):
                    await heartbeat_stop.wait()
                return
            except TimeoutError:
                elapsed = loop.time() - started_at
                try:
                    emit_wait = getattr(agent, "_emit_wait_notice", None)
                    if callable(emit_wait):
                        emit_wait(
                            "⏳ waiting on "
                            f"{api_kwargs.get('model', 'the provider')} — "
                            f"{int(elapsed)}s with no response yet (provider may "
                            "be slow or overloaded)"
                        )
                except Exception:
                    logger.debug(
                        "wait-notice construction failed", exc_info=True
                    )

    heartbeat_task = asyncio.create_task(
        _heartbeat(), name="provider-nonstream-heartbeat"
    )
    try:
        try:
            async with timeout_scope:
                response = await agent._execute_model_request(api_kwargs)
        finally:
            heartbeat_stop.set()
            await _finish_stream_heartbeat(heartbeat_task)
    except TimeoutError:
        if not timeout_scope.expired():
            raise
        _bump_stale_streak(agent)
        if callable(touch_activity):
            touch_activity(
                f"stale non-streaming call killed after {int(timeout or 0)}s"
            )
        raise TimeoutError(
            f"Non-streaming API call timed out after {int(timeout or 0)}s "
            f"with no response (threshold: {int(timeout or 0)}s)."
        ) from None
    _reset_stale_streak(agent)
    return response


def _finalize_chat_stream(accumulator: Any) -> Any:
    """Build upstream's response shape from a native async chat stream."""
    from agent.errors import EmptyStreamError

    if (
        accumulator.finish_reason is None
        and not accumulator.content_parts
        and not accumulator.reasoning_parts
        and not accumulator.tool_calls_acc
    ):
        raise EmptyStreamError(
            "Provider returned an empty stream with no finish_reason "
            "(possible upstream error or malformed SSE response)."
        )

    tool_calls = None
    has_truncated_tool_args = False
    if accumulator.tool_calls_acc:
        tool_calls = []
        for index in sorted(accumulator.tool_calls_acc):
            entry = accumulator.tool_calls_acc[index]
            arguments = "".join(entry["arguments"])
            tool_name = entry["name"] or "?"
            if arguments and arguments.strip():
                try:
                    json.loads(arguments)
                except json.JSONDecodeError:
                    repaired = _repair_tool_call_arguments(arguments, tool_name)
                    if repaired != "{}":
                        arguments = repaired
                    else:
                        has_truncated_tool_args = True
            tool_calls.append(
                SimpleNamespace(
                    id=entry["id"],
                    type="function",
                    extra_content=entry.get("extra_content"),
                    function=SimpleNamespace(
                        name=entry["name"],
                        arguments=arguments,
                    ),
                )
            )

    content = "".join(accumulator.content_parts) or None
    reasoning = "".join(accumulator.reasoning_parts) or None
    dropped_without_finish = (
        has_truncated_tool_args and accumulator.finish_reason is None
    )
    text_without_finish = (
        accumulator.finish_reason is None
        and bool(accumulator.content_parts)
        and not accumulator.tool_calls_acc
    )
    if dropped_without_finish or text_without_finish:
        dropped_names = None
        if dropped_without_finish:
            dropped_names = [
                accumulator.tool_calls_acc[index]["name"] or "?"
                for index in sorted(accumulator.tool_calls_acc)
            ]
            logger.warning(
                "Stream ended with no finish_reason while a tool call's "
                "arguments were still incomplete (tools=%s); treating as a "
                "mid-tool-call stream drop, not an output-length truncation.",
                dropped_names,
            )
        else:
            logger.warning(
                "Stream ended with no finish_reason after delivering text "
                "with no tool calls; treating as a mid-stream drop."
            )
        return SimpleNamespace(
            id=PARTIAL_STREAM_STUB_ID,
            model=accumulator.resp_model,
            choices=[
                SimpleNamespace(
                    index=0,
                    message=SimpleNamespace(
                        role="assistant",
                        content=content,
                        tool_calls=None,
                        reasoning_content=reasoning,
                    ),
                    finish_reason=FINISH_REASON_LENGTH,
                )
            ],
            usage=accumulator.usage,
            _dropped_tool_names=dropped_names,
        )

    finish_reason = accumulator.finish_reason or "stop"
    if has_truncated_tool_args:
        finish_reason = FINISH_REASON_LENGTH
    return SimpleNamespace(
        id="stream-" + str(uuid.uuid4()),
        model=accumulator.resp_model,
        choices=[
            SimpleNamespace(
                index=0,
                message=SimpleNamespace(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                    reasoning_content=reasoning,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=accumulator.usage,
    )


def _partial_stream_stub(
    agent: Any,
    error: Exception | None = None,
    *,
    delivered: bool = False,
) -> Any | None:
    """Return upstream's length-truncated stub after visible streamed text."""
    raw_content = str(
        getattr(agent, "_current_streamed_assistant_text", "") or ""
    )
    content = raw_content.strip() or None
    # Upstream only recovers a partial stream after visible text was already
    # delivered.  A tool-only failure is retried/raised; fabricating a warning
    # response for that case would change retry and trajectory behavior.
    if not delivered and not raw_content:
        return None
    dropped_tool_names = list(
        dict.fromkeys(
            str(name)
            for name in getattr(
                agent, "_current_stream_partial_tool_names", []
            )
            if name
        )
    )
    if dropped_tool_names:
        names = ", ".join(dropped_tool_names[:3])
        if len(dropped_tool_names) > 3:
            names += f", +{len(dropped_tool_names) - 3} more"
        warning = (
            f"\n\n⚠ Stream stalled mid tool-call ({names}); the action was not "
            "executed. Ask me to retry if you want to continue."
        )
        content = (content or "") + warning
        fire_delta = getattr(agent, "_fire_stream_delta", None)
        if callable(fire_delta):
            try:
                fire_delta(warning)
            except Exception:
                pass
    stub = SimpleNamespace(
        id=PARTIAL_STREAM_STUB_ID,
        model=getattr(agent, "model", "unknown"),
        choices=[
            SimpleNamespace(
                index=0,
                message=SimpleNamespace(
                    role="assistant",
                    content=content,
                    tool_calls=None,
                    reasoning_content=None,
                ),
                finish_reason="length",
            )
        ],
        usage=None,
        _dropped_tool_names=dropped_tool_names or None,
    )
    if error is not None:
        try:
            classified = classify_api_error(
                error,
                provider=str(getattr(agent, "provider", "") or ""),
                model=str(getattr(agent, "model", "") or ""),
            )
            if classified.reason == FailoverReason.content_policy_blocked:
                stub._content_filter_terminated = True
        except Exception:
            pass
    return stub


async def _finish_stream_heartbeat(task: asyncio.Task[Any]) -> None:
    """Finish an owned heartbeat task through repeated caller cancellation."""
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            await asyncio.shield(task)
            break
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised below
            if task.cancelled():
                raise
            if cancellation is None:
                cancellation = exc
        except Exception as exc:
            if cancellation is not None:
                raise cancellation from exc
            raise
    if cancellation is not None:
        raise cancellation


def _stream_request_timeout(agent: Any, stale_timeout: float) -> Any:
    """Build upstream's OpenAI-wire connect/read/write/pool timeout policy."""
    import httpx

    configured_timeout = getattr(agent, "_provider_request_timeout", None)
    base_timeout = (
        configured_timeout
        if configured_timeout is not None
        else env_float("HERMES_API_TIMEOUT", 1800.0)
    )
    if configured_timeout is not None:
        read_timeout = configured_timeout
        connection_timeout = min(base_timeout, 60.0)
    else:
        read_timeout = env_float("HERMES_STREAM_READ_TIMEOUT", 120.0)
        if read_timeout == 120.0 and getattr(agent, "base_url", None):
            from agent.model_metadata import is_local_endpoint

            if is_local_endpoint(agent.base_url):
                read_timeout = base_timeout
                logger.debug(
                    "Local provider detected (%s) — stream read timeout raised to %.0fs",
                    agent.base_url,
                    read_timeout,
                )
        if (
            read_timeout == 120.0
            and math.isfinite(stale_timeout)
            and stale_timeout > read_timeout
        ):
            read_timeout = stale_timeout
            logger.debug(
                "Cloud reasoning stream — read timeout raised to %.0fs to "
                "match stale-stream detector",
                read_timeout,
            )
        connection_timeout = 30.0
    return httpx.Timeout(
        connect=connection_timeout,
        read=read_timeout,
        write=base_timeout,
        pool=connection_timeout,
    )


async def interruptible_streaming_api_call(
    agent: Any,
    api_kwargs: dict,
    *,
    on_first_delta=None,
) -> Any:
    """Execute a native stream with upstream retry and stale-call policy."""
    _check_stale_giveup(agent)
    timeout = await _derive_stream_stale_timeout(agent, api_kwargs)
    loop = asyncio.get_running_loop()
    max_stream_retries = env_int("HERMES_STREAM_RETRIES", 2)

    for stream_attempt in range(max_stream_retries + 1):
        if getattr(agent, "_interrupt_requested", False):
            raise InterruptedError("Agent interrupted before stream retry")

        init_diag = getattr(agent, "_stream_diag_init", None)
        diag = init_diag() if callable(init_diag) else None
        timeout_scope = asyncio.timeout(
            timeout if math.isfinite(timeout) else None
        )
        last_stream_activity = loop.time()
        heartbeat_stop = asyncio.Event()
        text_delivery = {"yes": False}

        def _note_stream_activity() -> None:
            nonlocal last_stream_activity
            last_stream_activity = loop.time()
            if math.isfinite(timeout):
                timeout_scope.reschedule(loop.time() + timeout)

        def _note_stream_text() -> None:
            text_delivery["yes"] = True

        async def _heartbeat() -> None:
            while not heartbeat_stop.is_set():
                try:
                    async with asyncio.timeout(_STREAM_HEARTBEAT_INTERVAL):
                        await heartbeat_stop.wait()
                    return
                except TimeoutError:
                    waiting_elapsed = loop.time() - last_stream_activity
                    waiting_seconds = int(waiting_elapsed)
                    if waiting_elapsed >= _STREAM_HEARTBEAT_INTERVAL:
                        recovery = (
                            f"; auto-reconnect at {int(timeout)}s"
                            if math.isfinite(timeout)
                            else ""
                        )
                        emit_wait = getattr(agent, "_emit_wait_notice", None)
                        if callable(emit_wait):
                            emit_wait(
                                "⏳ waiting on "
                                f"{api_kwargs.get('model', 'the provider')} — "
                                f"{waiting_seconds}s with no output yet "
                                "(provider may be slow or overloaded, or the "
                                f"model is thinking{recovery})"
                            )
                    else:
                        touch_activity = getattr(agent, "_touch_activity", None)
                        if callable(touch_activity):
                            touch_activity(
                                "waiting for stream response "
                                f"({waiting_seconds}s, no chunks yet)"
                            )

        heartbeat_task = asyncio.create_task(
            _heartbeat(),
            name=f"provider-stream-heartbeat-{stream_attempt + 1}",
        )
        attempt_api_kwargs = api_kwargs
        if getattr(agent, "api_mode", None) == "chat_completions":
            attempt_api_kwargs = dict(api_kwargs)
            attempt_api_kwargs["timeout"] = _stream_request_timeout(
                agent,
                timeout,
            )
        try:
            try:
                async with timeout_scope:
                    response = await agent._execute_model_request(
                        attempt_api_kwargs,
                        use_streaming=True,
                        on_first_delta=on_first_delta,
                        on_stream_activity=_note_stream_activity,
                        _on_stream_text=_note_stream_text,
                        _stream_diag=diag,
                    )
            finally:
                heartbeat_stop.set()
                await _finish_stream_heartbeat(heartbeat_task)
        except asyncio.CancelledError:
            raise
        except InterruptedError:
            raise
        except Exception as caught:
            if getattr(agent, "_interrupt_requested", False):
                raise InterruptedError(
                    "Agent interrupted during streaming API call (post-worker)"
                ) from None
            error: Exception = caught
            watchdog_timeout = (
                isinstance(caught, TimeoutError) and timeout_scope.expired()
            )
            if watchdog_timeout:
                _bump_stale_streak(agent)
                stale_elapsed = timeout
                estimated_context = estimate_request_context_tokens(api_kwargs)
                logger.warning(
                    "Stream stale for %.0fs (threshold %.0fs) — no chunks "
                    "received. model=%s context=~%s tokens. Killing "
                    "connection.",
                    stale_elapsed,
                    timeout,
                    api_kwargs.get("model", "unknown"),
                    f"{estimated_context:,}",
                )
                buffer_status = getattr(agent, "_buffer_status", None)
                if callable(buffer_status):
                    buffer_status(
                        "⚠️ No response from provider for "
                        f"{int(stale_elapsed)}s (model: "
                        f"{api_kwargs.get('model', 'unknown')}, context: "
                        f"~{estimated_context:,} tokens). Reconnecting..."
                    )
                emit_wait = getattr(agent, "_emit_wait_notice", None)
                if callable(emit_wait):
                    emit_wait(
                        "⚠ no output from provider for "
                        f"{int(stale_elapsed)}s — reconnecting..."
                    )
                touch_activity = getattr(agent, "_touch_activity", None)
                if callable(touch_activity):
                    touch_activity(
                        f"stale stream detected after {int(timeout)}s, "
                        "reconnecting"
                    )
                error = TimeoutError(
                    f"Streaming API call produced no chunks for {int(timeout)}s "
                    f"(threshold: {int(timeout)}s)."
                )

            import httpx
            from agent.errors import EmptyStreamError

            is_timeout = watchdog_timeout or isinstance(
                error,
                (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.PoolTimeout),
            )
            is_connection = isinstance(
                error,
                (httpx.ConnectError, httpx.RemoteProtocolError, ConnectionError),
            )
            parse_check = getattr(agent, "_is_provider_stream_parse_error", None)
            is_parse_error = bool(
                callable(parse_check) and parse_check(error)
            )
            is_empty_stream = isinstance(error, EmptyStreamError)
            is_sse_connection = False
            if not is_timeout and not is_connection:
                from openai import APIError

                if isinstance(error, APIError) and not getattr(
                    error,
                    "status_code",
                    None,
                ):
                    message = str(error).lower()
                    is_sse_connection = any(
                        phrase in message
                        for phrase in (
                            "connection lost",
                            "connection reset",
                            "connection closed",
                            "connection terminated",
                            "network error",
                            "network connection",
                            "terminated",
                            "peer closed",
                            "broken pipe",
                            "upstream connect error",
                        )
                    )
            transient = (
                is_timeout
                or is_connection
                or is_sse_connection
                or is_parse_error
                or is_empty_stream
            )

            partial_raw_text = str(
                getattr(agent, "_current_streamed_assistant_text", "") or ""
            )
            partial_text_delivered = (
                text_delivery["yes"] or bool(partial_raw_text)
            )
            partial_tool_names = list(
                getattr(agent, "_current_stream_partial_tool_names", []) or []
            )
            can_retry_partial_tool = (
                partial_text_delivered
                and bool(partial_tool_names)
                and transient
                and stream_attempt < max_stream_retries
            )
            if can_retry_partial_tool:
                fire_delta = getattr(agent, "_fire_stream_delta", None)
                if callable(fire_delta):
                    try:
                        fire_delta(
                            "\n\n⚠ Connection dropped mid tool-call; "
                            "reconnecting…\n\n"
                        )
                    except Exception:
                        pass
                reset_delivery = getattr(
                    agent,
                    "_reset_stream_delivery_tracking",
                    None,
                )
                if callable(reset_delivery):
                    try:
                        reset_delivery()
                    except Exception:
                        pass
                emit_drop = getattr(agent, "_emit_stream_drop", None)
                if callable(emit_drop):
                    emit_drop(
                        error=error,
                        attempt=stream_attempt + 2,
                        max_attempts=max_stream_retries + 1,
                        mid_tool_call=True,
                        diag=diag,
                    )
                continue

            if partial_text_delivered:
                partial = _partial_stream_stub(
                    agent,
                    error,
                    delivered=True,
                )
                if partial is not None:
                    _reset_stale_streak(agent)
                    return partial

            if transient and stream_attempt < max_stream_retries:
                emit_drop = getattr(agent, "_emit_stream_drop", None)
                if callable(emit_drop):
                    emit_drop(
                        error=error,
                        attempt=stream_attempt + 2,
                        max_attempts=max_stream_retries + 1,
                        mid_tool_call=False,
                        diag=diag,
                    )
                continue

            if transient:
                log_retry = getattr(agent, "_log_stream_retry", None)
                if callable(log_retry):
                    log_retry(
                        kind="exhausted",
                        error=error,
                        attempt=max_stream_retries + 1,
                        max_attempts=max_stream_retries + 1,
                        mid_tool_call=False,
                        diag=diag,
                    )
                if is_parse_error:
                    status = (
                        "❌ Provider returned malformed streaming data after "
                        f"{max_stream_retries + 1} attempts. The provider may "
                        "be experiencing issues — try again in a moment."
                    )
                elif is_empty_stream:
                    status = (
                        "❌ Provider returned an empty response stream after "
                        f"{max_stream_retries + 1} attempts. The provider may "
                        "be experiencing issues — try again in a moment."
                    )
                else:
                    status = (
                        "❌ Connection to provider failed after "
                        f"{max_stream_retries + 1} attempts. The provider may "
                        "be experiencing issues — try again in a moment."
                    )
                buffer_status = getattr(agent, "_buffer_status", None)
                if callable(buffer_status):
                    buffer_status(status)
            else:
                lowered = str(error).lower()
                stream_unsupported = (
                    "stream" in lowered and "not supported" in lowered
                )
                bedrock_stream_denied = False
                if (
                    not stream_unsupported
                    and "invokemodelwithresponsestream" in lowered
                ):
                    from agent.bedrock_adapter import (
                        is_streaming_access_denied_error,
                    )

                    bedrock_stream_denied = is_streaming_access_denied_error(
                        error
                    )
                if stream_unsupported or bedrock_stream_denied:
                    agent._disable_streaming = True
                    safe_print = getattr(agent, "_safe_print", None)
                    if callable(safe_print):
                        safe_print(
                            "\n⚠  AWS IAM denied "
                            "bedrock:InvokeModelWithResponseStream. Switching "
                            "to non-streaming.\n   Grant that action to restore "
                            "streaming output.\n"
                            if bedrock_stream_denied
                            else "\n⚠  Streaming is not supported for this "
                            "model/provider. Switching to non-streaming.\n   "
                            "To avoid this delay, set display.streaming: false "
                            "in config.yaml\n"
                        )
                logger.exception("Streaming failed before delivery: %s", error)
            raise error

        if getattr(agent, "_interrupt_requested", False):
            raise InterruptedError(
                "Agent interrupted during streaming API call (post-worker)"
            )
        _reset_stale_streak(agent)
        return response

    raise AssertionError("unreachable stream retry state")


async def _derive_stream_stale_timeout(agent, api_kwargs: dict) -> float:
    """Resolve upstream's provider/context-aware stream idle timeout.

    Mirrors the main streaming path's derivation — provider config → env base
    → local-endpoint allowance → context-size scaling → reasoning-model floor.
    Every native async provider shares the same patience budget.
    """
    from hermes_cli.timeouts import get_provider_stale_timeout

    provider = getattr(agent, "provider", "") or ""
    model = (
        getattr(agent, "model", "")
        or api_kwargs.get("model")
        or api_kwargs.get("modelId")
        or ""
    )
    _cfg_stale = await get_provider_stale_timeout(provider, model)
    if _cfg_stale is not None:
        _base = _cfg_stale
    else:
        _base = env_float("HERMES_STREAM_STALE_TIMEOUT", 180.0)
    if _base == 180.0 and getattr(agent, "base_url", None):
        from agent.model_metadata import is_local_endpoint

        if is_local_endpoint(agent.base_url):
            local_default = 900.0
            try:
                from hermes_cli.config import load_config_readonly

                config = await load_config_readonly()
                agent_config = config.get("agent") if isinstance(config, dict) else None
                configured = (
                    agent_config.get("local_stream_stale_timeout")
                    if isinstance(agent_config, dict)
                    else None
                )
                if isinstance(configured, (int, float)):
                    local_default = float(configured)
            except Exception:
                pass
            return env_float("HERMES_LOCAL_STREAM_STALE_TIMEOUT", local_default)

    _est_tokens = estimate_request_context_tokens(api_kwargs)
    if _est_tokens > 100_000:
        _timeout = max(_base, 300.0)
    elif _est_tokens > 50_000:
        _timeout = max(_base, 240.0)
    else:
        _timeout = _base
    from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
    # Resolve the model id from BOTH the OpenAI/Anthropic key (``model``) and
    # the Bedrock key (``modelId``). OpenAI/Anthropic wins first via the ``or``
    # chain, so those paths are unchanged. Bedrock carries the model as a
    # dotted, region-prefixed inference-profile id (e.g.
    # ``us.anthropic.claude-opus-4-6-v1:0``) that the floor's start-of-slug
    # regex cannot match directly — normalize it to a canonical slug first.
    _model_id = api_kwargs.get("model") or api_kwargs.get("modelId") or ""
    _reasoning_floor = get_reasoning_stale_timeout_floor(_model_id)
    if _reasoning_floor is None and api_kwargs.get("modelId"):
        _reasoning_floor = _bedrock_reasoning_stale_floor(api_kwargs["modelId"])
    if _reasoning_floor is not None:
        _timeout = max(_timeout, _reasoning_floor)
    return _timeout


def _bedrock_reasoning_stale_floor(model_id: object) -> "float | None":
    """Map a Bedrock inference-profile id to its reasoning stale-timeout floor.

    Bedrock carries the model as a dotted, region-prefixed id such as
    ``us.anthropic.claude-opus-4-6-v1:0``, whereas
    :func:`get_reasoning_stale_timeout_floor` anchors its slug patterns at the
    start of a bare slug (``claude-opus-4``). Strip the region prefix
    (``us.``/``eu.``/``apac.``/...) and try two candidate slugs against the
    floor:

    * the segment after the provider namespace (``claude-opus-4-6-v1:0``) —
      matches Anthropic-style slugs whose floor key excludes the provider
      (``claude-opus-4``); and
    * the region-stripped id with the provider dot rewritten to a dash
      (``deepseek-r1-v1:0``) — matches provider-qualified floor keys
      (``deepseek-r1``).

    The floor's right-anchor (``$`` or ``-``/``.``/``_``) tolerates the
    trailing date-stamp / ``-v1:0`` version suffix, so no suffix stripping is
    needed. First non-None wins; returns None for unknown models.

    The floor table mixes version-separator conventions: some keys are
    keyed with a dashed version (``claude-opus-4``) while others embed a
    dotted version (``claude-sonnet-4.5``, ``claude-sonnet-4.6``). Bedrock
    always dashes the version (``claude-sonnet-4-5-v1:0``), so for every
    candidate slug we also try the alternate version-separator form —
    digit-dash-digit rewritten to digit-dot-digit and vice-versa — so a
    dashed Bedrock id matches a dotted floor key (and the reverse). The
    rewrite only touches version-number separators (a dash/dot flanked by
    digits), never other dashes in the slug, so ``claude-sonnet`` is left
    intact while ``4-5`` becomes ``4.5``.
    """
    from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor

    if not model_id or not isinstance(model_id, str):
        return None
    name = model_id.strip().lower()
    for prefix in (
        "global.", "us.", "eu.", "apac.", "ap.", "au.", "jp.",
        "ca.", "sa.", "me.", "af.",
    ):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    base_candidates = [name]
    if "." in name:
        base_candidates.append(name.rsplit(".", 1)[1])   # claude-opus-4-6-v1:0
        base_candidates.append(name.replace(".", "-", 1))  # deepseek-r1-v1:0
    candidates: list[str] = []
    for cand in base_candidates:
        # Try the slug as-is plus both alternate version-separator forms.
        # ``4-5`` <-> ``4.5`` only; a dash/dot not flanked by digits is
        # left alone (e.g. ``claude-sonnet`` stays dashed).
        dashed_to_dotted = re.sub(r"(?<=\d)-(?=\d)", ".", cand)
        dotted_to_dashed = re.sub(r"(?<=\d)\.(?=\d)", "-", cand)
        for form in (cand, dashed_to_dotted, dotted_to_dashed):
            if form not in candidates:
                candidates.append(form)
    for cand in candidates:
        floor = get_reasoning_stale_timeout_floor(cand)
        if floor is not None:
            return floor
    return None


async def build_api_kwargs(
    agent,
    api_messages: list,
    tools_for_api: list | None = None,
) -> dict:
    """Build the keyword arguments dict for the active API mode.

    Message preparation may need to consult async configuration/capability
    providers.  Keep this helper on the native async request path instead of
    reintroducing a synchronous adapter at the API boundary.
    """
    from providers import _ensure_provider_profiles_loaded

    await _ensure_provider_profiles_loaded()
    if tools_for_api is None:
        tools_for_api = agent.tools

    if agent.api_mode == "anthropic_messages":
        _transport = agent._get_transport()
        anthropic_messages = await agent._prepare_anthropic_messages_for_api(api_messages)
        ctx_len = getattr(agent, "context_compressor", None)
        ctx_len = ctx_len.context_length if ctx_len else None
        ephemeral_out = getattr(agent, "_ephemeral_max_output_tokens", None)
        if ephemeral_out is not None:
            agent._ephemeral_max_output_tokens = None  # consume immediately
        anthropic_kwargs = _transport.build_kwargs(
            model=agent.model,
            messages=anthropic_messages,
            tools=tools_for_api,
            max_tokens=ephemeral_out if ephemeral_out is not None else agent.max_tokens,
            reasoning_config=agent.reasoning_config,
            is_oauth=agent._is_anthropic_oauth,
            preserve_dots=agent._anthropic_preserve_dots(),
            context_length=ctx_len,
            base_url=getattr(agent, "_anthropic_base_url", None),
            fast_mode=(agent.request_overrides or {}).get("speed") == "fast",
            drop_context_1m_beta=bool(getattr(agent, "_oauth_1m_beta_disabled", False)),
        )
        # Nous Portal reads ``tags`` and ``session_id`` as top-level body fields
        # on its Messages route the same way it does on /chat/completions, but
        # the profile hook that produces them is only consulted by the
        # OpenAI-wire transport. Merge them here so Messages traffic keeps
        # product attribution and sticky routing.
        return _merge_nous_portal_messages_extra_body(agent, anthropic_kwargs)

    # AWS Bedrock native Converse API — bypasses the OpenAI client entirely.
    # The adapter handles message/tool conversion and native async AWS calls directly.
    if agent.api_mode == "bedrock_converse":
        _bt = agent._get_transport()
        region = getattr(agent, "_bedrock_region", None) or "us-east-1"
        guardrail = getattr(agent, "_bedrock_guardrail_config", None)
        return _bt.build_kwargs(
            model=agent.model,
            messages=api_messages,
            tools=tools_for_api,
            max_tokens=agent.max_tokens or 4096,
            region=region,
            guardrail_config=guardrail,
        )

    if agent.api_mode == "codex_responses":
        _ct = agent._get_transport()
        is_github_responses = (
            base_url_host_matches(agent.base_url, "models.github.ai")
            or base_url_host_matches(agent.base_url, "githubcopilot.com")
        )
        is_codex_backend = (
            agent.provider == "openai-codex"
            or (
                agent._base_url_hostname == "chatgpt.com"
                and "/backend-api/codex" in agent._base_url_lower
            )
        )
        is_xai_responses = agent.provider in {"xai", "xai-oauth"} or agent._base_url_hostname == "api.x.ai"
        _msgs_for_codex = await agent._prepare_messages_for_non_vision_model(api_messages)

        # xAI's /responses endpoint rejects ``pattern`` and ``format`` keywords
        # in tool schemas (HTTP 400 "Invalid arguments passed to the model").
        # Most commonly hit when MCP-derived tools carry JSON Schema validation
        # keywords through. Strip them before building kwargs. See #27197.
        # It also rejects ``enum`` values containing ``/`` (HuggingFace IDs
        # like ``Qwen/Qwen3.5-0.8B`` shipped by MCP servers) — same 400 with
        # the same opaque message; strip those enums too.
        #
        # Deep-copy ``tools_for_api`` before sanitizing: the sanitizers
        # mutate in place (documented contract on ``strip_slash_enum`` /
        # ``strip_pattern_and_format``), and ``tools_for_api`` is a direct
        # reference to ``agent.tools``.  Without the copy, the first xAI
        # request permanently strips constraints from the shared per-agent
        # tool registry — every subsequent non-xAI call from the same
        # agent (auxiliary task routed to Anthropic, OpenRouter fallback,
        # main-model swap) sees the already-stripped schema.  See #27907.
        if is_xai_responses:
            try:
                import copy as _copy
                from tools.schema_sanitizer import (
                    strip_pattern_and_format,
                    strip_slash_enum,
                )
                tools_for_api = _copy.deepcopy(tools_for_api)
                tools_for_api, _ = strip_pattern_and_format(tools_for_api)
                tools_for_api, _ = strip_slash_enum(tools_for_api)
            except Exception as exc:
                logger.warning(
                    "%s⚠️ Failed to sanitize tool schemas for xAI: %s",
                    getattr(agent, "log_prefix", ""), exc,
                )

        return _ct.build_kwargs(
            model=agent.model,
            messages=_msgs_for_codex,
            tools=tools_for_api,
            reasoning_config=agent.reasoning_config,
            session_id=getattr(agent, "session_id", None),
            base_url=agent.base_url,
            max_tokens=agent.max_tokens,
            timeout=await agent._resolved_api_call_timeout(),
            request_overrides=agent.request_overrides,
            is_github_responses=is_github_responses,
            is_codex_backend=is_codex_backend,
            is_xai_responses=is_xai_responses,
            github_reasoning_extra=agent._github_models_reasoning_extra_body() if is_github_responses else None,
            replay_encrypted_reasoning=bool(
                getattr(agent, "_codex_reasoning_replay_enabled", True)
            ),
        )

    # ── chat_completions (default) ─────────────────────────────────────
    _ct = agent._get_transport()

    # Provider detection flags
    _is_qwen = agent._is_qwen_portal()
    _is_or = agent._is_openrouter_url()
    _is_gh = (
        base_url_host_matches(agent._base_url_lower, "models.github.ai")
        or base_url_host_matches(agent._base_url_lower, "githubcopilot.com")
    )
    _is_nous = "nousresearch" in agent._base_url_lower
    _is_nvidia = "integrate.api.nvidia.com" in agent._base_url_lower
    _is_kimi = (
        base_url_host_matches(agent.base_url, "api.kimi.com")
        or base_url_host_matches(agent.base_url, "moonshot.ai")
        or base_url_host_matches(agent.base_url, "moonshot.cn")
    )
    _is_tokenhub = base_url_host_matches(agent._base_url_lower, "tokenhub.tencentmaas.com")
    _is_lmstudio = (agent.provider or "").strip().lower() == "lmstudio"

    # Temperature: _fixed_temperature_for_model may return OMIT_TEMPERATURE
    # sentinel (temperature omitted entirely), a numeric override, or None.
    try:
        from agent.auxiliary_client import _fixed_temperature_for_model, OMIT_TEMPERATURE
        _ft = _fixed_temperature_for_model(agent.model, agent.base_url)
        _omit_temp = _ft is OMIT_TEMPERATURE
        _fixed_temp = _ft if not _omit_temp else None
    except Exception:
        _omit_temp = False
        _fixed_temp = None

    # Provider preferences (aggregator profile decides whether to emit them).
    _prefs = _provider_preferences_for_agent(agent)

    # Anthropic-compatible max-output fallback (last resort only — applied in
    # build_kwargs *after* ephemeral/user/profile max_tokens, never overriding
    # an explicit value).  Model-gated, not URL-gated: any chat-completions
    # proxy serving a Claude/MiniMax/Qwen3 model needs max_tokens, because the
    # Anthropic Messages API treats it as mandatory and proxies that omit it
    # (AWS Bedrock, NVIDIA, LiteLLM, vLLM, corporate gateways) default as low
    # as 4096 output tokens — easily exhausted by thinking + large tool calls
    # like write_file/patch.  OpenRouter/Nous were the only routes covered
    # before; gating on _ANTHROPIC_OUTPUT_LIMITS membership covers them all.
    _ant_max = None
    try:
        from agent.anthropic_adapter import (
            _get_anthropic_max_output,
            _ANTHROPIC_OUTPUT_LIMITS,
        )
        _model_norm = (agent.model or "").lower().replace(".", "-")
        if any(key in _model_norm for key in _ANTHROPIC_OUTPUT_LIMITS):
            _ant_max = _get_anthropic_max_output(agent.model)
    except Exception:
        pass

    # Qwen session metadata
    _qwen_meta = None
    if _is_qwen:
        _qwen_meta = {
            "sessionId": agent.session_id or "hermes",
            "promptId": str(uuid.uuid4()),
        }

    # ── Provider profile path (registered providers) ───────────────────
    # Profiles handle per-provider quirks via hooks. When a profile is
    # found, delegate fully; otherwise fall through to the legacy flag path.
    try:
        from providers import _get_provider_profile_cached

        _profile = _get_provider_profile_cached(agent.provider)
    except Exception:
        _profile = None

    if _profile:
        _ephemeral_out = getattr(agent, "_ephemeral_max_output_tokens", None)
        if _ephemeral_out is not None:
            agent._ephemeral_max_output_tokens = None

        # Strip image parts for non-vision models that have provider profiles
        # (e.g. DeepSeek, Kimi). The legacy path below already does this, but
        # registered providers with profiles were bypassing the strip.
        api_messages = await agent._prepare_messages_for_non_vision_model(api_messages)

        return _ct.build_kwargs(
            model=agent.model,
            messages=api_messages,
            tools=tools_for_api,
            base_url=agent.base_url,
            timeout=await agent._resolved_api_call_timeout(),
            max_tokens=agent.max_tokens,
            ephemeral_max_output_tokens=_ephemeral_out,
            max_tokens_param_fn=agent._max_tokens_param,
            reasoning_config=agent.reasoning_config,
            request_overrides=agent.request_overrides,
            session_id=getattr(agent, "session_id", None),
            provider_profile=_profile,
            ollama_num_ctx=agent._ollama_num_ctx,
            # Context forwarded to profile hooks:
            provider_preferences=_prefs or None,
            openrouter_min_coding_score=agent.openrouter_min_coding_score,
            anthropic_max_output=_ant_max,
            supports_reasoning=await agent._supports_reasoning_extra_body(),
            qwen_session_metadata=_qwen_meta,
        )

    # ── Legacy flag path ────────────────────────────────────────────
    # Reached only when get_provider_profile() returns None — i.e. a
    # completely unknown provider not in providers/ registry.
    _ephemeral_out = getattr(agent, "_ephemeral_max_output_tokens", None)
    if _ephemeral_out is not None:
        agent._ephemeral_max_output_tokens = None

    # Strip image parts for non-vision models (no-op when vision-capable).
    _msgs_for_chat = await agent._prepare_messages_for_non_vision_model(api_messages)

    return _ct.build_kwargs(
        model=agent.model,
        messages=_msgs_for_chat,
        tools=tools_for_api,
        base_url=agent.base_url,
        timeout=await agent._resolved_api_call_timeout(),
        max_tokens=agent.max_tokens,
        ephemeral_max_output_tokens=_ephemeral_out,
        max_tokens_param_fn=agent._max_tokens_param,
        reasoning_config=agent.reasoning_config,
        request_overrides=agent.request_overrides,
        session_id=getattr(agent, "session_id", None),
        model_lower=(agent.model or "").lower(),
        is_openrouter=_is_or,
        is_nous=_is_nous,
        is_qwen_portal=_is_qwen,
        is_github_models=_is_gh,
        is_nvidia_nim=_is_nvidia,
        is_kimi=_is_kimi,
        is_tokenhub=_is_tokenhub,
        is_lmstudio=_is_lmstudio,
        is_custom_provider=agent.provider == "custom",
        ollama_num_ctx=agent._ollama_num_ctx,
        provider_preferences=_prefs or None,
        openrouter_min_coding_score=agent.openrouter_min_coding_score,
        qwen_prepare_fn=agent._qwen_prepare_chat_messages if _is_qwen else None,
        qwen_prepare_inplace_fn=agent._qwen_prepare_chat_messages_inplace if _is_qwen else None,
        qwen_session_metadata=_qwen_meta,
        fixed_temperature=_fixed_temp,
        omit_temperature=_omit_temp,
        supports_reasoning=await agent._supports_reasoning_extra_body(),
        github_reasoning_extra=agent._github_models_reasoning_extra_body() if _is_gh else None,
        lmstudio_reasoning_options=(
            await agent._lmstudio_reasoning_options_cached()
            if _is_lmstudio
            else None
        ),
        anthropic_max_output=_ant_max,
        provider_name=agent.provider,
    )



def build_assistant_message(agent, assistant_message, finish_reason: str) -> dict:
    """Build a normalized assistant message dict from an API response message.

    Handles reasoning extraction, reasoning_details, and optional tool_calls
    so both the tool-call path and the final-response path share one builder.
    """
    assistant_tool_calls = getattr(assistant_message, "tool_calls", None)
    reasoning_text = agent._extract_reasoning(assistant_message)
    _from_structured = bool(reasoning_text)

    # Fallback: extract inline <think> blocks from content when no structured
    # reasoning fields are present (some models/providers embed thinking
    # directly in the content rather than returning separate API fields).
    if not reasoning_text:
        content = flatten_message_text(getattr(assistant_message, "content", None))
        think_blocks = re.findall(r'<think>(.*?)</think>', content, flags=re.DOTALL)
        if think_blocks:
            combined = "\n\n".join(b.strip() for b in think_blocks if b.strip())
            reasoning_text = combined or None

    if reasoning_text and agent.verbose_logging:
        logging.debug(f"Captured reasoning ({len(reasoning_text)} chars): {reasoning_text}")

    if reasoning_text and agent.reasoning_callback:
        # Skip callback when streaming is active — reasoning was already
        # displayed during the stream via one of two paths:
        #   (a) _fire_reasoning_delta (structured reasoning_content deltas)
        #   (b) _stream_delta tag extraction (<think>/<REASONING_SCRATCHPAD>)
        # When streaming is NOT active, always fire so non-streaming modes
        # (gateway, batch, quiet) still get reasoning.
        # Any reasoning that wasn't shown during streaming is caught by the
        # CLI post-response display fallback (cli.py _reasoning_shown_this_turn).
        if not agent.stream_delta_callback and not agent._stream_callback:
            try:
                agent.reasoning_callback(reasoning_text)
            except Exception:
                pass

    # Sanitize surrogates from API response — some models (e.g. Kimi/GLM via Ollama)
    # can return invalid surrogate code points that crash json.dumps() on persist.
    _raw_content = flatten_message_text(getattr(assistant_message, "content", None))
    _san_content = _sanitize_surrogates(_raw_content)
    if reasoning_text:
        reasoning_text = _sanitize_surrogates(reasoning_text)

    # Strip inline reasoning tags (<think>…</think> etc.) from the stored
    # assistant content.  Reasoning was already captured into
    # ``reasoning_text`` above (either from structured fields or the
    # inline-block fallback), so the raw tags in content are redundant.
    # Leaving them in place caused reasoning to leak to messaging
    # platforms (#8878, #9568), inflate context on subsequent turns
    # (#9306 observed 16% content-size reduction on a real MiniMax
    # session), and pollute generated session titles.  One strip at the
    # storage boundary cleans content for every downstream consumer:
    # API replay, session transcript, gateway delivery, CLI display,
    # compression, title generation.
    if isinstance(_san_content, str) and _san_content:
        _san_content = agent._strip_think_blocks(_san_content).strip()

    # Defence-in-depth: redact credentials (PATs, API keys, Bearer tokens)
    # from assistant content BEFORE the message enters conversation history.
    # If the model accidentally inlines a secret in its natural-language
    # response, catch it here at the persistence boundary so it never
    # reaches state.db, session_*.json, gateway delivery, or compression.
    # Respects HERMES_REDACT_SECRETS via redact_sensitive_text — no-op
    # when disabled. (#19798)
    if isinstance(_san_content, str) and _san_content:
        from agent.redact import redact_sensitive_text
        _san_content = redact_sensitive_text(_san_content)

    # NOTE (empty-content class fix): textless assistant turns are NOT padded
    # here.  The single owner for "never send a turn strict wire validation
    # rejects as empty" is ``repair_empty_non_final_messages`` in
    # agent_runtime_helpers, which runs inside ``sanitize_api_messages`` — the
    # unconditional pre-send chokepoint for both the main loop and the summary
    # path.  Padding at write time was tried (a single-space pad, later a
    # placeholder) and rejected: it forked the concept across three sites,
    # broke codex commentary turns (content:'' is a designed state there), and
    # a DB-side pad can't survive ``_rows_to_conversation``'s whitespace strip
    # anyway.  Repair belongs at the send boundary, once.

    msg = {
        "role": "assistant",
        "content": _san_content,
        "reasoning": reasoning_text,
        "finish_reason": finish_reason,
    }

    raw_reasoning_content = getattr(assistant_message, "reasoning_content", None)
    if raw_reasoning_content is None and hasattr(assistant_message, "model_extra"):
        model_extra = getattr(assistant_message, "model_extra", None) or {}
        if isinstance(model_extra, dict) and "reasoning_content" in model_extra:
            raw_reasoning_content = model_extra["reasoning_content"]
    if raw_reasoning_content is not None:
        msg["reasoning_content"] = _sanitize_surrogates(raw_reasoning_content)
    elif assistant_tool_calls and agent._needs_thinking_reasoning_pad():
        # DeepSeek v4 thinking mode and Kimi / Moonshot thinking mode
        # both require reasoning_content on every assistant tool-call
        # message. Without it, replaying the persisted message causes
        # HTTP 400 ("The reasoning_content in the thinking mode must
        # be passed back to the API"). Include streamed reasoning
        # text when captured; otherwise pad with a single space —
        # DeepSeek V4 Pro tightened validation and rejects empty
        # string ("The reasoning content in the thinking mode must
        # be passed back to the API"). A space satisfies non-empty
        # checks everywhere without leaking fabricated reasoning.
        # Refs #15250, #17400, #17341.
        msg["reasoning_content"] = reasoning_text or " "

    # Additive fallback (refs #16844, #16884). Streaming-only providers
    # (glm, MiniMax, gpt-5.x via aigw, Anthropic via openai-compat shims)
    # accumulate reasoning through ``delta.reasoning_content`` chunks
    # but never land it on the message object as a top-level attribute,
    # so neither branch above fires and the chain-of-thought is stored
    # only under the internal ``reasoning`` key. When the user later
    # replays that history through a DeepSeek-v4 / Kimi thinking model,
    # the missing ``reasoning_content`` causes HTTP 400 ("The
    # reasoning_content in the thinking mode must be passed back to the
    # API.").
    #
    # Promote the already-sanitized streamed ``reasoning_text`` to
    # ``reasoning_content`` at write time, but ONLY when no prior branch
    # already set it AND we actually captured reasoning text. This
    # preserves every existing behavior:
    #   - SDK-exposed ``reasoning_content`` (OpenAI/Moonshot/DeepSeek SDK)
    #     still wins.
    #   - DeepSeek tool-call ""-pad (#15250) still fires.
    #   - Non-thinking turns with no reasoning leave the field absent,
    #     so ``_copy_reasoning_content_for_api``'s cross-provider leak
    #     guard (#15748) and ``reasoning``→``reasoning_content``
    #     promotion tiers still apply at replay time.
    if "reasoning_content" not in msg and reasoning_text:
        msg["reasoning_content"] = reasoning_text

    if hasattr(assistant_message, 'reasoning_details') and assistant_message.reasoning_details:
        # Pass reasoning_details back unmodified so providers (OpenRouter,
        # Anthropic, OpenAI) can maintain reasoning continuity across turns.
        # Each provider may include opaque fields (signature, encrypted_content)
        # that must be preserved exactly.
        raw_details = assistant_message.reasoning_details
        preserved = []
        for d in raw_details:
            if isinstance(d, dict):
                preserved.append(d)
            elif hasattr(d, "__dict__"):
                preserved.append(d.__dict__)
            elif hasattr(d, "model_dump"):
                preserved.append(d.model_dump())
        if preserved:
            msg["reasoning_details"] = preserved

    # Anthropic interleaved-thinking replay: when a turn interleaves signed
    # thinking blocks with tool_use, the parallel reasoning_details +
    # tool_calls fields lose the cross-type ordering, and reconstruction
    # front-loads thinking — reordering signed blocks and triggering HTTP 400
    # ("thinking ... blocks in the latest assistant message cannot be
    # modified"). Carry the verbatim ordered block list so the adapter can
    # replay the latest assistant message unchanged. See
    # agent/transports/anthropic.py and agent/anthropic_adapter.py.
    ordered_blocks = getattr(assistant_message, "anthropic_content_blocks", None)
    if ordered_blocks:
        msg["anthropic_content_blocks"] = ordered_blocks

    # Codex Responses API: preserve encrypted reasoning items for
    # multi-turn continuity. These get replayed as input on the next turn.
    codex_items = getattr(assistant_message, "codex_reasoning_items", None)
    if codex_items:
        msg["codex_reasoning_items"] = codex_items

    # Codex Responses API: preserve exact assistant message items (with
    # id/phase) so follow-up turns can replay structured items instead of
    # flattening to plain text. This is required for prefix cache hits.
    codex_message_items = getattr(assistant_message, "codex_message_items", None)
    if codex_message_items:
        msg["codex_message_items"] = codex_message_items

    if assistant_tool_calls:
        tool_calls = []
        for tool_call in assistant_tool_calls:
            raw_id = getattr(tool_call, "id", None)
            call_id = getattr(tool_call, "call_id", None)
            if not isinstance(call_id, str) or not call_id.strip():
                embedded_call_id, _ = agent._split_responses_tool_id(raw_id)
                call_id = embedded_call_id
            if not isinstance(call_id, str) or not call_id.strip():
                if isinstance(raw_id, str) and raw_id.strip():
                    call_id = raw_id.strip()
                else:
                    _fn = getattr(tool_call, "function", None)
                    _fn_name = getattr(_fn, "name", "") if _fn else ""
                    _fn_args = getattr(_fn, "arguments", "{}") if _fn else "{}"
                    call_id = agent._deterministic_call_id(_fn_name, _fn_args, len(tool_calls))
            call_id = call_id.strip()

            response_item_id = getattr(tool_call, "response_item_id", None)
            if not isinstance(response_item_id, str) or not response_item_id.strip():
                _, embedded_response_item_id = agent._split_responses_tool_id(raw_id)
                response_item_id = embedded_response_item_id

            response_item_id = agent._derive_responses_function_call_id(
                call_id,
                response_item_id if isinstance(response_item_id, str) else None,
            )

            tc_dict = {
                "id": call_id,
                "call_id": call_id,
                "response_item_id": response_item_id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                },
            }
            # Tool-call arguments are intentionally NOT redacted here. This
            # dict enters the in-memory conversation history that is replayed
            # to the model on every subsequent turn AND persisted to state.db,
            # which is itself replayed verbatim on session resume
            # (get_messages_as_conversation). Masking a credential to `***`
            # here poisons that replay: the model reads back its own
            # `PGPASSWORD='***' psql ...` call and copies the placeholder into
            # the next tool call, breaking every credential-dependent command
            # on the second turn (#43083). The masking also provided no real
            # protection — the same secret still leaks verbatim through tool
            # OUTPUT (file contents, command output, diffs, the compaction
            # block), none of which this pass ever touched. Keeping secrets
            # out of the replayable store is a separate tokenization/vault
            # concern, not something arg-redaction can deliver without
            # breaking replay. Storage-time redaction remains governed by the
            # `security.redact_secrets` toggle. (#19798 introduced this;
            # #43083 removed it.)
            # Preserve extra_content (e.g. Gemini thought_signature) so it
            # is sent back on subsequent API calls.  Without this, Gemini 3
            # thinking models reject the request with a 400 error.
            extra = getattr(tool_call, "extra_content", None)
            if extra is not None:
                if hasattr(extra, "model_dump"):
                    extra = extra.model_dump()
                tc_dict["extra_content"] = extra
            tool_calls.append(tc_dict)
        msg["tool_calls"] = tool_calls

    return msg



def rewrite_prompt_model_identity(agent, model: str, provider: str) -> None:
    """Point the cached system prompt's ``Model:``/``Provider:`` lines at
    the active runtime after a provider switch.

    The system prompt is session-stable and replayed verbatim for prefix-cache
    warmth, but after a failover the new backend's cache is cold anyway —
    while a stale identity line makes the agent misreport which model it is
    when asked.  Rewrite the lines in place WITHOUT persisting to the session
    DB: the stored row keeps the primary's labels, so when the primary is
    restored the prompt is byte-identical to the stored copy again and its
    prefix cache still matches.

    Only the LAST occurrence of each line is touched — the identity lines
    live in the volatile tail of the prompt, and earlier matches could be
    user content (memory snapshots, context files).
    """
    sp = getattr(agent, "_cached_system_prompt", None)
    if not isinstance(sp, str) or not sp:
        return
    for label, value in (("Model", model), ("Provider", provider)):
        if not value:
            continue
        matches = list(re.finditer(rf"(?m)^{label}: .*$", sp))
        if matches:
            last = matches[-1]
            sp = f"{sp[:last.start()]}{label}: {value}{sp[last.end():]}"
    agent._cached_system_prompt = sp


def _fallback_entry_key(fb: dict) -> tuple[str, str, str]:
    return (
        str(fb.get("provider") or "").strip().lower(),
        str(fb.get("model") or "").strip(),
        str(fb.get("base_url") or "").strip().rstrip("/"),
    )


async def try_activate_fallback(agent, reason: "FailoverReason | None" = None) -> bool:
    """Switch to the next fallback model/provider in the chain.

    Called when the current model is failing after retries.  Swaps the
    OpenAI client, model slug, and provider in-place so the retry loop
    can continue with the new backend.  Advances through the chain on
    each call; returns False when exhausted.

    Resolves credentials and native clients through the same deferred async
    lifecycle as a no-credential agent constructor. The synchronous legacy
    provider router is deliberately not available from an active turn.
    """
    if reason in {FailoverReason.rate_limit, FailoverReason.billing, FailoverReason.upstream_rate_limit}:
        # Only start cooldown when leaving the primary provider.  If we're
        # already on a fallback and chain-switching, the primary wasn't the
        # source of the 429 so the cooldown should not be reset/extended.
        fallback_already_active = bool(getattr(agent, "_fallback_activated", False))
        current_provider = (getattr(agent, "provider", "") or "").strip().lower()
        primary_provider = ((agent._primary_runtime or {}).get("provider") or "").strip().lower()
        if (not fallback_already_active) or (primary_provider and current_provider == primary_provider):
            agent._rate_limited_until = time.monotonic() + 60
    if agent._fallback_index >= len(agent._fallback_chain):
        # Chain exhausted.  If we actually walked a non-empty chain and the
        # failure was NOT a rate-limit/billing event (those already armed
        # their own 60s cooldown above), arm a short cooldown so the next
        # turn's restore_primary_runtime stays gated instead of resetting
        # _fallback_index=0 and re-marshaling the whole context across every
        # provider again.  Guards the cross-turn replay storm in #24996.
        if (
            len(agent._fallback_chain) > 0
            and reason not in {FailoverReason.rate_limit, FailoverReason.billing, FailoverReason.upstream_rate_limit}
        ):
            _existing_cooldown = getattr(agent, "_rate_limited_until", 0) or 0
            agent._rate_limited_until = max(
                _existing_cooldown,
                time.monotonic() + _FALLBACK_EXHAUSTED_COOLDOWN_S,
            )
        return False
    fb = agent._fallback_chain[agent._fallback_index]
    agent._fallback_index += 1
    fb_key = _fallback_entry_key(fb)
    unavailable = getattr(agent, "_unavailable_fallback_keys", None)
    if unavailable is None:
        unavailable = set()
        agent._unavailable_fallback_keys = unavailable
    if fb_key in unavailable:
        logger.debug("Fallback skip: %s previously marked unavailable", fb_key)
        return await agent._try_activate_fallback(reason)
    fb_provider = (fb.get("provider") or "").strip().lower()
    fb_model = (fb.get("model") or "").strip()
    if not fb_provider or not fb_model:
        return await agent._try_activate_fallback(reason)  # skip invalid, try next

    # Skip entries that resolve to the same backend that just failed —
    # falling back to it loops the failure. Identity semantics (which axes
    # distinguish two backends, shim aliases, first-class credential
    # surfaces, multi-endpoint pools) are owned by agent.backend_identity —
    # see #22548, #70893, #62984. Do not re-implement comparisons here.
    from agent.backend_identity import BackendIdentity, should_skip_candidate

    current_ident = BackendIdentity.build(
        provider=getattr(agent, "provider", ""),
        model=getattr(agent, "model", ""),
        base_url=str(getattr(agent, "base_url", "") or ""),
    )
    fb_ident = BackendIdentity.build(
        provider=fb_provider,
        model=fb_model,
        base_url=(fb.get("base_url") or ""),
    )
    if should_skip_candidate(fb_ident, current_ident):
        logger.warning(
            "Fallback skip: chain entry %s/%s resolves to the same backend "
            "as the current one (%s)",
            fb_provider, fb_model, current_ident.base_url or current_ident.provider,
        )
        return await agent._try_activate_fallback(reason)

    # Resolve the fallback with the same native async lifecycle used by a
    # no-credential AIAgent constructor.  ``resolve_provider_client`` is the
    # legacy router and may synchronously read/refresh credentials before it
    # converts a client to AsyncOpenAI, so it cannot run inside a turn.
    try:
        # Pass base_url and api_key from fallback config so custom
        # endpoints resolve without touching a legacy config/auth resolver.
        fb_base_url_hint = (fb.get("base_url") or "").strip() or None
        fb_api_key_hint = (fb.get("api_key") or "").strip() or None
        if not fb_api_key_hint:
            # key_env and api_key_env are both documented aliases (see
            # _normalize_custom_provider_entry in hermes_cli/config.py).
            fb_key_env = (fb.get("key_env") or fb.get("api_key_env") or "").strip()
            if fb_key_env:
                fb_api_key_hint = os.getenv(fb_key_env, "").strip() or None
        # For Ollama Cloud endpoints, pull OLLAMA_API_KEY from env
        # when no explicit key is in the fallback config. Host match
        # (not substring) — see GHSA-76xc-57q6-vm5m.
        if fb_base_url_hint and base_url_host_matches(fb_base_url_hint, "ollama.com") and not fb_api_key_hint:
            fb_api_key_hint = os.getenv("OLLAMA_API_KEY") or None
        try:
            from hermes_cli.model_normalize import normalize_model_for_provider

            fb_model = normalize_model_for_provider(fb_model, fb_provider)
        except Exception as _norm_err:
            logger.warning(
                "Could not normalize fallback model %r for provider %r: %s",
                fb_model, fb_provider, _norm_err,
            )

        old_model = agent.model
        old_provider = agent.provider
        agent._deferred_provider_runtime = {
            "provider": fb_provider,
            "model": fb_model,
            "api_key": fb_api_key_hint,
            "base_url": fb_base_url_hint,
            # A fallback must not overwrite the next-turn primary snapshot.
            "update_primary": False,
        }
        await agent._ensure_provider_runtime()
        fb_client = agent.client
        fb_base_url = str(agent.base_url)
        fb_api_mode = agent.api_mode

        # Preserve the existing transport-mode exceptions after native
        # credential resolution. These branches are pure URL/model policy.
        _fb_is_azure = agent._is_azure_openai_url(fb_base_url)
        if fb_provider == "openai-codex":
            fb_api_mode = "codex_responses"
        elif fb_provider in {"nous", "nous-portal", "nousresearch"}:
            # Portal is dual-wire: anthropic/* must land on /v1/messages.
            # The deferred resolver creates the matching native client;
            # preserve the provider's model-to-wire policy here.
            from hermes_cli.providers import nous_api_mode

            fb_api_mode = nous_api_mode(fb_model)
        elif (
            fb_provider == "anthropic"
            or fb_base_url.rstrip("/").lower().endswith("/anthropic")
            or base_url_hostname(fb_base_url) == "api.anthropic.com"
        ):
            # Custom providers (e.g. cron-anthropic) point at the native
            # api.anthropic.com host with no "/anthropic" path suffix, so the
            # name/suffix checks above miss them and they default to
            # chat_completions → POST /v1/chat/completions → 404. Match the
            # host the same way determine_api_mode() and _detect_api_mode_for_url()
            # do on the primary path. (#32243, #49247)
            fb_api_mode = "anthropic_messages"
        elif _fb_is_azure:
            # Azure OpenAI serves gpt-5.x on /chat/completions — does NOT
            # support the Responses API. Stay on chat_completions.
            fb_api_mode = "chat_completions"
        elif agent._is_direct_openai_url(fb_base_url):
            fb_api_mode = "codex_responses"
        elif agent._provider_model_requires_responses_api(
            fb_model,
            provider=fb_provider,
        ):
            # GPT-5.x models usually need Responses API, but keep
            # provider-specific exceptions like Copilot gpt-5-mini on
            # chat completions.
            fb_api_mode = "codex_responses"
        elif fb_provider == "bedrock" or (
            base_url_hostname(fb_base_url).startswith("bedrock-runtime.")
            and base_url_host_matches(fb_base_url, "amazonaws.com")
        ):
            fb_api_mode = "bedrock_converse"

        # Clear the per-config context_length override so the fallback
        # model's actual context window is resolved instead of inheriting
        # the stale value from the previous model.  See #22387.
        agent._config_context_length = None
        agent.model = fb_model
        agent.provider = fb_provider
        agent.requested_provider = fb_provider
        agent.base_url = fb_base_url
        agent.api_mode = fb_api_mode
        if hasattr(agent, "_transport_cache"):
            agent._transport_cache.clear()
        agent._fallback_activated = True

        # Rebind the credential pool to the fallback provider when the provider
        # changes.  Keeping the primary pool attached would make downstream
        # recovery (rate_limit / billing / auth) mutate the wrong credential
        # set and can overwrite the fallback's base_url back to the primary
        # endpoint.  See #33163.
        #
        # When the fallback shares the pool's provider (e.g. both openrouter
        # entries with different routing) the pool is preserved.  When the
        # providers differ, load the fallback provider's own pool if one exists
        # so provider-specific rotation continues to work after the switch.
        _existing_pool = getattr(agent, "_credential_pool", None)
        if _existing_pool is not None:
            _pool_provider = (getattr(_existing_pool, "provider", "") or "").strip().lower()
            if _pool_provider and _pool_provider != fb_provider:
                logger.info(
                    "Fallback to %s/%s: clearing primary credential pool "
                    "(pool_provider=%s) to prevent cross-provider contamination",
                    fb_provider, fb_model, _pool_provider,
                )
                agent._credential_pool = None
                agent._credential_pool_entry_id = None
        if getattr(agent, "_credential_pool", None) is None:
            try:
                from agent.credential_pool import load_pool

                fallback_pool = await load_pool(fb_provider)
                if fallback_pool and fallback_pool.entries():
                    agent._credential_pool = fallback_pool
                    logger.info(
                        "Fallback to %s/%s: attached fallback credential pool",
                        fb_provider, fb_model,
                    )
            except Exception as exc:
                logger.debug(
                    "Fallback to %s/%s: could not attach credential pool: %s",
                    fb_provider, fb_model, exc,
                )

        if fb_api_mode == "anthropic_messages":
            # The native deferred resolver has already created AsyncAnthropic
            # and installed the selected credential. Rebuilding it through the
            # legacy resolver would reintroduce synchronous OAuth/file I/O.
            agent.client = None
            agent._client_kwargs = {}
        else:
            if fb_client is None:
                raise RuntimeError(
                    f"Fallback {fb_provider!r} did not produce a native async client"
                )
            # The deferred resolver installed a native AsyncOpenAI client and
            # its rebuild recipe. Keep that instance rather than converting or
            # rebuilding a synchronous client.
            agent.client = fb_client

        # Bind the active entry ID through the pool's async API. This is only
        # needed for error attribution after a fallback shares the same pool;
        # the selected-pool path already writes the ID directly.
        active_pool = getattr(agent, "_credential_pool", None)
        if active_pool is not None:
            matching_entries = [
                entry
                for entry in active_pool.entries()
                if getattr(entry, "runtime_api_key", None) == agent.api_key
            ]
            agent._credential_pool_entry_id = (
                getattr(matching_entries[0], "id", None)
                if len(matching_entries) == 1
                else None
            )

        # Re-evaluate prompt caching for the new provider/model
        agent._use_prompt_caching, agent._use_native_cache_layout = (
            agent._anthropic_prompt_cache_policy(
                provider=fb_provider,
                base_url=fb_base_url,
                api_mode=fb_api_mode,
                model=fb_model,
            )
        )

        # Update context compressor limits for the fallback model.
        # Without this, compression decisions use the primary model's
        # context window (e.g. 200K) instead of the fallback's (e.g. 32K),
        # causing oversized sessions to overflow the fallback.
        # Also pass _config_context_length so the explicit config override
        # (model.context_length in config.yaml) is respected — without this,
        # the fallback activation drops to 128K even when config says 204800.
        if hasattr(agent, 'context_compressor') and agent.context_compressor:
            from agent.model_metadata import _get_static_context_length

            if fb_provider == "lmstudio":
                fb_context_length = agent._effective_lmstudio_context_length(
                    getattr(agent, "_config_context_length", None),
                    getattr(agent, "_lmstudio_runtime_context_length", None),
                )
            else:
                fb_context_length = None
            if fb_context_length is None:
                fb_context_length = _get_static_context_length(
                    agent.model, base_url=agent.base_url,
                    provider=agent.provider,
                    config_context_length=(
                        None
                        if fb_provider == "lmstudio"
                        else getattr(agent, "_config_context_length", None)
                    ),
                    custom_providers=getattr(agent, "_custom_providers", None),
                )
            agent.context_compressor.update_model(
                model=agent.model,
                context_length=fb_context_length,
                base_url=agent.base_url,
                api_key=getattr(agent, "api_key", ""),  # callable preserved → call_llm
                provider=agent.provider,
                api_mode=agent.api_mode,
            )

        # Keep the prompt's self-identity in sync with the model actually
        # answering, so "what model are you?" doesn't report the primary.
        rewrite_prompt_model_identity(agent, fb_model, fb_provider)

        agent._buffer_status(
            f"🔄 Primary model failed — switching to fallback: "
            f"{fb_model} via {fb_provider}"
        )
        # The buffered line above is dropped on successful recovery, but a
        # provider/model switch is a durable state change operators must see
        # even when the fallback succeeds.  Record a one-shot notice that the
        # success path surfaces exactly once via _emit_pending_fallback_notice
        # (see run_agent.py); it is discarded on terminal failure since the
        # buffered line is flushed instead.  See fallback-observability fix.
        agent._pending_fallback_notice = (
            f"🔄 Switched to fallback model: {old_model} via {old_provider} "
            f"→ {fb_model} via {fb_provider}"
        )
        logger.info(
            "Fallback activated: %s → %s (%s)",
            old_model, fb_model, fb_provider,
        )
        # Reset the stale-call circuit breaker (#58962): the streak measured
        # the OLD provider's unresponsiveness.  Carrying it over would
        # short-circuit the freshly activated fallback before it gets a
        # single stream attempt.
        _reset_stale_streak(agent)
        return True
    except Exception as e:
        if fb_provider == "nous":
            unavailable.add(fb_key)
        logger.error("Failed to activate fallback %s: %s", fb_model, e)
        return await agent._try_activate_fallback(reason)  # try next in chain



async def handle_max_iterations(agent, messages: list, api_call_count: int) -> str:
    """Request a summary when max iterations are reached. Returns the final response text."""
    print(f"⚠️  Reached maximum iterations ({agent.max_iterations}). Requesting summary...")


    summary_request = (
        "You've reached the maximum number of tool-calling iterations allowed. "
        "Please provide a final response summarizing what you've found and accomplished so far, "
        "without calling any more tools."
    )
    messages.append({"role": "user", "content": summary_request})

    try:
        # Build API messages, stripping internal-only fields
        # (finish_reason, reasoning) that strict APIs like Mistral reject with 422
        _needs_sanitize = agent._should_sanitize_tool_calls()
        api_messages = []
        for msg in messages:
            api_msg = msg.copy()
            agent._copy_reasoning_content_for_api(msg, api_msg)
            for internal_field in ("reasoning", "finish_reason", "_thinking_prefill"):
                api_msg.pop(internal_field, None)
            # Strict OpenAI-compatible gateways (Fireworks-backed OpenCode Go,
            # Mistral, Moonshot/Kimi) reject any message key outside the Chat
            # Completions schema. The main loop drops these via
            # ChatCompletionsTransport.convert_messages(), but the summary path
            # hand-builds messages and calls chat.completions.create() directly,
            # bypassing the transport — so mirror that sanitization here:
            # tool_name (SQLite FTS bookkeeping), the codex_* reasoning carriers,
            # timestamp (preserved on gateway user replay entries for the
            # stale-confirmation expiry check — #47868 rejection class),
            # and every Hermes-internal underscore-prefixed scaffolding key.
            for schema_foreign in ("tool_name", "codex_reasoning_items", "codex_message_items", "timestamp"):
                api_msg.pop(schema_foreign, None)
            # api_content (the persist-what-you-send sidecar) carries the
            # exact bytes every main-loop call sent for this message —
            # substitute it before dropping the key (Hermes bookkeeping,
            # never a provider field), mirroring the loop's api_messages
            # build. Popping without substituting would send CLEAN content
            # here, diverging the summary request's prefix at the EARLIEST
            # sidecar-carrying message and re-prefilling the whole transcript
            # at exactly the moment the context is largest.
            substitute_api_content(api_msg)
            for internal_key in [k for k in api_msg if isinstance(k, str) and k.startswith("_")]:
                api_msg.pop(internal_key, None)
            if _needs_sanitize:
                # In MoA mode, agent.model is the virtual preset name,
                # not the actual aggregator model.  Resolve the real
                # aggregator model so Gemini preserves thought_signature.
                _sanitize_model = agent.model
                if agent.provider == "moa":
                    _moa_client = getattr(agent, "client", None)
                    if _moa_client is not None:
                        _agg_slot = getattr(_moa_client, "last_aggregator_slot", None)
                        if _agg_slot and _agg_slot.get("model"):
                            _sanitize_model = _agg_slot["model"]
                agent._sanitize_tool_calls_for_strict_api(api_msg, model=_sanitize_model)
            api_messages.append(api_msg)

        effective_system = agent._cached_system_prompt or ""
        if agent.ephemeral_system_prompt:
            effective_system = (effective_system + "\n\n" + agent.ephemeral_system_prompt).strip()
        if effective_system:
            api_messages = [{"role": "system", "content": effective_system}] + api_messages
        if agent.prefill_messages:
            sys_offset = 1 if effective_system else 0
            for idx, pfm in enumerate(agent.prefill_messages):
                api_messages.insert(sys_offset + idx, pfm.copy())

        # Same safety net as the main loop: repair tool-call/result
        # pairing before asking for a final summary.  Compression and
        # session resume can leave a tool result whose parent assistant
        # tool_call was summarized away; Responses API rejects that as
        # "No tool call found for function call output".
        api_messages = agent._sanitize_api_messages(api_messages)

        # Same safety net as the main loop: drop thinking-only assistant
        # turns so Anthropic-family providers don't 400 the summary call.
        api_messages = agent._drop_thinking_only_and_merge_users(api_messages)

        summary_extra_body = {}
        try:
            from agent.auxiliary_client import _fixed_temperature_for_model, OMIT_TEMPERATURE as _OMIT_TEMP
        except Exception:
            _fixed_temperature_for_model = None
            _OMIT_TEMP = None
        _raw_summary_temp = (
            _fixed_temperature_for_model(agent.model, agent.base_url)
            if _fixed_temperature_for_model is not None
            else None
        )
        _omit_summary_temperature = _raw_summary_temp is _OMIT_TEMP
        _summary_temperature = None if _omit_summary_temperature else _raw_summary_temp
        _is_nous = "nousresearch" in agent._base_url_lower
        # LM Studio uses top-level `reasoning_effort` (not extra_body.reasoning).
        # Mirror ChatCompletionsTransport.build_kwargs() so the summary path
        # — which calls chat.completions.create() directly without going
        # through the transport — sends the same shape the transport does.
        _is_lmstudio_summary = (
            (agent.provider or "").strip().lower() == "lmstudio"
            and await agent._supports_reasoning_extra_body()
        )
        _lm_reasoning_effort: str | None = (
            await agent._resolve_lmstudio_summary_reasoning_effort()
            if _is_lmstudio_summary else None
        )
        if not _is_lmstudio_summary and await agent._supports_reasoning_extra_body():
            if agent.reasoning_config is not None:
                summary_extra_body["reasoning"] = agent.reasoning_config
            else:
                summary_extra_body["reasoning"] = {
                    "enabled": True,
                    "effort": "medium"
                }
        if _is_nous:
            from agent.portal_tags import nous_portal_tags as _portal_tags
            summary_extra_body["tags"] = _portal_tags()

        if agent.api_mode == "codex_responses":
            codex_kwargs = await agent._build_api_kwargs(api_messages)
            codex_kwargs.pop("tools", None)
            summary_response = await agent._execute_model_request(
                codex_kwargs, use_streaming=False
            )
            _ct_sum = agent._get_transport()
            _cnr_sum = _ct_sum.normalize_response(summary_response)
            final_response = (_cnr_sum.content or "").strip()
        else:
            summary_kwargs = {
                "model": agent.model,
                "messages": api_messages,
            }
            if _summary_temperature is not None:
                summary_kwargs["temperature"] = _summary_temperature
            if agent.max_tokens is not None:
                summary_kwargs.update(agent._max_tokens_param(agent.max_tokens))
            if _lm_reasoning_effort is not None:
                summary_kwargs["reasoning_effort"] = _lm_reasoning_effort

            # Merge the profile's canonical body even when routing is unset:
            # profiles may always emit required metadata such as Portal tags.
            provider_preferences = _provider_preferences_for_agent(agent)
            profile_extra_body = {}
            try:
                from providers import _get_provider_profile_cached

                provider_profile = _get_provider_profile_cached(agent.provider)
                if provider_profile is not None:
                    profile_extra_body = provider_profile.build_extra_body(
                        session_id=getattr(agent, "session_id", None),
                        provider_preferences=provider_preferences or None,
                        model=agent.model,
                        base_url=agent.base_url,
                        reasoning_config=agent.reasoning_config,
                    )
            except Exception:
                pass

            if profile_extra_body:
                summary_extra_body.update(profile_extra_body)
            if provider_preferences and "provider" not in profile_extra_body and (
                (agent.provider or "").strip().lower() == "openrouter"
                or agent._is_openrouter_url()
            ):
                summary_extra_body["provider"] = provider_preferences

            # Pareto Code router plugin — model-gated. Same shape as
            # the main-loop emission so summary calls on
            # openrouter/pareto-code respect the user's coding-score floor.
            if (
                agent.model == "openrouter/pareto-code"
                and (
                    (agent.provider or "").strip().lower() == "openrouter"
                    or agent._is_openrouter_url()
                )
                and agent.openrouter_min_coding_score is not None
                and agent.openrouter_min_coding_score != ""
            ):
                try:
                    _ps = float(agent.openrouter_min_coding_score)
                except (TypeError, ValueError):
                    _ps = None
                if _ps is not None and 0.0 <= _ps <= 1.0:
                    summary_extra_body["plugins"] = [
                        {"id": "pareto-router", "min_coding_score": _ps}
                    ]

            if summary_extra_body:
                summary_kwargs["extra_body"] = summary_extra_body

            if agent.api_mode == "anthropic_messages":
                _tsum = agent._get_transport()
                _ant_kw = _tsum.build_kwargs(
                    model=agent.model,
                    messages=api_messages,
                    tools=None,
                    max_tokens=agent.max_tokens,
                    reasoning_config=agent.reasoning_config,
                    is_oauth=agent._is_anthropic_oauth,
                    preserve_dots=agent._anthropic_preserve_dots(),
                    base_url=getattr(agent, "_anthropic_base_url", None),
                )
                _ant_kw = _merge_nous_portal_messages_extra_body(agent, _ant_kw)
                summary_response = await agent._execute_model_request(
                    _ant_kw, use_streaming=False
                )
                _summary_result = _tsum.normalize_response(summary_response, strip_tool_prefix=agent._is_anthropic_oauth)
                final_response = (_summary_result.content or "").strip()
            else:
                summary_response = await agent._execute_model_request(
                    summary_kwargs, use_streaming=False
                )
                _summary_result = agent._get_transport().normalize_response(summary_response)
                final_response = (_summary_result.content or "").strip()

        if final_response:
            if "<think>" in final_response:
                final_response = re.sub(r'<think>.*?</think>\s*', '', final_response, flags=re.DOTALL).strip()
            if final_response:
                messages.append({"role": "assistant", "content": final_response})
            else:
                final_response = "I reached the iteration limit and couldn't generate a summary."
        else:
            # Retry summary generation
            if agent.api_mode == "codex_responses":
                codex_kwargs = await agent._build_api_kwargs(api_messages)
                codex_kwargs.pop("tools", None)
                retry_response = await agent._execute_model_request(
                    codex_kwargs, use_streaming=False
                )
                _ct_retry = agent._get_transport()
                _cnr_retry = _ct_retry.normalize_response(retry_response)
                final_response = (_cnr_retry.content or "").strip()
            elif agent.api_mode == "anthropic_messages":
                _tretry = agent._get_transport()
                _ant_kw2 = _tretry.build_kwargs(
                    model=agent.model,
                    messages=api_messages,
                    tools=None,
                    is_oauth=agent._is_anthropic_oauth,
                    max_tokens=agent.max_tokens,
                    reasoning_config=agent.reasoning_config,
                    preserve_dots=agent._anthropic_preserve_dots(),
                    base_url=getattr(agent, "_anthropic_base_url", None),
                )
                _ant_kw2 = _merge_nous_portal_messages_extra_body(agent, _ant_kw2)
                retry_response = await agent._execute_model_request(
                    _ant_kw2, use_streaming=False
                )
                _retry_result = _tretry.normalize_response(retry_response, strip_tool_prefix=agent._is_anthropic_oauth)
                final_response = (_retry_result.content or "").strip()
            else:
                summary_kwargs = {
                    "model": agent.model,
                    "messages": api_messages,
                }
                if _summary_temperature is not None:
                    summary_kwargs["temperature"] = _summary_temperature
                if agent.max_tokens is not None:
                    summary_kwargs.update(agent._max_tokens_param(agent.max_tokens))
                if _lm_reasoning_effort is not None:
                    summary_kwargs["reasoning_effort"] = _lm_reasoning_effort
                if summary_extra_body:
                    summary_kwargs["extra_body"] = summary_extra_body

                summary_response = await agent._execute_model_request(
                    summary_kwargs, use_streaming=False
                )
                _retry_result = agent._get_transport().normalize_response(summary_response)
                final_response = (_retry_result.content or "").strip()

            if final_response:
                if "<think>" in final_response:
                    final_response = re.sub(r'<think>.*?</think>\s*', '', final_response, flags=re.DOTALL).strip()
                if final_response:
                    messages.append({"role": "assistant", "content": final_response})
                else:
                    final_response = "I reached the iteration limit and couldn't generate a summary."
            else:
                final_response = "I reached the iteration limit and couldn't generate a summary."

    except Exception as e:
        logger.warning(f"Failed to get summary response: {e}")
        final_response = f"I reached the maximum iterations ({agent.max_iterations}) but couldn't summarize. Error: {str(e)}"
    return final_response



async def cleanup_task_resources(agent, task_id: str) -> None:
    """Clean up terminal and browser resources for a given task.

    Skips ``cleanup_vm`` when the active terminal environment is marked
    persistent (``persistent_filesystem=True``) so that long-lived sandbox
    containers survive between turns. The idle reaper in
    ``terminal_tool._cleanup_inactive_envs`` still tears them down once
    ``terminal.lifetime_seconds`` is exceeded. Non-persistent backends are
    torn down per-turn as before to prevent resource leakage.

    Headed browser sessions remain visible between turns and are reclaimed by
    the browser inactivity task. Headless sessions are closed per turn.
    """
    try:
        if is_persistent_env(task_id):
            if agent.verbose_logging:
                logging.debug(
                    f"Skipping per-turn cleanup_vm for persistent env {task_id}; "
                    f"idle reaper will handle it."
                )
        else:
            await cleanup_vm(task_id)
    except Exception as exc:
        if agent.verbose_logging:
            logger.warning("Failed to cleanup VM for task %s: %s", task_id, exc)

    try:
        from tools.browser_tool import _is_headed_mode, cleanup_browser

        headed = await _is_headed_mode()
        if headed:
            if agent.verbose_logging:
                logging.debug(
                    "Skipping per-turn cleanup_browser for headed session %s; "
                    "idle reaper will handle it.",
                    task_id,
                )
        else:
            await cleanup_browser(task_id)
    except Exception as exc:
        if agent.verbose_logging:
            logger.warning("Failed to cleanup browser for task %s: %s", task_id, exc)


# ── Provider fallback ──────────────────────────────────────────────────



__all__ = [
    "interruptible_api_call",
    "build_api_kwargs",
    "build_assistant_message",
    "try_activate_fallback",
    "handle_max_iterations",
    "cleanup_task_resources",
    "interruptible_streaming_api_call",
]
