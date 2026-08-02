"""Core NeMo Relay adapters for physical Hermes provider attempts."""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from agent import relay_runtime

logger = logging.getLogger(__name__)


_PROVIDER_MESSAGE_EXTENSION_KEYS = frozenset(
    {"reasoning_content", "reasoning_details"}
)
_RELAY_INTERNAL_PROVIDER_HEADERS = frozenset(
    {"x-dynamo-parent-session-id", "x-dynamo-session-id"}
)


async def execute(
    request: dict[str, Any],
    callback: Callable[[dict[str, Any]], Any],
    *,
    session_id: str,
    name: str,
    model_name: str,
    metadata: dict[str, Any] | None = None,
    defer_logical_completion: bool = False,
) -> Any:
    """Run one native-async physical provider attempt through Relay."""
    runtime, session, parent = relay_runtime.resolve_execution_context(session_id)
    if runtime is None or session is None or not runtime.managed_execution_enabled():
        return await callback(request)
    logical = await _logical_parent(runtime, session, parent, metadata)
    parent = logical[1] if logical is not None else parent

    relay_request_body = _relay_request_body(request, metadata)
    relay_request = runtime.relay.LLMRequest({}, relay_request_body)
    codec_baseline_body = _codec_round_trip_request_body(
        runtime.relay,
        relay_request,
        relay_request_body=relay_request_body,
        metadata=metadata,
    )
    raw_response: dict[str, Any] = {}
    callback_error: BaseException | None = None
    callback_context = contextvars.copy_context()

    async def invoke(next_request: Any) -> Any:
        nonlocal callback_error
        try:
            final_request = _provider_request(
                request,
                next_request,
                relay_request_body=relay_request_body,
                codec_baseline_body=codec_baseline_body,
                metadata=metadata,
            )
            async def call_provider() -> Any:
                return await callback(final_request)

            task = callback_context.copy().run(
                asyncio.create_task,
                call_provider(),
            )
            raw = await task
        except BaseException as exc:
            callback_error = exc
            raise
        raw_response["value"] = raw
        raw_response["json"] = _jsonable(raw)
        return raw_response["json"]

    try:
        managed = await runtime.run_in_session_async(
            session,
            runtime.relay.llm.execute,
            name,
            relay_request,
            invoke,
            handle=parent,
            metadata=_jsonable(metadata or {}),
            model_name=model_name,
            codec=_codec(runtime.relay, metadata),
            response_codec=_codec(runtime.relay, metadata),
        )
    except BaseException as exc:
        if (
            callback_error is not None
            and relay_runtime._is_relay_wrapped_callback_error(exc, callback_error)
        ):
            raise callback_error
        if await _recover_successful_callback(
            raw_response,
            relay_error=exc,
            callback_error=callback_error,
            logical=logical,
            defer_logical_completion=defer_logical_completion,
        ):
            return raw_response["value"]
        raise

    if not defer_logical_completion:
        await _complete_logical(logical, outcome="success")
    if "value" in raw_response and _json_equal(managed, raw_response["json"]):
        return raw_response["value"]
    return _namespace(managed)


async def execute_current(
    request: dict[str, Any],
    callback: Callable[[dict[str, Any]], Any],
    *,
    name: str,
    model_name: str,
    metadata: dict[str, Any] | None = None,
    defer_logical_completion: bool = False,
) -> Any:
    """Run an async provider attempt under the inherited turn when present."""
    turn = relay_runtime.active_turn()
    if turn is None:
        return await callback(request)
    return await execute(
        request,
        callback,
        session_id=turn.lease.session_id,
        name=name,
        model_name=model_name,
        metadata=metadata,
        defer_logical_completion=defer_logical_completion,
    )


class AnthropicStreamAccumulator:
    """Rebuild an Anthropic Message from post-intercept SSE events."""

    def __init__(self) -> None:
        self._message: dict[str, Any] = {}
        self._blocks: dict[int, dict[str, Any]] = {}

    def observe(self, event: Any) -> None:
        payload = _jsonable(event)
        if not isinstance(payload, dict):
            return
        event_type = payload.get("type")
        if event_type == "message_start":
            message = payload.get("message")
            if isinstance(message, dict):
                for key in ("id", "type", "role", "model", "usage"):
                    if key in message:
                        self._message[key] = message[key]
            return
        if event_type == "content_block_start":
            index = payload.get("index")
            block = payload.get("content_block")
            if isinstance(index, int) and isinstance(block, dict):
                self._blocks[index] = dict(block)
            return
        if event_type == "content_block_delta":
            index = payload.get("index")
            delta = payload.get("delta")
            if not isinstance(index, int) or not isinstance(delta, dict):
                return
            block = self._blocks.setdefault(index, {})
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                block["text"] = str(block.get("text") or "") + str(
                    delta.get("text") or ""
                )
            elif delta_type == "thinking_delta":
                block["thinking"] = str(block.get("thinking") or "") + str(
                    delta.get("thinking") or ""
                )
            elif delta_type == "signature_delta":
                block["signature"] = str(block.get("signature") or "") + str(
                    delta.get("signature") or ""
                )
            elif delta_type == "input_json_delta":
                partial = str(block.pop("_partial_json", "")) + str(
                    delta.get("partial_json") or ""
                )
                block["_partial_json"] = partial
            elif delta_type == "citations_delta" and "citation" in delta:
                block.setdefault("citations", []).append(delta["citation"])
            return
        if event_type == "message_delta":
            delta = payload.get("delta")
            if isinstance(delta, dict):
                for key in ("stop_reason", "stop_sequence"):
                    if key in delta:
                        self._message[key] = delta[key]
            if "usage" in payload:
                usage = payload["usage"]
                current_usage = self._message.get("usage")
                if isinstance(current_usage, dict) and isinstance(usage, dict):
                    self._message["usage"] = {**current_usage, **usage}
                else:
                    self._message["usage"] = usage

    def finalize(self) -> dict[str, Any]:
        blocks = []
        for index in sorted(self._blocks):
            block = dict(self._blocks[index])
            partial = block.pop("_partial_json", None)
            if partial is not None:
                try:
                    block["input"] = json.loads(partial)
                except (TypeError, ValueError):
                    block["input"] = partial
            blocks.append(block)
        return {**self._message, "content": blocks}

    def response(self, base: Any = None) -> Any:
        """Return the attribute-shaped response consumed by Hermes."""
        assembled = self.finalize()
        base_payload = _jsonable(base)
        if not isinstance(base_payload, dict):
            base_payload = {}
        content = assembled.pop("content", [])
        merged = {**base_payload, **assembled}
        if content or "content" not in merged:
            merged["content"] = content
        return _namespace(merged)


async def _logical_parent(
    runtime: relay_runtime.RelayRuntime,
    session: Any,
    parent: Any,
    metadata: dict[str, Any] | None,
) -> tuple[relay_runtime.RelayTurnContext, Any, str] | None:
    """Open or reuse a logical LLM scope without entering sync Relay code.

    The async provider path must not call ``run_in_session`` while the event
    loop is active.  The bookkeeping locks are deliberately held only while
    reading/writing the handle map; the Relay operation itself is awaited
    outside those thread locks so a competing coroutine cannot block the
    event loop while this task is suspended.
    """
    turn = relay_runtime.active_turn(session.session_id)
    request_id = str((metadata or {}).get("api_request_id") or "")
    if turn is None or not request_id or turn.lease.host is not runtime:
        return None
    with turn.finalize_lock:
        if turn.closed:
            return None
    with turn.logical_llm_lock:
        handle = turn.logical_llm_calls.get(request_id)
    if handle is None:
        handle = await runtime.run_in_session_async(
            session,
            runtime.relay.scope.push,
            relay_runtime.LOGICAL_LLM_SCOPE,
            runtime.relay.ScopeType.Function,
            handle=parent,
            input={},
            metadata={
                relay_runtime.RUNTIME_SCHEMA_KEY: relay_runtime.RUNTIME_SCHEMA_VERSION,
                relay_runtime.RUNTIME_INSTANCE_KEY: runtime.runtime_id,
                "hermes.call_role": str(
                    (metadata or {}).get("call_role") or "primary"
                ),
            },
        )
        with turn.finalize_lock:
            if turn.closed:
                return None
        with turn.logical_llm_lock:
            existing = turn.logical_llm_calls.get(request_id)
            if existing is None:
                turn.logical_llm_calls[request_id] = handle
            else:
                handle = existing
    return turn, handle, request_id


async def _complete_logical(
    logical: tuple[relay_runtime.RelayTurnContext, Any, str] | None,
    *,
    outcome: str,
) -> None:
    """Complete a logical LLM scope through Relay's native async API."""
    if logical is None:
        return
    turn, handle, request_id = logical
    lease = turn.lease
    if not isinstance(lease.host, relay_runtime.RelayRuntime):
        return
    with turn.finalize_lock:
        with turn.logical_llm_lock:
            if turn.logical_llm_calls.get(request_id) is not handle:
                return
        if lease.session is None:
            return
    try:
        await lease.host.run_in_session_async(
            lease.session,
            lease.host.relay.scope.pop,
            handle,
            output={"outcome": outcome},
            metadata={
                relay_runtime.RUNTIME_SCHEMA_KEY: relay_runtime.RUNTIME_SCHEMA_VERSION,
                relay_runtime.RUNTIME_INSTANCE_KEY: lease.host.runtime_id,
            },
        )
    except Exception:
        # The provider result is authoritative. Retain the handle so turn
        # finalization can retry cleanup without changing that result.
        logger.warning(
            "Hermes Relay logical LLM finalization failed",
            exc_info=True,
        )
        return
    with turn.finalize_lock:
        with turn.logical_llm_lock:
            if turn.logical_llm_calls.get(request_id) is handle:
                turn.logical_llm_calls.pop(request_id, None)


async def _recover_successful_callback(
    raw_response: dict[str, Any],
    *,
    relay_error: BaseException,
    callback_error: BaseException | None,
    logical: tuple[relay_runtime.RelayTurnContext, Any, str] | None,
    defer_logical_completion: bool,
) -> bool:
    """Recover a provider result after async Relay post-processing fails."""
    if (
        not isinstance(relay_error, Exception)
        or callback_error is not None
        or "value" not in raw_response
    ):
        return False
    logger.warning(
        "NeMo Relay LLM post-processing failed after provider success; "
        "returning the provider response",
        exc_info=True,
    )
    if not defer_logical_completion:
        await _complete_logical(logical, outcome="success")
    return True


def _is_cancellation(error: BaseException) -> bool:
    return isinstance(
        error,
        (asyncio.CancelledError, InterruptedError, KeyboardInterrupt),
    )


async def complete_logical_call(api_request_id: str, *, outcome: str) -> None:
    """Complete the active turn's logical LLM call asynchronously."""
    turn = relay_runtime.active_turn()
    if turn is None or not api_request_id:
        return
    with turn.logical_llm_lock:
        handle = turn.logical_llm_calls.get(api_request_id)
    if handle is not None:
        await _complete_logical((turn, handle, api_request_id), outcome=outcome)


def _provider_request(
    original: dict[str, Any],
    request: Any,
    *,
    relay_request_body: dict[str, Any],
    codec_baseline_body: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    content = getattr(request, "content", request)
    if not isinstance(content, dict):
        content = relay_request_body
    if codec_baseline_body is None or _json_equal(content, relay_request_body):
        final = dict(original)
    else:
        baseline = codec_baseline_body
        intercepted = _provider_request_body(content, metadata)
        final = dict(original)
        # Typed codecs may not represent provider-specific fields. Overlay only
        # values that changed from the codec-facing baseline so unrelated
        # intercepts cannot delete or normalize unknown provider arguments.
        for key in baseline.keys() | intercepted.keys():
            if key not in intercepted:
                final.pop(key, None)
            elif key not in baseline or not _json_equal(
                intercepted[key],
                baseline[key],
            ):
                final[key] = intercepted[key]
        _restore_provider_message_extensions(
            original,
            final,
            baseline=baseline,
            intercepted=intercepted,
        )
    headers = getattr(request, "headers", None)
    if isinstance(headers, dict):
        headers = {
            key: value
            for key, value in headers.items()
            if str(key).lower() not in _RELAY_INTERNAL_PROVIDER_HEADERS
        }
    if headers:
        final["extra_headers"] = {
            **dict(final.get("extra_headers") or {}),
            **headers,
        }
    return final


def _relay_request_body(
    request: dict[str, Any], metadata: dict[str, Any] | None
) -> dict[str, Any]:
    body = _jsonable(request)
    if not isinstance(body, dict):
        return {}
    # The Responses SDK accepts ``tools=None`` as "no tools", while Relay's
    # typed Responses codec correctly expects either an array or an absent
    # field. Normalize only the codec-facing copy; the original provider
    # request is restored when no interceptor changes it.
    if str((metadata or {}).get("api_mode") or "") == "codex_responses":
        body = dict(body)
        if body.get("tools") is None:
            body.pop("tools", None)
        elif isinstance(body.get("tools"), list):
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        key: value
                        for key, value in tool.items()
                        if key != "type"
                    },
                }
                if isinstance(tool, dict)
                and tool.get("type") == "function"
                and "function" not in tool
                else tool
                for tool in body["tools"]
            ]
    elif str((metadata or {}).get("api_mode") or "") == "chat_completions":
        tools = body.get("tools")
        if isinstance(tools, list):
            body = dict(body)
            body["tools"] = [
                {"type": "function", **tool}
                if isinstance(tool, dict)
                and "function" in tool
                and "type" not in tool
                else tool
                for tool in tools
            ]
    return body


def _restore_provider_message_extensions(
    original: dict[str, Any],
    final: dict[str, Any],
    *,
    baseline: dict[str, Any],
    intercepted: dict[str, Any],
) -> None:
    """Restore provider wire fields that Relay's typed codec cannot represent."""
    original_messages = original.get("messages")
    final_messages = final.get("messages")
    baseline_messages = baseline.get("messages")
    intercepted_messages = intercepted.get("messages")
    if not all(
        isinstance(messages, list)
        for messages in (
            original_messages,
            final_messages,
            baseline_messages,
            intercepted_messages,
        )
    ):
        return
    if not (
        len(original_messages)
        == len(final_messages)
        == len(baseline_messages)
        == len(intercepted_messages)
    ):
        return
    for original_message, final_message, baseline_message, intercepted_message in zip(
        original_messages,
        final_messages,
        baseline_messages,
        intercepted_messages,
        strict=True,
    ):
        if not all(
            isinstance(message, dict)
            for message in (
                original_message,
                final_message,
                baseline_message,
                intercepted_message,
            )
        ):
            continue
        for key in _PROVIDER_MESSAGE_EXTENSION_KEYS:
            if (
                key in original_message
                and key not in baseline_message
                and key not in intercepted_message
                and key not in final_message
            ):
                final_message[key] = original_message[key]


def _codec_round_trip_request_body(
    relay: Any,
    relay_request: Any,
    *,
    relay_request_body: dict[str, Any],
    metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the codec-only request shape used to identify real rewrites."""
    codec = _codec(relay, metadata)
    if codec is None:
        return _provider_request_body(relay_request_body, metadata)
    try:
        annotated = codec.decode(relay_request)
        encoded = codec.encode(annotated, relay_request)
        content = getattr(encoded, "content", encoded)
        if isinstance(content, dict):
            return _provider_request_body(content, metadata)
    except Exception:
        logger.warning(
            "NeMo Relay request codec baseline failed; ignoring request rewrites",
            exc_info=True,
        )
        return None
    logger.warning(
        "NeMo Relay request codec returned an unsupported baseline; "
        "ignoring request rewrites"
    )
    return None


def _provider_request_body(
    content: dict[str, Any], metadata: dict[str, Any] | None
) -> dict[str, Any]:
    body = dict(content)
    if str((metadata or {}).get("api_mode") or "") != "codex_responses":
        return body
    tools = body.get("tools")
    if not isinstance(tools, list):
        return body
    body["tools"] = [
        {
            "type": "function",
            **dict(tool["function"]),
        }
        if isinstance(tool, dict)
        and tool.get("type") == "function"
        and isinstance(tool.get("function"), dict)
        else tool
        for tool in tools
    ]
    return body


def _codec(relay: Any, metadata: dict[str, Any] | None) -> Any:
    api_mode = str((metadata or {}).get("api_mode") or "")
    codecs = getattr(relay, "codecs", None)
    if codecs is None:
        return None
    if api_mode == "chat_completions":
        codec = getattr(codecs, "OpenAIChatCodec", None)
    elif api_mode == "anthropic_messages":
        codec = getattr(codecs, "AnthropicMessagesCodec", None)
    elif api_mode == "codex_responses":
        codec = getattr(codecs, "OpenAIResponsesCodec", None)
    else:
        codec = None
    return codec() if callable(codec) else None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    model_dump = getattr(type(value), "model_dump", None)
    if callable(model_dump):
        try:
            return _jsonable(value.model_dump(mode="json"))
        except Exception:
            pass
    try:
        attributes = {
            str(key): item
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    except (TypeError, AttributeError):
        return str(value)
    return _jsonable(attributes) if attributes else str(value)


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{
            str(key): _namespace(item) for key, item in value.items()
        })
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _json_equal(left: Any, right: Any) -> bool:
    try:
        return json.dumps(
            _jsonable(left), sort_keys=True, separators=(",", ":")
        ) == json.dumps(_jsonable(right), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return False
