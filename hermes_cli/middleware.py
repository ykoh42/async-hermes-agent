"""Hermes middleware contract helpers.

Observer hooks report what happened. Middleware can change what happens by
rewriting a request or wrapping the actual execution callback. Keep the small
contract helpers here so agent-loop call sites and plugins share one vocabulary.
"""

from __future__ import annotations

import logging
import inspect
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)

OBSERVER_SCHEMA_VERSION = "hermes.observer.v1"
MIDDLEWARE_SCHEMA_VERSION = "hermes.middleware.v1"

TOOL_REQUEST_MIDDLEWARE = "tool_request"
TOOL_EXECUTION_MIDDLEWARE = "tool_execution"
LLM_REQUEST_MIDDLEWARE = "llm_request"
LLM_EXECUTION_MIDDLEWARE = "llm_execution"

# Back-compat aliases for older PoC branches that used API terminology.
API_REQUEST_MIDDLEWARE = LLM_REQUEST_MIDDLEWARE
API_EXECUTION_MIDDLEWARE = LLM_EXECUTION_MIDDLEWARE

VALID_MIDDLEWARE: set[str] = {
    TOOL_REQUEST_MIDDLEWARE,
    TOOL_EXECUTION_MIDDLEWARE,
    LLM_REQUEST_MIDDLEWARE,
    LLM_EXECUTION_MIDDLEWARE,
}


@dataclass
class RequestMiddlewareResult:
    """Result of applying request middleware to a mutable payload."""

    payload: Any
    original_payload: Any
    changed: bool = False
    trace: List[Dict[str, Any]] = field(default_factory=list)


def observer_payload(**kwargs: Any) -> Dict[str, Any]:
    kwargs.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
    return kwargs


def middleware_payload(**kwargs: Any) -> Dict[str, Any]:
    kwargs.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
    kwargs.setdefault("middleware_schema_version", MIDDLEWARE_SCHEMA_VERSION)
    return kwargs


def _safe_copy(payload: Any) -> Any:
    """Deep-copy a request payload, tolerating non-deepcopyable members.

    Request payloads are normally plain JSON-shaped dicts, but an LLM request
    can occasionally carry non-deepcopyable objects (clients, callbacks, file
    handles). A hard ``deepcopy`` failure there would otherwise abort the whole
    request-middleware pass. Fall back to a shallow ``dict`` copy so middleware
    still runs and the original nested objects are shared by reference rather
    than corrupting the live payload.
    """
    try:
        return deepcopy(payload)
    except Exception as exc:  # pragma: no cover - exercised via fallback test
        logger.debug("deepcopy failed for request payload (%s); using shallow copy", exc)
        if isinstance(payload, dict):
            return dict(payload)
        return payload


async def apply_llm_request_middleware(
    request: Dict[str, Any],
    **context: Any,
) -> RequestMiddlewareResult:
    """Apply native-async LLM request middleware.

    Middleware may return ``{"request": {...}}`` to replace the effective
    provider kwargs before Hermes sends them. Callbacks must be coroutines;
    async Hermes does not execute synchronous plugin code in an active turn.
    """
    if not _has_middleware(LLM_REQUEST_MIDDLEWARE):
        return RequestMiddlewareResult(
            payload=request,
            original_payload=request,
            changed=False,
            trace=[],
        )

    original_request = _safe_copy(request)
    current_request = _safe_copy(original_request)
    trace: List[Dict[str, Any]] = []

    for result in await _invoke_middleware_async(
        LLM_REQUEST_MIDDLEWARE,
        request=current_request,
        original_request=original_request,
        **context,
    ):
        if not isinstance(result, dict):
            continue
        next_request = result.get("request")
        if not isinstance(next_request, dict):
            continue
        current_request = _safe_copy(next_request)
        trace.append(_trace_entry(result))

    return RequestMiddlewareResult(
        payload=current_request,
        original_payload=original_request,
        changed=bool(trace),
        trace=trace,
    )




async def apply_tool_request_middleware(
    tool_name: str,
    args: Dict[str, Any],
    **context: Any,
) -> RequestMiddlewareResult:
    """Apply native-async tool request middleware.

    Tool arguments are part of an active conversation turn, so middleware is
    not permitted to execute synchronously and stall the event loop.  A
    migrated callback receives the same payload as before and simply returns
    its directive from an ``async def`` function.
    """
    original_args = _safe_copy(args)
    current_args = _safe_copy(original_args)
    trace: List[Dict[str, Any]] = []

    session_id = str(context.get("session_id") or "")
    skip_relay = bool(context.pop("skip_relay", False))
    if session_id and not skip_relay:
        from agent import relay_runtime

        relay_args = relay_runtime.apply_tool_request_intercepts(
            session_id=session_id,
            tool_name=tool_name,
            args=current_args,
        )
        if relay_args != current_args:
            current_args = _safe_copy(relay_args)
            trace.append({"source": "nemo_relay"})

    if not _has_middleware(TOOL_REQUEST_MIDDLEWARE):
        return RequestMiddlewareResult(
            payload=args if not trace else current_args,
            original_payload=args,
            changed=bool(trace),
            trace=trace,
        )

    for result in await _invoke_middleware_async(
        TOOL_REQUEST_MIDDLEWARE,
        tool_name=tool_name,
        args=current_args,
        original_args=original_args,
        **context,
    ):
        if not isinstance(result, dict):
            continue
        next_args = result.get("args")
        if not isinstance(next_args, dict):
            continue
        current_args = _safe_copy(next_args)
        trace.append(_trace_entry(result))

    return RequestMiddlewareResult(
        payload=current_args,
        original_payload=original_args,
        changed=bool(trace),
        trace=trace,
    )
async def run_llm_execution_middleware(
    request: Dict[str, Any],
    next_call: Callable[[Dict[str, Any]], Any],
    **context: Any,
) -> Any:
    """Async execution chain for the coroutine-native agent loop.

    Middleware callbacks and ``next_call`` are native coroutines. The
    single-use ``next_call`` contract and failure fallback match the
    single-use continuation contract of the execution middleware.
    """
    callbacks = _get_middleware_callbacks(LLM_EXECUTION_MIDDLEWARE)
    if not callbacks:
        return await next_call(request)
    return await _run_execution_chain_async(
        LLM_EXECUTION_MIDDLEWARE,
        callbacks,
        next_call,
        request=request,
        original_request=context.pop("original_request", request),
        **context,
    )




async def run_tool_execution_middleware(
    tool_name: str,
    args: Dict[str, Any],
    next_call: Callable[[Dict[str, Any]], Any],
    **context: Any,
) -> Any:
    """Run a tool through coroutine-native execution middleware.

    Async Hermes never hides a tool call in a worker thread.  Middleware that
    wraps a tool call therefore has the same requirement: it must await its
    ``next_call`` continuation and return an awaitable itself.  The legacy
    synchronous chain remains private for code paths not exposed by this
    package's async agent API.
    """
    callbacks = _get_middleware_callbacks(TOOL_EXECUTION_MIDDLEWARE)
    if not callbacks:
        return await next_call(args)
    return await _run_execution_chain_async(
        TOOL_EXECUTION_MIDDLEWARE,
        callbacks,
        next_call,
        tool_name=tool_name,
        args=args,
        original_args=context.pop("original_args", args),
        **context,
    )


class AsyncMiddlewareCapabilityError(RuntimeError):
    """Raised when a synchronous plugin callback reaches the async agent."""


async def _invoke_middleware_async(kind: str, **kwargs: Any) -> List[Any]:
    """Invoke middleware callbacks without a sync fallback or worker bridge."""
    callbacks = _get_middleware_callbacks(kind)
    results: List[Any] = []
    payload = middleware_payload(**kwargs)
    for callback in callbacks:
        try:
            result = callback(**payload)
            if not inspect.isawaitable(result):
                raise AsyncMiddlewareCapabilityError(
                    "Async Hermes requires coroutine middleware callbacks; "
                    f"{getattr(callback, '__name__', repr(callback))} is synchronous"
                )
            result = await result
            if result is not None:
                results.append(result)
        except AsyncMiddlewareCapabilityError:
            raise
        except Exception as exc:
            logger.warning(
                "Middleware '%s' callback %s raised: %s",
                kind,
                getattr(callback, "__name__", repr(callback)),
                exc,
            )
    return results


def _has_middleware(kind: str) -> bool:
    from hermes_cli.plugins import has_middleware

    return has_middleware(kind)


def _get_middleware_callbacks(kind: str) -> List[Callable]:
    from hermes_cli.plugins import get_plugin_manager

    return list(get_plugin_manager()._middleware.get(kind, []))


def _run_execution_chain(
    kind: str,
    callbacks: List[Callable],
    terminal_call: Callable[[Any], Any],
    **kwargs: Any,
) -> Any:
    payload_key = "request" if "request" in kwargs else "args"

    class _DownstreamExecutionError(Exception):
        def __init__(self, original: BaseException) -> None:
            super().__init__(str(original))
            self.original = original

    def call_at(index: int, payload: Any) -> Any:
        if index >= len(callbacks):
            return terminal_call(payload)

        callback = callbacks[index]
        next_called = False
        next_succeeded = False
        next_result: Any = None

        def next_call(next_payload: Any = None) -> Any:
            nonlocal next_called, next_succeeded, next_result
            # ``next_call`` is single-use per middleware frame. Calling it more
            # than once would re-run the downstream provider/tool, so a second
            # invocation is a contract violation rather than a retry. Surface it
            # instead of silently executing the terminal call twice.
            if next_called:
                raise RuntimeError(
                    f"Middleware '{kind}' callback "
                    f"{getattr(callback, '__name__', repr(callback))} called "
                    "next_call() more than once; downstream execution is single-use"
                )
            next_called = True
            try:
                next_result = call_at(index + 1, payload if next_payload is None else next_payload)
                next_succeeded = True
                return next_result
            except Exception as exc:
                raise _DownstreamExecutionError(exc) from exc

        call_kwargs = middleware_payload(**kwargs)
        call_kwargs[payload_key] = payload
        call_kwargs["next_call"] = next_call
        try:
            return callback(**call_kwargs)
        except _DownstreamExecutionError as exc:
            raise exc.original
        except Exception as exc:
            logger.warning(
                "Middleware '%s' callback %s raised: %s",
                kind,
                getattr(callback, "__name__", repr(callback)),
                exc,
            )
            if next_succeeded:
                return next_result
            if next_called:
                raise
            return call_at(index + 1, payload)

    return call_at(0, kwargs[payload_key])


async def _run_execution_chain_async(
    kind: str,
    callbacks: List[Callable],
    terminal_call: Callable[[Any], Any],
    **kwargs: Any,
) -> Any:
    """Coroutine-native equivalent of :func:`_run_execution_chain`."""
    payload_key = "request" if "request" in kwargs else "args"

    class _DownstreamExecutionError(Exception):
        def __init__(self, original: BaseException) -> None:
            super().__init__(str(original))
            self.original = original

    async def call_at(index: int, payload: Any) -> Any:
        if index >= len(callbacks):
            return await terminal_call(payload)

        callback = callbacks[index]
        next_called = False
        next_succeeded = False
        next_result: Any = None

        async def next_call(next_payload: Any = None) -> Any:
            nonlocal next_called, next_succeeded, next_result
            if next_called:
                raise RuntimeError(
                    f"Middleware '{kind}' callback "
                    f"{getattr(callback, '__name__', repr(callback))} called "
                    "next_call() more than once; downstream execution is single-use"
                )
            next_called = True
            try:
                next_result = await call_at(
                    index + 1,
                    payload if next_payload is None else next_payload,
                )
                next_succeeded = True
                return next_result
            except Exception as exc:
                raise _DownstreamExecutionError(exc) from exc

        call_kwargs = middleware_payload(**kwargs)
        call_kwargs[payload_key] = payload
        call_kwargs["next_call"] = next_call
        try:
            result = callback(**call_kwargs)
            if not inspect.isawaitable(result):
                raise AsyncMiddlewareCapabilityError(
                    "Async Hermes requires coroutine middleware callbacks; "
                    f"{getattr(callback, '__name__', repr(callback))} is synchronous"
                )
            return await result
        except _DownstreamExecutionError as exc:
            raise exc.original
        except AsyncMiddlewareCapabilityError:
            raise
        except Exception as exc:
            logger.warning(
                "Middleware '%s' callback %s raised: %s",
                kind,
                getattr(callback, "__name__", repr(callback)),
                exc,
            )
            if next_succeeded:
                return next_result
            if next_called:
                raise
            return await call_at(index + 1, payload)

    return await call_at(0, kwargs[payload_key])


def _trace_entry(result: Dict[str, Any]) -> Dict[str, Any]:
    entry: Dict[str, Any] = {}
    for key in ("source", "reason", "name"):
        value = result.get(key)
        if isinstance(value, str) and value:
            entry[key] = value
    if not entry:
        entry["source"] = "plugin"
    return entry
