"""Native-async per-consumer observers for streaming model output.

The upstream observer contract is intentionally fire-and-forget.  This fork
keeps the same public helper names and queue/drop-oldest semantics, but uses
loop-owned asyncio tasks instead of a worker thread so a synchronous fallback
cannot hide blocking I/O in the retained runtime.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from hermes_cli.middleware import OBSERVER_SCHEMA_VERSION

logger = logging.getLogger(__name__)

_QUEUE_SIZE = 1024
_STOP = object()


@dataclass
class _ConsumerDispatcher:
    hook_name: str
    callback: Callable[..., Any]
    loop: asyncio.AbstractEventLoop
    events: asyncio.Queue[dict[str, Any] | object]
    task: asyncio.Task[None] | None = None


_dispatcher_lock = threading.Lock()
_dispatchers: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[tuple[str, int], _ConsumerDispatcher]
] = weakref.WeakKeyDictionary()


def _callback_name(callback: Callable[..., Any]) -> str:
    return getattr(callback, "__name__", repr(callback))


async def _worker(dispatcher: _ConsumerDispatcher) -> None:
    while True:
        item = await dispatcher.events.get()
        try:
            if item is _STOP:
                return
            payload = dict(item)
            payload.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
            try:
                result = dispatcher.callback(**payload)
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Hook '%s' callback %s raised: %s",
                    dispatcher.hook_name,
                    _callback_name(dispatcher.callback),
                    exc,
                )
        finally:
            dispatcher.events.task_done()


def _registered_callbacks(hook_name: str) -> tuple[Callable[..., Any], ...]:
    try:
        from hermes_cli import plugins

        return plugins.iter_hook_callbacks(hook_name)
    except Exception:
        logger.debug(
            "plugin stream hook callback lookup failed: %s",
            hook_name,
            exc_info=True,
        )
        return ()


def _remove_dispatcher(dispatcher: _ConsumerDispatcher) -> None:
    with _dispatcher_lock:
        per_loop = _dispatchers.get(dispatcher.loop)
        if per_loop is not None:
            per_loop.pop((dispatcher.hook_name, id(dispatcher.callback)), None)
            if not per_loop:
                _dispatchers.pop(dispatcher.loop, None)


def _dispatchers_for(hook_name: str) -> list[_ConsumerDispatcher]:
    callbacks = _registered_callbacks(hook_name)
    if not callbacks:
        return []
    loop = asyncio.get_running_loop()
    callback_ids = {id(callback) for callback in callbacks}
    stale: list[_ConsumerDispatcher] = []
    ready: list[_ConsumerDispatcher] = []
    with _dispatcher_lock:
        per_loop = _dispatchers.setdefault(loop, {})
        for key, dispatcher in list(per_loop.items()):
            key_hook_name, callback_id = key
            if key_hook_name == hook_name and callback_id not in callback_ids:
                stale.append(per_loop.pop(key))
        for callback in callbacks:
            key = (hook_name, id(callback))
            dispatcher = per_loop.get(key)
            if dispatcher is None or dispatcher.task is None or dispatcher.task.done():
                events: asyncio.Queue[dict[str, Any] | object] = asyncio.Queue(
                    maxsize=max(1, int(_QUEUE_SIZE))
                )
                dispatcher = _ConsumerDispatcher(
                    hook_name=hook_name,
                    callback=callback,
                    loop=loop,
                    events=events,
                )
                dispatcher.task = asyncio.create_task(
                    _worker(dispatcher),
                    name=f"plugin-stream-hook:{hook_name}",
                )
                dispatcher.task.add_done_callback(
                    lambda _task, d=dispatcher: _remove_dispatcher(d)
                )
                per_loop[key] = dispatcher
            ready.append(dispatcher)
    for dispatcher in stale:
        if dispatcher.task is not None:
            dispatcher.task.cancel()
    return ready


def enqueue_plugin_stream_hook(hook_name: str, **payload: Any) -> bool:
    """Queue an observer hook for each consumer without running it inline."""
    queued = False
    item = dict(payload)
    for dispatcher in _dispatchers_for(hook_name):
        try:
            dispatcher.events.put_nowait(item)
            queued = True
            continue
        except asyncio.QueueFull:
            try:
                dispatcher.events.get_nowait()
                dispatcher.events.task_done()
            except asyncio.QueueEmpty:
                pass
        try:
            dispatcher.events.put_nowait(item)
            queued = True
        except asyncio.QueueFull:
            logger.debug(
                "plugin stream hook queue full after drop-oldest: %s callback=%s",
                hook_name,
                _callback_name(dispatcher.callback),
            )
    return queued


def has_stream_observer_hooks() -> bool:
    return any(
        _registered_callbacks(name)
        for name in ("on_stream_start", "on_stream_delta", "on_stream_end")
    )


def has_reasoning_stream_observer_hooks() -> bool:
    return stream_reasoning_deltas_enabled() and bool(
        _registered_callbacks("on_stream_delta")
    )


def stream_reasoning_deltas_enabled() -> bool:
    """Return True only when the user opted plugins into reasoning deltas."""
    try:
        from hermes_cli import config as config_mod

        config = config_mod.load_config()
        return bool(
            config_mod.cfg_get(
                config,
                "plugins",
                "stream_reasoning_deltas",
                default=False,
            )
        )
    except Exception:
        logger.debug("failed to read plugins.stream_reasoning_deltas", exc_info=True)
        return False


def shutdown_plugin_stream_hook_dispatcher(timeout: float = 1.0) -> None:
    """Cancel loop-owned observer tasks; retained sync test/shutdown surface."""
    del timeout  # cancellation is non-blocking; callers can await loop progress
    with _dispatcher_lock:
        dispatchers = [
            dispatcher
            for per_loop in _dispatchers.values()
            for dispatcher in per_loop.values()
        ]
        _dispatchers.clear()
    for dispatcher in dispatchers:
        if dispatcher.task is not None:
            dispatcher.task.cancel()
