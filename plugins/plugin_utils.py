"""Shared native-async concurrency helpers for plugin authors.

The most common plugin footgun is a lazy process-wide singleton whose client
factory performs network or file setup synchronously.  These helpers preserve
the upstream lazy, first-value-wins, reset, and retry-after-failure semantics
while making the factory boundary awaitable and serializing it without worker
threads.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from collections.abc import Awaitable, Callable
from typing import Generic, Optional, TypeVar

__all__ = ["lazy_singleton", "SingletonSlot"]

T = TypeVar("T")


def _require_async_factory(factory: Callable[[], Awaitable[T]]) -> None:
    if not (
        inspect.iscoroutinefunction(factory)
        or inspect.iscoroutinefunction(getattr(factory, "__call__", None))
    ):
        raise TypeError("plugin singleton factory must be async")


def lazy_singleton(
    factory: Callable[[], Awaitable[T]],
) -> Callable[[], Awaitable[T]]:
    """Wrap a zero-argument async factory in a lazy singleton accessor.

    Concurrent first callers await one serialized factory invocation and all
    receive the same instance.  If the factory raises or is cancelled, no
    value is cached and the next call retries.  The attached async ``reset``
    method waits for an in-flight build before dropping the cached value.

    Example::

        @lazy_singleton
        async def get_client():
            config = await load_config_readonly()
            return AsyncClient(config)

        client = await get_client()
        await get_client.reset()
    """
    _require_async_factory(factory)
    lock = asyncio.Lock()
    box: list[T] = []

    @functools.wraps(factory)
    async def accessor() -> T:
        if box:
            return box[0]
        async with lock:
            if box:
                return box[0]
            instance = await factory()
            box.append(instance)
            return instance

    async def reset() -> None:
        async with lock:
            box.clear()

    accessor.reset = reset  # type: ignore[attr-defined]
    return accessor


class SingletonSlot(Generic[T]):
    """Lazy async slot for accessors whose factory depends on an argument.

    The first successfully built instance is cached and later factories are
    ignored, matching the upstream first-value-wins contract.
    """

    __slots__ = ("_lock", "_value", "_set")

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._value: Optional[T] = None
        self._set = False

    async def get(self, factory: Callable[[], Awaitable[T]]) -> T:
        if self._set:
            return self._value  # type: ignore[return-value]
        _require_async_factory(factory)
        async with self._lock:
            if self._set:
                return self._value  # type: ignore[return-value]
            value = await factory()
            self._value = value
            self._set = True
            return value

    def peek(self) -> Optional[T]:
        """Return the cached instance without building it (None if unset)."""
        return self._value if self._set else None

    async def reset(self) -> None:
        """Drop the cached instance so the next awaited ``get()`` rebuilds it."""
        async with self._lock:
            self._value = None
            self._set = False
