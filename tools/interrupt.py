"""Task-local interrupt signaling for the native async tool runtime.

Each ``AIAgent`` owns one :class:`asyncio.Event`. ``run_conversation`` binds
that event to the current context, and tool tasks inherit the same reference
through normal ``contextvars`` propagation. This isolates concurrent agents
that share one event-loop thread without a thread-local compatibility layer.
"""

from __future__ import annotations

import asyncio
import contextvars


_current_interrupt_event: contextvars.ContextVar[asyncio.Event | None] = (
    contextvars.ContextVar("hermes_interrupt_event", default=None)
)


def _bind_interrupt_event(
    event: asyncio.Event,
) -> contextvars.Token[asyncio.Event | None]:
    """Bind an agent-owned interrupt event to the current async context."""
    return _current_interrupt_event.set(event)


def _reset_interrupt_event(
    token: contextvars.Token[asyncio.Event | None],
) -> None:
    """Restore the interrupt binding that preceded a conversation turn."""
    _current_interrupt_event.reset(token)


def set_interrupt(
    active: bool,
    thread_id: asyncio.Event | None = None,
) -> None:
    """Set or clear an interrupt event without blocking the event loop.

    ``thread_id`` retains the upstream public parameter name. In the native
    async runtime it carries the agent-owned event supplied by cross-task
    controls such as
    :meth:`AIAgent.interrupt`. Tool code normally omits it and operates on the
    event bound to its inherited task context.
    """
    target = thread_id or _current_interrupt_event.get()
    if target is None:
        target = asyncio.Event()
        _current_interrupt_event.set(target)
    if active:
        target.set()
    else:
        target.clear()


def clear_current_thread_interrupt() -> None:
    """Clear the interrupt bound to the current async execution context."""
    set_interrupt(False)


def is_interrupted() -> bool:
    """Return whether the current agent task has been interrupted."""
    event = _current_interrupt_event.get()
    return bool(event and event.is_set())
