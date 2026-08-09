"""Native-async Language Server Protocol integration for Hermes Agent."""

from __future__ import annotations

import asyncio
import logging
import weakref
from typing import Optional

from agent.lsp.manager import LSPService


logger = logging.getLogger("agent.lsp")

_service: Optional[LSPService] = None
_service_lock: asyncio.Lock | None = None
_service_loop: asyncio.AbstractEventLoop | None = None
_service_shutdown_task: asyncio.Task[None] | None = None
_lifecycle_consumers: weakref.WeakSet[object] = weakref.WeakSet()
_lifecycle_lock = asyncio.Lock()


def _loop_lock() -> asyncio.Lock:
    global _service_lock, _service_loop
    loop = asyncio.get_running_loop()
    if _service_loop is not None and _service_loop is not loop:
        can_rebind = (
            not _lifecycle_consumers
            and _service_shutdown_task is None
            and (_service is None or not _service._has_owned_resources())
        )
        if not can_rebind:
            raise RuntimeError(
                "The process-wide LSP service belongs to another event loop; "
                "await shutdown_service() on its owning loop before reuse"
            )
        _service_lock = None
    _service_loop = loop
    if _service_lock is None:
        _service_lock = asyncio.Lock()
    return _service_lock


async def get_service() -> Optional[LSPService]:
    """Return the lazily created process-wide LSP service."""
    global _service
    while True:
        lock = _loop_lock()
        async with lock:
            shutdown_task = _service_shutdown_task
            if shutdown_task is None:
                if _service is None:
                    _service = await LSPService.create_from_config()
                return (
                    _service
                    if (_service is not None and _service.is_active())
                    else None
                )
        await _await_service_task(shutdown_task)


async def shutdown_service() -> None:
    """Tear down the process-wide LSP service and all owned subprocesses."""
    global _service, _service_shutdown_task
    # Preserve the fully detached state after an idempotent second shutdown.
    # There is no resource to synchronize here, and creating a fresh lock
    # would unnecessarily bind this state-only singleton to the caller's loop.
    if (
        _service is None
        and _service_shutdown_task is None
        and _service_lock is None
    ):
        return
    lock = _loop_lock()
    async with lock:
        shutdown_task = _service_shutdown_task
        if shutdown_task is None:
            service = _service
            _service = None
            if service is None:
                return
            shutdown_task = asyncio.create_task(
                _shutdown_service_owned(service),
                name="hermes-lsp-global-shutdown",
            )
            _service_shutdown_task = shutdown_task
    await _await_service_task(shutdown_task)


async def _shutdown_service_owned(service: LSPService) -> None:
    global _service_lock, _service_loop, _service_shutdown_task
    try:
        await service.shutdown()
    finally:
        _service_shutdown_task = None
        _service_lock = None
        _service_loop = None


async def _await_service_task(task: asyncio.Task[None]) -> None:
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


async def _retain_lsp_lifecycle(owner: object) -> None:
    """Keep the shared LSP service alive while an agent can use it."""
    async with _lifecycle_lock:
        _lifecycle_consumers.add(owner)


async def _release_lsp_lifecycle(owner: object) -> None:
    """Close shared LSP processes after the final agent releases them."""
    async with _lifecycle_lock:
        if owner not in _lifecycle_consumers:
            return
        _lifecycle_consumers.remove(owner)
        if not _lifecycle_consumers:
            await shutdown_service()


__all__ = ["get_service", "shutdown_service", "LSPService"]
