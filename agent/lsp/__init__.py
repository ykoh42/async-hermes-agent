"""Native-async Language Server Protocol integration for Hermes Agent."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import threading
import weakref
from dataclasses import dataclass, field
from typing import Optional

import aiofiles.os

from agent.lsp.manager import LSPService


logger = logging.getLogger("agent.lsp")

_LSPScopeKey = tuple[asyncio.AbstractEventLoop, str]
_lsp_scope_context: contextvars.ContextVar[tuple[str, str] | None] = (
    contextvars.ContextVar("lsp_profile_scope", default=None)
)
_lsp_scope_aliases: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, str]
] = weakref.WeakKeyDictionary()


@dataclass
class _LSPProfileState:
    """Resources shared by consumers on one loop and Hermes profile."""

    service: Optional[LSPService] = None
    service_lock_ref: weakref.ReferenceType[asyncio.Lock] | None = None
    shutdown_task: asyncio.Task[None] | None = None
    lifecycle_consumers: weakref.WeakSet[object] = field(
        default_factory=weakref.WeakSet
    )
    lifecycle_lock_ref: weakref.ReferenceType[asyncio.Lock] | None = None


_lsp_loop_states: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, _LSPProfileState]
] = weakref.WeakKeyDictionary()
_lsp_owner_scopes: weakref.WeakKeyDictionary[
    object,
    tuple[weakref.ReferenceType[asyncio.AbstractEventLoop], str],
] = weakref.WeakKeyDictionary()
_lsp_state_guard = threading.RLock()


def _lexical_lsp_profile_identity() -> str:
    """Return an environment-only profile marker without filesystem I/O."""
    from hermes_constants import get_hermes_home

    return os.path.normcase(os.fspath(get_hermes_home()))


def _prune_closed_lsp_loops() -> None:
    """Drop state that cannot own live async resources after loop closure."""
    with _lsp_state_guard:
        known_loops = set(_lsp_loop_states) | set(_lsp_scope_aliases)
        for loop in known_loops:
            if loop.is_closed():
                _lsp_loop_states.pop(loop, None)
                _lsp_scope_aliases.pop(loop, None)
        for owner, (loop_ref, _profile) in tuple(_lsp_owner_scopes.items()):
            loop = loop_ref()
            if loop is None or loop.is_closed():
                _lsp_owner_scopes.pop(owner, None)


async def _activate_lsp_scope() -> _LSPScopeKey:
    """Resolve the active loop and canonical Hermes profile."""
    _prune_closed_lsp_loops()
    loop = asyncio.get_running_loop()
    lexical = _lexical_lsp_profile_identity()
    active = _lsp_scope_context.get()
    if active is not None and active[0] == lexical:
        canonical = active[1]
    else:
        with _lsp_state_guard:
            aliases = _lsp_scope_aliases.get(loop)
            canonical = aliases.get(lexical) if aliases is not None else None
        if canonical is None:
            expanduser = aiofiles.os.wrap(os.path.expanduser)
            expanded = str(await expanduser(lexical))
            is_absolute = (
                expanded.startswith(("/", "\\\\"))
                or (
                    len(expanded) >= 3
                    and expanded[1] == ":"
                    and expanded[2] in "/\\"
                )
            )
            if not is_absolute:
                expanded = str(await aiofiles.os.getcwd()) + os.sep + expanded
            realpath = aiofiles.os.wrap(os.path.realpath)
            canonical = os.path.normcase(str(await realpath(expanded)))
        with _lsp_state_guard:
            _lsp_scope_aliases.setdefault(loop, {})[lexical] = canonical
    _lsp_scope_context.set((lexical, canonical))
    return loop, canonical


def _state_for_scope(scope: _LSPScopeKey) -> _LSPProfileState:
    loop, profile = scope
    with _lsp_state_guard:
        states = _lsp_loop_states.setdefault(loop, {})
        return states.setdefault(profile, _LSPProfileState())


def _existing_state_for_scope(
    scope: _LSPScopeKey,
) -> _LSPProfileState | None:
    loop, profile = scope
    with _lsp_state_guard:
        states = _lsp_loop_states.get(loop)
        return states.get(profile) if states is not None else None


def _state_lock(
    state: _LSPProfileState,
    attribute: str,
) -> asyncio.Lock:
    """Return a live lock without retaining its event loop after use."""
    with _lsp_state_guard:
        lock_ref = getattr(state, attribute)
        lock = lock_ref() if lock_ref is not None else None
        if lock is None:
            lock = asyncio.Lock()
            setattr(state, attribute, weakref.ref(lock))
        return lock


def _discard_scope_if_idle(
    scope: _LSPScopeKey,
    state: _LSPProfileState,
) -> None:
    lifecycle_lock = (
        state.lifecycle_lock_ref()
        if state.lifecycle_lock_ref is not None
        else None
    )
    if (
        state.service is not None
        or state.shutdown_task is not None
        or state.lifecycle_consumers
        or (lifecycle_lock is not None and lifecycle_lock.locked())
    ):
        return
    loop, profile = scope
    with _lsp_state_guard:
        states = _lsp_loop_states.get(loop)
        if states is None or states.get(profile) is not state:
            return
        states.pop(profile, None)
        if not states:
            _lsp_loop_states.pop(loop, None)


async def get_service() -> Optional[LSPService]:
    """Return the lazily created service for the active loop and profile."""
    scope = await _activate_lsp_scope()
    while True:
        state = _state_for_scope(scope)
        async with _state_lock(state, "service_lock_ref"):
            shutdown_task = state.shutdown_task
            if shutdown_task is None:
                if state.service is None:
                    try:
                        state.service = await LSPService.create_from_config()
                    except BaseException:
                        state.service = None
                        raise
                return (
                    state.service
                    if state.service is not None and state.service.is_active()
                    else None
                )
        await _await_service_task(shutdown_task)


async def shutdown_service() -> None:
    """Tear down the active profile's service and all owned subprocesses."""
    scope = await _activate_lsp_scope()
    state = _existing_state_for_scope(scope)
    if state is None:
        return
    await _shutdown_scope(scope, state)


async def _shutdown_scope(
    scope: _LSPScopeKey,
    state: _LSPProfileState,
) -> None:
    async with _state_lock(state, "service_lock_ref"):
        shutdown_task = state.shutdown_task
        if shutdown_task is None:
            service = state.service
            state.service = None
            if service is None:
                return
            shutdown_task = asyncio.create_task(
                _shutdown_service_owned(scope, state, service),
                name="hermes-lsp-profile-shutdown",
            )
            state.shutdown_task = shutdown_task
    await _await_service_task(shutdown_task)


async def _shutdown_service_owned(
    scope: _LSPScopeKey,
    state: _LSPProfileState,
    service: LSPService,
) -> None:
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    token = set_hermes_home_override(scope[1])
    try:
        await service.shutdown()
    finally:
        reset_hermes_home_override(token)
        state.shutdown_task = None
        _discard_scope_if_idle(scope, state)


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
    """Keep this profile's shared LSP service alive for ``owner``."""
    scope = await _activate_lsp_scope()
    state = _state_for_scope(scope)
    async with _state_lock(state, "lifecycle_lock_ref"):
        state.lifecycle_consumers.add(owner)
        with _lsp_state_guard:
            _lsp_owner_scopes[owner] = (weakref.ref(scope[0]), scope[1])


async def _release_lsp_lifecycle(owner: object) -> None:
    """Close only the released owner's profile after its final lease."""
    current_loop = asyncio.get_running_loop()
    with _lsp_state_guard:
        retained_scope = _lsp_owner_scopes.get(owner)
    if retained_scope is None:
        return
    retained_loop = retained_scope[0]()
    if retained_loop is not current_loop:
        raise RuntimeError(
            "The LSP lifecycle lease belongs to another event loop; "
            "release it on its owning loop"
        )
    scope = (current_loop, retained_scope[1])
    state = _state_for_scope(scope)
    async with _state_lock(state, "lifecycle_lock_ref"):
        if owner not in state.lifecycle_consumers:
            with _lsp_state_guard:
                _lsp_owner_scopes.pop(owner, None)
            return
        state.lifecycle_consumers.remove(owner)
        with _lsp_state_guard:
            _lsp_owner_scopes.pop(owner, None)
        final_consumer = not state.lifecycle_consumers
        if final_consumer:
            # Keep retain/release serialized until owned subprocess cleanup is
            # durable. A new same-profile consumer must not race the final
            # consumer and inherit a service already being shut down.
            await _shutdown_scope(scope, state)


__all__ = ["get_service", "shutdown_service", "LSPService"]
