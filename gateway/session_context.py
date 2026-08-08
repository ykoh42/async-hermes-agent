"""Task-local session metadata for concurrent async embedders.

The retained agent, tools, and subprocess bridge use ContextVars instead of
process-global environment mutation, so separate AIAgent instances can run
concurrently without leaking routing or session identity.
"""

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Iterator

_UNSET: Any = object()

_SESSION_PLATFORM: ContextVar = ContextVar("HERMES_SESSION_PLATFORM", default=_UNSET)
_SESSION_SOURCE: ContextVar = ContextVar("HERMES_SESSION_SOURCE", default=_UNSET)
_SESSION_CHAT_ID: ContextVar = ContextVar("HERMES_SESSION_CHAT_ID", default=_UNSET)
_SESSION_CHAT_TYPE: ContextVar = ContextVar("HERMES_SESSION_CHAT_TYPE", default=_UNSET)
_SESSION_CHAT_NAME: ContextVar = ContextVar("HERMES_SESSION_CHAT_NAME", default=_UNSET)
_SESSION_THREAD_ID: ContextVar = ContextVar("HERMES_SESSION_THREAD_ID", default=_UNSET)
_SESSION_USER_ID: ContextVar = ContextVar("HERMES_SESSION_USER_ID", default=_UNSET)
_SESSION_USER_NAME: ContextVar = ContextVar("HERMES_SESSION_USER_NAME", default=_UNSET)
_SESSION_KEY: ContextVar = ContextVar("HERMES_SESSION_KEY", default=_UNSET)
_SESSION_ID: ContextVar = ContextVar("HERMES_SESSION_ID", default=_UNSET)
_SESSION_PROFILE: ContextVar = ContextVar("HERMES_SESSION_PROFILE", default=_UNSET)

_VAR_MAP = {
    "HERMES_SESSION_PLATFORM": _SESSION_PLATFORM,
    "HERMES_SESSION_SOURCE": _SESSION_SOURCE,
    "HERMES_SESSION_CHAT_ID": _SESSION_CHAT_ID,
    "HERMES_SESSION_CHAT_TYPE": _SESSION_CHAT_TYPE,
    "HERMES_SESSION_CHAT_NAME": _SESSION_CHAT_NAME,
    "HERMES_SESSION_THREAD_ID": _SESSION_THREAD_ID,
    "HERMES_SESSION_USER_ID": _SESSION_USER_ID,
    "HERMES_SESSION_USER_NAME": _SESSION_USER_NAME,
    "HERMES_SESSION_KEY": _SESSION_KEY,
    "HERMES_SESSION_ID": _SESSION_ID,
    "HERMES_SESSION_PROFILE": _SESSION_PROFILE,
}
_SESSION_VARS = tuple(_VAR_MAP.values())
_session_context_engaged = False


def session_context_engaged() -> bool:
    """Return whether this process has bound task-local session metadata."""
    return _session_context_engaged


def set_current_session_id(session_id: str) -> None:
    """Bind the current task's durable session ID."""
    global _session_context_engaged
    _session_context_engaged = True
    _SESSION_ID.set(session_id)


@contextmanager
def scoped_current_session_id(session_id: str | None = None) -> Iterator[None]:
    """Bind a session ID for one scope and restore the previous value."""
    global _session_context_engaged
    _session_context_engaged = True
    token = _SESSION_ID.set(session_id) if session_id is not None else None
    try:
        yield
    finally:
        if token is not None:
            _SESSION_ID.reset(token)


def set_session_vars(
    platform: str = "",
    source: str = "",
    chat_id: str = "",
    chat_type: str = "",
    chat_name: str = "",
    thread_id: str = "",
    user_id: str = "",
    user_name: str = "",
    session_key: str = "",
    session_id: str = "",
    profile: str = "",
    cwd: str = "",
) -> list[Token]:
    """Bind request-scoped metadata and return tokens for restoration."""
    global _session_context_engaged
    _session_context_engaged = True
    values = (
        platform,
        source,
        chat_id,
        chat_type,
        chat_name,
        thread_id,
        user_id,
        user_name,
        session_key,
        session_id,
        profile,
    )
    tokens = [var.set(value) for var, value in zip(_SESSION_VARS, values)]
    try:
        from agent.runtime_cwd import set_session_cwd

        set_session_cwd(cwd)
    except Exception:
        pass
    return tokens


def clear_session_vars(tokens: list[Token]) -> None:
    """Restore metadata previously bound by set_session_vars."""
    for variable, token in zip(_SESSION_VARS, tokens):
        variable.reset(token)
    try:
        from agent.runtime_cwd import clear_session_cwd

        clear_session_cwd()
    except Exception:
        pass


def reset_session_vars() -> None:
    """Reset all session metadata in the current task."""
    for variable in _SESSION_VARS:
        variable.set(_UNSET)
    try:
        from agent.runtime_cwd import clear_session_cwd

        clear_session_cwd()
    except Exception:
        pass


def get_session_env(name: str, default: str = "") -> str:
    """Read task-local session metadata by its historical environment name."""
    variable = _VAR_MAP.get(name)
    if variable is None:
        return default
    value = variable.get()
    return default if value is _UNSET else str(value)
