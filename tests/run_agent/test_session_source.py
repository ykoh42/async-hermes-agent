import asyncio
import os
from unittest.mock import patch

import pytest

from gateway.session_context import (
    _UNSET,
    _SESSION_ASYNC_DELIVERY,
    _VAR_MAP,
    async_delivery_supported,
    clear_session_vars,
    get_session_env,
    set_current_session_id,
    set_session_vars,
)
from run_agent import _session_source_for_agent


@pytest.fixture(autouse=True)
def _reset_contextvars():
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    _SESSION_ASYNC_DELIVERY.set(_UNSET)
    yield
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    _SESSION_ASYNC_DELIVERY.set(_UNSET)


def test_session_source_context_overrides_platform(monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)

    tokens = set_session_vars(source="tool")
    try:
        assert _session_source_for_agent("web") == "tool"
    finally:
        clear_session_vars(tokens)


def test_session_source_falls_back_to_platform(monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)

    assert _session_source_for_agent("web") == "web"


def test_session_source_ignores_process_global_legacy_value(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "legacy-global")

    assert _session_source_for_agent("web") == "web"


def test_session_source_restores_upstream_env_fallback_on_context_error(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "legacy-global")

    with patch(
        "gateway.session_context.get_session_env",
        side_effect=RuntimeError("context unavailable"),
    ):
        assert _session_source_for_agent("web") == "legacy-global"


def test_set_session_vars_preserves_upstream_positional_order(tmp_path):
    tokens = set_session_vars(
        "platform",
        "source",
        "chat-id",
        "chat-type",
        "chat-name",
        "thread-id",
        "user-id",
        "user-name",
        "session-key",
        "session-id",
        "message-id",
        "profile-name",
        str(tmp_path),
        False,
        "ui-session-id",
        "1",
    )
    try:
        assert get_session_env("HERMES_SESSION_MESSAGE_ID") == "message-id"
        assert get_session_env("HERMES_SESSION_PROFILE") == "profile-name"
        assert get_session_env("HERMES_UI_SESSION_ID") == "ui-session-id"
        assert get_session_env("HERMES_CRON_SESSION") == "1"
        assert async_delivery_supported() is False
    finally:
        clear_session_vars(tokens)


def test_clear_session_vars_preserves_upstream_non_nestable_semantics():
    outer = set_session_vars(session_key="outer", async_delivery=False)
    try:
        inner = set_session_vars(session_key="inner", async_delivery=True)
        try:
            assert get_session_env("HERMES_SESSION_KEY") == "inner"
            assert async_delivery_supported() is True
        finally:
            clear_session_vars(inner)

        assert get_session_env("HERMES_SESSION_KEY") == ""
        assert async_delivery_supported() is True
    finally:
        clear_session_vars(outer)

    assert get_session_env("HERMES_SESSION_KEY") == ""
    assert async_delivery_supported() is True


@pytest.mark.asyncio
async def test_concurrent_tasks_keep_distinct_session_ids(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_ID", "process-global")
    ready = asyncio.Event()
    bound = 0
    lock = asyncio.Lock()

    async def bind_and_read(session_id: str) -> str:
        nonlocal bound
        set_current_session_id(session_id)
        async with lock:
            bound += 1
            if bound == 2:
                ready.set()
        await ready.wait()
        await asyncio.sleep(0)
        return get_session_env("HERMES_SESSION_ID")

    assert await asyncio.gather(bind_and_read("one"), bind_and_read("two")) == [
        "one",
        "two",
    ]
    assert os.environ["HERMES_SESSION_ID"] == "process-global"
