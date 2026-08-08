import asyncio
import os

import pytest

from gateway.session_context import (
    _UNSET,
    _VAR_MAP,
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
    yield
    for var in _VAR_MAP.values():
        var.set(_UNSET)


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

