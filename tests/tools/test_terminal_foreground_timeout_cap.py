"""Foreground timeouts are bounded without blocking the event loop."""
import json

import pytest

from tools.terminal_tool import FOREGROUND_MAX_TIMEOUT, TERMINAL_SCHEMA, terminal_tool


@pytest.mark.asyncio
async def test_foreground_timeout_above_cap_is_rejected():
    result = json.loads(await terminal_tool("printf no", timeout=FOREGROUND_MAX_TIMEOUT + 1))
    assert str(FOREGROUND_MAX_TIMEOUT) in result["error"]
    assert "background=true" in result["error"]


@pytest.mark.asyncio
async def test_timeout_at_cap_is_allowed(tmp_path):
    result = json.loads(
        await terminal_tool("printf ok", timeout=FOREGROUND_MAX_TIMEOUT, workdir=str(tmp_path))
    )
    assert result["output"] == "ok"


def test_schema_mentions_foreground_cap():
    description = TERMINAL_SCHEMA["parameters"]["properties"]["timeout"]["description"]
    assert str(FOREGROUND_MAX_TIMEOUT) in description
    assert "background=true" in description
