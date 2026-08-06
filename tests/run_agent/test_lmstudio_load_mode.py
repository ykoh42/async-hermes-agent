from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from hermes_cli.models import LMStudioLoadResult
from run_agent import AIAgent


def _agent(load_mode="explicit"):
    return SimpleNamespace(
        provider="lmstudio",
        model="test/model",
        base_url="http://127.0.0.1:1234/v1",
        api_key="",
        lmstudio_load_mode=load_mode,
        _config_context_length=None,
        context_compressor=None,
        api_mode="chat_completions",
    )


@pytest.mark.asyncio
async def test_lmstudio_jit_load_mode_skips_explicit_preload(monkeypatch):
    ensure = AsyncMock(return_value=LMStudioLoadResult(64_000))
    monkeypatch.setattr("hermes_cli.models.ensure_lmstudio_model_loaded", ensure)

    result = await AIAgent._ensure_lmstudio_runtime_loaded(
        cast(Any, _agent("jit"))
    )

    assert result is None
    ensure.assert_not_awaited()


@pytest.mark.asyncio
async def test_lmstudio_explicit_mode_awaits_native_preload(monkeypatch):
    expected = LMStudioLoadResult(64_000, load_attempted=True)
    ensure = AsyncMock(return_value=expected)
    monkeypatch.setattr("hermes_cli.models.ensure_lmstudio_model_loaded", ensure)

    result = await AIAgent._ensure_lmstudio_runtime_loaded(
        cast(Any, _agent("explicit")),
        80_000,
    )

    assert result == expected
    ensure.assert_awaited_once_with(
        "test/model",
        "http://127.0.0.1:1234/v1",
        "",
        80_000,
        return_load_result=True,
    )






def test_explicit_budget_below_loaded_runtime_limits_effective_context():
    result = AIAgent._effective_lmstudio_context_length(
        80_000,
        120_000,
    )

    assert result == 80_000
