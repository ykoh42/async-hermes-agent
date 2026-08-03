from types import SimpleNamespace
from typing import Any, cast

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


def test_lmstudio_jit_load_mode_skips_explicit_preload(monkeypatch):
    result = AIAgent._ensure_lmstudio_runtime_loaded(cast(Any, _agent("jit")))

    assert result is None






def test_explicit_budget_below_loaded_runtime_limits_effective_context():
    result = AIAgent._effective_lmstudio_context_length(
        80_000,
        120_000,
    )

    assert result == 80_000

