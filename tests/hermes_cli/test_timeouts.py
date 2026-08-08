from __future__ import annotations

import textwrap

import pytest

from hermes_cli.timeouts import (
    get_provider_request_timeout,
    get_provider_stale_timeout,
)


def _write_config(tmp_path, body: str) -> None:
    (tmp_path / "config.yaml").write_text(textwrap.dedent(body), encoding="utf-8")


@pytest.mark.asyncio
async def test_provider_timeout_helpers_preserve_model_and_provider_priority(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          openrouter:
            request_timeout_seconds: 77
            stale_timeout_seconds: 88
            models:
              openai/gpt-4o-mini:
                timeout_seconds: 42
                stale_timeout_seconds: 43
        """,
    )

    assert await get_provider_request_timeout(
        "openrouter", "openai/gpt-4o-mini"
    ) == 42.0
    assert await get_provider_stale_timeout(
        "openrouter", "openai/gpt-4o-mini"
    ) == 43.0
    assert await get_provider_request_timeout("openrouter", "other") == 77.0
    assert await get_provider_stale_timeout("openrouter", "other") == 88.0


@pytest.mark.asyncio
async def test_provider_timeout_helpers_fail_open(monkeypatch):
    async def fail_load():
        raise OSError("unreadable")

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", fail_load)

    assert await get_provider_request_timeout("openrouter") is None
    assert await get_provider_stale_timeout("openrouter") is None
    assert await get_provider_request_timeout("") is None
    assert await get_provider_stale_timeout("") is None










def test_anthropic_adapter_honors_timeout_kwarg():
    """build_anthropic_client(timeout=X) overrides the 900s default read timeout."""
    pytest = __import__("pytest")
    anthropic = pytest.importorskip("anthropic")  # skip if optional SDK missing
    from agent.anthropic_adapter import build_anthropic_client

    c_default = build_anthropic_client("sk-ant-dummy", None)
    c_custom = build_anthropic_client("sk-ant-dummy", None, timeout=45.0)
    c_invalid = build_anthropic_client("sk-ant-dummy", None, timeout=-1)

    # Default stays at 900s; custom overrides; invalid falls back to default
    assert c_default.timeout.read == 900.0
    assert c_custom.timeout.read == 45.0
    assert c_invalid.timeout.read == 900.0
    # Connect timeout always stays at 10s regardless
    assert c_default.timeout.connect == 10.0
    assert c_custom.timeout.connect == 10.0


@pytest.mark.asyncio
async def test_resolved_api_call_timeout_priority(monkeypatch, tmp_path):
    """AIAgent._resolved_api_call_timeout() honors config > env > default priority."""
    # Isolate HERMES_HOME
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")

    # Case A: config wins over env var
    _write_config(tmp_path, """\
        providers:
          openrouter:
            request_timeout_seconds: 77
            models:
              openai/gpt-4o-mini:
                timeout_seconds: 42
        """)
    monkeypatch.setenv("HERMES_API_TIMEOUT", "999")

    from run_agent import AIAgent
    agent = AIAgent(
        model="openai/gpt-4o-mini",
        provider="openrouter",
        api_key="sk-dummy",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    await agent._ensure_provider_runtime()
    # Per-model override wins
    assert agent._resolved_api_call_timeout() == 42.0

    # Provider-level (different model, no per-model override). Runtime timeout
    # policy is resolved when the route is constructed; direct attribute
    # mutation intentionally does not rebuild provider state.
    provider_agent = AIAgent(
        model="some/other-model",
        provider="openrouter",
        api_key="sk-dummy",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    await provider_agent._ensure_provider_runtime()
    assert provider_agent._resolved_api_call_timeout() == 77.0

    # Case B: no config → env wins
    _write_config(tmp_path, "")
    agent2 = AIAgent(
        model="some/model",
        provider="openrouter",
        api_key="sk-dummy",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    await agent2._ensure_provider_runtime()
    assert agent2._resolved_api_call_timeout() == 999.0

    # Case C: no config, no env → 1800.0 default
    monkeypatch.delenv("HERMES_API_TIMEOUT", raising=False)
    assert agent2._resolved_api_call_timeout() == 1800.0
