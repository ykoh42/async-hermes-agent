"""Profile-scoped settings for auxiliary provider routing."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agent import auxiliary_client as auxiliary
from agent.secret_scope import (
    UnscopedSecretError,
    is_multiplex_active,
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)


@pytest.fixture(autouse=True)
def _restore_multiplex_state():
    previous = is_multiplex_active()
    try:
        yield
    finally:
        set_multiplex_active(previous)


@pytest.mark.asyncio
async def test_concurrent_profiles_isolate_auxiliary_route_settings(monkeypatch):
    for key, value in {
        "HERMES_OPENROUTER_CACHE": "1",
        "HERMES_OPENROUTER_CACHE_TTL": "999",
        "OPENAI_API_KEY": "process-key",
        "OPENAI_BASE_URL": "https://process-openai.invalid",
        "NOUS_INFERENCE_BASE_URL": "https://process-nous.invalid",
    }.items():
        monkeypatch.setenv(key, value)
    set_multiplex_active(True)

    async def resolve(name: str, cache: str, ttl: str):
        token = set_secret_scope(
            {
                "HERMES_OPENROUTER_CACHE": cache,
                "HERMES_OPENROUTER_CACHE_TTL": ttl,
                "OPENAI_API_KEY": f"key-{name}",
                "OPENAI_BASE_URL": f"https://{name}-openai.example",
                "NOUS_INFERENCE_BASE_URL": f"https://{name}-nous.example",
            }
        )
        try:
            await asyncio.sleep(0)
            return (
                auxiliary.build_or_headers(or_config={}),
                auxiliary._resolve_custom_runtime(config={}),
                auxiliary._nous_base_url(),
            )
        finally:
            reset_secret_scope(token)

    result_a, result_b = await asyncio.gather(
        resolve("alpha", "1", "111"),
        resolve("beta", "0", "222"),
    )

    headers_a, custom_a, nous_a = result_a
    headers_b, custom_b, nous_b = result_b
    assert headers_a["X-OpenRouter-Cache"] == "true"
    assert headers_a["X-OpenRouter-Cache-TTL"] == "111"
    assert "X-OpenRouter-Cache" not in headers_b
    assert custom_a == (
        "https://alpha-openai.example",
        "key-alpha",
        None,
    )
    assert custom_b == (
        "https://beta-openai.example",
        "key-beta",
        None,
    )
    assert nous_a == "https://alpha-nous.example"
    assert nous_b == "https://beta-nous.example"


def test_single_profile_environment_precedence_is_unchanged(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "process-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://process-openai.example")
    monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", "")

    assert auxiliary._resolve_custom_runtime(config={}) == (
        "https://process-openai.example",
        "process-key",
        None,
    )
    assert auxiliary._nous_base_url() == ""


def test_unscoped_multiplex_route_setting_fails_closed(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://process.invalid")
    set_multiplex_active(True)

    with pytest.raises(UnscopedSecretError, match="OPENAI_BASE_URL"):
        auxiliary._resolve_custom_runtime(config={})


def test_nous_pool_base_url_uses_profile_scope_not_process_env(monkeypatch):
    monkeypatch.setenv(
        "NOUS_INFERENCE_BASE_URL", "https://process-nous.invalid/v1"
    )
    set_multiplex_active(True)
    entry = SimpleNamespace(
        provider="nous",
        runtime_base_url="https://pool-nous.example/v1",
        inference_base_url="https://pool-nous.example/v1",
        base_url="https://pool-nous.example/v1",
    )

    token = set_secret_scope(
        {"NOUS_INFERENCE_BASE_URL": "https://profile-nous.example/v1/"}
    )
    try:
        assert auxiliary._pool_runtime_base_url(entry) == (
            "https://profile-nous.example/v1"
        )
    finally:
        reset_secret_scope(token)

    empty_token = set_secret_scope({"NOUS_INFERENCE_BASE_URL": ""})
    try:
        assert auxiliary._pool_runtime_base_url(entry) == (
            "https://pool-nous.example/v1"
        )
    finally:
        reset_secret_scope(empty_token)

    with pytest.raises(UnscopedSecretError, match="NOUS_INFERENCE_BASE_URL"):
        auxiliary._pool_runtime_base_url(entry)


@pytest.mark.asyncio
async def test_xai_pool_base_url_does_not_swallow_unscoped_secret(monkeypatch):
    monkeypatch.setenv("HERMES_XAI_BASE_URL", "https://process-xai.invalid/v1")
    set_multiplex_active(True)
    entry = SimpleNamespace(
        runtime_api_key="pool-token",
        access_token="pool-token",
        runtime_base_url="https://pool-xai.example/v1",
        base_url="https://pool-xai.example/v1",
    )
    pool = SimpleNamespace(
        has_credentials=lambda: True,
        select=AsyncMock(return_value=entry),
    )

    with (
        patch.object(auxiliary, "load_pool", AsyncMock(return_value=pool)),
        patch(
            "hermes_cli.auth.resolve_xai_oauth_runtime_credentials",
            AsyncMock(
                return_value={
                    "api_key": "singleton-token",
                    "base_url": "https://singleton-xai.example/v1",
                }
            ),
        ) as singleton,
        pytest.raises(UnscopedSecretError, match="HERMES_XAI_BASE_URL"),
    ):
        await auxiliary._resolve_xai_oauth_for_aux()

    singleton.assert_not_awaited()
