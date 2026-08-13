import base64
import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.secret_scope import (
    is_multiplex_active,
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from hermes_cli import runtime_provider as rp

pytestmark = pytest.mark.asyncio


async def test_configured_api_key_provider_without_key_fails_closed(monkeypatch):
    """A saved provider must not resolve as another authenticated provider."""
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda *_args, **_kwargs: {"provider": "deepseek", "default": "deepseek-v4-pro"},
    )
    monkeypatch.setattr(
        rp,
        "load_pool",
        AsyncMock(return_value=SimpleNamespace(has_credentials=lambda: False)),
    )
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        AsyncMock(return_value={
            "provider": "deepseek",
            "api_key": "",
            "base_url": "https://api.deepseek.com/v1",
            "source": "default",
        }),
    )

    with pytest.raises(rp.AuthError, match="No usable credentials.*deepseek"):
        await rp.resolve_runtime_provider()


async def test_noauth_lmstudio_still_resolves(monkeypatch):
    """The fail-closed key guard preserves LM Studio's no-auth contract."""
    monkeypatch.setattr(
        rp,
        "load_pool",
        AsyncMock(return_value=SimpleNamespace(has_credentials=lambda: False)),
    )
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        AsyncMock(return_value={
            "provider": "lmstudio",
            "api_key": "lmstudio-noauth",
            "base_url": "http://localhost:1234/v1",
            "source": "default",
        }),
    )

    resolved = await rp.resolve_runtime_provider(requested="lmstudio")

    assert resolved["provider"] == "lmstudio"
    assert resolved["api_key"]


def _fake_invoke_jwt(ttl_seconds=3600):
    header = base64.urlsafe_b64encode(b'{"alg":"none","typ":"JWT"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "scope": "inference:invoke",
                "exp": int(time.time() + ttl_seconds),
            }
        ).encode()
    ).decode().rstrip("=")
    return f"{header}.{payload}.sig"


async def test_openai_codex_resolves_from_native_async_pool(monkeypatch):
    class _Entry:
        access_token = "pool-token"
        source = "manual"
        base_url = "https://chatgpt.com/backend-api/codex"

    class _Pool:
        def has_credentials(self):
            return True

        async def select(self):
            return _Entry()

    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="openai-codex"))
    monkeypatch.setattr(rp, "load_pool", AsyncMock(return_value=_Pool()))

    resolved = await rp.resolve_runtime_provider(requested="openai-codex")

    assert resolved["provider"] == "openai-codex"
    assert resolved["api_mode"] == "codex_responses"
    assert resolved["api_key"] == "pool-token"
    assert resolved["base_url"] == "https://chatgpt.com/backend-api/codex"


async def test_openai_codex_explicit_base_uses_native_async_pool(monkeypatch):
    class _Entry:
        runtime_api_key = "pool-token"
        runtime_base_url = "https://chatgpt.com/backend-api/codex"
        last_refresh = "2026-08-06T00:00:00+00:00"

    class _Pool:
        def has_credentials(self):
            return True

        async def select(self):
            return _Entry()

    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="openai-codex"))
    monkeypatch.setattr(rp, "load_pool", AsyncMock(return_value=_Pool()))

    resolved = await rp.resolve_runtime_provider(
        requested="openai-codex",
        explicit_base_url="https://codex.example.test/v1/",
    )

    assert resolved["api_key"] == "pool-token"
    assert resolved["base_url"] == "https://codex.example.test/v1"
    assert resolved["last_refresh"] == "2026-08-06T00:00:00+00:00"


async def test_openai_codex_uses_runtime_resolver_when_pool_is_empty(monkeypatch):
    pool = SimpleNamespace(has_credentials=lambda: False)
    resolve = AsyncMock(
        return_value={
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "singleton-token",
            "source": "hermes-auth-store",
            "last_refresh": "2026-08-08T00:00:00Z",
        }
    )
    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="openai-codex"))
    monkeypatch.setattr(rp, "load_pool", AsyncMock(return_value=pool))
    monkeypatch.setattr(rp.auth_mod, "resolve_codex_runtime_credentials", resolve)

    resolved = await rp.resolve_runtime_provider(requested="openai-codex")

    assert resolved["provider"] == "openai-codex"
    assert resolved["api_key"] == "singleton-token"
    resolve.assert_awaited_once_with()


async def test_xai_oauth_uses_runtime_resolver_when_pool_is_empty(monkeypatch):
    pool = SimpleNamespace(has_credentials=lambda: False)
    resolve = AsyncMock(
        return_value={
            "base_url": "https://api.x.ai/v1",
            "api_key": "singleton-token",
            "source": "hermes-auth-store",
            "last_refresh": "2026-08-08T00:00:00Z",
        }
    )
    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="xai-oauth"))
    monkeypatch.setattr(rp, "load_pool", AsyncMock(return_value=pool))
    monkeypatch.setattr(rp.auth_mod, "resolve_xai_oauth_runtime_credentials", resolve)

    resolved = await rp.resolve_runtime_provider(requested="xai-oauth")

    assert resolved["provider"] == "xai-oauth"
    assert resolved["api_key"] == "singleton-token"
    resolve.assert_awaited_once_with()


async def test_qwen_oauth_resolves_native_async_credentials(monkeypatch):
    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="qwen-oauth"))
    monkeypatch.setattr(rp, "_get_model_config", lambda *_args, **_kwargs: {})
    resolve = AsyncMock(
        return_value={
            "base_url": "https://portal.qwen.ai/v1",
            "api_key": "qwen-access",
            "source": "qwen-cli",
            "expires_at_ms": 123456789,
        }
    )
    monkeypatch.setattr(rp.auth_mod, "resolve_qwen_runtime_credentials", resolve)

    resolved = await rp.resolve_runtime_provider(requested="qwen-oauth")

    assert resolved == {
        "provider": "qwen-oauth",
        "api_mode": "chat_completions",
        "base_url": "https://portal.qwen.ai/v1",
        "api_key": "qwen-access",
        "source": "qwen-cli",
        "expires_at_ms": 123456789,
        "requested_provider": "qwen-oauth",
    }
    resolve.assert_awaited_once_with()


async def test_resolve_runtime_provider_ai_gateway(monkeypatch):
    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="ai-gateway"))
    monkeypatch.setattr(rp, "_get_model_config", lambda *_args, **_kwargs: {})
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-ai-gw-key")

    resolved = await rp.resolve_runtime_provider(requested="ai-gateway")

    assert resolved["provider"] == "ai-gateway"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["base_url"] == "https://ai-gateway.vercel.sh/v1"
    assert resolved["api_key"] == "test-ai-gw-key"
    assert resolved["requested_provider"] == "ai-gateway"


async def test_resolve_runtime_provider_lmstudio_uses_token_when_present(monkeypatch):
    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="lmstudio"))
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda *_args, **_kwargs: {
            "provider": "lmstudio",
            "base_url": "http://127.0.0.1:1234/v1",
            "default": "publisher/model-a",
        },
    )
    monkeypatch.setattr(
        rp,
        "load_pool",
        AsyncMock(return_value=type("Pool", (), {"has_credentials": lambda self: False})()),
    )
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        AsyncMock(return_value={
            "provider": "lmstudio",
            "api_key": "lm-token",
            "base_url": "http://127.0.0.1:1234/v1",
            "source": "LM_API_KEY",
        }),
    )

    resolved = await rp.resolve_runtime_provider(requested="lmstudio")

    assert resolved["provider"] == "lmstudio"
    assert resolved["api_key"] == "lm-token"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["base_url"] == "http://127.0.0.1:1234/v1"


async def test_resolve_runtime_provider_lmstudio_honors_saved_base_url(monkeypatch):
    """Pre-existing configs with `provider: lmstudio` + custom base_url must keep working.

    Before this PR, `lmstudio` aliased to `custom`, so a user with a remote
    LM Studio (e.g. lab box) could write `provider: "lmstudio"` plus
    `base_url: "http://192.168.1.10:1234/v1"` and the custom path honored it.
    Now that `lmstudio` is first-class with `inference_base_url=127.0.0.1`,
    the saved `base_url` from `model_cfg` must still win — otherwise this
    PR is a silent breaking change for those users.
    """
    monkeypatch.delenv("LM_API_KEY", raising=False)
    monkeypatch.delenv("LM_BASE_URL", raising=False)
    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="lmstudio"))
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda *_args, **_kwargs: {
            "provider": "lmstudio",
            "base_url": "http://192.168.1.10:1234/v1",
            "default": "qwen/qwen3-coder-30b",
        },
    )
    monkeypatch.setattr(
        rp,
        "load_pool",
        AsyncMock(return_value=type("Pool", (), {"has_credentials": lambda self: False})()),
    )
    # Don't mock resolve_api_key_provider_credentials — exercise the real
    # function so we test the end-to-end precedence between model_cfg and
    # the pconfig default.

    resolved = await rp.resolve_runtime_provider(requested="lmstudio")

    assert resolved["provider"] == "lmstudio"
    assert resolved["api_mode"] == "chat_completions"
    # The saved base_url must NOT be shadowed by the 127.0.0.1 default.
    assert resolved["base_url"] == "http://192.168.1.10:1234/v1"
    # No-auth LM Studio: missing LM_API_KEY substitutes the placeholder.
    assert resolved["api_key"] == "dummy-lm-api-key"


async def test_resolve_runtime_provider_lmstudio_saved_base_url_wins_over_env(monkeypatch):
    """Saved model.base_url takes precedence over LM_BASE_URL env var.

    This matches the established contract for all api_key providers: the
    explicit config value (model.base_url) wins over the env-derived
    default.  Users who saved a remote LM Studio URL must not have it
    silently overridden by a stale shell variable.
    """
    monkeypatch.delenv("LM_API_KEY", raising=False)
    monkeypatch.setenv("LM_BASE_URL", "http://override.local:9999/v1")
    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="lmstudio"))
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda *_args, **_kwargs: {
            "provider": "lmstudio",
            "base_url": "http://192.168.1.10:1234/v1",
            "default": "qwen/qwen3-coder-30b",
        },
    )
    monkeypatch.setattr(
        rp,
        "load_pool",
        AsyncMock(return_value=type("Pool", (), {"has_credentials": lambda self: False})()),
    )

    resolved = await rp.resolve_runtime_provider(requested="lmstudio")

    assert resolved["provider"] == "lmstudio"
    assert resolved["api_mode"] == "chat_completions"
    # Saved config base_url wins over env var (standard contract).
    assert resolved["base_url"] == "http://192.168.1.10:1234/v1"
    assert resolved["api_key"] == "dummy-lm-api-key"


async def test_resolve_runtime_provider_ai_gateway_explicit_override_skips_pool(monkeypatch):
    async def _unexpected_pool(provider):
        raise AssertionError(f"load_pool should not be called for {provider}")

    def _unexpected_provider_resolution(provider):
        raise AssertionError(f"resolve_api_key_provider_credentials should not be called for {provider}")

    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="ai-gateway"))
    monkeypatch.setattr(rp, "_get_model_config", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(rp, "load_pool", _unexpected_pool)
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        _unexpected_provider_resolution,
    )

    resolved = await rp.resolve_runtime_provider(
        requested="ai-gateway",
        explicit_api_key="ai-gateway-explicit-token",
        explicit_base_url="https://proxy.example.com/v1/",
    )

    assert resolved["provider"] == "ai-gateway"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["api_key"] == "ai-gateway-explicit-token"
    assert resolved["base_url"] == "https://proxy.example.com/v1"
    assert resolved["source"] == "explicit"
    assert resolved.get("credential_pool") is None


async def test_resolve_runtime_provider_openrouter_explicit(monkeypatch):
    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="openrouter"))
    monkeypatch.setattr(rp, "_get_model_config", lambda *_args, **_kwargs: {})
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    resolved = await rp.resolve_runtime_provider(
        requested="openrouter",
        explicit_api_key="test-key",
        explicit_base_url="https://example.com/v1/",
    )

    assert resolved["provider"] == "openrouter"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["api_key"] == "test-key"
    assert resolved["base_url"] == "https://example.com/v1"
    assert resolved["source"] == "explicit"


async def test_resolve_runtime_provider_auto_uses_openrouter_pool(monkeypatch):
    class _Entry:
        access_token = "pool-key"
        source = "manual"
        base_url = "https://openrouter.ai/api/v1"

    class _Pool:
        def has_credentials(self):
            return True

        async def select(self):
            return _Entry()

    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="openrouter"))
    monkeypatch.setattr(rp, "_get_model_config", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(rp, "load_pool", AsyncMock(return_value=_Pool()))
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    resolved = await rp.resolve_runtime_provider(requested="auto")

    assert resolved["provider"] == "openrouter"
    assert resolved["api_key"] == "pool-key"
    assert resolved["base_url"] == "https://openrouter.ai/api/v1"
    assert resolved["source"] == "manual"
    assert resolved.get("credential_pool") is not None


async def test_resolve_runtime_provider_openrouter_explicit_api_key_skips_pool(monkeypatch):
    class _Entry:
        access_token = "pool-key"
        source = "manual"
        base_url = "https://openrouter.ai/api/v1"

    class _Pool:
        def has_credentials(self):
            return True

        async def select(self):
            return _Entry()

    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="openrouter"))
    monkeypatch.setattr(rp, "_get_model_config", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(rp, "load_pool", AsyncMock(return_value=_Pool()))
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    resolved = await rp.resolve_runtime_provider(
        requested="openrouter",
        explicit_api_key="explicit-key",
    )

    assert resolved["provider"] == "openrouter"
    assert resolved["api_key"] == "explicit-key"
    assert resolved["base_url"] == rp.OPENROUTER_BASE_URL
    assert resolved["source"] == "explicit"
    assert resolved.get("credential_pool") is None


async def test_resolve_runtime_provider_openrouter_ignores_codex_config_base_url(monkeypatch):
    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="openrouter"))
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda *_args, **_kwargs: {
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
        },
    )
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    resolved = await rp.resolve_runtime_provider(requested="openrouter")

    assert resolved["provider"] == "openrouter"
    assert resolved["base_url"] == rp.OPENROUTER_BASE_URL


async def test_resolve_runtime_provider_auto_uses_custom_config_base_url(monkeypatch):
    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="openrouter"))
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda *_args, **_kwargs: {
            "provider": "auto",
            "base_url": "https://custom.example/v1/",
        },
    )
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    resolved = await rp.resolve_runtime_provider(requested="auto")

    assert resolved["provider"] == "openrouter"
    assert resolved["base_url"] == "https://custom.example/v1"


async def test_openrouter_key_takes_priority_over_openai_key(monkeypatch):
    """OPENROUTER_API_KEY should be used over OPENAI_API_KEY when both are set.

    Regression test for #289: users with OPENAI_API_KEY in .bashrc had it
    sent to OpenRouter instead of their OPENROUTER_API_KEY.
    """
    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="openrouter"))
    monkeypatch.setattr(rp, "_get_model_config", lambda *_args, **_kwargs: {})
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-should-lose")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-should-win")

    resolved = await rp.resolve_runtime_provider(requested="openrouter")

    assert resolved["api_key"] == "sk-or-should-win"


async def test_openai_key_used_when_no_openrouter_key(monkeypatch):
    """OPENAI_API_KEY is used as fallback when OPENROUTER_API_KEY is not set."""
    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="openrouter"))
    monkeypatch.setattr(rp, "_get_model_config", lambda *_args, **_kwargs: {})
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-fallback")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    resolved = await rp.resolve_runtime_provider(requested="openrouter")

    assert resolved["api_key"] == "sk-openai-fallback"


async def test_custom_endpoint_uses_saved_config_base_url_when_env_missing(monkeypatch):
    """Persisted custom endpoints in config.yaml must still resolve when
    OPENAI_BASE_URL is absent from the current environment.
    OPENAI_API_KEY / OPENROUTER_API_KEY must NOT leak to a non-OpenAI host
    (issue #28660) — local LLM servers get no-key-required instead."""
    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="openrouter"))
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda *_args, **_kwargs: {
            "provider": "custom",
            "base_url": "http://127.0.0.1:1234/v1",
        },
    )
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "local-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    resolved = await rp.resolve_runtime_provider(requested="custom")

    assert resolved["base_url"] == "http://127.0.0.1:1234/v1"
    # OPENAI_API_KEY must not leak to an unrelated host — local servers get
    # the no-key-required placeholder so the OpenAI SDK stays happy.
    assert resolved["api_key"] == "no-key-required"


async def test_custom_endpoint_explicit_custom_prefers_config_key(monkeypatch):
    """Explicit 'custom' provider with config base_url+api_key should use them.

    Updated for #4165: config.yaml is the source of truth, not OPENAI_BASE_URL.
    """
    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="openrouter"))
    monkeypatch.setattr(rp, "_get_model_config", lambda *_args, **_kwargs: {
        "provider": "custom",
        "base_url": "https://my-vllm-server.example.com/v1",
        "api_key": "sk-vllm-key",
    })
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-...leak")

    resolved = await rp.resolve_runtime_provider(requested="custom")

    assert resolved["base_url"] == "https://my-vllm-server.example.com/v1"
    assert resolved["api_key"] == "sk-vllm-key"


async def test_bare_custom_uses_loopback_model_base_url_when_provider_not_custom(monkeypatch):
    """Regression for #14676: /model can select Custom while YAML still lists another provider."""
    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="openrouter"))
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda *_args, **_kwargs: {
            "provider": "openrouter",
            "base_url": "http://127.0.0.1:8082/v1",
            "default": "my-local-model",
        },
    )
    monkeypatch.delenv("CUSTOM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    resolved = await rp.resolve_runtime_provider(requested="custom")

    assert resolved["provider"] == "custom"
    assert resolved["base_url"] == "http://127.0.0.1:8082/v1"
    # 127.0.0.1 is not openai.com — OPENAI_API_KEY must not leak here
    assert resolved["api_key"] == "no-key-required"


async def test_named_custom_provider_uses_saved_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        AsyncMock(return_value={
            "custom_providers": [
                {
                    "name": "Local",
                    "base_url": "http://1.2.3.4:1234/v1",
                    "api_key": "local-provider-key",
                }
            ]
        }),
    )
    monkeypatch.setattr(
        rp,
        "resolve_provider",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError(
                "resolve_provider should not be called for named custom providers"
            )
        ),
    )

    resolved = await rp.resolve_runtime_provider(requested="local")

    assert resolved["provider"] == "custom"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["base_url"] == "http://1.2.3.4:1234/v1"
    assert resolved["api_key"] == "local-provider-key"
    assert resolved["requested_provider"] == "local"
    assert resolved["source"] == "custom_provider:Local"


async def test_bare_custom_resolves_providers_dict_entry_named_custom(monkeypatch):
    """A request for bare ``provider="custom"`` must resolve a literal
    ``providers.custom`` entry (e.g. a cliproxy endpoint) instead of falling
    through to the global default. Regression for cron jobs stored with
    ``provider: "custom"`` failing with ``auth_unavailable: providers=codex``.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        AsyncMock(return_value={
            "providers": {
                "custom": {
                    "api": "https://cliproxy.example.com/v1",
                    "api_key": "cliproxy-key",
                    "default_model": "gpt-5.4",
                    "name": "CLIProxy",
                }
            }
        }),
    )
    # Reaching resolve_provider for bare custom with a matching entry means the
    # named-custom path was bypassed — that is the bug we are fixing.
    monkeypatch.setattr(
        rp,
        "resolve_provider",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError(
                "resolve_provider must not be called; providers.custom should match"
            )
        ),
    )

    resolved = await rp.resolve_runtime_provider(requested="custom")

    assert resolved["provider"] == "custom"
    assert resolved["base_url"] == "https://cliproxy.example.com/v1"
    assert resolved["api_key"] == "cliproxy-key"
    assert resolved["requested_provider"] == "custom"




async def test_named_custom_provider_same_url_uses_matching_key_env_and_api_mode(monkeypatch):
    """Named custom providers on one gateway must keep their own credentials and protocol."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("GPT_KEY", "gpt-secret")
    monkeypatch.setenv("CLAUDE_KEY", "claude-secret")
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        AsyncMock(return_value={
            "custom_providers": [
                {
                    "name": "gpt",
                    "base_url": "https://gateway.example.com",
                    "key_env": "GPT_KEY",
                    "api_mode": "codex_responses",
                    "model": "gpt-5.5",
                },
                {
                    "name": "claude",
                    "base_url": "https://gateway.example.com",
                    "key_env": "CLAUDE_KEY",
                    "api_mode": "anthropic_messages",
                    "model": "claude-opus-4-8",
                },
            ],
        }),
    )
    monkeypatch.setattr(
        rp,
        "resolve_provider",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError(
                "resolve_provider should not be called for named custom providers"
            )
        ),
    )

    resolved = await rp.resolve_runtime_provider(requested="custom:claude")

    assert resolved["provider"] == "custom"
    assert resolved["base_url"] == "https://gateway.example.com"
    assert resolved["api_key"] == "claude-secret"
    assert resolved["api_mode"] == "anthropic_messages"
    assert resolved["requested_provider"] == "custom:claude"
    assert resolved["model"] == "claude-opus-4-8"


async def test_named_custom_provider_falls_back_to_openai_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        AsyncMock(return_value={
            "custom_providers": [
                {
                    "name": "Local LLM",
                    "base_url": "http://localhost:1234/v1",
                }
            ]
        }),
    )
    monkeypatch.setattr(
        rp,
        "resolve_provider",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError(
                "resolve_provider should not be called for named custom providers"
            )
        ),
    )

    resolved = await rp.resolve_runtime_provider(requested="custom:local-llm")

    assert resolved["base_url"] == "http://localhost:1234/v1"
    # localhost is not openai.com — OPENAI_API_KEY must not leak to local endpoints (#28660)
    assert resolved["api_key"] == "no-key-required"
    assert resolved["requested_provider"] == "custom:local-llm"






async def test_named_custom_provider_wins_over_builtin_alias(monkeypatch):
    """A custom_providers entry named after a built-in *alias* (not a canonical
    provider name) must win over the built-in.  Regression guard for #15743:
    when users define ``custom_providers: [{name: kimi, ...}]`` and reference
    ``provider: kimi``, the built-in alias rewriting (``kimi`` → ``kimi-coding``)
    would otherwise hijack the request and send it to the wrong endpoint.
    """
    config = {
        "custom_providers": [
            {
                "name": "kimi",
                "base_url": "https://my-custom-kimi.example.com/v1",
                "api_key": "my-kimi-key",
            }
        ]
    }

    entry = rp._get_named_custom_provider("kimi", config=config)

    assert entry is not None
    assert entry["base_url"] == "https://my-custom-kimi.example.com/v1"
    assert entry["api_key"] == "my-kimi-key"


async def test_explicit_openrouter_skips_openai_base_url(monkeypatch):
    """When the user explicitly requests openrouter, OPENAI_BASE_URL
    (which may point to a custom endpoint) must not override the
    OpenRouter base URL.  Regression test for #874."""
    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="openrouter"))
    monkeypatch.setattr(rp, "_get_model_config", lambda *_args, **_kwargs: {})
    monkeypatch.setenv("OPENAI_BASE_URL", "https://my-custom-llm.example.com/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    resolved = await rp.resolve_runtime_provider(requested="openrouter")

    assert resolved["provider"] == "openrouter"
    assert "openrouter.ai" in resolved["base_url"]
    assert "my-custom-llm" not in resolved["base_url"]
    assert resolved["api_key"] == "or-test-key"






# ── api_mode config override tests ──────────────────────────────────────




async def test_minimax_config_base_url_overrides_hardcoded_default(monkeypatch):
    """model.base_url in config.yaml should override the hardcoded default (#6039)."""
    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="minimax"))
    monkeypatch.setattr(rp, "_get_model_config", lambda *_args, **_kwargs: {
        "provider": "minimax",
        "base_url": "https://api.minimaxi.com/anthropic",
    })
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
    monkeypatch.delenv("MINIMAX_BASE_URL", raising=False)

    resolved = await rp.resolve_runtime_provider(requested="minimax")

    assert resolved["provider"] == "minimax"
    assert resolved["base_url"] == "https://api.minimaxi.com/anthropic"
    assert resolved["api_mode"] == "anthropic_messages"


async def test_opencode_go_model_derivation_beats_stale_persisted_api_mode(monkeypatch):
    """opencode-zen/go re-derive api_mode from the effective model on every
    resolve, ignoring any persisted ``api_mode`` in config. Refs #16878 /
    PR #16888: the persisted mode from the previous default model must not
    leak across /model switches (a stale ``anthropic_messages`` on a
    chat_completions target would strip /v1 from base_url and 404).

    minimax-m2.5 is an Anthropic-routed model on opencode-go, so even when
    the config claims ``api_mode: chat_completions`` the runtime must pick
    ``anthropic_messages`` — the model dictates the mode, not the stale
    persisted setting.
    """
    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="opencode-go"))
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda *_args, **_kwargs: {
            "provider": "opencode-go",
            "default": "minimax-m2.5",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-opencode-go-key")
    monkeypatch.delenv("OPENCODE_GO_BASE_URL", raising=False)

    resolved = await rp.resolve_runtime_provider(requested="opencode-go")

    assert resolved["provider"] == "opencode-go"
    assert resolved["api_mode"] == "anthropic_messages"


# ------------------------------------------------------------------
# fix #2562 — resolve_provider("custom") must not remap to "openrouter"
# ------------------------------------------------------------------






async def test_auto_detected_stale_nous_falls_through_to_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-or-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)

    # resolve_provider returns "nous" (stale active_provider in auth.json)
    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="nous"))
    monkeypatch.setattr(
        rp,
        "load_pool",
        AsyncMock(
            return_value=type(
                "Pool",
                (),
                {"has_credentials": lambda self: False},
            )()
        ),
    )

    resolved = await rp.resolve_runtime_provider(requested="auto")

    assert resolved["provider"] == "openrouter"
    assert resolved["api_key"] == "test-or-key"


# ------------------------------------------------------------------
# fix #7828 — custom_providers model field must propagate to runtime
# ------------------------------------------------------------------




# ---------------------------------------------------------------------------
# GHSA-76xc-57q6-vm5m — Ollama URL substring leak
#
# Same bug class as the previously-fixed GHSA-xf8p-v2cg-h7h5 (OpenRouter).
# _resolve_openrouter_runtime's custom-endpoint branch selects OLLAMA_API_KEY
# when the base_url "looks like" ollama.com. Previous implementation used
# raw substring match; a custom base_url whose PATH or look-alike host
# merely contained "ollama.com" leaked OLLAMA_API_KEY to that endpoint.
# Fix: use base_url_host_matches (same helper as the OpenRouter sweep).
# ---------------------------------------------------------------------------

class TestOllamaUrlSubstringLeak:
    """Call-site regression tests for the fix in _resolve_openrouter_runtime."""

    def _make_cfg(self, base_url):
        return {"base_url": base_url, "api_key": "", "provider": "custom"}

    async def test_ollama_key_not_leaked_to_path_injection(self, monkeypatch):
        """http://127.0.0.1:9000/ollama.com/v1 — attacker endpoint with
        ollama.com in PATH. Must resolve to OPENAI_API_KEY, not OLLAMA_API_KEY."""
        monkeypatch.setenv("OPENAI_API_KEY", "oa-secret")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")
        monkeypatch.setenv("OLLAMA_API_KEY", "ol-SECRET-should-not-leak")
        monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="custom"))
        monkeypatch.setattr(rp, "_get_model_config", lambda *_args, **_kwargs: self._make_cfg(
            "http://127.0.0.1:9000/ollama.com/v1"
        ))
        monkeypatch.setattr(rp, "load_pool", AsyncMock(return_value=None))
        monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", AsyncMock(return_value=None))

        resolved = await rp.resolve_runtime_provider(requested="custom")

        assert "ol-SECRET" not in resolved["api_key"], (
            "OLLAMA_API_KEY must not be sent to an endpoint whose "
            "hostname is not ollama.com (GHSA-76xc-57q6-vm5m)"
        )
        # OPENAI_API_KEY must also not leak to non-openai.com hosts (#28660)
        assert resolved["api_key"] == "no-key-required"

    async def test_ollama_key_not_leaked_to_lookalike_host(self, monkeypatch):
        """ollama.com.attacker.test — look-alike host. OLLAMA_API_KEY
        must not be sent."""
        monkeypatch.setenv("OPENAI_API_KEY", "oa-secret")
        monkeypatch.setenv("OLLAMA_API_KEY", "ol-SECRET-should-not-leak")
        monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="custom"))
        monkeypatch.setattr(rp, "_get_model_config", lambda *_args, **_kwargs: self._make_cfg(
            "http://ollama.com.attacker.test:9000/v1"
        ))
        monkeypatch.setattr(rp, "load_pool", AsyncMock(return_value=None))
        monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", AsyncMock(return_value=None))

        resolved = await rp.resolve_runtime_provider(requested="custom")

        assert "ol-SECRET" not in resolved["api_key"]
        # OPENAI_API_KEY must also not leak to non-openai.com hosts (#28660)
        assert resolved["api_key"] == "no-key-required"

    async def test_ollama_key_sent_to_genuine_ollama_com(self, monkeypatch):
        """https://ollama.com/v1 — legit Ollama Cloud. OLLAMA_API_KEY
        should be used."""
        monkeypatch.setenv("OPENAI_API_KEY", "oa-secret")
        monkeypatch.setenv("OLLAMA_API_KEY", "ol-legit-key")
        monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="custom"))
        monkeypatch.setattr(rp, "_get_model_config", lambda *_args, **_kwargs: self._make_cfg(
            "https://ollama.com/v1"
        ))
        monkeypatch.setattr(rp, "load_pool", AsyncMock(return_value=None))
        monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", AsyncMock(return_value=None))

        resolved = await rp.resolve_runtime_provider(requested="custom")

        assert resolved["api_key"] == "ol-legit-key"



# =============================================================================
# Azure Foundry — both OpenAI-style and Anthropic-style endpoints
# =============================================================================

class TestAzureFoundryResolution:
    """Verify Azure Foundry resolves correctly for both API modes."""

    def _make_cfg(self, base_url: str, api_mode: str = "chat_completions"):
        return {
            "provider": "azure-foundry",
            "base_url": base_url,
            "api_mode": api_mode,
            # GPT-4 speaks chat completions on Azure, so this test's assertion
            # about chat_completions stays valid across the Apr 2026 fix that
            # upgrades GPT-5.x / codex deployments to codex_responses.
            "default": "gpt-4.1",
        }

    async def test_azure_foundry_reads_key_env_natively(self, monkeypatch):
        read_env = AsyncMock(return_value="dotenv-key")
        monkeypatch.setattr(rp, "get_env_value_prefer_dotenv", read_env)

        resolved = await rp._resolve_azure_foundry_runtime(
            requested_provider="azure-foundry",
            model_cfg={
                **self._make_cfg("https://example.openai.azure.com/openai/v1"),
                "key_env": "CUSTOM_AZURE_KEY",
            },
        )

        assert resolved["api_key"] == "dotenv-key"
        read_env.assert_awaited_once_with("CUSTOM_AZURE_KEY")


    async def test_azure_foundry_missing_base_url_raises(self, monkeypatch):
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "az-key")
        monkeypatch.delenv("AZURE_FOUNDRY_BASE_URL", raising=False)
        monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="azure-foundry"))
        monkeypatch.setattr(rp, "_get_model_config", lambda *_args, **_kwargs: {})
        monkeypatch.setattr(rp, "load_pool", AsyncMock(return_value=None))

        with pytest.raises(rp.AuthError, match="base URL"):
            await rp.resolve_runtime_provider(requested="azure-foundry")


    # -- Model-family api_mode inference -------------------------------------
    # Azure rejects /chat/completions on GPT-5.x / codex / o-series with
    # ``400 "The requested operation is unsupported."`` — the resolver must
    # upgrade api_mode to ``codex_responses`` for those models even when the
    # config was persisted as ``chat_completions`` (the default the setup
    # wizard writes when the user didn't pick explicitly).

    def _make_cfg_with_model(self, model: str, api_mode: str = "chat_completions"):
        return {
            "provider": "azure-foundry",
            "base_url": "https://synopsisse.openai.azure.com/openai/v1",
            "api_mode": api_mode,
            "default": model,
        }

    async def test_gpt5_codex_upgrades_chat_completions_to_responses(self, monkeypatch):
        """Reproduces Bob's April 2026 bug: gpt-5.3-codex on chat_completions."""
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "az-key")
        monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="azure-foundry"))
        monkeypatch.setattr(rp, "_get_model_config",
                            lambda *_args, **_kwargs: self._make_cfg_with_model("gpt-5.3-codex", "chat_completions"))
        monkeypatch.setattr(rp, "load_pool", AsyncMock(return_value=None))

        resolved = await rp.resolve_runtime_provider(requested="azure-foundry")

        assert resolved["api_mode"] == "codex_responses"
        assert resolved["base_url"] == "https://synopsisse.openai.azure.com/openai/v1"


    async def test_o3_mini_upgrades(self, monkeypatch):
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "az-key")
        monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="azure-foundry"))
        monkeypatch.setattr(rp, "_get_model_config",
                            lambda *_args, **_kwargs: self._make_cfg_with_model("o3-mini", "chat_completions"))
        monkeypatch.setattr(rp, "load_pool", AsyncMock(return_value=None))

        resolved = await rp.resolve_runtime_provider(requested="azure-foundry")

        assert resolved["api_mode"] == "codex_responses"


# ──────────────────────────────────────────────────────────────────────────
# Azure Anthropic — honor user-specified env var hints (key_env / api_key_env)
#
# When the user points provider=anthropic at an Azure Foundry base URL, the
# runtime resolver previously hardcoded `AZURE_ANTHROPIC_KEY` and
# `ANTHROPIC_API_KEY` as the only env var sources.  This meant
# `key_env: MY_CUSTOM_VAR` on the model config was silently ignored — and
# the Azure Foundry docs that showed `api_key_env:` were broken as a result.
#
# These tests lock in the priority chain:
#   1. model_cfg.key_env → os.getenv(value)
#   2. model_cfg.api_key_env → os.getenv(value) (docs alias)
#   3. model_cfg.api_key (inline value)
#   4. AZURE_ANTHROPIC_KEY env var
#   5. ANTHROPIC_API_KEY env var
# ──────────────────────────────────────────────────────────────────────────


class TestAzureAnthropicEnvVarHint:
    _AZURE_URL = "https://my-resource.services.ai.azure.com/anthropic"

    def _cfg(self, **overrides):
        base = {"provider": "anthropic", "base_url": self._AZURE_URL}
        base.update(overrides)
        return base

    async def test_key_env_hint_picks_custom_var(self, monkeypatch):
        """model.key_env names a non-default env var → that var's value is used."""
        monkeypatch.delenv("AZURE_ANTHROPIC_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("MY_CUSTOM_AZURE_KEY", "from-custom-var")
        monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="anthropic"))
        monkeypatch.setattr(rp, "_get_model_config",
                            lambda *_args, **_kwargs: self._cfg(key_env="MY_CUSTOM_AZURE_KEY"))
        monkeypatch.setattr(rp, "load_pool", AsyncMock(return_value=None))

        resolved = await rp.resolve_runtime_provider(requested="anthropic")

        assert resolved["api_key"] == "from-custom-var"
        assert resolved["base_url"] == self._AZURE_URL


    async def test_key_env_points_at_unset_var_falls_through(self, monkeypatch):
        """If key_env names an env var that isn't set, fall through to the
        historical fixed names rather than failing outright."""
        monkeypatch.setenv("AZURE_ANTHROPIC_KEY", "fallback-works")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("UNSET_VAR", raising=False)
        monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="anthropic"))
        monkeypatch.setattr(rp, "_get_model_config",
                            lambda *_args, **_kwargs: self._cfg(key_env="UNSET_VAR"))
        monkeypatch.setattr(rp, "load_pool", AsyncMock(return_value=None))

        resolved = await rp.resolve_runtime_provider(requested="anthropic")

        assert resolved["api_key"] == "fallback-works"


    async def test_non_azure_anthropic_path_ignores_key_env(self, monkeypatch):
        """key_env is only consulted on Azure endpoints — non-Azure Anthropic
        still goes through the regular resolve_anthropic_token chain."""
        monkeypatch.setenv("MY_KEY", "custom-key-value")
        monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="anthropic"))
        monkeypatch.setattr(rp, "_get_model_config", lambda *_args, **_kwargs: {
            "provider": "anthropic",
            "base_url": "https://api.anthropic.com",  # non-Azure
            "key_env": "MY_KEY",
        })
        monkeypatch.setattr(rp, "load_pool", AsyncMock(return_value=None))
        called = {"resolve_anthropic_token": False}
        async def _fake_resolve():
            called["resolve_anthropic_token"] = True
            return "token-from-resolver"
        monkeypatch.setattr(
            "agent.anthropic_adapter.resolve_anthropic_token",
            _fake_resolve,
        )

        resolved = await rp.resolve_runtime_provider(requested="anthropic")

        # The normal chain runs — key_env is not consulted off-Azure.
        assert called["resolve_anthropic_token"] is True
        assert resolved["api_key"] == "token-from-resolver"


# ──────────────────────────────────────────────────────────────────────────
# custom_providers / providers normalizer — api_key_env alias for key_env
# ──────────────────────────────────────────────────────────────────────────


class TestProviderEntryApiKeyEnvAlias:
    """The `providers.<name>` and `custom_providers[i]` normalizer must accept
    `api_key_env` as an alias for `key_env` so configs written against the
    documented Azure Foundry YAML shape (or imported from other tools that
    use `api_key_env`) resolve correctly."""

    async def test_snake_case_api_key_env_normalizes_to_key_env(self):
        from hermes_cli.config import _normalize_custom_provider_entry
        entry = {
            "name": "vendor",
            "base_url": "https://api.vendor.example.com/v1",
            "api_key_env": "MY_VENDOR_KEY",
        }
        normalized = _normalize_custom_provider_entry(dict(entry), provider_key="vendor")
        assert normalized is not None
        assert normalized.get("key_env") == "MY_VENDOR_KEY"


    async def test_valid_fields_set_lists_key_env(self):
        """The _VALID_CUSTOM_PROVIDER_FIELDS documentation set must include
        key_env so the set stays in sync with what the runtime actually reads."""
        from hermes_cli.config import _VALID_CUSTOM_PROVIDER_FIELDS
        assert "key_env" in _VALID_CUSTOM_PROVIDER_FIELDS

    async def test_extra_body_is_supported_schema(self):
        from hermes_cli.config import (
            _VALID_CUSTOM_PROVIDER_FIELDS,
            _normalize_custom_provider_entry,
        )
        entry = {
            "name": "vendor",
            "base_url": "https://api.vendor.example.com/v1",
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": True},
                "include_reasoning": True,
            },
        }
        normalized = _normalize_custom_provider_entry(dict(entry), provider_key="vendor")
        assert normalized is not None
        assert "extra_body" in _VALID_CUSTOM_PROVIDER_FIELDS
        assert normalized["extra_body"] == entry["extra_body"]
# =============================================================================
# Tencent TokenHub — API-key provider runtime resolution
# =============================================================================



# ---------------------------------------------------------------------------
# minimax-oauth runtime resolution tests (added by feat/minimax-oauth-provider)
# ---------------------------------------------------------------------------

async def test_minimax_oauth_resolves_native_async_credentials(monkeypatch):
    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="minimax-oauth"))
    monkeypatch.setattr(rp, "_get_model_config", lambda *_args, **_kwargs: {"provider": "minimax-oauth"})
    monkeypatch.setattr(rp, "load_pool", AsyncMock(return_value=None))
    monkeypatch.setattr(
        rp,
        "_resolve_named_custom_runtime",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        rp,
        "_resolve_explicit_runtime",
        AsyncMock(return_value=None),
    )

    resolve = AsyncMock(
        return_value={
            "provider": "minimax-oauth",
            "api_key": "oauth-token",
            "base_url": "https://api.minimax.io/anthropic",
            "source": "oauth",
        }
    )
    monkeypatch.setattr(
        rp.auth_mod,
        "resolve_minimax_oauth_runtime_credentials",
        resolve,
    )

    runtime = await rp.resolve_runtime_provider(requested="minimax-oauth")

    assert runtime == {
        "provider": "minimax-oauth",
        "api_mode": "anthropic_messages",
        "base_url": "https://api.minimax.io/anthropic",
        "api_key": "oauth-token",
        "source": "oauth",
        "requested_provider": "minimax-oauth",
    }
    resolve.assert_awaited_once_with()


async def test_minimax_oauth_runtime_does_not_use_stale_pool_token(monkeypatch):

    class _Entry:
        access_token = "oauth-token"
        source = "manual:minimax_oauth"
        base_url = "https://api.minimax.io/anthropic"

    class _Pool:
        def has_credentials(self):
            return True

        async def select(self):
            return _Entry()

    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="minimax-oauth"))
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda *_args, **_kwargs: {
            "provider": "minimax-oauth",
            "default": "MiniMax-M2.7",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr(rp, "load_pool", AsyncMock(return_value=_Pool()))
    monkeypatch.setattr(rp, "_resolve_named_custom_runtime", AsyncMock(return_value=None))
    monkeypatch.setattr(rp, "_resolve_explicit_runtime", AsyncMock(return_value=None))
    resolve = AsyncMock(
        return_value={
            "provider": "minimax-oauth",
            "api_key": "fresh-oauth-token",
            "base_url": "https://api.minimax.io/anthropic",
            "source": "oauth",
        }
    )
    monkeypatch.setattr(
        rp.auth_mod,
        "resolve_minimax_oauth_runtime_credentials",
        resolve,
    )

    runtime = await rp.resolve_runtime_provider(requested="minimax-oauth")

    assert runtime["api_key"] == "fresh-oauth-token"
    resolve.assert_awaited_once_with()


# ----------------------------------------------------------------------
# GitHub #27132 — provider aliases (ollama/vllm/llamacpp/llama-cpp) must
# follow the same base_url trust + routing rules as bare `provider: custom`.
# Without this, a YAML `provider: ollama` with a LAN/WireGuard `base_url`
# silently falls through to OpenRouter (HTTP 401).
# ----------------------------------------------------------------------






async def test_openai_key_only_sent_to_openai_host(monkeypatch):
    """OPENAI_API_KEY must only be forwarded to api.openai.com, not to
    arbitrary custom endpoints (issue #28660)."""
    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="openrouter"))
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda *_args, **_kwargs: {
            "provider": "custom",
            "base_url": "https://api.deepseek.com/v1",
        },
    )
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    resolved = await rp.resolve_runtime_provider(requested="custom")

    assert resolved["base_url"] == "https://api.deepseek.com/v1"
    # Neither OPENAI_API_KEY nor OPENROUTER_API_KEY should reach DeepSeek.
    assert resolved["api_key"] == "no-key-required"


async def test_openai_key_reaches_openai_host(monkeypatch):
    """OPENAI_API_KEY must be forwarded when the base_url is api.openai.com."""
    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="openrouter"))
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda *_args, **_kwargs: {
            "provider": "custom",
            "base_url": "https://api.openai.com/v1",
        },
    )
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")

    resolved = await rp.resolve_runtime_provider(requested="custom")

    assert resolved["api_key"] == "sk-openai-secret"


async def test_openrouter_key_reaches_openrouter_host(monkeypatch):
    """OPENROUTER_API_KEY must be forwarded when the base_url is openrouter.ai."""
    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="openrouter"))
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda *_args, **_kwargs: {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
        },
    )
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")

    resolved = await rp.resolve_runtime_provider(requested="openrouter")

    assert resolved["api_key"] == "or-secret"


# ----------------------------------------------------------------------
# Issue #28660 — bonus: `<VENDOR>_API_KEY` derivation from host.
# After the host-gating fix, users with a `DEEPSEEK_API_KEY` set and
# `base_url: https://api.deepseek.com/v1` should get the key picked up
# without needing to configure custom_providers.key_env first.
# ----------------------------------------------------------------------






async def test_host_derived_key_does_not_leak_to_lookalike_host(monkeypatch):
    """DEEPSEEK_API_KEY must NOT be sent to an attacker-controlled lookalike
    host (e.g. api.deepseek.com.attacker.test). The host-derive helper uses
    proper hostname parsing so it picks the *attacker's* vendor label, not
    DEEPSEEK — and any real DEEPSEEK_API_KEY stays put."""
    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="openrouter"))
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda *_args, **_kwargs: {
            "provider": "custom",
            "base_url": "https://api.deepseek.com.attacker.test/v1",
        },
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-secret")

    resolved = await rp.resolve_runtime_provider(requested="custom")

    assert "sk-deepseek-secret" not in (resolved["api_key"] or "")
    # No ATTACKER_API_KEY is set, so the chain falls through to no-key-required.
    assert resolved["api_key"] == "no-key-required"


async def test_host_derived_key_skips_already_handled_vendors(monkeypatch):
    """The host-derive helper must not double-resolve OPENAI / OPENROUTER /
    OLLAMA env vars — those are owned by their explicit host-gated paths.
    Specifically, OPENAI_API_KEY must not leak to a non-openai host via the
    `openai` label in a path or subdomain."""
    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="openrouter"))
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda *_args, **_kwargs: {
            "provider": "custom",
            # Hosts like proxy.openai.evil should derive nothing — but even
            # if "openai" were the registrable label, the explicit
            # OPENAI/OPENROUTER/OLLAMA filter blocks it.
            "base_url": "https://api.example.com/v1",
        },
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")

    resolved = await rp.resolve_runtime_provider(requested="custom")

    # example.com has no EXAMPLE_API_KEY set, and OPENAI/OPENROUTER are gated
    # on their own hosts — chain falls through to no-key-required.
    assert resolved["api_key"] == "no-key-required"




def _patch_bedrock(monkeypatch, config_default=""):
    """Stub the bedrock_adapter helpers resolve_runtime_provider imports.

    Leaves the real ``is_anthropic_bedrock_model`` in place so the dual-path
    routing decision is exercised for real. ``config_default`` is the persisted
    ``model.default`` — kept non-Claude here to prove ``target_model`` (not the
    stale config default) drives the api_mode decision.
    """
    import agent.bedrock_adapter as ba

    monkeypatch.setattr(rp, "resolve_provider", AsyncMock(return_value="bedrock"))
    monkeypatch.setattr(rp, "_get_model_config", lambda *_args, **_kwargs: {"default": config_default})
    monkeypatch.setattr(ba, "resolve_aws_auth_env_var", AsyncMock(return_value="AWS_PROFILE"))
    monkeypatch.setattr(ba, "resolve_bedrock_region", AsyncMock(return_value="eu-north-1"))


async def test_resolve_runtime_provider_bedrock_claude_uses_anthropic(monkeypatch):
    _patch_bedrock(monkeypatch, config_default="amazon.nova-pro-v1:0")

    runtime = await rp.resolve_runtime_provider(
        requested="bedrock",
        target_model="global.anthropic.claude-sonnet-4-6",
    )

    assert runtime["api_mode"] == "anthropic_messages"
    assert runtime["bedrock_anthropic"] is True
    assert runtime["region"] == "eu-north-1"


async def test_resolve_runtime_provider_bedrock_nova_uses_converse(monkeypatch):
    _patch_bedrock(monkeypatch, config_default="global.anthropic.claude-sonnet-4-6")

    runtime = await rp.resolve_runtime_provider(
        requested="bedrock",
        target_model="amazon.nova-pro-v1:0",
    )

    assert runtime["api_mode"] == "bedrock_converse"
    assert "bedrock_anthropic" not in runtime


async def test_bedrock_bearer_route_is_profile_scoped(monkeypatch):
    _patch_bedrock(monkeypatch)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "foreign-process-token")
    previous = is_multiplex_active()
    set_multiplex_active(True)

    async def resolve(secrets):
        token = set_secret_scope(secrets)
        try:
            await asyncio.sleep(0)
            return await rp.resolve_runtime_provider(
                requested="bedrock",
                target_model="global.anthropic.claude-sonnet-4-6",
            )
        finally:
            reset_secret_scope(token)

    try:
        bearer, sigv4 = await asyncio.gather(
            resolve({"AWS_BEARER_TOKEN_BEDROCK": "profile-token"}),
            resolve({}),
        )
    finally:
        set_multiplex_active(previous)

    assert bearer["api_mode"] == "bedrock_converse"
    assert "bedrock_anthropic" not in bearer
    assert sigv4["api_mode"] == "anthropic_messages"
    assert sigv4["bedrock_anthropic"] is True


async def test_auto_provider_with_local_base_url_bypasses_anthropic_key(monkeypatch):
    """provider:auto + base_url:localhost should NOT route to Anthropic even if
    ANTHROPIC_API_KEY is set in the environment. Regression test for #3846.

    Users running Ollama locally had requests silently sent to Anthropic's API
    because resolve_provider("auto") picked up ANTHROPIC_API_KEY before the
    config.yaml base_url was checked.
    """
    # ANTHROPIC_API_KEY is present in environment — should be ignored
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda *_args, **_kwargs: {
            "default": "ollama/minimax-m2.7:cloud",
            "provider": "auto",
            "base_url": "http://localhost:11434",
        },
    )

    resolved = await rp.resolve_runtime_provider()

    # Must NOT go to Anthropic's API
    assert "anthropic.com" not in resolved.get("base_url", ""), (
        f"Expected custom endpoint, got Anthropic: {resolved}"
    )
    assert resolved["base_url"] == "http://localhost:11434"
    # provider should be custom/openrouter (OpenAI-compat), not anthropic
    assert resolved["provider"] != "anthropic", (
        f"Should have routed to custom endpoint, not anthropic: {resolved}"
    )


async def test_auto_provider_with_known_cloud_base_url_still_uses_anthropic(monkeypatch):
    """provider:auto + base_url pointing to Anthropic should still use Anthropic.

    The local-endpoint bypass only applies to non-cloud endpoints; when the
    configured base_url IS a cloud API root, resolve_provider() must run
    normally and pick up ANTHROPIC_API_KEY.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda *_args, **_kwargs: {
            "provider": "auto",
            "base_url": "https://api.anthropic.com",
        },
    )

    resolved = await rp.resolve_runtime_provider()

    # Cloud base_url → bypass does NOT fire → resolve_provider picks anthropic.
    assert resolved["provider"] == "anthropic", (
        f"Cloud base_url must not be diverted to the custom resolver: {resolved}"
    )


async def test_auto_provider_lookalike_cloud_host_does_not_bypass_to_cloud(monkeypatch):
    """A look-alike host (api.anthropic.com.attacker.test) must be treated as a
    custom endpoint, NOT mistaken for the real Anthropic cloud.

    Guards the host-match (vs naive substring) check in the local-endpoint
    bypass: substring matching on "api.anthropic.com" would wrongly classify
    this attacker-controlled host as cloud and hand it the ANTHROPIC_API_KEY.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    lookalike = "http://api.anthropic.com.attacker.test/v1"
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda *_args, **_kwargs: {"provider": "auto", "base_url": lookalike},
    )

    resolved = await rp.resolve_runtime_provider()

    # Host doesn't actually match anthropic.com → bypass fires → custom route,
    # and the real Anthropic credential is NOT sent there.
    assert resolved["provider"] != "anthropic", (
        f"Look-alike host must not be classified as Anthropic cloud: {resolved}"
    )
    assert resolved["base_url"] == lookalike


# ---------------------------------------------------------------------------
# extra_headers support for named custom providers (#3526 salvage)
# ---------------------------------------------------------------------------






async def test_resolve_named_custom_runtime_pool_result_includes_extra_headers(monkeypatch):
    """extra_headers must survive the credential-pool path too."""
    pool_return_value = {
        "provider": "custom",
        "api_mode": "chat_completions",
        "base_url": "https://lmstudio.example.com/v1",
        "api_key": "pooled-key",
        "source": "pool:lmstudio-pool",
        "credential_pool": "fake-pool",
    }
    monkeypatch.setattr(
        rp,
        "_try_resolve_from_custom_pool",
        AsyncMock(return_value=pool_return_value),
    )
    monkeypatch.setattr(
        rp,
        "_get_named_custom_provider",
        lambda p, *_args, **_kwargs: {
            "name": "lmstudio",
            "base_url": "https://lmstudio.example.com/v1",
            "api_key": "not-used-when-pooled",
            "extra_headers": {
                "CF-Access-Client-Id": "xxx.access",
                "CF-Access-Client-Secret": "yyy",
            },
        },
    )

    # Exercise the public resolver: it is responsible for preserving the
    # original named identity after the pool path canonicalizes to "custom".
    resolved = await rp.resolve_runtime_provider(requested="custom:lmstudio")

    assert resolved is not None
    assert resolved["extra_headers"] == {
        "CF-Access-Client-Id": "xxx.access",
        "CF-Access-Client-Secret": "yyy",
    }
    assert resolved["api_key"] == "pooled-key"
    assert resolved["source"] == "pool:lmstudio-pool"
    assert resolved["provider"] == "custom"
    assert resolved["requested_provider"] == "custom:lmstudio"
