from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent import secret_scope
from hermes_cli import auth


@pytest.fixture(autouse=True)
def _restore_secret_scope():
    previous_multiplex = secret_scope.is_multiplex_active()
    outer_token = secret_scope.set_secret_scope(None)
    try:
        yield
    finally:
        secret_scope.reset_secret_scope(outer_token)
        secret_scope.set_multiplex_active(previous_multiplex)


async def _in_scope(secrets, operation):
    token = secret_scope.set_secret_scope(secrets)
    try:
        await asyncio.sleep(0)
        return await operation()
    finally:
        secret_scope.reset_secret_scope(token)


@pytest.mark.asyncio
async def test_auto_provider_detection_isolates_concurrent_profile_keys(
    monkeypatch,
):
    from agent import bedrock_adapter
    from hermes_cli import config
    import providers

    monkeypatch.setenv("OPENAI_API_KEY", "foreign-process-openai")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "foreign-process-deepseek")
    monkeypatch.setattr(config, "load_config_readonly", AsyncMock(return_value={}))
    monkeypatch.setattr(auth, "_load_auth_store", AsyncMock(return_value={}))
    monkeypatch.setattr(providers, "list_providers", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        bedrock_adapter,
        "has_aws_credentials",
        AsyncMock(return_value=False),
    )
    secret_scope.set_multiplex_active(True)

    async def resolve():
        return await auth.resolve_provider("auto")

    openai, deepseek = await asyncio.gather(
        _in_scope(
            {
                "OPENAI_API_KEY": "profile-openai",
                "DEEPSEEK_API_KEY": "",
            },
            resolve,
        ),
        _in_scope(
            {
                "OPENAI_API_KEY": "",
                "OPENROUTER_API_KEY": "",
                "DEEPSEEK_API_KEY": "profile-deepseek",
            },
            resolve,
        ),
    )

    assert openai == "openrouter"
    assert deepseek == "deepseek"


@pytest.mark.asyncio
async def test_auto_provider_detection_fails_closed_without_profile_scope(
    monkeypatch,
):
    from hermes_cli import config
    import providers

    monkeypatch.setenv("OPENAI_API_KEY", "foreign-process-openai")
    monkeypatch.setattr(config, "load_config_readonly", AsyncMock(return_value={}))
    monkeypatch.setattr(auth, "_load_auth_store", AsyncMock(return_value={}))
    monkeypatch.setattr(providers, "list_providers", AsyncMock(return_value=[]))
    secret_scope.set_multiplex_active(True)

    with pytest.raises(secret_scope.UnscopedSecretError, match="OPENAI_API_KEY"):
        await auth.resolve_provider("auto")


def _oauth_pool(provider: str):
    base_url = {
        "openai-codex": "https://entry-codex.example/codex",
        "xai-oauth": auth.DEFAULT_XAI_OAUTH_BASE_URL,
    }[provider]
    entry = SimpleNamespace(
        id=f"{provider}-entry",
        access_token=f"{provider}-access",
        runtime_api_key=f"{provider}-access",
        runtime_base_url=base_url,
        last_refresh="2026-08-13T00:00:00Z",
    )

    class Pool:
        def has_credentials(self):
            return True

        async def select(self):
            return entry

    return Pool()


@pytest.mark.asyncio
async def test_oauth_runtime_base_urls_isolate_concurrent_profiles(
    tmp_path, monkeypatch
):
    from agent import credential_pool

    monkeypatch.setenv("HERMES_QWEN_BASE_URL", "https://foreign-qwen.example/v1")
    monkeypatch.setenv("HERMES_CODEX_BASE_URL", "https://foreign-codex.example")
    monkeypatch.setenv("HERMES_XAI_BASE_URL", "https://foreign.x.ai/v1")
    monkeypatch.setattr(
        auth,
        "_read_qwen_cli_tokens",
        AsyncMock(
            return_value={
                "access_token": "qwen-access",
                "expiry_date": int((time.time() + 3600) * 1000),
            }
        ),
    )
    monkeypatch.setattr(auth, "_qwen_cli_auth_path", lambda: tmp_path / "qwen.json")
    monkeypatch.setattr(
        credential_pool,
        "load_pool",
        AsyncMock(side_effect=lambda provider: _oauth_pool(provider)),
    )
    monkeypatch.setattr(
        auth,
        "get_provider_auth_state",
        AsyncMock(return_value={}),
    )
    secret_scope.set_multiplex_active(True)

    async def resolve_all():
        qwen, codex, xai = await asyncio.gather(
            auth.resolve_qwen_runtime_credentials(refresh_if_expiring=False),
            auth.resolve_codex_runtime_credentials(refresh_if_expiring=False),
            auth.resolve_xai_oauth_runtime_credentials(refresh_if_expiring=False),
        )
        return qwen["base_url"], codex["base_url"], xai["base_url"]

    profile_a, profile_b = await asyncio.gather(
        _in_scope(
            {
                "QWEN_API_KEY": "qwen-a-access",
                "HERMES_QWEN_BASE_URL": "https://qwen-a.example/v1/",
                "HERMES_CODEX_BASE_URL": "https://codex-a.example/codex/",
                "HERMES_XAI_BASE_URL": "https://a.x.ai/v1/",
            },
            resolve_all,
        ),
        _in_scope(
            {
                "QWEN_API_KEY": "qwen-b-access",
                "HERMES_QWEN_BASE_URL": "https://qwen-b.example/v1",
                "HERMES_CODEX_BASE_URL": "https://codex-b.example/codex",
                "HERMES_XAI_BASE_URL": "https://b.x.ai/v1",
            },
            resolve_all,
        ),
    )

    assert profile_a == (
        "https://qwen-a.example/v1",
        "https://codex-a.example/codex",
        "https://a.x.ai/v1",
    )
    assert profile_b == (
        "https://qwen-b.example/v1",
        "https://codex-b.example/codex",
        "https://b.x.ai/v1",
    )


@pytest.mark.asyncio
async def test_oauth_runtime_explicit_empty_does_not_borrow_process_urls(
    tmp_path, monkeypatch
):
    from agent import credential_pool

    monkeypatch.setenv("HERMES_QWEN_BASE_URL", "https://foreign-qwen.example/v1")
    monkeypatch.setenv("HERMES_CODEX_BASE_URL", "https://foreign-codex.example")
    monkeypatch.setenv("HERMES_XAI_BASE_URL", "https://foreign.x.ai/v1")
    monkeypatch.setattr(
        auth,
        "_read_qwen_cli_tokens",
        AsyncMock(return_value={"access_token": "qwen-access"}),
    )
    monkeypatch.setattr(auth, "_qwen_cli_auth_path", lambda: tmp_path / "qwen.json")
    monkeypatch.setattr(
        credential_pool,
        "load_pool",
        AsyncMock(side_effect=lambda provider: _oauth_pool(provider)),
    )
    monkeypatch.setattr(
        auth,
        "get_provider_auth_state",
        AsyncMock(return_value={}),
    )
    secret_scope.set_multiplex_active(True)

    async def resolve_all():
        qwen, codex, xai = await asyncio.gather(
            auth.resolve_qwen_runtime_credentials(refresh_if_expiring=False),
            auth.resolve_codex_runtime_credentials(refresh_if_expiring=False),
            auth.resolve_xai_oauth_runtime_credentials(refresh_if_expiring=False),
        )
        return qwen["base_url"], codex["base_url"], xai["base_url"]

    assert await _in_scope(
        {
            "QWEN_API_KEY": "qwen-access",
            "HERMES_QWEN_BASE_URL": "",
            "HERMES_CODEX_BASE_URL": "",
            "HERMES_XAI_BASE_URL": "",
            "XAI_BASE_URL": "",
        },
        resolve_all,
    ) == (
        auth.DEFAULT_QWEN_BASE_URL,
        "https://entry-codex.example/codex",
        auth.DEFAULT_XAI_OAUTH_BASE_URL,
    )


@pytest.mark.asyncio
async def test_nous_and_codex_helpers_use_profile_scope(monkeypatch):
    monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", "https://foreign-nous.example/v1")
    monkeypatch.setenv("HERMES_PORTAL_BASE_URL", "https://foreign-portal.example")
    monkeypatch.setenv("HERMES_CODEX_BASE_URL", "https://foreign-codex.example")
    secret_scope.set_multiplex_active(True)

    async def resolve_helpers():
        return (
            auth._nous_inference_env_override(),
            auth._nous_portal_env_override(),
            auth._codex_usage_probe_url(None),
        )

    profile_a, profile_b = await asyncio.gather(
        _in_scope(
            {
                "NOUS_INFERENCE_BASE_URL": "https://nous-a.example/v1/",
                "HERMES_PORTAL_BASE_URL": "https://portal-a.example/",
                "HERMES_CODEX_BASE_URL": "https://codex-a.example/codex/",
            },
            resolve_helpers,
        ),
        _in_scope(
            {
                "NOUS_INFERENCE_BASE_URL": "",
                "HERMES_PORTAL_BASE_URL": "",
                "NOUS_PORTAL_BASE_URL": "",
                "HERMES_CODEX_BASE_URL": "",
            },
            resolve_helpers,
        ),
    )

    assert profile_a == (
        "https://nous-a.example/v1",
        "https://portal-a.example",
        "https://codex-a.example/api/codex/usage",
    )
    assert profile_b == (
        None,
        None,
        auth.DEFAULT_CODEX_BASE_URL.removesuffix("/codex") + "/wham/usage",
    )


@pytest.mark.asyncio
async def test_api_key_provider_base_urls_isolate_concurrent_profiles(
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "foreign-process-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://foreign.example/v1")
    secret_scope.set_multiplex_active(True)

    async def resolve_all():
        credentials, status = await asyncio.gather(
            auth.resolve_api_key_provider_credentials("openai-api"),
            auth.get_api_key_provider_status("openai-api"),
        )
        return credentials, status

    profile_a, profile_b = await asyncio.gather(
        _in_scope(
            {
                "OPENAI_API_KEY": "profile-a-key",
                "OPENAI_BASE_URL": "https://profile-a.example/v1/",
            },
            resolve_all,
        ),
        _in_scope(
            {
                "OPENAI_API_KEY": "profile-b-key",
                "OPENAI_BASE_URL": "https://profile-b.example/v1",
            },
            resolve_all,
        ),
    )

    assert profile_a[0]["api_key"] == "profile-a-key"
    assert profile_a[0]["base_url"] == "https://profile-a.example/v1"
    assert profile_a[1]["base_url"] == "https://profile-a.example/v1/"
    assert profile_b[0]["api_key"] == "profile-b-key"
    assert profile_b[0]["base_url"] == "https://profile-b.example/v1"
    assert profile_b[1]["base_url"] == "https://profile-b.example/v1"


@pytest.mark.asyncio
async def test_api_key_provider_explicit_empty_base_url_uses_default(
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://foreign.example/v1")
    secret_scope.set_multiplex_active(True)

    credentials = await _in_scope(
        {
            "OPENAI_API_KEY": "profile-key",
            "OPENAI_BASE_URL": "",
        },
        lambda: auth.resolve_api_key_provider_credentials("openai-api"),
    )

    assert credentials["base_url"] == "https://api.openai.com/v1"


@pytest.mark.asyncio
async def test_api_key_provider_base_url_fails_closed_without_profile_scope(
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://foreign.example/v1")
    monkeypatch.setattr(
        auth,
        "_resolve_api_key_provider_secret",
        AsyncMock(return_value=("profile-key", "profile-scope")),
    )
    secret_scope.set_multiplex_active(True)

    with pytest.raises(
        secret_scope.UnscopedSecretError,
        match="OPENAI_BASE_URL",
    ):
        await auth.resolve_api_key_provider_credentials("openai-api")
    with pytest.raises(
        secret_scope.UnscopedSecretError,
        match="OPENAI_BASE_URL",
    ):
        await auth.get_api_key_provider_status("openai-api")


@pytest.mark.asyncio
async def test_external_process_config_isolates_concurrent_profiles(monkeypatch):
    monkeypatch.setenv("COPILOT_ACP_BASE_URL", "acp+tcp://foreign.example:1")
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", "foreign-copilot")
    monkeypatch.setenv("HERMES_COPILOT_ACP_ARGS", "--foreign")
    secret_scope.set_multiplex_active(True)

    async def resolve():
        return await auth.resolve_external_process_provider_credentials(
            "copilot-acp"
        )

    profile_a, profile_b = await asyncio.gather(
        _in_scope(
            {
                "COPILOT_ACP_BASE_URL": "acp+tcp://profile-a.example:1",
                "HERMES_COPILOT_ACP_COMMAND": "copilot-a",
                "COPILOT_CLI_PATH": "",
                "HERMES_COPILOT_ACP_ARGS": "--profile a",
            },
            resolve,
        ),
        _in_scope(
            {
                "COPILOT_ACP_BASE_URL": "acp+tcp://profile-b.example:2",
                "HERMES_COPILOT_ACP_COMMAND": "",
                "COPILOT_CLI_PATH": "copilot-b",
                "HERMES_COPILOT_ACP_ARGS": "--profile b",
            },
            resolve,
        ),
    )

    assert profile_a["base_url"] == "acp+tcp://profile-a.example:1"
    assert profile_a["command"] == "copilot-a"
    assert profile_a["args"] == ["--profile", "a"]
    assert profile_b["base_url"] == "acp+tcp://profile-b.example:2"
    assert profile_b["command"] == "copilot-b"
    assert profile_b["args"] == ["--profile", "b"]


@pytest.mark.asyncio
async def test_external_process_config_fails_closed_without_profile_scope(
    monkeypatch,
):
    monkeypatch.setenv("COPILOT_ACP_BASE_URL", "acp+tcp://foreign.example:1")
    secret_scope.set_multiplex_active(True)

    with pytest.raises(
        secret_scope.UnscopedSecretError,
        match="COPILOT_ACP_BASE_URL",
    ):
        await auth.resolve_external_process_provider_credentials("copilot-acp")


@pytest.mark.asyncio
async def test_external_process_explicit_empty_uses_upstream_defaults(
    monkeypatch,
):
    monkeypatch.setenv("COPILOT_ACP_BASE_URL", "acp+tcp://foreign.example:1")
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", "foreign-copilot")
    monkeypatch.setenv("HERMES_COPILOT_ACP_ARGS", "--foreign")
    secret_scope.set_multiplex_active(True)

    original_wrap = auth.aiofiles.os.wrap

    def fake_wrap(function):
        if function is auth.shutil.which:
            async def resolve_command(command):
                assert command == "copilot"
                return "/resolved/copilot"

            return resolve_command
        return original_wrap(function)

    monkeypatch.setattr(auth.aiofiles.os, "wrap", fake_wrap)
    result = await _in_scope(
        {
            "COPILOT_ACP_BASE_URL": "",
            "HERMES_COPILOT_ACP_COMMAND": "",
            "COPILOT_CLI_PATH": "",
            "HERMES_COPILOT_ACP_ARGS": "",
        },
        lambda: auth.resolve_external_process_provider_credentials("copilot-acp"),
    )

    assert result["base_url"] == "acp://copilot"
    assert result["command"] == "/resolved/copilot"
    assert result["args"] == ["--acp", "--stdio"]


@pytest.mark.asyncio
async def test_external_process_resolution_propagates_cancellation(monkeypatch):
    entered = asyncio.Event()
    original_wrap = auth.aiofiles.os.wrap

    def fake_wrap(function):
        if function is auth.shutil.which:
            async def wait_for_command(_command):
                entered.set()
                await asyncio.Event().wait()

            return wait_for_command
        return original_wrap(function)

    monkeypatch.setattr(auth.aiofiles.os, "wrap", fake_wrap)
    task = asyncio.create_task(
        auth.resolve_external_process_provider_credentials("copilot-acp")
    )
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
