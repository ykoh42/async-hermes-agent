"""Profile isolation for retained image/video provider environment settings."""

from __future__ import annotations

import asyncio
import importlib

import pytest

from agent import secret_scope


@pytest.fixture(autouse=True)
def _restore_secret_scope():
    previous_multiplex = secret_scope.is_multiplex_active()
    scope_token = secret_scope.set_secret_scope(None)
    try:
        yield
    finally:
        secret_scope.reset_secret_scope(scope_token)
        secret_scope.set_multiplex_active(previous_multiplex)


async def _resolve_media_models(scope: dict[str, str]) -> dict[str, object]:
    from plugins.image_gen import deepinfra, krea, openai, openrouter, xai
    from plugins.video_gen import fal

    codex = importlib.import_module("plugins.image_gen.openai-codex")
    provider = openrouter.OpenRouterCompatImageProvider(
        provider_name="openrouter",
        display_name="OpenRouter",
        runtime_name="openrouter",
        config_key="openrouter",
        model_env_var="OPENROUTER_IMAGE_MODEL",
        setup_schema={},
    )

    token = secret_scope.set_secret_scope(scope)
    try:
        openai_model, _ = await openai._resolve_model()
        codex_model, _ = await codex._resolve_model()
        xai_model, _ = await xai._resolve_model()
        krea_model, _ = await krea._resolve_model()
        fal_model, _ = await fal._resolve_family(None)
        return {
            "openai": openai_model,
            "codex": codex_model,
            "xai": xai_model,
            "deepinfra": deepinfra._resolve_model([], {}),
            "openrouter": await provider._resolve_model(),
            "krea": krea_model,
            "fal": fal_model,
        }
    finally:
        secret_scope.reset_secret_scope(token)


@pytest.mark.asyncio
async def test_concurrent_profiles_isolate_media_model_overrides(monkeypatch):
    process_values = {
        "OPENAI_IMAGE_MODEL": "gpt-image-2-medium",
        "XAI_IMAGE_MODEL": "grok-imagine-image",
        "DEEPINFRA_IMAGE_MODEL": "process/deepinfra",
        "OPENROUTER_IMAGE_MODEL": "process/openrouter",
        "KREA_IMAGE_MODEL": "krea-2-medium",
        "FAL_VIDEO_MODEL": "pixverse-v6",
    }
    for name, value in process_values.items():
        monkeypatch.setenv(name, value)

    secret_scope.set_multiplex_active(True)
    profile_a, profile_b = await asyncio.gather(
        _resolve_media_models(
            {
                "OPENAI_IMAGE_MODEL": "gpt-image-2-high",
                "XAI_IMAGE_MODEL": "grok-imagine-image-quality",
                "DEEPINFRA_IMAGE_MODEL": "profile-a/deepinfra",
                "OPENROUTER_IMAGE_MODEL": "profile-a/openrouter",
                "KREA_IMAGE_MODEL": "krea-2-large",
                "FAL_VIDEO_MODEL": "ltx-2.3",
            }
        ),
        _resolve_media_models(
            {
                "OPENAI_IMAGE_MODEL": "gpt-image-2-low",
                "XAI_IMAGE_MODEL": "grok-imagine-image",
                "DEEPINFRA_IMAGE_MODEL": "profile-b/deepinfra",
                "OPENROUTER_IMAGE_MODEL": "profile-b/openrouter",
                "KREA_IMAGE_MODEL": "krea-2-medium",
                "FAL_VIDEO_MODEL": "veo3.1",
            }
        ),
    )

    assert profile_a == {
        "openai": "gpt-image-2-high",
        "codex": "gpt-image-2-high",
        "xai": "grok-imagine-image-quality",
        "deepinfra": "profile-a/deepinfra",
        "openrouter": "profile-a/openrouter",
        "krea": "krea-2-large",
        "fal": "ltx-2.3",
    }
    assert profile_b == {
        "openai": "gpt-image-2-low",
        "codex": "gpt-image-2-low",
        "xai": "grok-imagine-image",
        "deepinfra": "profile-b/deepinfra",
        "openrouter": "profile-b/openrouter",
        "krea": "krea-2-medium",
        "fal": "veo3.1",
    }


@pytest.mark.asyncio
async def test_xai_video_fallback_credentials_are_profile_scoped(monkeypatch):
    from plugins.video_gen import xai as xai_video

    async def no_shared_credentials():
        return {}

    monkeypatch.setattr(
        "tools.xai_http.resolve_xai_http_credentials",
        no_shared_credentials,
    )
    monkeypatch.setenv("XAI_API_KEY", "process-key")
    monkeypatch.setenv("XAI_BASE_URL", "https://process.xai.test/v1")
    secret_scope.set_multiplex_active(True)

    async def resolve(api_key: str, base_url: str) -> tuple[str, str]:
        token = secret_scope.set_secret_scope(
            {"XAI_API_KEY": api_key, "XAI_BASE_URL": base_url}
        )
        try:
            return await xai_video._resolve_xai_credentials()
        finally:
            secret_scope.reset_secret_scope(token)

    profile_a, profile_b = await asyncio.gather(
        resolve("profile-a-key", "https://profile-a.xai.test/v1/"),
        resolve("profile-b-key", "https://profile-b.xai.test/v1"),
    )

    assert profile_a == ("profile-a-key", "https://profile-a.xai.test/v1")
    assert profile_b == ("profile-b-key", "https://profile-b.xai.test/v1")


@pytest.mark.asyncio
async def test_legacy_fal_image_model_override_is_profile_scoped(monkeypatch):
    from tools import image_generation_tool

    async def no_config():
        return {}

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", no_config)
    monkeypatch.setenv("FAL_IMAGE_MODEL", "process-model")
    secret_scope.set_multiplex_active(True)

    async def resolve(model: str) -> str:
        token = secret_scope.set_secret_scope({"FAL_IMAGE_MODEL": model})
        try:
            model_id, _ = await image_generation_tool._resolve_fal_model()
            return model_id
        finally:
            secret_scope.reset_secret_scope(token)

    model_a, model_b = await asyncio.gather(
        resolve("fal-ai/flux-2-pro"),
        resolve("fal-ai/z-image/turbo"),
    )

    assert model_a == "fal-ai/flux-2-pro"
    assert model_b == "fal-ai/z-image/turbo"


@pytest.mark.asyncio
async def test_media_environment_reads_fail_closed_without_profile_scope(
    monkeypatch,
):
    from plugins.image_gen import openai
    from plugins.video_gen import xai as xai_video

    async def no_shared_credentials():
        return {}

    monkeypatch.setattr(
        "tools.xai_http.resolve_xai_http_credentials",
        no_shared_credentials,
    )
    monkeypatch.setenv("OPENAI_IMAGE_MODEL", "gpt-image-2-high")
    monkeypatch.setenv("XAI_API_KEY", "process-key")
    secret_scope.set_multiplex_active(True)

    with pytest.raises(secret_scope.UnscopedSecretError, match="OPENAI_IMAGE_MODEL"):
        await openai._resolve_model()
    with pytest.raises(secret_scope.UnscopedSecretError, match="XAI_API_KEY"):
        await xai_video._resolve_xai_credentials()


@pytest.mark.asyncio
async def test_openai_image_client_closes_through_repeated_cancellation(
    monkeypatch,
):
    from plugins.image_gen import openai as openai_image

    request_started = asyncio.Event()
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    closed = asyncio.Event()

    class Images:
        async def generate(self, **_kwargs):
            request_started.set()
            await asyncio.Future()

    class Client:
        images = Images()

        async def close(self) -> None:
            close_started.set()
            await allow_close.wait()
            closed.set()

    async def create_client(*_args, **_kwargs):
        return Client()

    monkeypatch.setattr(openai_image, "_create_openai_sdk_client", create_client)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    secret_scope.set_multiplex_active(False)

    task = asyncio.create_task(
        openai_image.OpenAIImageGenProvider().generate("a lighthouse")
    )
    await request_started.wait()
    task.cancel()
    await close_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    try:
        assert task.done() is False
    finally:
        allow_close.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert closed.is_set()


def test_deepinfra_uses_common_owned_client_close_boundary():
    from agent.image_gen_provider import _close_owned_client
    from plugins.image_gen import deepinfra

    assert deepinfra._close_owned_client is _close_owned_client
