import inspect
import base64

import aiofiles
import pytest

from agent import (
    browser_registry,
    image_gen_registry,
    transcription_registry,
    tts_registry,
    video_gen_registry,
)
from agent.browser_provider import BrowserProvider
from agent.image_gen_provider import ImageGenProvider, save_b64_image
from agent.transcription_provider import TranscriptionProvider
from agent.tts_provider import TTSProvider
from agent.video_gen_provider import VideoGenProvider, save_b64_video
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


class _ImageProvider(ImageGenProvider):
    name = "test-image"

    async def generate(self, prompt, aspect_ratio="landscape", **kwargs):
        return {"success": True, "image": prompt}


class _VideoProvider(VideoGenProvider):
    name = "test-video"

    async def generate(self, prompt, **kwargs):
        return {"success": True, "video": prompt}


class _TTSProvider(TTSProvider):
    name = "test-tts"

    async def synthesize(self, text, output_path, **kwargs):
        return output_path


class _TranscriptionProvider(TranscriptionProvider):
    name = "test-stt"

    async def transcribe(self, file_path, **kwargs):
        return {"success": True, "transcript": file_path, "provider": self.name}


class _BrowserProvider(BrowserProvider):
    name = "test-browser"

    def is_available(self):
        return True

    async def create_session(self, task_id):
        return {"session_name": task_id, "bb_session_id": task_id, "cdp_url": ""}

    async def close_session(self, session_id):
        return True

    async def emergency_cleanup(self, session_id):
        return None


@pytest.fixture(autouse=True)
def _reset_registries():
    registries = (
        browser_registry,
        image_gen_registry,
        video_gen_registry,
        tts_registry,
        transcription_registry,
    )
    for registry in registries:
        registry._reset_for_tests()
    yield
    for registry in registries:
        registry._reset_for_tests()


def test_media_and_browser_provider_io_contracts_are_native_async():
    assert inspect.iscoroutinefunction(ImageGenProvider.generate)
    assert inspect.iscoroutinefunction(ImageGenProvider.is_available)
    assert inspect.iscoroutinefunction(ImageGenProvider.list_models)
    assert inspect.iscoroutinefunction(ImageGenProvider.default_model)
    assert inspect.iscoroutinefunction(ImageGenProvider.get_setup_schema)
    assert inspect.iscoroutinefunction(ImageGenProvider.capabilities)
    assert inspect.iscoroutinefunction(VideoGenProvider.generate)
    assert inspect.iscoroutinefunction(VideoGenProvider.is_available)
    assert inspect.iscoroutinefunction(VideoGenProvider.list_models)
    assert inspect.iscoroutinefunction(VideoGenProvider.default_model)
    assert inspect.iscoroutinefunction(VideoGenProvider.get_setup_schema)
    assert inspect.iscoroutinefunction(TTSProvider.synthesize)
    assert inspect.isasyncgenfunction(TTSProvider.stream)
    assert inspect.iscoroutinefunction(TranscriptionProvider.transcribe)
    assert inspect.iscoroutinefunction(BrowserProvider.create_session)
    assert inspect.iscoroutinefunction(BrowserProvider.close_session)
    assert inspect.iscoroutinefunction(BrowserProvider.emergency_cleanup)


def test_native_async_provider_instances_register_on_original_boundaries():
    providers = (
        (image_gen_registry, _ImageProvider()),
        (video_gen_registry, _VideoProvider()),
        (tts_registry, _TTSProvider()),
        (transcription_registry, _TranscriptionProvider()),
        (browser_registry, _BrowserProvider()),
    )

    for registry, provider in providers:
        registry.register_provider(provider)
        assert registry.get_provider(provider.name) is provider


def test_plugin_context_original_registration_methods_are_operational():
    context = PluginContext(
        PluginManifest(name="async-provider-test"),
        PluginManager(),
    )
    providers = (
        (context.register_image_gen_provider, image_gen_registry, _ImageProvider()),
        (context.register_video_gen_provider, video_gen_registry, _VideoProvider()),
        (context.register_tts_provider, tts_registry, _TTSProvider()),
        (
            context.register_transcription_provider,
            transcription_registry,
            _TranscriptionProvider(),
        ),
        (context.register_browser_provider, browser_registry, _BrowserProvider()),
    )

    for register, registry, provider in providers:
        register(provider)
        assert registry.get_provider(provider.name) is provider


def test_sync_override_is_rejected_instead_of_hidden_behind_a_bridge():
    class SyncImageProvider(_ImageProvider):
        name = "sync-image"

        def generate(self, prompt, aspect_ratio="landscape", **kwargs):
            return {"success": True, "image": prompt}

    with pytest.raises(TypeError, match="generate must be async"):
        image_gen_registry.register_provider(SyncImageProvider())

    class SyncMetadataProvider(_ImageProvider):
        name = "sync-image-metadata"

        def is_available(self):
            return True

    with pytest.raises(TypeError, match="is_available must be async"):
        image_gen_registry.register_provider(SyncMetadataProvider())


@pytest.mark.asyncio
async def test_media_cache_helpers_preserve_bytes_with_async_io(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    raw = b"native-async-media"
    encoded = base64.b64encode(raw).decode()

    image_path = await save_b64_image(encoded, prefix="test")
    video_path = await save_b64_video(encoded, prefix="test")

    async with aiofiles.open(image_path, "rb") as image_file:
        assert await image_file.read() == raw
    async with aiofiles.open(video_path, "rb") as video_file:
        assert await video_file.read() == raw
