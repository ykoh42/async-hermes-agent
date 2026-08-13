"""Native-async parity tests for TTS provider registration."""

from __future__ import annotations

import inspect
import logging

import pytest

from agent import tts_registry
from agent.tts_provider import (
    DEFAULT_OUTPUT_FORMAT,
    TTSProvider,
    VALID_OUTPUT_FORMATS,
    resolve_output_format,
)


class _FakeProvider(TTSProvider):
    def __init__(self, name: str = "fake"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def synthesize(self, text: str, output_path: str, **kwargs) -> str:
        del text, kwargs
        return output_path


@pytest.fixture(autouse=True)
def _reset_registry():
    tts_registry._reset_for_tests()
    yield
    tts_registry._reset_for_tests()


def test_provider_io_contract_is_native_async():
    for name in (
        "is_available",
        "list_voices",
        "list_models",
        "get_setup_schema",
        "default_model",
        "default_voice",
        "synthesize",
    ):
        assert inspect.iscoroutinefunction(getattr(TTSProvider, name))
    assert inspect.isasyncgenfunction(TTSProvider.stream)


def test_registration_and_case_insensitive_lookup():
    provider = _FakeProvider("Cartesia")
    tts_registry.register_provider(provider)
    assert tts_registry.get_provider(" cartesia ") is provider
    assert tts_registry.list_providers() == [provider]


@pytest.mark.parametrize("name", ["edge", "openai", "mistral", "piper"])
def test_builtin_names_cannot_be_shadowed(name, caplog):
    with caplog.at_level(logging.WARNING):
        tts_registry.register_provider(_FakeProvider(name))
    assert "shadows a built-in name" in caplog.text
    assert tts_registry.get_provider(name) is None


def test_sync_provider_override_is_rejected():
    class SyncProvider(_FakeProvider):
        def is_available(self):
            return True

    with pytest.raises(TypeError, match="is_available must be async"):
        tts_registry.register_provider(SyncProvider("sync"))


@pytest.mark.asyncio
async def test_default_metadata_and_stream_contract():
    class CatalogProvider(_FakeProvider):
        async def list_models(self):
            return [{"id": "sonic-2"}, {"id": "sonic-1"}]

        async def list_voices(self):
            return [{"id": "aria"}, {"id": "jasper"}]

    provider = CatalogProvider("cartesia")
    assert await provider.is_available() is True
    assert await provider.default_model() == "sonic-2"
    assert await provider.default_voice() == "aria"
    assert (await provider.get_setup_schema())["name"] == "Cartesia"
    with pytest.raises(NotImplementedError, match="does not implement streaming"):
        await anext(provider.stream("hello"))


@pytest.mark.parametrize("value", sorted(VALID_OUTPUT_FORMATS))
def test_valid_output_formats_pass_through(value):
    assert resolve_output_format(value) == value


def test_invalid_output_formats_use_upstream_default():
    assert resolve_output_format("aac") == DEFAULT_OUTPUT_FORMAT
    assert resolve_output_format(None) == DEFAULT_OUTPUT_FORMAT


def test_registry_and_dispatch_builtin_sets_match():
    from tools.tts_tool import BUILTIN_TTS_PROVIDERS

    assert tts_registry._BUILTIN_NAMES == BUILTIN_TTS_PROVIDERS
