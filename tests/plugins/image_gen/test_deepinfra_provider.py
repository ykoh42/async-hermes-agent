"""Tests for the bundled DeepInfra image_gen plugin.

Invariants only — no snapshots of specific model ids. Most surface-level
contracts (network-failure → empty list, tag filtering, no-model error)
are covered by the shared tag-filter test in
``tests/hermes_cli/test_api_key_providers.py``; these two tests pin the
plugin-specific bits that wrapper doesn't reach.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pyleak import no_task_leaks
from pyleak.eventloop import LeakAction

import plugins.image_gen.deepinfra as deepinfra_plugin


# 1×1 transparent PNG — valid bytes for save_b64_image()
_PNG_HEX = (
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


def _b64_png() -> str:
    import base64

    return base64.b64encode(bytes.fromhex(_PNG_HEX)).decode()


@pytest.fixture(autouse=True)
def _isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_cli.models as _models_mod
    monkeypatch.setattr(_models_mod, "_deepinfra_catalog_cache", {})
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-key")
    yield


@pytest.mark.asyncio
async def test_list_models_filters_by_image_gen_tag(monkeypatch):
    """Plugin-side wiring: list_models() returns only ``image-gen``-tagged
    catalog entries and surfaces pricing + default dims when present."""
    import hermes_cli.models as models

    async def fetch_by_tag(tag, **_kwargs):
        assert tag == "image-gen"
        return [{"id": "vendor/img", "metadata": {
            "tags": ["image-gen"],
            "pricing": {"per_image_unit": 0.005},
            "default_width": 1024,
        }}]

    monkeypatch.setattr(models, "_fetch_deepinfra_models_by_tag", fetch_by_tag)
    rows = await deepinfra_plugin.DeepInfraImageGenProvider().list_models()
    ids = {row["id"] for row in rows}
    assert ids == {"vendor/img"}
    img = next(row for row in rows if row["id"] == "vendor/img")
    assert "price" in img and img["default_width"] == 1024


@pytest.mark.asyncio
async def test_generate_calls_openai_sdk_with_deepinfra_base_url(monkeypatch):
    """Happy path: pinned model → openai SDK called with DeepInfra
    base_url + Bearer key → b64 saved to cache."""
    monkeypatch.setenv("DEEPINFRA_IMAGE_MODEL", "vendor/test-img")
    captured: dict = {}

    class _FakeImages:
        async def generate(self, **kwargs):
            captured["kwargs"] = kwargs
            return SimpleNamespace(data=[SimpleNamespace(b64_json=_b64_png(), url=None)])

    class _FakeClient:
        def __init__(self, api_key=None, base_url=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            self.images = _FakeImages()

        async def close(self):
            pass

    fake_openai = MagicMock()
    fake_openai.AsyncOpenAI = _FakeClient

    async def create_client(client_class, **kwargs):
        return client_class(**kwargs)

    with (
        patch.dict("sys.modules", {"openai": fake_openai}),
        patch(
            "agent.ssl_verify._create_openai_sdk_client",
            side_effect=create_client,
        ),
    ):
        result = await deepinfra_plugin.DeepInfraImageGenProvider().generate(
            prompt="a cat", aspect_ratio="square",
        )

    assert result["success"] is True
    assert "deepinfra" in captured["base_url"]
    assert captured["api_key"] == "test-key"
    assert captured["kwargs"]["model"] == "vendor/test-img"


@pytest.mark.asyncio
async def test_capabilities_advertise_text_to_image_only():
    assert await deepinfra_plugin.DeepInfraImageGenProvider().capabilities() == {
        "modalities": ["text"],
        "max_reference_images": 0,
    }


@pytest.mark.asyncio
async def test_repeated_cancellation_closes_client_and_preserves_first_error(
    monkeypatch,
):
    request_started = asyncio.Event()
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    close_finished = asyncio.Event()

    class _Images:
        async def generate(self, **_kwargs):
            request_started.set()
            await asyncio.Event().wait()

    class _Client:
        images = _Images()

        async def close(self):
            close_started.set()
            await allow_close.wait()
            close_finished.set()

    async def _create_client(*_args, **_kwargs):
        return _Client()

    monkeypatch.setenv("DEEPINFRA_IMAGE_MODEL", "vendor/test-img")
    monkeypatch.setattr(deepinfra_plugin, "_live_models", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        deepinfra_plugin,
        "_load_deepinfra_image_config",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        "agent.ssl_verify._create_openai_sdk_client",
        _create_client,
    )

    async with no_task_leaks(action=LeakAction.RAISE):
        task = asyncio.create_task(
            deepinfra_plugin.DeepInfraImageGenProvider().generate("a cat")
        )
        await asyncio.wait_for(request_started.wait(), timeout=1.0)
        task.cancel("original-request-cancellation")
        await asyncio.wait_for(close_started.wait(), timeout=1.0)
        task.cancel("second-cleanup-cancellation")
        await asyncio.sleep(0)
        assert task.done() is False
        allow_close.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await task

    assert raised.value.args == ("original-request-cancellation",)
    assert close_finished.is_set()


@pytest.mark.asyncio
async def test_client_initialization_failure_propagates_without_leaking_tasks(
    monkeypatch,
):
    import openai

    initialization_error = RuntimeError("client initialization failed")
    close_finished = asyncio.Event()

    class _HttpClient:
        async def aclose(self):
            close_finished.set()

    class _FailingClient:
        def __init__(self, **_kwargs):
            raise initialization_error

    create_http_client = AsyncMock(return_value=_HttpClient())
    monkeypatch.setattr(openai, "AsyncOpenAI", _FailingClient)
    monkeypatch.setenv("DEEPINFRA_IMAGE_MODEL", "vendor/test-img")
    monkeypatch.setattr(deepinfra_plugin, "_live_models", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        deepinfra_plugin,
        "_load_deepinfra_image_config",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        "agent.ssl_verify._create_httpx_client",
        create_http_client,
    )

    async with no_task_leaks(action=LeakAction.RAISE):
        with pytest.raises(RuntimeError) as raised:
            await deepinfra_plugin.DeepInfraImageGenProvider().generate("a cat")

    assert raised.value is initialization_error
    create_http_client.assert_awaited_once()
    assert close_finished.is_set()


@pytest.mark.asyncio
async def test_api_error_keeps_upstream_response_shape_after_client_close(monkeypatch):
    close = AsyncMock()

    class _Images:
        async def generate(self, **_kwargs):
            raise RuntimeError("provider failed")

    client = SimpleNamespace(images=_Images(), close=close)
    monkeypatch.setenv("DEEPINFRA_IMAGE_MODEL", "vendor/test-img")
    monkeypatch.setattr(deepinfra_plugin, "_live_models", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        deepinfra_plugin,
        "_load_deepinfra_image_config",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        "agent.ssl_verify._create_openai_sdk_client",
        AsyncMock(return_value=client),
    )

    result = await deepinfra_plugin.DeepInfraImageGenProvider().generate("a cat")

    assert result == {
        "success": False,
        "image": None,
        "error": "DeepInfra image generation failed: provider failed",
        "error_type": "api_error",
        "model": "vendor/test-img",
        "prompt": "a cat",
        "aspect_ratio": "landscape",
        "provider": "deepinfra",
    }
    close.assert_awaited_once()
