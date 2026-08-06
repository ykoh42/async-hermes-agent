from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from agent import image_gen_registry
from agent.image_gen_provider import ImageGenProvider


@pytest.fixture(autouse=True)
def _reset_registry():
    image_gen_registry._reset_for_tests()
    yield
    image_gen_registry._reset_for_tests()


class _FakeCodexProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "codex"

    async def generate(self, prompt, aspect_ratio="landscape", **kwargs):
        return {
            "success": True,
            "image": "/tmp/codex-test.png",
            "model": "gpt-5.2-codex",
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "provider": "codex",
        }


class TestPluginDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_routes_to_codex_provider(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from agent import image_gen_registry as registry_module
        from hermes_cli import plugins as plugins_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("image_gen:\n  provider: codex\n")
        provider = _FakeCodexProvider()
        image_gen_registry.register_provider(provider)

        monkeypatch.setattr(
            image_generation_tool,
            "_read_configured_image_provider",
            AsyncMock(return_value="codex"),
        )
        monkeypatch.setattr(
            image_generation_tool,
            "_read_configured_image_model",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            plugins_module, "_ensure_plugins_discovered", AsyncMock(return_value=None)
        )
        monkeypatch.setattr(
            registry_module,
            "get_provider",
            lambda name: provider if name == "codex" else None,
        )

        dispatched = await image_generation_tool._dispatch_to_plugin_provider(
            "draw cat", "square"
        )
        payload = json.loads(dispatched)

        assert payload["success"] is True
        assert payload["provider"] == "codex"
        assert payload["image"] == "/tmp/codex-test.png"
        assert payload["aspect_ratio"] == "square"


    @pytest.mark.asyncio
    async def test_deepinfra_key_alone_does_not_select_image_backend(self, monkeypatch):
        """DeepInfra chat credentials do not imply consent to image billing."""
        from tools import image_generation_tool

        monkeypatch.setenv("DEEPINFRA_API_KEY", "«redacted:sk-…»")
        monkeypatch.delenv("FAL_KEY", raising=False)
        monkeypatch.setattr(
            image_generation_tool,
            "_read_configured_image_provider",
            AsyncMock(return_value=None),
        )
        assert await image_generation_tool._dispatch_to_plugin_provider("a cat", "square") is None

    @pytest.mark.asyncio
    async def test_requirements_ignore_unselected_paid_plugin(self, monkeypatch):
        from tools import image_generation_tool

        monkeypatch.setattr(
            image_generation_tool, "check_fal_api_key", AsyncMock(return_value=False)
        )
        monkeypatch.setattr(
            image_generation_tool,
            "_read_configured_image_provider",
            AsyncMock(return_value=None),
        )
        assert await image_generation_tool.check_image_generation_requirements() is False
