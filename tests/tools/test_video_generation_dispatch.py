"""Tests for the unified ``video_generate`` tool dispatch surface."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from agent import video_gen_registry
from agent.video_gen_provider import VideoGenProvider


@pytest.fixture(autouse=True)
def _reset_registry():
    video_gen_registry._reset_for_tests()
    yield
    video_gen_registry._reset_for_tests()


class _RecordingProvider(VideoGenProvider):
    """Captures the kwargs the tool layer hands it."""

    def __init__(self, name: str = "fake"):
        self._name = name
        self.last_kwargs: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    async def list_models(self) -> List[Dict[str, Any]]:
        return [{"id": "model-a"}]

    async def default_model(self) -> Optional[str]:
        return "model-a"

    def capabilities(self) -> Dict[str, Any]:
        return {"modalities": ["text", "image"]}

    async def generate(self, prompt, **kwargs):
        self.last_kwargs = {"prompt": prompt, **kwargs}
        modality = "image" if kwargs.get("image_url") else "text"
        return {
            "success": True,
            "video": "https://example.com/v.mp4",
            "model": kwargs.get("model") or "model-a",
            "prompt": prompt,
            "modality": modality,
            "aspect_ratio": kwargs.get("aspect_ratio", ""),
            "duration": kwargs.get("duration") or 0,
            "provider": self._name,
        }


class _RaisingProvider(VideoGenProvider):
    @property
    def name(self) -> str:
        return "raises"

    async def generate(self, prompt, **kwargs):
        raise RuntimeError("boom")


class TestUnifiedDispatch:
    async def _run(
        self, args: Dict[str, Any], *, configured: Optional[str] = None
    ) -> Dict[str, Any]:
        from tools import video_generation_tool
        import hermes_cli.plugins as plugins_module

        saved = video_generation_tool._read_configured_video_provider
        async def configured_provider():
            return configured

        video_generation_tool._read_configured_video_provider = configured_provider  # type: ignore
        saved_discover = plugins_module._ensure_plugins_discovered
        async def discover(*_a, **_k):
            return None

        plugins_module._ensure_plugins_discovered = discover  # type: ignore
        try:
            raw = await video_generation_tool._handle_video_generate(args)
        finally:
            video_generation_tool._read_configured_video_provider = saved  # type: ignore
            plugins_module._ensure_plugins_discovered = saved_discover  # type: ignore
        return json.loads(raw)

    @pytest.mark.asyncio
    async def test_no_provider_returns_clear_error(self):
        result = await self._run({"prompt": "a dog"})
        assert result["success"] is False
        assert result["error_type"] == "no_provider_configured"

    @pytest.mark.asyncio
    async def test_unknown_provider_returns_clear_error(self):
        result = await self._run({"prompt": "a dog"}, configured="ghost")
        assert result["success"] is False
        assert result["error_type"] == "provider_not_registered"


    def test_edit_extend_fields_not_in_schema(self):
        from tools.video_generation_tool import VIDEO_GENERATE_SCHEMA
        props = VIDEO_GENERATE_SCHEMA["parameters"]["properties"]
        assert "operation" not in props
        assert "video_url" not in props
