"""Native-async parity tests for ``vision_analyze(region=...)``."""

from __future__ import annotations

import base64
import io
import json

import pytest
from PIL import Image
from unittest.mock import AsyncMock


pytestmark = pytest.mark.asyncio


def _make_png(path, width=100, height=50):
    image = Image.new("RGB", (width, height), (200, 30, 30))
    image.save(path, format="PNG")
    return path


def _decoded_size(data_url: str):
    encoded = data_url.split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        return image.size


class TestCropImageRegion:
    async def test_crop_applied(self, tmp_path):
        from tools.vision_tools import _crop_image_region

        source = _make_png(tmp_path / "src.png", 100, 50)
        cropped_path, mime, error = await _crop_image_region(
            source, [10, 10, 60, 40]
        )
        assert error is None
        assert mime == "image/png"
        assert cropped_path is not None and cropped_path.exists()
        with Image.open(cropped_path) as image:
            assert image.size == (50, 30)

    async def test_out_of_bounds_clamped_to_image(self, tmp_path):
        from tools.vision_tools import _crop_image_region

        source = _make_png(tmp_path / "src.png", 100, 50)
        cropped_path, _mime, error = await _crop_image_region(
            source, [-10, -10, 200, 200]
        )
        assert error is None
        assert cropped_path is not None
        with Image.open(cropped_path) as image:
            assert image.size == (100, 50)

    async def test_zero_area_rejected_with_actual_dims_in_error(self, tmp_path):
        from tools.vision_tools import _crop_image_region

        source = _make_png(tmp_path / "src.png", 100, 50)
        cropped_path, _mime, error = await _crop_image_region(
            source, [200, 200, 300, 300]
        )
        assert cropped_path is None
        assert error is not None
        assert "100" in error and "50" in error

    async def test_inverted_coords_rejected_with_dims(self, tmp_path):
        from tools.vision_tools import _crop_image_region

        source = _make_png(tmp_path / "src.png", 100, 50)
        cropped_path, _mime, error = await _crop_image_region(
            source, [60, 40, 10, 10]
        )
        assert cropped_path is None
        assert error is not None and "100" in error and "50" in error

    async def test_malformed_region_rejected(self, tmp_path):
        from tools.vision_tools import _crop_image_region

        source = _make_png(tmp_path / "src.png", 100, 50)
        for bad in ([1, 2, 3], "10,10,60,40", [1, 2, 3, "x"], None):
            cropped_path, _mime, error = await _crop_image_region(source, bad)
            assert cropped_path is None
            assert error is not None


class TestNativePathRegion:
    async def test_region_crops_before_embed(self, tmp_path):
        from tools.vision_tools import _vision_analyze_native

        source = _make_png(tmp_path / "img.png", 100, 50)
        result = await _vision_analyze_native(
            str(source), "zoom", region=[10, 10, 60, 40]
        )
        assert isinstance(result, dict) and result.get("_multimodal") is True
        url = next(
            part["image_url"]["url"]
            for part in result["content"]
            if part.get("type") == "image_url"
        )
        assert _decoded_size(url) == (50, 30)

    async def test_no_region_behavior_unchanged(self, tmp_path):
        from tools.vision_tools import _vision_analyze_native

        source = _make_png(tmp_path / "img.png", 100, 50)
        result = await _vision_analyze_native(str(source), "full shot")
        assert isinstance(result, dict) and result.get("_multimodal") is True
        url = next(
            part["image_url"]["url"]
            for part in result["content"]
            if part.get("type") == "image_url"
        )
        assert _decoded_size(url) == (100, 50)

    async def test_zero_area_region_returns_error_with_dims(self, tmp_path):
        from tools.vision_tools import _vision_analyze_native

        source = _make_png(tmp_path / "img.png", 100, 50)
        result = await _vision_analyze_native(
            str(source), "zoom", region=[500, 500, 600, 600]
        )
        assert isinstance(result, str)
        payload = json.loads(result)
        assert payload.get("success") is False
        assert "100" in json.dumps(payload) and "50" in json.dumps(payload)

    async def test_crop_applied_before_downscale_gets_full_budget(self, tmp_path):
        from tools.vision_tools import _EMBED_MAX_DIMENSION, _vision_analyze_native

        source = tmp_path / "big.png"
        Image.new("RGB", (200, _EMBED_MAX_DIMENSION + 500), (0, 100, 0)).save(
            source, format="PNG"
        )
        result = await _vision_analyze_native(
            str(source), "zoom", region=[0, 0, 200, 300]
        )
        assert isinstance(result, dict) and result.get("_multimodal") is True
        url = next(
            part["image_url"]["url"]
            for part in result["content"]
            if part.get("type") == "image_url"
        )
        assert _decoded_size(url) == (200, 300)


class TestSchemaAndHandler:
    async def test_schema_declares_optional_region(self):
        from tools.vision_tools import VISION_ANALYZE_SCHEMA

        props = VISION_ANALYZE_SCHEMA["parameters"]["properties"]
        assert props["region"]["type"] == "array"
        assert "region" not in VISION_ANALYZE_SCHEMA["parameters"]["required"]
        assert "original" in props["region"]["description"].lower()

    async def test_handler_passes_region_to_native_path(self, tmp_path, monkeypatch):
        from tools import vision_tools
        from tools.vision_tools import _handle_vision_analyze

        source = _make_png(tmp_path / "img.png", 100, 50)
        seen = {}

        async def fake_native(image_url, question, task_id=None, region=None):
            seen["region"] = region
            return {"_multimodal": True, "content": []}

        monkeypatch.setattr(vision_tools, "_vision_analyze_native", fake_native)
        monkeypatch.setattr(
            vision_tools, "_should_use_native_vision_fast_path", AsyncMock(return_value=True)
        )
        await _handle_vision_analyze(
            {"image_url": str(source), "question": "q", "region": [1, 2, 30, 40]}
        )
        assert seen["region"] == [1, 2, 30, 40]
