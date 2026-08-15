"""Native-async coordinate disclosure tests for vision resizing/cropping."""

from __future__ import annotations

import io
import re

import pytest
from PIL import Image


def _marker_png(width=3024, height=1964):
    image = Image.new("RGB", (width, height), (0, 0, 0))
    for x in range(2400, 2410):
        for y in range(1500, 1510):
            image.putpixel((x, y), (255, 0, 0))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _red_center(image):
    rgb = image.convert("RGB")
    pixels = rgb.load()
    points = [
        (x, y)
        for x in range(rgb.width)
        for y in range(rgb.height)
        if pixels[x, y][0] > 100 and pixels[x, y][1] < 100 and pixels[x, y][2] < 100
    ]
    assert points
    xs, ys = zip(*points)
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


@pytest.mark.asyncio
async def test_computer_capture_downscale_note_recovers_position():
    from tools.computer_use.tool import _shrink_capture_for_vision

    raw = _marker_png()
    shrunk_bytes, note = _shrink_capture_for_vision(raw, ".png")
    assert note is not None and "downscaled" in note
    assert "3024x1964" in note and "2.08" in note
    match = re.search(r"downscaled from (\d+)x(\d+) to (\d+)x(\d+)", note)
    assert match
    original_width, original_height, new_width, new_height = (
        int(value) for value in match.groups()
    )
    assert (original_width, original_height) == (3024, 1964)
    with Image.open(io.BytesIO(shrunk_bytes)) as image:
        assert image.size == (new_width, new_height)
        center_x, center_y = _red_center(image)
    recovered_x = center_x * original_width / new_width
    recovered_y = center_y * original_height / new_height
    assert abs(recovered_x - 2404.5) <= 2
    assert abs(recovered_y - 1504.5) <= 2


@pytest.mark.asyncio
async def test_computer_capture_no_scale_note_when_under_cap():
    from tools.computer_use.tool import _shrink_capture_for_vision

    image = Image.new("RGB", (800, 600), (10, 20, 30))
    output = io.BytesIO()
    image.save(output, format="PNG")
    result, note = _shrink_capture_for_vision(output.getvalue(), ".png")
    assert result == output.getvalue()
    assert note is None


@pytest.mark.asyncio
async def test_computer_capture_no_scale_note_on_bad_bytes():
    from tools.computer_use.tool import _shrink_capture_for_vision

    result, note = _shrink_capture_for_vision(b"not an image", ".png")
    assert result == b"not an image"
    assert note is None


def test_build_scale_note_none_without_transform():
    from tools.vision_tools import _build_scale_note

    assert _build_scale_note(None, None) is None
    assert _build_scale_note({}, {}) is None


def test_build_scale_note_reports_scale_and_crop():
    from tools.vision_tools import _build_scale_note

    note = _build_scale_note(
        {"orig_width": 3024, "orig_height": 1964, "new_width": 1512, "new_height": 982},
        {"x": 300, "y": 200, "width": 500, "height": 400},
    )
    assert note is not None
    assert "3024x1964" in note and "1512x982" in note
    assert "2.00" in note
    assert "(300, 200)" in note
    assert "crop" in note.lower()


@pytest.mark.asyncio
async def test_resize_image_for_vision_populates_scale_info(tmp_path):
    from tools.vision_tools import _resize_image_for_vision

    source = tmp_path / "source.png"
    Image.new("RGB", (3024, 1964), (5, 5, 5)).save(source, format="PNG")
    scale_info = {}
    data_url = await _resize_image_for_vision(
        source,
        max_base64_bytes=10_000,
        max_dimension=1456,
        scale_out=scale_info,
    )
    assert data_url.startswith("data:image/png;base64,")
    assert scale_info["orig_width"] == 3024
    assert scale_info["orig_height"] == 1964
    assert scale_info["new_width"] <= 1456
    assert scale_info["new_height"] < 1964


@pytest.mark.asyncio
async def test_native_vision_result_includes_scale_note():
    from tools.vision_tools import _build_native_vision_tool_result

    result = _build_native_vision_tool_result(
        "file:///tmp/image.png",
        "describe",
        "data:image/png;base64,AAAA",
        100,
        scale_note="Image downscaled from 3024x1964 to 1456x945 for vision.",
    )
    assert "scale_note" in result["meta"]
    assert "3024x1964" in result["content"][0]["text"]
