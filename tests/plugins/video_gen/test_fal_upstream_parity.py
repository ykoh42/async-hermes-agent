"""Native-async ports of the upstream FAL catalog/payload regression tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def test_minimax_h3_int_duration_and_resolution_alias():
    from plugins.video_gen.fal import FAL_FAMILIES, _build_payload

    meta = FAL_FAMILIES["minimax-h3"]
    payload = _build_payload(
        meta,
        prompt="x",
        image_url=None,
        duration=7,
        aspect_ratio="16:9",
        resolution="720p",
        negative_prompt=None,
        audio=True,
        seed=None,
    )
    assert payload["duration"] == 7
    assert isinstance(payload["duration"], int)
    assert payload["resolution"] == "768P"
    assert payload["aspect_ratio"] == "16:9"
    assert "generate_audio" not in payload

    high = _build_payload(
        meta,
        prompt="x",
        image_url=None,
        duration=5,
        aspect_ratio="16:9",
        resolution="1080p",
        negative_prompt=None,
        audio=None,
        seed=None,
    )
    assert high["resolution"] == "2K"


def test_image_drop_keys_strip_aspect_ratio_on_i2v():
    from plugins.video_gen.fal import FAL_FAMILIES, _build_payload

    for family_id in ("seedance-2.5", "minimax-h3", "grok-imagine-1.5"):
        meta = FAL_FAMILIES[family_id]
        image_payload = _build_payload(
            meta,
            prompt="x",
            image_url="https://example.com/i.png",
            duration=5,
            aspect_ratio="16:9",
            resolution="480p",
            negative_prompt=None,
            audio=None,
            seed=None,
        )
        assert "aspect_ratio" not in image_payload
        text_payload = _build_payload(
            meta,
            prompt="x",
            image_url=None,
            duration=5,
            aspect_ratio="16:9",
            resolution="480p",
            negative_prompt=None,
            audio=None,
            seed=None,
        )
        assert text_payload.get("aspect_ratio") == "16:9"


def test_seedance_25_string_duration_up_to_30():
    from plugins.video_gen.fal import FAL_FAMILIES, _build_payload

    payload = _build_payload(
        FAL_FAMILIES["seedance-2.5"],
        prompt="x",
        image_url=None,
        duration=30,
        aspect_ratio="1:1",
        resolution="480p",
        negative_prompt=None,
        audio=True,
        seed=None,
    )
    assert payload["duration"] == "30"
    assert payload["generate_audio"] is True


def test_new_fal_family_catalog_metadata():
    from plugins.video_gen.fal import FAL_FAMILIES

    assert FAL_FAMILIES["gemini-omni-flash"]["text_endpoint"] is None
    assert FAL_FAMILIES["gemini-omni-flash"]["image_endpoint"]
    for family_id, meta in FAL_FAMILIES.items():
        assert meta.get("display"), family_id
        assert meta.get("tier") in {"cheap", "premium"}, family_id
        assert meta.get("text_endpoint") or meta.get("image_endpoint"), family_id


def test_family_endpoint_normalization_and_duration_capabilities():
    from plugins.video_gen.fal import FAL_FAMILIES, FALVideoGenProvider, _normalize_family_key

    for family_id, meta in FAL_FAMILIES.items():
        for field in ("text_endpoint", "image_endpoint"):
            endpoint = meta.get(field)
            if endpoint:
                assert _normalize_family_key(endpoint) == family_id
    assert _normalize_family_key("bytedance/seedance-2.0/mini") == "seedance-2.0-mini"
    assert _normalize_family_key("minimax/h3") == "minimax-h3"
    assert _normalize_family_key("xai/grok-imagine-video/v1.5") == "grok-imagine-1.5"
    assert FALVideoGenProvider().capabilities()["max_duration"] >= 30


@pytest.mark.parametrize(
    ("family_id", "expected"),
    [
        ("seedance-2.0", "7"),
        ("seedance-2.0-mini", "7"),
        ("seedance-2.5", "7"),
        ("minimax-h3", 7),
        ("flux-3", 7),
        ("grok-imagine-1.5", 7),
        ("gemini-omni-flash", 7),
        ("pixverse-v6", "7"),
        ("veo3.1", "6s"),
    ],
)
def test_new_fal_duration_and_seed_contracts(family_id, expected):
    from plugins.video_gen.fal import FAL_FAMILIES, _build_payload

    payload = _build_payload(
        FAL_FAMILIES[family_id],
        prompt="x",
        image_url=None,
        duration=7,
        aspect_ratio="16:9",
        resolution="720p",
        negative_prompt=None,
        audio=None,
        seed=42,
    )
    assert payload["duration"] == expected
    assert type(payload["duration"]) is type(expected)
    if family_id in {
        "seedance-2.0",
        "seedance-2.0-mini",
        "seedance-2.5",
        "minimax-h3",
        "flux-3",
        "grok-imagine-1.5",
        "gemini-omni-flash",
    }:
        assert "seed" not in payload


@pytest.mark.asyncio
async def test_gemini_omni_flash_text_job_fails_without_submission(monkeypatch):
    from plugins.video_gen import fal as fal_plugin
    from plugins.video_gen.fal import FALVideoGenProvider

    monkeypatch.setattr(fal_plugin, "_check_fal_video_available", AsyncMock(return_value=True))
    monkeypatch.setattr(fal_plugin, "_load_fal_client", lambda: object())
    submit = AsyncMock()
    monkeypatch.setattr(fal_plugin, "_submit_fal_video_request", submit)
    result = await FALVideoGenProvider().generate(
        "animate this", model="gemini-omni-flash"
    )
    assert result["success"] is False
    assert result["error_type"] == "modality_unsupported"
    submit.assert_not_awaited()


@pytest.mark.asyncio
async def test_upscale_chains_seedvr2_and_reports_result(monkeypatch):
    from plugins.video_gen import fal as fal_plugin
    from plugins.video_gen.fal import FALVideoGenProvider, UPSCALER_ENDPOINT

    calls: list[tuple[str, dict]] = []

    def submit_impl(endpoint, arguments, **kwargs):
        calls.append((endpoint, arguments))
        if endpoint == UPSCALER_ENDPOINT:
            return {"video": {"url": "https://fake/upscaled.mp4"}}
        return {"video": {"url": "https://fake/native.mp4"}}

    monkeypatch.setattr(fal_plugin, "_check_fal_video_available", AsyncMock(return_value=True))
    monkeypatch.setattr(fal_plugin, "_load_fal_client", lambda: object())
    monkeypatch.setattr(
        fal_plugin,
        "_submit_fal_video_request",
        AsyncMock(side_effect=submit_impl),
    )
    monkeypatch.setattr(
        fal_plugin,
        "_resolve_managed_fal_video_gateway",
        AsyncMock(return_value=None),
    )

    result = await FALVideoGenProvider().generate(
        "a dog", model="pixverse-v6", upscale=True
    )
    assert result["success"] is True
    assert result["video"] == "https://fake/upscaled.mp4"
    assert result["upscaled"] is True
    assert result["upscale_factor"] == 2
    assert [endpoint for endpoint, _ in calls] == [
        "fal-ai/pixverse/v6/text-to-video",
        UPSCALER_ENDPOINT,
    ]
    assert calls[1][1]["upscale_mode"] == "factor"


@pytest.mark.asyncio
async def test_upscale_failure_falls_back_to_native(monkeypatch):
    from plugins.video_gen import fal as fal_plugin
    from plugins.video_gen.fal import FALVideoGenProvider

    monkeypatch.setattr(fal_plugin, "_check_fal_video_available", AsyncMock(return_value=True))
    monkeypatch.setattr(fal_plugin, "_load_fal_client", lambda: object())
    monkeypatch.setattr(
        fal_plugin,
        "_submit_fal_video_request",
        AsyncMock(return_value={"video": {"url": "https://fake/native.mp4"}}),
    )
    monkeypatch.setattr(fal_plugin, "_upscale_video", AsyncMock(return_value=None))

    result = await FALVideoGenProvider().generate(
        "a dog", model="pixverse-v6", upscale=True
    )
    assert result["success"] is True
    assert result["video"] == "https://fake/native.mp4"
    assert result["upscaled"] is False


@pytest.mark.asyncio
async def test_managed_upscale_requires_source_request_id(monkeypatch):
    from plugins.video_gen import fal as fal_plugin

    monkeypatch.setattr(
        fal_plugin,
        "_resolve_managed_fal_video_gateway",
        AsyncMock(return_value=SimpleNamespace()),
    )
    submit = AsyncMock()
    monkeypatch.setattr(fal_plugin, "_submit_fal_video_request", submit)

    assert await fal_plugin._upscale_video("https://fake/native.mp4") is None
    submit.assert_not_awaited()
