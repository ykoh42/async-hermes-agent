"""Behavior parity tests for the xAI-specific video tool surface."""

from __future__ import annotations

import inspect
import json

import pytest

import tools.xai_video_tools as xai_video_tools
from tools.registry import registry


@pytest.mark.asyncio
async def test_xai_video_requirements_use_async_config_and_credentials(monkeypatch):
    calls: list[str] = []

    async def load_config_readonly():
        calls.append("config")
        return {"video_gen": {"provider": "xai"}}

    async def has_xai_video_credentials():
        calls.append("credentials")
        return True

    monkeypatch.setattr(xai_video_tools, "load_config_readonly", load_config_readonly)
    monkeypatch.setattr(
        xai_video_tools,
        "has_xai_video_credentials",
        has_xai_video_credentials,
    )

    assert await xai_video_tools._check_xai_video_requirements() is True
    assert calls == ["config", "credentials"]


@pytest.mark.asyncio
async def test_xai_video_requirements_short_circuit_without_xai_provider(monkeypatch):
    async def load_config_readonly():
        return {"video_gen": {"provider": "fal"}}

    async def unexpected_credentials_check():
        pytest.fail("credentials must not be checked for another provider")

    monkeypatch.setattr(xai_video_tools, "load_config_readonly", load_config_readonly)
    monkeypatch.setattr(
        xai_video_tools,
        "has_xai_video_credentials",
        unexpected_credentials_check,
    )

    assert await xai_video_tools._check_xai_video_requirements() is False


@pytest.mark.asyncio
async def test_xai_video_edit_normalizes_and_dispatches(monkeypatch):
    calls: list[dict[str, object]] = []

    async def configured():
        return True

    async def run_xai_video_edit(**kwargs):
        calls.append(kwargs)
        return {"success": True, "video": "https://files-cdn.x.ai/edit.mp4"}

    monkeypatch.setattr(xai_video_tools, "_configured_for_xai_video", configured)
    monkeypatch.setattr(xai_video_tools, "run_xai_video_edit", run_xai_video_edit)

    result = await xai_video_tools._handle_xai_video_edit({
        "prompt": "  make it snowy  ",
        "video_url": "  http://example.test/source.mp4  ",
        "model": "  grok-imagine-video  ",
    })

    assert result == json.dumps({
        "success": True,
        "video": "https://files-cdn.x.ai/edit.mp4",
    })
    assert calls == [{
        "prompt": "make it snowy",
        "video_url": "http://example.test/source.mp4",
        "model": "grok-imagine-video",
    }]


@pytest.mark.asyncio
async def test_xai_video_extend_normalizes_and_dispatches(monkeypatch):
    calls: list[dict[str, object]] = []

    async def configured():
        return True

    async def run_xai_video_extend(**kwargs):
        calls.append(kwargs)
        return {"success": True, "video": "https://files-cdn.x.ai/extend.mp4"}

    monkeypatch.setattr(xai_video_tools, "_configured_for_xai_video", configured)
    monkeypatch.setattr(xai_video_tools, "run_xai_video_extend", run_xai_video_extend)

    result = await xai_video_tools._handle_xai_video_extend({
        "prompt": "  continue into sunset  ",
        "video_url": "  https://example.test/source.mp4  ",
        "duration": "8",
        "model": " ",
    })

    assert result == json.dumps({
        "success": True,
        "video": "https://files-cdn.x.ai/extend.mp4",
    })
    assert calls == [{
        "prompt": "continue into sunset",
        "video_url": "https://example.test/source.mp4",
        "duration": 8,
        "model": None,
    }]


@pytest.mark.asyncio
async def test_xai_video_extend_rejects_bool_duration_without_rejecting_request(
    monkeypatch,
):
    async def configured():
        return True

    async def run_xai_video_extend(**kwargs):
        return kwargs

    monkeypatch.setattr(xai_video_tools, "_configured_for_xai_video", configured)
    monkeypatch.setattr(xai_video_tools, "run_xai_video_extend", run_xai_video_extend)

    result = await xai_video_tools._handle_xai_video_extend({
        "prompt": "continue",
        "video_url": "https://example.test/source.mp4",
        "duration": True,
    })

    assert json.loads(result)["duration"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler", "args", "message"),
    [
        (
            xai_video_tools._handle_xai_video_edit,
            {"video_url": "https://example.test/source.mp4"},
            "prompt is required for xAI video edit",
        ),
        (
            xai_video_tools._handle_xai_video_extend,
            {"prompt": "continue", "video_url": "file:///tmp/source.mp4"},
            (
                "video_url must be a public HTTPS MP4 URL "
                "(the `video`/`public_url` from a prior Imagine result)"
            ),
        ),
    ],
)
async def test_xai_video_validation_errors_match_upstream(handler, args, message):
    assert json.loads(await handler(args)) == {"error": message}


@pytest.mark.asyncio
async def test_xai_video_provider_not_configured_error_matches_upstream(monkeypatch):
    async def configured():
        return False

    monkeypatch.setattr(xai_video_tools, "_configured_for_xai_video", configured)

    result = await xai_video_tools._handle_xai_video_edit({
        "prompt": "edit",
        "video_url": "https://example.test/source.mp4",
    })

    assert json.loads(result) == {
        "success": False,
        "error": (
            "xAI video edit/extend tools require `video_gen.provider` to be "
            "configured as `xai` via `hermes tools` -> Video Generation."
        ),
        "error_type": "provider_not_configured",
        "provider": "xai",
    }


def test_xai_video_schemas_and_registry_metadata_match_upstream():
    assert xai_video_tools.XAI_VIDEO_EDIT_SCHEMA["name"] == "xai_video_edit"
    assert xai_video_tools.XAI_VIDEO_EDIT_SCHEMA["parameters"]["required"] == [
        "prompt",
        "video_url",
    ]
    assert xai_video_tools.XAI_VIDEO_EXTEND_SCHEMA["name"] == "xai_video_extend"
    assert xai_video_tools.XAI_VIDEO_EXTEND_SCHEMA["parameters"]["required"] == [
        "prompt",
        "video_url",
    ]

    for name, schema, handler in (
        (
            "xai_video_edit",
            xai_video_tools.XAI_VIDEO_EDIT_SCHEMA,
            xai_video_tools._handle_xai_video_edit,
        ),
        (
            "xai_video_extend",
            xai_video_tools.XAI_VIDEO_EXTEND_SCHEMA,
            xai_video_tools._handle_xai_video_extend,
        ),
    ):
        entry = registry.get_entry(name)
        assert entry is not None
        assert entry.toolset == "video_gen"
        assert entry.schema is schema
        assert entry.handler is handler
        assert entry.check_fn is xai_video_tools._check_xai_video_requirements
        assert entry.requires_env == []
        assert entry.is_async is True
        assert entry.emoji == "video"
        assert inspect.iscoroutinefunction(entry.handler)
