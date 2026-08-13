"""Async parity regressions for capture routing via auxiliary.vision."""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, patch

import pytest


pytestmark = pytest.mark.asyncio

_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAADUlEQVR4nG"
    "NgGAUgAAABCAABgukLHQAAAABJRU5ErkJggg=="
)


@pytest.fixture
def tmp_cache_dir(tmp_path):
    cache_dir = tmp_path / "cache_vision"
    cache_dir.mkdir()
    return cache_dir


def _make_capture(
    *,
    png_b64: str = _PNG_B64,
    mode: str = "som",
    elements=None,
    app: str = "Safari",
    window_title: str = "GitHub – Issue #24015",
    width: int = 1280,
    height: int = 800,
):
    from tools.computer_use.backend import CaptureResult, UIElement

    elements = list(
        elements
        or [
            UIElement(
                index=0,
                role="AXButton",
                label="Sign in",
                bounds=(10, 20, 80, 30),
            ),
            UIElement(
                index=1,
                role="AXTextField",
                label="username",
                bounds=(10, 60, 200, 24),
            ),
        ]
    )
    raw = base64.b64decode(png_b64, validate=False)
    return CaptureResult(
        mode=mode,
        width=width,
        height=height,
        png_b64=png_b64,
        elements=elements,
        app=app,
        window_title=window_title,
        png_bytes_len=len(raw),
    )


def _stub_aux_analysis(text: str):
    return json.dumps({"success": True, "analysis": text})


async def test_som_capture_returns_multimodal_envelope_when_native():
    from tools.computer_use import tool as cu_tool

    cap = _make_capture(png_b64=_PNG_B64, mode="som")
    with patch.object(
        cu_tool,
        "_should_route_through_aux_vision",
        AsyncMock(return_value=False),
    ):
        resp = await cu_tool._capture_response(cap)

    assert isinstance(resp, dict)
    assert resp.get("_multimodal") is True
    image_part = next(p for p in resp["content"] if p.get("type") == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert "vision_analysis" not in resp


async def test_ax_only_capture_returns_text_regardless_of_routing():
    from tools.computer_use import tool as cu_tool

    cap = _make_capture(mode="ax", png_b64="")
    routing = AsyncMock(return_value=True)
    with patch.object(cu_tool, "_should_route_through_aux_vision", routing):
        resp = await cu_tool._capture_response(cap)

    routing.assert_not_awaited()
    assert isinstance(resp, str)
    assert json.loads(resp)["mode"] == "ax"


async def test_som_capture_returns_text_with_vision_analysis(
    tmp_cache_dir, monkeypatch
):
    from tools.computer_use import tool as cu_tool
    import tools.vision_tools as vision_tools

    cap = _make_capture(mode="som")
    fake_vat = AsyncMock(
        return_value=_stub_aux_analysis(
            "A Safari window showing a GitHub issue page with a 'Sign in' "
            "button and a 'username' text field."
        )
    )
    monkeypatch.setattr(
        "hermes_constants.get_hermes_dir",
        AsyncMock(return_value=tmp_cache_dir),
    )
    monkeypatch.setattr(vision_tools, "vision_analyze_tool", fake_vat)
    with patch.object(
        cu_tool,
        "_should_route_through_aux_vision",
        AsyncMock(return_value=True),
    ):
        resp = await cu_tool._capture_response(cap)

    assert isinstance(resp, str)
    body = json.loads(resp)
    assert body["mode"] == "som"
    assert body["app"] == "Safari"
    assert "Sign in" in body["vision_analysis"]
    assert body["vision_analysis_routed_via"] == "auxiliary.vision"
    assert body["window_title"] == "GitHub – Issue #24015"
    assert len(body["elements"]) == 2
    args, _kwargs = fake_vat.await_args
    path_arg, prompt_arg = args[0], args[1]
    assert str(tmp_cache_dir) in path_arg
    assert "desktop application screenshot" in prompt_arg
    assert "Sign in" in prompt_arg


async def test_invalid_aux_response_degrades_to_text_payload(
    tmp_cache_dir, monkeypatch
):
    from tools.computer_use import tool as cu_tool
    import tools.vision_tools as vision_tools

    monkeypatch.setattr(
        "hermes_constants.get_hermes_dir",
        AsyncMock(return_value=tmp_cache_dir),
    )
    monkeypatch.setattr(
        vision_tools, "vision_analyze_tool", AsyncMock(return_value=1234)
    )
    with patch.object(
        cu_tool,
        "_should_route_through_aux_vision",
        AsyncMock(return_value=True),
    ):
        resp = await cu_tool._capture_response(_make_capture(mode="som"))

    assert isinstance(resp, str)
    assert json.loads(resp).get("vision_unavailable") is True


async def test_explicit_aux_vision_in_config_routes_to_aux(monkeypatch):
    from tools.computer_use import tool as cu_tool

    cfg = {
        "model": {"default": "tencent/hy3-preview", "provider": "openrouter"},
        "auxiliary": {
            "vision": {
                "provider": "openrouter",
                "model": "google/gemini-2.5-flash",
            }
        },
    }
    cu_tool._AUX_VISION_ROUTE_CACHE.clear()
    monkeypatch.setattr(
        "agent.auxiliary_client._read_main_provider",
        AsyncMock(return_value="openrouter"),
    )
    monkeypatch.setattr(
        "agent.auxiliary_client._read_main_model",
        AsyncMock(return_value="tencent/hy3-preview"),
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly", AsyncMock(return_value=cfg)
    )
    assert await cu_tool._should_route_through_aux_vision() is True


async def test_helper_decision_exception_is_swallowed(monkeypatch):
    from tools.computer_use import tool as cu_tool
    from tools.computer_use import vision_routing as vr_mod

    cu_tool._AUX_VISION_ROUTE_CACHE.clear()
    monkeypatch.setattr(
        "agent.auxiliary_client._read_main_provider",
        AsyncMock(return_value="openrouter"),
    )
    monkeypatch.setattr(
        "agent.auxiliary_client._read_main_model", AsyncMock(return_value="x")
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly", AsyncMock(return_value={})
    )
    monkeypatch.setattr(
        vr_mod,
        "should_route_capture_to_aux_vision",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("policy bug")),
    )
    assert await cu_tool._should_route_through_aux_vision() is False


async def test_non_vision_main_model_never_returns_image_url_when_routed(
    tmp_cache_dir, monkeypatch
):
    from tools.computer_use import tool as cu_tool
    import tools.vision_tools as vision_tools

    monkeypatch.setattr(
        "hermes_constants.get_hermes_dir",
        AsyncMock(return_value=tmp_cache_dir),
    )
    monkeypatch.setattr(
        vision_tools,
        "vision_analyze_tool",
        AsyncMock(
            return_value=_stub_aux_analysis(
                "Screenshot showing a GitHub.com window with a sign-in form."
            )
        ),
    )
    with patch.object(
        cu_tool,
        "_should_route_through_aux_vision",
        AsyncMock(return_value=True),
    ):
        resp = await cu_tool._capture_response(_make_capture(mode="som"))

    assert isinstance(resp, str)
    assert "data:image" not in resp
    assert "image_url" not in resp
