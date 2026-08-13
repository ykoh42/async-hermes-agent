"""Native-async behavior contracts for computer_use latency knobs."""

from unittest.mock import AsyncMock, patch

import pytest

from tools.computer_use import cua_backend
from tools.computer_use import tool as cu_tool


pytestmark = pytest.mark.asyncio


async def test_max_image_dimension_default():
    with patch("hermes_cli.config.load_config_readonly", AsyncMock(return_value={})):
        assert await cua_backend._computer_use_max_image_dimension() == 1456


async def test_capture_after_mode_default_som():
    with patch("hermes_cli.config.load_config_readonly", AsyncMock(return_value={})):
        assert await cu_tool._capture_after_mode() == "som"


async def test_aux_vision_route_caches_per_provider_model(monkeypatch):
    cu_tool._AUX_VISION_ROUTE_CACHE.clear()
    calls = {"n": 0}
    monkeypatch.setattr(
        "agent.auxiliary_client._read_main_provider",
        AsyncMock(return_value="openai"),
    )
    monkeypatch.setattr(
        "agent.auxiliary_client._read_main_model",
        AsyncMock(return_value="gpt-test"),
    )

    async def fake_load():
        calls["n"] += 1
        return {"auxiliary": {"vision": {}}}

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", fake_load)
    monkeypatch.setattr(
        "tools.computer_use.vision_routing.should_route_capture_to_aux_vision",
        lambda *args, **kwargs: True,
    )
    assert await cu_tool._should_route_through_aux_vision() is True
    assert await cu_tool._should_route_through_aux_vision() is True
    assert calls["n"] == 1
