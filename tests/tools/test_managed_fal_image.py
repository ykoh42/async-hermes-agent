from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tools import image_generation_tool as image_tool


@pytest.mark.asyncio
async def test_managed_fal_submit_preserves_gateway_request_and_result(
    monkeypatch,
):
    handle = SimpleNamespace(
        get=AsyncMock(return_value={"images": [{"url": "https://out.test/a.png"}]})
    )
    managed_client = SimpleNamespace(
        submit=AsyncMock(return_value=handle),
        close=AsyncMock(),
    )
    managed_gateway = SimpleNamespace(
        gateway_origin="https://fal-queue.gateway.test",
        nous_user_token="nous-token",
    )
    monkeypatch.setattr(image_tool, "_load_fal_client", lambda: object())
    monkeypatch.setattr(
        image_tool,
        "_resolve_managed_fal_gateway",
        AsyncMock(return_value=managed_gateway),
    )
    monkeypatch.setattr(
        image_tool,
        "_get_managed_fal_client",
        lambda managed_gateway: managed_client,
    )
    monkeypatch.setattr(image_tool.uuid, "uuid4", lambda: "call-123")

    result = await image_tool._submit_fal_request(
        "fal-ai/flux-2-pro",
        {"prompt": "test prompt", "num_images": 1},
    )

    assert result == {"images": [{"url": "https://out.test/a.png"}]}
    managed_client.submit.assert_awaited_once_with(
        "fal-ai/flux-2-pro",
        arguments={"prompt": "test prompt", "num_images": 1},
        headers={"x-idempotency-key": "call-123"},
    )
    handle.get.assert_awaited_once_with()
    managed_client.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_managed_fal_submit_translates_upstream_4xx(monkeypatch):
    error = RuntimeError("forbidden")
    error.response = SimpleNamespace(status_code=403)
    managed_client = SimpleNamespace(
        submit=AsyncMock(side_effect=error),
        close=AsyncMock(),
    )
    monkeypatch.setattr(image_tool, "_load_fal_client", lambda: object())
    monkeypatch.setattr(
        image_tool,
        "_resolve_managed_fal_gateway",
        AsyncMock(return_value=SimpleNamespace()),
    )
    monkeypatch.setattr(
        image_tool,
        "_get_managed_fal_client",
        lambda managed_gateway: managed_client,
    )
    guidance = AsyncMock(return_value="refresh account guidance")
    monkeypatch.setattr(
        image_tool,
        "nous_tool_gateway_unavailable_message",
        guidance,
    )

    with pytest.raises(ValueError) as exc_info:
        await image_tool._submit_fal_request("fal-ai/model", {"prompt": "x"})

    assert "HTTP 403" in str(exc_info.value)
    assert "refresh account guidance" in str(exc_info.value)
    assert exc_info.value.__cause__ is error
    guidance.assert_awaited_once_with(
        "managed FAL image generation",
        force_fresh=True,
    )
    managed_client.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_managed_fal_submit_preserves_non_http_exception(monkeypatch):
    error = ConnectionError("network down")
    managed_client = SimpleNamespace(
        submit=AsyncMock(side_effect=error),
        close=AsyncMock(),
    )
    monkeypatch.setattr(image_tool, "_load_fal_client", lambda: object())
    monkeypatch.setattr(
        image_tool,
        "_resolve_managed_fal_gateway",
        AsyncMock(return_value=SimpleNamespace()),
    )
    monkeypatch.setattr(
        image_tool,
        "_get_managed_fal_client",
        lambda managed_gateway: managed_client,
    )

    with pytest.raises(ConnectionError) as exc_info:
        await image_tool._submit_fal_request("fal-ai/model", {"prompt": "x"})
    assert exc_info.value is error
    managed_client.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_managed_fal_selection_preserves_direct_key_precedence(
    monkeypatch,
):
    monkeypatch.setattr(
        image_tool,
        "fal_key_is_configured",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        image_tool,
        "prefers_gateway",
        AsyncMock(return_value=False),
    )
    resolve = AsyncMock(return_value=SimpleNamespace())
    monkeypatch.setattr(image_tool, "resolve_managed_tool_gateway", resolve)

    assert await image_tool._resolve_managed_fal_gateway() is None
    resolve.assert_not_awaited()

    image_tool.prefers_gateway.return_value = True
    expected = SimpleNamespace()
    resolve.return_value = expected
    assert await image_tool._resolve_managed_fal_gateway() is expected
    resolve.assert_awaited_once_with("fal-queue")


@pytest.mark.asyncio
async def test_managed_fal_submit_closes_client_on_cancellation(monkeypatch):
    result_started = asyncio.Event()

    async def wait_for_result():
        result_started.set()
        await asyncio.Future()

    managed_client = SimpleNamespace(
        submit=AsyncMock(
            return_value=SimpleNamespace(get=wait_for_result)
        ),
        close=AsyncMock(),
    )
    monkeypatch.setattr(image_tool, "_load_fal_client", lambda: object())
    monkeypatch.setattr(
        image_tool,
        "_resolve_managed_fal_gateway",
        AsyncMock(return_value=SimpleNamespace()),
    )
    monkeypatch.setattr(
        image_tool,
        "_get_managed_fal_client",
        lambda managed_gateway: managed_client,
    )

    task = asyncio.create_task(
        image_tool._submit_fal_request("fal-ai/model", {"prompt": "x"})
    )
    await result_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    managed_client.close.assert_awaited_once_with()
