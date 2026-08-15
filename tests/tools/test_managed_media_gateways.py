"""Native-async ports of upstream managed media gateway tests.

The upstream transcription helper is outside this distribution's retained
surface; its STT contract is covered by the retained provider boundary tests.
FAL image/video and OpenAI audio remain retained and are exercised here.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_managed_fal_submit_uses_gateway_origin_and_nous_token(monkeypatch):
    from tools import image_generation_tool as image_tool

    handle = SimpleNamespace(
        get=AsyncMock(return_value={"images": [{"url": "https://out.test/a.png"}]}),
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
        lambda gateway: managed_client,
    )
    monkeypatch.setattr(image_tool.uuid, "uuid4", lambda: "fal-submit-123")

    result = await image_tool._submit_fal_request(
        "fal-ai/flux-2-pro",
        {"prompt": "test prompt", "num_images": 1},
    )

    assert result == {"images": [{"url": "https://out.test/a.png"}]}
    managed_client.submit.assert_awaited_once_with(
        "fal-ai/flux-2-pro",
        arguments={"prompt": "test prompt", "num_images": 1},
        headers={"x-idempotency-key": "fal-submit-123"},
    )
    handle.get.assert_awaited_once_with()
    managed_client.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_openai_tts_uses_managed_audio_gateway_when_direct_key_absent(
    monkeypatch,
    tmp_path,
):
    from tools import tts_tool

    captured: dict[str, object] = {}

    class Response:
        async def __aenter__(self):
            captured["response_entered"] = True
            return self

        async def __aexit__(self, *args):
            captured["response_closed"] = True

        async def stream_to_file(self, output_path):
            captured["stream_to_file"] = output_path

    class Speech:
        class Streaming:
            def create(self, **kwargs):
                captured["speech_kwargs"] = kwargs
                return Response()

        with_streaming_response = Streaming()

    class Client:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.audio = SimpleNamespace(speech=Speech())

        async def close(self):
            captured["close_calls"] = int(captured.get("close_calls", 0)) + 1

    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: Client)
    monkeypatch.setattr(
        tts_tool,
        "_resolve_openai_audio_client_config",
        AsyncMock(
            return_value=(
                "nous-token",
                "https://openai-audio-gateway.nousresearch.com/v1",
                True,
            )
        ),
    )
    monkeypatch.setattr(tts_tool.uuid, "uuid4", lambda: "tts-call-123")

    output_path = tmp_path / "speech.mp3"
    returned = await tts_tool._generate_openai_tts("hello world", str(output_path), {})

    assert returned == str(output_path)
    assert captured["client_kwargs"] == {
        "api_key": "nous-token",
        "base_url": "https://openai-audio-gateway.nousresearch.com/v1",
    }
    assert captured["speech_kwargs"]["model"] == "gpt-4o-mini-tts"
    assert captured["speech_kwargs"]["extra_headers"] == {
        "x-idempotency-key": "tts-call-123"
    }
    assert captured["stream_to_file"] == str(output_path)
    assert captured["close_calls"] == 1


def test_video_gen_happy_horse_uses_alibaba_namespace():
    from plugins.video_gen.fal import FAL_FAMILIES

    family = FAL_FAMILIES["happy-horse"]
    assert family["text_endpoint"] == "alibaba/happy-horse/text-to-video"
    assert family["image_endpoint"] == "alibaba/happy-horse/image-to-video"
