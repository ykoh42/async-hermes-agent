"""Tests for the vision-aware image preprocessing in run_agent.py.

Covers:

* ``_prepare_anthropic_messages_for_api`` — passes image parts through
  unchanged when the active model reports ``supports_vision=True`` (the
  adapter handles them natively), and falls back to text-description
  replacement when the model lacks vision.

* ``_prepare_messages_for_non_vision_model`` — the mirror method for the
  chat.completions / codex_responses paths. Same contract.
"""

from __future__ import annotations

import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from blockbuster import BlockBuster
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

from run_agent import AIAgent


def _make_agent() -> AIAgent:
    """Build a bare-bones AIAgent instance without running __init__.

    Avoids the heavy provider/credential setup for these pure-method tests.
    """
    agent = object.__new__(AIAgent)
    agent.provider = "anthropic"
    agent.model = "claude-sonnet-4"
    agent._anthropic_image_fallback_cache = {}
    return agent


IMG_PARTS_USER_MSG = {
    "role": "user",
    "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ],
}

PLAIN_USER_MSG = {"role": "user", "content": "hello, no images here"}


# ─── native text fallback ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_text_fallback_awaits_vision_tool_and_caches(monkeypatch):
    agent = _make_agent()
    analyze = AsyncMock(
        return_value=json.dumps({"success": True, "analysis": "a tabby cat"})
    )
    monkeypatch.setattr("tools.vision_tools.vision_analyze_tool", analyze)

    def reject_nested_loop(*_args, **_kwargs):
        raise AssertionError("vision fallback must not call asyncio.run")

    monkeypatch.setattr(asyncio, "run", reject_nested_loop)
    image_url = "https://example.test/cat.png"

    first = await agent._describe_image_for_anthropic_fallback(image_url, "user")
    second = await agent._describe_image_for_anthropic_fallback(image_url, "user")

    assert first == second
    assert first == (
        "[The user attached an image. Here's what it contains:\n"
        "a tabby cat]\n"
        "[If you need a closer look, use vision_analyze with image_url: "
        "https://example.test/cat.png]"
    )
    analyze.assert_awaited_once()
    assert analyze.await_args.kwargs["image_url"] == image_url


@pytest.mark.asyncio
async def test_text_fallback_propagates_cancellation(monkeypatch):
    agent = _make_agent()
    monkeypatch.setattr(
        "tools.vision_tools.vision_analyze_tool",
        AsyncMock(side_effect=asyncio.CancelledError),
    )

    with pytest.raises(asyncio.CancelledError):
        await agent._describe_image_for_anthropic_fallback(
            "data:image/png;base64,AAAA", "user"
        )


@pytest.mark.asyncio
async def test_text_fallback_runs_native_vision_path_without_blocking(
    monkeypatch, tmp_path
):
    agent = _make_agent()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = "native async description"
    response.choices = [choice]
    jpeg = base64.b64encode(b"\xff\xd8\xff" + b"\x00" * 32).decode("ascii")

    with (
        patch(
            "hermes_cli.config.load_config_readonly",
            new=AsyncMock(return_value={}),
        ),
        patch(
            "tools.vision_tools._image_to_base64_data_url",
            new=AsyncMock(return_value="data:image/jpeg;base64,abc"),
        ),
        patch(
            "tools.vision_tools.call_llm",
            new=AsyncMock(return_value=response),
        ),
    ):
        async with (
            no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.25),
            no_task_leaks(action=LeakAction.RAISE),
        ):
            blockbuster = BlockBuster()
            blockbuster.activate()
            try:
                note = await agent._describe_image_for_anthropic_fallback(
                    f"data:image/jpeg;base64,{jpeg}", "assistant"
                )
            finally:
                blockbuster.deactivate()

    assert note == (
        "[The assistant attached an image. Here's what it contains:\n"
        "native async description]"
    )


# ─── _prepare_anthropic_messages_for_api ─────────────────────────────────────


class TestPrepareAnthropicMessages:
    @pytest.mark.asyncio
    async def test_no_images_passes_through(self):
        agent = _make_agent()
        msgs = [PLAIN_USER_MSG]
        out = await agent._prepare_anthropic_messages_for_api(msgs)
        assert out is msgs  # unchanged reference

    @pytest.mark.asyncio
    async def test_vision_capable_passes_images_through(self):
        """The Anthropic adapter handles image_url/input_image natively."""
        agent = _make_agent()
        with patch.object(
            agent, "_model_supports_vision", new=AsyncMock(return_value=True)
        ):
            out = await agent._prepare_anthropic_messages_for_api([IMG_PARTS_USER_MSG])
        # Passes through unchanged — image_url parts still present.
        assert out[0]["content"][1]["type"] == "image_url"

    @pytest.mark.asyncio
    async def test_non_vision_replaces_images_with_text(self):
        agent = _make_agent()
        with patch.object(
            agent, "_model_supports_vision", new=AsyncMock(return_value=False)
        ), \
             patch.object(
                 agent,
                 "_describe_image_for_anthropic_fallback",
                 return_value="[Image description: a cat]",
             ):
            out = await agent._prepare_anthropic_messages_for_api([IMG_PARTS_USER_MSG])
        # Content collapsed to a string containing the description + user text.
        content = out[0]["content"]
        assert isinstance(content, str)
        assert "[Image description: a cat]" in content
        assert "What's in this image?" in content
        # No more image parts.
        assert "image_url" not in content


# ─── _prepare_messages_for_non_vision_model ──────────────────────────────────


class TestPrepareMessagesForNonVision:
    @pytest.mark.asyncio
    async def test_no_images_passes_through(self):
        agent = _make_agent()
        msgs = [PLAIN_USER_MSG]
        out = await agent._prepare_messages_for_non_vision_model(msgs)
        assert out is msgs

    @pytest.mark.asyncio
    async def test_vision_capable_passes_through(self):
        """For vision-capable models on chat.completions path, provider handles pixels."""
        agent = _make_agent()
        agent.provider = "openrouter"
        agent.model = "anthropic/claude-sonnet-4"
        with patch.object(
            agent, "_model_supports_vision", new=AsyncMock(return_value=True)
        ):
            out = await agent._prepare_messages_for_non_vision_model([IMG_PARTS_USER_MSG])
        assert out[0]["content"][1]["type"] == "image_url"

    @pytest.mark.asyncio
    async def test_non_vision_strips_images(self):
        agent = _make_agent()
        agent.provider = "openrouter"
        agent.model = "qwen/qwen3-235b-a22b"
        with patch.object(
            agent, "_model_supports_vision", new=AsyncMock(return_value=False)
        ), \
             patch.object(
                 agent,
                 "_describe_image_for_anthropic_fallback",
                 return_value="[Image description: a dog]",
             ):
            out = await agent._prepare_messages_for_non_vision_model([IMG_PARTS_USER_MSG])
        content = out[0]["content"]
        assert isinstance(content, str)
        assert "[Image description: a dog]" in content
        assert "image_url" not in content

    @pytest.mark.asyncio
    async def test_multiple_messages_with_mixed_content(self):
        agent = _make_agent()
        agent.model = "qwen/qwen3-235b"
        msgs = [
            {"role": "user", "content": "first turn"},
            {"role": "assistant", "content": "ack"},
            IMG_PARTS_USER_MSG,
        ]
        with patch.object(
            agent, "_model_supports_vision", new=AsyncMock(return_value=False)
        ), \
             patch.object(
                 agent,
                 "_describe_image_for_anthropic_fallback",
                 return_value="[Image: thing]",
             ):
            out = await agent._prepare_messages_for_non_vision_model(msgs)
        # First two messages unchanged (no images), third stripped.
        assert out[0]["content"] == "first turn"
        assert out[1]["content"] == "ack"
        assert isinstance(out[2]["content"], str)
        assert "[Image: thing]" in out[2]["content"]


# ─── _model_supports_vision ──────────────────────────────────────────────────


class TestModelSupportsVision:
    @pytest.mark.asyncio
    async def test_missing_provider_or_model_returns_false(self):
        agent = _make_agent()
        agent.provider = ""
        agent.model = "claude-sonnet-4"
        assert await agent._model_supports_vision() is False
        agent.provider = "anthropic"
        agent.model = ""
        assert await agent._model_supports_vision() is False

    @pytest.mark.asyncio
    async def test_uses_get_model_capabilities(self):
        agent = _make_agent()
        fake_caps = MagicMock()
        fake_caps.supports_vision = True
        with patch(
            "agent.models_dev.get_model_capabilities",
            new=AsyncMock(return_value=fake_caps),
        ):
            assert await agent._model_supports_vision() is True
        fake_caps.supports_vision = False
        with patch(
            "agent.models_dev.get_model_capabilities",
            new=AsyncMock(return_value=fake_caps),
        ):
            assert await agent._model_supports_vision() is False

    @pytest.mark.asyncio
    async def test_none_caps_returns_false(self):
        agent = _make_agent()
        with patch(
            "agent.models_dev.get_model_capabilities",
            new=AsyncMock(return_value=None),
        ):
            assert await agent._model_supports_vision() is False


    @pytest.mark.asyncio
    async def test_top_level_model_override_wins(self, tmp_path):
        agent = _make_agent()
        agent.provider = "custom"
        agent.model = "my-llava"
        config_path = tmp_path / "config.yaml"
        config_path.write_text("model:\n  supports_vision: true\n", encoding="utf-8")
        with patch("hermes_cli.config.get_config_path", return_value=config_path):
            assert await agent._model_supports_vision() is True


    @pytest.mark.asyncio
    async def test_named_custom_provider_resolved_via_config_provider(self, tmp_path):
        # Named custom providers get runtime self.provider rewritten to
        # "custom" while the config keeps the original name under
        # model.provider. The override must still resolve.
        agent = _make_agent()
        agent.provider = "custom"
        agent.model = "my-llava"
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "model:\n  provider: my-vllm\n  default: my-llava\n"
            "providers:\n  my-vllm:\n    models:\n      my-llava:\n"
            "        supports_vision: true\n",
            encoding="utf-8",
        )
        with patch("hermes_cli.config.get_config_path", return_value=config_path):
            assert await agent._model_supports_vision() is True
