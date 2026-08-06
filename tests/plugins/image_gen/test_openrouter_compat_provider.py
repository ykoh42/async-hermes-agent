#!/usr/bin/env python3
"""Tests for the OpenRouter-compatible image gen provider (OpenRouter + Nous)."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


pytestmark = pytest.mark.asyncio

_RUNTIME = "hermes_cli.runtime_provider.resolve_runtime_provider"
_PNG_DATA_URI = "data:image/png;base64,dGVzdC1pbWFnZS1kYXRh"  # "test-image-data"


def _runtime_ok(**over):
    base = {
        "provider": "openrouter",
        "api_mode": "chat_completions",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "sk-or-test",
        "source": "env",
    }
    base.update(over)
    return base


def _mock_chat_response(images):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "images": [
                        {"type": "image_url", "image_url": {"url": u}} for u in images
                    ],
                }
            }
        ]
    }
    return resp


def _patched_http_client(*, response=None, side_effect=None):
    post = AsyncMock(return_value=response, side_effect=side_effect)
    client = MagicMock()
    client.post = post
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=client)
    context.__aexit__ = AsyncMock(return_value=None)
    return patch("httpx.AsyncClient", return_value=context), post


def _openrouter():
    from plugins.image_gen.openrouter import OpenRouterCompatImageProvider

    return OpenRouterCompatImageProvider(
        provider_name="openrouter",
        display_name="OpenRouter",
        runtime_name="openrouter",
        config_key="openrouter",
        model_env_var="OPENROUTER_IMAGE_MODEL",
        setup_schema={"name": "OpenRouter (image)", "badge": "paid", "env_vars": []},
    )


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class TestProviderClass:
    async def test_names(self):
        from plugins.image_gen.openrouter import _build_providers

        names = {p.name for p in _build_providers()}
        assert names == {"openrouter", "nous"}

    async def test_display_names(self):
        from plugins.image_gen.openrouter import _build_providers

        by_name = {p.name: p for p in _build_providers()}
        assert by_name["openrouter"].display_name == "OpenRouter"
        assert by_name["nous"].display_name == "Nous Portal"

    async def test_capabilities_support_image_input(self):
        caps = await _openrouter().capabilities()
        assert "image" in caps["modalities"]
        assert caps["max_reference_images"] >= 1

    async def test_is_available_with_key(self):
        with patch(_RUNTIME, new=AsyncMock(return_value=_runtime_ok())):
            assert await _openrouter().is_available() is True


    async def test_default_model(self):
        from plugins.image_gen.openrouter import DEFAULT_MODEL

        with patch(
            "plugins.image_gen.openrouter._load_image_gen_config",
            new=AsyncMock(return_value={}),
        ):
            assert await _openrouter().default_model() == DEFAULT_MODEL
            # Default must be an image-output model id (provider/model form).
            assert "/" in DEFAULT_MODEL and "image" in DEFAULT_MODEL


    async def test_model_env_override(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_IMAGE_MODEL", "black-forest-labs/flux.2-pro")
        assert await _openrouter()._resolve_model() == "black-forest-labs/flux.2-pro"
        assert await _openrouter()._resolve_model_chain() == ["black-forest-labs/flux.2-pro"]


    async def test_nous_honors_top_level_model(self):
        from plugins.image_gen.openrouter import _build_providers

        cfg = {"model": "openai/gpt-image-2"}
        nous = {p.name: p for p in _build_providers()}["nous"]
        with patch(
            "plugins.image_gen.openrouter._load_image_gen_config",
            new=AsyncMock(return_value=cfg),
        ):
            assert await nous._resolve_model_chain() == ["openai/gpt-image-2"]

    async def test_explicit_model_kwarg_wins_over_config(self):
        cfg = {"model": "openai/gpt-image-2"}
        with patch(
            "plugins.image_gen.openrouter._load_image_gen_config",
            new=AsyncMock(return_value=cfg),
        ):
            assert await _openrouter()._resolve_model_chain("google/gemini-3-pro-image") == [
                "google/gemini-3-pro-image"
            ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    async def test_to_image_url_part_passthrough_url(self):
        from plugins.image_gen.openrouter import _to_image_url_part

        assert await _to_image_url_part("https://x/y.png") == "https://x/y.png"
        assert await _to_image_url_part("data:image/png;base64,AAAA") == "data:image/png;base64,AAAA"


    async def test_to_image_url_part_blocks_credential_store(self, tmp_path, monkeypatch):
        from plugins.image_gen.openrouter import _to_image_url_part

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        auth_json = hermes_home / "auth.json"
        auth_json.write_text('{"api_key":"sk-secret"}', encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        with pytest.raises(ValueError, match="credential store"):
            await _to_image_url_part(str(auth_json))


    async def test_extract_images(self):
        from plugins.image_gen.openrouter import _extract_images

        payload = {
            "choices": [
                {"message": {"images": [{"image_url": {"url": "data:image/png;base64,AA"}}]}}
            ]
        }
        assert _extract_images(payload) == ["data:image/png;base64,AA"]


    async def test_access_error_hint_for_gated_openai_model(self):
        from plugins.image_gen.openrouter import _FALLBACK_MODEL, _access_error_hint

        hint = _access_error_hint(
            "OpenRouter", "openai/gpt-5.4-image-2", "OPENROUTER_IMAGE_MODEL", 404, "No endpoints found"
        )
        assert hint is not None
        assert "openai/gpt-5.4-image-2" in hint
        assert "OPENROUTER_IMAGE_MODEL" in hint
        assert _FALLBACK_MODEL in hint
        # Stays a single line under the humanizer's 200-char truncation.
        assert "\n" not in hint and len(hint) <= 200


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


class TestGenerate:
    async def test_missing_credentials(self):
        with patch(_RUNTIME, new=AsyncMock(return_value=_runtime_ok(api_key=""))):
            result = await _openrouter().generate(prompt="a pet")
        assert result["success"] is False
        assert result["error_type"] == "missing_api_key"

    async def test_success_data_uri(self):
        http_patch, _ = _patched_http_client(response=_mock_chat_response([_PNG_DATA_URI]))
        with patch(_RUNTIME, new=AsyncMock(return_value=_runtime_ok())), \
             http_patch, \
             patch(
                 "plugins.image_gen.openrouter.save_b64_image",
                 new=AsyncMock(return_value=Path("/tmp/openrouter_gen.png")),
             ) as mock_save:
            result = await _openrouter().generate(prompt="a pet")

        assert result["success"] is True
        assert result["image"] == "/tmp/openrouter_gen.png"
        assert result["provider"] == "openrouter"
        mock_save.assert_called_once()


    async def test_empty_response(self):
        http_patch, _ = _patched_http_client(response=_mock_chat_response([]))
        with patch(_RUNTIME, new=AsyncMock(return_value=_runtime_ok())), http_patch:
            result = await _openrouter().generate(prompt="a pet")
        assert result["success"] is False
        assert result["error_type"] == "empty_response"

    async def test_payload_shape_and_references(self, tmp_path):
        """Wire payload must carry image modalities, aspect_ratio, and the
        reference image inlined as a data URI (this is what makes pet rows
        stay on-model)."""
        ref = tmp_path / "base.png"
        ref.write_bytes(b"\x89PNG\r\n")

        http_patch, mock_post = _patched_http_client(
            response=_mock_chat_response([_PNG_DATA_URI])
        )
        with patch(_RUNTIME, new=AsyncMock(return_value=_runtime_ok())), \
             http_patch, \
             patch(
                 "plugins.image_gen.openrouter.save_b64_image",
                 new=AsyncMock(return_value=Path("/tmp/x.png")),
             ):
            await _openrouter().generate(
                prompt="a pet", aspect_ratio="square", reference_images=[str(ref)]
            )

        payload = mock_post.call_args.kwargs["json"]
        assert payload["modalities"] == ["image", "text"]
        assert payload["image_config"]["aspect_ratio"] == "1:1"
        content = payload["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": "a pet"}
        image_parts = [c for c in content if c["type"] == "image_url"]
        assert len(image_parts) == 1
        assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")

    async def test_auth_header(self):
        http_patch, mock_post = _patched_http_client(
            response=_mock_chat_response([_PNG_DATA_URI])
        )
        with patch(_RUNTIME, new=AsyncMock(return_value=_runtime_ok())), \
             http_patch, \
             patch(
                 "plugins.image_gen.openrouter.save_b64_image",
                 new=AsyncMock(return_value=Path("/tmp/x.png")),
             ):
            await _openrouter().generate(prompt="a pet")

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer sk-or-test"

    async def test_generate_uses_model_kwarg_from_dispatch(self):
        """image_generate passes image_gen.model as a model kwarg — honor it."""
        http_patch, mock_post = _patched_http_client(
            response=_mock_chat_response([_PNG_DATA_URI])
        )
        with patch(_RUNTIME, new=AsyncMock(return_value=_runtime_ok())), \
             http_patch, \
             patch(
                 "plugins.image_gen.openrouter.save_b64_image",
                 new=AsyncMock(return_value=Path("/tmp/x.png")),
             ):
            result = await _openrouter().generate(prompt="a pet", model="openai/gpt-image-2")

        assert result["success"] is True
        assert result["model"] == "openai/gpt-image-2"
        assert mock_post.call_args.kwargs["json"]["model"] == "openai/gpt-image-2"

    async def test_posts_to_resolved_base_url(self):
        """Nous routes to its own base URL — proves the same code serves both."""
        nous_runtime = _runtime_ok(
            provider="nous", base_url="https://inference.nousresearch.com/v1", api_key="nous-tok"
        )
        http_patch, mock_post = _patched_http_client(
            response=_mock_chat_response([_PNG_DATA_URI])
        )
        with patch(_RUNTIME, new=AsyncMock(return_value=nous_runtime)), \
             http_patch, \
             patch(
                 "plugins.image_gen.openrouter.save_b64_image",
                 new=AsyncMock(return_value=Path("/tmp/x.png")),
             ):
            from plugins.image_gen.openrouter import _build_providers

            nous = {p.name: p for p in _build_providers()}["nous"]
            result = await nous.generate(prompt="a pet")

        assert result["success"] is True
        assert result["provider"] == "nous"
        url = mock_post.call_args[0][0]
        assert url == "https://inference.nousresearch.com/v1/chat/completions"

    async def test_api_error(self):
        import httpx

        resp = MagicMock()
        resp.status_code = 401
        resp.text = "Unauthorized"
        resp.json.return_value = {"error": {"message": "Invalid API key"}}
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401 Unauthorized", request=httpx.Request("POST", "https://example.test"), response=resp
        )

        http_patch, mock_post = _patched_http_client(response=resp)
        with patch(_RUNTIME, new=AsyncMock(return_value=_runtime_ok())), http_patch:
            result = await _openrouter().generate(prompt="a pet")
        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert mock_post.call_count == 1

    async def test_timeout(self):
        import httpx

        http_patch, _ = _patched_http_client(side_effect=httpx.TimeoutException("timeout"))
        with patch(_RUNTIME, new=AsyncMock(return_value=_runtime_ok())), http_patch:
            result = await _openrouter().generate(prompt="a pet")
        assert result["success"] is False
        assert result["error_type"] == "timeout"


# ---------------------------------------------------------------------------
# Registration + pet integration
# ---------------------------------------------------------------------------


class TestRegistration:
    async def test_register_both(self):
        from plugins.image_gen.openrouter import register

        ctx = MagicMock()
        register(ctx)
        registered = [c.args[0].name for c in ctx.register_image_gen_provider.call_args_list]
        assert set(registered) == {"openrouter", "nous"}
