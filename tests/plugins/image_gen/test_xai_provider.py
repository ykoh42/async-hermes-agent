#!/usr/bin/env python3
"""Tests for xAI image generation provider."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch, tmp_path):
    """Ensure XAI_API_KEY is set for all tests."""
    monkeypatch.setenv("XAI_API_KEY", "test-key-12345")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    try:
        import hermes_cli.config as cfg_mod

        if hasattr(cfg_mod, "_invalidate_load_config_cache"):
            cfg_mod._invalidate_load_config_cache()
    except Exception:
        pass

    import plugins.image_gen.xai as xai_plugin

    async def _credentials():
        return {
            "api_key": os.environ.get("XAI_API_KEY", ""),
            "provider": "xai",
            "base_url": "https://api.x.ai/v1",
        }

    monkeypatch.setattr(xai_plugin, "resolve_xai_http_credentials", _credentials)
    monkeypatch.setattr(
        xai_plugin,
        "build_xai_storage_options",
        AsyncMock(
            return_value={
                "public_url": True,
                "filename": "hermes-xai-image-test.png",
            }
        ),
    )
    monkeypatch.setattr(
        xai_plugin, "maybe_mark_xai_storage_notice_seen", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        xai_plugin, "read_xai_imagine_storage_config", AsyncMock(return_value={"enabled": True})
    )


def _patched_http_client(*, response=None, side_effect=None):
    post = AsyncMock(return_value=response, side_effect=side_effect)
    client = MagicMock()
    client.post = post
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=client)
    context.__aexit__ = AsyncMock(return_value=None)
    return patch("httpx.AsyncClient", return_value=context), post


# ---------------------------------------------------------------------------
# Provider class tests
# ---------------------------------------------------------------------------


class TestXAIImageGenProvider:
    async def test_name(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        provider = XAIImageGenProvider()
        assert provider.name == "xai"

    async def test_display_name(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        provider = XAIImageGenProvider()
        assert provider.display_name == "xAI (Grok)"

    async def test_is_available_with_key(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "sk-xxx")
        from plugins.image_gen.xai import XAIImageGenProvider

        provider = XAIImageGenProvider()
        assert await provider.is_available() is True


    async def test_list_models(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        provider = XAIImageGenProvider()
        models = await provider.list_models()
        assert len(models) >= 1
        assert models[0]["id"] == "grok-imagine-image"

    async def test_default_model(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        provider = XAIImageGenProvider()
        assert await provider.default_model() == "grok-imagine-image"

    async def test_get_setup_schema(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        provider = XAIImageGenProvider()
        schema = await provider.get_setup_schema()
        assert schema["name"] == "xAI Grok Imagine (image)"
        assert schema["badge"] == "paid"
        # Auth resolution is delegated to the shared "xai_grok" post_setup
        # hook so the picker doesn't blindly prompt for XAI_API_KEY when the
        # user is already signed in via xAI Grok OAuth.
        assert schema["env_vars"] == []
        assert schema["post_setup"] == "xai_grok"

    async def test_capabilities_expose_total_source_image_limit(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        caps = await XAIImageGenProvider().capabilities()
        assert caps["max_reference_images"] == 2
        assert caps["max_source_images"] == 3


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:


    async def test_custom_model(self, monkeypatch):
        monkeypatch.setenv("XAI_IMAGE_MODEL", "grok-imagine-image")
        from plugins.image_gen.xai import _resolve_model

        model_id, _ = await _resolve_model()
        assert model_id == "grok-imagine-image"


# ---------------------------------------------------------------------------
# Generate tests
# ---------------------------------------------------------------------------


class TestGenerate:
    async def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        from plugins.image_gen.xai import XAIImageGenProvider

        provider = XAIImageGenProvider()
        result = await provider.generate(prompt="test")
        assert result["success"] is False
        assert "XAI_API_KEY" in result["error"]

    async def test_successful_generation(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"b64_json": "dGVzdC1pbWFnZS1kYXRh"}],  # base64 "test-image-data"
        }

        http_patch, _ = _patched_http_client(response=mock_resp)
        with http_patch:
            with patch(
                "plugins.image_gen.xai.save_b64_image",
                new=AsyncMock(return_value="/tmp/test.png"),
            ):
                provider = XAIImageGenProvider()
                result = await provider.generate(prompt="A cat playing piano")

        assert result["success"] is True
        assert result["image"] == "/tmp/test.png"
        assert result["provider"] == "xai"
        assert result["model"] == "grok-imagine-image"


    async def test_url_response_falls_back_to_bare_url_when_download_fails(self):
        """If caching the URL fails (network blip, 404 in-flight), the
        provider must NOT hard-error — fall through to returning the bare
        URL so the agent surface at least sees *something*.  The gateway's
        existing URL-send fallback then has a chance to succeed; if it
        too 404s, the user gets the original (now legible) error rather
        than an opaque "image generation failed" tool result.
        """
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"url": "https://imgen.x.ai/xai-tmp-imgen-already-404.jpeg"}],
        }

        http_patch, _ = _patched_http_client(response=mock_resp)
        with http_patch, \
             patch(
                 "plugins.image_gen.xai.save_url_image",
                 new=AsyncMock(side_effect=RuntimeError("404 from CDN")),
             ):
            provider = XAIImageGenProvider()
            result = await provider.generate(prompt="A cat playing piano")

        assert result["success"] is True, (
            "Cache failure must not turn into a tool error — gateway gets a chance to retry"
        )
        assert result["image"] == "https://imgen.x.ai/xai-tmp-imgen-already-404.jpeg"

    async def test_api_error(self):
        import httpx
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        mock_resp.json.return_value = {"error": {"message": "Invalid API key"}}
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401 Unauthorized",
            request=httpx.Request("POST", "https://api.x.ai/v1/images/generations"),
            response=mock_resp,
        )

        http_patch, _ = _patched_http_client(response=mock_resp)
        with http_patch:
            provider = XAIImageGenProvider()
            result = await provider.generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "api_error"


    async def test_timeout(self):
        import httpx

        from plugins.image_gen.xai import XAIImageGenProvider

        http_patch, _ = _patched_http_client(side_effect=httpx.TimeoutException("timeout"))
        with http_patch:
            provider = XAIImageGenProvider()
            result = await provider.generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "timeout"

    async def test_empty_response(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": []}

        http_patch, _ = _patched_http_client(response=mock_resp)
        with http_patch:
            provider = XAIImageGenProvider()
            result = await provider.generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "empty_response"

    async def test_auth_header(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"url": "https://xai.image/test.png"}],
        }

        http_patch, mock_post = _patched_http_client(response=mock_resp)
        with http_patch, patch(
            "plugins.image_gen.xai.save_url_image",
            new=AsyncMock(return_value="/tmp/test.png"),
        ):
            provider = XAIImageGenProvider()
            await provider.generate(prompt="test")

        call_args = mock_post.call_args
        headers = call_args.kwargs.get("headers") or call_args[1].get("headers")
        assert "Bearer test-key-12345" in headers["Authorization"]
        assert "Hermes-Agent" in headers["User-Agent"]

    async def test_payload_resolution_is_literal_1k_or_2k(self):
        """Regression: xAI API rejects numeric resolutions ("1024"/"2048") with 422.

        The endpoint expects the literal strings "1k" or "2k". Ensure the wire
        payload carries that literal — not a numeric mapping. See PR #18678.
        """
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"url": "https://xai.image/test.png"}]}

        http_patch, mock_post = _patched_http_client(response=mock_resp)
        with http_patch, patch(
            "plugins.image_gen.xai.save_url_image",
            new=AsyncMock(return_value="/tmp/test.png"),
        ):
            provider = XAIImageGenProvider()
            await provider.generate(prompt="test")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["resolution"] in {"1k", "2k"}, (
            f"resolution must be the literal '1k' or '2k', got {payload['resolution']!r}"
        )

    async def test_image_edit_rejects_bare_file_id_input(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"url": "https://xai.image/edited.png"}]}

        http_patch, mock_post = _patched_http_client(response=mock_resp)
        with http_patch, patch(
            "plugins.image_gen.xai.save_url_image",
            new=AsyncMock(return_value="/tmp/edited.png"),
        ):
            provider = XAIImageGenProvider()
            result = await provider.generate(
                prompt="make the robot red",
                image_url="file_03eb65b1-aa97-482f-9ef0-b04f9172ea00",
            )

        assert result["success"] is False
        assert result["error_type"] == "invalid_image_url"
        mock_post.assert_not_called()


    async def test_multi_image_edit_rejects_bare_file_id_inputs(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"url": "https://xai.image/edited.png"}]}

        http_patch, mock_post = _patched_http_client(response=mock_resp)
        with http_patch, patch(
            "plugins.image_gen.xai.save_url_image",
            new=AsyncMock(return_value="/tmp/edited.png"),
        ):
            provider = XAIImageGenProvider()
            result = await provider.generate(
                prompt="combine these robots into one product shot",
                image_url="file_03eb65b1-aa97-482f-9ef0-b04f9172ea00",
                reference_image_urls=[
                    "file_54b48d6d-28ad-4982-9d72-bd3ac677c9bc",
                    "file_aa11bb22-cc33-44dd-88ee-ff0011223344",
                ],
            )

        assert result["success"] is False
        assert result["error_type"] == "invalid_image_url"
        mock_post.assert_not_called()


    async def test_storage_options_are_sent_by_default(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"b64_json": "dGVzdA=="}]}

        http_patch, mock_post = _patched_http_client(response=mock_resp)
        with http_patch, patch(
            "plugins.image_gen.xai.save_b64_image",
            new=AsyncMock(return_value="/tmp/test.png"),
        ):
            provider = XAIImageGenProvider()
            await provider.generate(prompt="test")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["storage_options"]["public_url"] is True
        assert "expires_after" not in payload["storage_options"]
        assert payload["storage_options"]["filename"].endswith(".png")

    async def test_public_url_file_output_wins_over_temporary_url(self):
        from plugins.image_gen.xai import XAIImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{
                "url": "https://imgen.x.ai/xai-tmp-imgen-test.jpeg",
                "file_output": {
                    "file_id": "file-123",
                    "filename": "stored.png",
                    "public_url": "https://xai-files.example/stored.png",
                    "public_url_expires_at": 1234567890,
                },
            }],
        }

        http_patch, _ = _patched_http_client(response=mock_resp)
        with http_patch, patch(
            "plugins.image_gen.xai.save_url_image", new=AsyncMock()
        ) as mock_save_url:
            provider = XAIImageGenProvider()
            result = await provider.generate(prompt="A cat playing piano")

        assert result["success"] is True
        assert result["image"] == "https://xai-files.example/stored.png"
        assert result["public_url"] == "https://xai-files.example/stored.png"
        assert "file_id" not in result
        mock_save_url.assert_not_called()


# ---------------------------------------------------------------------------
# Registration test
# ---------------------------------------------------------------------------


class TestRegistration:
    async def test_register(self):
        from plugins.image_gen.xai import XAIImageGenProvider, register

        mock_ctx = MagicMock()
        register(mock_ctx)
        mock_ctx.register_image_gen_provider.assert_called_once()
        provider = mock_ctx.register_image_gen_provider.call_args[0][0]
        assert isinstance(provider, XAIImageGenProvider)
        assert provider.name == "xai"


async def test_xai_image_field_expands_user_home(tmp_path, monkeypatch):
    """A ~-prefixed local image path must load (expanduser), not raise io_error.

    Pre-flight validation uses ``Path(source).expanduser()`` so a ``~/...`` path
    passes; ``_xai_image_field`` must expand it too or the load fails spuriously.
    """
    from plugins.image_gen.xai import _xai_image_field

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    img = tmp_path / "pic.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    field = await _xai_image_field("~/pic.png")
    assert field["type"] == "image_url"
    assert field["url"].startswith("data:image/png;base64,")


class TestXAIImageFieldReadGuard:
    """#57698: local image inputs must not read Hermes credential stores."""

    async def test_xai_image_field_blocks_credential_store(self, tmp_path, monkeypatch):
        from plugins.image_gen.xai import _xai_image_field

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        auth_json = hermes_home / "auth.json"
        auth_json.write_text('{"api_key":"sk-secret"}', encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        with pytest.raises(ValueError, match="credential store"):
            await _xai_image_field(str(auth_json))
