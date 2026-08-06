#!/usr/bin/env python3
"""Tests for Krea image generation provider."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    """Ensure KREA_API_KEY is set for all tests."""
    monkeypatch.setenv("KREA_API_KEY", "test-key-12345")



@pytest.fixture
def http_calls(monkeypatch):
    """Route each provider-created AsyncClient through awaitable test spies."""
    calls = SimpleNamespace(post=AsyncMock(), get=AsyncMock())

    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, *args, **kwargs):
            return await calls.post(*args, **kwargs)

        async def get(self, *args, **kwargs):
            return await calls.get(*args, **kwargs)

    import plugins.image_gen.krea as krea_mod

    monkeypatch.setattr(krea_mod.httpx, "AsyncClient", Client)
    monkeypatch.setattr(krea_mod.asyncio, "sleep", AsyncMock())
    return calls


def _completed_job(url: str = "https://krea.cdn/img.png") -> dict:
    return {
        "job_id": "00000000-0000-0000-0000-000000000abc",
        "status": "completed",
        "created_at": "2026-05-27T00:00:00Z",
        "completed_at": "2026-05-27T00:00:30Z",
        "result": {"urls": [url]},
    }


def _submit_response(job_id: str = "00000000-0000-0000-0000-000000000abc"):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "job_id": job_id,
        "status": "queued",
        "created_at": "2026-05-27T00:00:00Z",
        "completed_at": None,
        "result": None,
    }
    return resp


def _poll_response(body: dict):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = body
    return resp


# ---------------------------------------------------------------------------
# Provider class tests
# ---------------------------------------------------------------------------


class TestKreaImageGenProvider:
    async def test_name(self, http_calls):
        from plugins.image_gen.krea import KreaImageGenProvider

        assert KreaImageGenProvider().name == "krea"

    async def test_display_name(self, http_calls):
        from plugins.image_gen.krea import KreaImageGenProvider

        assert KreaImageGenProvider().display_name == "Krea"

    async def test_is_available_with_key(self, http_calls, monkeypatch):
        monkeypatch.setenv("KREA_API_KEY", "sk-test")
        from plugins.image_gen.krea import KreaImageGenProvider

        assert await KreaImageGenProvider().is_available() is True


    async def test_list_models(self, http_calls):
        from plugins.image_gen.krea import KreaImageGenProvider

        models = await KreaImageGenProvider().list_models()
        ids = {m["id"] for m in models}
        assert {"krea-2-medium", "krea-2-large"} <= ids
        # Each entry carries the picker fields the registry expects.
        for m in models:
            assert m["display"]
            assert m["speed"]
            assert m["strengths"]
            assert m["price"]

    async def test_default_model_is_medium(self, http_calls):
        from plugins.image_gen.krea import KreaImageGenProvider

        assert await KreaImageGenProvider().default_model() == "krea-2-medium"

    async def test_get_setup_schema(self, http_calls):
        from plugins.image_gen.krea import KreaImageGenProvider

        schema = await KreaImageGenProvider().get_setup_schema()
        assert schema["name"] == "Krea"
        assert schema["badge"] == "paid"
        env_vars = schema["env_vars"]
        assert len(env_vars) == 1
        assert env_vars[0]["key"] == "KREA_API_KEY"
        assert "krea.ai" in env_vars[0]["url"]


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------


class TestModelResolution:

    async def test_env_override_large(self, http_calls, monkeypatch):
        monkeypatch.setenv("KREA_IMAGE_MODEL", "krea-2-large")
        from plugins.image_gen.krea import _resolve_model

        model_id, meta = await _resolve_model()
        assert model_id == "krea-2-large"
        assert meta["path"] == "large"


    async def test_creativity_default(self, http_calls):
        from plugins.image_gen.krea import _resolve_creativity

        assert await _resolve_creativity(None) == "medium"


# ---------------------------------------------------------------------------
# Generate — main flow
# ---------------------------------------------------------------------------


class TestGenerate:
    async def test_missing_api_key(self, http_calls, monkeypatch):
        monkeypatch.delenv("KREA_API_KEY", raising=False)
        from plugins.image_gen.krea import KreaImageGenProvider

        result = await KreaImageGenProvider().generate(prompt="test")
        assert result["success"] is False
        assert "KREA_API_KEY" in result["error"]
        assert result["error_type"] == "auth_required"

    async def test_empty_prompt(self, http_calls):
        from plugins.image_gen.krea import KreaImageGenProvider

        result = await KreaImageGenProvider().generate(prompt="   ")
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"

    async def test_successful_generation(self, http_calls):
        """Happy path: submit → one poll → completed → URL downloaded."""
        from plugins.image_gen.krea import KreaImageGenProvider

        submit = _submit_response()
        poll = _poll_response(_completed_job("https://krea.cdn/result.png"))

        with patch.object(http_calls, "post", new_callable=AsyncMock, return_value=submit) as mock_post, \
             patch.object(http_calls, "get", new_callable=AsyncMock, return_value=poll) as mock_get, \
             patch(
                 "plugins.image_gen.krea.save_url_image",
                 new_callable=AsyncMock,
                 return_value=Path("/tmp/krea_krea-2-medium_test.png"),
             ) as mock_save, \
             patch("plugins.image_gen.krea.asyncio.sleep", new_callable=AsyncMock):  # skip real waits
            result = await KreaImageGenProvider().generate(prompt="A cinematic lamp")

        assert result["success"] is True
        assert result["image"] == "/tmp/krea_krea-2-medium_test.png"
        assert result["provider"] == "krea"
        assert result["model"] == "krea-2-medium"
        assert result["aspect_ratio"] == "landscape"
        assert result["job_id"] == "00000000-0000-0000-0000-000000000abc"
        assert result["resolution"] == "1K"
        assert result["creativity"] == "medium"
        # Submit hit the medium endpoint
        post_url = mock_post.call_args[0][0]
        assert post_url.endswith("/generate/image/krea/krea-2/medium")
        # Poll hit /jobs/{job_id}
        poll_url = mock_get.call_args[0][0]
        assert "/jobs/00000000-0000-0000-0000-000000000abc" in poll_url
        # URL was materialised once
        mock_save.assert_called_once()

    async def test_large_model_routes_to_large_endpoint(self, http_calls, monkeypatch):
        monkeypatch.setenv("KREA_IMAGE_MODEL", "krea-2-large")
        from plugins.image_gen.krea import KreaImageGenProvider

        submit = _submit_response()
        poll = _poll_response(_completed_job())

        with patch.object(http_calls, "post", new_callable=AsyncMock, return_value=submit) as mock_post, \
             patch.object(http_calls, "get", new_callable=AsyncMock, return_value=poll), \
             patch(
                 "plugins.image_gen.krea.save_url_image",
                 new_callable=AsyncMock,
                 return_value=Path("/tmp/x.png"),
             ), \
             patch("plugins.image_gen.krea.asyncio.sleep", new_callable=AsyncMock):
            await KreaImageGenProvider().generate(prompt="test")

        post_url = mock_post.call_args[0][0]
        assert post_url.endswith("/generate/image/krea/krea-2/large")

    async def test_aspect_ratio_mapping(self, http_calls):
        """Hermes 'square' must map to Krea '1:1' in the wire payload."""
        from plugins.image_gen.krea import KreaImageGenProvider

        submit = _submit_response()
        poll = _poll_response(_completed_job())

        with patch.object(http_calls, "post", new_callable=AsyncMock, return_value=submit) as mock_post, \
             patch.object(http_calls, "get", new_callable=AsyncMock, return_value=poll), \
             patch(
                 "plugins.image_gen.krea.save_url_image",
                 new_callable=AsyncMock,
                 return_value=Path("/tmp/x.png"),
             ), \
             patch("plugins.image_gen.krea.asyncio.sleep", new_callable=AsyncMock):
            await KreaImageGenProvider().generate(prompt="test", aspect_ratio="square")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["aspect_ratio"] == "1:1"
        assert payload["resolution"] == "1K"

    async def test_auth_header(self, http_calls):
        from plugins.image_gen.krea import KreaImageGenProvider

        submit = _submit_response()
        poll = _poll_response(_completed_job())

        with patch.object(http_calls, "post", new_callable=AsyncMock, return_value=submit) as mock_post, \
             patch.object(http_calls, "get", new_callable=AsyncMock, return_value=poll), \
             patch(
                 "plugins.image_gen.krea.save_url_image",
                 new_callable=AsyncMock,
                 return_value=Path("/tmp/x.png"),
             ), \
             patch("plugins.image_gen.krea.asyncio.sleep", new_callable=AsyncMock):
            await KreaImageGenProvider().generate(prompt="test")

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer test-key-12345"
        assert headers["Content-Type"] == "application/json"

    async def test_passthrough_seed_styles_moodboards(self, http_calls):
        from plugins.image_gen.krea import KreaImageGenProvider

        submit = _submit_response()
        poll = _poll_response(_completed_job())

        with patch.object(http_calls, "post", new_callable=AsyncMock, return_value=submit) as mock_post, \
             patch.object(http_calls, "get", new_callable=AsyncMock, return_value=poll), \
             patch(
                 "plugins.image_gen.krea.save_url_image",
                 new_callable=AsyncMock,
                 return_value=Path("/tmp/x.png"),
             ), \
             patch("plugins.image_gen.krea.asyncio.sleep", new_callable=AsyncMock):
            await KreaImageGenProvider().generate(
                prompt="test",
                seed=42,
                styles=[{"id": "lora-1", "strength": 0.7}],
                moodboards=[{"url": "https://x.com/mood.png"}, {"url": "https://x.com/mood2.png"}],
                image_style_references=[{"url": f"https://x.com/{i}.png"} for i in range(15)],
                creativity="high",
            )

        payload = mock_post.call_args.kwargs["json"]
        assert payload["seed"] == 42
        assert payload["styles"] == [{"id": "lora-1", "strength": 0.7}]
        assert len(payload["moodboards"]) == 1  # capped at 1
        assert len(payload["image_style_references"]) == 10  # capped at 10
        assert payload["creativity"] == "high"

    async def test_string_style_references_converted_to_objects(self, http_calls):
        """Krea requires {url, strength} objects; bare URL strings must be
        converted (a string yields a 422 'Expected object, received string')."""
        from plugins.image_gen.krea import KreaImageGenProvider

        submit = _submit_response()
        poll = _poll_response(_completed_job())

        with patch.object(http_calls, "post", new_callable=AsyncMock, return_value=submit) as mock_post, \
             patch.object(http_calls, "get", new_callable=AsyncMock, return_value=poll), \
             patch(
                 "plugins.image_gen.krea.save_url_image",
                 new_callable=AsyncMock,
                 return_value=Path("/tmp/x.png"),
             ), \
             patch("plugins.image_gen.krea.asyncio.sleep", new_callable=AsyncMock):
            await KreaImageGenProvider().generate(
                prompt="test",
                image_style_references=[
                    "https://x.com/a.png",
                    {"url": "https://x.com/b.png", "strength": 1.2},
                ],
            )

        payload = mock_post.call_args.kwargs["json"]
        assert payload["image_style_references"] == [
            {"url": "https://x.com/a.png", "strength": 0.6},
            {"url": "https://x.com/b.png", "strength": 1.2},
        ]

    async def test_unknown_kwargs_ignored(self, http_calls):
        """Forward-compat: unknown kwargs must not break generate()."""
        from plugins.image_gen.krea import KreaImageGenProvider

        submit = _submit_response()
        poll = _poll_response(_completed_job())

        with patch.object(http_calls, "post", new_callable=AsyncMock, return_value=submit), \
             patch.object(http_calls, "get", new_callable=AsyncMock, return_value=poll), \
             patch(
                 "plugins.image_gen.krea.save_url_image",
                 new_callable=AsyncMock,
                 return_value=Path("/tmp/x.png"),
             ), \
             patch("plugins.image_gen.krea.asyncio.sleep", new_callable=AsyncMock):
            result = await KreaImageGenProvider().generate(
                prompt="test",
                fictional_param="should be ignored",
                num_images=4,
            )

        assert result["success"] is True


# ---------------------------------------------------------------------------
# Generate — error paths
# ---------------------------------------------------------------------------


class TestGenerateErrors:
    async def test_submit_http_error(self, http_calls):
        import httpx
        from plugins.image_gen.krea import KreaImageGenProvider

        resp = httpx.Response(
            401,
            json={"error": {"message": "Invalid API key"}},
            request=httpx.Request("POST", "https://api.krea.ai/generate"),
        )

        with patch.object(http_calls, "post", new_callable=AsyncMock, return_value=resp):
            result = await KreaImageGenProvider().generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "401" in result["error"]
        assert "Invalid API key" in result["error"]


    async def test_job_failed(self, http_calls):
        from plugins.image_gen.krea import KreaImageGenProvider

        failed = {
            "job_id": "abc",
            "status": "failed",
            "completed_at": "2026-05-27T00:01:00Z",
            "result": {"error": "NSFW content"},
        }

        submit = _submit_response()
        with patch.object(http_calls, "post", new_callable=AsyncMock, return_value=submit), \
             patch.object(
                 http_calls,
                 "get",
                 new_callable=AsyncMock,
                 return_value=_poll_response(failed),
             ), \
             patch("plugins.image_gen.krea.asyncio.sleep", new_callable=AsyncMock):
            result = await KreaImageGenProvider().generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "NSFW" in result["error"]


    async def test_completed_but_missing_urls(self, http_calls):
        from plugins.image_gen.krea import KreaImageGenProvider

        completed_empty = {
            "job_id": "abc",
            "status": "completed",
            "completed_at": "2026-05-27T00:01:00Z",
            "result": {"urls": []},
        }

        with patch.object(http_calls, "post", new_callable=AsyncMock, return_value=_submit_response()), \
             patch.object(
                 http_calls,
                 "get",
                 new_callable=AsyncMock,
                 return_value=_poll_response(completed_empty),
             ), \
             patch("plugins.image_gen.krea.asyncio.sleep", new_callable=AsyncMock):
            result = await KreaImageGenProvider().generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "empty_response"

    async def test_url_download_failure_falls_back_to_bare_url(self, http_calls):
        """Mirror of xAI behaviour — if local cache fails, return the URL."""
        import httpx
        from plugins.image_gen.krea import KreaImageGenProvider

        url = "https://krea.cdn/expired-soon.png"
        submit = _submit_response()
        poll = _poll_response(_completed_job(url))

        with patch.object(http_calls, "post", new_callable=AsyncMock, return_value=submit), \
             patch.object(http_calls, "get", new_callable=AsyncMock, return_value=poll), \
             patch(
                 "plugins.image_gen.krea.save_url_image",
                 new_callable=AsyncMock,
                 side_effect=httpx.HTTPError("404"),
             ), \
             patch("plugins.image_gen.krea.asyncio.sleep", new_callable=AsyncMock):
            result = await KreaImageGenProvider().generate(prompt="test")

        assert result["success"] is True
        assert result["image"] == url

    async def test_polling_picks_up_completed_at_with_unknown_status(self, http_calls):
        """``completed_at`` set + unrecognised pending status → still terminal."""
        from plugins.image_gen.krea import KreaImageGenProvider

        # Use a status value that is NOT in our terminal set ("intermediate-complete")
        # but with completed_at populated — Krea's spec says completed_at is the
        # canonical terminal marker.
        oddball = {
            "job_id": "abc",
            "status": "intermediate-complete",
            "completed_at": "2026-05-27T00:01:00Z",
            "result": {"urls": ["https://krea.cdn/done.png"]},
        }

        with patch.object(http_calls, "post", new_callable=AsyncMock, return_value=_submit_response()), \
             patch.object(
                 http_calls,
                 "get",
                 new_callable=AsyncMock,
                 return_value=_poll_response(oddball),
             ), \
             patch(
                 "plugins.image_gen.krea.save_url_image",
                 new_callable=AsyncMock,
                 return_value=Path("/tmp/x.png"),
             ), \
             patch("plugins.image_gen.krea.asyncio.sleep", new_callable=AsyncMock):
            result = await KreaImageGenProvider().generate(prompt="test")

        assert result["success"] is True


class TestPollRetryPolicy:
    """Polling fail-fast on permanent 4xx, retry on transient 5xx/429."""

    def _http_error_response(self, status: int):
        import httpx

        return httpx.Response(
            status,
            json={"error": "boom"},
            request=httpx.Request("GET", "https://api.krea.ai/jobs/job-id"),
        )

    async def test_poll_fails_fast_on_401(self, http_calls):
        """Auth failure mid-poll should not wait the 180s deadline."""
        from plugins.image_gen.krea import KreaImageGenProvider

        bad_poll = self._http_error_response(401)

        with patch.object(http_calls, "post", new_callable=AsyncMock, return_value=_submit_response()), \
             patch.object(http_calls, "get", new_callable=AsyncMock, return_value=bad_poll) as mock_get, \
             patch("plugins.image_gen.krea.asyncio.sleep", new_callable=AsyncMock):
            result = await KreaImageGenProvider().generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "401" in result["error"]
        # One call — no retry on permanent auth failure.
        assert mock_get.call_count == 1


# ---------------------------------------------------------------------------
# Managed Nous gateway path
# ---------------------------------------------------------------------------


def _managed_cfg(
    origin: str = "https://krea-gateway.example.com",
    token: str = "nous-tok-abc",
):
    from types import SimpleNamespace

    return SimpleNamespace(
        vendor="krea",
        gateway_origin=origin,
        nous_user_token=token,
        managed_mode=True,
    )


class TestManagedGateway:
    async def test_managed_submit_uses_gateway_origin_and_nous_token(self, http_calls, monkeypatch):
        """Managed mode submits to the gateway origin with the Nous token."""
        import plugins.image_gen.krea as krea_mod
        from plugins.image_gen.krea import KreaImageGenProvider

        # Even with a direct key present, an active managed gateway wins.
        monkeypatch.setattr(krea_mod, "_resolve_managed_krea_gateway", AsyncMock(return_value=_managed_cfg()))

        submit = _submit_response()
        poll = _poll_response(_completed_job())
        with patch.object(http_calls, "post", new_callable=AsyncMock, return_value=submit) as mock_post, \
             patch.object(http_calls, "get", new_callable=AsyncMock, return_value=poll) as mock_get, \
             patch(
                 "plugins.image_gen.krea.save_url_image",
                 new_callable=AsyncMock,
                 return_value=Path("/tmp/x.png"),
             ), \
             patch("plugins.image_gen.krea.asyncio.sleep", new_callable=AsyncMock):
            result = await KreaImageGenProvider().generate(prompt="A managed lamp")

        assert result["success"] is True
        post_url = mock_post.call_args[0][0]
        assert post_url == (
            "https://krea-gateway.example.com/generate/image/krea/krea-2/medium"
        )
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer nous-tok-abc"
        # Idempotency key drives the gateway's per-generation billing boundary.
        assert headers["x-idempotency-key"]
        # Poll is bound to the same gateway + Nous token.
        poll_url = mock_get.call_args[0][0]
        assert poll_url.startswith("https://krea-gateway.example.com/jobs/")
        poll_headers = mock_get.call_args.kwargs["headers"]
        assert poll_headers["Authorization"] == "Bearer nous-tok-abc"


    async def test_managed_429_concurrency_hint(self, http_calls, monkeypatch):
        import httpx
        import plugins.image_gen.krea as krea_mod
        from plugins.image_gen.krea import KreaImageGenProvider

        monkeypatch.setattr(
            krea_mod,
            "_resolve_managed_krea_gateway",
            AsyncMock(return_value=_managed_cfg()),
        )

        resp = httpx.Response(
            429,
            json={"error": {"message": "maximum number of concurrent jobs"}},
            request=httpx.Request("POST", "https://krea-gateway.example.com/generate"),
        )

        with patch.object(http_calls, "post", new_callable=AsyncMock, return_value=resp):
            result = await KreaImageGenProvider().generate(prompt="test")

        assert result["success"] is False
        assert "429" in result["error"]
        assert "concurrency" in result["error"].lower()


class TestExplicitModelOverride:
    async def test_model_kwarg_overrides_config(self, http_calls, monkeypatch):
        """An explicit ``model`` kwarg (managed routing) wins over config/default."""
        from plugins.image_gen.krea import _resolve_model

        model_id, meta = await _resolve_model("krea-2-large")
        assert model_id == "krea-2-large"
        assert meta["path"] == "large"

    async def test_turbo_routes_to_medium_turbo_endpoint(self, http_calls):
        from plugins.image_gen.krea import KreaImageGenProvider

        submit = _submit_response()
        poll = _poll_response(_completed_job())
        with patch.object(http_calls, "post", new_callable=AsyncMock, return_value=submit) as mock_post, \
             patch.object(http_calls, "get", new_callable=AsyncMock, return_value=poll), \
             patch(
                 "plugins.image_gen.krea.save_url_image",
                 new_callable=AsyncMock,
                 return_value=Path("/tmp/x.png"),
             ), \
             patch("plugins.image_gen.krea.asyncio.sleep", new_callable=AsyncMock):
            result = await KreaImageGenProvider().generate(prompt="test", model="krea-2-medium-turbo")

        assert result["success"] is True
        assert result["model"] == "krea-2-medium-turbo"
        post_url = mock_post.call_args[0][0]
        assert post_url.endswith("/generate/image/krea/krea-2/medium-turbo")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    async def test_register(self, http_calls):
        from plugins.image_gen.krea import KreaImageGenProvider, register

        mock_ctx = MagicMock()
        register(mock_ctx)
        mock_ctx.register_image_gen_provider.assert_called_once()
        provider = mock_ctx.register_image_gen_provider.call_args[0][0]
        assert isinstance(provider, KreaImageGenProvider)
        assert provider.name == "krea"
