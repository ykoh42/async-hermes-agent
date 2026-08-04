"""Tests for probe-cache follow-ups on the #29988/#37595/#50572 salvage.

Covers:
- _query_ollama_api_show TTL caching (positive-only, namespaced key)
- persistent context-cache key normalization (trailing-slash dedup)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio



@pytest.fixture(autouse=True)
def _clear_probe_cache():
    """Module-level caches must not leak between tests."""
    from agent import model_metadata
    model_metadata._LOCAL_CTX_PROBE_CACHE.clear()
    model_metadata._endpoint_probe_path_cache.clear()
    yield
    model_metadata._LOCAL_CTX_PROBE_CACHE.clear()
    model_metadata._endpoint_probe_path_cache.clear()


def _mock_show_response(ctx=131072):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "model_info": {"llama.context_length": ctx},
        "parameters": "",
    }
    return resp


def _client_mock(resp):
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.post = AsyncMock(return_value=resp)
    client.get = AsyncMock()
    return client


class TestOllamaApiShowCaching:

    async def test_failure_never_memoized(self):
        """A down server must be re-probed on the next call (startup race)."""
        from agent.model_metadata import _query_ollama_api_show

        bad = MagicMock()
        bad.status_code = 404
        client = _client_mock(bad)
        with patch("httpx.AsyncClient", return_value=client):
            assert await _query_ollama_api_show("llama3", "http://127.0.0.1:11434") is None
            assert await _query_ollama_api_show("llama3", "http://127.0.0.1:11434") is None

        assert client.post.call_count == 2  # None was NOT cached

    async def test_ttl_expiry_reprobes(self):
        """After the 30s TTL lapses, the next call must hit the network again."""
        from agent import model_metadata
        from agent.model_metadata import _query_ollama_api_show
        import time as _time

        client = _client_mock(_mock_show_response(131072))
        with patch("httpx.AsyncClient", return_value=client):
            await _query_ollama_api_show("llama3", "http://127.0.0.1:11434")
            # Age the entry past the TTL.
            ((key, (val, _ts)),) = list(model_metadata._LOCAL_CTX_PROBE_CACHE.items())
            model_metadata._LOCAL_CTX_PROBE_CACHE[key] = (
                val, _time.monotonic() - model_metadata._LOCAL_CTX_PROBE_TTL_SECONDS - 1,
            )
            await _query_ollama_api_show("llama3", "http://127.0.0.1:11434")

        assert client.post.call_count == 2  # expired entry re-probed



class TestDetectLocalServerTypeCache:
    """#29988: detect_local_server_type memoized with a bounded TTL."""

    def _get_client(self, server_type="ollama"):
        ollama_resp = MagicMock()
        ollama_resp.status_code = 200
        ollama_resp.json.return_value = {"models": []}
        miss = MagicMock()
        miss.status_code = 404

        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        def _get(url, *a, **k):
            if url.endswith("/api/tags"):
                return ollama_resp
            return miss

        client.get = AsyncMock(side_effect=_get)
        return client

    async def test_second_call_served_from_cache(self):
        from agent.model_metadata import detect_local_server_type

        client = self._get_client()
        with patch("httpx.AsyncClient", return_value=client):
            first = await detect_local_server_type("http://127.0.0.1:11434")
            calls_after_first = client.get.call_count
            second = await detect_local_server_type("http://127.0.0.1:11434")

        assert first == second == "ollama"
        assert client.get.call_count == calls_after_first  # no new HTTP traffic

    async def test_ttl_expiry_allows_server_swap_redetection(self):
        """Stopping Ollama and starting LM Studio on the same port must be
        re-detected once the TTL lapses — the cache is bounded, not
        process-lifetime."""
        from agent import model_metadata
        from agent.model_metadata import detect_local_server_type
        import time as _time

        client = self._get_client()
        with patch("httpx.AsyncClient", return_value=client):
            assert await detect_local_server_type("http://127.0.0.1:11434") == "ollama"

        # Age the entry past the TTL, then swap the backend behind the URL.
        ((key, (val, _ts)),) = list(model_metadata._endpoint_probe_path_cache.items())
        model_metadata._endpoint_probe_path_cache[key] = (
            val, _time.monotonic() - model_metadata._ENDPOINT_PROBE_TTL_SECONDS - 1,
        )
        # Age the disk L2 entry too. Its TTL (300s) is much shorter than the
        # in-proc TTL (1h), so in real time-flow it always expires first —
        # this test compresses both expiries into one instant.
        import json as _json
        _disk = model_metadata._local_probe_disk_cache_path()
        if _disk.exists():
            _data = _json.loads(_disk.read_text(encoding="utf-8"))
            for _entry in _data.values():
                if isinstance(_entry, dict):
                    _entry["ts"] = (
                        _time.time() - model_metadata._LOCAL_PROBE_DISK_TTL_SECONDS - 1
                    )
            _disk.write_text(_json.dumps(_data), encoding="utf-8")

        lmstudio_resp = MagicMock()
        lmstudio_resp.status_code = 200
        lmstudio_resp.json.return_value = {"data": []}
        swap_client = MagicMock()
        swap_client.__aenter__ = AsyncMock(return_value=swap_client)
        swap_client.__aexit__ = AsyncMock(return_value=False)

        def _get(url, *a, **k):
            if url.endswith("/api/v1/models"):
                return lmstudio_resp
            miss = MagicMock(); miss.status_code = 404
            return miss

        swap_client.get = AsyncMock(side_effect=_get)
        with patch("httpx.AsyncClient", return_value=swap_client):
            assert await detect_local_server_type("http://127.0.0.1:11434") == "lm-studio"


class TestLocalhostIPv4SiblingSites:
    """#37595 widened: every probe helper rewrites localhost→127.0.0.1,
    not just detect_local_server_type."""


    async def test_rewrite_is_host_only_not_substring(self):
        """A URL that merely EMBEDS 'http://localhost' in its path/query must
        not be corrupted — only the URL's own host is rewritten."""
        from agent.model_metadata import _localhost_to_ipv4

        proxied = "https://proxy.example.com/route?upstream=http://localhost:11434"
        assert _localhost_to_ipv4(proxied) == proxied
        # Host must be a full label: localhost.example.com is NOT localhost.
        assert _localhost_to_ipv4("http://localhost.example.com/v1") == (
            "http://localhost.example.com/v1"
        )

    async def test_ollama_api_show_probes_ipv4(self):
        from agent.model_metadata import _query_ollama_api_show

        client = _client_mock(_mock_show_response(131072))
        with patch("httpx.AsyncClient", return_value=client):
            await _query_ollama_api_show("llama3", "http://localhost:11434")

        assert client.post.call_args[0][0].startswith("http://127.0.0.1:11434")

    async def test_fetch_endpoint_model_metadata_generic_probe_uses_ipv4(self):
        """The generic (non-LM-Studio) /models fetch loop must also rewrite
        localhost->127.0.0.1 before probing, like the LM Studio branch above."""
        from agent import model_metadata
        from agent.model_metadata import fetch_endpoint_model_metadata

        model_metadata._endpoint_model_metadata_cache.clear()
        model_metadata._endpoint_model_metadata_cache_time.clear()

        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"data": []}

        client = _client_mock(resp)
        client.get.return_value = resp
        with patch(
            "agent.model_metadata.detect_local_server_type",
            new=AsyncMock(return_value=None),
        ), patch("httpx.AsyncClient", return_value=client):
            await fetch_endpoint_model_metadata("http://localhost:8000/v1")

        assert client.get.call_args[0][0].startswith("http://127.0.0.1:8000")

    async def test_fetch_endpoint_model_metadata_llamacpp_props_followup_uses_ipv4(self):
        """The llama.cpp /props context-length follow-up must also rewrite
        localhost->127.0.0.1 before probing, not just the initial /models call."""
        from agent import model_metadata
        from agent.model_metadata import fetch_endpoint_model_metadata

        model_metadata._endpoint_model_metadata_cache.clear()
        model_metadata._endpoint_model_metadata_cache_time.clear()

        models_resp = MagicMock()
        models_resp.status_code = 200
        models_resp.raise_for_status = MagicMock()
        models_resp.json.return_value = {
            "data": [{"id": "llama-3-8b", "owned_by": "llamacpp"}],
        }

        props_resp = MagicMock()
        props_resp.ok = True
        props_resp.status_code = 200
        props_resp.json.return_value = {
            "default_generation_settings": {"n_ctx": 32768},
            "model_alias": "llama-3-8b",
        }

        client = _client_mock(models_resp)
        client.get.side_effect = [models_resp, props_resp]
        with patch(
            "agent.model_metadata.detect_local_server_type",
            new=AsyncMock(return_value=None),
        ), patch("httpx.AsyncClient", return_value=client):
            result = await fetch_endpoint_model_metadata(
                "http://localhost:8000/v1"
            )

        assert client.get.call_count == 2
        props_call_url = client.get.call_args_list[1][0][0]
        assert props_call_url.startswith("http://127.0.0.1:8000")
        assert result["llama-3-8b"]["context_length"] == 32768



class TestContextCacheKeyNormalization:
    async def test_trailing_slash_variants_share_one_entry(self, tmp_path, monkeypatch):
        from agent import model_metadata

        monkeypatch.setattr(
            model_metadata, "_get_context_cache_path",
            lambda: tmp_path / "context_lengths.yaml",
        )

        await model_metadata.save_context_length("m1", "http://host/v1/", 200_000)
        # Both slash variants resolve to the same row.
        assert await model_metadata.get_cached_context_length("m1", "http://host/v1") == 200_000
        assert await model_metadata.get_cached_context_length("m1", "http://host/v1/") == 200_000

        cache = await model_metadata._load_context_cache()
        assert list(cache.keys()) == ["m1@http://host/v1"]



    async def test_invalidate_clears_both_key_shapes(self, tmp_path, monkeypatch):
        import yaml
        from agent import model_metadata

        path = tmp_path / "context_lengths.yaml"
        monkeypatch.setattr(model_metadata, "_get_context_cache_path", lambda: path)
        path.write_text(yaml.dump({"context_lengths": {
            "m1@http://host/v1": 128_000,
            "m1@http://host/v1/": 64_000,
        }}))

        await model_metadata._invalidate_cached_context_length("m1", "http://host/v1/")
        cache = await model_metadata._load_context_cache()
        assert "m1@http://host/v1" not in cache
        assert "m1@http://host/v1/" not in cache
