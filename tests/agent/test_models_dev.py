"""Tests for agent.models_dev — models.dev registry integration."""
import asyncio
import gc
import json
import threading
import time
import weakref
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.models_dev import (
    PROVIDER_TO_MODELS_DEV,
    _extract_context,
    fetch_models_dev,
    get_model_capabilities,
    lookup_models_dev_context,
)


SAMPLE_REGISTRY = {
    "anthropic": {
        "id": "anthropic",
        "name": "Anthropic",
        "models": {
            "claude-opus-4-6": {
                "id": "claude-opus-4-6",
                "limit": {"context": 1000000, "output": 128000},
            },
            "claude-sonnet-4-6": {
                "id": "claude-sonnet-4-6",
                "limit": {"context": 1000000, "output": 64000},
            },
            "claude-sonnet-4-0": {
                "id": "claude-sonnet-4-0",
                "limit": {"context": 200000, "output": 64000},
            },
        },
    },
    "github-copilot": {
        "id": "github-copilot",
        "name": "GitHub Copilot",
        "models": {
            "claude-opus-4.6": {
                "id": "claude-opus-4.6",
                "limit": {"context": 128000, "output": 32000},
            },
        },
    },
    "xai": {
        "id": "xai",
        "name": "xAI",
        "models": {
            "grok-build-0.1": {
                "id": "grok-build-0.1",
                "limit": {"context": 256000, "output": 64000},
            },
        },
    },
    "kilo": {
        "id": "kilo",
        "name": "Kilo Gateway",
        "models": {
            "anthropic/claude-sonnet-4.6": {
                "id": "anthropic/claude-sonnet-4.6",
                "limit": {"context": 1000000, "output": 128000},
            },
        },
    },
    "deepseek": {
        "id": "deepseek",
        "name": "DeepSeek",
        "models": {
            "deepseek-chat": {
                "id": "deepseek-chat",
                "limit": {"context": 128000, "output": 8192},
            },
        },
    },
    "audio-only": {
        "id": "audio-only",
        "models": {
            "tts-model": {
                "id": "tts-model",
                "limit": {"context": 0, "output": 0},
            },
        },
    },
}


class TestProviderMapping:
    def test_all_mapped_providers_are_strings(self):
        for hermes_id, mdev_id in PROVIDER_TO_MODELS_DEV.items():
            assert isinstance(hermes_id, str)
            assert isinstance(mdev_id, str)

    def test_known_providers_mapped(self):
        assert PROVIDER_TO_MODELS_DEV["anthropic"] == "anthropic"
        assert PROVIDER_TO_MODELS_DEV["copilot"] == "github-copilot"
        assert PROVIDER_TO_MODELS_DEV["stepfun"] == "stepfun"
        assert PROVIDER_TO_MODELS_DEV["kilocode"] == "kilo"
        assert PROVIDER_TO_MODELS_DEV["ai-gateway"] == "vercel"

    def test_xai_oauth_uses_xai_catalog(self):
        assert PROVIDER_TO_MODELS_DEV["xai"] == "xai"
        assert PROVIDER_TO_MODELS_DEV["xai-oauth"] == "xai"

    def test_unmapped_provider_not_in_dict(self):
        assert "nous" not in PROVIDER_TO_MODELS_DEV



class TestExtractContext:
    def test_valid_entry(self):
        assert _extract_context({"limit": {"context": 128000}}) == 128000




    def test_non_dict_returns_none(self):
        assert _extract_context("not a dict") is None



class TestLookupModelsDevContext:
    @pytest.mark.asyncio
    @patch("agent.models_dev.fetch_models_dev")
    async def test_exact_match(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_REGISTRY
        assert await lookup_models_dev_context(
            "anthropic", "claude-opus-4-6"
        ) == 1000000






    @patch("agent.models_dev.fetch_models_dev")
    def test_zero_context_filtered(self, mock_fetch):
        mock_fetch.return_value = SAMPLE_REGISTRY
        # audio-only is not a mapped provider, but test the filtering directly
        data = SAMPLE_REGISTRY["audio-only"]["models"]["tts-model"]
        assert _extract_context(data) is None



class TestFetchModelsDev:
    @pytest.fixture(autouse=True)
    def _reset_fetch_state(self):
        import agent.models_dev as md

        md._models_dev_cache = {}
        md._models_dev_cache_time = 0
        md._models_dev_retry_after = 0
        md._models_dev_lock = None
        md._models_dev_refresh_task = None
        md._models_dev_update_claim = None
        yield
        md._models_dev_cache = {}
        md._models_dev_cache_time = 0
        md._models_dev_retry_after = 0
        md._models_dev_lock = None
        md._models_dev_refresh_task = None
        md._models_dev_update_claim = None

    @pytest.mark.asyncio
    async def test_disk_cache_short_circuits_network(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "models_dev_cache.json").write_text(
            json.dumps(SAMPLE_REGISTRY), encoding="utf-8"
        )

        with patch("httpx.AsyncClient") as client_cls:
            result = await fetch_models_dev()

        assert result == SAMPLE_REGISTRY
        client_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_network_disabled_never_fetches(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        with patch("httpx.AsyncClient") as client_cls:
            result = await fetch_models_dev(allow_network=False)

        assert result == {}
        client_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_stale_disk_cache_returns_without_foreground_network(self):
        import agent.models_dev as md

        with (
            patch.object(
                md,
                "_disk_cache_age_seconds",
                new=AsyncMock(return_value=md._MODELS_DEV_CACHE_TTL + 60),
            ),
            patch.object(
                md,
                "_load_disk_cache",
                new=AsyncMock(return_value=SAMPLE_REGISTRY),
            ),
            patch.object(md, "_start_background_refresh_models_dev") as refresh,
            patch.object(
                md,
                "_fetch_models_dev_from_network",
                new=AsyncMock(),
            ) as network,
        ):
            result = await fetch_models_dev()

        assert result == SAMPLE_REGISTRY
        refresh.assert_called_once_with()
        network.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stale_cache_failure_enters_backoff_and_suppresses_retry(self):
        import agent.models_dev as md

        md._models_dev_cache = SAMPLE_REGISTRY
        md._models_dev_cache_time = time.time() - md._MODELS_DEV_CACHE_TTL - 1
        network = AsyncMock(side_effect=OSError("models.dev unreachable"))

        with patch.object(md, "_fetch_models_dev_from_network", new=network):
            first = await fetch_models_dev()
            refresh_task = md._models_dev_refresh_task
            assert refresh_task is not None
            await refresh_task

            second = await fetch_models_dev()

        assert first == second == SAMPLE_REGISTRY
        assert md._models_dev_retry_after > time.time()
        network.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stale_cache_returns_before_background_refresh_finishes(self):
        import agent.models_dev as md

        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_refresh():
            started.set()
            await release.wait()
            return SAMPLE_REGISTRY

        md._models_dev_cache = {"stale": {}}
        md._models_dev_cache_time = time.time() - md._MODELS_DEV_CACHE_TTL - 1
        with (
            patch.object(md, "_fetch_models_dev_from_network", new=slow_refresh),
            patch.object(md, "_save_disk_cache", new=AsyncMock()),
        ):
            result = await asyncio.wait_for(fetch_models_dev(), timeout=0.1)
            assert result == {"stale": {}}
            await asyncio.wait_for(started.wait(), timeout=0.1)
            refresh_task = md._models_dev_refresh_task
            assert refresh_task is not None and not refresh_task.done()
            release.set()
            await asyncio.wait_for(refresh_task, timeout=0.1)

    @pytest.mark.asyncio
    async def test_cancelled_background_refresh_clears_task_without_backoff(self):
        import agent.models_dev as md

        started = asyncio.Event()

        async def blocked_refresh():
            started.set()
            await asyncio.Event().wait()

        md._models_dev_cache = {"stale": {}}
        md._models_dev_cache_time = time.time() - md._MODELS_DEV_CACHE_TTL - 1
        with patch.object(
            md,
            "_fetch_models_dev_from_network",
            new=blocked_refresh,
        ):
            assert await fetch_models_dev() == {"stale": {}}
            await asyncio.wait_for(started.wait(), timeout=0.1)
            refresh_task = md._models_dev_refresh_task
            assert refresh_task is not None
            refresh_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await refresh_task
            await asyncio.sleep(0)

        assert md._models_dev_refresh_task is None
        assert md._models_dev_retry_after == 0

    @pytest.mark.asyncio
    async def test_background_refresh_success_commits_registry(self):
        import agent.models_dev as md

        md._models_dev_cache = {"stale": {}}
        md._models_dev_cache_time = 0
        md._models_dev_retry_after = time.time() - 1

        with (
            patch.object(
                md,
                "_fetch_models_dev_from_network",
                new=AsyncMock(return_value=SAMPLE_REGISTRY),
            ),
            patch.object(md, "_save_disk_cache", new=AsyncMock()) as save,
        ):
            await md._background_refresh_models_dev()

        save.assert_awaited_once_with(SAMPLE_REGISTRY)
        assert md._models_dev_cache == SAMPLE_REGISTRY
        assert md._models_dev_cache_time > 0
        assert md._models_dev_retry_after == 0

    @pytest.mark.asyncio
    async def test_concurrent_refreshes_share_one_network_request(
        self,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        response = MagicMock()
        response.json.return_value = SAMPLE_REGISTRY
        response.raise_for_status.return_value = None
        client = MagicMock()
        client.get = AsyncMock(return_value=response)
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=client)
        context.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=context):
            results = await asyncio.gather(*(fetch_models_dev() for _ in range(6)))

        assert results == [SAMPLE_REGISTRY] * 6
        client.get.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_force_refresh_bypasses_failure_backoff(self):
        import agent.models_dev as md

        network = AsyncMock(
            side_effect=[OSError("models.dev unreachable"), SAMPLE_REGISTRY]
        )
        with (
            patch.object(
                md,
                "_disk_cache_age_seconds",
                new=AsyncMock(return_value=None),
            ),
            patch.object(md, "_load_disk_cache", new=AsyncMock(return_value={})),
            patch.object(md, "_save_disk_cache", new=AsyncMock()),
            patch.object(md, "_fetch_models_dev_from_network", new=network),
        ):
            assert await fetch_models_dev() == {}
            assert md._models_dev_retry_after > time.time()
            assert await fetch_models_dev(force_refresh=True) == SAMPLE_REGISTRY

        assert network.await_count == 2
        assert md._models_dev_retry_after == 0

    def test_cold_fetch_singleflight_is_loop_neutral(self, tmp_path, monkeypatch):
        """Two event-loop threads share one public catalogue request safely."""
        import agent.models_dev as md

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        entered = threading.Event()
        release = threading.Event()
        calls = 0
        calls_guard = threading.Lock()

        async def fetch_network():
            nonlocal calls
            with calls_guard:
                calls += 1
            entered.set()
            while not release.is_set():
                await asyncio.sleep(0.001)
            return SAMPLE_REGISTRY

        results = []
        errors = []
        loops = []

        def runner():
            loop = asyncio.new_event_loop()
            loops.append(weakref.ref(loop))
            try:
                results.append(loop.run_until_complete(fetch_models_dev()))
            except BaseException as exc:
                errors.append(exc)
            finally:
                loop.close()

        # Patch once in the owning test thread. Concurrent nested patch
        # contexts race their restoration order and can leak an AsyncMock into
        # later tests even when the runtime itself is correct.
        with (
            patch.object(md, "_disk_cache_age_seconds", new=AsyncMock(return_value=None)),
            patch.object(md, "_load_disk_cache", new=AsyncMock(return_value={})),
            patch.object(md, "_save_disk_cache", new=AsyncMock()),
            patch.object(md, "_fetch_models_dev_from_network", new=fetch_network),
        ):
            first = threading.Thread(target=runner)
            second = threading.Thread(target=runner)
            first.start()
            assert entered.wait(timeout=1)
            second.start()
            time.sleep(0.02)
            release.set()
            first.join(timeout=2)
            second.join(timeout=2)

        assert not first.is_alive() and not second.is_alive()
        assert errors == []
        assert results == [SAMPLE_REGISTRY, SAMPLE_REGISTRY]
        assert calls == 1
        gc.collect()
        assert all(loop_ref() is None for loop_ref in loops)

    def test_cancelled_owner_releases_cross_loop_waiter(self, tmp_path, monkeypatch):
        """Owner cancellation never strands a waiter on another event loop."""
        import agent.models_dev as md

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        owner_entered = threading.Event()
        cancel_owner = threading.Event()
        calls = 0
        calls_guard = threading.Lock()
        results = []
        errors = []

        async def fetch_network():
            nonlocal calls
            with calls_guard:
                calls += 1
                ordinal = calls
            if ordinal == 1:
                owner_entered.set()
                while not cancel_owner.is_set():
                    await asyncio.sleep(0.001)
                raise asyncio.CancelledError
            return SAMPLE_REGISTRY

        def owner_runner():
            try:
                asyncio.run(fetch_models_dev())
            except asyncio.CancelledError:
                results.append("cancelled")
            except BaseException as exc:
                errors.append(exc)

        def waiter_runner():
            try:
                results.append(asyncio.run(fetch_models_dev()))
            except BaseException as exc:
                errors.append(exc)

        with (
            patch.object(md, "_disk_cache_age_seconds", new=AsyncMock(return_value=None)),
            patch.object(md, "_load_disk_cache", new=AsyncMock(return_value={})),
            patch.object(md, "_save_disk_cache", new=AsyncMock()),
            patch.object(md, "_fetch_models_dev_from_network", new=fetch_network),
        ):
            owner = threading.Thread(target=owner_runner)
            waiter = threading.Thread(target=waiter_runner)
            owner.start()
            assert owner_entered.wait(timeout=1)
            waiter.start()
            time.sleep(0.02)
            cancel_owner.set()
            owner.join(timeout=2)
            waiter.join(timeout=2)

        assert not owner.is_alive() and not waiter.is_alive()
        assert errors == []
        assert "cancelled" in results
        assert SAMPLE_REGISTRY in results
        assert calls == 2


# ---------------------------------------------------------------------------
# get_model_capabilities — vision via modalities.input
# ---------------------------------------------------------------------------


CAPS_REGISTRY = {
    "google": {
        "id": "google",
        "models": {
            "gemma-4-31b-it": {
                "id": "gemma-4-31b-it",
                "attachment": False,
                "tool_call": True,
                "modalities": {"input": ["text", "image"]},
                "limit": {"context": 128000, "output": 8192},
            },
            "gemma-3-1b": {
                "id": "gemma-3-1b",
                "tool_call": True,
                "limit": {"context": 32000, "output": 8192},
            },
            "text-only-with-stale-attachment": {
                "id": "text-only-with-stale-attachment",
                "attachment": True,
                "tool_call": True,
                "modalities": {"input": ["text"]},
                "limit": {"context": 128000, "output": 8192},
            },
        },
    },
    "anthropic": {
        "id": "anthropic",
        "models": {
            "claude-sonnet-4": {
                "id": "claude-sonnet-4",
                "attachment": True,
                "tool_call": True,
                "limit": {"context": 200000, "output": 64000},
            },
        },
    },
}


class TestGetModelCapabilities:
    """Tests for get_model_capabilities vision detection."""

    @pytest.mark.asyncio
    async def test_vision_from_attachment_flag(self):
        """Models with attachment=True and no modalities should report supports_vision=True."""
        with patch("agent.models_dev.fetch_models_dev", return_value=CAPS_REGISTRY):
            caps = await get_model_capabilities("anthropic", "claude-sonnet-4")
        assert caps is not None
        assert caps.supports_vision is True




    @pytest.mark.asyncio
    async def test_modalities_non_dict_handled(self):
        """Non-dict modalities field should not crash."""
        registry = {
            "google": {"id": "google", "models": {
                "weird-model": {
                    "id": "weird-model",
                    "modalities": "text",  # not a dict
                    "limit": {"context": 200000, "output": 8192},
                },
            }},
        }
        with patch("agent.models_dev.fetch_models_dev", return_value=registry):
            caps = await get_model_capabilities("gemini", "weird-model")
        assert caps is not None
        assert caps.supports_vision is False
