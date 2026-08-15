"""Native-async port of upstream per-task auxiliary concurrency tests (#23324).

The upstream file also exercises a synchronous ``call_llm`` and a separate
``async_call_llm`` wrapper. This fork deliberately has one coroutine entry
point, so the same assertions are exercised through ``call_llm`` only; no
duplicate compatibility API is introduced.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from agent.auxiliary_client import (
    _acquire_async_aux_semaphore,
    _get_task_max_concurrency,
    _reset_aux_semaphores,
    call_llm,
)


@pytest.fixture(autouse=True)
def _clean_semaphore_cache():
    _reset_aux_semaphores()
    yield
    _reset_aux_semaphores()


class TestGetTaskMaxConcurrency:
    def test_returns_none_for_missing_task(self):
        assert _get_task_max_concurrency(None) is None
        assert _get_task_max_concurrency("") is None

    def test_returns_none_when_unset(self):
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config", return_value={}
        ):
            assert _get_task_max_concurrency("title_generation") is None

    def test_does_not_reuse_vision_cpu_limit_for_llm_calls(self):
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"max_concurrency": 1},
        ):
            assert _get_task_max_concurrency("vision") is None

    def test_returns_int_when_configured(self):
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"max_concurrency": 3},
        ):
            assert _get_task_max_concurrency("compression") == 3

    def test_returns_none_for_non_numeric(self):
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"max_concurrency": "not-a-number"},
        ):
            assert _get_task_max_concurrency("compression") is None

    def test_returns_none_for_zero_or_negative(self):
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"max_concurrency": 0},
        ):
            assert _get_task_max_concurrency("compression") is None
        with patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"max_concurrency": -2},
        ):
            assert _get_task_max_concurrency("compression") is None


class TestSemaphoreCache:
    @pytest.mark.asyncio
    async def test_async_reuses_semaphore_within_same_loop(self):
        config = {"auxiliary": {"compression": {"max_concurrency": 2}}}
        sem1 = _acquire_async_aux_semaphore("compression", config=config)
        sem2 = _acquire_async_aux_semaphore("compression", config=config)
        assert sem1 is sem2

    @pytest.mark.asyncio
    async def test_async_rebuilds_when_limit_changes(self):
        first = {"auxiliary": {"compression": {"max_concurrency": 2}}}
        second = {"auxiliary": {"compression": {"max_concurrency": 5}}}
        sem1 = _acquire_async_aux_semaphore("compression", config=first)
        sem2 = _acquire_async_aux_semaphore("compression", config=second)
        assert sem1 is not sem2

    def test_async_returns_none_with_no_running_loop(self):
        config = {"auxiliary": {"compression": {"max_concurrency": 2}}}
        assert _acquire_async_aux_semaphore("compression", config=config) is None


def _patch_call_setup(config: dict):
    return (
        patch(
            "providers._ensure_provider_profiles_loaded",
            new_callable=AsyncMock,
        ),
        patch(
            "hermes_cli.plugins.discover_plugins",
            new_callable=AsyncMock,
        ),
        patch(
            "hermes_cli.config.load_config_readonly",
            new_callable=AsyncMock,
            return_value=config,
        ),
    )


class TestCallLlmEnforcesLimit:
    @pytest.mark.asyncio
    async def test_call_llm_caps_concurrent_inflight(self):
        limit = 2
        n_callers = 6
        active = 0
        max_active = 0

        async def fake_impl(**_kwargs):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            try:
                await asyncio.sleep(0.02)
            finally:
                active -= 1
            return "ok"

        config = {"auxiliary": {"compression": {"max_concurrency": limit}}}
        setup = _patch_call_setup(config)
        with (
            patch("agent.auxiliary_client._call_llm_impl", side_effect=fake_impl),
            setup[0],
            setup[1],
            setup[2],
        ):
            results = await asyncio.gather(
                *[
                    call_llm(
                        task="compression",
                        messages=[{"role": "user", "content": "hi"}],
                    )
                    for _ in range(n_callers)
                ]
            )

        assert results == ["ok"] * n_callers
        assert max_active <= limit

    @pytest.mark.asyncio
    async def test_semaphore_released_on_exception(self):
        async def fail_impl(**_kwargs):
            raise RuntimeError("boom")

        config = {"auxiliary": {"compression": {"max_concurrency": 1}}}
        setup = _patch_call_setup(config)
        with (
            patch("agent.auxiliary_client._call_llm_impl", side_effect=fail_impl),
            setup[0],
            setup[1],
            setup[2],
        ):
            for _ in range(3):
                with pytest.raises(RuntimeError, match="boom"):
                    await call_llm(
                        task="compression",
                        messages=[{"role": "user", "content": "hi"}],
                    )

    @pytest.mark.asyncio
    async def test_unlimited_when_not_configured(self):
        entered = 0

        async def fake_impl(**_kwargs):
            nonlocal entered
            entered += 1
            return "ok"

        config = {"auxiliary": {"compression": {}}}
        setup = _patch_call_setup(config)
        with (
            patch("agent.auxiliary_client._call_llm_impl", side_effect=fake_impl),
            setup[0],
            setup[1],
            setup[2],
        ):
            await call_llm(
                task="compression",
                messages=[{"role": "user", "content": "hi"}],
            )
        assert entered == 1

    @pytest.mark.asyncio
    async def test_stream_options_and_api_mode_are_forwarded(self):
        seen = {}

        async def fake_impl(**kwargs):
            seen.update(kwargs)
            return "ok"

        config = {"auxiliary": {"compression": {"max_concurrency": 1}}}
        setup = _patch_call_setup(config)
        with (
            patch("agent.auxiliary_client._call_llm_impl", side_effect=fake_impl),
            setup[0],
            setup[1],
            setup[2],
        ):
            await call_llm(
                task="compression",
                messages=[{"role": "user", "content": "first"}],
                stream=True,
                stream_options={"include_usage": True},
                api_mode="codex_responses",
            )

        assert seen["stream"] is True
        assert seen["stream_options"] == {"include_usage": True}
        assert seen["api_mode"] == "codex_responses"
