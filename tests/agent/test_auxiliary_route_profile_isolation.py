"""Concurrency regressions for auxiliary provider request shaping."""

from __future__ import annotations

import asyncio
import contextvars

import pytest

import agent.auxiliary_client as auxiliary
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.mark.asyncio
async def test_concurrent_auto_routes_do_not_share_nous_request_metadata(
    monkeypatch,
):
    route_label: contextvars.ContextVar[str] = contextvars.ContextVar(
        "test_auxiliary_route_label",
        default="",
    )
    nous_selected = asyncio.Event()
    other_selected = asyncio.Event()

    async def no_task_fallback(*_args, **_kwargs):
        return None, None, None

    async def select_route():
        if route_label.get() == "nous":
            auxiliary._set_auxiliary_is_nous(True)
            nous_selected.set()
            await other_selected.wait()
        else:
            await nous_selected.wait()
            auxiliary._set_auxiliary_is_nous(False)
            other_selected.set()
        return object(), "route-model"

    monkeypatch.setattr(
        auxiliary,
        "_try_configured_fallback_chain",
        no_task_fallback,
    )
    monkeypatch.setattr(
        auxiliary,
        "_try_main_fallback_chain",
        no_task_fallback,
    )
    monkeypatch.setattr(
        auxiliary,
        "_get_provider_chain",
        lambda: [("test-route", select_route)],
    )
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    async def resolve(label: str) -> tuple[bool, dict]:
        token = route_label.set(label)
        try:
            client, model = await auxiliary._resolve_auto(config={})
            assert client is not None
            assert model == "route-model"
            return (
                auxiliary._is_auxiliary_nous_route(),
                auxiliary.get_auxiliary_extra_body(),
            )
        finally:
            route_label.reset(token)

    nous_result, other_result = await asyncio.gather(
        resolve("nous"),
        resolve("other"),
    )

    assert nous_result[0] is True
    assert nous_result[1] == auxiliary._nous_extra_body()
    assert other_result == (False, {})


def test_public_auxiliary_is_nous_snapshot_remains_compatible(monkeypatch):
    token = auxiliary._AUXILIARY_IS_NOUS_CONTEXT.set(None)
    try:
        monkeypatch.setattr(auxiliary, "auxiliary_is_nous", True)
        assert auxiliary.get_auxiliary_extra_body() == auxiliary._nous_extra_body()
    finally:
        auxiliary._AUXILIARY_IS_NOUS_CONTEXT.reset(token)


@pytest.mark.asyncio
async def test_payment_health_cache_is_scoped_to_profile(tmp_path):
    marked = asyncio.Event()
    checked = asyncio.Event()

    async def profile_a() -> tuple[bool, bool]:
        token = set_hermes_home_override(tmp_path / "profile-a")
        try:
            auxiliary._reset_aux_unhealthy_cache()
            auxiliary._mark_provider_unhealthy("openrouter", ttl=60)
            marked.set()
            await checked.wait()
            return (
                auxiliary._is_provider_unhealthy("openrouter"),
                "openrouter" in auxiliary._aux_unhealthy_until,
            )
        finally:
            auxiliary._reset_aux_unhealthy_cache()
            reset_hermes_home_override(token)

    async def profile_b() -> tuple[bool, bool]:
        token = set_hermes_home_override(tmp_path / "profile-b")
        try:
            await marked.wait()
            result = (
                auxiliary._is_provider_unhealthy("openrouter"),
                "openrouter" in auxiliary._aux_unhealthy_until,
            )
            checked.set()
            return result
        finally:
            auxiliary._reset_aux_unhealthy_cache()
            reset_hermes_home_override(token)

    a_result, b_result = await asyncio.gather(profile_a(), profile_b())

    assert a_result == (True, True)
    assert b_result == (False, False)
