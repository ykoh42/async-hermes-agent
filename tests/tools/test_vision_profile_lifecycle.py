"""Profile routing and owned probe-client lifecycle for vision tools."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from agent.secret_scope import (
    UnscopedSecretError,
    is_multiplex_active,
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import vision_tools
from tools.vision_tools import (
    _handle_video_analyze,
    _handle_vision_analyze,
    check_vision_requirements,
)


@pytest.fixture(autouse=True)
def _restore_multiplex_state():
    previous = is_multiplex_active()
    try:
        yield
    finally:
        set_multiplex_active(previous)


@pytest.mark.asyncio
async def test_concurrent_profiles_isolate_vision_and_video_model_overrides(
    monkeypatch,
):
    monkeypatch.setenv("AUXILIARY_VISION_MODEL", "process-vision")
    monkeypatch.setenv("AUXILIARY_VIDEO_MODEL", "process-video")
    set_multiplex_active(True)

    async def echo_vision(_url, _prompt, model, task_id=None):
        await asyncio.sleep(0)
        return (model, task_id)

    async def echo_video(_url, _prompt, model):
        await asyncio.sleep(0)
        return model

    async def resolve(name: str):
        token = set_secret_scope(
            {
                "AUXILIARY_VISION_MODEL": f"vision-{name}",
                "AUXILIARY_VIDEO_MODEL": f"video-{name}",
            }
        )
        try:
            vision = await _handle_vision_analyze(
                {"image_url": "image.png", "question": name},
                task_id=f"task-{name}",
            )
            video = await _handle_video_analyze(
                {"video_url": "video.mp4", "question": name}
            )
            return vision, video
        finally:
            reset_secret_scope(token)

    with (
        patch(
            "tools.vision_tools._should_use_native_vision_fast_path",
            AsyncMock(return_value=False),
        ),
        patch(
            "hermes_cli.config.load_config_readonly",
            AsyncMock(return_value={}),
        ),
        patch("tools.vision_tools.vision_analyze_tool", side_effect=echo_vision),
        patch("tools.vision_tools.video_analyze_tool", side_effect=echo_video),
    ):
        alpha, beta = await asyncio.gather(resolve("alpha"), resolve("beta"))

    assert alpha == (("vision-alpha", "task-alpha"), "video-alpha")
    assert beta == (("vision-beta", "task-beta"), "video-beta")


@pytest.mark.asyncio
async def test_video_empty_override_preserves_vision_fallback(monkeypatch):
    monkeypatch.setenv("AUXILIARY_VIDEO_MODEL", "process-video")
    monkeypatch.setenv("AUXILIARY_VISION_MODEL", "process-vision")
    set_multiplex_active(True)
    token = set_secret_scope(
        {
            "AUXILIARY_VIDEO_MODEL": "",
            "AUXILIARY_VISION_MODEL": "profile-vision",
        }
    )
    try:
        with (
            patch(
                "hermes_cli.config.load_config_readonly",
                AsyncMock(return_value={}),
            ),
            patch(
                "tools.vision_tools.video_analyze_tool",
                AsyncMock(return_value="ok"),
            ) as analyze,
        ):
            assert await _handle_video_analyze(
                {"video_url": "video.mp4", "question": "question"}
            ) == "ok"
    finally:
        reset_secret_scope(token)

    assert analyze.await_args.args[2] == "profile-vision"


@pytest.mark.asyncio
async def test_unscoped_model_override_fails_closed(monkeypatch):
    monkeypatch.setenv("AUXILIARY_VISION_MODEL", "process-vision")
    set_multiplex_active(True)

    with (
        patch(
            "tools.vision_tools._should_use_native_vision_fast_path",
            AsyncMock(return_value=False),
        ),
        patch(
            "hermes_cli.config.load_config_readonly",
            AsyncMock(return_value={}),
        ),
        pytest.raises(UnscopedSecretError, match="AUXILIARY_VISION_MODEL"),
    ):
        await _handle_vision_analyze(
            {"image_url": "image.png", "question": "question"}
        )


@pytest.mark.asyncio
async def test_debug_sessions_are_scoped_to_active_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("VISION_TOOLS_DEBUG", "true")

    async def resolve(name: str):
        home = tmp_path / name
        token = set_hermes_home_override(home)
        try:
            debug = vision_tools._active_debug_session()
            debug.log_call("profile", {"name": name})
            await asyncio.sleep(0)
            return debug, debug.log_dir, list(debug._calls)
        finally:
            reset_hermes_home_override(token)

    alpha, beta = await asyncio.gather(resolve("alpha"), resolve("beta"))
    assert alpha[0] is not beta[0]
    assert alpha[1] == tmp_path / "alpha" / "logs"
    assert beta[1] == tmp_path / "beta" / "logs"
    assert alpha[2][0]["name"] == "alpha"
    assert beta[2][0]["name"] == "beta"


@pytest.mark.asyncio
async def test_requirement_probe_closes_owned_client_before_returning():
    closed = 0

    class ProbeClient:
        async def aclose(self):
            nonlocal closed
            closed += 1

    resolve = AsyncMock(return_value=("openrouter", ProbeClient(), "vision-model"))
    with patch(
        "agent.auxiliary_client.resolve_vision_provider_client",
        resolve,
    ):
        assert await check_vision_requirements() is True

    assert closed == 1
    resolve.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_requirement_probe_finishes_close_through_repeated_cancellation():
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    close_finished = asyncio.Event()

    class ProbeClient:
        async def aclose(self):
            close_started.set()
            await release_close.wait()
            close_finished.set()

    with patch(
        "agent.auxiliary_client.resolve_vision_provider_client",
        AsyncMock(return_value=("openrouter", ProbeClient(), "vision-model")),
    ):
        probe = asyncio.create_task(check_vision_requirements())
        await close_started.wait()
        probe.cancel()
        await asyncio.sleep(0)
        probe.cancel()
        assert not probe.done()
        release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await probe

    assert close_finished.is_set()


@pytest.mark.asyncio
async def test_requirement_probe_does_not_swallow_unscoped_secret_error():
    error = UnscopedSecretError("missing vision profile scope")
    with patch(
        "agent.auxiliary_client.resolve_vision_provider_client",
        AsyncMock(side_effect=error),
    ):
        with pytest.raises(UnscopedSecretError, match="missing vision profile"):
            await check_vision_requirements()
