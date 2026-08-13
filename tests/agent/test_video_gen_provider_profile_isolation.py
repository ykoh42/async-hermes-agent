"""Profile-scoped credentials for the shared video-generation adapter."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from agent import secret_scope
from agent.video_gen_provider import OpenAICompatibleVideoGenProvider


class _Provider(OpenAICompatibleVideoGenProvider):
    name = "profile-video"
    _env_key = "PROFILE_VIDEO_API_KEY"
    _default_base_url = "https://default.video.test/v1"

    async def list_models(self):
        return [{"id": "profile-model"}]


@pytest.fixture(autouse=True)
def _restore_secret_scope():
    previous = secret_scope.is_multiplex_active()
    token = secret_scope.set_secret_scope(None)
    yield
    secret_scope.reset_secret_scope(token)
    secret_scope.set_multiplex_active(previous)


@pytest.mark.asyncio
async def test_video_provider_concurrent_profiles_use_own_credentials(
    monkeypatch,
):
    created: list[tuple[str, str]] = []
    closed: list[str] = []
    both_started = asyncio.Event()

    class Client:
        def __init__(self, api_key):
            self.api_key = api_key
            self.videos = SimpleNamespace()

        async def close(self):
            closed.append(self.api_key)

    async def create_client(_client_class, *, api_key, base_url):
        created.append((api_key, base_url))
        if len(created) == 2:
            both_started.set()
        return Client(api_key)

    async def create_and_poll(self, client, _call_kwargs):
        await both_started.wait()
        return SimpleNamespace(
            id=f"video-{client.api_key}",
            status="failed",
            error=f"failure-{client.api_key}",
            data=[],
        )

    async def generate(api_key, base_url):
        token = secret_scope.set_secret_scope({
            "PROFILE_VIDEO_API_KEY": api_key,
            "PROFILE-VIDEO_BASE_URL": base_url,
        })
        try:
            return await _Provider().generate("profile prompt")
        finally:
            secret_scope.reset_secret_scope(token)

    fake_openai = SimpleNamespace(AsyncOpenAI=object)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setattr(
        "agent.ssl_verify._create_openai_sdk_client",
        create_client,
    )
    monkeypatch.setattr(_Provider, "_create_and_poll", create_and_poll)
    monkeypatch.setenv("PROFILE_VIDEO_API_KEY", "process-profile-key")
    monkeypatch.setenv(
        "PROFILE-VIDEO_BASE_URL",
        "https://process-profile.video.test/v1",
    )
    secret_scope.set_multiplex_active(True)

    result_a, result_b = await asyncio.gather(
        generate("profile-a-key", "https://profile-a.video.test/v1"),
        generate("profile-b-key", "https://profile-b.video.test/v1"),
    )

    assert created == [
        ("profile-a-key", "https://profile-a.video.test/v1"),
        ("profile-b-key", "https://profile-b.video.test/v1"),
    ]
    assert result_a["error"] == "failure-profile-a-key"
    assert result_b["error"] == "failure-profile-b-key"
    assert sorted(closed) == ["profile-a-key", "profile-b-key"]


@pytest.mark.asyncio
async def test_video_provider_missing_scoped_key_does_not_borrow_process_env(
    monkeypatch,
):
    monkeypatch.setenv("PROFILE_VIDEO_API_KEY", "process-profile-key")
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({})
    try:
        provider = _Provider()
        assert await provider.is_available() is False
        result = await provider.generate("profile prompt", model="profile-model")
    finally:
        secret_scope.reset_secret_scope(token)

    assert result["success"] is False
    assert result["error_type"] == "missing_credentials"


@pytest.mark.asyncio
async def test_video_provider_closes_client_when_cancelled(monkeypatch):
    started = asyncio.Event()
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    closed = asyncio.Event()

    class Client:
        videos = SimpleNamespace()

        async def close(self):
            close_started.set()
            await release_close.wait()
            closed.set()

    async def create_client(_client_class, **_kwargs):
        return Client()

    async def create_and_poll(_self, _client, _call_kwargs):
        started.set()
        await asyncio.Future()

    fake_openai = SimpleNamespace(AsyncOpenAI=object)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setattr(
        "agent.ssl_verify._create_openai_sdk_client",
        create_client,
    )
    monkeypatch.setattr(_Provider, "_create_and_poll", create_and_poll)
    token = secret_scope.set_secret_scope({
        "PROFILE_VIDEO_API_KEY": "cancel-key",
    })
    try:
        task = asyncio.create_task(_Provider().generate("profile prompt"))
    finally:
        secret_scope.reset_secret_scope(token)
    await started.wait()

    task.cancel()
    await close_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    try:
        assert task.done() is False
    finally:
        release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert closed.is_set()
