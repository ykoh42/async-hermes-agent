"""Async artifact handoff tests for the retained image-generation tool.

These cases are the native-async equivalent of the upstream artifact tests:
the generated host cache path stays in the response while remote environments
receive an agent-visible path and an awaited forced sync.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_postprocess_adds_agent_visible_image_for_remote_env(
    monkeypatch, tmp_path
):
    from tools import image_generation_tool

    hermes_home = tmp_path / ".hermes"
    image_dir = hermes_home / "cache" / "images"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "generated.jpg"
    image_path.write_bytes(b"jpg")

    sync_calls: list[bool] = []

    class FakeSyncManager:
        async def sync(self, *, force=False):
            sync_calls.append(force)

    env = SimpleNamespace(
        _remote_home="/home/remotesshuser",
        _sync_manager=FakeSyncManager(),
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(image_generation_tool, "_active_terminal_env", lambda _task: env)

    raw = json.dumps({"success": True, "image": str(image_path)})
    result = json.loads(
        await image_generation_tool._postprocess_image_generate_result(
            raw, task_id="task-1"
        )
    )

    assert result["image"] == str(image_path)
    assert result["host_image"] == str(image_path)
    assert result["agent_visible_image"] == (
        "/home/remotesshuser/.hermes/cache/images/generated.jpg"
    )
    assert sync_calls == [True]


@pytest.mark.asyncio
async def test_postprocess_leaves_local_url_and_error_payloads_unchanged(
    monkeypatch,
):
    from tools import image_generation_tool

    monkeypatch.setattr(
        image_generation_tool,
        "_active_terminal_env",
        lambda _task: pytest.fail("remote env should not be inspected"),
    )

    url = json.dumps({"success": True, "image": "https://example.test/x.png"})
    error = json.dumps({"success": False, "image": None, "error": "bad"})
    assert await image_generation_tool._postprocess_image_generate_result(url) == url
    assert await image_generation_tool._postprocess_image_generate_result(error) == error


@pytest.mark.asyncio
async def test_handle_image_generate_postprocesses_plugin_result(monkeypatch, tmp_path):
    from tools import image_generation_tool

    hermes_home = tmp_path / ".hermes"
    image_dir = hermes_home / "cache" / "images"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "plugin.png"
    image_path.write_bytes(b"png")
    env = SimpleNamespace(_remote_home="/home/remote", _sync_manager=None)

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(image_generation_tool, "_active_terminal_env", lambda _task: env)
    monkeypatch.setattr(
        image_generation_tool,
        "_dispatch_to_plugin_provider",
        AsyncMock(
            return_value=json.dumps({"success": True, "image": str(image_path)})
        ),
    )

    result = json.loads(
        await image_generation_tool._handle_image_generate(
            {"prompt": "draw", "aspect_ratio": "square"},
            task_id="plugin-task",
        )
    )
    assert result["agent_visible_image"] == "/home/remote/.hermes/cache/images/plugin.png"
