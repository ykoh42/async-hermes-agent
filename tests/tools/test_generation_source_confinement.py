"""Source-image confinement tests ported to the native async boundary."""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import tools.image_generation_tool as igt

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@pytest.mark.asyncio
async def test_local_backend_is_passthrough(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    url, refs, error = await igt._confine_source_images(
        "/some/host/pic.png", ["/other/ref.png"], "t1"
    )
    assert (url, refs, error) == (
        "/some/host/pic.png",
        ["/other/ref.png"],
        None,
    )


@pytest.mark.asyncio
async def test_urls_pass_through_under_sandbox(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    url, refs, error = await igt._confine_source_images(
        "https://x/y.png", ["data:image/png;base64,AAAA"], "t1"
    )
    assert (url, refs, error) == (
        "https://x/y.png",
        ["data:image/png;base64,AAAA"],
        None,
    )


@pytest.mark.asyncio
async def test_path_resolves_to_data_url_under_nonlocal_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    source = tmp_path / "pic.png"
    source.write_bytes(PNG)
    async def execute(_command, **_kwargs):
        return {
            "returncode": 0,
            "output": base64.b64encode(PNG).decode("ascii"),
        }

    monkeypatch.setattr(
        "tools.image_source._get_active_env",
        lambda _task_id: SimpleNamespace(execute=execute),
    )
    url, refs, error = await igt._confine_source_images(str(source), None, "t1")
    assert error is None
    assert refs is None
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == PNG


@pytest.mark.asyncio
async def test_unreadable_path_returns_error_without_host_bytes(monkeypatch, tmp_path):
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    secret = tmp_path / "id_rsa"
    secret.write_bytes(b"HOST-PRIVATE-KEY")
    monkeypatch.setattr(
        "tools.image_source._permitted_host_read_target",
        AsyncMock(return_value=tmp_path / "missing.png"),
    )
    _url, _refs, error = await igt._confine_source_images(str(secret), None, "t1")
    assert error is not None
    payload = json.loads(error)
    assert payload["success"] is False
    assert "Could not read source image" in payload["error"]
    assert "HOST-PRIVATE-KEY" not in error


@pytest.mark.asyncio
async def test_image_handler_rejects_before_provider_dispatch(monkeypatch, tmp_path):
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    dispatched = AsyncMock(return_value=None)
    monkeypatch.setattr(igt, "_dispatch_to_plugin_provider", dispatched)
    out = await igt._handle_image_generate(
        {"prompt": "edit it", "image_url": str(tmp_path / "nope.png")},
        task_id="t1",
    )
    payload = json.loads(out)
    assert payload["success"] is False
    dispatched.assert_not_awaited()


@pytest.mark.asyncio
async def test_video_handler_uses_same_chokepoint(monkeypatch, tmp_path):
    import tools.video_generation_tool as vgt

    monkeypatch.setenv("TERMINAL_ENV", "docker")
    provider = object()
    monkeypatch.setattr(vgt, "_resolve_active_provider", AsyncMock(return_value=provider))
    out = await vgt._handle_video_generate(
        {"prompt": "animate", "image_url": str(tmp_path / "nope.png")},
        task_id="t1",
    )
    payload = json.loads(out)
    assert payload["success"] is False
    assert "Could not read source image" in payload["error"]
