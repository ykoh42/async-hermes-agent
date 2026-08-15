"""Native-async image-source resolver tests ported from upstream."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tools import image_source as source

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


@pytest.fixture(autouse=True)
def _skip_real_sandbox_bringup(monkeypatch):
    monkeypatch.setattr(source, "_ensure_container_env", AsyncMock())


@pytest.mark.asyncio
async def test_data_url_and_non_image_rejection():
    encoded = base64.b64encode(PNG).decode("ascii")
    resolved = await source.resolve_image_source(
        f"data:image/png;base64,{encoded}", source.ResolveContext()
    )
    assert resolved.data == PNG
    assert resolved.mime == "image/png"
    assert resolved.origin == "data"
    with pytest.raises(source.NotAnImage):
        await source.resolve_image_source(
            "data:text/plain;base64," + base64.b64encode(b"not image").decode(),
            source.ResolveContext(),
        )


@pytest.mark.asyncio
async def test_local_path_relative_path_and_svg(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    image = tmp_path / "pic.png"
    image.write_bytes(PNG)
    resolved = await source.resolve_image_source(str(image), source.ResolveContext())
    assert resolved.data == PNG
    monkeypatch.chdir(tmp_path)
    relative = await source.resolve_image_source("pic.png", source.ResolveContext())
    assert relative.origin == "file"

    svg = tmp_path / "art.svg"
    svg_bytes = b'<svg xmlns="http://www.w3.org/2000/svg"></svg>'
    svg.write_bytes(svg_bytes)
    svg_result = await source.resolve_image_source(str(svg), source.ResolveContext())
    assert svg_result.mime == "image/svg+xml"
    assert svg_result.data == svg_bytes


@pytest.mark.asyncio
async def test_nonlocal_media_cache_is_the_only_host_read(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path / "hermes")
    cached = tmp_path / "hermes" / "cache" / "images" / "inbound.png"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(PNG)
    resolved = await source.resolve_image_source(str(cached), source.ResolveContext())
    assert resolved.origin == "file"
    assert resolved.data == PNG


@pytest.mark.asyncio
async def test_nonlocal_uncached_path_fails_closed_without_native_sandbox(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    secret = tmp_path / "id_rsa"
    secret.write_bytes(b"HOST-PRIVATE-KEY")
    monkeypatch.setattr(source, "_get_active_env", lambda _task_id: None)
    with pytest.raises(source.SourceNotFound):
        await source.resolve_image_source(
            str(secret), source.ResolveContext(task_id="task-1")
        )


@pytest.mark.asyncio
async def test_container_read_is_bounded_and_retries_cold_start(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    attempts = 0
    encoded = base64.b64encode(PNG).decode("ascii")

    async def execute(command, **kwargs):
        nonlocal attempts
        assert f"head -c {source._MAX_INGEST_BYTES + 1}" in command
        assert " < " in command
        assert kwargs["timeout"] == 30
        attempts += 1
        if attempts == 1:
            return {"returncode": 1, "output": "cold start"}
        return {"returncode": 0, "output": encoded}

    monkeypatch.setattr(
        source,
        "_get_active_env",
        lambda _task_id: SimpleNamespace(execute=execute),
    )
    result = await source.resolve_image_source(
        "/workspace/picture.png", source.ResolveContext(task_id="task-1")
    )
    assert result.origin == "container"
    assert result.data == PNG
    assert attempts == 2


@pytest.mark.asyncio
async def test_container_failure_surfaces_diagnostic(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")

    async def execute(_command, **_kwargs):
        return {"returncode": 1, "output": "permission denied"}

    monkeypatch.setattr(
        source,
        "_get_active_env",
        lambda _task_id: SimpleNamespace(execute=execute),
    )
    with pytest.raises(source.SourceNotFound, match="permission denied"):
        await source.resolve_image_source(
            "/workspace/missing.png", source.ResolveContext(task_id="task-1")
        )


@pytest.mark.asyncio
async def test_container_requires_native_async_execute(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setattr(
        source,
        "_get_active_env",
        lambda _task_id: SimpleNamespace(execute=lambda *_a, **_k: {}),
    )
    with pytest.raises(RuntimeError, match="native async"):
        await source.resolve_image_source(
            "/workspace/missing.png", source.ResolveContext(task_id="task-1")
        )
