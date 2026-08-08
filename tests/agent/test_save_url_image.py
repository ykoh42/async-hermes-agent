"""Real-loop HTTP coverage for ``save_url_image``."""

from __future__ import annotations

import asyncio
from urllib.parse import urlsplit

import aiofiles
import aiofiles.os
import httpx
import pytest
import pytest_asyncio

from agent.image_gen_provider import _images_cache_dir, save_url_image

pytestmark = pytest.mark.asyncio

PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108020000009077"
    "53de00000010494441547801635c0e000000feff03000006000557bfabd400"
    "00000049454e44ae426082"
)


@pytest_asyncio.fixture
async def image_server(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        request_line = await reader.readline()
        target = request_line.decode("ascii", "replace").split(" ", 2)[1]
        while await reader.readline() not in {b"\r\n", b""}:
            pass

        route = urlsplit(target).path
        if route == "/image.png":
            status = "200 OK"
            content_type = "image/png"
            body = PNG_1PX
        elif route == "/empty":
            status = "200 OK"
            content_type = "image/png"
            body = b""
        elif route == "/oversize":
            status = "200 OK"
            content_type = "image/png"
            body = b"x" * 4096
        else:
            status = "404 Not Found"
            content_type = "text/plain"
            body = b"missing"

        writer.write(
            (
                f"HTTP/1.1 {status}\r\n"
                f"Content-Type: {content_type}\r\n"
                f"Content-Length: {len(body)}\r\n"
                "Connection: close\r\n\r\n"
            ).encode("ascii")
            + body
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.close()
        await server.wait_closed()


async def test_writes_real_bytes_to_profile_cache(image_server):
    path = await save_url_image(f"{image_server}/image.png", prefix="xai_test")
    async with aiofiles.open(path, "rb") as handle:
        assert await handle.read() == PNG_1PX
    assert "cache/images" in str(path)
    assert path.suffix == ".png"


async def test_http_error_propagates(image_server):
    with pytest.raises(httpx.HTTPStatusError):
        await save_url_image(f"{image_server}/missing")


async def test_empty_and_oversize_downloads_leave_no_partial_file(image_server):
    cache_dir = await _images_cache_dir()
    before = set(await aiofiles.os.listdir(cache_dir))

    with pytest.raises(ValueError, match="0 bytes"):
        await save_url_image(f"{image_server}/empty")
    with pytest.raises(ValueError, match="exceeds"):
        await save_url_image(f"{image_server}/oversize", max_bytes=1024)

    assert set(await aiofiles.os.listdir(cache_dir)) == before
