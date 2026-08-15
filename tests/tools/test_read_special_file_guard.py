"""Native-async guards for FIFO, socket, and device-like file reads."""

from __future__ import annotations

import json
import os
import socket
import uuid
from pathlib import Path

import pytest

from tools.file_tools import _special_file_kind, read_file_tool


@pytest.mark.asyncio
async def test_regular_directory_and_missing_paths_are_not_special(tmp_path):
    regular = tmp_path / "regular.txt"
    regular.write_text("hello")
    assert await _special_file_kind(regular) is None
    assert await _special_file_kind(tmp_path) is None
    assert await _special_file_kind(tmp_path / "missing") is None


@pytest.mark.asyncio
async def test_fifo_is_rejected_before_read(tmp_path):
    fifo = tmp_path / "live.pipe"
    os.mkfifo(fifo)
    assert await _special_file_kind(fifo) == "a FIFO (named pipe)"
    result = json.loads(await read_file_tool(str(fifo)))
    assert result["success"] is False
    assert "FIFO" in result["note"]


@pytest.mark.asyncio
async def test_socket_is_rejected_before_read(tmp_path):
    # AF_UNIX paths are capped at roughly 104 bytes on macOS; pytest's
    # temporary path can exceed that before the socket name is appended.
    socket_path = Path("/tmp") / f"hermes-special-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    server = socket.socket(socket.AF_UNIX)
    try:
        server.bind(str(socket_path))
        assert await _special_file_kind(socket_path) == "a socket"
        result = json.loads(await read_file_tool(str(socket_path)))
        assert result["success"] is False
        assert "socket" in result["note"]
    finally:
        server.close()
        socket_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_symlink_to_fifo_is_rejected(tmp_path):
    fifo = tmp_path / "live.pipe"
    os.mkfifo(fifo)
    alias = tmp_path / "alias"
    alias.symlink_to(fifo)
    result = json.loads(await read_file_tool(str(alias)))
    assert result["success"] is False
    assert "FIFO" in result["note"]


@pytest.mark.asyncio
async def test_regular_file_is_still_read(tmp_path):
    path = tmp_path / "normal.txt"
    path.write_text("hello\n")
    result = json.loads(await read_file_tool(str(path)))
    assert "hello" in result["content"]
