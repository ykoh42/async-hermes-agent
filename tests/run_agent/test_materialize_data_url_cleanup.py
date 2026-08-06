"""Regression coverage for async vision fallback data-URL materialization."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import aiofiles
import aiofiles.os
import pytest
from blockbuster import BlockBuster

from run_agent import AIAgent


async def _list_anthropic_tmpfiles(tmpdir: Path) -> list[str]:
    return [
        name
        for name in await aiofiles.os.listdir(tmpdir)
        if name.startswith("anthropic_image_")
    ]


@pytest.mark.asyncio
async def test_b64decode_failure_does_not_leak_tempfile(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    bad_url = "data:image/png;base64,!!!not-valid-base64!!!"
    with pytest.raises(Exception):
        await AIAgent._materialize_data_url_for_vision(bad_url)

    leftovers = await _list_anthropic_tmpfiles(tmp_path)
    assert leftovers == [], f"leaked temp files after decode failure: {leftovers}"


@pytest.mark.asyncio
async def test_successful_decode_returns_path_to_existing_file(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    payload = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    encoded = base64.b64encode(payload).decode("ascii")
    good_url = f"data:image/png;base64,{encoded}"

    blockbuster = BlockBuster()
    blockbuster.activate()
    path_obj: Path | None = None
    try:
        path_str, path_obj = await AIAgent._materialize_data_url_for_vision(
            good_url
        )
        assert isinstance(path_obj, Path)
        async with aiofiles.open(path_obj, "rb") as materialized:
            assert await materialized.read() == payload
        assert path_str == str(path_obj)
    finally:
        blockbuster.deactivate()
        if path_obj is not None:
            await aiofiles.os.remove(path_obj)
