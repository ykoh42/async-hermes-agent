"""Tests for MCP ResourceLink / EmbeddedResource / AudioContent handling.

Regression coverage for a customer report (2026-07): non-image binary
resources returned through MCP resource blocks were silently dropped from
tool results, so a PDF-returning MCP tool appeared to return metadata only.
"""

import asyncio
import base64
import json
from types import SimpleNamespace

import pytest
import pytest_asyncio


PDF_BYTES = b"%PDF-1.4 fake pdf payload for tests"


def _blob_resource(data: bytes, uri="slack://files/F123/report.pdf", mime="application/pdf"):
    return SimpleNamespace(
        uri=uri,
        mimeType=mime,
        blob=base64.b64encode(data).decode("ascii"),
        text=None,
    )


def _embedded(resource):
    return SimpleNamespace(type="resource", resource=resource)


@pytest.fixture()
def doc_cache(tmp_path, monkeypatch):
    """Point the document cache at a temp dir."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path / "cache" / "documents"


class TestRenderResourceBlock:
    @pytest.mark.asyncio
    async def test_embedded_pdf_blob_is_materialized(self, doc_cache):
        from tools.mcp_tool import _render_mcp_resource_block

        out = await _render_mcp_resource_block(
            _embedded(_blob_resource(PDF_BYTES)), "slack"
        )
        assert "saved to" in out
        assert "application/pdf" in out
        # Extract path and verify bytes round-trip
        path = out.split("saved to ", 1)[1].split(" (", 1)[0]
        with open(path, "rb") as fh:
            assert fh.read() == PDF_BYTES
        assert "report.pdf" in path


    @pytest.mark.asyncio
    async def test_malformed_base64_fails_explicitly(self):
        from tools.mcp_tool import _render_mcp_resource_block

        res = SimpleNamespace(uri="x://y", mimeType="application/pdf", blob="!!!not-base64!!!", text=None)
        out = await _render_mcp_resource_block(_embedded(res), "srv")
        assert "could not be decoded" in out

    @pytest.mark.asyncio
    async def test_non_resource_block_returns_empty(self):
        from tools.mcp_tool import _render_mcp_resource_block

        assert (
            await _render_mcp_resource_block(
                SimpleNamespace(type="text", text="hi"), "srv"
            )
            == ""
        )

    @pytest.mark.asyncio
    async def test_path_traversal_uri_is_neutralized(self, doc_cache):
        from tools.mcp_tool import _render_mcp_resource_block

        res = _blob_resource(PDF_BYTES, uri="evil://host/../../etc/passwd")
        out = await _render_mcp_resource_block(_embedded(res), "srv")
        assert "saved to" in out
        path = out.split("saved to ", 1)[1].split(" (", 1)[0]
        assert str(doc_cache) in path
        assert "/etc/passwd" not in path


class TestResourceFilename:
    def test_uri_last_segment_used(self):
        from tools.mcp_tool import _mcp_resource_filename

        assert _mcp_resource_filename("slack://f/ABC/quarterly.pdf", "application/pdf") == "quarterly.pdf"


    def test_long_filename_capped_preserving_extension(self):
        from tools.mcp_tool import _mcp_resource_filename

        name = _mcp_resource_filename("x://h/" + "a" * 500 + ".pdf", "application/pdf")
        assert len(name) <= 150
        assert name.endswith(".pdf")


class TestPreDecodeSizeCap:
    @pytest.mark.asyncio
    async def test_oversized_b64_rejected_before_decode(self, monkeypatch):
        import tools.mcp_tool as m

        monkeypatch.setattr(m, "_MCP_RESOURCE_MAX_B64_CHARS", 16)
        res = SimpleNamespace(
            uri="x://y/big.pdf", mimeType="application/pdf",
            blob="A" * 100, text=None,
        )
        called = []
        monkeypatch.setattr(base64, "b64decode", lambda *a, **k: called.append(1))
        out = await m._render_mcp_resource_block(_embedded(res), "srv")
        assert "too large" in out
        assert not called


class TestAudioBlock:
    @pytest.mark.asyncio
    async def test_non_audio_returns_empty(self):
        from tools.mcp_tool import _cache_mcp_audio_block

        block = SimpleNamespace(data=base64.b64encode(b"x").decode(), mimeType="application/pdf")
        assert await _cache_mcp_audio_block(block) == ""

    @pytest.mark.asyncio
    async def test_audio_block_cached_as_media(self, tmp_path, monkeypatch):
        from tools.mcp_tool import _cache_mcp_audio_block

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        block = SimpleNamespace(
            data=base64.b64encode(b"RIFFfakewav").decode(),
            mimeType="audio/wav",
        )
        out = await _cache_mcp_audio_block(block)
        assert out.startswith("MEDIA:")


class TestToolResultLoopOrdering:
    @pytest.mark.asyncio
    async def test_mixed_blocks_preserve_order(self, doc_cache):
        """Simulate the tool-result block loop with text + pdf resource."""
        from tools.mcp_tool import (
            _cache_mcp_image_block,
            _cache_mcp_audio_block,
            _render_mcp_resource_block,
        )

        blocks = [
            SimpleNamespace(type="text", text="File ID: F123\nMIME Type: application/pdf"),
            _embedded(_blob_resource(PDF_BYTES)),
        ]
        parts = []
        for block in blocks:
            if getattr(block, "text", None):
                parts.append(block.text)
                continue
            tag = await _cache_mcp_image_block(block)
            if not tag:
                tag = await _cache_mcp_audio_block(block)
            if tag:
                parts.append(tag)
                continue
            rendered = await _render_mcp_resource_block(block, "slack")
            if rendered:
                parts.append(rendered)
        assert len(parts) == 2
        assert parts[0].startswith("File ID")
        assert "saved to" in parts[1]

    @pytest.mark.asyncio
    async def test_existing_image_behavior_unchanged(self):
        from tools.mcp_tool import _cache_mcp_image_block

        block = SimpleNamespace(
            data=base64.b64encode(b"some bytes").decode("ascii"),
            mimeType="application/pdf",
        )
        assert await _cache_mcp_image_block(block) == ""


class TestErrorPathResourceText:
    """isError payloads must surface EmbeddedResource text, not drop it."""

    @pytest_asyncio.fixture()
    async def _handler(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock, patch as mock_patch

        from tools import mcp_tool

        fake_session = MagicMock()
        fake_server = SimpleNamespace(session=fake_session, _rpc_lock=asyncio.Lock())

        await mcp_tool._activate_mcp_scope()
        mcp_tool._reset_server_error("test-server")
        try:
            with mock_patch.dict(mcp_tool._servers, {"test-server": fake_server}):
                fake_session.call_tool = AsyncMock()
                yield fake_session, mcp_tool._make_tool_handler("test-server", "my-tool", 30.0)
        finally:
            mcp_tool._reset_server_error("test-server")

    @pytest.mark.asyncio
    async def test_error_embedded_resource_text_surfaced(self, _handler):
        from unittest.mock import AsyncMock

        session, handler = _handler
        res = SimpleNamespace(uri="mem://err", mimeType="text/plain",
                              text="quota exceeded for workspace W1", blob=None)
        session.call_tool = AsyncMock(return_value=SimpleNamespace(
            content=[_embedded(res)], isError=True, structuredContent=None,
        ))
        data = json.loads(await handler({}))
        assert "quota exceeded for workspace W1" in data["error"]

    @pytest.mark.asyncio
    async def test_error_mixed_text_and_resource(self, _handler):
        from unittest.mock import AsyncMock

        session, handler = _handler
        res = SimpleNamespace(uri="mem://err", mimeType="text/plain",
                              text=" — details in resource", blob=None)
        session.call_tool = AsyncMock(return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text="tool failed"), _embedded(res)],
            isError=True, structuredContent=None,
        ))
        data = json.loads(await handler({}))
        assert "tool failed" in data["error"]
        assert "details in resource" in data["error"]

    @pytest.mark.asyncio
    async def test_error_with_no_text_blocks_falls_back(self, _handler):
        from unittest.mock import AsyncMock

        session, handler = _handler
        session.call_tool = AsyncMock(return_value=SimpleNamespace(
            content=[], isError=True, structuredContent=None,
        ))
        data = json.loads(await handler({}))
        assert data["error"] == "MCP tool returned an error"
