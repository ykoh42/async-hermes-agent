"""Tests for the native-async OSV malware preflight."""

from unittest.mock import AsyncMock, patch

import pytest

from tools.osv_check import (
    _infer_ecosystem,
    _parse_npm_package,
    _parse_package_from_args,
    _parse_pypi_package,
    _query_osv,
    check_package_for_malware,
)


class TestInferEcosystem:
    def test_npx(self):
        assert _infer_ecosystem("npx") == "npm"
        assert _infer_ecosystem("/usr/bin/npx") == "npm"

    def test_unknown(self):
        assert _infer_ecosystem("node") is None
        assert _infer_ecosystem("python") is None
        assert _infer_ecosystem("/bin/bash") is None


class TestParsePackage:
    def test_npm(self):
        assert _parse_npm_package("react") == ("react", None)
        assert _parse_npm_package("react@latest") == ("react", None)

    def test_pypi(self):
        assert _parse_pypi_package("requests") == ("requests", None)
        assert _parse_pypi_package("mcp[cli]") == ("mcp", None)

    def test_args(self):
        assert _parse_package_from_args(["-y", "@scope/pkg@1.0"], "npm") == (
            "@scope/pkg",
            "1.0",
        )
        assert _parse_package_from_args(["--from", "mcp[cli]"], "PyPI")[0] == "mcp"
        assert _parse_package_from_args(["-y", "react@18.3.1"], "npm") == (
            "react",
            "18.3.1",
        )


class TestCheckPackageForMalware:
    @pytest.mark.asyncio
    async def test_clean_package(self):
        with patch("tools.osv_check._query_osv", new=AsyncMock(return_value=[])):
            assert await check_package_for_malware(
                "npx", ["-y", "@modelcontextprotocol/server-filesystem"]
            ) is None

    @pytest.mark.asyncio
    async def test_malware_is_blocked(self):
        malware = [
            {"id": "MAL-2023-7938", "summary": "Malicious code in evil-pkg"},
            {"id": "CVE-2023-1234", "summary": "Regular vulnerability"},
        ]
        with patch("tools.osv_check._query_osv", new=AsyncMock(return_value=malware)):
            result = await check_package_for_malware("npx", ["evil-pkg"])
        assert result is not None
        assert "BLOCKED" in result
        assert "MAL-2023-7938" in result
        assert "CVE-2023-1234" in result

    @pytest.mark.asyncio
    async def test_uvx_uses_pypi(self):
        query = AsyncMock(return_value=[])
        with patch("tools.osv_check._query_osv", new=query):
            await check_package_for_malware("uvx", ["mcp-server-fetch"])
        assert query.await_args.args[:2] == ("mcp-server-fetch", "PyPI")


class TestLiveOsvQuery:
    @pytest.mark.asyncio
    async def test_known_malware_package(self):
        try:
            result = await _query_osv("node-hide-console-windows", "npm")
        except Exception:
            pytest.skip("OSV API unreachable")
        assert result and result[0]["id"].startswith("MAL-")

    @pytest.mark.asyncio
    async def test_clean_package(self):
        try:
            result = await _query_osv("react", "npm")
        except Exception:
            pytest.skip("OSV API unreachable")
        assert result == []
