"""Tests for agent/file_safety.py read guards — env file blocking.

Run with:  python -m pytest tests/agent/test_file_safety.py -v
"""

import inspect
import os
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.asyncio

from agent.file_safety import (
    _BLOCKED_PROJECT_ENV_BASENAMES,
    build_write_denied_paths,
    build_write_denied_prefixes,
    get_read_block_error,
    get_safe_write_roots,
    is_write_denied,
)


async def test_public_write_safety_api_is_native_async(tmp_path, monkeypatch):
    """Keep the upstream public names while making their I/O awaitable."""
    functions = (
        build_write_denied_paths,
        build_write_denied_prefixes,
        get_safe_write_roots,
        is_write_denied,
    )
    assert all(inspect.iscoroutinefunction(function) for function in functions)

    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    roots = await get_safe_write_roots()
    assert str(tmp_path.resolve()) in roots
    assert await is_write_denied(str(tmp_path / "allowed.txt")) is False
    assert await is_write_denied(str(tmp_path.parent / "outside.txt")) is True


# ---------------------------------------------------------------------------
# Project-local .env file blocking (issue #20734)
# ---------------------------------------------------------------------------


class TestEnvFileReadBlocking:
    """Secret-bearing .env files must be blocked by get_read_block_error."""

    @pytest.mark.parametrize("basename", [
        ".env",
        ".env.local",
        ".env.development",
        ".env.production",
        ".env.test",
        ".env.staging",
        ".envrc",
    ])
    async def test_blocked_env_basenames(self, basename):
        """All secret-bearing .env basenames are blocked regardless of directory."""
        path = f"/tmp/project/{basename}"
        error = await get_read_block_error(path)
        assert error is not None, f"{basename} should be blocked"
        assert "Access denied" in error
        assert "secret-bearing" in error.lower() or "environment file" in error.lower()


    @pytest.mark.parametrize("basename", [
        ".ENV",
        ".Env.Local",
        ".ENV.PRODUCTION",
        ".ENVRC",
    ])
    async def test_blocked_env_basenames_case_insensitive(self, basename):
        """Secret-bearing .env basenames are blocked regardless of case."""
        error = await get_read_block_error(f"/tmp/project/{basename}")
        assert error is not None, f"{basename} should be blocked"
        assert "Access denied" in error
        assert "environment file" in error.lower()


    async def test_allowed_env_example(self):
        """"The .env.example file is explicitly allowed — it's documentation, not a secret."""
        error = await get_read_block_error("/tmp/project/.env.example")
        assert error is None






# ---------------------------------------------------------------------------
# Existing cache-file blocking (regression — must still work)
# ---------------------------------------------------------------------------


class TestCacheFileReadBlocking:
    """Internal Hermes cache files must remain blocked."""

    async def test_hub_index_cache_blocked(self, tmp_path):
        """Hub index-cache reads are blocked."""
        hermes_home = tmp_path / ".hermes"
        cache = hermes_home / "skills" / ".hub" / "index-cache" / "data.json"
        cache.parent.mkdir(parents=True)
        cache.write_text("{}")

        with patch("agent.file_safety._hermes_home_path", return_value=hermes_home):
            error = await get_read_block_error(str(cache))
            assert error is not None
            assert "internal Hermes cache" in error

    async def test_hub_directory_blocked(self, tmp_path):
        """Hub directory reads are blocked."""
        hermes_home = tmp_path / ".hermes"
        hub = hermes_home / "skills" / ".hub" / "metadata.json"
        hub.parent.mkdir(parents=True)
        hub.write_text("{}")

        with patch("agent.file_safety._hermes_home_path", return_value=hermes_home):
            error = await get_read_block_error(str(hub))
            assert error is not None


# ---------------------------------------------------------------------------
# Combined: env guard + cache guard don't interfere
# ---------------------------------------------------------------------------


class TestCombinedGuards:
    """Both guards should work independently without interference."""

    async def test_env_guard_works_regardless_of_hermes_home(self, tmp_path):
        """The env basename guard does not depend on HERMES_HOME resolution."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()

        with patch("agent.file_safety._hermes_home_path", return_value=hermes_home):
            # Regular project .env should still be blocked
            error = await get_read_block_error("/workspace/.env")
            assert error is not None

            # .env.example should still be allowed
            error = await get_read_block_error("/workspace/.env.example")
            assert error is None

    async def test_cache_guard_still_works_with_env_guard(self, tmp_path):
        """Cache file blocking still works when env guard is active."""
        hermes_home = tmp_path / ".hermes"
        cache = hermes_home / "skills" / ".hub" / "index-cache" / "x"
        cache.parent.mkdir(parents=True)
        cache.write_text("")

        with patch("agent.file_safety._hermes_home_path", return_value=hermes_home):
            error = await get_read_block_error(str(cache))
            assert error is not None
            assert "internal Hermes cache" in error
