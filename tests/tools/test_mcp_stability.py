"""Tests for MCP stability fixes — event loop handler, PID tracking, shutdown robustness."""

import asyncio
import os
import signal
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fix 1: MCP event loop exception handler
# ---------------------------------------------------------------------------

class TestMCPLoopExceptionHandler:
    """Compatibility shell left empty after removing the shared MCP loop."""





# ---------------------------------------------------------------------------
# Fix 2: stdio PID tracking
# ---------------------------------------------------------------------------

class TestStdioPidTracking:
    """_snapshot_child_pids and _stdio_pids track subprocess PIDs."""

    @pytest.mark.asyncio
    async def test_snapshot_returns_set(self):
        from tools.mcp_tool import _snapshot_child_pids
        result = await _snapshot_child_pids()
        assert isinstance(result, set)
        # All elements should be ints
        for pid in result:
            assert isinstance(pid, int)







# ---------------------------------------------------------------------------
# Fix 2b: stdio descendant reaping via process group (issue #23799)
# ---------------------------------------------------------------------------
#
# When a stdio MCP wrapper (e.g. ``openclaw mcp serve``) itself spawns a
# helper subprocess (``claude mcp serve``) and then exits, the helper
# reparents to systemd-user and is invisible to the per-pid orphan reaper.
# The fix captures the wrapper's pgid at spawn time and reaps via killpg,
# which reaches same-group descendants whether or not the direct pid is alive.

class TestStdioPgroupReaping:
    """_kill_orphaned_mcp_children reaps via killpg when a pgid is tracked."""

    def _reset_state(self):
        from tools.mcp_tool import (
            _orphan_stdio_pid_servers,
            _orphan_stdio_pids,
            _stdio_pgids,
            _stdio_pids,
            _lock,
        )
        with _lock:
            _stdio_pids.clear()
            _orphan_stdio_pids.clear()
            _orphan_stdio_pid_servers.clear()
            _stdio_pgids.clear()




# ---------------------------------------------------------------------------
# Fix 4: MCP initial connection retry with backoff
# (Ported from Kilo Code's MCP resilience fix)
# ---------------------------------------------------------------------------

class TestMCPInitialConnectionRetry:
    """MCPServerTask.run() retries initial connection failures instead of giving up."""

    def test_initial_connect_retries_constant_exists(self):
        """_MAX_INITIAL_CONNECT_RETRIES should be defined."""
        from tools.mcp_tool import _MAX_INITIAL_CONNECT_RETRIES
        assert _MAX_INITIAL_CONNECT_RETRIES >= 1


    @pytest.mark.asyncio
    async def test_initial_connect_retry_respects_shutdown(self):
        """Shutdown during initial retry backoff aborts cleanly."""
        from tools.mcp_tool import MCPServerTask

        async def _run():
            server = MCPServerTask("test-shutdown")
            attempt = 0

            async def fake_run_stdio(self_inner, config):
                nonlocal attempt
                attempt += 1
                if attempt == 1:
                    raise ConnectionError("transient failure")
                # Should not reach here because shutdown fires during sleep
                raise AssertionError("Should not attempt after shutdown")

            with patch.object(MCPServerTask, '_run_stdio', fake_run_stdio), \
                 patch('tools.mcp_tool._jittered', lambda s: 0.01):
                task = asyncio.ensure_future(server.run({"command": "fake"}))

                # Give the first attempt time to fail, then set shutdown
                # during the backoff sleep
                await asyncio.sleep(0.1)
                server._shutdown_event.set()
                await server._ready.wait()

                # Should have the error set and be done
                assert server._error is not None
                await task

        await _run()


# ---------------------------------------------------------------------------
# Fix: drain pending tasks before closing the MCP loop
# ---------------------------------------------------------------------------

class TestMCPLoopDrainOnStop:
    """Compatibility shell left empty after removing the shared MCP loop."""
