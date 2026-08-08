"""Tests for MCP stability fixes — event loop handler, PID tracking, shutdown robustness."""

import asyncio
import os
import signal
from unittest.mock import AsyncMock, MagicMock, patch

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

    @pytest.mark.asyncio
    async def test_kill_orphaned_handles_dead_pids(self):
        from tools.mcp_tool import (
            _kill_orphaned_mcp_children,
            _lock,
            _orphan_stdio_pid_servers,
            _orphan_stdio_pids,
        )

        fake_pid = 999_999_999
        with _lock:
            _orphan_stdio_pids.add(fake_pid)
            _orphan_stdio_pid_servers[fake_pid] = "orphan"

        with (
            patch("tools.mcp_tool.os.kill"),
            patch("tools.mcp_tool.asyncio.sleep", new=AsyncMock()),
            patch("gateway.status._pid_exists", new=AsyncMock(return_value=False)),
        ):
            await _kill_orphaned_mcp_children()

        with _lock:
            assert fake_pid not in _orphan_stdio_pids
            assert fake_pid not in _orphan_stdio_pid_servers

    @pytest.mark.asyncio
    async def test_kill_orphaned_can_filter_by_server_name(self):
        from tools.mcp_tool import (
            _kill_orphaned_mcp_children,
            _lock,
            _orphan_stdio_pid_servers,
            _orphan_stdio_pids,
        )

        target_pid = 454_545
        other_pid = 464_646
        with _lock:
            _orphan_stdio_pids.update({target_pid, other_pid})
            _orphan_stdio_pid_servers[target_pid] = "feishu"
            _orphan_stdio_pid_servers[other_pid] = "mimir"

        with (
            patch("tools.mcp_tool.os.kill") as mock_kill,
            patch("tools.mcp_tool.asyncio.sleep", new=AsyncMock()),
            patch("gateway.status._pid_exists", new=AsyncMock(return_value=False)),
        ):
            await _kill_orphaned_mcp_children(server_name="feishu")

        mock_kill.assert_called_once_with(target_pid, signal.SIGTERM)
        with _lock:
            assert target_pid not in _orphan_stdio_pids
            assert target_pid not in _orphan_stdio_pid_servers
            assert other_pid in _orphan_stdio_pids
            assert _orphan_stdio_pid_servers[other_pid] == "mimir"

        with _lock:
            _orphan_stdio_pids.discard(other_pid)
            _orphan_stdio_pid_servers.pop(other_pid, None)







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

    @pytest.mark.asyncio
    async def test_killpg_used_when_pgid_tracked(self):
        from tools.mcp_tool import (
            _kill_orphaned_mcp_children,
            _lock,
            _orphan_stdio_pids,
            _stdio_pgids,
        )

        self._reset_state()
        fake_pid = 525_252
        fake_pgid = 525_252
        with _lock:
            _orphan_stdio_pids.add(fake_pid)
            _stdio_pgids[fake_pid] = fake_pgid

        with (
            patch("tools.mcp_tool.os.getpgrp", return_value=111_111),
            patch("tools.mcp_tool.os.killpg") as mock_killpg,
            patch("tools.mcp_tool.os.kill") as mock_kill,
            patch("tools.mcp_tool.asyncio.sleep", new=AsyncMock()),
            patch("gateway.status._pid_exists", new=AsyncMock(return_value=True)),
        ):
            await _kill_orphaned_mcp_children()

        mock_killpg.assert_any_call(fake_pgid, signal.SIGTERM)
        mock_killpg.assert_any_call(fake_pgid, signal.SIGKILL)
        mock_kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_gateway_process_group_uses_per_pid_signals(self):
        from tools.mcp_tool import (
            _kill_orphaned_mcp_children,
            _lock,
            _orphan_stdio_pids,
            _stdio_pgids,
        )

        self._reset_state()
        own_pgid = 424_242
        fake_pid = 717_171
        with _lock:
            _orphan_stdio_pids.add(fake_pid)
            _stdio_pgids[fake_pid] = own_pgid

        with (
            patch("tools.mcp_tool.os.getpgrp", return_value=own_pgid),
            patch("tools.mcp_tool.os.killpg") as mock_killpg,
            patch("tools.mcp_tool.os.kill") as mock_kill,
            patch("tools.mcp_tool.asyncio.sleep", new=AsyncMock()),
            patch("gateway.status._pid_exists", new=AsyncMock(return_value=True)),
        ):
            await _kill_orphaned_mcp_children()

        mock_killpg.assert_not_called()
        mock_kill.assert_any_call(fake_pid, signal.SIGTERM)
        mock_kill.assert_any_call(fake_pid, signal.SIGKILL)

    @pytest.mark.asyncio
    async def test_no_pgid_uses_per_pid_signal(self):
        from tools.mcp_tool import (
            _kill_orphaned_mcp_children,
            _lock,
            _orphan_stdio_pids,
        )

        self._reset_state()
        fake_pid = 747_474
        with _lock:
            _orphan_stdio_pids.add(fake_pid)

        with (
            patch("tools.mcp_tool.os.kill") as mock_kill,
            patch("tools.mcp_tool.asyncio.sleep", new=AsyncMock()),
            patch("gateway.status._pid_exists", new=AsyncMock(return_value=False)),
        ):
            await _kill_orphaned_mcp_children()

        mock_kill.assert_called_once_with(fake_pid, signal.SIGTERM)




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
