#!/usr/bin/env python3
"""
Tests for file staleness detection in write_file and patch.

When a file is modified externally between the agent's read and write,
the write should include a warning so the agent can re-read and verify.

Run with:  python -m pytest tests/tools/test_file_staleness.py -v
"""

import json
import asyncio
import os
import tempfile
import unittest
from unittest.mock import patch

from tools import file_state
from tools.file_tools import (
    read_file_tool,
    write_file_tool,
    patch_tool,
    _check_file_staleness,
    _read_tracker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------









# ---------------------------------------------------------------------------
# Core staleness check
# ---------------------------------------------------------------------------

class TestStalenessCheck(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        _read_tracker.clear()
        file_state.get_registry().clear()
        self._tmpdir = tempfile.mkdtemp()
        self._tmpfile = os.path.join(self._tmpdir, "stale_test.txt")
        with open(self._tmpfile, "w") as f:
            f.write("original content\n")

    def tearDown(self):
        _read_tracker.clear()
        file_state.get_registry().clear()
        try:
            os.unlink(self._tmpfile)
            os.rmdir(self._tmpdir)
        except OSError:
            pass

    async def test_no_warning_when_file_unchanged(self):
        """Read then write with no external modification — no warning."""
        await read_file_tool(self._tmpfile, task_id="t1")

        result = json.loads(
            await write_file_tool(self._tmpfile, "new content", task_id="t1")
        )
        self.assertNotIn("_warning", result)


    async def test_relative_path_uses_recorded_session_cwd_for_staleness_tracking(self):
        """Relative-path stale tracking must follow the session's recorded cwd."""
        start_dir = os.path.join(self._tmpdir, "start")
        live_dir = os.path.join(self._tmpdir, "worktree")
        os.makedirs(start_dir, exist_ok=True)
        os.makedirs(live_dir, exist_ok=True)

        start_file = os.path.join(start_dir, "shared.txt")
        live_file = os.path.join(live_dir, "shared.txt")
        with open(start_file, "w") as f:
            f.write("start copy\n")
        with open(live_file, "w") as f:
            f.write("live copy\n")

        from tools import terminal_tool

        # The session cd'd into the worktree (recorded by the completed command).
        await terminal_tool.record_session_cwd("live_task", live_dir)

        try:
            with patch.dict(os.environ, {"TERMINAL_CWD": start_dir}, clear=False):
                await read_file_tool("shared.txt", task_id="live_task")

                await asyncio.sleep(0.05)
                with open(live_file, "w") as f:
                    f.write("live copy modified elsewhere\n")

                result = json.loads(
                    await write_file_tool(
                        "shared.txt", "replacement", task_id="live_task"
                    )
                )
        finally:
            terminal_tool.clear_session_cwd("live_task")

        self.assertIn("_warning", result)
        self.assertIn("modified since you last read", result["_warning"])


# ---------------------------------------------------------------------------
# Staleness in patch
# ---------------------------------------------------------------------------

class TestPatchStaleness(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        _read_tracker.clear()
        file_state.get_registry().clear()
        self._tmpdir = tempfile.mkdtemp()
        self._tmpfile = os.path.join(self._tmpdir, "patch_test.txt")
        with open(self._tmpfile, "w") as f:
            f.write("original line\n")

    def tearDown(self):
        _read_tracker.clear()
        file_state.get_registry().clear()
        try:
            os.unlink(self._tmpfile)
            os.rmdir(self._tmpdir)
        except OSError:
            pass

    async def test_patch_warns_on_stale_file(self):
        """Patch should warn if the target file changed since last read."""
        await read_file_tool(self._tmpfile, task_id="p1")

        await asyncio.sleep(0.05)
        with open(self._tmpfile, "w") as f:
            f.write("original line externally modified\n")

        result = json.loads(
            await patch_tool(
                mode="replace",
                path=self._tmpfile,
                old_string="original",
                new_string="patched",
                task_id="p1",
            )
        )
        self.assertIn("_warning", result)
        self.assertIn("modified since you last read", result["_warning"])

    async def test_patch_no_warning_when_fresh(self):
        """Patch with no external changes — no warning."""
        await read_file_tool(self._tmpfile, task_id="p2")

        result = json.loads(
            await patch_tool(
                mode="replace",
                path=self._tmpfile,
                old_string="original",
                new_string="patched",
                task_id="p2",
            )
        )
        self.assertNotIn("_warning", result)


# ---------------------------------------------------------------------------
# Unit test for the helper
# ---------------------------------------------------------------------------

class TestCheckFileStalenessHelper(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        _read_tracker.clear()
        file_state.get_registry().clear()

    def tearDown(self):
        _read_tracker.clear()
        file_state.get_registry().clear()

    async def test_returns_none_for_unknown_task(self):
        self.assertIsNone(await _check_file_staleness("/tmp/x.py", "nonexistent"))


    async def test_returns_none_when_stat_fails(self):
        from tools.file_tools import _read_tracker, _read_tracker_lock
        with _read_tracker_lock:
            _read_tracker["t1"] = {
                "last_key": None, "consecutive": 0,
                "read_history": set(), "dedup": {},
                "read_timestamps": {"/nonexistent/path": 99999.0},
            }
        # File doesn't exist → stat fails → returns None (let write handle it)
        self.assertIsNone(await _check_file_staleness("/nonexistent/path", "t1"))


if __name__ == "__main__":
    unittest.main()
