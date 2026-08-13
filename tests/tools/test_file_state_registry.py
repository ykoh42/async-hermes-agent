#!/usr/bin/env python3
"""Tests for the cross-agent FileStateRegistry (tools/file_state.py).

Covers the two registry layers used for safe concurrent subagent file edits:

  1. Cross-agent staleness detection via ``check_stale``
  2. Delegate-completion reminder via ``writes_since``

Plus integration through the real ``read_file_tool`` / ``write_file_tool``
/ ``patch_tool`` handlers so the full hook wiring is exercised.

Run:
    python -m pytest tests/tools/test_file_state_registry.py -v
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest

import aiofiles
import pytest
from blockbuster import BlockBuster
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

from tools import file_state
from tools.file_tools import (
    read_file_tool,
    write_file_tool,
    patch_tool,
)


def test_public_exports_match_upstream_contract():
    assert file_state.__all__ == [
        "FileStateRegistry",
        "get_registry",
        "record_read",
        "note_write",
        "check_stale",
        "lock_path",
        "writes_since",
        "known_reads",
    ]


def _tmp_file(content: str = "initial\n") -> str:
    fd, path = tempfile.mkstemp(prefix="hermes_file_state_test_", suffix=".txt")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


class FileStateRegistryUnitTests(unittest.IsolatedAsyncioTestCase):
    """Direct unit tests on the registry singleton."""

    def setUp(self) -> None:
        file_state.get_registry().clear()
        self._tmpfiles: list[str] = []

    def tearDown(self) -> None:
        for p in self._tmpfiles:
            try:
                os.unlink(p)
            except OSError:
                pass
        file_state.get_registry().clear()

    def _mk(self, content: str = "x\n") -> str:
        p = _tmp_file(content)
        self._tmpfiles.append(p)
        return p

    async def test_record_read_then_check_stale_returns_none(self):
        p = self._mk()
        await file_state.record_read("A", p)
        self.assertIsNone(await file_state.check_stale("A", p))

    async def test_sibling_write_flags_other_agent_as_stale(self):
        p = self._mk()
        await file_state.record_read("A", p)
        # Simulate sibling writing this file later
        await asyncio.sleep(0.01)  # ensure ts ordering across resolution
        await file_state.note_write("B", p)
        warn = await file_state.check_stale("A", p)
        self.assertIsNotNone(warn)
        self.assertIn("B", warn)
        self.assertIn("sibling", warn.lower())

    async def test_external_write_flags_original_reader_as_stale(self):
        p = self._mk()
        await file_state.record_read("A", p)
        await asyncio.sleep(0.01)
        async with aiofiles.open(p, "w", encoding="utf-8") as handle:
            await handle.write("externally changed\n")

        warn = await file_state.check_stale("A", p)

        self.assertIsNotNone(warn)
        self.assertIn("external edit", warn)

    async def test_kill_switch_env_var(self):
        p = self._mk()
        os.environ["HERMES_DISABLE_FILE_STATE_GUARD"] = "1"
        try:
            await file_state.record_read("A", p)
            await file_state.note_write("B", p)
            self.assertIsNone(await file_state.check_stale("A", p))
            self.assertEqual(file_state.known_reads("A"), [])
            self.assertEqual(
                file_state.writes_since("A", 0.0, [p]),
                {},
            )
        finally:
            del os.environ["HERMES_DISABLE_FILE_STATE_GUARD"]


@pytest.mark.asyncio
async def test_file_state_io_does_not_block_or_leak(tmp_path):
    path = tmp_path / "state.txt"
    async with aiofiles.open(path, "w", encoding="utf-8") as handle:
        await handle.write("initial\n")
    file_state.get_registry().clear()

    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        blockbuster = BlockBuster()
        blockbuster.activate()
        try:
            await file_state.record_read("agent", path)
            assert await file_state.check_stale("agent", path) is None
            async with file_state.lock_path(path):
                await file_state.note_write("agent", path)
        finally:
            blockbuster.deactivate()

    file_state.get_registry().clear()


class FileToolsIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Integration through the real file_tools handlers.

    These exercise the wiring: read_file_tool → registry.record_read,
    write_file_tool / patch_tool → check_stale + note_write.
    """

    def setUp(self) -> None:
        file_state.get_registry().clear()
        self._tmpdir = tempfile.mkdtemp(prefix="hermes_file_state_int_")

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        file_state.get_registry().clear()

    def _write_seed(self, name: str, content: str = "seed\n") -> str:
        p = os.path.join(self._tmpdir, name)
        with open(p, "w") as f:
            f.write(content)
        return p

    async def test_sibling_agent_write_surfaces_warning_through_handler(self):
        p = self._write_seed("shared.txt")
        r = json.loads(await read_file_tool(path=p, task_id="agentA"))
        self.assertNotIn("error", r)

        w_b = json.loads(await write_file_tool(path=p, content="B wrote\n", task_id="agentB"))
        self.assertNotIn("error", w_b)

        w_a = json.loads(await write_file_tool(path=p, content="A stale\n", task_id="agentA"))
        warn = w_a.get("_warning", "")
        self.assertTrue(warn, f"expected warning, got: {w_a}")
        # The cross-agent message names the sibling task_id.
        self.assertIn("agentB", warn)
        self.assertIn("sibling", warn.lower())

    async def test_staleness_is_checked_after_waiting_for_path_lock(self):
        p = self._write_seed("locked.txt")
        await read_file_tool(path=p, task_id="agentA")
        resolved = os.path.realpath(p)
        lock_held = asyncio.Event()
        release_lock = asyncio.Event()

        async def hold_lock():
            async with file_state.lock_path(resolved):
                lock_held.set()
                await release_lock.wait()

        holder = asyncio.create_task(hold_lock())
        await lock_held.wait()
        writer = asyncio.create_task(
            write_file_tool(path=p, content="agent write\n", task_id="agentA")
        )
        await asyncio.sleep(0.01)
        async with aiofiles.open(p, "w", encoding="utf-8") as handle:
            await handle.write("external write while waiting\n")
        release_lock.set()

        result = json.loads(await writer)
        await holder

        self.assertIn("_warning", result)
        self.assertIn("external edit", result["_warning"])

    async def test_v4a_patch_surfaces_sibling_staleness_warning(self):
        p = self._write_seed("v4a.txt")
        await read_file_tool(path=p, task_id="agentA")
        await write_file_tool(path=p, content="sibling edit\n", task_id="agentB")
        patch = (
            "*** Begin Patch\n"
            f"*** Update File: {p}\n"
            "@@\n"
            "-sibling edit\n"
            "+patched edit\n"
            "*** End Patch"
        )

        result = json.loads(await patch_tool(mode="patch", patch=patch, task_id="agentA"))

        self.assertNotIn("error", result)
        self.assertIn("_warning", result)
        self.assertIn("agentB", result["_warning"])


    async def test_net_new_file_no_warning(self):
        p = os.path.join(self._tmpdir, "brand_new.txt")
        # Nobody has read or written this before.
        w = json.loads(await write_file_tool(path=p, content="hi\n", task_id="agentX"))
        self.assertFalse(w.get("_warning"))
        self.assertNotIn("error", w)


if __name__ == "__main__":
    unittest.main()
