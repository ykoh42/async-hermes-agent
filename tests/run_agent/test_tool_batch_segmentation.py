"""Segment-aware mixed tool-batch dispatch.

A model response containing several parallel-safe reads plus one unsafe
tool used to lose ALL concurrency: `_should_parallelize_tool_batch` was
all-or-nothing, so one barrier call forced the entire batch onto the
sequential path.  `_plan_tool_batch_segments` now splits the batch into
ordered segments — maximal contiguous runs of parallel-safe calls execute
concurrently, barrier calls sequentially — while preserving:

  * model tool-result ordering (one result per call, in emission order),
  * side-effect boundaries (no call starts before an earlier barrier ends).
"""

import asyncio
import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from agent.tool_dispatch_helpers import (
    _plan_tool_batch_segments,
    _should_parallelize_tool_batch,
)


def _tc(name="web_search", arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _kinds(segments):
    return [kind for kind, _ in segments]


def _flatten_ids(segments):
    return [tc.id for _, calls in segments for tc in calls]


# ---------------------------------------------------------------------------
# Planner unit tests
# ---------------------------------------------------------------------------


class TestPlanToolBatchSegments:
    def test_all_safe_batch_is_single_parallel_segment(self):
        calls = [_tc("web_search"), _tc("read_file", '{"path":"a.py"}'), _tc("web_extract")]
        segments = _plan_tool_batch_segments(calls)
        assert _kinds(segments) == ["parallel"]
        assert _flatten_ids(segments) == [c.id for c in calls]

    def test_three_safe_reads_plus_trailing_unsafe_keeps_reads_parallel(self):
        """The headline case: 3 safe reads + 1 unsafe tool must NOT go fully sequential."""
        calls = [
            _tc("web_search", call_id="r1"),
            _tc("web_search", call_id="r2"),
            _tc("read_file", '{"path":"a.py"}', call_id="r3"),
            _tc("terminal", '{"command":"echo hi"}', call_id="b1"),
        ]
        segments = _plan_tool_batch_segments(calls)
        assert _kinds(segments) == ["parallel", "sequential"]
        assert [tc.id for tc in segments[0][1]] == ["r1", "r2", "r3"]
        assert [tc.id for tc in segments[1][1]] == ["b1"]

    def test_barrier_in_middle_splits_runs_and_preserves_order(self):
        calls = [
            _tc("web_search", call_id="r1"),
            _tc("web_search", call_id="r2"),
            _tc("terminal", '{"command":"make"}', call_id="b1"),
            _tc("web_search", call_id="r3"),
            _tc("web_search", call_id="r4"),
        ]
        segments = _plan_tool_batch_segments(calls)
        assert _kinds(segments) == ["parallel", "sequential", "parallel"]
        assert _flatten_ids(segments) == ["r1", "r2", "b1", "r3", "r4"]

    def test_single_safe_call_after_barrier_is_demoted_and_merged(self):
        # parallel run of 1 gains nothing — demote to sequential and merge
        # with the adjacent barrier segment.
        calls = [
            _tc("web_search", call_id="r1"),
            _tc("web_search", call_id="r2"),
            _tc("terminal", '{"command":"make"}', call_id="b1"),
            _tc("web_search", call_id="r3"),
        ]
        segments = _plan_tool_batch_segments(calls)
        assert _kinds(segments) == ["parallel", "sequential"]
        assert [tc.id for tc in segments[1][1]] == ["b1", "r3"]


    def test_never_parallel_tool_is_a_barrier(self):
        calls = [
            _tc("web_search", call_id="r1"),
            _tc("web_search", call_id="r2"),
            _tc("clarify", '{"question":"?"}', call_id="c1"),
        ]
        segments = _plan_tool_batch_segments(calls)
        assert _kinds(segments) == ["parallel", "sequential"]
        assert [tc.id for tc in segments[1][1]] == ["c1"]



    def test_overlapping_paths_split_across_segments(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        calls = [
            _tc("read_file", '{"path":"a.py"}', call_id="w1"),
            _tc("web_search", call_id="r1"),
            _tc("write_file", '{"path":"a.py","content":"x"}', call_id="w2"),
            _tc("web_search", call_id="r2"),
        ]
        segments = _plan_tool_batch_segments(calls)
        # w2 conflicts with w1 → closes the first run; w2+r2 form the second.
        assert _kinds(segments) == ["parallel", "parallel"]
        assert [tc.id for tc in segments[0][1]] == ["w1", "r1"]
        assert [tc.id for tc in segments[1][1]] == ["w2", "r2"]
        # Order and completeness preserved.
        assert _flatten_ids(segments) == ["w1", "r1", "w2", "r2"]

    def test_path_scoped_tool_without_path_is_a_barrier(self):
        calls = [
            _tc("read_file", "{}", call_id="nopath"),
            _tc("web_search", call_id="r1"),
            _tc("web_search", call_id="r2"),
        ]
        segments = _plan_tool_batch_segments(calls)
        assert _kinds(segments) == ["sequential", "parallel"]

    def test_flattened_segments_always_preserve_emission_order(self):
        calls = [
            _tc("terminal", '{"command":"x"}', call_id="b1"),
            _tc("web_search", call_id="r1"),
            _tc("clarify", '{"question":"?"}', call_id="c1"),
            _tc("read_file", '{"path":"a.py"}', call_id="r2"),
            _tc("read_file", '{"path":"b.py"}', call_id="r3"),
        ]
        segments = _plan_tool_batch_segments(calls)
        assert _flatten_ids(segments) == ["b1", "r1", "c1", "r2", "r3"]


class TestShouldParallelizeBackwardCompat:
    """The boolean gate is now a view over the planner — same answers as before."""

    def test_single_call_is_sequential(self):
        assert not _should_parallelize_tool_batch([_tc("web_search")])





# ---------------------------------------------------------------------------
# Dispatcher integration
# ---------------------------------------------------------------------------


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


@pytest.fixture()
def agent():
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("web_search", "terminal"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


class TestSegmentedDispatchIntegration:
    @staticmethod
    def _native_tool_entry():
        return patch(
            "tools.registry.registry.get_entry",
            return_value=SimpleNamespace(is_async=True, max_result_size_chars=None),
        )

    @pytest.mark.asyncio
    async def test_mixed_batch_runs_safe_prefix_concurrently_and_barrier_after(self, agent):
        """Safe calls overlap; a barrier starts only after their completion."""
        calls = [
            _tc("web_search", '{"query":"a"}', call_id="s1"),
            _tc("web_search", '{"query":"b"}', call_id="s2"),
            _tc("terminal", '{"command":"echo done"}', call_id="t1"),
        ]
        events = []
        searches_started = asyncio.Event()

        async def dispatch(name, _args, _task_id, **kwargs):
            events.append(("start", name, kwargs["tool_call_id"]))
            if name == "web_search":
                if len([event for event in events if event[:2] == ("start", "web_search")]) == 2:
                    searches_started.set()
                await asyncio.wait_for(searches_started.wait(), timeout=1)
            events.append(("end", name, kwargs["tool_call_id"]))
            return json.dumps({"ok": name})

        messages = []
        with (
            patch("model_tools.handle_function_call", side_effect=dispatch),
            self._native_tool_entry(),
        ):
            await agent._execute_tool_calls(
                SimpleNamespace(content="", tool_calls=calls), messages, "task-1"
            )

        assert [message["tool_call_id"] for message in messages] == ["s1", "s2", "t1"]
        terminal_start = events.index(("start", "terminal", "t1"))
        assert all(
            index < terminal_start
            for index, event in enumerate(events)
            if event[0] == "end" and event[1] == "web_search"
        )

    @pytest.mark.asyncio
    async def test_mixed_batch_preserves_order_with_barrier_in_middle(self, agent):
        calls = [
            _tc("web_search", call_id="s1"),
            _tc("web_search", call_id="s2"),
            _tc("terminal", '{"command":"touch x"}', call_id="t1"),
            _tc("web_search", call_id="s3"),
            _tc("web_search", call_id="s4"),
        ]
        executed = []

        async def dispatch(_name, _args, _task_id, **kwargs):
            executed.append(kwargs["tool_call_id"])
            return json.dumps({"ok": True})

        messages = []
        with (
            patch("model_tools.handle_function_call", side_effect=dispatch),
            self._native_tool_entry(),
        ):
            await agent._execute_tool_calls(
                SimpleNamespace(content="", tool_calls=calls), messages, "task-1"
            )

        assert [message["tool_call_id"] for message in messages] == ["s1", "s2", "t1", "s3", "s4"]
        terminal_index = executed.index("t1")
        assert set(executed[:terminal_index]) == {"s1", "s2"}
        assert set(executed[terminal_index + 1:]) == {"s3", "s4"}

    @pytest.mark.asyncio
    async def test_homogeneous_safe_batch_runs_concurrently(self, agent):
        calls = [_tc("web_search", call_id="s1"), _tc("web_search", call_id="s2")]
        starts = []
        release = asyncio.Event()

        async def dispatch(_name, _args, _task_id, **kwargs):
            starts.append(kwargs["tool_call_id"])
            if len(starts) == 2:
                release.set()
            await asyncio.wait_for(release.wait(), timeout=1)
            return json.dumps({"ok": True})

        with (
            patch("model_tools.handle_function_call", side_effect=dispatch),
            self._native_tool_entry(),
        ):
            await agent._execute_tool_calls(
                SimpleNamespace(content="", tool_calls=calls), [], "task-1"
            )

        assert starts == ["s1", "s2"]

    @pytest.mark.asyncio
    async def test_interrupt_during_barrier_drains_later_segments(self, agent):
        calls = [
            _tc("web_search", call_id="s1"),
            _tc("web_search", call_id="s2"),
            _tc("terminal", '{"command":"long"}', call_id="t1"),
            _tc("web_search", call_id="s3"),
            _tc("web_search", call_id="s4"),
        ]
        executed = []

        async def dispatch(_name, _args, _task_id, **kwargs):
            call_id = kwargs["tool_call_id"]
            executed.append(call_id)
            if call_id == "t1":
                agent._interrupt_requested = True
            return json.dumps({"ok": True})

        messages = []
        with (
            patch("model_tools.handle_function_call", side_effect=dispatch),
            self._native_tool_entry(),
        ):
            await agent._execute_tool_calls(
                SimpleNamespace(content="", tool_calls=calls), messages, "task-1"
            )

        assert [message["tool_call_id"] for message in messages] == ["s1", "s2", "t1", "s3", "s4"]
        assert "s3" not in executed and "s4" not in executed
        assert all("cancelled" in message["content"] for message in messages[-2:])

    @pytest.mark.asyncio
    async def test_steer_lands_exactly_once_in_mixed_batch(self, agent):
        calls = [
            _tc("web_search", call_id="s1"),
            _tc("web_search", call_id="s2"),
            _tc("terminal", '{"command":"echo hi"}', call_id="t1"),
        ]

        async def dispatch(_name, _args, _task_id, **_kwargs):
            return json.dumps({"ok": True})

        agent.steer("focus on the tests")
        messages = []
        with (
            patch("model_tools.handle_function_call", side_effect=dispatch),
            self._native_tool_entry(),
        ):
            await agent._execute_tool_calls(
                SimpleNamespace(content="", tool_calls=calls), messages, "task-1"
            )

        assert sum("focus on the tests" in message["content"] for message in messages) == 1


class TestPathCanonicalization:
    """Regression tests for _canonical_path / _extract_parallel_scope_path fixes.

    Verifies that symlink aliases, relative/absolute cwd mismatches, and
    (on Windows) case-insensitive aliases are never placed in the same
    parallel segment.
    """

    def test_relative_and_absolute_same_target_use_separate_segments(self, tmp_path):
        """A relative path resolved against execution_cwd and an absolute path
        pointing to the same file must be detected as overlapping."""
        from agent.tool_dispatch_helpers import (
            _canonical_path,
            _paths_overlap,
        )

        target = tmp_path / "config.json"
        target.touch()

        abs_path = _canonical_path(str(target))
        rel_path = _canonical_path("config.json", execution_cwd=tmp_path)

        assert _paths_overlap(abs_path, rel_path), (
            "Absolute and relative paths pointing to the same file must overlap"
        )

    def test_symlink_aliases_are_not_parallelized(self, tmp_path):
        """A symlink alias and the real path must be detected as overlapping
        so they are never placed in the same parallel segment."""
        import os
        from agent.tool_dispatch_helpers import (
            _canonical_path,
            _paths_overlap,
        )

        real_dir = tmp_path / "real"
        real_dir.mkdir()
        target = real_dir / "config.json"
        target.touch()

        alias_dir = tmp_path / "alias"
        alias_dir.symlink_to(real_dir)

        real_path = _canonical_path(str(target))
        alias_path = _canonical_path(str(alias_dir / "config.json"))

        assert _paths_overlap(real_path, alias_path), (
            "Symlink alias and real path must overlap — "
            "they must not be parallelized"
        )

    def test_execution_cwd_used_over_process_cwd(self, tmp_path, monkeypatch):
        """_extract_parallel_scope_path must use execution_cwd, not
        process cwd, when resolving relative paths."""
        from agent.tool_dispatch_helpers import (
            _extract_parallel_scope_path,
            _paths_overlap,
        )

        exec_cwd = tmp_path / "sub"
        exec_cwd.mkdir()
        (exec_cwd / "x.txt").touch()

        # Point process cwd somewhere else entirely.
        monkeypatch.chdir(tmp_path)

        # With execution_cwd supplied the relative path resolves under exec_cwd.
        path_with_cwd = _extract_parallel_scope_path(
            "write_file", {"path": "x.txt"}, execution_cwd=exec_cwd
        )
        # The absolute path under exec_cwd must match.
        path_absolute = _extract_parallel_scope_path(
            "write_file", {"path": str(exec_cwd / "x.txt")}
        )

        assert path_with_cwd is not None
        assert path_absolute is not None
        assert _paths_overlap(path_with_cwd, path_absolute), (
            "execution_cwd-relative path and absolute path must overlap; "
            "process cwd must not be used when execution_cwd is provided"
        )


    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="normcase() case-folding only matters on Windows",
    )
    def test_case_insensitive_paths_overlap_windows(self, tmp_path):
        """On Windows, FILE.txt and file.txt are the same file — they must
        be detected as overlapping after normcase() canonicalisation."""
        from agent.tool_dispatch_helpers import _canonical_path, _paths_overlap

        upper = _canonical_path(str(tmp_path / "FILE.txt"), execution_cwd=tmp_path)
        lower = _canonical_path(str(tmp_path / "file.txt"), execution_cwd=tmp_path)

        assert _paths_overlap(upper, lower), (
            "Case-insensitive aliases must overlap on Windows"
        )
