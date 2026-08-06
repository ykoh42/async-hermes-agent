"""Regression tests for file-tool path resolution base correctness.

The bug (observed in a worktree dev session, May 2026): when the resolution
base for a relative path is itself RELATIVE — e.g. ``TERMINAL_CWD="."`` from a
stale config — ``_resolve_path_for_task`` resolved the path against the agent's
PROCESS cwd instead of the intended workspace. In a git-worktree session this
silently routed ``patch``/``write_file`` edits into the *main* checkout: the
write landed, self-verified, and reported success — against the wrong file.
The agent then grepped the worktree, saw nothing, and concluded the patch tool
had silently no-op'd. It hadn't; it wrote to the wrong place.

Core invariant these tests pin:
  The resolution base for a relative path MUST always be absolute. A relative
  ``TERMINAL_CWD`` (``.``, ``./sub``, ``..``) must be anchored deterministically,
  never left to resolve against whatever the process cwd happens to be.
"""

import os
from pathlib import Path

import pytest
import pytest_asyncio

import tools.file_tools as ft
import tools.terminal_tool as terminal_tool


@pytest.fixture
def _isolated_cwd(tmp_path, monkeypatch):
    """Two checkouts: workspace (intended) + decoy (process cwd)."""
    workspace = tmp_path / "workspace"
    decoy = tmp_path / "decoy"
    workspace.mkdir()
    decoy.mkdir()
    (workspace / "target.py").write_text("WORKSPACE_ORIGINAL\n")
    (decoy / "target.py").write_text("DECOY_ORIGINAL\n")
    # Process cwd = decoy, analogous to "main repo" while the terminal is in
    # the worktree.
    monkeypatch.chdir(decoy)
    # No session cwd recorded yet (fresh-session condition).
    monkeypatch.setattr(terminal_tool, "_session_cwds", {})
    return workspace, decoy


@pytest.mark.asyncio
async def test_relative_terminal_cwd_anchors_to_absolute_not_process_cwd(
    _isolated_cwd, monkeypatch
):
    """TERMINAL_CWD='.' must NOT silently mean 'the agent process cwd'.

    A relative base is meaningless as a resolution anchor. The resolver must
    make it absolute deterministically. We assert the resolved path is
    absolute and stable regardless of where os.getcwd() points.
    """
    workspace, decoy = _isolated_cwd
    # Poison config: literal relative '.'
    monkeypatch.setenv("TERMINAL_CWD", ".")

    resolved = await ft._resolve_path_for_task("target.py", task_id="default")

    assert resolved.is_absolute(), f"resolution base leaked a relative path: {resolved}"
    # The exact anchor for a bare '.' is the process cwd resolved to absolute —
    # that is acceptable as long as it is ABSOLUTE and stable. The bug was that
    # a relative base produced surprising results; the fix is that the base is
    # always absolutised. (We do not require it to point at the workspace here —
    # that's what live-cwd tracking is for; see the next test.)
    assert str(resolved) == str((Path(os.getcwd()) / "target.py").resolve())


@pytest.mark.asyncio
async def test_live_tracking_cwd_wins_over_relative_terminal_cwd(
    _isolated_cwd, monkeypatch
):
    """When the terminal reports its absolute cwd, that is authoritative.

    This is the real-world fix: the terminal's tracked absolute cwd (the
    worktree) must override a stale relative TERMINAL_CWD so edits land where
    the agent is actually working.
    """
    workspace, decoy = _isolated_cwd
    monkeypatch.setenv("TERMINAL_CWD", ".")
    await terminal_tool.record_session_cwd("default", str(workspace))

    resolved = await ft._resolve_path_for_task("target.py", task_id="default")

    assert resolved == (workspace / "target.py")


@pytest.mark.asyncio
async def test_absolute_terminal_cwd_used_verbatim(_isolated_cwd, monkeypatch):
    """An absolute TERMINAL_CWD is the resolution base (no live tracking)."""
    workspace, decoy = _isolated_cwd
    monkeypatch.setenv("TERMINAL_CWD", str(workspace))

    resolved = await ft._resolve_path_for_task("target.py", task_id="default")

    assert resolved == (workspace / "target.py")


@pytest.mark.asyncio
async def test_resolution_base_always_absolute_no_terminal_cwd(
    _isolated_cwd, monkeypatch
):
    """With TERMINAL_CWD unset, the base falls back to an ABSOLUTE process cwd."""
    workspace, decoy = _isolated_cwd
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    resolved = await ft._resolve_path_for_task("target.py", task_id="default")

    assert resolved.is_absolute()
    assert str(resolved) == str((Path(os.getcwd()) / "target.py").resolve())


# ── B-(ii): workspace-divergence warning ────────────────────────────────────


# ── Fix C: sentinel TERMINAL_CWD + empty-registry worktree anchoring ─────────
# (May 2026 follow-up: PR #35399 made misroutes visible via resolved_path but
# the divergence warning only fired when the live terminal cwd was known. A
# worktree session whose terminal registry is still empty — no `cd` run yet —
# got neither a worktree anchor nor a warning, so a relative edit silently
# landed in main. These tests pin the sentinel handling + empty-registry
# anchoring + early warning.)


# ── Fix A: write_file / patch report the resolved ABSOLUTE path ──────────────


# ── Cross-session isolation: one session's cwd never leaks into another ──────
# (June 2026 bug class: two desktop sessions, each on its own worktree, shared
# the single "default" terminal environment and could inherit each other's cwd.
# The per-session record store solves this structurally: each session's cd
# state lives in its own record, keyed by the raw session id.)


@pytest_asyncio.fixture
async def _two_worktree_sessions(tmp_path, monkeypatch):
    """Two worktree sessions: B has cd'd (record), both registered overrides."""
    wt_a = tmp_path / "wt_a"
    wt_b = tmp_path / "wt_b"
    main = tmp_path / "main"
    for d in (wt_a, wt_b, main):
        d.mkdir()
        (d / "target.py").write_text(f"{d.name}\n")
    monkeypatch.chdir(main)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(terminal_tool, "_session_cwds", {})
    # Both sessions register their worktree cwd (TUI/desktop registration path;
    # registration seeds each session's record).
    await terminal_tool.register_task_env_overrides("sess-a", {"cwd": str(wt_a)})
    await terminal_tool.register_task_env_overrides("sess-b", {"cwd": str(wt_b)})
    # Session B ran the last command; the shared env's live cwd is wt_b but
    # only B's RECORD carries it.
    monkeypatch.setattr(
        terminal_tool,
        "_active_environments",
        {"default": _FakeEnv(str(wt_b))},
    )
    return wt_a, wt_b, main


class _FakeEnv:
    def __init__(self, cwd: str):
        self.cwd = cwd


@pytest.mark.asyncio
async def test_unregistered_session_never_inherits_another_sessions_record(
    _two_worktree_sessions, monkeypatch
):
    """Session C: no record, no override. Must NOT inherit A's or B's cwd."""
    wt_a, wt_b, main = _two_worktree_sessions
    resolved = await ft._resolve_path_for_task("target.py", task_id="sess-c")
    assert not str(resolved).startswith(str(wt_a))
    assert not str(resolved).startswith(str(wt_b))
    assert resolved == (main / "target.py").resolve()


@pytest.mark.asyncio
async def test_v4a_patch_applies_to_resolved_workspace_not_backend_cwd(
    _isolated_cwd, monkeypatch
):
    """V4A patch must edit the path the tool layer resolved, not the shell cwd.

    Regression for the git-worktree cwd bug: ``patch_tool`` resolved header
    paths against the task workspace for locking/staleness/reporting, but the
    raw (relative) patch text was handed to ``file_ops.patch_v4a``, which
    re-resolved it against the backend env's own cwd. A relative header then
    landed in a different directory than everything the tool reported. The fix
    rewrites headers to the resolved absolute paths before apply.
    """
    import json

    workspace, decoy = _isolated_cwd
    task_id = "sess-v4a"

    # Tool layer resolves against the workspace (worktree registration path).
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    await terminal_tool.register_task_env_overrides(
        task_id, {"cwd": str(workspace)}
    )

    out = json.loads(
        await ft.patch_tool(
            mode="patch",
            patch=(
                "*** Begin Patch\n"
                "*** Update File: target.py\n"
                "@@\n"
                "-WORKSPACE_ORIGINAL\n"
                "+WORKSPACE_PATCHED\n"
                "*** End Patch\n"
            ),
            task_id=task_id,
        )
    )

    expected = str((workspace / "target.py").resolve())
    assert not out.get("error"), out
    assert out.get("resolved_path") == expected
    assert out.get("files_modified") == [expected]
    # The workspace file — which the tool locked and reported — was edited.
    assert (workspace / "target.py").read_text() == "WORKSPACE_PATCHED\n"
    # The decoy (backend cwd) was left untouched.
    assert (decoy / "target.py").read_text() == "DECOY_ORIGINAL\n"
