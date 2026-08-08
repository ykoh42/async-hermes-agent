"""Tool-surface cwd contract tests for embedding workspaces.

These cover the platform-neutral part of #29265: once an embedder has resolved
``TERMINAL_CWD``, the user-visible tool surfaces should agree on that workspace.

Unlike the system-prompt readers fixed in the cwd-resolver cluster
(agent/runtime_cwd.py), these tool sites already read ``TERMINAL_CWD``-first and
were deliberately left out of scope. This file is a *characterization* guard: it
pins the already-correct behavior so the supersession of PR #29365 is airtight
and a future refactor of these sites can't silently regress the contract.
"""

from __future__ import annotations

import pytest

from tools import terminal_tool


@pytest.mark.asyncio
async def test_terminal_env_config_uses_terminal_cwd(monkeypatch, tmp_path):
    """The terminal tool's default cwd should come from TERMINAL_CWD."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(workspace))

    config = await terminal_tool._get_env_config()

    assert config["cwd"] == str(workspace)
