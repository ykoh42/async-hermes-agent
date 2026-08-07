"""Behavioral coverage for file-tool checkpoint path resolution."""

from types import SimpleNamespace

import pytest
from blockbuster import BlockBuster
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

from agent.tool_executor import _ensure_file_checkpoint
from tools.checkpoint_manager import CheckpointManager


@pytest.mark.asyncio
async def test_relative_file_checkpoint_uses_task_workspace(tmp_path, monkeypatch):
    """Checkpoint lookup must use the same cwd as a relative file mutation."""
    process_cwd = tmp_path / "opt" / "hermes"
    workspace_cwd = tmp_path / "opt" / "data" / "workspace"
    process_cwd.mkdir(parents=True)
    workspace_cwd.mkdir(parents=True)

    # Both directories contain content so checkpointing the wrong one would
    # still succeed and remain observable as the regression did in Docker.
    (process_cwd / "pyproject.toml").write_text("[project]\nname = 'hermes'\n")
    (workspace_cwd / "pyproject.toml").write_text(
        "[project]\nname = 'workspace'\n"
    )
    (workspace_cwd / "existing.txt").write_text("before\n")

    monkeypatch.chdir(process_cwd)
    monkeypatch.setenv("TERMINAL_CWD", str(workspace_cwd))
    monkeypatch.setattr(
        "tools.checkpoint_manager.CHECKPOINT_BASE",
        tmp_path / "checkpoints",
    )

    manager = CheckpointManager(enabled=True)
    agent = SimpleNamespace(_checkpoint_mgr=manager)

    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        blockbuster = BlockBuster()
        blockbuster.activate()
        try:
            await _ensure_file_checkpoint(
                agent,
                "write_file",
                {"path": "test_permissions2.txt"},
                "gateway-session",
            )
            workspace_checkpoints = await manager.list_checkpoints(
                str(workspace_cwd)
            )
            process_checkpoints = await manager.list_checkpoints(str(process_cwd))
        finally:
            blockbuster.deactivate()

    assert workspace_checkpoints
    assert process_checkpoints == []
