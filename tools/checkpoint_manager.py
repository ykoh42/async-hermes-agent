"""Compatibility boundary for the removed workspace checkpoint feature."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


CHECKPOINT_BASE = get_hermes_home() / "checkpoints"


class CheckpointManager:
    """No-op manager retained for agent lifecycle compatibility."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.enabled = False

    def new_turn(self) -> None:
        return None

    def ensure_checkpoint(self, working_dir: str, reason: str = "auto") -> bool:
        return False

    def get_working_dir_for_path(self, path: str) -> str:
        return str(Path(path).expanduser().resolve().parent)

    def list_checkpoints(self, working_dir: str) -> list[dict[str, Any]]:
        return []

    def diff(self, working_dir: str, commit_hash: str) -> dict[str, Any]:
        return {"success": False, "error": "Workspace checkpoints are not included in this runtime."}

    def restore(self, working_dir: str, commit_hash: str) -> dict[str, Any]:
        return {"success": False, "error": "Workspace checkpoints are not included in this runtime."}


def store_status() -> dict[str, Any]:
    return {"enabled": False, "message": "Workspace checkpoints are not included in this runtime."}


def prune_checkpoints(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return store_status()


def clear_all(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return store_status()


def clear_legacy(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return store_status()
