from __future__ import annotations

from tools import file_tools, file_state


def test_clear_file_ops_cache_releases_task_state() -> None:
    task_id = "cleanup-test"
    file_tools._read_state(task_id)
    with file_tools._patch_failure_lock:
        file_tools._patch_failure_tracker[task_id] = {"/tmp/example": 1}
    file_state.get_registry()._reads[task_id] = {
        "/tmp/example": (1.0, 1.0, False)
    }

    file_tools.clear_file_ops_cache(task_id)

    assert task_id not in file_tools._read_tracker
    assert task_id not in file_tools._patch_failure_tracker
    assert task_id not in file_state.get_registry()._reads
