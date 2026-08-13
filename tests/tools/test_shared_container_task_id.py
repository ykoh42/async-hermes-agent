"""Upstream task ids share sandboxes unless isolation is requested."""

import pytest

from tools import terminal_tool


@pytest.fixture(autouse=True)
def _clean_overrides():
    """Ensure no stray overrides from other tests leak in."""
    before = dict(terminal_tool._task_env_overrides)
    terminal_tool._task_env_overrides.clear()
    yield
    terminal_tool._task_env_overrides.clear()
    terminal_tool._task_env_overrides.update(before)


def test_none_task_id_maps_to_default():
    assert terminal_tool._resolve_container_task_id(None) == "default"


def test_task_id_argument_remains_required():
    with pytest.raises(TypeError):
        terminal_tool._resolve_container_task_id()


def test_empty_task_id_maps_to_default():
    assert terminal_tool._resolve_container_task_id("") == "default"


def test_cwd_override_collapses_to_shared_environment(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    terminal_tool.register_task_env_overrides(
        "session-abc", {"cwd": str(workspace)}
    )
    try:
        assert terminal_tool._resolve_container_task_id("session-abc") == "default"
    finally:
        terminal_tool.clear_task_env_overrides("session-abc")


@pytest.mark.parametrize(
    "override",
    [
        {"docker_image": "image"},
        {"modal_image": "image"},
        {"singularity_image": "image"},
        {"daytona_image": "image"},
        {"env_type": "docker"},
    ],
)
def test_backend_override_keeps_isolated_task_id(override):
    terminal_tool.register_task_env_overrides("rollout", override)

    assert terminal_tool._resolve_container_task_id("rollout") == "rollout"


def test_clear_override_removes_cwd_anchor(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    terminal_tool.register_task_env_overrides(
        "session", {"cwd": str(workspace)}
    )
    terminal_tool.clear_task_env_overrides("session")

    assert terminal_tool.resolve_task_overrides("session") == {}
    assert terminal_tool.get_session_cwd("session") != str(workspace)
