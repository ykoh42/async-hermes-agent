"""Parity coverage for delegated-child subprocess lineage isolation."""

from agent.delegation_context import delegated_child_context


def test_non_delegated_subprocess_env_preserves_inheritance_contract():
    from agent.delegation_context import delegated_child_subprocess_env

    assert delegated_child_subprocess_env() is None
    assert delegated_child_subprocess_env({"HERMES_KANBAN_TASK": "parent"}) == {
        "HERMES_KANBAN_TASK": "parent"
    }


def test_delegated_subprocess_env_scrubs_dispatcher_state_and_marks_lineage():
    from agent.delegation_context import (
        DELEGATED_CHILD_ENV_MARKER,
        delegated_child_subprocess_env,
        is_delegated_child_process_context,
    )

    with delegated_child_context():
        assert is_delegated_child_process_context() is True
        child_env = delegated_child_subprocess_env(
            {
                "PATH": "/bin",
                "HERMES_KANBAN_TASK": "parent-task",
                "HERMES_KANBAN_DB": "/tmp/parent.sqlite",
            }
        )

    assert child_env == {"PATH": "/bin", DELEGATED_CHILD_ENV_MARKER: "1"}


def test_process_marker_preserves_delegated_lineage(monkeypatch):
    from agent.delegation_context import (
        DELEGATED_CHILD_ENV_MARKER,
        is_delegated_child_process_context,
    )

    monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "1")
    assert is_delegated_child_process_context() is True
