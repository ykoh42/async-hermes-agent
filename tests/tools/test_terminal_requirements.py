"""The async distribution intentionally supports only the local terminal."""

from tools.terminal_tool import check_terminal_requirements


def test_local_terminal_requirements() -> None:
    assert check_terminal_requirements() is True
