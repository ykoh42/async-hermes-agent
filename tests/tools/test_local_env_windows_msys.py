import pytest

from unittest.mock import patch

from tools.environments.local import (
    _bash_safe_path,
    _msys_to_windows_path,
    _resolve_local_initial_cwd,
    _windows_to_msys_path,
)


def test_msys_drive_path_translation_is_idempotent():
    with patch("tools.environments.local._IS_WINDOWS", True):
        assert _msys_to_windows_path("/c/Users/test") == "C:\\Users\\test"
        assert _msys_to_windows_path("/mnt/d/work") == "D:\\work"
        assert _msys_to_windows_path("C:\\Users\\test") == "C:\\Users\\test"


def test_non_windows_path_is_unchanged():
    with patch("tools.environments.local._IS_WINDOWS", False):
        assert _msys_to_windows_path("/c/Users/test") == "/c/Users/test"


def test_windows_drive_path_is_translated_for_bash():
    with patch("tools.environments.local._IS_WINDOWS", True):
        assert _windows_to_msys_path(r"C:\Users\test") == "/c/Users/test"
        assert _windows_to_msys_path("D:/work/repo") == "/d/work/repo"
        assert _windows_to_msys_path("/tmp/repo") == "/tmp/repo"


def test_bash_safe_path_normalizes_mixed_windows_separators():
    with patch("tools.environments.local._IS_WINDOWS", True):
        assert _bash_safe_path(r"C:\Users\test\notes.txt") == (
            "/c/Users/test/notes.txt"
        )
        assert _bash_safe_path(r"/c/Users\test\notes.txt") == (
            "/c/Users/test/notes.txt"
        )


@pytest.mark.asyncio
async def test_initial_cwd_keeps_native_windows_absolute_path():
    with patch("tools.environments.local._IS_WINDOWS", True):
        assert await _resolve_local_initial_cwd(r"C:\Users\test") == r"C:\Users\test"
