from unittest.mock import patch

from tools.environments.local import _msys_to_windows_path, _resolve_local_initial_cwd


def test_msys_drive_path_translation_is_idempotent():
    with patch("tools.environments.local._IS_WINDOWS", True):
        assert _msys_to_windows_path("/c/Users/test") == "C:\\Users\\test"
        assert _msys_to_windows_path("/mnt/d/work") == "D:\\work"
        assert _msys_to_windows_path("C:\\Users\\test") == "C:\\Users\\test"


def test_non_windows_path_is_unchanged():
    with patch("tools.environments.local._IS_WINDOWS", False):
        assert _msys_to_windows_path("/c/Users/test") == "/c/Users/test"


def test_initial_cwd_keeps_native_windows_absolute_path():
    with patch("tools.environments.local._IS_WINDOWS", True):
        assert _resolve_local_initial_cwd(r"C:\Users\test") == r"C:\Users\test"
