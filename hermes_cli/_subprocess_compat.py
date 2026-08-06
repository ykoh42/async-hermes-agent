"""Small cross-platform subprocess helpers used by runtime modules."""

import os


def windows_hide_flags() -> int:
    """Hide a short-lived child console on Windows; return zero elsewhere."""
    if os.name != "nt":
        return 0
    return 0x08000000  # CREATE_NO_WINDOW
