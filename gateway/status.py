"""Process-liveness helper retained for MCP stdio cleanup."""

from __future__ import annotations

import psutil


def _pid_exists(pid: int) -> bool:
    """Return whether *pid* identifies a live, non-zombie process."""
    try:
        process = psutil.Process(int(pid))
        try:
            if process.status() == psutil.STATUS_ZOMBIE:
                return False
        except psutil.NoSuchProcess:
            return False
        except (psutil.AccessDenied, OSError):
            pass
        return psutil.pid_exists(process.pid)
    except (psutil.NoSuchProcess, TypeError, ValueError):
        return False
