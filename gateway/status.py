"""Process-liveness helper retained for MCP stdio cleanup."""

from __future__ import annotations

import sys

import aiofiles
import psutil


async def _pid_exists(pid: int) -> bool:
    """Return whether *pid* identifies a live, non-zombie process."""
    try:
        normalized_pid = int(pid)
    except (TypeError, ValueError):
        return False

    if normalized_pid < 0:
        return False

    if sys.platform.startswith("linux") and normalized_pid > 0:
        try:
            async with aiofiles.open(
                f"/proc/{normalized_pid}/stat", "rb"
            ) as stat_file:
                stat = await stat_file.read()
        except FileNotFoundError:
            return False
        except PermissionError:
            return psutil.pid_exists(normalized_pid)
        except OSError:
            return psutil.pid_exists(normalized_pid)

        _, separator, fields = stat.rpartition(b") ")
        if separator and fields[:1] == b"Z":
            return False
        return psutil.pid_exists(normalized_pid)

    try:
        process = psutil.Process(normalized_pid)
        try:
            if process.status() == psutil.STATUS_ZOMBIE:
                return False
        except psutil.NoSuchProcess:
            return False
        except (psutil.AccessDenied, OSError):
            pass
        return psutil.pid_exists(process.pid)
    except psutil.NoSuchProcess:
        return False
