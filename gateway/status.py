"""Process-liveness helper retained for MCP stdio cleanup."""

from __future__ import annotations

import asyncio
import os
import sys

import aiofiles

if os.name == "nt":  # Imported before the event loop, not on first PID probe.
    import ctypes
    from ctypes import wintypes


async def _ps_process_status(pid: int) -> str | None:
    """Return the platform ``ps`` status for *pid*, or ``None`` if absent."""
    process = await asyncio.create_subprocess_exec(
        "ps",
        "-o",
        "stat=",
        "-p",
        str(pid),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=2.0)
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    status = stdout.decode("utf-8", errors="replace").strip()
    return status or None


def _windows_pid_exists(pid: int) -> bool:
    """Non-blocking Windows process liveness check using a kernel handle."""
    synchronize = 0x00100000
    wait_object_0 = 0x00000000
    wait_timeout = 0x00000102
    error_access_denied = 5

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(synchronize, False, pid)
    if not handle:
        return ctypes.get_last_error() == error_access_denied
    try:
        result = kernel32.WaitForSingleObject(handle, 0)
        if result == wait_timeout:
            return True
        if result == wait_object_0:
            return False
        return True
    finally:
        kernel32.CloseHandle(handle)


async def _pid_exists(pid: int) -> bool:
    """Return whether *pid* identifies a live, non-zombie process."""
    try:
        normalized_pid = int(pid)
    except (TypeError, ValueError):
        return False

    if normalized_pid < 0:
        return False

    # Preserve psutil's historical contract for the platform kernel process.
    if normalized_pid == 0:
        return True

    if sys.platform.startswith("linux"):
        try:
            async with aiofiles.open(
                f"/proc/{normalized_pid}/stat", "rb"
            ) as stat_file:
                stat = await stat_file.read()
        except FileNotFoundError:
            return False
        except PermissionError:
            # A proc entry we cannot read still identifies a live process.
            return True
        except OSError:
            try:
                status = await _ps_process_status(normalized_pid)
            except (FileNotFoundError, OSError, asyncio.TimeoutError):
                return False
            return bool(status) and not status.startswith("Z")

        _, separator, fields = stat.rpartition(b") ")
        if separator and fields[:1] == b"Z":
            return False
        return True

    if os.name == "nt":
        return _windows_pid_exists(normalized_pid)

    try:
        status = await _ps_process_status(normalized_pid)
    except PermissionError:
        return True
    except (FileNotFoundError, OSError, asyncio.TimeoutError):
        return False
    return bool(status) and not status.startswith("Z")
