"""Process-liveness helper retained for MCP stdio cleanup."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Iterable

import aiofiles


_PSUTIL_PROBE_SOURCE = r"""
import json
import sys

import psutil

action = sys.argv[1]
payload = json.loads(sys.argv[2])

if action == "pid_exists":
    result = psutil.pid_exists(int(payload))
elif action == "inspect":
    result = {}
    env_keys = payload.get("env_keys", [])
    for raw_pid in payload.get("pids", []):
        pid = int(raw_pid)
        try:
            process = psutil.Process(pid)
            name = process.name() or ""
            cmdline = process.cmdline() or []
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
        try:
            process_env = (process.environ() or {}) if env_keys else {}
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            process_env = {}
        result[str(pid)] = {
            "name": name,
            "cmdline": cmdline,
            "environ": {key: process_env.get(key, "") for key in env_keys},
        }
elif action == "tree":
    pid = int(payload)
    try:
        parent = psutil.Process(pid)
        processes = [*parent.children(recursive=True), parent]
        result = [
            {"pid": process.pid, "create_time": process.create_time()}
            for process in processes
        ]
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        result = []
elif action == "running":
    result = []
    for identity in payload:
        try:
            process = psutil.Process(int(identity["pid"]))
            if (
                process.create_time() == float(identity["create_time"])
                and process.is_running()
            ):
                result.append(identity)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
elif action == "signal":
    mode = payload["mode"]
    if mode not in {"terminate", "kill"}:
        raise ValueError(f"unknown process signal mode: {mode}")
    result = []
    for identity in payload["processes"]:
        try:
            process = psutil.Process(int(identity["pid"]))
            if process.create_time() != float(identity["create_time"]):
                continue
            getattr(process, mode)()
            result.append(identity)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
else:
    raise ValueError(f"unknown process probe action: {action}")

print(json.dumps(result, separators=(",", ":")))
"""

if os.name == "nt":  # Imported before the event loop, not on first PID probe.
    import ctypes
    from ctypes import wintypes


async def _finish_process_communicate(
    process: asyncio.subprocess.Process,
    communicate_task: asyncio.Task[tuple[bytes | None, bytes | None]],
) -> tuple[bytes | None, bytes | None]:
    """Drain and reap one owned ps process through repeated cancellation."""
    async def drain_or_wait() -> tuple[bytes | None, bytes | None]:
        try:
            return await communicate_task
        except BaseException:
            await process.wait()
            raise

    cleanup_task = asyncio.create_task(drain_or_wait())
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            output = await asyncio.shield(cleanup_task)
            break
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised below
            if cleanup_task.cancelled():
                raise
            if cancellation is None:
                cancellation = exc
        except Exception as exc:
            if cancellation is not None:
                raise cancellation from exc
            raise
    if cancellation is not None:
        raise cancellation
    return output


async def _run_psutil_probe(action: str, payload: Any) -> Any:
    """Run blocking psutil process-table inspection outside the event loop."""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _PSUTIL_PROBE_SOURCE,
        action,
        json.dumps(payload, separators=(",", ":")),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    communicate_task = asyncio.create_task(process.communicate())
    try:
        stdout, _ = await asyncio.shield(communicate_task)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await _finish_process_communicate(process, communicate_task)
        raise
    if process.returncode != 0 or not stdout:
        raise RuntimeError("psutil process probe failed")
    try:
        return json.loads(stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("psutil process probe returned invalid JSON") from exc


async def _inspect_processes(
    pids: Iterable[int],
    *,
    env_keys: Iterable[str] = (),
) -> dict[int, dict[str, Any]]:
    """Return psutil-compatible identity snapshots without loop blocking."""
    normalized_set: set[int] = set()
    for value in pids:
        try:
            pid = int(value)
        except (TypeError, ValueError):
            continue
        if pid > 0:
            normalized_set.add(pid)
    normalized = sorted(normalized_set)
    if not normalized:
        return {}
    normalized_env_keys = sorted(
        {str(key) for key in env_keys if isinstance(key, str) and key}
    )
    raw = await _run_psutil_probe(
        "inspect",
        {"pids": normalized, "env_keys": normalized_env_keys},
    )
    if not isinstance(raw, dict):
        return {}
    snapshots: dict[int, dict[str, Any]] = {}
    for raw_pid, value in raw.items():
        try:
            pid = int(raw_pid)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            snapshots[pid] = value
    return snapshots


async def _process_tree_identities(pid: int) -> list[dict[str, int | float]]:
    """Return recursive children then parent with psutil PID-reuse identities."""
    try:
        raw = await _run_psutil_probe("tree", int(pid))
    except (FileNotFoundError, OSError, RuntimeError, TimeoutError, ValueError):
        return []
    if not isinstance(raw, list):
        return []
    return _normalize_process_identities(raw)


async def _process_tree_pids(pid: int) -> list[int]:
    """Return recursive child PIDs followed by the parent, matching psutil."""
    return [
        int(identity["pid"])
        for identity in await _process_tree_identities(pid)
    ]


async def _running_process_identities(
    processes: Iterable[dict[str, int | float]],
) -> list[dict[str, int | float]]:
    """Keep only live processes whose PID still has the captured identity."""
    payload = list(processes)
    if not payload:
        return []
    try:
        raw = await _run_psutil_probe("running", payload)
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return []
    if not isinstance(raw, list):
        return []
    return _normalize_process_identities(raw)


async def _signal_process_identities(
    processes: Iterable[dict[str, int | float]],
    mode: str,
) -> None:
    """Signal captured process identities without touching recycled PIDs."""
    payload = list(processes)
    if not payload:
        return
    try:
        await _run_psutil_probe(
            "signal", {"mode": mode, "processes": payload}
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return


def _normalize_process_identities(
    values: Iterable[Any],
) -> list[dict[str, int | float]]:
    result: list[dict[str, int | float]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        try:
            pid = int(value["pid"])
            create_time = float(value["create_time"])
        except (KeyError, TypeError, ValueError):
            continue
        if pid > 0 and create_time >= 0:
            result.append({"pid": pid, "create_time": create_time})
    return result


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
    communicate_task = asyncio.create_task(process.communicate())
    try:
        stdout, _ = await asyncio.wait_for(
            asyncio.shield(communicate_task), timeout=2.0
        )
    except BaseException:
        if process.returncode is None:
            process.kill()
        await _finish_process_communicate(process, communicate_task)
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


async def _pid_exists_including_zombie(pid: int) -> bool:
    """Run the canonical ``psutil.pid_exists`` check off the event loop."""
    try:
        normalized_pid = int(pid)
    except (TypeError, ValueError):
        return False
    if normalized_pid < 0:
        return False
    if normalized_pid == 0:
        return True
    return (await _run_psutil_probe("pid_exists", normalized_pid)) is True
