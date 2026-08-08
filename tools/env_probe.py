"""Native-async local Python toolchain probe for the system prompt."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from typing import Optional
from weakref import WeakKeyDictionary

import aiofiles.os

logger = logging.getLogger(__name__)

_CACHED_LINE: Optional[str] = None
_PROBE_TASKS: WeakKeyDictionary[
    asyncio.AbstractEventLoop, asyncio.Task[str]
] = WeakKeyDictionary()
_PROBE_GEN = 0
_PROBE_WAIT_TIMEOUT = 10.0
_WAIT_ALREADY_TIMED_OUT = False
_REMOTE_BACKENDS = frozenset(
    {
        "docker",
        "singularity",
        "modal",
        "daytona",
        "ssh",
        "managed_modal",
        "vercel_sandbox",
    }
)
_which = aiofiles.os.wrap(shutil.which)


async def _run(
    command: list[str],
    timeout: float = 3.0,
) -> tuple[int, str, str]:
    """Run a short subprocess without blocking the event loop."""
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return -1, "", "not found"
    except OSError as exc:
        return -1, "", f"oserror: {exc}"

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout,
        )
    except TimeoutError:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(process.wait(), timeout=1.0)
        except TimeoutError:
            logger.debug("environment probe process did not reap promptly")
        return -1, "", "timeout"
    except asyncio.CancelledError:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(process.wait(), timeout=1.0)
        except TimeoutError:
            logger.debug("cancelled environment probe did not reap promptly")
        raise
    return (
        int(process.returncode or 0),
        stdout.decode("utf-8", "replace").strip(),
        stderr.decode("utf-8", "replace").strip(),
    )


async def _python_version_of(binary: str) -> Optional[str]:
    if not await _which(binary):
        return None
    code = "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
    returncode, stdout, _stderr = await _run([binary, "-c", code])
    return stdout if returncode == 0 and stdout else None


async def _has_pip_module(binary: str) -> bool:
    if not await _which(binary):
        return False
    returncode, _stdout, _stderr = await _run(
        [binary, "-m", "pip", "--version"]
    )
    return returncode == 0


async def _detect_pep668(binary: str) -> bool:
    if not await _which(binary):
        return False
    code = (
        "import os; stdlib = os.path.dirname(os.__file__); "
        "marker = os.path.join(stdlib, 'EXTERNALLY-MANAGED'); "
        "print('yes' if os.path.exists(marker) else 'no')"
    )
    returncode, stdout, _stderr = await _run([binary, "-c", code])
    return returncode == 0 and stdout == "yes"


async def _pip_python_version() -> Optional[str]:
    if not await _which("pip"):
        return None
    returncode, stdout, _stderr = await _run(["pip", "--version"])
    if returncode != 0 or "(python " not in stdout or not stdout.endswith(")"):
        return None
    return stdout.rsplit("(python ", 1)[1][:-1].strip() or None


async def _build_probe_line() -> str:
    backend = (os.getenv("TERMINAL_ENV") or "local").strip().lower()
    if backend in _REMOTE_BACKENDS:
        return ""

    py3_version, py_version, py3_has_pip, pip_bound_to, py3_pep668, uv_path = (
        await asyncio.gather(
            _python_version_of("python3"),
            _python_version_of("python"),
            _has_pip_module("python3"),
            _pip_python_version(),
            _detect_pep668("python3"),
            _which("uv"),
        )
    )
    has_uv = uv_path is not None
    mismatch = bool(
        pip_bound_to
        and py3_version
        and not py3_version.startswith(pip_bound_to)
    )
    if (
        py3_version is not None
        and py3_has_pip
        and not mismatch
        and (not py3_pep668 or has_uv)
    ):
        return ""

    parts: list[str] = []
    if py3_version:
        py3_part = f"python3={py3_version}"
        if not py3_has_pip:
            py3_part += " (no pip module)"
        parts.append(py3_part)
    else:
        parts.append("python3=missing")

    if py_version and py_version != py3_version:
        parts.append(f"python={py_version}")
    elif not py_version and py3_version:
        parts.append("python=missing (use python3)")

    if pip_bound_to:
        if mismatch:
            parts.append(f"pip→python{pip_bound_to} (mismatch)")
        elif not py3_has_pip:
            parts.append(f"pip→python{pip_bound_to}")
    elif not py3_has_pip:
        parts.append("pip=missing")

    if py3_pep668:
        parts.append("PEP 668=yes (use venv or uv)")
    if has_uv:
        parts.append("uv=installed")
    return "Python toolchain: " + ", ".join(parts) + "." if parts else ""


async def _probe_worker(generation: int) -> str:
    global _CACHED_LINE
    try:
        line = await _build_probe_line()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.debug("env_probe failed: %s", exc)
        line = ""
    if generation == _PROBE_GEN:
        _CACHED_LINE = line
    return line


def _ensure_probe_started() -> asyncio.Task[str]:
    loop = asyncio.get_running_loop()
    task = _PROBE_TASKS.get(loop)
    if task is None or (task.done() and _CACHED_LINE is None):
        task = loop.create_task(
            _probe_worker(_PROBE_GEN),
            name="hermes-environment-probe",
        )
        _PROBE_TASKS[loop] = task
    return task


async def get_environment_probe_line(*, force_refresh: bool = False) -> str:
    """Return the cached probe line, failing open after a bounded wait."""
    global _CACHED_LINE, _PROBE_GEN, _WAIT_ALREADY_TIMED_OUT
    if force_refresh:
        await _reset_cache_for_tests()
    if _CACHED_LINE is not None:
        return _CACHED_LINE

    task = _ensure_probe_started()
    wait_timeout = 0.05 if _WAIT_ALREADY_TIMED_OUT else _PROBE_WAIT_TIMEOUT
    done, _pending = await asyncio.wait({task}, timeout=wait_timeout)
    if done:
        return await task
    if not _WAIT_ALREADY_TIMED_OUT:
        _WAIT_ALREADY_TIMED_OUT = True
        logger.warning(
            "env_probe did not finish within %.0fs; building the system "
            "prompt without the Python toolchain line",
            _PROBE_WAIT_TIMEOUT,
        )
    return ""


async def warm_environment_probe_async() -> None:
    """Populate the cached probe without leaving an unowned background task."""
    await _ensure_probe_started()


async def _reset_cache_for_tests() -> None:
    global _CACHED_LINE, _PROBE_GEN, _WAIT_ALREADY_TIMED_OUT
    _PROBE_GEN += 1
    _CACHED_LINE = None
    _WAIT_ALREADY_TIMED_OUT = False
    current_loop = asyncio.get_running_loop()
    tasks = [
        task
        for task in _PROBE_TASKS.values()
        if task.get_loop() is current_loop
    ]
    _PROBE_TASKS.clear()
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
