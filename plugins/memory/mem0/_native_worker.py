"""Owned native-async subprocess boundary for optional Mem0 transforms."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
from typing import Any

from hermes_cli.async_source_loader import locate_source_module

from ._native_oss import _finish_cleanup

_BUILTIN_WORKER_ERRORS = {
    error.__name__: error
    for error in (
        FileNotFoundError,
        IndexError,
        KeyError,
        LookupError,
        PermissionError,
        TimeoutError,
        TypeError,
        ValueError,
    )
}


class NativeWorker:
    """Serialize requests through one optional-dependency transform worker."""

    def __init__(
        self,
        dependency: str,
        *,
        worker_filename: str = "_transform_worker.py",
    ) -> None:
        self._dependency = dependency
        self._worker_filename = worker_filename
        self._process: Any = None
        self._available: bool | None = None
        self._initialize_lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()
        self._closed = False

    async def _start(self) -> Any:
        if self._closed:
            raise RuntimeError("Cannot use a closed Mem0 transform runtime")
        if self._available is False:
            return None
        if self._process is not None:
            if self._process.returncode is not None:
                raise RuntimeError("Mem0 transform worker exited unexpectedly")
            return self._process
        async with self._initialize_lock:
            if self._closed:
                raise RuntimeError("Cannot use a closed Mem0 transform runtime")
            if self._available is False:
                return None
            if self._process is not None:
                return self._process
            if await locate_source_module(self._dependency) is None:
                self._available = False
                return None

            environment = dict(os.environ)
            environment["PYTHONUNBUFFERED"] = "1"
            spawn = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    sys.executable,
                    str(Path(__file__).with_name(self._worker_filename)),
                    "--stdio",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                    env=environment,
                )
            )
            try:
                process = await asyncio.shield(spawn)
            except asyncio.CancelledError as cancellation:
                try:
                    process = await spawn
                except Exception as exc:
                    raise cancellation from exc
                await _finish_cleanup(
                    self._terminate(process),
                    error_message="Mem0 transform spawn cleanup failed",
                )
                raise cancellation
            if process.stdin is None or process.stdout is None:
                await self._terminate(process)
                raise RuntimeError("Mem0 transform worker has no stdio transport")
            self._available = True
            self._process = process
            return process

    @staticmethod
    async def _terminate(process: Any) -> None:
        if process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        await process.wait()

    @staticmethod
    async def _shutdown(process: Any) -> None:
        try:
            if process.returncode is None and process.stdin is not None:
                process.stdin.write(b'{"operation":"close"}\n')
                await process.stdin.drain()
                process.stdin.close()
                await process.stdin.wait_closed()
            await process.wait()
        except (BrokenPipeError, ConnectionResetError):
            await NativeWorker._terminate(process)

    async def request(
        self,
        operation: str,
        *,
        fallback: Any,
        **payload: Any,
    ) -> Any:
        async with self._request_lock:
            process = await self._start()
            if process is None:
                return fallback

            request = {"operation": operation, **payload}
            process.stdin.write(
                (json.dumps(request, ensure_ascii=False) + "\n").encode()
            )
            try:
                await process.stdin.drain()
                response_line = await process.stdout.readline()
            except asyncio.CancelledError:
                self._process = None
                await _finish_cleanup(
                    self._terminate(process),
                    error_message="Mem0 transform request cleanup failed",
                )
                raise
            if not response_line:
                self._process = None
                await self._terminate(process)
                raise RuntimeError("Mem0 transform worker exited without a response")
            try:
                response = json.loads(response_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError("Mem0 transform worker returned invalid JSON") from exc
            error = response.get("error")
            if error:
                if isinstance(error, dict):
                    error_type = _BUILTIN_WORKER_ERRORS.get(
                        str(error.get("builtin_type") or "")
                    )
                    error_args = error.get("args")
                    if error_type is not None and isinstance(error_args, list):
                        raise error_type(*error_args)
                    raise RuntimeError(str(error.get("message") or error))
                raise RuntimeError(str(error))
            return response.get("result")

    async def close(self) -> None:
        async with self._request_lock:
            async with self._initialize_lock:
                if self._closed:
                    return
                self._closed = True
                process = self._process
                self._process = None
                if process is not None:
                    await _finish_cleanup(
                        self._shutdown(process),
                        error_message="Mem0 transform cleanup failed",
                    )
