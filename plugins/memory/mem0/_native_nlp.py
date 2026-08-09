"""Native-async boundary for Mem0's optional synchronous spaCy pipeline."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
from typing import Any

from hermes_cli.async_source_loader import locate_source_module

from ._native_oss import _finish_cleanup


def _entities(value: Any) -> list[tuple[str, str]]:
    if not isinstance(value, list):
        raise RuntimeError("Mem0 NLP worker returned invalid entity data")
    entities: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise RuntimeError("Mem0 NLP worker returned invalid entity data")
        entity_type, entity_text = item
        if not isinstance(entity_type, str) or not isinstance(entity_text, str):
            raise RuntimeError("Mem0 NLP worker returned invalid entity data")
        entities.append((entity_type, entity_text))
    return entities


class NativeNLP:
    """Own one stdio worker so spaCy I/O never blocks the caller's loop."""

    def __init__(self) -> None:
        self._process: Any = None
        self._available: bool | None = None
        self._initialize_lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()
        self._closed = False

    async def _start(self) -> Any:
        if self._closed:
            raise RuntimeError("Cannot use a closed Mem0 NLP runtime")
        if self._available is False:
            return None
        if self._process is not None:
            if self._process.returncode is not None:
                raise RuntimeError("Mem0 NLP worker exited unexpectedly")
            return self._process
        async with self._initialize_lock:
            if self._closed:
                raise RuntimeError("Cannot use a closed Mem0 NLP runtime")
            if self._available is False:
                return None
            if self._process is not None:
                return self._process
            if await locate_source_module("spacy") is None:
                self._available = False
                return None

            environment = dict(os.environ)
            environment["PYTHONUNBUFFERED"] = "1"
            spawn = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    sys.executable,
                    str(Path(__file__).with_name("_nlp_worker.py")),
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
                    error_message="Mem0 NLP spawn cleanup failed",
                )
                raise cancellation
            if process.stdin is None or process.stdout is None:
                await self._terminate(process)
                raise RuntimeError("Mem0 NLP worker has no stdio transport")
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
            await NativeNLP._terminate(process)

    async def _request(self, operation: str, **payload: Any) -> Any:
        async with self._request_lock:
            process = await self._start()
            if process is None:
                if operation == "lemmatize":
                    return payload["text"]
                if operation == "extract_batch":
                    return [[] for _ in payload["texts"]]
                return []

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
                    error_message="Mem0 NLP request cleanup failed",
                )
                raise
            if not response_line:
                self._process = None
                await self._terminate(process)
                raise RuntimeError("Mem0 NLP worker exited without a response")
            try:
                response = json.loads(response_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError("Mem0 NLP worker returned invalid JSON") from exc
            if response.get("error"):
                raise RuntimeError(str(response["error"]))
            return response.get("result")

    async def lemmatize(self, text: str) -> str:
        result = await self._request("lemmatize", text=text)
        if not isinstance(result, str):
            raise RuntimeError("Mem0 NLP worker returned invalid lemma data")
        return result

    async def extract(self, text: str) -> list[tuple[str, str]]:
        return _entities(await self._request("extract", text=text))

    async def extract_batch(
        self,
        texts: list[str],
    ) -> list[list[tuple[str, str]]]:
        result = await self._request("extract_batch", texts=texts)
        if not isinstance(result, list):
            raise RuntimeError("Mem0 NLP worker returned invalid batch data")
        return [_entities(item) for item in result]

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
                        error_message="Mem0 NLP cleanup failed",
                    )
