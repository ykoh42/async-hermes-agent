"""Native-async direct Modal cloud execution environment."""

from __future__ import annotations

import asyncio
import base64
import codecs
import logging
import os
import shlex
import weakref
from collections import deque
from pathlib import Path
from typing import Any, Optional

import aiofiles
import aiofiles.os

from agent.secret_scope import get_secret, is_multiplex_active
from hermes_constants import get_hermes_home
from tools.environments.base import (
    BaseEnvironment,
    _BoundedOutputCollector,
    _UNBOUNDED_CAPTURE_CHARS,
    _load_json_store,
    _save_json_store,
    touch_activity_if_due,
)
from tools.environments.file_sync import (
    FileSyncManager,
    iter_sync_files,
    quoted_rm_command,
)

logger = logging.getLogger(__name__)

_SNAPSHOT_STORE = get_hermes_home() / "modal_snapshots.json"
_IMPORTED_SNAPSHOT_STORE = _SNAPSHOT_STORE
_DIRECT_SNAPSHOT_NAMESPACE = "direct"
_SNAPSHOT_STORE_LOCKS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    dict[str, weakref.ReferenceType[asyncio.Lock]],
] = weakref.WeakKeyDictionary()


class _DeferredStreamCapture:
    """Bound one concurrently drained stream until its ordered replay point."""

    def __init__(self, collector: _BoundedOutputCollector):
        self._prefix_limit = (
            collector._SPILL_CAP_CHARS
            if collector._spill_path is not None
            else 0
        )
        self._tail_limit = collector.max_chars
        self._prefix: list[str] = []
        self._tail: deque[str] = deque()
        self._prefix_chars = 0
        self._tail_chars = 0
        self.total_chars = 0

    def append(self, text: str) -> None:
        if not text:
            return
        self.total_chars += len(text)
        if self._prefix_chars < self._prefix_limit:
            take = min(self._prefix_limit - self._prefix_chars, len(text))
            if take:
                self._prefix.append(text[:take])
                self._prefix_chars += take
        self._tail.append(text)
        self._tail_chars += len(text)
        while self._tail_chars > self._tail_limit:
            excess = self._tail_chars - self._tail_limit
            first = self._tail[0]
            if len(first) <= excess:
                self._tail.popleft()
                self._tail_chars -= len(first)
            else:
                self._tail[0] = first[excess:]
                self._tail_chars -= excess

    def replay_into(self, collector: _BoundedOutputCollector) -> None:
        prefix = "".join(self._prefix)
        collector.append(prefix)
        remaining = self.total_chars - len(prefix)
        if remaining <= 0:
            return
        tail = "".join(self._tail)
        replayed_tail = tail[-min(len(tail), remaining) :]
        collector._account_discarded(remaining - len(replayed_tail))
        collector.append(replayed_tail)


def _snapshot_store_path() -> Path:
    configured = Path(_SNAPSHOT_STORE)
    if configured != _IMPORTED_SNAPSHOT_STORE:
        return configured
    return get_hermes_home() / "modal_snapshots.json"


async def _snapshot_store_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    locks = _SNAPSHOT_STORE_LOCKS.setdefault(loop, {})
    realpath = aiofiles.os.wrap(os.path.realpath)
    key = os.path.normcase(str(await realpath(_snapshot_store_path())))
    lock_ref = locks.get(key)
    lock = lock_ref() if lock_ref is not None else None
    if lock is None:
        lock = asyncio.Lock()
        locks[key] = weakref.ref(lock)
    return lock


async def _load_snapshots() -> dict:
    return await _load_json_store(_snapshot_store_path())


async def _save_snapshots(data: dict) -> None:
    await _save_json_store(_snapshot_store_path(), data)


def _direct_snapshot_key(task_id: str) -> str:
    return f"{_DIRECT_SNAPSHOT_NAMESPACE}:{task_id}"


async def _get_snapshot_restore_candidate(
    task_id: str,
) -> tuple[str | None, bool]:
    snapshots = await _load_snapshots()
    snapshot_id = snapshots.get(_direct_snapshot_key(task_id))
    if isinstance(snapshot_id, str) and snapshot_id:
        return snapshot_id, False
    legacy = snapshots.get(task_id)
    if isinstance(legacy, str) and legacy:
        return legacy, True
    return None, False


async def _store_direct_snapshot(task_id: str, snapshot_id: str) -> None:
    async with await _snapshot_store_lock():
        snapshots = await _load_snapshots()
        snapshots[_direct_snapshot_key(task_id)] = snapshot_id
        snapshots.pop(task_id, None)
        await _save_snapshots(snapshots)


async def _delete_direct_snapshot(
    task_id: str,
    snapshot_id: str | None = None,
) -> None:
    async with await _snapshot_store_lock():
        snapshots = await _load_snapshots()
        changed = False
        for key in (_direct_snapshot_key(task_id), task_id):
            value = snapshots.get(key)
            if value is not None and (snapshot_id is None or value == snapshot_id):
                snapshots.pop(key, None)
                changed = True
        if changed:
            await _save_snapshots(snapshots)


def _ensure_modal_sdk() -> None:
    try:
        import modal as _modal  # noqa: F401
    except ImportError as exc:
        raise ImportError("Direct Modal requires the 'modal==1.3.4' package") from exc


def _resolve_modal_image(image_spec: Any) -> Any:
    """Convert registry references or snapshot ids into Modal image objects."""
    _ensure_modal_sdk()
    import modal

    if not isinstance(image_spec, str):
        return image_spec
    if image_spec.startswith("im-"):
        return modal.Image.from_id(image_spec)
    lower = image_spec.lower()
    setup_commands = [
        "RUN rm -rf /usr/local/lib/python*/site-packages/pip* 2>/dev/null; "
        "python -m ensurepip --upgrade --default-pip 2>/dev/null || true",
    ]
    if any(base in lower for base in ("ubuntu", "debian")):
        setup_commands.insert(
            0,
            "RUN apt-get update -qq && apt-get install -y -qq "
            "python3 python3-venv > /dev/null 2>&1 || true",
        )
    return modal.Image.from_registry(
        image_spec,
        setup_dockerfile_commands=setup_commands,
    )


async def _await_owned(task: asyncio.Task[Any]) -> Any:
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised
            if task.cancelled():
                raise
            if cancellation is None:
                cancellation = exc
        except Exception as exc:
            if cancellation is not None:
                raise cancellation from exc
            raise
    if cancellation is not None:
        raise cancellation
    return result


class ModalEnvironment(BaseEnvironment):
    """Modal cloud execution via the SDK's native coroutine API."""

    _stdin_mode = "heredoc"
    _snapshot_timeout = 60
    _STDIN_CHUNK_SIZE = 1024 * 1024

    def __init__(
        self,
        image: str,
        cwd: str = "/root",
        timeout: int = 60,
        modal_sandbox_kwargs: Optional[dict[str, Any]] = None,
        persistent_filesystem: bool = True,
        task_id: str = "default",
    ):
        super().__init__(cwd=cwd, timeout=timeout)
        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._image = image
        self._sandbox_kwargs = dict(modal_sandbox_kwargs or {})
        self._sandbox = None
        self._app = None
        self._modal_client = None
        self._sync_manager: FileSyncManager | None = None
        self._initialize_lock = asyncio.Lock()
        self._cleanup_lock = asyncio.Lock()

    async def init_session(self) -> None:
        await self._ensure_initialized()

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._initialize_lock:
            if self._initialized:
                return
            _ensure_modal_sdk()
            try:
                await self._create_sandbox()
                self._sync_manager = FileSyncManager(
                    get_files_fn=lambda: iter_sync_files("/root/.hermes"),
                    upload_fn=self._modal_upload,
                    delete_fn=self._modal_delete,
                    bulk_upload_fn=self._modal_bulk_upload,
                    bulk_download_fn=self._modal_bulk_download,
                )
                await self._sync_manager.sync(force=True)
                await BaseEnvironment.init_session(self)
            except BaseException:
                if self._sandbox is not None or self._modal_client is not None:
                    cleanup = asyncio.create_task(self._terminate_sandbox())
                    await _await_owned(cleanup)  # noqa: ASYNC120
                raise

    async def _create_profile_client(self, modal: Any) -> Any | None:
        """Create a client for the active profile without consulting raw env."""
        token_id = get_secret("MODAL_TOKEN_ID")
        token_secret = get_secret("MODAL_TOKEN_SECRET") if token_id else None
        if not (token_id and token_secret) and is_multiplex_active():
            # Modal's ~/.modal.toml store is tied to the OS user and carries no
            # Hermes profile discriminator. Never borrow that account for a
            # multiplexed request; a caller-owned explicit SDK client remains
            # the only supported non-env credential path in this mode.
            raise RuntimeError(
                "Direct Modal requires non-empty profile-scoped "
                "MODAL_TOKEN_ID and MODAL_TOKEN_SECRET while Hermes profile "
                "multiplexing is active"
            )
        if token_id and token_secret:
            from modal_proto import api_pb2

            client = modal.Client(
                modal.config.config["server_url"],
                api_pb2.CLIENT_TYPE_CLIENT,
                (token_id, token_secret),
            )
            try:
                await client.__aenter__()
            except BaseException:
                cleanup = asyncio.create_task(
                    client.__aexit__(None, None, None),
                    name="modal-profile-client-init-cleanup",
                )
                await _await_owned(cleanup)  # noqa: ASYNC120
                raise
            return client
        return None

    async def _create_sandbox(self) -> None:
        import modal

        snapshot_id, restored_from_legacy = (
            await _get_snapshot_restore_candidate(self._task_id)
            if self._persistent
            else (None, False)
        )
        modal_client = self._sandbox_kwargs.get("client")
        if modal_client is None:
            self._modal_client = await self._create_profile_client(modal)
            modal_client = self._modal_client
        lookup_kwargs = {"create_if_missing": True}
        if modal_client is not None:
            lookup_kwargs["client"] = modal_client
        self._app = await modal.App.lookup.aio("hermes-agent", **lookup_kwargs)

        async def create(image_spec: Any):
            kwargs = dict(self._sandbox_kwargs)
            if modal_client is not None:
                kwargs["client"] = modal_client
            return await modal.Sandbox.create.aio(
                "sleep",
                "infinity",
                image=_resolve_modal_image(image_spec),
                app=self._app,
                timeout=int(kwargs.pop("timeout", 3600)),
                **kwargs,
            )

        try:
            self._sandbox = await create(snapshot_id or self._image)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not snapshot_id:
                raise
            logger.warning(
                "Modal: failed to restore snapshot %s, retrying with base image: %s",
                snapshot_id[:20],
                exc,
            )
            await _delete_direct_snapshot(self._task_id, snapshot_id)
            self._sandbox = await create(self._image)
        else:
            if snapshot_id and restored_from_legacy:
                await _store_direct_snapshot(self._task_id, snapshot_id)

    async def _write_process_stdin(self, process: Any, data: str) -> None:
        offset = 0
        while offset < len(data):
            process.stdin.write(data[offset : offset + self._STDIN_CHUNK_SIZE])
            await process.stdin.drain.aio()
            offset += self._STDIN_CHUNK_SIZE
        process.stdin.write_eof()
        await process.stdin.drain.aio()

    async def _modal_upload(self, host_path: str, remote_path: str) -> None:
        async with aiofiles.open(host_path, "rb") as handle:
            payload = base64.b64encode(await handle.read()).decode("ascii")
        remote_dir = str(Path(remote_path).parent)
        command = (
            f"mkdir -p {shlex.quote(remote_dir)} && "
            f"base64 -d > {shlex.quote(remote_path)}"
        )
        sandbox = self._require_sandbox()
        process = await sandbox.exec.aio("bash", "-c", command)
        await self._write_process_stdin(process, payload)
        exit_code = await process.wait.aio()
        if exit_code != 0:
            stderr = await process.stderr.read.aio()
            raise RuntimeError(f"Modal upload failed (exit {exit_code}): {stderr}")

    async def _modal_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        for host_path, remote_path in files:
            await self._modal_upload(host_path, remote_path)

    async def _modal_bulk_download(self, destination: Path) -> None:
        sandbox = self._require_sandbox()
        process = await sandbox.exec.aio(
            "bash",
            "-c",
            "tar cf - -C / root/.hermes",
        )
        data = await process.stdout.read.aio()
        exit_code = await process.wait.aio()
        if exit_code != 0:
            raise RuntimeError(f"Modal bulk download failed (exit {exit_code})")
        if isinstance(data, str):
            data = data.encode()
        async with aiofiles.open(destination, "wb") as handle:
            await handle.write(data)

    async def _modal_delete(self, remote_paths: list[str]) -> None:
        if not remote_paths:
            return
        sandbox = self._require_sandbox()
        process = await sandbox.exec.aio(
            "bash",
            "-c",
            quoted_rm_command(remote_paths),
        )
        await process.wait.aio()

    async def _before_execute(self) -> None:
        if self._sync_manager is not None:
            await self._sync_manager.sync()

    def _require_sandbox(self):
        if self._sandbox is None:
            raise RuntimeError("Modal sandbox is not initialized")
        return self._sandbox

    async def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int | float = 120,  # noqa: ASYNC109 - retained environment API
        stdin_data: str | None = None,
        bounded_capture: bool = False,
    ) -> dict[str, Any]:
        sandbox = self._require_sandbox()
        args = ["bash"]
        args.extend(["-l", "-c", cmd_string] if login else ["-c", cmd_string])
        collector = (
            await self._bounded_output_collector()
            if bounded_capture
            else _BoundedOutputCollector(_UNBOUNDED_CAPTURE_CHARS)
        )
        deferred_stderr = _DeferredStreamCapture(collector)
        try:
            process = await sandbox.exec.aio(*args, timeout=timeout)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Upstream's _ThreadedProcessHandle translates SDK failures into
            # an empty command result with return code 1.
            return {"output": "", "returncode": 1}

        async def collect_bounded() -> tuple[_BoundedOutputCollector, int]:
            async def drain(stream: Any, append: Any) -> None:
                decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                try:
                    async for chunk in stream:
                        if isinstance(chunk, bytes):
                            chunk = decoder.decode(chunk)
                        append(str(chunk or ""))
                finally:
                    append(decoder.decode(b"", final=True))

            try:
                async with asyncio.TaskGroup() as group:
                    stdout_task = group.create_task(
                        drain(process.stdout, collector.append)
                    )
                    stderr_task = group.create_task(
                        drain(process.stderr, deferred_stderr.append)
                    )
                    wait_task = group.create_task(process.wait.aio())
                    if stdin_data is not None:
                        group.create_task(
                            self._write_process_stdin(process, stdin_data)
                        )
            except* TimeoutError as timeout_group:
                raise TimeoutError from timeout_group
            await stdout_task
            await stderr_task
            if collector.total_chars and deferred_stderr.total_chars:
                collector.append("\n")
            deferred_stderr.replay_into(collector)
            return collector, int(await wait_task)

        def replay_partial_stderr() -> None:
            if collector.total_chars and deferred_stderr.total_chars:
                collector.append("\n")
            deferred_stderr.replay_into(collector)

        collection_task = asyncio.create_task(collect_bounded())

        async def cancel_collection() -> None:
            collection_task.cancel()
            await asyncio.gather(collection_task, return_exceptions=True)

        async def terminate_and_cancel_collection() -> None:
            try:
                await self._terminate_sandbox()
            finally:
                await cancel_collection()

        started = asyncio.get_running_loop().time()
        # Preserve upstream's worker.run_coroutine(..., timeout=timeout + 30)
        # transport grace period around Modal's own command timeout.
        deadline = started + float(timeout) + 30
        activity_state = {"last_touch": started, "start": started}
        try:
            while not collection_task.done():
                from tools.interrupt import is_interrupted

                if is_interrupted():
                    cleanup = asyncio.create_task(terminate_and_cancel_collection())
                    await _await_owned(cleanup)
                    replay_partial_stderr()
                    suffix = "\n[Command interrupted]"
                    rendered = collector.render(suffix=suffix)
                    if bounded_capture:
                        return await self._finalize_wait_result(
                            collector,
                            rendered,
                            130,
                        )
                    return {"output": rendered, "returncode": 130}
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError
                await asyncio.wait(
                    {collection_task},
                    timeout=min(0.25, remaining),
                )
                if collection_task.done():
                    break
                touch_activity_if_due(
                    activity_state,
                    "terminal command running",
                )
            collector, exit_code = await collection_task
        except TimeoutError:
            cleanup = asyncio.create_task(terminate_and_cancel_collection())
            await _await_owned(cleanup)  # noqa: ASYNC120
            replay_partial_stderr()
            suffix = f"\n[Command timed out after {timeout}s]"
            rendered = collector.render(suffix=suffix)
            if collector.total_chars == 0:
                rendered = rendered.lstrip()
            if bounded_capture:
                return await self._finalize_wait_result(
                    collector,
                    rendered,
                    124,
                )
            return {"output": rendered, "returncode": 124}
        except asyncio.CancelledError:
            cleanup = asyncio.create_task(terminate_and_cancel_collection())
            await _await_owned(cleanup)  # noqa: ASYNC120
            raise
        except Exception:
            # Match _ThreadedProcessHandle: an SDK execution failure becomes
            # an empty rc=1 result without invoking its cancellation hook.
            return {"output": "", "returncode": 1}
        if bounded_capture:
            return await self._finalize_wait_result(
                collector,
                collector.render(),
                exit_code,
            )
        return {"output": collector.render(), "returncode": exit_code}

    async def _terminate_sandbox(self) -> None:
        sandbox = self._sandbox
        modal_client = self._modal_client
        self._sandbox = None
        self._modal_client = None
        self._initialized = False
        try:
            if sandbox is None:
                return
            try:
                await sandbox.terminate.aio()
            except Exception:
                pass
        finally:
            if modal_client is not None:
                try:
                    await modal_client.__aexit__(None, None, None)
                except Exception:
                    pass

    async def cleanup(self) -> None:
        async with self._cleanup_lock:
            sandbox = self._sandbox
            if sandbox is None and self._modal_client is None:
                return
            if sandbox is None:
                cleanup = asyncio.create_task(self._terminate_sandbox())
            else:
                cleanup = asyncio.create_task(self._cleanup_owned(sandbox))
            await _await_owned(cleanup)

    async def _cleanup_owned(self, sandbox: Any) -> None:
        try:
            if self._sync_manager is not None:
                await self._sync_manager.sync_back()
            if self._persistent:
                try:
                    image = await sandbox.snapshot_filesystem.aio()
                    snapshot_id = getattr(image, "object_id", None)
                    if isinstance(snapshot_id, str) and snapshot_id:
                        await _store_direct_snapshot(self._task_id, snapshot_id)
                except Exception as exc:
                    logger.warning("Modal: filesystem snapshot failed: %s", exc)
        finally:
            await self._terminate_sandbox()
            self._app = None
            self._sync_manager = None
