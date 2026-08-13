"""Native-async Singularity/Apptainer persistent container environment."""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import logging
import os
import threading
import uuid
import weakref
from pathlib import Path
from typing import Any

import aiofiles.os

from hermes_constants import get_hermes_home
from tools.environments.base import (
    BaseEnvironment,
    _BoundedOutputCollector,
    _UNBOUNDED_CAPTURE_CHARS,
    _load_json_store,
    _save_json_store,
    get_sandbox_dir,
    touch_activity_if_due,
)
from tools.environments.local import _terminate_process

logger = logging.getLogger(__name__)

_SNAPSHOT_STORE = get_hermes_home() / "singularity_snapshots.json"
_IMPORTED_SNAPSHOT_STORE = _SNAPSHOT_STORE
_SIF_BUILD_LOCKS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    dict[str, weakref.ReferenceType[asyncio.Lock]],
] = weakref.WeakKeyDictionary()
_SIF_BUILD_LOCKS_GUARD = threading.RLock()
_SNAPSHOT_STORE_LOCKS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    dict[str, weakref.ReferenceType[asyncio.Lock]],
] = weakref.WeakKeyDictionary()


def _snapshot_store_path() -> Path:
    configured = Path(_SNAPSHOT_STORE)
    if configured != _IMPORTED_SNAPSHOT_STORE:
        return configured
    return get_hermes_home() / "singularity_snapshots.json"


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


async def _sif_build_lock(path: Path) -> asyncio.Lock:
    """Return a loop-local lock for one canonical SIF output path."""
    loop = asyncio.get_running_loop()
    realpath = aiofiles.os.wrap(os.path.realpath)
    key = os.path.normcase(str(await realpath(path)))
    with _SIF_BUILD_LOCKS_GUARD:
        for candidate in tuple(_SIF_BUILD_LOCKS):
            if candidate.is_closed():
                _SIF_BUILD_LOCKS.pop(candidate, None)
        locks = _SIF_BUILD_LOCKS.setdefault(loop, {})
        lock_ref = locks.get(key)
        lock = lock_ref() if lock_ref is not None else None
        if lock is None:
            lock = asyncio.Lock()
            locks[key] = weakref.ref(lock)
        return lock


async def _find_singularity_executable() -> str:
    """Locate the apptainer or singularity CLI binary."""
    path_value = os.getenv("PATH", os.defpath)
    for name in ("apptainer", "singularity"):
        for directory in path_value.split(os.pathsep):
            candidate = Path(directory or ".") / name
            if await aiofiles.os.path.isfile(candidate) and await aiofiles.os.access(
                candidate, os.X_OK
            ):
                return str(candidate)
    raise RuntimeError(
        "Neither 'apptainer' nor 'singularity' was found in PATH. "
        "Install Apptainer (https://apptainer.org/docs/admin/main/installation.html) "
        "or Singularity and ensure the CLI is available."
    )


async def _finish_process(
    process: asyncio.subprocess.Process,
    *,
    input_data: bytes | None = None,
    timeout: float,
    _output_collector: _BoundedOutputCollector | None = None,
) -> tuple[bytes, int]:
    if _output_collector is not None:
        assert process.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        async def communicate_bounded() -> tuple[bytes, int]:
            async def drain_output() -> None:
                while chunk := await process.stdout.read(64 * 1024):
                    _output_collector.append(decoder.decode(chunk))
                tail = decoder.decode(b"", final=True)
                if tail:
                    _output_collector.append(tail)

            async def write_input() -> None:
                if input_data is None or process.stdin is None:
                    return
                try:
                    process.stdin.write(input_data)
                    await process.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    process.stdin.close()
                    with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                        await process.stdin.wait_closed()

            await asyncio.gather(drain_output(), write_input(), process.wait())
            return (
                _output_collector.render().encode("utf-8"),
                int(process.returncode or 0),
            )

        communication = asyncio.create_task(communicate_bounded())
        try:
            return await asyncio.wait_for(
                asyncio.shield(communication),
                timeout=timeout,
            )
        except (asyncio.CancelledError, TimeoutError):
            from tools.environments.file_sync import _await_owned

            async def finish() -> None:
                await _terminate_process(process)
                await communication

            cleanup = asyncio.create_task(finish())
            await _await_owned(cleanup)
            raise

    try:
        output, _ = await asyncio.wait_for(
            process.communicate(input_data), timeout=timeout
        )
    except (asyncio.CancelledError, TimeoutError):
        from tools.environments.file_sync import _await_owned

        cleanup = asyncio.create_task(_terminate_process(process))
        await _await_owned(cleanup)
        raise
    return output or b"", int(process.returncode or 0)


async def _run_command(
    argv: list[str],
    *,
    timeout: float,
    input_data: str | None = None,
    env: dict[str, str] | None = None,
    _output_collector: _BoundedOutputCollector | None = None,
) -> tuple[str, int]:
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=(asyncio.subprocess.PIPE if input_data is not None else asyncio.subprocess.DEVNULL),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
        start_new_session=os.name == "posix",
    )
    output, returncode = await _finish_process(
        process,
        input_data=(input_data.encode("utf-8") if input_data is not None else None),
        timeout=timeout,
        _output_collector=_output_collector,
    )
    return output.decode("utf-8", errors="replace"), returncode


async def _ensure_singularity_available() -> str:
    """Resolve the executable and verify that it responds."""
    executable = await _find_singularity_executable()
    try:
        output, returncode = await _run_command(
            [executable, "version"], timeout=10
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"Singularity backend selected but '{executable}' could not be executed."
        )
    except TimeoutError:
        raise RuntimeError(f"'{executable} version' timed out.")
    if returncode != 0:
        raise RuntimeError(
            f"'{executable} version' failed (exit code {returncode}): "
            f"{output.strip()[:200]}"
        )
    return executable


async def _load_snapshots() -> dict:
    return await _load_json_store(_snapshot_store_path())


async def _save_snapshots(data: dict) -> None:
    await _save_json_store(_snapshot_store_path(), data)


async def _store_snapshot(task_id: str, overlay_dir: str) -> None:
    async with await _snapshot_store_lock():
        snapshots = await _load_snapshots()
        snapshots[task_id] = overlay_dir
        await _save_snapshots(snapshots)


async def _get_scratch_dir() -> Path:
    from agent.secret_scope import get_secret

    custom = get_secret("TERMINAL_SCRATCH_DIR")
    if custom:
        path = Path(custom)
        await aiofiles.os.makedirs(path, exist_ok=True)
        return path

    scratch = Path("/scratch")
    if await aiofiles.os.path.exists(scratch) and await aiofiles.os.access(
        scratch, os.W_OK
    ):
        user_scratch = scratch / os.getenv("USER", "hermes") / "hermes-agent"
        await aiofiles.os.makedirs(user_scratch, exist_ok=True)
        logger.info("Using /scratch for sandboxes: %s", user_scratch)
        return user_scratch

    sandbox = await get_sandbox_dir() / "singularity"
    await aiofiles.os.makedirs(sandbox, exist_ok=True)
    return sandbox


async def _get_apptainer_cache_dir() -> Path:
    configured = os.getenv("APPTAINER_CACHEDIR")
    cache = Path(configured) if configured else await _get_scratch_dir() / ".apptainer"
    await aiofiles.os.makedirs(cache, exist_ok=True)
    return cache


async def _get_or_build_sif(image: str, executable: str = "apptainer") -> str:
    if image.endswith(".sif") and await aiofiles.os.path.exists(image):
        return image
    if not image.startswith("docker://"):
        return image

    image_name = image.replace("docker://", "").replace("/", "-").replace(":", "-")
    cache_dir = await _get_apptainer_cache_dir()
    sif_path = cache_dir / f"{image_name}.sif"
    if await aiofiles.os.path.exists(sif_path):
        return str(sif_path)

    async with await _sif_build_lock(sif_path):
        if await aiofiles.os.path.exists(sif_path):
            return str(sif_path)
        tmp_dir = cache_dir / "tmp"
        await aiofiles.os.makedirs(tmp_dir, exist_ok=True)
        from tools.environments.local import build_subprocess_env

        env = await build_subprocess_env(
            scrub_secrets=False,
            inherit_profile_home=False,
        )
        env["APPTAINER_TMPDIR"] = str(tmp_dir)
        env["APPTAINER_CACHEDIR"] = str(cache_dir)
        try:
            output, returncode = await _run_command(
                [executable, "build", str(sif_path), image],
                timeout=600,
                env=env,
            )
        except TimeoutError:
            logger.warning("SIF build timed out, falling back to docker:// URL")
            try:
                await aiofiles.os.remove(sif_path)
            except FileNotFoundError:
                pass
            return image
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "SIF build error: %s, falling back to docker:// URL", exc
            )
            return image
        if returncode != 0:
            logger.warning("SIF build failed, falling back to docker:// URL")
            logger.warning("  Error: %s", output[:500])
            return image
        return str(sif_path)


class SingularityEnvironment(BaseEnvironment):
    """Hardened persistent Singularity/Apptainer environment."""

    def __init__(
        self,
        image: str,
        cwd: str = "~",
        timeout: int = 60,
        cpu: float = 0,
        memory: int = 0,
        disk: int = 0,
        persistent_filesystem: bool = False,
        task_id: str = "default",
    ):
        super().__init__(cwd=cwd, timeout=timeout)
        self.executable: str | None = None
        self.image = image
        self.instance_id = f"hermes_{uuid.uuid4().hex[:12]}"
        self._instance_started = False
        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._overlay_dir: Path | None = None
        self._cpu = cpu
        self._memory = memory
        self._disk = disk
        self._lifecycle_lock = asyncio.Lock()

    async def init_session(self) -> None:
        async with self._lifecycle_lock:
            if not self._instance_started:
                self.executable = await _ensure_singularity_available()
                self.image = await _get_or_build_sif(self.image, self.executable)
                if self._persistent:
                    overlay_base = await _get_scratch_dir() / "hermes-overlays"
                    await aiofiles.os.makedirs(overlay_base, exist_ok=True)
                    self._overlay_dir = overlay_base / f"overlay-{self._task_id}"
                    await aiofiles.os.makedirs(self._overlay_dir, exist_ok=True)
                await self._start_instance()
            if not self._snapshot_ready:
                await super().init_session()

    async def _start_instance(self) -> None:
        if self.executable is None:
            raise RuntimeError("Singularity executable was not initialized")
        command = [self.executable, "instance", "start", "--containall", "--no-home"]
        if self._persistent and self._overlay_dir:
            command.extend(["--overlay", str(self._overlay_dir)])
        else:
            command.append("--writable-tmpfs")
        try:
            from tools.credential_files import (
                get_credential_file_mounts,
                get_skills_directory_mount,
            )

            for entry in await get_credential_file_mounts():
                command.extend(
                    ["--bind", f"{entry['host_path']}:{entry['container_path']}:ro"]
                )
            for entry in await get_skills_directory_mount():
                command.extend(
                    ["--bind", f"{entry['host_path']}:{entry['container_path']}:ro"]
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(
                "Singularity: could not load credential/skills mounts: %s", exc
            )
        if self._memory > 0:
            command.extend(["--memory", f"{self._memory}M"])
        if self._cpu > 0:
            command.extend(["--cpus", str(self._cpu)])
        command.extend([str(self.image), self.instance_id])
        try:
            output, returncode = await _run_command(command, timeout=120)
        except TimeoutError:
            raise RuntimeError("Instance start timed out")
        if returncode != 0:
            raise RuntimeError(f"Failed to start instance: {output}")
        self._instance_started = True

    async def _run_bash(
        self,
        cmd_string: str,
        *,
        login: bool = False,
        timeout: int | float = 120,
        stdin_data: str | None = None,
        bounded_capture: bool = False,
    ) -> dict[str, Any]:
        if not self._instance_started or self.executable is None:
            raise RuntimeError("Singularity instance not started")
        collector = (
            await self._bounded_output_collector()
            if bounded_capture
            else _BoundedOutputCollector(_UNBOUNDED_CAPTURE_CHARS)
        )
        command = [
            self.executable,
            "exec",
            f"instance://{self.instance_id}",
            "bash",
        ]
        command.extend(["-l", "-c", cmd_string] if login else ["-c", cmd_string])
        command_task = asyncio.create_task(
            _run_command(
                command,
                timeout=float(timeout),
                input_data=stdin_data,
                _output_collector=collector,
            )
        )

        async def cancel_command() -> None:
            command_task.cancel()
            await asyncio.gather(command_task, return_exceptions=True)

        started = asyncio.get_running_loop().time()
        activity_state = {"last_touch": started, "start": started}
        try:
            await asyncio.sleep(0)
            while not command_task.done():
                from tools.interrupt import is_interrupted

                if is_interrupted():
                    from tools.environments.file_sync import _await_owned

                    cleanup = asyncio.create_task(cancel_command())
                    await _await_owned(cleanup)
                    suffix = "\n[Command interrupted]"
                    rendered = collector.render(suffix=suffix)
                    if bounded_capture:
                        return await self._finalize_wait_result(
                            collector,
                            rendered,
                            130,
                        )
                    return {"output": rendered, "returncode": 130}
                await asyncio.wait({command_task}, timeout=0.25)
                if command_task.done():
                    break
                touch_activity_if_due(
                    activity_state,
                    "terminal command running",
                )
            output, returncode = await command_task
        except TimeoutError:
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
            from tools.environments.file_sync import _await_owned

            cleanup = asyncio.create_task(cancel_command())
            await _await_owned(cleanup)  # noqa: ASYNC120
            raise
        if bounded_capture:
            return await self._finalize_wait_result(
                collector,
                collector.render(),
                returncode,
            )
        return {"output": output, "returncode": returncode}

    async def cleanup(self) -> None:
        async with self._lifecycle_lock:
            if self._instance_started and self.executable is not None:
                try:
                    await _run_command(
                        [
                            self.executable,
                            "instance",
                            "stop",
                            self.instance_id,
                        ],
                        timeout=30,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "Failed to stop Singularity instance %s: %s",
                        self.instance_id,
                        exc,
                    )
                self._instance_started = False
            if self._persistent and self._overlay_dir:
                await _store_snapshot(self._task_id, str(self._overlay_dir))
