"""Native-async behavior for the retained Singularity backend."""

from __future__ import annotations

import asyncio
import gc
import os
import sys
import weakref

import aiofiles
import pytest

from tools.environments import singularity
from tools.environments.base import BaseEnvironment, _BoundedOutputCollector


def test_sif_build_lock_does_not_cross_or_retain_event_loops(tmp_path):
    loop_refs = []
    lock_refs = []

    async def use_lock():
        loop_refs.append(weakref.ref(asyncio.get_running_loop()))
        lock = await singularity._sif_build_lock(tmp_path / "image.sif")
        lock_refs.append(weakref.ref(lock))
        async with lock:
            await asyncio.sleep(0)

    asyncio.run(use_lock())
    asyncio.run(use_lock())
    gc.collect()

    assert loop_refs[0]() is None
    assert lock_refs[0]() is None


@pytest.mark.asyncio
async def test_sif_build_lock_canonicalizes_aliases(tmp_path):
    cache = tmp_path / "cache"
    alias = tmp_path / "alias"
    cache.mkdir()
    alias.symlink_to(cache, target_is_directory=True)

    first = await singularity._sif_build_lock(cache / "image.sif")
    second = await singularity._sif_build_lock(alias / "image.sif")

    assert first is second


@pytest.mark.asyncio
async def test_constructor_is_state_only_and_init_preserves_start_argv(monkeypatch):
    calls: list[list[str]] = []

    async def executable():
        return "/usr/bin/apptainer"

    async def image(value, executable):
        assert executable == "/usr/bin/apptainer"
        return value

    async def run(command, **_kwargs):
        calls.append(command)
        return "", 0

    async def no_mounts(*_args, **_kwargs):
        return []

    monkeypatch.setattr(singularity, "_ensure_singularity_available", executable)
    monkeypatch.setattr(singularity, "_get_or_build_sif", image)
    monkeypatch.setattr(singularity, "_run_command", run)
    monkeypatch.setattr(
        "tools.credential_files.get_credential_file_mounts", no_mounts
    )
    monkeypatch.setattr(
        "tools.credential_files.get_skills_directory_mount", no_mounts
    )
    env = singularity.SingularityEnvironment(
        "docker://python:3.11",
        cpu=2,
        memory=1024,
    )
    assert calls == []

    await env.init_session()

    start = calls[0]
    assert start[:5] == [
        "/usr/bin/apptainer",
        "instance",
        "start",
        "--containall",
        "--no-home",
    ]
    assert ["--memory", "1024M"] == start[start.index("--memory") :][:2]
    assert ["--cpus", "2"] == start[start.index("--cpus") :][:2]
    assert start[-2:] == ["docker://python:3.11", env.instance_id]


@pytest.mark.asyncio
async def test_execute_and_cleanup_use_native_cli_transport(monkeypatch):
    commands: list[list[str]] = []

    async def run(command, **_kwargs):
        commands.append(command)
        if command[1:3] == ["instance", "stop"]:
            return "", 0
        return "hello", 0

    env = singularity.SingularityEnvironment("image.sif")
    env.executable = "/usr/bin/singularity"
    env._instance_started = True
    env._snapshot_ready = True
    monkeypatch.setattr(singularity, "_run_command", run)

    result = await env._run_bash("printf hello", timeout=3)
    await env.cleanup()

    assert result == {"output": "hello", "returncode": 0}
    assert commands[0] == [
        "/usr/bin/singularity",
        "exec",
        f"instance://{env.instance_id}",
        "bash",
        "-c",
        "printf hello",
    ]
    assert commands[-1] == [
        "/usr/bin/singularity",
        "instance",
        "stop",
        env.instance_id,
    ]


class _NeverFinishingProcess:
    def __init__(self):
        self.returncode = None
        self.started = asyncio.Event()
        self.terminated = False
        self.killed = False

    async def communicate(self, _input):
        self.started.set()
        await asyncio.Event().wait()

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        return self.returncode


@pytest.mark.asyncio
async def test_cancellation_terminates_owned_subprocess(monkeypatch):
    process = _NeverFinishingProcess()

    async def terminate(fake_process):
        fake_process.terminate()
        await fake_process.wait()

    monkeypatch.setattr(singularity, "_terminate_process", terminate)
    task = asyncio.create_task(singularity._finish_process(process, timeout=60))
    await asyncio.wait_for(process.started.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.terminated is True
    assert process.killed is False


@pytest.mark.asyncio
async def test_native_cli_bounded_capture_spills_full_output(tmp_path):
    full_output = "head-" + ("middle" * 1_000) + "-tail"
    spill = tmp_path / "singularity-output.log"
    collector = _BoundedOutputCollector(100, spill_path=spill)

    output, returncode = await singularity._run_command(
        [sys.executable, "-c", f"print({full_output!r}, end='')"],
        timeout=30,
        _output_collector=collector,
    )
    result = await BaseEnvironment._finalize_wait_result(
        collector,
        output,
        returncode,
    )

    assert len(result["output"]) <= 100
    assert result["output_total_chars"] == len(full_output)
    async with aiofiles.open(spill, encoding="utf-8") as handle:
        assert await handle.read() == full_output
    assert collector.buffered_chars <= 100


async def _wait_for_pid(path) -> int:
    for _ in range(200):
        try:
            async with aiofiles.open(path, encoding="utf-8") as handle:
                return int((await handle.read()).strip())
        except (FileNotFoundError, ValueError):
            await asyncio.sleep(0.005)
    raise AssertionError("subprocess did not publish its pid")


@pytest.mark.asyncio
async def test_bounded_capture_cancellation_reaps_native_cli(tmp_path):
    pid_path = tmp_path / "singularity-cancel.pid"
    script = (
        "import os,sys,time;"
        f"open({str(pid_path)!r},'w').write(str(os.getpid()));"
        "sys.stdout.write('x'*10000);sys.stdout.flush();time.sleep(30)"
    )
    collector = _BoundedOutputCollector(
        100,
        spill_path=tmp_path / "cancelled-output.log",
    )
    task = asyncio.create_task(
        singularity._run_command(
            [sys.executable, "-c", script],
            timeout=30,
            _output_collector=collector,
        )
    )
    pid = await _wait_for_pid(pid_path)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert collector.buffered_chars <= 100


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group semantics")
async def test_bounded_cancellation_kills_term_ignoring_grandchild(tmp_path):
    child_pid_path = tmp_path / "singularity-grandchild.pid"
    child = (
        "import os,signal,time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        f"open({str(child_pid_path)!r},'w').write(str(os.getpid()));"
        "time.sleep(30)"
    )
    parent = (
        "import subprocess,sys,time;"
        f"subprocess.Popen([sys.executable,'-c',{child!r}]);"
        "time.sleep(30)"
    )
    collector = _BoundedOutputCollector(
        100,
        spill_path=tmp_path / "grandchild-output.log",
    )
    task = asyncio.create_task(
        singularity._run_command(
            [sys.executable, "-c", parent],
            timeout=30,
            _output_collector=collector,
        )
    )
    child_pid = await _wait_for_pid(child_pid_path)

    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)

    for _ in range(100):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("TERM-ignoring grandchild survived Singularity cancellation")


@pytest.mark.asyncio
async def test_sif_build_awaits_native_subprocess_environment(
    monkeypatch,
    tmp_path,
):
    sif_path = tmp_path / "python-3.11.sif"
    captured_env = None

    async def build_env(**kwargs):
        assert kwargs == {
            "scrub_secrets": False,
            "inherit_profile_home": False,
        }
        return {"BASE": "value"}

    async def run(command, *, env=None, **_kwargs):
        nonlocal captured_env
        captured_env = env
        sif_path.write_bytes(b"sif")
        return "", 0

    monkeypatch.setattr(
        singularity,
        "_get_apptainer_cache_dir",
        lambda: asyncio.sleep(0, result=tmp_path),
    )
    monkeypatch.setattr(
        "tools.environments.local.build_subprocess_env",
        build_env,
    )
    monkeypatch.setattr(singularity, "_run_command", run)

    result = await singularity._get_or_build_sif(
        "docker://python:3.11",
        "/usr/bin/apptainer",
    )

    assert result == str(sif_path)
    assert captured_env == {
        "BASE": "value",
        "APPTAINER_TMPDIR": str(tmp_path / "tmp"),
        "APPTAINER_CACHEDIR": str(tmp_path),
    }


@pytest.mark.asyncio
async def test_bounded_run_bash_timeout_preserves_partial_spill(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr("tools.tool_output_limits.get_max_bytes", lambda: 100)
    environment = singularity.SingularityEnvironment("image.sif")
    environment.executable = "/usr/bin/singularity"
    environment._instance_started = True
    partial = "partial-" + ("x" * 500)

    async def timeout(*_args, _output_collector=None, **_kwargs):
        assert _output_collector is not None
        _output_collector.append(partial)
        raise TimeoutError

    monkeypatch.setattr(singularity, "_run_command", timeout)

    result = await environment._run_bash(
        "sleep 30",
        timeout=7,
        bounded_capture=True,
    )

    assert result["returncode"] == 124
    assert result["output"].endswith("[Command timed out after 7s]")
    assert len(result["output"]) <= 100
    assert result["output_total_chars"] == len(partial)
    async with aiofiles.open(
        result["full_output_path"],
        encoding="utf-8",
    ) as handle:
        assert await handle.read() == partial


@pytest.mark.asyncio
async def test_run_bash_interrupt_cleans_up_and_preserves_partial_output(
    monkeypatch,
):
    from tools.interrupt import _bind_interrupt_event, _reset_interrupt_event

    started = asyncio.Event()
    finalized = asyncio.Event()

    async def blocked(*_args, _output_collector=None, **_kwargs):
        assert _output_collector is not None
        _output_collector.append("partial")
        started.set()
        try:
            await asyncio.Future()
        finally:
            finalized.set()

    monkeypatch.setattr(singularity, "_run_command", blocked)
    environment = singularity.SingularityEnvironment("image.sif")
    environment.executable = "/usr/bin/singularity"
    environment._instance_started = True
    interrupt = asyncio.Event()
    token = _bind_interrupt_event(interrupt)
    try:
        command = asyncio.create_task(environment._run_bash("sleep 30"))
        await asyncio.wait_for(started.wait(), timeout=1)
        interrupt.set()
        result = await asyncio.wait_for(command, timeout=1)
    finally:
        _reset_interrupt_event(token)

    assert result == {
        "output": "partial\n[Command interrupted]",
        "returncode": 130,
    }
    assert finalized.is_set()
