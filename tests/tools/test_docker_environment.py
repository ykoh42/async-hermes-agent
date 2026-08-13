"""Focused upstream-parity tests for the native-async Docker backend."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import aiofiles
import aiofiles.os
import pytest


@pytest.fixture
def docker_runtime(monkeypatch, tmp_path):
    from tools import credential_files
    from tools.environments import base
    from tools.environments import docker as module

    calls: list[list[str]] = []
    responses: dict[str, tuple[int, str, str] | BaseException] = {}

    async def _fake_run(
        argv,
        *,
        timeout,
        check=False,
        stdin_data=None,
        merge_stderr=False,
        _output_collector=None,
    ):
        argv = list(argv)
        calls.append(argv)
        operation = argv[1] if len(argv) > 1 else ""
        configured = responses.get(operation)
        if isinstance(configured, BaseException):
            raise configured
        if configured is not None:
            returncode, stdout, stderr = configured
        elif operation == "run":
            returncode, stdout, stderr = 0, "fresh-container-id\n", ""
        else:
            returncode, stdout, stderr = 0, "", ""
        if _output_collector is not None:
            _output_collector.append(stdout)
        completed = subprocess.CompletedProcess(
            argv, returncode, stdout=stdout, stderr=stderr
        )
        if check and returncode:
            raise subprocess.CalledProcessError(
                returncode, argv, output=stdout, stderr=stderr
            )
        return completed

    monkeypatch.setattr(module, "_run_docker_command", _fake_run)
    monkeypatch.setattr(
        module, "_ensure_docker_available", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        module, "_cgroup_limits_available", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(
        module, "_image_uses_init_entrypoint", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(
        module, "_egress_proxy_args_for_docker", AsyncMock(return_value=([], {}, []))
    )
    monkeypatch.setattr(
        module, "_egress_enforce_on_docker", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        module, "_get_active_profile_name", AsyncMock(return_value="default")
    )
    monkeypatch.setattr(
        module, "find_docker", AsyncMock(return_value="/usr/bin/docker")
    )
    monkeypatch.setattr(
        credential_files,
        "get_credential_file_mounts",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        credential_files,
        "get_skills_directory_mount",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        credential_files,
        "get_cache_directory_mounts",
        AsyncMock(return_value=[]),
    )
    sandbox = tmp_path / "sandboxes"
    monkeypatch.setattr(base, "get_sandbox_dir", AsyncMock(return_value=sandbox))
    return module, calls, responses, sandbox


@pytest.mark.asyncio
async def test_constructor_is_state_only_and_run_argv_matches_upstream(docker_runtime):
    module, calls, _responses, _sandbox = docker_runtime

    environment = module.DockerEnvironment(
        image="hermes-agent:test",
        task_id="task-a",
        persist_across_processes=False,
    )
    assert calls == []

    await environment.init_session()

    run = next(argv for argv in calls if argv[1] == "run")
    assert run[:4] == ["/usr/bin/docker", "run", "-d", "--init"]
    assert ["--name", environment._container_name] == run[4:6]
    assert "hermes-agent=1" in run
    assert "hermes-task-id=task-a" in run
    assert "hermes-profile=default" in run
    assert ["-w", "/root"] == run[run.index("-w") : run.index("-w") + 2]
    assert ["--shm-size", "1g"] == run[
        run.index("--shm-size") : run.index("--shm-size") + 2
    ]
    assert run[-3:] == ["hermes-agent:test", "sleep", "infinity"]
    assert environment._container_id == "fresh-container-id"


@pytest.mark.asyncio
async def test_running_persistent_container_is_reused_without_docker_run(
    docker_runtime,
):
    module, calls, responses, _sandbox = docker_runtime
    responses["ps"] = (0, "reused-cid\trunning\t\n", "")
    environment = module.DockerEnvironment(
        image="hermes-agent:test",
        task_id="reuse-task",
        persist_across_processes=True,
    )

    await environment.init_session()

    assert environment._container_id == "reused-cid"
    assert any(argv[1] == "ps" for argv in calls)
    assert not any(argv[1] == "run" for argv in calls)
    assert not any(argv[1] == "start" for argv in calls)


@pytest.mark.asyncio
async def test_stopped_persistent_container_is_started_before_exec(docker_runtime):
    module, calls, responses, _sandbox = docker_runtime
    responses["ps"] = (0, "reused-cid\texited\t\n", "")
    environment = module.DockerEnvironment(
        image="hermes-agent:test",
        task_id="reuse-task",
        persist_across_processes=True,
    )

    await environment.init_session()

    start_index = next(i for i, argv in enumerate(calls) if argv[1] == "start")
    exec_index = next(i for i, argv in enumerate(calls) if argv[1] == "exec")
    assert calls[start_index] == ["/usr/bin/docker", "start", "reused-cid"]
    assert start_index < exec_index


@pytest.mark.asyncio
async def test_execute_preserves_docker_exec_argv_and_stdin(docker_runtime):
    module, calls, _responses, _sandbox = docker_runtime
    environment = module.DockerEnvironment(
        image="hermes-agent:test",
        persist_across_processes=False,
    )
    await environment.init_session()
    calls.clear()

    result = await environment.execute(
        "printf hello",
        cwd="/workspace",
        stdin_data="input-data",
    )

    assert result["returncode"] == 0
    argv = calls[0]
    assert argv[:3] == ["/usr/bin/docker", "exec", "-i"]
    assert argv[-3:-1] == ["bash", "-c"]
    assert "builtin cd -- /workspace" in argv[-1]
    assert "printf hello" in argv[-1]


@pytest.mark.asyncio
async def test_persistent_filesystem_mounts_task_directories(docker_runtime):
    module, calls, _responses, sandbox = docker_runtime
    environment = module.DockerEnvironment(
        image="hermes-agent:test",
        persistent_filesystem=True,
        task_id="persist-task",
        persist_across_processes=False,
    )

    await environment.init_session()

    home = sandbox / "docker" / "persist-task" / "home"
    workspace = sandbox / "docker" / "persist-task" / "workspace"
    assert await aiofiles.os.path.isdir(home)
    assert await aiofiles.os.path.isdir(workspace)
    run = next(argv for argv in calls if argv[1] == "run")
    assert f"{home}:/root" in run
    assert f"{workspace}:/workspace" in run


@pytest.mark.asyncio
async def test_auto_mount_cwd_preserves_upstream_volume_argv(
    docker_runtime, tmp_path
):
    module, calls, _responses, _sandbox = docker_runtime
    host_cwd = tmp_path / "project"
    await aiofiles.os.makedirs(host_cwd)
    environment = module.DockerEnvironment(
        image="hermes-agent:test",
        host_cwd=str(host_cwd),
        auto_mount_cwd=True,
        persist_across_processes=False,
    )

    await environment.init_session()

    run = next(argv for argv in calls if argv[1] == "run")
    mount_index = run.index(f"{host_cwd}:/workspace")
    assert run[mount_index - 1] == "-v"
    assert "/workspace:rw,exec,size=10g" not in run


@pytest.mark.asyncio
async def test_async_credential_file_discovery_preserves_read_only_mount(
    docker_runtime, monkeypatch, tmp_path
):
    from tools import credential_files

    module, calls, _responses, _sandbox = docker_runtime
    credential = tmp_path / "oauth.json"
    async with aiofiles.open(credential, "w", encoding="utf-8") as handle:
        await handle.write("{}")
    monkeypatch.setattr(
        credential_files,
        "get_credential_file_mounts",
        AsyncMock(
            return_value=[
                {
                    "host_path": str(credential),
                    "container_path": "/root/.hermes/oauth.json",
                }
            ]
        ),
    )
    environment = module.DockerEnvironment(
        image="hermes-agent:test", persist_across_processes=False
    )

    await environment.init_session()

    run = next(argv for argv in calls if argv[1] == "run")
    mount = f"{credential}:/root/.hermes/oauth.json:ro"
    assert run[run.index(mount) - 1 : run.index(mount) + 1] == ["-v", mount]


@pytest.mark.asyncio
async def test_forwarded_env_is_refreshed_for_each_docker_exec(
    docker_runtime, monkeypatch
):
    module, calls, _responses, _sandbox = docker_runtime
    monkeypatch.setenv("SERVICE_TOKEN", "first-token")
    monkeypatch.setattr(module, "_load_hermes_env_vars", AsyncMock(return_value={}))
    environment = module.DockerEnvironment(
        image="hermes-agent:test",
        forward_env=["SERVICE_TOKEN"],
        persist_across_processes=False,
    )

    await environment.init_session()
    initial_exec = next(argv for argv in calls if argv[1] == "exec")
    assert "SERVICE_TOKEN=first-token" in initial_exec
    calls.clear()
    monkeypatch.setenv("SERVICE_TOKEN", "second-token")

    await environment.execute("true")

    runtime_exec = calls[0]
    assert "SERVICE_TOKEN=second-token" in runtime_exec
    assert "SERVICE_TOKEN=first-token" not in runtime_exec


@pytest.mark.asyncio
async def test_concurrent_first_use_creates_only_one_container(docker_runtime):
    module, calls, _responses, _sandbox = docker_runtime
    environment = module.DockerEnvironment(
        image="hermes-agent:test", persist_across_processes=False
    )

    await asyncio.gather(environment.init_session(), environment.init_session())

    assert sum(argv[1] == "run" for argv in calls) == 1


@pytest.mark.asyncio
async def test_cancelled_snapshot_removes_newly_owned_container(
    docker_runtime, monkeypatch
):
    from tools.environments import base

    module, calls, _responses, _sandbox = docker_runtime
    snapshot_started = asyncio.Event()

    async def _cancel_snapshot(self):
        snapshot_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(base.BaseEnvironment, "init_session", _cancel_snapshot)
    environment = module.DockerEnvironment(
        image="hermes-agent:test", persist_across_processes=True
    )
    initialization = asyncio.create_task(environment.init_session())
    await asyncio.wait_for(snapshot_started.wait(), timeout=1)
    container_id = environment._container_id

    initialization.cancel()
    with pytest.raises(asyncio.CancelledError):
        await initialization

    assert ["/usr/bin/docker", "stop", "-t", "10", container_id] in calls
    assert ["/usr/bin/docker", "rm", "-f", container_id] in calls
    assert environment._container_id is None


@pytest.mark.asyncio
async def test_cancelled_snapshot_does_not_remove_reused_container(
    docker_runtime, monkeypatch
):
    from tools.environments import base

    module, calls, responses, _sandbox = docker_runtime
    responses["ps"] = (0, "reused-cid\trunning\t\n", "")
    snapshot_started = asyncio.Event()

    async def _cancel_snapshot(self):
        snapshot_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(base.BaseEnvironment, "init_session", _cancel_snapshot)
    environment = module.DockerEnvironment(
        image="hermes-agent:test", persist_across_processes=True
    )
    initialization = asyncio.create_task(environment.init_session())
    await asyncio.wait_for(snapshot_started.wait(), timeout=1)

    initialization.cancel()
    with pytest.raises(asyncio.CancelledError):
        await initialization

    assert not any(argv[1] in {"stop", "rm"} for argv in calls)
    assert environment._container_id is None


@pytest.mark.asyncio
async def test_cleanup_preserves_or_removes_container_per_upstream_mode(
    docker_runtime,
):
    module, calls, _responses, _sandbox = docker_runtime
    persistent = module.DockerEnvironment(
        image="hermes-agent:test", persist_across_processes=True
    )
    await persistent.init_session()
    calls.clear()

    await persistent.cleanup()

    assert calls == []
    assert persistent._container_id is None

    disposable = module.DockerEnvironment(
        image="hermes-agent:test", persist_across_processes=False
    )
    await disposable.init_session()
    container_id = disposable._container_id
    calls.clear()

    await disposable.cleanup()

    assert calls == [
        ["/usr/bin/docker", "stop", "-t", "10", container_id],
        ["/usr/bin/docker", "rm", "-f", container_id],
    ]
    assert disposable._container_id is None


@pytest.mark.asyncio
async def test_failed_docker_run_removes_named_orphan_and_preserves_error(
    docker_runtime,
):
    module, calls, responses, _sandbox = docker_runtime
    responses["run"] = subprocess.CalledProcessError(
        125, ["docker", "run"], output="", stderr="daemon unavailable"
    )
    environment = module.DockerEnvironment(
        image="hermes-agent:test", persist_across_processes=False
    )

    with pytest.raises(subprocess.CalledProcessError) as error:
        await environment.init_session()

    assert error.value.returncode == 125
    assert ["/usr/bin/docker", "rm", "-f", environment._container_name] in calls


@pytest.mark.asyncio
async def test_cancelled_docker_run_removes_named_orphan_then_reraises(
    docker_runtime,
):
    module, calls, responses, _sandbox = docker_runtime
    responses["run"] = asyncio.CancelledError()
    environment = module.DockerEnvironment(
        image="hermes-agent:test", persist_across_processes=False
    )

    with pytest.raises(asyncio.CancelledError):
        await environment.init_session()

    assert ["/usr/bin/docker", "rm", "-f", environment._container_name] in calls


@pytest.mark.asyncio
async def test_cancelled_cleanup_finishes_stop_and_remove_before_reraising(
    docker_runtime, monkeypatch
):
    module, calls, _responses, _sandbox = docker_runtime
    environment = module.DockerEnvironment(
        image="hermes-agent:test", persist_across_processes=False
    )
    await environment.init_session()
    container_id = environment._container_id
    calls.clear()
    stop_started = asyncio.Event()
    release_stop = asyncio.Event()

    async def _controlled_run(argv, **_kwargs):
        argv = list(argv)
        calls.append(argv)
        if argv[1] == "stop":
            stop_started.set()
            await release_stop.wait()
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(module, "_run_docker_command", _controlled_run)
    cleanup = asyncio.create_task(environment.cleanup())
    await asyncio.wait_for(stop_started.wait(), timeout=1)

    cleanup.cancel()
    release_stop.set()
    with pytest.raises(asyncio.CancelledError):
        await cleanup

    assert calls == [
        ["/usr/bin/docker", "stop", "-t", "10", container_id],
        ["/usr/bin/docker", "rm", "-f", container_id],
    ]
    assert environment._container_id is None


@pytest.mark.asyncio
async def test_run_bash_timeout_preserves_partial_output_and_upstream_message(
    docker_runtime, monkeypatch
):
    module, _calls, _responses, _sandbox = docker_runtime
    environment = module.DockerEnvironment(
        image="hermes-agent:test", persist_across_processes=False
    )
    environment._container_id = "container-id"
    environment._docker_exe = "/usr/bin/docker"
    environment._profile_scoped_passthrough = False

    async def _timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(
            ["docker", "exec"], 7, output="partial output\n"
        )

    monkeypatch.setattr(module, "_run_docker_command", _timeout)

    result = await environment._run_bash("sleep 30", timeout=7)

    assert result == {
        "output": "partial output\n[Command timed out after 7s]",
        "returncode": 124,
    }


@pytest.mark.asyncio
async def test_bounded_run_bash_timeout_preserves_partial_spill(
    docker_runtime,
    monkeypatch,
    tmp_path,
):
    module, _calls, _responses, _sandbox = docker_runtime
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr("tools.tool_output_limits.get_max_bytes", lambda: 100)
    environment = module.DockerEnvironment(
        image="hermes-agent:test",
        persist_across_processes=False,
    )
    environment._container_id = "container-id"
    environment._docker_exe = "/usr/bin/docker"
    environment._profile_scoped_passthrough = False
    partial = "partial-" + ("x" * 500)

    async def _timeout(*_args, _output_collector=None, **_kwargs):
        assert _output_collector is not None
        _output_collector.append(partial)
        raise subprocess.TimeoutExpired(
            ["docker", "exec"],
            7,
            output=partial,
        )

    monkeypatch.setattr(module, "_run_docker_command", _timeout)

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


async def _wait_for_pid(path: Path) -> int:
    for _ in range(200):
        try:
            async with aiofiles.open(path, "r", encoding="utf-8") as handle:
                return int((await handle.read()).strip())
        except (FileNotFoundError, ValueError):
            await asyncio.sleep(0.005)
    raise AssertionError("subprocess did not publish its pid")


def _assert_process_reaped(pid: int) -> None:
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.asyncio
async def test_native_subprocess_timeout_terminates_and_reaps_child(tmp_path):
    from tools.environments.docker import _run_docker_command

    pid_path = tmp_path / "timeout.pid"
    script = (
        "import os,time;"
        f"open({str(pid_path)!r},'w').write(str(os.getpid()));"
        "time.sleep(30)"
    )

    with pytest.raises(subprocess.TimeoutExpired):
        await _run_docker_command(
            [sys.executable, "-c", script],
            timeout=0.1,
        )

    pid = await _wait_for_pid(pid_path)
    _assert_process_reaped(pid)


@pytest.mark.asyncio
async def test_native_subprocess_cancellation_terminates_and_reaps_child(tmp_path):
    from tools.environments.docker import _run_docker_command

    pid_path = tmp_path / "cancel.pid"
    script = (
        "import os,time;"
        f"open({str(pid_path)!r},'w').write(str(os.getpid()));"
        "time.sleep(30)"
    )
    task = asyncio.create_task(
        _run_docker_command([sys.executable, "-c", script], timeout=30)
    )
    pid = await _wait_for_pid(pid_path)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    _assert_process_reaped(pid)


@pytest.mark.asyncio
async def test_native_subprocess_bounded_cancellation_reaps_child(tmp_path):
    from tools.environments.base import _BoundedOutputCollector
    from tools.environments.docker import _run_docker_command

    pid_path = tmp_path / "bounded-cancel.pid"
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
        _run_docker_command(
            [sys.executable, "-c", script],
            timeout=30,
            merge_stderr=True,
            _output_collector=collector,
        )
    )
    pid = await _wait_for_pid(pid_path)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    _assert_process_reaped(pid)
    assert collector.buffered_chars <= 100


@pytest.mark.asyncio
async def test_native_subprocess_bounded_capture_spills_full_output(tmp_path):
    from tools.environments.base import BaseEnvironment, _BoundedOutputCollector
    from tools.environments.docker import _run_docker_command

    full_output = "head-" + ("middle" * 1_000) + "-tail"
    spill = tmp_path / "docker-output.log"
    collector = _BoundedOutputCollector(100, spill_path=spill)
    completed = await _run_docker_command(
        [sys.executable, "-c", f"print({full_output!r}, end='')"],
        timeout=30,
        merge_stderr=True,
        _output_collector=collector,
    )
    result = await BaseEnvironment._finalize_wait_result(
        collector,
        completed.stdout,
        completed.returncode,
    )

    assert len(result["output"]) <= 100
    assert result["output_total_chars"] == len(full_output)
    assert result["full_output_path"] == str(spill)
    async with aiofiles.open(spill, encoding="utf-8") as handle:
        assert await handle.read() == full_output
    assert collector.buffered_chars <= 100


@pytest.mark.asyncio
async def test_docker_bounded_run_bash_returns_upstream_spill_metadata(
    docker_runtime,
    monkeypatch,
    tmp_path,
):
    module, _calls, responses, _sandbox = docker_runtime
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr("tools.tool_output_limits.get_max_bytes", lambda: 100)
    environment = module.DockerEnvironment(
        image="hermes-agent:test",
        persist_across_processes=False,
    )
    await environment.init_session()
    full_output = "head-" + ("x" * 500) + "-tail"
    responses["exec"] = (7, full_output, "")

    result = await environment._run_bash("printf output", bounded_capture=True)

    assert result["returncode"] == 7
    assert len(result["output"]) <= 100
    assert result["output_total_chars"] == len(full_output)
    async with aiofiles.open(
        result["full_output_path"],
        encoding="utf-8",
    ) as handle:
        assert await handle.read() == full_output


@pytest.mark.asyncio
async def test_run_bash_interrupt_cleans_up_and_preserves_partial_output(
    docker_runtime,
    monkeypatch,
):
    from tools.interrupt import _bind_interrupt_event, _reset_interrupt_event

    module, _calls, _responses, _sandbox = docker_runtime
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

    monkeypatch.setattr(module, "_run_docker_command", blocked)
    environment = module.DockerEnvironment(
        image="hermes-agent:test",
        persist_across_processes=False,
    )
    environment._container_id = "container-id"
    environment._docker_exe = "/usr/bin/docker"
    environment._profile_scoped_passthrough = False
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


@pytest.mark.asyncio
async def test_natural_docker_rc130_has_no_interrupt_marker(
    docker_runtime,
):
    module, _calls, responses, _sandbox = docker_runtime
    environment = module.DockerEnvironment(
        image="hermes-agent:test",
        persist_across_processes=False,
    )
    environment._container_id = "container-id"
    environment._docker_exe = "/usr/bin/docker"
    environment._profile_scoped_passthrough = False
    responses["exec"] = (130, "natural", "")

    result = await environment._run_bash("exit 130")

    assert result == {"output": "natural", "returncode": 130}


@pytest.mark.asyncio
async def test_spill_finalize_discards_raw_file_after_repeated_cancellation(tmp_path):
    from tools.environments.base import BaseEnvironment, _BoundedOutputCollector

    spill = tmp_path / "cancelled-finalize.log"
    collector = _BoundedOutputCollector(10, spill_path=spill)
    collector.append("output-" * 100)
    original_close = collector._close_spill
    close_started = asyncio.Event()
    release_close = asyncio.Event()

    async def controlled_close():
        close_started.set()
        await release_close.wait()
        return await original_close()

    collector._close_spill = controlled_close
    finalize = asyncio.create_task(
        BaseEnvironment._finalize_wait_result(
            collector,
            collector.render(),
            0,
        )
    )
    await asyncio.wait_for(close_started.wait(), timeout=1)
    finalize.cancel()
    finalize.cancel()
    release_close.set()

    with pytest.raises(asyncio.CancelledError):
        await finalize
    assert not spill.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_upstream_pure_argument_helpers_remain_unchanged():
    from tools.environments.docker import (
        _build_security_args,
        _extra_args_set_shm_size,
        _normalize_env_dict,
        _normalize_forward_env_names,
        _sanitize_label_value,
    )

    assert _normalize_forward_env_names([" A ", "A", "bad-name", 1]) == ["A"]
    assert _normalize_env_dict({"A": 1, "B": True, "bad-name": "x"}) == {
        "A": "1",
        "B": "True",
    }
    assert _sanitize_label_value("unsafe/profile value") == "unsafe_profile_value"
    assert _extra_args_set_shm_size(["--shm-size=2g"])
    security = _build_security_args(run_as_host_user=False)
    assert security[-4:] == ["--cap-add", "SETUID", "--cap-add", "SETGID"]
