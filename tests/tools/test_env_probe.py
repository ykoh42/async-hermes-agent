"""Native-async local Python toolchain probe behavior."""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

from tools import env_probe

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture(autouse=True)
async def reset_probe_cache():
    await env_probe._reset_cache_for_tests()
    yield
    await env_probe._reset_cache_for_tests()


def _async_value(value):
    async def result(*_args, **_kwargs):
        return value

    return result


async def test_clean_environment_is_silent(monkeypatch):
    monkeypatch.setattr(
        env_probe,
        "_python_version_of",
        _async_value("3.13.3"),
    )
    monkeypatch.setattr(env_probe, "_has_pip_module", _async_value(True))
    monkeypatch.setattr(env_probe, "_detect_pep668", _async_value(False))
    monkeypatch.setattr(env_probe, "_pip_python_version", _async_value("3.13"))
    monkeypatch.setattr(env_probe, "_which", _async_value(None))

    assert await env_probe.get_environment_probe_line() == ""


async def test_problematic_environment_emits_one_line(monkeypatch):
    async def python_version(binary):
        return "3.11.15" if binary == "python3" else None

    monkeypatch.setattr(env_probe, "_python_version_of", python_version)
    monkeypatch.setattr(env_probe, "_has_pip_module", _async_value(False))
    monkeypatch.setattr(env_probe, "_detect_pep668", _async_value(True))
    monkeypatch.setattr(env_probe, "_pip_python_version", _async_value("3.12"))
    async def which(name):
        return None if name == "uv" else f"/usr/bin/{name}"

    monkeypatch.setattr(env_probe, "_which", which)

    line = await env_probe.get_environment_probe_line()
    assert "\n" not in line
    assert "3.11.15" in line
    assert "no pip module" in line
    assert "mismatch" in line
    assert "PEP 668" in line


async def test_remote_backend_is_silent(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")

    async def should_not_run(*_args, **_kwargs):
        raise AssertionError("remote backends must skip the local probe")

    monkeypatch.setattr(env_probe, "_python_version_of", should_not_run)
    assert await env_probe.get_environment_probe_line() == ""


async def test_result_is_cached(monkeypatch):
    calls = 0

    async def python_version(_binary):
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(env_probe, "_python_version_of", python_version)
    monkeypatch.setattr(env_probe, "_has_pip_module", _async_value(False))
    monkeypatch.setattr(env_probe, "_detect_pep668", _async_value(False))
    monkeypatch.setattr(env_probe, "_pip_python_version", _async_value(None))
    monkeypatch.setattr(env_probe, "_which", _async_value(None))

    first = await env_probe.get_environment_probe_line()
    second = await env_probe.get_environment_probe_line()
    assert first == second
    assert calls == 2


async def test_stuck_probe_fails_open_without_cancelling_worker(monkeypatch):
    release = asyncio.Event()

    async def stuck_probe():
        await release.wait()
        return "Python toolchain: recovered."

    monkeypatch.setattr(env_probe, "_build_probe_line", stuck_probe)
    monkeypatch.setattr(env_probe, "_PROBE_WAIT_TIMEOUT", 0.01)

    assert await env_probe.get_environment_probe_line() == ""
    task = env_probe._PROBE_TASKS[asyncio.get_running_loop()]
    assert not task.cancelled()

    release.set()
    await task
    assert (
        await env_probe.get_environment_probe_line()
        == "Python toolchain: recovered."
    )


async def test_run_times_out_without_blocking_event_loop():
    heartbeats = 0

    async def heartbeat():
        nonlocal heartbeats
        while True:
            heartbeats += 1
            await asyncio.sleep(0)

    pulse = asyncio.create_task(heartbeat())
    try:
        returncode, _stdout, stderr = await env_probe._run(
            ["python", "-c", "import time; time.sleep(30)"],
            timeout=0.02,
        )
    finally:
        pulse.cancel()
        await asyncio.gather(pulse, return_exceptions=True)

    assert returncode == -1
    assert stderr == "timeout"
    assert heartbeats > 1


async def test_run_repeated_cancellation_drains_process(monkeypatch):
    communicate_started = asyncio.Event()
    release_communicate = asyncio.Event()
    communicate_completed = asyncio.Event()

    class BlockingProcess:
        returncode = None
        killed = False

        async def communicate(self):
            communicate_started.set()
            await release_communicate.wait()
            communicate_completed.set()
            self.returncode = -9
            return b"", b""

        async def wait(self):
            return self.returncode

        def kill(self):
            self.killed = True

    process = BlockingProcess()

    async def create_process(*_args, **_kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

    task = asyncio.create_task(env_probe._run(["python", "--version"]))
    await communicate_started.wait()
    task.cancel()
    while not process.killed:
        await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)

    assert task.done() is False
    release_communicate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert communicate_completed.is_set()
