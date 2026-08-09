"""Native-async parity tests for the command secret source."""

from __future__ import annotations

import asyncio
import os
import stat
import time
from pathlib import Path

import pytest

from agent.secret_sources import registry
from agent.secret_sources.base import FetchResult, SecretSource
from agent.secret_sources.command import (
    _run_helper,
    apply_command_secrets,
    get_command_secret,
    parse_secret_output,
)
from hermes_cli import env_loader


pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="the command secret provider is POSIX-only"
)


def _write_helper(tmp_path: Path, body: str, name: str = "helper.sh") -> Path:
    script = tmp_path / name
    script.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    registry._reset_registry_for_tests()
    env_loader.reset_secret_source_cache()
    for name in ("CMDTEST_API_KEY", "CMDTEST_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    yield
    registry._reset_registry_for_tests()
    env_loader.reset_secret_source_cache()


def test_parse_base64_padding_not_misclassified_as_dotenv():
    assert parse_secret_output("dGVzdA==\n", "CMDTEST_API_KEY") == "dGVzdA=="


@pytest.mark.asyncio
async def test_real_helper_resolves_without_blocking_event_loop(tmp_path):
    helper = _write_helper(
        tmp_path,
        "sleep 0.1\nprintf 'sk-test-bare-12345'",
    )
    heartbeats = 0

    async def heartbeat() -> None:
        nonlocal heartbeats
        while True:
            heartbeats += 1
            await asyncio.sleep(0.01)

    pulse = asyncio.create_task(heartbeat())
    try:
        value = await get_command_secret(command=str(helper), key="CMDTEST_API_KEY")
    finally:
        pulse.cancel()
        await asyncio.gather(pulse, return_exceptions=True)

    assert value == "sk-test-bare-12345"
    assert heartbeats >= 5


@pytest.mark.asyncio
async def test_timeout_kills_helper_and_returns_none(tmp_path):
    helper = _write_helper(tmp_path, "sleep 30")
    started = time.monotonic()
    value = await get_command_secret(
        command=str(helper),
        key="CMDTEST_API_KEY",
        timeout_seconds=0.1,
    )
    assert value is None
    assert time.monotonic() - started < 2


@pytest.mark.asyncio
async def test_cancellation_stops_helper_and_propagates(tmp_path):
    helper = _write_helper(tmp_path, "sleep 30")
    task = asyncio.create_task(_run_helper(str(helper), "", 60, 1024))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)


@pytest.mark.asyncio
async def test_repeated_cancellation_finishes_helper_pipe_cleanup(monkeypatch):
    from agent.secret_sources import command

    communicate_started = asyncio.Event()
    release_communicate = asyncio.Event()
    communicate_completed = asyncio.Event()

    class BlockingProcess:
        pid = 12345
        returncode = None
        killed = False

        async def communicate(self):
            communicate_started.set()
            await release_communicate.wait()
            communicate_completed.set()
            return b"", b""

        def kill(self):
            self.killed = True
            self.returncode = -9

    process = BlockingProcess()

    async def create_process(*_args, **_kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(command.os, "getpgid", lambda _pid: process.pid)

    def kill_process_group(_pid, _signal):
        process.killed = True
        process.returncode = -9

    monkeypatch.setattr(command.os, "killpg", kill_process_group)

    task = asyncio.create_task(_run_helper("unused", "", 60, 1024))
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


@pytest.mark.asyncio
async def test_failure_logging_never_leaks_command_or_secret(tmp_path, capfd):
    secret = "sk-super-secret-value-do-not-log"
    helper = _write_helper(
        tmp_path,
        f"echo '{secret}' >&2\nexit 7",
        name="my-distinctive-helper-name.sh",
    )
    assert await get_command_secret(command=str(helper), key="CMDTEST_API_KEY") is None
    captured = capfd.readouterr()
    combined = captured.out + captured.err
    assert "my-distinctive-helper-name" not in combined
    assert secret not in combined
    assert "code=7" in combined


@pytest.mark.asyncio
async def test_apply_dotenv_blob_preserves_public_result_shape(tmp_path):
    helper = _write_helper(
        tmp_path,
        "printf 'CMDTEST_API_KEY=sk-applied\\nCMDTEST_TOKEN=tok-applied\\n'",
    )
    result = await apply_command_secrets(command=str(helper))
    assert isinstance(result, FetchResult)
    assert sorted(result.applied) == ["CMDTEST_API_KEY", "CMDTEST_TOKEN"]
    assert os.environ["CMDTEST_API_KEY"] == "sk-applied"
    assert os.environ["CMDTEST_TOKEN"] == "tok-applied"


@pytest.mark.asyncio
async def test_env_loader_applies_once_and_records_provenance(
    tmp_path, monkeypatch, capsys
):
    helper = _write_helper(
        tmp_path,
        "printf 'CMDTEST_API_KEY=sk-dispatch\\nCMDTEST_TOKEN=tok-dispatch\\n'",
    )
    (tmp_path / "config.yaml").write_text(
        f"secrets:\n  command:\n    enabled: true\n    command: {helper}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    await env_loader.load_hermes_dotenv(hermes_home=tmp_path)
    await env_loader.load_hermes_dotenv(hermes_home=tmp_path)

    assert os.environ["CMDTEST_API_KEY"] == "sk-dispatch"
    assert env_loader.get_secret_source("CMDTEST_API_KEY") == "command"
    assert (
        env_loader.format_secret_source_suffix("CMDTEST_API_KEY")
        == " (from Command helper)"
    )
    assert capsys.readouterr().err.count("Command helper: applied 2 secrets") == 1


@pytest.mark.asyncio
async def test_registry_rejects_sync_fetch(monkeypatch):
    monkeypatch.setattr(registry, "_ensure_builtin_sources", lambda: None)

    class SyncSource(SecretSource):
        name = "sync_source"
        label = "Sync source"

        def fetch(self, cfg, home_path):
            return FetchResult()

    assert registry.register_source(SyncSource()) is False
