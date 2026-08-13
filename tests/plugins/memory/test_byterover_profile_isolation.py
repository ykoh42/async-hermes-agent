from __future__ import annotations

import asyncio
import gc
import os
import queue
import sys
import threading
import weakref
from pathlib import Path

import pytest
from blockbuster import BlockBuster

from agent import secret_scope
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from plugins.memory import byterover


@pytest.fixture(autouse=True)
def _isolate_byterover_state(monkeypatch: pytest.MonkeyPatch):
    token = byterover._brv_scope_context.set(None)
    byterover._brv_path_states.clear()
    byterover._brv_scope_aliases.clear()
    monkeypatch.setattr(byterover, "_cached_brv_path", None)
    secret_scope.set_multiplex_active(False)
    yield
    secret_scope.set_multiplex_active(False)
    byterover._brv_path_states.clear()
    byterover._brv_scope_aliases.clear()
    byterover._brv_scope_context.reset(token)


def _make_brv_probe(path: Path) -> Path:
    executable = path / "brv"
    executable.write_text(
        "#!/bin/sh\n"
        "printf '%s|%s|%s\\n' \"${BRV_API_KEY-<missing>}\" "
        '"${HERMES_HOME-<missing>}" "$PWD"\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


async def _run_in_profile(home: Path, secrets: dict[str, str], *args: str) -> dict:
    home_token = set_hermes_home_override(home)
    secret_token = secret_scope.set_secret_scope(secrets)
    try:
        await asyncio.sleep(0)
        return await byterover._run_brv(list(args))
    finally:
        secret_scope.reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


@pytest.mark.asyncio
async def test_concurrent_profiles_isolate_key_home_and_working_tree(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    _make_brv_probe(executable_dir)
    monkeypatch.setenv("PATH", f"{executable_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("BRV_API_KEY", "process-leak")
    secret_scope.set_multiplex_active(True)
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"

    result_a, result_b = await asyncio.gather(
        _run_in_profile(home_a, {"BRV_API_KEY": "key-a"}, "status"),
        _run_in_profile(home_b, {"BRV_API_KEY": "key-b"}, "status"),
    )

    assert result_a == {
        "success": True,
        "output": f"key-a|{home_a}|{home_a / 'byterover'}",
    }
    assert result_b == {
        "success": True,
        "output": f"key-b|{home_b}|{home_b / 'byterover'}",
    }


@pytest.mark.asyncio
async def test_sequential_profiles_do_not_reuse_child_credentials_or_home(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    _make_brv_probe(executable_dir)
    monkeypatch.setenv("PATH", f"{executable_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("BRV_API_KEY", "process-leak")
    secret_scope.set_multiplex_active(True)
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"

    result_a = await _run_in_profile(home_a, {"BRV_API_KEY": "key-a"}, "status")
    result_b = await _run_in_profile(home_b, {"BRV_API_KEY": "key-b"}, "status")

    assert result_a == {
        "success": True,
        "output": f"key-a|{home_a}|{home_a / 'byterover'}",
    }
    assert result_b == {
        "success": True,
        "output": f"key-b|{home_b}|{home_b / 'byterover'}",
    }


@pytest.mark.asyncio
async def test_missing_scoped_key_never_falls_through_to_process_secret(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    _make_brv_probe(executable_dir)
    monkeypatch.setenv("PATH", f"{executable_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("BRV_API_KEY", "other-profile-secret")
    secret_scope.set_multiplex_active(True)
    home = tmp_path / "profile"

    blocker = BlockBuster()
    blocker.activate()
    try:
        result = await _run_in_profile(home, {}, "status")
    finally:
        blocker.deactivate()

    assert result == {
        "success": True,
        "output": f"<missing>|{home}|{home / 'byterover'}",
    }


@pytest.mark.asyncio
async def test_unscoped_multiplex_call_fails_before_spawning_child(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _make_brv_probe(tmp_path)
    marker = tmp_path / "spawned"
    executable.write_text(
        "#!/bin/sh\n"
        f"touch {str(marker)!r}\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setattr(byterover, "_cached_brv_path", str(executable))
    monkeypatch.setenv("BRV_API_KEY", "other-profile-secret")
    secret_scope.set_multiplex_active(True)

    result = await byterover._run_brv(["status"], cwd=str(tmp_path))

    assert result["success"] is False
    assert "no profile secret scope active" in result["error"]
    assert not marker.exists()


@pytest.mark.asyncio
async def test_binary_cache_is_canonical_profile_scoped_and_path_sensitive(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "profile"
    home.mkdir()
    alias = tmp_path / "profile-alias"
    alias.symlink_to(home, target_is_directory=True)
    other = tmp_path / "other-profile"
    other.mkdir()
    monkeypatch.setenv("PATH", "/first/bin")
    monkeypatch.setenv("HOME", str(tmp_path))
    calls: list[tuple[str, str]] = []

    async def which(_name: str, *, path: str):
        calls.append((os.fspath(get_hermes_home()), path))
        await asyncio.sleep(0)
        return f"{path}/brv"

    monkeypatch.setattr(byterover, "_which", which)

    async def resolve(profile: Path):
        token = set_hermes_home_override(profile)
        try:
            return await byterover._resolve_brv_path()
        finally:
            reset_hermes_home_override(token)

    first, sibling = await asyncio.gather(resolve(home), resolve(alias))
    other_result = await resolve(other)
    monkeypatch.setenv("PATH", "/rotated/bin")
    rotated = await resolve(home)

    assert first == sibling == "/first/bin/brv"
    assert other_result == "/first/bin/brv"
    assert rotated == "/rotated/bin/brv"
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_provider_prompt_state_tracks_its_profile_availability(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    _make_brv_probe(executable_dir)
    monkeypatch.setenv("PATH", f"{executable_dir}{os.pathsep}{os.environ['PATH']}")
    provider = byterover.ByteRoverMemoryProvider()

    assert await provider.is_available() is True
    assert provider.system_prompt_block().startswith("# ByteRover Memory")


def test_sequential_event_loops_do_not_retain_cache_owners(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/test/bin")
    monkeypatch.setenv("HOME", str(tmp_path))
    loop_refs: list[weakref.ReferenceType[asyncio.AbstractEventLoop]] = []

    async def which(_name: str, *, path: str):
        return f"{path}/brv"

    monkeypatch.setattr(byterover, "_which", which)

    for index in range(2):
        home = tmp_path / f"profile-{index}"
        home.mkdir()

        async def cycle() -> None:
            loop_refs.append(weakref.ref(asyncio.get_running_loop()))
            token = set_hermes_home_override(home)
            try:
                assert await byterover._resolve_brv_path() == "/test/bin/brv"
            finally:
                reset_hermes_home_override(token)

        asyncio.run(cycle())

    gc.collect()
    assert all(loop_ref() is None for loop_ref in loop_refs)
    assert not byterover._brv_path_states
    assert not byterover._brv_scope_aliases


def test_concurrent_event_loops_keep_profile_subprocesses_isolated(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    _make_brv_probe(executable_dir)
    monkeypatch.setenv("PATH", f"{executable_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("BRV_API_KEY", "process-leak")
    secret_scope.set_multiplex_active(True)
    barrier = threading.Barrier(2)
    results: queue.Queue[tuple[str, dict | BaseException]] = queue.Queue()

    def worker(label: str) -> None:
        home = tmp_path / f"profile-{label}"
        try:
            barrier.wait(timeout=5)
            result = asyncio.run(
                _run_in_profile(
                    home,
                    {"BRV_API_KEY": f"key-{label}"},
                    "status",
                )
            )
        except BaseException as exc:
            results.put((label, exc))
        else:
            results.put((label, result))

    threads = [
        threading.Thread(target=worker, args=(label,), daemon=True)
        for label in ("a", "b")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    observed = dict(results.get_nowait() for _ in range(2))
    assert observed == {
        "a": {
            "success": True,
            "output": (
                f"key-a|{tmp_path / 'profile-a'}|"
                f"{tmp_path / 'profile-a' / 'byterover'}"
            ),
        },
        "b": {
            "success": True,
            "output": (
                f"key-b|{tmp_path / 'profile-b'}|"
                f"{tmp_path / 'profile-b' / 'byterover'}"
            ),
        },
    }


async def _wait_for_pid(path: Path) -> int:
    for _ in range(500):
        if path.exists():
            return int(path.read_text(encoding="utf-8"))
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out waiting for {path}")


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group semantics")
async def test_timeout_kills_term_ignoring_grandchild(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_pid_path = tmp_path / "grandchild.pid"
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
    monkeypatch.setattr(byterover, "_cached_brv_path", sys.executable)

    task = asyncio.create_task(
        byterover._run_brv(
            ["-c", parent],
            timeout=0.5,
            cwd=str(tmp_path),
        )
    )
    child_pid = await _wait_for_pid(child_pid_path)
    result = await asyncio.wait_for(task, timeout=5)

    assert result == {"success": False, "error": "brv timed out after 0.5s"}
    for _ in range(100):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("TERM-ignoring ByteRover grandchild survived timeout")


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group semantics")
async def test_repeated_cancellation_kills_term_ignoring_grandchild(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_pid_path = tmp_path / "cancelled-grandchild.pid"
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
    monkeypatch.setattr(byterover, "_cached_brv_path", sys.executable)
    task = asyncio.create_task(
        byterover._run_brv(
            ["-c", parent],
            timeout=30,
            cwd=str(tmp_path),
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
        pytest.fail("TERM-ignoring ByteRover grandchild survived cancellation")
