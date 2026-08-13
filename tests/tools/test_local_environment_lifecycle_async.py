"""Native-async parity regressions for the retained local environment."""

from __future__ import annotations

import asyncio
import os
import signal
from unittest.mock import AsyncMock, patch

import pytest

from tools.environments import local as local_mod
from tools.environments.local import (
    LocalEnvironment,
    _find_bash,
    _finish_stream_readers,
    _resolve_shell_init_files,
    _terminate_and_reap,
)


async def _wait_for_group_exit(pgid: int, grace_seconds: float = 5.0) -> bool:
    deadline = asyncio.get_running_loop().time() + grace_seconds
    while asyncio.get_running_loop().time() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        await asyncio.sleep(0.05)
    return False


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group semantics")
async def test_terminate_reaps_term_ignoring_group_after_wrapper_exits(
    monkeypatch,
) -> None:
    shell = await _find_bash()
    process = await asyncio.create_subprocess_exec(
        shell,
        "-c",
        "trap '' TERM; sleep 60 & exit 0",
        start_new_session=True,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    process._hermes_pgid = process.pid
    pgid = process.pid
    signals = []
    real_killpg = os.killpg

    def record_killpg(group: int, sig: int) -> None:
        signals.append(sig)
        real_killpg(group, sig)

    monkeypatch.setattr(local_mod.os, "killpg", record_killpg)
    try:
        await process.wait()
        assert process.returncode == 0
        os.killpg(pgid, 0)

        await _terminate_and_reap(process)

        assert await _wait_for_group_exit(pgid)
        term_index = signals.index(signal.SIGTERM)
        kill_index = signals.index(signal.SIGKILL)
        assert term_index < kill_index
        assert 0 in signals[term_index + 1 : kill_index]
    finally:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group semantics")
async def test_stdin_drain_cancellation_reaps_process_and_readers(
    tmp_path,
    monkeypatch,
) -> None:
    original_create = asyncio.create_subprocess_exec
    spawned = []

    async def capture_process(*args, **kwargs):
        process = await original_create(*args, **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr(local_mod.asyncio, "create_subprocess_exec", capture_process)
    environment = LocalEnvironment(cwd=str(tmp_path), timeout=60)
    task = asyncio.create_task(
        environment._run_bash(
            "trap '' TERM; while :; do sleep 60; done",
            timeout=60,
            stdin_data="x" * (8 * 1024 * 1024),
        )
    )
    try:
        while not spawned:
            await asyncio.sleep(0)
        process = spawned[0]
        pgid = process.pid
        await asyncio.sleep(0.05)
        assert not task.done()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)

        assert await _wait_for_group_exit(pgid)
        assert process.returncode is not None
        assert not [
            pending
            for pending in asyncio.all_tasks()
            if not pending.done()
            and pending is not asyncio.current_task()
            and pending.get_name().startswith(("local-output-", "local-wait-"))
        ]
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if spawned:
            try:
                os.killpg(spawned[0].pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        await environment.cleanup()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group semantics")
async def test_foreground_interrupt_reaps_real_group_and_marks_result(
    tmp_path,
    monkeypatch,
) -> None:
    from tools.interrupt import _bind_interrupt_event, _reset_interrupt_event

    original_create = asyncio.create_subprocess_exec
    spawned = []

    async def capture_process(*args, **kwargs):
        process = await original_create(*args, **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr(local_mod.asyncio, "create_subprocess_exec", capture_process)
    environment = LocalEnvironment(cwd=str(tmp_path), timeout=60)
    interrupt = asyncio.Event()
    token = _bind_interrupt_event(interrupt)
    try:
        run = asyncio.create_task(
            environment._run_bash(
                "trap '' TERM; while :; do sleep 60; done",
                timeout=60,
            )
        )
        while not spawned:
            await asyncio.sleep(0)
        pgid = spawned[0].pid
        interrupt.set()

        result = await asyncio.wait_for(run, timeout=5)

        assert result == {
            "output": "[Command interrupted]",
            "returncode": 130,
        }
        assert await _wait_for_group_exit(pgid)
        assert spawned[0].returncode is not None
    finally:
        _reset_interrupt_event(token)
        if "run" in locals() and not run.done():
            run.cancel()
            await asyncio.gather(run, return_exceptions=True)
        if spawned:
            try:
                os.killpg(spawned[0].pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        await environment.cleanup()


@pytest.mark.asyncio
async def test_natural_exit_130_has_no_interrupt_marker(tmp_path) -> None:
    environment = LocalEnvironment(cwd=str(tmp_path), timeout=5)
    try:
        result = await environment._run_bash("exit 130", timeout=5)
    finally:
        await environment.cleanup()

    assert result == {"output": "", "returncode": 130}


@pytest.mark.asyncio
async def test_reader_collection_finishes_when_its_caller_is_cancelled() -> None:
    never = asyncio.Event()
    reader = asyncio.create_task(never.wait(), name="test-local-reader")
    cleanup = asyncio.create_task(_finish_stream_readers([reader]))
    await asyncio.sleep(0)

    cleanup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cleanup

    assert reader.cancelled()


@pytest.mark.asyncio
async def test_shell_init_candidate_failure_does_not_hide_later_file(
    tmp_path,
) -> None:
    valid = tmp_path / "valid.sh"
    valid.write_text("export VALID=1\n", encoding="utf-8")
    config = {
        "terminal": {
            "shell_init_files": ["invalid\0path", str(valid)],
            "auto_source_bashrc": False,
        }
    }
    with patch(
        "hermes_cli.config.load_config_readonly",
        new=AsyncMock(return_value=config),
    ):
        assert await _resolve_shell_init_files() == [str(valid)]


@pytest.mark.asyncio
async def test_login_shell_keeps_profile_path_precedence(tmp_path) -> None:
    init_file = tmp_path / "init.sh"
    init_file.write_text(
        'export PATH="/profile-path-sentinel:$PATH"\n',
        encoding="utf-8",
    )
    environment = LocalEnvironment(
        cwd=str(tmp_path),
        env={"PATH": f"/bootstrap-path-sentinel{os.pathsep}{os.defpath}"},
    )
    try:
        with patch(
            "tools.environments.local._resolve_shell_init_files",
            new=AsyncMock(return_value=[str(init_file)]),
        ):
            result = await environment._run_bash(
                'printf "%s" "$PATH"',
                login=True,
                timeout=5,
            )
    finally:
        await environment.cleanup()

    assert result["returncode"] == 0
    assert result["output"].split(":", 1)[0] == "/profile-path-sentinel"
    assert "_HERMES_BOOTSTRAP_PATH" not in environment.env


@pytest.mark.asyncio
async def test_windows_marker_normalizes_and_invalid_marker_rolls_back(
    tmp_path,
) -> None:
    environment = LocalEnvironment(cwd=r"C:\Users\old")
    marker = environment._cwd_marker

    with (
        patch("tools.environments.local._IS_WINDOWS", True),
        patch(
            "tools.environments.local.aiofiles.os.path.isdir",
            new=AsyncMock(side_effect=lambda path: path == r"D:\work"),
        ),
    ):
        previous = environment.cwd
        valid = {"output": f"x\n{marker}/d/work{marker}\n", "returncode": 0}
        environment._extract_cwd_from_output(valid)
        await environment._validate_cwd_update(previous)
        assert environment.cwd == r"D:\work"
        assert marker not in valid["output"]

        previous = environment.cwd
        invalid = {
            "output": f"x\n{marker}/e/deleted{marker}\n",
            "returncode": 0,
        }
        environment._extract_cwd_from_output(invalid)
        await environment._validate_cwd_update(previous)
        assert environment.cwd == r"D:\work"
        assert marker not in invalid["output"]

    await environment.cleanup()


@pytest.mark.asyncio
async def test_invalid_marker_does_not_overwrite_concurrent_valid_cwd(
    tmp_path,
) -> None:
    original = tmp_path / "original"
    concurrent = tmp_path / "concurrent"
    original.mkdir()
    concurrent.mkdir()
    environment = LocalEnvironment(cwd=str(original))
    marker = environment._cwd_marker
    validation_started = asyncio.Event()
    release_validation = asyncio.Event()

    async def invalid_after_release(_path) -> bool:
        validation_started.set()
        await release_validation.wait()
        return False

    async def validate_invalid_marker() -> None:
        previous = environment.cwd
        result = {
            "output": f"x\n{marker}{tmp_path / 'deleted'}{marker}\n",
            "returncode": 0,
        }
        environment._extract_cwd_from_output(result)
        await environment._validate_cwd_update(previous)

    with patch(
        "tools.environments.local.aiofiles.os.path.isdir",
        new=invalid_after_release,
    ):
        task = asyncio.create_task(validate_invalid_marker())
        await validation_started.wait()
        environment.cwd = str(concurrent)
        release_validation.set()
        await task

    assert environment.cwd == str(concurrent)
    await environment.cleanup()


@pytest.mark.asyncio
async def test_reused_environment_reresolves_passthrough_for_active_profile(
    tmp_path,
) -> None:
    from agent.secret_scope import (
        reset_secret_scope,
        set_multiplex_active,
        set_secret_scope,
    )
    from tools.env_passthrough import (
        clear_env_passthrough,
        register_env_passthrough,
    )

    register_env_passthrough(["PROFILE_SCOPED_TOKEN"])
    set_multiplex_active(True)
    environment = LocalEnvironment(
        cwd=str(tmp_path),
        env={"PROFILE_SCOPED_TOKEN": "foreign-global-value"},
    )
    alpha = set_secret_scope({"PROFILE_SCOPED_TOKEN": "alpha-secret"})
    try:
        await environment._ensure_initialized()
    finally:
        reset_secret_scope(alpha)

    beta = set_secret_scope({"PROFILE_SCOPED_TOKEN": "beta-secret"})
    try:
        result = await environment.execute(
            'printf "%s" "$PROFILE_SCOPED_TOKEN"'
        )
    finally:
        reset_secret_scope(beta)

    missing = set_secret_scope({})
    try:
        missing_result = await environment.execute(
            'printf "%s" "${PROFILE_SCOPED_TOKEN-unset}"'
        )
    finally:
        reset_secret_scope(missing)
        await environment.cleanup()
        set_multiplex_active(False)
        clear_env_passthrough()

    assert result["returncode"] == 0
    assert result["output"] == "beta-secret"
    assert missing_result["returncode"] == 0
    assert missing_result["output"] == "unset"


@pytest.mark.asyncio
async def test_concurrent_reused_environment_keeps_profile_passthrough_isolated(
    tmp_path,
) -> None:
    from agent.secret_scope import (
        reset_secret_scope,
        set_multiplex_active,
        set_secret_scope,
    )
    from tools.env_passthrough import (
        clear_env_passthrough,
        register_env_passthrough,
    )

    register_env_passthrough(["PROFILE_SCOPED_TOKEN"])
    set_multiplex_active(True)
    environment = LocalEnvironment(
        cwd=str(tmp_path),
        env={"PROFILE_SCOPED_TOKEN": "foreign-global-value"},
    )
    initial = set_secret_scope({"PROFILE_SCOPED_TOKEN": "initial-secret"})
    try:
        await environment._ensure_initialized()
    finally:
        reset_secret_scope(initial)

    async def run_for(value: str) -> dict:
        token = set_secret_scope({"PROFILE_SCOPED_TOKEN": value})
        try:
            return await environment.execute(
                'printf "%s" "$PROFILE_SCOPED_TOKEN"'
            )
        finally:
            reset_secret_scope(token)

    try:
        alpha_result, beta_result = await asyncio.gather(
            run_for("alpha-secret"),
            run_for("beta-secret"),
        )
    finally:
        await environment.cleanup()
        set_multiplex_active(False)
        clear_env_passthrough()

    assert alpha_result["returncode"] == 0
    assert beta_result["returncode"] == 0
    assert alpha_result["output"] == "alpha-secret"
    assert beta_result["output"] == "beta-secret"


@pytest.mark.asyncio
async def test_reused_environment_keeps_profile_home_runtime_values_isolated(
    tmp_path,
) -> None:
    from agent.secret_scope import set_multiplex_active
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    homes = [tmp_path / name for name in ("alpha", "beta", "gamma", "delta")]
    for home in homes:
        (home / "home").mkdir(parents=True)
    set_multiplex_active(True)
    environment = LocalEnvironment(
        cwd=str(tmp_path),
        env={"TERMINAL_HOME_MODE": "profile"},
    )
    alpha_token = set_hermes_home_override(homes[0])
    try:
        await environment._ensure_initialized()
    finally:
        reset_hermes_home_override(alpha_token)

    async def run_for(home) -> dict:
        token = set_hermes_home_override(home)
        try:
            return await environment.execute(
                'printf "%s|%s|%s" "$HERMES_HOME" '
                '"$HERMES_REAL_HOME" "$HOME"'
            )
        finally:
            reset_hermes_home_override(token)

    try:
        beta = await run_for(homes[1])
        gamma, delta = await asyncio.gather(
            run_for(homes[2]),
            run_for(homes[3]),
        )
    finally:
        await environment.cleanup()
        set_multiplex_active(False)

    real_home = environment.env["HERMES_REAL_HOME"]
    assert beta["output"] == f"{homes[1]}|{real_home}|{homes[1] / 'home'}"
    assert gamma["output"] == f"{homes[2]}|{real_home}|{homes[2] / 'home'}"
    assert delta["output"] == f"{homes[3]}|{real_home}|{homes[3] / 'home'}"


@pytest.mark.asyncio
async def test_each_spawn_overlays_snapshot_on_current_host_environment(
    tmp_path,
    monkeypatch,
) -> None:
    environment = LocalEnvironment(
        cwd=str(tmp_path),
        env={"HERMES_SNAPSHOT_VALUE": "snapshot"},
    )
    try:
        await environment._ensure_initialized()
        monkeypatch.setenv("HERMES_LATE_HARMLESS", "late-host")
        monkeypatch.setenv("HERMES_SNAPSHOT_VALUE", "wrong-host")

        result = await environment.execute(
            'printf "%s|%s" "$HERMES_LATE_HARMLESS" '
            '"$HERMES_SNAPSHOT_VALUE"'
        )
    finally:
        await environment.cleanup()

    assert result == {"output": "late-host|snapshot", "returncode": 0}
