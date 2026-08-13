"""Native-async parity tests for the upstream execute_code tool."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agent import secret_scope
from tools import code_execution_tool as code_tool


@pytest.fixture(autouse=True)
def _local_execute_code(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr(
        "tools.terminal_tool._get_env_config",
        AsyncMock(return_value={"env_type": "local", "docker_volumes": []}),
    )
    monkeypatch.setattr(
        "tools.approval.check_execute_code_guard",
        AsyncMock(return_value={"approved": True, "message": None}),
    )
    monkeypatch.setattr(
        code_tool,
        "_load_config",
        AsyncMock(
            return_value={"timeout": 10, "max_tool_calls": 50, "mode": "strict"}
        ),
    )


@pytest.mark.asyncio
async def test_upstream_schema_and_availability_contract():
    assert await code_tool.check_sandbox_requirements() is True
    assert code_tool.EXECUTE_CODE_SCHEMA["name"] == "execute_code"
    assert code_tool.EXECUTE_CODE_SCHEMA["parameters"]["required"] == ["code"]

    schema = await code_tool.build_execute_code_schema({"terminal", "read_file"})
    assert "terminal(" in schema["description"]
    assert "read_file(" in schema["description"]
    assert "web_search(" not in schema["description"]


@pytest.mark.asyncio
async def test_mode_resolution_preserves_upstream_config_contract(monkeypatch):
    monkeypatch.setattr(code_tool, "_load_config", AsyncMock(return_value={}))
    assert await code_tool._get_execution_mode() == "project"

    code_tool._load_config.return_value = {"mode": "strict"}
    assert await code_tool._get_execution_mode() == "strict"

    code_tool._load_config.return_value = {"mode": "invalid"}
    assert await code_tool._get_execution_mode() == "project"


@pytest.mark.asyncio
async def test_child_python_and_cwd_mode_contract(monkeypatch, tmp_path):
    assert await code_tool._resolve_child_python("strict") == sys.executable
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    assert await code_tool._resolve_child_python("project") == sys.executable

    staging = str(tmp_path / "staging")
    assert await code_tool._resolve_child_cwd("strict", staging) == staging
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    assert await code_tool._resolve_child_cwd("project", staging) == str(tmp_path)


@pytest.mark.asyncio
async def test_project_child_cwd_uses_active_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path / "foreign"))
    profile_cwds = {label: tmp_path / label for label in ("a", "b")}
    for cwd in profile_cwds.values():
        cwd.mkdir()
    previous_multiplex = secret_scope.is_multiplex_active()
    outer_scope = secret_scope.set_secret_scope(None)
    secret_scope.set_multiplex_active(True)

    async def resolve(label: str):
        token = secret_scope.set_secret_scope(
            {"TERMINAL_CWD": str(profile_cwds[label])}
        )
        try:
            await asyncio.sleep(0)
            return await code_tool._resolve_child_cwd(
                "project", str(tmp_path / "staging")
            )
        finally:
            secret_scope.reset_secret_scope(token)

    try:
        assert await asyncio.gather(resolve("a"), resolve("b")) == [
            str(profile_cwds["a"]),
            str(profile_cwds["b"]),
        ]
        with pytest.raises(secret_scope.UnscopedSecretError):
            await code_tool._resolve_child_cwd(
                "project", str(tmp_path / "staging")
            )
    finally:
        secret_scope.set_multiplex_active(previous_multiplex)
        secret_scope.reset_secret_scope(outer_scope)


@pytest.mark.asyncio
async def test_project_mode_runs_in_session_cwd_without_leaking_secrets(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        code_tool,
        "_load_config",
        AsyncMock(
            return_value={"timeout": 10, "max_tool_calls": 50, "mode": "project"}
        ),
    )
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    result = json.loads(
        await code_tool.execute_code(
            "import os\n"
            "import hermes_tools\n"
            "print(os.getcwd())\n"
            "print(os.environ.get('OPENAI_API_KEY', 'MISSING'))\n"
            "print(hasattr(hermes_tools, 'execute_code'))\n"
        )
    )

    assert result["status"] == "success"
    assert result["output"] == f"{tmp_path}\nMISSING\nFalse\n"
    assert "must-not-leak" not in result["output"]


@pytest.mark.asyncio
async def test_child_env_windows_allowlist_and_secret_scrub():
    source = {
        "PATH": os.defpath,
        "SYSTEMROOT": r"C:\\Windows",
        "windir": r"C:\\Windows",
        "OPENAI_API_KEY": "secret",
        "GITHUB_TOKEN": "secret",
    }
    scrubbed = await code_tool._scrub_child_env(
        source,
        is_passthrough=lambda _name: False,
        is_windows=True,
    )

    assert scrubbed["SYSTEMROOT"] == r"C:\\Windows"
    assert scrubbed["windir"] == r"C:\\Windows"
    assert scrubbed["PATH"] == os.defpath
    assert "OPENAI_API_KEY" not in scrubbed
    assert "GITHUB_TOKEN" not in scrubbed


@pytest.mark.asyncio
async def test_explicit_passthrough_precedes_secret_substring_block(monkeypatch):
    monkeypatch.setattr(
        "tools.env_passthrough.resolve_passthrough_value",
        lambda _name, fallback: fallback,
    )
    scrubbed = await code_tool._scrub_child_env(
        {"CUSTOM_API_KEY": "allowed-by-opt-in"},
        is_passthrough=lambda name: name == "CUSTOM_API_KEY",
        is_windows=False,
    )
    assert scrubbed == {"CUSTOM_API_KEY": "allowed-by-opt-in"}


def test_generated_module_preserves_upstream_public_stubs():
    source = code_tool.generate_hermes_tools_module(
        list(code_tool.SANDBOX_ALLOWED_TOOLS)
    )
    compile(source, "hermes_tools.py", "exec")
    for tool_name in code_tool.SANDBOX_ALLOWED_TOOLS:
        assert f"def {tool_name}(" in source
    assert "HERMES_RPC_TOKEN" in source
    assert "_call_lock = threading.Lock()" in source


@pytest.mark.asyncio
async def test_basic_print_preserves_trajectory_visible_json_result():
    result = json.loads(await code_tool.execute_code('print("hello world")'))

    assert result["status"] == "success"
    assert result["output"] == "hello world\n"
    assert result["exit_code"] == 0
    assert result["tool_calls_made"] == 0
    assert result["stdout_truncated"] is False
    assert result["stdout_bytes_captured"] == 12
    assert result["stdout_bytes_total"] == 12
    assert result["stdout_bytes_omitted"] == 0
    assert isinstance(result["duration_seconds"], float)


@pytest.mark.asyncio
async def test_tool_call_observation_round_trips_in_order(monkeypatch):
    calls = []

    async def handle(name, args, task_id=None, **_kwargs):
        await asyncio.sleep(0)
        calls.append((name, args, task_id))
        return json.dumps({"output": f"observed: {args['command']}"})

    monkeypatch.setattr("model_tools.handle_function_call", handle)
    code = """
from hermes_tools import terminal
first = terminal("echo first")
second = terminal("echo second")
print(first["output"])
print(second["output"])
"""
    result = json.loads(
        await code_tool.execute_code(
            code,
            task_id="trajectory-task",
            enabled_tools=["terminal"],
        )
    )

    assert result["status"] == "success"
    assert result["output"] == "observed: echo first\nobserved: echo second\n"
    assert result["tool_calls_made"] == 2
    assert calls == [
        ("terminal", {"command": "echo first", "timeout": None, "workdir": None}, "trajectory-task"),
        ("terminal", {"command": "echo second", "timeout": None, "workdir": None}, "trajectory-task"),
    ]


@pytest.mark.asyncio
async def test_concurrent_child_calls_keep_responses_matched(monkeypatch):
    async def handle(name, args, **_kwargs):
        assert name == "terminal"
        await asyncio.sleep(0.005)
        return json.dumps({"output": args["command"].removeprefix("echo ")})

    monkeypatch.setattr("model_tools.handle_function_call", handle)
    code = """
from concurrent.futures import ThreadPoolExecutor
from hermes_tools import terminal

def call(i):
    return i, terminal(f"echo TAG-{i}")["output"]

with ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(call, range(8)))
print(results)
assert all(value == f"TAG-{index}" for index, value in results)
"""
    result = json.loads(
        await code_tool.execute_code(code, enabled_tools=["terminal"])
    )

    assert result["status"] == "success"
    assert result["tool_calls_made"] == 8


@pytest.mark.asyncio
async def test_rpc_token_rejects_request_before_dispatch(monkeypatch):
    handler = AsyncMock(return_value="{}")
    monkeypatch.setattr("model_tools.handle_function_call", handler)
    response = json.loads(
        await code_tool._dispatch_rpc_request(
            {"tool": "terminal", "args": {"command": "echo never"}},
            task_id="task-1",
            tool_call_log=[],
            tool_call_counter=[0],
            max_tool_calls=50,
            allowed_tools=frozenset({"terminal"}),
            rpc_token="expected-token",
        )
    )
    assert response == {"error": "Unauthorized RPC request"}
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_timeout_kills_child_and_preserves_json_contract(monkeypatch):
    monkeypatch.setattr(
        code_tool,
        "_load_config",
        AsyncMock(
            return_value={"timeout": 0.05, "max_tool_calls": 50, "mode": "strict"}
        ),
    )
    result = json.loads(
        await code_tool.execute_code(
            "import time\nprint('before', flush=True)\ntime.sleep(60)"
        )
    )

    assert result["status"] == "timeout"
    assert result["exit_code"] != 0
    assert "before" in result["output"]
    assert "timed out after 0.05s and was killed" in result["error"]
    assert "⏰" in result["output"]


@pytest.mark.asyncio
async def test_large_stdout_preserves_head_tail_and_metadata():
    result = json.loads(
        await code_tool.execute_code(
            "print('HEAD')\nprint('x' * 80000)\nprint('TAIL')"
        )
    )

    assert result["status"] == "success"
    assert result["stdout_truncated"] is True
    assert result["stdout_bytes_total"] > result["stdout_bytes_captured"]
    assert result["stdout_bytes_omitted"] > 0
    assert "HEAD" in result["output"]
    assert "TAIL" in result["output"]
    assert "execute_code stdout was truncated" in result["warning"]


@pytest.mark.asyncio
async def test_tool_whitelist_is_enforced_in_generated_child_module():
    result = json.loads(
        await code_tool.execute_code(
            "from hermes_tools import web_search",
            enabled_tools=["terminal"],
        )
    )
    assert result["status"] == "error"
    assert "cannot import name 'web_search'" in result["error"]


@pytest.mark.asyncio
async def test_non_ascii_child_stdio_remains_utf8():
    result = json.loads(await code_tool.execute_code("print('café → 한글')"))
    assert result["status"] == "success"
    assert result["output"] == "café → 한글\n"


@pytest.mark.asyncio
async def test_cancellation_reaps_process_and_owned_tasks(monkeypatch, tmp_path):
    original_spawn = asyncio.create_subprocess_exec
    spawned = []

    async def capture_spawn(*args, **kwargs):
        process = await original_spawn(*args, **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr(code_tool.asyncio, "create_subprocess_exec", capture_spawn)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    task = asyncio.create_task(
        code_tool.execute_code("import time\ntime.sleep(60)"),
        name="execute-code-caller",
    )
    for _ in range(100):
        if spawned:
            break
        await asyncio.sleep(0.01)
    assert spawned

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert spawned[0].returncode is not None
    assert not [
        path for path in tmp_path.iterdir() if path.name.startswith("hermes_sandbox_")
    ]
    owned = {
        pending.get_name()
        for pending in asyncio.all_tasks()
        if pending is not asyncio.current_task()
    }
    assert not {name for name in owned if name.startswith("execute-code")}


@pytest.mark.asyncio
async def test_child_sleep_does_not_block_event_loop():
    ticks = 0
    stop = asyncio.Event()

    async def ticker():
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            await asyncio.sleep(0.005)

    ticker_task = asyncio.create_task(ticker())
    try:
        result = json.loads(
            await code_tool.execute_code("import time\ntime.sleep(0.15)\nprint('done')")
        )
    finally:
        stop.set()
        await ticker_task

    assert result["status"] == "success"
    assert ticks >= 10


@pytest.mark.asyncio
async def test_approval_guard_receives_upstream_arguments(monkeypatch):
    guard = AsyncMock(
        return_value={"approved": False, "message": "blocked by test"}
    )
    monkeypatch.setattr("tools.approval.check_execute_code_guard", guard)
    monkeypatch.setattr(
        "tools.terminal_tool._get_env_config",
        AsyncMock(
            return_value={
                "env_type": "docker",
                "docker_volumes": ["/host:/workspace"],
            }
        ),
    )

    result = json.loads(await code_tool.execute_code("print('never runs')"))

    assert result == {
        "status": "error",
        "error": "blocked by test",
        "tool_calls_made": 0,
        "duration_seconds": 0,
    }
    guard.assert_awaited_once_with(
        "print('never runs')",
        "docker",
        has_host_access=True,
    )


class _RemoteEnvironment:
    def __init__(self, script_result=None):
        self.commands = []
        self.script_result = script_result or {
            "output": "remote output\n",
            "returncode": 0,
        }

    async def get_temp_dir(self):
        return "/remote/tmp"

    async def execute(self, command, cwd=None, **kwargs):
        timeout = kwargs.get("timeout")
        self.commands.append((command, cwd, timeout))
        if "command -v python3" in command:
            return {"output": "OK\n", "returncode": 0}
        if "python3 script.py" in command:
            return self.script_result
        return {"output": "", "returncode": 0}


@pytest.mark.asyncio
async def test_remote_execution_preserves_commands_and_result(monkeypatch):
    env = _RemoteEnvironment()
    monkeypatch.setattr(
        code_tool,
        "_get_or_create_env",
        AsyncMock(return_value=(env, "ssh")),
    )

    result = json.loads(
        await code_tool._execute_remote("print('remote')", "task-1", ["terminal"])
    )

    assert result["status"] == "success"
    assert result["output"] == "remote output\n"
    assert result["exit_code"] == 0
    assert result["tool_calls_made"] == 0
    commands = [command for command, _, _ in env.commands]
    assert any("mkdir -p /remote/tmp/hermes_exec_" in command for command in commands)
    assert any("HERMES_RPC_DIR=/remote/tmp/hermes_exec_" in command for command in commands)
    assert any("rm -rf /remote/tmp/hermes_exec_" in command for command in commands)


@pytest.mark.asyncio
async def test_remote_cancellation_cleans_sandbox_and_reraises(monkeypatch):
    script_started = asyncio.Event()
    release_script = asyncio.Event()

    class SlowRemoteEnvironment(_RemoteEnvironment):
        async def execute(self, command, cwd=None, **kwargs):
            timeout = kwargs.get("timeout")
            self.commands.append((command, cwd, timeout))
            if "command -v python3" in command:
                return {"output": "OK\n", "returncode": 0}
            if "python3 script.py" in command:
                script_started.set()
                await release_script.wait()
                return {"output": "late", "returncode": 0}
            return {"output": "", "returncode": 0}

    env = SlowRemoteEnvironment()
    monkeypatch.setattr(
        code_tool,
        "_get_or_create_env",
        AsyncMock(return_value=(env, "ssh")),
    )
    task = asyncio.create_task(
        code_tool._execute_remote("print('remote')", "task-1", ["terminal"])
    )
    await asyncio.wait_for(script_started.wait(), timeout=2)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    commands = [command for command, _, _ in env.commands]
    assert any("rm -rf /remote/tmp/hermes_exec_" in command for command in commands)
    owned = {
        pending.get_name()
        for pending in asyncio.all_tasks()
        if pending is not asyncio.current_task()
    }
    assert not {name for name in owned if name.startswith("execute-code")}


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group assertion")
async def test_nonzero_exit_preserves_stderr_and_recovery_shape():
    result = json.loads(
        await code_tool.execute_code(
            "print('before error')\nraise RuntimeError('deliberate crash')"
        )
    )

    assert result["status"] == "error"
    assert result["exit_code"] != 0
    assert "before error" in result["output"]
    assert "--- stderr ---" in result["output"]
    assert "RuntimeError: deliberate crash" in result["error"]


@pytest.mark.asyncio
async def test_empty_code_keeps_upstream_tool_error_shape():
    result = json.loads(await code_tool.execute_code("   "))
    assert result == {"error": "No code provided."}


def test_parent_runtime_has_no_thread_or_sync_subprocess_fallback():
    source = Path(code_tool.__file__).read_text(encoding="utf-8")
    parent_source = source.split("_HERMES_TOOLS_HEADER = r'''", 1)[0]
    parent_source += source.split("# RPC server", 1)[1]
    assert "asyncio.to_thread" not in parent_source
    assert "run_in_executor" not in parent_source
    assert "run_until_complete" not in parent_source
    assert "subprocess.Popen" not in parent_source
    assert "threading.Thread" not in parent_source
