"""Native-async parity tests for BatchRunner per-row environment overrides."""

from __future__ import annotations

import asyncio
import subprocess

import pytest

import batch_runner
from agent import secret_scope


def _config(*, verbose: bool = False) -> dict:
    return {
        "distribution": "test",
        "model": "test-model",
        "max_iterations": 2,
        "verbose": verbose,
    }


class _Agent:
    instances = []
    events = []
    run_gate: asyncio.Event | None = None

    def __init__(self, **_kwargs):
        self.closed = False
        type(self).instances.append(self)

    async def run_conversation(self, prompt, task_id=None):
        type(self).events.append(("run", task_id, prompt))
        if type(self).run_gate is not None:
            await type(self).run_gate.wait()
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning": "reasoning",
                },
            ],
            "completed": True,
            "partial": False,
            "api_calls": 1,
        }

    def _convert_to_trajectory_format(self, _messages, prompt, completed):
        return [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": "answer"},
            {"completed": completed},
        ]

    async def close(self):
        self.closed = True
        type(self).events.append(("close",))


@pytest.fixture(autouse=True)
def _reset_agent():
    _Agent.instances = []
    _Agent.events = []
    _Agent.run_gate = None


@pytest.fixture
def fake_agent(monkeypatch):
    monkeypatch.setattr(batch_runner, "AIAgent", _Agent)
    monkeypatch.setattr(
        batch_runner,
        "sample_toolsets_from_distribution",
        lambda _distribution: ["terminal"],
    )
    return _Agent


@pytest.mark.asyncio
@pytest.mark.parametrize("image_key", ["image", "docker_image"])
async def test_row_image_and_cwd_register_exact_backend_overrides(
    image_key,
    monkeypatch,
    fake_agent,
):
    events = fake_agent.events

    def register(task_id, overrides):
        events.append(("register", task_id, dict(overrides)))

    def clear(task_id):
        events.append(("clear", task_id))

    monkeypatch.setenv("TERMINAL_ENV", "modal")
    monkeypatch.setattr(
        "tools.terminal_tool.register_task_env_overrides",
        register,
    )
    monkeypatch.setattr(
        "tools.terminal_tool.clear_task_env_overrides",
        clear,
    )

    result = await batch_runner._process_single_prompt(
        7,
        {"prompt": "question", image_key: "registry/image:tag", "cwd": "/work"},
        3,
        _config(),
    )

    assert events == [
        (
            "register",
            "task_7",
            {
                "docker_image": "registry/image:tag",
                "modal_image": "registry/image:tag",
                "singularity_image": "docker://registry/image:tag",
                "daytona_image": "registry/image:tag",
                "cwd": "/work",
            },
        ),
        ("run", "task_7", "question"),
        ("close",),
        ("clear", "task_7"),
    ]
    assert result["success"] is True
    assert result["prompt_index"] == 7
    assert result["trajectory"] == [
        {"from": "human", "value": "question"},
        {"from": "gpt", "value": "answer"},
        {"completed": True},
    ]
    assert result["completed"] is True
    assert result["partial"] is False
    assert result["api_calls"] == 1
    assert result["toolsets_used"] == ["terminal"]


@pytest.mark.asyncio
async def test_row_image_probe_uses_concurrent_profile_backend(
    monkeypatch,
    fake_agent,
):
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    previous_multiplex = secret_scope.is_multiplex_active()
    outer_scope = secret_scope.set_secret_scope(None)
    secret_scope.set_multiplex_active(True)
    docker_calls: list[list[str]] = []

    async def docker_command(argv, **_kwargs):
        docker_calls.append(list(argv))
        await asyncio.sleep(0)
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(batch_runner, "_run_docker_image_command", docker_command)

    async def run(label: str, backend: str):
        token = secret_scope.set_secret_scope({"TERMINAL_ENV": backend})
        try:
            return await batch_runner._process_single_prompt(
                1 if label == "a" else 2,
                {"prompt": label, "image": f"registry/{label}:tag"},
                0,
                _config(),
            )
        finally:
            secret_scope.reset_secret_scope(token)

    try:
        profile_a, profile_b = await asyncio.gather(
            run("a", "docker"),
            run("b", "modal"),
        )
        assert profile_a["success"] is True
        assert profile_b["success"] is True
        assert docker_calls == [
            ["docker", "image", "inspect", "registry/a:tag"]
        ]

        unscoped = await batch_runner._process_single_prompt(
            3,
            {"prompt": "unscoped", "image": "registry/unscoped:tag"},
            0,
            _config(),
        )
        assert unscoped["success"] is False
        assert "get_secret('TERMINAL_ENV')" in unscoped["error"]
    finally:
        secret_scope.set_multiplex_active(previous_multiplex)
        secret_scope.reset_secret_scope(outer_scope)


@pytest.mark.asyncio
async def test_cwd_without_image_preserves_upstream_no_override_behavior(
    monkeypatch,
    fake_agent,
):
    register = monkeypatch.setattr(
        "tools.terminal_tool.register_task_env_overrides",
        lambda *_args: pytest.fail("cwd-only rows must not register an override"),
    )

    result = await batch_runner._process_single_prompt(
        1,
        {"prompt": "question", "cwd": "/work"},
        0,
        _config(),
    )

    assert register is None
    assert result["success"] is True


class _CompletedProcess:
    def __init__(self, returncode: int, stdout: bytes = b"", stderr: bytes = b""):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self):
        return self._stdout, self._stderr


@pytest.mark.asyncio
async def test_docker_cache_miss_pull_failure_preserves_error_shape(
    monkeypatch,
    fake_agent,
):
    processes = [
        _CompletedProcess(1),
        _CompletedProcess(9, stderr=("pull failed " + "x" * 600).encode()),
    ]
    calls = []

    async def create(*argv, **kwargs):
        calls.append((list(argv), kwargs))
        return processes.pop(0)

    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setattr(batch_runner.asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(
        "tools.terminal_tool.register_task_env_overrides",
        lambda *_args: pytest.fail("failed pulls must not register an override"),
    )

    result = await batch_runner._process_single_prompt(
        4,
        {"prompt": "question", "image": "missing:image"},
        2,
        _config(),
    )

    assert [call[0] for call in calls] == [
        ["docker", "image", "inspect", "missing:image"],
        ["docker", "pull", "missing:image"],
    ]
    assert all(
        call[1]
        == {
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
        }
        for call in calls
    )
    assert result == {
        "success": False,
        "prompt_index": 4,
        "error": "Docker image not available: missing:image\n"
        + ("pull failed " + "x" * 600)[:500],
        "trajectory": None,
        "tool_stats": {},
        "toolsets_used": [],
        "metadata": {
            "batch_num": 2,
            "timestamp": result["metadata"]["timestamp"],
        },
    }
    assert fake_agent.instances == []


class _PendingProcess:
    def __init__(self, *, release_on_kill: bool = True):
        self.returncode = None
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.killed = False
        self.waited = False
        self.release_on_kill = release_on_kill

    async def communicate(self):
        self.started.set()
        await self.release.wait()
        self.waited = True
        return b"partial-out", b"partial-err"

    def kill(self):
        self.killed = True
        self.returncode = -9
        if self.release_on_kill:
            self.release.set()

    async def wait(self):
        await self.release.wait()
        self.waited = True
        return self.returncode


@pytest.mark.asyncio
async def test_docker_probe_timeout_kills_drains_and_raises_timeout(monkeypatch):
    process = _PendingProcess()
    monkeypatch.setattr(
        batch_runner.asyncio,
        "create_subprocess_exec",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=process),
    )

    with pytest.raises(subprocess.TimeoutExpired) as raised:
        await batch_runner._run_docker_image_command(
            ["docker", "image", "inspect", "slow:image"],
            timeout=0.01,
        )

    assert raised.value.cmd == ["docker", "image", "inspect", "slow:image"]
    assert raised.value.timeout == 0.01
    assert raised.value.output == b"partial-out"
    assert raised.value.stderr == b"partial-err"
    assert process.killed is True
    assert process.waited is True


@pytest.mark.asyncio
async def test_docker_probe_cancellation_kills_and_reaps_without_task_leak(
    monkeypatch,
):
    process = _PendingProcess(release_on_kill=False)
    monkeypatch.setattr(
        batch_runner.asyncio,
        "create_subprocess_exec",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=process),
    )
    run = asyncio.create_task(
        batch_runner._run_docker_image_command(
            ["docker", "image", "inspect", "slow:image"],
            timeout=600,
        )
    )
    await process.started.wait()
    run.cancel()
    while not process.killed:
        await asyncio.sleep(0)
    run.cancel()
    await asyncio.sleep(0)
    run.cancel()
    process.release.set()

    with pytest.raises(asyncio.CancelledError):
        await run

    assert process.killed is True
    assert process.waited is True
    assert not [
        task
        for task in asyncio.all_tasks()
        if not task.done()
        and task is not asyncio.current_task()
        and task.get_name().startswith("batch-docker-probe-")
    ]


@pytest.mark.asyncio
async def test_row_cancellation_closes_agent_and_clears_registered_override(
    monkeypatch,
    fake_agent,
):
    events = fake_agent.events
    fake_agent.run_gate = asyncio.Event()
    monkeypatch.setenv("TERMINAL_ENV", "modal")
    monkeypatch.setattr(
        "tools.terminal_tool.register_task_env_overrides",
        lambda task_id, overrides: events.append(
            ("register", task_id, dict(overrides))
        ),
    )
    monkeypatch.setattr(
        "tools.terminal_tool.clear_task_env_overrides",
        lambda task_id: events.append(("clear", task_id)),
    )
    run = asyncio.create_task(
        batch_runner._process_single_prompt(
            8,
            {"prompt": "question", "image": "registry/image:tag"},
            3,
            _config(),
        )
    )
    while not any(event[0] == "run" for event in events):
        await asyncio.sleep(0)
    run.cancel()

    with pytest.raises(asyncio.CancelledError):
        await run

    assert events[-2:] == [("close",), ("clear", "task_8")]
    assert fake_agent.instances[0].closed is True


@pytest.mark.asyncio
async def test_row_repeated_cancellation_finishes_owned_agent_close(
    monkeypatch,
):
    run_started = asyncio.Event()
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    close_finished = asyncio.Event()

    class _ClosingAgent(_Agent):
        async def run_conversation(self, prompt, task_id=None):
            run_started.set()
            await asyncio.Event().wait()

        async def close(self):
            close_started.set()
            await release_close.wait()
            self.closed = True
            close_finished.set()

    monkeypatch.setattr(batch_runner, "AIAgent", _ClosingAgent)
    monkeypatch.setattr(
        batch_runner,
        "sample_toolsets_from_distribution",
        lambda _distribution: ["terminal"],
    )
    run = asyncio.create_task(
        batch_runner._process_single_prompt(
            9,
            {"prompt": "question"},
            3,
            _config(),
        )
    )
    await run_started.wait()
    run.cancel()
    await close_started.wait()
    run.cancel()
    await asyncio.sleep(0)

    assert run.done() is False
    release_close.set()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert close_finished.is_set()
    assert _ClosingAgent.instances[0].closed is True


@pytest.mark.asyncio
async def test_concurrent_runners_namespace_terminal_overrides_by_run_name(
    monkeypatch,
):
    from tools.terminal_tool import resolve_task_overrides

    both_running = asyncio.Event()
    entered = 0
    observed = {}

    class _ConcurrentAgent(_Agent):
        async def run_conversation(self, prompt, task_id=None):
            nonlocal entered
            entered += 1
            if entered == 2:
                both_running.set()
            await both_running.wait()
            observed[prompt] = (task_id, resolve_task_overrides(task_id))
            return await super().run_conversation(prompt, task_id=task_id)

    monkeypatch.setenv("TERMINAL_ENV", "modal")
    monkeypatch.setattr(batch_runner, "AIAgent", _ConcurrentAgent)
    monkeypatch.setattr(
        batch_runner,
        "sample_toolsets_from_distribution",
        lambda _distribution: ["terminal"],
    )
    first_config = _config() | {"run_name": "first-run"}
    second_config = _config() | {"run_name": "second-run"}

    first, second = await asyncio.gather(
        batch_runner._process_single_prompt(
            0,
            {"prompt": "first", "image": "registry/first:tag"},
            0,
            first_config,
        ),
        batch_runner._process_single_prompt(
            0,
            {"prompt": "second", "image": "registry/second:tag"},
            0,
            second_config,
        ),
    )

    assert first["success"] is True
    assert second["success"] is True
    assert observed == {
        "first": (
            "first-run:task_0",
            {
                "docker_image": "registry/first:tag",
                "modal_image": "registry/first:tag",
                "singularity_image": "docker://registry/first:tag",
                "daytona_image": "registry/first:tag",
            },
        ),
        "second": (
            "second-run:task_0",
            {
                "docker_image": "registry/second:tag",
                "modal_image": "registry/second:tag",
                "singularity_image": "docker://registry/second:tag",
                "daytona_image": "registry/second:tag",
            },
        ),
    }
