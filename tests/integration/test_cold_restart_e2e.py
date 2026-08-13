"""Hermetic acceptance coverage for recovery in a fresh OS process."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

import aiofiles
import aiofiles.os
import pytest

from tests.integration._cold_restart_worker import (
    BATCH_PROMPTS as _BATCH_PROMPTS,
    MEMORY_MARKER as _MEMORY_MARKER,
    RESULT_PREFIX as _RESULT_PREFIX,
    SESSION_MARKER as _SESSION_MARKER,
)


pytestmark = pytest.mark.integration

_REPO = Path(__file__).resolve().parents[2]
_WORKER = Path(__file__).with_name("_cold_restart_worker.py")


async def _prepare_root(root: Path) -> None:
    for directory in (root / "os-home", root / "hermes-home", root / "tmp"):
        await aiofiles.os.makedirs(directory, exist_ok=True)


def _worker_env(root: Path) -> dict[str, str]:
    """Build a credential-free environment for the acceptance worker."""
    env = {
        "PATH": os.environ.get("PATH", os.defpath),
        "PYTHONPATH": str(_REPO),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "PYTEST_CURRENT_TEST": "cold-restart-worker",
        "HOME": str(root / "os-home"),
        "HERMES_HOME": str(root / "hermes-home"),
        "TMPDIR": str(root / "tmp"),
        "TEMP": str(root / "tmp"),
        "TMP": str(root / "tmp"),
        "NO_PROXY": "127.0.0.1,localhost",
        "NO_COLOR": "1",
        "TERM": "dumb",
    }
    for name in ("COMSPEC", "PATHEXT", "SYSTEMROOT", "WINDIR"):
        if value := os.environ.get(name):
            env[name] = value
    return env


async def _start_worker(action: str, root: Path) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        sys.executable,
        str(_WORKER),
        action,
        str(root),
        cwd=root,
        env=_worker_env(root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


async def _finish_worker(
    process: asyncio.subprocess.Process,
    *,
    expected_code: int,
    timeout: float = 60,
) -> dict:
    communicate_task = asyncio.create_task(process.communicate())
    try:
        stdout, stderr = await asyncio.wait_for(
            asyncio.shield(communicate_task), timeout=timeout
        )
    except TimeoutError:
        if process.returncode is None:
            process.kill()
        stdout, stderr = await communicate_task
        pytest.fail(
            f"worker timed out after {timeout}s\n"
            f"stdout:\n{stdout.decode('utf-8', errors='replace')[-8000:]}\n"
            f"stderr:\n{stderr.decode('utf-8', errors='replace')[-8000:]}"
        )
    decoded_stdout = stdout.decode("utf-8", errors="replace")
    decoded_stderr = stderr.decode("utf-8", errors="replace")
    assert process.returncode == expected_code, (
        f"worker exited {process.returncode}\n"
        f"stdout:\n{decoded_stdout[-8000:]}\n"
        f"stderr:\n{decoded_stderr[-8000:]}"
    )
    result_lines = [
        line.removeprefix(_RESULT_PREFIX)
        for line in decoded_stdout.splitlines()
        if line.startswith(_RESULT_PREFIX)
    ]
    assert len(result_lines) == 1, decoded_stdout[-8000:]
    return json.loads(result_lines[0])


async def _run_worker(action: str, root: Path, *, expected_code: int = 0) -> dict:
    return await _finish_worker(
        await _start_worker(action, root),
        expected_code=expected_code,
    )


async def _wait_for_path(
    path: Path,
    process: asyncio.subprocess.Process,
    *,
    timeout: float = 30,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not await aiofiles.os.path.exists(path):
        if process.returncode is not None:
            stdout, stderr = await process.communicate()
            pytest.fail(
                f"worker exited {process.returncode} before creating {path.name}\n"
                f"stdout:\n{stdout.decode('utf-8', errors='replace')[-8000:]}\n"
                f"stderr:\n{stderr.decode('utf-8', errors='replace')[-8000:]}"
            )
        if asyncio.get_running_loop().time() >= deadline:
            if process.returncode is None:
                process.kill()
            stdout, stderr = await process.communicate()
            pytest.fail(
                f"worker did not create {path.name} within {timeout}s\n"
                f"stdout:\n{stdout.decode('utf-8', errors='replace')[-8000:]}\n"
                f"stderr:\n{stderr.decode('utf-8', errors='replace')[-8000:]}"
            )
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("git") is None, reason="checkpoints require git")
async def test_session_memory_checkpoint_and_trajectory_survive_cold_restart(
    tmp_path: Path,
) -> None:
    root = tmp_path / "session-cold-restart"
    await _prepare_root(root)

    origin = await _run_worker("session-origin", root)
    resumed = await _run_worker("session-resume", root)

    assert origin["pid"] != os.getpid()
    assert resumed["pid"] != os.getpid()
    assert origin["completed"] is True
    assert origin["final"] == "COLD_SESSION_ORIGIN_SAVED"
    assert origin["session_roles"] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert _MEMORY_MARKER in origin["memory_text"]
    assert origin["artifact"] == "changed-after-checkpoint"
    assert origin["checkpoint_hashes"]
    assert origin["trajectory_count"] == 1
    assert origin["trajectory_roles"] == [
        "system",
        "human",
        "gpt",
        "tool",
        "gpt",
        "tool",
        "gpt",
    ]
    assert origin["provider_request_count"] == 3

    assert resumed["completed"] is True
    assert resumed["final"] == _SESSION_MARKER
    assert resumed["memory_final"] == _MEMORY_MARKER
    assert resumed["session_roles"] == [
        *origin["session_roles"],
        "user",
        "assistant",
    ]
    assert resumed["stored_system_hash"] == origin["system_hash"]
    assert resumed["resumed_system_hash"] == origin["system_hash"]
    assert resumed["resume_request_system_hash"] == origin["system_hash"]
    assert resumed["resume_request_has_session_marker"] is True
    assert resumed["fresh_system_has_memory"] is True
    assert resumed["fresh_request_has_memory"] is True
    assert resumed["checkpoint_hashes"] == origin["checkpoint_hashes"]
    assert resumed["trajectory_count"] == 2
    assert resumed["trajectory_roles"] == [
        origin["trajectory_roles"],
        [*origin["trajectory_roles"], "human", "gpt"],
    ]


@pytest.mark.asyncio
async def test_batch_cancellation_then_cold_resume_is_exactly_once(
    tmp_path: Path,
) -> None:
    root = tmp_path / "batch-cold-restart"
    await _prepare_root(root)
    async with aiofiles.open(root / "dataset.jsonl", "w", encoding="utf-8") as output:
        for prompt in _BATCH_PROMPTS:
            await output.write(json.dumps({"prompt": prompt}) + "\n")

    interrupted_process = await _start_worker("batch-interrupt", root)
    await _wait_for_path(root / "active-tool.pid", interrupted_process)
    async with aiofiles.open(root / "cancel.request", "w", encoding="utf-8") as output:
        await output.write("cancel active terminal tool")
        await output.flush()
    interrupted = await _finish_worker(interrupted_process, expected_code=75)

    assert interrupted["cancelled"] is True
    assert interrupted["child_pid"] != interrupted["pid"]
    assert interrupted["child_reaped"] is True
    assert interrupted["checkpoint"] == [0]
    assert [row["prompt_index"] for row in interrupted["shard_rows"]] == [0]
    assert interrupted["request_count"] == 2

    resumed = await _run_worker("batch-resume", root)
    assert interrupted["pid"] != os.getpid()
    assert resumed["pid"] != os.getpid()
    assert resumed["checkpoint"] == [0, 1]
    assert set(resumed["request_prompts"]) == {_BATCH_PROMPTS[1]}
    assert len(resumed["request_prompts"]) == 2

    shard_rows = resumed["shard_rows"]
    assert [row["prompt_index"] for row in shard_rows] == [0, 1]
    assert len({row["prompt_index"] for row in shard_rows}) == len(shard_rows)
    assert shard_rows[0] == interrupted["shard_rows"][0]
    assert resumed["merged_rows"] == shard_rows
    assert [turn["from"] for turn in shard_rows[0]["conversations"]] == [
        "system",
        "human",
        "gpt",
    ]
    assert [turn["from"] for turn in shard_rows[1]["conversations"]] == [
        "system",
        "human",
        "gpt",
        "tool",
        "gpt",
    ]
    assert shard_rows[1]["tool_stats"]["terminal"] == {
        "count": 1,
        "success": 1,
        "failure": 0,
    }
    assert shard_rows[1]["conversations"][-1]["value"].endswith("COLD_BATCH_FINAL_1")
