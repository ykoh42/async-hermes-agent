"""Real-process recovery for durable async-delegation delivery."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.asyncio


async def _run_script(source: str, *, cwd: Path, env: dict[str, str]) -> str:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        source,
        cwd=cwd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=20)
    assert process.returncode == 0, stderr.decode()
    return stdout.decode().strip().splitlines()[-1]


async def test_real_process_restart_restores_owned_completion_once(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    env = {**os.environ, "HERMES_HOME": str(tmp_path), "PYTHONPATH": str(repo)}
    producer = """
import asyncio
from tools import async_delegation as ad
async def main():
    async def runner():
        return {"status": "completed", "summary": "after restart"}
    result = await ad.dispatch_async_delegation(
        goal="restart", context=None, toolsets=None, role="leaf", model="m",
        session_key="owner-session", parent_session_id="durable-parent",
        runner=runner,
    )
    while ad.active_count():
        await asyncio.sleep(.01)
    print(result["delegation_id"])
asyncio.run(main())
"""
    delegation_id = await _run_script(producer, cwd=repo, env=env)

    consumer = """
import asyncio, json
from tools import async_delegation as ad
from tools.process_registry import process_registry
async def main():
    await ad.restore_undelivered_completions(process_registry.completion_queue)
    print(json.dumps(process_registry.completion_queue.get_nowait(), sort_keys=True))
asyncio.run(main())
"""
    event = json.loads(await _run_script(consumer, cwd=repo, env=env))
    assert event["delegation_id"] == delegation_id
    assert event["session_key"] == "owner-session"
    assert event["parent_session_id"] == "durable-parent"
    assert event["summary"] == "after restart"
    assert event["restored"] is True

    acker = f"""
import asyncio
from tools import async_delegation as ad
asyncio.run(ad.mark_completion_delivered({delegation_id!r}))
"""
    await _run_script(acker + "\nprint('acked')", cwd=repo, env=env)
    probe = """
import asyncio
from tools import async_delegation as ad
from tools.process_registry import process_registry
async def main():
    print(await ad.restore_undelivered_completions(process_registry.completion_queue))
asyncio.run(main())
"""
    assert await _run_script(probe, cwd=repo, env=env) == "0"
