"""Hermetic source-checkout CLI coverage with a loopback OpenAI stream."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiofiles
from aiohttp import web
import pytest


pytestmark = pytest.mark.integration


def _chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
    return {
        "id": "batch-cli-e2e",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "batch-cli-model",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


@asynccontextmanager
async def _loopback_provider() -> AsyncIterator[tuple[str, list[dict[str, Any]]]]:
    requests: list[dict[str, Any]] = []

    async def completions(request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        requests.append(payload)
        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream"},
        )
        await response.prepare(request)
        chunks = [
            _chunk(
                {
                    "role": "assistant",
                    "reasoning_content": "loopback reasoning",
                }
            ),
            _chunk({"content": "BATCH_CLI_COMPLETE"}),
            _chunk({}, finish_reason="stop"),
        ]
        for chunk in chunks:
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    application = web.Application()
    application.router.add_post("/v1/chat/completions", completions)
    server = web.AppRunner(application)
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", 0)
    await site.start()
    sockets = getattr(site, "_server").sockets
    port = sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/v1", requests
    finally:
        await server.cleanup()


async def _run_cli(
    root: Path,
    dataset: Path,
    output_root: Path,
    base_url: str,
    *,
    resume: bool,
) -> tuple[int, str, str]:
    env = os.environ.copy()
    env.update(
        {
            "HERMES_HOME": str(output_root / "hermes-home"),
            "PYTHONUNBUFFERED": "1",
            "HERMES_LIVE_TESTS": "0",
            "OPENAI_API_KEY": "",
            "OPENROUTER_API_KEY": "",
        }
    )
    output_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(root / "batch_runner.py"),
        f"--dataset_file={dataset}",
        "--batch_size=1",
        "--run_name=cli-e2e",
        "--distribution=terminal_only",
        "--model=batch-cli-model",
        "--api_key=loopback-key",
        f"--base_url={base_url}",
        "--max_turns=1",
        "--num_workers=1",
        "--reasoning_effort=low",
    ]
    if resume:
        command.append("--resume")
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=output_root,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return process.returncode, stdout.decode(), stderr.decode()


@pytest.mark.asyncio
async def test_source_cli_runs_loopback_batch_and_resumes(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    dataset = tmp_path / "prompts.jsonl"
    output_root = tmp_path / "runtime"
    dataset.write_text(json.dumps({"prompt": "first CLI prompt"}) + "\n")

    async with _loopback_provider() as (base_url, requests):
        first = await _run_cli(root, dataset, output_root, base_url, resume=False)
        assert first[0] == 0, first[1] + first[2]

        async with aiofiles.open(dataset, "a", encoding="utf-8") as handle:
            await handle.write(json.dumps({"prompt": "second CLI prompt"}) + "\n")

        second = await _run_cli(root, dataset, output_root, base_url, resume=True)
        assert second[0] == 0, second[1] + second[2]

    output_dir = output_root / "data" / "cli-e2e"
    async with aiofiles.open(output_dir / "checkpoint.json", encoding="utf-8") as handle:
        checkpoint = json.loads(await handle.read())
    async with aiofiles.open(output_dir / "batch_0.jsonl", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in (await handle.read()).splitlines()]

    assert checkpoint["completed_prompts"] == [0, 1]
    assert [row["prompt_index"] for row in rows] == [0, 1]
    assert all(
        row["conversations"][-1]["value"].endswith("BATCH_CLI_COMPLETE")
        for row in rows
    )
    assert len(requests) == 2
