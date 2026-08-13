"""Child-process protocol for hermetic cold-restart acceptance tests."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any
from collections.abc import Awaitable, Callable

import aiofiles
import aiofiles.os
import psutil
from aiohttp import web


MODEL = "cold-restart-model"
SESSION_ID = "cold-restart-session"
MEMORY_MARKER = "COLD_MEMORY_ORCHID_731"
SESSION_MARKER = "COLD_SESSION_COBALT_942"
BATCH_PROMPTS = ("COLD_BATCH_PROMPT_0", "COLD_BATCH_PROMPT_1")
RESULT_PREFIX = "COLD_RESTART_RESULT="


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _tool_chunks(name: str, arguments: dict[str, Any], call_id: str) -> list[dict]:
    return [
        {
            "id": call_id,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "reasoning_content": f"reasoning before {name}",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": call_id,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": MODEL,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        },
    ]


def _final_chunks(content: str) -> list[dict]:
    return [
        {
            "id": "cold-final",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "reasoning_content": "cold restart final reasoning",
                        "content": content,
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "cold-final",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": MODEL,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]


async def _stream(request: web.Request, chunks: list[dict]) -> web.StreamResponse:
    response = web.StreamResponse(
        status=200,
        headers={"Content-Type": "text/event-stream"},
    )
    await response.prepare(request)
    chunks.append({
        "id": chunks[-1]["id"],
        "object": "chat.completion.chunk",
        "created": 1,
        "model": MODEL,
        "choices": [],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 4,
            "total_tokens": 16,
        },
    })
    for chunk in chunks:
        await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
    await response.write(b"data: [DONE]\n\n")
    await response.write_eof()
    return response


class LocalProvider:
    def __init__(self, responder: Callable[[dict], Awaitable[list[dict]]]):
        self.responder = responder
        self.requests: list[dict] = []
        application = web.Application()
        application.router.add_post("/v1/chat/completions", self._handle)
        self.runner = web.AppRunner(application)
        self.base_url = ""

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        self.requests.append(payload)
        return await _stream(request, await self.responder(payload))

    async def __aenter__(self) -> LocalProvider:
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        sockets = getattr(site, "_server").sockets
        self.base_url = f"http://127.0.0.1:{sockets[0].getsockname()[1]}/v1"
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.runner.cleanup()


async def _read_json(path: Path) -> Any:
    async with aiofiles.open(path, encoding="utf-8") as source:
        return json.loads(await source.read())


async def _read_jsonl(path: Path) -> list[dict]:
    async with aiofiles.open(path, encoding="utf-8") as source:
        return [json.loads(line) for line in (await source.read()).splitlines()]


async def _write_runtime_config(root: Path) -> None:
    home = Path(os.environ["HERMES_HOME"])
    await aiofiles.os.makedirs(home, exist_ok=True)
    async with aiofiles.open(home / "config.yaml", "w", encoding="utf-8") as output:
        await output.write(
            "model:\n"
            "  provider: custom\n"
            f"  default: {MODEL}\n"
            "memory:\n"
            "  memory_enabled: true\n"
            "  user_profile_enabled: false\n"
            "compression:\n"
            "  enabled: false\n"
            "terminal:\n"
            "  backend: local\n"
            f"  cwd: {json.dumps(str(root))}\n"
            "  timeout: 120\n"
        )


def _payload_system(payload: dict) -> str:
    return next(
        (
            str(message.get("content") or "")
            for message in payload.get("messages", [])
            if message.get("role") == "system"
        ),
        "",
    )


async def _session_origin(root: Path) -> dict:
    from hermes_state import SessionDB
    from run_agent import AIAgent

    workspace = root / "workspace"
    await aiofiles.os.makedirs(workspace, exist_ok=True)
    async with aiofiles.open(
        workspace / "artifact.txt", "w", encoding="utf-8"
    ) as output:
        await output.write("seed-before-checkpoint")
    os.chdir(workspace)

    async def respond(payload: dict) -> list[dict]:
        tool_results = [
            message
            for message in payload.get("messages", [])
            if message.get("role") == "tool"
        ]
        if not tool_results:
            return _tool_chunks(
                "memory",
                {"target": "memory", "action": "add", "content": MEMORY_MARKER},
                "call-memory",
            )
        if len(tool_results) == 1:
            return _tool_chunks(
                "write_file",
                {
                    "path": str(workspace / "artifact.txt"),
                    "content": "changed-after-checkpoint",
                },
                "call-write",
            )
        return _final_chunks("COLD_SESSION_ORIGIN_SAVED")

    database = SessionDB(root / "state.db")
    async with LocalProvider(respond) as provider:
        async with AIAgent(
            provider="custom",
            api_key="integration-key",
            base_url=provider.base_url,
            model=MODEL,
            max_iterations=5,
            enabled_toolsets=["memory", "file"],
            quiet_mode=True,
            skip_context_files=True,
            save_trajectories=True,
            session_db=database,
            session_id=SESSION_ID,
            checkpoints_enabled=True,
        ) as agent:
            agent.compression_enabled = False
            result = await agent.run_conversation(
                f"Persist {MEMORY_MARKER}; keep this session marker {SESSION_MARKER}."
            )
            system_prompt = agent._cached_system_prompt or ""
            checkpoints = await agent._checkpoint_mgr.list_checkpoints(str(workspace))
            messages = await database.get_messages(SESSION_ID)
    await database.close()

    memory_path = Path(os.environ["HERMES_HOME"]) / "memories" / "MEMORY.md"
    async with aiofiles.open(memory_path, encoding="utf-8") as source:
        memory_text = await source.read()
    async with aiofiles.open(workspace / "artifact.txt", encoding="utf-8") as source:
        artifact = await source.read()
    trajectories = await _read_jsonl(workspace / "trajectory_samples.jsonl")
    return {
        "pid": os.getpid(),
        "completed": result["completed"],
        "final": result["final_response"],
        "session_roles": [message["role"] for message in messages],
        "memory_text": memory_text,
        "artifact": artifact,
        "system_hash": _sha256(system_prompt),
        "checkpoint_hashes": [item["hash"] for item in checkpoints],
        "trajectory_roles": [turn["from"] for turn in trajectories[0]["conversations"]],
        "trajectory_count": len(trajectories),
        "provider_request_count": len(provider.requests),
    }


async def _session_resume(root: Path) -> dict:
    from hermes_state import SessionDB
    from run_agent import AIAgent

    workspace = root / "workspace"
    os.chdir(workspace)
    database = SessionDB(root / "state.db")
    tip = await database.resolve_resume_session_id(SESSION_ID)
    model_history, _display_history = await database.get_resume_conversations(tip)
    stored_session = await database.get_session(tip)
    stored_prompt = str(stored_session.get("system_prompt") or "")

    async def respond(payload: dict) -> list[dict]:
        last_user = next(
            (
                str(message.get("content") or "")
                for message in reversed(payload.get("messages", []))
                if message.get("role") == "user"
            ),
            "",
        )
        if last_user == "FRESH_MEMORY_PROBE":
            return _final_chunks(MEMORY_MARKER)
        return _final_chunks(SESSION_MARKER)

    async with LocalProvider(respond) as provider:
        async with AIAgent(
            provider="custom",
            api_key="integration-key",
            base_url=provider.base_url,
            model=MODEL,
            max_iterations=2,
            disabled_toolsets=["*"],
            quiet_mode=True,
            skip_context_files=True,
            save_trajectories=True,
            session_db=database,
            session_id=tip,
            checkpoints_enabled=True,
        ) as resumed_agent:
            resumed_agent.compression_enabled = False
            resumed = await resumed_agent.run_conversation(
                "Return the prior session marker.",
                conversation_history=model_history,
            )
            resumed_prompt = resumed_agent._cached_system_prompt or ""
            checkpoints = await resumed_agent._checkpoint_mgr.list_checkpoints(
                str(workspace)
            )
            messages = await database.get_messages(tip)

        memory_database = SessionDB(root / "memory-probe.db")
        async with AIAgent(
            provider="custom",
            api_key="integration-key",
            base_url=provider.base_url,
            model=MODEL,
            max_iterations=2,
            disabled_toolsets=["*"],
            quiet_mode=True,
            skip_context_files=True,
            session_db=memory_database,
            session_id="cold-memory-probe",
        ) as memory_agent:
            memory_agent.compression_enabled = False
            memory_result = await memory_agent.run_conversation("FRESH_MEMORY_PROBE")
            fresh_prompt = memory_agent._cached_system_prompt or ""
        await memory_database.close()
    await database.close()

    trajectories = await _read_jsonl(workspace / "trajectory_samples.jsonl")
    resume_request, fresh_request = provider.requests
    return {
        "pid": os.getpid(),
        "completed": resumed["completed"],
        "final": resumed["final_response"],
        "memory_final": memory_result["final_response"],
        "session_roles": [message["role"] for message in messages],
        "stored_system_hash": _sha256(stored_prompt),
        "resumed_system_hash": _sha256(resumed_prompt),
        "resume_request_system_hash": _sha256(_payload_system(resume_request)),
        "resume_request_has_session_marker": SESSION_MARKER
        in json.dumps(resume_request),
        "fresh_system_has_memory": MEMORY_MARKER in fresh_prompt,
        "fresh_request_has_memory": MEMORY_MARKER in _payload_system(fresh_request),
        "checkpoint_hashes": [item["hash"] for item in checkpoints],
        "trajectory_roles": [
            [turn["from"] for turn in row["conversations"]] for row in trajectories
        ],
        "trajectory_count": len(trajectories),
    }


async def _wait_for_checkpoint(path: Path, completed: int) -> None:
    for _attempt in range(500):
        try:
            checkpoint = await _read_json(path)
        except (FileNotFoundError, json.JSONDecodeError):
            checkpoint = {}
        if completed in checkpoint.get("completed_prompts", []):
            return
        await asyncio.sleep(0.01)
    raise TimeoutError(f"checkpoint never recorded prompt {completed}")


def _terminal_command(*, block: bool, marker: Path | None = None) -> str:
    if block:
        if marker is None:
            raise ValueError("blocking terminal command requires a marker path")
        source = (
            "from pathlib import Path; import os, time; "
            f"Path({str(marker)!r}).write_text(str(os.getpid()), encoding='utf-8'); "
            "time.sleep(120)"
        )
    else:
        source = "print('COLD_RESUME_OBSERVATION')"
    return shlex.join([sys.executable, "-c", source])


async def _batch_resume(root: Path) -> tuple[dict, int]:
    from batch_runner import BatchRunner

    os.chdir(root)

    async def respond(payload: dict) -> list[dict]:
        messages = payload.get("messages", [])
        prompt = next(
            (
                str(message.get("content") or "")
                for message in messages
                if message.get("role") == "user"
            ),
            "",
        )
        has_tool_result = any(message.get("role") == "tool" for message in messages)
        if prompt == BATCH_PROMPTS[0]:
            return _final_chunks("COLD_BATCH_FINAL_0")
        if not has_tool_result:
            return _tool_chunks(
                "terminal",
                {
                    "command": _terminal_command(block=False),
                    "timeout": 120,
                    "force": True,
                },
                "call-batch-terminal",
            )
        return _final_chunks("COLD_BATCH_FINAL_1")

    dataset = root / "dataset.jsonl"
    runner_args = {
        "dataset_file": str(dataset),
        "batch_size": 1,
        "run_name": "cold-batch",
        "distribution": "terminal_only",
        "model": MODEL,
        "base_url": "",
        "api_key": "integration-key",
        "max_iterations": 4,
        "num_workers": 1,
        "verbose": False,
        "reasoning_config": {"enabled": True, "effort": "low"},
    }

    async with LocalProvider(respond) as provider:
        runner_args["base_url"] = provider.base_url
        runner = BatchRunner(**runner_args)
        await runner.run(resume=True)

    output_dir = root / "data" / "cold-batch"
    checkpoint = await _read_json(output_dir / "checkpoint.json")
    shard_rows = await _read_jsonl(output_dir / "batch_0.jsonl")
    merged_rows = await _read_jsonl(output_dir / "trajectories.jsonl")
    report = {
        "pid": os.getpid(),
        "checkpoint": checkpoint["completed_prompts"],
        "shard_rows": shard_rows,
        "merged_rows": merged_rows,
        "request_prompts": [
            next(
                (
                    str(message.get("content") or "")
                    for message in request.get("messages", [])
                    if message.get("role") == "user"
                ),
                "",
            )
            for request in provider.requests
        ],
    }
    return report, 0


async def _batch_interrupt(root: Path) -> tuple[dict, int]:
    """Run until the parent requests cancellation while terminal is active."""
    from batch_runner import BatchRunner

    os.chdir(root)
    checkpoint_path = root / "data" / "cold-batch" / "checkpoint.json"
    active_marker = root / "active-tool.pid"
    cancel_request = root / "cancel.request"

    async def respond(payload: dict) -> list[dict]:
        prompt = next(
            (
                str(message.get("content") or "")
                for message in payload.get("messages", [])
                if message.get("role") == "user"
            ),
            "",
        )
        if prompt == BATCH_PROMPTS[0]:
            return _final_chunks("COLD_BATCH_FINAL_0")
        await _wait_for_checkpoint(checkpoint_path, 0)
        return _tool_chunks(
            "terminal",
            {
                "command": _terminal_command(block=True, marker=active_marker),
                "timeout": 120,
                "force": True,
            },
            "call-batch-terminal",
        )

    async with LocalProvider(respond) as provider:
        runner = BatchRunner(
            dataset_file=str(root / "dataset.jsonl"),
            batch_size=1,
            run_name="cold-batch",
            distribution="terminal_only",
            model=MODEL,
            base_url=provider.base_url,
            api_key="integration-key",
            max_iterations=4,
            num_workers=1,
            verbose=False,
            reasoning_config={"enabled": True, "effort": "low"},
        )
        run_task = asyncio.create_task(runner.run(), name="interrupted-batch")
        while not await aiofiles.os.path.exists(cancel_request):
            if run_task.done():
                await run_task
                raise RuntimeError("batch ended before cancellation request")
            await asyncio.sleep(0.01)
        run_task.cancel()
        cancellation_propagated = False
        try:
            await run_task
        except asyncio.CancelledError:
            cancellation_propagated = True
        if not cancellation_propagated:
            raise RuntimeError("BatchRunner.run() swallowed external cancellation")

        async with aiofiles.open(active_marker, encoding="utf-8") as source:
            child_pid = int((await source.read()).strip())
        for _attempt in range(500):
            if not psutil.pid_exists(child_pid):
                break
            await asyncio.sleep(0.01)
        child_reaped = not psutil.pid_exists(child_pid)

    output_dir = root / "data" / "cold-batch"
    checkpoint = await _read_json(output_dir / "checkpoint.json")
    shard_rows = await _read_jsonl(output_dir / "batch_0.jsonl")
    return {
        "pid": os.getpid(),
        "cancelled": cancellation_propagated,
        "child_pid": child_pid,
        "child_reaped": child_reaped,
        "checkpoint": checkpoint["completed_prompts"],
        "shard_rows": shard_rows,
        "request_count": len(provider.requests),
    }, 75


async def _main(args: argparse.Namespace) -> tuple[dict, int]:
    root = Path(args.root).resolve()
    await _write_runtime_config(root)
    if args.action == "session-origin":
        return await _session_origin(root), 0
    if args.action == "session-resume":
        return await _session_resume(root), 0
    if args.action == "batch-interrupt":
        return await _batch_interrupt(root)
    return await _batch_resume(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("session-origin", "session-resume", "batch-interrupt", "batch-resume"),
    )
    parser.add_argument("root")
    parsed = parser.parse_args()
    payload, exit_code = asyncio.run(_main(parsed))
    print(f"{RESULT_PREFIX}{json.dumps(payload, sort_keys=True)}", flush=True)
    raise SystemExit(exit_code)
