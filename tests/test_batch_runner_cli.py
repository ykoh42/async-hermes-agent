"""Upstream-compatible CLI and async-host coverage for ``batch_runner.py``."""

from __future__ import annotations

import asyncio
import json
import inspect
from pathlib import Path
import sys
from typing import Any

import pytest

import batch_runner


_ROOT = Path(__file__).parents[1]
_UPSTREAM_MAIN_PARAMETERS = (
    ("dataset_file", None),
    ("batch_size", None),
    ("run_name", None),
    ("distribution", "default"),
    ("model", "anthropic/claude-sonnet-4.6"),
    ("api_key", None),
    ("base_url", "https://openrouter.ai/api/v1"),
    ("max_turns", 10),
    ("num_workers", 4),
    ("resume", False),
    ("verbose", False),
    ("list_distributions", False),
    ("ephemeral_system_prompt", None),
    ("log_prefix_chars", 100),
    ("providers_allowed", None),
    ("providers_ignored", None),
    ("providers_order", None),
    ("provider_sort", None),
    ("max_tokens", None),
    ("reasoning_effort", None),
    ("reasoning_disabled", False),
    ("prefill_messages_file", None),
    ("max_samples", None),
)


async def _run_python(*arguments: str) -> tuple[int, str]:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        *arguments,
        cwd=_ROOT,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    output, _ = await asyncio.wait_for(process.communicate(), timeout=30)
    return process.returncode or 0, output.decode("utf-8", errors="replace")


def test_batch_main_preserves_upstream_signature_and_defaults() -> None:
    parameters = inspect.signature(batch_runner.main).parameters
    assert tuple((name, parameter.default) for name, parameter in parameters.items()) == (
        _UPSTREAM_MAIN_PARAMETERS
    )
    assert inspect.iscoroutinefunction(batch_runner.main)


def test_private_cli_boundary_preserves_main_metadata() -> None:
    assert not inspect.iscoroutinefunction(batch_runner._run_cli)
    assert inspect.signature(batch_runner._run_cli) == inspect.signature(
        batch_runner.main
    )
    assert batch_runner._run_cli.__wrapped__ is batch_runner.main


@pytest.mark.asyncio
async def test_fire_cli_maps_upstream_flags_to_async_main() -> None:
    # Run Fire in a fresh process. This verifies the real process boundary and
    # prevents unrelated tests from sharing Fire's parser/introspection state.
    code = """
import asyncio
import json
import inspect
import batch_runner
import fire

original_main = batch_runner.main
received = {}

async def fake_main(*args, **kwargs):
    bound = inspect.signature(original_main).bind(*args, **kwargs)
    bound.apply_defaults()
    received.update(bound.arguments)

batch_runner.main = fake_main
batch_runner._run_cli.__wrapped__ = original_main
fire.Fire(batch_runner._run_cli, command=[
    "--dataset_file=data.jsonl",
    "--batch_size=3",
    "--run_name=web-compatible",
    "--resume",
    "--verbose=False",
    "--reasoning_disabled",
    "--providers_allowed=anthropic,openai",
    "--max_tokens=2048",
])
print(json.dumps(received))
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        code,
        cwd=_ROOT,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    output, _ = await asyncio.wait_for(process.communicate(), timeout=30)
    assert process.returncode == 0, output.decode()
    received = json.loads(output.decode().splitlines()[-1])

    assert received["dataset_file"] == "data.jsonl"
    assert received["batch_size"] == 3
    assert received["run_name"] == "web-compatible"
    assert received["resume"] is True
    assert received["verbose"] is False
    assert received["reasoning_disabled"] is True
    assert received["providers_allowed"] == ["anthropic", "openai"]
    assert received["max_tokens"] == 2048
    assert received["distribution"] == "default"
    assert received["num_workers"] == 4


@pytest.mark.asyncio
async def test_source_script_exposes_upstream_fire_help() -> None:
    returncode, output = await _run_python("batch_runner.py", "--help")
    assert returncode == 0
    for name, _default in _UPSTREAM_MAIN_PARAMETERS:
        flag = f"--{name}"
        assert flag in output


@pytest.mark.asyncio
async def test_installed_module_style_lists_distributions() -> None:
    returncode, output = await _run_python(
        "-m", "batch_runner", "--list_distributions"
    )
    assert returncode == 0
    assert "Available Toolset Distributions" in output
    assert "default" in output


@pytest.mark.asyncio
async def test_async_main_runs_inside_host_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host_loop = asyncio.get_running_loop()
    received: dict[str, Any] = {}

    class FakeBatchRunner:
        def __init__(self, **kwargs: Any) -> None:
            received["init"] = kwargs

        async def run(self, resume: bool = False) -> None:
            await asyncio.sleep(0)
            received["loop"] = asyncio.get_running_loop()
            received["resume"] = resume

    monkeypatch.setattr(batch_runner, "BatchRunner", FakeBatchRunner)
    result = await batch_runner.main(
        dataset_file="data.jsonl",
        batch_size=2,
        run_name="host-loop",
        model="loopback-model",
        resume=True,
        providers_allowed=("anthropic", "openai"),  # type: ignore[arg-type]
    )

    assert result is None
    assert received["loop"] is host_loop
    assert received["resume"] is True
    assert received["init"]["run_name"] == "host-loop"
    assert received["init"]["num_workers"] == 4
    assert received["init"]["providers_allowed"] == ["anthropic", "openai"]
