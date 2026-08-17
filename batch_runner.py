#!/usr/bin/env python3
"""
Batch Agent Runner

This module provides parallel batch processing capabilities for running the agent
across multiple prompts from a dataset. It includes:
- Dataset loading and batching
- Bounded parallel batch processing with asyncio tasks
- Checkpointing for fault tolerance and resumption
- Trajectory saving in the proper format (from/value pairs)
- Tool usage statistics aggregation across all batches

Usage:
    # From a source checkout (upstream-compatible CLI):
    python batch_runner.py \
        --dataset_file=data/prompts.jsonl \
        --batch_size=10 \
        --run_name=my_run \
        --model=anthropic/claude-sonnet-4.6

    # Or from an installed package:
    python -m batch_runner --list_distributions

    # Programmatic async use from an existing event loop:
    runner = BatchRunner(
        dataset_file="data.jsonl",
        batch_size=10,
        run_name="my_run",
    )
    await runner.run()

    # Resume an interrupted run.
    await runner.run(resume=True)
"""

# IMPORTANT: hermes_bootstrap must be the very first import — UTF-8 stdio
# on Windows.  No-op on POSIX.  See hermes_bootstrap.py for full rationale.
try:
    import hermes_bootstrap  # noqa: F401
except ModuleNotFoundError:
    # Graceful fallback when hermes_bootstrap isn't registered in the venv
    # yet — happens during partial ``hermes update`` where git-reset landed
    # new code but ``uv pip install -e .`` didn't finish.  Missing bootstrap
    # means UTF-8 stdio setup is skipped on Windows; POSIX is unaffected.
    pass

import json
import asyncio
import aiofiles
import aiofiles.os
import aiofiles.tempfile
import errno
from functools import wraps
import logging
import os
import stat
import subprocess
import time
from pathlib import Path
from typing import Any
from datetime import datetime
import traceback
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console

logger = logging.getLogger(__name__)
import fire

from run_agent import AIAgent
from toolset_distributions import (
    list_distributions, 
    sample_toolsets_from_distribution,
    validate_distribution
)
from model_tools import TOOL_TO_TOOLSET_MAP, get_all_tool_names


# Global configuration for worker processes
_WORKER_CONFIG = {}

# All possible tools - auto-derived from the master mapping in model_tools.py.
# This stays in sync automatically when new tools are added to TOOL_TO_TOOLSET_MAP.
# Used for consistent schema in Arrow/Parquet (HuggingFace datasets) and for
# filtering corrupted entries during trajectory combination.
ALL_POSSIBLE_TOOLS = set(TOOL_TO_TOOLSET_MAP.keys())

# Default stats for tools that weren't used
DEFAULT_TOOL_STATS = {'count': 0, 'success': 0, 'failure': 0}


async def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish a JSON checkpoint without blocking batch workers."""
    await aiofiles.os.makedirs(path.parent, exist_ok=True)
    target = str(path)
    if await aiofiles.os.path.islink(target):
        target = await aiofiles.os.wrap(os.path.realpath)(target)

    original_mode: int | None = None
    original_owner: tuple[int, int] | None = None
    try:
        target_stat = await aiofiles.os.stat(target)
        original_mode = stat.S_IMODE(target_stat.st_mode)
        if os.name == "posix":
            original_owner = (target_stat.st_uid, target_stat.st_gid)
    except OSError:
        pass

    serialized = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    temporary = ""
    try:
        async with aiofiles.tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}_",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = output.name
            await output.write(serialized)
            await output.flush()
            await aiofiles.os.wrap(os.fsync)(output.fileno())
        try:
            await aiofiles.os.replace(temporary, target)
        except OSError as exc:
            if exc.errno not in (errno.EXDEV, errno.EBUSY):
                raise
            temporary_stat = await aiofiles.os.stat(temporary)
            async with (
                aiofiles.open(temporary, "rb") as source,
                aiofiles.open(target, "wb") as destination,
            ):
                while chunk := await source.read(1024 * 1024):
                    await destination.write(chunk)
                await destination.flush()
                await aiofiles.os.wrap(os.fsync)(destination.fileno())
            try:
                await aiofiles.os.wrap(os.chmod)(
                    target, stat.S_IMODE(temporary_stat.st_mode)
                )
                await aiofiles.os.wrap(os.utime)(
                    target,
                    ns=(temporary_stat.st_atime_ns, temporary_stat.st_mtime_ns),
                )
            except OSError:
                pass
            await aiofiles.os.remove(temporary)
        temporary = ""

        if original_owner is not None and hasattr(os, "chown"):
            try:
                await aiofiles.os.wrap(os.chown)(
                    target,
                    original_owner[0],
                    original_owner[1],
                )
            except OSError:
                pass
        if original_mode is not None:
            try:
                await aiofiles.os.wrap(os.chmod)(target, original_mode)
            except OSError:
                pass
    except BaseException:
        if temporary:
            try:
                await aiofiles.os.remove(temporary)
            except OSError:
                pass
        raise


async def _append_jsonl_line(path: Path, payload: dict[str, Any]) -> None:
    """Append one complete trajectory row before propagating cancellation.

    A batch shard has exactly one owning coroutine, so rows cannot interleave.
    Shielding the short open/write/flush transaction means a cancellation does
    not leave that owner midway through a JSON object.  The caller still sees
    ``CancelledError`` as soon as the durable row is complete, and resume can
    reliably discover it from the shard.
    """
    line = json.dumps(payload, ensure_ascii=False) + "\n"

    async def _append() -> None:
        async with aiofiles.open(path, "a", encoding="utf-8") as output:
            await output.write(line)
            await output.flush()
            await aiofiles.os.wrap(os.fsync)(output.fileno())

    write_task = asyncio.create_task(_append())
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            await asyncio.shield(write_task)
            break
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised below
            # Complete the already-started row before the worker is cancelled.
            # Keep waiting through repeated caller cancellation; otherwise the
            # owned write task can outlive BatchRunner.run().  A cancellation
            # of the write task itself is different and must still propagate.
            if write_task.cancelled():
                raise
            if cancellation is None:
                cancellation = exc
    if cancellation is not None:
        raise cancellation


async def _await_owned_batch_task(task: asyncio.Task[Any]) -> Any:
    """Finish one owned batch subprocess task through repeated cancellation."""
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised below
            if task.cancelled():
                raise
            if cancellation is None:
                cancellation = exc
        except Exception as exc:
            if cancellation is not None:
                raise cancellation from exc
            raise
    if cancellation is not None:
        raise cancellation
    return result


async def _finish_batch_subprocess(
    process: asyncio.subprocess.Process,
    communicate_task: asyncio.Task[tuple[bytes | None, bytes | None]],
) -> tuple[bytes | None, bytes | None]:
    """Kill, drain, and reap one Docker probe owned by a batch row."""
    async def _cleanup() -> tuple[bytes | None, bytes | None]:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        try:
            return await communicate_task
        except BaseException:
            await process.wait()
            raise

    return await _await_owned_batch_task(
        asyncio.create_task(_cleanup(), name="batch-docker-probe-cleanup")
    )


async def _run_docker_image_command(
    argv: list[Any],
    *,
    timeout: int,
    text: bool = False,
) -> subprocess.CompletedProcess:
    """Run an upstream Docker image probe with native asyncio subprocess I/O."""
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    communicate_task = asyncio.create_task(
        process.communicate(),
        name="batch-docker-probe-communicate",
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            asyncio.shield(communicate_task),
            timeout=timeout,
        )
    except TimeoutError as exc:
        stdout, stderr = await _finish_batch_subprocess(  # noqa: ASYNC120 - cancellation wins
            process, communicate_task
        )
        raise subprocess.TimeoutExpired(
            argv,
            timeout,
            output=stdout,
            stderr=stderr,
        ) from exc
    except asyncio.CancelledError as cancellation:
        cleanup = asyncio.create_task(
            _finish_batch_subprocess(process, communicate_task),
            name="batch-docker-probe-cancel-cleanup",
        )
        try:
            await _await_owned_batch_task(cleanup)
        except asyncio.CancelledError:  # noqa: ASYNC103 - original re-raised
            pass
        raise cancellation
    except BaseException:
        await _finish_batch_subprocess(  # noqa: ASYNC120 - cancellation wins
            process, communicate_task
        )
        raise

    if text:
        stdout = (stdout or b"").decode("utf-8", errors="replace")
        stderr = (stderr or b"").decode("utf-8", errors="replace")
    return subprocess.CompletedProcess(
        argv,
        int(process.returncode or 0),
        stdout,
        stderr,
    )


async def _list_batch_files(directory: Path) -> list[Path]:
    """List batch JSONL shards without a synchronous directory glob."""
    try:
        names = await aiofiles.os.listdir(directory)
    except OSError:
        return []
    paths: list[Path] = []
    for name in names:
        path = directory / name
        if (
            name.startswith("batch_")
            and name.endswith(".jsonl")
            and await aiofiles.os.path.isfile(path)
        ):
            paths.append(path)
    return sorted(paths)


def _normalize_tool_stats(tool_stats: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    """
    Normalize tool_stats to include all possible tools with consistent schema.
    
    This ensures HuggingFace datasets can load the JSONL without schema mismatch errors.
    Tools that weren't used get zero counts.
    
    Args:
        tool_stats (Dict): Raw tool statistics from extraction
        
    Returns:
        Dict: Normalized tool statistics with all tools present
    """
    normalized = {}
    
    # Add all possible tools with defaults
    for tool in ALL_POSSIBLE_TOOLS:
        if tool in tool_stats:
            normalized[tool] = tool_stats[tool].copy()
        else:
            normalized[tool] = DEFAULT_TOOL_STATS.copy()
    
    # Also include any unexpected tools (in case new tools are added)
    for tool, stats in tool_stats.items():
        if tool not in normalized:
            normalized[tool] = stats.copy()
    
    return normalized


def _normalize_tool_error_counts(tool_error_counts: dict[str, int]) -> dict[str, int]:
    """
    Normalize tool_error_counts to include all possible tools.
    
    Args:
        tool_error_counts (Dict): Raw error counts mapping
        
    Returns:
        Dict: Normalized error counts with all tools present
    """
    normalized = {}
    
    # Add all possible tools with zero defaults
    for tool in ALL_POSSIBLE_TOOLS:
        normalized[tool] = tool_error_counts.get(tool, 0)
    
    # Also include any unexpected tools
    for tool, count in tool_error_counts.items():
        if tool not in normalized:
            normalized[tool] = count
    
    return normalized


def _extract_tool_stats(messages: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """
    Extract tool usage statistics from message history.
    
    Args:
        messages (list[dict]): Message history
        
    Returns:
        Dict: Tool statistics with counts and success/failure rates
    """
    tool_stats = {}
    
    # Track tool calls and their results
    tool_calls_map = {}  # Map tool_call_id to tool name
    
    for msg in messages:
        # Track tool calls from assistant messages
        if msg["role"] == "assistant" and "tool_calls" in msg and msg["tool_calls"]:
            for tool_call in msg["tool_calls"]:
                if not tool_call or not isinstance(tool_call, dict): continue
                tool_name = tool_call["function"]["name"]
                tool_call_id = tool_call["id"]
                
                # Initialize stats for this tool if not exists
                if tool_name not in tool_stats:
                    tool_stats[tool_name] = {
                        "count": 0,
                        "success": 0,
                        "failure": 0
                    }
                
                tool_stats[tool_name]["count"] += 1
                tool_calls_map[tool_call_id] = tool_name
        
        # Track tool responses
        elif msg["role"] == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            content = msg.get("content", "")
            
            # Determine if tool call was successful
            is_success = True
            try:
                # Try to parse as JSON and check for actual error values
                content_json = json.loads(content) if isinstance(content, str) else content
                
                if isinstance(content_json, dict):
                    # Check if error field exists AND has a non-null value
                    if "error" in content_json and content_json["error"] is not None:
                        is_success = False
                    
                    # Special handling for terminal tool responses
                    # Terminal wraps its response in a "content" field
                    if "content" in content_json and isinstance(content_json["content"], dict):
                        inner_content = content_json["content"]
                        # Check for actual error (non-null error field)
                        # Note: non-zero exit codes are not failures - the model can self-correct
                        if inner_content.get("error") is not None:
                            is_success = False
                    
                    # Check for "success": false pattern used by some tools
                    if content_json.get("success") is False:
                        is_success = False
                        
            except (json.JSONDecodeError, ValueError, TypeError):
                # If not JSON, check if content is empty or explicitly states an error
                # Note: We avoid simple substring matching to prevent false positives
                if not content:
                    is_success = False
                # Only mark as failure if it explicitly starts with "Error:" or "ERROR:"
                elif content.strip().lower().startswith("error:"):
                    is_success = False
            
            # Update success/failure count
            if tool_call_id in tool_calls_map:
                tool_name = tool_calls_map[tool_call_id]
                if is_success:
                    tool_stats[tool_name]["success"] += 1
                else:
                    tool_stats[tool_name]["failure"] += 1
    
    return tool_stats


def _extract_reasoning_stats(messages: list[dict[str, Any]]) -> dict[str, int]:
    """
    Count how many assistant turns have reasoning vs no reasoning.
    
    Checks for <REASONING_SCRATCHPAD> in content or a non-empty 'reasoning' field
    (native thinking tokens). Returns counts for tracking reasoning coverage.
    
    Args:
        messages: Message history
        
    Returns:
        Dict with 'total_assistant_turns', 'turns_with_reasoning', 'turns_without_reasoning'
    """
    total = 0
    with_reasoning = 0
    
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        total += 1
        
        content = msg.get("content", "") or ""
        has_scratchpad = "<REASONING_SCRATCHPAD>" in content
        has_native_reasoning = bool(msg.get("reasoning", "").strip()) if msg.get("reasoning") else False
        
        if has_scratchpad or has_native_reasoning:
            with_reasoning += 1
    
    return {
        "total_assistant_turns": total,
        "turns_with_reasoning": with_reasoning,
        "turns_without_reasoning": total - with_reasoning,
        "has_any_reasoning": with_reasoning > 0,
    }


async def _process_single_prompt(
    prompt_index: int,
    prompt_data: dict[str, Any],
    batch_num: int,
    config: dict[str, Any]
) -> dict[str, Any]:
    """
    Process a single prompt with the agent.
    
    Args:
        prompt_index (int): Index of prompt in dataset
        prompt_data (Dict): Prompt data containing 'prompt' field and optional 'image' field
        batch_num (int): Batch number
        config (Dict): Configuration dict with agent parameters
        
    Returns:
        Dict: Result containing trajectory, stats, and metadata
    """
    prompt = prompt_data["prompt"]
    run_name = config.get("run_name")
    task_id = (
        f"{run_name}:task_{prompt_index}"
        if run_name
        else f"task_{prompt_index}"
    )

    agent = None
    overrides_registered = False
    try:
        # Per-prompt container image override: if the dataset row has an
        # ``image`` field, register it for this task's sandbox.  This is the
        # retained upstream edge for Docker, Modal, Singularity, and Daytona.
        container_image = prompt_data.get("image") or prompt_data.get(
            "docker_image"
        )
        if container_image:
            # Docker checks its local cache and pulls on a miss.  Other
            # backends resolve the image server-side and skip this host probe.
            from agent.secret_scope import get_secret

            env_type = get_secret("TERMINAL_ENV", "local")
            if env_type == "docker":
                try:
                    probe = await _run_docker_image_command(
                        ["docker", "image", "inspect", container_image],
                        timeout=10,
                    )
                    if probe.returncode != 0:
                        if config.get("verbose"):
                            print(
                                f"   Prompt {prompt_index}: Pulling docker image "
                                f"{container_image}...",
                                flush=True,
                            )
                        pull = await _run_docker_image_command(
                            ["docker", "pull", container_image],
                            timeout=600,
                            text=True,
                        )
                        if pull.returncode != 0:
                            return {
                                "success": False,
                                "prompt_index": prompt_index,
                                "error": (
                                    "Docker image not available: "
                                    f"{container_image}\n{pull.stderr[:500]}"
                                ),
                                "trajectory": None,
                                "tool_stats": {},
                                "toolsets_used": [],
                                "metadata": {
                                    "batch_num": batch_num,
                                    "timestamp": datetime.now().isoformat(),
                                },
                            }
                except FileNotFoundError:
                    pass
                except Exception as img_err:
                    if config.get("verbose"):
                        print(
                            f"   Prompt {prompt_index}: Docker image check "
                            f"failed: {img_err}",
                            flush=True,
                        )

            from tools.terminal_tool import register_task_env_overrides

            overrides = {
                "docker_image": container_image,
                "modal_image": container_image,
                "singularity_image": f"docker://{container_image}",
                "daytona_image": container_image,
            }
            if prompt_data.get("cwd"):
                overrides["cwd"] = prompt_data["cwd"]
            register_task_env_overrides(task_id, overrides)
            overrides_registered = True
            if config.get("verbose"):
                print(
                    f"   Prompt {prompt_index}: Using container image "
                    f"{container_image}"
                )

        # Sample toolsets from distribution for this prompt
        selected_toolsets = sample_toolsets_from_distribution(config["distribution"])
        
        if config.get("verbose"):
            print(f"   Prompt {prompt_index}: Using toolsets {selected_toolsets}")
        
        # Initialize agent with sampled toolsets and log prefix for identification
        log_prefix = f"[B{batch_num}:P{prompt_index}]"
        agent = AIAgent(
            base_url=config.get("base_url"),
            api_key=config.get("api_key"),
            model=config["model"],
            max_iterations=config["max_iterations"],
            enabled_toolsets=selected_toolsets,
            save_trajectories=False,  # We handle saving ourselves
            verbose_logging=config.get("verbose", False),
            ephemeral_system_prompt=config.get("ephemeral_system_prompt"),
            log_prefix_chars=config.get("log_prefix_chars", 100),
            log_prefix=log_prefix,
            providers_allowed=config.get("providers_allowed"),
            providers_ignored=config.get("providers_ignored"),
            providers_order=config.get("providers_order"),
            provider_sort=config.get("provider_sort"),
            openrouter_min_coding_score=config.get("openrouter_min_coding_score"),
            max_tokens=config.get("max_tokens"),
            reasoning_config=config.get("reasoning_config"),
            prefill_messages=config.get("prefill_messages"),
            skip_context_files=True,  # Don't pollute trajectories with SOUL.md/AGENTS.md
            skip_memory=True,  # Don't use persistent memory in batch runs
        )

        # Run the agent with task_id to ensure each task gets its own isolated VM
        result = await agent.run_conversation(prompt, task_id=task_id)
        
        # Extract tool usage statistics
        tool_stats = _extract_tool_stats(result["messages"])
        
        # Extract reasoning coverage stats
        reasoning_stats = _extract_reasoning_stats(result["messages"])
        
        # Convert to trajectory format (using existing method)
        trajectory = agent._convert_to_trajectory_format(
            result["messages"],
            prompt,
            result["completed"]
        )
        
        return {
            "success": True,
            "prompt_index": prompt_index,
            "trajectory": trajectory,
            "tool_stats": tool_stats,
            "reasoning_stats": reasoning_stats,
            "completed": result["completed"],
            "partial": result.get("partial", False),
            "api_calls": result["api_calls"],
            "toolsets_used": selected_toolsets,
            "metadata": {
                "batch_num": batch_num,
                "timestamp": datetime.now().isoformat(),
                "model": config["model"]
            }
        }
    
    except Exception as e:
        print(f"❌ Error processing prompt {prompt_index}: {e}")
        if config.get("verbose"):
            traceback.print_exc()
        
        return {
            "success": False,
            "prompt_index": prompt_index,
            "error": str(e),
            "trajectory": None,
            "tool_stats": {},
            "toolsets_used": [],
            "metadata": {
                "batch_num": batch_num,
                "timestamp": datetime.now().isoformat()
            }
        }
    finally:
        try:
            if agent is not None:
                close_task = asyncio.create_task(
                    agent.close(),
                    name=f"batch-agent-close-{task_id}",
                )
                await _await_owned_batch_task(close_task)
        finally:
            if overrides_registered:
                from tools.terminal_tool import clear_task_env_overrides

                clear_task_env_overrides(task_id)


async def _process_batch_worker(args: tuple) -> dict[str, Any]:
    """
    Worker function to process a single batch of prompts.
    
    Args:
        args (Tuple): (batch_num, batch_data, output_dir, completed_prompts, config)
        
    Returns:
        Dict: Batch results with statistics
    """
    batch_num, batch_data, output_dir, completed_prompts_set, config = args
    
    output_dir = Path(output_dir)
    print(f"\n🔄 Batch {batch_num}: Starting ({len(batch_data)} prompts)")
    
    # Output file for this batch
    batch_output_file = output_dir / f"batch_{batch_num}.jsonl"
    
    # Filter out already completed prompts
    prompts_to_process = [
        (idx, data) for idx, data in batch_data
        if idx not in completed_prompts_set
    ]
    
    if not prompts_to_process:
        print(f"✅ Batch {batch_num}: Already completed (skipping)")
        return {
            "batch_num": batch_num,
            "processed": 0,
            "skipped": len(batch_data),
            "tool_stats": {},
            "completed_prompts": []
        }
    
    print(f"   Processing {len(prompts_to_process)} prompts (skipping {len(batch_data) - len(prompts_to_process)} already completed)")
    
    # Initialize aggregated stats for this batch
    batch_tool_stats = {}
    batch_reasoning_stats = {"total_assistant_turns": 0, "turns_with_reasoning": 0, "turns_without_reasoning": 0}
    completed_in_batch = []
    discarded_no_reasoning = 0
    
    # Process each prompt sequentially in this batch
    for prompt_index, prompt_data in prompts_to_process:
        # Process the prompt
        result = await _process_single_prompt(
            prompt_index,
            prompt_data,
            batch_num,
            config
        )
        
        # Save trajectory if successful
        if result["success"] and result["trajectory"]:
            # Discard samples with zero reasoning across all turns
            reasoning = result.get("reasoning_stats", {})
            if not reasoning.get("has_any_reasoning", True):
                print(f"   🚫 Prompt {prompt_index} discarded (no reasoning in any turn)")
                discarded_no_reasoning += 1
                completed_in_batch.append(prompt_index)
                continue

            # Get and normalize tool stats for consistent schema across all entries
            raw_tool_stats = result.get("tool_stats", {})
            tool_stats = _normalize_tool_stats(raw_tool_stats)
            
            # Create normalized tool_error_counts mapping tool names to their failure counts
            raw_error_counts = {
                tool_name: stats.get("failure", 0) 
                for tool_name, stats in raw_tool_stats.items()
            }
            tool_error_counts = _normalize_tool_error_counts(raw_error_counts)
            
            trajectory_entry = {
                "prompt_index": prompt_index,
                "conversations": result["trajectory"],
                "metadata": result["metadata"],
                "completed": result["completed"],
                "partial": result.get("partial", False),  # True if stopped due to invalid tool calls
                "api_calls": result["api_calls"],
                "toolsets_used": result["toolsets_used"],
                "tool_stats": tool_stats,  # Full stats: {tool: {count, success, failure}} - normalized
                "tool_error_counts": tool_error_counts  # Simple: {tool: failure_count} - normalized
            }
            
            # Each batch owns its own append-only shard, so concurrent batches
            # cannot interleave a JSONL row.  The helper also completes a row
            # before cancellation reaches the worker, keeping resume JSONL-safe.
            await _append_jsonl_line(batch_output_file, trajectory_entry)
        
        # Aggregate tool statistics
        for tool_name, stats in result.get("tool_stats", {}).items():
            if tool_name not in batch_tool_stats:
                batch_tool_stats[tool_name] = {
                    "count": 0,
                    "success": 0,
                    "failure": 0
                }
            
            batch_tool_stats[tool_name]["count"] += stats["count"]
            batch_tool_stats[tool_name]["success"] += stats["success"]
            batch_tool_stats[tool_name]["failure"] += stats["failure"]
        
        # Aggregate reasoning stats
        for key in batch_reasoning_stats:
            batch_reasoning_stats[key] += result.get("reasoning_stats", {}).get(key, 0)
        
        # Only mark as completed if successfully saved (failed prompts can be retried on resume)
        if result["success"] and result["trajectory"]:
            completed_in_batch.append(prompt_index)
            status = "⚠️  partial" if result.get("partial") else "✅"
            print(f"   {status} Prompt {prompt_index} completed")
        else:
            print(f"   ❌ Prompt {prompt_index} failed (will retry on resume)")
    
    print(f"✅ Batch {batch_num}: Completed ({len(prompts_to_process)} prompts processed)")
    
    return {
        "batch_num": batch_num,
        "processed": len(prompts_to_process),
        "skipped": len(batch_data) - len(prompts_to_process),
        "tool_stats": batch_tool_stats,
        "reasoning_stats": batch_reasoning_stats,
        "discarded_no_reasoning": discarded_no_reasoning,
        "completed_prompts": completed_in_batch
    }


class BatchRunner:
    """
    Manages batch processing of agent prompts with checkpointing and statistics.
    """
    
    def __init__(
        self,
        dataset_file: str,
        batch_size: int,
        run_name: str,
        distribution: str = "default",
        max_iterations: int = 10,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = "claude-opus-4-20250514",
        num_workers: int = 4,
        verbose: bool = False,
        ephemeral_system_prompt: str | None = None,
        log_prefix_chars: int = 100,
        providers_allowed: list[str] | None = None,
        providers_ignored: list[str] | None = None,
        providers_order: list[str] | None = None,
        provider_sort: str | None = None,
        openrouter_min_coding_score: float | None = None,
        max_tokens: int | None = None,
        reasoning_config: dict[str, Any] | None = None,
        prefill_messages: list[dict[str, Any]] | None = None,
        max_samples: int | None = None,
    ):
        """
        Initialize the batch runner.

        Args:
            dataset_file (str): Path to the dataset JSONL file with 'prompt' field
            batch_size (int): Number of prompts per batch
            run_name (str): Name for this run (used for checkpointing and output)
            distribution (str): Toolset distribution to use (default: "default")
            max_iterations (int): Max iterations per agent run
            base_url (str): Base URL for model API
            api_key (str): API key for model
            model (str): Model name to use
            num_workers (int): Number of parallel workers
            verbose (bool): Enable verbose logging
            ephemeral_system_prompt (str): System prompt used during agent execution but NOT saved to trajectories (optional)
            log_prefix_chars (int): Number of characters to show in log previews for tool calls/responses (default: 20)
            providers_allowed (list[str]): OpenRouter providers to allow (optional)
            providers_ignored (list[str]): OpenRouter providers to ignore (optional)
            providers_order (list[str]): OpenRouter providers to try in order (optional)
            provider_sort (str): Sort providers by price/throughput/latency (optional)
            max_tokens (int): Maximum tokens for model responses (optional, uses model default if not set)
            reasoning_config (Dict): OpenRouter reasoning config override (e.g. {"effort": "none"} to disable thinking)
            prefill_messages (list[dict]): Messages to prepend as prefilled conversation context (few-shot priming).
                NOTE: Anthropic Sonnet 4.6+ and Opus 4.6+ reject a trailing assistant-role prefill
                (400 error).  For those models use output_config.format or structured-output
                schemas instead.  Safe here for user-role priming and for older Claude / non-Claude models.
            max_samples (int): Only process the first N samples from the dataset (optional, processes all if not set)
        """
        self.dataset_file = Path(dataset_file)
        self.batch_size = batch_size
        self.run_name = run_name
        self.distribution = distribution
        self.max_iterations = max_iterations
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.num_workers = num_workers
        self.verbose = verbose
        self.ephemeral_system_prompt = ephemeral_system_prompt
        self.log_prefix_chars = log_prefix_chars
        self.providers_allowed = providers_allowed
        self.providers_ignored = providers_ignored
        self.providers_order = providers_order
        self.provider_sort = provider_sort
        self.openrouter_min_coding_score = openrouter_min_coding_score
        self.max_tokens = max_tokens
        self.reasoning_config = reasoning_config
        self.prefill_messages = prefill_messages
        self.max_samples = max_samples
        
        # Validate distribution
        if not validate_distribution(distribution):
            raise ValueError(f"Unknown distribution: {distribution}. Available: {list(list_distributions().keys())}")
        
        # Output paths are resolved synchronously, but directory creation and
        # dataset/checkpoint reads are deferred to ``run()``.  Constructing a
        # runner is therefore safe in an async service request path.
        self.output_dir = Path("data") / run_name
        
        # Checkpoint file
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        
        # Statistics file
        self.stats_file = self.output_dir / "statistics.json"
        
        self.dataset: list[dict[str, Any]] = []
        self.batches: list[list[tuple[int, dict[str, Any]]]] = []
        self._initialized = False
    
    async def _load_dataset(self) -> list[dict[str, Any]]:
        """
        Load dataset from JSONL file.
        
        Returns:
            list[dict]: List of dataset entries
        """
        dataset = []
        try:
            async with aiofiles.open(self.dataset_file, encoding="utf-8") as source:
                line_num = 0
                async for line in source:
                    line_num += 1
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)
                        if "prompt" not in entry:
                            print(f"⚠️  Warning: Line {line_num} missing 'prompt' field, skipping")
                            continue
                        dataset.append(entry)
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Warning: Invalid JSON on line {line_num}: {e}")
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_file}") from exc
        
        if not dataset:
            raise ValueError(f"No valid entries found in dataset file: {self.dataset_file}")
        
        return dataset
    
    def _create_batches(self) -> list[list[tuple[int, dict[str, Any]]]]:
        """
        Split dataset into batches with indices.
        
        Returns:
            List of batches, where each batch is a list of (index, entry) tuples
        """
        batches = []
        for i in range(0, len(self.dataset), self.batch_size):
            batch = [(idx, entry) for idx, entry in enumerate(self.dataset[i:i + self.batch_size], start=i)]
            batches.append(batch)
        
        return batches
    
    async def _load_checkpoint(self) -> dict[str, Any]:
        """
        Load checkpoint data if it exists.
        
        Returns:
            Dict: Checkpoint data with completed prompt indices
        """
        try:
            async with aiofiles.open(self.checkpoint_file, encoding="utf-8") as source:
                return json.loads(await source.read())
        except FileNotFoundError:
            return {
                "run_name": self.run_name,
                "completed_prompts": [],
                "batch_stats": {},
                "last_updated": None,
            }
        except Exception as e:
            print(f"⚠️  Warning: Failed to load checkpoint: {e}")
            return {
                "run_name": self.run_name,
                "completed_prompts": [],
                "batch_stats": {},
                "last_updated": None
            }
    
    async def _save_checkpoint(
        self,
        checkpoint_data: dict[str, Any],
        lock: asyncio.Lock | None = None,
    ):
        """
        Save checkpoint data.
        
        Args:
            checkpoint_data (Dict): Checkpoint data to save
        """
        checkpoint_data["last_updated"] = datetime.now().isoformat()
        if lock is None:
            await _atomic_json_write(self.checkpoint_file, checkpoint_data)
            return
        if not isinstance(lock, asyncio.Lock):
            raise TypeError("checkpoint lock must be an asyncio.Lock")
        async with lock:
            await _atomic_json_write(self.checkpoint_file, checkpoint_data)
    
    async def _scan_completed_prompts_by_content(self) -> set:
        """
        Scan all batch files and extract completed prompts by their actual content.
        
        This provides a more robust resume mechanism that matches on prompt text
        rather than indices, allowing recovery even if indices don't match.
        
        Returns:
            set: Set of prompt texts that have been successfully processed
        """
        completed_prompts = set()
        batch_files = await _list_batch_files(self.output_dir)
        
        if not batch_files:
            return completed_prompts
        
        print(f"📂 Scanning {len(batch_files)} batch files for completed prompts...")
        
        for batch_file in batch_files:
            try:
                async with aiofiles.open(batch_file, encoding="utf-8") as source:
                    async for line in source:
                        try:
                            entry = json.loads(line.strip())
                            
                            # Skip failed entries - we want to retry these
                            if entry.get("failed", False):
                                continue
                            
                            # Extract the human/user prompt from conversations
                            conversations = entry.get("conversations", [])
                            for msg in conversations:
                                if msg.get("from") == "human":
                                    prompt_text = msg.get("value", "").strip()
                                    if prompt_text:
                                        completed_prompts.add(prompt_text)
                                    break  # Only need the first human message
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"  ⚠️  Warning: Error reading {batch_file.name}: {e}")
        
        return completed_prompts

    def _filter_dataset_by_completed(self, completed_prompts: set) -> tuple[list[dict], list[int]]:
        """
        Filter the dataset to exclude prompts that have already been completed.
        
        Args:
            completed_prompts: Set of prompt texts that have been completed
            
        Returns:
            Tuple of (filtered_dataset, skipped_indices)
        """
        filtered_dataset = []
        skipped_indices = []
        
        for idx, entry in enumerate(self.dataset):
            # Extract prompt from the dataset entry
            prompt_text = entry.get("prompt", "").strip()
            
            # Also check conversations format
            if not prompt_text:
                conversations = entry.get("conversations", [])
                for msg in conversations:
                    role = msg.get("role") or msg.get("from")
                    if role in {"user", "human"}:
                        prompt_text = (msg.get("content") or msg.get("value", "")).strip()
                        break
            
            if prompt_text in completed_prompts:
                skipped_indices.append(idx)
            else:
                # Keep original index for tracking
                filtered_dataset.append((idx, entry))
        
        return filtered_dataset, skipped_indices
    
    async def run(self, resume: bool = False):
        """
        Run the batch processing pipeline.
        
        Args:
            resume (bool): Whether to resume from checkpoint
        """
        # Built-in discovery is an awaited lazy boundary. Mutate the existing
        # public set so helpers that imported it retain their object identity.
        ALL_POSSIBLE_TOOLS.clear()
        ALL_POSSIBLE_TOOLS.update(await get_all_tool_names())
        if not self._initialized:
            await aiofiles.os.makedirs(self.output_dir, exist_ok=True)
            self.dataset = await self._load_dataset()
            if self.max_samples and self.max_samples < len(self.dataset):
                full_count = len(self.dataset)
                self.dataset = self.dataset[:self.max_samples]
                print(f"✂️  Truncated dataset from {full_count} to {self.max_samples} samples (--max_samples)")
            self.batches = self._create_batches()
            self._initialized = True

            print("📊 Batch Runner Initialized")
            print(f"   Dataset: {self.dataset_file} ({len(self.dataset)} prompts)")
            print(f"   Batch size: {self.batch_size}")
            print(f"   Total batches: {len(self.batches)}")
            print(f"   Run name: {self.run_name}")
            print(f"   Distribution: {self.distribution}")
            print(f"   Output directory: {self.output_dir}")
            print(f"   Workers: {self.num_workers}")
            if self.ephemeral_system_prompt:
                prompt_preview = self.ephemeral_system_prompt[:60] + "..." if len(self.ephemeral_system_prompt) > 60 else self.ephemeral_system_prompt
                print(f"   🔒 Ephemeral system prompt: '{prompt_preview}'")

        print("\n" + "=" * 70)
        print("🚀 Starting Batch Processing")
        print("=" * 70)
        
        # Smart resume: scan batch files by content to find completed prompts
        completed_prompt_texts = set()
        if resume:
            completed_prompt_texts = await self._scan_completed_prompts_by_content()
            if completed_prompt_texts:
                print(f"   Found {len(completed_prompt_texts)} already-completed prompts by content matching")
        
        # Filter dataset to only include unprocessed prompts
        if resume and completed_prompt_texts:
            filtered_entries, skipped_indices = self._filter_dataset_by_completed(completed_prompt_texts)
            
            if not filtered_entries:
                print("\n✅ All prompts have already been processed!")
                return
            
            # Recreate batches from filtered entries (keeping original indices for tracking)
            batches_to_process = []
            for i in range(0, len(filtered_entries), self.batch_size):
                batch = filtered_entries[i:i + self.batch_size]
                batches_to_process.append(batch)
            
            self.batches = batches_to_process
            
            # Print prominent resume summary
            print("\n" + "=" * 70)
            print("📊 RESUME SUMMARY")
            print("=" * 70)
            print(f"   Original dataset size:     {len(self.dataset):,} prompts")
            print(f"   Already completed:         {len(skipped_indices):,} prompts")
            print("   ─────────────────────────────────────────")
            print(f"   🎯 RESUMING WITH:          {len(filtered_entries):,} prompts")
            print(f"   New batches created:       {len(batches_to_process)}")
            print("=" * 70 + "\n")
        
        # Load existing checkpoint (so resume doesn't clobber prior progress)
        checkpoint_data = await self._load_checkpoint()
        if checkpoint_data.get("run_name") != self.run_name:
            checkpoint_data = {
                "run_name": self.run_name,
                "completed_prompts": [],
                "batch_stats": {},
                "last_updated": None
            }
        
        # Prepare configuration for batch tasks.
        #
        # ``self.api_key`` may be a zero-arg callable (Azure Foundry Entra ID
        # credential provider). Batch
        # tasks share the event loop, so keep the callable out of the task
        # configuration and let the agent resolve credentials from config when
        # needed, just as the old process worker did.
        if callable(self.api_key) and not isinstance(self.api_key, str):
            worker_api_key = None
            print(
                "ℹ️  Detected Entra ID bearer provider — batch tasks will rebuild "
                "credentials from config.yaml.",
                flush=True,
            )
        else:
            worker_api_key = self.api_key

        config = {
            "run_name": self.run_name,
            "distribution": self.distribution,
            "model": self.model,
            "max_iterations": self.max_iterations,
            "base_url": self.base_url,
            "api_key": worker_api_key,
            "verbose": self.verbose,
            "ephemeral_system_prompt": self.ephemeral_system_prompt,
            "log_prefix_chars": self.log_prefix_chars,
            "providers_allowed": self.providers_allowed,
            "providers_ignored": self.providers_ignored,
            "providers_order": self.providers_order,
            "provider_sort": self.provider_sort,
            "openrouter_min_coding_score": self.openrouter_min_coding_score,
            "max_tokens": self.max_tokens,
            "reasoning_config": self.reasoning_config,
            "prefill_messages": self.prefill_messages,
        }
        
        # For backward compatibility, still track by index (but this is secondary to content matching)
        completed_prompts_set = set(checkpoint_data.get("completed_prompts", []))
        
        # Aggregate statistics across all batches
        total_tool_stats = {}
        
        start_time = time.time()
        
        print(f"\n🔧 Initializing {self.num_workers} async batch workers...")

        # Each worker is an asyncio task.  A worker processes the prompts in
        # its batch sequentially (preserving per-batch JSONL ordering), while
        # up to ``num_workers`` batches overlap their model/tool I/O.  Nothing
        # wraps the agent loop in ``to_thread``.
        tasks = [
            (
                batch_num,
                batch_data,
                str(self.output_dir),
                completed_prompts_set,
                config,
            )
            for batch_num, batch_data in enumerate(self.batches)
        ]
        print(f"✅ Created {len(tasks)} async batch tasks")
        print("🚀 Starting async batch processing...\n")

        semaphore = asyncio.Semaphore(max(1, self.num_workers))

        async def run_batch(task_args: tuple) -> dict[str, Any]:
            async with semaphore:
                return await _process_batch_worker(task_args)

        batch_tasks = [asyncio.create_task(run_batch(task_args)) for task_args in tasks]
        results = []
        console = Console(force_terminal=True)
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]📦 Batches"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=2,
            transient=False,
            redirect_stdout=False,
            redirect_stderr=False,
        ) as progress:
            progress_task = progress.add_task("Processing", total=len(batch_tasks))
            root_logger = logging.getLogger()
            original_level = root_logger.level
            root_logger.setLevel(logging.WARNING)
            try:
                for completed_task in asyncio.as_completed(batch_tasks):
                    try:
                        result = await completed_task
                    except BaseException as exc:
                        if not isinstance(exc, asyncio.CancelledError):
                            logger.error(
                                "Async batch worker failed: %s", exc, exc_info=True
                            )
                        raise

                    results.append(result)
                    progress.update(progress_task, advance=1)

                    # Checkpoint writes happen only in this parent coroutine,
                    # so no process/thread lock is needed. The atomic write
                    # uses the native async file path and cannot interleave
                    # with another batch's checkpoint update.
                    try:
                        batch_num = result.get("batch_num")
                        completed = result.get("completed_prompts", []) or []
                        completed_prompts_set.update(completed)

                        if isinstance(batch_num, int):
                            checkpoint_data.setdefault("batch_stats", {})[str(batch_num)] = {
                                "processed": result.get("processed", 0),
                                "skipped": result.get("skipped", 0),
                                "discarded_no_reasoning": result.get("discarded_no_reasoning", 0),
                            }

                        checkpoint_data["completed_prompts"] = sorted(completed_prompts_set)
                        await self._save_checkpoint(checkpoint_data)
                    except Exception as ckpt_err:
                        # Don't fail the run if checkpoint write fails.
                        print(f"⚠️  Warning: Failed to save incremental checkpoint: {ckpt_err}")
            finally:
                root_logger.setLevel(original_level)
                pending_tasks = [task for task in batch_tasks if not task.done()]
                for pending in pending_tasks:
                    pending.cancel()
                if batch_tasks:
                    cleanup = asyncio.gather(*batch_tasks, return_exceptions=True)
                    cleanup_cancellation: asyncio.CancelledError | None = None
                    while True:
                        try:
                            await asyncio.shield(cleanup)
                            break
                        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised after cleanup
                            if cleanup.cancelled():
                                raise
                            if cleanup_cancellation is None:
                                cleanup_cancellation = exc
                    if cleanup_cancellation is not None:
                        raise cleanup_cancellation
        
        # Aggregate all batch statistics and update checkpoint
        total_reasoning_stats = {"total_assistant_turns": 0, "turns_with_reasoning": 0, "turns_without_reasoning": 0}

        for batch_result in results:
            # Aggregate tool stats
            for tool_name, stats in batch_result.get("tool_stats", {}).items():
                if tool_name not in total_tool_stats:
                    total_tool_stats[tool_name] = {
                        "count": 0,
                        "success": 0,
                        "failure": 0
                    }
                
                total_tool_stats[tool_name]["count"] += stats["count"]
                total_tool_stats[tool_name]["success"] += stats["success"]
                total_tool_stats[tool_name]["failure"] += stats["failure"]
            
            # Aggregate reasoning stats
            for key in total_reasoning_stats:
                total_reasoning_stats[key] += batch_result.get("reasoning_stats", {}).get(key, 0)
        
        # Save final checkpoint (best-effort; incremental writes already happened)
        try:
            checkpoint_data["completed_prompts"] = sorted(completed_prompts_set)
            await self._save_checkpoint(checkpoint_data)
        except Exception as ckpt_err:
            print(f"âš ï¸  Warning: Failed to save final checkpoint: {ckpt_err}")
        
        # Calculate success rates
        for tool_name in total_tool_stats:
            stats = total_tool_stats[tool_name]
            total_calls = stats["success"] + stats["failure"]
            if total_calls > 0:
                stats["success_rate"] = round(stats["success"] / total_calls * 100, 2)
                stats["failure_rate"] = round(stats["failure"] / total_calls * 100, 2)
            else:
                stats["success_rate"] = 0.0
                stats["failure_rate"] = 0.0
        
        # Combine ALL batch files in directory into a single trajectories.jsonl file
        # This includes both old batches (from previous runs) and new batches (from resume)
        # Also filter out corrupted entries (where model generated invalid tool names)
        combined_file = self.output_dir / "trajectories.jsonl"
        print(f"\n📦 Combining ALL batch files into {combined_file.name}...")
        
        # Valid tools auto-derived from model_tools.py — no manual updates needed
        VALID_TOOLS = ALL_POSSIBLE_TOOLS
        
        total_entries = 0
        filtered_entries = 0
        batch_files_found = 0
        
        # Find ALL batch files in the output directory (handles resume merging old + new)
        all_batch_files = await _list_batch_files(self.output_dir)
        
        combined_temporary_file = combined_file.with_name(f".{combined_file.name}.tmp")
        async with aiofiles.open(combined_temporary_file, "w", encoding="utf-8") as outfile:
            for batch_file in all_batch_files:
                batch_files_found += 1
                batch_num = batch_file.stem.split("_")[1]  # Extract batch number for logging
                
                async with aiofiles.open(batch_file, encoding="utf-8") as infile:
                    async for line in infile:
                        total_entries += 1
                        try:
                            data = json.loads(line)
                            tool_stats = data.get('tool_stats', {})
                            
                            # Check for invalid tool names (model hallucinations)
                            invalid_tools = [k for k in tool_stats if k not in VALID_TOOLS]
                            
                            if invalid_tools:
                                filtered_entries += 1
                                invalid_preview = invalid_tools[0][:50] + "..." if len(invalid_tools[0]) > 50 else invalid_tools[0]
                                print(f"   ⚠️  Filtering corrupted entry (batch {batch_num}): invalid tool '{invalid_preview}'")
                                continue
                            
                            await outfile.write(line)
                        except json.JSONDecodeError:
                            filtered_entries += 1
                            print(f"   ⚠️  Filtering invalid JSON entry (batch {batch_num})")
        await aiofiles.os.replace(combined_temporary_file, combined_file)
        
        if filtered_entries > 0:
            print(f"⚠️  Filtered {filtered_entries} corrupted entries out of {total_entries} total")
        print(f"✅ Combined {batch_files_found} batch files into trajectories.jsonl ({total_entries - filtered_entries} entries)")
        
        # Save final statistics
        final_stats = {
            "run_name": self.run_name,
            "distribution": self.distribution,
            "total_prompts": len(self.dataset),
            "total_batches": len(self.batches),
            "batch_size": self.batch_size,
            "model": self.model,
            "completed_at": datetime.now().isoformat(),
            "duration_seconds": round(time.time() - start_time, 2),
            "tool_statistics": total_tool_stats,
            "reasoning_statistics": total_reasoning_stats,
        }
        
        await _atomic_json_write(self.stats_file, final_stats)
        
        # Print summary
        print("\n" + "=" * 70)
        print("📊 BATCH PROCESSING COMPLETE")
        print("=" * 70)
        print(f"✅ Prompts processed this run: {sum(r.get('processed', 0) for r in results)}")
        print(f"✅ Total trajectories in merged file: {total_entries - filtered_entries}")
        print(f"✅ Total batch files merged: {batch_files_found}")
        print(f"⏱️  Total duration: {round(time.time() - start_time, 2)}s")
        print("\n📈 Tool Usage Statistics:")
        print("-" * 70)
        
        if total_tool_stats:
            # Sort by count descending
            sorted_tools = sorted(
                total_tool_stats.items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )
            
            print(f"{'Tool Name':<25} {'Count':<10} {'Success':<10} {'Failure':<10} {'Success Rate':<12}")
            print("-" * 70)
            for tool_name, stats in sorted_tools:
                print(
                    f"{tool_name:<25} "
                    f"{stats['count']:<10} "
                    f"{stats['success']:<10} "
                    f"{stats['failure']:<10} "
                    f"{stats['success_rate']:.1f}%"
                )
        else:
            print("No tool calls were made during this run.")
        
        # Print reasoning coverage stats
        total_discarded = sum(r.get("discarded_no_reasoning", 0) for r in results)

        print("\n🧠 Reasoning Coverage:")
        print("-" * 70)
        total_turns = total_reasoning_stats["total_assistant_turns"]
        with_reasoning = total_reasoning_stats["turns_with_reasoning"]
        without_reasoning = total_reasoning_stats["turns_without_reasoning"]
        if total_turns > 0:
            pct_with = round(with_reasoning / total_turns * 100, 1)
            pct_without = round(without_reasoning / total_turns * 100, 1)
            print(f"   Total assistant turns:    {total_turns:,}")
            print(f"   With reasoning:           {with_reasoning:,} ({pct_with}%)")
            print(f"   Without reasoning:        {without_reasoning:,} ({pct_without}%)")
        else:
            print("   No assistant turns recorded.")
        if total_discarded > 0:
            print(f"   🚫 Samples discarded (zero reasoning): {total_discarded:,}")

        print(f"\n💾 Results saved to: {self.output_dir}")
        print("   - Trajectories: trajectories.jsonl (combined)")
        print("   - Individual batches: batch_*.jsonl (for debugging)")
        print(f"   - Statistics: {self.stats_file.name}")
        print(f"   - Checkpoint: {self.checkpoint_file.name}")


def _parse_provider_preferences(value: Any) -> list[str] | None:
    """Normalize Fire's tuple form of an upstream comma-separated CLI value."""
    if not value:
        return None
    values = value.split(",") if isinstance(value, str) else value
    return [provider.strip() for provider in values]


async def main(
    dataset_file: str | None = None,
    batch_size: int | None = None,
    run_name: str | None = None,
    distribution: str = "default",
    model: str = "anthropic/claude-sonnet-4.6",
    api_key: str | None = None,
    base_url: str = "https://openrouter.ai/api/v1",
    max_turns: int = 10,
    num_workers: int = 4,
    resume: bool = False,
    verbose: bool = False,
    list_distributions: bool = False,
    ephemeral_system_prompt: str | None = None,
    log_prefix_chars: int = 100,
    providers_allowed: str | None = None,
    providers_ignored: str | None = None,
    providers_order: str | None = None,
    provider_sort: str | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    reasoning_disabled: bool = False,
    prefill_messages_file: str | None = None,
    max_samples: int | None = None,
):
    """
    Run batch processing of agent prompts from a dataset.

    Args:
        dataset_file (str): Path to JSONL file with 'prompt' field in each entry
        batch_size (int): Number of prompts per batch
        run_name (str): Name for this run (used for output and checkpointing)
        distribution (str): Toolset distribution to use (default: "default")
        model (str): Model name to use (default: "claude-opus-4-20250514")
        api_key (str): API key for model authentication
        base_url (str): Base URL for model API
        max_turns (int): Maximum number of tool calling iterations per prompt (default: 10)
        num_workers (int): Maximum number of concurrent async batch tasks (default: 4)
        resume (bool): Resume from checkpoint if run was interrupted (default: False)
        verbose (bool): Enable verbose logging (default: False)
        list_distributions (bool): List available toolset distributions and exit
        ephemeral_system_prompt (str): System prompt used during agent execution but NOT saved to trajectories (optional)
        log_prefix_chars (int): Number of characters to show in log previews for tool calls/responses (default: 20)
        providers_allowed (str): Comma-separated list of OpenRouter providers to allow (e.g. "anthropic,openai")
        providers_ignored (str): Comma-separated list of OpenRouter providers to ignore (e.g. "together,deepinfra")
        providers_order (str): Comma-separated list of OpenRouter providers to try in order (e.g. "anthropic,openai,google")
        provider_sort (str): Sort providers by "price", "throughput", or "latency" (OpenRouter only)
        max_tokens (int): Maximum tokens for model responses (optional, uses model default if not set)
        reasoning_effort (str): Reasoning effort: "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra" (default: "medium")
        reasoning_disabled (bool): Completely disable reasoning/thinking tokens (default: False)
        prefill_messages_file (str): Path to JSON file containing prefill messages (list of {role, content} dicts)
        max_samples (int): Only process the first N samples from the dataset (optional, processes all if not set)
        
    Example:
        await main(
            dataset_file="data.jsonl",
            batch_size=10,
            run_name="my_run",
        )

        # Resume an interrupted run.
        await main(
            dataset_file="data.jsonl",
            batch_size=10,
            run_name="my_run",
            resume=True,
        )
    """
    # Handle list distributions
    if list_distributions:
        from toolset_distributions import (
            list_distributions as _list_distributions,
            print_distribution_info,
        )

        print("📊 Available Toolset Distributions")
        print("=" * 70)

        all_dists = _list_distributions()
        for dist_name in sorted(all_dists.keys()):
            print_distribution_info(dist_name)
        
        print("\n💡 Usage:")
        print("  python batch_runner.py --dataset_file=data.jsonl --batch_size=10 \\")
        print("                         --run_name=my_run --distribution=<name>")
        return
    
    # Validate required arguments
    if not dataset_file:
        print("❌ Error: --dataset_file is required")
        raise SystemExit(1)
    
    if not batch_size or batch_size < 1:
        print("❌ Error: --batch_size must be a positive integer")
        raise SystemExit(1)
    
    if not run_name:
        print("❌ Error: --run_name is required")
        raise SystemExit(1)
    
    # Parse provider preferences (comma-separated strings to lists)
    providers_allowed_list = _parse_provider_preferences(providers_allowed)
    providers_ignored_list = _parse_provider_preferences(providers_ignored)
    providers_order_list = _parse_provider_preferences(providers_order)
    
    # Build reasoning_config from CLI flags
    # --reasoning_disabled takes priority, then --reasoning_effort, then default (medium)
    reasoning_config = None
    if reasoning_disabled:
        # Completely disable reasoning/thinking tokens
        reasoning_config = {"effort": "none"}
        print("🧠 Reasoning: DISABLED (effort=none)")
    elif reasoning_effort:
        # Use specified effort level
        valid_efforts = ["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"]
        if reasoning_effort not in valid_efforts:
            print(f"❌ Error: --reasoning_effort must be one of: {', '.join(valid_efforts)}")
            raise SystemExit(1)
        reasoning_config = {"enabled": True, "effort": reasoning_effort}
        print(f"🧠 Reasoning effort: {reasoning_effort}")
    
    # Load prefill messages from JSON file if provided
    prefill_messages = None
    if prefill_messages_file:
        try:
            async with aiofiles.open(
                prefill_messages_file, encoding="utf-8"
            ) as source:
                prefill_messages = json.loads(await source.read())
            if not isinstance(prefill_messages, list):
                print("❌ Error: prefill_messages_file must contain a JSON array of messages")
                raise SystemExit(1)
            print(f"💬 Loaded {len(prefill_messages)} prefill messages from {prefill_messages_file}")
        except Exception as e:
            print(f"❌ Error loading prefill messages: {e}")
            raise SystemExit(1)
    
    # Initialize and run batch runner
    try:
        runner = BatchRunner(
            dataset_file=dataset_file,
            batch_size=batch_size,
            run_name=run_name,
            distribution=distribution,
            max_iterations=max_turns,
            base_url=base_url,
            api_key=api_key,
            model=model,
            num_workers=num_workers,
            verbose=verbose,
            ephemeral_system_prompt=ephemeral_system_prompt,
            log_prefix_chars=log_prefix_chars,
            providers_allowed=providers_allowed_list,
            providers_ignored=providers_ignored_list,
            providers_order=providers_order_list,
            provider_sort=provider_sort,
            max_tokens=max_tokens,
            reasoning_config=reasoning_config,
            prefill_messages=prefill_messages,
            max_samples=max_samples,
        )

        await runner.run(resume=resume)
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if verbose:
            traceback.print_exc()
        raise SystemExit(1)


@wraps(main)
def _run_cli(*args: Any, **kwargs: Any) -> Any:
    """Run the upstream Fire entry point at the process-level async boundary."""
    return asyncio.run(main(*args, **kwargs))


if __name__ == "__main__":
    fire.Fire(_run_cli)
