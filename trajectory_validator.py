"""Audit and export Hermes trajectories for interleaved-thinking training.

The batch runner already owns generation, concurrency, checkpoints, and resume.
This module is the fail-closed boundary between its JSONL output and a training
pipeline: malformed rows are reported with line numbers, and export only
publishes a complete ShareGPT JSONL file when every source row is valid.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os


_THINK_BLOCK = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TOOL_RESPONSE_BLOCK = re.compile(
    r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL
)
_ALLOWED_ROLES = frozenset({"system", "human", "gpt", "tool"})


def validate_trajectory_entry(
    entry: Any,
    *,
    require_reasoning: bool = True,
    require_tools: bool = False,
) -> list[str]:
    """Return invariant violations for one batch trajectory entry."""
    if not isinstance(entry, dict):
        return ["entry must be a JSON object"]

    errors: list[str] = []
    conversations = entry.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return ["conversations must be a non-empty list"]

    roles: list[str | None] = []
    reasoning_turns = 0
    tool_turns = 0
    for index, turn in enumerate(conversations):
        if not isinstance(turn, dict):
            errors.append(f"conversation {index} must be an object")
            roles.append(None)
            continue

        role = turn.get("from")
        value = turn.get("value")
        roles.append(role if isinstance(role, str) else None)
        if role not in _ALLOWED_ROLES:
            errors.append(f"conversation {index} has invalid role {role!r}")
        if not isinstance(value, str) or not value.strip():
            errors.append(f"conversation {index} must have a non-empty string value")
            continue

        if role == "gpt":
            if value.count("<think>") != value.count("</think>"):
                errors.append(f"conversation {index} has an unbalanced think block")
            think_blocks = _THINK_BLOCK.findall(value)
            if not think_blocks:
                errors.append(f"conversation {index} is missing a think block")
            reasoning_turns += sum(bool(block.strip()) for block in think_blocks)
            for block in _TOOL_CALL_BLOCK.findall(value):
                try:
                    tool_call = json.loads(block)
                except json.JSONDecodeError:
                    errors.append(f"conversation {index} has invalid tool-call JSON")
                    continue
                if not isinstance(tool_call, dict):
                    errors.append(f"conversation {index} tool call must be an object")
                    continue
                if not isinstance(tool_call.get("name"), str) or not tool_call["name"]:
                    errors.append(f"conversation {index} tool call is missing a name")
                if not isinstance(tool_call.get("arguments"), dict):
                    errors.append(
                        f"conversation {index} tool call arguments must be an object"
                    )

        if role == "tool":
            tool_turns += 1
            previous = conversations[index - 1] if index > 0 else None
            following = conversations[index + 1] if index + 1 < len(conversations) else None
            previous_value = previous.get("value", "") if isinstance(previous, dict) else ""
            call_count = len(_TOOL_CALL_BLOCK.findall(previous_value))
            response_count = len(_TOOL_RESPONSE_BLOCK.findall(value))
            if index == 0 or roles[index - 1] != "gpt" or call_count == 0:
                errors.append(
                    f"conversation {index} tool observation is not preceded by a tool call"
                )
            if not isinstance(following, dict) or following.get("from") != "gpt":
                errors.append(
                    f"conversation {index} tool observation is not followed by a model turn"
                )
            if response_count == 0:
                errors.append(f"conversation {index} is missing a tool_response block")
            for block in _TOOL_RESPONSE_BLOCK.findall(value):
                try:
                    tool_response = json.loads(block)
                except json.JSONDecodeError:
                    errors.append(f"conversation {index} has invalid tool-response JSON")
                    continue
                if not isinstance(tool_response, dict):
                    errors.append(
                        f"conversation {index} tool response must be an object"
                    )
            if call_count and response_count and call_count != response_count:
                errors.append(
                    f"conversation {index} has {call_count} tool calls but "
                    f"{response_count} tool responses"
                )

    if roles[:2] != ["system", "human"]:
        errors.append("trajectory must start with system then human")
    if roles[-1:] != ["gpt"]:
        errors.append("trajectory must end with a model turn")
    elif isinstance(conversations[-1], dict):
        final_value = conversations[-1].get("value", "")
        if isinstance(final_value, str):
            visible_answer = _TOOL_CALL_BLOCK.sub("", _THINK_BLOCK.sub("", final_value))
            if not visible_answer.strip():
                errors.append("final model turn has no answer outside think/tool blocks")
    if entry.get("completed") is not True:
        errors.append("trajectory is not marked completed")
    if require_reasoning and reasoning_turns == 0:
        errors.append("trajectory has no non-empty reasoning turn")
    if require_tools and tool_turns == 0:
        errors.append("trajectory has no tool observation")
    return errors


async def audit_trajectory_file(
    source: Path,
    *,
    require_reasoning: bool = True,
    require_tools: bool = False,
) -> dict[str, Any]:
    """Stream and audit a trajectory JSONL file without loading it into memory."""
    total_rows = 0
    valid_rows = 0
    reasoning_turns = 0
    tool_observations = 0
    tool_names: dict[str, int] = {}
    seen_prompt_indices: set[Any] = set()
    failures: list[dict[str, Any]] = []

    async with aiofiles.open(source, "r", encoding="utf-8") as handle:
        line_number = 0
        async for raw_line in handle:
            line_number += 1
            if not raw_line.strip():
                continue
            total_rows += 1
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                failures.append(
                    {"line": line_number, "errors": [f"invalid JSON: {exc.msg}"]}
                )
                continue

            errors = validate_trajectory_entry(
                entry,
                require_reasoning=require_reasoning,
                require_tools=require_tools,
            )
            prompt_index = entry.get("prompt_index") if isinstance(entry, dict) else None
            if prompt_index is not None:
                if prompt_index in seen_prompt_indices:
                    errors.append(f"duplicate prompt_index {prompt_index!r}")
                seen_prompt_indices.add(prompt_index)

            conversations = entry.get("conversations", []) if isinstance(entry, dict) else []
            for turn in conversations if isinstance(conversations, list) else []:
                if not isinstance(turn, dict):
                    continue
                value = turn.get("value", "")
                if not isinstance(value, str):
                    continue
                if turn.get("from") == "gpt":
                    reasoning_turns += sum(
                        bool(block.strip()) for block in _THINK_BLOCK.findall(value)
                    )
                    for block in _TOOL_CALL_BLOCK.findall(value):
                        try:
                            name = json.loads(block).get("name")
                        except (json.JSONDecodeError, AttributeError):
                            name = None
                        if isinstance(name, str) and name:
                            tool_names[name] = tool_names.get(name, 0) + 1
                elif turn.get("from") == "tool":
                    tool_observations += len(_TOOL_RESPONSE_BLOCK.findall(value))

            if errors:
                failures.append({"line": line_number, "errors": errors})
            else:
                valid_rows += 1

    return {
        "source": str(source),
        "valid": not failures and total_rows > 0,
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "invalid_rows": len(failures),
        "reasoning_turns": reasoning_turns,
        "tool_observations": tool_observations,
        "tool_names": dict(sorted(tool_names.items())),
        "failures": failures,
    }


async def export_training_file(
    source: Path,
    destination: Path,
    *,
    require_reasoning: bool = True,
    require_tools: bool = False,
) -> dict[str, Any]:
    """Validate *source* and atomically export ShareGPT-only JSONL rows."""
    audit = await audit_trajectory_file(
        source,
        require_reasoning=require_reasoning,
        require_tools=require_tools,
    )
    if not audit["valid"]:
        raise ValueError(
            f"refusing to export {audit['invalid_rows']} invalid trajectory rows"
        )

    await aiofiles.os.makedirs(destination.parent, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        async with (
            aiofiles.open(source, "r", encoding="utf-8") as input_file,
            aiofiles.open(temporary, "w", encoding="utf-8") as output_file,
        ):
            async for raw_line in input_file:
                if not raw_line.strip():
                    continue
                entry = json.loads(raw_line)
                await output_file.write(
                    json.dumps(
                        {"conversations": entry["conversations"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            await output_file.flush()
            await aiofiles.os.wrap(os.fsync)(output_file.fileno())
        await aiofiles.os.replace(temporary, destination)
    except BaseException:
        try:
            await aiofiles.os.remove(temporary)
        except FileNotFoundError:
            pass
        raise

    audit["exported_to"] = str(destination)
    return audit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--export", type=Path, dest="destination")
    parser.add_argument("--allow-no-reasoning", action="store_true")
    parser.add_argument("--require-tools", action="store_true")
    return parser


async def _run_cli(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    kwargs = {
        "require_reasoning": not args.allow_no_reasoning,
        "require_tools": args.require_tools,
    }
    if args.destination:
        try:
            report = await export_training_file(
                args.source, args.destination, **kwargs
            )
        except ValueError:
            report = await audit_trajectory_file(args.source, **kwargs)
    else:
        report = await audit_trajectory_file(args.source, **kwargs)
    return report, 0 if report["valid"] else 1


def main() -> int:
    report, exit_code = asyncio.run(_run_cli(_build_parser().parse_args()))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
