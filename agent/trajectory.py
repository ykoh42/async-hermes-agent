"""Trajectory saving utilities and static helpers.

_convert_to_trajectory_format stays as an AIAgent method (batch_runner.py
calls agent._convert_to_trajectory_format). Only the static helpers and
the file-write logic live here.
"""

import json
import asyncio
from collections import deque
import concurrent.futures
import logging
import os
import threading
import aiofiles
import aiofiles.os
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

_TRAJECTORY_FILE_CLAIMS_GUARD = threading.RLock()
_TRAJECTORY_FILE_CLAIMS: dict[
    str,
    deque[concurrent.futures.Future[bool]],
] = {}


def _claim_trajectory_file(
    filename: str,
) -> tuple[str, bool, concurrent.futures.Future[bool]]:
    """Queue one append ticket without binding ownership to an event loop."""
    path_key = os.path.normcase(os.path.normpath(os.fspath(filename)))
    with _TRAJECTORY_FILE_CLAIMS_GUARD:
        claims = _TRAJECTORY_FILE_CLAIMS.setdefault(path_key, deque())
        owner = not claims
        claim: concurrent.futures.Future[bool] = concurrent.futures.Future()
        claims.append(claim)
        if owner:
            claim.set_result(True)
        return path_key, owner, claim


def _finish_trajectory_file_claim(
    path_key: str,
    claim: concurrent.futures.Future[bool],
) -> None:
    """Remove one ticket and hand ownership to the next FIFO waiter."""
    with _TRAJECTORY_FILE_CLAIMS_GUARD:
        claims = _TRAJECTORY_FILE_CLAIMS.get(path_key)
        if not claims:
            return
        was_owner = claims[0] is claim
        if was_owner:
            claims.popleft()
        else:
            try:
                claims.remove(claim)
            except ValueError:
                return
            if not claim.done():
                claim.cancel()
        while claims and claims[0].cancelled():
            claims.popleft()
        if not claims:
            _TRAJECTORY_FILE_CLAIMS.pop(path_key, None)
        elif was_owner and not claims[0].done():
            claims[0].set_result(True)


def convert_scratchpad_to_think(content: str) -> str:
    """Convert <REASONING_SCRATCHPAD> tags to <think> tags."""
    if not content or "<REASONING_SCRATCHPAD>" not in content:
        return content
    return content.replace("<REASONING_SCRATCHPAD>", "<think>").replace("</REASONING_SCRATCHPAD>", "</think>")


def has_incomplete_scratchpad(content: str) -> bool:
    """Check if content has an opening <REASONING_SCRATCHPAD> without a closing tag."""
    if not content:
        return False
    return "<REASONING_SCRATCHPAD>" in content and "</REASONING_SCRATCHPAD>" not in content


async def save_trajectory(trajectory: list[dict[str, Any]], model: str,
                          completed: bool, filename: str | None = None):
    """Append a trajectory entry to a JSONL file.

    Args:
        trajectory: The ShareGPT-format conversation list.
        model: Model name for metadata.
        completed: Whether the conversation completed successfully.
        filename: Override output filename. Defaults to trajectory_samples.jsonl
                  or failed_trajectories.jsonl based on ``completed``.
    """
    if filename is None:
        filename = "trajectory_samples.jsonl" if completed else "failed_trajectories.jsonl"

    entry = {
        "conversations": trajectory,
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "completed": completed,
    }

    line = json.dumps(entry, ensure_ascii=False) + "\n"

    async def _append() -> None:
        claim_filename = os.fspath(filename)
        if not os.path.isabs(claim_filename):
            claim_filename = os.path.join(
                await aiofiles.os.getcwd(),
                claim_filename,
            )
        path_key, owner, claim = _claim_trajectory_file(claim_filename)
        try:
            if not owner:
                await asyncio.wrap_future(claim)
            async with aiofiles.open(filename, "a", encoding="utf-8") as handle:
                await handle.write(line)
                await handle.flush()
        finally:
            _finish_trajectory_file_claim(path_key, claim)

    cancellation: asyncio.CancelledError | None = None
    try:
        # Keep one JSONL record intact when the owning turn is cancelled. The
        # write task is short and remains shielded long enough to close the
        # file before cancellation is propagated to the caller. Keep waiting
        # through repeated caller cancellation so the owned task cannot leak.
        write_task = asyncio.create_task(_append())
        while True:
            try:
                await asyncio.shield(write_task)
                break
            except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised below
                if write_task.cancelled():
                    raise
                if cancellation is None:
                    cancellation = exc
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning("Failed to save trajectory: %s", e)
        if cancellation is not None:
            raise cancellation
        return

    if cancellation is not None:
        raise cancellation
    logger.info("Trajectory saved to %s", filename)
