"""Small OS-process worker used by PostgreSQL production-readiness tests."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import aiofiles
import aiofiles.os

from hermes_state_postgres import SessionDB


RESULT_PREFIX = "POSTGRES_READINESS_RESULT="


async def _touch(path: str | None) -> None:
    if path is None:
        return
    target = Path(path)
    await aiofiles.os.makedirs(target.parent, exist_ok=True)
    async with aiofiles.open(target, "w", encoding="utf-8") as marker:
        await marker.write("ready\n")
        await marker.flush()


async def _run(args: argparse.Namespace) -> dict[str, object]:
    database = SessionDB(os.environ["HERMES_POSTGRES_TEST_DSN"])
    started = time.perf_counter()
    written = 0
    lease_holder = None
    lease_acquired = False
    lease_wait_seconds = None
    try:
        if args.mode == "append":
            if await database.get_session(args.session_id) is None:
                raise RuntimeError(f"missing session: {args.session_id}")
            await _touch(args.ready_path)
            for sequence in range(args.count):
                await database.append_message(
                    args.session_id,
                    "user",
                    f"worker={args.worker};sequence={sequence}",
                )
                written += 1
                if args.delay:
                    await asyncio.sleep(args.delay)
        elif args.mode == "lease":
            if await database.get_session(args.session_id) is None:
                raise RuntimeError(f"missing session: {args.session_id}")
            lease_holder = f"pid={os.getpid()}:worker={args.worker}"
            await _touch(args.ready_path)
            acquire_started = time.perf_counter()
            lease_acquired = await database.acquire_session_turn_lease(
                args.session_id,
                lease_holder,
                ttl_seconds=args.ttl,
                wait_seconds=args.wait,
                poll_interval_seconds=0.02,
            )
            lease_wait_seconds = time.perf_counter() - acquire_started
            if lease_acquired:
                await _touch(args.acquired_path)
                await database.append_message(
                    args.session_id,
                    "user",
                    f"lease={args.worker}:start",
                    turn_lease_holder=lease_holder,
                )
                written += 1
                if args.delay:
                    await asyncio.sleep(args.delay)
                await database.append_message(
                    args.session_id,
                    "user",
                    f"lease={args.worker}:end",
                    turn_lease_holder=lease_holder,
                )
                written += 1
        elif args.mode == "resume":
            if await database.get_session(args.session_id) is None:
                raise RuntimeError(f"missing session: {args.session_id}")
            marker = f"restart:{args.worker}"
            messages = await database.get_messages(args.session_id)
            if not any(message.get("content") == marker for message in messages):
                await database.append_message(args.session_id, "user", marker)
                written = 1
    finally:
        if lease_acquired and lease_holder is not None:
            await database.release_session_turn_lease(
                args.session_id,
                lease_holder,
            )
        await database.close()
    return {
        "mode": args.mode,
        "pid": os.getpid(),
        "written": written,
        "lease_acquired": lease_acquired,
        "lease_wait_seconds": (
            round(max(0.0, lease_wait_seconds), 6)
            if lease_wait_seconds is not None
            else None
        ),
        "duration_seconds": round(time.perf_counter() - started, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("append", "lease", "resume"))
    parser.add_argument("session_id")
    parser.add_argument("--worker", default="worker")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--ready-path")
    parser.add_argument("--acquired-path")
    parser.add_argument("--ttl", type=float, default=300.0)
    parser.add_argument("--wait", type=float, default=1800.0)
    args = parser.parse_args()
    print(f"{RESULT_PREFIX}{json.dumps(asyncio.run(_run(args)))}", flush=True)


if __name__ == "__main__":
    main()
