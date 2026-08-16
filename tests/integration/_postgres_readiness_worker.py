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
        elif args.mode == "resume":
            if await database.get_session(args.session_id) is None:
                raise RuntimeError(f"missing session: {args.session_id}")
            marker = f"restart:{args.worker}"
            messages = await database.get_messages(args.session_id)
            if not any(message.get("content") == marker for message in messages):
                await database.append_message(args.session_id, "user", marker)
                written = 1
    finally:
        await database.close()
    return {
        "mode": args.mode,
        "pid": os.getpid(),
        "written": written,
        "duration_seconds": round(time.perf_counter() - started, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("append", "resume"))
    parser.add_argument("session_id")
    parser.add_argument("--worker", default="worker")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--ready-path")
    args = parser.parse_args()
    print(f"{RESULT_PREFIX}{json.dumps(asyncio.run(_run(args)))}", flush=True)


if __name__ == "__main__":
    main()
