"""Regression: compaction must move the LOGGING session context with the id.

When ``compress_context`` rotates ``agent.session_id`` it updates the
gateway/tools session context (``gateway.session_context.set_current_session_id``,
which moves ``HERMES_SESSION_ID`` env + ContextVar). The ``[session_id]`` tag on
log lines comes from a SEPARATE mechanism — ``hermes_logging._session_id_var``
(a ContextVar read by the global LogRecord factory), set once per turn in
``conversation_loop.py``. Before the fix, the rotation block never updated it, so
log lines emitted after a mid-turn compaction carried the STALE old id while the
message body / session DB / gateway state carried the new one (see #34089). This
asserts the logging context follows the rotation.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import hermes_logging
import pytest
from hermes_state import SessionDB


def _build_agent_with_db(db: SessionDB, session_id: str):
    """Mirror tests/agent/test_compression_concurrent_fork.py's harness."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )

    compressor = MagicMock()
    compressor.compress = AsyncMock(return_value=[
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "user", "content": "tail"},
    ])
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    # This test covers the ROTATION fallback (logging session-context follows
    # the id rotation) — pin in_place=False so it keeps exercising rotation
    # regardless of the global default (flipped to True in #38763).
    agent.compression_in_place = False
    return agent


@pytest.mark.asyncio
async def test_logging_session_context_follows_compression_rotation(tmp_path: Path) -> None:
    db = SessionDB(tmp_path / "state.db")
    parent_sid = "PARENT_LOGCTX_SESSION"
    await db.create_session(parent_sid, source="cli")

    agent = _build_agent_with_db(db, parent_sid)

    # conversation_loop.py pins the logging tag to the ORIGINAL id at turn start.
    hermes_logging.set_session_context(parent_sid)
    try:
        messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        await agent._compress_context(messages, "sys", approx_tokens=120_000)

        # The id actually rotated (sanity — otherwise the assertion is vacuous).
        assert agent.session_id != parent_sid

        # The logging context must now match the NEW id, not the stale one.
        current = hermes_logging._session_id_var.get()
        assert current == agent.session_id, (
            "Logging session context did not follow the compaction rotation: "
            f"log tag still {current!r}, agent.session_id is {agent.session_id!r} "
            "(see #34089)."
        )
    finally:
        hermes_logging.clear_session_context()
        await agent.close()
        await db.close()


@pytest.mark.asyncio
async def test_logging_session_context_is_isolated_between_tasks() -> None:
    ready = [asyncio.Event(), asyncio.Event()]
    release = asyncio.Event()

    async def observe(index: int, session_id: str) -> str | None:
        hermes_logging.set_session_context(session_id)
        ready[index].set()
        await release.wait()
        return hermes_logging._session_id_var.get()

    tasks = [
        asyncio.create_task(observe(0, "session-a")),
        asyncio.create_task(observe(1, "session-b")),
    ]
    await asyncio.gather(*(event.wait() for event in ready))
    release.set()

    assert await asyncio.gather(*tasks) == ["session-a", "session-b"]
