"""Restart-safety regressions for proactive tool-result pruning."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.context_compressor import _estimate_msg_budget_tokens
from hermes_state import SessionDB


_REARM_KEY = "_proactive_prune_rearm_tokens"


def _assistant_call(call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": "terminal", "arguments": '{"cmd":"ls"}'},
        }],
    }


def _tool_result(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _history(*, large_chars: int = 24_000) -> list[dict]:
    messages: list[dict] = [{"role": "user", "content": "start"}]
    for index in range(8):
        call_id = f"call_{index}"
        messages.append(_assistant_call(call_id))
        content = chr(65 + index) * large_chars if index < 3 else "ok"
        messages.append(_tool_result(call_id, content))
    return messages


async def _build_agent(db: SessionDB, session_id: str, *, platform: str = "telegram"):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            platform=platform,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.context_compressor.bind_session_state(db, session_id)
    from agent.conversation_compression import _hydrate_persisted_compression_guards

    await _hydrate_persisted_compression_guards(agent.context_compressor, db, session_id)
    return agent


def _configure_pruning(agent) -> None:
    compressor = agent.context_compressor
    compressor.proactive_prune_tokens = 48_000
    compressor.proactive_prune_min_result_chars = 8_000
    compressor.proactive_prune_min_reclaim_tokens = 4_096
    compressor.protect_first_n = 2
    compressor.protect_last_n = 4


async def _model_config(db: SessionDB, session_id: str) -> dict:
    raw = (await db.get_session(session_id))["model_config"]
    return json.loads(raw) if raw else {}


@pytest.mark.asyncio
async def test_gateway_eviction_reload_keeps_prune_and_durable_runway(tmp_path: Path) -> None:
    """A fresh gateway agent must reload both the pruned body and its runway."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "GATEWAY_PRUNE_RESTART"
    await db.create_session(
        session_id, source="telegram", model_config={"keep": "value"},
    )
    await db.append_messages_batch(session_id, _history())

    first_agent = await _build_agent(db, session_id)
    _configure_pruning(first_agent)
    before = await db.get_messages_as_conversation(session_id)
    pruned, count = first_agent.context_compressor.prune_tool_results_only(
        before, current_tokens=120_000,
    )

    assert count >= 1
    await first_agent.context_compressor._commit_proactive_prune(before, pruned, count)
    durable = await db.get_messages_as_conversation(session_id)
    assert [message["content"] for message in durable] == [
        message["content"] for message in pruned
    ]
    assert len(durable[2]["content"]) < 24_000
    stored_runway = (await _model_config(db, session_id))[_REARM_KEY]
    assert (await _model_config(db, session_id))["keep"] == "value"
    assert stored_runway > sum(map(_estimate_msg_budget_tokens, durable))

    # Simulate gateway cache eviction / process restart: construct a wholly
    # new AIAgent and load the active transcript from SQLite.
    resumed_agent = await _build_agent(db, session_id)
    _configure_pruning(resumed_agent)
    assert resumed_agent.context_compressor._proactive_prune_rearm_tokens == stored_runway
    reloaded = await db.get_messages_as_conversation(session_id)
    archived_before = len(await db.get_messages(session_id, include_inactive=True))
    result, second_count = resumed_agent.context_compressor.prune_tool_results_only(
        reloaded, current_tokens=1_000_000,
    )

    assert result is reloaded
    assert second_count == 0
    assert len(await db.get_messages(session_id, include_inactive=True)) == archived_before
    await db.close()


@pytest.mark.asyncio
async def test_fresh_agent_rearms_after_durable_history_regrowth_once(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "PRUNE_DURABLE_REGROWTH"
    await db.create_session(session_id, source="telegram")
    await db.append_messages_batch(session_id, _history())
    first_agent = await _build_agent(db, session_id)
    _configure_pruning(first_agent)
    first, first_count = first_agent.context_compressor.prune_tool_results_only(
        await db.get_messages_as_conversation(session_id), current_tokens=120_000,
    )
    assert first_count >= 1
    await first_agent.context_compressor._commit_proactive_prune(
        await db.get_messages_as_conversation(session_id), first, first_count
    )
    first_runway = (await _model_config(db, session_id))[_REARM_KEY]

    growth = [
        _assistant_call("regrown_large"),
        _tool_result("regrown_large", "z" * 240_000),
        _assistant_call("tail_1"),
        _tool_result("tail_1", "ok"),
        _assistant_call("tail_2"),
        _tool_result("tail_2", "ok"),
    ]
    await db.append_messages_batch(session_id, growth)

    resumed = await _build_agent(db, session_id)
    _configure_pruning(resumed)
    grown = await db.get_messages_as_conversation(session_id)
    assert sum(map(_estimate_msg_budget_tokens, grown)) >= first_runway
    second, second_count = resumed.context_compressor.prune_tool_results_only(
        grown, current_tokens=1_000_000,
    )

    assert second_count >= 1
    await resumed.context_compressor._commit_proactive_prune(grown, second, second_count)
    second_runway = (await _model_config(db, session_id))[_REARM_KEY]
    assert second_runway > first_runway

    restarted = await _build_agent(db, session_id)
    _configure_pruning(restarted)
    durable = await db.get_messages_as_conversation(session_id)
    result, third_count = restarted.context_compressor.prune_tool_results_only(
        durable, current_tokens=1_000_000,
    )
    assert result is durable
    assert third_count == 0
    assert restarted.context_compressor._proactive_prune_rearm_tokens == second_runway
    await db.close()


@pytest.mark.asyncio
async def test_prune_persistence_failure_is_a_noop(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "PRUNE_PERSISTENCE_FAILURE"
    await db.create_session(session_id, source="telegram")
    await db.append_messages_batch(session_id, _history())
    agent = await _build_agent(db, session_id)
    _configure_pruning(agent)
    messages = await db.get_messages_as_conversation(session_id)
    original_contents = [message["content"] for message in messages]

    with patch.object(
        db, "archive_and_compact", side_effect=RuntimeError("disk full"),
    ):
            result, count = agent.context_compressor.prune_tool_results_only(
                messages, current_tokens=120_000,
            )
            if count:
                    committed = await agent.context_compressor._commit_proactive_prune(
                        messages, result, count
                    )
                    if not committed:
                        result = messages
                        count = 0

    assert result is messages
    assert count == 0
    assert agent.context_compressor._proactive_prune_rearm_tokens == 0
    assert [message["content"] for message in messages] == original_contents
    assert [message["content"] for message in await db.get_messages_as_conversation(session_id)] == original_contents
    assert _REARM_KEY not in await _model_config(db, session_id)
    await db.close()


@pytest.mark.asyncio
async def test_archive_model_config_patch_rolls_back_with_transcript(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "PRUNE_ATOMIC_ARCHIVE_FAILURE"
    await db.create_session(
        session_id,
        source="telegram",
        model_config={"keep": "value", _REARM_KEY: 120_000},
    )
    original = [{"role": "user", "content": "original"}]
    await db.append_messages_batch(session_id, original)

    with patch.object(db, "_execute_write", side_effect=RuntimeError("insert failed")):
        with pytest.raises(RuntimeError, match="insert failed"):
            await db.archive_and_compact(
                session_id,
                [{"role": "user", "content": "replacement"}],
                model_config_patch={_REARM_KEY: None},
            )

    assert (await db.get_messages_as_conversation(session_id))[0]["content"] == "original"
    assert await _model_config(db, session_id) == {"keep": "value", _REARM_KEY: 120_000}
    await db.close()


@pytest.mark.asyncio
async def test_model_switch_clears_durable_runway(tmp_path: Path) -> None:
    """update_model must clear BOTH the in-memory and the durable runway."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "MODEL_SWITCH_CLEARS_RUNWAY"
    await db.create_session(
        session_id,
        source="telegram",
        model_config={"keep": "value", _REARM_KEY: 120_000},
    )
    agent = await _build_agent(db, session_id)
    compressor = agent.context_compressor
    assert compressor._proactive_prune_rearm_tokens == 120_000

    compressor.update_model("other/model", 200_000)
    await compressor.clear_durable_proactive_prune_rearm()

    assert compressor._proactive_prune_rearm_tokens == 0
    assert _REARM_KEY not in await _model_config(db, session_id)
    assert (await _model_config(db, session_id))["keep"] == "value"
    await db.close()


@pytest.mark.asyncio
async def test_patch_session_model_config_merge_and_delete(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "PATCH_MODEL_CONFIG"
    await db.create_session(
        session_id, source="cli", model_config={"keep": "value", "drop": 1},
    )

    await db.patch_session_model_config(session_id, {"drop": None, "added": 7})
    assert await _model_config(db, session_id) == {"keep": "value", "added": 7}

    # Missing rows and empty patches are no-ops, never errors.
    await db.patch_session_model_config("NO_SUCH_SESSION", {"x": 1})
    await db.patch_session_model_config(session_id, {})
    await db.close()


@pytest.mark.asyncio
async def test_incapable_store_short_circuits_before_prune_scan(tmp_path: Path) -> None:
    """A bound store without archive_and_compact must not pay the prune scan."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "INCAPABLE_STORE_FAST_NOOP"
    await db.create_session(session_id, source="telegram")
    await db.append_messages_batch(session_id, _history())
    agent = await _build_agent(db, session_id)
    _configure_pruning(agent)
    compressor = agent.context_compressor

    class _NoArchiveStore:
        pass

    compressor.bind_session_state(_NoArchiveStore(), session_id)
    messages = await db.get_messages_as_conversation(session_id)
    with patch.object(
        type(compressor), "_prune_old_tool_results",
        side_effect=AssertionError("scan must not run for incapable stores"),
    ):
        result, count = compressor.prune_tool_results_only(
            messages, current_tokens=120_000,
        )

    assert result is messages
    assert count == 0
    await db.close()
