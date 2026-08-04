"""Native-async auxiliary model usage accounting tests."""

from types import SimpleNamespace

import pytest
import pytest_asyncio

from hermes_state import SessionDB


@pytest_asyncio.fixture
async def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    yield database
    await database.close()


def _response(model="aux-model", prompt=100, completion=20):
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
    )


async def _usage_rows(db, session_id):
    connection = await db._get_connection()
    cursor = await connection.execute(
        "SELECT * FROM session_model_usage WHERE session_id = ? ORDER BY task",
        (session_id,),
    )
    return [dict(row) for row in await cursor.fetchall()]


@pytest.mark.asyncio
async def test_records_and_accumulates_auxiliary_usage(db):
    await db.create_session("s1", source="library")
    for _ in range(3):
        await db.record_auxiliary_usage(
            "s1",
            "compression",
            model="glm-5",
            billing_provider="openrouter",
            input_tokens=1_000,
            output_tokens=100,
        )

    rows = await _usage_rows(db, "s1")
    assert len(rows) == 1
    assert rows[0]["task"] == "compression"
    assert rows[0]["input_tokens"] == 3_000
    assert rows[0]["output_tokens"] == 300
    assert rows[0]["api_call_count"] == 3


@pytest.mark.asyncio
async def test_main_loop_and_auxiliary_rows_remain_distinct(db):
    await db.create_session("s1", source="library")
    await db.update_token_counts(
        "s1",
        input_tokens=100,
        output_tokens=10,
        model="main-model",
        billing_provider="nous",
        api_call_count=1,
    )
    await db.record_auxiliary_usage(
        "s1",
        "title_generation",
        model="main-model",
        billing_provider="nous",
        input_tokens=40,
        output_tokens=8,
    )

    rows = await _usage_rows(db, "s1")
    assert [row["task"] for row in rows] == ["", "title_generation"]


@pytest.mark.asyncio
async def test_ambient_accounting_context_records_usage(db):
    from agent.aux_accounting import (
        record_aux_usage,
        reset_accounting_context,
        set_accounting_context,
    )

    await db.create_session("s1", source="library")
    token = set_accounting_context(db, "s1")
    try:
        await record_aux_usage(_response(model="aux-m"), "vision", provider="gemini")
    finally:
        reset_accounting_context(token)

    rows = await _usage_rows(db, "s1")
    assert len(rows) == 1
    assert rows[0]["task"] == "vision"
    assert rows[0]["model"] == "aux-m"
    assert rows[0]["input_tokens"] == 100
    assert rows[0]["output_tokens"] == 20


@pytest.mark.asyncio
async def test_moa_usage_is_not_double_counted(db):
    from agent.aux_accounting import (
        record_aux_usage,
        reset_accounting_context,
        set_accounting_context,
    )

    await db.create_session("s1", source="library")
    token = set_accounting_context(db, "s1")
    try:
        await record_aux_usage(_response(), "moa_reference")
        await record_aux_usage(_response(), "moa_aggregator")
    finally:
        reset_accounting_context(token)

    assert await _usage_rows(db, "s1") == []


@pytest.mark.asyncio
async def test_auxiliary_validation_chokepoint_records_usage(db):
    from agent.aux_accounting import reset_accounting_context, set_accounting_context
    from agent.auxiliary_client import _validate_llm_response

    await db.create_session("s1", source="library")
    token = set_accounting_context(db, "s1")
    try:
        result = await _validate_llm_response(
            _response(), "web_extract", provider="openrouter"
        )
    finally:
        reset_accounting_context(token)

    assert result is not None
    rows = await _usage_rows(db, "s1")
    assert len(rows) == 1
    assert rows[0]["task"] == "web_extract"
    assert rows[0]["billing_provider"] == "openrouter"
