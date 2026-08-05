import argparse
import asyncio

import pytest

from benchmarks.openrouter import (
    LiveResponse,
    parse_stages,
    run_benchmark,
    validate_model,
    validate_request_budget,
)


def test_parse_stages_requires_strictly_increasing_positive_values():
    assert parse_stages("1, 2,4") == (1, 2, 4)
    with pytest.raises(argparse.ArgumentTypeError, match="strictly increasing"):
        parse_stages("1,4,2")
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        parse_stages("0,1")


def test_request_budget_and_paid_model_guards():
    validate_request_budget((1, 2, 4), 1, 7)
    with pytest.raises(ValueError, match="exceeding"):
        validate_request_budget((1, 2, 4), 2, 7)
    validate_model("openai/gpt-oss-20b:free", allow_paid_model=False)
    with pytest.raises(ValueError, match="non-free"):
        validate_model("openai/gpt-5", allow_paid_model=False)


@pytest.mark.asyncio
async def test_stage_requests_overlap_and_leave_no_benchmark_tasks():
    active = 0
    max_active = 0
    lock = asyncio.Lock()

    async def runner(_request_id: str) -> LiveResponse:
        nonlocal active, max_active
        async with lock:
            active += 1
            max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        async with lock:
            active -= 1
        return LiveResponse(response_ok=True)

    result = await run_benchmark(
        runner,
        model="fake:free",
        stages=(4,),
        timeout_seconds=1,
        cooldown_seconds=0,
    )

    assert max_active == 4
    assert result.stages[0].statuses == {"success": 4}
    assert result.stages[0].pending_task_delta == 0
    assert not [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and task.get_name().startswith("benchmark-")
    ]


@pytest.mark.asyncio
async def test_all_timeouts_stop_later_stages():
    calls = 0

    async def runner(_request_id: str) -> LiveResponse:
        nonlocal calls
        calls += 1
        await asyncio.sleep(1)
        return LiveResponse(response_ok=True)

    result = await run_benchmark(
        runner,
        model="fake:free",
        stages=(1, 2),
        timeout_seconds=0.01,
        cooldown_seconds=0,
    )

    assert calls == 1
    assert result.stop_reason == "stage_had_no_successes"
    assert result.stages[0].statuses == {"timeout": 1}


@pytest.mark.asyncio
async def test_benchmark_cancellation_cleans_up_child_tasks():
    started = asyncio.Event()

    async def runner(_request_id: str) -> LiveResponse:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    benchmark = asyncio.create_task(
        run_benchmark(
            runner,
            model="fake:free",
            stages=(2,),
            timeout_seconds=10,
            cooldown_seconds=0,
        )
    )
    await started.wait()
    benchmark.cancel()

    with pytest.raises(asyncio.CancelledError):
        await benchmark
    assert not [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and task.get_name().startswith("benchmark-")
    ]


@pytest.mark.asyncio
async def test_structured_agent_failure_is_not_reported_as_invalid_response():
    async def runner(_request_id: str) -> LiveResponse:
        return LiveResponse(
            response_ok=False,
            failure_status="provider_error",
            error_type="AIAgentResultError",
        )

    result = await run_benchmark(
        runner,
        model="fake:free",
        stages=(1, 2),
        timeout_seconds=1,
        cooldown_seconds=0,
    )

    request = result.stages[0].results[0]
    assert request.status == "provider_error"
    assert request.error_type == "AIAgentResultError"
    assert result.stop_reason == "stage_had_no_successes"
