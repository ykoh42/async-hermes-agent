"""Rate-limit-aware live concurrency benchmark for the async Hermes loop.

The live path is intentionally opt-in and conservative.  It creates one
``AIAgent`` per request, so a stage measures inter-agent concurrency without
violating the per-agent turn serialization contract.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import math
import os
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os
import psutil
from dotenv import dotenv_values


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_FREE_MODEL = "openai/gpt-oss-20b:free"
DEFAULT_STAGES = (1, 2, 4)


@dataclass(slots=True)
class LiveResponse:
    response_ok: bool
    rate_limits: dict[str, Any] | None = None
    failure_status: str | None = None
    error_type: str | None = None


@dataclass(slots=True)
class RequestResult:
    request_id: str
    status: str
    latency_seconds: float
    status_code: int | None = None
    retry_after_seconds: float | None = None
    response_ok: bool = False
    rate_limits: dict[str, Any] | None = None
    error_type: str | None = None


@dataclass(slots=True)
class StageResult:
    concurrency: int
    requests: int
    wall_seconds: float
    requests_per_second: float
    statuses: dict[str, int]
    latency_p50_seconds: float
    latency_p95_seconds: float
    event_loop_lag_p95_ms: float
    event_loop_lag_max_ms: float
    rss_delta_mib: float
    open_resource_delta: int | None
    pending_task_delta: int
    results: list[RequestResult]


@dataclass(slots=True)
class BenchmarkResult:
    model: str
    configured_stages: list[int]
    requests_per_worker: int
    timeout_seconds: float
    stages: list[StageResult]
    stop_reason: str | None


RequestRunner = Callable[[str], Awaitable[LiveResponse]]


def parse_stages(value: str) -> tuple[int, ...]:
    try:
        stages = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("concurrency stages must be integers") from exc
    if not stages or any(stage < 1 for stage in stages):
        raise argparse.ArgumentTypeError("concurrency stages must be positive")
    if tuple(sorted(set(stages))) != stages:
        raise argparse.ArgumentTypeError(
            "concurrency stages must be unique and strictly increasing"
        )
    return stages


def validate_request_budget(
    stages: Sequence[int], requests_per_worker: int, max_requests: int
) -> None:
    if requests_per_worker < 1:
        raise ValueError("requests per worker must be positive")
    requested = sum(stages) * requests_per_worker
    if requested > max_requests:
        raise ValueError(
            f"benchmark would make {requested} requests, exceeding the "
            f"--max-requests limit of {max_requests}"
        )


def validate_model(model: str, *, allow_paid_model: bool) -> None:
    if not model:
        raise ValueError("an OpenRouter model is required")
    if not allow_paid_model and not model.endswith(":free"):
        raise ValueError(
            "refusing a non-free model; pass --allow-paid-model to opt in explicitly"
        )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after")
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _status_code(exc: BaseException) -> int | None:
    value = getattr(exc, "status_code", None)
    if value is None:
        value = getattr(getattr(exc, "response", None), "status_code", None)
    return value if isinstance(value, int) else None


def _classify_exception(exc: BaseException) -> str:
    status_code = _status_code(exc)
    if status_code == 429:
        return "rate_limited"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if status_code is not None:
        return "provider_error"
    return "error"


async def _run_request(
    request_runner: RequestRunner,
    request_id: str,
    timeout_seconds: float,
) -> RequestResult:
    started = time.perf_counter()
    try:
        async with asyncio.timeout(timeout_seconds):
            response = await request_runner(request_id)
        status = response.failure_status or (
            "success" if response.response_ok else "invalid_response"
        )
        return RequestResult(
            request_id=request_id,
            status=status,
            latency_seconds=time.perf_counter() - started,
            response_ok=response.response_ok,
            rate_limits=response.rate_limits,
            error_type=response.error_type,
        )
    except Exception as exc:
        return RequestResult(
            request_id=request_id,
            status=_classify_exception(exc),
            latency_seconds=time.perf_counter() - started,
            status_code=_status_code(exc),
            retry_after_seconds=_retry_after_seconds(exc),
            error_type=type(exc).__name__,
        )


async def _event_loop_probe(stop: asyncio.Event, samples: list[float]) -> None:
    interval = 0.01
    while not stop.is_set():
        target = asyncio.get_running_loop().time() + interval
        await asyncio.sleep(interval)
        samples.append(max(0.0, asyncio.get_running_loop().time() - target))


def _open_resource_count(process: psutil.Process) -> int | None:
    try:
        if hasattr(process, "num_fds"):
            return process.num_fds()
        if hasattr(process, "num_handles"):
            return process.num_handles()
    except (psutil.Error, OSError):
        pass
    return None


def _pending_task_count() -> int:
    current = asyncio.current_task()
    return sum(task is not current and not task.done() for task in asyncio.all_tasks())


async def _run_stage(
    request_runner: RequestRunner,
    stage_index: int,
    concurrency: int,
    requests_per_worker: int,
    timeout_seconds: float,
) -> StageResult:
    process = psutil.Process()
    rss_before = process.memory_info().rss
    resources_before = _open_resource_count(process)
    tasks_before = _pending_task_count()
    lag_samples: list[float] = []
    stop_probe = asyncio.Event()
    probe = asyncio.create_task(
        _event_loop_probe(stop_probe, lag_samples), name="benchmark-event-loop-probe"
    )

    tasks: list[asyncio.Task[RequestResult]] = []
    started = time.perf_counter()
    try:
        async with asyncio.TaskGroup() as group:
            for worker in range(concurrency):
                for request_number in range(requests_per_worker):
                    request_id = f"s{stage_index}-w{worker}-r{request_number}"
                    tasks.append(
                        group.create_task(
                            _run_request(request_runner, request_id, timeout_seconds),
                            name=f"benchmark-{request_id}",
                        )
                    )
    finally:
        wall_seconds = time.perf_counter() - started
        stop_probe.set()
        await probe

    results = [task.result() for task in tasks]
    await asyncio.sleep(0)
    gc.collect()
    rss_after = process.memory_info().rss
    resources_after = _open_resource_count(process)
    statuses: dict[str, int] = {}
    for result in results:
        statuses[result.status] = statuses.get(result.status, 0) + 1
    latencies = [result.latency_seconds for result in results]

    return StageResult(
        concurrency=concurrency,
        requests=len(results),
        wall_seconds=wall_seconds,
        requests_per_second=len(results) / wall_seconds if wall_seconds else 0.0,
        statuses=statuses,
        latency_p50_seconds=_percentile(latencies, 0.50),
        latency_p95_seconds=_percentile(latencies, 0.95),
        event_loop_lag_p95_ms=_percentile(lag_samples, 0.95) * 1000,
        event_loop_lag_max_ms=max(lag_samples, default=0.0) * 1000,
        rss_delta_mib=(rss_after - rss_before) / (1024 * 1024),
        open_resource_delta=(
            resources_after - resources_before
            if resources_before is not None and resources_after is not None
            else None
        ),
        pending_task_delta=_pending_task_count() - tasks_before,
        results=results,
    )


async def run_benchmark(
    request_runner: RequestRunner,
    *,
    model: str,
    stages: Sequence[int] = DEFAULT_STAGES,
    requests_per_worker: int = 1,
    timeout_seconds: float = 120.0,
    cooldown_seconds: float = 10.0,
) -> BenchmarkResult:
    stage_results: list[StageResult] = []
    stop_reason: str | None = None

    for index, concurrency in enumerate(stages, start=1):
        stage = await _run_stage(
            request_runner,
            index,
            concurrency,
            requests_per_worker,
            timeout_seconds,
        )
        stage_results.append(stage)
        if stage.statuses.get("rate_limited", 0):
            stop_reason = "rate_limited"
            break
        if stage.statuses.get("success", 0) == 0:
            stop_reason = "stage_had_no_successes"
            break
        if index < len(stages) and cooldown_seconds > 0:
            await asyncio.sleep(cooldown_seconds)

    return BenchmarkResult(
        model=model,
        configured_stages=list(stages),
        requests_per_worker=requests_per_worker,
        timeout_seconds=timeout_seconds,
        stages=stage_results,
        stop_reason=stop_reason,
    )


def _normalise_rate_limits(state: Any) -> dict[str, Any] | None:
    if state is None or not getattr(state, "has_data", False):
        return None

    def bucket(name: str) -> dict[str, int | float]:
        value = getattr(state, name)
        return {
            "limit": value.limit,
            "remaining": value.remaining,
            "reset_seconds": value.reset_seconds,
        }

    return {
        "provider": state.provider,
        "requests_min": bucket("requests_min"),
        "requests_hour": bucket("requests_hour"),
        "tokens_min": bucket("tokens_min"),
        "tokens_hour": bucket("tokens_hour"),
    }


def make_openrouter_runner(
    *, api_key: str, model: str, work_dir: Path
) -> RequestRunner:
    from hermes_state import SessionDB
    from run_agent import AIAgent

    async def run(request_id: str) -> LiveResponse:
        marker = f"HERMES_BENCH_OK_{request_id}"
        session_db = SessionDB(work_dir / f"{request_id}.db")
        agent = AIAgent(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            provider="openrouter",
            model=model,
            max_iterations=1,
            max_tokens=64,
            enabled_toolsets=[],
            save_trajectories=False,
            quiet_mode=True,
            reasoning_config={"enabled": True, "effort": "low"},
            skip_context_files=True,
            skip_memory=True,
            session_db=session_db,
        )
        async with agent:
            # The benchmark request budget counts real HTTP attempts, not
            # conversation calls hidden behind Hermes' normal retry policy.
            agent._api_max_retries = 1
            result = await agent.run_conversation(
                f"Reply with exactly this text and nothing else: {marker}"
            )
            final_response = str(result.get("final_response") or "")
            failure_reason = str(result.get("failure_reason") or "")
            if result.get("failed"):
                if failure_reason in {"rate_limit", "upstream_rate_limit"}:
                    failure_status = "rate_limited"
                elif failure_reason == "timeout":
                    failure_status = "timeout"
                else:
                    failure_status = "provider_error"
            else:
                failure_status = None
            return LiveResponse(
                response_ok=marker in final_response,
                rate_limits=_normalise_rate_limits(agent.get_rate_limit_state()),
                failure_status=failure_status,
                error_type="AIAgentResultError" if failure_status else None,
            )

    return run


def _load_live_config(env_file: Path, model_override: str | None) -> tuple[str, str]:
    values: Mapping[str, str | None] = (
        dotenv_values(env_file) if env_file.exists() else {}
    )
    api_key = str(
        values.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY") or ""
    ).strip()
    model = str(
        model_override
        or values.get("OPENROUTER_MODEL")
        or os.getenv("OPENROUTER_MODEL")
        or DEFAULT_FREE_MODEL
    ).strip()
    if not api_key:
        raise ValueError(
            f"OPENROUTER_API_KEY was not found in {env_file} or the process environment"
        )
    return api_key, model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="confirm that real OpenRouter requests may be sent",
    )
    parser.add_argument(
        "--env-file", type=Path, default=Path("~/.hermes/.env").expanduser()
    )
    parser.add_argument("--model")
    parser.add_argument("--concurrency", type=parse_stages, default=DEFAULT_STAGES)
    parser.add_argument("--requests-per-worker", type=int, default=1)
    parser.add_argument("--max-requests", type=int, default=7)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--cooldown-seconds", type=float, default=10.0)
    parser.add_argument("--allow-paid-model", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


async def _async_main(
    args: argparse.Namespace,
    *,
    api_key: str,
    model: str,
    work_dir: Path,
) -> BenchmarkResult:
    validate_model(model, allow_paid_model=args.allow_paid_model)
    validate_request_budget(
        args.concurrency, args.requests_per_worker, args.max_requests
    )
    if args.timeout_seconds <= 0 or args.cooldown_seconds < 0:
        raise ValueError("timeout must be positive and cooldown must not be negative")

    result = await run_benchmark(
        make_openrouter_runner(api_key=api_key, model=model, work_dir=work_dir),
        model=model,
        stages=args.concurrency,
        requests_per_worker=args.requests_per_worker,
        timeout_seconds=args.timeout_seconds,
        cooldown_seconds=args.cooldown_seconds,
    )
    if args.output:
        await aiofiles.os.makedirs(args.output.parent, exist_ok=True)
        async with aiofiles.open(args.output, "w", encoding="utf-8") as handle:
            await handle.write(json.dumps(asdict(result), indent=2, sort_keys=True))
            await handle.write("\n")
    return result


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if not args.live:
        parser.error("live requests are disabled unless --live is supplied")
    try:
        api_key, model = _load_live_config(args.env_file, args.model)
        previous_hermes_home = os.environ.get("HERMES_HOME")
        with tempfile.TemporaryDirectory(
            prefix="async-hermes-openrouter-bench-"
        ) as temp_dir:
            os.environ["HERMES_HOME"] = temp_dir
            try:
                result = asyncio.run(
                    _async_main(
                        args,
                        api_key=api_key,
                        model=model,
                        work_dir=Path(temp_dir),
                    )
                )
            finally:
                if previous_hermes_home is None:
                    os.environ.pop("HERMES_HOME", None)
                else:
                    os.environ["HERMES_HOME"] = previous_hermes_home
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
