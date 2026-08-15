"""Tests for concurrent @-reference expansion in context_references.

This is the upstream test module carried to the native-async boundary. The
barrier proves concurrency without relying on elapsed-time thresholds.
"""

from __future__ import annotations

import asyncio

import pytest

from agent.context_references import preprocess_context_references_async


async def _slow_fetcher(url: str) -> str:
    await asyncio.sleep(0.2)
    return f"CONTENT[{url}]"


@pytest.mark.asyncio
async def test_refs_expand_concurrently(tmp_path):
    message = (
        "see @url:https://a.example/x @url:https://b.example/y "
        "@url:https://c.example/z please"
    )
    ref_count = 3
    rendezvous = asyncio.Barrier(ref_count)
    entered: list[str] = []
    overlapped = asyncio.Event()

    async def barrier_fetcher(url: str) -> str:
        entered.append(url)
        async with asyncio.timeout(10):
            await rendezvous.wait()
        overlapped.set()
        return f"CONTENT[{url}]"

    try:
        result = await preprocess_context_references_async(
            message,
            cwd=tmp_path,
            context_length=100_000,
            url_fetcher=barrier_fetcher,
        )
    except (asyncio.BrokenBarrierError, TimeoutError):  # pragma: no cover
        pytest.fail(
            "references did not expand concurrently: a fetch reached the "
            f"rendezvous alone (entered: {entered})"
        )

    assert overlapped.is_set(), f"references never overlapped (entered: {entered})"
    assert len(entered) == ref_count
    assert result.expanded
    assert result.message.index("a.example") < result.message.index("b.example")
    assert result.message.index("b.example") < result.message.index("c.example")


@pytest.mark.asyncio
async def test_concurrent_preserves_output_contract(tmp_path):
    result = await preprocess_context_references_async(
        "@url:https://one.example/p @url:https://two.example/q",
        cwd=tmp_path,
        context_length=100_000,
        url_fetcher=_slow_fetcher,
    )
    assert "CONTENT[https://one.example/p]" in result.message
    assert "CONTENT[https://two.example/q]" in result.message
    assert result.message.index("one.example") < result.message.index("two.example")
    assert result.injected_tokens > 0
