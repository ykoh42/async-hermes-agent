"""Async LLM execution middleware contract tests."""

import pytest

from hermes_cli.middleware import run_llm_execution_middleware_async


@pytest.mark.asyncio
async def test_async_middleware_can_await_next_call(monkeypatch):
    seen = []

    async def callback(request, next_call, **_context):
        seen.append(request["value"])
        response = await next_call({"value": request["value"] + 1})
        return {"response": response}

    monkeypatch.setattr(
        "hermes_cli.middleware._get_middleware_callbacks",
        lambda _kind: [callback],
    )

    async def terminal(request):
        return request["value"] * 2

    result = await run_llm_execution_middleware_async(
        {"value": 2}, terminal, original_request={"value": 1}
    )

    assert result == {"response": 6}
    assert seen == [2]
