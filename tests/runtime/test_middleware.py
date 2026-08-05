"""LLM execution middleware contract tests."""

import pytest


@pytest.mark.asyncio
async def test_middleware_can_await_next_call(monkeypatch):
    import importlib

    middleware = importlib.import_module("hermes_cli.middleware")
    seen = []

    async def callback(request, next_call, **_context):
        seen.append(request["value"])
        response = await next_call({"value": request["value"] + 1})
        return {"response": response}

    monkeypatch.setattr(
        middleware,
        "_get_middleware_callbacks",
        lambda _kind: [callback],
    )

    async def terminal(request):
        return request["value"] * 2

    result = await middleware.run_llm_execution_middleware(
        {"value": 2}, terminal, original_request={"value": 1}
    )

    assert result == {"response": 6}
    assert seen == [2]
