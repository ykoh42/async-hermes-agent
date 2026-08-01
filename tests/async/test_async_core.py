"""Regression coverage for the async-first public core."""

import inspect

import pytest

from run_agent import AIAgent
from model_tools import async_handle_function_call
from tools.registry import registry


def test_conversation_and_chat_are_coroutines():
    assert inspect.iscoroutinefunction(AIAgent.run_conversation)
    assert inspect.iscoroutinefunction(AIAgent.chat)


@pytest.mark.asyncio
async def test_async_registry_handler_is_awaited(monkeypatch):
    name = "__async_core_test_tool__"

    async def handler(args, **kwargs):
        return '{"value": %d}' % (args["value"] + 1)

    registry.register(
        name=name,
        toolset="async-core-test",
        schema={
            "name": name,
            "description": "test-only async handler",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
        },
        handler=handler,
        is_async=True,
    )
    try:
        result = await async_handle_function_call(name, {"value": 4})
        assert result == '{"value": 5}'
    finally:
        monkeypatch.setattr(registry, "_tools", {
            key: value for key, value in registry._tools.items() if key != name
        })
