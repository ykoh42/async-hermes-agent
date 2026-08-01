"""Service-boundary tests for the async-only HTTP adapter."""

import inspect

import httpx
import pytest

from async_service import (
    AsyncHermesService,
    ConversationRequest,
    create_app,
)


class FakeAgent:
    def __init__(self):
        self.calls = []

    async def run_conversation(self, message, **kwargs):
        self.calls.append((message, kwargs))
        return {
            "final_response": "done",
            "completed": True,
            "partial": False,
            "api_calls": 1,
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "done"},
            ],
        }

    def _convert_to_trajectory_format(self, messages, prompt, completed):
        return [{"from": "human", "value": prompt}, {"from": "gpt", "value": "done"}]


@pytest.mark.asyncio
async def test_service_awaits_agent_and_returns_trajectory():
    agent = FakeAgent()
    service = AsyncHermesService(lambda request: agent)

    response = await service.converse(
        ConversationRequest(message="hello", task_id="task-1")
    )

    assert response.final_response == "done"
    assert response.trajectory[0]["value"] == "hello"
    assert agent.calls == [("hello", {
        "system_message": None,
        "conversation_history": None,
        "task_id": "task-1",
    })]


def test_fastapi_routes_are_async():
    app = create_app(lambda request: FakeAgent())
    route = next(route for route in app.routes if route.path == "/v1/conversations")
    assert inspect.iscoroutinefunction(route.endpoint)


@pytest.mark.asyncio
async def test_fastapi_route_returns_async_trajectory_payload():
    app = create_app(lambda request: FakeAgent())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/conversations", json={"message": "hello"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["final_response"] == "done"
    assert payload["trajectory"][0]["value"] == "hello"
