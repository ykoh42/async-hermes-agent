"""Async HTTP/service boundary for the async Hermes agent.

This module intentionally contains no second conversation implementation.  A
request is translated into one ``AIAgent`` instance and the existing async
``run_conversation`` contract is awaited directly.  Keeping this boundary
small makes it useful from FastAPI, an ASGI worker, or an embedding service
without bringing back the removed gateway stack.
"""

from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from run_agent import AIAgent


class ConversationRequest(BaseModel):
    """Input accepted by the async conversation endpoint."""

    message: str = Field(min_length=1)
    system_message: Optional[str] = None
    conversation_history: Optional[List[Dict[str, Any]]] = None
    task_id: Optional[str] = None
    session_id: Optional[str] = None


class ConversationResponse(BaseModel):
    """Stable service response containing both answer and training data."""

    final_response: Any = None
    completed: bool = False
    partial: bool = False
    api_calls: int = 0
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    trajectory: List[Dict[str, Any]] = Field(default_factory=list)


AgentFactory = Callable[
    [ConversationRequest],
    AIAgent | Awaitable[AIAgent],
]


class AsyncHermesService:
    """Small async application service around :class:`AIAgent`.

    ``agent_factory`` is injectable so a FastAPI deployment can provide model,
    provider, toolset, and session configuration without making those details
    part of the wire protocol.  A synchronous factory is accepted because
    constructing an agent is local configuration work; model/tool I/O is
    always performed by the awaited conversation call.
    """

    def __init__(self, agent_factory: Optional[AgentFactory] = None):
        self._agent_factory = agent_factory or self._default_agent_factory

    @staticmethod
    def _default_agent_factory(request: ConversationRequest) -> AIAgent:
        kwargs: Dict[str, Any] = {}
        if request.session_id:
            kwargs["session_id"] = request.session_id
        return AIAgent(**kwargs)

    async def converse(self, request: ConversationRequest) -> ConversationResponse:
        """Run one turn and return the raw messages plus trajectory format."""
        agent = self._agent_factory(request)
        if inspect.isawaitable(agent):
            agent = await agent

        result = await agent.run_conversation(
            request.message,
            system_message=request.system_message,
            conversation_history=request.conversation_history,
            task_id=request.task_id,
        )
        messages = result.get("messages", []) or []
        trajectory = agent._convert_to_trajectory_format(
            messages,
            request.message,
            bool(result.get("completed", False)),
        )
        return ConversationResponse(
            final_response=result.get("final_response"),
            completed=bool(result.get("completed", False)),
            partial=bool(result.get("partial", False)),
            api_calls=int(result.get("api_calls", 0) or 0),
            messages=messages,
            trajectory=trajectory,
        )


def create_app(agent_factory: Optional[AgentFactory] = None) -> FastAPI:
    """Create an ASGI app exposing the async conversation endpoint."""
    service = AsyncHermesService(agent_factory=agent_factory)
    app = FastAPI(title="Async Hermes Agent")

    @app.get("/healthz")
    async def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/conversations", response_model=ConversationResponse)
    async def conversations(request: ConversationRequest) -> ConversationResponse:
        return await service.converse(request)

    return app


__all__ = [
    "AsyncHermesService",
    "ConversationRequest",
    "ConversationResponse",
    "create_app",
]
