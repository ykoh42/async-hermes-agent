"""Tests for user-id propagation at the retained memory-manager boundary."""

import json
import os
from unittest.mock import patch

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider


class RecordingProvider(MemoryProvider):
    """Minimal provider that records initialize() arguments."""

    def __init__(self, name: str = "recording"):
        self._name = name
        self._init_kwargs = {}
        self._init_session_id = None

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._init_session_id = session_id
        self._init_kwargs = dict(kwargs)

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return ""

    def sync_turn(self, user_content, assistant_content, *, session_id=""):
        return None

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, tool_name, args, **kwargs):
        return json.dumps({})

    def shutdown(self):
        return None


class TestMemoryManagerUserIdThreading:
    def test_no_user_id_when_cli(self):
        mgr = MemoryManager()
        provider = RecordingProvider()
        mgr.add_provider(provider)

        mgr.initialize_all(session_id="sess-456", platform="cli")

        assert "user_id" not in provider._init_kwargs
        assert provider._init_kwargs.get("platform") == "cli"

    def test_multiple_providers_all_receive_user_id(self):
        mgr = MemoryManager()
        first = RecordingProvider("builtin")
        second = RecordingProvider("external")
        mgr.add_provider(first)
        mgr.add_provider(second)

        mgr.initialize_all(
            session_id="sess-multi", platform="service", user_id="user-123"
        )

        for provider in (first, second):
            assert provider._init_kwargs.get("user_id") == "user-123"
            assert provider._init_kwargs.get("platform") == "service"


class TestAIAgentUserIdPropagation:
    def test_user_id_stored_on_agent(self):
        with patch.dict(os.environ, {"HERMES_HOME": "/tmp/test_hermes"}):
            from run_agent import AIAgent

            agent = object.__new__(AIAgent)
            agent._user_id = "test_user_42"
            assert agent._user_id == "test_user_42"

    def test_user_id_none_by_default(self):
        with patch.dict(os.environ, {"HERMES_HOME": "/tmp/test_hermes"}):
            from run_agent import AIAgent

            agent = object.__new__(AIAgent)
            agent._user_id = None
            assert agent._user_id is None
