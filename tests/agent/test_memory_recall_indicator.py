"""MemoryManager.describe_recall deterministic indicator tests."""

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider, RecallStatus


class _FakeProvider(MemoryProvider):
    def __init__(self, name: str, status: RecallStatus | None, *, raises: bool = False):
        self._name = name
        self._status = status
        self._raises = raises

    @property
    def name(self) -> str:
        return self._name

    async def is_available(self) -> bool:
        return True

    async def initialize(self, session_id: str = "", **kwargs) -> None:
        return None

    def get_tool_schemas(self) -> list[dict]:
        return []

    async def handle_tool_call(self, tool_name, args, **kwargs) -> str:
        return ""

    def recall_status(self) -> RecallStatus | None:
        if self._raises:
            raise RuntimeError("boom")
        return self._status


def test_no_status_returns_empty_string():
    mgr = MemoryManager()
    mgr.add_provider(_FakeProvider("hindsight", None))
    assert mgr.describe_recall() == ""


def test_no_providers_returns_empty_string():
    assert MemoryManager().describe_recall() == ""


def test_single_memory_is_singular():
    mgr = MemoryManager()
    mgr.add_provider(_FakeProvider("hindsight", RecallStatus("Hindsight", 1)))
    assert mgr.describe_recall() == "👁️ Hindsight — recalled 1 memory"


def test_multiple_memories_are_plural():
    mgr = MemoryManager()
    mgr.add_provider(_FakeProvider("hindsight", RecallStatus("Hindsight", 3)))
    assert mgr.describe_recall() == "👁️ Hindsight — recalled 3 memories"


def test_zero_count_renders_generic():
    mgr = MemoryManager()
    mgr.add_provider(_FakeProvider("hindsight", RecallStatus("Hindsight", 0)))
    assert mgr.describe_recall() == "👁️ Hindsight — recalled relevant memory"


def test_aggregates_multiple_providers():
    mgr = MemoryManager()
    mgr.add_provider(_FakeProvider("builtin", RecallStatus("Notes", 2)))
    mgr.add_provider(_FakeProvider("hindsight", RecallStatus("Hindsight", 5)))
    result = mgr.describe_recall()
    assert "👁️ Notes — recalled 2 memories" in result
    assert "👁️ Hindsight — recalled 5 memories" in result


def test_failing_provider_is_skipped_not_fatal():
    mgr = MemoryManager()
    mgr.add_provider(_FakeProvider("builtin", None, raises=True))
    mgr.add_provider(_FakeProvider("hindsight", RecallStatus("Hindsight", 1)))
    assert mgr.describe_recall() == "👁️ Hindsight — recalled 1 memory"
