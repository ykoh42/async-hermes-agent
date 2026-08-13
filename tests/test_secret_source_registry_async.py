"""Behavioral parity tests for the native-async secret-source registry."""

from __future__ import annotations

import asyncio

import pytest

from agent.secret_sources import registry
from agent.secret_sources.base import ErrorKind, FetchResult, SecretSource


class StubSource(SecretSource):
    def __init__(
        self,
        name: str,
        secrets: dict[str, str] | None = None,
        *,
        shape: str = "mapped",
        override: bool = False,
        protected: tuple[str, ...] = (),
        delay: float = 0,
    ) -> None:
        self.name = name
        self.label = name.title()
        self.secrets = secrets or {}
        self.shape = shape
        self._override = override
        self._protected = protected
        self._delay = delay

    async def fetch(self, cfg, home_path):
        if self._delay:
            await asyncio.sleep(self._delay)
        return FetchResult(secrets=dict(self.secrets))

    def override_existing(self, cfg):
        return self._override

    def protected_env_vars(self, cfg):
        return frozenset(self._protected)


def test_error_kind_preserves_legacy_stringification():
    assert str(ErrorKind.TIMEOUT) == "ErrorKind.TIMEOUT"
    assert ErrorKind.TIMEOUT.value == "timeout"


@pytest.fixture(autouse=True)
def _empty_registry(monkeypatch):
    registry._reset_registry_for_tests()
    monkeypatch.setattr(registry, "_ensure_builtin_sources", lambda: None)
    yield
    registry._reset_registry_for_tests()


@pytest.mark.asyncio
async def test_mapped_source_beats_bulk_regardless_of_configured_order(tmp_path):
    bulk = StubSource("bulk", {"SHARED": "bulk", "BULK_ONLY": "bulk"}, shape="bulk")
    mapped = StubSource("mapped", {"SHARED": "mapped"}, shape="mapped")
    assert registry.register_source(bulk)
    assert registry.register_source(mapped)
    environment: dict[str, str] = {}

    report = await registry.apply_all(
        {
            "sources": ["bulk", "mapped"],
            "bulk": {"enabled": True},
            "mapped": {"enabled": True},
        },
        tmp_path,
        environ=environment,
    )

    assert environment == {"SHARED": "mapped", "BULK_ONLY": "bulk"}
    assert report.provenance["SHARED"].source == "mapped"
    assert report.conflicts


@pytest.mark.asyncio
async def test_existing_override_and_protected_guards_are_preserved(tmp_path):
    source = StubSource(
        "source",
        {"EXISTING": "new", "PROTECTED": "bad"},
        override=True,
        protected=("PROTECTED",),
    )
    assert registry.register_source(source)
    environment = {"EXISTING": "old", "PROTECTED": "safe"}

    report = await registry.apply_all(
        {"source": {"enabled": True}}, tmp_path, environ=environment
    )

    assert environment == {"EXISTING": "new", "PROTECTED": "safe"}
    assert report.provenance["EXISTING"].overrode_env is True
    assert report.sources[0].skipped_protected == ["PROTECTED"]


@pytest.mark.asyncio
async def test_timeout_is_reported_without_blocking_other_sources(tmp_path):
    slow = StubSource("slow", delay=0.2)
    working = StubSource("working", {"KEY": "value"})
    assert registry.register_source(slow)
    assert registry.register_source(working)

    report = await registry.apply_all(
        {
            "slow": {"enabled": True, "timeout_seconds": 0.01},
            "working": {"enabled": True},
        },
        tmp_path,
        environ={},
    )

    assert report.sources[0].result.error_kind is ErrorKind.TIMEOUT
    assert report.provenance["KEY"].source == "working"


@pytest.mark.asyncio
async def test_external_cancellation_is_never_converted_to_source_error(tmp_path):
    source = StubSource("slow", delay=30)
    assert registry.register_source(source)
    task = asyncio.create_task(
        registry.apply_all({"slow": {"enabled": True}}, tmp_path, environ={})
    )
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
