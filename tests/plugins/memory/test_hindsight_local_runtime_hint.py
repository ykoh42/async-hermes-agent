"""Upstream Hindsight local-runtime hint tests at the native async boundary."""

import sys
from unittest.mock import AsyncMock

import pytest

import plugins.memory.hindsight as hs
from plugins.memory.hindsight import HindsightMemoryProvider, _local_runtime_hint


def test_hint_for_missing_hindsight_all():
    hint = _local_runtime_hint("No module named 'hindsight'")
    assert "hindsight-all" in hint
    assert "hermes memory setup" in hint
    assert sys.executable in hint


def test_hint_for_missing_hindsight_embed():
    hint = _local_runtime_hint("No module named 'hindsight_embed.daemon_embed_manager'")
    assert "hindsight-all" in hint


def test_no_hint_for_unrelated_runtime_error():
    assert _local_runtime_hint("Illegal instruction (NumPy SIMD)") == ""
    assert _local_runtime_hint(None) == ""


@pytest.mark.asyncio
async def test_unavailable_reason_surfaces_hint_after_async_probe(monkeypatch):
    monkeypatch.setattr(hs, "_load_config", AsyncMock(return_value={"mode": "local_embedded"}))
    monkeypatch.setattr(
        hs,
        "_check_local_runtime",
        AsyncMock(return_value=(False, "No module named 'hindsight'")),
    )
    provider = HindsightMemoryProvider()
    assert await provider.is_available() is False
    reason = provider.unavailable_reason()
    assert "hindsight-all" in reason
    assert reason == reason.strip()


@pytest.mark.asyncio
async def test_unavailable_reason_empty_for_cloud(monkeypatch):
    monkeypatch.setattr(hs, "_load_config", AsyncMock(return_value={"mode": "cloud"}))
    probe = AsyncMock(side_effect=AssertionError("probed"))
    monkeypatch.setattr(hs, "_check_local_runtime", probe)
    provider = HindsightMemoryProvider()
    # No key/URL means unavailable, but cloud must not run the local probe.
    assert await provider.is_available() is False
    probe.assert_not_awaited()
    assert provider.unavailable_reason() == ""


@pytest.mark.asyncio
async def test_unavailable_reason_empty_when_runtime_present(monkeypatch):
    monkeypatch.setattr(hs, "_load_config", AsyncMock(return_value={"mode": "local_embedded"}))
    monkeypatch.setattr(hs, "_check_local_runtime", AsyncMock(return_value=(True, None)))
    provider = HindsightMemoryProvider()
    assert await provider.is_available() is True
    assert provider.unavailable_reason() == ""
