"""Configured unavailable memory providers should produce one clear warning."""

import logging

from agent import agent_init


def test_warns_once_and_dedupes(caplog):
    agent_init._warned_unavailable_providers.clear()
    with caplog.at_level(logging.WARNING, logger="run_agent"):
        agent_init._warn_memory_provider_unavailable("hindsight")
        agent_init._warn_memory_provider_unavailable("hindsight")

    warnings = [r for r in caplog.records if "unavailable" in r.getMessage()]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "hindsight" in msg
    assert "hermes memory status" in msg
    assert ".env" in msg


def test_distinct_providers_each_warn(caplog):
    agent_init._warned_unavailable_providers.clear()
    with caplog.at_level(logging.WARNING, logger="run_agent"):
        agent_init._warn_memory_provider_unavailable("hindsight")
        agent_init._warn_memory_provider_unavailable("mem0")

    warnings = [r for r in caplog.records if "unavailable" in r.getMessage()]
    assert len(warnings) == 2


def test_provider_reason_is_appended(caplog):
    agent_init._warned_unavailable_providers.clear()
    hint = "Install the embedded runtime with: uv pip install hindsight-all."
    with caplog.at_level(logging.WARNING, logger="run_agent"):
        agent_init._warn_memory_provider_unavailable("hindsight", hint)

    assert hint in caplog.records[-1].getMessage()
