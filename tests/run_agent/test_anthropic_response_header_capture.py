"""Anthropic Messages path must capture rate-limit + credits headers.

Portal Claude moved off /chat/completions onto /v1/messages. The OpenAI-wire
streaming path captured both header families from ``stream.response``; the
Messages path must do the same via ``on_response`` or /status and the Nous
429 classifier lose last-known state.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from run_agent import AIAgent


def test_capture_anthropic_response_headers_forwards_to_both_captures():
    agent = object.__new__(AIAgent)
    agent._capture_rate_limits = MagicMock()
    agent._capture_credits = MagicMock()

    response = SimpleNamespace(headers={"x-ratelimit-remaining-requests": "10"})
    agent._capture_anthropic_response_headers(response)

    agent._capture_rate_limits.assert_called_once_with(response)
    agent._capture_credits.assert_called_once_with(response)
