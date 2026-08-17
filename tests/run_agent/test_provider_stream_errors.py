"""Provider-encoded streaming errors retain normal classifier semantics."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.chat_completion_helpers import (
    ProviderStreamError,
    _provider_stream_error_from_json_decode_error,
    _provider_stream_error_from_text,
)
from agent.error_classifier import FailoverReason, classify_api_error


def test_sse_error_event_is_materialized_with_status_and_body():
    error = _provider_stream_error_from_text(
        'event: error\ndata: {"code":"invalid_request_error",'
        '"message":"request validation failed: bad field"}\n\n',
        "error",
    )
    assert isinstance(error, ProviderStreamError)
    assert error.status_code is None
    assert error.body["error"]["code"] == "invalid_request_error"
    assert "request validation failed" in str(error)


def test_plain_json_error_event_is_not_silently_retried():
    error = _provider_stream_error_from_json_decode_error(
        json.JSONDecodeError(
            "invalid",
            "request validation failed: unsupported field",
            0,
        )
    )
    classified = classify_api_error(error, provider="openrouter", model="x")
    assert classified.reason is FailoverReason.format_error
    assert classified.should_fallback is True
    assert classified.retryable is False


def test_non_error_finish_does_not_turn_normal_text_into_error():
    assert _provider_stream_error_from_text("ordinary answer", "stop") is None


def test_chunk_error_shape_has_provider_status():
    from run_agent import _chat_completion_helpers

    chunk = SimpleNamespace(error_type="400", error_message="bad request")
    status = _chat_completion_helpers._status_code_from_payload(
        {"code": chunk.error_type, "message": chunk.error_message}
    )
    assert status == 400


@pytest.mark.parametrize("payload", ["not json", "event: error\\ndata: nope"])
def test_plain_stream_error_is_bounded(payload):
    error = _provider_stream_error_from_text(payload, "error")
    assert error is not None
    assert len(error.raw_text) <= 4096
