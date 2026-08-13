"""Unit tests for the shared session activity observation contract."""

from types import SimpleNamespace

from agent.session_activity import (
    ACTIVITY_DESCRIPTION_MAX,
    ActivityProvenance,
    bound_activity_description,
    build_activity_snapshot,
    normalize_activity_provenance,
    reset_session_activity_persist_window,
)


def test_bound_activity_description_truncates():
    output = bound_activity_description("x" * (ACTIVITY_DESCRIPTION_MAX + 80))
    assert len(output) == ACTIVITY_DESCRIPTION_MAX
    assert output.endswith("…")


def test_reset_session_activity_persist_window_clears_rate_limit():
    agent = SimpleNamespace(_session_activity_last_persist_mono=1234.5)
    reset_session_activity_persist_window(agent)
    assert agent._session_activity_last_persist_mono == 0.0


def test_reset_session_activity_persist_window_swallows_missing_attr():
    reset_session_activity_persist_window(object())


def test_normalize_activity_provenance_defaults_to_unknown():
    assert normalize_activity_provenance(None) is ActivityProvenance.UNKNOWN
    assert normalize_activity_provenance("") is ActivityProvenance.UNKNOWN
    assert (
        normalize_activity_provenance("not-a-real-source")
        is ActivityProvenance.UNKNOWN
    )
    assert (
        normalize_activity_provenance(ActivityProvenance.AGENT_COMPRESSION)
        is ActivityProvenance.AGENT_COMPRESSION
    )
    assert (
        normalize_activity_provenance("agent.compression_timeout")
        is ActivityProvenance.AGENT_COMPRESSION_TIMEOUT
    )


def test_activity_provenance_preserves_legacy_stringification():
    provenance = ActivityProvenance.AGENT_COMPRESSION
    assert str(provenance) == "ActivityProvenance.AGENT_COMPRESSION"
    assert provenance.value == "agent.compression"


def test_build_activity_snapshot_includes_compat_aliases():
    snapshot = build_activity_snapshot(
        last_activity_at=100.0,
        last_activity_description="starting API call #1",
        last_activity_provenance=ActivityProvenance.UNKNOWN,
        now=110.0,
        extra={"api_call_count": 1},
    )
    assert snapshot["last_activity_at"] == 100.0
    assert snapshot["last_activity_description"] == "starting API call #1"
    assert snapshot["last_activity_provenance"] == "unknown"
    assert snapshot["seconds_since_activity"] == 10.0
    assert snapshot["last_activity_ts"] == 100.0
    assert snapshot["last_activity_desc"] == "starting API call #1"
    assert snapshot["description"] == "starting API call #1"
    assert snapshot["api_call_count"] == 1
    assert "phase" not in snapshot
    assert "last_progress_at" not in snapshot


def test_build_activity_snapshot_preserves_compression_provenances():
    for provenance in (
        ActivityProvenance.AGENT_COMPRESSION,
        ActivityProvenance.AGENT_COMPRESSION_TIMEOUT,
        ActivityProvenance.AGENT_COMPRESSION_COOLDOWN,
    ):
        snapshot = build_activity_snapshot(
            last_activity_at=50.0,
            last_activity_description="compression state",
            last_activity_provenance=provenance,
            now=55.0,
        )
        assert snapshot["last_activity_provenance"] == provenance.value
        assert snapshot["provenance"] == provenance.value
