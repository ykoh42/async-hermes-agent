from agent.context_compressor import (
    ContextCompressor,
    _prune_stale_reasoning_replay,
    _summarize_tool_result_unguarded,
    is_user_originated_turn,
    reference_handoff_would_drive_next_model_call,
)
from agent.conversation_loop import _INTERRUPT_SCAFFOLD_MARKER


def test_stale_codex_replay_is_pruned_but_compaction_item_survives():
    messages = [
        {
            "role": "assistant",
            "codex_reasoning_items": [
                {"type": "message", "encrypted_content": "old"},
                {"type": "compaction", "encrypted_content": "carrier"},
            ],
        },
        {"role": "user", "content": "current ask"},
        {"role": "assistant", "codex_reasoning_items": [{"type": "message"}]},
    ]

    assert _prune_stale_reasoning_replay(messages) == 1
    assert messages[0]["codex_reasoning_items"] == [
        {"type": "compaction", "encrypted_content": "carrier"}
    ]
    assert messages[2]["codex_reasoning_items"] == [{"type": "message"}]


def test_clarify_summary_preserves_real_answer_but_not_timeout_sentinel():
    answered = _summarize_tool_result_unguarded(
        "clarify", "{}", '{"user_response":"use postgres"}'
    )
    timed_out = _summarize_tool_result_unguarded(
        "clarify", "{}", '{"user_response":"[user did not respond within 5m]"}'
    )

    assert answered == '[clarify] user responded: "use postgres"'
    assert timed_out == "[clarify] asked user a question"


def test_reference_only_handoff_does_not_drive_next_turn():
    handoff = {
        "role": "user",
        "content": "[CONTEXT COMPACTION — REFERENCE ONLY] summary",
        "_compressed_summary": True,
    }
    assert reference_handoff_would_drive_next_model_call([handoff]) is True
    assert reference_handoff_would_drive_next_model_call(
        [handoff, {"role": "user", "content": "new ask"}]
    ) is False


def test_user_originated_turn_excludes_synthetic_rows():
    assert is_user_originated_turn({"role": "user", "content": "real"}) is True
    assert is_user_originated_turn(
        {"role": "user", "content": "internal", "_compressed_summary": True}
    ) is False


def test_interrupt_scaffold_marker_is_not_a_public_user_turn():
    assert _INTERRUPT_SCAFFOLD_MARKER.startswith("[This response was interrupted")
    assert ContextCompressor._is_synthetic_compression_user_turn(
        {"role": "user", "content": _INTERRUPT_SCAFFOLD_MARKER}
    ) is False
