"""Retained tool-observation summary parity with upstream v2026.8.3."""

import json
from types import SimpleNamespace

from agent.agent_runtime_helpers import convert_to_trajectory_format
from agent.context_compressor import _summarize_tool_result
from agent.context_compressor import ContextCompressor


def test_execute_code_summary_preserves_upstream_preview_and_line_count():
    code = "first = 1\n" + "second = 2 # " + ("x" * 80)

    summary = _summarize_tool_result(
        "execute_code",
        json.dumps({"code": code}),
        "line one\nline two\n",
    )

    expected_preview = code[:60].replace("\n", " ") + "..."
    assert summary == f"[execute_code] `{expected_preview}` (3 lines output)"


def test_text_to_speech_summary_preserves_upstream_observation_shape():
    content = '{"audio_path":"/tmp/generated.wav"}'

    summary = _summarize_tool_result(
        "text_to_speech",
        json.dumps({"text": "hello"}),
        content,
    )

    assert summary == f"[text_to_speech] generated audio ({len(content):,} chars)"


def test_pruned_execute_code_observation_keeps_order_in_saved_trajectory():
    code = "print('training observation')"
    messages = [
        {"role": "user", "content": "run the code"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-code",
                    "type": "function",
                    "function": {
                        "name": "execute_code",
                        "arguments": json.dumps({"code": code}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-code",
            "content": "line\n" * 100,
        },
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "done"},
    ]
    compressor = ContextCompressor(
        model="test-model",
        config_context_length=128_000,
        quiet_mode=True,
    )

    pruned, count = compressor._prune_old_tool_results(
        messages,
        protect_tail_count=2,
    )

    assert count == 1
    assert pruned[2] == {
        "role": "tool",
        "tool_call_id": "call-code",
        "content": f"[execute_code] `{code}` (101 lines output)",
    }
    trajectory = convert_to_trajectory_format(
        SimpleNamespace(_format_tools_for_system_message=lambda: ""),
        pruned,
        "run the code",
        completed=True,
    )
    assert [row["from"] for row in trajectory[:4]] == [
        "system",
        "human",
        "gpt",
        "tool",
    ]
    assert f'"content": "[execute_code] `{code}` (101 lines output)"' in (
        trajectory[3]["value"]
    )
