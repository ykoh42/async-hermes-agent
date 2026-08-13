"""Deterministic coverage for strict live trajectory parsing."""

from __future__ import annotations

import json

import pytest

from tests.e2e.trajectory_assertions import assert_exact_terminal_trajectory


@pytest.mark.parametrize("nested_content", [False, True])
def test_exact_terminal_trajectory_accepts_retained_runner_formats(
    nested_content: bool,
) -> None:
    prompt = "strict prompt"
    command = "printf OBSERVATION"
    call = json.dumps(
        {"name": "terminal", "arguments": {"command": command}}
    )
    terminal_result = {"output": "OBSERVATION", "exit_code": 0, "error": None}
    response = json.dumps(
        {
            "tool_call_id": "call-1",
            "name": "terminal",
            "content": (
                {"content": terminal_result} if nested_content else terminal_result
            ),
        }
    )
    conversations = [
        {"from": "system", "value": "system"},
        {"from": "human", "value": prompt},
        {
            "from": "gpt",
            "value": f"<think>reasoning</think><tool_call>\n{call}\n</tool_call>",
        },
        {
            "from": "tool",
            "value": f"<tool_response>\n{response}\n</tool_response>",
        },
        {"from": "gpt", "value": "<think>\nsummary\n</think>\nFINAL"},
    ]

    assert_exact_terminal_trajectory(
        conversations,
        prompt=prompt,
        command=command,
        observation="OBSERVATION",
        final="FINAL",
    )


def test_exact_terminal_trajectory_rejects_extra_visible_tool_call_text() -> None:
    conversations = [
        {"from": "system", "value": "system"},
        {"from": "human", "value": "strict prompt"},
        {
            "from": "gpt",
            "value": (
                "extra text\n<tool_call>\n"
                '{"name":"terminal","arguments":{"command":"printf O"}}\n'
                "</tool_call>"
            ),
        },
        {
            "from": "tool",
            "value": (
                "<tool_response>\n"
                '{"tool_call_id":"c","name":"terminal","content":'
                '{"output":"O","exit_code":0,"error":null}}\n'
                "</tool_response>"
            ),
        },
        {"from": "gpt", "value": "FINAL"},
    ]

    with pytest.raises(AssertionError):
        assert_exact_terminal_trajectory(
            conversations,
            prompt="strict prompt",
            command="printf O",
            observation="O",
            final="FINAL",
        )
