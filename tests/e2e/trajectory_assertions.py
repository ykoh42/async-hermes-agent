"""Strict parsers shared by opt-in live trajectory acceptance tests."""

from __future__ import annotations

import json
import re
from typing import Any


_TOOL_CALL = re.compile(r"<tool_call>\n(?P<payload>.+)\n</tool_call>", re.DOTALL)
_TOOL_RESPONSE = re.compile(
    r"<tool_response>\n(?P<payload>.+)\n</tool_response>", re.DOTALL
)


def split_optional_think(value: str) -> tuple[str | None, str]:
    """Return an optional leading thinking block and the exact visible suffix."""
    if not value.startswith("<think>"):
        assert "<think>" not in value
        assert "</think>" not in value
        return None, value

    assert value.count("<think>") == 1
    assert value.count("</think>") == 1
    closing = value.index("</think>")
    thinking = value[len("<think>") : closing]
    if thinking.startswith("\n"):
        thinking = thinking[1:]
    if thinking.endswith("\n"):
        thinking = thinking[:-1]
    visible = value[closing + len("</think>") :]
    if visible.startswith("\n"):
        visible = visible[1:]
    assert "<think>" not in visible
    assert "</think>" not in visible
    return thinking, visible


def _parse_single_xml_json(value: str, pattern: re.Pattern[str]) -> dict[str, Any]:
    match = pattern.fullmatch(value)
    assert match is not None, value
    payload = json.loads(match.group("payload"))
    assert isinstance(payload, dict), payload
    return payload


def assert_exact_terminal_trajectory(
    conversations: list[dict[str, Any]],
    *,
    prompt: str,
    command: str,
    observation: str,
    final: str,
) -> None:
    """Assert one terminal call, its matching observation, and one final turn."""
    assert [turn["from"] for turn in conversations] == [
        "system",
        "human",
        "gpt",
        "tool",
        "gpt",
    ]
    assert conversations[1]["value"] == prompt

    _thinking, visible_call = split_optional_think(conversations[2]["value"])
    call = _parse_single_xml_json(visible_call, _TOOL_CALL)
    assert call == {
        "name": "terminal",
        "arguments": {"command": command},
    }

    response = _parse_single_xml_json(
        conversations[3]["value"], _TOOL_RESPONSE
    )
    assert set(response) == {"tool_call_id", "name", "content"}
    assert isinstance(response["tool_call_id"], str)
    assert response["tool_call_id"]
    assert response["name"] == call["name"]
    content = response["content"]
    if (
        isinstance(content, dict)
        and set(content) == {"content"}
        and isinstance(content["content"], dict)
    ):
        content = content["content"]
    assert isinstance(content, dict), content
    assert content["output"] == observation
    assert content["exit_code"] == 0
    assert content["error"] is None

    _final_thinking, visible_final = split_optional_think(
        conversations[4]["value"]
    )
    assert visible_final == final
