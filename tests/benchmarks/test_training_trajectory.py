import json

import pytest

from trajectory_validator import (
    audit_trajectory_file,
    export_training_file,
    validate_trajectory_entry,
)


def _entry(prompt_index=0):
    return {
        "prompt_index": prompt_index,
        "completed": True,
        "conversations": [
            {"from": "system", "value": "tools"},
            {"from": "human", "value": "calculate"},
            {
                "from": "gpt",
                "value": (
                    "<think>use the terminal</think>\n"
                    '<tool_call>{"name":"terminal","arguments":{}}</tool_call>'
                ),
            },
            {
                "from": "tool",
                "value": (
                    "<tool_response>"
                    '{"name":"terminal","content":"42"}'
                    "</tool_response>"
                ),
            },
            {"from": "gpt", "value": "<think>verify 42</think>\n42"},
        ],
    }


def test_valid_interleaved_trajectory_contract():
    assert validate_trajectory_entry(_entry(), require_tools=True) == []


def test_rejects_orphaned_observation():
    entry = _entry()
    entry["conversations"][2]["value"] = "<think></think>\nno call"

    errors = validate_trajectory_entry(entry, require_tools=True)

    assert any("not preceded by a tool call" in error for error in errors)


def test_rejects_malformed_tool_json_and_empty_final_answer():
    entry = _entry()
    entry["conversations"][2]["value"] = (
        "<think>call it</think>\n<tool_call>{broken}</tool_call>"
    )
    entry["conversations"][-1]["value"] = "<think>only reasoning</think>"

    errors = validate_trajectory_entry(entry, require_tools=True)

    assert any("invalid tool-call JSON" in error for error in errors)
    assert "final model turn has no answer outside think/tool blocks" in errors


@pytest.mark.asyncio
async def test_audit_detects_duplicate_prompt_and_invalid_json(tmp_path):
    source = tmp_path / "trajectories.jsonl"
    source.write_text(
        json.dumps(_entry()) + "\n" + json.dumps(_entry()) + "\n{" + "\n",
        encoding="utf-8",
    )

    report = await audit_trajectory_file(source, require_tools=True)

    assert report["valid"] is False
    assert report["total_rows"] == 3
    assert report["valid_rows"] == 1
    assert report["invalid_rows"] == 2
    assert report["tool_names"] == {"terminal": 2}
    assert report["tool_observations"] == 2
    assert report["reasoning_turns"] == 4
    assert report["failures"][0]["errors"] == ["duplicate prompt_index 0"]
    assert report["failures"][1]["errors"][0].startswith("invalid JSON:")


@pytest.mark.asyncio
async def test_export_is_sharegpt_only_and_fail_closed(tmp_path):
    source = tmp_path / "trajectories.jsonl"
    destination = tmp_path / "training.jsonl"
    source.write_text(json.dumps(_entry(7)) + "\n", encoding="utf-8")

    report = await export_training_file(source, destination, require_tools=True)

    exported = json.loads(destination.read_text(encoding="utf-8"))
    assert report["valid"] is True
    assert exported == {"conversations": _entry(7)["conversations"]}

    invalid_source = tmp_path / "invalid.jsonl"
    invalid_source.write_text("{}\n", encoding="utf-8")
    destination.write_text("preserve-existing-output\n", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to export"):
        await export_training_file(invalid_source, destination)
    assert destination.read_text(encoding="utf-8") == "preserve-existing-output\n"
