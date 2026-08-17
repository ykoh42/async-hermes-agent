"""Keep the upstream-verified safe wording in the skills guidance."""

import inspect
import re

import pytest

from agent.prompt_builder import SKILLS_GUIDANCE


REJECTED_FRAGMENTS = (
    "After completing a complex task",
    "5+ tool calls",
    "fixing a tricky error",
    "save the approach as a",
    "so you can reuse it next time",
)


@pytest.mark.parametrize("fragment", REJECTED_FRAGMENTS)
def test_rejected_skill_guidance_fragment_is_absent(fragment: str) -> None:
    assert fragment not in SKILLS_GUIDANCE


def test_verified_reword_and_behavior_remain_present() -> None:
    assert SKILLS_GUIDANCE.split("\n", 1)[0] == (
        "When you work out a non-trivial workflow, record it with skill_manage "
        "for future reuse."
    )
    assert "skill_manage(action='patch')" in SKILLS_GUIDANCE
    assert "## Skill Safety Rule" in SKILLS_GUIDANCE


def test_guidance_is_wired_into_system_prompt() -> None:
    import agent.system_prompt as system_prompt

    source = inspect.getsource(system_prompt)
    assert re.search(r"tool_guidance\.append\(\s*SKILLS_GUIDANCE\s*\)", source)
