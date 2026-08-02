"""Historical slash-skill messages remain readable after the async migration."""

from agent.skill_commands import (
    SKILL_EXCERPT_JOINT,
    SKILL_SCAFFOLD_SQL_LIKE,
    describe_skill_invocation,
)


_SINGLE_TURN = (
    '[IMPORTANT: The user has invoked the "work" skill, indicating they want '
    'you to follow its instructions. The full skill content is loaded below.]\n\n'
    'Skill body that must not become a preview.\n\n'
    'The user has provided the following instruction alongside the skill invocation: '
    'fix the title leak'
)
_BUNDLE_TURN = (
    '[IMPORTANT: The user has invoked the "/work /clean" skill bundle, loading '
    '2 skills together. Treat every skill below as active guidance for this turn.]\n\n'
    'User instruction: fix the title leak\n\n'
    '[Loaded as part of the "demo" skill bundle.]\n\nSkill body'
)


def test_ignores_non_skill_content() -> None:
    assert describe_skill_invocation(None) is None
    assert describe_skill_invocation([{"type": "text", "text": "hi"}]) is None


def test_recovers_single_skill_command_and_instruction() -> None:
    assert describe_skill_invocation(_SINGLE_TURN) == "/work — fix the title leak"


def test_recovers_bundle_instruction_without_exposing_body() -> None:
    preview = describe_skill_invocation(_BUNDLE_TURN)
    assert preview == "/work /clean — fix the title leak"
    assert "Skill body" not in preview


def test_excerpt_never_joins_skill_body_to_instruction() -> None:
    excerpt = _SINGLE_TURN[:120] + SKILL_EXCERPT_JOINT + _SINGLE_TURN[-220:]
    preview = describe_skill_invocation(excerpt)
    assert preview == "/work — fix the title leak"
    assert SKILL_EXCERPT_JOINT not in preview


def test_sql_like_pattern_matches_historical_prefix() -> None:
    assert _SINGLE_TURN.startswith(SKILL_SCAFFOLD_SQL_LIKE.rstrip("%"))
    assert "%" not in SKILL_SCAFFOLD_SQL_LIKE[:-1]
    assert "_" not in SKILL_SCAFFOLD_SQL_LIKE[:-1]
