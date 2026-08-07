"""Behavior contracts for native-async skill command helpers."""

import inspect
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import agent.skill_commands as skill_commands


def _write_skill(root: Path, name: str, body: str) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        f"description: Description for {name}.\n"
        "---\n\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return skill_dir


@pytest.fixture(autouse=True)
def _clear_command_cache():
    skill_commands._skill_commands = {}
    skill_commands._skill_commands_platform = None
    yield
    skill_commands._skill_commands = {}
    skill_commands._skill_commands_platform = None


def test_io_backed_skill_command_interfaces_are_coroutines():
    for name in (
        "scan_skill_commands",
        "get_skill_commands",
        "reload_skills",
        "resolve_skill_command_key",
        "build_skill_invocation_message",
        "split_stacked_skill_commands",
        "build_stacked_skill_invocation_message",
        "build_preloaded_skills_prompt",
    ):
        assert inspect.iscoroutinefunction(getattr(skill_commands, name)), name


@pytest.mark.asyncio
async def test_scan_resolve_and_build_single_skill_message(tmp_path):
    skill_dir = _write_skill(
        tmp_path,
        "code-review",
        "Review files under ${HERMES_SKILL_DIR} for ${HERMES_SESSION_ID}.",
    )
    with (
        patch("tools.skills_tool.SKILLS_DIR", tmp_path),
        patch(
            "tools.skills_tool._external_skills_dirs",
            AsyncMock(return_value=[]),
        ),
        patch(
            "tools.skills_tool._get_disabled_skill_names",
            AsyncMock(return_value=set()),
        ),
    ):
        commands = await skill_commands.scan_skill_commands()
        assert list(commands) == ["/code-review"]
        assert await skill_commands.resolve_skill_command_key("code_review") == (
            "/code-review"
        )
        message = await skill_commands.build_skill_invocation_message(
            "/code-review",
            "check cancellation safety",
            task_id="turn-7",
        )

    assert message is not None
    assert f"Review files under {skill_dir} for turn-7." in message
    assert "check cancellation safety" in message
    assert skill_commands.extract_user_instruction_from_skill_message(message) == (
        "check cancellation safety"
    )


@pytest.mark.asyncio
async def test_stacked_skills_preserve_order_and_memory_instruction(tmp_path):
    _write_skill(tmp_path, "alpha", "Alpha guidance.")
    _write_skill(tmp_path, "beta", "Beta guidance.")
    with (
        patch("tools.skills_tool.SKILLS_DIR", tmp_path),
        patch(
            "tools.skills_tool._external_skills_dirs",
            AsyncMock(return_value=[]),
        ),
        patch(
            "tools.skills_tool._get_disabled_skill_names",
            AsyncMock(return_value=set()),
        ),
    ):
        await skill_commands.scan_skill_commands()
        extra, instruction = await skill_commands.split_stacked_skill_commands(
            "/beta investigate the failure"
        )
        result = await skill_commands.build_stacked_skill_invocation_message(
            ["/alpha", *extra],
            instruction,
            task_id="turn-8",
        )

    assert extra == ["/beta"]
    assert instruction == "investigate the failure"
    assert result is not None
    message, loaded, missing = result
    assert loaded == ["alpha", "beta"]
    assert missing == []
    assert message.index("Alpha guidance.") < message.index("Beta guidance.")
    assert skill_commands.extract_user_instruction_from_skill_message(message) == (
        "investigate the failure"
    )


@pytest.mark.asyncio
async def test_preloaded_skills_respect_disabled_names(tmp_path):
    _write_skill(tmp_path, "enabled", "Enabled guidance.")
    _write_skill(tmp_path, "disabled", "Disabled guidance.")
    with (
        patch("tools.skills_tool.SKILLS_DIR", tmp_path),
        patch(
            "tools.skills_tool._external_skills_dirs",
            AsyncMock(return_value=[]),
        ),
        patch(
            "tools.skills_tool._get_disabled_skill_names",
            AsyncMock(return_value={"disabled"}),
        ),
    ):
        prompt, loaded, missing = await skill_commands.build_preloaded_skills_prompt(
            ["enabled", "disabled"], task_id="session"
        )

    assert loaded == ["enabled"]
    assert missing == ["disabled"]
    assert "Enabled guidance." in prompt
    assert "Disabled guidance." not in prompt


@pytest.mark.asyncio
async def test_reload_reports_added_and_removed_skills(tmp_path):
    alpha = _write_skill(tmp_path, "alpha", "Alpha guidance.")
    with (
        patch("tools.skills_tool.SKILLS_DIR", tmp_path),
        patch(
            "tools.skills_tool._external_skills_dirs",
            AsyncMock(return_value=[]),
        ),
        patch(
            "tools.skills_tool._get_disabled_skill_names",
            AsyncMock(return_value=set()),
        ),
    ):
        await skill_commands.scan_skill_commands()
        (alpha / "SKILL.md").unlink()
        _write_skill(tmp_path, "beta", "Beta guidance.")
        diff = await skill_commands.reload_skills()

    assert [entry["name"] for entry in diff["added"]] == ["beta"]
    assert [entry["name"] for entry in diff["removed"]] == ["alpha"]
    assert diff["total"] == 1


@pytest.mark.asyncio
async def test_skill_config_is_injected_from_async_config(tmp_path):
    skill_dir = tmp_path / "configured"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: configured\n"
        "description: Configured skill.\n"
        "metadata:\n"
        "  hermes:\n"
        "    config:\n"
        "      - key: wiki.path\n"
        "        description: Wiki path\n"
        "        default: ~/wiki\n"
        "---\n\n"
        "Use the configured wiki.\n",
        encoding="utf-8",
    )
    with (
        patch("tools.skills_tool.SKILLS_DIR", tmp_path),
        patch(
            "tools.skills_tool._external_skills_dirs",
            AsyncMock(return_value=[]),
        ),
        patch(
            "tools.skills_tool._get_disabled_skill_names",
            AsyncMock(return_value=set()),
        ),
        patch(
            "hermes_cli.config.load_config_readonly",
            AsyncMock(
                return_value={
                    "skills": {"config": {"wiki": {"path": "/srv/wiki"}}}
                }
            ),
        ),
    ):
        await skill_commands.scan_skill_commands()
        message = await skill_commands.build_skill_invocation_message(
            "/configured"
        )

    assert message is not None
    assert "wiki.path = /srv/wiki" in message
