"""Prompt-byte parity for the retained computer_use tool surface."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.prompt_builder import COMPUTER_USE_GUIDANCE, computer_use_guidance
from agent.system_prompt import build_system_prompt_parts


@pytest.mark.parametrize(
    ("platform_name", "length", "digest", "os_name", "save", "app"),
    [
        (
            "darwin",
            5267,
            "1f00d63b8fd901e4cef25c7ca485460ca9d94f375da0470b48d83de5b195f531",
            "macOS",
            "cmd+s",
            "Safari",
        ),
        (
            "win32",
            5382,
            "30b82ccd1ddf16c8f3bb1e7f40dcd817f284cceb2db6336d3e87fd53d9d2584b",
            "Windows",
            "ctrl+s",
            "Chrome",
        ),
        (
            "linux",
            5243,
            "f86b527057069e96c242f5b2b7b6da8ad709326eb20d4b3ae5cf86e02246daac",
            "Linux",
            "ctrl+s",
            "Firefox",
        ),
    ],
)
def test_platform_guidance_bytes_match_upstream(
    platform_name,
    length,
    digest,
    os_name,
    save,
    app,
):
    guidance = computer_use_guidance(platform_name)
    encoded = guidance.encode("utf-8")
    assert len(encoded) == length
    assert hashlib.sha256(encoded).hexdigest() == digest
    assert f"# Computer Use ({os_name} background control)" in guidance
    assert f"keys='{save}'" in guidance
    assert f"app='{app}'" in guidance


def test_backwards_compatible_constant_is_exact_macos_variant():
    assert COMPUTER_USE_GUIDANCE.encode("utf-8") == computer_use_guidance(
        "darwin"
    ).encode("utf-8")


def _agent(valid_tool_names):
    return SimpleNamespace(
        load_soul_identity=False,
        skip_context_files=True,
        valid_tool_names=valid_tool_names,
        _task_completion_guidance=False,
        _parallel_tool_call_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        model="test-model",
        provider="test-provider",
        platform="",
        pass_session_id=False,
        session_id="",
    )


async def _stable_prompt(valid_tool_names):
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        return (
            await build_system_prompt_parts(_agent(valid_tool_names))
        )["stable"]


@pytest.mark.asyncio
async def test_guidance_is_injected_once_only_when_tool_is_available(monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    guidance = computer_use_guidance("linux")

    enabled = await _stable_prompt(["computer_use"])
    disabled = await _stable_prompt([])

    assert enabled.count(guidance) == 1
    assert guidance.encode("utf-8") in enabled.encode("utf-8")
    assert "# Computer Use (" not in disabled
