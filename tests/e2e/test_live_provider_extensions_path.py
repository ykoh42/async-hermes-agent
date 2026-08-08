"""Opt-in live acceptance test for provider → skill → MCP → trajectory.

Run with an authenticated provider, for example::

    HERMES_LIVE_TESTS=1 HERMES_LIVE_PROVIDER=copilot \
      pytest -q tests/e2e/test_live_provider_extensions_path.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction


LIVE = os.environ.get("HERMES_LIVE_TESTS") == "1"
PROVIDER = (os.environ.get("HERMES_LIVE_PROVIDER") or "copilot").strip().lower()
DEFAULT_MODELS = {
    "copilot": "gpt-4.1",
    "openrouter": "google/gemini-2.5-flash",
}

pytestmark = pytest.mark.skipif(
    not LIVE,
    reason="live-only — set HERMES_LIVE_TESTS=1",
)


@pytest.mark.asyncio
async def test_live_provider_skill_mcp_observations_and_trajectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tools import mcp_tool

    monkeypatch.chdir(tmp_path)
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    model = os.environ.get("HERMES_LIVE_MODEL") or DEFAULT_MODELS.get(PROVIDER)
    if not model:
        pytest.fail(
            f"No default model for live provider {PROVIDER!r}; set HERMES_LIVE_MODEL"
        )

    skill_dir = hermes_home / "skills" / "live-extension"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: live-extension\n"
        "description: Expose a unique live extension observation.\n"
        "---\n"
        "# Live Extension\n\n"
        "The required observation is LIVE_SKILL_OBSERVATION.\n",
        encoding="utf-8",
    )
    server_script = tmp_path / "live_mcp_server.py"
    server_script.write_text(
        "from mcp.server.fastmcp import FastMCP\n"
        "server = FastMCP('live-extension-mcp')\n"
        "@server.tool()\n"
        "async def echo(value: str) -> str:\n"
        "    return f'LIVE_MCP_OBSERVATION:{value}'\n"
        "if __name__ == '__main__':\n"
        "    server.run(transport='stdio')\n",
        encoding="utf-8",
    )

    database = SessionDB(tmp_path / "state.db")
    kwargs = {
        "provider": PROVIDER,
        "model": model,
        "max_iterations": 6,
        "quiet_mode": True,
        "skip_context_files": True,
        "skip_memory": True,
        "enabled_toolsets": ["skills", "live-extension-mcp"],
        "save_trajectories": True,
        "session_db": database,
    }
    if PROVIDER == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            pytest.fail("OPENROUTER_API_KEY is required for the live OpenRouter test")
        kwargs.update(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    result = None
    registered: list[str] = []
    agent = None
    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.25),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        try:
            registered = await mcp_tool.register_mcp_servers(
                {
                    "live-extension-mcp": {
                        "command": sys.executable,
                        "args": [str(server_script)],
                        "timeout": 10,
                    }
                }
            )
            echo_tool = next(name for name in registered if name.endswith("__echo"))
            agent = AIAgent(**kwargs)
            async with agent:
                result = await agent.run_conversation(
                    "Use tools in this exact order: "
                    "(1) call skills_list, "
                    "(2) call skill_view for live-extension, "
                    "(3) call tool_call with name "
                    f"{echo_tool} and arguments {{\"value\": \"LIVE_MCP\"}}. "
                    "Do not skip a tool or infer its observation. After reading all "
                    "three tool results, reply exactly LIVE_EXTENSIONS_FINAL."
                )
        finally:
            if agent is not None:
                await agent.close()
            await mcp_tool.shutdown_mcp_servers()
            await database.close()

    assert result is not None
    assert result["completed"] is True
    assert result["final_response"].strip() == "LIVE_EXTENSIONS_FINAL"
    assert [message["role"] for message in result["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    observations = [
        message["content"]
        for message in result["messages"]
        if message["role"] == "tool"
    ]
    assert "live-extension" in observations[0]
    assert "LIVE_SKILL_OBSERVATION" in observations[1]
    assert "LIVE_MCP_OBSERVATION:LIVE_MCP" in observations[2]

    rows = [
        json.loads(line)
        for line in (tmp_path / "trajectory_samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 1
    trajectory = rows[0]["conversations"]
    assert [turn["from"] for turn in trajectory] == [
        "system",
        "human",
        "gpt",
        "tool",
        "gpt",
        "tool",
        "gpt",
        "tool",
        "gpt",
    ]
    assert '"name": "skills_list"' in trajectory[2]["value"]
    assert '"name": "skill_view"' in trajectory[4]["value"]
    assert '"name": "tool_call"' in trajectory[6]["value"]
    assert "LIVE_MCP_OBSERVATION:LIVE_MCP" in trajectory[7]["value"]

    await asyncio.sleep(0)
    assert mcp_tool.get_registered_mcp_server_names() == set()
