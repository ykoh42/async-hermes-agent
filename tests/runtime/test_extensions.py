"""End-to-end contracts for the retained skill and MCP extension waist."""

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest
import pytest_asyncio
from blockbuster import BlockBuster
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

from hermes_state import SessionDB
from run_agent import AIAgent
from tools import mcp_tool


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _response(response_id, *, reasoning, content="", tool_calls=None):
    return SimpleNamespace(
        id=response_id,
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls" if tool_calls else "stop",
                message=SimpleNamespace(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                    reasoning=reasoning,
                    reasoning_content=None,
                ),
            )
        ],
        usage=None,
    )


@pytest_asyncio.fixture
async def echo_mcp(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    server_script = tmp_path / "echo_server.py"
    server_script.write_text(
        "from mcp.server.fastmcp import Context, FastMCP\n"
        "from pydantic import BaseModel\n"
        "server = FastMCP('training-mcp')\n"
        "class Consent(BaseModel):\n"
        "    pass\n"
        "@server.tool()\n"
        "async def echo(value: str, ctx: Context) -> str:\n"
        "    consent = await ctx.elicit('Allow the echo observation?', Consent)\n"
        "    return f'mcp-observation:{value}:{consent.action}'\n"
        "if __name__ == '__main__':\n"
        "    server.run(transport='stdio')\n",
        encoding="utf-8",
    )

    def fail_if_threaded(*_args, **_kwargs):
        raise AssertionError("skill/MCP runtime must not call asyncio.to_thread")

    monkeypatch.setattr(asyncio, "to_thread", fail_if_threaded)
    registered = await mcp_tool.register_mcp_servers(
        {
            "training-mcp": {
                "command": sys.executable,
                "args": [str(server_script)],
                "timeout": 10,
            }
        }
    )
    try:
        yield next(name for name in registered if name.endswith("__echo"))
    finally:
        await mcp_tool.shutdown_mcp_servers()

    await asyncio.sleep(0)
    assert mcp_tool.get_registered_mcp_server_names() == set()
    assert not [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and "mcp" in task.get_name().lower()
    ]


@pytest.mark.asyncio
async def test_skill_and_stdio_mcp_calls_are_preserved_in_trajectory(
    echo_mcp, tmp_path
):
    skill_dir = tmp_path / "home" / "skills" / "trajectory-training"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: trajectory-training\n"
        "description: Preserve tool observations in trajectories.\n---\n"
        "# Trajectory Training\n\nAlways retain tool observations.\n",
        encoding="utf-8",
    )
    database = SessionDB(tmp_path / "state.db")
    elicitation_calls = []

    async def clarify(question, choices):
        elicitation_calls.append((question, choices))
        return "Approve"

    agent = AIAgent(
        api_key="synthetic",
        base_url="https://example.invalid/v1",
        model="synthetic-model",
        max_iterations=4,
        enabled_toolsets=["skills", "training-mcp"],
        quiet_mode=True,
        save_trajectories=True,
        skip_context_files=True,
        skip_memory=True,
        session_db=database,
        clarify_callback=clarify,
    )
    responses = iter(
        [
            _response(
                "skill-list",
                reasoning="discover the available skill",
                tool_calls=[_tool_call("skill-list-call", "skills_list", {})],
            ),
            _response(
                "skill-view",
                reasoning="read the selected skill",
                tool_calls=[
                    _tool_call(
                        "skill-view-call",
                        "skill_view",
                        {"name": "trajectory-training"},
                    )
                ],
            ),
            _response(
                "mcp-call",
                reasoning="call the discovered MCP tool",
                tool_calls=[
                    _tool_call(
                        "mcp-echo-call",
                        "tool_call",
                        {"name": echo_mcp, "arguments": {"value": "hello"}},
                    )
                ],
            ),
            _response(
                "final",
                reasoning="all extension observations are available",
                content="EXTENSIONS_OK",
            ),
        ]
    )

    async def model_response(*_args, **_kwargs):
        return next(responses)

    agent._execute_model_request = model_response
    agent.compression_enabled = False
    try:
        # Warm the allowed aiosqlite worker before overlapping detectors; cold
        # SessionDB startup is covered by test_session_lifecycle_does_not_block_or_leak.
        await database.session_count()
        async with (
            no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
            no_task_leaks(action=LeakAction.RAISE),
            agent,
        ):
            blockbuster = BlockBuster()
            blockbuster.activate()
            try:
                result = await agent.run_conversation(
                    "Read the trajectory-training skill, then call MCP echo."
                )
            finally:
                blockbuster.deactivate()
    finally:
        await database.close()

    assert result["final_response"] == "EXTENSIONS_OK"
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
    assert "trajectory-training" in observations[0]
    assert "Always retain tool observations" in observations[1]
    assert "mcp-observation:hello:accept" in observations[2]
    assert '<untrusted_tool_result source="mcp__training_mcp__echo">' in observations[2]
    assert elicitation_calls == [
        (
            "Allow the echo observation?\n\nApproval requested by MCP server "
            "'training-mcp'.",
            ["Approve", "Decline"],
        )
    ]

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
    assert echo_mcp in trajectory[6]["value"]
    assert "mcp-observation:hello:accept" in trajectory[7]["value"]
