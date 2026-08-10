"""First-use import coverage for the process-level async bootstrap."""

import subprocess
import sys


def _run_fresh_interpreter(source: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_process_bootstrap_preloads_httpcore_in_a_fresh_interpreter():
    _run_fresh_interpreter(
        "import sys; import agent.process_bootstrap; "
        "assert 'httpcore' in sys.modules"
    )


def test_mini_swe_runner_preloads_client_dependencies_before_event_loop():
    _run_fresh_interpreter(
        """
import asyncio
import os
import sys
from unittest.mock import patch

from mini_swe_runner import MiniSWERunner

assert "agent.process_bootstrap" in sys.modules


async def main():
    runner = MiniSWERunner(
        model="test/model",
        base_url="http://127.0.0.1:1/v1",
        api_key="test-key",
    )
    with patch.object(
        os,
        "listdir",
        side_effect=AssertionError("synchronous directory scan in event loop"),
    ):
        client = await runner._ensure_client()
    assert client is runner.client
    await runner._close_owned_client()
    assert runner.client is None


asyncio.run(main())
"""
    )


def test_agent_provider_init_does_not_cold_import_inside_event_loop():
    _run_fresh_interpreter(
        """
import asyncio
import builtins
import os
import sys
import tempfile

workspace = tempfile.TemporaryDirectory(prefix="async-hermes-import-audit-")
os.environ["HERMES_HOME"] = workspace.name

from run_agent import AIAgent

original_import = builtins.__import__
cold_imports = []


def audit_import(name, globals=None, locals=None, fromlist=(), level=0):
    loaded_before = set(sys.modules)
    result = original_import(name, globals, locals, fromlist, level)
    cold_imports.extend(sorted(set(sys.modules) - loaded_before))
    return result


async def main():
    agent = None
    builtins.__import__ = audit_import
    try:
        agent = AIAgent(
            provider="openrouter",
            model="test/model",
            base_url="http://127.0.0.1:1/v1",
            api_key="test-key",
            enabled_toolsets=["terminal"],
            quiet_mode=True,
        )
        assert await agent._ensure_provider_runtime()
    finally:
        try:
            if agent is not None:
                await agent.close()
        finally:
            builtins.__import__ = original_import


asyncio.run(main())
workspace.cleanup()
assert not cold_imports, cold_imports
"""
    )


def test_terminal_lifecycle_does_not_cold_import_inside_event_loop():
    _run_fresh_interpreter(
        """
import asyncio
import builtins
import json
import os
import sys
import tempfile

workspace = tempfile.TemporaryDirectory(prefix="async-hermes-terminal-import-")
os.environ["HERMES_HOME"] = workspace.name

from tools.terminal_tool import cleanup_vm, terminal_tool

original_import = builtins.__import__
cold_imports = []


def audit_import(name, globals=None, locals=None, fromlist=(), level=0):
    loaded_before = set(sys.modules)
    result = original_import(name, globals, locals, fromlist, level)
    cold_imports.extend(sorted(set(sys.modules) - loaded_before))
    return result


async def main():
    builtins.__import__ = audit_import
    try:
        result = await terminal_tool(
            "printf ASYNC_COLD_TOOL",
            task_id="cold-import-audit",
        )
        assert json.loads(result)["output"] == "ASYNC_COLD_TOOL"
    finally:
        try:
            await cleanup_vm("cold-import-audit")
        finally:
            builtins.__import__ = original_import


asyncio.run(main())
workspace.cleanup()
assert not cold_imports, cold_imports
"""
    )


def test_retained_state_memory_skill_and_mcp_paths_do_not_cold_import():
    _run_fresh_interpreter(
        """
import asyncio
import builtins
import json
import os
import sys
import tempfile
from pathlib import Path

workspace = tempfile.TemporaryDirectory(prefix="async-hermes-retained-import-")
os.environ["HERMES_HOME"] = workspace.name
skill_dir = Path(workspace.name) / "skills" / "cold-skill"
skill_dir.mkdir(parents=True)
(skill_dir / "SKILL.md").write_text(
    "---\\nname: cold-skill\\ndescription: cold audit\\n---\\n"
    "Use native async.\\n"
)

from hermes_state import SessionDB
from tools.mcp_tool import discover_mcp_tools, shutdown_mcp_servers
from tools.memory_tool import MemoryStore
from tools.skills_tool import skill_view, skills_list

original_import = builtins.__import__
cold_imports = []


def audit_import(name, globals=None, locals=None, fromlist=(), level=0):
    loaded_before = set(sys.modules)
    result = original_import(name, globals, locals, fromlist, level)
    cold_imports.extend(sorted(set(sys.modules) - loaded_before))
    return result


async def main():
    database = SessionDB(Path(workspace.name) / "state.db")
    store = MemoryStore()
    builtins.__import__ = audit_import
    try:
        session_id = await database.create_session("cold-session", "test")
        assert session_id == "cold-session"
        assert (await database.get_session(session_id))["id"] == session_id
        await store.load_from_disk()
        assert (await store.add("memory", "cold memory"))["success"]
        assert json.loads(await skills_list())["count"] == 1
        assert json.loads(await skill_view("cold-skill"))["success"]
        assert await discover_mcp_tools() == []
    finally:
        try:
            await shutdown_mcp_servers()
            await database.close()
        finally:
            builtins.__import__ = original_import


asyncio.run(main())
workspace.cleanup()
assert not cold_imports, cold_imports
"""
    )


def test_subagent_plugin_lifecycle_does_not_use_source_file_loader():
    _run_fresh_interpreter(
        """
import asyncio
import importlib._bootstrap_external as bootstrap_external
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

workspace = tempfile.TemporaryDirectory(prefix="async-hermes-subagent-import-")
os.environ["HERMES_HOME"] = workspace.name

from agent.subagent_lifecycle import SubagentLaunchRequest, SubagentLifecycleService
import tools.delegate_tool as delegate_tool


class Child:
    _subagent_id = "sa-cold-audit"
    _delegate_role = "leaf"
    _delegate_depth = 1
    provider = "test"
    model = "test-model"
    session_id = "child-cold-audit"

    async def close(self):
        return None


async def build_child(**_kwargs):
    return Child()


async def run_child(*_args, **_kwargs):
    return {
        "task_index": 0,
        "status": "completed",
        "summary": "ok",
        "api_calls": 1,
        "duration_seconds": 0.01,
        "_child_role": "leaf",
        "_child_cost_usd": 0.0,
    }


delegate_tool._build_child_agent = build_child
delegate_tool._run_single_child = run_child

repo_root = Path(delegate_tool.__file__).resolve().parent.parent
venv_root = Path(sys.prefix).resolve()
source_reads = []
original_get_data = bootstrap_external.SourceFileLoader.get_data


def audit_get_data(loader, path):
    resolved = Path(os.fsdecode(path)).resolve()
    if resolved.is_relative_to(repo_root) and not resolved.is_relative_to(venv_root):
        source_reads.append(str(resolved))
    return original_get_data(loader, path)


async def main():
    parent = SimpleNamespace(
        session_id="parent-cold-audit",
        enabled_toolsets=["file"],
        _current_turn_id="turn-cold-audit",
        session_estimated_cost_usd=0.0,
        session_cost_source="none",
        session_cost_status="unknown",
    )
    service = SubagentLifecycleService(lambda: parent)
    bootstrap_external.SourceFileLoader.get_data = audit_get_data
    try:
        handle = await service.launch(SubagentLaunchRequest(goal="audit"))
        terminal = await service.wait(handle, timeout_seconds=5)
        assert terminal.completed
        assert "plugins.browser.browser_use.provider" in sys.modules
    finally:
        bootstrap_external.SourceFileLoader.get_data = original_get_data


asyncio.run(main())
workspace.cleanup()
assert not source_reads, source_reads
"""
    )


def test_process_bootstrap_preloads_installed_optional_provider_sdks():
    _run_fresh_interpreter(
        """
import importlib.util
import sys

def present(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False

names = (
    "aiobotocore.session",
    "azure.identity.aio",
    "google.auth.transport.aiohttp_requests",
    "supermemory",
    "honcho",
    "psycopg",
    "psycopg_pool",
    "qdrant_client",
    "ollama",
    "parallel",
)
available = {name for name in names if present(name)}
import agent.process_bootstrap
assert available <= sys.modules.keys()
"""
    )


def test_provider_adapters_do_not_defer_installed_sdk_imports():
    _run_fresh_interpreter(
        """
import importlib.util
import sys

def present(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False

bedrock = present("aiobotocore.session")
entra = present("azure.identity.aio")
import agent.bedrock_adapter
import agent.azure_identity_adapter
assert not bedrock or "aiobotocore.session" in sys.modules
assert not entra or "azure.identity.aio" in sys.modules
"""
    )


def test_memory_plugins_do_not_defer_installed_sdk_imports():
    _run_fresh_interpreter(
        """
import importlib.util
import sys

def present(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False

expected = {
    name
    for name in (
        "supermemory",
        "honcho",
        "psycopg",
        "psycopg_pool",
        "qdrant_client",
        "ollama",
    )
    if present(name)
}
import plugins.memory.supermemory
import plugins.memory.honcho.client
import plugins.memory.mem0._native_vector
import plugins.memory.mem0._native_oss
assert expected <= sys.modules.keys()
"""
    )


def test_parallel_web_plugin_does_not_defer_installed_sdk_import():
    _run_fresh_interpreter(
        """
import importlib.util
import sys

try:
    installed = importlib.util.find_spec("parallel") is not None
except (ImportError, AttributeError, ValueError):
    installed = False
import plugins.web.parallel.provider
assert not installed or "parallel" in sys.modules
"""
    )
