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
