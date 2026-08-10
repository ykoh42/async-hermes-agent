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
