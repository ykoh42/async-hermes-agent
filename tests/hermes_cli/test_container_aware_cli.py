"""Tests for container-aware CLI routing (NixOS container mode).

When container.enable = true in the NixOS module, the activation script
writes a .container-mode metadata file. The host CLI detects this and
execs into the container instead of running locally.
"""
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.config import (
    get_container_exec_info,
)


# =============================================================================
# get_container_exec_info
# =============================================================================


@pytest.fixture
def container_env(tmp_path, monkeypatch):
    """Set up a fake HERMES_HOME with .container-mode file."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_DEV", raising=False)

    container_mode = hermes_home / ".container-mode"
    container_mode.write_text(
        "# Written by NixOS activation script. Do not edit manually.\n"
        "backend=podman\n"
        "container_name=hermes-agent\n"
        "exec_user=hermes\n"
        "hermes_bin=/data/current-package/bin/hermes\n"
    )
    return hermes_home


def test_get_container_exec_info_returns_metadata(container_env):
    """Reads .container-mode and returns all fields including exec_user."""
    with patch("hermes_constants.is_container", return_value=False):
        info = get_container_exec_info()

    assert info is not None
    assert info["backend"] == "podman"
    assert info["container_name"] == "hermes-agent"
    assert info["exec_user"] == "hermes"
    assert info["hermes_bin"] == "/data/current-package/bin/hermes"








# =============================================================================
# _exec_in_container
# =============================================================================


@pytest.fixture
def docker_container_info():
    return {
        "backend": "docker",
        "container_name": "hermes-agent",
        "exec_user": "hermes",
        "hermes_bin": "/data/current-package/bin/hermes",
    }
@pytest.fixture
def podman_container_info():
    return {
        "backend": "podman",
        "container_name": "hermes-agent",
        "exec_user": "hermes",
        "hermes_bin": "/data/current-package/bin/hermes",
    }


