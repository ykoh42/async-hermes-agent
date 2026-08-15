"""Native async file-write safety and atomicity tests."""

import json
import os

import pytest

from agent.file_safety import (
    get_write_denied_error,
    is_write_approval_required,
    is_write_denied,
)
from tools.file_operations import _strip_bom
from tools.file_tools import _check_sensitive_path, patch_tool, write_file_tool


pytestmark = pytest.mark.asyncio


async def test_regular_temp_file_allowed(tmp_path):
    assert await get_write_denied_error(str(tmp_path / "regular.txt")) is None


async def test_credential_path_denied():
    error = await get_write_denied_error(os.path.expanduser("~/.ssh/id_rsa"))
    assert error is not None
    assert "protected system/credential file" in error


async def test_ssh_config_is_approval_gated_not_hard_denied(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / ".ssh" / "config"
    assert await is_write_denied(str(target)) is False
    assert await get_write_denied_error(str(target)) is None
    assert await is_write_approval_required(str(target)) is True


@pytest.mark.parametrize(
    "name",
    ["id_rsa", "id_ed25519", "authorized_keys", "id_rsa.pub"],
)
async def test_other_ssh_files_remain_hard_denied(tmp_path, monkeypatch, name):
    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / ".ssh" / name
    assert await is_write_denied(str(target)) is True
    assert await is_write_approval_required(str(target)) is False


async def test_safe_write_root_bounds_native_write(tmp_path, monkeypatch):
    safe_root = tmp_path / "workspace"
    safe_root.mkdir()
    inside = safe_root / "inside.txt"
    outside = tmp_path / "outside.txt"
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))

    allowed = json.loads(await write_file_tool(str(inside), "inside"))
    denied = json.loads(await write_file_tool(str(outside), "outside"))

    assert "error" not in allowed
    assert inside.read_text() == "inside"
    assert "outside HERMES_WRITE_SAFE_ROOT" in denied["error"]
    assert not outside.exists()


@pytest.mark.parametrize(
    "path",
    [
        "/etc/hosts",
        "/private/etc/hosts",
        "/private/var/db/example",
        "/boot/grub/grub.cfg",
    ],
)
async def test_sensitive_system_paths_blocked(path):
    assert await _check_sensitive_path(path) is not None


async def test_safe_path_passes_sensitive_guard(tmp_path):
    assert await _check_sensitive_path(str(tmp_path / "safe.txt")) is None


async def test_native_write_uses_atomic_replace(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("before")
    inode_before = target.stat().st_ino

    result = json.loads(await write_file_tool(str(target), "after"))

    assert "error" not in result
    assert target.read_text() == "after"
    assert target.stat().st_ino != inode_before
    assert not list(tmp_path.glob(".*.hermes-*.tmp"))


async def test_native_patch_preserves_line_endings(tmp_path):
    target = tmp_path / "windows.txt"
    target.write_bytes(b"first\r\nsecond\r\n")

    result = json.loads(
        await patch_tool(
            mode="replace",
            path=str(target),
            old_string="second",
            new_string="changed",
        )
    )

    assert result["success"] is True
    assert target.read_bytes() == b"first\r\nchanged\r\n"


async def test_strip_bom_is_pure_and_precise():
    assert _strip_bom("\ufeffhello") == ("hello", True)
    assert _strip_bom("a\ufeffb") == ("a\ufeffb", False)
