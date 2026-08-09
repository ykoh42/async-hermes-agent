"""Contracts retained from the v2026.8.3 file-operations implementation."""

import asyncio
import inspect
import json
import os

import pytest

from agent.file_safety import is_write_denied
import tools.file_operations as file_operations_module

from tools.file_operations import (
    FileOperations,
    LintResult,
    PatchResult,
    ReadResult,
    SearchMatch,
    SearchResult,
    ShellFileOperations,
    WriteResult,
    normalize_read_pagination,
)


def test_module_import_does_not_create_unawaited_coroutines():
    assert not [
        name
        for name, value in vars(file_operations_module).items()
        if inspect.iscoroutine(value)
    ]


def test_result_dataclasses_preserve_upstream_shapes():
    assert "error" not in ReadResult().to_dict()
    assert "similar_files" not in ReadResult().to_dict()
    assert WriteResult(bytes_written=100).to_dict()["bytes_written"] == 100
    assert PatchResult(success=True, files_modified=["a.py"]).to_dict() == {
        "success": True,
        "files_modified": ["a.py"],
    }
    assert LintResult(skipped=True, message="none").to_dict() == {
        "status": "skipped",
        "message": "none",
    }


def test_search_result_densification_preserves_paths_with_spaces():
    result = SearchResult(
        matches=[
            SearchMatch("dir with spaces/a.py", line, f"value {line}")
            for line in range(1, 6)
        ],
        total_count=5,
    ).to_dict(densify=True)

    assert result["matches_text"].startswith("dir with spaces/a.py\n")
    assert "matches" not in result


def test_read_pagination_clamps_invalid_values():
    offset, limit = normalize_read_pagination("bad", -5)
    assert offset == 1
    assert limit == 1


def test_upstream_file_operations_surface_is_native_async():
    method_names = {
        "read_file",
        "read_file_raw",
        "write_file",
        "patch_replace",
        "patch_v4a",
        "delete_file",
        "delete_path",
        "move_file",
        "search",
    }

    for cls in (FileOperations, ShellFileOperations):
        for method_name in method_names:
            assert inspect.iscoroutinefunction(getattr(cls, method_name)), (
                cls.__name__,
                method_name,
            )

    assert str(inspect.signature(ShellFileOperations.read_file)) == (
        "(self, path: str, offset: int = 1, limit: int = 2000) -> "
        "tools.file_operations.ReadResult"
    )
    assert str(inspect.signature(ShellFileOperations.search)) == (
        "(self, pattern: str, path: str = '.', target: str = 'content', "
        "file_glob: Optional[str] = None, limit: int = 50, offset: int = 0, "
        "output_mode: str = 'content', context: int = 0) -> "
        "tools.file_operations.SearchResult"
    )


@pytest.mark.asyncio
async def test_shell_file_operations_real_native_async_workflow(tmp_path):
    from tools.environments.local import LocalEnvironment

    env = LocalEnvironment(cwd=str(tmp_path))
    ops = ShellFileOperations(env)
    try:
        write = await ops.write_file("notes.txt", "old value\n")
        assert write.error is None
        assert write.verified is True

        read = await ops.read_file("notes.txt")
        assert read.to_dict()["content"] == "1|old value"

        patch = await ops.patch_replace("notes.txt", "old value", "new value")
        assert patch.success is True
        assert patch.files_modified == ["notes.txt"]

        search = await ops.search("new value", path="notes.txt")
        assert search.to_dict()["matches"] == [
            {"path": "notes.txt", "line": 1, "content": "new value"}
        ]

        delete = await ops.delete_file("notes.txt")
        assert delete.error is None
        assert not (tmp_path / "notes.txt").exists()
    finally:
        await env.cleanup()


@pytest.mark.asyncio
async def test_shell_file_operations_tracks_live_environment_cwd(tmp_path):
    from tools.environments.local import LocalEnvironment

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "target.txt").write_text("first\n")
    (second / "target.txt").write_text("second\n")

    env = LocalEnvironment(cwd=str(first))
    ops = ShellFileOperations(env, cwd=str(first))
    try:
        result = await env.execute(f"cd {second}")
        assert result["returncode"] == 0
        assert env.cwd == str(second)
        assert ops.cwd == str(first)

        read = await ops.read_file("target.txt")
        assert read.content == "1|second"
    finally:
        await env.cleanup()


@pytest.mark.asyncio
async def test_shell_file_operations_preserves_external_cancellation():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class BlockingEnvironment:
        cwd = "/"

        async def execute(self, command, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    task = asyncio.create_task(
        ShellFileOperations(BlockingEnvironment()).read_file("x")
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_shell_file_operations_rejects_sync_backend_without_thread_fallback():
    class SyncEnvironment:
        cwd = "/"

        def execute(self, command, **kwargs):
            return {"output": "", "returncode": 0}

    with pytest.raises(TypeError, match="can't be used in 'await' expression"):
        await ShellFileOperations(SyncEnvironment()).read_file("x")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "relative_path",
    [
        ".ssh/authorized_keys",
        ".netrc",
        ".pgpass",
        ".npmrc",
        ".pypirc",
        ".aws/credentials",
    ],
)
async def test_upstream_sensitive_home_paths_are_write_denied(
    tmp_path, monkeypatch, relative_path
):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert await is_write_denied(str(tmp_path / relative_path))


@pytest.mark.asyncio
async def test_upstream_oauth_path_is_write_denied(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert await is_write_denied(str(tmp_path / ".anthropic_oauth.json"))


@pytest.mark.asyncio
async def test_upstream_profile_and_root_token_pairing_paths_are_denied(
    tmp_path, monkeypatch
):
    root = tmp_path / "hermes"
    profile = root / "profiles" / "coder"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))

    for path in (
        profile / "mcp-tokens" / "tok.json",
        root / "mcp-tokens" / "tok.json",
        root / "mcp-tokens",
        profile / "pairing" / "telegram-approved.json",
        profile / "pairing",
        root / "pairing" / "telegram-approved.json",
        root / "pairing",
    ):
        assert await is_write_denied(str(path)), path


@pytest.mark.asyncio
@pytest.mark.parametrize("test_umask", [0o022, 0o002, 0o077])
async def test_new_file_uses_umask_default_permissions(tmp_path, test_umask):
    from tools.file_tools import write_file_tool

    target = tmp_path / "new_file.txt"
    old_umask = os.umask(test_umask)
    try:
        result = json.loads(await write_file_tool(str(target), "test content\n"))
    finally:
        os.umask(old_umask)

    assert "error" not in result
    assert target.stat().st_mode & 0o777 == 0o666 & ~test_umask


@pytest.mark.asyncio
async def test_atomic_write_follows_symlink_and_preserves_link(tmp_path):
    from tools.file_tools import write_file_tool

    target = tmp_path / "real.txt"
    link = tmp_path / "link.txt"
    target.write_text("original\n")
    link.symlink_to(target)

    result = json.loads(await write_file_tool(str(link), "updated\n"))

    assert "error" not in result
    assert link.is_symlink()
    assert target.read_text() == "updated\n"
    assert os.path.realpath(link) == str(target)


@pytest.mark.asyncio
async def test_atomic_write_through_broken_symlink_creates_target(tmp_path):
    from tools.file_tools import write_file_tool

    target = tmp_path / "target.txt"
    link = tmp_path / "broken.lnk"
    link.symlink_to(target)

    result = json.loads(await write_file_tool(str(link), "data\n"))

    assert "error" not in result
    assert link.is_symlink()
    assert target.read_text() == "data\n"


@pytest.mark.asyncio
async def test_overwrite_preserves_existing_mode(tmp_path):
    from tools.file_tools import write_file_tool

    target = tmp_path / "executable.sh"
    target.write_text("old\n")
    target.chmod(0o751)

    result = json.loads(await write_file_tool(str(target), "new\n"))

    assert "error" not in result
    assert target.stat().st_mode & 0o777 == 0o751


@pytest.mark.asyncio
async def test_non_utf8_content_is_reported_as_binary(tmp_path):
    from tools.file_tools import read_file_tool

    target = tmp_path / "latin1.txt"
    target.write_bytes(b"caf\xe9 r\xe9sum\xe9\n")

    result = json.loads(await read_file_tool(str(target)))

    assert result["is_binary"] is True
    assert "Binary file" in result["error"]


@pytest.mark.asyncio
async def test_plain_utf8_content_is_not_reported_as_binary(tmp_path):
    from tools.file_tools import read_file_tool

    target = tmp_path / "utf8.txt"
    target.write_text("café résumé\n")

    result = json.loads(await read_file_tool(str(target)))

    assert "error" not in result
    assert result.get("is_binary") is not True
    assert result["content"] == "1|café résumé"
