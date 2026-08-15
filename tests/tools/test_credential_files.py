"""Async parity tests for credential, skill, and cache passthrough files."""

from __future__ import annotations

import asyncio
import inspect
import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio
from blockbuster import BlockBuster

import tools.credential_files as credential_files
from agent import secret_scope
from hermes_constants import (
    reset_hermes_home_override,
    set_hermes_home_override,
)
from tools.credential_files import (
    clear_credential_files,
    from_agent_visible_cache_path,
    get_cache_directory_mounts,
    get_credential_file_mounts,
    get_skills_directory_mount,
    iter_cache_files,
    iter_skills_files,
    map_cache_path_to_container,
    register_credential_file,
    register_credential_files,
    to_agent_visible_cache_path,
)


@pytest_asyncio.fixture(autouse=True)
async def _clean_state():
    clear_credential_files()
    credential_files._config_files = None
    credential_files._config_files_by_home.clear()
    credential_files._config_file_locks.clear()
    credential_files._cache_mounts_by_scope.clear()
    yield
    clear_credential_files()
    credential_files._config_files = None
    credential_files._config_files_by_home.clear()
    credential_files._config_file_locks.clear()
    credential_files._cache_mounts_by_scope.clear()
    safe_dir = credential_files._safe_skills_tempdir
    if safe_dir is not None:
        await credential_files._remove_tree(safe_dir)
        credential_files._safe_skills_tempdir = None


def test_io_api_is_async_and_cache_translation_stays_sync():
    for function in (
        register_credential_file,
        register_credential_files,
        get_credential_file_mounts,
        get_skills_directory_mount,
        iter_skills_files,
        get_cache_directory_mounts,
        iter_cache_files,
    ):
        assert inspect.iscoroutinefunction(function), function.__name__
    for function in (
        clear_credential_files,
        map_cache_path_to_container,
        from_agent_visible_cache_path,
        to_agent_visible_cache_path,
    ):
        assert not inspect.iscoroutinefunction(function), function.__name__


@pytest.mark.asyncio
async def test_registers_dict_entry_and_path_precedes_name(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "real.json").write_text("{}")
    monkeypatch.setenv("HERMES_HOME", str(home))

    missing = await register_credential_files(
        [{"path": "real.json", "name": "wrong.json"}]
    )
    mounts = await get_credential_file_mounts()

    assert missing == []
    assert mounts == [
        {
            "host_path": str(home / "real.json"),
            "container_path": "/root/.hermes/real.json",
        }
    ]


@pytest.mark.asyncio
async def test_custom_container_base_and_deleted_file_recheck(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    token = home / "token.json"
    token.write_text("{}")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert await register_credential_file("token.json", "/home/user/.hermes")
    token.unlink()
    assert await get_credential_file_mounts() == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "relative_path",
    ["../sensitive.json", "../../.ssh/id_rsa"],
)
async def test_path_traversal_is_rejected(tmp_path, monkeypatch, relative_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (tmp_path / "sensitive.json").write_text("secret")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert await register_credential_file(relative_path) is False
    assert await get_credential_file_mounts() == []


@pytest.mark.asyncio
async def test_absolute_path_is_rejected(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    sensitive = tmp_path / "absolute.json"
    sensitive.write_text("secret")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert await register_credential_file(str(sensitive)) is False


@pytest.mark.asyncio
async def test_symlink_escape_is_rejected(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    sensitive = tmp_path / "sensitive.json"
    sensitive.write_text("secret")
    try:
        (home / "evil.json").symlink_to(sensitive)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert await register_credential_file("evil.json") is False


@pytest.mark.asyncio
async def test_nested_service_token_is_allowed(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "creds").mkdir(parents=True)
    (home / "creds" / "oauth.json").write_text("{}")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert await register_credential_file("creds/oauth.json") is True


def _credential_home(tmp_path: Path) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / ".env").write_text("OPENAI_API_KEY=secret\n")
    (home / "auth.json").write_text("{}")
    (home / ".anthropic_oauth.json").write_text("{}")
    (home / "webhook_subscriptions.json").write_text("{}")
    (home / "cache").mkdir()
    (home / "cache" / "bws_cache.json").write_text("{}")
    (home / "mcp-tokens").mkdir()
    (home / "mcp-tokens" / "server.json").write_text("{}")
    (home / "google_token.json").write_text("{}")
    return home


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "relative_path",
    [
        ".env",
        "auth.json",
        ".anthropic_oauth.json",
        "webhook_subscriptions.json",
        "cache/bws_cache.json",
        "mcp-tokens/server.json",
    ],
)
async def test_master_credential_stores_are_never_mountable(
    tmp_path,
    monkeypatch,
    relative_path,
):
    home = _credential_home(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert await register_credential_file(relative_path) is False
    assert await get_credential_file_mounts() == []


@pytest.mark.asyncio
async def test_refused_batch_entry_does_not_block_service_token(tmp_path, monkeypatch):
    home = _credential_home(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    missing = await register_credential_files([".env", "google_token.json"])
    mounts = await get_credential_file_mounts()

    assert missing == [".env"]
    assert [entry["container_path"] for entry in mounts] == [
        "/root/.hermes/google_token.json"
    ]


@pytest.mark.asyncio
async def test_missing_read_guard_fails_closed_with_diagnostic(
    tmp_path,
    monkeypatch,
    caplog,
):
    home = _credential_home(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(credential_files, "get_read_block_error", None)

    with caplog.at_level("ERROR", logger="tools.credential_files"):
        assert await register_credential_file("google_token.json") is False
    assert "deny-list cannot be consulted" in caplog.text


@pytest.mark.asyncio
async def test_raising_read_guard_fails_closed_with_traceback(
    tmp_path,
    monkeypatch,
    caplog,
):
    home = _credential_home(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    async def fail(_path):  # noqa: ASYNC124 - coroutine-shaped test double
        raise RuntimeError("guard exploded")

    monkeypatch.setattr(credential_files, "get_read_block_error", fail)
    with caplog.at_level("ERROR", logger="tools.credential_files"):
        assert await register_credential_file("google_token.json") is False
    record = next(r for r in caplog.records if "read guard raised" in r.message)
    assert record.exc_info is not None


def _write_config(home: Path, credential_paths: list[str]) -> None:
    import yaml

    (home / "config.yaml").write_text(
        yaml.safe_dump({"terminal": {"credential_files": credential_paths}})
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("entry_kind", ["traversal", "absolute"])
async def test_config_paths_cannot_escape_home(
    tmp_path,
    monkeypatch,
    entry_kind,
):
    home = tmp_path / ".hermes"
    home.mkdir()
    sensitive = tmp_path / "secret.json"
    sensitive.write_text("{}")
    monkeypatch.setenv("HERMES_HOME", str(home))
    entry = "../secret.json" if entry_kind == "traversal" else str(sensitive)
    _write_config(home, [entry])

    assert await get_credential_file_mounts() == []


@pytest.mark.asyncio
async def test_legitimate_config_file_is_mounted(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "oauth.json").write_text("{}")
    _write_config(home, ["oauth.json"])
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert await get_credential_file_mounts() == [
        {
            "host_path": str(home / "oauth.json"),
            "container_path": "/root/.hermes/oauth.json",
        }
    ]


@pytest.mark.asyncio
async def test_config_credential_mount_cache_is_profile_isolated(tmp_path):
    homes = {name: tmp_path / name for name in ("a", "b")}
    for name, home in homes.items():
        home.mkdir()
        (home / f"token-{name}.json").write_text(name)
        _write_config(home, [f"token-{name}.json"])

    async def load(name: str):
        token = set_hermes_home_override(str(homes[name]))
        try:
            return await get_credential_file_mounts()
        finally:
            reset_hermes_home_override(token)

    mounts_a, mounts_b = await asyncio.gather(load("a"), load("b"))

    assert mounts_a == [
        {
            "host_path": str(homes["a"] / "token-a.json"),
            "container_path": "/root/.hermes/token-a.json",
        }
    ]
    assert mounts_b == [
        {
            "host_path": str(homes["b"] / "token-b.json"),
            "container_path": "/root/.hermes/token-b.json",
        }
    ]


@pytest.mark.asyncio
async def test_config_credential_mount_cache_uses_canonical_profile(tmp_path):
    home = tmp_path / "profile"
    home.mkdir()
    (home / "token.json").write_text("token")
    _write_config(home, ["token.json"])
    alias = tmp_path / "profile-alias"
    alias.symlink_to(home, target_is_directory=True)

    async def load(active_home: Path):
        token = set_hermes_home_override(str(active_home))
        try:
            return await get_credential_file_mounts()
        finally:
            reset_hermes_home_override(token)

    assert await load(alias) == await load(home)
    assert len(credential_files._config_files_by_home) == 1


@pytest.mark.asyncio
async def test_config_read_cancellation_is_preserved(monkeypatch):
    async def cancelled(*_args, **_kwargs):  # noqa: ASYNC124 - test double
        raise asyncio.CancelledError

    monkeypatch.setattr("hermes_cli.config.read_user_config_raw", cancelled)
    with pytest.raises(asyncio.CancelledError):
        await credential_files._load_config_files()
    assert credential_files._config_files is None


@pytest.mark.asyncio
async def test_skills_directory_mount_and_custom_base(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    skills = home / "skills"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("# skill")
    monkeypatch.setenv("HERMES_HOME", str(home))

    mounts = await get_skills_directory_mount("/home/user/.hermes")

    assert mounts[0] == {
        "host_path": str(skills),
        "container_path": "/home/user/.hermes/skills",
    }


@pytest.mark.asyncio
async def test_skill_symlinks_are_sanitized_and_mode_is_preserved(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / ".hermes"
    skills = home / "skills"
    scripts = skills / "sample" / "scripts"
    scripts.mkdir(parents=True)
    executable = scripts / "run.sh"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    os.utime(executable, ns=(1_700_000_000_000_000_000,) * 2)
    secret = tmp_path / "secret"
    secret.write_text("TOP SECRET")
    try:
        (skills / "evil").symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported")
    monkeypatch.setenv("HERMES_HOME", str(home))

    mount = (await get_skills_directory_mount())[0]
    safe_path = Path(mount["host_path"])

    assert safe_path != skills
    assert not (safe_path / "evil").exists()
    copied = safe_path / "sample" / "scripts" / "run.sh"
    assert copied.read_text() == "#!/bin/sh\n"
    assert stat.S_IMODE(copied.stat().st_mode) == 0o755
    assert copied.stat().st_mtime_ns == executable.stat().st_mtime_ns


@pytest.mark.asyncio
async def test_iter_skills_files_skips_symlinks(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    skills = home / "skills" / "cat" / "sample"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("# skill")
    secret = tmp_path / "secret"
    secret.write_text("secret")
    try:
        (skills / "evil").symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported")
    monkeypatch.setenv("HERMES_HOME", str(home))

    paths = {entry["container_path"] for entry in await iter_skills_files()}

    assert "/root/.hermes/skills/cat/sample/SKILL.md" in paths
    assert not any("evil" in path for path in paths)


@pytest.mark.asyncio
async def test_external_skills_directory_is_included(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (external / "SKILL.md").write_text("# external")
    monkeypatch.setenv("HERMES_HOME", str(home))

    async def external_dirs():  # noqa: ASYNC124 - coroutine-shaped test double
        return [external]

    monkeypatch.setattr("agent.skill_utils.get_external_skills_dirs", external_dirs)
    mounts = await get_skills_directory_mount()
    files = await iter_skills_files()

    assert mounts == [
        {
            "host_path": str(external),
            "container_path": "/root/.hermes/external_skills/0",
        }
    ]
    assert files[0]["container_path"] == (
        "/root/.hermes/external_skills/0/SKILL.md"
    )


@pytest.mark.asyncio
async def test_cache_mounts_include_new_legacy_and_flat_images(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "cache" / "audio").mkdir(parents=True)
    legacy = home / "document_cache"
    legacy.mkdir()
    (legacy / "cached.txt").write_text("x")
    (home / "images").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    mounts = await get_cache_directory_mounts()
    by_container = {entry["container_path"]: entry["host_path"] for entry in mounts}

    assert by_container["/root/.hermes/cache/audio"] == str(home / "cache" / "audio")
    assert by_container["/root/.hermes/cache/documents"] == str(legacy)
    assert by_container["/root/.hermes/images"] == str(home / "images")


@pytest.mark.asyncio
async def test_cache_mapping_is_pure_after_awaited_mount_discovery(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / ".hermes"
    images = home / "cache" / "images"
    images.mkdir(parents=True)
    upload = images / "generated.png"
    monkeypatch.setenv("HERMES_HOME", str(home))
    await get_cache_directory_mounts()

    container_path = map_cache_path_to_container(str(upload))

    assert container_path == "/root/.hermes/cache/images/generated.png"
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    assert to_agent_visible_cache_path(str(upload)) == container_path
    assert from_agent_visible_cache_path(container_path) == str(upload)


@pytest.mark.asyncio
async def test_sync_cache_translation_uses_active_profile_backend(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / ".hermes"
    images = home / "cache" / "images"
    images.mkdir(parents=True)
    upload = images / "generated.png"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    await get_cache_directory_mounts()
    container_path = "/root/.hermes/cache/images/generated.png"

    previous_multiplex = secret_scope.is_multiplex_active()
    outer_token = secret_scope.set_secret_scope(None)
    secret_scope.set_multiplex_active(True)
    try:
        async def translate(backend: str):
            token = secret_scope.set_secret_scope({"TERMINAL_ENV": backend})
            try:
                await asyncio.sleep(0)
                return (
                    to_agent_visible_cache_path(str(upload)),
                    from_agent_visible_cache_path(container_path),
                )
            finally:
                secret_scope.reset_secret_scope(token)

        docker, local = await asyncio.gather(
            translate("docker"),
            translate("local"),
        )
        assert docker == (container_path, str(upload))
        assert local == (str(upload), container_path)

        with pytest.raises(secret_scope.UnscopedSecretError):
            to_agent_visible_cache_path(str(upload))
        with pytest.raises(secret_scope.UnscopedSecretError):
            from_agent_visible_cache_path(container_path)
    finally:
        secret_scope.set_multiplex_active(previous_multiplex)
        secret_scope.reset_secret_scope(outer_token)


@pytest.mark.asyncio
async def test_cache_mount_discovery_creates_empty_staging_dirs(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    await get_cache_directory_mounts()

    assert map_cache_path_to_container(str(home / "cache" / "images" / "x")) == (
        "/root/.hermes/cache/images/x"
    )


@pytest.mark.asyncio
async def test_iter_cache_files_enumerates_regular_files_and_skips_links(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / ".hermes"
    documents = home / "cache" / "documents"
    documents.mkdir(parents=True)
    regular = documents / "report.pdf"
    regular.write_bytes(b"%PDF")
    try:
        (documents / "linked.pdf").symlink_to(regular)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported")
    monkeypatch.setenv("HERMES_HOME", str(home))

    entries = await iter_cache_files()
    names = {Path(entry["container_path"]).name for entry in entries}

    assert names == {"report.pdf"}


@pytest.mark.asyncio
async def test_empty_skill_and_cache_homes_return_empty(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert await get_skills_directory_mount() == []
    assert await iter_skills_files() == []
    mounts = await get_cache_directory_mounts()
    assert mounts
    assert all(Path(entry["host_path"]).is_dir() for entry in mounts)
    assert await iter_cache_files() == []


@pytest.mark.asyncio
async def test_retained_io_paths_do_not_block_the_event_loop(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    skills = home / "skills"
    documents = home / "cache" / "documents"
    skills.mkdir(parents=True)
    documents.mkdir(parents=True)
    (home / "token.json").write_text("{}")
    (skills / "SKILL.md").write_text("# skill")
    (documents / "upload.txt").write_text("payload")
    monkeypatch.setenv("HERMES_HOME", str(home))

    blocker = BlockBuster()
    blocker.activate()
    try:
        assert await register_credential_file("token.json")
        assert await get_credential_file_mounts()
        assert await get_skills_directory_mount()
        assert await iter_skills_files()
        assert await get_cache_directory_mounts()
        assert await iter_cache_files()
    finally:
        blocker.deactivate()
