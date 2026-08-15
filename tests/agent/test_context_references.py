"""Upstream context-reference behavior adapted to the native async boundary."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Hermes Tests")
    _git(repo, "config", "user.email", "tests@example.com")

    (repo / "src").mkdir()
    (repo / "src" / "main.py").write_text(
        "def alpha():\n"
        "    return 'a'\n\n"
        "def beta():\n"
        "    return 'b'\n",
        encoding="utf-8",
    )
    (repo / "src" / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "README.md").write_text("# Demo\n", encoding="utf-8")
    (repo / "blob.bin").write_bytes(b"\x00\x01\x02binary")

    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")

    (repo / "src" / "main.py").write_text(
        "def alpha():\n"
        "    return 'changed'\n\n"
        "def beta():\n"
        "    return 'b'\n",
        encoding="utf-8",
    )
    (repo / "src" / "helper.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(repo, "add", "src/helper.py")
    return repo


def test_parse_typed_references_ignores_emails_and_handles():
    from agent.context_references import parse_context_references

    message = (
        "email me at user@example.com and ping @teammate "
        "but include @file:src/main.py:1-2 plus @diff and @git:2 "
        "and @url:https://example.com/docs"
    )

    refs = parse_context_references(message)

    assert [ref.kind for ref in refs] == ["file", "diff", "git", "url"]
    assert refs[0].target == "src/main.py"
    assert refs[0].line_start == 1
    assert refs[0].line_end == 2
    assert refs[2].target == "2"


@pytest.mark.asyncio
async def test_folder_listing_falls_back_when_rg_is_blocked(sample_repo: Path):
    from agent.context_references import preprocess_context_references

    real_create = asyncio.create_subprocess_exec

    async def blocked_rg(*args, **kwargs):
        if args and args[0] == "rg":
            raise PermissionError("rg blocked by policy")
        return await real_create(*args, **kwargs)

    with patch(
        "agent.context_references.asyncio.create_subprocess_exec",
        side_effect=blocked_rg,
    ):
        result = await preprocess_context_references(
            "Review @folder:src/",
            cwd=sample_repo,
            context_length=100_000,
        )

    assert result.expanded
    assert "src/" in result.message
    assert "main.py" in result.message
    assert "helper.py" in result.message
    assert not result.warnings


@pytest.mark.asyncio
async def test_missing_file_becomes_warning(sample_repo: Path):
    from agent.context_references import preprocess_context_references

    result = await preprocess_context_references(
        "Check @file:nope.txt",
        cwd=sample_repo,
        context_length=100_000,
    )

    assert result.expanded
    assert len(result.warnings) == 1
    assert "not found" in result.message.lower()


@pytest.mark.asyncio
async def test_blocks_canonical_read_denylist_credential_stores(
    tmp_path: Path, monkeypatch,
):
    """Context references must honor the canonical async read deny-list."""
    from agent.context_references import preprocess_context_references_async

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True)
    auth_json = hermes_home / "auth.json"
    auth_json.write_text('{"openai": "sk-AUTHJSON-SECRET"}\n', encoding="utf-8")
    oauth = hermes_home / ".anthropic_oauth.json"
    oauth.write_text('{"access_token": "OAUTH-SECRET"}\n', encoding="utf-8")
    mcp_token = hermes_home / "mcp-tokens" / "github.json"
    mcp_token.parent.mkdir(parents=True)
    mcp_token.write_text('{"token": "MCP-TOKEN-SECRET"}\n', encoding="utf-8")
    project_env = tmp_path / "project" / ".env"
    project_env.parent.mkdir(parents=True)
    project_env.write_text("DB_PASSWORD=ENV-SECRET\n", encoding="utf-8")

    result = await preprocess_context_references_async(
        "inspect @file:.hermes/auth.json and @file:.hermes/.anthropic_oauth.json "
        "and @file:.hermes/mcp-tokens/github.json and @file:project/.env",
        cwd=tmp_path,
        allowed_root=tmp_path,
        context_length=100_000,
    )

    assert result.expanded
    for secret in (
        "sk-AUTHJSON-SECRET",
        "OAUTH-SECRET",
        "MCP-TOKEN-SECRET",
        "ENV-SECRET",
    ):
        assert secret not in result.message
    assert sum("sensitive credential" in warning for warning in result.warnings) == 4


@pytest.mark.asyncio
async def test_canonical_guard_fails_closed_when_lookup_raises(
    tmp_path: Path, monkeypatch,
):
    """A failing canonical guard must refuse the reference."""
    from agent.context_references import preprocess_context_references_async

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True)
    auth_json = hermes_home / "auth.json"
    auth_json.write_text('{"openai": "sk-AUTHJSON-SECRET"}\n', encoding="utf-8")

    def _boom(_path):
        raise RuntimeError("guard resolution failed")

    monkeypatch.setattr("agent.file_safety.get_read_block_error", _boom)
    result = await preprocess_context_references_async(
        "inspect @file:.hermes/auth.json",
        cwd=tmp_path,
        allowed_root=tmp_path,
        context_length=100_000,
    )

    assert "sk-AUTHJSON-SECRET" not in result.message
    assert any(
        "credential deny-list" in warning or "sensitive credential" in warning
        for warning in result.warnings
    )


@pytest.mark.parametrize(
    "value",
    [
        "/tmp/plain.png",
        "/Users/me/Library/Application Support/Hermes/composer-images/a.png",
        r"C:\Users\John Doe\Pictures\cat.png",
        "/tmp/report (final).pdf",
        "/tmp/it's here.png",
        '/tmp/say "hi".png',
    ],
)
def test_format_reference_value_round_trips_through_the_parser(value):
    from agent.context_references import REFERENCE_PATTERN, format_reference_value

    match = REFERENCE_PATTERN.search(f"@file:{format_reference_value(value)}")

    assert match is not None
    assert match.group("value").strip("`\"'") == value


async def _slow_url_fetcher(url: str) -> str:
    await asyncio.sleep(0.01)
    return f"CONTENT[{url}]"


@pytest.mark.asyncio
async def test_url_references_expand_concurrently_and_preserve_order(tmp_path):
    """The upstream gather contract must survive the native async port."""
    from agent.context_references import preprocess_context_references_async

    refs = [
        "https://a.example/x",
        "https://b.example/y",
        "https://c.example/z",
    ]
    barrier = asyncio.Barrier(len(refs))
    entered: list[str] = []

    async def fetch(url: str) -> str:
        entered.append(url)
        async with asyncio.timeout(5):
            await barrier.wait()
        return f"CONTENT[{url}]"

    result = await preprocess_context_references_async(
        "see " + " ".join(f"@url:{url}" for url in refs),
        cwd=tmp_path,
        context_length=100_000,
        url_fetcher=fetch,
    )
    assert result.expanded
    assert entered == refs
    assert result.message.index("a.example") < result.message.index("b.example")
    assert result.message.index("b.example") < result.message.index("c.example")


@pytest.mark.asyncio
async def test_url_reference_concurrency_preserves_output_contract(tmp_path):
    from agent.context_references import preprocess_context_references_async

    result = await preprocess_context_references_async(
        "@url:https://one.example/p @url:https://two.example/q",
        cwd=tmp_path,
        context_length=100_000,
        url_fetcher=_slow_url_fetcher,
    )
    assert "CONTENT[https://one.example/p]" in result.message
    assert "CONTENT[https://two.example/q]" in result.message
    assert result.message.index("one.example") < result.message.index("two.example")
    assert result.injected_tokens > 0
