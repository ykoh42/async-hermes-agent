"""Parity tests for auth helpers retained by the async runtime."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    return home


@pytest.mark.asyncio
async def test_get_anthropic_key_prefers_dotenv_over_stale_environment(
    hermes_home,
    monkeypatch,
):
    (hermes_home / ".env").write_text(
        "ANTHROPIC_API_KEY=dotenv-fresh-anthropic\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "stale-shell-anthropic")

    from hermes_cli.auth import get_anthropic_key

    assert await get_anthropic_key() == "dotenv-fresh-anthropic"


@pytest.mark.asyncio
async def test_get_anthropic_key_preserves_upstream_lookup_order(hermes_home):
    (hermes_home / ".env").write_text(
        "ANTHROPIC_TOKEN=token-second\n"
        "CLAUDE_CODE_OAUTH_TOKEN=oauth-third\n",
        encoding="utf-8",
    )

    from hermes_cli.auth import get_anthropic_key

    assert await get_anthropic_key() == "token-second"


@pytest.mark.asyncio
async def test_unsuppress_credential_source_clears_marker(hermes_home):
    from hermes_cli.auth import (
        is_source_suppressed,
        suppress_credential_source,
        unsuppress_credential_source,
    )

    await suppress_credential_source("openai-codex", "device_code")
    assert await is_source_suppressed("openai-codex", "device_code") is True

    assert (
        await unsuppress_credential_source("openai-codex", "device_code") is True
    )
    assert await is_source_suppressed("openai-codex", "device_code") is False

    payload = json.loads((hermes_home / "auth.json").read_text(encoding="utf-8"))
    assert "suppressed_sources" not in payload


@pytest.mark.asyncio
async def test_unsuppress_credential_source_preserves_other_markers(hermes_home):
    from hermes_cli.auth import (
        is_source_suppressed,
        suppress_credential_source,
        unsuppress_credential_source,
    )

    await suppress_credential_source("openai-codex", "device_code")
    await suppress_credential_source("anthropic", "claude_code")

    assert (
        await unsuppress_credential_source("openai-codex", "device_code") is True
    )
    assert await is_source_suppressed("anthropic", "claude_code") is True


@pytest.mark.asyncio
async def test_concurrent_suppressions_preserve_both_sources(hermes_home):
    from hermes_cli.auth import is_source_suppressed, suppress_credential_source

    await asyncio.gather(
        suppress_credential_source("anthropic", "claude_code"),
        suppress_credential_source("anthropic", "env:ANTHROPIC_API_KEY"),
    )

    assert await is_source_suppressed("anthropic", "claude_code") is True
    assert await is_source_suppressed("anthropic", "env:ANTHROPIC_API_KEY") is True


@pytest.mark.asyncio
async def test_is_source_suppressed_preserves_cancellation(monkeypatch):
    from hermes_cli import auth

    monkeypatch.setattr(
        auth,
        "_load_auth_store",
        AsyncMock(side_effect=asyncio.CancelledError),
    )

    with pytest.raises(asyncio.CancelledError):
        await auth.is_source_suppressed("anthropic", "claude_code")
