"""Profile-secret isolation for command TTS env passthrough."""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import sys

import pytest

from agent.secret_scope import (
    UnscopedSecretError,
    is_multiplex_active,
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from tools import tts_tool


async def _run_scoped(value: str):
    token = set_secret_scope({"PROFILE_TTS_KEY": value})
    try:
        script = "import os; print(os.environ.get('PROFILE_TTS_KEY', 'missing'))"
        command = (
            subprocess.list2cmdline([sys.executable, "-c", script])
            if os.name == "nt"
            else " ".join(
                (shlex.quote(sys.executable), "-c", shlex.quote(script))
            )
        )
        return await tts_tool._run_command_tts(
            command,
            timeout=5,
            env_passthrough=["PROFILE_TTS_KEY"],
        )
    finally:
        reset_secret_scope(token)


@pytest.mark.asyncio
async def test_command_passthrough_uses_each_concurrent_profile_scope(monkeypatch):
    previous = is_multiplex_active()
    set_multiplex_active(True)
    monkeypatch.setenv("PROFILE_TTS_KEY", "foreign-process-value")
    try:
        result_a, result_b = await asyncio.gather(
            _run_scoped("profile-a"),
            _run_scoped("profile-b"),
        )
    finally:
        set_multiplex_active(previous)

    assert result_a.stdout.strip() == "profile-a"
    assert result_b.stdout.strip() == "profile-b"


@pytest.mark.asyncio
async def test_command_passthrough_fails_closed_without_profile_scope(monkeypatch):
    previous = is_multiplex_active()
    set_multiplex_active(True)
    monkeypatch.setenv("PROFILE_TTS_KEY", "foreign-process-value")
    try:
        with pytest.raises(UnscopedSecretError):
            await tts_tool._run_command_tts(
                "true",
                timeout=5,
                env_passthrough=["PROFILE_TTS_KEY"],
            )
    finally:
        set_multiplex_active(previous)
