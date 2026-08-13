"""Tests for agent/runtime_cwd.py — the single source of truth for the agent working directory."""

import asyncio
import importlib.util
import os
from pathlib import Path

import pytest
from blockbuster import BlockBuster

import agent.runtime_cwd as rt
from agent import secret_scope
from agent.runtime_cwd import (
    clear_session_cwd,
    resolve_agent_cwd,
    resolve_context_cwd,
    set_session_cwd,
)


async def _raise_oserror(*args, **kwargs):
    raise OSError("cwd gone")


@pytest.fixture(autouse=True)
def _restore_secret_scope_state():
    previous_multiplex = secret_scope.is_multiplex_active()
    token = secret_scope.set_secret_scope(None)
    secret_scope.set_multiplex_active(False)
    try:
        yield
    finally:
        secret_scope.set_multiplex_active(previous_multiplex)
        secret_scope.reset_secret_scope(token)


def test_module_initialization_does_not_resolve_package_root(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "_runtime_cwd_import_probe",
        rt.__file__,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    def forbidden_resolve(*_args, **_kwargs):
        raise AssertionError("runtime_cwd resolved a path during module execution")

    monkeypatch.setattr(Path, "resolve", forbidden_resolve)
    spec.loader.exec_module(module)


@pytest.mark.asyncio
async def test_install_tree_comparison_canonicalizes_lazy_package_root(
    monkeypatch, tmp_path
):
    package_root = tmp_path / "package"
    package_root.mkdir()
    alias = tmp_path / "package-alias"
    alias.symlink_to(package_root, target_is_directory=True)
    monkeypatch.setattr(rt, "_PACKAGE_ROOT", alias)

    assert await rt._is_install_tree(package_root) is True


class TestResolveAgentCwd:
    @pytest.mark.asyncio
    async def test_prefers_terminal_cwd_over_getcwd(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        monkeypatch.chdir(os.path.expanduser("~"))
        assert await resolve_agent_cwd() == tmp_path





    @pytest.mark.asyncio
    async def test_propagates_oserror_from_getcwd(self, monkeypatch):
        # The fallback arm calls os.getcwd(), which can raise OSError (deleted cwd).
        # The resolver must NOT swallow it — build_environment_hints owns the
        # try/except OSError guard at the call site (prompt_builder.py:805).
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.setattr(rt.aiofiles.os, "getcwd", _raise_oserror)
        with pytest.raises(OSError):
            await resolve_agent_cwd()


class TestResolveContextCwd:
    @pytest.mark.asyncio
    async def test_returns_dir_when_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        assert await resolve_context_cwd() == tmp_path




    @pytest.mark.asyncio
    async def test_expands_leading_tilde(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_CWD", "~")
        expected = Path(os.path.expanduser("~"))
        blocker = BlockBuster()
        blocker.activate()
        try:
            assert await resolve_context_cwd() == expected
        finally:
            blocker.deactivate()


@pytest.mark.asyncio
async def test_concurrent_profiles_resolve_their_own_terminal_cwd(
    monkeypatch,
    tmp_path,
):
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    monkeypatch.setenv("TERMINAL_CWD", str(foreign))
    secret_scope.set_multiplex_active(True)
    profile_cwds = {label: tmp_path / label for label in ("a", "b")}
    for cwd in profile_cwds.values():
        cwd.mkdir()

    async def resolve(label: str):
        token = secret_scope.set_secret_scope(
            {"TERMINAL_CWD": str(profile_cwds[label])}
        )
        try:
            await asyncio.sleep(0)
            return await resolve_agent_cwd(), await resolve_context_cwd()
        finally:
            secret_scope.reset_secret_scope(token)

    assert await asyncio.gather(resolve("a"), resolve("b")) == [
        (profile_cwds["a"], profile_cwds["a"]),
        (profile_cwds["b"], profile_cwds["b"]),
    ]


@pytest.mark.asyncio
async def test_unscoped_multiplex_cwd_resolution_fails_closed(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    secret_scope.set_multiplex_active(True)

    with pytest.raises(secret_scope.UnscopedSecretError):
        await resolve_agent_cwd()
    with pytest.raises(secret_scope.UnscopedSecretError):
        await resolve_context_cwd()



class TestSessionCwdOverride:
    """The #29531 per-session arm: a contextvar cwd wins over TERMINAL_CWD so a
    multi-session gateway can pin each session to its own folder."""

    @pytest.mark.asyncio
    async def test_session_cwd_overrides_terminal_cwd(self, monkeypatch, tmp_path):
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        token = set_session_cwd(str(other))
        try:
            assert await resolve_agent_cwd() == other
            assert await resolve_context_cwd() == other
        finally:
            rt._SESSION_CWD.reset(token)


    @pytest.mark.asyncio
    async def test_clear_session_cwd_restores_terminal_cwd(self, monkeypatch, tmp_path):
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        token = set_session_cwd(str(other))
        try:
            clear_session_cwd()
            assert await resolve_agent_cwd() == tmp_path
        finally:
            rt._SESSION_CWD.reset(token)
