"""Tests for follow-up fixes to the LSP integration (PR after #24168).

Covers:

1. ``typescript-language-server`` install recipe pulls in ``typescript``
   alongside the server, so the npm install command targets both.
2. ``_check_lint`` returns ``skipped`` (not ``error``) when the linter
   command exists on PATH but couldn't actually run — e.g. ``npx tsc``
   without the typescript SDK installed.  This is what unblocks the
   LSP semantic tier on TypeScript files when the user doesn't also
   have a project-level ``tsc``.
"""
from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.lsp.install import INSTALL_RECIPES


@pytest.mark.asyncio
async def test_cancelled_installer_reaps_owned_subprocess(monkeypatch):
    from agent.lsp import install as install_mod

    captured = []
    create_subprocess_exec = asyncio.create_subprocess_exec

    async def capture_process(*args, **kwargs):
        process = await create_subprocess_exec(*args, **kwargs)
        captured.append(process)
        return process

    monkeypatch.setattr(
        install_mod.asyncio,
        "create_subprocess_exec",
        capture_process,
    )
    install_task = asyncio.create_task(
        install_mod._run_install(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            timeout=30,
        )
    )
    while not captured:
        await asyncio.sleep(0)

    install_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await install_task

    assert captured[0].returncode is not None


@pytest.mark.asyncio
async def test_timed_out_installer_reaps_owned_subprocess(monkeypatch):
    from agent.lsp import install as install_mod

    captured = []
    create_subprocess_exec = asyncio.create_subprocess_exec

    async def capture_process(*args, **kwargs):
        process = await create_subprocess_exec(*args, **kwargs)
        captured.append(process)
        return process

    monkeypatch.setattr(
        install_mod.asyncio,
        "create_subprocess_exec",
        capture_process,
    )
    with pytest.raises(TimeoutError):
        await install_mod._run_install(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            timeout=0.01,
        )

    assert captured[0].returncode is not None


# ---------------------------------------------------------------------------
# Fix 1: typescript install recipe carries the typescript SDK
# ---------------------------------------------------------------------------






@pytest.mark.asyncio
async def test_install_npm_works_without_extras(tmp_path, monkeypatch):
    """Backwards compat: pyright-style recipes (no extras) still install."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    captured = {}

    async def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return 0, ""

    from agent.lsp import install as install_mod

    monkeypatch.setattr(install_mod, "_run_install", fake_run)
    monkeypatch.setattr(
        install_mod,
        "_existing_binary",
        AsyncMock(side_effect=lambda name: "/usr/bin/npm" if name == "npm" else None),
    )

    await install_mod._install_npm("pyright", "pyright-langserver")

    cmd = captured["cmd"]
    assert "pyright" in cmd
    # Should not blow up when extra_pkgs is omitted/None
    install_targets = [c for c in cmd if not c.startswith("-") and c not in {
        "install", "--prefix", str((await install_mod.hermes_lsp_bin_dir()).parent),
        "/usr/bin/npm",
    }]
    assert install_targets == ["pyright"]




@pytest.mark.asyncio
async def test_install_pip_finds_windows_scripts_launcher(tmp_path, monkeypatch):
    """pip console scripts can land in Scripts/ on native Windows."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from agent.lsp import install as install_mod

    async def fake_run(cmd, **kwargs):
        scripts_dir = (
            (await install_mod.hermes_lsp_bin_dir()).parent
            / "python-packages"
            / "Scripts"
        )
        scripts_dir.mkdir(parents=True, exist_ok=True)
        launcher = scripts_dir / "fake-language-server.exe"
        launcher.write_text("launcher\n")
        launcher.chmod(0o755)
        return 0, ""

    monkeypatch.setattr(install_mod, "_is_windows", lambda: True)
    monkeypatch.setattr(install_mod, "_run_install", fake_run)

    resolved = await install_mod._install_pip("fake-lsp", "fake-language-server")

    assert resolved is not None
    assert resolved.endswith("fake-language-server.exe")
    assert (
        (await install_mod.hermes_lsp_bin_dir()) / "fake-language-server.exe"
    ).exists()


# ---------------------------------------------------------------------------
# Fix 2: tier-1 lint treats unusable linters as ``skipped``, not ``error``
# ---------------------------------------------------------------------------










@pytest.mark.asyncio
async def test_check_lint_returns_error_for_real_ts_type_errors(tmp_path):
    """Sanity: real TypeScript errors still go through the error path."""
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations

    ts_file = tmp_path / "bad.ts"
    ts_file.write_text("const x: string = 42;\n")

    env = LocalEnvironment()
    fops = ShellFileOperations(env)

    real_tsc_error = (
        "bad.ts:1:7 - error TS2322: Type 'number' is not assignable to type 'string'.\n"
        "1 const x: string = 42;\n"
        "        ~\n"
        "Found 1 error.\n"
    )

    def fake_exec(cmd, **kwargs):
        result = MagicMock()
        result.exit_code = 1
        result.stdout = real_tsc_error
        return result

    with patch.object(fops, "_exec", side_effect=fake_exec), \
         patch.object(fops, "_has_command", return_value=True):
        lint = await fops._check_lint(str(ts_file))

    assert lint.skipped is False
    assert lint.success is False
    assert "TS2322" in lint.output


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
