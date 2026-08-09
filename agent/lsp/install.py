"""Auto-installation of LSP server binaries.

Tries to install missing servers using whatever package manager is
appropriate.  All installs go to a Hermes-owned bin staging dir,
``<HERMES_HOME>/lsp/bin/``, so we don't pollute the user's global
toolchain.

Strategies:

- ``auto`` — attempt to install with the best available package
  manager.  This is the default.
- ``manual`` — never install; if a binary is missing, the server is
  silently skipped and the missing binary is reported in the LSP log.
- ``off`` — same as ``manual`` for now (kept distinct so we can
  evolve behavior later, e.g. logging differently).

The actual installs happen asynchronously the first time a server is
needed and concurrent calls to :func:`try_install` for the same
package are deduplicated via a per-package lock.

Failure modes are non-fatal: every install path is wrapped in
try/except and returns ``None`` on failure.  The tool layer then
falls back to its in-process syntax checker, exactly as if the user
hadn't enabled LSP at all.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import aiofiles.os

from hermes_cli._subprocess_compat import windows_hide_flags

logger = logging.getLogger("agent.lsp.install")

# Package-name → install-strategy hint registry.  Each entry is a
# tuple of strategy name + package name + executable name.  When the
# install completes, we look for the executable in
# ``<HERMES_HOME>/lsp/bin/`` first, then on PATH.
#
# Optional fields:
#   - ``extra_pkgs``: list of sibling packages to install alongside
#     ``pkg`` in the same node_modules tree.  Used when an LSP server
#     has a runtime peer dependency that npm doesn't auto-pull (e.g.
#     typescript-language-server needs ``typescript``).
INSTALL_RECIPES: Dict[str, Dict[str, Any]] = {
    # Python
    "pyright": {"strategy": "npm", "pkg": "pyright", "bin": "pyright-langserver"},
    # JS/TS family
    "typescript-language-server": {
        "strategy": "npm",
        "pkg": "typescript-language-server",
        "bin": "typescript-language-server",
        # typescript-language-server requires the `typescript` SDK
        # (tsserver) to be importable from the same node_modules tree;
        # otherwise initialize() fails with "Could not find a valid
        # TypeScript installation".  Install them together.
        "extra_pkgs": ["typescript"],
    },
    "@vue/language-server": {
        "strategy": "npm",
        "pkg": "@vue/language-server",
        "bin": "vue-language-server",
    },
    "svelte-language-server": {
        "strategy": "npm",
        "pkg": "svelte-language-server",
        "bin": "svelteserver",
    },
    "@astrojs/language-server": {
        "strategy": "npm",
        "pkg": "@astrojs/language-server",
        "bin": "astro-ls",
    },
    "yaml-language-server": {
        "strategy": "npm",
        "pkg": "yaml-language-server",
        "bin": "yaml-language-server",
    },
    "bash-language-server": {
        "strategy": "npm",
        "pkg": "bash-language-server",
        "bin": "bash-language-server",
    },
    "intelephense": {"strategy": "npm", "pkg": "intelephense", "bin": "intelephense"},
    "dockerfile-language-server-nodejs": {
        "strategy": "npm",
        "pkg": "dockerfile-language-server-nodejs",
        "bin": "docker-langserver",
    },
    # Go
    "gopls": {"strategy": "go", "pkg": "golang.org/x/tools/gopls@latest", "bin": "gopls"},
    # Rust — too heavy (hundreds of MB to bootstrap).  We do NOT
    # auto-install rust-analyzer; users install via rustup.
    "rust-analyzer": {"strategy": "manual", "pkg": "", "bin": "rust-analyzer"},
    # C/C++ — manual (clangd ships with LLVM, very heavy)
    "clangd": {"strategy": "manual", "pkg": "", "bin": "clangd"},
    # Lua — manual (LuaLS is platform-specific binaries from GitHub
    # releases; complex enough that we punt to the user)
    "lua-language-server": {"strategy": "manual", "pkg": "", "bin": "lua-language-server"},
    # PowerShell — PowerShellEditorServices ships as a GitHub release
    # zip driven by a pwsh bootstrap script, not a single binary.  We
    # require a manual bundle install and probe for the pwsh host.
    "powershell": {"strategy": "manual", "pkg": "", "bin": "pwsh"},
}


_install_locks: Dict[str, asyncio.Lock] = {}
_install_results: Dict[str, Optional[str]] = {}
_WINDOWS_WRAPPER_SUFFIXES = (".cmd", ".exe", ".bat")


def _is_windows() -> bool:
    return os.name == "nt"


async def _await_cleanup_task(
    task: asyncio.Task[tuple[bytes, bytes]],
) -> tuple[bytes, bytes]:
    """Finish installer cleanup through repeated caller cancellation."""
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised below
            if task.cancelled():
                raise
            if cancellation is None:
                cancellation = exc
        except Exception as exc:
            if cancellation is not None:
                raise cancellation from exc
            raise
    if cancellation is not None:
        raise cancellation
    return result


async def hermes_lsp_bin_dir() -> Path:
    """Return the Hermes-owned bin staging dir for LSP servers."""
    from hermes_constants import get_hermes_home

    path = get_hermes_home() / "lsp" / "bin"
    await aiofiles.os.makedirs(path, exist_ok=True)
    return path


def _native_binary_candidates(base: Path) -> list[Path]:
    """Return platform-native executable candidates for a staged binary."""
    candidates = [base]
    if _is_windows():
        existing = {str(base).lower()}
        for suffix in _WINDOWS_WRAPPER_SUFFIXES:
            candidate = Path(str(base) + suffix)
            key = str(candidate).lower()
            if key not in existing:
                candidates.append(candidate)
                existing.add(key)
    return candidates


async def _which(name: str) -> str | None:
    return await aiofiles.os.wrap(shutil.which)(name)


async def _existing_binary(name: str) -> Optional[str]:
    """Probe the staging dir and PATH for a binary named ``name``."""
    staging = await hermes_lsp_bin_dir()
    for candidate in _native_binary_candidates(staging / name):
        if await aiofiles.os.path.exists(candidate) and await aiofiles.os.access(
            candidate, os.X_OK
        ):
            return str(candidate)
    on_path = await _which(name)
    if on_path:
        return on_path
    if _is_windows():
        for suffix in _WINDOWS_WRAPPER_SUFFIXES:
            on_path = await _which(f"{name}{suffix}")
            if on_path:
                return on_path
    return None


def _get_lock(pkg: str) -> asyncio.Lock:
    lock = _install_locks.get(pkg)
    if lock is None:
        lock = asyncio.Lock()
        _install_locks[pkg] = lock
    return lock


async def try_install(pkg: str, strategy: str = "auto") -> Optional[str]:
    """Install ``pkg`` once and return its binary path when available."""
    recipe = INSTALL_RECIPES.get(pkg, {})
    bin_name = recipe.get("bin", pkg)
    if strategy != "auto":
        return await _existing_binary(bin_name)
    if pkg in _install_results:
        return _install_results[pkg]

    async with _get_lock(pkg):
        if pkg in _install_results:
            return _install_results[pkg]
        result = await _do_install(pkg)
        _install_results[pkg] = result
        return result


async def _finish_process(
    process: asyncio.subprocess.Process,
    communicate_task: asyncio.Task[tuple[bytes, bytes]],
) -> tuple[bytes, bytes]:
    """Terminate, drain, and reap an installer before propagating cancellation."""
    if process.returncode is None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass
    try:
        return await asyncio.wait_for(asyncio.shield(communicate_task), timeout=2.0)
    except TimeoutError:
        if process.returncode is None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
        return await asyncio.shield(communicate_task)


async def _run_install(
    command: list[str],
    *,
    timeout: float,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        start_new_session=os.name == "posix",
        creationflags=windows_hide_flags(),
    )
    communicate_task = asyncio.create_task(process.communicate())
    failure: asyncio.CancelledError | TimeoutError | None = None
    try:
        _stdout, stderr = await asyncio.wait_for(
            asyncio.shield(communicate_task), timeout=timeout
        )
    except (
        asyncio.CancelledError,  # noqa: ASYNC103 - cleanup precedes re-raise below
        TimeoutError,
    ) as exc:
        failure = exc
    if failure is not None:
        cleanup_task = asyncio.create_task(
            _finish_process(process, communicate_task)
        )
        await _await_cleanup_task(cleanup_task)
        raise failure.with_traceback(failure.__traceback__)
    return process.returncode or 0, stderr.decode("utf-8", errors="replace")


async def _do_install(pkg: str) -> Optional[str]:
    recipe = INSTALL_RECIPES.get(pkg)
    if recipe is None:
        return await _which(pkg)

    install_kind = recipe.get("strategy", "manual")
    bin_name = recipe.get("bin", pkg)
    existing = await _existing_binary(bin_name)
    if existing:
        return existing
    if install_kind == "manual":
        logger.debug("[install] %s requires manual install (recipe=%s)", pkg, recipe)
        return None
    if install_kind == "npm":
        return await _install_npm(
            recipe.get("pkg", pkg),
            bin_name,
            extra_pkgs=recipe.get("extra_pkgs") or [],
        )
    if install_kind == "go":
        return await _install_go(recipe.get("pkg", pkg), bin_name)
    if install_kind == "pip":
        return await _install_pip(recipe.get("pkg", pkg), bin_name)
    logger.warning("[install] unknown strategy %r for %s", install_kind, pkg)
    return None


async def _link_or_copy(source: Path, destination: Path) -> str:
    if not await aiofiles.os.path.exists(destination):
        try:
            await aiofiles.os.symlink(source, destination)
        except (OSError, NotImplementedError):
            try:
                await aiofiles.os.wrap(shutil.copy2)(source, destination)
            except OSError:
                return str(source)
    return str(destination if await aiofiles.os.path.exists(destination) else source)


async def _install_npm(
    pkg: str,
    bin_name: str,
    extra_pkgs: Optional[list] = None,
) -> Optional[str]:
    npm = await _existing_binary("npm")
    if npm is None:
        logger.info("[install] cannot install %s: no usable npm found", pkg)
        return None
    staging = (await hermes_lsp_bin_dir()).parent
    install_targets = [pkg] + list(extra_pkgs or [])
    try:
        logger.info("[install] npm install --prefix %s %s", staging, " ".join(install_targets))
        returncode, stderr = await _run_install(
            [npm, "install", "--prefix", str(staging), "--silent", "--no-fund", "--no-audit", *install_targets],
            timeout=300,
        )
    except TimeoutError:
        logger.warning("[install] npm install timed out for %s", pkg)
        return None
    except OSError as exc:
        logger.warning("[install] npm install errored for %s: %s", pkg, exc)
        return None
    if returncode != 0:
        logger.warning("[install] npm install failed for %s: %s", pkg, stderr.strip()[:500])
        return None

    node_bin = staging / "node_modules" / ".bin" / bin_name
    for candidate in _native_binary_candidates(node_bin):
        if await aiofiles.os.path.exists(candidate):
            return await _link_or_copy(
                candidate, (await hermes_lsp_bin_dir()) / candidate.name
            )
    logger.warning("[install] npm install for %s succeeded but bin %s not found", pkg, bin_name)
    return None


async def _install_go(pkg: str, bin_name: str) -> Optional[str]:
    go = await _which("go")
    if go is None:
        logger.info("[install] cannot install %s: go not on PATH", pkg)
        return None
    staging = await hermes_lsp_bin_dir()
    env = dict(os.environ)
    env["GOBIN"] = str(staging)
    try:
        returncode, stderr = await _run_install(
            [go, "install", pkg], timeout=600, env=env
        )
    except TimeoutError:
        logger.warning("[install] go install timed out for %s", pkg)
        return None
    except OSError as exc:
        logger.warning("[install] go install errored for %s: %s", pkg, exc)
        return None
    if returncode != 0:
        logger.warning("[install] go install failed for %s: %s", pkg, stderr.strip()[:500])
        return None
    bin_path = staging / bin_name
    if _is_windows():
        bin_path = bin_path.with_suffix(".exe")
    return str(bin_path) if await aiofiles.os.path.exists(bin_path) else None


async def _install_pip(pkg: str, bin_name: str) -> Optional[str]:
    pip_target = (await hermes_lsp_bin_dir()).parent / "python-packages"
    await aiofiles.os.makedirs(pip_target, exist_ok=True)
    try:
        returncode, stderr = await _run_install(
            [sys.executable, "-m", "pip", "install", "--target", str(pip_target), "--quiet", pkg],
            timeout=300,
        )
    except TimeoutError:
        logger.warning("[install] pip install timed out for %s", pkg)
        return None
    except OSError as exc:
        logger.warning("[install] pip install errored for %s: %s", pkg, exc)
        return None
    if returncode != 0:
        logger.warning("[install] pip install failed for %s: %s", pkg, stderr.strip()[:500])
        return None

    script_dirs = [pip_target / "bin"]
    if _is_windows():
        script_dirs.append(pip_target / "Scripts")
    for script_dir in script_dirs:
        for bin_path in _native_binary_candidates(script_dir / bin_name):
            if await aiofiles.os.path.exists(bin_path):
                return await _link_or_copy(
                    bin_path, (await hermes_lsp_bin_dir()) / bin_path.name
                )
    return None


async def detect_status(pkg: str) -> str:
    """Return ``installed``, ``missing``, or ``manual-only``."""
    recipe = INSTALL_RECIPES.get(pkg)
    bin_name = recipe.get("bin", pkg) if recipe else pkg
    if await _existing_binary(bin_name):
        return "installed"
    if recipe and recipe.get("strategy") == "manual":
        return "manual-only"
    return "missing"


__all__ = ["INSTALL_RECIPES", "try_install", "detect_status", "hermes_lsp_bin_dir"]
