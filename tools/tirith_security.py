"""Tirith pre-exec security scanning wrapper.

Runs the tirith binary as a subprocess to scan commands for content-level
threats (homograph URLs, pipe-to-interpreter, terminal injection, etc.).

Exit code is the verdict source of truth:
  0 = allow, 1 = block, 2 = warn

JSON stdout enriches findings/summary but never overrides the verdict.
Operational failures (spawn error, timeout, unknown exit code) respect
the fail_open config setting. Programming errors propagate.

Auto-install: if tirith is not found on PATH or at the configured path,
it is automatically downloaded from GitHub releases to $HERMES_HOME/bin/tirith.
The download always verifies SHA-256 checksums.  When cosign is available on
PATH, provenance verification (GitHub Actions workflow signature) is also
performed.  If cosign is not installed, the download proceeds with SHA-256
verification only — still secure via HTTPS + checksum, just without supply
chain provenance proof. Installation runs in a tracked asyncio task so startup
never blocks.
"""

import asyncio
import hashlib
import io
import json
import logging
import os
import platform
import shutil
import stat
import tarfile
import time

import aiofiles
import aiofiles.os
import aiofiles.tempfile
import httpx

from agent.ssl_verify import _create_httpx_client

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_REPO = "sheeki03/tirith"

# Cosign provenance verification — pinned to the specific release workflow
_COSIGN_IDENTITY_REGEXP = f"^https://github.com/{_REPO}/\\.github/workflows/release\\.yml@refs/tags/v"
_COSIGN_ISSUER = "https://token.actions.githubusercontent.com"

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes"}


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


async def _load_security_config() -> dict:
    """Load security settings from config.yaml, with env var overrides."""
    defaults = {
        "tirith_enabled": True,
        "tirith_path": "tirith",
        "tirith_timeout": 5,
        "tirith_fail_open": True,
    }
    try:
        from hermes_cli.config import load_config_readonly

        cfg = (await load_config_readonly()).get("security", {}) or {}
    except Exception:
        cfg = {}

    return {
        "tirith_enabled": _env_bool("TIRITH_ENABLED", cfg.get("tirith_enabled", defaults["tirith_enabled"])),
        "tirith_path": os.getenv("TIRITH_BIN", cfg.get("tirith_path", defaults["tirith_path"])),
        "tirith_timeout": _env_int("TIRITH_TIMEOUT", cfg.get("tirith_timeout", defaults["tirith_timeout"])),
        "tirith_fail_open": _env_bool("TIRITH_FAIL_OPEN", cfg.get("tirith_fail_open", defaults["tirith_fail_open"])),
    }


# ---------------------------------------------------------------------------
# Auto-install
# ---------------------------------------------------------------------------

# Cached path after first resolution (avoids repeated shutil.which per command).
# _INSTALL_FAILED means "we tried and failed" — prevents retry on every command.
_resolved_path: str | None | bool = None
_INSTALL_FAILED = False  # sentinel: distinct from "not yet tried"
_install_failure_reason: str = ""  # reason tag when _resolved_path is _INSTALL_FAILED

# Circuit breaker: after _CRASH_LIMIT consecutive spawn/execution failures,
# disable tirith for the rest of the process to prevent agent hangs (#41400).
# Reset on successful execution (see _record_tirith_crash / check_command_security).
#
# ``_record_tirith_crash`` contains no await, so each update is atomic with
# respect to other tasks on the owning event loop. This matches the lock-free
# error counters in ``mcp_tool.py``.
_CRASH_LIMIT = 3
_crash_count: int = 0
_circuit_open: bool = False


async def _finish_process_communicate(
    process: asyncio.subprocess.Process,
    communicate_task: asyncio.Task[tuple[bytes, bytes]],
) -> tuple[bytes, bytes]:
    """Drain and reap one owned security helper through repeated cancellation."""
    async def drain_or_wait() -> tuple[bytes, bytes]:
        try:
            return await communicate_task
        except BaseException:
            await process.wait()
            raise

    cleanup_task = asyncio.create_task(drain_or_wait())
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            output = await asyncio.shield(cleanup_task)
            break
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised below
            if cleanup_task.cancelled():
                raise
            if cancellation is None:
                cancellation = exc
        except Exception as exc:
            if cancellation is not None:
                raise cancellation from exc
            raise
    if cancellation is not None:
        raise cancellation
    return output


def _record_tirith_crash() -> None:
    """Increment the crash counter and open the circuit breaker if needed."""
    global _crash_count, _circuit_open
    _crash_count += 1
    if _crash_count >= _CRASH_LIMIT:
        _circuit_open = True
        logger.warning(
            "tirith circuit breaker opened after %d consecutive failures; "
            "disabling for the rest of the process",
            _crash_count,
        )

# Background install task coordination
_install_lock = asyncio.Lock()
_install_task: asyncio.Task[str | None] | None = None

# Warning de-duplication. The spawn/path warnings live in the hot path —
# without this dedupe set, a Windows install where ``tirith`` isn't on PATH
# (e.g. background install thread still running, or install marked failed)
# spams ``tirith spawn failed: [WinError 2]...`` once per terminal command,
# easily filling errors.log with hundreds of identical lines.
_warned_messages: set[str] = set()
def _warn_once(key: str, message: str, *args) -> None:
    """``logger.warning`` but at-most-once per ``key`` for the process
    lifetime. Used to avoid drowning the log when a fail-open tirith
    misconfiguration fires on every command."""
    if key in _warned_messages:
        return
    _warned_messages.add(key)
    logger.warning(message, *args)


def _reset_spawn_warning_state() -> None:
    """Clear the warn-once dedupe set. Called when tirith is freshly
    (re)installed so a subsequent failure surfaces again — e.g. user
    deletes the binary mid-session.
    """
    _warned_messages.clear()

# Disk-persistent failure marker — avoids retry across process restarts
_MARKER_TTL = 86400  # 24 hours


def _get_hermes_home() -> str:
    """Return the Hermes home directory, respecting HERMES_HOME env var."""
    return str(get_hermes_home())


def _failure_marker_path() -> str:
    """Return the path to the install-failure marker file."""
    return os.path.join(_get_hermes_home(), ".tirith-install-failed")


async def _read_failure_reason() -> str | None:
    """Read the failure reason from the disk marker.

    Returns the reason string, or None if the marker doesn't exist or is
    older than _MARKER_TTL.
    """
    try:
        p = _failure_marker_path()
        mtime = (await aiofiles.os.stat(p)).st_mtime
        if (time.time() - mtime) >= _MARKER_TTL:
            return None
        async with aiofiles.open(p, "r", encoding="utf-8") as f:
            return (await f.read()).strip()
    except OSError:
        return None


async def _is_install_failed_on_disk() -> bool:
    """Check if a recent install failure was persisted to disk.

    Returns False (allowing retry) when:
    - No marker exists
    - Marker is older than _MARKER_TTL (24h)
    - Marker reason is 'cosign_missing' and cosign is now on PATH
    """
    reason = await _read_failure_reason()
    if reason is None:
        return False
    if reason == "cosign_missing" and await aiofiles.os.wrap(shutil.which)("cosign"):
        await _clear_install_failed()
        return False
    return True


async def _mark_install_failed(reason: str = ""):
    """Persist install failure to disk to avoid retry on next process.

    Args:
        reason: Short tag identifying the failure cause. Use "cosign_missing"
                when cosign is not on PATH so the marker can be auto-cleared
                once cosign becomes available.
    """
    try:
        p = _failure_marker_path()
        await aiofiles.os.makedirs(os.path.dirname(p), exist_ok=True)
        async with aiofiles.open(p, "w", encoding="utf-8") as f:
            await f.write(reason)
    except OSError:
        pass


async def _clear_install_failed():
    """Remove the failure marker after successful install."""
    # Reset the warn-once dedupe set so a subsequent failure (e.g. user
    # deletes the binary) surfaces in the log again instead of being
    # silently suppressed by a stale dedupe key from before the fix.
    _reset_spawn_warning_state()
    try:
        await aiofiles.os.remove(_failure_marker_path())
    except OSError:
        pass


async def _hermes_bin_dir() -> str:
    """Return $HERMES_HOME/bin, creating it if needed."""
    d = os.path.join(_get_hermes_home(), "bin")
    await aiofiles.os.makedirs(d, exist_ok=True)
    return d


def _detect_target() -> str | None:
    """Return the Rust target triple for the current platform, or None.

    Windows is intentionally unsupported — tirith does not ship a Windows
    build. Callers should treat `None` as "this platform will never have
    tirith" and silently fall back to pattern-matching guards.
    """
    system = platform.system()
    machine = platform.machine().lower()

    # Android (Termux) is ABI-compatible with Linux — reuse Linux binaries.
    if system == "Darwin":
        plat = "apple-darwin"
    elif system in {"Linux", "Android"}:
        plat = "unknown-linux-gnu"
    else:
        return None

    if machine in {"x86_64", "amd64"}:
        arch = "x86_64"
    elif machine in {"aarch64", "arm64"}:
        arch = "aarch64"
    else:
        return None

    return f"{arch}-{plat}"


def is_platform_supported() -> bool:
    """True when tirith ships a prebuilt binary for this OS+arch.

    Used by callers (CLI banner, etc.) to distinguish "tirith failed to
    install" from "tirith was never going to install here" — the latter
    is silent because there is nothing the user can do about it.
    """
    return _detect_target() is not None


async def _download_file(url: str, dest: str, timeout: int = 10):
    """Download a URL to a local file."""
    from agent.secret_scope import get_secret

    token = get_secret("GITHUB_TOKEN")
    headers = {"Authorization": f"token {token}"} if token else None
    async with (await _create_httpx_client(
        follow_redirects=True,
        timeout=timeout,
        headers=headers,
    )) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            async with aiofiles.open(dest, "wb") as handle:
                async for chunk in response.aiter_bytes():
                    await handle.write(chunk)


async def _verify_cosign(
    checksums_path: str,
    sig_path: str,
    cert_path: str,
) -> bool | None:
    """Verify cosign provenance signature on checksums.txt.

    Returns:
        True  — cosign verified successfully
        False — cosign found but verification failed
        None  — cosign not available (not on PATH, or execution failed)

    The caller treats both False and None as "abort auto-install" — only
    True allows the install to proceed.
    """
    cosign = await aiofiles.os.wrap(shutil.which)("cosign")
    if not cosign:
        logger.info("cosign not found on PATH")
        return None

    try:
        process = await asyncio.create_subprocess_exec(
            cosign,
            "verify-blob",
            "--certificate",
            cert_path,
            "--signature",
            sig_path,
            "--certificate-identity-regexp",
            _COSIGN_IDENTITY_REGEXP,
            "--certificate-oidc-issuer",
            _COSIGN_ISSUER,
            checksums_path,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        communicate_task = asyncio.create_task(process.communicate())
        timeout_error: TimeoutError | None = None
        try:
            _, stderr = await asyncio.wait_for(
                asyncio.shield(communicate_task), timeout=15
            )
        except asyncio.CancelledError:
            process.kill()
            try:
                await _finish_process_communicate(process, communicate_task)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("cosign cleanup after cancellation failed", exc_info=True)
            raise
        except TimeoutError as exc:
            timeout_error = exc

        if timeout_error is not None:
            process.kill()
            try:
                await _finish_process_communicate(process, communicate_task)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("cosign cleanup after timeout failed", exc_info=True)
            logger.warning("cosign execution failed: timed out")
            return None
        if process.returncode == 0:
            logger.info("cosign provenance verification passed")
            return True
        else:
            logger.warning("cosign verification failed (exit %d): %s",
                          process.returncode, stderr.decode("utf-8", errors="replace").strip())
            return False
    except OSError as exc:
        logger.warning("cosign execution failed: %s", exc)
        return None


async def _verify_checksum(
    archive_path: str,
    checksums_path: str,
    archive_name: str,
) -> bool:
    """Verify SHA-256 of the archive against checksums.txt."""
    expected = None
    async with aiofiles.open(checksums_path, encoding="utf-8") as f:
        async for line in f:
            # Format: "<hash>  <filename>"
            parts = line.strip().split("  ", 1)
            if len(parts) == 2 and parts[1] == archive_name:
                expected = parts[0]
                break
    if not expected:
        logger.warning("No checksum entry for %s", archive_name)
        return False

    sha = hashlib.sha256()
    async with aiofiles.open(archive_path, "rb") as f:
        while chunk := await f.read(8192):
            sha.update(chunk)
    actual = sha.hexdigest()
    if actual != expected:
        logger.warning("Checksum mismatch: expected %s, got %s", expected, actual)
        return False
    return True


async def _extract_tirith_binary(
    tar: tarfile.TarFile,
    dest_dir: str,
    log,
) -> tuple[str | None, str]:
    """Extract the tirith binary from a release archive into dest_dir."""
    for member in tar.getmembers():
        if member.name == "tirith" or member.name.endswith("/tirith"):
            if ".." in member.name:
                continue
            if not member.isfile():
                log("tirith archive member is not a regular file: %s", member.name)
                return None, "binary_not_regular_file"
            src_file = tar.extractfile(member)
            if src_file is None:
                log("tirith binary could not be read from archive")
                return None, "binary_extract_failed"

            dest_path = os.path.join(dest_dir, "tirith")
            try:
                binary = src_file.read()
                async with aiofiles.open(dest_path, "wb") as out:
                    await out.write(binary)
            finally:
                src_file.close()
            return dest_path, ""

    log("tirith binary not found in archive")
    return None, "binary_not_in_archive"


async def _install_tirith(
    *,
    log_failures: bool = True,
) -> tuple[str | None, str]:
    """Download and install tirith to $HERMES_HOME/bin/tirith.

    Verifies provenance via cosign and SHA-256 checksum.
    Returns (installed_path, failure_reason).  On success failure_reason is "".
    failure_reason is a short tag used by the disk marker to decide if the
    failure is retryable (e.g. "cosign_missing" clears when cosign appears).
    """
    log = logger.warning if log_failures else logger.debug

    target = _detect_target()
    if not target:
        logger.info("tirith auto-install: unsupported platform %s/%s",
                     platform.system(), platform.machine())
        return None, "unsupported_platform"

    archive_name = f"tirith-{target}.tar.gz"
    base_url = f"https://github.com/{_REPO}/releases/latest/download"

    try:
        temp_context = aiofiles.tempfile.TemporaryDirectory(prefix="tirith-install-")
        async with temp_context as tmpdir:
            archive_path = os.path.join(tmpdir, archive_name)
            checksums_path = os.path.join(tmpdir, "checksums.txt")
            sig_path = os.path.join(tmpdir, "checksums.txt.sig")
            cert_path = os.path.join(tmpdir, "checksums.txt.pem")

            logger.info(
                "tirith not found — downloading latest release for %s...",
                target,
            )

            try:
                await _download_file(f"{base_url}/{archive_name}", archive_path)
                await _download_file(f"{base_url}/checksums.txt", checksums_path)
            except Exception as exc:
                log("tirith download failed: %s", exc)
                return None, "download_failed"

            # Cosign provenance verification — preferred but not mandatory.
            cosign_verified = False
            if await aiofiles.os.wrap(shutil.which)("cosign"):
                try:
                    await _download_file(
                        f"{base_url}/checksums.txt.sig",
                        sig_path,
                    )
                    await _download_file(
                        f"{base_url}/checksums.txt.pem",
                        cert_path,
                    )
                except Exception as exc:
                    logger.info(
                        "cosign artifacts unavailable (%s), proceeding with "
                        "SHA-256 only",
                        exc,
                    )
                else:
                    cosign_result = await _verify_cosign(
                        checksums_path,
                        sig_path,
                        cert_path,
                    )
                    if cosign_result is True:
                        cosign_verified = True
                    elif cosign_result is False:
                        log(
                            "tirith install aborted: cosign provenance "
                            "verification failed"
                        )
                        return None, "cosign_verification_failed"
                    else:
                        logger.info(
                            "cosign execution failed, proceeding with SHA-256 only"
                        )
            else:
                logger.info(
                    "cosign not on PATH — installing tirith with SHA-256 "
                    "verification only (install cosign for full supply chain "
                    "verification)"
                )

            if not await _verify_checksum(
                archive_path,
                checksums_path,
                archive_name,
            ):
                return None, "checksum_failed"

            async with aiofiles.open(archive_path, "rb") as archive:
                archive_data = await archive.read()
            with tarfile.open(fileobj=io.BytesIO(archive_data), mode="r:gz") as tar:
                src, reason = await _extract_tirith_binary(tar, tmpdir, log)
                if src is None:
                    return None, reason

            dest = os.path.join(await _hermes_bin_dir(), "tirith")
            try:
                await aiofiles.os.replace(src, dest)
            except OSError:
                try:
                    async with (
                        aiofiles.open(src, "rb") as source,
                        aiofiles.open(dest, "wb") as target_file,
                    ):
                        while chunk := await source.read(64 * 1024):
                            await target_file.write(chunk)
                except OSError:
                    try:
                        await aiofiles.os.remove(dest)
                    except OSError:
                        pass
                    return None, "cross_device_copy_failed"
            mode = (await aiofiles.os.stat(dest)).st_mode
            await aiofiles.os.wrap(os.chmod)(
                dest,
                mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
            )

            verification = (
                "cosign + SHA-256" if cosign_verified else "SHA-256 only"
            )
            logger.info("tirith installed to %s (%s)", dest, verification)
            return dest, ""

    except OSError as exc:
        log("tirith install failed: cannot create temp dir: %s", exc)
        return None, "no_space"


def _is_explicit_path(configured_path: str) -> bool:
    """Return True if the user explicitly configured a non-default tirith path."""
    return configured_path != "tirith"


async def _resolve_tirith_path(configured_path: str) -> str:
    """Resolve the tirith binary path, auto-installing if necessary.

    If the user explicitly set a path (anything other than the bare "tirith"
    default), that path is authoritative — we never fall through to
    auto-download a different binary.

    For the default "tirith":
    1. PATH lookup via shutil.which
    2. $HERMES_HOME/bin/tirith (previously auto-installed)
    3. Auto-install from GitHub releases → $HERMES_HOME/bin/tirith

    Failed installs are cached for the process lifetime (and persisted to
    disk for 24h) to avoid repeated network attempts.
    """
    global _resolved_path, _install_failure_reason

    # Fast path: successfully resolved on a previous call.
    if _resolved_path is not None and _resolved_path is not _INSTALL_FAILED:
        return _resolved_path

    expanded = os.path.expanduser(configured_path)
    explicit = _is_explicit_path(configured_path)
    install_failed = _resolved_path is _INSTALL_FAILED

    # Platform has no tirith build (Windows etc.). Cache the verdict and
    # return the unexpanded configured path — the spawn loop will fail-open
    # via the dedupe'd OSError handler, but only after the first call; on
    # subsequent calls the fast-path above short-circuits before spawning.
    if not explicit and not is_platform_supported():
        _resolved_path = _INSTALL_FAILED
        _install_failure_reason = "unsupported_platform"
        return expanded

    # Explicit path: check it and stop. Never auto-download a replacement.
    if explicit:
        if await aiofiles.os.path.isfile(expanded) and await aiofiles.os.wrap(
            os.access
        )(expanded, os.X_OK):
            _resolved_path = expanded
            return expanded
        # Also try shutil.which in case it's a bare name on PATH
        found = await aiofiles.os.wrap(shutil.which)(expanded)
        if found:
            _resolved_path = found
            return found
        logger.warning("Configured tirith path %r not found; scanning disabled", configured_path)
        _resolved_path = _INSTALL_FAILED
        _install_failure_reason = "explicit_path_missing"
        return expanded

    # Default "tirith" — always re-run cheap local checks so a manual
    # install is picked up even after a previous network failure (P2 fix:
    # long-lived gateway/CLI recovers without restart).
    found = await aiofiles.os.wrap(shutil.which)("tirith")
    if found:
        _resolved_path = found
        _install_failure_reason = ""
        await _clear_install_failed()
        return found

    hermes_bin = os.path.join(await _hermes_bin_dir(), "tirith")
    if await aiofiles.os.path.isfile(hermes_bin) and await aiofiles.os.wrap(
        os.access
    )(hermes_bin, os.X_OK):
        _resolved_path = hermes_bin
        _install_failure_reason = ""
        await _clear_install_failed()
        return hermes_bin

    # Local checks failed.  If a previous install attempt already failed,
    # skip the network retry — UNLESS the failure was "cosign_missing" and
    # cosign is now available (retryable cause resolved in-process).
    if install_failed:
        if (
            _install_failure_reason == "cosign_missing"
            and await aiofiles.os.wrap(shutil.which)("cosign")
        ):
            # Retryable cause resolved — clear sentinel and fall through to retry
            _resolved_path = None
            _install_failure_reason = ""
            await _clear_install_failed()
            install_failed = False
        else:
            return expanded

    # Check disk failure marker before attempting network download.
    # Preserve the marker's real reason so in-memory retry logic can
    # detect retryable causes (e.g. cosign_missing) without restart.
    disk_reason = await _read_failure_reason()
    if disk_reason is not None and await _is_install_failed_on_disk():
        _resolved_path = _INSTALL_FAILED
        _install_failure_reason = disk_reason
        return expanded

    installed = await _background_install()
    if installed:
        return installed
    return expanded


async def _background_install(*, log_failures: bool = True) -> str | None:
    """Download and install tirith while serializing concurrent attempts."""
    global _resolved_path, _install_failure_reason
    async with _install_lock:
        # Double-check after acquiring the lock; another task may have resolved it.
        if _resolved_path is _INSTALL_FAILED:
            return None
        if _resolved_path is not None:
            return _resolved_path

        # Re-check local paths (may have been installed by another process)
        found = await aiofiles.os.wrap(shutil.which)("tirith")
        if found:
            _resolved_path = found
            _install_failure_reason = ""
            return found

        hermes_bin = os.path.join(await _hermes_bin_dir(), "tirith")
        if await aiofiles.os.path.isfile(hermes_bin) and await aiofiles.os.wrap(
            os.access
        )(hermes_bin, os.X_OK):
            _resolved_path = hermes_bin
            _install_failure_reason = ""
            return hermes_bin

        installed, reason = await _install_tirith(log_failures=log_failures)
        if installed:
            _resolved_path = installed
            _install_failure_reason = ""
            await _clear_install_failed()
        else:
            _resolved_path = _INSTALL_FAILED
            _install_failure_reason = reason
            await _mark_install_failed(reason)
        return installed


async def ensure_installed(*, log_failures: bool = True):
    """Ensure tirith is available, downloading in background if needed.

    Quick PATH/local checks are awaited without blocking the loop; network
    download runs in a native task. Safe to call multiple times.
    Returns the resolved path immediately if available, or None.
    """
    global _resolved_path, _install_task, _install_failure_reason

    cfg = await _load_security_config()
    if not cfg["tirith_enabled"]:
        return None

    # Already resolved from a previous call
    if _resolved_path is not None and _resolved_path is not _INSTALL_FAILED:
        path = _resolved_path
        if await aiofiles.os.path.isfile(path) and await aiofiles.os.wrap(
            os.access
        )(path, os.X_OK):
            return path
        return None

    # Platform has no tirith build (e.g. Windows) — don't probe PATH,
    # don't start a download thread, don't write a disk failure marker.
    # Pattern-matching guards still run; this path stays silent.
    if not is_platform_supported():
        _resolved_path = _INSTALL_FAILED
        _install_failure_reason = "unsupported_platform"
        return None

    configured_path = cfg["tirith_path"]
    explicit = _is_explicit_path(configured_path)
    expanded = os.path.expanduser(configured_path)

    # Explicit path: synchronous check only, no download
    if explicit:
        if await aiofiles.os.path.isfile(expanded) and await aiofiles.os.wrap(
            os.access
        )(expanded, os.X_OK):
            _resolved_path = expanded
            return expanded
        found = await aiofiles.os.wrap(shutil.which)(expanded)
        if found:
            _resolved_path = found
            return found
        _resolved_path = _INSTALL_FAILED
        _install_failure_reason = "explicit_path_missing"
        return None

    # Default "tirith" — quick local checks first (no network)
    found = await aiofiles.os.wrap(shutil.which)("tirith")
    if found:
        _resolved_path = found
        _install_failure_reason = ""
        await _clear_install_failed()
        return found

    hermes_bin = os.path.join(await _hermes_bin_dir(), "tirith")
    if await aiofiles.os.path.isfile(hermes_bin) and await aiofiles.os.wrap(
        os.access
    )(hermes_bin, os.X_OK):
        _resolved_path = hermes_bin
        _install_failure_reason = ""
        await _clear_install_failed()
        return hermes_bin

    # If previously failed in-memory, check if the cause is now resolved
    if _resolved_path is _INSTALL_FAILED:
        if (
            _install_failure_reason == "cosign_missing"
            and await aiofiles.os.wrap(shutil.which)("cosign")
        ):
            _resolved_path = None
            _install_failure_reason = ""
            await _clear_install_failed()
        else:
            return None

    # Check disk failure marker (skip network attempt for 24h, unless
    # the cosign_missing reason was resolved — handled by _is_install_failed_on_disk).
    # Preserve the marker's real reason for in-memory retry logic.
    disk_reason = await _read_failure_reason()
    if disk_reason is not None and await _is_install_failed_on_disk():
        _resolved_path = _INSTALL_FAILED
        _install_failure_reason = disk_reason
        return None

    # Need to download — launch one tracked native task so startup does not block.
    if _install_task is None or _install_task.done():
        _install_task = asyncio.create_task(
            _background_install(log_failures=log_failures),
            name="tirith-install",
        )

    return None  # Not available yet; commands will fail-open until ready


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

_MAX_FINDINGS = 50
_MAX_SUMMARY_LEN = 500


async def check_command_security(command: str) -> dict:
    """Run tirith security scan on a command.

    Exit code determines action (0=allow, 1=block, 2=warn). JSON enriches
    findings/summary. Spawn failures and timeouts respect fail_open config.
    Programming errors propagate.

    Returns:
        {"action": "allow"|"warn"|"block", "findings": [...], "summary": str}
    """
    global _crash_count, _circuit_open

    cfg = await _load_security_config()

    if not cfg["tirith_enabled"]:
        return {"action": "allow", "findings": [], "summary": ""}

    # Circuit breaker: if tirith has crashed _CRASH_LIMIT times in a row,
    # stop trying for the rest of the process.  Without this, a corrupted
    # or missing binary causes every tool call to hit the same spawn failure
    # → fail-open → agent retry loop, hanging the user for 20+ minutes
    # (issue #41400).
    if _circuit_open:
        return {"action": "allow", "findings": [], "summary": "tirith disabled (circuit breaker)"}

    # Unsupported platform (Windows etc.) — tirith has no binary here and
    # never will. Skip the resolver entirely so we don't even try to spawn.
    # Pattern-matching guards still run via the rest of approval.py.
    if not is_platform_supported():
        return {"action": "allow", "findings": [], "summary": ""}

    tirith_path = await _resolve_tirith_path(cfg["tirith_path"])
    timeout = cfg["tirith_timeout"]
    fail_open = cfg["tirith_fail_open"]

    if tirith_path is None:
        _warn_once(
            "tirith_path_none",
            "tirith path resolved to None; scanning disabled",
        )
        if fail_open:
            return {"action": "allow", "findings": [], "summary": "tirith path unavailable"}
        return {"action": "block", "findings": [], "summary": "tirith path unavailable (fail-closed)"}

    try:
        process = await asyncio.create_subprocess_exec(
            tirith_path,
            "check",
            "--json",
            "--non-interactive",
            "--shell",
            "posix",
            "--",
            command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        communicate_task = asyncio.create_task(process.communicate())
        timeout_error: TimeoutError | None = None
        try:
            stdout_bytes, _ = await asyncio.wait_for(
                asyncio.shield(communicate_task),
                timeout=timeout,
            )
        except asyncio.CancelledError:
            process.kill()
            try:
                await _finish_process_communicate(process, communicate_task)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("tirith cleanup after cancellation failed", exc_info=True)
            raise
        except TimeoutError as exc:
            timeout_error = exc

        if timeout_error is not None:
            process.kill()
            try:
                await _finish_process_communicate(process, communicate_task)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("tirith cleanup after timeout failed", exc_info=True)
            raise TimeoutError from timeout_error
    except OSError as exc:
        # Covers FileNotFoundError, PermissionError, exec format error.
        # Dedupe by ``(errno, exc class)`` so a transient failure mode
        # surfaces once but doesn't drown the log on every command —
        # commonly seen on Windows when the configured path "tirith"
        # isn't on PATH yet (background install still running, or
        # install marked failed for the day).
        spawn_key = f"tirith_spawn_failed:{type(exc).__name__}:{getattr(exc, 'errno', '')}"
        _warn_once(spawn_key, "tirith spawn failed: %s", exc)
        _record_tirith_crash()
        if fail_open:
            return {"action": "allow", "findings": [], "summary": f"tirith unavailable: {exc}"}
        return {"action": "block", "findings": [], "summary": f"tirith spawn failed (fail-closed): {exc}"}
    except TimeoutError:
        _warn_once(
            f"tirith_timeout:{timeout}",
            "tirith timed out after %ds",
            timeout,
        )
        _record_tirith_crash()
        if fail_open:
            return {"action": "allow", "findings": [], "summary": f"tirith timed out ({timeout}s)"}
        return {"action": "block", "findings": [], "summary": "tirith timed out (fail-closed)"}

    # Map exit code to action
    exit_code = process.returncode
    if exit_code == 0:
        action = "allow"
        # Successful execution — reset circuit breaker
        _crash_count = 0
    elif exit_code == 1:
        action = "block"
    elif exit_code == 2:
        action = "warn"
    else:
        # Unknown exit code (includes signal-killed processes like -11/SIGSEGV)
        # — respect fail_open
        logger.warning("tirith returned unexpected exit code %d", exit_code)
        _record_tirith_crash()
        if fail_open:
            return {"action": "allow", "findings": [], "summary": f"tirith exit code {exit_code} (fail-open)"}
        return {"action": "block", "findings": [], "summary": f"tirith exit code {exit_code} (fail-closed)"}

    # Parse JSON for enrichment (never overrides the exit code verdict)
    findings = []
    summary = ""
    try:
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        data = json.loads(stdout) if stdout.strip() else {}
        raw_findings = data.get("findings", [])
        findings = raw_findings[:_MAX_FINDINGS]
        summary = (data.get("summary", "") or "")[:_MAX_SUMMARY_LEN]
    except (json.JSONDecodeError, AttributeError):
        # JSON parse failure degrades findings/summary, not the verdict
        logger.debug("tirith JSON parse failed, using exit code only")
        if action == "block":
            summary = "security issue detected (details unavailable)"
        elif action == "warn":
            summary = "security warning detected (details unavailable)"

    # Suppress warn verdicts that consist solely of a lookalike_tld finding for
    # the .app TLD.  .app is a legitimate gTLD used by many production services
    # and the "can be confused with file extensions" heuristic generates false
    # positives for normal API calls.  Any other finding (including other
    # lookalike_tld entries for non-.app TLDs) preserves the warn action.
    if action == "warn" and findings:
        non_suppressible = [f for f in findings if not _is_app_tld_finding(f)]
        if not non_suppressible:
            action = "allow"
            findings = []
            summary = ""

    return {"action": action, "findings": findings, "summary": summary}


def _is_app_tld_finding(finding: dict) -> bool:
    """Return True if this finding is a lookalike_tld warning for the .app TLD only.

    Checks the rule_id and inspects common value/detail field names that
    Tirith may use to carry the TLD string.
    """
    if not isinstance(finding, dict):
        return False
    if finding.get("rule_id") != "lookalike_tld":
        return False
    for field in ("value", "tld", "detail", "description", "message"):
        val = finding.get(field)
        if val is not None and ".app" in str(val).lower():
            return True
    return False
