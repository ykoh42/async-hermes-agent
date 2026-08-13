"""GitHub Copilot authentication utilities.

Implements the OAuth device code flow used by the Copilot CLI and handles
token validation/exchange for the Copilot API without blocking the event loop.

Token type support (per GitHub docs):
  gho_          OAuth token           ✓  (default via copilot login)
  github_pat_   Fine-grained PAT      ✓  (needs Copilot Requests permission)
  ghu_          GitHub App token      ✓  (via environment variable)
  ghp_          Classic PAT           ✗  NOT SUPPORTED

Credential search order (matching Copilot CLI behaviour):
  1. COPILOT_GITHUB_TOKEN env var
  2. GH_TOKEN env var
  3. GITHUB_TOKEN env var
  4. gh auth token  CLI fallback
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import shutil
import threading
import time
import weakref
from collections.abc import Hashable, Iterator, MutableMapping
from pathlib import Path
from typing import Optional

import aiofiles
import aiofiles.os
import httpx

from agent.secret_scope import (
    get_secret as _get_secret,
    is_multiplex_active as _is_multiplex_active,
)
from agent.ssl_verify import _create_httpx_client

logger = logging.getLogger(__name__)

# OAuth device code flow constants — VS Code's GitHub App client ID.
# The previous opencode OAuth App ID (Ov23li8tweQw6odWQebz) produces gho_*
# tokens that cannot be exchanged for Copilot API JWTs (404 on
# /copilot_internal/v2/token). VS Code's App ID produces ghu_* tokens
# that support exchange, which is required to access internal-only models
# (e.g. claude-opus-4.6-1m) and enterprise endpoints.
# Tested on Individual and Enterprise accounts.
COPILOT_OAUTH_CLIENT_ID = "Iv1.b507a08c87ecfe98"
# Token type prefixes
_CLASSIC_PAT_PREFIX = "ghp_"
_SUPPORTED_PREFIXES = ("gho_", "github_pat_", "ghu_")

# Env var search order (matches Copilot CLI)
COPILOT_ENV_VARS = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")

# Polling constants
_DEVICE_CODE_POLL_INTERVAL = 5  # seconds
_DEVICE_CODE_POLL_SAFETY_MARGIN = 3  # seconds


async def _finish_process_wait(process: asyncio.subprocess.Process) -> int:
    """Reap one owned gh process before propagating cancellation."""
    wait_task = asyncio.create_task(process.wait())
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            return_code = await asyncio.shield(wait_task)
            break
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised below
            if wait_task.cancelled():
                raise
            if cancellation is None:
                cancellation = exc
        except Exception as exc:
            if cancellation is not None:
                raise cancellation from exc
            raise
    if cancellation is not None:
        raise cancellation
    return return_code


async def _finish_process_communicate(
    process: asyncio.subprocess.Process,
    communicate_task: asyncio.Task[tuple[bytes | None, bytes | None]],
) -> tuple[bytes | None, bytes | None]:
    """Drain pipes and reap one owned gh process."""
    async def drain_or_wait() -> tuple[bytes | None, bytes | None]:
        try:
            return await communicate_task
        except BaseException:
            await _finish_process_wait(process)
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


def validate_copilot_token(token: str) -> tuple[bool, str]:
    """Validate that a token is usable with the Copilot API.

    Returns (valid, message).
    """
    token = token.strip()
    if not token:
        return False, "Empty token"

    if token.startswith(_CLASSIC_PAT_PREFIX):
        return False, (
            "Classic Personal Access Tokens (ghp_*) are not supported by the "
            "Copilot API. Use one of:\n"
            "  → `copilot login` or `hermes model` to authenticate via OAuth\n"
            "  → A fine-grained PAT (github_pat_*) with Copilot Requests permission\n"
            "  → `gh auth login` with the default device code flow (produces gho_* tokens)"
        )

    return True, "OK"


async def resolve_copilot_token() -> tuple[str, str]:
    """Resolve a GitHub token suitable for Copilot API use.

    Returns (token, source) where source describes where the token came from.
    Raises ValueError if only a classic PAT is available.
    """
    # 1. Check env vars in priority order
    any_env_var_set = False
    for env_var in COPILOT_ENV_VARS:
        val = (_get_secret(env_var, "") or "").strip()
        if val:
            any_env_var_set = True
            valid, msg = validate_copilot_token(val)
            if not valid:
                logger.warning(
                    "Token from %s is not supported: %s", env_var, msg
                )
                continue
            return val, env_var

    if any_env_var_set:
        logger.debug(
            "Copilot env var(s) set but none held a supported token; "
            "skipping `gh auth token` fallback to honor explicit env-var "
            "intent (and avoid the subprocess cost on cold start, #60800)."
        )
        return "", ""

    # ``gh auth token`` reads the process user's global credential store. It
    # cannot identify the active Hermes profile, so falling through here in a
    # multiplexed service could authenticate profile B as profile A. Scoped
    # environment tokens above remain supported; the global CLI store does not.
    if _is_multiplex_active():
        logger.debug(
            "Skipping `gh auth token` fallback while profile multiplexing is active"
        )
        return "", ""

    # 2. Fall back to gh auth token
    token = await _try_gh_cli_token()
    if token:
        valid, msg = validate_copilot_token(token)
        if not valid:
            raise ValueError(
                f"Token from `gh auth token` is a classic PAT (ghp_*). {msg}"
            )
        return token, "gh auth token"

    return "", ""


async def _gh_cli_candidates() -> list[str]:
    """Return candidate ``gh`` binary paths, including common Homebrew installs."""
    candidates: list[str] = []

    resolved = await aiofiles.os.wrap(shutil.which)("gh")
    if resolved:
        candidates.append(resolved)

    for candidate in (
        "/opt/homebrew/bin/gh",
        "/usr/local/bin/gh",
        str(Path.home() / ".local" / "bin" / "gh"),
    ):
        if candidate in candidates:
            continue
        if await aiofiles.os.path.isfile(candidate) and await aiofiles.os.access(
            candidate, os.X_OK
        ):
            candidates.append(candidate)

    return candidates


async def _try_gh_cli_token() -> Optional[str]:
    """Return a token from ``gh auth token`` when the GitHub CLI is available.

    When COPILOT_GH_HOST is set, passes ``--hostname`` so gh returns the
    correct host's token.  Also strips GITHUB_TOKEN / GH_TOKEN from the
    subprocess environment so ``gh`` reads from its own credential store
    (hosts.yml) instead of just echoing the env var back.
    """
    if _is_multiplex_active():
        logger.debug(
            "Skipping process-global GitHub CLI credentials while profile "
            "multiplexing is active"
        )
        return None

    hostname = (_get_secret("COPILOT_GH_HOST", "") or "").strip()

    # Build a clean env so gh doesn't short-circuit on GITHUB_TOKEN / GH_TOKEN
    clean_env = {k: v for k, v in os.environ.items()
                 if k not in {"GITHUB_TOKEN", "GH_TOKEN"}}

    for gh_path in await _gh_cli_candidates():
        cmd = [gh_path, "auth", "token"]
        if hostname:
            cmd += ["--hostname", hostname]
        process = None
        communicate_task = None
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=clean_env,
            )
            communicate_task = asyncio.create_task(process.communicate())
            stdout, _ = await asyncio.wait_for(
                asyncio.shield(communicate_task), timeout=5
            )
        except asyncio.TimeoutError:
            process.kill()
            await _finish_process_communicate(process, communicate_task)
            logger.debug("gh CLI token lookup timed out (%s)", gh_path)
            continue
        except asyncio.CancelledError:
            if process is not None and process.returncode is None:
                process.kill()
            if process is not None and communicate_task is not None:
                await _finish_process_communicate(process, communicate_task)
            raise
        except FileNotFoundError as exc:
            logger.debug("gh CLI token lookup failed (%s): %s", gh_path, exc)
            continue
        token = stdout.decode("utf-8", errors="replace").strip()
        if process.returncode == 0 and token:
            return token
    return None


# ─── OAuth Device Code Flow ────────────────────────────────────────────────

async def copilot_device_code_login(
    *,
    host: str = "github.com",
    timeout_seconds: float = 300,
) -> Optional[str]:
    """Run the GitHub OAuth device code flow for Copilot.

    Prints instructions for the user, polls for completion, and returns
    the OAuth access token on success, or None on failure/cancellation.

    This replicates the flow used by opencode and the Copilot CLI.
    """
    domain = host.rstrip("/")
    device_code_url = f"https://{domain}/login/device/code"
    access_token_url = f"https://{domain}/login/oauth/access_token"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "HermesAgent/1.0",
    }

    try:
        async with (await _create_httpx_client(timeout=15)) as client:
            response = await client.post(
                device_code_url,
                data={"client_id": COPILOT_OAUTH_CLIENT_ID, "scope": "read:user"},
                headers=headers,
            )
            response.raise_for_status()
            device_data = response.json()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("Failed to initiate device authorization: %s", exc)
        print(f"  ✗ Failed to start device authorization: {exc}")
        return None

    verification_uri = device_data.get(
        "verification_uri", "https://github.com/login/device"
    )
    user_code = device_data.get("user_code", "")
    device_code = device_data.get("device_code", "")
    interval = max(device_data.get("interval", _DEVICE_CODE_POLL_INTERVAL), 1)

    if not device_code or not user_code:
        print("  ✗ GitHub did not return a device code.")
        return None

    print()
    print(f"  Open this URL in your browser: {verification_uri}")
    print(f"  Enter this code: {user_code}")
    print()
    print("  Waiting for authorization...", end="", flush=True)

    deadline = time.monotonic() + timeout_seconds
    async with (await _create_httpx_client(timeout=10)) as client:
        while time.monotonic() < deadline:
            await asyncio.sleep(interval + _DEVICE_CODE_POLL_SAFETY_MARGIN)
            try:
                response = await client.post(
                    access_token_url,
                    data={
                        "client_id": COPILOT_OAUTH_CLIENT_ID,
                        "device_code": device_code,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    },
                    headers=headers,
                )
                response.raise_for_status()
                result = response.json()
            except asyncio.CancelledError:
                raise
            except Exception:
                print(".", end="", flush=True)
                continue

            if result.get("access_token"):
                print(" ✓")
                return result["access_token"]

            error = result.get("error", "")
            if error == "authorization_pending":
                print(".", end="", flush=True)
                continue
            if error == "slow_down":
                server_interval = result.get("interval")
                if isinstance(server_interval, (int, float)) and server_interval > 0:
                    interval = int(server_interval)
                else:
                    interval += 5
                print(".", end="", flush=True)
                continue
            if error == "expired_token":
                print()
                print("  ✗ Device code expired. Please try again.")
                return None
            if error == "access_denied":
                print()
                print("  ✗ Authorization was denied.")
                return None
            if error:
                print()
                print(f"  ✗ Authorization failed: {error}")
                return None

    print()
    print("  ✗ Timed out waiting for authorization.")
    return None


# ─── Copilot Token Exchange ────────────────────────────────────────────────

_COPILOT_NO_LOOP = object()
_CopilotScope = tuple[asyncio.AbstractEventLoop | object, str]
_copilot_scope_context: contextvars.ContextVar[tuple[str, str] | None] = (
    contextvars.ContextVar("copilot_auth_profile_scope", default=None)
)
_copilot_scope_aliases: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, str]
] = weakref.WeakKeyDictionary()
_copilot_scope_guard = threading.RLock()


def _lexical_copilot_profile() -> str:
    """Return the active profile marker without filesystem access."""
    from hermes_constants import get_hermes_home

    return os.path.normcase(os.fspath(get_hermes_home()))


def _current_copilot_scope() -> _CopilotScope:
    """Return the activated profile or a lexical staging scope."""
    lexical = _lexical_copilot_profile()
    try:
        loop: asyncio.AbstractEventLoop | object = asyncio.get_running_loop()
    except RuntimeError:
        return _COPILOT_NO_LOOP, lexical
    active = _copilot_scope_context.get()
    if active is not None and active[0] == lexical:
        return loop, active[1]
    with _copilot_scope_guard:
        aliases = _copilot_scope_aliases.get(loop)
        canonical = aliases.get(lexical, lexical) if aliases is not None else lexical
    return loop, canonical


class _ScopedCopilotCache(MutableMapping):
    """Dict-compatible active-loop/profile cache for retained private hooks."""

    def __init__(self) -> None:
        self._loop_values: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, dict[str, dict]
        ] = weakref.WeakKeyDictionary()
        self._staged_values: dict[str, dict] = {}

    def _scoped(self, scope: _CopilotScope) -> dict:
        loop, profile = scope
        with _copilot_scope_guard:
            if loop is _COPILOT_NO_LOOP:
                return self._staged_values.setdefault(profile, {})
            assert isinstance(loop, asyncio.AbstractEventLoop)
            return self._loop_values.setdefault(loop, {}).setdefault(profile, {})

    def _active(self) -> dict:
        return self._scoped(_current_copilot_scope())

    def migrate(self, source: _CopilotScope, target: _CopilotScope) -> None:
        if source == target:
            return
        with _copilot_scope_guard:
            source_values = self._scoped(source)
            target_values = self._scoped(target)
            if source_values is target_values:
                return
            target_values.update(source_values)
            source_values.clear()

    def __getitem__(self, key):
        return self._active()[key]

    def __setitem__(self, key, value) -> None:
        self._active()[key] = value

    def __delitem__(self, key) -> None:
        del self._active()[key]

    def __iter__(self) -> Iterator:
        return iter(tuple(self._active()))

    def __len__(self) -> int:
        return len(self._active())

    def clear(self) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            with _copilot_scope_guard:
                self._loop_values.clear()
                self._staged_values.clear()
            return
        self._active().clear()


class _CopilotAsyncLocks:
    """Weak loop-local locks for exchange and disk critical sections."""

    def __init__(self) -> None:
        self._locks: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop,
            dict[Hashable, weakref.ReferenceType[asyncio.Lock]],
        ] = weakref.WeakKeyDictionary()

    def for_key(self, key: Hashable) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        with _copilot_scope_guard:
            locks = self._locks.setdefault(loop, {})
            for stale_key, stale_ref in tuple(locks.items()):
                if stale_ref() is None:
                    locks.pop(stale_key, None)
            lock_ref = locks.get(key)
            lock = lock_ref() if lock_ref is not None else None
            if lock is None:
                lock = asyncio.Lock()
                locks[key] = weakref.ref(lock)
            return lock


async def _activate_copilot_scope() -> _CopilotScope:
    """Resolve the active event loop and canonical Hermes profile."""
    loop = asyncio.get_running_loop()
    lexical = _lexical_copilot_profile()
    active = _copilot_scope_context.get()
    if active is not None and active[0] == lexical:
        canonical = active[1]
    else:
        with _copilot_scope_guard:
            aliases = _copilot_scope_aliases.get(loop)
            canonical = aliases.get(lexical) if aliases is not None else None
        if canonical is None:
            expanduser = aiofiles.os.wrap(os.path.expanduser)
            expanded = str(await expanduser(lexical))
            is_absolute = (
                expanded.startswith(("/", "\\\\"))
                or (
                    len(expanded) >= 3
                    and expanded[1] == ":"
                    and expanded[2] in "/\\"
                )
            )
            if not is_absolute:
                expanded = str(await aiofiles.os.getcwd()) + os.sep + expanded
            realpath = aiofiles.os.wrap(os.path.realpath)
            canonical = os.path.normcase(str(await realpath(expanded)))
        with _copilot_scope_guard:
            _copilot_scope_aliases.setdefault(loop, {})[lexical] = canonical
    scope = (loop, canonical)
    _copilot_scope_context.set((lexical, canonical))
    for source in ((loop, lexical), (_COPILOT_NO_LOOP, lexical)):
        for cache in (_jwt_cache, _exchange_failure_cache):
            migrate = getattr(cache, "migrate", None)
            if callable(migrate):
                migrate(source, scope)
    return scope


_exchange_locks = _CopilotAsyncLocks()
_jwt_disk_locks = _CopilotAsyncLocks()

# Module-level cache for exchanged Copilot API tokens.
# Maps raw_token_fingerprint -> (api_token, expires_at_epoch, base_url).
_jwt_cache: MutableMapping[str, tuple[str, float, Optional[str]]] = (
    _ScopedCopilotCache()
)
_JWT_REFRESH_MARGIN_SECONDS = 120  # refresh 2 min before expiry

# Token exchange endpoint and headers (matching VS Code / Copilot CLI)
_TOKEN_EXCHANGE_URL = "https://api.github.com/copilot_internal/v2/token"
_EDITOR_VERSION = "vscode/1.104.1"
_EXCHANGE_USER_AGENT = "GitHubCopilotChat/0.26.7"

# Transient-failure hardening for the token exchange. Gateway startup often
# races network readiness (launchd relaunch, DHCP/VPN settling); a single-shot
# exchange that fails there silently degrades to the RAW GitHub token, which the
# Copilot server routes to the "copilot-language-server" integrator whose model
# allowlist omits enterprise-only models (e.g. claude-opus-4.8) → HTTP 400 on
# every turn until the next restart. Retry a few times, and persist the last
# good exchanged JWT to disk so a restart during a blip reuses the still-valid
# ~30-min token instead of degrading.
_EXCHANGE_MAX_ATTEMPTS = 3
_EXCHANGE_BACKOFF_BASE_SECONDS = 1.5
_JWT_DISK_FILENAME = ".copilot_jwt.json"
_JWT_DISK_MAX_BYTES = 1_048_576

# Negative cache for failed exchanges. Without it, every load_pool("copilot")
# call re-runs the full exchange. Maps raw-token fingerprint to the epoch until
# which exchange attempts are skipped.
_exchange_failure_cache: MutableMapping[str, float] = _ScopedCopilotCache()
_EXCHANGE_FAILURE_TTL_TRANSIENT_SECONDS = 60.0
_EXCHANGE_FAILURE_TTL_PERMANENT_SECONDS = 1800.0
_EXCHANGE_PERMANENT_HTTP_STATUSES = frozenset({401, 403, 404})


def _token_fingerprint(raw_token: str) -> str:
    """Short fingerprint of a raw token for cache keying (avoids storing full token)."""
    import hashlib
    return hashlib.sha256(raw_token.encode()).hexdigest()[:16]


async def _read_jwt_store(path: Path) -> Optional[dict]:
    """Bounded read of the on-disk JWT store, or None if unusable."""
    try:
        stat = await aiofiles.os.stat(path)
        if stat.st_size > _JWT_DISK_MAX_BYTES:
            logger.debug(
                "Persisted Copilot JWT store exceeds %d bytes; ignoring",
                _JWT_DISK_MAX_BYTES,
            )
            return None
        async with aiofiles.open(path, encoding="utf-8") as handle:
            loaded = json.loads(await handle.read())
        return loaded if isinstance(loaded, dict) else None
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.debug("Failed to read persisted Copilot JWT store: %s", exc)
        return None


async def _jwt_disk_lock(path: Path) -> asyncio.Lock:
    realpath = aiofiles.os.wrap(os.path.realpath)
    key = os.path.normcase(str(await realpath(path)))
    return _jwt_disk_locks.for_key(key)


async def _remove_jwt_temp(path: Path) -> None:
    try:
        await aiofiles.os.remove(path)
    except FileNotFoundError:
        pass


async def _cleanup_jwt_temp_after_cancellation(
    path: Path,
    cancellation: asyncio.CancelledError,
) -> None:
    """Remove a secret-bearing temp file before propagating cancellation."""
    cleanup_task = asyncio.create_task(_remove_jwt_temp(path))
    while True:
        try:
            await asyncio.shield(cleanup_task)
            break
        except asyncio.CancelledError:  # noqa: ASYNC103 - original re-raised below
            if cleanup_task.cancelled():
                break  # noqa: ASYNC104 - original re-raised below
        except Exception as exc:
            logger.debug("Failed to clean Copilot JWT temp file: %s", exc)
            break
    raise cancellation


async def _write_jwt_store(path: Path, store: dict) -> None:
    """Atomically replace one bounded JWT store with private permissions."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        async with aiofiles.open(temporary, "w", encoding="utf-8") as handle:
            # Tighten a reused/crash-left temp file before writing secrets.
            try:
                await aiofiles.os.wrap(os.chmod)(temporary, 0o600)
            except Exception:
                pass
            await handle.write(json.dumps(store))
        await aiofiles.os.replace(temporary, path)
        try:
            await aiofiles.os.wrap(os.chmod)(path, 0o600)
        except Exception:
            pass
    except asyncio.CancelledError as cancellation:  # noqa: ASYNC103
        await _cleanup_jwt_temp_after_cancellation(temporary, cancellation)
    except Exception:
        await _remove_jwt_temp(temporary)  # noqa: ASYNC120 - then re-raise
        raise


async def evict_cached_exchanged_token(raw_token: str) -> None:
    """Drop any cached exchanged JWT for ``raw_token`` in both cache tiers."""
    if not raw_token:
        return
    scope = await _activate_copilot_scope()
    fp = _token_fingerprint(raw_token)
    async with _exchange_locks.for_key((scope[1], fp)):
        _jwt_cache.pop(fp, None)
        _exchange_failure_cache.pop(fp, None)
        path = _jwt_disk_path()
        if not path:
            return
        async with await _jwt_disk_lock(path):
            if not await aiofiles.os.path.exists(path):
                return
            try:
                store = await _read_jwt_store(path)
                if store is not None and fp in store:
                    del store[fp]
                    await _write_jwt_store(path, store)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Failed to evict cached Copilot JWT: %s", exc)


def _jwt_disk_path() -> Optional[Path]:
    """Path to the on-disk exchanged-JWT cache (profile-aware), or None."""
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home()) / _JWT_DISK_FILENAME
    except Exception:
        return None


async def _load_jwt_from_disk(
    fp: str,
) -> Optional[tuple[str, float, Optional[str]]]:
    """Load a persisted exchanged JWT for ``fp``."""
    path = _jwt_disk_path()
    if not path or not await aiofiles.os.path.exists(path):
        return None
    try:
        store = await _read_jwt_store(path)
        entry = store.get(fp) if store is not None else None
        if not isinstance(entry, dict):
            return None
        api_token = entry.get("api_token", "")
        expires_at = float(entry.get("expires_at", 0) or 0)
        base_url = entry.get("base_url")
        if api_token and expires_at:
            return api_token, expires_at, base_url
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.debug("Failed to load persisted Copilot JWT: %s", exc)
    return None


async def _save_jwt_to_disk(
    fp: str,
    api_token: str,
    expires_at: float,
    base_url: Optional[str],
) -> None:
    """Persist an exchanged JWT (0o600), pruning expired entries."""
    await _activate_copilot_scope()
    path = _jwt_disk_path()
    if not path:
        return
    async with await _jwt_disk_lock(path):
        try:
            store: dict = {}
            if await aiofiles.os.path.exists(path):
                store = await _read_jwt_store(path) or {}
            now = time.time()
            store = {
                key: value
                for key, value in store.items()
                if isinstance(value, dict)
                and float(value.get("expires_at", 0) or 0) > now
            }
            store[fp] = {
                "api_token": api_token,
                "expires_at": expires_at,
                "base_url": base_url,
            }
            await _write_jwt_store(path, store)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Failed to persist Copilot JWT: %s", exc)


async def exchange_copilot_token(
    raw_token: str,
    *,
    timeout: float = 10.0,
) -> tuple[str, float, Optional[str]]:
    """Exchange a raw GitHub token for a short-lived Copilot API token.

    Calls ``GET https://api.github.com/copilot_internal/v2/token`` with
    the raw GitHub token and returns ``(api_token, expires_at, base_url)``.

    The returned token is a semicolon-separated string (not a standard JWT)
    used as ``Authorization: Bearer <token>`` for Copilot API requests.
    ``base_url`` is the account-specific API host: the authoritative
    ``endpoints.api`` advertised by the exchange (enterprise/proxied
    accounts), falling back to a host derived from the token's ``proxy-ep``
    field. Individual accounts have neither, so ``base_url`` is None.

    Results are cached in-process and reused until close to expiry.
    Raises ``ValueError`` on failure.
    """
    scope = await _activate_copilot_scope()
    fp = _token_fingerprint(raw_token)
    async with _exchange_locks.for_key((scope[1], fp)):
        return await _exchange_copilot_token_locked(
            raw_token,
            timeout=timeout,
            fp=fp,
        )


async def _exchange_copilot_token_locked(
    raw_token: str,
    *,
    timeout: float,
    fp: str,
) -> tuple[str, float, Optional[str]]:
    """Exchange after the active profile's fingerprint lock is held."""

    # Check in-process cache first
    cached = _jwt_cache.get(fp)
    if cached:
        api_token, expires_at, base_url = cached
        if time.time() < expires_at - _JWT_REFRESH_MARGIN_SECONDS:
            return api_token, expires_at, base_url

    disk_cached = await _load_jwt_from_disk(fp)
    if disk_cached:
        api_token, expires_at, base_url = disk_cached
        if time.time() < expires_at - _JWT_REFRESH_MARGIN_SECONDS:
            _jwt_cache[fp] = (api_token, expires_at, base_url)
            return api_token, expires_at, base_url

    fail_until = _exchange_failure_cache.get(fp, 0.0)
    if time.time() < fail_until:
        raise ValueError(
            "Copilot token exchange recently failed; skipping re-attempt "
            f"for another {int(fail_until - time.time())}s"
        )

    headers = {
        "Authorization": f"token {raw_token}",
        "User-Agent": _EXCHANGE_USER_AGENT,
        "Accept": "application/json",
        "Editor-Version": _EDITOR_VERSION,
    }
    data = None
    last_exc: Optional[Exception] = None
    permanent_failure = False
    async with (await _create_httpx_client(timeout=timeout)) as client:
        for attempt in range(_EXCHANGE_MAX_ATTEMPTS):
            try:
                response = await client.get(_TOKEN_EXCHANGE_URL, headers=headers)
                response.raise_for_status()
                data = response.json()
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — retry all, re-raise below
                last_exc = exc
                response = getattr(exc, "response", None)
                status = getattr(response, "status_code", None)
                if status is None:
                    status = getattr(exc, "code", None) or getattr(exc, "status", None)
                if status in _EXCHANGE_PERMANENT_HTTP_STATUSES:
                    permanent_failure = True
                    logger.debug(
                        "Copilot token exchange rejected (HTTP %s); not retrying",
                        status,
                    )
                    break
                if attempt < _EXCHANGE_MAX_ATTEMPTS - 1:
                    sleep_seconds = _EXCHANGE_BACKOFF_BASE_SECONDS * (attempt + 1)
                    logger.debug(
                        "Copilot token exchange attempt %d/%d failed (%s); "
                        "retrying in %.1fs",
                        attempt + 1,
                        _EXCHANGE_MAX_ATTEMPTS,
                        exc,
                        sleep_seconds,
                    )
                    await asyncio.sleep(sleep_seconds)

    if data is None:
        ttl = (
            _EXCHANGE_FAILURE_TTL_PERMANENT_SECONDS
            if permanent_failure
            else _EXCHANGE_FAILURE_TTL_TRANSIENT_SECONDS
        )
        _exchange_failure_cache[fp] = time.time() + ttl
        raise ValueError(
            f"Copilot token exchange failed after {_EXCHANGE_MAX_ATTEMPTS} "
            f"attempts: {last_exc}"
        ) from last_exc
    _exchange_failure_cache.pop(fp, None)

    api_token = data.get("token", "")
    expires_at = data.get("expires_at", 0)
    if not api_token:
        raise ValueError("Copilot token exchange returned empty token")

    # Convert expires_at to float if needed
    expires_at = float(expires_at) if expires_at else time.time() + 1800

    # Resolve the account-specific API base URL. GitHub advertises the
    # authoritative endpoint under ``endpoints.api`` in the exchange response
    # (it differs for Copilot Enterprise / proxied accounts). When the
    # response omits it, fall back to deriving the host from the ``proxy-ep``
    # field embedded in the exchanged token. Individual accounts have neither,
    # so ``base_url`` stays None and callers use the registry default.
    base_url: Optional[str] = None
    endpoints = data.get("endpoints")
    if isinstance(endpoints, dict):
        api_endpoint = str(endpoints.get("api") or "").strip().rstrip("/")
        if api_endpoint:
            base_url = api_endpoint
    if not base_url:
        base_url = _derive_base_url_from_proxy_ep(api_token)

    await _save_jwt_to_disk(fp, api_token, expires_at, base_url)
    _jwt_cache[fp] = (api_token, expires_at, base_url)
    logger.debug(
        "Copilot token exchanged, expires_at=%s, base_url=%s",
        expires_at,
        base_url,
    )
    return api_token, expires_at, base_url


def _derive_base_url_from_proxy_ep(token: str) -> Optional[str]:
    """Derive the Copilot API base URL from a proxy-ep field in the token.

    The exchanged Copilot token is a semicolon-separated string like
    ``tid=xxx;exp=xxx;proxy-ep=proxy.enterprise.githubcopilot.com;...``.
    This extracts ``proxy-ep`` and converts it to an API base URL by
    replacing the leading ``proxy.`` with ``api.``.

    Returns ``https://{api_hostname}`` or None if proxy-ep is absent.
    """
    import re
    m = re.search(r'(?:^|;)\s*proxy-ep=([^;\s]+)', token)
    if not m:
        return None

    proxy_ep = m.group(1)
    # Strip scheme if present
    for prefix in ("https://", "http://"):
        if proxy_ep.startswith(prefix):
            proxy_ep = proxy_ep[len(prefix):]
            break
    proxy_ep = proxy_ep.rstrip("/")

    # Replace leading "proxy." with "api."
    if proxy_ep.startswith("proxy."):
        api_host = "api." + proxy_ep[len("proxy."):]
    else:
        api_host = proxy_ep

    return f"https://{api_host}"


async def get_copilot_api_token(raw_token: str) -> tuple[str, Optional[str]]:
    """Exchange a raw GitHub token for a Copilot API token, with fallback.

    Convenience wrapper: returns ``(api_token, base_url)`` on success, or
    ``(raw_token, None)`` if the exchange fails (e.g. network error, unsupported
    account type). This preserves existing behaviour for accounts that don't
    need exchange while enabling access to internal-only models for those that do.

    ``base_url`` is the account-specific API endpoint advertised by the
    exchange (``endpoints.api``, with a ``proxy-ep`` fallback), or None for
    individual accounts.
    """
    if not raw_token:
        return raw_token, None
    try:
        api_token, _, base_url = await exchange_copilot_token(raw_token)
        return api_token, base_url
    except Exception as exc:
        logger.debug("Copilot token exchange failed, using raw token: %s", exc)
        return raw_token, None


# ─── Copilot API Headers ───────────────────────────────────────────────────

def copilot_request_headers(
    *,
    is_agent_turn: bool = True,
    is_vision: bool = False,
) -> dict[str, str]:
    """Build the standard headers for Copilot API requests.

    Replicates the header set used by opencode and the Copilot CLI.
    """
    headers: dict[str, str] = {
        "Editor-Version": "vscode/1.104.1",
        "User-Agent": "HermesAgent/1.0",
        "Copilot-Integration-Id": "vscode-chat",
        "Openai-Intent": "conversation-edits",
        "x-initiator": "agent" if is_agent_turn else "user",
    }
    if is_vision:
        headers["Copilot-Vision-Request"] = "true"

    return headers
