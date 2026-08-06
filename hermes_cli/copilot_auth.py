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
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import aiofiles
import aiofiles.os
import httpx

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
        val = os.getenv(env_var, "").strip()
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
    hostname = os.getenv("COPILOT_GH_HOST", "").strip()

    # Build a clean env so gh doesn't short-circuit on GITHUB_TOKEN / GH_TOKEN
    clean_env = {k: v for k, v in os.environ.items()
                 if k not in {"GITHUB_TOKEN", "GH_TOKEN"}}

    for gh_path in await _gh_cli_candidates():
        cmd = [gh_path, "auth", "token"]
        if hostname:
            cmd += ["--hostname", hostname]
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=clean_env,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=5)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logger.debug("gh CLI token lookup timed out (%s)", gh_path)
            continue
        except asyncio.CancelledError:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
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
        async with httpx.AsyncClient(timeout=15) as client:
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
    async with httpx.AsyncClient(timeout=10) as client:
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

# Module-level cache for exchanged Copilot API tokens.
# Maps raw_token_fingerprint -> (api_token, expires_at_epoch, base_url).
_jwt_cache: dict[str, tuple[str, float, Optional[str]]] = {}
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
_exchange_failure_cache: dict[str, float] = {}
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


async def evict_cached_exchanged_token(raw_token: str) -> None:
    """Drop any cached exchanged JWT for ``raw_token`` in both cache tiers."""
    if not raw_token:
        return
    fp = _token_fingerprint(raw_token)
    _jwt_cache.pop(fp, None)
    _exchange_failure_cache.pop(fp, None)
    path = _jwt_disk_path()
    if not path or not await aiofiles.os.path.exists(path):
        return
    try:
        store = await _read_jwt_store(path)
        if store is not None and fp in store:
            del store[fp]
            tmp = path.with_suffix(path.suffix + ".tmp")
            async with aiofiles.open(tmp, "w", encoding="utf-8") as handle:
                await handle.write(json.dumps(store))
            try:
                await aiofiles.os.wrap(os.chmod)(tmp, 0o600)
            except Exception:
                pass
            await aiofiles.os.replace(tmp, path)
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
    path = _jwt_disk_path()
    if not path:
        return
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
        tmp = path.with_suffix(path.suffix + ".tmp")
        async with aiofiles.open(tmp, "w", encoding="utf-8") as handle:
            await handle.write(json.dumps(store))
        try:
            await aiofiles.os.wrap(os.chmod)(tmp, 0o600)
        except Exception:
            pass
        await aiofiles.os.replace(tmp, path)
        try:
            await aiofiles.os.wrap(os.chmod)(path, 0o600)
        except Exception:
            pass
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
    fp = _token_fingerprint(raw_token)

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
    async with httpx.AsyncClient(timeout=timeout) as client:
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

    _jwt_cache[fp] = (api_token, expires_at, base_url)
    await _save_jwt_to_disk(fp, api_token, expires_at, base_url)
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
