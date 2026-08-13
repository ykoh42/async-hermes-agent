"""OAuth credential storage and refresh for the Honcho memory provider.

An access token authenticates exactly like a scoped API key, so it is stored
as the host's ``apiKey``; this module exchanges the refresh token before
expiry to keep it live.

Refresh tokens rotate with single-use reuse detection: a replayed stale token
revokes the whole grant. So every refresh must persist the rotated token
atomically and be serialized — and a failed refresh never raises into the
agent (stale token stays; the fail-open path absorbs the eventual 401).
"""

from __future__ import annotations

import json
import logging
import os
import asyncio
import threading
import time
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Callable

import aiofiles
import aiofiles.os

logger = logging.getLogger(__name__)

ACCESS_TOKEN_PREFIX = "hch-at-"
REFRESH_TOKEN_PREFIX = "hch-rt-"

# Refresh this many seconds before the access token actually expires, so an
# in-flight request never races the expiry boundary.
_REFRESH_SKEW_SECONDS = 120

# Default HTTP timeout for the token exchange. Kept short — the refresh happens
# on the path to a memory call, and a stalled auth server must not hang it.
_REFRESH_TIMEOUT_SECONDS = 15.0

# Serializes refresh within one event loop for each physical config. Re-checked
# under the lock (double-checked) so racing callers don't replay a rotated
# refresh token and trip reuse detection. Cross-process serialization remains
# owned by ``_config_refresh_lock`` below.
_refresh_locks: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    dict[str, weakref.ReferenceType[asyncio.Lock]],
] = weakref.WeakKeyDictionary()
_refresh_locks_guard = threading.RLock()
_realpath = aiofiles.os.wrap(os.path.realpath)


async def _refresh_lock(path: Path) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    key = os.path.normcase(str(await _realpath(path)))
    with _refresh_locks_guard:
        for candidate in tuple(_refresh_locks):
            if candidate.is_closed():
                _refresh_locks.pop(candidate, None)
        locks = _refresh_locks.setdefault(loop, {})
        lock_ref = locks.get(key)
        lock = lock_ref() if lock_ref is not None else None
        if lock is None:
            lock = asyncio.Lock()
            locks[key] = weakref.ref(lock)
        return lock


@asynccontextmanager
async def _config_refresh_lock(path: Path):
    """Machine-wide advisory lock around read-refresh-persist.

    The in-process ``_refresh_lock`` can't stop a second process (a sibling
    Hermes profile or the desktop app sharing this honcho.json) from replaying
    the single-use refresh token and tripping reuse-detection — which revokes
    the whole grant. An OS file lock on ``<config>.lock`` serializes rotation
    across processes; best-effort, so a platform without flock degrades to
    in-process serialization only.
    """
    lock_path = Path(f"{path}.lock")
    handle = None
    try:
        await aiofiles.os.makedirs(lock_path.parent, exist_ok=True)
        handle = await aiofiles.open(lock_path, "a+b")
        if os.name == "nt":
            import msvcrt

            await handle.seek(0)
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    await asyncio.sleep(0.05)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    await asyncio.sleep(0.05)
    except asyncio.CancelledError:
        if handle is not None:
            await handle.close()
        raise
    except Exception:
        logger.debug("Honcho OAuth cross-process lock unavailable; in-process only", exc_info=True)
        if handle is not None:
            await handle.close()
            handle = None
    try:
        yield
    finally:
        if handle is not None:
            try:
                if os.name == "nt":
                    import msvcrt

                    await handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            await handle.close()

# In-memory expiry cache keyed by (config path, host) → (expires_at, access).
# Lets the hot path (every memory access calls this) skip the honcho.json read
# while the token is comfortably live; disk is only touched near expiry, on a
# cache miss, or when an explicit ``raw`` is supplied. Single-key dict ops are
# atomic under the GIL, so no separate lock is needed. An access token stays
# valid until its own expiry regardless of out-of-band rotation, so a stale
# cache entry can't break auth — it just defers picking up external changes
# until the token nears expiry and disk is read again.
_expiry_cache: dict[tuple[str, str], tuple[float, str]] = {}


def is_oauth_access_token(value: str | None) -> bool:
    """True when ``value`` is an OAuth access token (vs a static API key)."""
    return bool(value) and value.startswith(ACCESS_TOKEN_PREFIX)


@dataclass
class OAuthCredential:
    """An OAuth grant as stored in a honcho.json host block.

    ``access_token`` mirrors the host's ``apiKey``; the remaining fields live in
    the host's ``oauth`` sub-block. ``expires_at`` is absolute epoch seconds.
    """

    access_token: str
    refresh_token: str
    expires_at: float
    client_id: str
    token_endpoint: str
    scope: str = "write"
    token_type: str = "Bearer"
    # Transient consent peer name — set only on a fresh grant, never persisted.
    consent_peer_name: str | None = None

    @classmethod
    def from_host_block(cls, block: dict[str, Any]) -> OAuthCredential | None:
        """Build a credential from a honcho.json host block, or None if incomplete."""
        oauth = block.get("oauth")
        access = block.get("apiKey")
        if not isinstance(oauth, dict) or not is_oauth_access_token(access):
            return None
        refresh = oauth.get("refreshToken")
        endpoint = oauth.get("tokenEndpoint")
        client_id = oauth.get("clientId")
        if not (refresh and endpoint and client_id):
            return None
        try:
            expires_at = float(oauth.get("expiresAt", 0))
        except (TypeError, ValueError):
            expires_at = 0.0
        return cls(
            access_token=access,
            refresh_token=str(refresh),
            expires_at=expires_at,
            client_id=str(client_id),
            token_endpoint=str(endpoint),
            scope=str(oauth.get("scope", "write")),
            token_type=str(oauth.get("tokenType", "Bearer")),
        )

    def oauth_block(self) -> dict[str, Any]:
        """The ``oauth`` sub-block to persist (the access token lives in apiKey)."""
        return {
            "refreshToken": self.refresh_token,
            "expiresAt": int(self.expires_at),
            "clientId": self.client_id,
            "tokenEndpoint": self.token_endpoint,
            "scope": self.scope,
            "tokenType": self.token_type,
        }

    def is_expired(self, *, now: float, skew: float = _REFRESH_SKEW_SECONDS) -> bool:
        """True when the access token is within ``skew`` seconds of expiry."""
        return now >= (self.expires_at - skew)


# Indirection so tests can drive the exchange without a live server.
async def _http_post_form(url: str, data: dict[str, str], timeout: float) -> dict[str, Any]:
    """POST form-encoded ``data`` to ``url`` and return the parsed JSON body."""
    import httpx

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, data=data)
    resp.raise_for_status()
    return resp.json()


async def _http_post_form_status(
    url: str, data: dict[str, str], timeout: float
) -> tuple[int, dict[str, Any]]:
    """POST form-encoded ``data``; return ``(status, parsed JSON body)``.

    Unlike ``_http_post_form``, 4xx does not raise — RFC 8628 polling reads the
    OAuth error body off a 400. A non-JSON body parses to ``{}``.
    """
    import httpx

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, data=data)
    try:
        body = resp.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    return resp.status_code, body


async def _http_get_json(url: str, timeout: float) -> dict[str, Any]:
    """GET ``url`` and return the parsed JSON body. Raises on non-2xx/non-JSON."""
    import httpx

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url)
    resp.raise_for_status()
    body = resp.json()
    return body if isinstance(body, dict) else {}


async def _exchange_refresh_token(cred: OAuthCredential, *, now: float) -> OAuthCredential:
    """Run the refresh_token grant and return the rotated credential.

    Raises on any transport/protocol failure; callers fail open.
    """
    body = await _http_post_form(
        cred.token_endpoint,
        {
            "grant_type": "refresh_token",
            "client_id": cred.client_id,
            "refresh_token": cred.refresh_token,
        },
        _REFRESH_TIMEOUT_SECONDS,
    )
    access = body.get("access_token")
    refresh = body.get("refresh_token")
    if not is_oauth_access_token(access) or not refresh:
        raise ValueError("refresh response missing access_token/refresh_token")
    try:
        expires_in = int(body.get("expires_in", 0))
    except (TypeError, ValueError):
        expires_in = 0
    return OAuthCredential(
        access_token=access,
        refresh_token=str(refresh),
        expires_at=now + expires_in,
        client_id=cred.client_id,
        token_endpoint=cred.token_endpoint,
        scope=str(body.get("scope", cred.scope)),
        token_type=str(body.get("token_type", cred.token_type)),
    )


async def _read_config(path: Path) -> dict[str, Any]:
    try:
        async with aiofiles.open(path, encoding="utf-8") as handle:
            return json.loads(await handle.read())
    except (OSError, json.JSONDecodeError):
        return {}


async def _atomic_write_config(path: Path, raw: dict[str, Any]) -> None:
    """Write ``raw`` to ``path`` atomically, preserving 0600 on the new file."""
    await aiofiles.os.makedirs(path.parent, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    text = json.dumps(raw, indent=2) + "\n"
    replaced = False

    def _owner_only_opener(filename: str, flags: int) -> int:
        return os.open(filename, flags, 0o600)

    try:
        async with aiofiles.open(
            tmp,
            "w",
            encoding="utf-8",
            opener=_owner_only_opener,
        ) as handle:
            await handle.write(text)
            await handle.flush()
            await aiofiles.os.wrap(os.fsync)(handle.fileno())
        await aiofiles.os.replace(tmp, path)
        replaced = True
    finally:
        if not replaced:
            try:
                await aiofiles.os.remove(tmp)
            except FileNotFoundError:
                pass


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base`` (overlay wins on scalars/lists)."""
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


async def _persist_credential(path: Path, host: str, cred: OAuthCredential) -> None:
    """Persist ``cred`` into ``host``'s block (apiKey + oauth), leaving all else intact."""
    raw = await _read_config(path)
    hosts = raw.setdefault("hosts", {})
    block = hosts.setdefault(host, {})
    block["apiKey"] = cred.access_token
    block["oauth"] = cred.oauth_block()
    await _atomic_write_config(path, raw)
    _expiry_cache[(str(path), host)] = (cred.expires_at, cred.access_token)


async def ensure_fresh_token(
    path: Path,
    host: str,
    raw: dict[str, Any] | None = None,
    *,
    now: float | None = None,
) -> tuple[str | None, bool]:
    """Return ``(access_token, refreshed)`` for ``host``, refreshing if near expiry.

    Returns ``(None, False)`` when the host has no OAuth credential (e.g. a plain
    API key) so callers leave the existing token untouched. Refresh failures are
    swallowed: the current (possibly stale) token is returned with
    ``refreshed=False`` and the fail-open path handles any resulting 401.
    """
    now = time.time() if now is None else now
    key = (str(path), host)

    # Hot path: trust the cached expiry while the token is well clear of the
    # skew window — no disk read. Bypassed when an explicit ``raw`` is supplied.
    if raw is None:
        cached = _expiry_cache.get(key)
        if cached is not None and now < cached[0] - _REFRESH_SKEW_SECONDS:
            return cached[1], False

    source = raw if raw is not None else await _read_config(path)
    block = (source.get("hosts") or {}).get(host) or {}
    cred = OAuthCredential.from_host_block(block)
    if cred is None:
        _expiry_cache.pop(key, None)
        return None, False

    _expiry_cache[key] = (cred.expires_at, cred.access_token)
    if not cred.is_expired(now=now):
        return cred.access_token, False

    async with await _refresh_lock(path), _config_refresh_lock(path):
        # Re-read under both locks: another thread or process may have just
        # rotated the token — adopt theirs instead of replaying the old one.
        fresh_block = ((await _read_config(path)).get("hosts") or {}).get(host) or {}
        current = OAuthCredential.from_host_block(fresh_block) or cred
        if not current.is_expired(now=now):
            return current.access_token, current.access_token != cred.access_token
        try:
            rotated = await _exchange_refresh_token(current, now=now)
        except Exception as exc:
            logger.warning("Honcho OAuth refresh failed for host %s: %s", host, exc)
            return current.access_token, False
        await _persist_credential(path, host, rotated)
        logger.info("Honcho OAuth token refreshed for host %s", host)
        return rotated.access_token, True


async def install_grant(
    path: Path,
    host: str,
    grant: dict[str, Any],
    *,
    client_id: str,
    token_endpoint: str,
    apply_config: bool = True,
    now: float | None = None,
) -> OAuthCredential:
    """Apply a fresh OAuth grant to ``path`` for ``host``.

    Deep-merges the grant's ``config`` (the manifest default_config) into the
    file root — preserving other hosts and root keys — then writes the host's
    ``apiKey`` and ``oauth`` block. ``grant`` is an OAuthTokenResponse dict
    (access_token, refresh_token, expires_in, scope, config).
    ``apply_config=False`` skips the config merge and stores tokens only.
    """
    now = time.time() if now is None else now
    access = grant.get("access_token")
    refresh = grant.get("refresh_token")
    if not is_oauth_access_token(access) or not refresh:
        raise ValueError("grant missing access_token/refresh_token")
    try:
        expires_in = int(grant.get("expires_in", 0))
    except (TypeError, ValueError):
        expires_in = 0

    cred = OAuthCredential(
        access_token=access,
        refresh_token=str(refresh),
        expires_at=now + expires_in,
        client_id=client_id,
        token_endpoint=token_endpoint,
        scope=str(grant.get("scope", "write")),
        token_type=str(grant.get("token_type", "Bearer")),
    )

    raw = await _read_config(path)
    granted_config = grant.get("config")
    if isinstance(granted_config, dict):
        cred.consent_peer_name = granted_config.get("peerName")
        if apply_config:
            _deep_merge(raw, granted_config)
    _expiry_cache[(str(path), host)] = (cred.expires_at, cred.access_token)
    hosts = raw.setdefault("hosts", {})
    block = hosts.setdefault(host, {})
    block["apiKey"] = cred.access_token
    block["oauth"] = cred.oauth_block()
    await _atomic_write_config(path, raw)
    return cred


def apply_token_to_client(client: Any, token: str) -> bool:
    """Rotate the live Honcho client's Bearer in place. Returns success.

    The SDK builds its auth header per request from the HTTP client's
    ``api_key``, so mutating it rotates every holder of the singleton without a
    rebuild. Guarded: an SDK shape change degrades to False and the caller can
    fall back to resetting the client.
    """
    http = getattr(client, "_async_http", None)
    if http is None or not hasattr(http, "api_key"):
        return False
    http.api_key = token
    return True
