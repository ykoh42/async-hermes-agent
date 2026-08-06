"""Persistent multi-credential pool for same-provider failover."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
import uuid
import re
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from hermes_constants import OPENROUTER_BASE_URL
from hermes_cli.config import (
    get_env_value_prefer_dotenv,
    load_config_readonly,
)
from agent.credential_persistence import (
    is_borrowed_credential_source,
    sanitize_borrowed_credential_payload,
)
import hermes_cli.auth as auth_mod
from hermes_cli.auth import (
    CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    PROVIDER_REGISTRY,
    _codex_access_token_is_expiring,
    _decode_jwt_claims,
    _load_auth_store,
    _load_provider_state,
    _resolve_kimi_base_url,
    _resolve_zai_base_url,
    _store_provider_state,
    read_credential_pool,
    write_credential_pool,
)

logger = logging.getLogger(__name__)


# --- Status and type constants ---

STATUS_OK = "ok"
STATUS_EXHAUSTED = "exhausted"
# Terminal failure — the credential will never recover on its own.  Used for
# upstream-permanent OAuth states like ``token_invalidated`` / ``token_revoked``
# where retrying after a TTL cooldown is guaranteed to fail.  ``DEAD`` entries
# are excluded from rotation unconditionally and only clear when an explicit
# write-side sync (e.g. ``_save_codex_tokens`` after a fresh device-code
# login) rewrites the tokens.
STATUS_DEAD = "dead"

# OAuth error reasons that indicate the credential is permanently invalid
# server-side and cannot be recovered by retry/refresh.  Sourced from
# OpenAI Codex Responses API, Anthropic, xAI, and Google OAuth spec.
_TERMINAL_AUTH_REASONS = frozenset({
    "token_invalidated",   # OpenAI Codex: "Your authentication token has been invalidated."
    "token_revoked",        # OAuth 2.0 RFC 7009: token explicitly revoked
    "invalid_token",        # RFC 6750: bearer token is malformed/expired/revoked
    "invalid_grant",        # RFC 6749: refresh_token rejected during refresh
    "unauthorized_client",  # RFC 6749: client no longer authorized
    "refresh_token_reused", # Single-use refresh token consumed by another process
})

# How long a DEAD manual credential is preserved before being pruned.
# Manual entries (``manual:*``) are independent credentials with no singleton
# to re-seed from, so pruning them after a quiet window cleans up dead state
# without losing recoverability — the user always has the option to re-add
# via ``hermes auth add``.
#
# Singleton-seeded entries (``device_code``, ``claude_code``)
# are NOT pruned because ``_seed_from_singletons`` would just re-create them
# on the next ``load_pool()`` with the same stale singleton tokens, defeating
# the cleanup.  They remain in the pool marked DEAD until an explicit re-auth
# write-side sync (``_save_codex_tokens`` etc.) clears the status.
DEAD_MANUAL_PRUNE_TTL_SECONDS = 24 * 60 * 60  # 24 hours

AUTH_TYPE_OAUTH = "oauth"
AUTH_TYPE_API_KEY = "api_key"

SOURCE_MANUAL = "manual"
SOURCE_MANUAL_DEVICE_CODE = f"{SOURCE_MANUAL}:device_code"

STRATEGY_FILL_FIRST = "fill_first"
STRATEGY_ROUND_ROBIN = "round_robin"
STRATEGY_RANDOM = "random"
STRATEGY_LEAST_USED = "least_used"
SUPPORTED_POOL_STRATEGIES = {
    STRATEGY_FILL_FIRST,
    STRATEGY_ROUND_ROBIN,
    STRATEGY_RANDOM,
    STRATEGY_LEAST_USED,
}

# Cooldown before retrying an exhausted credential.
# Transient 401 auth failures cool down briefly so single-key setups can recover.
# 429 (rate-limited), 402 (billing/quota), and other failures cool down after 1 hour.
# Provider-supplied reset_at timestamps override these defaults.
EXHAUSTED_TTL_401_SECONDS = 5 * 60           # 5 minutes
EXHAUSTED_TTL_429_SECONDS = 60 * 60          # 1 hour
EXHAUSTED_TTL_DEFAULT_SECONDS = 60 * 60      # 1 hour

# Throttle window for the "no available entries" INFO line. Credential
# selection runs on a hot path (every model call, plus auxiliary tasks like
# compression/moa/titles), so when a pool is empty or fully exhausted the
# un-throttled log fires on *every* selection. On Windows several Hermes
# processes share one rotating log guarded by concurrent-log-handler's
# cross-process lock; that per-selection volume storms the lock
# (``RuntimeError: Cannot acquire lock after 20 attempts``), pegs a core, and
# stalls the asyncio event loop long enough to fail the Desktop backend
# readiness handshake ("Timed out connecting to Hermes backend after
# 15000ms"). Logging the condition at most once per window preserves the
# signal while removing the storm — same class of fix as the warn-once
# dedup in #58265.
NO_AVAILABLE_ENTRIES_LOG_THROTTLE_SECONDS = 60.0

# Pool key prefix for custom OpenAI-compatible endpoints.
# Custom endpoints all share provider='custom' but are keyed by their
# custom_providers name: 'custom:<normalized_name>'.
CUSTOM_POOL_PREFIX = "custom:"


# Fields that are only round-tripped through JSON — never used for logic as attributes.
_EXTRA_KEYS = frozenset({
    "token_type", "scope", "client_id", "portal_base_url", "obtained_at",
    "expires_in", "agent_key_id", "agent_key_expires_in", "agent_key_reused",
    "agent_key_obtained_at", "tls", "secret_source", "secret_fingerprint",
})


def _normalize_pool_auth_type(provider: str, token: Any, auth_type: Any) -> str:
    """Infer pool auth metadata for token formats with one unambiguous meaning."""
    if (
        provider == "anthropic"
        and isinstance(token, str)
        and token.startswith("sk-ant-oat")
    ):
        return AUTH_TYPE_OAUTH
    return str(auth_type or AUTH_TYPE_API_KEY)


@dataclass
class PooledCredential:
    provider: str
    id: str
    label: str
    auth_type: str
    priority: int
    source: str
    access_token: str
    refresh_token: Optional[str] = None
    last_status: Optional[str] = None
    last_status_at: Optional[float] = None
    last_error_code: Optional[int] = None
    last_error_reason: Optional[str] = None
    last_error_message: Optional[str] = None
    last_error_reset_at: Optional[float] = None
    base_url: Optional[str] = None
    expires_at: Optional[str] = None
    expires_at_ms: Optional[int] = None
    last_refresh: Optional[str] = None
    inference_base_url: Optional[str] = None
    agent_key: Optional[str] = None
    agent_key_expires_at: Optional[str] = None
    request_count: int = 0
    extra: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}
        self.auth_type = _normalize_pool_auth_type(
            self.provider,
            self.access_token,
            self.auth_type,
        )

    def __getattr__(self, name: str):
        if name in _EXTRA_KEYS:
            return self.extra.get(name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute {name!r}")

    @classmethod
    def from_dict(cls, provider: str, payload: Dict[str, Any]) -> "PooledCredential":
        field_names = {f.name for f in fields(cls) if f.name != "provider"}
        data = {k: payload.get(k) for k in field_names if k in payload}
        # Rehydrated last_status_at may be an ISO string from to_dict() — normalize to float epoch
        if "last_status_at" in data and isinstance(data["last_status_at"], str):
            data["last_status_at"] = _parse_absolute_timestamp(data["last_status_at"])
        extra = {k: payload[k] for k in _EXTRA_KEYS if k in payload and payload[k] is not None}
        data["extra"] = extra
        data.setdefault("id", uuid.uuid4().hex[:6])
        data.setdefault("label", payload.get("source", provider))
        data.setdefault("auth_type", AUTH_TYPE_API_KEY)
        data.setdefault("priority", 0)
        data.setdefault("source", SOURCE_MANUAL)
        data.setdefault("access_token", "")
        return cls(provider=provider, **data)

    def to_dict(self) -> Dict[str, Any]:
        _ALWAYS_EMIT = {
            "last_status",
            "last_status_at",
            "last_error_code",
            "last_error_reason",
            "last_error_message",
            "last_error_reset_at",
        }
        result: Dict[str, Any] = {}
        for field_def in fields(self):
            if field_def.name in {"provider", "extra"}:
                continue
            value = getattr(self, field_def.name)
            if value is not None or field_def.name in _ALWAYS_EMIT:
                result[field_def.name] = value
        for k, v in self.extra.items():
            if v is not None:
                result[k] = v
        return sanitize_borrowed_credential_payload(result, self.provider)

    @property
    def runtime_api_key(self) -> str:
        if self.provider == "nous":
            # Nous stores the runtime inference credential in agent_key for
            # compatibility. It must be a NAS invoke JWT.
            for token, expires_at in (
                (self.agent_key, self.agent_key_expires_at),
                (self.access_token, self.expires_at),
            ):
                if (
                    isinstance(token, str)
                    and token.strip()
                    and auth_mod._nous_invoke_jwt_is_usable(
                        token,
                        scope=getattr(self, "scope", None),
                        expires_at=expires_at,
                    )
                ):
                    return token.strip()
            return ""
        return str(self.access_token or "")

    @property
    def runtime_base_url(self) -> Optional[str]:
        if self.provider == "nous":
            return self.inference_base_url or self.base_url
        return self.base_url


def label_from_token(token: str, fallback: str) -> str:
    claims = _decode_jwt_claims(token)
    for key in ("email", "preferred_username", "upn"):
        value = claims.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _next_priority(entries: List[PooledCredential]) -> int:
    return max((entry.priority for entry in entries), default=-1) + 1


def _is_manual_source(source: str) -> bool:
    normalized = (source or "").strip().lower()
    return normalized == SOURCE_MANUAL or normalized.startswith(f"{SOURCE_MANUAL}:")


def _exhausted_ttl(error_code: Optional[int]) -> int:
    """Return cooldown seconds based on the HTTP status that caused exhaustion."""
    if error_code == 401:
        return EXHAUSTED_TTL_401_SECONDS
    if error_code == 429:
        return EXHAUSTED_TTL_429_SECONDS
    return EXHAUSTED_TTL_DEFAULT_SECONDS


def _parse_absolute_timestamp(value: Any) -> Optional[float]:
    """Best-effort parse for provider reset timestamps.

    Accepts epoch seconds, epoch milliseconds, and ISO-8601 strings.
    Returns seconds since epoch.
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric <= 0:
            return None
        return numeric / 1000.0 if numeric > 1_000_000_000_000 else numeric
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            numeric = float(raw)
        except ValueError:
            numeric = None
        if numeric is not None:
            return numeric / 1000.0 if numeric > 1_000_000_000_000 else numeric
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _extract_retry_delay_seconds(message: str) -> Optional[float]:
    if not message:
        return None
    delay_match = re.search(r"quotaResetDelay[:\s\"]+(\d+(?:\.\d+)?)(ms|s)", message, re.IGNORECASE)
    if delay_match:
        value = float(delay_match.group(1))
        return value / 1000.0 if delay_match.group(2).lower() == "ms" else value
    sec_match = re.search(r"retry\s+(?:after\s+)?(\d+(?:\.\d+)?)\s*(?:sec|secs|seconds|s\b)", message, re.IGNORECASE)
    if sec_match:
        return float(sec_match.group(1))
    # "Resets in 4hr 5min" format used by OpenCode Go weekly usage limits
    hr_min_match = re.search(r"resets?\s+in\s+(\d+)\s*hr\s+(\d+)\s*min", message, re.IGNORECASE)
    if hr_min_match:
        return int(hr_min_match.group(1)) * 3600 + int(hr_min_match.group(2)) * 60
    hr_only_match = re.search(r"resets?\s+in\s+(\d+)\s*hr\b", message, re.IGNORECASE)
    if hr_only_match:
        return int(hr_only_match.group(1)) * 3600
    min_only_match = re.search(r"resets?\s+in\s+(\d+)\s*min\b", message, re.IGNORECASE)
    if min_only_match:
        return int(min_only_match.group(1)) * 60
    return None


def _normalize_error_context(error_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(error_context, dict):
        return {}
    normalized: Dict[str, Any] = {}
    reason = error_context.get("reason")
    if isinstance(reason, str) and reason.strip():
        normalized["reason"] = reason.strip()
    message = error_context.get("message")
    if isinstance(message, str) and message.strip():
        normalized["message"] = message.strip()
    reset_at = (
        error_context.get("reset_at")
        or error_context.get("resets_at")
        or error_context.get("retry_until")
    )
    parsed_reset_at = _parse_absolute_timestamp(reset_at)
    if parsed_reset_at is None and isinstance(message, str):
        retry_delay_seconds = _extract_retry_delay_seconds(message)
        if retry_delay_seconds is not None:
            parsed_reset_at = time.time() + retry_delay_seconds
    if parsed_reset_at is not None:
        normalized["reset_at"] = parsed_reset_at
    return normalized


def _exhausted_until(entry: PooledCredential) -> Optional[float]:
    if entry.last_status != STATUS_EXHAUSTED:
        return None
    reset_at = _parse_absolute_timestamp(getattr(entry, "last_error_reset_at", None))
    if reset_at is not None:
        return reset_at
    if entry.last_status_at:
        return entry.last_status_at + _exhausted_ttl(entry.last_error_code)
    return None


def _normalize_custom_pool_name(name: str) -> str:
    """Normalize a custom provider name for use as a pool key suffix."""
    return name.strip().lower().replace(" ", "-")


def _iter_custom_providers(config: Optional[dict] = None):
    """Yield (normalized_name, entry_dict) for each valid custom_providers entry."""
    if config is None:
        return
    custom_providers = config.get("custom_providers")
    if not isinstance(custom_providers, list):
        # Fall back to the v12+ providers dict via the compatibility layer
        try:
            from hermes_cli.config import get_compatible_custom_providers

            custom_providers = get_compatible_custom_providers(config)
        except Exception:
            return
    if not custom_providers:
        return
    for entry in custom_providers:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str):
            continue
        yield _normalize_custom_pool_name(name), entry


async def get_custom_provider_pool_key(
    base_url: Optional[str],
    provider_name: Optional[str] = None,
    *,
    config: Optional[dict] = None,
) -> Optional[str]:
    """Look up the custom_providers list in config.yaml and return 'custom:<name>' for a matching base_url.

    When provider_name is given, prefer matching by name first (solving the case where
    multiple custom providers share the same base_url but have different API keys).
    Falls back to base_url matching when no name match is found.

    Returns None if no match is found.
    """
    if not base_url:
        return None
    if config is None:
        config = await load_config_readonly()
    normalized_url = base_url.strip().rstrip("/")

    # When a provider name is given, try to match by name first.
    # This fixes the P1 bug where two custom providers sharing the same
    # base_url always resolve to the first one's credentials.
    if provider_name:
        normalized_name = _normalize_custom_pool_name(provider_name)
        for norm_name, entry in _iter_custom_providers(config):
            if norm_name == normalized_name:
                return f"{CUSTOM_POOL_PREFIX}{norm_name}"

    # Fall back to base_url matching (original behavior)
    for norm_name, entry in _iter_custom_providers(config):
        entry_url = str(entry.get("base_url") or "").strip().rstrip("/")
        if entry_url and entry_url == normalized_url:
            return f"{CUSTOM_POOL_PREFIX}{norm_name}"
    return None


async def list_custom_pool_providers() -> List[str]:
    """Return all 'custom:*' pool keys that have entries in auth.json."""
    pool_data = await read_credential_pool(None)
    return sorted(
        key for key in pool_data
        if key.startswith(CUSTOM_POOL_PREFIX)
        and isinstance(pool_data.get(key), list)
        and pool_data[key]
    )


def _get_custom_provider_config(
    pool_key: str, config: Optional[dict] = None,
) -> Optional[Dict[str, Any]]:
    """Return the custom_providers config entry matching a pool key like 'custom:together.ai'."""
    if not pool_key.startswith(CUSTOM_POOL_PREFIX):
        return None
    suffix = pool_key[len(CUSTOM_POOL_PREFIX):]
    for norm_name, entry in _iter_custom_providers(config):
        if norm_name == suffix:
            return entry
    return None


async def get_pool_strategy(provider: str) -> str:
    """Return the configured selection strategy for a provider."""
    config = await load_config_readonly()
    if config is None:
        return STRATEGY_FILL_FIRST

    strategies = config.get("credential_pool_strategies")
    if not isinstance(strategies, dict):
        return STRATEGY_FILL_FIRST

    strategy = str(strategies.get(provider, "") or "").strip().lower()
    if strategy in SUPPORTED_POOL_STRATEGIES:
        return strategy
    return STRATEGY_FILL_FIRST


def credential_pool_matches_provider(
    pool_or_provider: Any,
    provider: Optional[str],
    *,
    base_url: Optional[str] = None,
) -> bool:
    """Return whether a pool belongs to the requested runtime provider.

    Named custom endpoints intentionally use two identities: the live agent is
    ``custom`` while its pool is keyed ``custom:<name>``. Accept that pair only
    when the runtime base URL resolves to the exact same custom pool key.
    Empty string identities fail closed. Legacy pool adapters without a
    ``provider`` attribute remain compatible; production pools are scoped.
    """
    raw_pool_provider = getattr(pool_or_provider, "provider", None)
    if raw_pool_provider is None:
        if isinstance(pool_or_provider, str):
            raw_pool_provider = pool_or_provider
        else:
            # Backward compatibility for lightweight/unscoped pool adapters.
            # Production CredentialPool instances always carry ``provider``;
            # old plugins and tests may expose only select()/has_credentials().
            return True
    pool_provider = str(raw_pool_provider or "").strip().lower()
    provider_norm = str(provider or "").strip().lower()
    if not pool_provider or not provider_norm:
        return False
    if pool_provider == provider_norm:
        return True
    if provider_norm != "custom" or not pool_provider.startswith(CUSTOM_POOL_PREFIX):
        return False
    target_url = str(base_url or "").strip().rstrip("/")
    if not target_url:
        return False
    entries = getattr(pool_or_provider, "entries", None)
    if not callable(entries):
        return False
    return any(
        str(getattr(entry, "base_url", "") or "").strip().rstrip("/") == target_url
        for entry in entries()
    )


DEFAULT_MAX_CONCURRENT_PER_CREDENTIAL = 1


async def _write_through_provider_state_to_global_root(
    provider_id: str, state: Dict[str, Any]
) -> None:
    """Persist a rotated OAuth ``state`` into the global-root auth.json.

    Best-effort write-through for the multi-profile rotation hazard
    (#48415 / #43589): nous, openai-codex, and xai-oauth rotate the
    refresh_token on refresh, so when a profile pool refresh rotates a grant
    it resolved from the root fallback, the rotated chain must land back in
    root. Otherwise root keeps a now-revoked refresh token and every other
    profile reading the stale root grant dies with ``refresh_token_reused`` /
    ``invalid_grant`` once its access token expires.

    Only updates ``providers.<provider_id>`` in the root store; never touches
    the profile store (the caller already saved that). Swallows all errors — a
    failed write-through degrades to the pre-existing behavior (root stale), it
    must never break the profile's own successful save. Mirrors
    ``hermes_cli.auth._write_through_xai_oauth_to_global_root`` (which covers
    the non-pool xAI refresh path) for the credential-pool refresh path.
    """
    try:
        global_path = auth_mod._global_auth_file_path()
    except Exception:
        return
    if global_path is None:
        # Classic mode (profile == root); the profile save already hit root.
        return
    # Seat belt: under pytest, refuse to write the real user's
    # ~/.hermes/auth.json even when HERMES_HOME points at a profile path
    # (mirrors the read-side guard in _load_global_auth_store). Uses the
    # unmodified HOME env, not Path.home() which fixtures may monkeypatch.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        real_home_env = os.environ.get("HOME", "")
        if real_home_env:
            real_root = Path(real_home_env) / ".hermes" / "auth.json"
            try:
                if os.path.abspath(global_path) == os.path.abspath(real_root):
                    return
            except Exception:
                return
    try:
        async with auth_mod._auth_store_transaction(global_path):
            auth_store = await auth_mod._load_auth_store(global_path)
            auth_mod._store_provider_state(
                auth_store,
                provider_id,
                state,
                set_active=False,
            )
            await auth_mod._save_auth_store(auth_store, global_path)
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug(
            "%s pool refresh: write-through to global root failed: %s",
            provider_id,
            exc,
        )


class CredentialPool:
    def __init__(
        self,
        provider: str,
        entries: List[PooledCredential],
        *,
        strategy: str = STRATEGY_FILL_FIRST,
    ):
        self.provider = provider
        self._entries = sorted(entries, key=lambda entry: entry.priority)
        self._current_id: Optional[str] = None
        self._strategy = strategy
        self._lock = asyncio.Lock()
        self._refresh_lock = asyncio.Lock()
        self._active_leases: Dict[str, int] = {}
        self._max_concurrent = DEFAULT_MAX_CONCURRENT_PER_CREDENTIAL
        # Monotonic timestamp of the last "no available entries" log, used to
        # throttle that message so an empty/exhausted pool cannot storm the
        # shared rotating log (see NO_AVAILABLE_ENTRIES_LOG_THROTTLE_SECONDS).
        # Re-armed to None on every successful selection so a recover→re-exhaust
        # transition logs promptly instead of being swallowed by a stale window.
        self._last_no_entries_log_at: Optional[float] = None
        # #70401: consecutive mark_exhausted_and_rotate() calls whose supplied
        # credential identity matched no pool entry (OAuth wrappers whose
        # runtime key rotates, entries pruned by another process, ...).  These
        # rotations mark nothing exhausted, so without a cap the pool can
        # never converge to "no available entries" and the caller's 401 retry
        # loop runs unbounded and non-interruptible.  Reset whenever a real
        # entry is identified or an escape path returns None.
        self._unmatched_rotation_streak: int = 0

    def has_credentials(self) -> bool:
        return bool(self._entries)

    def entries(self) -> List[PooledCredential]:
        return list(self._entries)

    def _current_unlocked(self) -> Optional[PooledCredential]:
        if not self._current_id:
            return None
        return next((entry for entry in self._entries if entry.id == self._current_id), None)

    def current(self) -> Optional[PooledCredential]:
        return self._current_unlocked()

    def entry_id_for_api_key(self, api_key_hint: Any = None) -> Optional[str]:
        """Return the stable id for the runtime credential in use.

        Prefer the current selection when it still supplies ``api_key_hint``.
        If the cursor was cleared, fall back to an unambiguous key match.
        """
        current = self._current_unlocked()
        if current is not None and (
            api_key_hint is None
            or current.runtime_api_key == api_key_hint
        ):
            return current.id
        if api_key_hint is None:
            return None
        matches = [
            entry
            for entry in self._entries
            if entry.runtime_api_key == api_key_hint
        ]
        return matches[0].id if len(matches) == 1 else None

    def _replace_entry(self, old: PooledCredential, new: PooledCredential) -> None:
        """Swap an entry in-place by id, preserving sort order."""
        for idx, entry in enumerate(self._entries):
            if entry.id == old.id:
                self._entries[idx] = new
                return

    def _is_terminal_auth_failure(
        self,
        status_code: Optional[int],
        normalized_error: Dict[str, Any],
    ) -> bool:
        """Detect upstream-permanent OAuth failures that won't recover on TTL.

        Only fires for 401 responses whose error code/reason matches a known
        terminal OAuth state (token_invalidated, token_revoked, invalid_grant,
        etc.).  Distinguishes permanent failures from transient ones like
        token_expired (refreshable) or generic 401 without a specific reason
        (could be a server-side glitch worth retrying).

        Returns False for non-401 status codes — 429 rate limits and 402
        billing failures are transient by nature and should keep TTL semantics.
        """
        if status_code != 401:
            return False
        reason = normalized_error.get("reason")
        if not isinstance(reason, str):
            return False
        return reason.strip().lower() in _TERMINAL_AUTH_REASONS

    def _mark_exhausted(
        self,
        entry: PooledCredential,
        status_code: Optional[int],
        error_context: Optional[Dict[str, Any]] = None,
    ) -> PooledCredential:
        normalized_error = _normalize_error_context(error_context)
        # Permanent OAuth failures (token_invalidated, token_revoked, etc.)
        # transition to STATUS_DEAD instead of STATUS_EXHAUSTED.  Without this,
        # a revoked credential gets a 1-hour TTL cooldown and then re-enters
        # rotation, failing immediately every hour until the user manually
        # removes it (issue #32849).  DEAD entries are excluded from rotation
        # unconditionally and only clear via an explicit re-auth write-side
        # sync (``_save_codex_tokens`` after a fresh device-code login).
        if self._is_terminal_auth_failure(status_code, normalized_error):
            terminal_status = STATUS_DEAD
        else:
            terminal_status = STATUS_EXHAUSTED
        updated = replace(
            entry,
            last_status=terminal_status,
            last_status_at=time.time(),
            last_error_code=status_code,
            last_error_reason=normalized_error.get("reason"),
            last_error_message=normalized_error.get("message"),
            last_error_reset_at=normalized_error.get("reset_at"),
        )
        self._replace_entry(entry, updated)
        return updated
    def _entry_needs_refresh(self, entry: PooledCredential) -> bool:
        if entry.auth_type != AUTH_TYPE_OAUTH:
            return False
        if self.provider == "anthropic":
            if entry.expires_at_ms is None:
                return False
            return int(entry.expires_at_ms) <= int(time.time() * 1000) + 120_000
        if self.provider == "openai-codex":
            return _codex_access_token_is_expiring(
                entry.access_token,
                CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
            )
        if self.provider == "xai-oauth":
            return auth_mod._xai_access_token_is_expiring(
                entry.access_token,
                auth_mod._xai_proactive_refresh_skew_seconds(entry.access_token),
            )
        if self.provider == "nous":
            # Nous refresh can require network access and should happen when
            # runtime credentials are actually resolved, not merely when the pool
            # is enumerated for listing, migration, or selection.
            return False
        return False

    async def _sync_anthropic_entry_from_credentials_file(
        self, entry: PooledCredential,
    ) -> PooledCredential:
        """Adopt a newer Claude Code credential file snapshot when present."""
        if self.provider != "anthropic" or entry.source != "claude_code":
            return entry

        from agent.anthropic_adapter import read_claude_code_credentials

        creds = await read_claude_code_credentials()
        if not creds:
            return entry

        access_token = str(creds.get("accessToken") or "")
        refresh_token = str(creds.get("refreshToken") or "")
        expires_at = creds.get("expiresAt")
        if not access_token and not refresh_token:
            return entry
        if (
            access_token == (entry.access_token or "")
            and refresh_token == (entry.refresh_token or "")
        ):
            return entry

        updated = replace(
            entry,
            access_token=access_token or entry.access_token,
            refresh_token=refresh_token or entry.refresh_token,
            expires_at_ms=expires_at or entry.expires_at_ms,
            last_status=None,
            last_status_at=None,
            last_error_code=None,
            last_error_reason=None,
            last_error_message=None,
            last_error_reset_at=None,
        )
        self._replace_entry(entry, updated)
        await self._persist()
        return updated

    async def _persist(self, *, removed_ids: Optional[List[str]] = None) -> None:
        """Persist the current snapshot without blocking an agent turn."""
        await write_credential_pool(
            self.provider,
            [entry.to_dict() for entry in self._entries],
            removed_ids=removed_ids,
        )

    async def _sync_xai_oauth_entry_from_auth_store(
        self, entry: PooledCredential,
    ) -> PooledCredential:
        """Adopt a newer singleton xAI token pair without blocking the loop."""
        if entry.source != "device_code":
            return entry
        try:
            auth_store = await auth_mod._load_auth_store()
            state = auth_mod._load_provider_state(auth_store, "xai-oauth")
            tokens = state.get("tokens") if isinstance(state, dict) else None
            if not isinstance(tokens, dict):
                return entry
            access_token = str(tokens.get("access_token") or "").strip()
            refresh_token = str(tokens.get("refresh_token") or "").strip()
            if not access_token or (
                access_token == entry.access_token
                and refresh_token == str(entry.refresh_token or "")
            ):
                return entry
            updated = replace(
                entry,
                access_token=access_token,
                refresh_token=refresh_token or entry.refresh_token,
                last_refresh=state.get("last_refresh") or entry.last_refresh,
                last_status=STATUS_OK,
                last_status_at=None,
                last_error_code=None,
                last_error_reason=None,
                last_error_message=None,
                last_error_reset_at=None,
            )
            self._replace_entry(entry, updated)
            return updated
        except Exception as exc:
            logger.debug("Failed to sync xAI OAuth entry from auth store: %s", exc)
            return entry

    def _sync_codex_entry_from_auth_store(
        self,
        entry: PooledCredential,
        auth_store: Dict[str, Any],
    ) -> PooledCredential:
        """Adopt a newer singleton Codex token pair from a locked snapshot."""
        if entry.source not in {"device_code", "manual:device_code"}:
            return entry
        state = auth_mod._load_provider_state(auth_store, "openai-codex")
        tokens = state.get("tokens") if isinstance(state, dict) else None
        if not isinstance(tokens, dict):
            return entry
        store_access = str(tokens.get("access_token") or "").strip()
        store_refresh = str(tokens.get("refresh_token") or "").strip()
        should_adopt = bool(
            store_access
            and (
                store_access != entry.access_token
                or (store_refresh and store_refresh != str(entry.refresh_token or ""))
            )
        ) or bool(
            store_refresh
            and store_refresh != str(entry.refresh_token or "")
            and not store_access
        )
        if not should_adopt:
            return entry
        updated = replace(
            entry,
            access_token=store_access or entry.access_token,
            refresh_token=store_refresh or entry.refresh_token,
            last_refresh=state.get("last_refresh") or entry.last_refresh,
            last_status=STATUS_OK,
            last_status_at=None,
            last_error_code=None,
            last_error_reason=None,
            last_error_message=None,
            last_error_reset_at=None,
        )
        self._replace_entry(entry, updated)
        return updated

    async def _sync_device_code_entry_to_auth_store(
        self, entry: PooledCredential,
    ) -> None:
        """Write a rotated singleton-seeded OAuth token pair back to auth.json."""
        if entry.source != "device_code" or self.provider not in {
            "nous", "openai-codex", "xai-oauth",
        }:
            return
        state: Optional[Dict[str, Any]] = None
        write_through_to_root = False
        try:
            async with auth_mod._auth_store_transaction():
                auth_store = await auth_mod._load_auth_store()
                providers = auth_store.get("providers")
                write_through_to_root = not (
                    isinstance(providers, dict)
                    and isinstance(providers.get(self.provider), dict)
                )
                state = auth_mod._load_provider_state(auth_store, self.provider)
                if not isinstance(state, dict):
                    return
                if self.provider == "nous":
                    state["access_token"] = entry.access_token
                    if entry.refresh_token:
                        state["refresh_token"] = entry.refresh_token
                    if entry.expires_at:
                        state["expires_at"] = entry.expires_at
                    if entry.agent_key:
                        state["agent_key"] = entry.agent_key
                    if entry.agent_key_expires_at:
                        state["agent_key_expires_at"] = entry.agent_key_expires_at
                    if entry.inference_base_url:
                        state["inference_base_url"] = entry.inference_base_url
                else:
                    tokens = state.get("tokens")
                    if not isinstance(tokens, dict):
                        return
                    tokens["access_token"] = entry.access_token
                    if entry.refresh_token:
                        tokens["refresh_token"] = entry.refresh_token
                    if entry.last_refresh:
                        state["last_refresh"] = entry.last_refresh
                auth_mod._store_provider_state(
                    auth_store,
                    self.provider,
                    state,
                    set_active=False,
                )
                await auth_mod._save_auth_store(auth_store)
            if write_through_to_root and state is not None:
                await _write_through_provider_state_to_global_root(
                    self.provider,
                    state,
                )
        except Exception as exc:
            logger.debug(
                "Failed to sync %s pool entry back to auth store: %s",
                self.provider,
                exc,
            )

    async def _available_entries(
        self,
        *,
        clear_expired: bool = False,
        allow_refresh: bool = True,
    ) -> List[PooledCredential]:
        """Return entries usable without invoking a synchronous refresher.

        API-key credentials require only local status bookkeeping.  OAuth
        credentials are admitted while their access token is still valid; a
        required refresh is intentionally rejected here instead of silently
        calling the legacy blocking refresh path.  The OAuth-native refresh
        implementation is added at the provider boundary.
        """
        now = time.time()
        changed = False
        entries_to_prune: List[str] = []
        available: List[PooledCredential] = []
        for entry in list(self._entries):
            # Reference-only API-key rows are hydrated by the legacy CLI
            # loader.  They must never be selected by the async turn with an
            # empty token merely because they survived in auth.json.
            if entry.auth_type == AUTH_TYPE_API_KEY and not entry.runtime_api_key:
                continue
            if entry.last_status == STATUS_DEAD:
                if _is_manual_source(entry.source):
                    dead_at = entry.last_status_at or 0
                    if dead_at and now - dead_at > DEAD_MANUAL_PRUNE_TTL_SECONDS:
                        entries_to_prune.append(entry.id)
                        changed = True
                continue
            if entry.last_status == STATUS_EXHAUSTED:
                exhausted_until = _exhausted_until(entry)
                if exhausted_until is not None and now < exhausted_until:
                    continue
                if clear_expired:
                    entry = replace(
                        entry,
                        last_status=STATUS_OK,
                        last_status_at=None,
                        last_error_code=None,
                        last_error_reason=None,
                        last_error_message=None,
                        last_error_reset_at=None,
                    )
                    self._replace_entry(
                        next(candidate for candidate in self._entries if candidate.id == entry.id),
                        entry,
                    )
                    changed = True
            if entry.auth_type == AUTH_TYPE_OAUTH and self._entry_needs_refresh(entry):
                if not allow_refresh:
                    continue
                refreshed = await self._refresh_entry(entry, force=False)
                if refreshed is None:
                    continue
                entry = refreshed
            available.append(entry)
        if entries_to_prune:
            removed = set(entries_to_prune)
            self._entries = [entry for entry in self._entries if entry.id not in removed]
        if changed:
            await self._persist(removed_ids=entries_to_prune)
        return available

    async def _select_unlocked(
        self, *, allow_refresh: bool = True
    ) -> Optional[PooledCredential]:
        available = await self._available_entries(
            clear_expired=True,
            allow_refresh=allow_refresh,
        )
        if not available:
            self._current_id = None
            self._log_no_available_entries()
            return None
        self._last_no_entries_log_at = None
        if self._strategy == STRATEGY_RANDOM:
            entry = random.choice(available)
        elif self._strategy == STRATEGY_LEAST_USED and len(available) > 1:
            entry = min(available, key=lambda candidate: candidate.request_count)
            updated = replace(entry, request_count=entry.request_count + 1)
            self._replace_entry(entry, updated)
            entry = updated
        elif self._strategy == STRATEGY_ROUND_ROBIN and len(available) > 1:
            entry = available[0]
            rotated = [candidate for candidate in self._entries if candidate.id != entry.id]
            rotated.append(replace(entry, priority=len(self._entries) - 1))
            self._entries = [
                replace(candidate, priority=index)
                for index, candidate in enumerate(rotated)
            ]
            await self._persist()
            entry = self._current_unlocked() or entry
        else:
            entry = available[0]
        self._current_id = entry.id
        return entry

    def _log_no_available_entries(self) -> None:
        """Emit the empty-pool INFO line at most once per throttle window."""
        now = time.monotonic()
        last = self._last_no_entries_log_at
        if last is not None and (now - last) < NO_AVAILABLE_ENTRIES_LOG_THROTTLE_SECONDS:
            return
        self._last_no_entries_log_at = now
        logger.info("credential pool: no available entries (all exhausted or empty)")

    async def select(self) -> Optional[PooledCredential]:
        """Select a credential, refreshing OAuth outside the pool lock."""
        while True:
            async with self._lock:
                refresh_entry = next(
                    (
                        entry
                        for entry in self._entries
                        if entry.last_status != STATUS_DEAD
                        and self._entry_needs_refresh(entry)
                    ),
                    None,
                )
                if refresh_entry is None:
                    entry = await self._select_unlocked(allow_refresh=False)
                    if entry is not None:
                        self._unmatched_rotation_streak = 0
                    return entry

            async with self._refresh_lock:
                async with self._lock:
                    current = next(
                        (
                            entry
                            for entry in self._entries
                            if entry.id == refresh_entry.id
                        ),
                        None,
                    )
                    if current is None or not self._entry_needs_refresh(current):
                        continue
                await self._refresh_entry(current, force=False)

    async def peek(self) -> Optional[PooledCredential]:
        """Return the current or next usable credential without selecting it."""
        async with self._lock:
            return self._current_unlocked() or next(
                iter(await self._available_entries(allow_refresh=False)), None,
            )

    async def has_available(self) -> bool:
        """Check pool availability without a synchronous token refresh."""
        async with self._lock:
            return bool(await self._available_entries(allow_refresh=False))

    async def next_available_at(self) -> Optional[float]:
        """Return the earliest known epoch when an exhausted entry recovers."""
        async with self._lock:
            available = await self._available_entries(allow_refresh=False)
            if available:
                return None
            candidates = [
                exhausted_until
                for entry in self._entries
                if entry.last_status == STATUS_EXHAUSTED
                and (exhausted_until := _exhausted_until(entry)) is not None
            ]
            return min(candidates) if candidates else None

    async def mark_exhausted_and_rotate(
        self,
        *,
        status_code: Optional[int],
        error_context: Optional[Dict[str, Any]] = None,
        api_key_hint: Optional[str] = None,
        credential_id: Optional[str] = None,
    ) -> Optional[PooledCredential]:
        """Async-agent implementation of the existing rotate-after-failure rule.

        It deliberately mirrors the identity attribution and sibling-key
        quarantine rules of ``mark_exhausted_and_rotate`` while its only disk
        write goes through the awaitable auth-store boundary.
        """
        async with self._lock:
            entry = None
            identity_supplied = bool(credential_id or api_key_hint)
            if credential_id:
                entry = next(
                    (candidate for candidate in self._entries if candidate.id == credential_id),
                    None,
                )
            if entry is None and api_key_hint:
                entry = next(
                    (
                        candidate
                        for candidate in self._entries
                        if candidate.runtime_api_key == api_key_hint
                    ),
                    None,
                )
            if entry is None and identity_supplied:
                self._unmatched_rotation_streak += 1
                available_count = len(await self._available_entries())
                if self._unmatched_rotation_streak > max(available_count, 1):
                    logger.warning(
                        "credential pool: failed credential identity matched no %s entry "
                        "for %d consecutive rotations (pool size %d) — surfacing error",
                        self.provider,
                        self._unmatched_rotation_streak,
                        available_count,
                    )
                    self._unmatched_rotation_streak = 0
                    self._current_id = None
                    return None
                self._current_id = None
                next_entry = await self._select_unlocked()
                if next_entry is not None and len(await self._available_entries()) == 1:
                    self._unmatched_rotation_streak = 0
                    self._current_id = None
                    return None
                return next_entry

            self._unmatched_rotation_streak = 0
            if entry is None:
                entry = self._current_unlocked() or await self._select_unlocked()
            if entry is None:
                return None
            label = entry.label or entry.id[:8]
            self._mark_exhausted(entry, status_code, error_context)
            failed_runtime_key = entry.runtime_api_key
            if identity_supplied and failed_runtime_key:
                for sibling in list(self._entries):
                    if sibling.id != entry.id and sibling.runtime_api_key == failed_runtime_key:
                        self._mark_exhausted(sibling, status_code, error_context)
            await self._persist()
            updated_entry = next(
                (candidate for candidate in self._entries if candidate.id == entry.id), entry,
            )
            logger.info(
                "credential pool: marking %s %s (status=%s), rotating",
                label,
                "DEAD" if updated_entry.last_status == STATUS_DEAD else "exhausted",
                status_code,
            )
            self._current_id = None
            next_entry = await self._select_unlocked()
            if next_entry:
                logger.info("credential pool: rotated to %s", next_entry.label or next_entry.id[:8])
            return next_entry

    async def try_refresh_matching(
        self,
        api_key_hint: Optional[str] = None,
        credential_id: Optional[str] = None,
    ) -> Optional[PooledCredential]:
        """Refresh a matching OAuth entry when the native refresher exists.

        API-key entries have nothing to refresh and are left for the normal
        rotation path.  Keeping this explicit prevents accidental calls into
        the blocking legacy OAuth implementation.
        """
        async with self._lock:
            entry = None
            if credential_id:
                entry = next(
                    (candidate for candidate in self._entries if candidate.id == credential_id),
                    None,
                )
            if entry is None and api_key_hint:
                entry = next(
                    (
                        candidate
                        for candidate in self._entries
                        if candidate.runtime_api_key == api_key_hint
                    ),
                    None,
                )
            entry = entry or self._current_unlocked()
            if entry is None or entry.auth_type != AUTH_TYPE_OAUTH:
                return None
            return await self._refresh_entry(entry, force=True)

    async def _refresh_entry(
        self, entry: PooledCredential, *, force: bool,
    ) -> Optional[PooledCredential]:
        """Refresh an OAuth entry only through a native provider transport."""
        # This cannot be a module import because runtime helpers already import
        # this module.  Bind it before the try block so the exception handler
        # is valid for the Anthropic branch as well.
        from agent.agent_runtime_helpers import UnsupportedCapabilityError

        if entry.auth_type != AUTH_TYPE_OAUTH or not entry.refresh_token:
            if force:
                self._mark_exhausted(entry, None)
                await self._persist()
            return None

        if self.provider not in {
            "anthropic",
            "minimax-oauth",
            "openai-codex",
            "xai-oauth",
        }:
            raise UnsupportedCapabilityError(
                f"Credential pool OAuth refresh for {self.provider} is not native async yet."
            )

        codex_target_path = None
        try:
            if self.provider == "minimax-oauth":
                if entry.source != "oauth":
                    raise UnsupportedCapabilityError(
                        "Manual MiniMax OAuth pool entries cannot be refreshed; "
                        "log in through the Hermes MiniMax OAuth flow."
                    )
                await auth_mod.resolve_minimax_oauth_runtime_credentials(
                    force_refresh=force,
                )
                auth_store = await auth_mod._load_auth_store()
                state = auth_mod._load_provider_state(
                    auth_store,
                    "minimax-oauth",
                )
                if not state or not state.get("access_token"):
                    return None
                expires_at_ms = None
                try:
                    expires_at_ms = int(
                        datetime.fromisoformat(
                            str(state.get("expires_at") or "")
                        ).timestamp()
                        * 1000
                    )
                except Exception:
                    pass
                updated = replace(
                    entry,
                    access_token=state["access_token"],
                    refresh_token=state.get("refresh_token"),
                    expires_at_ms=expires_at_ms,
                    last_status=STATUS_OK,
                    last_status_at=None,
                    last_error_code=None,
                    last_error_reason=None,
                    last_error_message=None,
                    last_error_reset_at=None,
                )
                self._replace_entry(entry, updated)
                await self._persist()
                return updated

            if self.provider == "openai-codex":
                refresh_timeout = auth_mod.env_float(
                    "HERMES_CODEX_REFRESH_TIMEOUT_SECONDS",
                    20,
                )
                lock_timeout = max(
                    float(auth_mod.AUTH_LOCK_TIMEOUT_SECONDS),
                    float(refresh_timeout) + 5.0,
                )
                local_store = await auth_mod._load_auth_store()
                local_pool = local_store.get("credential_pool")
                local_entries = (
                    local_pool.get(self.provider)
                    if isinstance(local_pool, dict)
                    else None
                )
                local_owns_entry = bool(
                    isinstance(local_entries, list)
                    and any(
                        isinstance(candidate, dict)
                        and candidate.get("id") == entry.id
                        for candidate in local_entries
                    )
                )
                local_state = auth_mod._load_provider_state(
                    local_store,
                    self.provider,
                )
                if not local_owns_entry and local_state is None:
                    global_path = auth_mod._global_auth_file_path()
                    if global_path is not None:
                        global_store = await auth_mod._load_global_auth_store()
                        global_pool = global_store.get("credential_pool")
                        global_entries = (
                            global_pool.get(self.provider)
                            if isinstance(global_pool, dict)
                            else None
                        )
                        if (
                            isinstance(global_entries, list)
                            and any(
                                isinstance(candidate, dict)
                                and candidate.get("id") == entry.id
                                for candidate in global_entries
                            )
                        ) or auth_mod._load_provider_state(
                            global_store,
                            self.provider,
                        ) is not None:
                            codex_target_path = global_path
                async with auth_mod._auth_store_transaction(
                    codex_target_path,
                    timeout_seconds=lock_timeout,
                ):
                    auth_store = await auth_mod._load_auth_store(codex_target_path)
                    entry = self._sync_codex_entry_from_auth_store(entry, auth_store)
                    if not force and not self._entry_needs_refresh(entry):
                        auth_mod._merge_credential_pool_entries(
                            auth_store,
                            self.provider,
                            [candidate.to_dict() for candidate in self._entries],
                        )
                        await auth_mod._save_auth_store(
                            auth_store,
                            codex_target_path,
                        )
                        return entry

                    refreshed = await auth_mod.refresh_codex_oauth_pure(
                        entry.access_token,
                        entry.refresh_token or "",
                        timeout_seconds=refresh_timeout,
                    )
                    updated = replace(
                        entry,
                        access_token=refreshed["access_token"],
                        refresh_token=refreshed["refresh_token"],
                        last_refresh=refreshed.get("last_refresh"),
                        last_status=STATUS_OK,
                        last_status_at=None,
                        last_error_code=None,
                        last_error_reason=None,
                        last_error_message=None,
                        last_error_reset_at=None,
                    )
                    self._replace_entry(entry, updated)
                    auth_mod._merge_credential_pool_entries(
                        auth_store,
                        self.provider,
                        [candidate.to_dict() for candidate in self._entries],
                    )
                    if updated.source == "device_code":
                        state = auth_mod._load_provider_state(
                            auth_store,
                            self.provider,
                        )
                        if isinstance(state, dict):
                            tokens = state.get("tokens")
                            if isinstance(tokens, dict):
                                tokens["access_token"] = updated.access_token
                                tokens["refresh_token"] = updated.refresh_token
                                state["last_refresh"] = updated.last_refresh
                                auth_mod._store_provider_state(
                                    auth_store,
                                    self.provider,
                                    state,
                                    set_active=False,
                                )
                    await auth_mod._save_auth_store(auth_store, codex_target_path)
                return updated

            if self.provider == "xai-oauth":
                synced = await self._sync_xai_oauth_entry_from_auth_store(entry)
                if synced is not entry:
                    entry = synced
                    if not force and not self._entry_needs_refresh(entry):
                        return entry
                import httpx

                async with httpx.AsyncClient(timeout=20) as client:
                    discovery_response = await client.get(
                        auth_mod.XAI_OAUTH_DISCOVERY_URL,
                        headers={"Accept": "application/json"},
                    )
                    discovery_response.raise_for_status()
                    discovery = discovery_response.json()
                    token_endpoint = str(
                        discovery.get("token_endpoint", "")
                        if isinstance(discovery, dict)
                        else ""
                    ).strip()
                    auth_mod._xai_validate_oauth_endpoint(
                        token_endpoint,
                        field="token_endpoint",
                    )
                    response = await client.post(
                        token_endpoint,
                        headers={
                            "Accept": "application/json",
                            "Content-Type": "application/x-www-form-urlencoded",
                        },
                        data={
                            "grant_type": "refresh_token",
                            "client_id": auth_mod.XAI_OAUTH_CLIENT_ID,
                            "refresh_token": entry.refresh_token,
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
                access_token = str(
                    payload.get("access_token", "")
                    if isinstance(payload, dict)
                    else ""
                ).strip()
                if not access_token:
                    raise ValueError("xAI OAuth refresh response omitted access_token")
                updated = replace(
                    entry,
                    access_token=access_token,
                    refresh_token=str(
                        payload.get("refresh_token") or entry.refresh_token
                    ).strip(),
                    last_refresh=datetime.now(timezone.utc).isoformat(),
                    last_status=STATUS_OK,
                    last_status_at=None,
                    last_error_code=None,
                    last_error_reason=None,
                    last_error_message=None,
                    last_error_reset_at=None,
                )
                self._replace_entry(entry, updated)
                await self._persist()
                await self._sync_device_code_entry_to_auth_store(updated)
                return updated

            from agent.anthropic_adapter import (
                _write_claude_code_credentials,
                refresh_anthropic_oauth_pure,
            )

            entry = await self._sync_anthropic_entry_from_credentials_file(entry)
            refreshed = await refresh_anthropic_oauth_pure(
                entry.refresh_token,
                use_json=entry.source.endswith("hermes_pkce"),
            )
            updated = replace(
                entry,
                access_token=refreshed["access_token"],
                refresh_token=refreshed["refresh_token"],
                expires_at_ms=refreshed["expires_at_ms"],
                last_status=STATUS_OK,
                last_status_at=None,
                last_error_code=None,
                last_error_reason=None,
                last_error_message=None,
                last_error_reset_at=None,
            )
            self._replace_entry(entry, updated)
            await self._persist()
            if entry.source == "claude_code":
                await _write_claude_code_credentials(
                    updated.access_token,
                    updated.refresh_token or entry.refresh_token,
                    updated.expires_at_ms or 0,
                )
            return updated
        except UnsupportedCapabilityError:
            # Never silently downgrade an unsupported OAuth refresh into an
            # exhausted credential.  The async distribution has no sync or
            # worker-thread fallback for this path, so callers need a clear
            # capability failure instead of a misleading quota error.
            raise
        except Exception as exc:
            if self.provider == "openai-codex":
                async with auth_mod._auth_store_transaction(codex_target_path):
                    auth_store = await auth_mod._load_auth_store(codex_target_path)
                    synced = self._sync_codex_entry_from_auth_store(entry, auth_store)
                    if synced.refresh_token != entry.refresh_token:
                        auth_mod._merge_credential_pool_entries(
                            auth_store,
                            self.provider,
                            [candidate.to_dict() for candidate in self._entries],
                        )
                        await auth_mod._save_auth_store(
                            auth_store,
                            codex_target_path,
                        )
                        return synced
                    if auth_mod._is_terminal_codex_oauth_refresh_error(exc):
                        state = auth_mod._load_provider_state(
                            auth_store,
                            self.provider,
                        )
                        if isinstance(state, dict):
                            tokens = state.get("tokens")
                            if isinstance(tokens, dict):
                                store_refresh = str(
                                    tokens.get("refresh_token") or ""
                                ).strip()
                                entry_refresh = str(entry.refresh_token or "").strip()
                                if not store_refresh or store_refresh == entry_refresh:
                                    tokens.pop("access_token", None)
                                    tokens.pop("refresh_token", None)
                                    state["last_auth_error"] = {
                                        "provider": self.provider,
                                        "code": getattr(exc, "code", "unknown"),
                                        "message": str(exc),
                                        "reason": "credential_pool_refresh_failure",
                                        "relogin_required": True,
                                        "at": datetime.now(timezone.utc).isoformat(),
                                    }
                                    auth_mod._store_provider_state(
                                        auth_store,
                                        self.provider,
                                        state,
                                        set_active=False,
                                    )
                        removed_ids = [
                            candidate.id
                            for candidate in self._entries
                            if candidate.source == "device_code"
                        ]
                        self._entries = [
                            candidate
                            for candidate in self._entries
                            if candidate.source != "device_code"
                        ]
                        if self._current_id in removed_ids:
                            self._current_id = None
                        auth_mod._merge_credential_pool_entries(
                            auth_store,
                            self.provider,
                            [candidate.to_dict() for candidate in self._entries],
                            removed_ids=removed_ids,
                        )
                        await auth_mod._save_auth_store(
                            auth_store,
                            codex_target_path,
                        )
                        return None
                    self._mark_exhausted(entry, None)
                    auth_mod._merge_credential_pool_entries(
                        auth_store,
                        self.provider,
                        [candidate.to_dict() for candidate in self._entries],
                    )
                    await auth_mod._save_auth_store(
                        auth_store,
                        codex_target_path,
                    )
                    return None
            if self.provider == "xai-oauth":
                synced = await self._sync_xai_oauth_entry_from_auth_store(entry)
                if synced is not entry:
                    await self._persist()
                    return synced
            logger.debug(
                "Async credential refresh failed for %s/%s: %s",
                self.provider,
                entry.id,
                exc,
            )
            self._mark_exhausted(entry, None)
            await self._persist()
            return None

    async def acquire_lease(self, credential_id: Optional[str] = None) -> Optional[str]:
        """Acquire a soft lease on a credential.

        If a specific credential_id is provided, lease that entry directly.
        Otherwise prefer the least-leased available credential, using priority as
        a stable tie-breaker. When every credential is already at the soft cap,
        still return the least-leased one instead of blocking.
        """
        async with self._lock:
            if credential_id:
                self._active_leases[credential_id] = self._active_leases.get(credential_id, 0) + 1
                self._current_id = credential_id
                return credential_id

            available = await self._available_entries(clear_expired=True)
            if not available:
                return None

            below_cap = [
                entry for entry in available
                if self._active_leases.get(entry.id, 0) < self._max_concurrent
            ]
            candidates = below_cap if below_cap else available
            chosen = min(
                candidates,
                key=lambda entry: (self._active_leases.get(entry.id, 0), entry.priority),
            )
            self._active_leases[chosen.id] = self._active_leases.get(chosen.id, 0) + 1
            self._current_id = chosen.id
            return chosen.id

    async def release_lease(self, credential_id: str) -> None:
        """Release a previously acquired credential lease."""
        async with self._lock:
            count = self._active_leases.get(credential_id, 0)
            if count <= 1:
                self._active_leases.pop(credential_id, None)
            else:
                self._active_leases[credential_id] = count - 1

    async def try_refresh_current(self) -> Optional[PooledCredential]:
        async with self._lock:
            entry = self._current_unlocked()
            if entry is None:
                return None
            refreshed = await self._refresh_entry(entry, force=True)
            if refreshed is not None:
                self._current_id = refreshed.id
            return refreshed

    async def reset_statuses(self) -> int:
        async with self._lock:
            count = 0
            new_entries = []
            for entry in self._entries:
                if entry.last_status or entry.last_status_at or entry.last_error_code:
                    new_entries.append(
                        replace(
                            entry,
                            last_status=None,
                            last_status_at=None,
                            last_error_code=None,
                            last_error_reason=None,
                            last_error_message=None,
                            last_error_reset_at=None,
                        )
                    )
                    count += 1
                else:
                    new_entries.append(entry)
            if count:
                self._entries = new_entries
                await self._persist()
            return count

    async def remove_index(self, index: int) -> Optional[PooledCredential]:
        async with self._lock:
            if index < 1 or index > len(self._entries):
                return None
            removed = self._entries.pop(index - 1)
            self._entries = [
                replace(entry, priority=new_priority)
                for new_priority, entry in enumerate(self._entries)
            ]
            await self._persist(removed_ids=[removed.id])
            if self._current_id == removed.id:
                self._current_id = None
            return removed

    def resolve_target(self, target: Any) -> Tuple[Optional[int], Optional[PooledCredential], Optional[str]]:
        raw = str(target or "").strip()
        if not raw:
            return None, None, "No credential target provided."

        for idx, entry in enumerate(self._entries, start=1):
            if entry.id == raw:
                return idx, entry, None

        label_matches = [
            (idx, entry)
            for idx, entry in enumerate(self._entries, start=1)
            if entry.label.strip().lower() == raw.lower()
        ]
        if len(label_matches) == 1:
            return label_matches[0][0], label_matches[0][1], None
        if len(label_matches) > 1:
            return None, None, f'Ambiguous credential label "{raw}". Use the numeric index or entry id instead.'
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(self._entries):
                return index, self._entries[index - 1], None
            return None, None, f"No credential #{index}."
        return None, None, f'No credential matching "{raw}".'

    async def add_entry(self, entry: PooledCredential) -> PooledCredential:
        async with self._lock:
            entry = replace(entry, priority=_next_priority(self._entries))
            self._entries.append(entry)
            await self._persist()
            return entry


def _upsert_entry(entries: List[PooledCredential], provider: str, source: str, payload: Dict[str, Any]) -> bool:
    matching_indices = []
    for idx, entry in enumerate(entries):
        if entry.source == source:
            matching_indices.append(idx)

    existing_idx = matching_indices[0] if matching_indices else None
    duplicate_indices = set(matching_indices[1:])
    if duplicate_indices:
        entries[:] = [entry for idx, entry in enumerate(entries) if idx not in duplicate_indices]

    if existing_idx is None:
        payload.setdefault("id", uuid.uuid4().hex[:6])
        payload.setdefault("priority", _next_priority(entries))
        payload.setdefault("label", payload.get("label") or source)
        entries.append(PooledCredential.from_dict(provider, payload))
        return True

    existing = entries[existing_idx]
    field_updates = {}
    extra_updates = {}
    _field_names = {f.name for f in fields(existing)}
    for key, value in payload.items():
        if key in {"id", "priority"} or value is None:
            continue
        if key == "label" and existing.label:
            continue
        if key in _field_names:
            if getattr(existing, key) != value:
                field_updates[key] = value
        elif key in _EXTRA_KEYS:
            if existing.extra.get(key) != value:
                extra_updates[key] = value
    if field_updates or extra_updates:
        if extra_updates:
            field_updates["extra"] = {**existing.extra, **extra_updates}
        updated = replace(existing, **field_updates)
        entries[existing_idx] = updated
        # Runtime-only borrowed secret updates should refresh the in-memory
        # entry without forcing auth.json churn when the disk-safe payload is
        # unchanged (for example env keys with the same fingerprint).
        return bool(duplicate_indices) or existing.to_dict() != updated.to_dict()
    return bool(duplicate_indices)


def _normalize_pool_priorities(provider: str, entries: List[PooledCredential]) -> bool:
    if provider != "anthropic":
        return False

    source_rank = {
        "env:ANTHROPIC_TOKEN": 0,
        "env:CLAUDE_CODE_OAUTH_TOKEN": 1,
        "hermes_pkce": 2,
        "claude_code": 3,
        "env:ANTHROPIC_API_KEY": 4,
    }
    manual_entries = sorted(
        (entry for entry in entries if _is_manual_source(entry.source)),
        key=lambda entry: entry.priority,
    )
    seeded_entries = sorted(
        (entry for entry in entries if not _is_manual_source(entry.source)),
        key=lambda entry: (
            source_rank.get(entry.source, len(source_rank)),
            entry.priority,
            entry.label,
        ),
    )

    ordered = [*manual_entries, *seeded_entries]
    id_to_idx = {entry.id: idx for idx, entry in enumerate(entries)}
    changed = False
    for new_priority, entry in enumerate(ordered):
        if entry.priority != new_priority:
            entries[id_to_idx[entry.id]] = replace(entry, priority=new_priority)
            changed = True
    return changed


async def _is_provider_explicitly_configured(
    provider_id: str,
    *,
    auth_store: Optional[Dict[str, Any]] = None,
) -> bool:
    """Check provider opt-in without entering synchronous auth/config paths.

    The synchronous auth helper is intentionally retained for CLI/setup code,
    but ``load_pool`` runs on the agent event loop.  Keep this check local to
    the async pool path so external credential discovery cannot block it.
    """
    normalized = (provider_id or "").strip().lower()
    if not normalized:
        return False

    store = auth_store if isinstance(auth_store, dict) else await _load_auth_store()
    active = str(store.get("active_provider") or "").strip().lower()
    if active == normalized:
        return True

    try:
        config = await load_config_readonly()
    except Exception:
        config = {}
    model_cfg = config.get("model") if isinstance(config, dict) else None
    if isinstance(model_cfg, dict):
        configured = str(model_cfg.get("provider") or "").strip().lower()
        if configured == normalized:
            return True

    def _slot_matches(slot: Any) -> bool:
        return (
            isinstance(slot, dict)
            and str(slot.get("provider") or "").strip().lower() == normalized
        )

    moa_cfg = config.get("moa") if isinstance(config, dict) else None
    if isinstance(moa_cfg, dict):
        if any(_slot_matches(slot) for slot in moa_cfg.get("reference_models") or []):
            return True
        if _slot_matches(moa_cfg.get("aggregator")):
            return True
        presets = moa_cfg.get("presets")
        if isinstance(presets, dict):
            for preset in presets.values():
                if not isinstance(preset, dict):
                    continue
                if any(_slot_matches(slot) for slot in preset.get("reference_models") or []):
                    return True
                if _slot_matches(preset.get("aggregator")):
                    return True

    pconfig = PROVIDER_REGISTRY.get(normalized)
    if pconfig and pconfig.auth_type == AUTH_TYPE_API_KEY:
        for env_var in pconfig.api_key_env_vars:
            if env_var == "CLAUDE_CODE_OAUTH_TOKEN":
                continue
            value = await get_env_value_prefer_dotenv(env_var)
            if value and len(value.strip()) >= 4:
                return True

    try:
        persisted = await read_credential_pool(normalized)
    except Exception:
        persisted = []
    for entry in persisted:
        if not isinstance(entry, dict):
            continue
        source = str(entry.get("source") or "").strip().lower()
        if source in {"device_code", "loopback_pkce", "hermes_pkce", "manual"}:
            return True
        if source.startswith("manual:"):
            return True
        if source.startswith("env:"):
            env_var = source.split(":", 1)[1].strip()
            if env_var and await get_env_value_prefer_dotenv(env_var):
                return True
    return False


async def _seed_from_singletons(
    provider: str, entries: List[PooledCredential],
) -> Tuple[bool, Set[str]]:
    changed = False
    active_sources: Set[str] = set()
    auth_store = await _load_auth_store()
    suppressed_sources = auth_store.get("suppressed_sources", {})

    def _is_suppressed(source: str) -> bool:
        provider_sources = (
            suppressed_sources.get(provider, [])
            if isinstance(suppressed_sources, dict)
            else []
        )
        return source in provider_sources

    if provider == "anthropic":
        # Only auto-discover external credentials (Claude Code, Hermes PKCE)
        # when the user has explicitly configured anthropic as their provider.
        # Without this gate, auxiliary client fallback chains silently read
        # ~/.claude/.credentials.json without user consent.  See PR #4210.
        try:
            if not await _is_provider_explicitly_configured(
                "anthropic", auth_store=auth_store
            ):
                return changed, active_sources
        except Exception:
            return changed, active_sources

        # API-key vs OAuth is a user-visible choice at `hermes setup` ("Claude
        # Pro/Max subscription" vs "Anthropic API key").  The signal that the
        # user picked the API-key path is: ANTHROPIC_API_KEY set in the env,
        # AND no OAuth env vars set — `save_anthropic_api_key()` writes the
        # API key and zeros ANTHROPIC_TOKEN; `save_anthropic_oauth_token()`
        # does the inverse.  When that signal is present we MUST NOT seed
        # autodiscovered OAuth tokens (~/.claude/.credentials.json from the
        # Claude Code CLI, hermes_pkce creds from a previous OAuth login)
        # into the anthropic pool — otherwise rotation on a 401/429 silently
        # flips the session onto an OAuth credential, which forces the Claude
        # Code identity injection, `mcp_` tool-name rewrite, and claude-cli
        # User-Agent header (`agent/anthropic_adapter.py:2128`).  Users who
        # explicitly opted into the API-key path are explicitly opting OUT of
        # that masquerade.  Prefer ~/.hermes/.env over os.environ for the
        # same reason `_seed_from_env` does — that's the authoritative file
        # that `hermes setup` writes.
        anthropic_api_key = (
            await get_env_value_prefer_dotenv("ANTHROPIC_API_KEY") or ""
        ).strip()
        anthropic_oauth_env = (
            await get_env_value_prefer_dotenv("ANTHROPIC_TOKEN") or ""
        ).strip() or (
            await get_env_value_prefer_dotenv("CLAUDE_CODE_OAUTH_TOKEN") or ""
        )
        anthropic_oauth_env = anthropic_oauth_env.strip()
        api_key_path_explicit = bool(anthropic_api_key and not anthropic_oauth_env)

        if api_key_path_explicit:
            # Prune any stale autodiscovered OAuth entries that may have been
            # seeded into the on-disk pool during a previous OAuth session.
            # Without this, switching OAuth -> API key at setup leaves the
            # OAuth entries dormant in auth.json forever and rotation on a
            # transient 401 could revive them.
            retained = [
                entry for entry in entries
                if entry.source not in {"hermes_pkce", "claude_code"}
            ]
            if len(retained) != len(entries):
                entries[:] = retained
                changed = True
            return changed, active_sources

        from agent.anthropic_adapter import read_claude_code_credentials, read_hermes_oauth_credentials

        hermes_creds, claude_creds = await asyncio.gather(
            read_hermes_oauth_credentials(),
            read_claude_code_credentials(),
        )
        for source_name, creds in (
            ("hermes_pkce", hermes_creds),
            ("claude_code", claude_creds),
        ):
            if creds and creds.get("accessToken"):
                if _is_suppressed(source_name):
                    continue
                active_sources.add(source_name)
                changed |= _upsert_entry(
                    entries,
                    provider,
                    source_name,
                    {
                        "source": source_name,
                        "auth_type": AUTH_TYPE_OAUTH,
                        "access_token": creds.get("accessToken", ""),
                        "refresh_token": creds.get("refreshToken"),
                        "expires_at_ms": creds.get("expiresAt"),
                        "label": label_from_token(creds.get("accessToken", ""), source_name),
                    },
                )

    elif provider == "nous":
        state = _load_provider_state(auth_store, "nous")
        has_runtime_material = bool(
            isinstance(state, dict)
            and (
                str(state.get("access_token") or "").strip()
                or str(state.get("agent_key") or "").strip()
            )
        )
        if state and not has_runtime_material:
            retained = [
                entry for entry in entries
                if entry.source not in {"device_code", "manual:device_code"}
            ]
            if len(retained) != len(entries):
                entries[:] = retained
                changed = True
        if state and has_runtime_material and not _is_suppressed("device_code"):
            active_sources.add("device_code")
            # Prefer a user-supplied label embedded in the singleton state
            # (set by persist_nous_credentials(label=...) when the user ran
            # `hermes auth add nous --label <name>`).  Fall back to the
            # auto-derived token fingerprint for logins that didn't supply one.
            custom_label = str(state.get("label") or "").strip()
            seeded_label = custom_label or label_from_token(
                state.get("access_token", ""), "device_code"
            )
            changed |= _upsert_entry(
                entries,
                provider,
                "device_code",
                {
                    "source": "device_code",
                    "auth_type": AUTH_TYPE_OAUTH,
                    "access_token": state.get("access_token", ""),
                    "refresh_token": state.get("refresh_token"),
                    "expires_at": state.get("expires_at"),
                    "token_type": state.get("token_type"),
                    "scope": state.get("scope"),
                    "client_id": state.get("client_id"),
                    "portal_base_url": state.get("portal_base_url"),
                    "inference_base_url": state.get("inference_base_url"),
                    "agent_key": state.get("agent_key"),
                    "agent_key_expires_at": state.get("agent_key_expires_at"),
                    # Carry the refresh timestamps into the pool so
                    # freshness-sensitive consumers (self-heal hooks, pool
                    # pruning by age) can distinguish just-refreshed credentials
                    # from stale ones.  Without these, fresh device_code
                    # entries get obtained_at=None and look older than they
                    # are (#15099).
                    "obtained_at": state.get("obtained_at"),
                    "expires_in": state.get("expires_in"),
                    "agent_key_id": state.get("agent_key_id"),
                    "agent_key_expires_in": state.get("agent_key_expires_in"),
                    "agent_key_reused": state.get("agent_key_reused"),
                    "agent_key_obtained_at": state.get("agent_key_obtained_at"),
                    "tls": state.get("tls") if isinstance(state.get("tls"), dict) else None,
                    "label": seeded_label,
                },
            )

    elif provider == "copilot":
        try:
            from hermes_cli.copilot_auth import (
                COPILOT_ENV_VARS,
                get_copilot_api_token,
                resolve_copilot_token,
            )

            sources = ["gh_cli", *[f"env:{name}" for name in COPILOT_ENV_VARS]]
            if all(_is_suppressed(source) for source in sources):
                return changed, active_sources
            token, source = await resolve_copilot_token()
            if token:
                source_name = (
                    "gh_cli" if source == "gh auth token" else f"env:{source}"
                )
                if _is_suppressed(source_name):
                    return changed, active_sources
                api_token, enterprise_base_url = await get_copilot_api_token(token)
                active_sources.add(source_name)
                provider_config = PROVIDER_REGISTRY.get(provider)
                base_url = enterprise_base_url or (
                    provider_config.inference_base_url if provider_config else ""
                )
                changed |= _upsert_entry(
                    entries,
                    provider,
                    source_name,
                    {
                        "source": source_name,
                        "auth_type": AUTH_TYPE_API_KEY,
                        "access_token": api_token,
                        "base_url": base_url,
                        "label": source,
                    },
                )
        except Exception as exc:
            logger.debug("Copilot token seed failed: %s", exc)

    elif provider == "qwen-oauth":
        try:
            creds = await auth_mod.resolve_qwen_runtime_credentials(
                refresh_if_expiring=False,
            )
            token = creds.get("api_key", "")
            if token:
                source_name = creds.get("source", "qwen-cli")
                if not _is_suppressed(source_name):
                    active_sources.add(source_name)
                    changed |= _upsert_entry(
                        entries,
                        provider,
                        source_name,
                        {
                            "source": source_name,
                            "auth_type": AUTH_TYPE_OAUTH,
                            "access_token": token,
                            "expires_at_ms": creds.get("expires_at_ms"),
                            "base_url": creds.get("base_url", ""),
                            "label": creds.get("auth_file", source_name),
                        },
                    )
        except Exception as exc:
            logger.debug("Qwen OAuth token seed failed: %s", exc)

    elif provider == "minimax-oauth":
        state = _load_provider_state(auth_store, provider)
        if state and state.get("access_token"):
            source_name = "oauth"
            if not _is_suppressed(source_name):
                active_sources.add(source_name)
                expires_at_ms = None
                try:
                    raw_expiry = state.get("expires_at", "")
                    if raw_expiry:
                        expires_at_ms = int(
                            datetime.fromisoformat(raw_expiry).timestamp() * 1000
                        )
                except Exception:
                    pass
                changed |= _upsert_entry(
                    entries,
                    provider,
                    source_name,
                    {
                        "source": source_name,
                        "auth_type": AUTH_TYPE_OAUTH,
                        "access_token": state["access_token"],
                        "refresh_token": state.get("refresh_token"),
                        "expires_at_ms": expires_at_ms,
                        "base_url": str(
                            state.get("inference_base_url", "") or ""
                        ).rstrip("/"),
                        "label": state.get("label", "")
                        or label_from_token(state["access_token"], source_name),
                    },
                )

    elif provider == "openai-codex":
        # Respect user suppression — `hermes auth remove openai-codex` marks
        # the device_code source as suppressed so it won't be re-seeded from
        # the Hermes auth store.  Without this gate the removal is instantly
        # undone on the next load_pool() call.
        if _is_suppressed("device_code"):
            return changed, active_sources

        state = _load_provider_state(auth_store, "openai-codex")
        tokens = state.get("tokens") if isinstance(state, dict) else None
        # Hermes owns its own Codex auth state — we do NOT auto-import from
        # ~/.codex/auth.json at pool-load time.  OAuth refresh tokens are
        # single-use, so sharing them with Codex CLI / VS Code causes
        # refresh_token_reused race failures.  Users who want to adopt
        # existing Codex CLI credentials get a one-time, explicit prompt
        # via `hermes auth openai-codex`.
        if isinstance(tokens, dict) and tokens.get("access_token"):
            active_sources.add("device_code")
            custom_label = str(state.get("label") or "").strip()
            changed |= _upsert_entry(
                entries,
                provider,
                "device_code",
                {
                    "source": "device_code",
                    "auth_type": AUTH_TYPE_OAUTH,
                    "access_token": tokens.get("access_token", ""),
                    "refresh_token": tokens.get("refresh_token"),
                    "base_url": "https://chatgpt.com/backend-api/codex",
                    "last_refresh": state.get("last_refresh"),
                    "label": custom_label or label_from_token(tokens.get("access_token", ""), "device_code"),
                },
            )

    elif provider == "xai-oauth":
        # When the user logs in via ``hermes model`` -> xAI Grok OAuth,
        # tokens are written to the auth.json singleton
        # (``providers["xai-oauth"]``).  Surface them in the pool too so
        # ``hermes auth list`` reflects the logged-in state and so the pool
        # is the single source of truth for refresh during runtime resolution.
        state = _load_provider_state(auth_store, "xai-oauth")
        tokens = state.get("tokens") if isinstance(state, dict) else None
        if isinstance(tokens, dict) and tokens.get("access_token"):
            # Device code is the only supported xAI OAuth flow; the singleton is
            # always surfaced as ``device_code`` (consistent with nous/codex).
            source = "device_code"
            if _is_suppressed(source):
                return changed, active_sources
            active_sources.add(source)
            from hermes_cli.auth import DEFAULT_XAI_OAUTH_BASE_URL

            base_url = DEFAULT_XAI_OAUTH_BASE_URL
            changed |= _upsert_entry(
                entries,
                provider,
                source,
                {
                    "source": source,
                    "auth_type": AUTH_TYPE_OAUTH,
                    "access_token": tokens.get("access_token", ""),
                    "refresh_token": tokens.get("refresh_token"),
                    "base_url": base_url,
                    "last_refresh": state.get("last_refresh"),
                    "label": label_from_token(tokens.get("access_token", ""), source),
                },
            )

    return changed, active_sources


async def _seed_from_env(
    provider: str, entries: List[PooledCredential],
) -> Tuple[bool, Set[str]]:
    changed = False
    active_sources: Set[str] = set()
    auth_store = await _load_auth_store()
    suppressed_sources = auth_store.get("suppressed_sources", {})

    # Prefer ~/.hermes/.env over os.environ — the user's config file is the
    # authoritative source for Hermes credentials. Stale env vars from parent
    # processes (Codex CLI, test scripts, etc.) should not override deliberate
    # changes to the .env file.
    async def _get_env_prefer_dotenv(key: str) -> str:
        return (await get_env_value_prefer_dotenv(key) or "").strip()

    # Honour user suppression — `hermes auth remove <provider> <N>` for an
    # env-seeded credential marks the env:<VAR> source as suppressed so it
    # won't be re-seeded from the user's shell environment or ~/.hermes/.env.
    # Without this gate the removal is silently undone on the next
    # load_pool() call whenever the var is still exported by the shell.
    def _is_source_suppressed(source: str) -> bool:
        provider_sources = (
            suppressed_sources.get(provider, [])
            if isinstance(suppressed_sources, dict)
            else []
        )
        return source in provider_sources

    def _env_payload(
        *,
        source: str,
        env_var: str,
        token: str,
        base_url: str,
        auth_type: str = AUTH_TYPE_API_KEY,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "source": source,
            "auth_type": auth_type,
            "access_token": token,
            "base_url": base_url,
            "label": env_var,
        }
        return payload

    if provider == "openrouter":
        # Prefer ~/.hermes/.env over os.environ
        token = await _get_env_prefer_dotenv("OPENROUTER_API_KEY")
        if token:
            source = "env:OPENROUTER_API_KEY"
            if _is_source_suppressed(source):
                return changed, active_sources
            active_sources.add(source)
            changed |= _upsert_entry(
                entries,
                provider,
                source,
                _env_payload(
                    source=source,
                    env_var="OPENROUTER_API_KEY",
                    token=token,
                    base_url=OPENROUTER_BASE_URL,
                ),
            )
        return changed, active_sources

    pconfig = PROVIDER_REGISTRY.get(provider)
    if not pconfig or pconfig.auth_type != AUTH_TYPE_API_KEY:
        return changed, active_sources

    env_url = ""
    if pconfig.base_url_env_var:
        env_url = (await _get_env_prefer_dotenv(pconfig.base_url_env_var)).rstrip("/")

    env_vars = list(pconfig.api_key_env_vars)
    if provider == "anthropic":
        env_vars = [
            "ANTHROPIC_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "ANTHROPIC_API_KEY",
        ]

    for env_var in env_vars:
        # Prefer ~/.hermes/.env over os.environ
        token = await _get_env_prefer_dotenv(env_var)
        if not token:
            continue
        source = f"env:{env_var}"
        if _is_source_suppressed(source):
            continue
        active_sources.add(source)
        base_url = env_url or pconfig.inference_base_url
        if provider == "kimi-coding":
            base_url = _resolve_kimi_base_url(token, pconfig.inference_base_url, env_url)
        elif provider == "zai":
            base_url = await _resolve_zai_base_url(
                token,
                pconfig.inference_base_url,
                env_url,
            )
        changed |= _upsert_entry(
            entries,
            provider,
            source,
            _env_payload(
                source=source,
                env_var=env_var,
                token=token,
                base_url=base_url,
            ),
        )
    return changed, active_sources


def _prune_stale_seeded_entries(
    entries: List[PooledCredential],
    active_sources: Set[str],
    *,
    prune_env_sources: bool = True,
) -> bool:
    def _is_prunable(entry: PooledCredential) -> bool:
        # ``env:*`` entries are persisted references that get re-hydrated from
        # the environment on every load. A process that merely lacks the env
        # var this call must NOT delete the on-disk entry for every other
        # process — that destructive read is the bug behind #9331. Only prune
        # an env source when ``prune_env_sources`` is explicitly requested
        # (e.g. an `hermes auth` command that confirmed the source is gone).
        if entry.source.startswith("env:"):
            return prune_env_sources
        # File-backed singletons (device-code OAuth, claude_code) and Hermes
        # PKCE should disappear from the pool when their backing file is gone.
        return (
            is_borrowed_credential_source(entry.source, entry.provider)
            or entry.source == "hermes_pkce"
        )

    retained = [
        entry
        for entry in entries
        if _is_manual_source(entry.source)
        or entry.source in active_sources
        or not _is_prunable(entry)
    ]
    if len(retained) == len(entries):
        return False
    entries[:] = retained
    return True


async def _seed_custom_pool(
    pool_key: str, entries: List[PooledCredential],
) -> Tuple[bool, Set[str]]:
    """Seed a custom endpoint pool from custom_providers config and model config."""
    changed = False
    active_sources: Set[str] = set()
    auth_store = await _load_auth_store()
    suppressed_sources = auth_store.get("suppressed_sources", {})
    config = await load_config_readonly()

    # Shared suppression gate — same pattern as _seed_from_env/_seed_from_singletons.
    def _is_suppressed(source: str) -> bool:
        provider_sources = (
            suppressed_sources.get(pool_key, [])
            if isinstance(suppressed_sources, dict)
            else []
        )
        return source in provider_sources

    # Seed from the custom_providers config entry's api_key field
    cp_config = _get_custom_provider_config(pool_key, config)
    if cp_config:
        api_key = str(cp_config.get("api_key") or "").strip()
        base_url = str(cp_config.get("base_url") or "").strip().rstrip("/")
        name = str(cp_config.get("name") or "").strip()
        if api_key:
            source = f"config:{name}"
            if not _is_suppressed(source):
                active_sources.add(source)
                changed |= _upsert_entry(
                    entries,
                    pool_key,
                    source,
                    {
                        "source": source,
                        "auth_type": AUTH_TYPE_API_KEY,
                        "access_token": api_key,
                        "base_url": base_url,
                        "label": name or source,
                    },
                )

    # Seed from model.api_key if model.provider=='custom' and model.base_url matches
    try:
        model_cfg = config.get("model") if config else None
        if isinstance(model_cfg, dict):
            model_provider = str(model_cfg.get("provider") or "").strip().lower()
            model_base_url = str(model_cfg.get("base_url") or "").strip().rstrip("/")
            model_api_key = ""
            for k in ("api_key", "api"):
                v = model_cfg.get(k)
                if isinstance(v, str) and v.strip():
                    model_api_key = v.strip()
                    break
            if model_provider == "custom" and model_base_url and model_api_key:
                # Check if this model's base_url matches our custom provider
                matched_key = await get_custom_provider_pool_key(
                    model_base_url, config=config
                )
                if matched_key == pool_key:
                    source = "model_config"
                    if not _is_suppressed(source):
                        active_sources.add(source)
                        changed |= _upsert_entry(
                            entries,
                            pool_key,
                            source,
                            {
                                "source": source,
                                "auth_type": AUTH_TYPE_API_KEY,
                                "access_token": model_api_key,
                                "base_url": model_base_url,
                                "label": "model_config",
                            },
                        )
    except Exception:
        pass

    return changed, active_sources


async def load_pool(provider: str) -> CredentialPool:
    provider = (provider or "").strip().lower()
    raw_entries = await read_credential_pool(provider)
    disk_ids = {
        entry.get("id")
        for entry in raw_entries
        if isinstance(entry, dict) and entry.get("id")
    }
    raw_needs_sanitization = any(
        isinstance(payload, dict)
        and sanitize_borrowed_credential_payload(payload, provider) != payload
        for payload in raw_entries
    )
    entries = [PooledCredential.from_dict(provider, payload) for payload in raw_entries]
    raw_needs_auth_normalization = any(
        isinstance(payload, dict)
        and _normalize_pool_auth_type(
            provider,
            payload.get("access_token"),
            payload.get("auth_type", AUTH_TYPE_API_KEY),
        ) != payload.get("auth_type", AUTH_TYPE_API_KEY)
        for payload in raw_entries
    )
    if raw_needs_auth_normalization:
        # A profile may be reading this provider from the global-root fallback.
        # Keep that fallback read-only: only the store that owns these rows may
        # rewrite them. Loading the default/root profile will heal global rows.
        active_pool = (await _load_auth_store()).get("credential_pool")
        active_entries = active_pool.get(provider) if isinstance(active_pool, dict) else None
        raw_needs_auth_normalization = bool(active_entries)

    if provider.startswith(CUSTOM_POOL_PREFIX):
        # Custom endpoint pool — seed from custom_providers config and model config
        custom_changed, custom_sources = await _seed_custom_pool(provider, entries)
        changed = raw_needs_sanitization or raw_needs_auth_normalization or custom_changed
        changed |= _prune_stale_seeded_entries(entries, custom_sources)
    else:
        singleton_changed, singleton_sources = await _seed_from_singletons(provider, entries)
        env_changed, env_sources = await _seed_from_env(provider, entries)
        changed = (
            raw_needs_sanitization
            or raw_needs_auth_normalization
            or singleton_changed
            or env_changed
        )
        # ``load_pool()`` is a non-destructive read for env-seeded entries: a
        # process missing a provider env var must not delete the persisted
        # pool entry for every other process (#9331). File-backed singletons
        # still prune when their backing file is gone.
        changed |= _prune_stale_seeded_entries(
            entries,
            singleton_sources | env_sources,
            prune_env_sources=False,
        )
        changed |= _normalize_pool_priorities(provider, entries)

    if changed:
        new_ids = {entry.id for entry in entries}
        await write_credential_pool(
            provider,
            [entry.to_dict() for entry in sorted(entries, key=lambda item: item.priority)],
            removed_ids=disk_ids - new_ids,
        )
    return CredentialPool(
        provider,
        entries,
        strategy=await get_pool_strategy(provider),
    )
