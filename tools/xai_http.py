"""Shared helpers for direct xAI HTTP integrations."""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import uuid
from typing import Any, Dict, Optional

import aiofiles
import aiofiles.os


MAX_XAI_STORAGE_EXPIRES_AFTER_SECONDS = 30 * 24 * 60 * 60
SAFE_XAI_STORAGE_EXPIRES_AFTER_SECONDS = 2 * 24 * 60 * 60


async def has_xai_credentials() -> bool:
    """Cheap probe — return True when xAI credentials are *likely* usable.

    Deliberately avoids awaiting :func:`resolve_xai_http_credentials` so callers in
    hot-paint paths (``hermes tools`` repaint, tool-registration scans,
    ``WebSearchProvider.is_available()``) don't incur disk locks or — in
    the OAuth path — a network token refresh. The ABC contract on
    :meth:`agent.web_search_provider.WebSearchProvider.is_available`
    explicitly forbids network calls for exactly this reason.

    Resolution order, fast-to-slow:

    1. ``XAI_API_KEY`` env var (cheapest; covers explicit-key users).
    2. ``~/.hermes/auth.json`` has a non-empty ``providers.xai-oauth.tokens.access_token``
       (single file read, no expiry check, no refresh).
    3. ``credential_pool.xai-oauth`` has any entry with a non-empty
       ``access_token`` (covers multi-account ``hermes auth add xai-oauth``
       grants that are pool-only / ``manual:device_code``).

    Returns False on any exception so a corrupted auth store can't block
    other availability scans. Truthful refresh and expiry handling happens in
    the async request path.
    """
    try:
        from agent.secret_scope import get_secret
    except ImportError:  # pragma: no cover — secret_scope is in-repo
        if os.environ.get("XAI_API_KEY", "").strip():
            return True
    else:
        if (get_secret("XAI_API_KEY", "") or "").strip():
            return True
    try:
        from hermes_constants import get_hermes_home

        auth_path = get_hermes_home() / "auth.json"
        if not await aiofiles.os.path.exists(auth_path):
            return False
        async with aiofiles.open(auth_path, encoding="utf-8") as auth_file:
            store = json.loads(await auth_file.read())
        providers = store.get("providers") if isinstance(store, dict) else None
        xai_state = providers.get("xai-oauth") if isinstance(providers, dict) else None
        tokens = xai_state.get("tokens") if isinstance(xai_state, dict) else None
        access_token = tokens.get("access_token") if isinstance(tokens, dict) else None
        if str(access_token or "").strip():
            return True
        # Pool-only grants (multi-account ``auth add``) never write the
        # providers singleton; still count as present credentials.
        credential_pool = store.get("credential_pool") if isinstance(store, dict) else None
        entries = (
            credential_pool.get("xai-oauth")
            if isinstance(credential_pool, dict)
            else None
        )
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("access_token", "") or "").strip():
                    return True
        return False
    except Exception:
        return False


def get_env_value(name: str, default=None):
    """Read ``name`` from ``~/.hermes/.env`` first, then ``os.environ``.

    Wraps :func:`hermes_cli.config.get_env_value` so tests can patch
    ``tools.xai_http.get_env_value`` to inject dotenv-only secrets into the
    xAI credential resolver.
    """
    try:
        from hermes_cli.config import get_env_value as _hermes_get_env_value
    except ImportError:
        return os.environ.get(name, default)

    value = _hermes_get_env_value(name)
    return value if value is not None else default


def hermes_xai_user_agent() -> str:
    """Return a stable Hermes-specific User-Agent for xAI HTTP calls."""
    try:
        from hermes_cli import __version__ as imported_version

        package_version = str(imported_version)
    except Exception:
        package_version = "unknown"
    return f"Hermes-Agent/{package_version}"


def hermes_xai_default_headers() -> Dict[str, str]:
    """Default headers for OpenAI-SDK and raw HTTP clients talking to xAI.

    Replaces the OpenAI Python SDK's identifying ``User-Agent: OpenAI/Python …``
    so chat/completions and Responses traffic is attributed as Hermes Agent,
    matching the direct HTTP integrations (search, TTS, STT, image, video).
    """
    return {"User-Agent": hermes_xai_user_agent()}


async def _load_config_section(section_name: str) -> Dict[str, Any]:
    """Return a top-level Hermes config section as a dict, or empty."""
    try:
        from hermes_cli.config import load_config_readonly

        cfg = await load_config_readonly()
        section = cfg.get(section_name) if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


def _coerce_expires_after(value: Any) -> Optional[int]:
    """Normalize an xAI storage TTL."""
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "default", "none", "null", "never", "permanent", "forever", "0"}:
            return None
        try:
            value = int(normalized)
        except ValueError:
            return SAFE_XAI_STORAGE_EXPIRES_AFTER_SECONDS
    if isinstance(value, (int, float)):
        seconds = int(value)
        if seconds <= 0:
            return None
        return min(seconds, MAX_XAI_STORAGE_EXPIRES_AFTER_SECONDS)
    return SAFE_XAI_STORAGE_EXPIRES_AFTER_SECONDS


async def read_xai_imagine_storage_config(section_name: str) -> Dict[str, Any]:
    """Read xAI Imagine storage settings from image/video provider config."""
    section = await _load_config_section(section_name)
    xai_section = section.get("xai") if isinstance(section, dict) else None
    storage = xai_section.get("storage") if isinstance(xai_section, dict) else None
    storage = storage if isinstance(storage, dict) else {}
    return {
        "enabled": _coerce_bool(storage.get("enabled"), True),
        "public_url": _coerce_bool(storage.get("public_url"), True),
        "expires_after": _coerce_expires_after(storage.get("expires_after")),
    }


async def build_xai_storage_options(
    section_name: str,
    *,
    filename_prefix: str,
    extension: str,
) -> Optional[Dict[str, Any]]:
    """Return an xAI ``storage_options`` payload, or None when disabled."""
    cfg = await read_xai_imagine_storage_config(section_name)
    if not cfg["enabled"]:
        return None

    now = datetime.datetime.now(datetime.UTC)
    ts = now.strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:8]
    ext = extension.lstrip(".") or "bin"
    payload: Dict[str, Any] = {
        "filename": f"{filename_prefix}-{ts}-{short}.{ext}",
        "public_url": bool(cfg["public_url"]),
    }
    if cfg["expires_after"] is not None:
        payload["expires_after"] = cfg["expires_after"]
    return payload


async def xai_storage_notice_text(section_name: str) -> str:
    """Return the user-facing notice for enabled xAI Imagine storage."""
    cfg = await read_xai_imagine_storage_config(section_name)
    if not cfg["enabled"]:
        return ""
    if cfg["expires_after"] is None:
        retention = "without an automatic expiry"
    else:
        days = cfg["expires_after"] / (24 * 60 * 60)
        retention = f"for about {days:g} day{'s' if days != 1 else ''}"
    return (
        "xAI Imagine storage is enabled so generated media gets a reusable "
        f"public URL {retention}. xAI may bill for stored files and public URL "
        f"hosting. Disable this with `{section_name}.xai.storage.enabled: false` "
        "or set `expires_after` to change the retention."
    )


async def maybe_mark_xai_storage_notice_seen(section_name: str) -> Optional[str]:
    """Return the storage notice once per Hermes home, then mark it seen."""
    notice = await xai_storage_notice_text(section_name)
    if not notice:
        return None
    try:
        from hermes_constants import get_hermes_home

        marker_dir = get_hermes_home() / "state"
        await aiofiles.os.makedirs(marker_dir, exist_ok=True)
        marker = marker_dir / f"{section_name}_xai_storage_notice_seen"
        if await aiofiles.os.path.exists(marker):
            return None
        async with aiofiles.open(marker, "w", encoding="utf-8") as marker_file:
            await marker_file.write(datetime.datetime.now(datetime.UTC).isoformat() + "\n")
        return notice
    except Exception:
        return notice


async def resolve_xai_http_credentials(
    *,
    force_refresh: bool = False,
    api_key_hint: Optional[str] = None,
) -> Dict[str, str]:
    """Resolve xAI credentials through coroutine-native pool operations."""
    from agent.secret_scope import (
        UnscopedSecretError,
        get_secret,
        is_multiplex_active,
    )

    if is_multiplex_active():
        # Validate the request scope before the OAuth pool reads or refreshes
        # profile-owned auth state.  The value itself remains a fallback: OAuth
        # keeps its upstream precedence when the active profile has both.
        get_secret("XAI_API_KEY")

    try:
        from agent.credential_pool import load_pool
        import hermes_cli.auth as auth_mod

        pool = await load_pool("xai-oauth")
        entry = (
            await pool.try_refresh_matching(api_key_hint)
            if force_refresh
            else await pool.select()
        )
        if force_refresh and entry is None:
            entry = await pool.select()
        access_token = str(
            getattr(entry, "runtime_api_key", None)
            or getattr(entry, "access_token", "")
        ).strip()
        fallback_base_url = str(
            getattr(entry, "runtime_base_url", None)
            or getattr(entry, "base_url", "")
            or auth_mod.DEFAULT_XAI_OAUTH_BASE_URL
        ).strip().rstrip("/")
        from hermes_cli.config import get_env_value_prefer_dotenv

        override_base_url = str(
            await get_env_value_prefer_dotenv("HERMES_XAI_BASE_URL")
            or await get_env_value_prefer_dotenv("XAI_BASE_URL")
            or ""
        ).strip().rstrip("/")
        base_url = auth_mod._xai_validate_inference_base_url(
            override_base_url,
            fallback=fallback_base_url,
        )
        if access_token:
            return {
                "provider": "xai-oauth",
                "api_key": access_token,
                "base_url": base_url,
            }
    except asyncio.CancelledError:
        raise
    except UnscopedSecretError:
        raise
    except Exception:
        # Match the synchronous resolver's contract: an unavailable OAuth
        # pool falls through to the explicit API-key resolver.
        pass

    from hermes_cli.config import get_env_value_prefer_dotenv

    api_key = str(
        await get_env_value_prefer_dotenv("XAI_API_KEY") or ""
    ).strip()
    base_url = str(
        await get_env_value_prefer_dotenv("XAI_BASE_URL")
        or "https://api.x.ai/v1"
    ).strip().rstrip("/")
    return {
        "provider": "xai",
        "api_key": api_key,
        "base_url": base_url,
    }
