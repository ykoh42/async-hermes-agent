"""Shared helpers for direct xAI HTTP integrations."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


def has_xai_credentials() -> bool:
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
        if not auth_path.exists():
            return False
        store = json.loads(auth_path.read_text(encoding="utf-8"))
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
        from hermes_cli import __version__
    except Exception:
        __version__ = "unknown"
    return f"Hermes-Agent/{__version__}"


def hermes_xai_default_headers() -> Dict[str, str]:
    """Default headers for OpenAI-SDK and raw HTTP clients talking to xAI.

    Replaces the OpenAI Python SDK's identifying ``User-Agent: OpenAI/Python …``
    so chat/completions and Responses traffic is attributed as Hermes Agent,
    matching the direct HTTP integrations (search, TTS, STT, image, video).
    """
    return {"User-Agent": hermes_xai_user_agent()}


async def resolve_xai_http_credentials(
    *,
    force_refresh: bool = False,
    api_key_hint: Optional[str] = None,
) -> Dict[str, str]:
    """Resolve xAI credentials through coroutine-native pool operations."""
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
