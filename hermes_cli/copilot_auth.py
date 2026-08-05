"""GitHub Copilot authentication utilities.

Resolves existing GitHub credentials and exchanges them for Copilot API
tokens without blocking the event loop.

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
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Token type prefixes
_CLASSIC_PAT_PREFIX = "ghp_"

# Env var search order (matches Copilot CLI)
COPILOT_ENV_VARS = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")


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


def _gh_cli_candidates() -> list[str]:
    """Return candidate ``gh`` binary paths, including common Homebrew installs."""
    candidates: list[str] = []

    resolved = shutil.which("gh")
    if resolved:
        candidates.append(resolved)

    for candidate in (
        "/opt/homebrew/bin/gh",
        "/usr/local/bin/gh",
        str(Path.home() / ".local" / "bin" / "gh"),
    ):
        if candidate in candidates:
            continue
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
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

    for gh_path in _gh_cli_candidates():
        cmd = [gh_path, "auth", "token"]
        if hostname:
            cmd += ["--hostname", hostname]
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
        except FileNotFoundError as exc:
            logger.debug("gh CLI token lookup failed (%s): %s", gh_path, exc)
            continue
        token = stdout.decode("utf-8", errors="replace").strip()
        if process.returncode == 0 and token:
            return token
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


def _token_fingerprint(raw_token: str) -> str:
    """Short fingerprint of a raw token for cache keying (avoids storing full token)."""
    import hashlib
    return hashlib.sha256(raw_token.encode()).hexdigest()[:16]


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

    # Check cache first
    cached = _jwt_cache.get(fp)
    if cached:
        api_token, expires_at, base_url = cached
        if time.time() < expires_at - _JWT_REFRESH_MARGIN_SECONDS:
            return api_token, expires_at, base_url

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                _TOKEN_EXCHANGE_URL,
                headers={
                    "Authorization": f"token {raw_token}",
                    "User-Agent": _EXCHANGE_USER_AGENT,
                    "Accept": "application/json",
                    "Editor-Version": _EDITOR_VERSION,
                },
            )
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        raise ValueError(f"Copilot token exchange failed: {exc}") from exc

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
