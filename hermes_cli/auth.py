"""
Multi-provider authentication system for Hermes Agent.

Supports OAuth device code flows (Nous Portal, future: OpenAI Codex) and
traditional API key providers (OpenRouter, custom endpoints). Auth state
is persisted in ~/.hermes/auth.json with cross-process file locking.

Architecture:
- ProviderConfig registry defines known OAuth providers
- Auth store (auth.json) holds per-provider credential state
- resolve_provider() picks the active provider via priority chain
- resolve_*_runtime_credentials() handles token refresh and runtime keys
- logout_command() is the CLI entry point for clearing auth

Nous authentication paths:
- Invoke JWT (preferred): use a scoped access_token directly for inference.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import shlex
import ssl
import stat
import sys
import base64
import hashlib
import subprocess
import threading
import time
import uuid
import webbrowser
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import aiofiles

from hermes_cli.config import (
    get_hermes_home,
    get_config_path,
    read_raw_config,
    require_readable_config_before_write,
)
from hermes_constants import OPENROUTER_BASE_URL, secure_parent_dir
from agent.credential_persistence import sanitize_borrowed_credential_payload
from utils import env_float, is_truthy_value

logger = logging.getLogger(__name__)

try:
    import fcntl
except Exception:
    fcntl = None
try:
    import msvcrt
except Exception:
    msvcrt = None

# =============================================================================
# Constants
# =============================================================================

AUTH_STORE_VERSION = 1
AUTH_LOCK_TIMEOUT_SECONDS = 15.0

# Nous Portal defaults
DEFAULT_NOUS_PORTAL_URL = "https://portal.nousresearch.com"
DEFAULT_NOUS_INFERENCE_URL = "https://inference-api.nousresearch.com/v1"
DEFAULT_NOUS_CLIENT_ID = "hermes-cli"
NOUS_INFERENCE_INVOKE_SCOPE = "inference:invoke"
NOUS_BILLING_MANAGE_SCOPE = "billing:manage"
DEFAULT_NOUS_SCOPE = NOUS_INFERENCE_INVOKE_SCOPE
NOUS_DEVICE_CODE_SOURCE = "device_code"
NOUS_AUTH_PATH_INVOKE_JWT = "invoke_jwt"
ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120       # refresh 2 min before expiry
NOUS_INVOKE_JWT_MIN_TTL_SECONDS = ACCESS_TOKEN_REFRESH_SKEW_SECONDS
DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS = 1     # poll at most every 1s
DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
DEFAULT_XAI_OAUTH_BASE_URL = "https://api.x.ai/v1"
MINIMAX_OAUTH_CLIENT_ID = "78257093-7e40-4613-99e0-527b14b39113"
MINIMAX_OAUTH_SCOPE = "group_id profile model.completion"
MINIMAX_OAUTH_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:user_code"
MINIMAX_OAUTH_GLOBAL_BASE = "https://api.minimax.io"
MINIMAX_OAUTH_CN_BASE = "https://api.minimaxi.com"
MINIMAX_OAUTH_GLOBAL_INFERENCE = "https://api.minimax.io/anthropic"
MINIMAX_OAUTH_CN_INFERENCE = "https://api.minimaxi.com/anthropic"
MINIMAX_OAUTH_REFRESH_SKEW_SECONDS = 60
DEFAULT_QWEN_BASE_URL = "https://portal.qwen.ai/v1"
DEFAULT_GITHUB_MODELS_BASE_URL = "https://api.githubcopilot.com"
DEFAULT_COPILOT_ACP_BASE_URL = "acp://copilot"
DEFAULT_OLLAMA_CLOUD_BASE_URL = "https://ollama.com/v1"
STEPFUN_STEP_PLAN_INTL_BASE_URL = "https://api.stepfun.ai/step_plan/v1"
STEPFUN_STEP_PLAN_CN_BASE_URL = "https://api.stepfun.com/step_plan/v1"
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
try:  # Version tag for the Codex token-endpoint User-Agent; fall back if unavailable.
    from hermes_cli import __version__ as _HERMES_CLI_VERSION
except Exception:  # pragma: no cover - version import should always succeed
    _HERMES_CLI_VERSION = "unknown"
CODEX_OAUTH_USER_AGENT = f"hermes-cli/{_HERMES_CLI_VERSION}"
CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120
XAI_OAUTH_ISSUER = "https://auth.x.ai"
XAI_OAUTH_DISCOVERY_URL = f"{XAI_OAUTH_ISSUER}/.well-known/openid-configuration"
XAI_OAUTH_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
XAI_OAUTH_SCOPE = "openid profile email offline_access grok-cli:access api:access"
XAI_OAUTH_DEVICE_CODE_URL = f"{XAI_OAUTH_ISSUER}/oauth2/device/code"
# xAI/Grok OAuth access tokens are intentionally short-lived (about 6h in
# current SuperGrok flows). A two-minute refresh window is too narrow for
# gateway/cron workloads that may only touch the provider every 30 minutes,
# leaving brief but noisy credential-expiry gaps. Refresh up to one hour
# early so ordinary runtime calls keep the token warm without user reauth.
XAI_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 3600
QWEN_OAUTH_CLIENT_ID = "f0304373b74a44d2b584a3fb70ca9e56"
QWEN_OAUTH_TOKEN_URL = "https://chat.qwen.ai/api/v1/oauth2/token"
QWEN_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120
DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL = "https://accounts.spotify.com"
DEFAULT_SPOTIFY_API_BASE_URL = "https://api.spotify.com/v1"
DEFAULT_SPOTIFY_REDIRECT_URI = "http://127.0.0.1:43827/spotify/callback"
SPOTIFY_DOCS_URL = "https://hermes-agent.nousresearch.com/docs/user-guide/features/spotify"
SPOTIFY_DASHBOARD_URL = "https://developer.spotify.com/dashboard"
SPOTIFY_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120

OAUTH_OVER_SSH_DOCS_URL = "https://hermes-agent.nousresearch.com/docs/guides/oauth-over-ssh"
DEFAULT_SPOTIFY_SCOPE = " ".join((
    "user-modify-playback-state",
    "user-read-playback-state",
    "user-read-currently-playing",
    "user-read-recently-played",
    "playlist-read-private",
    "playlist-read-collaborative",
    "playlist-modify-public",
    "playlist-modify-private",
    "user-library-read",
    "user-library-modify",
))
SERVICE_PROVIDER_NAMES: Dict[str, str] = {
    "spotify": "Spotify",
}

# LM Studio's default no-auth mode still requires *some* non-empty bearer for
# the API-key code paths (auxiliary_client, runtime resolver) to treat the
# provider as configured. This sentinel is sent only to LM Studio, never to
# any remote service.
LMSTUDIO_NOAUTH_PLACEHOLDER = "dummy-lm-api-key"


# =============================================================================
# Provider Registry
# =============================================================================

@dataclass
class ProviderConfig:
    """Describes a known inference provider."""
    id: str
    name: str
    auth_type: str  # "oauth_device_code", "oauth_external", "oauth_minimax", or "api_key"
    portal_base_url: str = ""
    inference_base_url: str = ""
    client_id: str = ""
    scope: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    # For API-key providers: env vars to check (in priority order)
    api_key_env_vars: tuple = ()
    # Optional env var for base URL override
    base_url_env_var: str = ""


PROVIDER_REGISTRY: Dict[str, ProviderConfig] = {
    "nous": ProviderConfig(
        id="nous",
        name="Nous Portal",
        auth_type="oauth_device_code",
        portal_base_url=DEFAULT_NOUS_PORTAL_URL,
        inference_base_url=DEFAULT_NOUS_INFERENCE_URL,
        client_id=DEFAULT_NOUS_CLIENT_ID,
        scope=DEFAULT_NOUS_SCOPE,
    ),
    "openai-codex": ProviderConfig(
        id="openai-codex",
        name="OpenAI Codex",
        auth_type="oauth_external",
        inference_base_url=DEFAULT_CODEX_BASE_URL,
    ),
    "openai-api": ProviderConfig(
        id="openai-api",
        name="OpenAI API",
        auth_type="api_key",
        inference_base_url="https://api.openai.com/v1",
        api_key_env_vars=("OPENAI_API_KEY",),
        base_url_env_var="OPENAI_BASE_URL",
    ),
    "xai-oauth": ProviderConfig(
        id="xai-oauth",
        name="xAI Grok OAuth (SuperGrok / Premium+)",
        auth_type="oauth_external",
        inference_base_url=DEFAULT_XAI_OAUTH_BASE_URL,
    ),
    "qwen-oauth": ProviderConfig(
        id="qwen-oauth",
        name="Qwen OAuth",
        auth_type="oauth_external",
        inference_base_url=DEFAULT_QWEN_BASE_URL,
    ),
    "lmstudio": ProviderConfig(
        id="lmstudio",
        name="LM Studio",
        auth_type="api_key",
        inference_base_url="http://127.0.0.1:1234/v1",
        api_key_env_vars=("LM_API_KEY",),
        base_url_env_var="LM_BASE_URL",
    ),
    "copilot": ProviderConfig(
        id="copilot",
        name="GitHub Copilot",
        auth_type="api_key",
        inference_base_url=DEFAULT_GITHUB_MODELS_BASE_URL,
        api_key_env_vars=("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"),
        base_url_env_var="COPILOT_API_BASE_URL",
    ),
    "copilot-acp": ProviderConfig(
        id="copilot-acp",
        name="GitHub Copilot ACP",
        auth_type="external_process",
        inference_base_url=DEFAULT_COPILOT_ACP_BASE_URL,
        base_url_env_var="COPILOT_ACP_BASE_URL",
    ),
    "gemini": ProviderConfig(
        id="gemini",
        name="Google AI Studio",
        auth_type="api_key",
        inference_base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key_env_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        base_url_env_var="GEMINI_BASE_URL",
    ),
    "zai": ProviderConfig(
        id="zai",
        name="Z.AI / GLM",
        auth_type="api_key",
        inference_base_url="https://api.z.ai/api/paas/v4",
        api_key_env_vars=("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
        base_url_env_var="GLM_BASE_URL",
    ),
    "kimi-coding": ProviderConfig(
        id="kimi-coding",
        name="Kimi / Moonshot",
        auth_type="api_key",
        # Legacy platform.moonshot.ai keys use this endpoint (OpenAI-compat).
        # sk-kimi- (Kimi Code) keys are auto-redirected to api.kimi.com/coding
        # by _resolve_kimi_base_url() below.
        inference_base_url="https://api.moonshot.ai/v1",
        api_key_env_vars=("KIMI_API_KEY", "KIMI_CODING_API_KEY"),
        base_url_env_var="KIMI_BASE_URL",
    ),
    "kimi-coding-cn": ProviderConfig(
        id="kimi-coding-cn",
        name="Kimi / Moonshot (China)",
        auth_type="api_key",
        inference_base_url="https://api.moonshot.cn/v1",
        api_key_env_vars=("KIMI_CN_API_KEY",),
    ),
    "stepfun": ProviderConfig(
        id="stepfun",
        name="StepFun Step Plan",
        auth_type="api_key",
        inference_base_url=STEPFUN_STEP_PLAN_INTL_BASE_URL,
        api_key_env_vars=("STEPFUN_API_KEY",),
        base_url_env_var="STEPFUN_BASE_URL",
    ),
    "arcee": ProviderConfig(
        id="arcee",
        name="Arcee AI",
        auth_type="api_key",
        inference_base_url="https://api.arcee.ai/api/v1",
        api_key_env_vars=("ARCEEAI_API_KEY",),
        base_url_env_var="ARCEE_BASE_URL",
    ),
    "gmi": ProviderConfig(
        id="gmi",
        name="GMI Cloud",
        auth_type="api_key",
        inference_base_url="https://api.gmi-serving.com/v1",
        api_key_env_vars=("GMI_API_KEY",),
        base_url_env_var="GMI_BASE_URL",
    ),
    "minimax": ProviderConfig(
        id="minimax",
        name="MiniMax",
        auth_type="api_key",
        inference_base_url="https://api.minimax.io/anthropic",
        api_key_env_vars=("MINIMAX_API_KEY",),
        base_url_env_var="MINIMAX_BASE_URL",
    ),
    "minimax-oauth": ProviderConfig(
        id="minimax-oauth",
        name="MiniMax (OAuth \u00b7 minimax.io)",
        auth_type="oauth_minimax",
        portal_base_url=MINIMAX_OAUTH_GLOBAL_BASE,
        inference_base_url=MINIMAX_OAUTH_GLOBAL_INFERENCE,
        client_id=MINIMAX_OAUTH_CLIENT_ID,
        scope=MINIMAX_OAUTH_SCOPE,
        extra={"region": "global", "cn_portal_base_url": MINIMAX_OAUTH_CN_BASE,
               "cn_inference_base_url": MINIMAX_OAUTH_CN_INFERENCE},
    ),
    "anthropic": ProviderConfig(
        id="anthropic",
        name="Anthropic",
        auth_type="api_key",
        inference_base_url="https://api.anthropic.com",
        api_key_env_vars=("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"),
        base_url_env_var="ANTHROPIC_BASE_URL",
    ),
    "alibaba": ProviderConfig(
        id="alibaba",
        name="Qwen Cloud",
        auth_type="api_key",
        inference_base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        api_key_env_vars=("DASHSCOPE_API_KEY",),
        base_url_env_var="DASHSCOPE_BASE_URL",
    ),
    "alibaba-coding-plan": ProviderConfig(
        id="alibaba-coding-plan",
        name="Alibaba Cloud (Coding Plan)",
        auth_type="api_key",
        inference_base_url="https://coding-intl.dashscope.aliyuncs.com/v1",
        api_key_env_vars=("ALIBABA_CODING_PLAN_API_KEY", "DASHSCOPE_API_KEY"),
        base_url_env_var="ALIBABA_CODING_PLAN_BASE_URL",
    ),
    "minimax-cn": ProviderConfig(
        id="minimax-cn",
        name="MiniMax (China)",
        auth_type="api_key",
        inference_base_url="https://api.minimaxi.com/anthropic",
        api_key_env_vars=("MINIMAX_CN_API_KEY",),
        base_url_env_var="MINIMAX_CN_BASE_URL",
    ),
    "deepseek": ProviderConfig(
        id="deepseek",
        name="DeepSeek",
        auth_type="api_key",
        inference_base_url="https://api.deepseek.com/v1",
        api_key_env_vars=("DEEPSEEK_API_KEY",),
        base_url_env_var="DEEPSEEK_BASE_URL",
    ),
    "xai": ProviderConfig(
        id="xai",
        name="xAI",
        auth_type="api_key",
        inference_base_url="https://api.x.ai/v1",
        api_key_env_vars=("XAI_API_KEY",),
        base_url_env_var="XAI_BASE_URL",
    ),
    "nvidia": ProviderConfig(
        id="nvidia",
        name="NVIDIA NIM",
        auth_type="api_key",
        inference_base_url="https://integrate.api.nvidia.com/v1",
        api_key_env_vars=("NVIDIA_API_KEY",),
        base_url_env_var="NVIDIA_BASE_URL",
    ),
    "ai-gateway": ProviderConfig(
        id="ai-gateway",
        name="Vercel AI Gateway",
        auth_type="api_key",
        inference_base_url="https://ai-gateway.vercel.sh/v1",
        api_key_env_vars=("AI_GATEWAY_API_KEY",),
        base_url_env_var="AI_GATEWAY_BASE_URL",
    ),
    "opencode-zen": ProviderConfig(
        id="opencode-zen",
        name="OpenCode Zen",
        auth_type="api_key",
        inference_base_url="https://opencode.ai/zen/v1",
        api_key_env_vars=("OPENCODE_ZEN_API_KEY",),
        base_url_env_var="OPENCODE_ZEN_BASE_URL",
    ),
    "opencode-go": ProviderConfig(
        id="opencode-go",
        name="OpenCode Go",
        auth_type="api_key",
        # OpenCode Go mixes API surfaces by model:
        # - GLM / Kimi use OpenAI-compatible chat completions under /v1
        # - MiniMax models use Anthropic Messages under /v1/messages
        # - Qwen 3.7 uses Anthropic Messages under /v1/messages
        # Keep the provider base at /v1 and select api_mode per-model.
        inference_base_url="https://opencode.ai/zen/go/v1",
        api_key_env_vars=("OPENCODE_GO_API_KEY",),
        base_url_env_var="OPENCODE_GO_BASE_URL",
    ),
    "kilocode": ProviderConfig(
        id="kilocode",
        name="Kilo Code",
        auth_type="api_key",
        inference_base_url="https://api.kilo.ai/api/gateway",
        api_key_env_vars=("KILOCODE_API_KEY",),
        base_url_env_var="KILOCODE_BASE_URL",
    ),
    "huggingface": ProviderConfig(
        id="huggingface",
        name="Hugging Face",
        auth_type="api_key",
        inference_base_url="https://router.huggingface.co/v1",
        api_key_env_vars=("HF_TOKEN",),
        base_url_env_var="HF_BASE_URL",
    ),
    "xiaomi": ProviderConfig(
        id="xiaomi",
        name="Xiaomi MiMo",
        auth_type="api_key",
        inference_base_url="https://api.xiaomimimo.com/v1",
        api_key_env_vars=("XIAOMI_API_KEY",),
        base_url_env_var="XIAOMI_BASE_URL",
    ),
    "tencent-tokenhub": ProviderConfig(
        id="tencent-tokenhub",
        name="Tencent TokenHub",
        auth_type="api_key",
        inference_base_url="https://tokenhub.tencentmaas.com/v1",
        api_key_env_vars=("TOKENHUB_API_KEY",),
        base_url_env_var="TOKENHUB_BASE_URL",
    ),
    "ollama-cloud": ProviderConfig(
        id="ollama-cloud",
        name="Ollama Cloud",
        auth_type="api_key",
        inference_base_url=DEFAULT_OLLAMA_CLOUD_BASE_URL,
        api_key_env_vars=("OLLAMA_API_KEY",),
        base_url_env_var="OLLAMA_BASE_URL",
    ),
    "bedrock": ProviderConfig(
        id="bedrock",
        name="AWS Bedrock",
        auth_type="aws_sdk",
        inference_base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        api_key_env_vars=(),
        base_url_env_var="BEDROCK_BASE_URL",
    ),
    "vertex": ProviderConfig(
        id="vertex",
        name="Google Vertex AI",
        auth_type="vertex",
        # No static inference_base_url: Vertex's endpoint is computed per
        # request from project_id + region (agent/vertex_adapter.py's
        # build_vertex_base_url), not a fixed host like the other entries.
        inference_base_url="",
        api_key_env_vars=(),  # OAuth2 (service-account JSON / ADC), not a key
        base_url_env_var="",
    ),
    "azure-foundry": ProviderConfig(
        id="azure-foundry",
        name="Azure Foundry",
        auth_type="api_key",
        inference_base_url="",  # User-provided endpoint
        api_key_env_vars=("AZURE_FOUNDRY_API_KEY",),
        base_url_env_var="AZURE_FOUNDRY_BASE_URL",
    ),
}

# Auto-extend PROVIDER_REGISTRY with any api-key provider registered in
# providers/ that is not already declared above.  New providers only need a
# plugins/model-providers/<name>/ plugin — no edits to this file required.
try:
    from providers import list_providers as _list_providers_for_registry
    for _pp in _list_providers_for_registry():
        if _pp.name in PROVIDER_REGISTRY:
            continue
        if _pp.auth_type != "api_key" or not _pp.env_vars:
            continue
        # Skip providers that need custom token resolution or are special-cased
        # in resolve_provider() (copilot/kimi/zai have bespoke token refresh;
        # openrouter/custom are aggregator/user-supplied and handled outside
        # the registry — adding them here breaks runtime_provider resolution
        # that relies on `openrouter not in PROVIDER_REGISTRY`).
        if _pp.name in {"copilot", "kimi-coding", "kimi-coding-cn", "zai", "openrouter", "custom"}:
            continue
        _api_key_vars = tuple(v for v in _pp.env_vars if not v.endswith("_BASE_URL") and not v.endswith("_URL"))
        _base_url_var = next((v for v in _pp.env_vars if v.endswith("_BASE_URL") or v.endswith("_URL")), None)
        PROVIDER_REGISTRY[_pp.name] = ProviderConfig(
            id=_pp.name,
            name=_pp.display_name or _pp.name,
            auth_type="api_key",
            inference_base_url=_pp.base_url,
            api_key_env_vars=_api_key_vars or _pp.env_vars,
            base_url_env_var=_base_url_var or "",
        )
        # Also register aliases so resolve_provider() resolves them
        for _alias in _pp.aliases:
            if _alias not in PROVIDER_REGISTRY:
                PROVIDER_REGISTRY[_alias] = PROVIDER_REGISTRY[_pp.name]
except Exception:
    pass


# =============================================================================
# Anthropic Key Helper
# =============================================================================

# =============================================================================
# Kimi Code Endpoint Detection
# =============================================================================

# Kimi Code (kimi.com/code) issues keys prefixed "sk-kimi-" that only work
# on api.kimi.com/coding.  Legacy keys from platform.moonshot.ai work on
# api.moonshot.ai/v1 (the old default).  Auto-detect when user hasn't set
# KIMI_BASE_URL explicitly.
#
# Note: the base URL intentionally has NO /v1 suffix.  The /coding endpoint
# speaks the Anthropic Messages protocol, and the anthropic SDK appends
# "/v1/messages" internally — so "/coding" + SDK suffix → "/coding/v1/messages"
# (the correct target). Using "/coding/v1" here would produce
# "/coding/v1/v1/messages" (a 404).
KIMI_CODE_BASE_URL = "https://api.kimi.com/coding"


def _resolve_kimi_base_url(api_key: str, default_url: str, env_override: str) -> str:
    """Return the correct Kimi base URL based on the API key prefix.

    If the user has explicitly set KIMI_BASE_URL, that always wins.
    Otherwise, sk-kimi- prefixed keys route to api.kimi.com/coding/v1.
    """
    if env_override:
        return env_override
    # No key → nothing to infer from.  Return default without inspecting.
    if not api_key:
        return default_url
    if api_key.startswith("sk-kimi-"):
        return KIMI_CODE_BASE_URL
    return default_url



_PLACEHOLDER_SECRET_VALUES = {
    "*",
    "**",
    "***",
    "changeme",
    "your_api_key",
    "your_api_key_here",
    "your-api-key",
    "placeholder",
    "example",
    "dummy",
    "null",
    "none",
}


def has_usable_secret(value: Any, *, min_length: int = 4) -> bool:
    """Return True when a configured secret looks usable, not empty/placeholder."""
    if not isinstance(value, str):
        return False
    cleaned = value.strip()
    if len(cleaned) < min_length:
        return False
    if cleaned.lower() in _PLACEHOLDER_SECRET_VALUES:
        return False
    return True


async def _resolve_api_key_provider_secret(
    provider_id: str,
    pconfig: ProviderConfig,
) -> tuple[str, str]:
    """Resolve API credentials without blocking the running event loop."""
    if provider_id == "copilot":
        from hermes_cli.copilot_auth import resolve_copilot_token

        return await resolve_copilot_token()

    from hermes_cli.config import get_env_value_prefer_dotenv

    for env_var in pconfig.api_key_env_vars:
        value = (
            await get_env_value_prefer_dotenv(env_var) or ""
        ).strip()
        if has_usable_secret(value):
            return value, env_var
    return "", ""


async def resolve_api_key_provider_credentials(
    provider_id: str,
) -> Dict[str, Any]:
    """Async counterpart used by native provider/client resolution."""
    pconfig = PROVIDER_REGISTRY.get(provider_id)
    if not pconfig or pconfig.auth_type != "api_key":
        raise AuthError(
            f"Provider '{provider_id}' is not an API-key provider.",
            provider=provider_id,
            code="invalid_provider",
        )

    api_key, key_source = await _resolve_api_key_provider_secret(provider_id, pconfig)
    copilot_base_url = ""
    if provider_id == "copilot" and api_key:
        from hermes_cli.copilot_auth import get_copilot_api_token

        api_key, advertised_base_url = await get_copilot_api_token(api_key)
        copilot_base_url = str(advertised_base_url or "").strip()
    if not api_key and provider_id == "lmstudio":
        api_key = LMSTUDIO_NOAUTH_PLACEHOLDER
        key_source = key_source or "default"

    env_url = ""
    if pconfig.base_url_env_var:
        env_url = os.getenv(pconfig.base_url_env_var, "").strip()
    if provider_id in {"kimi-coding", "kimi-coding-cn"}:
        base_url = _resolve_kimi_base_url(
            api_key, pconfig.inference_base_url, env_url,
        )
    elif provider_id == "zai":
        base_url = await _resolve_zai_base_url(
            api_key, pconfig.inference_base_url, env_url,
        )
    elif copilot_base_url:
        base_url = copilot_base_url.rstrip("/")
    elif env_url:
        base_url = env_url.rstrip("/")
    else:
        base_url = pconfig.inference_base_url
    if provider_id == "lmstudio":
        base_url = _normalize_lmstudio_runtime_base_url(base_url)
    if not isinstance(base_url, str) or not base_url.strip():
        base_url = pconfig.inference_base_url
    return {
        "provider": provider_id,
        "api_key": api_key,
        "base_url": base_url.rstrip("/"),
        "source": key_source or "default",
    }


# =============================================================================
# Z.AI Endpoint Detection
# =============================================================================

# Z.AI has separate billing for general vs coding plans, and global vs China
# endpoints.  A key that works on one may return "Insufficient balance" on
# another.  We probe at setup time and store the working endpoint.
# Each entry lists candidate models to try in order — newer coding plan accounts
# may only have access to recent models (glm-5.1, glm-5v-turbo) while older
# ones still use glm-4.7.

ZAI_ENDPOINTS = [
    # (id, base_url, probe_models, label)
    ("global",        "https://api.z.ai/api/paas/v4",        ["glm-5"],   "Global"),
    ("cn",            "https://open.bigmodel.cn/api/paas/v4", ["glm-5"],   "China"),
    ("coding-global", "https://api.z.ai/api/coding/paas/v4",  ["glm-5.2", "glm-5.1", "glm-5v-turbo", "glm-4.7"], "Global (Coding Plan)"),
    ("coding-cn",     "https://open.bigmodel.cn/api/coding/paas/v4", ["glm-5.2", "glm-5.1", "glm-5v-turbo", "glm-4.7"], "China (Coding Plan)"),
]


async def detect_zai_endpoint(
    api_key: str,
    timeout: float = 8.0,
) -> Optional[Dict[str, str]]:
    """Probe z.ai endpoints to find one that accepts this API key.

    Returns {"id": ..., "base_url": ..., "model": ..., "label": ...} for the
    first working endpoint, or None if all fail.  For endpoints with multiple
    candidate models, tries each in order and returns the first that succeeds.
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        for ep_id, base_url, probe_models, label in ZAI_ENDPOINTS:
            for model in probe_models:
                try:
                    resp = await client.post(
                        f"{base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model,
                            "stream": False,
                            "max_tokens": 1,
                            "messages": [{"role": "user", "content": "ping"}],
                        },
                    )
                    if resp.status_code == 200:
                        logger.debug(
                            "Z.AI endpoint probe: %s (%s) model=%s OK",
                            ep_id,
                            base_url,
                            model,
                        )
                        return {
                            "id": ep_id,
                            "base_url": base_url,
                            "model": model,
                            "label": label,
                        }
                    logger.debug(
                        "Z.AI endpoint probe: %s model=%s returned %s",
                        ep_id,
                        model,
                        resp.status_code,
                    )
                except Exception as exc:
                    logger.debug(
                        "Z.AI endpoint probe: %s model=%s failed: %s",
                        ep_id,
                        model,
                        exc,
                    )
    return None


async def _resolve_zai_base_url(
    api_key: str,
    default_url: str,
    env_override: str,
) -> str:
    """Return the correct Z.AI base URL by probing endpoints.

    If the user has explicitly set GLM_BASE_URL, that always wins.
    Otherwise, probe the candidate endpoints to find one that accepts the
    key.  The detected endpoint is cached in provider state (auth.json) keyed
    on a hash of the API key so subsequent starts skip the probe.
    """
    if env_override:
        return env_override

    # No API key set → don't probe (would fire N×M HTTPS requests with an
    # empty Bearer token, all returning 401).  This path is hit during
    # auxiliary-client auto-detection when the user has no Z.AI credentials
    # at all — the caller discards the result immediately, so the probe is
    # pure latency for every AIAgent construction.
    if not api_key:
        return default_url

    # Check provider-state cache for a previously-detected endpoint.
    auth_store = await _load_auth_store()
    state = _load_provider_state(auth_store, "zai") or {}
    cached = state.get("detected_endpoint")
    if isinstance(cached, dict) and cached.get("base_url"):
        key_hash = cached.get("key_hash", "")
        if key_hash == hashlib.sha256(api_key.encode()).hexdigest()[:16]:
            logger.debug("Z.AI: using cached endpoint %s", cached["base_url"])
            return cached["base_url"]

    # Probe — may take up to ~8s per endpoint.
    detected = await detect_zai_endpoint(api_key)
    if detected and detected.get("base_url"):
        # Persist the detection result keyed on the API key hash.
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        detected_endpoint = {
            "base_url": detected["base_url"],
            "endpoint_id": detected.get("id", ""),
            "model": detected.get("model", ""),
            "label": detected.get("label", ""),
            "key_hash": key_hash,
        }
        # Persist failure (disk full, permissions, lock timeout) must not
        # break resolution — detection already succeeded; worst case the
        # next start re-probes.
        try:
            async with _auth_store_transaction():
                # Reload auth_store under lock to avoid overwriting concurrent changes
                auth_store = await _load_auth_store()
                state_under_lock = _load_provider_state(auth_store, "zai") or {}
                state_under_lock["detected_endpoint"] = detected_endpoint
                # set_active=False: this runs from credential-pool env seeding
                # (agent/credential_pool.py) for ANY user with a Z.AI key in env,
                # and caching a probe result must not flip their active provider.
                _store_provider_state(auth_store, "zai", state_under_lock, set_active=False)
                await _save_auth_store(auth_store)
        except Exception as exc:
            logger.warning("Z.AI: could not persist detected endpoint (%s); will re-probe next start", exc)
        logger.info("Z.AI: auto-detected endpoint %s (%s)", detected["label"], detected["base_url"])
        return detected["base_url"]

    logger.debug("Z.AI: probe failed, falling back to default %s", default_url)
    return default_url


def _normalize_lmstudio_runtime_base_url(base_url: str) -> str:
    """Return the OpenAI-compatible LM Studio runtime base URL.

    LM Studio's native management API lives under ``/api/v1`` while its
    OpenAI-compatible chat endpoint lives under ``/v1``. Users often paste
    either form into ``LM_BASE_URL`` or ``model.base_url``; normalize before
    the OpenAI SDK appends ``/chat/completions``.
    """
    root = str(base_url or "").strip().rstrip("/")
    for suffix in ("/api/v1", "/api", "/v1"):
        if root.endswith(suffix):
            root = root[: -len(suffix)].rstrip("/")
            break
    return (root or "http://127.0.0.1:1234") + "/v1"


# =============================================================================
# Error Types
# =============================================================================

# Error code marking upstream rate-limit / usage-quota exhaustion (HTTP 429).
# Such failures are transient and re-authenticating cannot resolve them, so
# they must be kept distinct from missing/expired-credential errors.
CODEX_RATE_LIMITED_CODE = "codex_rate_limited"


class AuthError(RuntimeError):
    """Structured auth error with UX mapping hints."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        code: Optional[str] = None,
        relogin_required: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.code = code
        self.relogin_required = relogin_required


def is_rate_limited_auth_error(error: Exception) -> bool:
    """True when an :class:`AuthError` represents upstream rate-limiting / quota
    exhaustion rather than missing or invalid credentials.

    These failures are transient — re-authenticating cannot resolve them — so
    callers should surface a "retry later" notice and prefer a fallback chain
    instead of prompting the operator to run ``hermes auth``.
    """
    # ``auth`` may be reloaded by long-lived hosts while an exception from the
    # previous module instance is still in flight.  Its class is semantically
    # the same ``AuthError`` but no longer passes ``isinstance`` against the
    # newly imported class.  The structured fields are the public contract for
    # this classification, so use them directly.
    return (
        not bool(getattr(error, "relogin_required", True))
        and getattr(error, "code", None) == CODEX_RATE_LIMITED_CODE
    )


def _parse_retry_after_seconds(headers: Any) -> Optional[int]:
    """Best-effort parse of a ``Retry-After`` header into whole seconds.

    Thin wrapper around :func:`agent.retry_utils.parse_retry_after_seconds`
    (delta-seconds and HTTP-date forms; negatives clamp to 0; missing or
    unparseable values return ``None``).
    """
    from agent.retry_utils import parse_retry_after_seconds

    seconds = parse_retry_after_seconds(headers)
    return None if seconds is None else int(seconds)


def format_auth_error(error: Exception) -> str:
    """Map auth failures to concise user-facing guidance."""
    if not isinstance(error, AuthError):
        return str(error)

    # Rate-limit / quota errors are not credential problems — never append the
    # "re-authenticate" remediation, which would mislead the operator.
    if is_rate_limited_auth_error(error):
        return str(error)

    if error.relogin_required:
        return f"{error} Run `hermes model` to re-authenticate."

    if error.code == "subscription_required":
        if error.provider == "nous":
            return _format_nous_entitlement_auth_error(error)
        return "No active paid subscription found. Please purchase/activate a subscription, then retry."

    if error.code == "insufficient_credits":
        if error.provider == "nous":
            return _format_nous_entitlement_auth_error(error)
        return "Subscription credits are exhausted. Top up/renew credits, then retry."

    if error.code in {"subscription_expired", "no_usable_credits", "account_missing"}:
        if error.provider == "nous":
            return _format_nous_entitlement_auth_error(error)

    if error.code == "temporarily_unavailable":
        return f"{error} Please retry in a few seconds."

    return str(error)


def _format_nous_entitlement_auth_error(error: AuthError) -> str:
    return f"{error} Check credits or billing in Nous Portal, then retry."


def _token_fingerprint(token: Any) -> Optional[str]:
    """Return a short hash fingerprint for telemetry without leaking token bytes."""
    if not isinstance(token, str):
        return None
    cleaned = token.strip()
    if not cleaned:
        return None
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:12]


def _oauth_trace_enabled() -> bool:
    raw = os.getenv("HERMES_OAUTH_TRACE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _oauth_trace(event: str, *, sequence_id: Optional[str] = None, **fields: Any) -> None:
    if not _oauth_trace_enabled():
        return
    payload: Dict[str, Any] = {"event": event}
    if sequence_id:
        payload["sequence_id"] = sequence_id
    payload.update(fields)
    logger.info("oauth_trace %s", json.dumps(payload, sort_keys=True, ensure_ascii=False))


# =============================================================================
# Auth Store — persistence layer for ~/.hermes/auth.json
# =============================================================================

def _auth_file_path() -> Path:
    path = get_hermes_home() / "auth.json"
    # Seat belt: if pytest is running and HERMES_HOME resolves to the real
    # user's auth store, refuse rather than silently corrupt it. This catches
    # tests that forgot to monkeypatch HERMES_HOME, tests invoked without the
    # hermetic conftest, or sandbox escapes via threads/subprocesses. In
    # production (no PYTEST_CURRENT_TEST) this is a single dict lookup.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        real_home_auth = (Path.home() / ".hermes" / "auth.json").resolve(strict=False)
        try:
            resolved = path.resolve(strict=False)
        except Exception:
            resolved = path
        if resolved == real_home_auth:
            raise RuntimeError(
                f"Refusing to touch real user auth store during test run: {path}. "
                "Set HERMES_HOME to a tmp_path in your test fixture, or run "
                "via scripts/run_tests.sh for hermetic CI-parity env."
            )
    return path


def _global_auth_file_path() -> Optional[Path]:
    """Return the global-root auth.json when the process is in profile mode.

    Returns ``None`` when the profile and global root resolve to the same
    directory (classic mode, or custom HERMES_HOME that is not a profile).
    Used by read-only fallback paths so providers authed at the root are
    visible to profile processes that haven't configured them locally.

    See issue #18594 follow-up (credential_pool shadowing).
    """
    try:
        from hermes_constants import get_default_hermes_root
        global_root = get_default_hermes_root()
    except Exception:
        return None
    profile_home = get_hermes_home()
    try:
        if profile_home.resolve(strict=False) == global_root.resolve(strict=False):
            return None
    except Exception:
        if profile_home == global_root:
            return None
    # No pytest seat belt here: this is a pure read-only path, and
    # ``_load_global_auth_store()`` wraps the read in a try/except so an
    # unreadable global file can never break the profile process.  The
    # write-side seat belt still lives on ``_auth_file_path()`` where it
    # belongs (that's what protects the real user's auth store from being
    # corrupted by a mis-configured test).
    return global_root / "auth.json"





_auth_target_lock_holders: Dict[str, threading.local] = {}
_auth_target_lock_holders_guard = threading.Lock()
_auth_store_locks: Dict[Tuple[int, str], asyncio.Lock] = {}
_auth_store_locks_guard = threading.Lock()






def _auth_store_lock_for(target_path: Path) -> asyncio.Lock:
    """Return this event loop's lock for one auth-store path.

    The synchronous CLI still owns ``_auth_store_lock``.  The async agent must
    not acquire that blocking lock from its turn loop, so its native path uses
    a task lock plus a non-blocking ``flock`` transaction below.  Locks are
    keyed by event loop as well as path: test suites and embedding hosts often
    create more than one loop in a process.
    """
    loop = asyncio.get_running_loop()
    try:
        path_key = str(target_path.resolve(strict=False))
    except Exception:
        path_key = str(target_path)
    key = (id(loop), path_key)
    with _auth_store_locks_guard:
        return _auth_store_locks.setdefault(key, asyncio.Lock())


@asynccontextmanager
async def _auth_store_transaction(
    target_path: Optional[Path] = None,
    *,
    timeout_seconds: float = AUTH_LOCK_TIMEOUT_SECONDS,
):
    """Awaitable cross-process transaction guard for ``auth.json``.

    ``fcntl.flock(..., LOCK_NB)`` is retried with ``asyncio.sleep`` so another
    process holding the auth-store transaction never stalls the event loop.
    Windows has no compatible non-blocking primitive in the stdlib; the
    process-local asyncio lock still serializes all async-hermes writers there
    and the atomic replace below prevents torn JSON files.
    """
    auth_path = target_path or _auth_file_path()
    task_lock = _auth_store_lock_for(auth_path)
    async with task_lock:
        lock_path = auth_path.with_suffix(".lock")
        await aiofiles.os.makedirs(lock_path.parent, exist_ok=True)
        lock_handle = None
        acquired = False
        try:
            if fcntl is not None:
                lock_handle = await aiofiles.open(lock_path, "a+")
                fd = lock_handle.fileno()
                deadline = time.monotonic() + max(0.0, float(timeout_seconds))
                while True:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired = True
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Timed out waiting for auth store lock")
                        await asyncio.sleep(0.05)
            yield
        finally:
            if lock_handle is not None:
                try:
                    if acquired:
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                finally:
                    await lock_handle.close()












def _load_provider_state(auth_store: Dict[str, Any], provider_id: str) -> Optional[Dict[str, Any]]:
    """Return a provider state from an already-loaded auth snapshot."""
    providers = auth_store.get("providers")
    if not isinstance(providers, dict):
        return None
    state = providers.get(provider_id)
    return dict(state) if isinstance(state, dict) else None


def _save_provider_state(auth_store: Dict[str, Any], provider_id: str, state: Dict[str, Any]) -> None:
    providers = auth_store.setdefault("providers", {})
    if not isinstance(providers, dict):
        auth_store["providers"] = {}
        providers = auth_store["providers"]
    providers[provider_id] = state
    auth_store["active_provider"] = provider_id




def _store_provider_state(
    auth_store: Dict[str, Any],
    provider_id: str,
    state: Dict[str, Any],
    *,
    set_active: bool = True,
) -> None:
    providers = auth_store.setdefault("providers", {})
    if not isinstance(providers, dict):
        auth_store["providers"] = {}
        providers = auth_store["providers"]
    providers[provider_id] = state
    if set_active:
        auth_store["active_provider"] = provider_id






def is_known_auth_provider(provider_id: str) -> bool:
    normalized = (provider_id or "").strip().lower()
    return normalized in PROVIDER_REGISTRY or normalized in SERVICE_PROVIDER_NAMES


def get_auth_provider_display_name(provider_id: str) -> str:
    normalized = (provider_id or "").strip().lower()
    if normalized in PROVIDER_REGISTRY:
        return PROVIDER_REGISTRY[normalized].name
    return SERVICE_PROVIDER_NAMES.get(normalized, provider_id)


async def is_runtime_provider_routable(provider_id: str) -> bool:
    """Return whether runtime resolution recognizes a provider identity.

    This is a capability check, not a credential check. It follows the same
    alias/plugin-aware normalization as ``resolve_provider`` while preserving
    special runtime identities that intentionally live outside the registry.
    """
    normalized = (provider_id or "").strip().lower()
    if not normalized:
        return False
    if normalized in {"auto", "openrouter", "custom", "moa"}:
        return True
    if normalized.startswith("custom:"):
        return True
    try:
        await resolve_provider(normalized)
    except AuthError:
        return False
    return True



async def _load_auth_store(auth_file: Optional[Path] = None) -> Dict[str, Any]:
    """Load the auth store without masking transient filesystem failures."""
    auth_file = auth_file or _auth_file_path()
    if not await aiofiles.os.path.exists(auth_file):
        return {"version": AUTH_STORE_VERSION, "providers": {}}
    try:
        async with aiofiles.open(auth_file, "rb") as handle:
            raw_bytes = await handle.read()
    except OSError:
        logger.warning(
            "auth: could not read %s, leaving the store on disk untouched "
            "rather than degrading to an empty one",
            auth_file,
            exc_info=True,
        )
        raise

    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:
        corrupt_path = auth_file.with_suffix(".json.corrupt")
        preserved = False
        try:
            async with aiofiles.open(corrupt_path, "wb") as handle:
                await handle.write(raw_bytes)
            preserved = True
        except OSError:
            logger.debug(
                "auth: could not preserve a copy of the corrupt store at %s",
                corrupt_path,
                exc_info=True,
            )
        if preserved:
            logger.warning(
                "auth: failed to parse %s (%s), starting with empty store. "
                "Corrupt file preserved at %s",
                auth_file,
                exc,
                corrupt_path,
            )
        else:
            logger.warning(
                "auth: failed to parse %s (%s), starting with empty store. "
                "A copy could NOT be preserved at %s",
                auth_file,
                exc,
                corrupt_path,
            )
        return {"version": AUTH_STORE_VERSION, "providers": {}}

    if isinstance(raw, dict) and (
        isinstance(raw.get("providers"), dict)
        or isinstance(raw.get("credential_pool"), dict)
    ):
        raw.setdefault("providers", {})
        if isinstance(raw.get("providers"), dict):
            _migrate_stale_nous_portal_url(raw["providers"])
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("systems"), dict):
        systems = raw["systems"]
        providers = {}
        if "nous_portal" in systems:
            providers["nous"] = systems["nous_portal"]
        return {
            "version": AUTH_STORE_VERSION,
            "providers": providers,
            "active_provider": "nous" if providers else None,
        }
    return {"version": AUTH_STORE_VERSION, "providers": {}}


async def _load_global_auth_store() -> Dict[str, Any]:
    """Read the profile fallback store without blocking an async turn."""
    global_path = _global_auth_file_path()
    if global_path is None or not await aiofiles.os.path.exists(global_path):
        return {}
    if os.environ.get("PYTEST_CURRENT_TEST"):
        real_home_env = os.environ.get("HOME", "")
        if real_home_env:
            real_root = Path(real_home_env) / ".hermes" / "auth.json"
            try:
                if os.path.abspath(global_path) == os.path.abspath(real_root):
                    return {}
            except Exception:
                pass
    try:
        return await _load_auth_store(global_path)
    except Exception:
        return {}


async def read_credential_pool(provider_id: Optional[str] = None) -> Dict[str, Any]:
    """Awaitably read one credential-pool slice with profile shadowing."""
    auth_store, global_store = await asyncio.gather(
        _load_auth_store(), _load_global_auth_store(),
    )
    pool = auth_store.get("credential_pool")
    if not isinstance(pool, dict):
        pool = {}
    global_pool = global_store.get("credential_pool")
    if not isinstance(global_pool, dict):
        global_pool = {}

    if provider_id is None:
        merged = dict(pool)
        for provider_key, entries in global_pool.items():
            if not isinstance(entries, list) or not entries:
                continue
            current = merged.get(provider_key)
            if not isinstance(current, list) or not current:
                merged[provider_key] = list(entries)
        return merged

    provider_entries = pool.get(provider_id)
    if isinstance(provider_entries, list) and provider_entries:
        return list(provider_entries)
    global_entries = global_pool.get(provider_id)
    return list(global_entries) if isinstance(global_entries, list) else []


_POOL_STATUS_FIELDS = (
    "last_status",
    "last_status_at",
    "last_error_code",
    "last_error_reason",
    "last_error_message",
    "last_error_reset_at",
)


def _merge_disk_cooldown_state(
    entry: Dict[str, Any],
    disk_entry: Optional[Dict[str, Any]],
    provider_id: str,
) -> Dict[str, Any]:
    """Keep a newer on-disk cooldown/quarantine over a stale in-memory one.

    ``write_credential_pool`` callers persist an in-memory snapshot that may
    predate another process marking the same credential exhausted or dead
    (last-writer-wins lost update).  Without this merge, process B's later
    rewrite resurrects a rate-limited key as healthy and both processes
    resume hammering it.  Adopt the on-disk status fields only when they are
    strictly more recent (by ``last_status_at``) AND still binding — a DEAD
    marker, or an EXHAUSTED cooldown that has not yet expired.  Expired
    cooldowns are not resurrected, so the pool's own expiry-clear (which
    resets ``last_status_at`` to None) is never overridden.
    """
    if not isinstance(disk_entry, dict):
        return entry
    try:
        from agent.credential_pool import (
            PooledCredential,
            STATUS_DEAD,
            STATUS_EXHAUSTED,
            _exhausted_until,
            _parse_absolute_timestamp,
        )

        disk_status = disk_entry.get("last_status")
        if disk_status not in (STATUS_DEAD, STATUS_EXHAUSTED):
            return entry
        # A token change means the caller re-authed/refreshed this entry and
        # intentionally cleared its status (e.g. _sync_codex_entry_from_
        # auth_store after a fresh device-code login) — never resurrect the
        # old cooldown onto fresh credentials.
        mem_access = entry.get("access_token") or ""
        disk_access = disk_entry.get("access_token") or ""
        if mem_access and disk_access and mem_access != disk_access:
            return entry
        disk_ts = _parse_absolute_timestamp(disk_entry.get("last_status_at")) or 0.0
        mem_ts = _parse_absolute_timestamp(entry.get("last_status_at")) or 0.0
        if disk_ts <= mem_ts:
            return entry
        if disk_status == STATUS_EXHAUSTED:
            until = _exhausted_until(
                PooledCredential.from_dict(provider_id, disk_entry)
            )
            if until is None or until <= time.time():
                return entry
        merged_entry = dict(entry)
        for status_field in _POOL_STATUS_FIELDS:
            merged_entry[status_field] = disk_entry.get(status_field)
        return merged_entry
    except Exception:  # pragma: no cover - best-effort merge
        return entry



async def _save_auth_store(
    auth_store: Dict[str, Any], target_path: Optional[Path] = None,
) -> Path:
    """Atomically persist ``auth.json`` through ``aiofiles``.

    The temporary file is created with owner-only permissions before any
    credential bytes are written.  ``os.replace`` is intentionally the final
    tiny synchronous syscall: it is atomic and cannot wait on filesystem I/O.
    """
    auth_file = target_path or _auth_file_path()
    await aiofiles.os.makedirs(auth_file.parent, exist_ok=True)
    secure_parent_dir(auth_file)
    auth_store["version"] = AUTH_STORE_VERSION
    auth_store["updated_at"] = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(auth_store, indent=2) + "\n"
    tmp_path = auth_file.with_name(f"{auth_file.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    def secure_opener(file: str, flags: int) -> int:
        return os.open(
            file,
            flags | os.O_EXCL,
            stat.S_IRUSR | stat.S_IWUSR,
        )

    try:
        async with aiofiles.open(
            tmp_path,
            "w",
            encoding="utf-8",
            opener=secure_opener,
        ) as handle:
            await handle.write(payload)
            await handle.flush()
        await aiofiles.os.replace(tmp_path, auth_file)
        return auth_file
    finally:
        try:
            await aiofiles.os.remove(tmp_path)
        except FileNotFoundError:
            pass


async def write_credential_pool(
    provider_id: str,
    entries: List[Dict[str, Any]],
    *,
    removed_ids: Optional[Iterable[str]] = None,
) -> Path:
    """Awaitably merge and persist one provider's pool slice.

    The merge is identical to ``write_credential_pool``: concurrent entries
    are retained and a newer on-disk cooldown wins over stale in-memory state.
    The transaction itself is non-blocking for the event loop.
    """
    removed = {rid for rid in (removed_ids or ()) if rid}
    async with _auth_store_transaction():
        auth_store = await _load_auth_store()
        pool = auth_store.get("credential_pool")
        if not isinstance(pool, dict):
            pool = {}
            auth_store["credential_pool"] = pool
        sanitized_entries = [
            sanitize_borrowed_credential_payload(entry, provider_id)
            if isinstance(entry, dict) else entry
            for entry in entries
        ]
        existing = pool.get(provider_id)
        existing_list = existing if isinstance(existing, list) else []
        existing_by_id = {
            entry.get("id"): entry
            for entry in existing_list
            if isinstance(entry, dict) and entry.get("id")
        }
        new_ids = {
            entry.get("id")
            for entry in sanitized_entries
            if isinstance(entry, dict) and entry.get("id")
        }
        merged: List[Dict[str, Any]] = [
            _merge_disk_cooldown_state(
                entry, existing_by_id.get(entry.get("id")), provider_id
            )
            if isinstance(entry, dict)
            else entry
            for entry in sanitized_entries
        ]
        for disk_entry in existing_list:
            if not isinstance(disk_entry, dict):
                continue
            disk_id = disk_entry.get("id")
            if not disk_id or disk_id in new_ids or disk_id in removed:
                continue
            merged.append(sanitize_borrowed_credential_payload(disk_entry, provider_id))
        pool[provider_id] = merged
        return await _save_auth_store(auth_store)


















# =============================================================================
# Provider Resolution — picks which provider to use
# =============================================================================




async def resolve_provider(
    requested: Optional[str] = None,
    *,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
) -> str:
    """
    Determine which inference provider to use.

    Priority (when requested="auto" or None) — explicit user intent wins over a
    stale logged-in OAuth provider (#29285):
    1. Explicit CLI api_key/base_url -> "openrouter"
    2. config.yaml `model.provider`
    3. OPENAI_API_KEY / OPENROUTER_API_KEY env vars -> "openrouter"
    4. OpenRouter credential pool
    5. Provider-specific API keys (GLM, Kimi, MiniMax, ...) -> that provider
    6. auth.json `active_provider` (logged-in OAuth) — last-resort fallback
    7. AWS Bedrock credential chain
    8. Error (no provider configured)
    """
    normalized = (requested or "auto").strip().lower()

    # Normalize provider aliases
    _PROVIDER_ALIASES = {
        "glm": "zai", "z-ai": "zai", "z.ai": "zai", "zhipu": "zai",
        "google": "gemini", "google-gemini": "gemini", "google-ai-studio": "gemini",
        "x-ai": "xai", "x.ai": "xai", "grok": "xai",
        "xai-oauth": "xai-oauth", "x-ai-oauth": "xai-oauth",
        "grok-oauth": "xai-oauth", "xai-grok-oauth": "xai-oauth",
        "kimi": "kimi-coding", "kimi-for-coding": "kimi-coding", "moonshot": "kimi-coding",
        "kimi-cn": "kimi-coding-cn", "moonshot-cn": "kimi-coding-cn",
        "step": "stepfun", "stepfun-coding-plan": "stepfun",
        "arcee-ai": "arcee", "arceeai": "arcee",
        "gmi-cloud": "gmi", "gmicloud": "gmi",
        "minimax-china": "minimax-cn", "minimax_cn": "minimax-cn",
        "minimax-portal": "minimax-oauth", "minimax-global": "minimax-oauth", "minimax_oauth": "minimax-oauth",
        "alibaba_coding": "alibaba-coding-plan", "alibaba-coding": "alibaba-coding-plan",
        "alibaba_coding_plan": "alibaba-coding-plan",
        "claude": "anthropic", "claude-code": "anthropic",
        "github": "copilot", "github-copilot": "copilot",
        "github-models": "copilot", "github-model": "copilot",
        "github-copilot-acp": "copilot-acp", "copilot-acp-agent": "copilot-acp",
        "aigateway": "ai-gateway", "vercel": "ai-gateway", "vercel-ai-gateway": "ai-gateway",
        "opencode": "opencode-zen", "zen": "opencode-zen",
        "qwen-portal": "qwen-oauth", "qwen-cli": "qwen-oauth", "qwen-oauth": "qwen-oauth",
        "hf": "huggingface", "hugging-face": "huggingface", "huggingface-hub": "huggingface",
        "mimo": "xiaomi", "xiaomi-mimo": "xiaomi",
        "tencent": "tencent-tokenhub", "tokenhub": "tencent-tokenhub",
        "tencent-cloud": "tencent-tokenhub", "tencentmaas": "tencent-tokenhub",
        "aws": "bedrock", "aws-bedrock": "bedrock", "amazon-bedrock": "bedrock", "amazon": "bedrock",
        "go": "opencode-go", "opencode-go-sub": "opencode-go",
        "kilo": "kilocode", "kilo-code": "kilocode", "kilo-gateway": "kilocode",
        "lmstudio": "lmstudio", "lm-studio": "lmstudio", "lm_studio": "lmstudio",
        # Local server aliases — route through the generic custom provider
        "ollama": "custom", "ollama_cloud": "ollama-cloud",
        "vllm": "custom", "llamacpp": "custom",
        "llama.cpp": "custom", "llama-cpp": "custom",
    }
    # Extend with aliases declared in plugins/model-providers/<name>/ that aren't already mapped.
    # This keeps providers/ as the single source for new aliases while the
    # hardcoded dict above remains authoritative for existing ones.
    try:
        from providers import list_providers as _lp
        for _pp in _lp():
            for _alias in _pp.aliases:
                if _alias not in _PROVIDER_ALIASES:
                    _PROVIDER_ALIASES[_alias] = _pp.name
    except Exception:
        pass
    normalized = _PROVIDER_ALIASES.get(normalized, normalized)

    if normalized == "openrouter":
        return "openrouter"
    if normalized == "custom":
        return "custom"
    if normalized in PROVIDER_REGISTRY:
        return normalized
    if normalized != "auto":
        raise AuthError(
            f"Unknown provider '{normalized}'.",
            code="invalid_provider",
        )

    # Explicit one-off CLI creds always mean openrouter/custom
    if explicit_api_key or explicit_base_url:
        return "openrouter"

    # Provider precedence for the auto-path (#29285): explicit user intent must
    # win over a stale logged-in OAuth `active_provider`. Order matches the
    # docstring: 1. explicit CLI creds  2. config.yaml `model.provider`
    # 3. OPENAI/OPENROUTER env keys  4. OpenRouter pool  5. provider-specific
    # env keys  6. auth.json `active_provider` (OAuth)  7. Bedrock  8. error.
    # The normal chat/gateway path resolves config.provider upstream in
    # resolve_requested_provider() before ever reaching "auto"; this duplicate
    # check is the safety net for the lone direct caller (main.py resolve_provider
    # ("auto")) and any future bypass of that stage.
    _model_cfg: Any = None
    try:
        from hermes_cli.config import load_config_readonly

        _model_cfg = (await load_config_readonly() or {}).get("model")
        if isinstance(_model_cfg, dict):
            _cfg_provider = _model_cfg.get("provider")
            if isinstance(_cfg_provider, str) and _cfg_provider.strip().lower() in PROVIDER_REGISTRY:
                return _cfg_provider.strip().lower()
    except Exception as e:
        logger.debug("Could not read config.yaml model.provider for auto-resolution: %s", e)

    if has_usable_secret(os.getenv("OPENAI_API_KEY")) or has_usable_secret(os.getenv("OPENROUTER_API_KEY")):
        return "openrouter"

    # Credential-pool discovery belongs to the native-async runtime resolver.
    # This synchronous config/env classifier must not instantiate the async
    # pool coroutine and silently discard it; callers that need pool-only
    # credentials resolve them through ``await load_pool("openrouter")``.

    # Determine the logged-in OAuth provider up front so the env-key loop below
    # can WARN when an exported API key preempts it (#29285 transparency). The
    # actual OAuth fallback (tier 6) still happens later if nothing else matches.
    _oauth_active: Optional[str] = None
    try:
        _store = await _load_auth_store()
        _maybe = _store.get("active_provider")
        providers = _store.get("providers")
        pool = _store.get("credential_pool")
        provider_state = providers.get(_maybe) if isinstance(providers, dict) else None
        pool_entries = pool.get(_maybe) if isinstance(pool, dict) else None
        if _maybe in PROVIDER_REGISTRY and (
            isinstance(provider_state, dict)
            or (isinstance(pool_entries, list) and bool(pool_entries))
        ):
            _oauth_active = _maybe
    except Exception as e:
        logger.debug("Could not pre-read active auth provider: %s", e)

    # Auto-detect API-key providers by checking their env vars
    for pid, pconfig in PROVIDER_REGISTRY.items():
        if pconfig.auth_type != "api_key":
            continue
        # GitHub tokens are commonly present for repo/tool access but should not
        # hijack inference auto-selection unless the user explicitly chooses
        # Copilot/GitHub Models as the provider. LM Studio is a local server
        # whose availability isn't implied by LM_API_KEY presence (it may be
        # offline, and the no-auth setup uses a placeholder value), so it
        # also requires explicit selection.
        if pid in {"copilot", "lmstudio"}:
            continue
        for env_var in pconfig.api_key_env_vars:
            if has_usable_secret(os.getenv(env_var, "")):
                # An exported API key now wins over a logged-in OAuth provider
                # (the #29285 fix). Surface that so a user who deliberately uses
                # OAuth but has a stale key in ~/.hermes/.env isn't silently
                # switched without knowing why.
                if _oauth_active and _oauth_active != pid:
                    logger.warning(
                        "Provider resolved to %r via %s, preempting your "
                        "logged-in OAuth provider %r. If you meant to use the "
                        "OAuth login, unset %s or set `model.provider` "
                        "explicitly.",
                        pid, env_var, _oauth_active, env_var,
                    )
                return pid

    # Logged-in OAuth provider (auth.json `active_provider`) — a LAST-RESORT
    # fallback, chosen only when the user expressed no other preference above.
    # Previously this sat ABOVE the env-var/config checks, so a stale OAuth
    # login silently overrode an explicit `model.provider` or an exported API
    # key (#29285). Demoted here so explicit intent always wins.
    if _oauth_active:
        # Surface the silent-override case the issue reported: a populated
        # `model` config that lacks a `provider` key falls through to OAuth.
        if isinstance(_model_cfg, dict) and _model_cfg and not _model_cfg.get("provider"):
            logger.warning(
                "Provider resolved to logged-in OAuth provider %r because "
                "config.yaml `model` has no `provider` key. If you meant a "
                "different provider, set `model.provider` explicitly.",
                _oauth_active,
            )
        return _oauth_active

    raise AuthError(
        "No inference provider configured. Run 'hermes model' to choose a "
        "provider and model, or set an API key (OPENROUTER_API_KEY, "
        "OPENAI_API_KEY, etc.) in ~/.hermes/.env.",
        code="no_provider_configured",
    )


# =============================================================================
# Timestamp / TTL helpers
# =============================================================================

def _parse_iso_timestamp(value: Any) -> Optional[float]:
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _is_expiring(expires_at_iso: Any, skew_seconds: int) -> bool:
    expires_epoch = _parse_iso_timestamp(expires_at_iso)
    if expires_epoch is None:
        return True
    return expires_epoch <= (time.time() + skew_seconds)


def _coerce_ttl_seconds(expires_in: Any) -> int:
    try:
        ttl = int(expires_in)
    except Exception:
        ttl = 0
    return max(0, ttl)


def _optional_base_url(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().rstrip("/")
    return cleaned if cleaned else None


_NOUS_STALE_PORTAL_HOSTS: FrozenSet[str] = frozenset({
    "api.nousresearch.com",
})

# Allowlist of valid Nous Portal hosts. A portal_base_url outside this
# set is treated as a misconfiguration and falls back to the default.
# "localhost" / "127.0.0.1" are valid for local development and testing.
_NOUS_PORTAL_ALLOWED_HOSTS: FrozenSet[str] = frozenset({
    "portal.nousresearch.com",
    "localhost",
    "127.0.0.1",
})


def _migrate_stale_nous_portal_url(providers: Dict[str, Any]) -> None:
    nous = providers.get("nous")
    if not isinstance(nous, dict):
        return
    stored = (nous.get("portal_base_url") or "").strip()
    if stored:
        parsed = urlparse(stored)
        if parsed.hostname in _NOUS_STALE_PORTAL_HOSTS:
            logger.warning(
                "auth: migrating stale nous portal_base_url %s -> %s",
                stored, DEFAULT_NOUS_PORTAL_URL,
            )
            nous["portal_base_url"] = DEFAULT_NOUS_PORTAL_URL


# Allowlist of hosts the Nous Portal proxy is willing to forward inference
# JWTs to. Sending a bearer anywhere else would leak it.
#
# This is consulted only for URLs coming from the NETWORK side (Portal
# refresh responses). User-controlled env-var overrides
# (NOUS_INFERENCE_BASE_URL) bypass validation — that's the documented
# dev/staging escape hatch and the env source is already trusted (the
# user set it themselves).
_ALLOWED_NOUS_INFERENCE_HOSTS: FrozenSet[str] = frozenset({
    "inference-api.nousresearch.com",
})




def _nous_inference_env_override() -> Optional[str]:
    """Return the user-set ``NOUS_INFERENCE_BASE_URL`` override, if any.

    This is the documented dev/staging escape hatch. The env source is
    trusted (the OS user set it themselves), so it is intentionally NOT
    gated by the network host allowlist — unlike Portal-returned URLs.

    Returns a trailing-slash-stripped non-empty string, or ``None`` when
    the env var is unset/blank.
    """
    return _optional_base_url(os.getenv("NOUS_INFERENCE_BASE_URL"))




def _decode_jwt_claims(token: Any) -> Dict[str, Any]:
    if not isinstance(token, str) or token.count(".") != 2:
        return {}
    payload = token.split(".")[1]
    payload += "=" * ((4 - len(payload) % 4) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload.encode("utf-8"))
        claims = json.loads(raw.decode("utf-8"))
    except Exception:
        return {}
    return claims if isinstance(claims, dict) else {}


def _scope_values(raw_scope: Any) -> set[str]:
    # OAuth token responses normally return a space-separated string. Keep
    # collection support for JWT ``scp`` claims and older stored test fixtures.
    scopes: set[str] = set()
    if isinstance(raw_scope, str):
        for part in raw_scope.replace(",", " ").split():
            cleaned = part.strip()
            if cleaned:
                scopes.add(cleaned)
    elif isinstance(raw_scope, (list, tuple, set, frozenset)):
        for item in raw_scope:
            if isinstance(item, str):
                scopes.update(_scope_values(item))
    return scopes


def _nous_invoke_jwt_status(
    token: Any,
    *,
    scope: Any = None,
    expires_at: Any = None,
    min_ttl_seconds: int = NOUS_INVOKE_JWT_MIN_TTL_SECONDS,
) -> Optional[str]:
    """Return None when the token can be used for inference, else a reason."""
    claims = _decode_jwt_claims(token)
    if not claims:
        return "access_token_not_jwt"
    scopes = (
        _scope_values(scope)
        | _scope_values(claims.get("scope"))
        | _scope_values(claims.get("scp"))
    )
    if NOUS_INFERENCE_INVOKE_SCOPE not in scopes:
        return "missing_inference_invoke_scope"
    exp = claims.get("exp")
    skew = max(0, int(min_ttl_seconds))
    if isinstance(exp, (int, float)):
        if float(exp) <= (time.time() + skew):
            return "invoke_jwt_expiring"
        return None
    if _is_expiring(expires_at, skew):
        return "invoke_jwt_expiry_unknown_or_expiring"
    return None


def _nous_invoke_jwt_is_usable(
    token: Any,
    *,
    scope: Any = None,
    expires_at: Any = None,
    min_ttl_seconds: int = NOUS_INVOKE_JWT_MIN_TTL_SECONDS,
) -> bool:
    return (
        _nous_invoke_jwt_status(
            token,
            scope=scope,
            expires_at=expires_at,
            min_ttl_seconds=min_ttl_seconds,
        )
        is None
    )




def _log_nous_invoke_jwt_selected(
    *,
    access_token: Any,
    sequence_id: Optional[str] = None,
) -> None:
    logger.debug("Nous inference auth: using NAS invoke JWT")
    _oauth_trace(
        "nous_invoke_jwt_selected",
        sequence_id=sequence_id,
        access_token_fp=_token_fingerprint(access_token),
    )


def _nous_jwt_expires_at(token: Any, fallback_expires_at: Any = None) -> Optional[str]:
    claims = _decode_jwt_claims(token)
    exp = claims.get("exp")
    if isinstance(exp, (int, float)):
        try:
            return datetime.fromtimestamp(float(exp), tz=timezone.utc).isoformat()
        except Exception:
            pass
    return fallback_expires_at if isinstance(fallback_expires_at, str) else None


def _set_nous_agent_key_from_invoke_jwt(
    state: Dict[str, Any],
    *,
    obtained_at: Optional[str] = None,
) -> None:
    access_token = state.get("access_token")
    if not isinstance(access_token, str) or not access_token.strip():
        return
    now = datetime.now(timezone.utc)
    existing_obtained_at = state.get("agent_key_obtained_at")
    if obtained_at:
        effective_obtained_at = obtained_at
    elif (
        state.get("agent_key") == access_token
        and isinstance(existing_obtained_at, str)
        and existing_obtained_at.strip()
    ):
        effective_obtained_at = existing_obtained_at
    else:
        effective_obtained_at = now.isoformat()
    expires_at = _nous_jwt_expires_at(access_token, state.get("expires_at"))
    expires_epoch = _parse_iso_timestamp(expires_at)
    expires_in = (
        max(0, int(expires_epoch - time.time()))
        if expires_epoch is not None
        else _coerce_ttl_seconds(state.get("expires_in"))
    )
    if expires_at:
        state["expires_at"] = expires_at
        state["expires_in"] = expires_in
    state["agent_key"] = access_token
    state["agent_key_id"] = None
    state["agent_key_expires_at"] = expires_at
    state["agent_key_expires_in"] = expires_in
    state["agent_key_reused"] = False
    state["agent_key_obtained_at"] = effective_obtained_at


def _select_nous_invoke_jwt(
    state: Dict[str, Any],
    *,
    access_token: Any = None,
    sequence_id: Optional[str] = None,
) -> None:
    if isinstance(access_token, str) and access_token.strip():
        state["access_token"] = access_token
    _set_nous_agent_key_from_invoke_jwt(state)
    _log_nous_invoke_jwt_selected(
        access_token=state.get("access_token"),
        sequence_id=sequence_id,
    )


_NOUS_EFFECTIVE_STATE_IGNORED_KEYS = frozenset({
    # These are derived from expires_at/JWT exp and naturally tick down between
    # reads. Persisting only these changes makes auth.json noisy and defeats
    # the mtime-keyed auth-status cache.
    "expires_in",
    "agent_key_expires_in",
})




def _codex_access_token_is_expiring(access_token: Any, skew_seconds: int) -> bool:
    claims = _decode_jwt_claims(access_token)
    exp = claims.get("exp")
    if not isinstance(exp, (int, float)):
        return False
    return float(exp) <= (time.time() + max(0, int(skew_seconds)))


















# =============================================================================
# Spotify auth — PKCE tokens stored in ~/.hermes/auth.json
# =============================================================================











































# =============================================================================
# SSH / remote session detection
# =============================================================================



# Console/text-mode browsers that ``webbrowser`` will happily launch INSIDE
# the terminal.  Opening one of these is worse than not opening anything —
# it hijacks the user's TTY with an unusable text browser (the xAI OAuth
# "Account Management" page rendered in w3m, reported May 2026) instead of
# letting them copy the URL to a real browser.  When the resolved browser is
# one of these we refuse to auto-open and fall back to the print-the-URL
# path, same as a remote session.
_CONSOLE_BROWSER_NAMES: FrozenSet[str] = frozenset(
    {
        "w3m",
        "lynx",
        "links",
        "links2",
        "elinks",
        "www-browser",
        "browsh",  # TUI browser — still hijacks the terminal
    }
)








# =============================================================================
# OpenAI Codex auth — tokens stored in ~/.hermes/auth.json (not ~/.codex/)
#
# Hermes maintains its own Codex OAuth session separate from the Codex CLI
# and VS Code extension. This prevents refresh token rotation conflicts
# where one app's refresh invalidates the other's session.
# =============================================================================



















# Throttle for the live Codex quota probe below.  The probe runs on the hot
# credential-selection path while the pool is exhausted, so without a floor a
# busy gateway would hammer the usage endpoint on every model/auxiliary call.
CODEX_QUOTA_PROBE_MIN_INTERVAL_SECONDS = 300  # 5 minutes
_codex_quota_probe_cache: Dict[str, Tuple[float, Optional[bool]]] = {}
_codex_quota_probe_lock = threading.Lock()












# =============================================================================
# xAI Grok OAuth — tokens stored in ~/.hermes/auth.json
# =============================================================================













def _xai_access_token_is_expiring(access_token: str, skew_seconds: int = 0) -> bool:
    if not isinstance(access_token, str) or "." not in access_token:
        return False
    try:
        parts = access_token.split(".")
        if len(parts) < 2:
            return False
        payload_b64 = parts[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64.encode("ascii")).decode("utf-8"))
        exp = payload.get("exp")
        if not isinstance(exp, (int, float)):
            return False
        return float(exp) <= (time.time() + max(0, int(skew_seconds)))
    except Exception:
        return False


def _xai_proactive_refresh_skew_seconds(access_token: str) -> int:
    """How far before JWT ``exp`` to proactively refresh xAI OAuth tokens.

    SuperGrok sessions can still ship multi-hour access tokens, where the
    gateway-oriented :data:`XAI_ACCESS_TOKEN_REFRESH_SKEW_SECONDS` window
    makes sense. Device-code logins often return ~15-minute JWTs; applying
    the full hour-long skew to those forces a refresh on *every* credential
    resolution (chat turn, Imagine tool call, ``hermes auth status``, …),
    which burns single-use refresh tokens and races concurrent callers into
    ``invalid_grant`` quarantine.
    """
    max_skew = XAI_ACCESS_TOKEN_REFRESH_SKEW_SECONDS
    if not isinstance(access_token, str) or "." not in access_token:
        return max_skew
    try:
        parts = access_token.split(".")
        if len(parts) < 2:
            return max_skew
        payload_b64 = parts[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64.encode("ascii")).decode("utf-8"))
        exp = payload.get("exp")
        if not isinstance(exp, (int, float)):
            return max_skew
        remaining = float(exp) - time.time()
        if remaining <= 0:
            return max_skew
        if remaining <= 45 * 60:
            return min(120, max_skew)
        return max_skew
    except Exception:
        return max_skew


def _xai_validate_oauth_endpoint(url: str, *, field: str) -> str:
    """Refuse any OIDC discovery endpoint that isn't HTTPS on the xAI origin.

    The OIDC discovery response is a long-lived, low-frequency request whose
    output is cached in ``~/.hermes/auth.json``. A single MITM during initial
    login could substitute a malicious ``token_endpoint``; that URL would
    then receive the refresh_token on every subsequent refresh — a permanent
    credential leak from a one-time MITM. Validating scheme + host pins the
    cached endpoint to the xAI auth origin (or a future ``*.x.ai`` subdomain
    if xAI migrates) so the cache poisoning loses its persistence guarantee.

    RFC 8414 §2 requires the issuer to be ``https://`` and SHOULD-keeps the
    token_endpoint on the same origin; we enforce both. ``x.ai`` is the
    bare apex, so we accept either exact host match or any ``.x.ai`` suffix.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise AuthError(
            f"xAI OIDC discovery returned a non-HTTPS {field}: {url!r}.",
            provider="xai-oauth",
            code="xai_discovery_invalid",
        )
    host = (parsed.hostname or "").lower()
    if not host:
        raise AuthError(
            f"xAI OIDC discovery {field} is missing a hostname: {url!r}.",
            provider="xai-oauth",
            code="xai_discovery_invalid",
        )
    if host != "x.ai" and not host.endswith(".x.ai"):
        raise AuthError(
            f"xAI OIDC discovery {field} host {host!r} is not on the xAI origin "
            f"(expected x.ai or a *.x.ai subdomain). Refusing to use a cached "
            f"endpoint that may have been substituted by a MITM during initial "
            f"discovery; re-authenticate with `hermes model` to re-fetch.",
            provider="xai-oauth",
            code="xai_discovery_invalid",
        )
    return url


def _xai_validate_inference_base_url(value: str, *, fallback: str) -> str:
    """Refuse a non-xAI base_url for the OAuth-authenticated inference path.

    The xAI Grok OAuth bearer is a high-value, long-lived credential tied to
    the user's SuperGrok subscription. ``XAI_BASE_URL`` / ``HERMES_XAI_BASE_URL``
    let users repoint the inference endpoint (handy for staging or a local
    proxy), but the env override is also a credential-leak vector: a tampered
    ``.env`` or hostile shell init that sets
    ``XAI_BASE_URL=https://attacker.example/v1`` would ship the OAuth access
    token to a third party on every request, silently.

    Pin the inference origin to ``api.x.ai`` (or any ``*.x.ai`` subdomain xAI
    may add). On rejection, fall back to the default and log a warning rather
    than raise — a bad env var should not deadlock authentication, but it
    should also never leak the bearer.

    ``value`` is the already-stripped, trailing-slash-trimmed candidate from
    env. Empty input returns ``fallback`` unchanged.
    """
    candidate = (value or "").strip().rstrip("/")
    if not candidate:
        return fallback
    try:
        parsed = urlparse(candidate)
    except Exception:
        logger.warning(
            "Ignoring malformed xAI base_url override %r; using %s instead.",
            candidate, fallback,
        )
        return fallback
    if parsed.scheme != "https":
        logger.warning(
            "Refusing non-HTTPS xAI base_url override %r (xai-oauth bearer would "
            "be sent in cleartext); falling back to %s.",
            candidate, fallback,
        )
        return fallback
    host = (parsed.hostname or "").lower()
    if not host:
        logger.warning(
            "Ignoring xAI base_url override %r with no hostname; using %s instead.",
            candidate, fallback,
        )
        return fallback
    if host != "x.ai" and not host.endswith(".x.ai"):
        logger.warning(
            "Refusing xAI base_url override %r — host %r is not on the xAI origin "
            "(expected x.ai or a *.x.ai subdomain). The xai-oauth bearer is only "
            "valid against xAI's inference API; sending it elsewhere would leak "
            "the credential. Falling back to %s.",
            candidate, host, fallback,
        )
        return fallback
    return candidate










# =============================================================================
# TLS verification helper
# =============================================================================





# =============================================================================
# OAuth Device Code Flow — generic, parameterized by provider
# =============================================================================





# =============================================================================
# Nous Portal — token refresh and model discovery
# =============================================================================

# -----------------------------------------------------------------------------
# Shared Nous token store — lets OAuth credentials persist across profiles
# so a new `hermes --profile <name> auth add nous --type oauth` can one-tap
# import instead of running the full device-code flow every time.
#
# File lives at ${HERMES_SHARED_AUTH_DIR}/nous_auth.json, defaulting to
# ``<hermes-root>/shared/nous_auth.json`` where ``<hermes-root>`` is what
# ``get_default_hermes_root()`` returns — ``~/.hermes`` on Linux/macOS,
# ``%LOCALAPPDATA%\hermes`` on native Windows, or the Docker/custom root.
# It is OUTSIDE any named profile's HERMES_HOME so named profiles (which
# typically live under ``<hermes-root>/profiles/<name>/``) all see the
# same file.
#
# Written on successful login and on every runtime refresh so the stored
# refresh_token stays current even if one profile refreshes and rotates it.
# If ever the stored refresh_token does go stale server-side, import fails
# gracefully and the user falls back to the normal device-code flow.
# -----------------------------------------------------------------------------

NOUS_SHARED_STORE_FILENAME = "nous_auth.json"
_nous_shared_lock_holder = threading.local()
































def _agent_key_is_usable(state: Dict[str, Any], min_ttl_seconds: int) -> bool:
    key = state.get("agent_key")
    if not isinstance(key, str) or not key.strip():
        return False
    return _nous_invoke_jwt_is_usable(
        key,
        scope=state.get("scope"),
        expires_at=state.get("agent_key_expires_at"),
        min_ttl_seconds=max(0, int(min_ttl_seconds)),
    )












# =============================================================================
# Status helpers
# =============================================================================





# ── Process-level memo for get_nous_auth_status() ──
# get_nous_auth_status() validates state by calling resolve_nous_runtime_credentials(),
# which does a synchronous OAuth refresh POST to portal.nousresearch.com. That can take
# ~350ms even on the failure path, and read-only UI surfaces (`hermes tools`, status panels,
# subscription-feature checks) call it many times per render — `hermes tools` → "All Platforms"
# was firing the refresh ~31× during one menu paint, racking up >13s of HTTP and burning
# single-use refresh tokens. Cache the snapshot for a few seconds, keyed on the auth.json
# path + mtime so that profile switches do not share a process memo and
# `hermes auth login/logout/add/remove` invalidate naturally on the next call.
_NOUS_AUTH_STATUS_CACHE_TTL = 15.0  # seconds
_nous_auth_status_cache: Optional[Tuple[float, str, Optional[float], Dict[str, Any]]] = None












# Enum values reported on the dashboard /api/status as ``nous_session_valid``.
# NAS's health sweep re-mints the bootstrap session ONLY on "terminal"; "valid"
# and "unknown" are no-ops. Keep this set small and stable — NAS parses it with
# a permissive schema, so new members are non-breaking but should stay rare.
NOUS_SESSION_VALID = "valid"
NOUS_SESSION_TERMINAL = "terminal"
NOUS_SESSION_UNKNOWN = "unknown"








async def get_api_key_provider_status(provider_id: str) -> Dict[str, Any]:
    """Status snapshot for API-key providers (z.ai, Kimi, MiniMax)."""
    pconfig = PROVIDER_REGISTRY.get(provider_id)
    if not pconfig or pconfig.auth_type != "api_key":
        return {"configured": False}

    api_key = ""
    key_source = ""
    api_key, key_source = await _resolve_api_key_provider_secret(provider_id, pconfig)

    env_url = ""
    if pconfig.base_url_env_var:
        env_url = os.getenv(pconfig.base_url_env_var, "").strip()

    if provider_id in {"kimi-coding", "kimi-coding-cn"}:
        base_url = _resolve_kimi_base_url(api_key, pconfig.inference_base_url, env_url)
    elif env_url:
        base_url = env_url
    else:
        base_url = pconfig.inference_base_url

    return {
        "configured": bool(api_key),
        "provider": provider_id,
        "name": pconfig.name,
        "key_source": key_source,
        "base_url": base_url,
        "logged_in": bool(api_key),  # compat with OAuth status shape
    }










# =============================================================================
# CLI Commands — login / logout
# =============================================================================

































# ==================== MiniMax Portal OAuth ====================
