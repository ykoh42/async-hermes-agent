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
import inspect
import json
import logging
import os
import shutil
import shlex
import ssl
import stat
import base64
import hashlib
import subprocess
import time
import uuid
import webbrowser
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Tuple,
)
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import aiofiles
import aiofiles.os

from hermes_cli.config import (
    get_hermes_home,
    get_config_path,
    read_raw_config,
    require_readable_config_before_write,
)
from hermes_constants import OPENROUTER_BASE_URL, secure_parent_dir
from agent.credential_persistence import sanitize_borrowed_credential_payload
from agent.ssl_verify import (
    _create_httpx_client,
    _resolve_httpx_client_verify,
    resolve_httpx_verify,
)
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
ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120  # refresh 2 min before expiry
NOUS_INVOKE_JWT_MIN_TTL_SECONDS = ACCESS_TOKEN_REFRESH_SKEW_SECONDS
DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS = 1  # poll at most every 1s
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
# long-lived workloads that may only touch the provider every 30 minutes,
# leaving brief but noisy credential-expiry gaps. Refresh up to one hour
# early so ordinary runtime calls keep the token warm without user reauth.
XAI_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 3600
QWEN_OAUTH_CLIENT_ID = "f0304373b74a44d2b584a3fb70ca9e56"
QWEN_OAUTH_TOKEN_URL = "https://chat.qwen.ai/api/v1/oauth2/token"
QWEN_ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120
OAUTH_OVER_SSH_DOCS_URL = (
    "https://hermes-agent.nousresearch.com/docs/guides/oauth-over-ssh"
)

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
    auth_type: (
        str  # "oauth_device_code", "oauth_external", "oauth_minimax", or "api_key"
    )
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
        extra={
            "region": "global",
            "cn_portal_base_url": MINIMAX_OAUTH_CN_BASE,
            "cn_inference_base_url": MINIMAX_OAUTH_CN_INFERENCE,
        },
    ),
    "anthropic": ProviderConfig(
        id="anthropic",
        name="Anthropic",
        auth_type="api_key",
        inference_base_url="https://api.anthropic.com",
        api_key_env_vars=(
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
        ),
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

def _inject_profile_provider_registry() -> None:
    """Register API-key provider profiles after async discovery.

    The module-level call only projects profiles already present in memory.
    The agent calls this helper again immediately after awaited discovery so
    dynamic profiles retain their upstream registry behavior without blocking
    import-time callers.
    """
    try:
        from providers import _list_providers_cached

        for _pp in _list_providers_cached():
            if _pp.name in PROVIDER_REGISTRY:
                continue
            if _pp.auth_type != "api_key" or not _pp.env_vars:
                continue
            # Skip providers that need custom token resolution or are
            # special-cased in resolve_provider().
            if _pp.name in {
                "copilot",
                "kimi-coding",
                "kimi-coding-cn",
                "zai",
                "openrouter",
                "custom",
            }:
                continue
            _api_key_vars = tuple(
                v
                for v in _pp.env_vars
                if not v.endswith("_BASE_URL") and not v.endswith("_URL")
            )
            _base_url_var = next(
                (
                    v
                    for v in _pp.env_vars
                    if v.endswith("_BASE_URL") or v.endswith("_URL")
                ),
                None,
            )
            PROVIDER_REGISTRY[_pp.name] = ProviderConfig(
                id=_pp.name,
                name=_pp.display_name or _pp.name,
                auth_type="api_key",
                inference_base_url=_pp.base_url,
                api_key_env_vars=_api_key_vars or _pp.env_vars,
                base_url_env_var=_base_url_var or "",
            )
            for _alias in _pp.aliases:
                if _alias not in PROVIDER_REGISTRY:
                    PROVIDER_REGISTRY[_alias] = PROVIDER_REGISTRY[_pp.name]
    except Exception:
        return


_inject_profile_provider_registry()


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
        value = (await get_env_value_prefer_dotenv(env_var) or "").strip()
        if has_usable_secret(value):
            return value, env_var
    return "", ""


async def resolve_api_key_provider_credentials(
    provider_id: str,
) -> Dict[str, Any]:
    """Async counterpart used by native provider/client resolution."""
    from providers import _ensure_provider_profiles_loaded

    await _ensure_provider_profiles_loaded()
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
            api_key,
            pconfig.inference_base_url,
            env_url,
        )
    elif provider_id == "zai":
        base_url = await _resolve_zai_base_url(
            api_key,
            pconfig.inference_base_url,
            env_url,
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


async def resolve_external_process_provider_credentials(
    provider_id: str,
) -> Dict[str, Any]:
    """Resolve runtime details for a native-async subprocess provider."""
    pconfig = PROVIDER_REGISTRY.get(provider_id)
    if not pconfig or pconfig.auth_type != "external_process":
        raise AuthError(
            f"Provider '{provider_id}' is not an external-process provider.",
            provider=provider_id,
            code="invalid_provider",
        )

    base_url = (
        os.getenv(pconfig.base_url_env_var, "").strip()
        if pconfig.base_url_env_var
        else ""
    ) or pconfig.inference_base_url
    command = (
        os.getenv("HERMES_COPILOT_ACP_COMMAND", "").strip()
        or os.getenv("COPILOT_CLI_PATH", "").strip()
        or "copilot"
    )
    raw_args = os.getenv("HERMES_COPILOT_ACP_ARGS", "").strip()
    args = shlex.split(raw_args) if raw_args else ["--acp", "--stdio"]
    resolved_command = (
        await aiofiles.os.wrap(shutil.which)(command) if command else None
    )
    if not resolved_command and not base_url.startswith("acp+tcp://"):
        raise AuthError(
            f"Could not find the Copilot CLI command '{command}'. "
            "Install GitHub Copilot CLI or set "
            "HERMES_COPILOT_ACP_COMMAND/COPILOT_CLI_PATH.",
            provider=provider_id,
            code="missing_copilot_cli",
        )

    return {
        "provider": provider_id,
        "api_key": "copilot-acp",
        "base_url": base_url.rstrip("/"),
        "command": resolved_command or command,
        "args": args,
        "source": "process",
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
    ("global", "https://api.z.ai/api/paas/v4", ["glm-5"], "Global"),
    ("cn", "https://open.bigmodel.cn/api/paas/v4", ["glm-5"], "China"),
    (
        "coding-global",
        "https://api.z.ai/api/coding/paas/v4",
        ["glm-5.2", "glm-5.1", "glm-5v-turbo", "glm-4.7"],
        "Global (Coding Plan)",
    ),
    (
        "coding-cn",
        "https://open.bigmodel.cn/api/coding/paas/v4",
        ["glm-5.2", "glm-5.1", "glm-5v-turbo", "glm-4.7"],
        "China (Coding Plan)",
    ),
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
    async with (await _create_httpx_client(
        timeout=timeout,
        verify=await _resolve_httpx_client_verify(),
    )) as client:
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
                _store_provider_state(
                    auth_store, "zai", state_under_lock, set_active=False
                )
                await _save_auth_store(auth_store)
        except Exception as exc:
            logger.warning(
                "Z.AI: could not persist detected endpoint (%s); will re-probe next start",
                exc,
            )
        logger.info(
            "Z.AI: auto-detected endpoint %s (%s)",
            detected["label"],
            detected["base_url"],
        )
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


def _oauth_trace(
    event: str, *, sequence_id: Optional[str] = None, **fields: Any
) -> None:
    if not _oauth_trace_enabled():
        return
    payload: Dict[str, Any] = {"event": event}
    if sequence_id:
        payload["sequence_id"] = sequence_id
    payload.update(fields)
    logger.info(
        "oauth_trace %s", json.dumps(payload, sort_keys=True, ensure_ascii=False)
    )


# =============================================================================
# Auth Store — persistence layer for ~/.hermes/auth.json
# =============================================================================


async def _auth_file_path() -> Path:
    path = get_hermes_home() / "auth.json"
    # Seat belt: if pytest is running and HERMES_HOME resolves to the real
    # user's auth store, refuse rather than silently corrupt it. This catches
    # tests that forgot to monkeypatch HERMES_HOME, tests invoked without the
    # hermetic conftest, or sandbox escapes via threads/subprocesses. In
    # production (no PYTEST_CURRENT_TEST) this is a single dict lookup.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        realpath = aiofiles.os.wrap(os.path.realpath)
        real_home_auth = Path(await realpath(Path.home() / ".hermes" / "auth.json"))
        try:
            resolved = Path(await realpath(path))
        except Exception:
            resolved = path
        if resolved == real_home_auth:
            raise RuntimeError(
                f"Refusing to touch real user auth store during test run: {path}. "
                "Set HERMES_HOME to a tmp_path in your test fixture, or run "
                "via scripts/run_tests.sh for hermetic CI-parity env."
            )
    return path


async def _global_auth_file_path() -> Optional[Path]:
    """Return the global-root auth.json when the process is in profile mode.

    Returns ``None`` when the profile and global root resolve to the same
    directory (classic mode, or custom HERMES_HOME that is not a profile).
    Used by read-only fallback paths so providers authed at the root are
    visible to profile processes that haven't configured them locally.

    See issue #18594 follow-up (credential_pool shadowing).
    """
    try:
        from hermes_constants import get_default_hermes_root

        global_root = await get_default_hermes_root()
    except Exception:
        return None
    profile_home = get_hermes_home()
    try:
        realpath = aiofiles.os.wrap(os.path.realpath)
        if await realpath(profile_home) == await realpath(global_root):
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


_auth_store_locks: Dict[Tuple[int, str], asyncio.Lock] = {}


async def _auth_store_lock_for(target_path: Path) -> asyncio.Lock:
    """Return this event loop's lock for one auth-store path.

    The native path uses a task lock plus the non-blocking ``flock``
    transaction below. Locks are keyed by event loop as well as path because
    test suites and embedding hosts often create more than one loop in a
    process.
    """
    loop = asyncio.get_running_loop()
    try:
        path_key = await aiofiles.os.wrap(os.path.realpath)(target_path)
    except Exception:
        path_key = str(target_path)
    key = (id(loop), path_key)
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
    auth_path = target_path or await _auth_file_path()
    task_lock = await _auth_store_lock_for(auth_path)
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


def _load_provider_state(
    auth_store: Dict[str, Any], provider_id: str
) -> Optional[Dict[str, Any]]:
    """Return a provider state from an already-loaded auth snapshot."""
    providers = auth_store.get("providers")
    if not isinstance(providers, dict):
        return None
    state = providers.get(provider_id)
    return dict(state) if isinstance(state, dict) else None


def _save_provider_state(
    auth_store: Dict[str, Any], provider_id: str, state: Dict[str, Any]
) -> None:
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
    return normalized in PROVIDER_REGISTRY


def get_auth_provider_display_name(provider_id: str) -> str:
    normalized = (provider_id or "").strip().lower()
    if normalized in PROVIDER_REGISTRY:
        return PROVIDER_REGISTRY[normalized].name
    return provider_id


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
    auth_file = auth_file or await _auth_file_path()
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
    global_path = await _global_auth_file_path()
    if global_path is None or not await aiofiles.os.path.exists(global_path):
        return {}
    if os.environ.get("PYTEST_CURRENT_TEST"):
        real_home_env = os.environ.get("HOME", "")
        if real_home_env:
            real_root = Path(real_home_env) / ".hermes" / "auth.json"
            try:
                if (
                    await aiofiles.os.path.abspath(global_path)
                    == await aiofiles.os.path.abspath(real_root)
                ):
                    return {}
            except Exception:
                pass
    try:
        return await _load_auth_store(global_path)
    except Exception:
        return {}


async def get_provider_auth_state(provider_id: str) -> Optional[Dict[str, Any]]:
    """Return profile-local provider state, falling back to the global root."""
    auth_store = await _load_auth_store()
    state = _load_provider_state(auth_store, provider_id)
    if state is not None:
        return state
    global_store = await _load_global_auth_store()
    return _load_provider_state(global_store, provider_id)


async def is_provider_explicitly_configured(provider_id: str) -> bool:
    """Return whether the user explicitly configured a provider."""
    normalized = (provider_id or "").strip().lower()
    if not normalized:
        return False

    try:
        auth_store = await _load_auth_store()
        active = str(auth_store.get("active_provider") or "").strip().lower()
        if active == normalized:
            return True
    except Exception:
        pass

    try:
        from hermes_cli.config import load_config_readonly

        config = await load_config_readonly()
    except Exception:
        config = {}
    model_cfg = config.get("model") if isinstance(config, dict) else None
    if isinstance(model_cfg, dict):
        configured = str(model_cfg.get("provider") or "").strip().lower()
        if configured == normalized:
            return True

    def slot_matches(slot: Any) -> bool:
        return (
            isinstance(slot, dict)
            and str(slot.get("provider") or "").strip().lower() == normalized
        )

    moa_cfg = config.get("moa") if isinstance(config, dict) else None
    if isinstance(moa_cfg, dict):
        if any(slot_matches(slot) for slot in moa_cfg.get("reference_models") or []):
            return True
        if slot_matches(moa_cfg.get("aggregator")):
            return True
        presets = moa_cfg.get("presets")
        if isinstance(presets, dict):
            for preset in presets.values():
                if not isinstance(preset, dict):
                    continue
                if any(
                    slot_matches(slot)
                    for slot in preset.get("reference_models") or []
                ):
                    return True
                if slot_matches(preset.get("aggregator")):
                    return True

    pconfig = PROVIDER_REGISTRY.get(normalized)
    if pconfig is None:
        try:
            from hermes_cli.providers import get_provider

            pconfig = get_provider(normalized)
        except Exception:
            pconfig = None
    if pconfig and pconfig.auth_type == "api_key":
        from hermes_cli.config import get_env_value_prefer_dotenv

        for env_var in pconfig.api_key_env_vars:
            if env_var == "CLAUDE_CODE_OAUTH_TOKEN":
                continue
            if has_usable_secret(await get_env_value_prefer_dotenv(env_var)):
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
            if env_var:
                from hermes_cli.config import get_env_value_prefer_dotenv

                if await get_env_value_prefer_dotenv(env_var):
                    return True
    return False


async def read_credential_pool(provider_id: Optional[str] = None) -> Dict[str, Any]:
    """Awaitably read one credential-pool slice with profile shadowing."""
    auth_store, global_store = await asyncio.gather(
        _load_auth_store(),
        _load_global_auth_store(),
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
    auth_store: Dict[str, Any],
    target_path: Optional[Path] = None,
) -> Path:
    """Atomically persist ``auth.json`` through awaitable file operations."""
    auth_file = target_path or await _auth_file_path()
    await aiofiles.os.makedirs(auth_file.parent, exist_ok=True)
    await secure_parent_dir(auth_file)
    auth_store["version"] = AUTH_STORE_VERSION
    auth_store["updated_at"] = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(auth_store, indent=2) + "\n"
    tmp_path = auth_file.with_name(
        f"{auth_file.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )

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
    async with _auth_store_transaction():
        auth_store = await _load_auth_store()
        _merge_credential_pool_entries(
            auth_store,
            provider_id,
            entries,
            removed_ids=removed_ids,
        )
        return await _save_auth_store(auth_store)


def _merge_credential_pool_entries(
    auth_store: Dict[str, Any],
    provider_id: str,
    entries: List[Dict[str, Any]],
    *,
    removed_ids: Optional[Iterable[str]] = None,
) -> None:
    """Apply the credential-pool merge to an already locked auth snapshot."""
    removed = {rid for rid in (removed_ids or ()) if rid}
    pool = auth_store.get("credential_pool")
    if not isinstance(pool, dict):
        pool = {}
        auth_store["credential_pool"] = pool
    sanitized_entries = [
        sanitize_borrowed_credential_payload(entry, provider_id)
        if isinstance(entry, dict)
        else entry
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
        "glm": "zai",
        "z-ai": "zai",
        "z.ai": "zai",
        "zhipu": "zai",
        "google": "gemini",
        "google-gemini": "gemini",
        "google-ai-studio": "gemini",
        "x-ai": "xai",
        "x.ai": "xai",
        "grok": "xai",
        "xai-oauth": "xai-oauth",
        "x-ai-oauth": "xai-oauth",
        "grok-oauth": "xai-oauth",
        "xai-grok-oauth": "xai-oauth",
        "kimi": "kimi-coding",
        "kimi-for-coding": "kimi-coding",
        "moonshot": "kimi-coding",
        "kimi-cn": "kimi-coding-cn",
        "moonshot-cn": "kimi-coding-cn",
        "step": "stepfun",
        "stepfun-coding-plan": "stepfun",
        "arcee-ai": "arcee",
        "arceeai": "arcee",
        "gmi-cloud": "gmi",
        "gmicloud": "gmi",
        "minimax-china": "minimax-cn",
        "minimax_cn": "minimax-cn",
        "minimax-portal": "minimax-oauth",
        "minimax-global": "minimax-oauth",
        "minimax_oauth": "minimax-oauth",
        "alibaba_coding": "alibaba-coding-plan",
        "alibaba-coding": "alibaba-coding-plan",
        "alibaba_coding_plan": "alibaba-coding-plan",
        "claude": "anthropic",
        "claude-code": "anthropic",
        "github": "copilot",
        "github-copilot": "copilot",
        "github-models": "copilot",
        "github-model": "copilot",
        "github-copilot-acp": "copilot-acp",
        "copilot-acp-agent": "copilot-acp",
        "aigateway": "ai-gateway",
        "vercel": "ai-gateway",
        "vercel-ai-gateway": "ai-gateway",
        "opencode": "opencode-zen",
        "zen": "opencode-zen",
        "qwen-portal": "qwen-oauth",
        "qwen-cli": "qwen-oauth",
        "qwen-oauth": "qwen-oauth",
        "hf": "huggingface",
        "hugging-face": "huggingface",
        "huggingface-hub": "huggingface",
        "mimo": "xiaomi",
        "xiaomi-mimo": "xiaomi",
        "tencent": "tencent-tokenhub",
        "tokenhub": "tencent-tokenhub",
        "tencent-cloud": "tencent-tokenhub",
        "tencentmaas": "tencent-tokenhub",
        "aws": "bedrock",
        "aws-bedrock": "bedrock",
        "amazon-bedrock": "bedrock",
        "amazon": "bedrock",
        "go": "opencode-go",
        "opencode-go-sub": "opencode-go",
        "kilo": "kilocode",
        "kilo-code": "kilocode",
        "kilo-gateway": "kilocode",
        "lmstudio": "lmstudio",
        "lm-studio": "lmstudio",
        "lm_studio": "lmstudio",
        # Local server aliases — route through the generic custom provider
        "ollama": "custom",
        "ollama_cloud": "ollama-cloud",
        "vllm": "custom",
        "llamacpp": "custom",
        "llama.cpp": "custom",
        "llama-cpp": "custom",
    }
    # Extend with aliases declared in plugins/model-providers/<name>/ that aren't already mapped.
    # This keeps providers/ as the single source for new aliases while the
    # hardcoded dict above remains authoritative for existing ones.
    try:
        from providers import list_providers as _lp

        for _pp in await _lp():
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
            if (
                isinstance(_cfg_provider, str)
                and _cfg_provider.strip().lower() in PROVIDER_REGISTRY
            ):
                return _cfg_provider.strip().lower()
    except Exception as e:
        logger.debug(
            "Could not read config.yaml model.provider for auto-resolution: %s", e
        )

    if has_usable_secret(os.getenv("OPENAI_API_KEY")) or has_usable_secret(
        os.getenv("OPENROUTER_API_KEY")
    ):
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
                        pid,
                        env_var,
                        _oauth_active,
                        env_var,
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
        if (
            isinstance(_model_cfg, dict)
            and _model_cfg
            and not _model_cfg.get("provider")
        ):
            logger.warning(
                "Provider resolved to logged-in OAuth provider %r because "
                "config.yaml `model` has no `provider` key. If you meant a "
                "different provider, set `model.provider` explicitly.",
                _oauth_active,
            )
        return _oauth_active

    # AWS Bedrock — detect through the native async credential chain. This
    # remains after API-key and logged-in providers so explicit credentials
    # preserve their upstream precedence.
    try:
        from agent.bedrock_adapter import has_aws_credentials

        if await has_aws_credentials():
            return "bedrock"
    except ImportError:
        pass  # The optional Bedrock transport is not installed.

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
                stored,
                DEFAULT_NOUS_PORTAL_URL,
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


def _validate_nous_inference_url_from_network(url: Optional[str]) -> Optional[str]:
    """Validate a Portal-returned inference URL before persisting it."""
    if not isinstance(url, str):
        return None
    cleaned = url.strip()
    if not cleaned:
        return None
    try:
        parsed = urlparse(cleaned)
    except Exception:
        return None
    if parsed.scheme != "https":
        logger.warning(
            "nous: refusing non-https inference URL scheme %r from Portal response",
            parsed.scheme,
        )
        return None
    if parsed.hostname not in _ALLOWED_NOUS_INFERENCE_HOSTS:
        logger.warning(
            "nous: refusing inference URL host %r from Portal response "
            "(not in allowlist); falling back to default",
            parsed.hostname,
        )
        return None
    return cleaned.rstrip("/")


def _nous_inference_env_override() -> Optional[str]:
    """Return the user-set ``NOUS_INFERENCE_BASE_URL`` override, if any.

    This is the documented dev/staging escape hatch. The env source is
    trusted (the OS user set it themselves), so it is intentionally NOT
    gated by the network host allowlist — unlike Portal-returned URLs.

    Returns a trailing-slash-stripped non-empty string, or ``None`` when
    the env var is unset/blank.
    """
    return _optional_base_url(os.getenv("NOUS_INFERENCE_BASE_URL"))


def _nous_portal_env_override() -> Optional[str]:
    """Return the trusted operator Portal URL override, if configured."""
    return _optional_base_url(
        os.getenv("HERMES_PORTAL_BASE_URL") or os.getenv("NOUS_PORTAL_BASE_URL")
    )


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


def _assert_nous_inference_jwt_usable(
    state: Dict[str, Any],
    *,
    access_token: Any = None,
) -> None:
    token = state.get("access_token") if access_token is None else access_token
    reason = _nous_invoke_jwt_status(
        token,
        scope=state.get("scope"),
        expires_at=state.get("expires_at"),
    )
    if reason is None:
        return
    raise AuthError(
        "Nous Portal access token is not a usable inference JWT "
        f"({reason}). Re-authenticate with: hermes auth add nous",
        provider="nous",
        code=reason,
        relogin_required=True,
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


def _nous_effective_provider_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in state.items()
        if key not in _NOUS_EFFECTIVE_STATE_IGNORED_KEYS
    }


def _codex_access_token_is_expiring(access_token: Any, skew_seconds: int) -> bool:
    claims = _decode_jwt_claims(access_token)
    exp = claims.get("exp")
    if not isinstance(exp, (int, float)):
        return False
    return float(exp) <= (time.time() + max(0, int(skew_seconds)))


def _qwen_cli_auth_path() -> Path:
    return Path.home() / ".qwen" / "oauth_creds.json"


async def _read_qwen_cli_tokens() -> Dict[str, Any]:
    auth_path = _qwen_cli_auth_path()
    if not await aiofiles.os.path.exists(auth_path):
        raise AuthError(
            "Qwen CLI credentials not found. Run 'qwen auth qwen-oauth' first.",
            provider="qwen-oauth",
            code="qwen_auth_missing",
        )
    try:
        async with aiofiles.open(auth_path, encoding="utf-8") as handle:
            data = json.loads(await handle.read())
    except Exception as exc:
        raise AuthError(
            f"Failed to read Qwen CLI credentials from {auth_path}: {exc}",
            provider="qwen-oauth",
            code="qwen_auth_read_failed",
        ) from exc
    if not isinstance(data, dict):
        raise AuthError(
            f"Invalid Qwen CLI credentials in {auth_path}.",
            provider="qwen-oauth",
            code="qwen_auth_invalid",
        )
    return data


async def _save_qwen_cli_tokens(tokens: Dict[str, Any]) -> Path:
    auth_path = _qwen_cli_auth_path()
    await aiofiles.os.makedirs(auth_path.parent, exist_ok=True)
    await secure_parent_dir(auth_path)
    tmp_path = auth_path.with_name(
        f"{auth_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )

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
            await handle.write(json.dumps(tokens, indent=2, sort_keys=True) + "\n")
            await handle.flush()
            await aiofiles.os.wrap(os.fsync)(handle.fileno())
        await aiofiles.os.replace(tmp_path, auth_path)
        return auth_path
    finally:
        try:
            await aiofiles.os.remove(tmp_path)
        except FileNotFoundError:
            pass


def _qwen_access_token_is_expiring(
    expiry_date_ms: Any,
    skew_seconds: int = QWEN_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
) -> bool:
    try:
        expiry_ms = int(expiry_date_ms)
    except Exception:
        return True
    return (time.time() + max(0, int(skew_seconds))) * 1000 >= expiry_ms


async def _refresh_qwen_cli_tokens(
    tokens: Dict[str, Any],
    timeout_seconds: float = 20.0,
) -> Dict[str, Any]:
    refresh_token = str(tokens.get("refresh_token", "") or "").strip()
    if not refresh_token:
        raise AuthError(
            "Qwen OAuth refresh token missing. Re-run 'qwen auth qwen-oauth'.",
            provider="qwen-oauth",
            code="qwen_refresh_token_missing",
        )

    try:
        async with (await _create_httpx_client(
            timeout=timeout_seconds,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            verify=await _resolve_httpx_client_verify(),
        )) as client:
            response = await client.post(
                QWEN_OAUTH_TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": QWEN_OAUTH_CLIENT_ID,
                },
            )
    except Exception as exc:
        raise AuthError(
            f"Qwen OAuth refresh failed: {exc}",
            provider="qwen-oauth",
            code="qwen_refresh_failed",
        ) from exc

    if response.status_code >= 400:
        body = response.text.strip()
        raise AuthError(
            "Qwen OAuth refresh failed. Re-run 'qwen auth qwen-oauth'."
            + (f" Response: {body}" if body else ""),
            provider="qwen-oauth",
            code="qwen_refresh_failed",
        )

    try:
        payload = response.json()
    except Exception as exc:
        raise AuthError(
            f"Qwen OAuth refresh returned invalid JSON: {exc}",
            provider="qwen-oauth",
            code="qwen_refresh_invalid_json",
        ) from exc

    if (
        not isinstance(payload, dict)
        or not str(payload.get("access_token", "") or "").strip()
    ):
        raise AuthError(
            "Qwen OAuth refresh response missing access_token.",
            provider="qwen-oauth",
            code="qwen_refresh_invalid_response",
        )

    try:
        expires_in_seconds = int(payload.get("expires_in"))
    except Exception:
        expires_in_seconds = 6 * 60 * 60

    refreshed = {
        "access_token": str(payload.get("access_token", "") or "").strip(),
        "refresh_token": str(
            payload.get("refresh_token", refresh_token) or refresh_token
        ).strip(),
        "token_type": str(
            payload.get("token_type", tokens.get("token_type", "Bearer")) or "Bearer"
        ).strip()
        or "Bearer",
        "resource_url": str(
            payload.get(
                "resource_url",
                tokens.get("resource_url", "portal.qwen.ai"),
            )
            or "portal.qwen.ai"
        ).strip(),
        "expiry_date": int(time.time() * 1000) + max(1, expires_in_seconds) * 1000,
    }
    await _save_qwen_cli_tokens(refreshed)
    return refreshed


async def resolve_qwen_runtime_credentials(
    *,
    force_refresh: bool = False,
    refresh_if_expiring: bool = True,
    refresh_skew_seconds: int = QWEN_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
) -> Dict[str, Any]:
    tokens = await _read_qwen_cli_tokens()
    access_token = str(tokens.get("access_token", "") or "").strip()
    should_refresh = bool(force_refresh)
    if not should_refresh and refresh_if_expiring:
        should_refresh = _qwen_access_token_is_expiring(
            tokens.get("expiry_date"),
            refresh_skew_seconds,
        )
    if should_refresh:
        tokens = await _refresh_qwen_cli_tokens(tokens)
        access_token = str(tokens.get("access_token", "") or "").strip()
    if not access_token:
        raise AuthError(
            "Qwen OAuth access token missing. Re-run 'qwen auth qwen-oauth'.",
            provider="qwen-oauth",
            code="qwen_access_token_missing",
        )

    base_url = (
        os.getenv("HERMES_QWEN_BASE_URL", "").strip().rstrip("/")
        or DEFAULT_QWEN_BASE_URL
    )
    return {
        "provider": "qwen-oauth",
        "base_url": base_url,
        "api_key": access_token,
        "source": "qwen-cli",
        "expires_at_ms": tokens.get("expiry_date"),
        "auth_file": str(_qwen_cli_auth_path()),
    }


async def refresh_codex_oauth_pure(
    access_token: str,
    refresh_token: str,
    *,
    timeout_seconds: float = 20.0,
) -> Dict[str, Any]:
    """Refresh Codex OAuth tokens without mutating Hermes auth state."""
    del access_token
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        raise AuthError(
            "Codex auth is missing refresh_token. Run `hermes auth` to re-authenticate.",
            provider="openai-codex",
            code="codex_auth_missing_refresh_token",
            relogin_required=True,
        )

    timeout = httpx.Timeout(max(5.0, float(timeout_seconds)))
    async with (await _create_httpx_client(
        timeout=timeout,
        headers={
            "Accept": "application/json",
            "User-Agent": CODEX_OAUTH_USER_AGENT,
        },
        verify=await _resolve_httpx_client_verify(),
    )) as client:
        response = await client.post(
            CODEX_OAUTH_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CODEX_OAUTH_CLIENT_ID,
            },
        )

    if response.status_code == 429:
        retry_after = _parse_retry_after_seconds(getattr(response, "headers", None))
        if retry_after is not None:
            message = (
                f"Codex provider quota exhausted (429); retry after {retry_after}s. "
                "Credentials are still valid."
            )
        else:
            message = (
                "Codex provider quota exhausted (429). Credentials are still valid; "
                "retry after the usage limit resets."
            )
        raise AuthError(
            message,
            provider="openai-codex",
            code=CODEX_RATE_LIMITED_CODE,
            relogin_required=False,
        )

    if response.status_code != 200:
        code = "codex_refresh_failed"
        message = f"Codex token refresh failed with status {response.status_code}."
        relogin_required = False
        try:
            error_payload = response.json()
            if isinstance(error_payload, dict):
                error = error_payload.get("error")
                if isinstance(error, dict):
                    nested_code = error.get("code") or error.get("type")
                    if isinstance(nested_code, str) and nested_code.strip():
                        code = nested_code.strip()
                    nested_message = error.get("message")
                    if isinstance(nested_message, str) and nested_message.strip():
                        message = (
                            f"Codex token refresh failed: {nested_message.strip()}"
                        )
                elif isinstance(error, str) and error.strip():
                    code = error.strip()
                    description = error_payload.get(
                        "error_description"
                    ) or error_payload.get("message")
                    if isinstance(description, str) and description.strip():
                        message = f"Codex token refresh failed: {description.strip()}"
        except Exception:
            pass
        if code in {"invalid_grant", "invalid_token", "invalid_request"}:
            relogin_required = True
        if code == "refresh_token_reused":
            message = (
                "Codex refresh token was already consumed by another client "
                "(e.g. Codex CLI or VS Code extension). "
                "Run `codex` in your terminal to generate fresh tokens, "
                "then run `hermes auth` to re-authenticate."
            )
            relogin_required = True
        if response.status_code in {401, 403} and not relogin_required:
            relogin_required = True
        raise AuthError(
            message,
            provider="openai-codex",
            code=code,
            relogin_required=relogin_required,
        )

    try:
        refresh_payload = response.json()
    except Exception as exc:
        raise AuthError(
            "Codex token refresh returned invalid JSON.",
            provider="openai-codex",
            code="codex_refresh_invalid_json",
            relogin_required=True,
        ) from exc

    refreshed_access = refresh_payload.get("access_token")
    if not isinstance(refreshed_access, str) or not refreshed_access.strip():
        raise AuthError(
            "Codex token refresh response was missing access_token.",
            provider="openai-codex",
            code="codex_refresh_missing_access_token",
            relogin_required=True,
        )

    updated = {
        "access_token": refreshed_access.strip(),
        "refresh_token": refresh_token.strip(),
        "last_refresh": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    next_refresh = refresh_payload.get("refresh_token")
    if isinstance(next_refresh, str) and next_refresh.strip():
        updated["refresh_token"] = next_refresh.strip()
    return updated


def _is_terminal_codex_oauth_refresh_error(exc: Exception) -> bool:
    """Return whether retrying the same Codex refresh token cannot succeed."""
    return (
        isinstance(exc, AuthError)
        and exc.provider == "openai-codex"
        and exc.code
        in {
            "codex_refresh_failed",
            "codex_auth_missing_refresh_token",
            "invalid_grant",
            "invalid_token",
            "refresh_token_reused",
        }
        and bool(exc.relogin_required)
    )


def _is_terminal_xai_oauth_refresh_error(exc: Exception) -> bool:
    """Return whether retrying the same xAI refresh token cannot succeed."""
    return (
        isinstance(exc, AuthError)
        and exc.provider == "xai-oauth"
        and exc.code in {"xai_refresh_failed", "xai_auth_missing_refresh_token"}
        and bool(exc.relogin_required)
    )


def _is_codex_rate_limit_shaped(
    code: Any,
    reason: Any,
    message: Any,
) -> bool:
    """Return whether persisted pool metadata represents quota exhaustion."""
    reason_l = str(reason or "").lower()
    message_l = str(message or "").lower()
    return (
        code == 429
        or "rate_limit" in reason_l
        or "usage_limit" in reason_l
        or "quota" in reason_l
        or "rate limit" in message_l
        or "usage limit" in message_l
        or "quota" in message_l
    )


CODEX_QUOTA_PROBE_MIN_INTERVAL_SECONDS = 300
_codex_quota_probe_cache: Dict[str, Tuple[float, Optional[bool]]] = {}


def _codex_usage_probe_url(base_url: Optional[str]) -> str:
    """Resolve the Codex usage endpoint using the upstream path-style rule."""
    normalized = str(base_url or "").strip().rstrip("/")
    if not normalized:
        normalized = (
            os.getenv("HERMES_CODEX_BASE_URL", "").strip().rstrip("/")
            or DEFAULT_CODEX_BASE_URL
        )
    if normalized.endswith("/codex"):
        normalized = normalized[: -len("/codex")]
    prefix = normalized + ("/wham" if "/backend-api" in normalized else "/api/codex")
    return prefix + "/usage"


async def _probe_codex_quota_restored(
    access_token: Any,
    *,
    base_url: Optional[str] = None,
    min_interval_seconds: float = CODEX_QUOTA_PROBE_MIN_INTERVAL_SECONDS,
) -> Optional[bool]:
    """Return whether every reported Codex quota window is usable again."""
    token = str(access_token or "").strip()
    claims = _decode_jwt_claims(token) if token else {}
    if not claims:
        return None

    cache_key = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
    now = time.monotonic()
    cached = _codex_quota_probe_cache.get(cache_key)
    if cached is not None and (now - cached[0]) < min_interval_seconds:
        return cached[1]
    # There is no await before this assignment, so concurrent tasks on the
    # event loop cannot stampede the endpoint.
    _codex_quota_probe_cache[cache_key] = (now, None)

    result: Optional[bool] = None
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "codex-cli",
        }
        auth_claims = claims.get("https://api.openai.com/auth")
        account_id = (
            auth_claims.get("chatgpt_account_id")
            if isinstance(auth_claims, dict)
            else None
        )
        if isinstance(account_id, str) and account_id.strip():
            headers["ChatGPT-Account-Id"] = account_id.strip()
        async with (await _create_httpx_client(
            timeout=10.0,
            verify=await _resolve_httpx_client_verify(),
        )) as client:
            response = await client.get(
                _codex_usage_probe_url(base_url),
                headers=headers,
            )
        if response.status_code == 200:
            payload = response.json() or {}
            rate_limit = payload.get("rate_limit") or {}
            used_values = [
                (rate_limit.get(key) or {}).get("used_percent")
                for key in ("primary_window", "secondary_window")
            ]
            numeric_values = [
                float(value)
                for value in used_values
                if isinstance(value, (int, float))
            ]
            if numeric_values:
                result = max(numeric_values) < 100.0
        elif response.status_code == 429:
            result = False
    except asyncio.CancelledError:
        if _codex_quota_probe_cache.get(cache_key) == (now, None):
            _codex_quota_probe_cache.pop(cache_key, None)
        raise
    except Exception:
        logger.debug("Codex quota probe failed", exc_info=True)

    _codex_quota_probe_cache[cache_key] = (now, result)
    return result


async def clear_codex_pool_quota_cooldowns(
    access_token: Optional[str] = None,
) -> int:
    """Clear persisted Codex 429 cooldowns after quota recovery is confirmed."""
    cleared = 0
    targets: list[Path | None] = [None]
    global_path = await _global_auth_file_path()
    if global_path is not None:
        targets.append(global_path)

    for target_path in targets:
        try:
            async with _auth_store_transaction(target_path):
                auth_store = await _load_auth_store(target_path)
                pool = auth_store.get("credential_pool")
                entries = (
                    pool.get("openai-codex") if isinstance(pool, dict) else None
                )
                if not isinstance(entries, list):
                    continue
                changed = 0
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    if entry.get("last_status") != "exhausted":
                        continue
                    if (
                        access_token
                        and str(entry.get("access_token") or "") != access_token
                    ):
                        continue
                    if not _is_codex_rate_limit_shaped(
                        entry.get("last_error_code"),
                        entry.get("last_error_reason"),
                        entry.get("last_error_message"),
                    ):
                        continue
                    for field_name in (
                        "last_status",
                        "last_status_at",
                        "last_error_code",
                        "last_error_reason",
                        "last_error_message",
                        "last_error_reset_at",
                    ):
                        entry[field_name] = None
                    changed += 1
                if changed:
                    await _save_auth_store(auth_store, target_path)
                    cleared += changed
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("Failed to clear Codex pool quota cooldowns", exc_info=True)
    return cleared


async def resolve_codex_runtime_credentials(
    *,
    force_refresh: bool = False,
    refresh_if_expiring: bool = True,
    refresh_skew_seconds: int = CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
) -> Dict[str, Any]:
    """Resolve runtime credentials from Hermes's Codex credential pool."""
    from agent.credential_pool import STATUS_EXHAUSTED, load_pool

    pool = await load_pool("openai-codex")
    if not pool.has_credentials():
        state = await get_provider_auth_state("openai-codex")
        if not state:
            raise AuthError(
                "No Codex credentials stored. Run `hermes auth` to authenticate.",
                provider="openai-codex",
                code="codex_auth_missing",
                relogin_required=True,
            )
        tokens = state.get("tokens")
        if not isinstance(tokens, dict):
            code = "codex_auth_invalid_shape"
            message = "Codex auth state is missing tokens."
        elif not str(tokens.get("access_token") or "").strip():
            code = "codex_auth_missing_access_token"
            message = "Codex auth is missing access_token."
        else:
            code = "codex_auth_missing_refresh_token"
            message = "Codex auth is missing refresh_token."
        raise AuthError(
            f"{message} Run `hermes auth` to re-authenticate.",
            provider="openai-codex",
            code=code,
            relogin_required=True,
        )

    entry = await pool.select()
    if entry is None:
        rate_limited = next(
            (
                candidate
                for candidate in pool.entries()
                if candidate.last_status == STATUS_EXHAUSTED
                and (
                    candidate.last_error_code == 429
                    or "rate" in str(candidate.last_error_reason or "").lower()
                    or "usage" in str(candidate.last_error_reason or "").lower()
                    or "quota" in str(candidate.last_error_message or "").lower()
                )
            ),
            None,
        )
        if rate_limited is not None:
            if await _probe_codex_quota_restored(
                rate_limited.access_token,
                base_url=rate_limited.runtime_base_url,
            ):
                await clear_codex_pool_quota_cooldowns()
                pool = await load_pool("openai-codex")
                entry = await pool.select()
            if entry is None:
                reset_at = rate_limited.last_error_reset_at
                if isinstance(reset_at, (int, float)) and reset_at > time.time():
                    remaining = int(reset_at - time.time())
                    message = (
                        "Codex provider quota exhausted (429); "
                        f"retry after {remaining}s. Credentials are still valid."
                    )
                else:
                    message = (
                        "Codex provider quota exhausted (429). Credentials are still "
                        "valid; retry after the usage limit resets."
                    )
                raise AuthError(
                    message,
                    provider="openai-codex",
                    code=CODEX_RATE_LIMITED_CODE,
                    relogin_required=False,
                )
        if entry is None:
            raise AuthError(
                "No usable Codex credentials stored. Run `hermes auth` to re-authenticate.",
                provider="openai-codex",
                code="codex_auth_missing",
                relogin_required=True,
            )

    should_refresh = force_refresh or (
        refresh_if_expiring
        and _codex_access_token_is_expiring(
            entry.access_token,
            refresh_skew_seconds,
        )
    )
    if should_refresh:
        refreshed = await pool.try_refresh_matching(credential_id=entry.id)
        if refreshed is None:
            raise AuthError(
                "Codex credentials could not be refreshed. Run `hermes auth` "
                "to re-authenticate.",
                provider="openai-codex",
                code="codex_refresh_failed",
                relogin_required=True,
            )
        entry = refreshed

    state = await get_provider_auth_state("openai-codex") or {}
    tokens = state.get("tokens") if isinstance(state, dict) else None
    singleton_token = (
        str(tokens.get("access_token") or "").strip()
        if isinstance(tokens, dict)
        else ""
    )
    source = (
        "hermes-auth-store"
        if singleton_token and singleton_token == entry.access_token
        else "credential_pool"
    )
    return {
        "provider": "openai-codex",
        "base_url": (
            os.getenv("HERMES_CODEX_BASE_URL", "").strip().rstrip("/")
            or entry.runtime_base_url
            or DEFAULT_CODEX_BASE_URL
        ),
        "api_key": entry.runtime_api_key,
        "source": source,
        "last_refresh": entry.last_refresh,
        "auth_mode": "chatgpt",
    }


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
_CONSOLE_BROWSER_NAMES: FrozenSet[str] = frozenset({
    "w3m",
    "lynx",
    "links",
    "links2",
    "elinks",
    "www-browser",
    "browsh",  # TUI browser — still hijacks the terminal
})


# =============================================================================
# OpenAI Codex auth — tokens stored in ~/.hermes/auth.json (not ~/.codex/)
#
# Hermes maintains its own Codex OAuth session separate from the Codex CLI
# and VS Code extension. This prevents refresh token rotation conflicts
# where one app's refresh invalidates the other's session.
# =============================================================================


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
        payload = json.loads(
            base64.urlsafe_b64decode(payload_b64.encode("ascii")).decode("utf-8")
        )
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
        payload = json.loads(
            base64.urlsafe_b64decode(payload_b64.encode("ascii")).decode("utf-8")
        )
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
            candidate,
            fallback,
        )
        return fallback
    if parsed.scheme != "https":
        logger.warning(
            "Refusing non-HTTPS xAI base_url override %r (xai-oauth bearer would "
            "be sent in cleartext); falling back to %s.",
            candidate,
            fallback,
        )
        return fallback
    host = (parsed.hostname or "").lower()
    if not host:
        logger.warning(
            "Ignoring xAI base_url override %r with no hostname; using %s instead.",
            candidate,
            fallback,
        )
        return fallback
    if host != "x.ai" and not host.endswith(".x.ai"):
        logger.warning(
            "Refusing xAI base_url override %r — host %r is not on the xAI origin "
            "(expected x.ai or a *.x.ai subdomain). The xai-oauth bearer is only "
            "valid against xAI's inference API; sending it elsewhere would leak "
            "the credential. Falling back to %s.",
            candidate,
            host,
            fallback,
        )
        return fallback
    return candidate


async def _xai_oauth_discovery(timeout_seconds: float = 15.0) -> Dict[str, str]:
    try:
        async with (await _create_httpx_client(
            timeout=httpx.Timeout(timeout_seconds),
            headers={"Accept": "application/json"},
            verify=await _resolve_httpx_client_verify(),
        )) as client:
            response = await client.get(XAI_OAUTH_DISCOVERY_URL)
    except Exception as exc:
        raise AuthError(
            f"xAI OIDC discovery failed: {exc}",
            provider="xai-oauth",
            code="xai_discovery_failed",
        ) from exc
    if response.status_code != 200:
        raise AuthError(
            f"xAI OIDC discovery returned status {response.status_code}.",
            provider="xai-oauth",
            code="xai_discovery_failed",
        )
    try:
        payload = response.json()
    except Exception as exc:
        raise AuthError(
            f"xAI OIDC discovery returned invalid JSON: {exc}",
            provider="xai-oauth",
            code="xai_discovery_invalid_json",
        ) from exc
    if not isinstance(payload, dict):
        raise AuthError(
            "xAI OIDC discovery response was not a JSON object.",
            provider="xai-oauth",
            code="xai_discovery_incomplete",
        )
    authorization_endpoint = str(
        payload.get("authorization_endpoint", "") or ""
    ).strip()
    token_endpoint = str(payload.get("token_endpoint", "") or "").strip()
    if not authorization_endpoint or not token_endpoint:
        raise AuthError(
            "xAI OIDC discovery response was missing required endpoints.",
            provider="xai-oauth",
            code="xai_discovery_incomplete",
        )
    _xai_validate_oauth_endpoint(
        authorization_endpoint,
        field="authorization_endpoint",
    )
    _xai_validate_oauth_endpoint(token_endpoint, field="token_endpoint")
    return {
        "authorization_endpoint": authorization_endpoint,
        "token_endpoint": token_endpoint,
    }


async def refresh_xai_oauth_pure(
    access_token: str,
    refresh_token: str,
    *,
    token_endpoint: str = "",
    timeout_seconds: float = 20.0,
) -> Dict[str, Any]:
    del access_token
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        raise AuthError(
            "xAI OAuth is missing refresh_token. Re-authenticate with `hermes model`.",
            provider="xai-oauth",
            code="xai_auth_missing_refresh_token",
            relogin_required=True,
        )
    endpoint = token_endpoint.strip()
    if not endpoint:
        endpoint = (await _xai_oauth_discovery(timeout_seconds))["token_endpoint"]
    _xai_validate_oauth_endpoint(endpoint, field="token_endpoint")
    timeout = httpx.Timeout(max(5.0, float(timeout_seconds)))
    async with (await _create_httpx_client(
        timeout=timeout,
        headers={"Accept": "application/json"},
        verify=await _resolve_httpx_client_verify(),
    )) as client:
        response = await client.post(
            endpoint,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "client_id": XAI_OAUTH_CLIENT_ID,
                "refresh_token": refresh_token,
            },
        )
    if response.status_code != 200:
        detail = response.text.strip()
        if response.status_code == 403:
            raise AuthError(
                "xAI token refresh failed with HTTP 403."
                + (f" Response: {detail}" if detail else "")
                + " This OAuth account is not authorized for xAI API"
                " access — xAI may be restricting API/OAuth use to"
                " specific SuperGrok tiers despite the in-app"
                " subscription being active. Re-logging in won't"
                " change that; set ``XAI_API_KEY`` and switch to"
                " ``provider: xai`` (API-key path) if available, or"
                " upgrade your subscription at https://x.ai/grok.",
                provider="xai-oauth",
                code="xai_oauth_tier_denied",
                relogin_required=False,
            )
        raise AuthError(
            "xAI token refresh failed." + (f" Response: {detail}" if detail else ""),
            provider="xai-oauth",
            code="xai_refresh_failed",
            relogin_required=(response.status_code in {400, 401}),
        )
    try:
        payload = response.json()
    except Exception as exc:
        raise AuthError(
            f"xAI token refresh returned invalid JSON: {exc}",
            provider="xai-oauth",
            code="xai_refresh_invalid_json",
        ) from exc
    if not isinstance(payload, dict):
        raise AuthError(
            "xAI token refresh response was not a JSON object.",
            provider="xai-oauth",
            code="xai_refresh_invalid_response",
            relogin_required=True,
        )
    refreshed_access = str(payload.get("access_token", "") or "").strip()
    if not refreshed_access:
        raise AuthError(
            "xAI token refresh response was missing access_token.",
            provider="xai-oauth",
            code="xai_refresh_missing_access_token",
            relogin_required=True,
        )
    return {
        "access_token": refreshed_access,
        "refresh_token": str(payload.get("refresh_token") or refresh_token).strip(),
        "id_token": str(payload.get("id_token") or "").strip(),
        "expires_in": payload.get("expires_in"),
        "token_type": str(payload.get("token_type") or "Bearer").strip() or "Bearer",
        "last_refresh": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


async def resolve_xai_oauth_runtime_credentials(
    *,
    force_refresh: bool = False,
    refresh_if_expiring: bool = True,
    refresh_skew_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """Resolve a usable xAI OAuth credential through the native async pool."""
    from agent.credential_pool import load_pool

    pool = await load_pool("xai-oauth")
    if not pool.has_credentials():
        state = await get_provider_auth_state("xai-oauth")
        if not state:
            raise AuthError(
                "No xAI OAuth credentials stored. Select xAI Grok OAuth "
                "(SuperGrok / Premium+) in `hermes model`.",
                provider="xai-oauth",
                code="xai_auth_missing",
                relogin_required=True,
            )
        tokens = state.get("tokens")
        if not isinstance(tokens, dict):
            code = "xai_auth_invalid_shape"
            message = "xAI OAuth state is missing tokens."
        elif not str(tokens.get("access_token") or "").strip():
            code = "xai_auth_missing_access_token"
            message = "xAI OAuth state is missing access_token."
        else:
            code = "xai_auth_missing_refresh_token"
            message = "xAI OAuth state is missing refresh_token."
        raise AuthError(
            f"{message} Re-authenticate with `hermes model`.",
            provider="xai-oauth",
            code=code,
            relogin_required=True,
        )

    entry = await pool.select()
    if entry is None:
        raise AuthError(
            "No usable xAI OAuth credentials stored. Re-authenticate with "
            "`hermes model`.",
            provider="xai-oauth",
            code="xai_auth_missing",
            relogin_required=True,
        )
    effective_skew = (
        int(refresh_skew_seconds)
        if refresh_skew_seconds is not None
        else _xai_proactive_refresh_skew_seconds(entry.access_token)
    )
    should_refresh = force_refresh or (
        refresh_if_expiring
        and _xai_access_token_is_expiring(entry.access_token, effective_skew)
    )
    if should_refresh:
        refreshed = await pool.try_refresh_matching(credential_id=entry.id)
        if refreshed is None:
            raise AuthError(
                "xAI OAuth credentials could not be refreshed. Re-authenticate "
                "with `hermes model`.",
                provider="xai-oauth",
                code="xai_refresh_failed",
                relogin_required=True,
            )
        entry = refreshed

    base_url = _xai_validate_inference_base_url(
        os.getenv("HERMES_XAI_BASE_URL", "").strip().rstrip("/")
        or os.getenv("XAI_BASE_URL", "").strip().rstrip("/")
        or entry.runtime_base_url,
        fallback=DEFAULT_XAI_OAUTH_BASE_URL,
    )
    state = await get_provider_auth_state("xai-oauth") or {}
    tokens = state.get("tokens") if isinstance(state, dict) else None
    singleton_token = (
        str(tokens.get("access_token") or "").strip()
        if isinstance(tokens, dict)
        else ""
    )
    return {
        "provider": "xai-oauth",
        "base_url": base_url,
        "api_key": entry.runtime_api_key,
        "source": (
            "hermes-auth-store"
            if singleton_token and singleton_token == entry.access_token
            else "credential_pool"
        ),
        "last_refresh": entry.last_refresh,
        "auth_mode": "oauth_device_code",
    }


# =============================================================================
# TLS verification helper
# =============================================================================


async def _default_verify() -> bool | ssl.SSLContext:
    return await resolve_httpx_verify()


async def _resolve_verify(
    *,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    auth_state: Optional[Dict[str, Any]] = None,
) -> bool | ssl.SSLContext:
    tls_state = auth_state.get("tls") if isinstance(auth_state, dict) else {}
    tls_state = tls_state if isinstance(tls_state, dict) else {}
    effective_insecure = (
        is_truthy_value(insecure, default=False)
        if insecure is not None
        else is_truthy_value(tls_state.get("insecure", False), default=False)
    )
    effective_ca = (
        ca_bundle
        or tls_state.get("ca_bundle")
        or os.getenv("HERMES_CA_BUNDLE")
        or os.getenv("SSL_CERT_FILE")
        or os.getenv("REQUESTS_CA_BUNDLE")
    )
    return await resolve_httpx_verify(
        ca_bundle=str(effective_ca) if effective_ca else None,
        ssl_verify=False if effective_insecure else None,
    )


async def _resolve_client_verify(
    *,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    auth_state: Optional[Dict[str, Any]] = None,
) -> bool | ssl.SSLContext:
    """Resolve auth TLS settings and prebuild httpx's default context."""
    verify = await _resolve_verify(
        insecure=insecure,
        ca_bundle=ca_bundle,
        auth_state=auth_state,
    )
    if verify is True:
        return await _resolve_httpx_client_verify()
    return verify


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


async def _nous_shared_auth_dir() -> Path:
    override = os.getenv("HERMES_SHARED_AUTH_DIR", "").strip()
    if override:
        return await aiofiles.os.wrap(Path.expanduser)(Path(override))
    from hermes_constants import get_default_hermes_root

    return (await get_default_hermes_root()) / "shared"


async def _nous_shared_store_path() -> Path:
    path = (await _nous_shared_auth_dir()) / NOUS_SHARED_STORE_FILENAME
    if os.environ.get("PYTEST_CURRENT_TEST"):
        from hermes_constants import get_default_hermes_root

        realpath = aiofiles.os.wrap(os.path.realpath)
        default_root = await get_default_hermes_root()
        real_store = Path(
            await realpath(
                default_root / "shared" / NOUS_SHARED_STORE_FILENAME
            )
        )
        try:
            resolved = Path(await realpath(path))
        except Exception:
            resolved = path
        if resolved == real_store:
            raise RuntimeError(
                "Refusing to touch real user shared Nous auth store during test run: "
                f"{path}. Set HERMES_SHARED_AUTH_DIR to a tmp_path in your test fixture."
            )
    return path


async def _read_shared_nous_state() -> Optional[Dict[str, Any]]:
    try:
        path = await _nous_shared_store_path()
    except RuntimeError:
        return None
    if not await aiofiles.os.path.isfile(path):
        return None
    try:
        async with aiofiles.open(path, encoding="utf-8") as handle:
            payload = json.loads(await handle.read())
    except (OSError, ValueError) as exc:
        logger.debug("Shared Nous auth store at %s is unreadable: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        return None
    refresh_token = payload.get("refresh_token")
    access_token = payload.get("access_token")
    if not (isinstance(refresh_token, str) and refresh_token.strip()):
        return None
    if not (isinstance(access_token, str) and access_token.strip()):
        return None
    return payload


async def _save_shared_nous_state(state: Dict[str, Any]) -> None:
    """Write shared state while the caller holds its transaction lock."""
    path = await _nous_shared_store_path()
    refresh_token = state.get("refresh_token")
    access_token = state.get("access_token")
    if not (isinstance(refresh_token, str) and refresh_token.strip()):
        return
    if not (isinstance(access_token, str) and access_token.strip()):
        return
    shared = {
        "_schema": 1,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": state.get("token_type") or "Bearer",
        "scope": state.get("scope") or DEFAULT_NOUS_SCOPE,
        "client_id": state.get("client_id") or DEFAULT_NOUS_CLIENT_ID,
        "portal_base_url": state.get("portal_base_url") or DEFAULT_NOUS_PORTAL_URL,
        "inference_base_url": state.get("inference_base_url")
        or DEFAULT_NOUS_INFERENCE_URL,
        "obtained_at": state.get("obtained_at"),
        "expires_at": state.get("expires_at"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    await aiofiles.os.makedirs(path.parent, exist_ok=True)
    await secure_parent_dir(path)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")

    def secure_opener(file: str, flags: int) -> int:
        return os.open(file, flags | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)

    try:
        async with aiofiles.open(
            tmp,
            "w",
            encoding="utf-8",
            opener=secure_opener,
        ) as handle:
            await handle.write(json.dumps(shared, indent=2, sort_keys=True))
            await handle.flush()
        await aiofiles.os.replace(tmp, path)
    finally:
        try:
            await aiofiles.os.remove(tmp)
        except FileNotFoundError:
            pass
    _oauth_trace(
        "nous_shared_store_written",
        path=str(path),
        refresh_token_fp=_token_fingerprint(refresh_token),
    )


async def _write_shared_nous_state(state: Dict[str, Any]) -> None:
    """Best-effort cross-profile mirror of the Nous OAuth token chain."""
    try:
        path = await _nous_shared_store_path()
        async with _auth_store_transaction(path):
            await _save_shared_nous_state(state)
    except Exception as exc:
        logger.debug("Failed to write shared Nous auth store: %s", exc)


async def _merge_shared_nous_oauth_state(state: Dict[str, Any]) -> bool:
    shared = await _read_shared_nous_state()
    if not shared:
        return False
    shared_refresh = shared.get("refresh_token")
    if not isinstance(shared_refresh, str) or not shared_refresh.strip():
        return False
    local_refresh = state.get("refresh_token")
    shared_access_exp = _parse_iso_timestamp(shared.get("expires_at")) or 0.0
    local_access_exp = _parse_iso_timestamp(state.get("expires_at")) or 0.0
    refresh_changed = shared_refresh.strip() != str(local_refresh or "").strip()
    if not refresh_changed and shared_access_exp <= local_access_exp:
        return False
    for key in (
        "access_token",
        "refresh_token",
        "token_type",
        "scope",
        "client_id",
        "portal_base_url",
        "inference_base_url",
        "obtained_at",
        "expires_at",
    ):
        value = shared.get(key)
        if value not in {None, ""}:
            state[key] = value
    return True


async def _clear_shared_nous_state(reason: str) -> None:
    try:
        path = await _nous_shared_store_path()
        async with _auth_store_transaction(path):
            try:
                await aiofiles.os.remove(path)
            except FileNotFoundError:
                pass
        _oauth_trace("nous_shared_store_cleared", reason=reason)
    except Exception as exc:
        logger.debug("Failed to clear shared Nous auth store: %s", exc)


def _is_terminal_nous_refresh_error(exc: Exception) -> bool:
    return (
        isinstance(exc, AuthError)
        and exc.provider == "nous"
        and exc.code in {"invalid_grant", "invalid_token", "refresh_token_reused"}
        and bool(exc.relogin_required)
    )


def _quarantine_nous_oauth_state(
    state: Dict[str, Any],
    error: AuthError,
    *,
    reason: str,
) -> None:
    logger.warning(
        "Nous OAuth state quarantined (terminal auth death): %s",
        json.dumps(
            {
                "reason": reason,
                "error_code": error.code,
                "client_id": state.get("client_id"),
                "agent_key_id": state.get("agent_key_id"),
                "refresh_token_fp": _token_fingerprint(state.get("refresh_token")),
            },
            sort_keys=True,
            ensure_ascii=False,
        ),
    )
    for key in (
        "access_token",
        "refresh_token",
        "expires_at",
        "expires_in",
        "obtained_at",
        "agent_key",
        "agent_key_id",
        "agent_key_expires_at",
        "agent_key_expires_in",
        "agent_key_reused",
        "agent_key_obtained_at",
    ):
        state.pop(key, None)
    state["last_auth_error"] = {
        "provider": "nous",
        "code": error.code,
        "message": str(error),
        "reason": reason,
        "relogin_required": True,
        "at": datetime.now(timezone.utc).isoformat(),
    }


def _quarantine_nous_pool_entries(
    auth_store: Dict[str, Any],
    error: AuthError,
    *,
    reason: str,
) -> bool:
    pool = auth_store.get("credential_pool")
    if not isinstance(pool, dict):
        return False
    entries = pool.get("nous")
    if not isinstance(entries, list):
        return False
    singleton_sources = {NOUS_DEVICE_CODE_SOURCE, f"manual:{NOUS_DEVICE_CODE_SOURCE}"}
    retained = [
        entry
        for entry in entries
        if not (isinstance(entry, dict) and entry.get("source") in singleton_sources)
    ]
    if len(retained) == len(entries):
        return False
    pool["nous"] = retained
    _oauth_trace(
        "nous_pool_device_code_quarantined",
        reason=reason,
        error_code=error.code,
    )
    return True


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


async def _refresh_access_token(
    *,
    client: httpx.AsyncClient,
    portal_base_url: str,
    client_id: str,
    refresh_token: str,
) -> Dict[str, Any]:
    response = await client.post(
        f"{portal_base_url}/api/oauth/token",
        headers={"x-nous-refresh-token": refresh_token},
        data={
            "grant_type": "refresh_token",
            "client_id": client_id,
        },
    )
    if response.status_code == 200:
        payload = response.json()
        if "access_token" not in payload:
            raise AuthError(
                "Refresh response missing access_token",
                provider="nous",
                code="invalid_token",
                relogin_required=True,
            )
        return payload
    try:
        error_payload = response.json()
    except Exception as exc:
        raise AuthError(
            "Refresh token exchange failed",
            provider="nous",
            relogin_required=True,
        ) from exc
    code = str(error_payload.get("error", "invalid_grant"))
    description = str(
        error_payload.get("error_description") or "Refresh token exchange failed"
    )
    relogin = code in {"invalid_grant", "invalid_token", "refresh_token_reused"}
    lowered = description.lower()
    if code == "refresh_token_reused" or "reuse" in lowered:
        description = (
            "Nous Portal detected refresh-token reuse and revoked this session.\n"
            "This usually means another process used Hermes's single-use refresh "
            "token without persisting the rotated token.\n"
            "Re-authenticate with: hermes auth add nous"
        )
        relogin = True
    raise AuthError(
        description,
        provider="nous",
        code=code,
        relogin_required=relogin,
    )


async def refresh_nous_oauth_pure(
    access_token: str,
    refresh_token: str,
    client_id: str,
    portal_base_url: str,
    inference_base_url: str,
    *,
    token_type: str = "Bearer",
    scope: str = DEFAULT_NOUS_SCOPE,
    obtained_at: Optional[str] = None,
    expires_at: Optional[str] = None,
    agent_key: Optional[str] = None,
    agent_key_expires_at: Optional[str] = None,
    timeout_seconds: float = 15.0,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    force_refresh: bool = False,
    on_state_update: Optional[
        Callable[[Dict[str, Any], str], Optional[Awaitable[None]]]
    ] = None,
) -> Dict[str, Any]:
    """Refresh Nous OAuth without owning a particular credential store."""
    state: Dict[str, Any] = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "client_id": client_id or DEFAULT_NOUS_CLIENT_ID,
        "portal_base_url": (portal_base_url or DEFAULT_NOUS_PORTAL_URL).rstrip("/"),
        "inference_base_url": (inference_base_url or DEFAULT_NOUS_INFERENCE_URL).rstrip(
            "/"
        ),
        "token_type": token_type or "Bearer",
        "scope": scope or DEFAULT_NOUS_SCOPE,
        "obtained_at": obtained_at,
        "expires_at": expires_at,
        "agent_key": agent_key,
        "agent_key_expires_at": agent_key_expires_at,
        "tls": {"insecure": bool(insecure), "ca_bundle": ca_bundle},
    }
    verify = await _resolve_client_verify(
        insecure=insecure,
        ca_bundle=ca_bundle,
        auth_state=state,
    )
    timeout = httpx.Timeout(timeout_seconds if timeout_seconds else 15.0)
    async with (await _create_httpx_client(
        timeout=timeout,
        headers={"Accept": "application/json"},
        verify=verify,
    )) as client:
        invoke_status = _nous_invoke_jwt_status(
            state.get("access_token"),
            scope=state.get("scope"),
            expires_at=state.get("expires_at"),
        )
        if force_refresh or invoke_status is not None:
            refresh_value = state.get("refresh_token")
            if not isinstance(refresh_value, str) or not refresh_value:
                reason = invoke_status or "force_refresh"
                raise AuthError(
                    "Nous Portal access token is not a usable inference JWT "
                    f"({reason}) and no refresh token is available. "
                    "Re-authenticate with: hermes auth add nous",
                    provider="nous",
                    code=reason,
                    relogin_required=True,
                )
            refreshed = await _refresh_access_token(
                client=client,
                portal_base_url=state["portal_base_url"],
                client_id=state["client_id"],
                refresh_token=refresh_value,
            )
            now = datetime.now(timezone.utc)
            access_ttl = _coerce_ttl_seconds(refreshed.get("expires_in"))
            state["access_token"] = refreshed["access_token"]
            state["refresh_token"] = refreshed.get("refresh_token") or refresh_value
            state["token_type"] = (
                refreshed.get("token_type") or state.get("token_type") or "Bearer"
            )
            state["scope"] = refreshed.get("scope") or state.get("scope")
            refreshed_url = _validate_nous_inference_url_from_network(
                refreshed.get("inference_base_url")
            )
            state["inference_base_url"] = refreshed_url or DEFAULT_NOUS_INFERENCE_URL
            state["obtained_at"] = now.isoformat()
            state["expires_in"] = access_ttl
            state["expires_at"] = datetime.fromtimestamp(
                now.timestamp() + access_ttl,
                tz=timezone.utc,
            ).isoformat()
            if on_state_update is not None:
                update_result = on_state_update(
                    dict(state),
                    "post_refresh_access_token",
                )
                if inspect.isawaitable(update_result):
                    await update_result
        _assert_nous_inference_jwt_usable(state)
        _select_nous_invoke_jwt(state)
    return state


async def refresh_nous_oauth_from_state(
    state: Dict[str, Any],
    *,
    timeout_seconds: float = 15.0,
    force_refresh: bool = False,
    on_state_update: Optional[
        Callable[[Dict[str, Any], str], Optional[Awaitable[None]]]
    ] = None,
) -> Dict[str, Any]:
    tls = state.get("tls") or {}
    return await refresh_nous_oauth_pure(
        state.get("access_token", ""),
        state.get("refresh_token", ""),
        state.get("client_id", DEFAULT_NOUS_CLIENT_ID),
        state.get("portal_base_url", DEFAULT_NOUS_PORTAL_URL),
        state.get("inference_base_url", DEFAULT_NOUS_INFERENCE_URL),
        token_type=state.get("token_type", "Bearer"),
        scope=state.get("scope", DEFAULT_NOUS_SCOPE),
        obtained_at=state.get("obtained_at"),
        expires_at=state.get("expires_at"),
        agent_key=state.get("agent_key"),
        agent_key_expires_at=state.get("agent_key_expires_at"),
        timeout_seconds=timeout_seconds,
        insecure=tls.get("insecure"),
        ca_bundle=tls.get("ca_bundle"),
        force_refresh=force_refresh,
        on_state_update=on_state_update,
    )


async def resolve_nous_runtime_credentials(
    *,
    timeout_seconds: float = 15.0,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Resolve and persist a usable Nous inference JWT with native async I/O."""
    local_path = await _auth_file_path()
    local_store = await _load_auth_store(local_path)
    target_path: Optional[Path] = local_path
    if _load_provider_state(local_store, "nous") is None:
        global_path = await _global_auth_file_path()
        global_store = await _load_global_auth_store()
        if global_path is not None and _load_provider_state(global_store, "nous"):
            target_path = global_path

    lock_timeout = max(
        float(AUTH_LOCK_TIMEOUT_SECONDS),
        float(timeout_seconds) + 5.0,
    )
    sequence_id = uuid.uuid4().hex[:12]
    async with _auth_store_transaction(
        target_path,
        timeout_seconds=lock_timeout,
    ):
        auth_store = await _load_auth_store(target_path)
        state = _load_provider_state(auth_store, "nous")
        if not state:
            raise AuthError(
                "Hermes is not logged into Nous Portal.",
                provider="nous",
                relogin_required=True,
            )
        persisted_state = dict(state)

        def resolve_routing() -> tuple[str, str, str, str]:
            portal_url = (
                _optional_base_url(state.get("portal_base_url"))
                or DEFAULT_NOUS_PORTAL_URL
            ).rstrip("/")
            env_portal = _nous_portal_env_override()
            if env_portal:
                portal_url = env_portal.rstrip("/")
            else:
                parsed = urlparse(portal_url)
                loopback_http = parsed.scheme == "http" and parsed.hostname in {
                    "localhost",
                    "127.0.0.1",
                }
                if (
                    not parsed.hostname
                    or parsed.hostname not in _NOUS_PORTAL_ALLOWED_HOSTS
                    or (parsed.scheme != "https" and not loopback_http)
                ):
                    logger.warning(
                        "auth: ignoring invalid portal_base_url %r "
                        "(host %r or scheme not allowed), using default",
                        portal_url,
                        parsed.hostname,
                    )
                    portal_url = DEFAULT_NOUS_PORTAL_URL
            stored_inference = (
                _validate_nous_inference_url_from_network(
                    _optional_base_url(state.get("inference_base_url"))
                )
                or DEFAULT_NOUS_INFERENCE_URL
            )
            effective_inference = _nous_inference_env_override() or stored_inference
            return (
                portal_url,
                stored_inference,
                effective_inference,
                str(state.get("client_id") or DEFAULT_NOUS_CLIENT_ID),
            )

        try:
            shared_path: Optional[Path] = await _nous_shared_store_path()
        except RuntimeError:
            shared_path = None

        async with AsyncExitStack() as stack:
            if shared_path is not None:
                await stack.enter_async_context(
                    _auth_store_transaction(
                        shared_path,
                        timeout_seconds=lock_timeout,
                    )
                )
                await _merge_shared_nous_oauth_state(state)

            portal_url, stored_inference, inference_url, client_id = resolve_routing()
            state["portal_base_url"] = portal_url
            state["inference_base_url"] = stored_inference
            state["client_id"] = client_id
            tls = state.get("tls") if isinstance(state.get("tls"), dict) else {}
            effective_insecure = (
                insecure if insecure is not None else tls.get("insecure")
            )
            effective_ca = ca_bundle or tls.get("ca_bundle")
            state["tls"] = {
                "insecure": bool(effective_insecure),
                "ca_bundle": effective_ca,
            }

            async def persist_refresh(
                updated: Dict[str, Any],
                _reason: str,
            ) -> None:
                merged = dict(state)
                merged.update(updated)
                state.clear()
                state.update(merged)
                _save_provider_state(auth_store, "nous", state)
                await _save_auth_store(auth_store, target_path)
                if shared_path is not None:
                    try:
                        await _save_shared_nous_state(state)
                    except Exception as exc:
                        logger.debug("Failed to mirror refreshed Nous state: %s", exc)

            refresh_error: AuthError | None = None
            try:
                refreshed = await refresh_nous_oauth_from_state(
                    state,
                    timeout_seconds=timeout_seconds,
                    force_refresh=force_refresh,
                    on_state_update=persist_refresh,
                )
            except AuthError as exc:
                refresh_error = exc

            if refresh_error is not None:
                if _is_terminal_nous_refresh_error(refresh_error):
                    failed_refresh = str(state.get("refresh_token") or "")
                    current = _load_provider_state(auth_store, "nous") or {}
                    current_refresh = str(current.get("refresh_token") or "")
                    if not current_refresh or current_refresh == failed_refresh:
                        _quarantine_nous_oauth_state(
                            state,
                            refresh_error,
                            reason="runtime_access_refresh_failure",
                        )
                        _quarantine_nous_pool_entries(
                            auth_store,
                            refresh_error,
                            reason="runtime_access_refresh_failure",
                        )
                        _save_provider_state(auth_store, "nous", state)
                        await _save_auth_store(auth_store, target_path)
                        if shared_path is not None:
                            try:
                                await aiofiles.os.remove(shared_path)
                            except FileNotFoundError:
                                pass
                            except OSError as remove_exc:
                                logger.debug(
                                    "Failed to clear shared Nous auth store: %s",
                                    remove_exc,
                                )
                raise refresh_error

            state.update(refreshed)
            state["portal_base_url"] = portal_url
            state["inference_base_url"] = stored_inference
            state["client_id"] = client_id
            verify = await _resolve_client_verify(
                insecure=effective_insecure,
                ca_bundle=effective_ca,
                auth_state=state,
            )
            state["tls"] = {
                "insecure": verify is False,
                "ca_bundle": effective_ca,
            }
            if _nous_effective_provider_state(state) != _nous_effective_provider_state(
                persisted_state
            ):
                _save_provider_state(auth_store, "nous", state)
                await _save_auth_store(auth_store, target_path)
                if shared_path is not None:
                    try:
                        await _save_shared_nous_state(state)
                    except Exception as exc:
                        logger.debug("Failed to mirror resolved Nous state: %s", exc)

    api_key = state.get("agent_key")
    if not isinstance(api_key, str) or not api_key:
        raise AuthError(
            "Failed to resolve a Nous inference API key",
            provider="nous",
            code="server_error",
        )
    expires_at = state.get("agent_key_expires_at")
    expires_epoch = _parse_iso_timestamp(expires_at)
    expires_in = (
        max(0, int(expires_epoch - time.time()))
        if expires_epoch is not None
        else _coerce_ttl_seconds(state.get("agent_key_expires_in"))
    )
    _oauth_trace(
        "nous_runtime_credentials_resolved",
        sequence_id=sequence_id,
        access_token_fp=_token_fingerprint(api_key),
    )
    return {
        "provider": "nous",
        "base_url": inference_url,
        "api_key": api_key,
        "key_id": state.get("agent_key_id"),
        "expires_at": expires_at,
        "expires_in": expires_in,
        "source": NOUS_AUTH_PATH_INVOKE_JWT,
        "auth_path": NOUS_AUTH_PATH_INVOKE_JWT,
        "state_path": str(target_path or local_path),
    }


_RESOLVE_TOKEN_CACHE: tuple[float, str] | None = None
_RESOLVE_TOKEN_CACHE_TTL_S = 5.0


async def resolve_nous_access_token(
    *,
    timeout_seconds: float = 15.0,
    insecure: Optional[bool] = None,
    ca_bundle: Optional[str] = None,
    refresh_skew_seconds: int = ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
) -> str:
    """Resolve a refresh-aware Nous Portal access token for account lookup."""
    global _RESOLVE_TOKEN_CACHE
    if not insecure and ca_bundle is None and _RESOLVE_TOKEN_CACHE is not None:
        cached_at, cached_token = _RESOLVE_TOKEN_CACHE
        if (time.monotonic() - cached_at) < _RESOLVE_TOKEN_CACHE_TTL_S:
            return cached_token

    local_path = await _auth_file_path()
    local_store = await _load_auth_store(local_path)
    target_path: Optional[Path] = local_path
    if _load_provider_state(local_store, "nous") is None:
        global_path = await _global_auth_file_path()
        global_store = await _load_global_auth_store()
        if global_path is not None and _load_provider_state(global_store, "nous"):
            target_path = global_path

    lock_timeout = max(
        float(AUTH_LOCK_TIMEOUT_SECONDS),
        float(timeout_seconds) + 5.0,
    )
    async with _auth_store_transaction(
        target_path,
        timeout_seconds=lock_timeout,
    ):
        if not insecure and ca_bundle is None and _RESOLVE_TOKEN_CACHE is not None:
            cached_at, cached_token = _RESOLVE_TOKEN_CACHE
            if (time.monotonic() - cached_at) < _RESOLVE_TOKEN_CACHE_TTL_S:
                return cached_token

        auth_store = await _load_auth_store(target_path)
        state = _load_provider_state(auth_store, "nous")
        if not state:
            raise AuthError(
                "Hermes is not logged into Nous Portal.",
                provider="nous",
                relogin_required=True,
            )
        persisted_state = dict(state)

        try:
            shared_path: Optional[Path] = await _nous_shared_store_path()
        except RuntimeError:
            shared_path = None

        async with AsyncExitStack() as stack:
            if shared_path is not None:
                await stack.enter_async_context(
                    _auth_store_transaction(
                        shared_path,
                        timeout_seconds=lock_timeout,
                    )
                )
                await _merge_shared_nous_oauth_state(state)

            env_portal_override = _nous_portal_env_override()
            if env_portal_override:
                portal_base_url = env_portal_override.rstrip("/")
            else:
                portal_base_url = (
                    _optional_base_url(state.get("portal_base_url"))
                    or DEFAULT_NOUS_PORTAL_URL
                ).rstrip("/")
                parsed_portal_url = urlparse(portal_base_url)
                loopback_http = (
                    parsed_portal_url.scheme == "http"
                    and parsed_portal_url.hostname in {"localhost", "127.0.0.1"}
                )
                if (
                    not parsed_portal_url.hostname
                    or parsed_portal_url.hostname not in _NOUS_PORTAL_ALLOWED_HOSTS
                    or (parsed_portal_url.scheme != "https" and not loopback_http)
                ):
                    logger.warning(
                        "auth: ignoring invalid portal_base_url %r "
                        "(host %r or scheme not allowed), using default",
                        portal_base_url,
                        parsed_portal_url.hostname,
                    )
                    portal_base_url = DEFAULT_NOUS_PORTAL_URL

            client_id = str(state.get("client_id") or DEFAULT_NOUS_CLIENT_ID)
            tls = state.get("tls") if isinstance(state.get("tls"), dict) else {}
            effective_insecure = (
                insecure if insecure is not None else tls.get("insecure")
            )
            effective_ca = ca_bundle or tls.get("ca_bundle")
            verify = await _resolve_client_verify(
                insecure=effective_insecure,
                ca_bundle=effective_ca,
                auth_state=state,
            )
            access_token = state.get("access_token")
            refresh_token = state.get("refresh_token")
            if not isinstance(access_token, str) or not access_token:
                raise AuthError(
                    "No access token found for Nous Portal login.",
                    provider="nous",
                    relogin_required=True,
                )

            if _is_expiring(state.get("expires_at"), refresh_skew_seconds):
                if not isinstance(refresh_token, str) or not refresh_token:
                    raise AuthError(
                        "Session expired and no refresh token is available.",
                        provider="nous",
                        relogin_required=True,
                    )
                timeout = httpx.Timeout(timeout_seconds if timeout_seconds else 15.0)
                async with (await _create_httpx_client(
                    timeout=timeout,
                    headers={"Accept": "application/json"},
                    verify=verify,
                )) as client:
                    refresh_error: AuthError | None = None
                    try:
                        refreshed = await _refresh_access_token(
                            client=client,
                            portal_base_url=portal_base_url,
                            client_id=client_id,
                            refresh_token=refresh_token,
                        )
                    except AuthError as exc:
                        refresh_error = exc

                    if refresh_error is not None:
                        if _is_terminal_nous_refresh_error(refresh_error):
                            _quarantine_nous_oauth_state(
                                state,
                                refresh_error,
                                reason="managed_access_token_refresh_failure",
                            )
                            _quarantine_nous_pool_entries(
                                auth_store,
                                refresh_error,
                                reason="managed_access_token_refresh_failure",
                            )
                            _save_provider_state(auth_store, "nous", state)
                            await _save_auth_store(auth_store, target_path)
                            if shared_path is not None:
                                try:
                                    await aiofiles.os.remove(shared_path)
                                except FileNotFoundError:
                                    pass
                                except OSError as remove_exc:
                                    logger.debug(
                                        "Failed to clear shared Nous auth store: %s",
                                        remove_exc,
                                    )
                        raise refresh_error

                now = datetime.now(timezone.utc)
                access_ttl = _coerce_ttl_seconds(refreshed.get("expires_in"))
                access_token = refreshed["access_token"]
                state["access_token"] = access_token
                state["refresh_token"] = refreshed.get("refresh_token") or refresh_token
                state["token_type"] = (
                    refreshed.get("token_type") or state.get("token_type") or "Bearer"
                )
                state["scope"] = refreshed.get("scope") or state.get("scope")
                state["obtained_at"] = now.isoformat()
                state["expires_in"] = access_ttl
                state["expires_at"] = datetime.fromtimestamp(
                    now.timestamp() + access_ttl,
                    tz=timezone.utc,
                ).isoformat()

            state["portal_base_url"] = portal_base_url
            state["client_id"] = client_id
            state["tls"] = {
                "insecure": verify is False,
                "ca_bundle": effective_ca,
            }
            if _nous_effective_provider_state(state) != _nous_effective_provider_state(
                persisted_state
            ):
                _save_provider_state(auth_store, "nous", state)
                await _save_auth_store(auth_store, target_path)
                if shared_path is not None:
                    try:
                        await _save_shared_nous_state(state)
                    except Exception as exc:
                        logger.debug("Failed to mirror resolved Nous state: %s", exc)

    if not insecure and ca_bundle is None:
        _RESOLVE_TOKEN_CACHE = (time.monotonic(), access_token)
    return access_token


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
_nous_auth_status_cache: Optional[
    Tuple[float, str, Optional[float], Dict[str, Any]]
] = None


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

_MINIMAX_OAUTH_ERROR_BODY_LIMIT = 16 * 1024


async def _minimax_response_error_text(
    response: httpx.Response,
    *,
    limit: int = _MINIMAX_OAUTH_ERROR_BODY_LIMIT,
) -> str:
    """Read and close a streamed MiniMax OAuth error response with a bound."""
    limit = max(0, int(limit))
    chunks: list[bytes] = []
    total = 0
    truncated = False
    try:
        if response.is_stream_consumed:
            text = response.text
            return text[:limit] + ("...[truncated]" if len(text) > limit else "")
        async for chunk in response.aiter_bytes():
            if not chunk:
                continue
            remaining = limit + 1 - total
            if remaining <= 0:
                truncated = True
                break
            if len(chunk) > remaining:
                chunks.append(chunk[:remaining])
                total += remaining
                truncated = True
                break
            chunks.append(chunk)
            total += len(chunk)
        raw = b"".join(chunks)
        if len(raw) > limit:
            raw = raw[:limit]
            truncated = True
        encoding = response.encoding or "utf-8"
        text = raw.decode(encoding, errors="replace")
        return text + ("...[truncated]" if truncated else "")
    finally:
        await response.aclose()


async def _minimax_post_form(
    client: httpx.AsyncClient,
    url: str,
    *,
    data: Dict[str, Any],
    headers: Dict[str, str],
) -> httpx.Response:
    """POST a MiniMax OAuth form without eagerly reading error bodies."""
    request = client.build_request("POST", url, data=data, headers=headers)
    response = await client.send(request, stream=True)
    if response.status_code == 200:
        await response.aread()
    return response


def _minimax_expired_in_looks_like_unix_ms(
    expired_in: int,
    *,
    now_ms: int,
) -> bool:
    """Return whether MiniMax supplied an absolute millisecond timestamp."""
    return int(expired_in) > (now_ms // 2)


def _minimax_resolve_token_expiry_unix(
    expired_in: int,
    *,
    now: datetime,
) -> float:
    raw = int(expired_in)
    now_ms = int(now.timestamp() * 1000)
    if _minimax_expired_in_looks_like_unix_ms(raw, now_ms=now_ms):
        return raw / 1000.0
    return now.timestamp() + max(1, raw)


async def _refresh_minimax_oauth_state(
    state: Dict[str, Any],
    *,
    timeout_seconds: float = 15.0,
    force: bool = False,
) -> Dict[str, Any]:
    """Refresh MiniMax OAuth state through native async file and HTTP I/O."""
    lock_timeout = max(
        float(AUTH_LOCK_TIMEOUT_SECONDS),
        float(timeout_seconds) + 5.0,
    )
    async with _auth_store_transaction(timeout_seconds=lock_timeout):
        auth_store = await _load_auth_store()
        current = _load_provider_state(auth_store, "minimax-oauth") or dict(state)
        refresh_token = str(current.get("refresh_token") or "").strip()
        if not refresh_token:
            raise AuthError(
                "MiniMax OAuth state has no refresh_token; please re-login.",
                provider="minimax-oauth",
                code="no_refresh_token",
                relogin_required=True,
            )
        try:
            expires_at = datetime.fromisoformat(
                str(current.get("expires_at") or "")
            ).timestamp()
        except Exception:
            expires_at = 0.0
        if not force and expires_at - time.time() > MINIMAX_OAUTH_REFRESH_SKEW_SECONDS:
            return current

        portal_base_url = str(current.get("portal_base_url") or "").rstrip("/")
        client_id = str(current.get("client_id") or "").strip()
        if not portal_base_url or not client_id:
            raise AuthError(
                "MiniMax OAuth state is missing portal_base_url or client_id; please re-login.",
                provider="minimax-oauth",
                code="refresh_state_incomplete",
                relogin_required=True,
            )

        async with (await _create_httpx_client(
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=True,
            verify=await _resolve_httpx_client_verify(),
        )) as client:
            response = await _minimax_post_form(
                client,
                f"{portal_base_url}/oauth/token",
                data={
                    "grant_type": "refresh_token",
                    "client_id": client_id,
                    "refresh_token": refresh_token,
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
            )
            if response.status_code != 200:
                body = (await _minimax_response_error_text(response)).strip()
                body_lower = body.lower()
                relogin = any(
                    marker in body_lower
                    for marker in (
                        "invalid_grant",
                        "refresh_token_reused",
                        "invalid_refresh_token",
                    )
                )
                raise AuthError(
                    f"MiniMax OAuth refresh failed: {body or response.reason_phrase}",
                    provider="minimax-oauth",
                    code="refresh_failed",
                    relogin_required=relogin,
                )

        payload = response.json()
        if payload.get("status") != "success":
            raise AuthError(
                "MiniMax OAuth refresh did not return success.",
                provider="minimax-oauth",
                code="refresh_failed",
                relogin_required=True,
            )
        now = datetime.now(timezone.utc)
        expires_at_unix = _minimax_resolve_token_expiry_unix(
            int(payload["expired_in"]),
            now=now,
        )
        refreshed = dict(current)
        refreshed.update({
            "access_token": payload["access_token"],
            "refresh_token": payload.get("refresh_token", refresh_token),
            "obtained_at": now.isoformat(),
            "expires_at": datetime.fromtimestamp(
                expires_at_unix,
                tz=timezone.utc,
            ).isoformat(),
            "expires_in": max(0, int(expires_at_unix - now.timestamp())),
        })
        _save_provider_state(auth_store, "minimax-oauth", refreshed)
        await _save_auth_store(auth_store)
        return refreshed


async def _minimax_oauth_quarantine_on_terminal_refresh(
    state: Dict[str, Any],
    exc: AuthError,
) -> None:
    if not (exc.relogin_required and state.get("refresh_token")):
        return
    async with _auth_store_transaction():
        auth_store = await _load_auth_store()
        current = _load_provider_state(auth_store, "minimax-oauth")
        if not current:
            return
        failed_refresh = str(state.get("refresh_token") or "").strip()
        current_refresh = str(current.get("refresh_token") or "").strip()
        if current_refresh and current_refresh != failed_refresh:
            return
        for key in (
            "access_token",
            "refresh_token",
            "expires_at",
            "expires_in",
            "obtained_at",
        ):
            current.pop(key, None)
        current["last_auth_error"] = {
            "provider": "minimax-oauth",
            "code": exc.code or "refresh_failed",
            "message": str(exc),
            "reason": "runtime_refresh_failure",
            "relogin_required": True,
            "at": datetime.now(timezone.utc).isoformat(),
        }
        _save_provider_state(auth_store, "minimax-oauth", current)
        await _save_auth_store(auth_store)


def build_minimax_oauth_token_provider() -> Callable[[], Awaitable[str]]:
    """Return an awaitable token provider for native async consumers."""

    async def _provide() -> str:
        credentials = await resolve_minimax_oauth_runtime_credentials()
        return str(credentials["api_key"])

    return _provide


async def _resolve_minimax_oauth_runtime_credentials(
    *,
    min_token_ttl_seconds: int = MINIMAX_OAUTH_REFRESH_SKEW_SECONDS,
    as_token_provider: bool = False,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Resolve MiniMax OAuth credentials with internal refresh control."""
    auth_store = await _load_auth_store()
    state = _load_provider_state(auth_store, "minimax-oauth")
    if not state or not state.get("access_token"):
        raise AuthError(
            "Not logged into MiniMax OAuth. Run `hermes model` and select "
            "MiniMax (OAuth).",
            provider="minimax-oauth",
            code="not_logged_in",
            relogin_required=True,
        )
    refresh_error: AuthError | None = None
    try:
        state = await _refresh_minimax_oauth_state(
            state,
            force=force_refresh,
        )
    except AuthError as exc:
        refresh_error = exc
    if refresh_error is not None:
        await _minimax_oauth_quarantine_on_terminal_refresh(state, refresh_error)
        raise refresh_error

    api_key: Any
    if as_token_provider:
        api_key = build_minimax_oauth_token_provider()
    else:
        api_key = state["access_token"]
    return {
        "provider": "minimax-oauth",
        "api_key": api_key,
        "base_url": str(state["inference_base_url"]).rstrip("/"),
        "source": "oauth",
    }


async def resolve_minimax_oauth_runtime_credentials(
    *,
    min_token_ttl_seconds: int = MINIMAX_OAUTH_REFRESH_SKEW_SECONDS,
    as_token_provider: bool = False,
) -> Dict[str, Any]:
    """Resolve a current MiniMax OAuth credential without blocking the loop."""
    return await _resolve_minimax_oauth_runtime_credentials(
        min_token_ttl_seconds=min_token_ttl_seconds,
        as_token_provider=as_token_provider,
    )
