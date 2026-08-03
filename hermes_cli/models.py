"""
Canonical model catalogs and lightweight validation helpers.

Add, remove, or reorder entries here — both `hermes setup` and
`hermes` provider-selection will pick up the change automatically.
"""

from __future__ import annotations

import json
import asyncio
import os
import re
import urllib.parse
import urllib.request
import urllib.error
import time
import aiofiles
import aiofiles.os
import httpx
from difflib import get_close_matches
from pathlib import Path
from typing import Any, NamedTuple, Optional

from hermes_cli import __version__ as _HERMES_VERSION
from hermes_cli.urllib_security import open_credentialed_url

# Identify ourselves so endpoints fronted by Cloudflare's Browser Integrity
# Check (error 1010) don't reject the default ``Python-urllib/*`` signature.
_HERMES_USER_AGENT = f"hermes-cli/{_HERMES_VERSION}"

COPILOT_BASE_URL = "https://api.githubcopilot.com"
COPILOT_MODELS_URL = f"{COPILOT_BASE_URL}/models"
COPILOT_EDITOR_VERSION = "vscode/1.104.1"
COPILOT_REASONING_EFFORTS_GPT5 = ["minimal", "low", "medium", "high"]
COPILOT_REASONING_EFFORTS_O_SERIES = ["low", "medium", "high"]



# Fallback OpenRouter snapshot used when the live catalog is unavailable.
# (model_id, display description shown in menus)
OPENROUTER_MODELS: list[tuple[str, str]] = [
    # Anthropic
    ("anthropic/claude-fable-5",               ""),
    ("anthropic/claude-opus-5",                ""),
    ("anthropic/claude-opus-5-fast",           "2x price, higher output speed"),
    ("anthropic/claude-opus-4.8",              ""),
    ("anthropic/claude-opus-4.8-fast",         "2x price, higher output speed"),
    ("anthropic/claude-sonnet-5",              ""),
    ("anthropic/claude-haiku-4.5",             ""),
    # OpenAI
    ("openai/gpt-5.6-sol",                     ""),
    ("openai/gpt-5.6-sol-pro",                 ""),
    ("openai/gpt-5.6-terra",                   ""),
    ("openai/gpt-5.6-terra-pro",               ""),
    ("openai/gpt-5.6-luna",                    ""),
    ("openai/gpt-5.6-luna-pro",                ""),
    ("openai/gpt-5.5",                         ""),
    ("openai/gpt-5.5-pro",                     ""),
    ("openai/gpt-5.4-mini",                    ""),
    # Google
    ("google/gemini-3.1-pro-preview",          ""),
    ("google/gemini-3.6-flash",                ""),
    # xAI
    ("x-ai/grok-4.5",                          ""),
    # DeepSeek
    ("deepseek/deepseek-v4-pro",               ""),
    ("deepseek/deepseek-v4-flash",             ""),
    # Qwen
    ("qwen/qwen3.7-max",                       ""),
    # MoonshotAI
    ("moonshotai/kimi-k3",                     "recommended"),
    # MiniMax
    ("minimax/minimax-m3",                     ""),
    # Z-AI
    ("z-ai/glm-5.2",                           "default"),
    ("z-ai/glm-5.1",                           ""),
    # Xiaomi
    ("xiaomi/mimo-v2.5-pro",                   ""),
    # Tencent
    ("tencent/hy3",                            ""),
    # StepFun
    ("stepfun/step-3.7-flash",                 ""),
    # NVIDIA
    ("nvidia/nemotron-3-super-120b-a12b",      ""),
    # Sakana
    ("sakana/fugu-ultra",                      ""),
    # OpenRouter routers
    ("openrouter/pareto-code",                 "auto-routes to cheapest coder meeting openrouter.min_coding_score"),
    # Free tier
    ("openrouter/elephant-alpha",              "free"),
    ("poolside/laguna-m.1:free",               "free"),
    ("tencent/hy3:free",                       "free"),
    ("nvidia/nemotron-3-super-120b-a12b:free", "free"),
    ("nvidia/nemotron-3-ultra-550b-a55b:free", "free"),
    ("inclusionai/ring-2.6-1t:free",           "free"),
]

_openrouter_catalog_cache: list[tuple[str, str]] | None = None


# Fallback Vercel AI Gateway snapshot used when the live catalog is unavailable.
# OSS / open-weight models prioritized first, then closed-source by family.
# Slugs match Vercel's actual /v1/models catalog (e.g. alibaba/ for Qwen,
# zai/ and xai/ without hyphens).
VERCEL_AI_GATEWAY_MODELS: list[tuple[str, str]] = [
    ("moonshotai/kimi-k2.6",                 "recommended"),
    ("alibaba/qwen3.6-plus",                 ""),
    ("zai/glm-5.1",                          ""),
    ("minimax/minimax-m2.7",                 ""),
    ("anthropic/claude-sonnet-4.6",          ""),
    ("anthropic/claude-opus-4.7",            ""),
    ("anthropic/claude-opus-4.6",            ""),
    ("anthropic/claude-haiku-4.5",           ""),
    ("openai/gpt-5.4",                       ""),
    ("openai/gpt-5.4-mini",                  ""),
    ("openai/gpt-5.3-codex",                 ""),
    ("google/gemini-3.1-pro-preview",        ""),
    ("google/gemini-3-flash",                ""),
    ("google/gemini-3.1-flash-lite-preview", ""),
    ("xai/grok-4.20-reasoning",              ""),
]

_ai_gateway_catalog_cache: list[tuple[str, str]] | None = None




# Static fallback for xAI when the models.dev disk cache is empty (fresh
# install, offline first run, etc.). Mirrors the xAI-direct model IDs from
# $HERMES_HOME/models_dev_cache.json as of 2026-04-28. Whenever xAI renames
# or retires a model, the disk cache picks it up on the next refresh and the
# fallback here only matters until that refresh lands.
#
# Models retired by xAI on May 15, 2026 are excluded — see
# https://docs.x.ai/developers/migration/may-15-retirement
# (grok-4, grok-4-0709, grok-4-fast{,-reasoning,-non-reasoning},
#  grok-4-1-fast{,-reasoning,-non-reasoning}, grok-code-fast-1 → grok-4.3).
_XAI_STATIC_FALLBACK: list[str] = [
    "grok-build-0.1",
    "grok-4.5",
    "grok-4.3",
    "grok-4.20-0309-reasoning",
    "grok-4.20-0309-non-reasoning",
    "grok-4.20-multi-agent-0309",
]

# Callable via xAI OAuth but omitted from models.dev and /v1/models listings.
_XAI_CURATED_EXTRAS: list[str] = [
    "grok-4.5",  # GA 2026-07 — kept until the models.dev disk cache refreshes
    "grok-composer-2.5-fast",
]


_XAI_TOP_MODEL = "grok-build-0.1"


def _codex_curated_models() -> list[str]:
    from hermes_cli.codex_models import (
        DEFAULT_CODEX_MODELS,
        _add_forward_compat_models,
    )

    return _add_forward_compat_models(list(DEFAULT_CODEX_MODELS))


def _xai_curated_models() -> list[str]:
    models = [_XAI_TOP_MODEL, *_XAI_CURATED_EXTRAS, *_XAI_STATIC_FALLBACK]
    return list(dict.fromkeys(models))








_PROVIDER_MODELS: dict[str, list[str]] = {
    "moa": ["default"],
    "nous": [
        # Anthropic
        "anthropic/claude-fable-5",
        "anthropic/claude-opus-5",
        "anthropic/claude-opus-4.8",
        "anthropic/claude-sonnet-5",
        "anthropic/claude-haiku-4.5",
        # OpenAI
        "openai/gpt-5.6-sol",
        "openai/gpt-5.6-sol-pro",
        "openai/gpt-5.6-terra",
        "openai/gpt-5.6-terra-pro",
        "openai/gpt-5.6-luna",
        "openai/gpt-5.6-luna-pro",
        "openai/gpt-5.5",
        "openai/gpt-5.5-pro",
        "openai/gpt-5.4-mini",
        # Google
        "google/gemini-3.1-pro-preview",
        "google/gemini-3.6-flash",
        # xAI
        "x-ai/grok-4.5",
        # DeepSeek
        "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-v4-flash",
        # Qwen
        "qwen/qwen3.7-max",
        # MoonshotAI
        "moonshotai/kimi-k3",
        # MiniMax
        "minimax/minimax-m3",
        # Z-AI
        "z-ai/glm-5.2",
        "z-ai/glm-5.1",
        # Xiaomi
        "xiaomi/mimo-v2.5-pro",
        # Tencent
        "tencent/hy3",
        # StepFun
        "stepfun/step-3.7-flash",
        # NVIDIA
        "nvidia/nemotron-3-super-120b-a12b",
        # Sakana
        "sakana/fugu-ultra",
    ],
    # Native OpenAI Chat Completions (api.openai.com). Used by /model counts and
    # provider_model_ids fallback when /v1/models is unavailable.
    "openai": [
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5-mini",
        "gpt-5.3-codex",
        "gpt-5.2-codex",
        "gpt-4.1",
        "gpt-4o",
        "gpt-4o-mini",
    ],
    "openai-api": [
        "gpt-5.6-sol",
        "gpt-5.6-sol-pro",
        "gpt-5.6-terra",
        "gpt-5.6-terra-pro",
        "gpt-5.6-luna",
        "gpt-5.6-luna-pro",
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5-mini",
        "gpt-5.3-codex",
        "gpt-4.1",
        "gpt-4o",
        "gpt-4o-mini",
    ],
    "openai-codex": _codex_curated_models(),
    "xai-oauth": _xai_curated_models(),
    "copilot-acp": [
        "copilot-acp",
    ],
    "copilot": [
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5-mini",
        "gpt-5.3-codex",
        "gpt-5.2-codex",
        "gpt-4.1",
        "gpt-4o",
        "gpt-4o-mini",
        "claude-sonnet-4.6",
        "claude-sonnet-5",
        "claude-sonnet-4",
        "claude-sonnet-4.5",
        "claude-haiku-4.5",
        "gemini-3.1-pro-preview",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
    ],
    "gemini": [
        "gemini-3.1-pro-preview",
        "gemini-3-pro-preview",
        "gemini-3.6-flash",
        "gemini-3.1-flash-lite-preview",
    ],
    "zai": [
        "glm-5.2",
        "glm-5.1",
        "glm-5",
        "glm-5v-turbo",
        "glm-5-turbo",
        "glm-4.7",
        "glm-4.5",
        "glm-4.5-flash",
    ],
    "xai": _xai_curated_models(),
    "nvidia": [
        # NVIDIA flagship reasoning models
        "nvidia/nemotron-3-ultra-550b-a55b",
        "nvidia/nemotron-3-super-120b-a12b",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        # Third-party agentic models hosted on build.nvidia.com
        # (map to OpenRouter defaults — users get familiar picks on NIM)
        "z-ai/glm-5.2",
        "moonshotai/kimi-k2.6",
        "minimaxai/minimax-m3",
    ],
    "kimi-coding": [
        "kimi-k3",
        "kimi-k2.7-code",
        "kimi-k2.6",
        "kimi-k2.5",
        "kimi-for-coding",
        "kimi-for-coding-highspeed",
        "kimi-k2-thinking",
        "kimi-k2-thinking-turbo",
        "kimi-k2-turbo-preview",
        "kimi-k2-0905-preview",
    ],
    "kimi-coding-cn": [
        "kimi-k3",
        "kimi-k2.7-code",
        "kimi-k2.7-code-highspeed",
        "kimi-k2.6",
        "kimi-k2.5",
        "kimi-k2-thinking",
        "kimi-k2-turbo-preview",
        "kimi-k2-0905-preview",
    ],
    "stepfun": [
        "step-3.5-flash",
        "step-3.5-flash-2603",
    ],
    "moonshot": [
        "kimi-k3",
        "kimi-k2.6",
        "kimi-k2.5",
        "kimi-k2-thinking",
        "kimi-k2-turbo-preview",
        "kimi-k2-0905-preview",
    ],
    "minimax": [
        "MiniMax-M3",
        "MiniMax-M2.7",
        "MiniMax-M2.5",
        "MiniMax-M2.1",
        "MiniMax-M2",
    ],
    "minimax-oauth": [
        "MiniMax-M3",
        "MiniMax-M2.7",
        "MiniMax-M2.7-highspeed",
    ],
    "minimax-cn": [
        "MiniMax-M3",
        "MiniMax-M2.7",
        "MiniMax-M2.5",
        "MiniMax-M2.1",
        "MiniMax-M2",
    ],
    "anthropic": [
        "claude-fable-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-haiku-4-5-20251001",
    ],
    "deepseek": [
        "deepseek-v4-pro",
        "deepseek-v4-flash",
    ],
    "xiaomi": [
        "mimo-v2.5-pro",
        "mimo-v2.5",
        "mimo-v2-pro",
        "mimo-v2-omni",
        "mimo-v2-flash",
    ],
    "tencent-tokenhub": [
        "hy3-preview",
    ],
    "arcee": [
        "trinity-large-thinking",
        "trinity-large-preview",
        "trinity-mini",
    ],
    "gmi": [
        "zai-org/GLM-5.1-FP8",
        "deepseek-ai/DeepSeek-V3.2",
        "moonshotai/Kimi-K2.5",
        "google/gemini-3.1-flash-lite-preview",
        "anthropic/claude-sonnet-5",
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5.4",
    ],
    "opencode-zen": [
        "kimi-k2.5",
        "kimi-k2.6",
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.4-pro",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.3-codex",
        "gpt-5.3-codex-spark",
        "gpt-5.2",
        "gpt-5.2-codex",
        "gpt-5.1",
        "gpt-5.1-codex",
        "gpt-5.1-codex-max",
        "gpt-5.1-codex-mini",
        "gpt-5",
        "gpt-5-codex",
        "gpt-5-nano",
        "claude-fable-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-opus-4-5",
        "claude-opus-4-1",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-sonnet-4",
        "claude-haiku-4-5",
        "gemini-3.5-flash",
        "gemini-3.1-pro",
        "gemini-3-flash",
        "minimax-m3",
        "minimax-m2.7",
        "minimax-m2.5",
        "glm-5.2",
        "glm-5.1",
        "glm-5",
        "kimi-k2.7-code",
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "deepseek-v4-flash-free",
        "qwen3.7-plus",
        "qwen3.6-plus",
        "qwen3.5-plus",
        "grok-build-0.1",
        "big-pickle",
        "mimo-v2.5-free",
        "north-mini-code-free",
        "nemotron-3-ultra-free",
    ],
    "opencode-go": [
        "kimi-k3",
        "kimi-k2.7-code",
        "kimi-k2.6",
        "kimi-k2.5",
        "glm-5.2",
        "glm-5.1",
        "glm-5",
        "mimo-v2.5-pro",
        "mimo-v2.5",
        "mimo-v2-pro",
        "mimo-v2-omni",
        "minimax-m3",
        "minimax-m2.7",
        "minimax-m2.5",
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.6-plus",
        "qwen3.5-plus",
    ],
    "kilocode": [
        "anthropic/claude-opus-4.6",
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5.4",
        "google/gemini-3-pro-preview",
        "google/gemini-3-flash-preview",
    ],
    # Alibaba DashScope Coding platform (coding-intl) — default endpoint.
    # Supports Qwen models + third-party providers (GLM, Kimi, MiniMax).
    # Users with classic DashScope keys should override DASHSCOPE_BASE_URL
    # to https://dashscope-intl.aliyuncs.com/compatible-mode/v1 (OpenAI-compat)
    # or https://dashscope-intl.aliyuncs.com/apps/anthropic (Anthropic-compat).
    "alibaba": [
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.6-plus",
        "kimi-k2.5",
        "qwen3.5-plus",
        "qwen3-coder-plus",
        "qwen3-coder-next",
        # Third-party models available on coding-intl
        "glm-5",
        "glm-4.7",
        "MiniMax-M2.5",
    ],
    # Alibaba Coding Plan — same platform as alibaba (DashScope coding-intl),
    # separate provider ID with its own base_url_env_var.
    "alibaba-coding-plan": [
        "qwen3.7-plus",
        "qwen3.6-plus",
        "qwen3.5-plus",
        "qwen3-max-2026-01-23",
        "qwen3-coder-plus",
        "qwen3-coder-next",
        "kimi-k2.5",
        "glm-5",
        "glm-4.7",
        "MiniMax-M2.5",
    ],
    # Curated HF model list — only agentic models that map to OpenRouter defaults.
    "huggingface": [
        "moonshotai/Kimi-K2.5",
        "Qwen/Qwen3.5-397B-A17B",
        "Qwen/Qwen3.5-35B-A3B",
        "deepseek-ai/DeepSeek-V3.2",
        "MiniMaxAI/MiniMax-M2.5",
        "zai-org/GLM-5",
        "XiaomiMiMo/MiMo-V2-Flash",
        "moonshotai/Kimi-K2-Thinking",
        "moonshotai/Kimi-K2.6",
    ],
    # AWS Bedrock — static fallback list used when dynamic discovery is
    # unavailable (no boto3, no credentials, or API error).  The agent
    # prefers live discovery via ListFoundationModels + ListInferenceProfiles.
    # Use inference profile IDs (us.*) since most models require them.
    "bedrock": [
        "us.anthropic.claude-sonnet-5",
        "us.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-opus-4-6-v1",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "us.amazon.nova-pro-v1:0",
        "us.amazon.nova-lite-v1:0",
        "us.amazon.nova-micro-v1:0",
        "deepseek.v3.2",
        "us.meta.llama4-maverick-17b-instruct-v1:0",
        "us.meta.llama4-scout-17b-instruct-v1:0",
    ],
    # Azure Foundry: user-provided endpoint and model.
    # Empty list because models depend on the endpoint configuration.
    "azure-foundry": [],
    # Google Vertex AI — static curated list.  Vertex's OpenAI-compatible
    # endpoint has no /models listing route, so without this entry the
    # /model picker only ever shows the currently-configured model.
    # Model IDs use the "google/" publisher prefix Vertex's openapi
    # endpoint expects (see hermes_cli/model_setup_flows.py).
    # Entries validated live against a GCP project (global region,
    # HTTP 200) as of 2026-07-21 (PR #68767).
    "vertex": [
        "google/gemini-3.1-pro-preview",
        "google/gemini-3-pro-preview",
        "google/gemini-3.6-flash",
        "google/gemini-3.5-flash",
        "google/gemini-3.5-flash-lite",
        "google/gemini-3-flash-preview",
        "google/gemini-3.1-flash-lite-preview",
        "google/gemini-3.1-flash-lite",
    ],
    "novita": [
        "moonshotai/kimi-k2.5",
        "minimax/minimax-m2.7",
        "zai-org/glm-5",
        "deepseek/deepseek-v3-0324",
        "deepseek/deepseek-r1-0528",
        "qwen/qwen3-235b-a22b-fp8",
    ],
}

# Vercel AI Gateway: derive the bare-model-id catalog from the curated
# ``VERCEL_AI_GATEWAY_MODELS`` snapshot so both the picker (tuples with descriptions)
# and the static fallback catalog (bare ids) stay in sync from a single
# source of truth.
_PROVIDER_MODELS["ai-gateway"] = [mid for mid, _ in VERCEL_AI_GATEWAY_MODELS]

# ---------------------------------------------------------------------------
# Nous Portal free-model helper
# ---------------------------------------------------------------------------
# The Nous Portal models endpoint is the source of truth for which models
# are currently offered (free or paid). We trust whatever it returns and
# surface it to users as-is — no local allowlist filtering.


def _is_model_free(model_id: str, pricing: dict[str, dict[str, str]]) -> bool:
    """Return True if *model_id* has zero-cost prompt AND completion pricing."""
    p = pricing.get(model_id)
    if not p:
        return False
    try:
        return float(p.get("prompt", "1")) == 0 and float(p.get("completion", "1")) == 0
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Nous Portal account tier detection
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# TTL cache for free-tier detection — avoids repeated API calls within a
# session while still picking up upgrades quickly.
# ---------------------------------------------------------------------------
_FREE_TIER_CACHE_TTL: int = 180  # seconds (3 minutes)
_free_tier_cache: tuple[bool, float] | None = None  # (result, timestamp)




# ---------------------------------------------------------------------------
# Nous Portal recommended models
#
# The Portal publishes a curated list of suggested models (separated into
# paid and free tiers) plus dedicated recommendations for compaction (text
# summarisation / auxiliary) and vision tasks. We fetch it once per process
# with a TTL cache so callers can ask "what's the best aux model right now?"
# without hitting the network on every lookup.
#
# Shape of the response (fields we care about):
#   {
#     "paidRecommendedModels":     [ {modelName, ...}, ... ],
#     "freeRecommendedModels":     [ {modelName, ...}, ... ],
#     "paidRecommendedCompactionModel":  {modelName, ...} | null,
#     "paidRecommendedVisionModel":      {modelName, ...} | null,
#     "freeRecommendedCompactionModel":  {modelName, ...} | null,
#     "freeRecommendedVisionModel":      {modelName, ...} | null,
#   }
# ---------------------------------------------------------------------------

NOUS_RECOMMENDED_MODELS_PATH = "/api/nous/recommended-models"
_NOUS_RECOMMENDED_CACHE_TTL: int = 600  # seconds (10 minutes)
# (result_dict, timestamp) keyed by portal_base_url so staging vs prod don't collide.
_nous_recommended_cache: dict[str, tuple[dict[str, Any], float]] = {}












# ---------------------------------------------------------------------------
# Canonical provider list — single source of truth for provider identity.
# Every code path that lists, displays, or iterates providers derives from
# this list:  hermes model, /model, list_authenticated_providers.
#
# Fields:
#   slug        — internal provider ID (used in config.yaml, --provider flag)
#   label       — short display name
#   tui_desc    — longer description for the `hermes model` interactive picker
# ---------------------------------------------------------------------------

class ProviderEntry(NamedTuple):
    slug: str
    label: str
    tui_desc: str   # detailed description for `hermes model` TUI

CANONICAL_PROVIDERS: list[ProviderEntry] = [
    ProviderEntry("nous",           "Nous Portal",              "Nous Portal (Everything your agent needs, 300+ models with bundled tool use)"),
    ProviderEntry("fireworks",      "Fireworks AI",             "Fireworks AI (OpenAI-compatible direct model API)"),
    ProviderEntry("openrouter",     "OpenRouter",               "OpenRouter (Pay-per-use API aggregator)"),
    ProviderEntry("moa",            "Mixture of Agents",        "Mixture of Agents (named presets; aggregator acts after reference models)"),
    ProviderEntry("novita",         "NovitaAI",                 "NovitaAI (Cloud: Model API, Agent Sandbox, GPU Cloud)"),
    ProviderEntry("lmstudio",       "LM Studio",                "LM Studio (Local desktop app with built-in model server)"),
    ProviderEntry("anthropic",      "Anthropic",                "Anthropic (Claude models via API key or Claude Code)"),
    ProviderEntry("openai-codex",   "OpenAI Codex",             "OpenAI Codex (Codex CLI via ChatGPT subscription or API key)"),
    ProviderEntry("openai-api",     "OpenAI API",               "OpenAI API (api.openai.com, API key)"),
    ProviderEntry("alibaba",        "Qwen Cloud",               "Qwen Cloud / DashScope (Qwen + multi-provider)"),
    ProviderEntry("xai-oauth",      "xAI Grok OAuth (SuperGrok / Premium+)", "xAI Grok OAuth (SuperGrok / Premium+ subscription)"),
    ProviderEntry("xiaomi",         "Xiaomi MiMo",              "Xiaomi MiMo (MiMo-V2.5 and V2 models: pro, omni, flash)"),
    ProviderEntry("tencent-tokenhub", "Tencent TokenHub",       "Tencent TokenHub (Hy3 Preview via tokenhub.tencentmaas.com)"),
    ProviderEntry("nvidia",         "NVIDIA NIM",               "NVIDIA NIM (Nemotron models via build.nvidia.com or local NIM)"),
    ProviderEntry("copilot",        "GitHub Copilot",           "GitHub Copilot (Uses GITHUB_TOKEN or gh auth token)"),
    ProviderEntry("copilot-acp",    "GitHub Copilot ACP",       "GitHub Copilot ACP (Spawns copilot --acp --stdio)"),
    ProviderEntry("huggingface",    "Hugging Face",             "Hugging Face Inference Providers"),
    ProviderEntry("gemini",         "Google AI Studio",         "Google AI Studio (Native Gemini API)"),
    ProviderEntry("vertex",         "Google Vertex AI",         "Google Vertex AI (Gemini via GCP; OAuth2 service account or ADC, GCP billing/quotas)"),
    ProviderEntry("deepseek",       "DeepSeek",                 "DeepSeek (V3, R1, coder, direct API)"),
    ProviderEntry("xai",            "xAI",                      "xAI Grok (Direct API)"),
    ProviderEntry("zai",            "Z.AI / GLM",               "Z.AI / GLM (Zhipu direct API)"),
    ProviderEntry("kimi-coding",    "Kimi / Kimi Coding Plan",  "Kimi Coding Plan (api.kimi.com & Moonshot API)"),
    ProviderEntry("kimi-coding-cn", "Kimi / Moonshot (China)",  "Kimi / Moonshot China (Domestic direct API)"),
    ProviderEntry("stepfun",        "StepFun Step Plan",       "StepFun Step Plan (Agent / coding models via Step Plan API)"),
    ProviderEntry("minimax",        "MiniMax",                  "MiniMax (Global direct API)"),
    ProviderEntry("minimax-oauth",  "MiniMax (OAuth)",          "MiniMax via OAuth browser login (Coding Plan, minimax.io)"),
    ProviderEntry("minimax-cn",     "MiniMax (China)",          "MiniMax China (Domestic direct API)"),
    ProviderEntry("ollama-cloud",   "Ollama Cloud",             "Ollama Cloud (Cloud-hosted open models, ollama.com)"),
    ProviderEntry("arcee",          "Arcee AI",                 "Arcee AI (Trinity models, direct API)"),
    ProviderEntry("gmi",            "GMI Cloud",                "GMI Cloud (Multi-model direct API)"),
    ProviderEntry("kilocode",       "Kilo Code",                "Kilo Code (Kilo Gateway API)"),
    ProviderEntry("opencode-zen",   "OpenCode Zen",             "OpenCode Zen (Curated models, pay-as-you-go)"),
    ProviderEntry("opencode-go",    "OpenCode Go",              "OpenCode Go (Open models subscription)"),
    ProviderEntry("bedrock",        "AWS Bedrock",              "AWS Bedrock (Claude, Nova, Llama, DeepSeek; IAM or API key)"),
    ProviderEntry("azure-foundry",  "Azure Foundry",            "Azure Foundry (OpenAI-style or Anthropic-style endpoint, your Azure AI deployment)"),
    ProviderEntry("ai-gateway",     "Vercel AI Gateway",        "Vercel AI Gateway (Multi-model aggregator)"),
    ProviderEntry("qwen-oauth",     "Qwen OAuth (Portal)",      "Qwen OAuth (Reuses local Qwen CLI login)"),
]

# Auto-extend CANONICAL_PROVIDERS with any provider registered in providers/
# that is not already in the list above.  Adding plugins/model-providers/<name>/
# is sufficient to expose a new provider in the model picker, /model, and all
# downstream consumers — no edits to this file needed.
_canonical_slugs = {p.slug for p in CANONICAL_PROVIDERS}
try:
    from providers import list_providers as _list_providers_for_canonical
    for _pp in _list_providers_for_canonical():
        if _pp.name in _canonical_slugs:
            continue
        if _pp.auth_type in {"oauth_device_code", "oauth_external", "external_process", "aws_sdk", "copilot", "vertex"}:
            continue  # non-api-key flows need bespoke picker UX; skip auto-inject
        _label = _pp.display_name or _pp.name
        _desc = _pp.description or f"{_label} (direct API)"
        CANONICAL_PROVIDERS.append(ProviderEntry(_pp.name, _label, _desc))
        _canonical_slugs.add(_pp.name)
except Exception:
    pass

# Derived dicts — used throughout the codebase
_PROVIDER_LABELS = {p.slug: p.label for p in CANONICAL_PROVIDERS}
_PROVIDER_LABELS["custom"] = "Custom endpoint"  # special case: not a named provider


# ---------------------------------------------------------------------------
# Provider groups — DISPLAY ONLY
#
# Some vendors expose several Hermes provider slugs (one per endpoint /
# auth method: global API, China API, OAuth coding plan, ...). Listing every
# slug as a top-level row in the interactive `hermes model` / setup wizard /
# Telegram `/model` pickers makes that list long and noisy.
#
# These groups fold related slugs under one top-level row in INTERACTIVE
# PICKERS only. They do NOT change ``CANONICAL_PROVIDERS``, slug identity,
# the ``--provider`` flag, ``/model <provider:model>``, or any typed path —
# every member slug remains individually addressable. Grouping is a pure
# display affordance; ``group_providers()`` is the single fold used by all
# three picker surfaces so they stay consistent.
#
#   group_id -> (display_label, group_description, [member_slug, ...])
#
# ``group_description`` is a short blurb shown on the collapsed top-level group
# row in the interactive pickers (alongside the label). Member-specific detail
# lives in each member's ``tui_desc`` and shows in the drill-down sub-picker.
# Member order is the order shown inside the group submenu.
# ---------------------------------------------------------------------------
PROVIDER_GROUPS: dict[str, tuple[str, str, list[str]]] = {
    "kimi":     ("Kimi / Moonshot", "Coding Plan, Moonshot global & China endpoints", ["kimi-coding", "kimi-coding-cn"]),
    "minimax":  ("MiniMax",         "Global, OAuth Coding Plan & China endpoints",     ["minimax", "minimax-oauth", "minimax-cn"]),
    "xai":      ("xAI Grok",        "Direct API or SuperGrok / Premium+ OAuth",        ["xai", "xai-oauth"]),
    "google":   ("Google Gemini",   "Google AI Studio (API key)",                     ["gemini"]),
    "openai":   ("OpenAI",          "Codex CLI or direct OpenAI API",                  ["openai-codex", "openai-api"]),
    "qwen":     ("Qwen",            "Qwen Cloud / DashScope, Coding Plan & Qwen CLI OAuth", ["alibaba", "alibaba-coding-plan", "qwen-oauth"]),
    "opencode": ("OpenCode",        "Zen pay-as-you-go or Go subscription",            ["opencode-zen", "opencode-go"]),
    "copilot":  ("GitHub Copilot",  "GitHub token API or copilot --acp process",       ["copilot", "copilot-acp"]),
}

# Reverse index: member slug -> group_id. Built once at import.
_SLUG_TO_GROUP: dict[str, str] = {
    slug: gid for gid, (_label, _desc, members) in PROVIDER_GROUPS.items() for slug in members
}


def provider_group_for_slug(slug: str) -> str:
    """Return the group_id a provider slug belongs to, or "" if ungrouped."""
    return _SLUG_TO_GROUP.get(str(slug or "").strip().lower(), "")


def group_providers(slugs):
    """Fold a flat ordered slug iterable into picker rows by provider group.

    DISPLAY ONLY. Used by every interactive picker (``hermes model``, the
    setup wizard, the Telegram ``/model`` keyboard) so grouping is identical
    across surfaces.

    Each returned row is a dict::

        {"kind": "single", "slug": <slug>}                       # ungrouped, or
                                                                  # 1-member group
        {"kind": "group", "group_id": <gid>, "label": <label>,
         "description": <desc>, "members": [<slug>, ...]}        # 2+ members

    Rules:
      * A group row appears at the position of its FIRST present member, in
        the input order. Subsequent members fold into that row (and are not
        emitted again).
      * Member order inside a group follows ``PROVIDER_GROUPS`` declaration,
        restricted to the members actually present in ``slugs``.
      * A group reduced to a single present member degrades to a ``single``
        row — no pointless one-item submenu.
      * Slugs not in any group pass through as ``single`` rows, order
        preserved.
      * Duplicate slugs in the input are ignored after first sight.
    """
    seen: set[str] = set()
    # Which present members each group has, in declaration order.
    group_members: dict[str, list[str]] = {}
    for gid, (_label, _desc, members) in PROVIDER_GROUPS.items():
        present = [m for m in members if m in set(slugs)]
        if present:
            group_members[gid] = present

    rows = []
    emitted_groups: set[str] = set()
    for slug in slugs:
        s = str(slug or "").strip().lower()
        if not s or s in seen:
            continue
        seen.add(s)
        gid = _SLUG_TO_GROUP.get(s, "")
        if not gid:
            rows.append({"kind": "single", "slug": s})
            continue
        if gid in emitted_groups:
            continue  # already folded at the first member's position
        emitted_groups.add(gid)
        members = group_members.get(gid, [s])
        if len(members) <= 1:
            rows.append({"kind": "single", "slug": members[0]})
        else:
            label, desc, _ = PROVIDER_GROUPS[gid]
            rows.append(
                {"kind": "group", "group_id": gid, "label": label,
                 "description": desc, "members": list(members)}
            )
    return rows


_PROVIDER_ALIASES = {
    "glm": "zai",
    "z-ai": "zai",
    "z.ai": "zai",
    "zhipu": "zai",
    "github": "copilot",
    "github-copilot": "copilot",
    "github-models": "copilot",
    "github-model": "copilot",
    "github-copilot-acp": "copilot-acp",
    "copilot-acp-agent": "copilot-acp",
    "google": "gemini",
    "google-gemini": "gemini",
    "google-ai-studio": "gemini",
    "google-vertex": "vertex",
    "vertex-ai": "vertex",
    "gcp-vertex": "vertex",
    "vertexai": "vertex",
    "kimi": "kimi-coding",
    "moonshot": "kimi-coding",
    "kimi-cn": "kimi-coding-cn",
    "moonshot-cn": "kimi-coding-cn",
    "step": "stepfun",
    "stepfun-coding-plan": "stepfun",
    "arcee-ai": "arcee",
    "arceeai": "arcee",
    "gmi-cloud": "gmi",
    "gmicloud": "gmi",
    "fireworks-ai": "fireworks",
    "fw": "fireworks",
    "minimax-china": "minimax-cn",
    "minimax_cn": "minimax-cn",
    "minimax-portal": "minimax-oauth",
    "minimax-global": "minimax-oauth",
    "minimax_oauth": "minimax-oauth",
    "claude": "anthropic",
    "claude-code": "anthropic",
    "deep-seek": "deepseek",
    "opencode": "opencode-zen",
    "zen": "opencode-zen",
    "go": "opencode-go",
    "opencode-go-sub": "opencode-go",
    "aigateway": "ai-gateway",
    "vercel": "ai-gateway",
    "vercel-ai-gateway": "ai-gateway",
    "kilo": "kilocode",
    "kilo-code": "kilocode",
    "kilo-gateway": "kilocode",
    "dashscope": "alibaba",
    "aliyun": "alibaba",
    "qwen": "alibaba",
    "alibaba-cloud": "alibaba",
    "qwen-portal": "qwen-oauth",
    "hf": "huggingface",
    "hugging-face": "huggingface",
    "huggingface-hub": "huggingface",
    "novita-ai": "novita",
    "novitaai": "novita",
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
    "grok": "xai",
    "grok-oauth": "xai-oauth",
    "xai-oauth": "xai-oauth",
    "x-ai-oauth": "xai-oauth",
    "xai-grok-oauth": "xai-oauth",
    "x-ai": "xai",
    "x.ai": "xai",
    "nim": "nvidia",
    "nvidia-nim": "nvidia",
    "build-nvidia": "nvidia",
    "nemotron": "nvidia",
    "lmstudio": "lmstudio",
    "lm-studio": "lmstudio",
    "lm_studio": "lmstudio",
    "ollama": "custom",  # bare "ollama" = local; use "ollama-cloud" for cloud
    "ollama_cloud": "ollama-cloud",
}


# In-repo fallback for the model Hermes silently lands on when the user never
# picked one (GUI onboarding confirm card, empty ``model.default``,
# provider-set-but-model-missing resolution). The AUTHORITATIVE source is the
# remote model catalog: the manifest labels exactly one entry per provider
# with ``"default": true`` (see get_default_model_from_cache in
# model_catalog.py), so maintainers can rotate the default without shipping a
# release. This constant is the offline/fresh-install fallback and MUST match
# the labeled entry in website/static/api/model-catalog.json. Deliberately a
# capable low-cost model rather than the curated lists' entry [0]: aggregator
# lists are ordered most-capable-first, so [0] is the priciest Anthropic
# flagship (claude-fable-5 / opus) — silently billing the most expensive model
# for traffic the user never opted into.
PREFERRED_SILENT_DEFAULT_MODEL = "z-ai/glm-5.2"






# Providers whose *silent* auto-default must go through the cost-safe
# catalog-labeled default (``get_preferred_silent_default_model``) instead of
# curated-list entry [0]. Metered aggregators (Nous Portal, OpenRouter) order
# their lists best-/most-capable-first — entry [0] is the priciest flagship
# (``anthropic/claude-fable-5``). Using that as the non-interactive fallback
# when a profile sets a provider with no model silently bills the most
# expensive model for traffic the user never opted into (a missing default
# escalated to Opus and billed 863 requests before the user noticed). The
# catalog manifest labels the default entry (``"default": true``) so it can
# rotate without a release; a missing model must never escalate to the
# flagship.
#
# This is deliberately a network-free lookup for the hot resolution path
# (cache-only catalog read). The *interactive* default (GUI onboarding /
# ``hermes model``) uses the richer free/paid-tier-aware resolver — see
# ``get_recommended_default_model`` in hermes_cli/web_server.py and
# ``partition_nous_models_by_tier`` — which can hit the Portal.
_SILENT_DEFAULT_PROVIDERS: frozenset[str] = frozenset({"nous", "openrouter"})


def get_default_model_for_provider(provider: str) -> str:
    """Return the network-free default model for *provider*, if known."""
    normalized = normalize_provider(provider)
    if normalized in _SILENT_DEFAULT_PROVIDERS:
        return PREFERRED_SILENT_DEFAULT_MODEL
    models = _PROVIDER_MODELS.get(normalized, ())
    return models[0] if models else ""






















# ---------------------------------------------------------------------------
# Pricing helpers — fetch live pricing from OpenRouter-compatible /v1/models
# ---------------------------------------------------------------------------

# Cache: maps model_id → {"prompt": str, "completion": str} per endpoint
_pricing_cache: dict[str, dict[str, dict[str, str]]] = {}












_DEFAULT_NOUS_INFERENCE_BASE = "https://inference-api.nousresearch.com"










# All provider IDs and aliases that are valid for the provider:model syntax.
_KNOWN_PROVIDER_NAMES: set[str] = (
    set(_PROVIDER_LABELS.keys())
    | set(_PROVIDER_ALIASES.keys())
    | {"openrouter", "custom"}
)


















# Retired model IDs kept for /model auto-detect only — not shown in pickers.
# DeepSeek cut these off on 2026-07-24; model_normalize remaps them on the wire.
_PROVIDER_RETIRED_ALIASES: dict[str, tuple[str, ...]] = {
    "deepseek": ("deepseek-chat", "deepseek-reasoner"),
}






_AGGREGATOR_PROVIDERS = frozenset(
    {"nous", "openrouter", "ai-gateway", "copilot", "kilocode"}
)

# Subscription/OAuth providers whose catalogs RE-EXPOSE other vendors' models
# would be listed here (tried only as a last resort for bare short-alias
# resolution, after every native-vendor catalog, so they never hijack an alias
# away from the model's native vendor). None are currently defined.
_BORROWED_MODEL_PROVIDERS: frozenset[str] = frozenset()

# Providers whose live /v1/models endpoint is the authoritative catalog, so the
# curated list is a discovery-only fallback. For these, the picker merges
# live-first (live entries lead, curated-only entries append). Every OTHER
# provider keeps curated-first (commit 658ac1d86, #46309) so a deliberately
# surfaced newest model stays at the top even when the live API lags. OpenCode
# Zen / Go re-expose dozens of upstream vendors and rotate them frequently, so
# their stale curated entries must not pollute the top of the picker. (#49129)
_LIVE_FIRST_PICKER_PROVIDERS: frozenset[str] = frozenset(
    {"opencode-zen", "opencode-go"}
)










def normalize_provider(provider: Optional[str]) -> str:
    """Normalize provider aliases to Hermes' canonical provider ids.

    Note: ``"auto"`` passes through unchanged — use
    ``hermes_cli.auth.resolve_provider()`` to resolve it to a concrete
    provider based on credentials and environment.
    """
    normalized = (provider or "openrouter").strip().lower()
    return _PROVIDER_ALIASES.get(normalized, normalized)




# Models that support OpenAI Priority Processing (service_tier="priority").
# See https://openai.com/api-priority-processing/ for the canonical list.
#
# Pattern-based matching — any OpenAI flagship model (gpt-*, o1*, o3*, o4*)
# is assumed to support Priority Processing. service_tier=priority is silently
# ignored by non-OpenAI endpoints (OpenRouter/Copilot/opencode-zen proxies
# strip the field), so false positives are harmless. Codex-series models
# (gpt-5-codex, gpt-5.3-codex, etc.) are excluded — they don't expose the
# service_tier parameter through the Codex Responses API.
_OPENAI_FAST_MODE_PREFIXES: tuple[str, ...] = (
    "gpt-",
    "o1",
    "o3",
    "o4",
)




# Models that support Anthropic Fast Mode (speed="fast").
# See https://platform.claude.com/docs/en/build-with-claude/fast-mode
#
# Pattern-based matching — any claude-* model is eligible. The anthropic
# adapter gates speed=fast on native Anthropic endpoints only (see
# _is_third_party_anthropic_endpoint in agent/anthropic_adapter.py), so
# third-party proxies that would reject the beta header are protected.










# Providers where models.dev is treated as authoritative: curated static
# lists are kept only as an offline fallback and to capture custom additions
# the registry doesn't publish yet. Adding a provider here causes its
# curated list to be merged with fresh models.dev entries (fresh first, any
# curated-only names appended) for both the CLI and the gateway /model picker.
#
# DELIBERATELY EXCLUDED:
#   - "openrouter": curated list is already a hand-picked agentic subset of
#     OpenRouter's 400+ catalog. Blindly merging would dump everything.
#   - "nous": curated list and Portal /models endpoint are the source of
#     truth for the subscription tier.
# Also excluded: providers that already have dedicated live-endpoint
# branches below (copilot, anthropic, ai-gateway, ollama-cloud, custom,
# stepfun, openai-codex) — those paths handle freshness themselves.
_MODELS_DEV_PREFERRED: frozenset[str] = frozenset({
    "opencode-go",
    "opencode-zen",
    "deepseek",
    "kilocode",
    "fireworks",
    "mistral",
    "togetherai",
    "cohere",
    "perplexity",
    "groq",
    "nvidia",
    "huggingface",
    "zai",
    "gemini",
    "google",
})






# ---------------------------------------------------------------------------
# Generic disk cache for provider_model_ids() — keeps /model picker fast.
# ---------------------------------------------------------------------------
#
# Without this layer, every /model picker open re-fetches every authed
# provider's /v1/models endpoint. On a well-configured user (anthropic +
# openai + copilot + gemini + huggingface + ...) that's 2+ seconds of cold
# HTTP roundtrips just to render the provider list.
#
# Cache strategy:
#   - One JSON file at $HERMES_HOME/provider_models_cache.json
#   - Per-provider entries keyed by (provider, credential fingerprint)
#   - Credential fingerprint = sha256 of env-var values that the provider
#     normally reads. Swap your OPENAI_API_KEY and the entry invalidates.
#   - 1h TTL by default. `force_refresh=True` skips the cache entirely
#     and overwrites it on success.
#   - Only NON-EMPTY results are cached. An empty/None response from a
#     transient network error never gets pinned.
#   - Cache file is best-effort. Any read/write error degrades silently
#     to a live fetch — the picker keeps working.

_PROVIDER_MODELS_CACHE_TTL = 3600  # 1h


















def copilot_default_headers(*, is_agent_turn: bool = True) -> dict[str, str]:
    """Standard headers for Copilot API requests.

    Includes Openai-Intent and x-initiator headers that opencode and the
    Copilot CLI send on every request.
    """
    try:
        from hermes_cli.copilot_auth import copilot_request_headers
        return copilot_request_headers(is_agent_turn=is_agent_turn)
    except ImportError:
        return {
            "Editor-Version": COPILOT_EDITOR_VERSION,
            "User-Agent": "HermesAgent/1.0",
            "Openai-Intent": "conversation-edits",
            "x-initiator": "agent" if is_agent_turn else "user",
        }






# ─── Copilot catalog context-window helpers ─────────────────────────────────

# Module-level cache: {model_id: max_prompt_tokens}
_copilot_context_cache: dict[str, int] = {}
_copilot_context_cache_time: float = 0.0
_COPILOT_CONTEXT_CACHE_TTL = 3600  # 1 hour






def _lmstudio_server_root(base_url: Optional[str]) -> Optional[str]:
    """Return the LM Studio server root for native ``/api/v1`` endpoints.

    Users commonly copy either the OpenAI-compatible runtime URL
    (``.../v1``) or the native API prefix (``.../api`` / ``.../api/v1``).
    Native probes append ``/api/v1/...`` themselves, so normalize all accepted
    forms back to the bare server root to avoid ``/api/api/v1`` requests.
    Returns ``None`` when the base URL is empty/invalid.
    """
    root = (base_url or "").strip().rstrip("/")
    for suffix in ("/api/v1", "/api", "/v1"):
        if root.endswith(suffix):
            root = root[: -len(suffix)].rstrip("/")
            break
    return root or None


def _lmstudio_request_headers(api_key: Optional[str] = None) -> dict:
    """Build HTTP headers for LM Studio native API requests."""
    headers = {"User-Agent": _HERMES_USER_AGENT}
    token = str(api_key or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers












async def lmstudio_model_reasoning_options(
    model: str,
    base_url: Optional[str],
    api_key: Optional[str] = None,
    timeout: float = 5.0,
) -> list[str]:
    """Return the reasoning ``allowed_options`` LM Studio publishes for ``model``.

    Pulls ``capabilities.reasoning.allowed_options`` from ``/api/v1/models``.
    Returns ``[]`` when the model is unknown, the endpoint is unreachable,
    or the model does not declare a reasoning capability.
    """
    server_root = _lmstudio_server_root(base_url)
    if not server_root:
        return []
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            headers=_lmstudio_request_headers(api_key),
        ) as client:
            response = await client.get(f"{server_root}/api/v1/models")
            response.raise_for_status()
            payload = response.json()
        raw_models = payload.get("models") if isinstance(payload, dict) else None
    except Exception:
        raw_models = None
    if not raw_models:
        return []

    for raw in raw_models:
        if not isinstance(raw, dict):
            continue
        if raw.get("key") != model and raw.get("id") != model:
            continue
        caps = raw.get("capabilities")
        reasoning = caps.get("reasoning") if isinstance(caps, dict) else None
        opts = reasoning.get("allowed_options") if isinstance(reasoning, dict) else None
        if isinstance(opts, list):
            return [str(o).strip().lower() for o in opts if isinstance(o, str)]
        return []
    return []


async def ollama_model_supports_thinking(
    model: str,
    base_url: Optional[str],
    api_key: Optional[str] = None,
    timeout: float = 5.0,
) -> Optional[bool]:
    """Return True if an Ollama (Cloud or local) model advertises ``thinking``.

    Probes the native ``/api/show`` endpoint and checks the ``capabilities``
    list, which Ollama populates from the model's metadata (e.g.
    ``deepseek-v4-pro`` → ``["completion", "tools", "thinking"]`` while
    ``gemma3:27b`` → ``["completion", "vision"]``). This is the authoritative
    capability source — the OpenAI-compat ``/v1/models`` endpoint omits it.

    Returns:
        True  — the model declares the ``thinking`` capability.
        False — ``/api/show`` succeeded but the model has no ``thinking`` cap.
        None  — the probe failed (unreachable / non-Ollama / error); the caller
                decides the fallback (we treat None as "don't emit").
    """
    import httpx

    server_url = (base_url or "").strip().rstrip("/")
    if server_url.endswith("/v1"):
        server_url = server_url[:-3]
    if not server_url:
        return None

    bare_model = _strip_ollama_cloud_suffix((model or "").strip())
    if not bare_model:
        return None

    token = str(api_key or "").strip()
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            resp = await client.post(
                f"{server_url}/api/show",
                json={"name": bare_model},
            )
            if resp.status_code != 200:
                return None
            caps = resp.json().get("capabilities")
            if isinstance(caps, list):
                return "thinking" in caps
    except Exception:
        return None
    return None




_COPILOT_MODEL_ALIASES = {
    "openai/gpt-5": "gpt-5-mini",
    "openai/gpt-5-chat": "gpt-5-mini",
    "openai/gpt-5-mini": "gpt-5-mini",
    "openai/gpt-5-nano": "gpt-5-mini",
    "openai/gpt-4.1": "gpt-4.1",
    "openai/gpt-4.1-mini": "gpt-4.1",
    "openai/gpt-4.1-nano": "gpt-4.1",
    "openai/gpt-4o": "gpt-4o",
    "openai/gpt-4o-mini": "gpt-4o-mini",
    "openai/o1": "gpt-5.2",
    "openai/o1-mini": "gpt-5-mini",
    "openai/o1-preview": "gpt-5.2",
    "openai/o3": "gpt-5.3-codex",
    "openai/o3-mini": "gpt-5-mini",
    "openai/o4-mini": "gpt-5-mini",
    "anthropic/claude-opus-4.6": "claude-opus-4.6",
    "anthropic/claude-sonnet-5": "claude-sonnet-5",
    "anthropic/claude-sonnet-4.6": "claude-sonnet-4.6",
    "anthropic/claude-sonnet-4": "claude-sonnet-4",
    "anthropic/claude-sonnet-4.5": "claude-sonnet-4.5",
    "anthropic/claude-haiku-4.5": "claude-haiku-4.5",
    # Dash-notation fallbacks: Hermes' default Claude IDs elsewhere use
    # hyphens (anthropic native format), but Copilot's API only accepts
    # dot-notation.  Accept both so users who configure copilot + a
    # default hyphenated Claude model don't hit HTTP 400
    # "model_not_supported".  See issue #6879.
    "claude-sonnet-5": "claude-sonnet-5",
    "claude-opus-4-6": "claude-opus-4.6",
    "claude-sonnet-4-6": "claude-sonnet-4.6",
    "claude-sonnet-4-0": "claude-sonnet-4",
    "claude-sonnet-4-5": "claude-sonnet-4.5",
    "claude-haiku-4-5": "claude-haiku-4.5",
    "anthropic/claude-opus-4-6": "claude-opus-4.6",
    "anthropic/claude-sonnet-5": "claude-sonnet-5",
    "anthropic/claude-sonnet-4-6": "claude-sonnet-4.6",
    "anthropic/claude-sonnet-4-0": "claude-sonnet-4",
    "anthropic/claude-sonnet-4-5": "claude-sonnet-4.5",
    "anthropic/claude-haiku-4-5": "claude-haiku-4.5",
}


def _copilot_catalog_ids(
    catalog: Optional[list[dict[str, Any]]] = None,
    api_key: Optional[str] = None,
) -> set[str]:
    if not catalog:
        return set()
    return {
        str(item.get("id") or "").strip()
        for item in catalog
        if str(item.get("id") or "").strip()
    }


def normalize_copilot_model_id(
    model_id: Optional[str],
    *,
    catalog: Optional[list[dict[str, Any]]] = None,
    api_key: Optional[str] = None,
) -> str:
    raw = str(model_id or "").strip()
    if not raw:
        return ""

    catalog_ids = _copilot_catalog_ids(catalog=catalog, api_key=api_key)
    alias = _COPILOT_MODEL_ALIASES.get(raw)
    if alias:
        return alias

    candidates = [raw]
    if "/" in raw:
        candidates.append(raw.split("/", 1)[1].strip())

    if raw.endswith("-mini"):
        candidates.append(raw[:-5])
    if raw.endswith("-nano"):
        candidates.append(raw[:-5])
    if raw.endswith("-chat"):
        candidates.append(raw[:-5])

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if candidate in _COPILOT_MODEL_ALIASES:
            return _COPILOT_MODEL_ALIASES[candidate]
        if candidate in catalog_ids:
            return candidate

    if "/" in raw:
        return raw.split("/", 1)[1].strip()
    return raw


def _github_reasoning_efforts_for_model_id(model_id: str) -> list[str]:
    raw = (model_id or "").strip().lower()
    if raw.startswith(("openai/o1", "openai/o3", "openai/o4", "o1", "o3", "o4")):
        return list(COPILOT_REASONING_EFFORTS_O_SERIES)
    normalized = normalize_copilot_model_id(model_id).lower()
    if normalized.startswith("gpt-5"):
        return list(COPILOT_REASONING_EFFORTS_GPT5)
    return []


def _is_github_models_base_url(base_url: Optional[str]) -> bool:
    """Return whether *base_url* is one of GitHub Models' API roots."""
    normalized = (base_url or "").strip().rstrip("/").lower()
    return (
        normalized.startswith(COPILOT_BASE_URL)
        or normalized.startswith("https://models.github.ai/inference")
        or normalized.startswith("https://models.inference.ai.azure.com")
    )


def _should_use_copilot_responses_api(model_id: str) -> bool:
    """Decide whether a Copilot model should use the Responses API.

    Replicates opencode's ``shouldUseCopilotResponsesApi`` logic:
    GPT-5+ models use Responses API, except ``gpt-5-mini`` which uses
    Chat Completions.  All non-GPT models (Claude, Gemini, etc.) use
    Chat Completions.
    """
    import re

    match = re.match(r"^gpt-(\d+)", model_id)
    if not match:
        return False
    major = int(match.group(1))
    return major >= 5 and not model_id.startswith("gpt-5-mini")


def copilot_model_api_mode(
    model_id: Optional[str],
    *,
    catalog: Optional[list[dict[str, Any]]] = None,
    api_key: Optional[str] = None,
) -> str:
    """Determine the API mode for a Copilot model.

    Uses the model ID pattern (matching opencode's approach) as the
    primary signal.  Falls back to the catalog's ``supported_endpoints``
    only for models not covered by the pattern check.
    """
    normalized = normalize_copilot_model_id(model_id, catalog=catalog, api_key=api_key)
    if not normalized:
        return "chat_completions"

    # Primary: model ID pattern (matches opencode's shouldUseCopilotResponsesApi)
    if _should_use_copilot_responses_api(normalized):
        return "codex_responses"

    # Copilot's Claude models are exposed through its OpenAI-compatible chat
    # endpoint, not through Hermes' native Anthropic adapter. The live catalog may
    # advertise /v1/messages, but the Copilot token/header scheme is handled by
    # the OpenAI client path; selecting anthropic_messages would send the wrong
    # auth/wire shape. Keep non-GPT Copilot slots on chat_completions.
    return "chat_completions"


# Azure Foundry model families that require the Responses API.  Azure
# rejects /chat/completions against these deployments with
# ``400 "The requested operation is unsupported."`` — the same payload Bob
# Dobolina hit in April 2026 on ``gpt-5.3-codex`` while ``gpt-4o-pure`` on
# the same endpoint worked fine.  Keep the patterns broad enough to cover
# vendor-renamed deployments (e.g. ``gpt-5.3-codex``, ``gpt-5-codex``,
# ``gpt-5.4``, ``o1-preview``) but tight enough to leave GPT-4 / 3.5 / Llama /
# Mistral / Grok deployments on chat completions.
_AZURE_FOUNDRY_RESPONSES_PREFIXES = (
    "codex",       # codex-*, codex-mini
    "gpt-5",       # gpt-5, gpt-5.x, gpt-5-codex, gpt-5.x-codex
    "o1",          # o1, o1-preview, o1-mini
    "o3",          # o3, o3-mini
    "o4",          # o4, o4-mini
)


def azure_foundry_model_api_mode(model_name: Optional[str]) -> Optional[str]:
    """Infer Azure Foundry api_mode from a deployment/model name.

    Returns ``"codex_responses"`` when the model name matches a family that
    only accepts the Responses API on Azure Foundry (GPT-5.x, codex, o1/o3/o4
    reasoning models).  Returns ``None`` otherwise — the caller should fall
    back to the configured/default api_mode (typically ``chat_completions``)
    so GPT-4o, GPT-4 Turbo, Llama, Mistral, etc. keep working.

    Intentionally does NOT return ``anthropic_messages``; Anthropic-style
    Azure endpoints are disambiguated by URL (``/anthropic`` suffix) in
    ``runtime_provider._detect_api_mode_for_url`` and by the user setting
    ``model.api_mode: anthropic_messages`` explicitly.
    """
    raw = str(model_name or "").strip().lower()
    if not raw:
        return None
    # Strip any vendor/ prefix a user may have copied from OpenRouter / Copilot.
    if "/" in raw:
        raw = raw.rsplit("/", 1)[-1]
    # gpt-5-mini speaks chat completions on Copilot but Azure Foundry deploys
    # the full gpt-5 family uniformly on Responses API — don't carve an
    # exception here.
    for prefix in _AZURE_FOUNDRY_RESPONSES_PREFIXES:
        if raw.startswith(prefix):
            return "codex_responses"
    return None


def normalize_opencode_model_id(provider_id: Optional[str], model_id: Optional[str]) -> str:
    """Normalize OpenCode config IDs to the bare model slug used in API requests."""
    provider = normalize_provider(provider_id)
    current = str(model_id or "").strip()
    if not current or provider not in {"opencode-zen", "opencode-go"}:
        return current

    prefix = f"{provider}/"
    if current.lower().startswith(prefix):
        return current[len(prefix):]
    return current


def opencode_model_api_mode(provider_id: Optional[str], model_id: Optional[str]) -> str:
    """Determine the API mode for an OpenCode Zen / Go model.

    OpenCode routes different models behind different API surfaces:

    - GPT-5 / Codex models on Zen use ``/v1/responses``
    - Claude models on Zen use ``/v1/messages``
    - MiniMax and Qwen models on Go use ``/v1/messages``
    - GLM / Kimi / DeepSeek / MiMo on Go use ``/v1/chat/completions``
    - Qwen models on Zen use ``/v1/messages``
    - Other Zen models (Gemini, GLM, Kimi, MiniMax, DeepSeek, etc.) use
      ``/v1/chat/completions``

    This follows the published OpenCode docs for Zen and Go endpoints
    (https://opencode.ai/docs/zen/ and https://opencode.ai/docs/go/).
    """
    provider = normalize_provider(provider_id)
    normalized = normalize_opencode_model_id(provider_id, model_id).lower()
    if not normalized:
        return "chat_completions"

    if provider == "opencode-go":
        if normalized.startswith("minimax-"):
            return "anthropic_messages"
        if normalized.startswith("qwen"):
            # All Qwen models on Go (qwen3.7-max, qwen3.7-plus, qwen3.6-plus)
            # are served via /v1/messages per the published Go endpoint table.
            return "anthropic_messages"
        return "chat_completions"

    if provider == "opencode-zen":
        if normalized.startswith("claude-"):
            return "anthropic_messages"
        if normalized.startswith("gpt-"):
            return "codex_responses"
        if normalized.startswith("qwen"):
            # Qwen models on Zen moved to /v1/messages per the published
            # Zen endpoint table.
            return "anthropic_messages"
        return "chat_completions"

    return "chat_completions"


def normalize_opencode_base_url(
    provider_id: Optional[str], api_mode: Optional[str], base_url: Optional[str]
) -> str:
    """Normalize an OpenCode Zen / Go base URL for the target API mode.

    OpenCode's OpenAI-compatible endpoints live under ``/v1`` (the OpenAI SDK
    appends ``/chat/completions`` or ``/responses``), while the Anthropic SDK
    appends its own ``/v1/messages`` — so anthropic_messages needs the ``/v1``
    suffix stripped.

    Crucially this must be SYMMETRIC.  The stripped URL gets persisted to
    config (``model.base_url``) by the TUI/desktop and gateway after switching
    into an anthropic-routed model (e.g. minimax-m2.7 on Go).  A later switch
    to a chat_completions model (glm, deepseek, kimi) then inherited the
    stripped URL and POSTed to ``https://opencode.ai/zen/go/chat/completions``
    — a 404 (the marketing site).  Re-append ``/v1`` for non-anthropic modes
    so previously-stripped URLs heal themselves.

    Only opencode.ai-hosted URLs are re-suffixed; custom proxy overrides via
    ``OPENCODE_*_BASE_URL`` are left alone unless they already carry ``/v1``.
    """
    url = str(base_url or "").strip().rstrip("/")
    if not url:
        return url
    provider = normalize_provider(provider_id)
    if provider not in {"opencode-zen", "opencode-go"}:
        return url

    import re as _re

    if api_mode == "anthropic_messages":
        return _re.sub(r"/v1$", "", url)

    # chat_completions / codex_responses: ensure the /v1 suffix is present on
    # official opencode.ai hosts (heals a persisted anthropic-stripped URL).
    if url.endswith("/v1"):
        return url
    try:
        host = urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        host = ""
    if host == "opencode.ai" or host.endswith(".opencode.ai"):
        return url + "/v1"
    return url


def github_model_reasoning_efforts(
    model_id: Optional[str],
    *,
    catalog: Optional[list[dict[str, Any]]] = None,
    api_key: Optional[str] = None,
) -> list[str]:
    """Return supported reasoning-effort levels for a Copilot-visible model."""
    normalized = normalize_copilot_model_id(model_id, catalog=catalog, api_key=api_key)
    if not normalized:
        return []

    catalog_entry = None
    if catalog is not None:
        catalog_entry = next((item for item in catalog if item.get("id") == normalized), None)

    if catalog_entry is not None:
        capabilities = catalog_entry.get("capabilities")
        if isinstance(capabilities, dict):
            supports = capabilities.get("supports")
            if isinstance(supports, dict):
                efforts = supports.get("reasoning_effort")
                if isinstance(efforts, list):
                    normalized_efforts = [
                        str(effort).strip().lower()
                        for effort in efforts
                        if str(effort).strip()
                    ]
                    return list(dict.fromkeys(normalized_efforts))
            return []
        legacy_capabilities = {
            str(capability).strip().lower()
            for capability in catalog_entry.get("capabilities", [])
            if str(capability).strip()
        }
        if "reasoning" not in legacy_capabilities:
            return []

    return _github_reasoning_efforts_for_model_id(str(model_id or normalized))




# Legacy filter — used when an item has no surface tag (rolling out
# 2026-05). Once every model returned by the catalog endpoint carries an
# explicit surface tag (``chat``/``embed``/``image-gen``/``tts``/``stt``)
# the regex path becomes unreachable and can be removed.
_DEEPINFRA_EXCLUDE_RE = re.compile(
    r"(?i)(embed|rerank|whisper|stable-diffusion|flux|sdxl|"
    r"tts|bark|speech|image-gen|clip|vit-|dpt-)",
)

# Surface tags announce *what kind of model* this is. When none of these
# are present on a catalog entry, the tags array only carries capability
# tags (``reasoning``, ``vision``, ``prompt_cache``, …) and we have to
# fall back to id-regex inference for the chat surface.
_DEEPINFRA_SURFACE_TAGS: frozenset[str] = frozenset({
    "chat", "embed", "image-gen", "tts", "stt", "video-gen",
})

_DEEPINFRA_DEFAULT_BASE_URL = "https://api.deepinfra.com/v1/openai"
_DEEPINFRA_MODELS_QUERY = "filter=true&sort_by=hermes"

# Module-level cache for the full tagged catalog response, keyed by base URL.
# Each value is the parsed ``data`` list. Surface-specific filters read from
# this cache so a single network round-trip serves chat / image-gen / tts /
# stt callers across the whole process lifetime.
_deepinfra_catalog_cache: dict[str, list[dict]] = {}

# Negative cache: monotonic timestamp of the last failed fetch, keyed by base
# URL. Without this, an unreachable catalog (offline / DNS / firewall) makes
# every surface helper (chat picker, pricing, image/video/tts/stt defaults,
# vision) re-attempt a fresh blocking fetch that eats the full timeout each
# time — several sequential stalls in one user-visible operation. A short TTL
# lets connectivity recover without a process restart.
_deepinfra_catalog_neg_cache: dict[str, float] = {}
_DEEPINFRA_CATALOG_NEG_TTL = 60.0  # seconds




















# ---------------------------------------------------------------------------
# Ollama Cloud — merged model discovery with disk cache
# ---------------------------------------------------------------------------



_OLLAMA_CLOUD_CACHE_TTL = 3600  # 1 hour


def _strip_ollama_cloud_suffix(model_id: str) -> str:
    """Strip :cloud / -cloud suffixes that models.dev appends to Ollama Cloud IDs.

    The live API uses clean IDs (e.g. 'kimi-k2.6') while models.dev sometimes
    returns them as 'kimi-k2.6:cloud'. Normalising before the dedup merge
    prevents duplicate entries in the merged model list.
    """
    for suffix in (":cloud", "-cloud"):
        if model_id.endswith(suffix):
            return model_id[: -len(suffix)]
    return model_id
