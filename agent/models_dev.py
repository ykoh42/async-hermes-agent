"""Models.dev registry integration — primary database for providers and models.

Fetches from https://models.dev/api.json — a community-maintained database
of 4000+ models across 109+ providers.  Provides:

- **Provider metadata**: name, base URL, env vars, documentation link
- **Model metadata**: context window, max output, cost/M tokens, capabilities
  (reasoning, tools, vision, PDF, audio), modalities, knowledge cutoff,
  open-weights flag, family grouping, deprecation status

Data resolution order:
  1. In-memory cache (fresh, or stale served immediately while a single
     background daemon thread refreshes)
  2. Disk cache (~/.hermes/models_dev_cache.json — any age; stale data is
     served rather than blocking callers on the network)
  3. Network fetch (https://models.dev/api.json) — only when no cache
     exists at all; failed refreshes back off for 5 minutes process-wide
Latency-sensitive callers (gateway route-identity checks) pass
``allow_network=False`` and never touch the network.

Other modules should import the dataclasses and query functions from here
rather than parsing the raw JSON themselves.
"""

import json
import logging
import time
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

MODELS_DEV_URL = "https://models.dev/api.json"
_MODELS_DEV_CACHE_TTL = 3600  # 1 hour in-memory
_MODELS_DEV_RETRY_DELAY = 300  # 5 minutes after a failed refresh

# In-memory cache
_models_dev_cache: Dict[str, Any] = {}
_models_dev_cache_time: float = 0
_models_dev_retry_after: float = 0
_models_dev_lock: Optional[asyncio.Lock] = None


# ---------------------------------------------------------------------------
# Dataclasses — rich metadata for providers and models
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    """Full metadata for a single model from models.dev."""

    id: str
    name: str
    family: str
    provider_id: str        # models.dev provider ID (e.g. "anthropic")

    # Capabilities
    reasoning: bool = False
    tool_call: bool = False
    attachment: bool = False       # supports image/file attachments (vision)
    temperature: bool = False
    structured_output: bool = False
    open_weights: bool = False

    # Modalities
    input_modalities: Tuple[str, ...] = ()    # ("text", "image", "pdf", ...)
    output_modalities: Tuple[str, ...] = ()

    # Limits
    context_window: int = 0
    max_output: int = 0
    max_input: Optional[int] = None

    # Cost (per million tokens, USD)
    cost_input: float = 0.0
    cost_output: float = 0.0
    cost_cache_read: Optional[float] = None
    cost_cache_write: Optional[float] = None

    # Metadata
    knowledge_cutoff: str = ""
    release_date: str = ""
    status: str = ""          # "alpha", "beta", "deprecated", or ""
    interleaved: Any = False  # True or {"field": "reasoning_content"}

    def has_cost_data(self) -> bool:
        return self.cost_input > 0 or self.cost_output > 0

    def supports_vision(self) -> bool:
        return self.attachment or "image" in self.input_modalities

    def supports_pdf(self) -> bool:
        return "pdf" in self.input_modalities

    def supports_audio_input(self) -> bool:
        return "audio" in self.input_modalities

    def format_cost(self) -> str:
        """Human-readable cost string, e.g. '$3.00/M in, $15.00/M out'."""
        if not self.has_cost_data():
            return "unknown"
        parts = [f"${self.cost_input:.2f}/M in", f"${self.cost_output:.2f}/M out"]
        if self.cost_cache_read is not None:
            parts.append(f"cache read ${self.cost_cache_read:.2f}/M")
        return ", ".join(parts)

    def format_capabilities(self) -> str:
        """Human-readable capabilities, e.g. 'reasoning, tools, vision, PDF'."""
        caps = []
        if self.reasoning:
            caps.append("reasoning")
        if self.tool_call:
            caps.append("tools")
        if self.supports_vision():
            caps.append("vision")
        if self.supports_pdf():
            caps.append("PDF")
        if self.supports_audio_input():
            caps.append("audio")
        if self.structured_output:
            caps.append("structured output")
        if self.open_weights:
            caps.append("open weights")
        return ", ".join(caps) if caps else "basic"


@dataclass
class ProviderInfo:
    """Full metadata for a provider from models.dev."""

    id: str                         # models.dev provider ID
    name: str                       # display name
    env: Tuple[str, ...]            # env var names for API key
    api: str                        # base URL
    doc: str = ""                   # documentation URL
    model_count: int = 0


# ---------------------------------------------------------------------------
# Provider ID mapping: Hermes ↔ models.dev
# ---------------------------------------------------------------------------

# Hermes provider names → models.dev provider IDs
PROVIDER_TO_MODELS_DEV: Dict[str, str] = {
    "openrouter": "openrouter",
    "novita": "novita-ai",
    "anthropic": "anthropic",
    "openai": "openai",
    "openai-codex": "openai",
    "zai": "zai",
    "kimi": "kimi-for-coding",
    "kimi-coding": "kimi-for-coding",
    "moonshot": "kimi-for-coding",
    "stepfun": "stepfun",
    "kimi-coding-cn": "kimi-for-coding",
    "minimax": "minimax",
    "minimax-oauth": "minimax",
    "minimax-cn": "minimax-cn",
    "deepseek": "deepseek",
    "alibaba": "alibaba",
    "qwen-oauth": "alibaba",
    "copilot": "github-copilot",
    "ai-gateway": "vercel",
    "opencode-zen": "opencode",
    "opencode-go": "opencode-go",
    "kilocode": "kilo",
    "fireworks": "fireworks-ai",
    "huggingface": "huggingface",
    "gemini": "google",
    "google": "google",
    "xai": "xai",
    # xAI OAuth is an authentication/transport path for the same xAI model
    # catalog, so model metadata should resolve through the xAI provider.
    "xai-oauth": "xai",
    "xiaomi": "xiaomi",
    "nvidia": "nvidia",
    "groq": "groq",
    "mistral": "mistral",
    "togetherai": "togetherai",
    "perplexity": "perplexity",
    "cohere": "cohere",
    "ollama-cloud": "ollama-cloud",
}

# Reverse mapping: models.dev → Hermes (built lazily)
_MODELS_DEV_TO_PROVIDER: Optional[Dict[str, str]] = None



def _get_cache_path() -> Path:
    """Return path to disk cache file."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "models_dev_cache.json"


async def lookup_models_dev_context(provider: str, model: str) -> Optional[int]:
    """Look up context_length for a provider+model combo in models.dev.

    Returns the context window in tokens, or None if not found.
    Handles case-insensitive matching and filters out context=0 entries.
    """
    mdev_provider_id = PROVIDER_TO_MODELS_DEV.get(provider)
    if not mdev_provider_id:
        return None

    data = await fetch_models_dev()
    provider_data = data.get(mdev_provider_id)
    if not isinstance(provider_data, dict):
        return None

    models = provider_data.get("models", {})
    if not isinstance(models, dict):
        return None

    # Exact match
    entry = models.get(model)
    if entry:
        ctx = _extract_context(entry)
        if ctx:
            return ctx

    # Case-insensitive match
    model_lower = model.lower()
    for mid, mdata in models.items():
        if mid.lower() == model_lower:
            ctx = _extract_context(mdata)
            if ctx:
                return ctx

    # Suffix-aware fallback: some providers (e.g. ollama-cloud) store
    # model IDs with :cloud / -cloud suffixes in models.dev while the
    # live API returns bare names.  Without this, kimi-k2.6 misses the
    # kimi-k2.6:cloud entry and falls through to stale OpenRouter metadata
    # reporting 32768 — tripping the 64k minimum-context guard.
    # The suffix-stripping in fetch_ollama_cloud_models() handles the
    # model-picker UX; this handles the context-length lookup path.
    for suffix in (":cloud", "-cloud"):
        suffixed_key = model + suffix
        entry = models.get(suffixed_key)
        if entry:
            ctx = _extract_context(entry)
            if ctx:
                return ctx
        # Also try case-insensitive
        suffixed_lower = model_lower + suffix
        for mid, mdata in models.items():
            if mid.lower() == suffixed_lower:
                ctx = _extract_context(mdata)
                if ctx:
                    return ctx

    return None


def _extract_context(entry: Dict[str, Any]) -> Optional[int]:
    """Extract context_length from a models.dev model entry.

    Returns None for invalid/zero values (some audio/image models have context=0).
    """
    if not isinstance(entry, dict):
        return None
    limit = entry.get("limit")
    if not isinstance(limit, dict):
        return None
    ctx = limit.get("context")
    if isinstance(ctx, (int, float)) and ctx > 0:
        return int(ctx)
    return None


# ---------------------------------------------------------------------------
# Model capability metadata
# ---------------------------------------------------------------------------


@dataclass
class ModelCapabilities:
    """Structured capability metadata for a model from models.dev."""

    supports_tools: bool = True
    supports_vision: bool = False
    supports_reasoning: bool = False
    context_window: int = 200000
    max_output_tokens: int = 8192
    model_family: str = ""


async def _get_provider_models(provider: str) -> Optional[Dict[str, Any]]:
    """Resolve a Hermes provider ID to its models dict from models.dev.

    Returns the models dict or None if the provider is unknown or has no data.
    """
    mdev_provider_id = PROVIDER_TO_MODELS_DEV.get(provider)
    if not mdev_provider_id:
        return None

    data = await fetch_models_dev()
    provider_data = data.get(mdev_provider_id)
    if not isinstance(provider_data, dict):
        return None

    models = provider_data.get("models", {})
    if not isinstance(models, dict):
        return None

    return models


def _find_model_entry(models: Dict[str, Any], model: str) -> Optional[Dict[str, Any]]:
    """Find a model entry by exact match, then case-insensitive fallback."""
    # Exact match
    entry = models.get(model)
    if isinstance(entry, dict):
        return entry

    # Case-insensitive match
    model_lower = model.lower()
    for mid, mdata in models.items():
        if mid.lower() == model_lower and isinstance(mdata, dict):
            return mdata

    return None


async def fetch_models_dev(
    force_refresh: bool = False,
    *,
    allow_network: bool = True,
) -> Dict[str, Any]:
    """Fetch the models.dev registry without blocking the event loop.

    CLI/setup callers continue to use :func:`fetch_models_dev`, while the
    agent turn uses this native transport.  Stale capability data is safe to
    serve, so the async path never starts a background thread as a side
    effect of a request.
    """
    global _models_dev_cache, _models_dev_cache_time, _models_dev_retry_after
    global _models_dev_lock

    if (
        not force_refresh
        and _models_dev_cache
        and (time.time() - _models_dev_cache_time) < _MODELS_DEV_CACHE_TTL
    ):
        return _models_dev_cache

    if _models_dev_lock is None:
        _models_dev_lock = asyncio.Lock()

    async with _models_dev_lock:
        if (
            not force_refresh
            and _models_dev_cache
            and (time.time() - _models_dev_cache_time) < _MODELS_DEV_CACHE_TTL
        ):
            return _models_dev_cache

        import aiofiles
        import aiofiles.os

        cache_path = _get_cache_path()
        if not force_refresh:
            try:
                if await aiofiles.os.path.exists(cache_path):
                    async with aiofiles.open(cache_path, encoding="utf-8") as fh:
                        disk_data = json.loads(await fh.read())
                    if isinstance(disk_data, dict) and disk_data:
                        _models_dev_cache = disk_data
                        _models_dev_cache_time = time.time()
                        return _models_dev_cache
            except Exception as exc:
                logger.debug("Failed to load async models.dev cache: %s", exc)

        if not allow_network or (
            not force_refresh and time.time() < _models_dev_retry_after
        ):
            return _models_dev_cache

        import httpx

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
                response = await client.get(MODELS_DEV_URL)
                response.raise_for_status()
                data = response.json()
            if not isinstance(data, dict) or not data:
                raise ValueError("models.dev returned an empty or invalid registry")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _models_dev_retry_after = time.time() + _MODELS_DEV_RETRY_DELAY
            logger.debug("Async models.dev refresh failed: %s", exc)
            return _models_dev_cache

        _models_dev_cache = data
        _models_dev_cache_time = time.time()
        _models_dev_retry_after = 0
        try:
            await aiofiles.os.makedirs(cache_path.parent, exist_ok=True)
            async with aiofiles.open(cache_path, "w", encoding="utf-8") as fh:
                await fh.write(json.dumps(data, separators=(",", ":")))
        except Exception as exc:
            logger.debug("Failed to persist async models.dev cache: %s", exc)
        return _models_dev_cache


async def get_model_capabilities(
    provider: str,
    model: str,
) -> Optional[ModelCapabilities]:
    """Async counterpart of :func:`get_model_capabilities`."""
    mdev_provider_id = PROVIDER_TO_MODELS_DEV.get(provider)
    if not mdev_provider_id:
        return None
    data = await fetch_models_dev()
    provider_data = data.get(mdev_provider_id)
    if not isinstance(provider_data, dict):
        return None
    models = provider_data.get("models", {})
    if not isinstance(models, dict):
        return None
    entry = _find_model_entry(models, model)
    if entry is None:
        return None

    input_mods = entry.get("modalities", {})
    input_mods = input_mods.get("input") if isinstance(input_mods, dict) else None
    supports_vision = (
        "image" in input_mods
        if isinstance(input_mods, list)
        else bool(entry.get("attachment", False))
    )
    limit = entry.get("limit", {})
    limit = limit if isinstance(limit, dict) else {}
    context = limit.get("context")
    output = limit.get("output")
    return ModelCapabilities(
        supports_tools=bool(entry.get("tool_call", False)),
        supports_vision=supports_vision,
        supports_reasoning=bool(entry.get("reasoning", False)),
        context_window=int(context) if isinstance(context, (int, float)) and context > 0 else 200000,
        max_output_tokens=int(output) if isinstance(output, (int, float)) and output > 0 else 8192,
        model_family=entry.get("family", "") or "",
    )


async def list_provider_models(provider: str) -> List[str]:
    """Return all model IDs for a provider from models.dev.

    Returns an empty list if the provider is unknown or has no data.
    """
    from hermes_cli.models import normalize_provider
    provider = normalize_provider(provider) or provider
    
    models = await _get_provider_models(provider)
    if models is None:
        return []
    return [
        mid for mid in models.keys()
        if not _should_hide_from_provider_catalog(provider, mid)
    ]


# Patterns that indicate non-agentic or noise models (TTS, embedding,
# dated preview snapshots, live/streaming-only, image-only).
import re
_NOISE_PATTERNS: re.Pattern = re.compile(
    r"-tts\b|embedding|live-|-(preview|exp)-\d{2,4}[-_]|"
    r"-image\b|-image-preview\b|-customtools\b",
    re.IGNORECASE,
)

# Google's live Gemini catalogs currently include a mix of stale slugs and
# Gemma models whose TPM quotas are too small for normal Hermes agent traffic.
# Keep capability metadata available for direct/manual use, but hide these from
# the Gemini model catalogs we surface in setup and model selection.
_GOOGLE_HIDDEN_MODELS = frozenset({
    # Low-TPM Gemma models that trip Google input-token quota walls under
    # agent-style traffic despite advertising large context windows.
    "gemma-4-31b-it",
    "gemma-4-26b-it",
    "gemma-4-26b-a4b-it",
    "gemma-3-1b",
    "gemma-3-1b-it",
    "gemma-3-2b",
    "gemma-3-2b-it",
    "gemma-3-4b",
    "gemma-3-4b-it",
    "gemma-3-12b",
    "gemma-3-12b-it",
    "gemma-3-27b",
    "gemma-3-27b-it",
    # Stale/retired Google slugs that still surface through models.dev-backed
    # Gemini selection but 404 on the current Google endpoints.
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash-8b",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
})


def _should_hide_from_provider_catalog(provider: str, model_id: str) -> bool:
    provider_lower = (provider or "").strip().lower()
    model_lower = (model_id or "").strip().lower()
    if provider_lower in {"gemini", "google"} and model_lower in _GOOGLE_HIDDEN_MODELS:
        return True
    return False


async def list_agentic_models(provider: str) -> List[str]:
    """Return model IDs suitable for agentic use from models.dev.

    Filters for tool_call=True and excludes noise (TTS, embedding,
    dated preview snapshots, live/streaming, image-only models).
    Returns an empty list on any failure.
    """
    models = await _get_provider_models(provider)
    if models is None:
        return []

    result = []
    for mid, entry in models.items():
        if not isinstance(entry, dict):
            continue
        if _should_hide_from_provider_catalog(provider, mid):
            continue
        if not entry.get("tool_call", False):
            continue
        if _NOISE_PATTERNS.search(mid):
            continue
        result.append(mid)
    return result



# ---------------------------------------------------------------------------
# Rich dataclass constructors — parse raw models.dev JSON into dataclasses
# ---------------------------------------------------------------------------

def _parse_model_info(model_id: str, raw: Dict[str, Any], provider_id: str) -> ModelInfo:
    """Convert a raw models.dev model entry dict into a ModelInfo dataclass."""
    limit = raw.get("limit") or {}
    if not isinstance(limit, dict):
        limit = {}

    cost = raw.get("cost") or {}
    if not isinstance(cost, dict):
        cost = {}

    modalities = raw.get("modalities") or {}
    if not isinstance(modalities, dict):
        modalities = {}

    input_mods = modalities.get("input") or []
    output_mods = modalities.get("output") or []

    ctx = limit.get("context")
    ctx_int = int(ctx) if isinstance(ctx, (int, float)) and ctx > 0 else 0
    out = limit.get("output")
    out_int = int(out) if isinstance(out, (int, float)) and out > 0 else 0
    inp = limit.get("input")
    inp_int = int(inp) if isinstance(inp, (int, float)) and inp > 0 else None

    return ModelInfo(
        id=model_id,
        name=raw.get("name", "") or model_id,
        family=raw.get("family", "") or "",
        provider_id=provider_id,
        reasoning=bool(raw.get("reasoning", False)),
        tool_call=bool(raw.get("tool_call", False)),
        attachment=bool(raw.get("attachment", False)),
        temperature=bool(raw.get("temperature", False)),
        structured_output=bool(raw.get("structured_output", False)),
        open_weights=bool(raw.get("open_weights", False)),
        input_modalities=tuple(input_mods) if isinstance(input_mods, list) else (),
        output_modalities=tuple(output_mods) if isinstance(output_mods, list) else (),
        context_window=ctx_int,
        max_output=out_int,
        max_input=inp_int,
        cost_input=float(cost.get("input", 0) or 0),
        cost_output=float(cost.get("output", 0) or 0),
        cost_cache_read=float(cost["cache_read"]) if "cache_read" in cost and cost["cache_read"] is not None else None,
        cost_cache_write=float(cost["cache_write"]) if "cache_write" in cost and cost["cache_write"] is not None else None,
        knowledge_cutoff=raw.get("knowledge", "") or "",
        release_date=raw.get("release_date", "") or "",
        status=raw.get("status", "") or "",
        interleaved=raw.get("interleaved", False),
    )


def _parse_provider_info(provider_id: str, raw: Dict[str, Any]) -> ProviderInfo:
    """Convert a raw models.dev provider entry dict into a ProviderInfo."""
    env = raw.get("env") or []
    models = raw.get("models") or {}
    return ProviderInfo(
        id=provider_id,
        name=raw.get("name", "") or provider_id,
        env=tuple(env) if isinstance(env, list) else (),
        api=raw.get("api", "") or "",
        doc=raw.get("doc", "") or "",
        model_count=len(models) if isinstance(models, dict) else 0,
    )


# ---------------------------------------------------------------------------
# Provider-level queries
# ---------------------------------------------------------------------------

async def get_provider_info(
    provider_id: str, *, allow_network: bool = True
) -> Optional[ProviderInfo]:
    """Get full provider metadata from models.dev.

    Accepts either a Hermes provider ID (e.g. "kilocode") or a models.dev
    ID (e.g. "kilo").  Returns None if the provider is not in the catalog.
    """
    # Resolve Hermes ID → models.dev ID
    mdev_id = PROVIDER_TO_MODELS_DEV.get(provider_id, provider_id)

    # NOTE: keep the zero-argument call on the default path. Dozens of test
    # sites monkeypatch fetch_models_dev with zero-arg lambdas; passing the
    # kwarg unconditionally would break them all (they raise TypeError).
    data = await fetch_models_dev(allow_network=allow_network)
    raw = data.get(mdev_id)
    if not isinstance(raw, dict):
        return None

    return _parse_provider_info(mdev_id, raw)


# ---------------------------------------------------------------------------
# Model-level queries (rich ModelInfo)
# ---------------------------------------------------------------------------

async def get_model_info(
    provider_id: str, model_id: str
) -> Optional[ModelInfo]:
    """Get full model metadata from models.dev.

    Accepts Hermes or models.dev provider ID.  Tries exact match then
    case-insensitive fallback.  Returns None if not found.
    """
    mdev_id = PROVIDER_TO_MODELS_DEV.get(provider_id, provider_id)

    data = await fetch_models_dev()
    pdata = data.get(mdev_id)
    if not isinstance(pdata, dict):
        return None

    models = pdata.get("models", {})
    if not isinstance(models, dict):
        return None

    # Exact match
    raw = models.get(model_id)
    if isinstance(raw, dict):
        return _parse_model_info(model_id, raw, mdev_id)

    # Case-insensitive fallback
    model_lower = model_id.lower()
    for mid, mdata in models.items():
        if mid.lower() == model_lower and isinstance(mdata, dict):
            return _parse_model_info(mid, mdata, mdev_id)

    return None
