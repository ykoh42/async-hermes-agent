"""Models.dev registry integration — primary database for providers and models.

Fetches from https://models.dev/api.json — a community-maintained database
of 4000+ models across 109+ providers.  Provides:

- **Provider metadata**: name, base URL, env vars, documentation link
- **Model metadata**: context window, max output, cost/M tokens, capabilities
  (reasoning, tools, vision, PDF, audio), modalities, knowledge cutoff,
  open-weights flag, family grouping, deprecation status

Data resolution order:
  1. In-memory cache (fresh, or stale served immediately while a single
     background async task refreshes)
  2. Disk cache (~/.hermes/models_dev_cache.json — any age; stale data is
     served rather than blocking callers on the network)
  3. Network fetch (https://models.dev/api.json) — only when no cache
     exists at all; failed refreshes back off for 5 minutes process-wide
Latency-sensitive callers pass ``allow_network=False`` and never touch the
network.

Other modules should import the dataclasses and query functions from here
rather than parsing the raw JSON themselves.
"""

import json
import logging
import time
import asyncio
import concurrent.futures
import contextvars
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MODELS_DEV_URL = "https://models.dev/api.json"
_MODELS_DEV_CACHE_TTL = 3600  # 1 hour in-memory
_MODELS_DEV_RETRY_DELAY = 300  # 5 minutes after a failed refresh

# In-memory cache
_models_dev_cache: dict[str, Any] = {}
_models_dev_cache_time: float = 0
_models_dev_retry_after: float = 0
# Retained as a private compatibility seam for upstream-derived tests. The
# native runtime uses the loop-neutral completion claim below instead of
# binding an asyncio.Lock to the first event loop that reaches the cache.
_models_dev_lock: asyncio.Lock | None = None
_models_dev_refresh_task: asyncio.Task[None] | None = None
_models_dev_update_guard = threading.RLock()
_models_dev_update_claim: concurrent.futures.Future[bool] | None = None
_models_dev_profile_context: contextvars.ContextVar[
    tuple[str, str] | None
] = contextvars.ContextVar("models_dev_profile_scope", default=None)
_models_dev_profile_aliases: dict[str, str] = {}


@dataclass
class _ModelsDevProfileState:
    cache: dict[str, Any] = field(default_factory=dict)
    cache_time: float = 0
    retry_after: float = 0
    refresh_task: asyncio.Task[None] | None = None
    refresh_claim: concurrent.futures.Future[bool] | None = None
    update_claim: concurrent.futures.Future[bool] | None = None


_models_dev_profile_states: dict[str, _ModelsDevProfileState] = {}
_models_dev_legacy_snapshot: tuple = (
    id(_models_dev_cache),
    _models_dev_cache_time,
    _models_dev_retry_after,
    _models_dev_refresh_task,
    _models_dev_update_claim,
)


def _lexical_models_dev_profile() -> str:
    """Return the active profile marker without filesystem access."""
    from hermes_constants import get_hermes_home

    return os.path.normcase(os.fspath(get_hermes_home()))


def _current_models_dev_profile() -> str:
    lexical = _lexical_models_dev_profile()
    active = _models_dev_profile_context.get()
    if active is not None and active[0] == lexical:
        return active[1]
    with _models_dev_update_guard:
        return _models_dev_profile_aliases.get(lexical, lexical)


def _models_dev_state() -> _ModelsDevProfileState:
    profile = _current_models_dev_profile()
    with _models_dev_update_guard:
        return _models_dev_profile_states.setdefault(
            profile,
            _ModelsDevProfileState(),
        )


def _publish_models_dev_legacy_state(state: _ModelsDevProfileState) -> None:
    """Expose the active profile through retained private scalar seams."""
    global _models_dev_cache, _models_dev_cache_time, _models_dev_retry_after
    global _models_dev_refresh_task, _models_dev_update_claim
    global _models_dev_legacy_snapshot
    with _models_dev_update_guard:
        _models_dev_cache = state.cache
        _models_dev_cache_time = state.cache_time
        _models_dev_retry_after = state.retry_after
        _models_dev_refresh_task = state.refresh_task
        _models_dev_update_claim = state.update_claim
        _models_dev_legacy_snapshot = (
            id(_models_dev_cache),
            _models_dev_cache_time,
            _models_dev_retry_after,
            _models_dev_refresh_task,
            _models_dev_update_claim,
        )


def _capture_models_dev_legacy_overrides(state: _ModelsDevProfileState) -> None:
    """Apply direct assignments made through upstream-derived test seams."""
    current = (
        id(_models_dev_cache),
        _models_dev_cache_time,
        _models_dev_retry_after,
        _models_dev_refresh_task,
        _models_dev_update_claim,
    )
    previous = _models_dev_legacy_snapshot
    if current != previous:
        # Test fixtures reset these coupled cache fields through direct module
        # assignments. Treat any observed change as one atomic legacy state so
        # reusing the same registry object identity cannot miss the reset.
        state.cache = _models_dev_cache
        state.cache_time = _models_dev_cache_time
        state.retry_after = _models_dev_retry_after
        state.refresh_task = _models_dev_refresh_task
        state.update_claim = _models_dev_update_claim
    _publish_models_dev_legacy_state(state)


async def _activate_models_dev_profile() -> _ModelsDevProfileState:
    """Activate the canonical Hermes profile at an existing awaited edge."""
    lexical = _lexical_models_dev_profile()
    active = _models_dev_profile_context.get()
    if active is not None and active[0] == lexical:
        canonical = active[1]
    else:
        with _models_dev_update_guard:
            canonical = _models_dev_profile_aliases.get(lexical)
        if canonical is None:
            import aiofiles.os

            expanduser = aiofiles.os.wrap(os.path.expanduser)
            expanded = str(await expanduser(lexical))
            is_absolute = (
                expanded.startswith(("/", "\\\\"))
                or (
                    len(expanded) >= 3
                    and expanded[1] == ":"
                    and expanded[2] in "/\\"
                )
            )
            if not is_absolute:
                expanded = str(await aiofiles.os.getcwd()) + os.sep + expanded
            realpath = aiofiles.os.wrap(os.path.realpath)
            canonical = os.path.normcase(str(await realpath(expanded)))
    with _models_dev_update_guard:
        _models_dev_profile_aliases[lexical] = canonical
        target = _models_dev_profile_states.setdefault(
            canonical,
            _ModelsDevProfileState(),
        )
        if lexical != canonical:
            staged = _models_dev_profile_states.pop(lexical, None)
            if staged is not None and staged is not target:
                if staged.cache:
                    target.cache = staged.cache
                    target.cache_time = staged.cache_time
                target.retry_after = max(target.retry_after, staged.retry_after)
                if target.refresh_task is None:
                    target.refresh_task = staged.refresh_task
                    target.refresh_claim = staged.refresh_claim
                if target.update_claim is None:
                    target.update_claim = staged.update_claim
        _models_dev_profile_context.set((lexical, canonical))
        _capture_models_dev_legacy_overrides(target)
        return target


def _claim_models_dev_update(
) -> tuple[bool, concurrent.futures.Future[bool]]:
    """Claim one profile-local models.dev update without loop affinity."""
    state = _models_dev_state()
    with _models_dev_update_guard:
        claim = state.update_claim
        if claim is None or claim.done():
            claim = concurrent.futures.Future()
            state.update_claim = claim
            _publish_models_dev_legacy_state(state)
            return True, claim
        return False, claim


def _finish_models_dev_update(
    claim: concurrent.futures.Future[bool],
    *,
    completed: bool,
) -> None:
    """Publish update completion and release the profile-local claim."""
    state = _models_dev_state()
    with _models_dev_update_guard:
        if state.update_claim is claim:
            state.update_claim = None
        if not claim.done():
            claim.set_result(completed)
        _publish_models_dev_legacy_state(state)


async def _wait_for_models_dev_update(
    claim: concurrent.futures.Future[bool],
) -> bool:
    """Await another loop's update without letting waiter cancellation own it."""
    return await asyncio.shield(asyncio.wrap_future(claim))


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
    input_modalities: tuple[str, ...] = ()    # ("text", "image", "pdf", ...)
    output_modalities: tuple[str, ...] = ()

    # Limits
    context_window: int = 0
    max_output: int = 0
    max_input: int | None = None

    # Cost (per million tokens, USD)
    cost_input: float = 0.0
    cost_output: float = 0.0
    cost_cache_read: float | None = None
    cost_cache_write: float | None = None

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
    env: tuple[str, ...]            # env var names for API key
    api: str                        # base URL
    doc: str = ""                   # documentation URL
    model_count: int = 0


# ---------------------------------------------------------------------------
# Provider ID mapping: Hermes ↔ models.dev
# ---------------------------------------------------------------------------

# Hermes provider names → models.dev provider IDs
PROVIDER_TO_MODELS_DEV: dict[str, str] = {
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
_MODELS_DEV_TO_PROVIDER: dict[str, str] | None = None



def _get_cache_path() -> Path:
    """Return path to disk cache file."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "models_dev_cache.json"


async def _load_disk_cache() -> dict[str, Any]:
    """Load models.dev data from disk cache using native async file I/O."""
    try:
        import aiofiles
        import aiofiles.os

        cache_path = _get_cache_path()
        if await aiofiles.os.path.exists(cache_path):
            async with aiofiles.open(cache_path, encoding="utf-8") as fh:
                data = json.loads(await fh.read())
            return data if isinstance(data, dict) else {}
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.debug("Failed to load models.dev disk cache: %s", exc)
    return {}


async def _disk_cache_age_seconds() -> float | None:
    """Return disk-cache age, or None when it cannot be determined."""
    try:
        import aiofiles.os

        cache_path = _get_cache_path()
        if not await aiofiles.os.path.exists(cache_path):
            return None
        stat_result = await aiofiles.os.stat(cache_path)
        age = time.time() - stat_result.st_mtime
        return age if age >= 0 else None
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.debug("Failed to stat models.dev disk cache: %s", exc)
        return None


async def _save_disk_cache(data: dict[str, Any]) -> None:
    """Atomically save models.dev data with native async file operations."""
    import aiofiles
    import aiofiles.os

    cache_path = _get_cache_path()
    temp_path = cache_path.with_name(f".{cache_path.name}.tmp")
    try:
        await aiofiles.os.makedirs(cache_path.parent, exist_ok=True)
        async with aiofiles.open(temp_path, "w", encoding="utf-8") as fh:
            await fh.write(json.dumps(data, separators=(",", ":")))
        await aiofiles.os.replace(temp_path, cache_path)
    except asyncio.CancelledError:
        try:
            await aiofiles.os.remove(temp_path)
        except OSError:
            pass
        raise
    except Exception as exc:
        try:
            await aiofiles.os.remove(temp_path)
        except OSError:
            pass
        logger.debug("Failed to save models.dev disk cache: %s", exc)


async def _fetch_models_dev_from_network() -> dict[str, Any]:
    """Fetch the live registry without touching local caches."""
    import httpx

    from agent.ssl_verify import _create_httpx_client

    async with (await _create_httpx_client(
        timeout=httpx.Timeout(10.0, connect=5.0),
    )) as client:
        response = await client.get(MODELS_DEV_URL)
        response.raise_for_status()
        data = response.json()
    if not isinstance(data, dict) or not data:
        raise ValueError("models.dev returned an empty or invalid registry")
    return data


def _mark_stale_cache_grace() -> None:
    """Give stale cache data a short in-memory grace before retrying."""
    state = _models_dev_state()
    grace_time = time.time() - _MODELS_DEV_CACHE_TTL + _MODELS_DEV_RETRY_DELAY
    if grace_time > state.cache_time:
        state.cache_time = grace_time
        _publish_models_dev_legacy_state(state)


async def _commit_registry(data: dict[str, Any], *, where: str) -> None:
    """Persist a freshly fetched registry and clear failure backoff."""
    state = _models_dev_state()
    await _save_disk_cache(data)
    state.cache = data
    state.cache_time = time.time()
    state.retry_after = 0
    _publish_models_dev_legacy_state(state)
    logger.debug(
        "Refreshed models.dev registry (%s): %d providers, %d total models",
        where,
        len(data),
        sum(
            len(provider.get("models", {}))
            for provider in data.values()
            if isinstance(provider, dict)
        ),
    )


def _note_refresh_failure(exc: Exception, *, where: str) -> None:
    """Arm the active profile's retry backoff after a failed refresh."""
    state = _models_dev_state()
    state.retry_after = time.time() + _MODELS_DEV_RETRY_DELAY
    _publish_models_dev_legacy_state(state)
    logger.debug(
        "models.dev refresh failed (%s); retry suppressed for %ds: %s",
        where,
        _MODELS_DEV_RETRY_DELAY,
        exc,
    )


async def _background_refresh_models_dev() -> None:
    """Best-effort native async refresh after serving stale cache data."""
    owned_claim: concurrent.futures.Future[bool] | None = None
    completed = False
    try:
        state = await _activate_models_dev_profile()
        current_task = asyncio.current_task()
        if state.refresh_task is current_task:
            owned_claim = state.refresh_claim
        if owned_claim is None:
            owner, claim = _claim_models_dev_update()
            if not owner:
                await _wait_for_models_dev_update(claim)
                return
            owned_claim = claim
        data = await _fetch_models_dev_from_network()
        await _commit_registry(data, where="background")
        completed = True
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _note_refresh_failure(exc, where="background")
        completed = True
    finally:
        if owned_claim is not None:
            _finish_models_dev_update(owned_claim, completed=completed)


def _consume_refresh_task(task: asyncio.Task[None]) -> None:
    """Clear the tracked refresh task and consume terminal exceptions."""
    state = _models_dev_state()
    with _models_dev_update_guard:
        if state.refresh_task is task:
            state.refresh_task = None
            claim = state.refresh_claim
            state.refresh_claim = None
            # A task cancelled before its coroutine starts cannot execute the
            # coroutine's finally block. Release only its own claim here; a
            # later foreground owner may already hold a different claim.
            if claim is not None and state.update_claim is claim:
                state.update_claim = None
                if not claim.done():
                    claim.set_result(False)
        _publish_models_dev_legacy_state(state)
    if task.cancelled():
        return
    try:
        task.exception()
    except Exception:
        logger.debug("models.dev background refresh failed", exc_info=True)


def _start_background_refresh_models_dev() -> None:
    """Start at most one event-loop-owned refresh task outside backoff."""
    state = _models_dev_state()
    if time.time() < state.retry_after:
        return
    owner, claim = _claim_models_dev_update()
    if not owner:
        return
    try:
        task = asyncio.create_task(
            _background_refresh_models_dev(),
            name="models-dev-refresh",
        )
    except BaseException:
        _finish_models_dev_update(claim, completed=False)
        raise
    with _models_dev_update_guard:
        state.refresh_task = task
        state.refresh_claim = claim
        _publish_models_dev_legacy_state(state)
    task.add_done_callback(_consume_refresh_task)


async def lookup_models_dev_context(provider: str, model: str) -> int | None:
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


def _extract_context(entry: dict[str, Any]) -> int | None:
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


async def _get_provider_models(provider: str) -> dict[str, Any] | None:
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


def _find_model_entry(models: dict[str, Any], model: str) -> dict[str, Any] | None:
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
) -> dict[str, Any]:
    """Fetch models.dev with upstream cache semantics and native async I/O.

    Fresh memory wins. Stale memory or disk data is returned immediately
    while one tracked async refresh runs. With no cache, callers share one
    foreground request. Failed automatic refreshes back off for five minutes;
    ``force_refresh`` bypasses both caches and the backoff.
    """
    state = await _activate_models_dev_profile()

    if not allow_network:
        if state.cache:
            return state.cache
        disk_data = await _load_disk_cache()
        if disk_data:
            state.cache = disk_data
            disk_age = await _disk_cache_age_seconds()
            state.cache_time = (
                time.time() - disk_age if disk_age is not None else 0
            )
            _publish_models_dev_legacy_state(state)
        return state.cache

    if (
        not force_refresh
        and state.cache
        and (time.time() - state.cache_time) < _MODELS_DEV_CACHE_TTL
    ):
        return state.cache

    if not force_refresh and state.cache:
        _mark_stale_cache_grace()
        _start_background_refresh_models_dev()
        logger.debug(
            "Using stale in-memory models.dev cache; refreshing in background"
        )
        return state.cache

    if not force_refresh:
        disk_age = await _disk_cache_age_seconds()
        if disk_age is not None:
            disk_data = await _load_disk_cache()
            if disk_data:
                state.cache = disk_data
                if disk_age < _MODELS_DEV_CACHE_TTL:
                    state.cache_time = time.time() - disk_age
                    _publish_models_dev_legacy_state(state)
                    logger.debug(
                        "Loaded models.dev from fresh disk cache "
                        "(%d providers, age=%.0fs)",
                        len(disk_data),
                        disk_age,
                    )
                else:
                    _mark_stale_cache_grace()
                    _start_background_refresh_models_dev()
                    logger.debug(
                        "Using stale models.dev disk cache (age=%.0fs); "
                        "refreshing in background",
                        disk_age,
                    )
                return state.cache

    if not force_refresh and time.time() < state.retry_after:
        return state.cache

    owner, claim = _claim_models_dev_update()
    if not owner:
        await _wait_for_models_dev_update(claim)
        # A forced refresh retains upstream's serialized refresh semantics:
        # every explicit caller performs its own fetch after the predecessor.
        if force_refresh:
            return await fetch_models_dev(force_refresh=True)
        if state.cache or time.time() < state.retry_after:
            return state.cache
        return await fetch_models_dev()

    completed = False
    try:
        if not force_refresh:
            if state.cache:
                completed = True
                return state.cache
            if time.time() < state.retry_after:
                completed = True
                return state.cache

        try:
            data = await _fetch_models_dev_from_network()
            await _commit_registry(data, where="foreground")
            completed = True
            return data
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _note_refresh_failure(exc, where="foreground")

        if not state.cache:
            state.cache = await _load_disk_cache()
            state.cache_time = 0
            _publish_models_dev_legacy_state(state)
            if state.cache:
                logger.debug(
                    "Loaded stale models.dev disk cache (%d providers)",
                    len(state.cache),
                )
        completed = True
        return state.cache
    finally:
        _finish_models_dev_update(claim, completed=completed)


async def get_model_capabilities(
    provider: str,
    model: str,
) -> ModelCapabilities | None:
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


async def list_provider_models(provider: str) -> list[str]:
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


async def list_agentic_models(provider: str) -> list[str]:
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

def _parse_model_info(model_id: str, raw: dict[str, Any], provider_id: str) -> ModelInfo:
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


def _parse_provider_info(provider_id: str, raw: dict[str, Any]) -> ProviderInfo:
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
) -> ProviderInfo | None:
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
) -> ModelInfo | None:
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
