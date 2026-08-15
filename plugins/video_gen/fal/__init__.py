"""FAL.ai video generation backend.

User-facing surface: pick a **model family** (e.g. "Pixverse v6",
"Veo 3.1", "Seedance 2.0", "Kling v3 4K", "LTX 2.3", "Happy Horse").
The plugin auto-routes to the family's text-to-video endpoint when
called without ``image_url``, and to its image-to-video endpoint when
``image_url`` is provided. The agent never sees the routing — it just
calls ``video_generate(prompt=..., image_url=...)``.

Model families (most expose both t2v + i2v; gemini-omni-flash is image-to-video only):

  Cheap tier:
    ltx-2.3            fal-ai/ltx-2.3-22b/text-to-video           /  fal-ai/ltx-2.3-22b/image-to-video
    pixverse-v6        fal-ai/pixverse/v6/text-to-video           /  fal-ai/pixverse/v6/image-to-video
    seedance-2.0-mini  bytedance/seedance-2.0/mini/text-to-video  /  bytedance/seedance-2.0/mini/image-to-video

  Premium tier:
    veo3.1             fal-ai/veo3.1                              /  fal-ai/veo3.1/image-to-video
    seedance-2.0       bytedance/seedance-2.0/text-to-video       /  bytedance/seedance-2.0/image-to-video
    seedance-2.5       bytedance/seedance-2.5/text-to-video       /  bytedance/seedance-2.5/image-to-video
    minimax-h3         minimax/h3/text-to-video                   /  minimax/h3/image-to-video
    flux-3             blackforestlabs/flux-3/text-to-video       /  blackforestlabs/flux-3/image-to-video
    grok-imagine-1.5   xai/grok-imagine-video/v1.5/text-to-video  /  xai/grok-imagine-video/v1.5/image-to-video
    kling-v3-4k        fal-ai/kling-video/v3/4k/text-to-video     /  fal-ai/kling-video/v3/4k/image-to-video
    happy-horse        alibaba/happy-horse/text-to-video          /  alibaba/happy-horse/image-to-video

  Image-to-video only (no text_endpoint):
    gemini-omni-flash  google/gemini-omni-flash/image-to-video

Selection precedence for the active family:
    1. ``model=`` arg from the tool call
    2. ``FAL_VIDEO_MODEL`` env var
    3. ``video_gen.fal.model`` in ``config.yaml``
    4. ``video_gen.model`` in ``config.yaml`` (when it's one of our family IDs)
    5. ``DEFAULT_MODEL``

Authentication via ``FAL_KEY`` or the managed Nous gateway. Output is an
HTTPS URL from FAL's CDN; the gateway downloads and delivers it.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from agent.secret_scope import get_secret
from agent.video_gen_provider import (
    VideoGenProvider,
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Family catalog
# ---------------------------------------------------------------------------
#
# Each family declares both endpoints (when available) plus a per-family
# capability sheet derived from FAL's OpenAPI schemas. Capability flags
# drive which keys get added to the request payload — keys a family doesn't
# advertise are dropped before send.
#
# Capabilities:
#   aspect_ratios  : tuple of supported ratios (None = endpoint decides)
#   resolutions    : tuple of supported resolutions (None = endpoint decides)
#   durations      : tuple of supported durations OR (min, max) range
#                    (heuristic: 2-element with gap > 1 is a range)
#   audio          : True if generate_audio is supported
#   negative       : True if negative_prompt is supported
#   seed           : False when the endpoint declares no `seed` field
#                    (absent = True, so existing families keep sending it)
#   duration_int   : True when FAL types duration as an integer rather than
#                    the usual queue-API string

FAL_FAMILIES: dict[str, dict[str, Any]] = {
    # ─── Cheap / fast tier ─────────────────────────────────────────────
    "ltx-2.3": {
        "display": "LTX 2.3 (22B)",
        "speed": "~30-60s",
        "price": "cheap",
        "strengths": "22B model with native audio generation. Affordable.",
        "tier": "cheap",
        "text_endpoint": "fal-ai/ltx-2.3-22b/text-to-video",
        "image_endpoint": "fal-ai/ltx-2.3-22b/image-to-video",
        # LTX docs don't expose duration/aspect/resolution enums — leave
        # blank so we don't send unrecognized payload keys.
        "aspect_ratios": None,
        "resolutions": None,
        "durations": None,
        "audio": True,
        "negative": True,
    },
    "pixverse-v6": {
        "display": "Pixverse v6",
        "speed": "~30-90s",
        "price": "cheap",
        "strengths": "Affordable. Negative prompts. 1-15s durations.",
        "tier": "cheap",
        "text_endpoint": "fal-ai/pixverse/v6/text-to-video",
        "image_endpoint": "fal-ai/pixverse/v6/image-to-video",
        "aspect_ratios": None,
        "resolutions": ("360p", "540p", "720p", "1080p"),
        "durations": (1, 15),
        "audio": True,
        "negative": True,
    },
    "seedance-2.0-mini": {
        "display": "Seedance 2.0 Mini",
        "speed": "~30-90s",
        "price": "cheap",
        "strengths": "ByteDance. Faster/cheaper Seedance tier, audio + lip-sync, 4-15s.",
        "tier": "cheap",
        "text_endpoint": "bytedance/seedance-2.0/mini/text-to-video",
        "image_endpoint": "bytedance/seedance-2.0/mini/image-to-video",
        "aspect_ratios": ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16"),
        "resolutions": ("480p", "720p"),
        "durations": (4, 15),
        "audio": True,
        "negative": False,
        "seed": False,
    },
    # ─── Expensive / premium tier ──────────────────────────────────────
    "veo3.1": {
        "display": "Veo 3.1",
        "speed": "~60-120s",
        "price": "premium",
        "strengths": "Google DeepMind. Cinematic, native audio, strong prompt adherence.",
        "tier": "premium",
        "text_endpoint": "fal-ai/veo3.1",
        "image_endpoint": "fal-ai/veo3.1/image-to-video",
        "aspect_ratios": ("16:9", "9:16"),
        "resolutions": ("720p", "1080p", "4k"),
        "durations": (4, 6, 8),
        "duration_suffix": "s",  # FAL veo3.1 wants "4s" not "4"
        "audio": True,
        "negative": True,
    },
    "seedance-2.0": {
        "display": "Seedance 2.0",
        "speed": "~60-120s",
        "price": "premium",
        "strengths": "ByteDance. Cinematic, synchronized audio + lip-sync, 4-15s.",
        "tier": "premium",
        "text_endpoint": "bytedance/seedance-2.0/text-to-video",
        "image_endpoint": "bytedance/seedance-2.0/image-to-video",
        # Seedance accepts "auto" too — we omit it from the enum so the
        # agent can't pass it; the endpoint defaults handle the rest.
        "aspect_ratios": ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16"),
        "resolutions": ("480p", "720p", "1080p"),
        "durations": (4, 15),
        "audio": True,
        "negative": False,
        # FAL input schema has no `seed` (only returned on output).
        "seed": False,
    },
    "seedance-2.5": {
        "display": "Seedance 2.5",
        "speed": "~60-180s",
        "price": "premium",
        "strengths": "ByteDance flagship. Native 30s single-pass, audio in the same latent space, lip-sync.",
        "tier": "premium",
        "text_endpoint": "bytedance/seedance-2.5/text-to-video",
        "image_endpoint": "bytedance/seedance-2.5/image-to-video",
        "image_drop_keys": ("aspect_ratio",),
        "aspect_ratios": ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16"),
        "resolutions": ("480p", "720p"),
        "durations": (4, 30),
        "audio": True,
        "negative": False,
        "seed": False,
    },
    "minimax-h3": {
        "display": "MiniMax H3",
        "speed": "~60-180s",
        "price": "premium",
        "strengths": "MiniMax frontier. Native 2K (up to 4K), 5-15s, seven aspect ratios.",
        "tier": "premium",
        "text_endpoint": "minimax/h3/text-to-video",
        "image_endpoint": "minimax/h3/image-to-video",
        "duration_int": True,
        "image_drop_keys": ("aspect_ratio",),
        "aspect_ratios": ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16"),
        "resolutions": ("768P", "2K", "4K"),
        "resolution_aliases": {
            "480p": "768P", "540p": "768P", "720p": "768P", "768p": "768P",
            "1080p": "2K", "2k": "2K", "4k": "4K", "2160p": "4K",
        },
        "durations": (5, 15),
        "audio": False,
        "negative": False,
        "seed": False,
    },
    "flux-3": {
        "display": "FLUX 3 (via FAL)",
        "speed": "~60-120s",
        "price": "premium",
        "strengths": "Black Forest Labs frontier video. Native audio, 5-20s, 8 aspect ratios.",
        "tier": "premium",
        "text_endpoint": "blackforestlabs/flux-3/text-to-video",
        "image_endpoint": "blackforestlabs/flux-3/image-to-video",
        "duration_int": True,
        "aspect_ratios": ("21:9", "2:1", "16:9", "4:3", "1:1", "3:4", "9:16"),
        "resolutions": ("720p", "1080p"),
        "durations": (5, 20),
        "audio": True,
        "negative": False,
        "seed": False,
    },
    "grok-imagine-1.5": {
        "display": "Grok Imagine 1.5 (via FAL)",
        "speed": "~30-90s",
        "price": "premium",
        "strengths": "xAI. Fast stylized video with audio, 1-15s, cheap per second.",
        "tier": "premium",
        "text_endpoint": "xai/grok-imagine-video/v1.5/text-to-video",
        "image_endpoint": "xai/grok-imagine-video/v1.5/image-to-video",
        "duration_int": True,
        "image_drop_keys": ("aspect_ratio",),
        "aspect_ratios": ("16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16"),
        "resolutions": ("480p", "720p", "1080p"),
        "durations": (1, 15),
        "audio": False,
        "negative": False,
        "seed": False,
    },
    "gemini-omni-flash": {
        "display": "Gemini Omni Flash (via FAL)",
        "speed": "~60-120s",
        "price": "premium",
        "strengths": "Google. Image-to-video with audio, physics-grounded motion, 3-10s.",
        "tier": "premium",
        "text_endpoint": None,
        "image_endpoint": "google/gemini-omni-flash/image-to-video",
        "duration_int": True,
        "aspect_ratios": ("16:9", "9:16"),
        "resolutions": None,
        "durations": (3, 10),
        "audio": False,
        "negative": False,
        "seed": False,
    },
    "kling-v3-4k": {
        "display": "Kling v3 4K",
        "speed": "~120-300s",
        "price": "premium",
        "strengths": "4K output, native audio (Chinese/English), 3-15s.",
        "tier": "premium",
        "text_endpoint": "fal-ai/kling-video/v3/4k/text-to-video",
        "image_endpoint": "fal-ai/kling-video/v3/4k/image-to-video",
        # Kling 4K image-to-video uses `start_image_url` instead of
        # `image_url`. Handled in _build_payload via image_param_key.
        "image_param_key": "start_image_url",
        "aspect_ratios": ("16:9", "9:16", "1:1"),
        "resolutions": None,  # 4K is implicit
        "durations": (3, 15),
        "audio": True,
        "negative": True,
    },
    "happy-horse": {
        "display": "Happy Horse 1.0",
        "speed": "~60-120s",
        "price": "premium",
        "strengths": "Alibaba. New model, sparse public docs — conservative defaults.",
        "tier": "premium",
        "text_endpoint": "alibaba/happy-horse/text-to-video",
        "image_endpoint": "alibaba/happy-horse/image-to-video",
        # Docs don't expose duration/aspect/resolution — let the endpoint
        # apply its own defaults.
        "aspect_ratios": None,
        "resolutions": None,
        "durations": None,
        "audio": False,
        "negative": False,
    },
}

DEFAULT_MODEL = "pixverse-v6"  # cheap, both modalities, sane defaults


def _is_duration_range(durations: Any) -> bool:
    """Heuristic: a 2-tuple of ints with a gap > 1 is treated as ``(min, max)``."""
    if not isinstance(durations, tuple) or len(durations) != 2:
        return False
    if not all(isinstance(d, int) for d in durations):
        return False
    return durations[1] - durations[0] > 1


def _clamp_duration(family: dict[str, Any], duration: int | None) -> int | None:
    durations = family.get("durations")
    if not durations:
        return duration
    if duration is None:
        # Range families (e.g. pixverse-v6 (1,15)) should omit the field so
        # the FAL endpoint applies its own default rather than receiving the
        # minimum value.  Enum families (e.g. veo3.1 (4,6,8)) keep sending
        # their first entry as the default.
        return None if _is_duration_range(durations) else durations[0]
    if _is_duration_range(durations):
        lo, hi = durations
        return max(lo, min(hi, duration))
    # enum
    if duration in durations:
        return duration
    return min(durations, key=lambda d: abs(d - duration))


# ---------------------------------------------------------------------------
# Config / model resolution
# ---------------------------------------------------------------------------


async def _load_video_gen_section() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        cfg = await load_config_readonly()
        section = cfg.get("video_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load video_gen config: %s", exc)
        return {}


_ENDPOINT_MODALITY_LEAVES = frozenset({"text-to-video", "image-to-video"})


def _normalize_family_key(candidate: str) -> str | None:
    """Extract a known family from a bare id or a full FAL endpoint path."""
    value = candidate.strip()
    if not value:
        return None
    if value in FAL_FAMILIES:
        return value

    # Exact endpoint match is unambiguous and must win over segment matching.
    for family_id, meta in FAL_FAMILIES.items():
        if value in (meta.get("text_endpoint"), meta.get("image_endpoint")):
            return family_id

    # A configured endpoint stem such as ``minimax/h3`` or
    # ``bytedance/seedance-2.0/mini`` is accepted only when the following
    # segment is a declared modality leaf.
    stem_hits: list[tuple[int, str]] = []
    for family_id, meta in FAL_FAMILIES.items():
        for endpoint in (meta.get("text_endpoint"), meta.get("image_endpoint")):
            if not isinstance(endpoint, str) or not endpoint.startswith(value + "/"):
                continue
            next_segment = endpoint[len(value) + 1 :].split("/", 1)[0]
            if next_segment in _ENDPOINT_MODALITY_LEAVES:
                stem_hits.append((len(value), family_id))
                break
    if stem_hits:
        return max(stem_hits)[1]

    # Finally match the longest known family id appearing as a path segment.
    best_family: str | None = None
    best_length = -1
    parts = set(value.split("/"))
    for family_id in FAL_FAMILIES:
        if family_id in parts and len(family_id) > best_length:
            best_family = family_id
            best_length = len(family_id)
    return best_family


async def _resolve_family(explicit: str | None) -> tuple[str, dict[str, Any]]:
    """Decide which FAL family to use. Returns ``(family_id, meta)``."""
    candidates: list[str | None] = []
    candidates.append(explicit)
    candidates.append(get_secret("FAL_VIDEO_MODEL"))

    cfg = await _load_video_gen_section()
    fal_cfg = cfg.get("fal") if isinstance(cfg.get("fal"), dict) else {}
    if isinstance(fal_cfg, dict):
        candidates.append(fal_cfg.get("model"))
    top = cfg.get("model")
    if isinstance(top, str):
        candidates.append(top)

    for c in candidates:
        if isinstance(c, str) and c.strip():
            family_id = _normalize_family_key(c)
            if family_id:
                return family_id, FAL_FAMILIES[family_id]

    return DEFAULT_MODEL, FAL_FAMILIES[DEFAULT_MODEL]


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------


def _build_payload(
    family: dict[str, Any],
    *,
    prompt: str,
    image_url: str | None,
    duration: int | None,
    aspect_ratio: str,
    resolution: str,
    negative_prompt: str | None,
    audio: bool | None,
    seed: int | None,
) -> dict[str, Any]:
    """Build a family-specific payload, dropping keys the family doesn't declare."""
    payload: dict[str, Any] = {}

    if prompt:
        payload["prompt"] = prompt
    if image_url:
        # Some endpoints (e.g. Kling v3 4K image-to-video) expect
        # `start_image_url` instead of `image_url`. The family entry can
        # declare an override.
        key = family.get("image_param_key") or "image_url"
        payload[key] = image_url
    # Newer FAL schemas do not accept a seed input. Existing families omit the
    # flag and therefore retain the historical default of sending it.
    if seed is not None and family.get("seed", True):
        payload["seed"] = seed

    if family.get("aspect_ratios"):
        if aspect_ratio in family["aspect_ratios"]:
            payload["aspect_ratio"] = aspect_ratio
        # otherwise let the endpoint auto-crop / use its default

    if family.get("resolutions"):
        aliases = family.get("resolution_aliases") or {}
        resolved_resolution = aliases.get((resolution or "").lower(), resolution)
        if resolved_resolution in family["resolutions"]:
            payload["resolution"] = resolved_resolution
        # else: let the endpoint default

    clamped = _clamp_duration(family, duration)
    if clamped is not None and family.get("durations"):
        if family.get("duration_int"):
            payload["duration"] = clamped
        else:
            # FAL exposes duration as a string in the queue API ("8" not 8).
            # Some families (e.g. veo3.1) require a unit suffix ("4s").
            suffix = family.get("duration_suffix", "")
            payload["duration"] = f"{clamped}{suffix}"

    if family.get("audio") and audio is not None:
        payload["generate_audio"] = bool(audio)

    if family.get("negative") and negative_prompt:
        payload["negative_prompt"] = negative_prompt

    if image_url:
        for key in family.get("image_drop_keys", ()):
            payload.pop(key, None)

    return payload


# ---------------------------------------------------------------------------
# fal_client lazy import (shared with image_generation_tool via fal_common)
# ---------------------------------------------------------------------------

_fal_client: Any = None


def _load_fal_client() -> Any:
    """Lazy-load the ``fal_client`` SDK and cache it on this module.

    Delegates the actual import to :func:`tools.fal_common.import_fal_client`
    so the ``lazy_deps`` ensure-install handling stays in one place.

    Python's import lock already serializes concurrent first imports.
    """
    global _fal_client
    if _fal_client is not None:
        return _fal_client
    from tools.fal_common import import_fal_client

    _fal_client = import_fal_client()
    return _fal_client


# ---------------------------------------------------------------------------
async def _resolve_managed_fal_video_gateway():
    """Resolve managed FAL when requested or direct credentials are absent."""
    from tools.tool_backend_helpers import fal_key_is_configured, prefers_gateway

    if await fal_key_is_configured() and not await prefers_gateway("video_gen"):
        return None
    from tools.managed_tool_gateway import resolve_managed_tool_gateway

    return await resolve_managed_tool_gateway("fal-queue")


def _get_managed_fal_video_client(managed_gateway):
    """Create an owned native-async client for one managed FAL request."""
    from tools.fal_common import _ManagedFalClient

    _load_fal_client()
    return _ManagedFalClient(
        _fal_client,
        key=managed_gateway.nous_user_token,
        queue_run_origin=managed_gateway.gateway_origin,
    )


async def _submit_fal_video_request(
    endpoint: str,
    arguments: dict[str, Any],
    *,
    return_request_id: bool = False,
):
    """Submit through direct FAL or the managed queue.

    Returns the completed queue result without blocking the event loop.
    """
    _load_fal_client()
    from tools.fal_common import (
        _close_fal_client,
        _create_fal_client,
        _extract_http_status,
    )

    request_headers = {"x-idempotency-key": str(uuid.uuid4())}
    managed_gateway = await _resolve_managed_fal_video_gateway()
    if managed_gateway is None:
        client = await _create_fal_client(_fal_client)
        try:
            handle = await client.submit(
                endpoint,
                arguments=arguments,
                headers=request_headers,
            )
            result = await handle.get()
            if return_request_id:
                return result, getattr(handle, "request_id", None)
            return result
        finally:
            await _close_fal_client(client)

    managed_client = _get_managed_fal_video_client(managed_gateway)
    try:
        try:
            handle = await managed_client.submit(
                endpoint,
                arguments=arguments,
                headers=request_headers,
            )
        except Exception as exc:
            status = _extract_http_status(exc)
            if status is not None and 400 <= status < 500:
                raise ValueError(
                    f"Nous Subscription gateway rejected endpoint '{endpoint}' "
                    f"(HTTP {status}). This model may not yet be enabled on "
                    f"the Nous Portal's FAL proxy. Either:\n"
                    f"  • Set FAL_KEY in your environment to use FAL.ai "
                    f"directly, or\n"
                    f"  • Configure a different video_gen.model in "
                    f"config.yaml."
                ) from exc
            raise
        result = await handle.get()
        if return_request_id:
            return result, getattr(handle, "request_id", None)
        return result
    finally:
        await managed_client.close()


async def _check_fal_video_available() -> bool:
    """True if direct FAL or the managed gateway is available."""
    from tools.tool_backend_helpers import fal_key_is_configured

    if await fal_key_is_configured():
        return True
    return await _resolve_managed_fal_video_gateway() is not None


# ---------------------------------------------------------------------------
# Upscaler (SeedVR2 — video upscale pass)
# ---------------------------------------------------------------------------

UPSCALER_ENDPOINT = "fal-ai/seedvr/upscale/video"
UPSCALER_FACTOR = 2


async def _upscale_video(
    video_url: str,
    source_request_id: str | None = None,
) -> str | None:
    """Best-effort SeedVR2 upscale, preserving a successful native result."""
    try:
        arguments: dict[str, Any] = {
            "video_url": video_url,
            "upscale_mode": "factor",
            "upscale_factor": UPSCALER_FACTOR,
        }
        if await _resolve_managed_fal_video_gateway() is not None:
            if not source_request_id:
                raise RuntimeError(
                    "Managed SeedVR upscale requires the source FAL request id"
                )
            arguments["source_request_id"] = source_request_id
        result = await _submit_fal_video_request(UPSCALER_ENDPOINT, arguments)
    except Exception as exc:  # noqa: BLE001 - upscale is best effort
        logger.warning("Video upscale failed: %s", exc)
        return None

    video = result.get("video") if isinstance(result, dict) else None
    if isinstance(video, dict) and video.get("url"):
        return video["url"]
    if isinstance(video, str) and video:
        return video
    logger.warning("Video upscaler returned no URL")
    return None


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class FALVideoGenProvider(VideoGenProvider):
    """FAL.ai multi-family video generation backend.

    Routes between text-to-video and image-to-video endpoints automatically
    based on whether ``image_url`` was provided.
    """

    @property
    def name(self) -> str:
        return "fal"

    @property
    def display_name(self) -> str:
        return "FAL"

    async def is_available(self) -> bool:
        try:
            return await _check_fal_video_available()
        except Exception:  # noqa: BLE001 — never break the picker
            return False

    async def list_models(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for fid, meta in FAL_FAMILIES.items():
            modalities: list[str] = []
            if meta.get("text_endpoint"):
                modalities.append("text")
            if meta.get("image_endpoint"):
                modalities.append("image")
            out.append({
                "id": fid,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": meta["price"],
                "tier": meta.get("tier", "premium"),
                "modalities": modalities,
            })
            durations = meta.get("durations")
            if durations:
                if _is_duration_range(durations):
                    out[-1]["min_duration"], out[-1]["max_duration"] = durations
                else:
                    out[-1]["min_duration"] = min(durations)
                    out[-1]["max_duration"] = max(durations)
        return out

    async def default_model(self) -> str | None:
        return DEFAULT_MODEL

    async def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": "FAL",
            "badge": "paid",
            "tag": "LTX, Pixverse, Seedance 2.0/2.5/Mini, Veo 3.1, MiniMax H3, FLUX 3, Kling 4K, Happy Horse, Grok Imagine, Gemini Omni — text-to-video & image-to-video",
            "env_vars": [
                {
                    "key": "FAL_KEY",
                    "prompt": "FAL.ai API key",
                    "url": "https://fal.ai/dashboard/keys",
                },
            ],
        }

    def capabilities(self) -> dict[str, Any]:
        max_duration = 1
        min_duration: int | None = None
        for meta in FAL_FAMILIES.values():
            durations = meta.get("durations")
            if not durations:
                continue
            if _is_duration_range(durations):
                low, high = durations
            else:
                low, high = min(durations), max(durations)
            max_duration = max(max_duration, high)
            min_duration = low if min_duration is None else min(min_duration, low)
        return {
            "modalities": ["text", "image"],
            "aspect_ratios": ["16:9", "9:16", "1:1"],
            "resolutions": ["360p", "540p", "720p", "1080p"],
            "max_duration": max_duration,
            "min_duration": min_duration if min_duration is not None else 1,
            "supports_audio": True,
            "supports_negative_prompt": True,
            "max_reference_images": 0,
        }

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        image_url: str | None = None,
        reference_image_urls: list[str] | None = None,
        duration: int | None = None,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        negative_prompt: str | None = None,
        audio: bool | None = None,
        seed: int | None = None,
        upscale: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not await _check_fal_video_available():
            return error_response(
                error=(
                    "No FAL backend available. Set FAL_KEY to use FAL.ai."
                ),
                error_type="auth_required",
                provider="fal",
                prompt=prompt,
            )

        try:
            _load_fal_client()
        except ImportError:
            return error_response(
                error="fal_client Python package not installed (pip install fal-client)",
                error_type="missing_dependency",
                provider="fal",
                prompt=prompt,
            )

        prompt = (prompt or "").strip()
        family_id, family = await _resolve_family(model)

        # Route: image_url → image-to-video endpoint; else → text-to-video.
        image_url_norm = (image_url or "").strip() or None
        if image_url_norm:
            endpoint = family.get("image_endpoint")
            modality_used = "image"
            if not endpoint:
                return error_response(
                    error=(
                        f"FAL family {family_id} has no image-to-video "
                        f"endpoint. Pick a family with image-to-video support "
                        f"via `hermes tools` → Video Generation."
                    ),
                    error_type="modality_unsupported",
                    provider="fal", model=family_id, prompt=prompt,
                )
        else:
            endpoint = family.get("text_endpoint")
            modality_used = "text"
            if not endpoint:
                return error_response(
                    error=(
                        f"FAL family {family_id} has no text-to-video "
                        f"endpoint. Pass an image_url to use its "
                        f"image-to-video endpoint, or pick a different family."
                    ),
                    error_type="modality_unsupported",
                    provider="fal", model=family_id, prompt=prompt,
                )

        if not prompt:
            return error_response(
                error="prompt is required.",
                error_type="missing_prompt",
                provider="fal", model=family_id, prompt=prompt,
            )

        payload = _build_payload(
            family,
            prompt=prompt,
            image_url=image_url_norm,
            duration=duration,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            negative_prompt=negative_prompt,
            audio=audio,
            seed=seed,
        )

        try:
            # Preserve the upstream request-id handoff needed by managed
            # SeedVR2 while keeping the ordinary two-argument private helper
            # contract unchanged for callers/tests that do not upscale.
            if upscale:
                submitted = await _submit_fal_video_request(
                    endpoint,
                    payload,
                    return_request_id=True,
                )
                if isinstance(submitted, tuple) and len(submitted) == 2:
                    result, source_request_id = submitted
                else:
                    # Keep monkeypatched/private callers that return the
                    # ordinary result shape compatible with the opt-in path.
                    result, source_request_id = submitted, None
            else:
                result = await _submit_fal_video_request(endpoint, payload)
                source_request_id = None
        except Exception as exc:
            logger.warning(
                "FAL video gen failed (family=%s, endpoint=%s): %s",
                family_id, endpoint, exc, exc_info=True,
            )
            return error_response(
                error=f"FAL video generation failed: {exc}",
                error_type="api_error",
                provider="fal", model=family_id, prompt=prompt,
                aspect_ratio=aspect_ratio,
            )

        video = (result or {}).get("video") if isinstance(result, dict) else None
        url: str | None = None
        if isinstance(video, dict):
            url = video.get("url")
        elif isinstance(video, str):
            url = video

        if not url:
            return error_response(
                error="FAL returned no video URL in response",
                error_type="empty_response",
                provider="fal", model=family_id, prompt=prompt,
            )

        upscaled = False
        if upscale:
            upscaled_url = await _upscale_video(url, source_request_id)
            if upscaled_url:
                url = upscaled_url
                upscaled = True

        extra: dict[str, Any] = {"endpoint": endpoint, "upscaled": upscaled}
        if upscaled:
            extra["upscale_factor"] = UPSCALER_FACTOR
        if isinstance(video, dict):
            if video.get("file_size"):
                extra["file_size"] = video["file_size"]
            if video.get("content_type"):
                extra["content_type"] = video["content_type"]

        return success_response(
            video=url,
            model=family_id,
            prompt=prompt,
            modality=modality_used,
            aspect_ratio=aspect_ratio if "aspect_ratio" in payload else "",
            duration=int("".join(c for c in payload["duration"] if c.isdigit()) or "0") if "duration" in payload else 0,
            provider="fal",
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``FALVideoGenProvider`` into the registry."""
    ctx.register_video_gen_provider(FALVideoGenProvider())
