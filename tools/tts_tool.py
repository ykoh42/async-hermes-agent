#!/usr/bin/env python3
"""
Text-to-Speech Tool Module

Built-in TTS providers:
- Edge TTS (default, free, no API key): Microsoft Edge neural voices
- ElevenLabs (premium): High-quality voices, needs ELEVENLABS_API_KEY
- OpenAI TTS: Good quality, needs OPENAI_API_KEY
- MiniMax TTS: High-quality with voice cloning, needs the selected region's key
- Mistral (Voxtral TTS): Multilingual, native Opus, needs MISTRAL_API_KEY
- Google Gemini TTS: Controllable, 30 prebuilt voices, needs GEMINI_API_KEY
- xAI TTS: Grok voices, uses xAI Grok OAuth credentials or XAI_API_KEY
- NeuTTS (local, free, no API key): On-device TTS via neutts
- KittenTTS (local, free, no API key): On-device 25MB model
- Piper (local, free, no API key): OHF-Voice/piper1-gpl neural VITS, 44 languages

Custom command providers:
- Users can declare any number of named providers with ``type: command``
  under ``tts.providers.<name>`` in ``~/.hermes/config.yaml``. Hermes
  writes the input text to a temp file and runs the configured shell
  command, which must produce the audio file at the expected path.
  See the Local Command section of ``website/docs/user-guide/features/tts.md``.

Output formats:
- Opus (.ogg) for Telegram voice bubbles (requires ffmpeg for Edge TTS)
- MP3 (.mp3) for everything else (CLI, Discord, WhatsApp)

Configuration is loaded from ~/.hermes/config.yaml under the 'tts:' key.
The user chooses the provider and voice; the model just sends text.

Usage:
    from tools.tts_tool import text_to_speech_tool, check_tts_requirements

    result = await text_to_speech_tool(text="Hello world")
"""

import asyncio
import base64
import contextvars
import datetime
import inspect
import json
import logging
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Callable
from urllib.parse import urljoin, urlparse

import aiofiles
import aiofiles.os
import aiofiles.tempfile
from hermes_constants import display_hermes_home

logger = logging.getLogger(__name__)


def __getattr__(name: str):
    """Resolve httpx lazily while preserving the module patch surface."""
    if name == "httpx":
        module = _get_httpx_module()
        globals()["httpx"] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _get_httpx_module():
    import httpx as httpx_module

    return httpx_module


async def get_env_value(name, default=None):
    """Read env values through the live config module.

    Tests may monkeypatch and later restore ``hermes_cli.config.get_env_value``
    before this module is imported. Resolve the helper at call time so TTS does
    not keep a stale imported function for the rest of the test process.
    """
    try:
        from hermes_cli.config import get_env_value_prefer_dotenv
    except ImportError:
        return os.getenv(name, default)
    value = await get_env_value_prefer_dotenv(name)
    return default if value is None else value


async def _resolve_provider_key(env_var: str, provider_id: str) -> str:
    """Resolve a TTS provider API key via the shared voice-key resolver.

    Delegates to ``tools.tool_backend_helpers.resolve_provider_secret`` —
    the single owner of STT/TTS key resolution (config > env/.env > the
    credential pool populated by ``hermes auth add <provider_id>``).
    Resolved at call time so tests that reload the helpers module see the
    live function.
    """
    from tools.tool_backend_helpers import resolve_provider_secret

    return await resolve_provider_secret(
        env_var,
        provider_id,
        env_getter=get_env_value,
    )

from tools.managed_tool_gateway import resolve_managed_tool_gateway
from tools.tool_backend_helpers import (
    managed_nous_tools_enabled,
    nous_tool_gateway_unavailable_message,
    prefers_gateway,
)
from tools.xai_http import hermes_xai_user_agent

# ---------------------------------------------------------------------------
# Lazy imports -- providers are imported only when actually used to avoid
# crashing in headless environments (SSH, Docker, WSL, no PortAudio).
# ---------------------------------------------------------------------------

def _import_edge_tts():
    """Lazy import edge_tts. Returns the module or raises ImportError."""
    import edge_tts
    return edge_tts

def _import_elevenlabs():
    """Lazy import the coroutine-native ElevenLabs client."""
    from elevenlabs.client import AsyncElevenLabs
    return AsyncElevenLabs


def _elevenlabs_environment_kwargs(el_config: dict[str, Any]) -> dict[str, Any]:
    """Build ElevenLabs client kwargs honoring config base_url/wss_url.

    ``tts.elevenlabs.base_url`` (and optionally ``wss_url``) redirect the SDK
    to a self-hosted / proxy endpoint via an ``ElevenLabsEnvironment``. When
    neither is set the SDK default environment is used. ``wss_url`` defaults
    to the ``base_url`` host with a ``wss://`` scheme when omitted.
    """
    base_url = (el_config.get("base_url") or "").rstrip("/")
    if not base_url:
        return {}
    wss_url = (el_config.get("wss_url") or "").rstrip("/")
    if not wss_url:
        wss_url = re.sub(r"^http", "ws", base_url)
    from elevenlabs.environment import ElevenLabsEnvironment
    return {"environment": ElevenLabsEnvironment(base=base_url, wss=wss_url)}

def _import_openai_client():
    """Lazy import OpenAI client. Returns the class or raises ImportError."""
    from openai import AsyncOpenAI as OpenAIClient
    return OpenAIClient

def _import_mistral_client():
    """Lazy import the Mistral client when its async audio API is available."""
    from mistralai.client import Mistral
    return Mistral


def _import_kittentts():
    """Lazy import KittenTTS. Returns the class or raises ImportError."""
    from kittentts import KittenTTS
    return KittenTTS


def _import_piper():
    """Lazy import Piper. Returns the PiperVoice class or raises ImportError.

    Piper is an optional, fully-local neural TTS engine (Home Assistant /
    Open Home Foundation). ``pip install piper-tts`` provides cross-platform
    wheels (Linux / macOS / Windows, x86_64 + ARM64) with embedded espeak-ng.
    Voice models (.onnx + .onnx.json) are downloaded on first use.
    """
    from piper import PiperVoice
    return PiperVoice


# ===========================================================================
# Defaults
# ===========================================================================
DEFAULT_PROVIDER = "edge"
DEFAULT_EDGE_VOICE = "en-US-AriaNeural"
DEFAULT_ELEVENLABS_VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # Adam
DEFAULT_ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_ELEVENLABS_STREAMING_MODEL_ID = "eleven_flash_v2_5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini-tts"
# The managed OpenAI audio gateway (Nous portal proxy) only proxies these speech
# models. A user's tts.openai.model set for *direct* OpenAI (e.g. "tts-1-hd")
# is rejected with a 400 "Unsupported managed OpenAI speech model", so it must be
# coerced to a supported model when routing through the gateway.
MANAGED_OPENAI_TTS_MODELS = frozenset({"gpt-4o-mini-tts"})
DEFAULT_KITTENTTS_MODEL = "KittenML/kitten-tts-nano-0.8-int8"  # 25MB
DEFAULT_KITTENTTS_VOICE = "Jasper"
DEFAULT_PIPER_VOICE = "en_US-lessac-medium"  # balanced size/quality
DEFAULT_OPENAI_VOICE = "alloy"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MINIMAX_MODEL = "speech-02-hd"
DEFAULT_MINIMAX_VOICE_ID = "English_expressive_narrator"
DEFAULT_MINIMAX_BASE_URL = "https://api.minimax.io/v1/t2a_v2"
DEFAULT_MINIMAX_CN_BASE_URL = "https://api.minimaxi.com/v1/t2a_v2"
DEFAULT_MISTRAL_TTS_MODEL = "voxtral-mini-tts-2603"
DEFAULT_MISTRAL_TTS_VOICE_ID = "c69964a6-ab8b-4f8a-9465-ec0925096ec8"  # Paul - Neutral
DEFAULT_XAI_VOICE_ID = "eve"
DEFAULT_XAI_LANGUAGE = "en"
DEFAULT_XAI_SAMPLE_RATE = 24000
DEFAULT_XAI_BIT_RATE = 128000
DEFAULT_XAI_AUTO_SPEECH_TAGS = False
DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
# xAI TTS `speed` accepts 0.7..1.5; 1.0 is the API default (omitted => default).
DEFAULT_XAI_SPEED_MIN = 0.7
DEFAULT_XAI_SPEED_MAX = 1.5
DEFAULT_XAI_SPEED_DEFAULT = 1.0
# xAI TTS `optimize_streaming_latency` accepts 0, 1, or 2; 0 (best quality) is
# the API default (omitted => default). Values >0 trade quality for time-to-first-audio.
DEFAULT_XAI_OPTIMIZE_STREAMING_LATENCY_DEFAULT = 0
# xAI TTS `text_normalization` is a boolean (default False). When enabled,
# the model normalizes written-form text (numbers, abbreviations, symbols)
# into spoken-form before generating audio.
DEFAULT_XAI_TEXT_NORMALIZATION_DEFAULT = False
DEFAULT_GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"
DEFAULT_GEMINI_TTS_VOICE = "Kore"
DEFAULT_GEMINI_TTS_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GEMINI_AUDIO_TAGS = False
GEMINI_AUDIO_TAG_REWRITE_TASK = "tts_audio_tags"
# Base URL now resolved via hermes_cli.models.deepinfra_base_url (shared).
DEFAULT_DEEPINFRA_TTS_VOICE = "default"
# PCM output specs for Gemini TTS (fixed by the API)
GEMINI_TTS_SAMPLE_RATE = 24000
GEMINI_TTS_CHANNELS = 1
GEMINI_TTS_SAMPLE_WIDTH = 2  # 16-bit PCM (L16)
TTS_RESPONSE_BODY_LIMIT_BYTES = 16 * 1024 * 1024
TTS_RESPONSE_BODY_CHUNK_BYTES = 64 * 1024

async def _get_default_output_dir() -> str:
    from hermes_constants import get_hermes_dir
    return str(await get_hermes_dir("cache/audio", "audio_cache"))


# Kept as an override point for callers/tests that used the upstream constant.
# The default is resolved lazily because legacy-directory discovery performs I/O.
DEFAULT_OUTPUT_DIR: str | None = None

# ---------------------------------------------------------------------------
# Per-provider input-character limits (from official provider docs).
# A single global cap was wrong: OpenAI is 4096, xAI is 15k, MiniMax is 10k,
# ElevenLabs is model-dependent (5k / 10k / 30k / 40k), Gemini has a 32k-token
# context window.  Users can override any of these via
# ``tts.<provider>.max_text_length`` in config.yaml.
# ---------------------------------------------------------------------------
PROVIDER_MAX_TEXT_LENGTH: dict[str, int] = {
    "edge": 5000,         # edge-tts practical sync limit
    "openai": 4096,       # https://platform.openai.com/docs/guides/text-to-speech
    "xai": 15000,         # https://docs.x.ai/developers/model-capabilities/audio/text-to-speech
    "minimax": 10000,     # https://platform.minimax.io/docs/api-reference/speech-t2a-http (sync)
    "mistral": 4000,      # conservative; no published per-request cap
    "gemini": 32000,      # Gemini TTS has a 32k-token context window; char cap is conservative
    "elevenlabs": 10000,  # fallback when model-aware lookup can't resolve (multilingual_v2)
    "neutts": 2000,       # local model, quality falls off on long text
    "kittentts": 2000,    # local 25MB model
    "piper": 5000,        # local VITS model, phoneme-based; practical cap
}

# ElevenLabs caps vary by model_id. https://elevenlabs.io/docs/overview/models
ELEVENLABS_MODEL_MAX_TEXT_LENGTH: dict[str, int] = {
    "eleven_v3": 5000,
    "eleven_ttv_v3": 5000,
    "eleven_multilingual_v2": 10000,
    "eleven_multilingual_v1": 10000,
    "eleven_english_sts_v2": 10000,
    "eleven_english_sts_v1": 10000,
    "eleven_flash_v2": 30000,
    "eleven_flash_v2_5": 40000,
}


def _config_bool(value: Any, default: bool = False) -> bool:
    """Coerce common YAML/env bool spellings without treating random strings as true."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


def _response_has_explicit_stream(response: Any) -> bool:
    return callable(getattr(response, "aiter_bytes", None))


async def _close_response(response: Any) -> None:
    close = getattr(response, "aclose", None)
    if callable(close):
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except Exception:
            pass


async def _read_tts_response_bytes(
    response: Any,
    *,
    label: str,
    limit: int | None = None,
) -> bytes:
    """Read an upstream TTS response with a hard byte cap."""
    limit = TTS_RESPONSE_BODY_LIMIT_BYTES if limit is None else limit
    chunks: list[bytes] = []
    total = 0
    try:
        if _response_has_explicit_stream(response):
            async for chunk in response.aiter_bytes(
                TTS_RESPONSE_BODY_CHUNK_BYTES
            ):
                if not chunk:
                    continue
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8", errors="replace")
                chunk = bytes(chunk)
                total += len(chunk)
                if total > limit:
                    await _close_response(response)
                    raise RuntimeError(f"{label} response exceeds {limit} bytes")
                chunks.append(chunk)
        else:
            content = getattr(response, "content", b"")
            if isinstance(content, str):
                content = content.encode("utf-8", errors="replace")
            if isinstance(content, (bytes, bytearray)) and content:
                total = len(content)
                if total > limit:
                    await _close_response(response)
                    raise RuntimeError(f"{label} response exceeds {limit} bytes")
                chunks.append(bytes(content))
        return b"".join(chunks)
    finally:
        await _close_response(response)


async def _read_tts_response_json(
    response: Any,
    *,
    label: str,
    limit: int | None = None,
) -> dict[str, Any]:
    raw = await _read_tts_response_bytes(response, label=label, limit=limit)
    if raw:
        return json.loads(raw.decode("utf-8"))

    # Unit-test doubles often only provide `.json()`. Real requests.Response
    # objects use the streaming path above, so this fallback does not re-open
    # the production eager-buffering behavior.
    if not _response_has_explicit_stream(response):
        json_reader = getattr(response, "json", None)
        if callable(json_reader):
            parsed = json_reader()
            return parsed if isinstance(parsed, dict) else {}
    return {}


async def _write_tts_response_to_file(
    response: Any,
    output_path: str,
    *,
    label: str,
    limit: int | None = None,
) -> None:
    audio_bytes = await _read_tts_response_bytes(
        response,
        label=label,
        limit=limit,
    )
    async with aiofiles.open(output_path, "wb") as output_file:
        await output_file.write(audio_bytes)

# Final fallback when provider isn't recognised at all.
FALLBACK_MAX_TEXT_LENGTH = 4000

# Back-compat alias. Prefer ``_resolve_max_text_length()`` for new code.
MAX_TEXT_LENGTH = FALLBACK_MAX_TEXT_LENGTH


def _resolve_max_text_length(
    provider: str | None,
    tts_config: dict[str, Any] | None = None,
) -> int:
    """Return the input-character cap for *provider*.

    Resolution order:
      1. ``tts.<provider>.max_text_length`` (user override in config.yaml)
      2. ``tts.providers.<provider>.max_text_length`` for user-declared
         command providers
      3. ElevenLabs model-aware table (keyed on configured ``model_id``)
      4. ``PROVIDER_MAX_TEXT_LENGTH`` default
      5. ``DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH`` when the provider is a
         command-type user provider without an explicit cap
      6. ``FALLBACK_MAX_TEXT_LENGTH`` (4000)

    Non-positive or non-integer overrides fall through to the default so a
    broken config can't accidentally disable truncation entirely.
    """
    if not provider:
        return FALLBACK_MAX_TEXT_LENGTH
    key = provider.lower().strip()
    cfg = tts_config or {}

    # Built-in-style override at tts.<provider>.max_text_length wins first,
    # matching historical behavior.
    prov_cfg = cfg.get(key) if isinstance(cfg.get(key), dict) else {}
    override = prov_cfg.get("max_text_length") if prov_cfg else None
    if isinstance(override, bool):
        override = None
    if isinstance(override, int) and override > 0:
        return override

    if key == "elevenlabs":
        model_id = (prov_cfg or {}).get("model_id") or DEFAULT_ELEVENLABS_MODEL_ID
        mapped = ELEVENLABS_MODEL_MAX_TEXT_LENGTH.get(str(model_id).strip())
        if mapped:
            return mapped

    if key in PROVIDER_MAX_TEXT_LENGTH:
        return PROVIDER_MAX_TEXT_LENGTH[key]

    # User-declared command provider (under tts.providers.<name>)
    if key not in BUILTIN_TTS_PROVIDERS:
        named = _get_named_provider_config(cfg, key)
        if _is_command_provider_config(named):
            named_override = named.get("max_text_length")
            if isinstance(named_override, bool):
                named_override = None
            if isinstance(named_override, int) and named_override > 0:
                return named_override
            return DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH

    return FALLBACK_MAX_TEXT_LENGTH


# ===========================================================================
# Long-form chunking and native async delivery packing
# ===========================================================================

@dataclass(frozen=True)
class AudioDeliveryProfile:
    """Destination-platform constraints for generated TTS audio."""

    platform: str
    max_file_bytes: int
    safety_ratio: float = 0.85

    @property
    def target_file_bytes(self) -> int:
        return max(1, int(self.max_file_bytes * self.safety_ratio))


_PLATFORM_AUDIO_DEFAULTS = {
    "discord": {"max_file_bytes": 10 * 1024 * 1024, "safety_ratio": 0.85},
    "telegram": {"max_file_bytes": 50 * 1024 * 1024, "safety_ratio": 0.85},
    "default": {"max_file_bytes": 10 * 1024 * 1024, "safety_ratio": 0.85},
}


def _resolve_audio_delivery_profile(
    platform: str | None,
    tts_config: dict[str, Any] | None = None,
) -> AudioDeliveryProfile:
    """Resolve upload limits, including validated per-platform overrides."""
    key = (platform or "default").lower().strip() or "default"
    defaults = dict(_PLATFORM_AUDIO_DEFAULTS.get(key, _PLATFORM_AUDIO_DEFAULTS["default"]))
    profiles = (tts_config or {}).get("delivery_profiles")
    overrides = profiles.get(key, {}) if isinstance(profiles, dict) else {}
    if isinstance(overrides, dict):
        defaults.update({k: v for k, v in overrides.items() if v is not None})
    max_bytes = defaults.get("max_file_bytes")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        max_bytes = _PLATFORM_AUDIO_DEFAULTS["default"]["max_file_bytes"]
    ratio = defaults.get("safety_ratio", 0.85)
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0 < ratio <= 1:
        ratio = 0.85
    return AudioDeliveryProfile(key, max_bytes, float(ratio))


def _split_oversized_sentence(sentence: str, max_chars: int) -> list[str]:
    """Split one over-limit sentence on word boundaries, then hard boundaries."""
    words = sentence.split()
    chunks: list[str] = []
    current = ""
    for word in words:
        if len(word) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(word[i:i + max_chars] for i in range(0, len(word), max_chars))
            continue
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _split_text_for_tts(text: str, max_chars: int) -> list[str]:
    """Split normalized text under a provider cap without dropping content."""
    if max_chars <= 0:
        max_chars = FALLBACK_MAX_TEXT_LENGTH
    normalized = " ".join((text or "").split())
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?;:,])\s+", normalized)
        if sentence.strip()
    ]
    expanded: list[str] = []
    for sentence in sentences:
        expanded.extend(
            [sentence]
            if len(sentence) <= max_chars
            else _split_oversized_sentence(sentence, max_chars)
        )
    chunks: list[str] = []
    current = ""
    for sentence in expanded:
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


async def _pack_audio_files_for_delivery(
    audio_paths: list[str],
    profile: AudioDeliveryProfile,
) -> list[list[str]]:
    """Group final-encoded chunks under the conservative size target."""
    groups: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    current_suffix = ""
    for path in audio_paths:
        size = (await aiofiles.os.stat(path)).st_size
        suffix = Path(path).suffix.lower()
        if current and (
            current_size + size > profile.target_file_bytes
            or suffix != current_suffix
        ):
            groups.append(current)
            current, current_size, current_suffix = [], 0, ""
        current.append(path)
        current_size += size
        current_suffix = suffix
    if current:
        groups.append(current)
    return groups


async def _concat_audio_files(
    audio_paths: list[str],
    output_path: str,
    *,
    voice_compatible: bool = False,
) -> str | None:
    """Combine encoded chunks with native asyncio subprocess/file I/O."""
    if not audio_paths:
        raise ValueError("No audio chunks to combine")
    if len(audio_paths) == 1:
        if os.path.abspath(audio_paths[0]) != os.path.abspath(output_path):
            async with aiofiles.open(audio_paths[0], "rb") as source:
                data = await source.read()
            async with aiofiles.open(output_path, "wb") as destination:
                await destination.write(data)
        return output_path
    if not await _has_ffmpeg():
        return None
    ffmpeg = "ffmpeg"
    destination = Path(output_path)
    await aiofiles.os.makedirs(destination.parent, exist_ok=True)
    concat_path = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.concat.txt")
    temp_output = destination.with_name(
        f".{destination.stem}.{uuid.uuid4().hex}.combining{destination.suffix}"
    )
    try:
        async with aiofiles.open(concat_path, "w", encoding="utf-8") as handle:
            for path in audio_paths:
                await handle.write(f"file {shlex.quote(os.path.abspath(path))}\n")
        command = [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat_path), "-vn"]
        suffix = destination.suffix.lower()
        if voice_compatible or suffix in {".ogg", ".opus"}:
            command.extend(["-c:a", "libopus", "-ac", "1", "-b:a", "64k", "-vbr", "off"])
        elif suffix == ".mp3" and all(Path(path).suffix.lower() == ".mp3" for path in audio_paths):
            command.extend(["-c:a", "copy"])
        command.append(str(temp_output))
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=await _non_model_tts_subprocess_env(),
        )
        try:
            _stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
        except asyncio.CancelledError:
            await _finish_owned_tts_cleanup(_kill_and_reap_tts_process(process))
            raise
        if process.returncode == 0 and await aiofiles.os.path.exists(temp_output):
            if (await aiofiles.os.stat(temp_output)).st_size > 0:
                await aiofiles.os.replace(temp_output, destination)
                return str(destination)
        logger.warning("ffmpeg audio combine failed: %s", (stderr or b"").decode(errors="ignore")[:500])
    except (OSError, TimeoutError) as exc:
        logger.warning("ffmpeg audio combine failed: %s", exc)
    finally:
        for path in (concat_path, temp_output):
            try:
                await aiofiles.os.remove(path)
            except OSError:
                pass
    return None


async def _build_audio_delivery_files(
    audio_paths: list[str],
    output_path: str,
    profile: AudioDeliveryProfile,
    *,
    voice_compatible: bool = False,
) -> tuple[list[str], bool]:
    """Pack final-encoded chunks and enforce the hard upload limit."""
    if not audio_paths:
        raise ValueError("No final-encoded TTS audio chunks")
    for path in audio_paths:
        size = (await aiofiles.os.stat(path)).st_size
        if size > profile.max_file_bytes:
            raise ValueError(
                f"Final-encoded TTS chunk exceeds {profile.platform} delivery "
                f"limit ({size} > {profile.max_file_bytes} bytes): {path}"
            )
    base = Path(output_path)
    scratch_outputs: list[str] = []
    combined_any = False

    async def emit(group: list[str], index: int) -> list[str]:
        nonlocal combined_any
        if len(group) == 1:
            return list(group)
        scratch = base.with_name(
            f".{base.stem}.delivery{index:03d}.{uuid.uuid4().hex}{base.suffix}"
        )
        combined = await _concat_audio_files(group, str(scratch), voice_compatible=voice_compatible)
        if not combined:
            return list(group)
        scratch_outputs.append(combined)
        if (await aiofiles.os.stat(combined)).st_size <= profile.max_file_bytes:
            combined_any = True
            return [combined]
        try:
            await aiofiles.os.remove(combined)
        except OSError:
            pass
        midpoint = max(1, len(group) // 2)
        return await emit(group[:midpoint], index + 1) + await emit(group[midpoint:], index + 1)

    packed: list[str] = []
    for index, group in enumerate(await _pack_audio_files_for_delivery(audio_paths, profile), start=1):
        packed.extend(await emit(group, index))
    final_paths: list[str] = []
    try:
        for index, source in enumerate(packed, start=1):
            suffix = Path(source).suffix or base.suffix
            destination = base if len(packed) == 1 else base.with_name(f"{base.stem}.part{index:02d}{suffix}")
            if os.path.abspath(source) != os.path.abspath(destination):
                await aiofiles.os.makedirs(destination.parent, exist_ok=True)
                await aiofiles.os.replace(source, destination)
            if (await aiofiles.os.stat(destination)).st_size > profile.max_file_bytes:
                raise ValueError(f"Final TTS deliverable exceeds {profile.platform} delivery limit: {destination}")
            final_paths.append(str(destination))
        return final_paths, combined_any
    finally:
        final_set = {os.path.abspath(path) for path in final_paths}
        for scratch in scratch_outputs:
            if os.path.abspath(scratch) not in final_set:
                try:
                    await aiofiles.os.remove(scratch)
                except OSError:
                    pass


# ===========================================================================
# Config loader -- reads tts: section from ~/.hermes/config.yaml
# ===========================================================================
async def _load_tts_config() -> dict[str, Any]:
    """
    Load TTS configuration from ~/.hermes/config.yaml.

    Returns a dict with provider settings. Falls back to defaults
    for any missing fields.
    """
    try:
        from hermes_cli.config import load_config_readonly
        config = await load_config_readonly()
        return config.get("tts") or {}
    except ImportError:
        logger.debug("hermes_cli.config not available, using default TTS config")
        return {}
    except Exception as e:
        logger.warning("Failed to load TTS config: %s", e, exc_info=True)
        return {}


def _get_provider(tts_config: dict[str, Any]) -> str:
    """Get the explicitly configured TTS provider or the free default.

    Inference credentials do not imply consent to paid speech generation.
    Users opt into cloud TTS by setting ``tts.provider`` (normally through
    ``hermes tools``); otherwise the historical Edge backend remains active.
    """
    return (tts_config.get("provider") or DEFAULT_PROVIDER).lower().strip()


@dataclass(frozen=True)
class _MiniMaxTTSRuntime:
    """A region-bound MiniMax endpoint and credential.

    The credential is excluded from ``repr`` so diagnostics cannot expose it
    accidentally.
    """

    region: str
    endpoint: str
    credential_source: str
    api_key: str = field(repr=False)


async def _resolve_minimax_tts_runtime(
    tts_config: dict[str, Any],
) -> _MiniMaxTTSRuntime:
    """Select MiniMax TTS region, endpoint, and credential atomically.

    An explicit ``tts.minimax.region`` wins. Without one, the legacy global
    credential wins when present; a China credential is selected only when it
    is the sole configured MiniMax credential.
    """
    mm_config = tts_config.get("minimax", {})
    if not isinstance(mm_config, dict):
        mm_config = {}

    credentials = {
        "global": (
            "MINIMAX_API_KEY",
            str(
                await _resolve_provider_key("MINIMAX_API_KEY", "minimax")
                or ""
            ).strip(),
        ),
        "cn": (
            "MINIMAX_CN_API_KEY",
            str(
                await _resolve_provider_key("MINIMAX_CN_API_KEY", "minimax")
                or ""
            ).strip(),
        ),
    }
    endpoints = {
        "global": DEFAULT_MINIMAX_BASE_URL,
        "cn": DEFAULT_MINIMAX_CN_BASE_URL,
    }

    configured_region = str(mm_config.get("region") or "").strip().lower()
    if configured_region and configured_region not in endpoints:
        raise ValueError("tts.minimax.region must be 'global' or 'cn'")

    if configured_region:
        region = configured_region
    elif credentials["global"][1]:
        region = "global"
    elif credentials["cn"][1]:
        region = "cn"
    else:
        region = "global"

    credential_source, api_key = credentials[region]
    if not api_key:
        raise ValueError(
            f"{credential_source} not set for MiniMax TTS region {region!r}"
        )

    endpoint = str(mm_config.get("base_url") or endpoints[region]).strip()
    endpoint_host = (urlparse(endpoint).hostname or "").lower()
    official_region_hosts = {
        "global": frozenset({"api.minimax.io", "api.minimax.chat"}),
        "cn": frozenset({"api.minimaxi.com"}),
    }
    other_region = "cn" if region == "global" else "global"
    if endpoint_host in official_region_hosts[other_region]:
        raise ValueError(
            f"tts.minimax.base_url points to the {other_region!r} MiniMax endpoint "
            f"but region is {region!r}"
        )

    return _MiniMaxTTSRuntime(
        region=region,
        endpoint=endpoint,
        credential_source=credential_source,
        api_key=api_key,
    )


# ===========================================================================
# Custom command providers (type: command under tts.providers.<name>)
# ===========================================================================
#
# Users can declare any number of command-type providers alongside the
# built-ins so they can plug any local CLI (Piper, VoxCPM, Kokoro CLIs,
# custom voice-cloning scripts, etc.) into Hermes without any Python code
# changes. The config shape is::
#
#     tts:
#       provider: piper-en
#       providers:
#         piper-en:
#           type: command
#           command: "piper -m ~/model.onnx -f {output_path} < {input_path}"
#           output_format: wav
#
# Hermes writes the input text to a temp UTF-8 file, runs the command with
# placeholder substitution, and reads the audio file the command wrote to
# ``{output_path}``. Supported placeholders: ``{input_path}``,
# ``{text_path}`` (alias for input_path), ``{output_path}``, ``{format}``,
# ``{voice}``, ``{model}``, ``{speed}``. Use ``{{`` / ``}}`` for literal braces.
#
# Built-in provider names always win over an entry with the same name under
# ``tts.providers``, so user config can't silently shadow ``edge`` etc.
#
# Placeholder values are shell-quoted for their surrounding context
# (bare / single / double quote), so paths with spaces work transparently.

# Built-in provider names. Any ``tts.provider`` value NOT in this set is
# interpreted as a reference to ``tts.providers.<name>``.
BUILTIN_TTS_PROVIDERS = frozenset({
    "edge",
    "elevenlabs",
    "openai",
    "minimax",
    "xai",
    "mistral",
    "gemini",
    "neutts",
    "kittentts",
    "piper",
    "deepinfra",
})

DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS = 120
DEFAULT_COMMAND_TTS_OUTPUT_FORMAT = "mp3"
COMMAND_TTS_OUTPUT_FORMATS = frozenset(
    {"mp3", "wav", "ogg", "flac", "m4a", "aac", "amr", "opus"}
)
DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH = 5000

# Platforms whose native voice-bubble delivery requires Ogg/Opus audio.
# Previously only Telegram was recognized, so Matrix/Feishu/WhatsApp/Signal
# voice replies were synthesized as MP3 and rendered as broken attachments
# (#14841, #45557 and siblings).
OPUS_VOICE_PLATFORMS = frozenset({
    "telegram",
    "matrix",
    "feishu",
    "whatsapp",
    "signal",
})


def _get_provider_section(tts_config: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a provider config block if it's a dict, else an empty dict."""
    if not isinstance(tts_config, dict):
        return {}
    section = tts_config.get(name)
    return section if isinstance(section, dict) else {}


def _get_named_provider_config(
    tts_config: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    """Return the config dict for a user-declared provider.

    Looks up ``tts.providers.<name>`` first (the canonical location), and
    falls back to ``tts.<name>`` so users who followed the built-in layout
    still work. Returns an empty dict when the provider is not declared.
    """
    providers = _get_provider_section(tts_config, "providers")
    section = providers.get(name) if isinstance(providers, dict) else None
    if isinstance(section, dict):
        return section
    # Back-compat: allow ``tts.<name>`` for user-declared providers too,
    # but only when the name is not a built-in (so a user's ``tts.openai``
    # block still means the OpenAI provider, not a custom command).
    if name.lower() not in BUILTIN_TTS_PROVIDERS:
        legacy = _get_provider_section(tts_config, name)
        if legacy:
            return legacy
    return {}


def _is_command_provider_config(config: dict[str, Any]) -> bool:
    """Return True when *config* declares a command-type provider."""
    if not isinstance(config, dict):
        return False
    ptype = str(config.get("type") or "").strip().lower()
    if ptype and ptype != "command":
        return False
    command = config.get("command")
    return isinstance(command, str) and bool(command.strip())


def _resolve_command_provider_config(
    provider: str,
    tts_config: dict[str, Any],
) -> dict[str, Any] | None:
    """Return the provider config if *provider* resolves to a command type.

    Built-in provider names are rejected (they have native handlers).
    Returns None when the name is a built-in, unknown, or not a command
    type.
    """
    if not provider:
        return None
    key = provider.lower().strip()
    if key in BUILTIN_TTS_PROVIDERS:
        return None
    config = _get_named_provider_config(tts_config, key)
    if _is_command_provider_config(config):
        return config
    return None


async def _dispatch_to_plugin_provider(
    text: str,
    output_path: str,
    provider: str,
    tts_config: dict[str, Any],
) -> str | None:
    """Route the call to a plugin-registered TTS provider, or return None.

    Returns the path to the written audio file on dispatch, or ``None``
    to fall through to the next resolution layer (built-in dispatch or
    Edge TTS default).

    Resolution invariants enforced here (matches issue #30398):

    1. Built-in provider names short-circuit — never reach the plugin
       registry. The caller is responsible for the elif chain that
       handles ``edge``/``openai``/etc.; this function explicitly
       rejects those names defensively.
    2. Command-type providers declared under
       ``tts.providers.<name>: type: command`` (PR #17843) win over a
       plugin with the same name. The caller passes us only when its
       own command-provider check returned None — we re-verify here so
       a refactor of the caller can't silently break the invariant.
    3. Plugin dispatch fires only when ``provider`` matches a registered
       :class:`TTSProvider` whose ``name`` equals the configured value.
       Unknown names return None (caller falls through to Edge default).

    Plugin exceptions are caught and re-raised — the outer
    ``text_to_speech_tool`` try/except converts them to the standard
    error envelope, matching how command-provider failures surface.
    """
    if not provider:
        return None
    key = provider.lower().strip()
    if key in BUILTIN_TTS_PROVIDERS:
        return None
    # Defense in depth: command-provider check should already have
    # short-circuited the caller. If a same-name command config exists,
    # bail so the command path wins.
    if _is_command_provider_config(_get_named_provider_config(tts_config, key)):
        return None
    try:
        from agent.tts_registry import get_provider
        from hermes_cli.plugins import _ensure_plugins_discovered

        await _ensure_plugins_discovered()
        plugin_provider = get_provider(key)
        if plugin_provider is None:
            # Long-lived sessions may have discovered plugins before the
            # bundled backend was patched in or before config changed.
            # Retry once with a forced refresh before surfacing fall-
            # through. Mirrors the image_gen / browser dispatcher
            # recovery pattern.
            await _ensure_plugins_discovered(force=True)
            plugin_provider = get_provider(key)
    except Exception as exc:  # noqa: BLE001 — discovery failure is non-fatal
        logger.debug("tts plugin dispatch skipped (discovery failed): %s", exc)
        return None
    if plugin_provider is None:
        return None

    # Resolve voice / model / format from tts_config — providers should
    # treat all of these as optional and fall back to their own defaults
    # when None is passed (matches the ABC contract documented on
    # ``TTSProvider.synthesize``).
    voice = tts_config.get("voice") if isinstance(tts_config, dict) else None
    model = tts_config.get("model") if isinstance(tts_config, dict) else None
    speed = tts_config.get("speed") if isinstance(tts_config, dict) else None
    fmt = (
        tts_config.get("output_format", DEFAULT_COMMAND_TTS_OUTPUT_FORMAT)
        if isinstance(tts_config, dict)
        else DEFAULT_COMMAND_TTS_OUTPUT_FORMAT
    )

    logger.info(
        "Generating speech with plugin TTS provider '%s'...", key,
    )
    written = await plugin_provider.synthesize(
        text,
        output_path,
        voice=voice if isinstance(voice, str) and voice else None,
        model=model if isinstance(model, str) and model else None,
        speed=float(speed) if isinstance(speed, (int, float)) else None,
        format=str(fmt).lower() if fmt else "mp3",
    )
    # Provider contract: returns the (possibly rewritten) output path.
    # Defensive against a provider returning None or a non-string —
    # fall back to the caller's expected output_path.
    return written if isinstance(written, str) and written else output_path


def _plugin_provider_is_voice_compatible(provider: str) -> bool:
    """Return True when the registered plugin provider opts into voice
    bubble delivery via its ``voice_compatible`` property.

    Defensive: any registry or property access failure means False
    (matches the safe default for the command-provider path).
    """
    if not provider:
        return False
    key = provider.lower().strip()
    if key in BUILTIN_TTS_PROVIDERS:
        return False
    try:
        from agent.tts_registry import get_provider

        plugin_provider = get_provider(key)
        if plugin_provider is None:
            return False
        return bool(plugin_provider.voice_compatible)
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "tts plugin voice_compatible check failed for '%s': %s", key, exc,
        )
        return False


def _iter_command_providers(tts_config: dict[str, Any]):
    """Yield (name, config) pairs for every declared command-type provider."""
    if not isinstance(tts_config, dict):
        return
    providers = _get_provider_section(tts_config, "providers")
    for name, cfg in (providers or {}).items():
        if isinstance(name, str) and name.lower() not in BUILTIN_TTS_PROVIDERS:
            if _is_command_provider_config(cfg):
                yield name, cfg


def _get_command_tts_timeout(config: dict[str, Any]) -> float:
    """Return timeout in seconds, falling back when invalid."""
    raw = config.get("timeout", config.get("timeout_seconds", DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS)
    if value <= 0:
        return float(DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS)
    return value


def _get_command_tts_output_format(
    config: dict[str, Any],
    output_path: str | None = None,
) -> str:
    """Return the validated output format (mp3/wav/ogg/flac)."""
    if output_path:
        suffix = Path(output_path).suffix.lower().strip().lstrip(".")
        if suffix in COMMAND_TTS_OUTPUT_FORMATS:
            return suffix
    raw = (
        config.get("format")
        or config.get("output_format")
        or DEFAULT_COMMAND_TTS_OUTPUT_FORMAT
    )
    fmt = str(raw).lower().strip().lstrip(".")
    return fmt if fmt in COMMAND_TTS_OUTPUT_FORMATS else DEFAULT_COMMAND_TTS_OUTPUT_FORMAT


def _is_command_tts_voice_compatible(config: dict[str, Any]) -> bool:
    """Return True only when the user explicitly opted in to voice delivery."""
    value = config.get("voice_compatible", False)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _shell_quote_context(command_template: str, position: int) -> str | None:
    """Return the shell quote character active right before *position*.

    Returns ``"'"`` / ``'"'`` when inside a single- / double-quoted region
    of the template, ``None`` for bare context.
    """
    quote: str | None = None
    escaped = False
    i = 0
    while i < position:
        char = command_template[i]
        if quote == "'":
            if char == "'":
                quote = None
        elif quote == '"':
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = None
        elif char == "'":
            quote = "'"
        elif char == '"':
            quote = '"'
        elif char == "\\":
            i += 1
        i += 1
    return quote


def _quote_command_tts_placeholder(value: str, quote_context: str | None) -> str:
    """Quote a placeholder value for its position in a shell command template."""
    if quote_context == "'":
        return value.replace("'", r"'\''")
    if quote_context == '"':
        return (
            value
            .replace("\\", "\\\\")
            .replace('"', r'\"')
            .replace("$", r"\$")
            .replace("`", r"\`")
        )
    if os.name == "nt":
        return subprocess.list2cmdline([value])
    return shlex.quote(value)


def _render_command_tts_template(
    command_template: str,
    placeholders: dict[str, str],
) -> str:
    """Replace supported placeholders while preserving ``{{`` / ``}}``."""
    names = "|".join(re.escape(name) for name in placeholders)
    pattern = re.compile(
        rf"(?<!\$)(?:\{{\{{(?P<double>{names})\}}\}}|\{{(?P<single>{names})\}})"
    )
    replacements: list[tuple[str, str]] = []

    def replace_match(match: re.Match[str]) -> str:
        name = match.group("double") or match.group("single")
        token = f"__HERMES_TTS_PLACEHOLDER_{len(replacements)}__"
        replacements.append((
            token,
            _quote_command_tts_placeholder(
                placeholders[name],
                _shell_quote_context(command_template, match.start()),
            ),
        ))
        return token

    rendered = pattern.sub(replace_match, command_template)
    rendered = rendered.replace("{{", "{").replace("}}", "}")
    for token, value in replacements:
        rendered = rendered.replace(token, value)
    return rendered


async def _terminate_command_tts_process_tree(
    proc: asyncio.subprocess.Process,
) -> None:
    """Best-effort termination of a shell process and all of its children."""
    if proc.returncode is not None:
        return

    if os.name == "nt":
        try:
            killer = await asyncio.create_subprocess_exec(
                "taskkill", "/F", "/T", "/PID", str(proc.pid),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                stdin=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(killer.wait(), timeout=5)
        except Exception:
            proc.kill()
            await proc.wait()
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except (PermissionError, OSError):
        proc.terminate()

    try:
        await asyncio.wait_for(proc.wait(), timeout=2)
        return
    except TimeoutError:
        pass

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except (PermissionError, OSError):
        proc.kill()
    await proc.wait()


async def _finish_owned_tts_cleanup(cleanup) -> None:
    """Finish owned resource cleanup before propagating cancellation."""
    cleanup_task = asyncio.create_task(cleanup)
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            await asyncio.shield(cleanup_task)
            break
        except asyncio.CancelledError as exc:  # noqa: ASYNC103
            if cleanup_task.cancelled():
                raise
            if cancellation is None:
                cancellation = exc
        except Exception as exc:
            if cancellation is not None:
                raise cancellation from exc
            raise
    if cancellation is not None:
        raise cancellation


async def _kill_and_reap_tts_process(
    process: asyncio.subprocess.Process,
) -> None:
    if process.returncode is None:
        process.kill()
    await process.wait()


def _command_provider_env_passthrough(config: dict[str, Any]) -> list:
    """Return the provider's ``env_passthrough`` allowlist (opt-out of scrub).

    Command providers legitimately reference their own API keys in the shell
    template (curl one-liners). The child env is scrubbed of Hermes secrets by
    default; ``env_passthrough: [MY_API_KEY, ...]`` copies the named variables
    from the active profile secret scope (or the legacy process environment in
    single-profile mode) so a trusted template keeps working.
    """
    raw = config.get("env_passthrough")
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


async def _run_command_tts(
    command: str,
    timeout: float,
    env_passthrough: list | None = None,
) -> subprocess.CompletedProcess:
    """Run a command-provider shell command with process-tree idle cleanup.

    Child env is scrubbed of Hermes secrets (salvage of #56332) while still
    propagating delegated-child lineage markers when applicable.
    """
    from agent.delegation_context import delegated_child_subprocess_env
    from agent.secret_scope import get_secret
    from tools.environments.local import hermes_subprocess_env

    scrubbed = await hermes_subprocess_env(inherit_credentials=False)
    for key in env_passthrough or []:
        value = get_secret(key)
        if value is not None:
            scrubbed[key] = value
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.DEVNULL,
        env=delegated_child_subprocess_env(scrubbed),
        start_new_session=os.name != "nt",
        creationflags=(
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            if os.name == "nt"
            else 0
        ),
    )
    assert proc.stdout is not None
    assert proc.stderr is not None
    output_queue: asyncio.Queue[tuple[str, str | None]] = asyncio.Queue()
    chunks: dict[str, list[str]] = {"stdout": [], "stderr": []}
    open_streams = {"stdout", "stderr"}

    async def read_stream(name: str, stream: asyncio.StreamReader) -> None:
        try:
            while True:
                data = await stream.read(65536)
                if not data:
                    break
                await output_queue.put(
                    (name, data.decode("utf-8", errors="replace"))
                )
        finally:
            await output_queue.put((name, None))

    readers = [
        asyncio.create_task(
            read_stream("stdout", proc.stdout),
            name="tts-command-stdout",
        ),
        asyncio.create_task(
            read_stream("stderr", proc.stderr),
            name="tts-command-stderr",
        ),
    ]

    deadline = time.monotonic() + timeout
    timed_out = False
    try:
        while open_streams:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                name, chunk = await asyncio.wait_for(
                    output_queue.get(),
                    timeout=remaining,
                )
            except TimeoutError:
                timed_out = True
                break
            if chunk is None:
                open_streams.discard(name)
                continue
            chunks[name].append(chunk)
            deadline = time.monotonic() + timeout

        if not timed_out:
            try:
                await asyncio.wait_for(
                    proc.wait(),
                    timeout=max(0.0, deadline - time.monotonic()),
                )
            except TimeoutError:
                timed_out = True

        if timed_out:
            await _terminate_command_tts_process_tree(proc)
            await asyncio.gather(*readers, return_exceptions=True)
            while not output_queue.empty():
                name, chunk = output_queue.get_nowait()
                if chunk:
                    chunks[name].append(chunk)
            stdout = "".join(chunks["stdout"])
            stderr = "".join(chunks["stderr"])
            raise subprocess.TimeoutExpired(
                command,
                timeout,
                output=stdout,
                stderr=stderr,
            )
    finally:
        async def cleanup_command() -> None:
            try:
                if proc.returncode is None:
                    await _terminate_command_tts_process_tree(proc)
            finally:
                for reader in readers:
                    if not reader.done():
                        reader.cancel()
                await asyncio.gather(*readers, return_exceptions=True)

        await _finish_owned_tts_cleanup(cleanup_command())

    stdout = "".join(chunks["stdout"])
    stderr = "".join(chunks["stderr"])

    if proc.returncode:
        raise subprocess.CalledProcessError(
            proc.returncode,
            command,
            output=stdout,
            stderr=stderr,
        )
    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)


def _configured_command_tts_output_path(path: Path, config: dict[str, Any]) -> Path:
    """Return an output path whose extension matches the provider's output_format."""
    fmt = _get_command_tts_output_format(config)
    return path.with_suffix(f".{fmt}")


async def _generate_command_tts(
    text: str,
    output_path: str,
    provider_name: str,
    config: dict[str, Any],
    tts_config: dict[str, Any],
) -> str:
    """Generate speech by running a user-configured shell command.

    Returns the absolute path of the audio file the command wrote.
    Raises ``ValueError`` when the provider config is invalid, and
    ``RuntimeError`` for timeouts / non-zero exits / empty output.
    """
    command_template = str(config.get("command") or "").strip()
    if not command_template:
        raise ValueError(
            f"tts.providers.{provider_name}.command is not configured"
        )

    output = Path(output_path).expanduser()
    await aiofiles.os.makedirs(output.parent, exist_ok=True)
    if await aiofiles.os.path.exists(output):
        await aiofiles.os.remove(output)

    timeout = _get_command_tts_timeout(config)
    output_format = _get_command_tts_output_format(config, str(output))
    speed = config.get("speed", tts_config.get("speed", ""))

    async with aiofiles.tempfile.TemporaryDirectory() as tmpdir:
        text_path = Path(tmpdir) / "input.txt"
        async with aiofiles.open(text_path, "w", encoding="utf-8") as input_file:
            await input_file.write(text)

        placeholders = {
            "input_path": str(text_path),
            "text_path": str(text_path),
            "output_path": str(output),
            "format": output_format,
            "voice": str(config.get("voice", "")),
            "model": str(config.get("model", "")),
            "speed": str(speed),
        }
        command = _render_command_tts_template(command_template, placeholders)

        try:
            await _run_command_tts(
                command,
                timeout,
                env_passthrough=_command_provider_env_passthrough(config),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"TTS provider '{provider_name}' timed out after {timeout:g}s"
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail_parts = []
            if exc.stderr:
                detail_parts.append(f"stderr: {exc.stderr.strip()}")
            if exc.stdout:
                detail_parts.append(f"stdout: {exc.stdout.strip()}")
            detail = "; ".join(detail_parts) or "no command output"
            raise RuntimeError(
                f"TTS provider '{provider_name}' exited with code "
                f"{exc.returncode}: {detail}"
            ) from exc

    if (
        not await aiofiles.os.path.exists(output)
        or (await aiofiles.os.stat(output)).st_size <= 0
    ):
        raise RuntimeError(
            f"TTS provider '{provider_name}' produced no output at {output}"
        )
    return str(output)


async def _has_any_command_tts_provider(
    tts_config: dict[str, Any] | None = None,
) -> bool:
    """Return True when any command-type TTS provider is configured."""
    if tts_config is None:
        tts_config = await _load_tts_config()
    for _name, _cfg in _iter_command_providers(tts_config):
        return True
    return False


async def _non_model_tts_subprocess_env(
    *,
    include_hf_token: bool = False,
) -> dict[str, str]:
    """Return a credential-scrubbed environment for trusted TTS helpers.

    Local module probes, ffmpeg, and the public Piper voice downloader never
    need model-provider credentials. NeuTTS is the sole exception: its model
    repository can be private or gated, so only the active profile's
    Hugging Face token names are restored for that child.
    """
    from agent.secret_scope import get_secret
    from tools.environments.local import hermes_subprocess_env

    env = await hermes_subprocess_env(inherit_credentials=False)
    hf_token_names = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
    for name in hf_token_names:
        env.pop(name, None)
    if include_hf_token:
        for name in hf_token_names:
            value = get_secret(name)
            if value is not None:
                env[name] = value
    return env


# ===========================================================================
# ffmpeg Opus conversion (Edge TTS MP3 -> OGG Opus for Telegram)
# ===========================================================================
async def _has_ffmpeg() -> bool:
    """Check if ffmpeg is available on the system."""
    try:
        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-version",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            stdin=asyncio.subprocess.DEVNULL,
            env=await _non_model_tts_subprocess_env(),
        )
        await asyncio.wait_for(process.wait(), timeout=5)
        return process.returncode == 0
    except asyncio.CancelledError:
        if "process" in locals():
            await _finish_owned_tts_cleanup(
                _kill_and_reap_tts_process(process)
            )
        raise
    except TimeoutError:
        await _finish_owned_tts_cleanup(_kill_and_reap_tts_process(process))
        return False
    except FileNotFoundError:
        return False


async def _convert_to_opus(mp3_path: str) -> str | None:
    """
    Convert an audio file (MP3/WAV/anything ffmpeg reads) to OGG Opus
    format for Telegram voice bubbles.

    Args:
        mp3_path: Path to the input audio file.

    Returns:
        Path to the .ogg file, or None if conversion fails.
    """
    if not await _has_ffmpeg():
        return None

    ogg_path = mp3_path.rsplit(".", 1)[0] + ".ogg"
    return await _ffmpeg_transcode_to_opus(mp3_path, ogg_path)


async def _ffmpeg_transcode_to_opus(
    input_path: str,
    ogg_path: str,
) -> str | None:
    """Transcode *input_path* to real Ogg/Opus at *ogg_path* via ffmpeg.

    Safe when ``input_path == ogg_path`` (writes to a temp file, then
    replaces). Returns the output path on success, None on failure.
    """
    if not await _has_ffmpeg():
        return None

    in_place = os.path.abspath(input_path) == os.path.abspath(ogg_path)
    work_path = ogg_path + ".tmp.ogg" if in_place else ogg_path
    try:
        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", input_path, "-acodec", "libopus",
            "-ac", "1", "-b:a", "48k", "-vbr", "on",
            "-application", "voip", "-compression_level", "10", "-f", "ogg",
            work_path, "-y",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
            env=await _non_model_tts_subprocess_env(),
        )
        try:
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
        except TimeoutError:
            await _finish_owned_tts_cleanup(
                _kill_and_reap_tts_process(process)
            )
            logger.warning("ffmpeg OGG conversion timed out after 30s")
            return None
        if process.returncode != 0:
            logger.warning("ffmpeg conversion failed with return code %d: %s",
                          process.returncode, stderr.decode('utf-8', errors='ignore')[:200])
            return None
        if (
            await aiofiles.os.path.exists(work_path)
            and (await aiofiles.os.stat(work_path)).st_size > 0
        ):
            if in_place:
                await aiofiles.os.replace(work_path, ogg_path)
            return ogg_path
    except asyncio.CancelledError:
        if "process" in locals() and process.returncode is None:
            await _finish_owned_tts_cleanup(
                _kill_and_reap_tts_process(process)
            )
        raise
    except FileNotFoundError:
        logger.warning("ffmpeg not found in PATH")
    except Exception as e:
        logger.warning("ffmpeg OGG conversion failed: %s", e, exc_info=True)
    finally:
        if in_place and await aiofiles.os.path.exists(work_path):
            try:
                await aiofiles.os.remove(work_path)
            except OSError:
                pass
    return None


# ---------------------------------------------------------------------------
# Container sniffing — class-level guard against "MP3/WAV bytes in a .ogg
# file". Several TTS backends silently ignore the requested opus format
# (Edge only emits MP3, Piper writes WAV, xAI writes MP3, some
# OpenAI-compatible servers reject/ignore response_format="opus"), which
# breaks native voice bubbles on Telegram/Matrix/Feishu/WhatsApp. Rather
# than special-casing every provider, sniff the magic bytes once after
# synthesis and repair the container when it doesn't match the extension.
# ---------------------------------------------------------------------------

async def _sniff_audio_container(path: str) -> str:
    """Return a container id ('ogg', 'wav', 'mp3', 'flac', ...) or 'unknown'.

    Delegates to the shared magic-byte sniffer in ``tools.audio_container``
    (one module owns container detection for both this outbound repair and
    the inbound gateway audio cache).
    """
    from tools.audio_container import sniff_container

    try:
        async with aiofiles.open(path, "rb") as audio_file:
            head = await audio_file.read(12)
    except OSError:
        return "unknown"
    return sniff_container(head) or "unknown"


async def _repair_ogg_container(file_str: str) -> str:
    """Ensure a path claiming ``.ogg`` actually contains an Ogg container.

    When the bytes are MP3/WAV/FLAC (a backend ignored the opus request),
    transcode in place to real Ogg/Opus. On any failure, rename to the
    sniffed real extension so downstream players/platforms at least get an
    honest file instead of a 0-second voice bubble. Returns the (possibly
    updated) path.
    """
    if not file_str.endswith(".ogg"):
        return file_str
    container = await _sniff_audio_container(file_str)
    if container in ("ogg", "unknown"):
        return file_str

    logger.info(
        "TTS wrote %s bytes into a .ogg path (%s) — transcoding to real Ogg/Opus",
        container, file_str,
    )
    repaired = await _ffmpeg_transcode_to_opus(file_str, file_str)
    if repaired:
        return repaired

    # ffmpeg unavailable/failed: rename to the honest extension.
    honest = file_str[:-4] + "." + container
    try:
        await aiofiles.os.replace(file_str, honest)
        logger.warning(
            "Could not transcode %s to Ogg/Opus — renamed to %s so the "
            "file is delivered with its real format", file_str, honest,
        )
        return honest
    except OSError:
        return file_str


# ===========================================================================
# Provider: Edge TTS (free)
# ===========================================================================
async def _generate_edge_tts(text: str, output_path: str, tts_config: dict[str, Any]) -> str:
    """
    Generate audio using Edge TTS.

    Args:
        text: Text to convert.
        output_path: Where to save the MP3 file.
        tts_config: TTS config dict.

    Returns:
        Path to the saved audio file.
    """
    _edge_tts = _import_edge_tts()
    edge_config = tts_config.get("edge") or {}
    voice = edge_config.get("voice", DEFAULT_EDGE_VOICE)
    speed = float(edge_config.get("speed", tts_config.get("speed", 1.0)))

    kwargs = {"voice": voice}
    if speed != 1.0:
        pct = round((speed - 1.0) * 100)
        kwargs["rate"] = f"{pct:+d}%"

    communicate = _edge_tts.Communicate(text, **kwargs)
    await communicate.save(output_path)
    return output_path


# ===========================================================================
# Provider: ElevenLabs (premium)
# ===========================================================================
async def _generate_elevenlabs(
    text: str,
    output_path: str,
    tts_config: dict[str, Any],
) -> str:
    """
    Generate audio using ElevenLabs.

    Args:
        text: Text to convert.
        output_path: Where to save the audio file.
        tts_config: TTS config dict.

    Returns:
        Path to the saved audio file.
    """
    api_key = (
        await _resolve_provider_key("ELEVENLABS_API_KEY", "elevenlabs")
        or ""
    )
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY not set. Get one at https://elevenlabs.io/")

    el_config = tts_config.get("elevenlabs") or {}
    voice_id = el_config.get("voice_id", DEFAULT_ELEVENLABS_VOICE_ID)
    model_id = el_config.get("model_id", DEFAULT_ELEVENLABS_MODEL_ID)

    # Determine output format based on file extension
    if output_path.endswith(".ogg"):
        output_format = "opus_48000_64"
    else:
        output_format = "mp3_44100_128"

    ElevenLabs = _import_elevenlabs()
    client = ElevenLabs(api_key=api_key, **_elevenlabs_environment_kwargs(el_config))
    try:
        audio_generator = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
        )
        if inspect.isawaitable(audio_generator):
            audio_generator = await audio_generator
        if not hasattr(audio_generator, "__aiter__"):
            raise RuntimeError(
                "ElevenLabs SDK did not provide a native async audio stream"
            )
        async with aiofiles.open(output_path, "wb") as output_file:
            async for chunk in audio_generator:
                await output_file.write(chunk)
    finally:
        close = getattr(client, "aclose", None)
        if callable(close):
            await close()

    return output_path


def _tts_response_format_from_path(output_path: str) -> str:
    """Pick an OpenAI-compatible TTS response format from the output extension."""
    if output_path.endswith(".ogg"):
        return "opus"
    if output_path.endswith(".wav"):
        return "wav"
    if output_path.endswith(".flac"):
        return "flac"
    return "mp3"


# ===========================================================================
# Provider: OpenAI TTS (also used by every OpenAI-compatible TTS endpoint —
# DeepInfra delegates here via _generate_deepinfra_tts).
# ===========================================================================
async def _generate_openai_tts(
    text: str,
    output_path: str,
    tts_config: dict[str, Any],
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    voice: str | None = None,
    speed: float | None = None,
    instructions: str | None = None,
) -> str:
    """Generate audio via the OpenAI ``audio.speech.create`` SDK shape.

    Optional kwargs let OpenAI-compatible backends (DeepInfra etc.) reuse
    this function — they resolve credentials/model themselves and pass
    them through, skipping the OpenAI-only ``_resolve_openai_audio_client_config``.

    Args:
        text: Text to convert.
        output_path: Where to save the audio file.
        tts_config: TTS config dict (used for ``tts.openai`` sub-block
            and the global ``speed`` default).
        api_key: Bearer token. When None, resolved from the OpenAI auth
            chain (config → env → managed gateway).
        base_url: API base URL. When None, falls back to
            ``tts.openai.base_url`` then the OpenAI default.
        model: Model id. When None, reads ``tts.openai.model``.
        voice: Voice id. When None, reads ``tts.openai.voice``.
        speed: Playback speed. When None, reads ``tts.openai.speed`` /
            ``tts.speed``.
        instructions: Optional voice-design guidance (tone, emotion, pacing,
            accent, whispering). Forwarded to `audio.speech.create` when
            truthy; omitted otherwise so ``tts-1``/``tts-1-hd`` and strict
            OpenAI-compatible servers that reject unknown kwargs are
            unaffected.

    Returns:
        Path to the saved audio file.
    """
    # Only resolve the OpenAI auth chain when the caller didn't pass explicit
    # credentials. OpenAI-compatible backends (DeepInfra) pass api_key /
    # base_url / model / voice through and never hit the managed-gateway path.
    fallback_base: str | None = None
    is_managed = False
    explicit_base_url = base_url is not None
    if api_key is None:
        api_key, fallback_base, is_managed = (
            await _resolve_openai_audio_client_config()
        )

    # ``tts.openai: null`` in YAML yields None — coalesce so .get() is safe.
    oai_config = (tts_config.get("openai") if isinstance(tts_config, dict) else None) or {}
    if model is None:
        model = oai_config.get("model", DEFAULT_OPENAI_MODEL)
    if voice is None:
        voice = oai_config.get("voice", DEFAULT_OPENAI_VOICE)
    config_base_url = oai_config.get("base_url")
    if base_url is None:
        # Config override wins over the auth-chain fallback (restores the
        # pre-refactor precedence, where tts.openai.base_url beat the resolved
        # default); the auth-chain value is the last-resort default. An
        # explicit base_url arg from an OpenAI-compatible caller (DeepInfra)
        # skips this block entirely and always wins.
        base_url = config_base_url or fallback_base or DEFAULT_OPENAI_BASE_URL
    if speed is None:
        speed_default = tts_config.get("speed", 1.0) if isinstance(tts_config, dict) else 1.0
        speed = float(oai_config.get("speed", speed_default))
    language = oai_config.get("language")

    # The managed OpenAI audio gateway only proxies MANAGED_OPENAI_TTS_MODELS.
    # A model set for direct OpenAI (e.g. "tts-1-hd") 400s there with
    # "Unsupported managed OpenAI speech model", so coerce it — unless the user
    # redirected base_url to their own endpoint, in which case respect it.
    if (
        is_managed
        and not explicit_base_url
        and not config_base_url
        and model not in MANAGED_OPENAI_TTS_MODELS
    ):
        logger.warning(
            "TTS: managed OpenAI audio gateway does not support model %r; "
            "falling back to %s. Set VOICE_TOOLS_OPENAI_KEY or OPENAI_API_KEY "
            "to use %r directly.",
            model, DEFAULT_OPENAI_MODEL, model,
        )
        model = DEFAULT_OPENAI_MODEL

    response_format = _tts_response_format_from_path(output_path)

    OpenAIClient = _import_openai_client()
    client = OpenAIClient(api_key=api_key, base_url=base_url)
    try:
        create_kwargs: dict[str, Any] = {
            "model": model,
            "voice": voice,
            "input": text,
            "response_format": response_format,
            "extra_headers": {"x-idempotency-key": str(uuid.uuid4())},
        }
        if speed != 1.0:
            create_kwargs["speed"] = max(0.25, min(4.0, speed))
        if instructions:
            create_kwargs["instructions"] = instructions
        if language:
            create_kwargs["extra_body"] = {"lang_code": language}
        async with client.audio.speech.with_streaming_response.create(
            **create_kwargs
        ) as response:
            await response.stream_to_file(output_path)
        return output_path
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            await close()


# ===========================================================================
# Provider: DeepInfra TTS
# ===========================================================================
#
# DeepInfra serves TTS over an OpenAI-compatible /v1/openai/audio/speech
# endpoint. Models are discovered live via the shared catalog helper
# (filtered by the ``tts`` surface tag) — no hardcoded model ids in this
# file, so retired models disappear from hermes the next time the
# catalog is fetched without a patch.


async def _generate_deepinfra_tts(
    text: str,
    output_path: str,
    tts_config: dict[str, Any],
) -> str:
    """Resolve DeepInfra credentials/model, then delegate to the OpenAI handler.

    DeepInfra's audio endpoint is OpenAI-compatible, so there's no need
    to duplicate the SDK call — we just pass an explicit api_key /
    base_url / model / voice through. Model ids and the base URL come from
    the shared ``hermes_cli.models`` helpers so every DeepInfra surface
    resolves them identically.
    """
    api_key = await _resolve_provider_key("DEEPINFRA_API_KEY", "deepinfra")
    if not api_key:
        raise ValueError(
            "DEEPINFRA_API_KEY not set. Run `hermes setup` to configure, "
            "or set the env var directly."
        )

    # ``tts.deepinfra: null`` in YAML yields None, not {} — coalesce so the
    # ``.get`` calls below don't raise AttributeError (there is no
    # tts.deepinfra block in DEFAULT_CONFIG to deep-merge over the null).
    di_config = tts_config.get("deepinfra") if isinstance(tts_config, dict) else None
    if not isinstance(di_config, dict):
        di_config = {}

    from hermes_cli.models import (
        _fetch_deepinfra_models_by_tag,
        deepinfra_base_url,
    )

    model = di_config.get("model")
    if not isinstance(model, str) or not model.strip():
        entries = await _fetch_deepinfra_models_by_tag("tts")
        candidates = [entry["id"] for entry in entries or []]
        if not candidates:
            raise ValueError(
                "No DeepInfra TTS model available. Pin one in config.yaml "
                "under tts.deepinfra.model, or check connectivity to "
                "api.deepinfra.com so the live catalog can be fetched."
            )
        model = candidates[0]
    return await _generate_openai_tts(
        text,
        output_path,
        tts_config,
        api_key=api_key,
        base_url=deepinfra_base_url(di_config),
        model=model,
        voice=di_config.get("voice", DEFAULT_DEEPINFRA_TTS_VOICE),
        speed=float(di_config.get("speed", tts_config.get("speed", 1.0))),
    )


# ===========================================================================
# Provider: xAI TTS
# ===========================================================================
_XAI_INLINE_SPEECH_TAGS = (
    "pause",
    "long-pause",
    "hum-tune",
    "laugh",
    "chuckle",
    "giggle",
    "cry",
    "tsk",
    "tongue-click",
    "lip-smack",
    "breath",
    "inhale",
    "exhale",
    "sigh",
)
_XAI_WRAPPING_SPEECH_TAGS = (
    "soft",
    "whisper",
    "loud",
    "build-intensity",
    "decrease-intensity",
    "higher-pitch",
    "lower-pitch",
    "slow",
    "fast",
    "sing-song",
    "singing",
    "laugh-speak",
    "emphasis",
)
_XAI_SPEECH_TAG_RE = re.compile(
    r"(\[(?:" + "|".join(_XAI_INLINE_SPEECH_TAGS) + r")\]|</?(?:" + "|".join(_XAI_WRAPPING_SPEECH_TAGS) + r")>)",
    flags=re.IGNORECASE,
)
_XAI_FIRST_SENTENCE_RE = re.compile(r"^(.{12,120}?[.!?…])\s+(?=\S)", flags=re.DOTALL)


def _xai_bool_config(value: Any, default: bool = False) -> bool:
    return _config_bool(value, default=default)


async def _apply_xai_auto_speech_tags(text: str) -> str:
    """Add xAI speech tags for more natural voice-mode replies.

    First applies a conservative local transform (inserts [pause] between
    paragraphs and after the first sentence). Then, if the result contains
    no explicit user/model speech tags, asks the configured auxiliary model
    to rewrite the transcript with a richer set of xAI-supported tags
    (laughs, sighs, whispers, soft/loud, slow/fast, etc.) so the voice
    output sounds more expressive. Falls back to the local result on any
    auxiliary-model failure.
    """
    clean = text.strip()
    if not clean:
        return text

    # Local conservative pass: pauses only.
    local = clean
    local = re.sub(r"\n\s*\n+", " [pause] ", local)
    local = re.sub(r"\s*\n\s*", " ", local)
    if not _XAI_SPEECH_TAG_RE.search(local):
        local = _XAI_FIRST_SENTENCE_RE.sub(r"\1 [pause] ", local, count=1)
    local = re.sub(r"\s{2,}", " ", local).strip()

    # If the user/model already supplied explicit speech tags, trust them
    # and don't re-rewrite.
    if _XAI_SPEECH_TAG_RE.search(clean):
        return local

    # Auxiliary rewrite for richer emotion tags (mirrors the Gemini path).
    inline = ", ".join(_XAI_INLINE_SPEECH_TAGS)
    wrapping = ", ".join(_XAI_WRAPPING_SPEECH_TAGS)
    system_prompt = (
        "You rewrite transcripts for the xAI /v1/tts endpoint by inserting "
        "expressive speech tags.\n\n"
        "Valid inline tags (use as `[tag]`): " + inline + ".\n"
        "Valid wrapping tags (use as `[tag]...[/tag]`): " + wrapping + ".\n\n"
        "Rules:\n"
        "- Preserve the spoken words, order, and meaning.\n"
        "- Do not add new spoken sentences or remove existing spoken words.\n"
        "- Use inline `[tag]` for short modifiers (laughs, sighs, pause, etc.).\n"
        "- Use wrapping `[tag]...[/tag]` for sustained effects (whisper, soft, slow, fast, loud, etc.).\n"
        "- Do not use angle-bracket tags like `<tag>...</tag>` — xAI uses BBCode-style closing tags with `[/tag]`.\n"
        "- Do not use SSML.\n"
        "- Do not explain or comment.\n"
        "- Return only the tagged TTS script."
    )
    try:
        from agent.auxiliary_client import call_llm

        response = await call_llm(
            task="tts_audio_tags",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"TRANSCRIPT TO TAG:\n{local}"},
            ],
            temperature=0.7,
        )
        tagged = _extract_auxiliary_message_content(response).strip()
        # Strip markdown fences if the LLM wrapped the response.
        fence = re.fullmatch(r"```(?:[A-Za-z0-9_-]+)?\s*(.*?)\s*```", tagged, flags=re.DOTALL)
        if fence:
            tagged = fence.group(1).strip()
        return tagged or local
    except Exception as exc:
        logger.debug("xAI TTS audio tag rewrite failed; using locally-tagged text: %s", exc)
        return local


async def _generate_xai_tts(
    text: str,
    output_path: str,
    tts_config: dict[str, Any],
) -> str:
    """
    Generate audio using xAI TTS.

    xAI exposes a dedicated /v1/tts endpoint instead of the OpenAI audio.speech
    API shape, so this is implemented as a separate backend.
    """
    httpx = _get_httpx_module()
    from tools.xai_http import resolve_xai_http_credentials

    creds = await resolve_xai_http_credentials()
    api_key = str(creds.get("api_key") or "").strip()
    if not api_key:
        raise ValueError("No xAI credentials found. Configure xAI OAuth in `hermes model` or set XAI_API_KEY.")

    xai_config = tts_config.get("xai") or {}
    voice_id = str(xai_config.get("voice_id", DEFAULT_XAI_VOICE_ID)).strip() or DEFAULT_XAI_VOICE_ID
    language = str(xai_config.get("language", DEFAULT_XAI_LANGUAGE)).strip() or DEFAULT_XAI_LANGUAGE
    sample_rate = int(xai_config.get("sample_rate", DEFAULT_XAI_SAMPLE_RATE))
    bit_rate = int(xai_config.get("bit_rate", DEFAULT_XAI_BIT_RATE))
    auto_speech_tags = _xai_bool_config(
        xai_config.get("auto_speech_tags", xai_config.get("speech_tags")),
        DEFAULT_XAI_AUTO_SPEECH_TAGS,
    )
    # ``tts.xai.speed`` overrides global ``tts.speed``; the xAI TTS API
    # accepts 0.7..1.5 (1.0 = normal). Out-of-range values are clamped so a
    # misconfigured agent can't 400 the request — the API would reject
    # anything outside the band.
    speed = xai_config.get("speed", tts_config.get("speed"))
    if speed is not None and speed != "":
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            speed = None
    if speed is not None:
        speed = max(DEFAULT_XAI_SPEED_MIN, min(DEFAULT_XAI_SPEED_MAX, speed))
    # ``tts.xai.optimize_streaming_latency`` is 0, 1, or 2 (xAI-specific;
    # trades chunk-boundary quality for time-to-first-audio).
    optimize_streaming_latency = xai_config.get(
        "optimize_streaming_latency",
        tts_config.get("optimize_streaming_latency"),
    )
    if optimize_streaming_latency is not None and optimize_streaming_latency != "":
        try:
            optimize_streaming_latency = int(optimize_streaming_latency)
        except (TypeError, ValueError):
            optimize_streaming_latency = None
    if optimize_streaming_latency is not None:
        optimize_streaming_latency = max(0, min(2, optimize_streaming_latency))
    # ``tts.xai.text_normalization`` enables spoken-form normalization
    # (numbers, abbreviations, symbols → words). Defaults to False.
    text_normalization = _xai_bool_config(
        xai_config.get("text_normalization"),
        DEFAULT_XAI_TEXT_NORMALIZATION_DEFAULT,
    )
    if auto_speech_tags:
        text = await _apply_xai_auto_speech_tags(text)
    if creds.get("provider") == "xai-oauth":
        base_url = str(creds.get("base_url") or DEFAULT_XAI_BASE_URL).strip().rstrip("/")
    else:
        base_url = str(
            xai_config.get("base_url")
            or creds.get("base_url")
            or await get_env_value("XAI_BASE_URL")
            or DEFAULT_XAI_BASE_URL
        ).strip().rstrip("/")

    # Match the documented minimal POST /v1/tts shape by default. Only send
    # output_format when Hermes actually needs a non-default format/override.
    codec = "wav" if output_path.endswith(".wav") else "mp3"
    payload: dict[str, Any] = {
        "text": text,
        "voice_id": voice_id,
        "language": language,
    }
    if (
        codec != "mp3"
        or sample_rate != DEFAULT_XAI_SAMPLE_RATE
        or (codec == "mp3" and bit_rate != DEFAULT_XAI_BIT_RATE)
    ):
        output_format: dict[str, Any] = {"codec": codec}
        if sample_rate:
            output_format["sample_rate"] = sample_rate
        if codec == "mp3" and bit_rate:
            output_format["bit_rate"] = bit_rate
        payload["output_format"] = output_format
    # Only attach `speed` when the caller asked for something other than the
    # API default (1.0). Keeps the existing minimal-payload contract for
    # users who never touch the knob.
    if speed is not None and speed != DEFAULT_XAI_SPEED_DEFAULT:
        payload["speed"] = speed
    # Only attach `optimize_streaming_latency` when the caller explicitly
    # opts in to a non-default value (anything other than 0).
    if (
        optimize_streaming_latency is not None
        and optimize_streaming_latency != DEFAULT_XAI_OPTIMIZE_STREAMING_LATENCY_DEFAULT
    ):
        payload["optimize_streaming_latency"] = optimize_streaming_latency
    # Only attach `text_normalization` when explicitly enabled (default is False).
    if text_normalization:
        payload["text_normalization"] = True

    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream(
            "POST",
            f"{base_url}/tts",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": hermes_xai_user_agent(),
            },
            json=payload,
        ) as response:
            response.raise_for_status()
            await _write_tts_response_to_file(
                response,
                output_path,
                label="xAI TTS",
            )

    return output_path


# ===========================================================================
# Provider: MiniMax TTS
# ===========================================================================
async def _generate_minimax_tts(
    text: str,
    output_path: str,
    tts_config: dict[str, Any],
) -> str:
    """
    Generate audio using MiniMax TTS API.

    Supports two endpoints:
    - v1/text_to_speech: simple payload, returns raw audio (Content-Type: audio/mpeg)
    - v1/t2a_v2: nested voice_setting/audio_setting, returns JSON with hex-encoded audio

    Args:
        text: Text to convert (max 10,000 characters).
        output_path: Where to save the audio file.
        tts_config: TTS config dict.

    Returns:
        Path to the saved audio file.
    """
    httpx = _get_httpx_module()
    runtime = await _resolve_minimax_tts_runtime(tts_config)

    mm_config = tts_config.get("minimax", {})
    if not isinstance(mm_config, dict):
        mm_config = {}
    model = mm_config.get("model", DEFAULT_MINIMAX_MODEL)
    voice_id = mm_config.get("voice_id", DEFAULT_MINIMAX_VOICE_ID)
    base_url = runtime.endpoint
    speed = mm_config.get("speed", 1.0)
    vol = mm_config.get("vol", 1.0)
    pitch = mm_config.get("pitch", 0)
    emotion = mm_config.get("emotion", "neutral")
    sample_rate = mm_config.get("sample_rate", 32000)
    bitrate = mm_config.get("bitrate", 128000)

    # MiniMax accounts scope TTS requests by GroupId.  When present, the docs
    # show it as a ?GroupId=<id> query param on the t2a_v2 URL.  Accept it
    # from config or from the MINIMAX_GROUP_ID env var; only attach when the
    # URL doesn't already carry one.
    group_id = (
        str(mm_config.get("group_id") or "").strip()
        or (await get_env_value("MINIMAX_GROUP_ID") or "").strip()
    )
    if group_id and "GroupId=" not in base_url:
        sep = "&" if "?" in base_url else "?"
        base_url = f"{base_url}{sep}GroupId={group_id}"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {runtime.api_key}",
    }

    # Detect endpoint from URL
    is_t2a_v2 = "t2a_v2" in base_url

    if is_t2a_v2:
        # t2a_v2 endpoint: nested voice_setting/audio_setting structure
        payload = {
            "model": model,
            "text": text,
            "voice_setting": {
                "voice_id": voice_id,
                "speed": speed,
                "vol": vol,
                "pitch": pitch,
                "emotion": emotion,
            },
            "audio_setting": {
                "sample_rate": sample_rate,
                "bitrate": bitrate,
                "format": "mp3",
                "channel": 1,
            },
        }
    else:
        # text_to_speech endpoint: flat payload
        payload = {
            "model": model,
            "text": text,
            "voice_id": voice_id,
        }

    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream(
            "POST",
            base_url,
            json=payload,
            headers=headers,
        ) as response:
            if is_t2a_v2:
                response.raise_for_status()
                result = await _read_tts_response_json(
                    response,
                    label="MiniMax TTS",
                )
                base_resp = result.get("base_resp", {})
                status_code = base_resp.get("status_code", -1)
                if status_code != 0:
                    status_msg = base_resp.get("status_msg", "unknown error")
                    raise RuntimeError(
                        f"MiniMax TTS API error (code {status_code}): {status_msg}"
                    )
                hex_audio = result.get("data", {}).get("audio", "")
                if not hex_audio:
                    raise RuntimeError("MiniMax TTS returned empty audio data")
                async with aiofiles.open(output_path, "wb") as output_file:
                    await output_file.write(bytes.fromhex(hex_audio))
                return output_path

            content_type = response.headers.get("Content-Type", "")
            if "audio/" in content_type:
                await _write_tts_response_to_file(
                    response,
                    output_path,
                    label="MiniMax TTS",
                )
                return output_path

            raw_body = await _read_tts_response_bytes(
                response,
                label="MiniMax TTS",
            )
            try:
                result = json.loads(raw_body.decode("utf-8")) if raw_body else {}
                base_resp = result.get("base_resp", {})
                status_code = base_resp.get("status_code", -1)
                if status_code != 0:
                    status_msg = base_resp.get("status_msg", "unknown error")
                    raise RuntimeError(
                        f"MiniMax TTS API error (code {status_code}): {status_msg}"
                    )
            except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
                response.raise_for_status()
                raise RuntimeError(
                    f"MiniMax TTS returned unexpected Content-Type '{content_type}' "
                    f"({len(raw_body)} bytes)"
                )
            raise RuntimeError("MiniMax TTS returned no audio data")


# ===========================================================================
# Provider: Mistral (Voxtral TTS)
# ===========================================================================
async def _generate_mistral_tts(
    text: str,
    output_path: str,
    tts_config: dict[str, Any],
) -> str:
    """Generate audio using Mistral Voxtral TTS API.

    The API returns base64-encoded audio; this function decodes it
    and writes the raw bytes to *output_path*.
    Supports native Opus output for Telegram voice bubbles.
    """
    api_key = await _resolve_provider_key("MISTRAL_API_KEY", "mistral") or ""
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not set. Get one at https://console.mistral.ai/")

    mi_config = tts_config.get("mistral") or {}
    model = mi_config.get("model", DEFAULT_MISTRAL_TTS_MODEL)
    voice_id = mi_config.get("voice_id") or DEFAULT_MISTRAL_TTS_VOICE_ID
    # Class-level base_url parity: every cloud TTS provider section supports
    # base_url. The Mistral SDK calls it server_url.
    base_url = mi_config.get("base_url")

    if output_path.endswith(".ogg"):
        response_format = "opus"
    elif output_path.endswith(".wav"):
        response_format = "wav"
    elif output_path.endswith(".flac"):
        response_format = "flac"
    else:
        response_format = "mp3"

    Mistral = _import_mistral_client()
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["server_url"] = base_url
    client = Mistral(**client_kwargs)
    try:
        complete_async = getattr(client.audio.speech, "complete_async", None)
        if not callable(complete_async):
            raise RuntimeError(
                "Mistral SDK does not expose native async TTS synthesis"
            )
        response = await complete_async(
            model=model,
            input=text,
            voice_id=voice_id,
            response_format=response_format,
        )
        audio_bytes = base64.b64decode(response.audio_data)
    except ValueError:
        raise
    except Exception as e:
        logger.error("Mistral TTS failed: %s", e, exc_info=True)
        raise RuntimeError(f"Mistral TTS failed: {type(e).__name__}") from e

    finally:
        close = getattr(client, "close_async", None) or getattr(
            client,
            "aclose",
            None,
        )
        if callable(close):
            await close()

    async with aiofiles.open(output_path, "wb") as output_file:
        await output_file.write(audio_bytes)

    return output_path


# ===========================================================================
# Provider: Google Gemini TTS
# ===========================================================================
def _wrap_pcm_as_wav(
    pcm_bytes: bytes,
    sample_rate: int = GEMINI_TTS_SAMPLE_RATE,
    channels: int = GEMINI_TTS_CHANNELS,
    sample_width: int = GEMINI_TTS_SAMPLE_WIDTH,
) -> bytes:
    """Wrap raw signed-little-endian PCM with a standard WAV RIFF header.

    Gemini TTS returns audio/L16;codec=pcm;rate=24000 -- raw PCM samples with
    no container. We add a minimal WAV header so the file is playable and
    ffmpeg can re-encode it to MP3/Opus downstream.
    """
    import struct

    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    data_size = len(pcm_bytes)
    fmt_chunk = struct.pack(
        "<4sIHHIIHH",
        b"fmt ",
        16,             # fmt chunk size (PCM)
        1,              # audio format (PCM)
        channels,
        sample_rate,
        byte_rate,
        block_align,
        sample_width * 8,
    )
    data_chunk_header = struct.pack("<4sI", b"data", data_size)
    riff_size = 4 + len(fmt_chunk) + len(data_chunk_header) + data_size
    riff_header = struct.pack("<4sI4s", b"RIFF", riff_size, b"WAVE")
    return riff_header + fmt_chunk + data_chunk_header + pcm_bytes


async def _resolve_gemini_persona_prompt_path(
    gemini_config: dict[str, Any],
) -> Path | None:
    """Return the configured persona prompt file path, if any."""
    raw = gemini_config.get("persona_prompt_file")
    if not isinstance(raw, str) or not raw.strip():
        return None

    expanded = os.path.expandvars(raw.strip())
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        try:
            from hermes_constants import get_hermes_home
            path = get_hermes_home() / path
        except Exception:
            path = Path(await aiofiles.os.getcwd()) / path
    return path


async def _read_gemini_persona_prompt(gemini_config: dict[str, Any]) -> str:
    """Read the Gemini persona prompt file, failing soft on config mistakes."""
    path = await _resolve_gemini_persona_prompt_path(gemini_config)
    if path is None:
        return ""
    try:
        async with aiofiles.open(path, encoding="utf-8") as prompt_file:
            return (await prompt_file.read()).strip()
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning(
            "Gemini TTS persona prompt file unavailable at %s: %s",
            path,
            exc,
        )
        return ""


def _gemini_model_supports_audio_tags(model: str) -> bool:
    """Return True for Gemini TTS models known to support expressive audio tags."""
    normalized = (model or "").strip().lower().rsplit("/", 1)[-1]
    return "gemini-3.1" in normalized and "tts" in normalized


def _gemini_audio_tags_enabled(gemini_config: dict[str, Any], model: str) -> bool:
    raw = gemini_config.get("audio_tags")
    if isinstance(raw, dict):
        raw = raw.get("enabled")
    enabled = _config_bool(raw, default=DEFAULT_GEMINI_AUDIO_TAGS)
    if not enabled:
        return False
    if not _gemini_model_supports_audio_tags(model):
        logger.warning(
            "Gemini TTS audio_tags enabled, but model %s is not known to support "
            "Gemini audio tags; skipping hidden tag rewrite",
            model,
        )
        return False
    return True


def _clean_gemini_audio_tag_rewrite(content: str) -> str:
    clean = (content or "").strip()
    fence = re.fullmatch(r"```(?:[A-Za-z0-9_-]+)?\s*(.*?)\s*```", clean, flags=re.DOTALL)
    if fence:
        clean = fence.group(1).strip()
    return clean


def _extract_auxiliary_message_content(response: Any) -> str:
    try:
        choice = response.choices[0]
        message = getattr(choice, "message", None)
        if isinstance(message, dict):
            return str(message.get("content") or "")
        return str(getattr(message, "content", "") or "")
    except Exception:
        return ""


async def _rewrite_gemini_tts_audio_tags(
    text: str,
    persona_prompt: str = "",
) -> str:
    """Use the configured auxiliary model to insert Gemini audio tags."""
    transcript = text.strip()
    if not transcript:
        return text

    system_prompt = (
        "You rewrite transcripts for Gemini 3.1 Flash TTS by inserting expressive "
        "audio tags.\n\n"
        "Audio tags are inline square-bracket modifiers such as [whispers], "
        "[excitedly], [very slow], [sarcastically], [laughs], [sighs], or [gasp]. "
        "There is no fixed allowlist. Use creative freeform tags generously but "
        "naturally to control tone, pace, emotional vibe, emphasis, section-level "
        "delivery, and non-verbal sounds. Use English audio tags even when the "
        "spoken transcript is not English.\n\n"
        "Rules:\n"
        "- Preserve the spoken words, order, and meaning.\n"
        "- Do not add new spoken sentences or remove existing spoken words.\n"
        "- Use square brackets for every audio tag.\n"
        "- Do not use SSML or XML tags.\n"
        "- Do not explain or comment.\n"
        "- Return only the tagged TTS script."
    )
    context = persona_prompt.strip() or "(none)"
    user_prompt = (
        "PERSONA AND DIRECTOR CONTEXT:\n"
        f"{context}\n\n"
        "TRANSCRIPT TO TAG:\n"
        f"{transcript}"
    )
    try:
        from agent.auxiliary_client import call_llm

        response = await call_llm(
            task=GEMINI_AUDIO_TAG_REWRITE_TASK,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        tagged = _clean_gemini_audio_tag_rewrite(_extract_auxiliary_message_content(response))
        return tagged or text
    except Exception as exc:
        logger.warning("Gemini TTS audio tag rewrite failed; using untagged text: %s", exc)
        return text


async def _compose_gemini_tts_prompt(
    text: str,
    gemini_config: dict[str, Any],
    persona_prompt: str | None = None,
) -> str:
    """Build the Gemini prompt from persona direction plus the live transcript."""
    transcript = text.strip()
    if persona_prompt is None:
        persona_prompt = await _read_gemini_persona_prompt(gemini_config)
    if not persona_prompt:
        return transcript

    preamble = (
        "Synthesize speech from the TRANSCRIPT only. Treat AUDIO PROFILE, "
        "SCENE, DIRECTOR'S NOTES, and SAMPLE CONTEXT as performance direction; "
        "do not speak those sections aloud."
    )

    placeholder_patterns = (
        re.compile(r"\{\{\s*transcript\s*\}\}", flags=re.IGNORECASE),
        re.compile(r"\{\s*transcript\s*\}", flags=re.IGNORECASE),
    )
    prompt = persona_prompt
    for pattern in placeholder_patterns:
        if pattern.search(prompt):
            prompt = pattern.sub(transcript, prompt)
            return f"{preamble}\n\n{prompt}".strip()

    return f"{preamble}\n\n{persona_prompt}\n\n#### TRANSCRIPT\n{transcript}".strip()


async def _generate_gemini_tts(
    text: str,
    output_path: str,
    tts_config: dict[str, Any],
) -> str:
    """Generate audio using Google Gemini TTS.

    Gemini's generateContent endpoint with responseModalities=["AUDIO"] returns
    raw 24kHz mono 16-bit PCM (L16) as base64. We wrap it with a WAV RIFF
    header to produce a playable file, then ffmpeg-convert to MP3 / Opus if
    the caller requested those formats (same pattern as NeuTTS).

    Args:
        text: Text to convert (prompt-style; supports inline direction like
              "Say cheerfully:" and audio tags like [whispers]).
        output_path: Where to save the audio file (.wav, .mp3, or .ogg).
        tts_config: TTS config dict.

    Returns:
        Path to the saved audio file.
    """
    httpx = _get_httpx_module()
    api_key = (
        await _resolve_provider_key("GEMINI_API_KEY", "gemini")
        or await _resolve_provider_key("GOOGLE_API_KEY", "gemini")
    )
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not set. Get one at https://aistudio.google.com/app/apikey"
        )

    raw_gemini_config = tts_config.get("gemini") or {}
    gemini_config = raw_gemini_config if isinstance(raw_gemini_config, dict) else {}
    model = str(gemini_config.get("model", DEFAULT_GEMINI_TTS_MODEL)).strip() or DEFAULT_GEMINI_TTS_MODEL
    voice = str(gemini_config.get("voice", DEFAULT_GEMINI_TTS_VOICE)).strip() or DEFAULT_GEMINI_TTS_VOICE
    base_url = str(
        gemini_config.get("base_url")
        or await get_env_value("GEMINI_BASE_URL")
        or DEFAULT_GEMINI_TTS_BASE_URL
    ).strip().rstrip("/")
    persona_prompt = await _read_gemini_persona_prompt(gemini_config)
    tts_script = text
    if _gemini_audio_tags_enabled(gemini_config, model):
        tts_script = await _rewrite_gemini_tts_audio_tags(
            text,
            persona_prompt=persona_prompt,
        )
    prompt_text = await _compose_gemini_tts_prompt(
        tts_script,
        gemini_config,
        persona_prompt=persona_prompt,
    )
    max_len = _resolve_max_text_length("gemini", tts_config)
    if len(prompt_text) > max_len:
        logger.warning(
            "Gemini TTS composed prompt too long (%d chars), truncating to %d",
            len(prompt_text), max_len,
        )
        prompt_text = prompt_text[:max_len]

    payload: dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": voice},
                },
            },
        },
    }

    headers = {"Content-Type": "application/json"}
    if urlparse(base_url).hostname == "generativelanguage.googleapis.com":
        try:
            import hermes_cli as _hermes_cli

            _hermes_version = str(_hermes_cli.__version__)
        except Exception:
            _hermes_version = "0.0.0"
        # Include Hermes client context following Gemini's partner
        # integration guidance:
        # https://ai.google.dev/gemini-api/docs/partner-integration
        headers["X-Goog-Api-Client"] = f"hermes-agent/{_hermes_version}"

    endpoint = f"{base_url}/models/{model}:generateContent"
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream(
            "POST",
            endpoint,
            params={"key": api_key},
            headers=headers,
            json=payload,
        ) as response:
            if response.status_code != 200:
                raw_body = await _read_tts_response_bytes(
                    response,
                    label="Gemini TTS",
                )
                try:
                    err = (
                        json.loads(raw_body.decode("utf-8")).get("error", {})
                        if raw_body
                        else {}
                    )
                    detail = err.get("message") or raw_body.decode(
                        "utf-8",
                        errors="replace",
                    )[:300]
                except Exception:
                    detail = raw_body.decode("utf-8", errors="replace")[:300]
                raise RuntimeError(
                    f"Gemini TTS API error (HTTP {response.status_code}): {detail}"
                )
            data = await _read_tts_response_json(response, label="Gemini TTS")
    try:
        parts = data["candidates"][0]["content"]["parts"]
        audio_part = next((p for p in parts if "inlineData" in p or "inline_data" in p), None)
        if audio_part is None:
            raise RuntimeError("Gemini TTS response contained no audio data")
        inline = audio_part.get("inlineData") or audio_part.get("inline_data") or {}
        audio_b64 = inline.get("data", "")
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Gemini TTS response was malformed: {e}") from e

    if not audio_b64:
        raise RuntimeError("Gemini TTS returned empty audio data")

    pcm_bytes = base64.b64decode(audio_b64)
    wav_bytes = _wrap_pcm_as_wav(pcm_bytes)

    # Fast path: caller wants WAV directly, just write.
    if output_path.lower().endswith(".wav"):
        async with aiofiles.open(output_path, "wb") as output_file:
            await output_file.write(wav_bytes)
        return output_path

    # Otherwise write WAV to a temp file and ffmpeg-convert to the target
    # format (.mp3 or .ogg). If ffmpeg is missing, fall back to renaming the
    # WAV -- this matches the NeuTTS behavior and keeps the tool usable on
    # systems without ffmpeg (audio still plays, just with a misleading
    # extension).
    async with aiofiles.tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    ) as tmp:
        await tmp.write(wav_bytes)
        wav_path = tmp.name

    try:
        if await _has_ffmpeg():
            # For .ogg output, force libopus encoding (Telegram voice bubbles
            # require Opus specifically; ffmpeg's default for .ogg is Vorbis).
            if output_path.lower().endswith(".ogg"):
                cmd = [
                    "ffmpeg", "-i", wav_path,
                    "-acodec", "libopus", "-ac", "1",
                    "-b:a", "48k", "-vbr", "on",
                    "-application", "voip", "-compression_level", "10",
                    "-y", "-loglevel", "error",
                    output_path,
                ]
            else:
                cmd = ["ffmpeg", "-i", wav_path, "-y", "-loglevel", "error", output_path]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                env=await _non_model_tts_subprocess_env(),
            )
            try:
                _, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=30,
                )
            except asyncio.CancelledError:
                await _finish_owned_tts_cleanup(
                    _kill_and_reap_tts_process(process)
                )
                raise
            except TimeoutError as exc:
                timeout_error = exc
            else:
                timeout_error = None
            if timeout_error is not None:
                await _finish_owned_tts_cleanup(
                    _kill_and_reap_tts_process(process)
                )
                raise RuntimeError(
                    "ffmpeg conversion timed out after 30s"
                ) from timeout_error
            if process.returncode != 0:
                stderr = stderr.decode("utf-8", errors="ignore")[:300]
                raise RuntimeError(f"ffmpeg conversion failed: {stderr}")
        else:
            logger.warning(
                "ffmpeg not found; writing raw WAV to %s (extension may be misleading)",
                output_path,
            )
            async with aiofiles.open(wav_path, "rb") as source_file:
                wav_data = await source_file.read()
            async with aiofiles.open(output_path, "wb") as output_file:
                await output_file.write(wav_data)
    finally:
        try:
            await aiofiles.os.remove(wav_path)
        except OSError:
            pass

    return output_path


# ===========================================================================
# Local SDK providers
# ===========================================================================
# NeuTTS, KittenTTS, and Piper expose synchronous synthesis APIs. They run in
# owned child processes so model and file I/O never blocks the agent event
# loop. The parent awaits, times out, cancels, and reaps every child.


async def _check_python_module_available(module_name: str) -> bool:
    import sys

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        (
            "import importlib.util,sys;"
            f"sys.exit(0 if importlib.util.find_spec({module_name!r}) else 1)"
        ),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        env=await _non_model_tts_subprocess_env(),
    )
    try:
        await asyncio.wait_for(process.wait(), timeout=10)
    except asyncio.CancelledError:
        await _finish_owned_tts_cleanup(_kill_and_reap_tts_process(process))
        raise
    except TimeoutError:
        await _finish_owned_tts_cleanup(_kill_and_reap_tts_process(process))
        return False
    return process.returncode == 0


async def _check_neutts_available() -> bool:
    """Check if the neutts engine is importable (installed locally)."""
    return await _check_python_module_available("neutts")


async def _check_kittentts_available() -> bool:
    """Check if the kittentts engine is importable (installed locally)."""
    return await _check_python_module_available("kittentts")


async def _check_piper_available() -> bool:
    """Check whether the piper-tts package is importable."""
    return await _check_python_module_available("piper")


def _default_neutts_ref_audio() -> str:
    return str(Path(__file__).parent / "neutts_samples" / "jo.wav")


def _default_neutts_ref_text() -> str:
    return str(Path(__file__).parent / "neutts_samples" / "jo.txt")


async def _communicate_tts_process(
    process: asyncio.subprocess.Process,
    timeout: float | None,
    input_data: bytes | None = None,
) -> tuple[bytes, bytes]:
    try:
        communicate = (
            process.communicate()
            if input_data is None
            else process.communicate(input=input_data)
        )
        if timeout is None:
            stdout, stderr = await communicate
        else:
            stdout, stderr = await asyncio.wait_for(
                communicate,
                timeout=timeout,
            )
        return stdout or b"", stderr or b""
    except asyncio.CancelledError:
        await _finish_owned_tts_cleanup(_kill_and_reap_tts_process(process))
        raise
    except TimeoutError as exc:
        timeout_error = exc
    await _finish_owned_tts_cleanup(_kill_and_reap_tts_process(process))
    raise timeout_error


async def _generate_neutts(
    text: str,
    output_path: str,
    tts_config: dict[str, Any],
) -> str:
    import sys

    neutts_config = tts_config.get("neutts") or {}
    ref_audio = (
        neutts_config.get("ref_audio", "") or _default_neutts_ref_audio()
    )
    ref_text = (
        neutts_config.get("ref_text", "") or _default_neutts_ref_text()
    )
    model = neutts_config.get("model", "neuphonic/neutts-air-q4-gguf")
    device = neutts_config.get("device", "cpu")
    wav_path = (
        output_path
        if output_path.endswith(".wav")
        else output_path.rsplit(".", 1)[0] + ".wav"
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(Path(__file__).parent / "neutts_synth.py"),
        "--text",
        text,
        "--out",
        wav_path,
        "--ref-audio",
        str(ref_audio),
        "--ref-text",
        str(ref_text),
        "--model",
        str(model),
        "--device",
        str(device),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=await _non_model_tts_subprocess_env(include_hf_token=True),
    )
    try:
        _, stderr = await _communicate_tts_process(process, 120)
    except TimeoutError as exc:
        raise RuntimeError("NeuTTS synthesis timed out after 120s") from exc
    if process.returncode != 0:
        error_lines = [
            line
            for line in stderr.decode("utf-8", errors="replace").splitlines()
            if not line.startswith("OK:")
        ]
        raise RuntimeError(
            "NeuTTS synthesis failed: "
            + ("\n".join(error_lines) or "unknown error")
        )

    if wav_path != output_path:
        if await _has_ffmpeg():
            converter = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-i",
                wav_path,
                "-y",
                "-loglevel",
                "error",
                output_path,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=await _non_model_tts_subprocess_env(),
            )
            try:
                _, conversion_error = await _communicate_tts_process(
                    converter,
                    30,
                )
            except TimeoutError as exc:
                raise RuntimeError(
                    "NeuTTS ffmpeg conversion timed out after 30s"
                ) from exc
            if converter.returncode != 0:
                raise RuntimeError(
                    "NeuTTS ffmpeg conversion failed: "
                    + conversion_error.decode("utf-8", errors="replace")[:300]
                )
            await aiofiles.os.remove(wav_path)
        else:
            await aiofiles.os.replace(wav_path, output_path)
    return output_path


# Each cache entry is a loaded local model. Keep the upstream LRU helper and
# cache names for behavior-level parity. The synchronous SDK objects live in
# the profile-scoped broker child, which applies this same LRU contract.
_TTS_MODEL_CACHE_MAX = 3


def _tts_cache_get_or_load(
    cache: dict[str, Any],
    key: str,
    load: Callable[[], Any],
) -> Any:
    """Get ``key`` from ``cache`` or load it, keeping the cache LRU-bounded."""
    if key in cache:
        cache[key] = cache.pop(key)
        return cache[key]
    value = load()
    cache[key] = value
    while len(cache) > _TTS_MODEL_CACHE_MAX:
        cache.pop(next(iter(cache)), None)
    return value


_piper_voice_cache: dict[str, Any] = {}
_kittentts_model_cache: dict[str, Any] = {}


async def _get_piper_voices_dir() -> Path:
    """Return the profile-scoped Piper voice cache directory."""
    from hermes_constants import get_hermes_dir

    root = Path(
        await get_hermes_dir("cache/piper-voices", "piper_voices_cache")
    )
    await aiofiles.os.makedirs(root, exist_ok=True)
    return root


async def _resolve_piper_voice_path(voice: str, download_dir: Path) -> str:
    """Resolve a Piper voice path, downloading a named voice when needed."""
    import sys

    if not voice:
        voice = DEFAULT_PIPER_VOICE

    candidate = Path(voice).expanduser()
    if (
        candidate.suffix.lower() == ".onnx"
        and await aiofiles.os.path.exists(candidate)
    ):
        return str(candidate)

    cached = download_dir / f"{voice}.onnx"
    cached_config = download_dir / f"{voice}.onnx.json"
    if (
        await aiofiles.os.path.exists(cached)
        and await aiofiles.os.path.exists(cached_config)
    ):
        return str(cached)

    await aiofiles.os.makedirs(download_dir, exist_ok=True)
    logger.info(
        "[Piper] Downloading voice '%s' to %s (first use)",
        voice,
        download_dir,
    )
    command = [
        sys.executable,
        "-m",
        "piper.download_voices",
        voice,
        "--download-dir",
        str(download_dir),
    ]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=await _non_model_tts_subprocess_env(),
    )
    try:
        _, stderr = await _communicate_tts_process(process, 300)
    except TimeoutError as exc:
        raise RuntimeError(
            f"Piper voice download timed out after 300s for '{voice}'"
        ) from exc
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Piper voice download failed for '{voice}': "
            f"{(detail or 'no stderr output')[:400]}"
        )
    if not await aiofiles.os.path.exists(cached):
        raise RuntimeError(
            f"Piper voice download completed but {cached} is missing — "
            "check voice name (see: https://github.com/OHF-Voice/piper1-gpl/"
            "blob/main/docs/VOICES.md)"
        )
    return str(cached)


_LOCAL_TTS_RESPONSE_PREFIX = "HERMES_TTS_RESPONSE:"
_LocalTTSScope = tuple[asyncio.AbstractEventLoop, str]


async def _finish_local_tts_task(
    task: asyncio.Task[Any],
    cancellation: asyncio.CancelledError | None = None,
) -> Any:
    """Finish an owned broker task and preserve the first cancellation."""
    first_cancellation = cancellation
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as exc:  # noqa: ASYNC103
            if task.cancelled():
                if first_cancellation is not None:
                    raise first_cancellation from exc  # noqa: ASYNC104
                raise
            if first_cancellation is None:
                first_cancellation = exc
        except BaseException as exc:
            if first_cancellation is not None:
                raise first_cancellation from exc
            raise
    if first_cancellation is not None:
        raise first_cancellation
    return result


class _LocalTTSBroker:
    """One persistent sync-SDK worker owned by an async profile scope."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self.process = process
        self._responses: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._diagnostic_tail = ""
        self._closed = False
        self._stdout_task = asyncio.create_task(
            self._read_stdout(),
            name="local-tts-broker-stdout",
        )
        self._stderr_task = asyncio.create_task(
            self._read_stderr(),
            name="local-tts-broker-stderr",
        )

    @classmethod
    async def spawn(cls) -> "_LocalTTSBroker":
        from tools.environments.local import hermes_subprocess_env

        env = await hermes_subprocess_env(inherit_credentials=False)
        env["PYTHONUNBUFFERED"] = "1"
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__).parent / "local_tts_synth.py"),
            "--broker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            start_new_session=os.name != "nt",
            creationflags=(
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                if os.name == "nt"
                else 0
            ),
            limit=1024 * 1024,
        )
        return cls(process)

    @property
    def closed(self) -> bool:
        return self._closed or self.process.returncode is not None

    def _append_diagnostic(self, value: str) -> None:
        if not value:
            return
        self._diagnostic_tail = (self._diagnostic_tail + value)[-4000:]

    async def _read_stdout(self) -> None:
        stream = self.process.stdout
        assert stream is not None
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace")
                marker = decoded.find(_LOCAL_TTS_RESPONSE_PREFIX)
                if marker < 0:
                    self._append_diagnostic(decoded)
                    continue
                self._append_diagnostic(decoded[:marker])
                encoded_response = decoded[
                    marker + len(_LOCAL_TTS_RESPONSE_PREFIX) :
                ].strip()
                try:
                    response = json.loads(encoded_response)
                except (TypeError, ValueError):
                    self._append_diagnostic(decoded[marker:])
                    return
                if not isinstance(response, dict):
                    self._append_diagnostic(decoded[marker:])
                    return
                await self._responses.put(response)
        except Exception as exc:
            self._append_diagnostic(
                "local TTS stdout protocol failed: "
                f"{type(exc).__name__}: {exc}\n"
            )
        finally:
            await self._responses.put(None)

    async def _read_stderr(self) -> None:
        stream = self.process.stderr
        assert stream is not None
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                return
            self._append_diagnostic(chunk.decode("utf-8", errors="replace"))

    async def _finish_readers(self) -> None:
        await asyncio.gather(
            self._stdout_task,
            self._stderr_task,
            return_exceptions=True,
        )

    def _close_stdin(self) -> None:
        writer = self.process.stdin
        if writer is not None and not writer.is_closing():
            writer.close()

    async def _abort_unprotected(self) -> None:
        self._closed = True
        self._close_stdin()
        if self.process.returncode is None:
            await _terminate_command_tts_process_tree(self.process)
        else:
            await self.process.wait()
        await self._finish_readers()

    async def abort(
        self,
        cancellation: asyncio.CancelledError | None = None,
    ) -> None:
        cleanup = asyncio.create_task(
            self._abort_unprotected(),
            name="local-tts-broker-abort",
        )
        await _finish_local_tts_task(cleanup, cancellation)

    async def close(self) -> None:
        if self._closed:
            await self._finish_readers()
            return
        self._closed = True
        writer = self.process.stdin
        try:
            if self.process.returncode is None and writer is not None:
                writer.write(b'{"command":"shutdown"}\n')
                await writer.drain()
                self._close_stdin()
                await asyncio.wait_for(self.process.wait(), timeout=5)
            else:
                await self.process.wait()
            await self._finish_readers()
        except asyncio.CancelledError as cancellation:  # noqa: ASYNC103
            self._closed = False
            await self.abort(cancellation)
        except Exception:
            self._closed = False
            await self.abort()

    async def request(self, provider: str, payload: dict[str, Any]) -> None:
        if self.closed:
            detail = self._diagnostic_tail.strip() or "worker exited unexpectedly"
            raise RuntimeError(f"{provider} synthesis failed: {detail}")
        request_id = uuid.uuid4().hex
        request = json.dumps(
            {
                "id": request_id,
                "provider": provider,
                "payload": payload,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        writer = self.process.stdin
        assert writer is not None
        try:
            writer.write(request)
            await writer.drain()
            while True:
                response = await self._responses.get()
                if response is None:
                    await self.abort()
                    detail = (
                        self._diagnostic_tail.strip()
                        or "worker exited unexpectedly"
                    )
                    raise RuntimeError(
                        f"{provider} synthesis failed: {detail}"
                    )
                if str(response.get("id", "")) != request_id:
                    await self.abort()
                    raise RuntimeError(
                        f"{provider} synthesis failed: worker protocol "
                        "returned an unexpected response id"
                    )
                if response.get("ok") is True:
                    return
                detail = str(response.get("error") or "unknown error")
                error_type = str(response.get("error_type") or "")
                if error_type == "ValueError":
                    raise ValueError(detail)
                if error_type == "FileNotFoundError":
                    raise FileNotFoundError(detail)
                raise RuntimeError(detail)
        except asyncio.CancelledError as cancellation:  # noqa: ASYNC103
            await self.abort(cancellation)
        except (BrokenPipeError, ConnectionError):
            await self.abort()  # noqa: ASYNC120
            detail = self._diagnostic_tail.strip() or "worker exited unexpectedly"
            raise RuntimeError(f"{provider} synthesis failed: {detail}")


@dataclass
class _LocalTTSScopeState:
    profile_home: str
    broker: _LocalTTSBroker | None = None
    consumers: weakref.WeakSet[object] = field(default_factory=weakref.WeakSet)
    lock: asyncio.Lock | None = None
    lock_users: int = 0


_local_tts_scope_states: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, _LocalTTSScopeState]
] = weakref.WeakKeyDictionary()
_local_tts_owner_scopes: weakref.WeakKeyDictionary[
    object, tuple[weakref.ReferenceType[asyncio.AbstractEventLoop], str]
] = weakref.WeakKeyDictionary()
_local_tts_scope_guard = threading.RLock()
_local_tts_scope_context: contextvars.ContextVar[
    tuple[str, str] | None
] = contextvars.ContextVar("local_tts_profile_scope", default=None)


class _LocalTTSStateLock:
    def __init__(self, state: _LocalTTSScopeState) -> None:
        self.state = state
        self.lock: asyncio.Lock | None = None

    async def __aenter__(self) -> None:
        with _local_tts_scope_guard:
            lock = self.state.lock
            if lock is None:
                lock = asyncio.Lock()
                self.state.lock = lock
            self.state.lock_users += 1
            self.lock = lock
        try:
            await lock.acquire()
        except BaseException:
            with _local_tts_scope_guard:
                self.state.lock_users -= 1
                if self.state.lock_users == 0:
                    self.state.lock = None
            raise

    async def __aexit__(self, *_exc: Any) -> None:
        assert self.lock is not None
        self.lock.release()
        with _local_tts_scope_guard:
            self.state.lock_users -= 1
            if self.state.lock_users == 0:
                self.state.lock = None


def _lexical_local_tts_home() -> str:
    from hermes_constants import get_hermes_home

    return os.path.normcase(os.fspath(get_hermes_home()))


async def _activate_local_tts_scope() -> tuple[
    _LocalTTSScope,
    _LocalTTSScopeState,
]:
    loop = asyncio.get_running_loop()
    lexical = _lexical_local_tts_home()
    active = _local_tts_scope_context.get()
    if active is not None and active[0] == lexical:
        canonical = active[1]
    else:
        expanduser = aiofiles.os.wrap(os.path.expanduser)
        expanded = str(await expanduser(lexical))
        if not os.path.isabs(expanded):
            expanded = str(await aiofiles.os.getcwd()) + os.sep + expanded
        realpath = aiofiles.os.wrap(os.path.realpath)
        canonical = os.path.normcase(str(await realpath(expanded)))
        _local_tts_scope_context.set((lexical, canonical))
    with _local_tts_scope_guard:
        for closed_loop in tuple(_local_tts_scope_states):
            if closed_loop.is_closed():
                _local_tts_scope_states.pop(closed_loop, None)
        states = _local_tts_scope_states.setdefault(loop, {})
        state = states.setdefault(canonical, _LocalTTSScopeState(canonical))
    return (loop, canonical), state


def _existing_local_tts_state(
    scope: _LocalTTSScope,
) -> _LocalTTSScopeState | None:
    loop, profile = scope
    with _local_tts_scope_guard:
        states = _local_tts_scope_states.get(loop)
        return states.get(profile) if states is not None else None


def _discard_local_tts_scope_if_idle(
    scope: _LocalTTSScope,
    state: _LocalTTSScopeState,
) -> None:
    if state.broker is not None or state.consumers or state.lock_users:
        return
    loop, profile = scope
    with _local_tts_scope_guard:
        states = _local_tts_scope_states.get(loop)
        if states is None or states.get(profile) is not state:
            return
        states.pop(profile, None)
        if not states:
            _local_tts_scope_states.pop(loop, None)


async def _retain_local_tts_lifecycle(owner: object) -> None:
    scope, state = await _activate_local_tts_scope()
    try:
        async with _LocalTTSStateLock(state):
            with _local_tts_scope_guard:
                retained = _local_tts_owner_scopes.get(owner)
                if retained is not None:
                    retained_loop = retained[0]()
                    if retained_loop is not scope[0] or retained[1] != scope[1]:
                        raise RuntimeError(
                            "The local TTS lifecycle lease belongs to another "
                            "event loop or Hermes profile"
                        )
                state.consumers.add(owner)
                _local_tts_owner_scopes[owner] = (
                    weakref.ref(scope[0]),
                    scope[1],
                )
    except BaseException:
        _discard_local_tts_scope_if_idle(scope, state)
        raise


async def _release_local_tts_lifecycle_owned(
    owner: object,
    scope: _LocalTTSScope,
    state: _LocalTTSScopeState,
) -> None:
    async with _LocalTTSStateLock(state):
        state.consumers.discard(owner)
        with _local_tts_scope_guard:
            retained = _local_tts_owner_scopes.get(owner)
            if (
                retained is not None
                and retained[0]() is scope[0]
                and retained[1] == scope[1]
            ):
                _local_tts_owner_scopes.pop(owner, None)
        if not state.consumers and state.broker is not None:
            broker = state.broker
            state.broker = None
            await broker.close()
    _discard_local_tts_scope_if_idle(scope, state)


async def _release_local_tts_lifecycle(owner: object) -> None:
    loop = asyncio.get_running_loop()
    with _local_tts_scope_guard:
        retained = _local_tts_owner_scopes.get(owner)
    if retained is None:
        return
    if retained[0]() is not loop:
        raise RuntimeError(
            "The local TTS lifecycle lease belongs to another event loop; "
            "release it on its owning loop"
        )
    scope = (loop, retained[1])
    state = _existing_local_tts_state(scope)
    if state is None:
        with _local_tts_scope_guard:
            _local_tts_owner_scopes.pop(owner, None)
        return
    cleanup = asyncio.create_task(
        _release_local_tts_lifecycle_owned(owner, scope, state),
        name="local-tts-lifecycle-release",
    )
    await _finish_local_tts_task(cleanup)


async def _retain_active_agent_local_tts_lifecycle() -> None:
    from agent.subagent_lifecycle import get_active_subagent_parent

    owner = get_active_subagent_parent()
    if owner is None or getattr(owner, "_local_tts_lifecycle_retained", False):
        return
    await _retain_local_tts_lifecycle(owner)
    owner._local_tts_lifecycle_retained = True


async def _run_local_tts_synth(
    provider: str,
    payload: dict[str, Any],
) -> None:
    """Synthesize through the active profile's persistent native-async broker."""
    await _retain_active_agent_local_tts_lifecycle()
    scope, state = await _activate_local_tts_scope()
    try:
        async with _LocalTTSStateLock(state):
            broker = state.broker
            if broker is None or broker.closed:
                if broker is not None:
                    # Detach before awaiting cleanup. If cancellation arrives
                    # while an idle worker that already exited is being reaped,
                    # the closed broker must not remain reachable from the
                    # loop/profile state and keep that loop alive.
                    state.broker = None
                    await broker.close()
                broker = await _LocalTTSBroker.spawn()
                state.broker = broker
            try:
                await broker.request(provider, payload)
            finally:
                if broker.closed and state.broker is broker:
                    state.broker = None
                if not state.consumers and state.broker is not None:
                    standalone_broker = state.broker
                    state.broker = None
                    cleanup = asyncio.create_task(
                        standalone_broker.close(),
                        name="local-tts-standalone-close",
                    )
                    await _finish_local_tts_task(cleanup)
    finally:
        _discard_local_tts_scope_if_idle(scope, state)


async def _finalize_local_wav(wav_path: str, output_path: str) -> str:
    """Convert an SDK-produced WAV to the requested container when needed."""
    if wav_path == output_path:
        return output_path
    if not await _has_ffmpeg():
        await aiofiles.os.replace(wav_path, output_path)
        return output_path

    command = [
        "ffmpeg",
        "-i",
        wav_path,
        "-y",
        "-loglevel",
        "error",
        output_path,
    ]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=await _non_model_tts_subprocess_env(),
    )
    try:
        stdout, stderr = await _communicate_tts_process(process, 30)
    except TimeoutError as exc:
        raise subprocess.TimeoutExpired(command, 30) from exc
    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode,
            command,
            output=stdout,
            stderr=stderr,
        )
    try:
        await aiofiles.os.remove(wav_path)
    except OSError:
        pass
    return output_path


async def _generate_kittentts(
    text: str,
    output_path: str,
    tts_config: dict[str, Any],
) -> str:
    """Generate speech with KittenTTS in an owned subprocess."""
    kt_config = tts_config.get("kittentts", {})
    wav_path = (
        output_path
        if output_path.endswith(".wav")
        else output_path.rsplit(".", 1)[0] + ".wav"
    )
    await _run_local_tts_synth(
        "kittentts",
        {
            "text": text,
            "output_path": wav_path,
            "model": kt_config.get("model", DEFAULT_KITTENTTS_MODEL),
            "voice": kt_config.get("voice", DEFAULT_KITTENTTS_VOICE),
            "speed": kt_config.get("speed", 1.0),
            "clean_text": kt_config.get("clean_text", True),
        },
    )
    return await _finalize_local_wav(wav_path, output_path)


async def _generate_piper_tts(
    text: str,
    output_path: str,
    tts_config: dict[str, Any],
) -> str:
    """Generate speech with Piper in an owned subprocess."""
    piper_config = (
        (tts_config.get("piper") or {})
        if isinstance(tts_config, dict)
        else {}
    )
    voice_name = piper_config.get("voice") or DEFAULT_PIPER_VOICE
    voices_dir = piper_config.get("voices_dir")
    download_dir = (
        Path(voices_dir).expanduser()
        if voices_dir
        else await _get_piper_voices_dir()
    )
    model_path = await _resolve_piper_voice_path(voice_name, download_dir)
    raw_speaker = piper_config.get("speaker_id", 0)
    speaker_id = (
        raw_speaker
        if isinstance(raw_speaker, int) and not isinstance(raw_speaker, bool)
        else 0
    )
    has_advanced = any(
        key in piper_config
        for key in (
            "length_scale",
            "noise_scale",
            "noise_w_scale",
            "volume",
            "normalize_audio",
            "speaker_id",
        )
    )
    wav_path = (
        output_path
        if output_path.endswith(".wav")
        else output_path.rsplit(".", 1)[0] + ".wav"
    )
    await _run_local_tts_synth(
        "piper",
        {
            "text": text,
            "output_path": wav_path,
            "model_path": model_path,
            "use_cuda": bool(piper_config.get("use_cuda", False)),
            "speaker_id": speaker_id,
            "has_advanced": has_advanced,
            "length_scale": piper_config.get("length_scale", 1.0),
            "noise_scale": piper_config.get("noise_scale", 0.667),
            "noise_w_scale": piper_config.get("noise_w_scale", 0.8),
            "volume": piper_config.get("volume", 1.0),
            "normalize_audio": piper_config.get("normalize_audio", True),
        },
    )
    return await _finalize_local_wav(wav_path, output_path)


# ===========================================================================
# Single provider-safe synthesis pass
# ===========================================================================
async def _text_to_speech_single(
    text: str,
    output_path: str | None = None,
    speed: float | None = None,
    instructions: str | None = None,
    provider: str | None = None,
    *,
    tts_config_override: dict[str, Any] | None = None,
) -> str:
    """
    Convert text to speech audio.

    Reads provider/voice config from ~/.hermes/config.yaml (tts: section).
    The model sends text; the user configures voice and provider.

    On messaging platforms, the returned MEDIA:<path> tag is intercepted
    by the send pipeline and delivered as a native voice message.
    In CLI mode, the file is saved to ~/voice-memos/.

    Args:
        text: The text to convert to speech.
        output_path: Optional custom save path. Defaults to ~/voice-memos/<timestamp>.mp3
        speed: Optional playback speed multiplier (0.25-4.0). Overrides config.yaml.
        instructions: Optional voice-design guidance (tone, emotion, pacing,
            accent, whispering). Forwarded to the OpenAI backend
            (gpt-4o-mini-tts and OpenAI-compatible servers). Silently
            ignored by backends that don't support it.
        provider: Optional TTS provider override. When set, bypasses the
            configured ``tts.provider`` and uses this provider instead.
            Accepts built-in names (``edge``, ``openai``, ``elevenlabs``,
            ``minimax``, ``xai``, ``mistral``, ``gemini``, ``neutts``,
            ``kittentts``, ``piper``), user-declared command provider names
            from ``tts.providers.<name>``, or plugin-registered provider
            names.  When ``None`` (the default), the configured provider
            from ``tts.provider`` in config.yaml is used.

    Returns:
        str: JSON result with success, file_path, and optionally MEDIA tag.
    """
    if not text or not text.strip():
        return tool_error("Text is required", success=False)

    try:
        from tools.tts_text_normalize import prepare_spoken_text
        text = prepare_spoken_text(text, max_chars=None)
    except Exception:
        text = text.strip()
    if not text:
        return tool_error("Text is empty after TTS cleanup", success=False)

    tts_config = (
        tts_config_override
        if tts_config_override is not None
        else await _load_tts_config()
    )

    # When the model supplies a speed parameter, inject it into the config
    # so all downstream provider functions pick it up uniformly.
    if speed is not None:
        clamped = max(0.25, min(4.0, float(speed)))
        tts_config = dict(tts_config)  # shallow copy to avoid mutating the cache
        tts_config["speed"] = clamped

    # Allow per-call provider override; fall back to the configured default.
    if provider:
        provider = provider.lower().strip()
    else:
        provider = _get_provider(tts_config)

    # User-declared command provider (type: command under tts.providers.<name>)
    # resolves BEFORE the built-in dispatch. Built-in names short-circuit here
    # so a user's ``tts.providers.openai.command`` can't override the real
    # OpenAI handler.
    command_provider_config = _resolve_command_provider_config(provider, tts_config)

    # The public wrapper splits text before calling this function. Keep this
    # boundary explicit for private callers rather than silently truncating.
    max_len = _resolve_max_text_length(provider, tts_config)
    if len(text) > max_len:
        logger.warning(
            "TTS text exceeds provider %s cap (%d > %d chars) — "
            "use text_to_speech_tool() for automatic chunking",
            provider, len(text), max_len,
        )

    # Detect platform from gateway env var to choose the best output format.
    # Several platforms deliver native voice bubbles only for Ogg/Opus
    # (Telegram, Matrix, Feishu/Lark, WhatsApp, Signal); OpenAI and
    # ElevenLabs can produce Opus natively (no ffmpeg needed). Edge TTS
    # always outputs MP3 and needs ffmpeg for conversion.
    from gateway.session_context import get_session_env
    platform = get_session_env("HERMES_SESSION_PLATFORM", "").lower()
    want_opus = platform in OPUS_VOICE_PLATFORMS

    # Determine output path
    if output_path:
        # Reject '..' traversal components in the user-supplied path. An
        # explicit absolute path is fine (the agent legitimately writes
        # audio to user-specified locations), but a path that uses ``..``
        # to escape its declared base is almost always either a bug or
        # prompt-injection-controlled — e.g.
        # ``output_path="audio/../../etc/cron.d/x"``. The terminal tool
        # can still write anywhere with approval; this just keeps the
        # unattended TTS surface from materializing files via traversal.
        from tools.path_security import has_traversal_component
        if has_traversal_component(output_path):
            return json.dumps({
                "success": False,
                "error": (
                    f"output_path contains '..' traversal component: "
                    f"{output_path}. Use an absolute path or one relative "
                    "to the current directory without '..'."
                ),
            }, ensure_ascii=False)
        file_path = Path(output_path).expanduser()
        if command_provider_config is not None:
            # Respect caller-supplied path but align the extension with the
            # provider's configured output_format so the command writes to a
            # path the caller actually expects.
            file_path = _configured_command_tts_output_path(
                file_path, command_provider_config
            )
        from agent.file_safety import is_write_denied

        if await is_write_denied(str(file_path)):
            return json.dumps({
                "success": False,
                "error": (
                    f"output_path targets a protected credential or system path: "
                    f"{file_path}. Choose a normal audio output location."
                ),
            }, ensure_ascii=False)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_dir = Path(DEFAULT_OUTPUT_DIR or await _get_default_output_dir())
        await aiofiles.os.makedirs(out_dir, exist_ok=True)
        if command_provider_config is not None:
            fmt = _get_command_tts_output_format(command_provider_config)
            file_path = out_dir / f"tts_{timestamp}.{fmt}"
        # Use .ogg for Telegram with providers that support native Opus output,
        # otherwise fall back to .mp3 (Edge TTS will attempt ffmpeg conversion later).
        elif want_opus and provider in {"openai", "elevenlabs", "mistral", "gemini"}:
            file_path = out_dir / f"tts_{timestamp}.ogg"
        else:
            file_path = out_dir / f"tts_{timestamp}.mp3"

    # Ensure parent directory exists
    await aiofiles.os.makedirs(file_path.parent, exist_ok=True)
    file_str = str(file_path)

    try:
        # Generate audio with the configured provider
        if command_provider_config is not None:
            logger.info(
                "Generating speech with command TTS provider '%s'...", provider,
            )
            file_str = await _generate_command_tts(
                text, file_str, provider, command_provider_config, tts_config,
            )

        # Plugin-registered TTS backend (issue #30398). Fires when the
        # configured provider is neither a built-in nor a command-type
        # entry, AND a plugin is registered under that name. The walrus
        # binds `_plugin_path` only when the dispatcher returns a path
        # (i.e. a plugin was actually found); a None return falls
        # through to the built-in elif chain so unknown names hit the
        # Edge TTS default at the bottom. The dispatcher itself enforces
        # built-ins-always-win + command-wins-over-plugin defensively.
        elif provider not in BUILTIN_TTS_PROVIDERS and (
            _plugin_path := await _dispatch_to_plugin_provider(
                text, file_str, provider, tts_config,
            )
        ) is not None:
            file_str = _plugin_path

        elif provider == "elevenlabs":
            try:
                _import_elevenlabs()
            except ImportError:
                return json.dumps({
                    "success": False,
                    "error": "ElevenLabs provider selected but 'elevenlabs' package not installed. Run: pip install elevenlabs"
                }, ensure_ascii=False)
            logger.info("Generating speech with ElevenLabs...")
            await _generate_elevenlabs(text, file_str, tts_config)

        elif provider == "openai":
            try:
                _import_openai_client()
            except ImportError:
                return json.dumps({
                    "success": False,
                    "error": "OpenAI provider selected but 'openai' package not installed."
                }, ensure_ascii=False)
            logger.info("Generating speech with OpenAI TTS...")
            await _generate_openai_tts(
                text,
                file_str,
                tts_config,
                instructions=instructions,
            )

        elif provider == "deepinfra":
            try:
                _import_openai_client()
            except ImportError:
                return json.dumps({
                    "success": False,
                    "error": "DeepInfra TTS uses the 'openai' SDK but it isn't installed."
                }, ensure_ascii=False)
            logger.info("Generating speech with DeepInfra TTS...")
            await _generate_deepinfra_tts(text, file_str, tts_config)

        elif provider == "minimax":
            logger.info("Generating speech with MiniMax TTS...")
            await _generate_minimax_tts(text, file_str, tts_config)

        elif provider == "xai":
            logger.info("Generating speech with xAI TTS...")
            await _generate_xai_tts(text, file_str, tts_config)

        elif provider == "mistral":
            try:
                _import_mistral_client()
            except ImportError:
                return json.dumps({
                    "success": False,
                    "error": "Mistral provider selected but 'mistralai' package not installed. "
                             "Run `hermes setup` to install Mistral support."
                }, ensure_ascii=False)
            logger.info("Generating speech with Mistral Voxtral TTS...")
            await _generate_mistral_tts(text, file_str, tts_config)

        elif provider == "gemini":
            logger.info("Generating speech with Google Gemini TTS...")
            await _generate_gemini_tts(text, file_str, tts_config)

        elif provider == "neutts":
            if not await _check_neutts_available():
                return json.dumps({
                    "success": False,
                    "error": (
                        "NeuTTS not installed. Run: python -m pip install "
                        "-U 'neutts[all]'"
                    ),
                }, ensure_ascii=False)
            await _generate_neutts(text, file_str, tts_config)

        elif provider == "kittentts":
            if not await _check_kittentts_available():
                return json.dumps({
                    "success": False,
                    "error": (
                        "KittenTTS provider selected but the compatible package is not "
                        "installed. Install the official KittenML 0.8.1 wheel with: "
                        "python -m pip install 'https://github.com/KittenML/KittenTTS/"
                        "releases/download/0.8.1/kittentts-0.8.1-py3-none-any.whl"
                        "#sha256=482a436c4f1f3192153710376e459ff3689517ebcda7c2b051e2fd"
                        "4187b41851'. The public-index 'kittentts' package is not the "
                        "compatible KittenML 0.8.1 artifact."
                    ),
                }, ensure_ascii=False)
            logger.info("Generating speech with KittenTTS (local, ~25MB)...")
            await _generate_kittentts(text, file_str, tts_config)

        elif provider == "piper":
            if not await _check_piper_available():
                return json.dumps({
                    "success": False,
                    "error": "Piper provider selected but 'piper-tts' package not installed. "
                             "Install the local Piper backend with: "
                             "python -m pip install 'async-hermes-agent[piper-tts]'",
                }, ensure_ascii=False)
            logger.info("Generating speech with Piper (local)...")
            await _generate_piper_tts(text, file_str, tts_config)

        else:
            # Default: Edge TTS (free), with NeuTTS as local fallback
            edge_available = True
            try:
                _import_edge_tts()
            except ImportError:
                edge_available = False

            if edge_available:
                logger.info("Generating speech with Edge TTS...")
                await _generate_edge_tts(text, file_str, tts_config)
            elif await _check_neutts_available():
                logger.info(
                    "Edge TTS not available, falling back to NeuTTS (local)..."
                )
                provider = "neutts"
                await _generate_neutts(text, file_str, tts_config)
            else:
                return json.dumps({
                    "success": False,
                    "error": (
                        "No TTS provider available. Install edge-tts "
                        "(pip install edge-tts) or set up NeuTTS for local "
                        "synthesis."
                    ),
                }, ensure_ascii=False)

        # Check the file was actually created
        if (
            not await aiofiles.os.path.exists(file_str)
            or (await aiofiles.os.stat(file_str)).st_size == 0
        ):
            return json.dumps({
                "success": False,
                "error": f"TTS generation produced no output (provider: {provider})"
            }, ensure_ascii=False)

        # Class-level container repair: several backends silently write
        # MP3/WAV bytes into a .ogg output path (Edge, Piper, xAI,
        # OpenAI-compatible servers without opus support), which platforms
        # like Telegram render as broken 0-second voice bubbles. Sniff the
        # magic bytes once here — covering every current and future
        # provider — and transcode in place when they don't match.
        file_str = await _repair_ogg_container(file_str)

        # Try Opus conversion for Telegram compatibility.
        # Edge TTS outputs MP3, NeuTTS/KittenTTS output WAV. Keep those native
        # formats for local/CLI playback and only convert when the current
        # platform actually needs Opus voice delivery.
        voice_compatible = False
        if command_provider_config is not None:
            # Command providers are documents by default. Voice-bubble
            # delivery only kicks in when the user explicitly opts in
            # via ``voice_compatible: true`` in their provider config.
            if _is_command_tts_voice_compatible(command_provider_config):
                if not file_str.endswith(".ogg"):
                    opus_path = await _convert_to_opus(file_str)
                    if opus_path:
                        file_str = opus_path
                voice_compatible = file_str.endswith(".ogg")
        elif provider not in BUILTIN_TTS_PROVIDERS:
            # Plugin-registered provider (issue #30398). Voice-bubble
            # delivery opts in via ``TTSProvider.voice_compatible``
            # (mirrors the command-provider opt-in). Plugins that
            # already write Opus skip the ffmpeg conversion.
            plugin_voice_compatible = _plugin_provider_is_voice_compatible(provider)
            if plugin_voice_compatible:
                if not file_str.endswith(".ogg"):
                    opus_path = await _convert_to_opus(file_str)
                    if opus_path:
                        file_str = opus_path
                voice_compatible = file_str.endswith(".ogg")
        elif (
            want_opus
            and provider in {"edge", "neutts", "minimax", "xai", "kittentts", "piper"}
            and not file_str.endswith(".ogg")
        ):
            opus_path = await _convert_to_opus(file_str)
            if opus_path:
                file_str = opus_path
                voice_compatible = True
        elif provider in {"elevenlabs", "openai", "mistral", "gemini"}:
            voice_compatible = want_opus and file_str.endswith(".ogg")

        file_size = (await aiofiles.os.stat(file_str)).st_size
        logger.info("TTS audio saved: %s (%s bytes, provider: %s)", file_str, f"{file_size:,}", provider)

        # Build response with MEDIA tag for platform delivery
        media_tag = f"MEDIA:{file_str}"
        if voice_compatible:
            media_tag = f"[[audio_as_voice]]\n{media_tag}"

        return json.dumps({
            "success": True,
            "file_path": file_str,
            "media_tag": media_tag,
            "provider": provider,
            "voice_compatible": voice_compatible,
        }, ensure_ascii=False)

    except asyncio.CancelledError:
        raise
    except ValueError as e:
        # Configuration errors (missing API keys, etc.)
        error_msg = f"TTS configuration error ({provider}): {e}"
        logger.error("%s", error_msg)
        return tool_error(error_msg, success=False)
    except FileNotFoundError as e:
        # Missing dependencies or files
        error_msg = f"TTS dependency missing ({provider}): {e}"
        logger.error("%s", error_msg, exc_info=True)
        return tool_error(error_msg, success=False)
    except Exception as e:
        # Unexpected errors
        error_msg = f"TTS generation failed ({provider}): {e}"
        logger.error("%s", error_msg, exc_info=True)
        return tool_error(error_msg, success=False)


# ===========================================================================
# Public long-form tool wrapper
# ===========================================================================
async def text_to_speech_tool(
    text: str,
    output_path: str | None = None,
    speed: float | None = None,
    instructions: str | None = None,
    provider: str | None = None,
) -> str:
    """Convert text to speech with provider-safe long-form chunking.

    The public upstream contract remains one coroutine with the original
    arguments.  A single provider request keeps the existing result shape;
    longer input is split into ordered requests and the final encoded files
    are packed under the active platform's upload limit.
    """
    if not text or not text.strip():
        return tool_error("Text is required", success=False)

    try:
        from tools.tts_text_normalize import prepare_spoken_text

        text = prepare_spoken_text(text, max_chars=None)
    except Exception:
        text = text.strip()
    if not text:
        return tool_error("Text is empty after TTS cleanup", success=False)

    tts_config = await _load_tts_config()
    if speed is not None:
        clamped = max(0.25, min(4.0, float(speed)))
        tts_config = dict(tts_config)
        tts_config["speed"] = clamped

    if provider:
        provider = provider.lower().strip()
    else:
        provider = _get_provider(tts_config)
    command_provider_config = _resolve_command_provider_config(provider, tts_config)
    max_len = _resolve_max_text_length(provider, tts_config)
    chunks = _split_text_for_tts(text, max_len)
    if not chunks:
        return tool_error("Text is required", success=False)
    if len(chunks) > 1:
        logger.info(
            "TTS text for provider %s split into %d chunks (input=%d chars, cap=%d)",
            provider,
            len(chunks),
            len(text),
            max_len,
        )

    from gateway.session_context import get_session_env

    platform = get_session_env("HERMES_SESSION_PLATFORM", "").lower()
    want_opus = platform in OPUS_VOICE_PLATFORMS
    delivery_profile = _resolve_audio_delivery_profile(platform, tts_config)

    if output_path:
        from tools.path_security import has_traversal_component

        if has_traversal_component(output_path):
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        f"output_path contains '..' traversal component: {output_path}. "
                        "Use an absolute path or one relative to the current directory "
                        "without '..'."
                    ),
                },
                ensure_ascii=False,
            )
        base_path = Path(output_path).expanduser()
        if command_provider_config is not None:
            base_path = _configured_command_tts_output_path(
                base_path, command_provider_config
            )
        from agent.file_safety import is_write_denied

        if await is_write_denied(str(base_path)):
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        "output_path targets a protected credential or system path: "
                        f"{base_path}. Choose a normal audio output location."
                    ),
                },
                ensure_ascii=False,
            )
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_dir = Path(DEFAULT_OUTPUT_DIR or await _get_default_output_dir())
        await aiofiles.os.makedirs(out_dir, exist_ok=True)
        if command_provider_config is not None:
            fmt = _get_command_tts_output_format(command_provider_config)
            base_path = out_dir / f"tts_{timestamp}.{fmt}"
        elif want_opus and provider in {"openai", "elevenlabs", "mistral", "gemini"}:
            base_path = out_dir / f"tts_{timestamp}.ogg"
        else:
            base_path = out_dir / f"tts_{timestamp}.mp3"
    await aiofiles.os.makedirs(base_path.parent, exist_ok=True)

    generated_artifacts: set[str] = set()
    final_paths: list[str] = []
    chunk_results: list[dict[str, Any]] = []
    try:
        encoded_paths: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            chunk_path = (
                base_path
                if len(chunks) == 1
                else base_path.with_name(
                    f"{base_path.stem}.chunk{index:03d}{base_path.suffix}"
                )
            )
            generated_artifacts.add(str(chunk_path))
            raw_result = await _text_to_speech_single(
                text=chunk,
                output_path=str(chunk_path),
                speed=speed,
                instructions=instructions,
                provider=provider,
                tts_config_override=tts_config,
            )
            try:
                chunk_result = json.loads(raw_result)
            except (json.JSONDecodeError, TypeError) as exc:
                raise RuntimeError(
                    f"TTS chunk {index} returned invalid JSON: {str(raw_result)[:200]}"
                ) from exc
            if not chunk_result.get("success"):
                error_msg = chunk_result.get("error", "unknown error")
                return tool_error(
                    f"TTS chunk {index} failed ({provider}): {error_msg}",
                    success=False,
                )
            actual_path = str(chunk_result.get("file_path") or chunk_path)
            if not await aiofiles.os.path.exists(actual_path):
                raise RuntimeError(
                    f"TTS chunk {index} produced no final audio: {actual_path}"
                )
            if (await aiofiles.os.stat(actual_path)).st_size <= 0:
                raise RuntimeError(
                    f"TTS chunk {index} produced no final audio: {actual_path}"
                )
            generated_artifacts.add(actual_path)
            encoded_paths.append(actual_path)
            chunk_results.append(chunk_result)

        voice_compatible = bool(chunk_results) and all(
            bool(result.get("voice_compatible")) for result in chunk_results
        )
        delivery_base = base_path.with_suffix(Path(encoded_paths[0]).suffix)
        final_paths, combined_chunks = await _build_audio_delivery_files(
            encoded_paths,
            str(delivery_base),
            delivery_profile,
            voice_compatible=voice_compatible,
        )
        for path in final_paths:
            file_size = (await aiofiles.os.stat(path)).st_size
            logger.info(
                "TTS audio saved: %s (%s bytes, provider: %s)",
                path,
                f"{file_size:,}",
                provider,
            )
        media_tag = "\n".join(f"MEDIA:{path}" for path in final_paths)
        if voice_compatible:
            media_tag = f"[[audio_as_voice]]\n{media_tag}"
        return json.dumps(
            {
                "success": True,
                "file_path": final_paths[0],
                "file_paths": final_paths,
                "media_tag": media_tag,
                "provider": chunk_results[0].get("provider", provider),
                "voice_compatible": voice_compatible,
                "chunk_count": len(chunks),
                "delivery_file_count": len(final_paths),
                "combined_chunks": bool(combined_chunks),
                "delivery_profile": {
                    "platform": delivery_profile.platform,
                    "max_file_bytes": delivery_profile.max_file_bytes,
                    "target_file_bytes": delivery_profile.target_file_bytes,
                },
            },
            ensure_ascii=False,
        )
    except asyncio.CancelledError:
        raise
    except ValueError as exc:
        error_msg = f"TTS delivery error ({provider}): {exc}"
        logger.error("%s", error_msg)
        return tool_error(error_msg, success=False)
    except Exception as exc:
        error_msg = f"TTS long-form generation failed ({provider}): {exc}"
        logger.error("%s", error_msg, exc_info=True)
        return tool_error(error_msg, success=False)
    finally:
        final_absolute = {os.path.abspath(path) for path in final_paths}
        for artifact in generated_artifacts:
            if os.path.abspath(artifact) in final_absolute:
                continue
            try:
                await aiofiles.os.remove(artifact)
            except OSError:
                pass


# ===========================================================================
# Requirements check
# ===========================================================================
async def check_tts_requirements() -> bool:
    """Return whether the explicitly resolved TTS provider can run.

    Availability must mirror :func:`text_to_speech_tool` dispatch. Unrelated
    cloud credentials do not make the default Edge backend usable, and an
    explicitly selected backend is checked on its own requirements.
    """
    tts_config = await _load_tts_config()
    provider = _get_provider(tts_config)
    command_config = _resolve_command_provider_config(provider, tts_config)
    if command_config is not None:
        return True

    if provider == "edge":
        try:
            _import_edge_tts()
            return True
        except ImportError:
            return await _check_neutts_available()
    if provider == "elevenlabs":
        try:
            _import_elevenlabs()
        except ImportError:
            return False
        return bool(
            await _resolve_provider_key("ELEVENLABS_API_KEY", "elevenlabs")
        )
    if provider == "openai":
        try:
            _import_openai_client()
        except ImportError:
            return False
        return await _has_openai_audio_backend()
    if provider == "deepinfra":
        try:
            _import_openai_client()
        except ImportError:
            return False
        return bool(
            await _resolve_provider_key("DEEPINFRA_API_KEY", "deepinfra")
        )
    if provider == "minimax":
        try:
            await _resolve_minimax_tts_runtime(tts_config)
        except ValueError:
            return False
        return True
    if provider == "xai":
        try:
            from tools.xai_http import resolve_xai_http_credentials

            return bool(
                (await resolve_xai_http_credentials()).get("api_key")
            )
        except Exception:
            return False
    if provider == "gemini":
        return bool(
            await _resolve_provider_key("GEMINI_API_KEY", "gemini")
            or await _resolve_provider_key("GOOGLE_API_KEY", "gemini")
        )
    if provider == "mistral":
        try:
            _import_mistral_client()
        except ImportError:
            return False
        return bool(
            await _resolve_provider_key("MISTRAL_API_KEY", "mistral")
        )
    if provider == "neutts":
        return await _check_neutts_available()
    if provider == "kittentts":
        return await _check_kittentts_available()
    if provider == "piper":
        return await _check_piper_available()

    try:
        from agent.tts_registry import get_provider
        from hermes_cli.plugins import _ensure_plugins_discovered

        await _ensure_plugins_discovered()
        plugin = get_provider(provider)
        return bool(plugin and await plugin.is_available())
    except Exception:
        return False


async def _resolve_openai_audio_client_config() -> tuple[str, str, bool]:
    """Return ``(api_key, base_url, is_managed)`` for the OpenAI audio client.

    ``is_managed`` is True when the config resolves to the Nous managed audio
    gateway (a restricted proxy), so callers can coerce the request to what the
    gateway supports. When ``tts.use_gateway`` is set the gateway is preferred
    even if direct OpenAI credentials are present.

    Resolution order (mirrors the STT resolver):
    1. ``tts.openai.api_key`` / ``tts.openai.base_url`` from ``config.yaml``
    2. ``VOICE_TOOLS_OPENAI_KEY`` / ``OPENAI_API_KEY`` environment variables
       (still honoring ``tts.openai.base_url`` when set)
    3. Managed OpenAI audio tool gateway
    """
    tts_config = await _load_tts_config()
    openai_cfg = (tts_config.get("openai") if isinstance(tts_config, dict) else None) or {}
    cfg_api_key = openai_cfg.get("api_key") or ""
    cfg_base_url = openai_cfg.get("base_url") or ""
    if cfg_api_key and not await prefers_gateway("tts"):
        return cfg_api_key, (cfg_base_url or DEFAULT_OPENAI_BASE_URL), False

    direct_api_key = (
        await _resolve_provider_key("VOICE_TOOLS_OPENAI_KEY", "openai")
        or await _resolve_provider_key("OPENAI_API_KEY", "openai")
    )
    if direct_api_key and not await prefers_gateway("tts"):
        return direct_api_key, (cfg_base_url or DEFAULT_OPENAI_BASE_URL), False

    managed_gateway = await resolve_managed_tool_gateway("openai-audio")
    if managed_gateway is None:
        message = (
            "Neither tts.openai.api_key in config nor "
            "VOICE_TOOLS_OPENAI_KEY/OPENAI_API_KEY is set"
        )
        if await managed_nous_tools_enabled() or await prefers_gateway("tts"):
            message += (
                ". "
                + await nous_tool_gateway_unavailable_message(
                    "managed OpenAI audio for TTS",
                )
            )
        raise ValueError(message)

    return (
        managed_gateway.nous_user_token,
        urljoin(f"{managed_gateway.gateway_origin.rstrip('/')}/", "v1"),
        True,
    )


async def _has_openai_audio_backend() -> bool:
    """Return True when OpenAI audio can use config/env credentials or the managed gateway."""
    openai_cfg = ((await _load_tts_config()).get("openai") or {})
    if openai_cfg.get("api_key"):
        return True
    direct = (
        await _resolve_provider_key("VOICE_TOOLS_OPENAI_KEY", "openai")
        or await _resolve_provider_key("OPENAI_API_KEY", "openai")
    )
    return bool(direct or await resolve_managed_tool_gateway("openai-audio"))


# ===========================================================================
# Streaming TTS: sentence-by-sentence pipeline
# ===========================================================================
# Markdown stripping patterns (same as cli.py _voice_speak_response)
_MD_CODE_BLOCK = re.compile(r'```[\s\S]*?```')
_MD_LINK = re.compile(r'\[([^\]]+)\]\([^)]+\)')
_MD_URL = re.compile(r'https?://\S+')
_MD_BOLD = re.compile(r'\*\*(.+?)\*\*')
_MD_ITALIC = re.compile(r'\*(.+?)\*')
_MD_INLINE_CODE = re.compile(r'`(.+?)`')
_MD_HEADER = re.compile(r'^#+\s*', flags=re.MULTILINE)
_MD_LIST_ITEM = re.compile(r'^\s*[-*]\s+', flags=re.MULTILINE)
_MD_HR = re.compile(r'---+')
_MD_EXCESS_NL = re.compile(r'\n{3,}')
# Emoji + variation selectors/ZWJ — TTS providers render these as awkward
# pauses or literal descriptions ("smiling face"), breaking the speech flow.
_EMOJI = re.compile(
    '[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D\U000E0020-\U000E007F]+'
)

# Strip <think>...</think> reasoning blocks before TTS — models with
# /reasoning show enabled produce think blocks that shouldn't be spoken.
_THINK_BLOCK = re.compile(r'<think[\s>].*?</think>', flags=re.DOTALL)


def _strip_markdown_for_tts(text: str) -> str:
    """Prepare text for speech via the shared cleaner in tts_text_normalize.

    One cleaner for every TTS path (tool, gateway auto-TTS, voice-mode
    streaming, web dashboard): strips <think> reasoning blocks, the
    file-mutation verifier footer, markdown, and emoji; expands units and
    symbols; and flattens newlines to sentence breaks so newline-sensitive
    providers (Kokoro) speak the whole script.  Falls back to the legacy
    regex pipeline if the normalizer ever fails.
    """
    try:
        from tools.tts_text_normalize import prepare_spoken_text
        return prepare_spoken_text(text, max_chars=None)
    except Exception:
        pass
    text = _THINK_BLOCK.sub(' ', text)
    text = _MD_CODE_BLOCK.sub(' ', text)
    text = _MD_LINK.sub(r'\1', text)
    text = _MD_URL.sub('', text)
    text = _MD_BOLD.sub(r'\1', text)
    text = _MD_ITALIC.sub(r'\1', text)
    text = _MD_INLINE_CODE.sub(r'\1', text)
    text = _MD_HEADER.sub('', text)
    text = _MD_LIST_ITEM.sub('', text)
    text = _MD_HR.sub('', text)
    text = _EMOJI.sub(' ', text)
    text = _MD_EXCESS_NL.sub('\n\n', text)
    return text.strip()



# The upstream speaker-playback pipeline belongs to the removed CLI/gateway
# surface. The retained library exposes synthesis through text_to_speech_tool.

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error

TTS_SCHEMA = {
    "name": "text_to_speech",
    "description": "Convert text to speech audio. Returns a MEDIA: path that the platform delivers as native audio. Compatible providers render as a voice bubble on Telegram; otherwise audio is sent as a regular attachment. In CLI mode, saves to ~/voice-memos/. Voice and provider are user-configured (built-in providers like edge/openai or custom command providers under tts.providers.<name>), not model-selected.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to convert to speech. Provider-specific character caps apply and are enforced automatically (OpenAI 4096, xAI 15000, MiniMax 10000, ElevenLabs 5k-40k depending on model); over-long input is truncated."
            },
            "output_path": {
                "type": "string",
                "description": f"Optional custom file path to save the audio. Defaults to {display_hermes_home()}/audio_cache/<timestamp>.mp3"
            },
            "speed": {
                "type": "number",
                "description": "Playback speed multiplier. 1.0 = normal, 0.5 = very slow (language learning), 2.0 = fast. Range: 0.25-4.0. Overrides the speed configured in config.yaml."
            },
            "instructions": {
                "type": "string",
                "description": (
                    "Optional voice-design guidance: tone, emotion, pacing, accent, "
                    "whispering, impressions (e.g. 'Speak in a cheerful, excited whisper'). "
                    "Forwarded to the OpenAI backend (gpt-4o-mini-tts and OpenAI-compatible "
                    "voice-design servers). Silently ignored by backends that don't support it."
                )
            },
            "provider": {
                "type": "string",
                "description": (
                    "Optional TTS provider override. Accepts built-in names "
                    "(edge, openai, elevenlabs, minimax, xai, mistral, gemini, "
                    "neutts, kittentts, piper), user-declared command provider "
                    "names from tts.providers.<name>, or plugin-registered names. "
                    "When omitted, the configured tts.provider from config.yaml is used."
                )
            }
        },
        "required": ["text"]
    }
}

async def _handle_text_to_speech(args, **kw):
    return await text_to_speech_tool(
        text=args.get("text", ""),
        output_path=args.get("output_path"),
        speed=args.get("speed"),
        instructions=args.get("instructions"),
        provider=args.get("provider"),
    )


registry.register(
    name="text_to_speech",
    toolset="tts",
    schema=TTS_SCHEMA,
    handler=_handle_text_to_speech,
    check_fn=check_tts_requirements,
    emoji="🔊",
)
