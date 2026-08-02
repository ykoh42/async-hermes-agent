"""Small compatibility boundary retained for the lean agent runtime.

The messaging adapters and gateway runner are intentionally excluded.  A few
core paths still share media-cache and network-safety helpers through this
historical module path, so those helpers stay here until the async service
runtime owns them directly.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import uuid
from pathlib import Path

import aiofiles
import aiofiles.os

from hermes_constants import get_hermes_home


GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE = (
    "Interactive secret capture is unavailable in this runtime. "
    "Load this skill in the local CLI to be prompted, or configure the "
    "secret before starting the agent."
)

_MAX_CACHE_BYTES = 128 * 1024 * 1024
_SAFE_EXTENSION = re.compile(r"^\.[A-Za-z0-9]{1,12}$")


def is_network_accessible(host: str) -> bool:
    """Return whether *host* is not a loopback-only listener address."""
    value = str(host or "").strip().strip("[]")
    if not value or value.lower() == "localhost":
        return False
    try:
        return not ipaddress.ip_address(value).is_loopback
    except ValueError:
        # A hostname may resolve to both loopback and public addresses.  It is
        # safer to treat it as externally reachable for startup auditing.
        return True


async def _cache_dir(kind: str) -> Path:
    path = get_hermes_home() / "cache" / kind
    await aiofiles.os.makedirs(path, exist_ok=True)
    return path


def _safe_extension(value: str, fallback: str) -> str:
    extension = str(value or fallback).strip().lower()
    return extension if _SAFE_EXTENSION.fullmatch(extension) else fallback


def _validate_size(data: bytes, media_type: str) -> None:
    if len(data) > _MAX_CACHE_BYTES:
        raise ValueError(
            f"{media_type} payload is too large ({len(data)} bytes > {_MAX_CACHE_BYTES})"
        )


def _looks_like_image(data: bytes) -> bool:
    return bool(
        data[:8] == b"\x89PNG\r\n\x1a\n"
        or data[:3] == b"\xff\xd8\xff"
        or data[:6] in {b"GIF87a", b"GIF89a"}
        or data[:2] == b"BM"
        or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")
    )


async def _write_cached(
    kind: str, prefix: str, data: bytes, extension: str
) -> str:
    cache_dir = await _cache_dir(kind)
    path = cache_dir / f"{prefix}_{uuid.uuid4().hex[:12]}{extension}"
    async with aiofiles.open(path, "wb") as handle:
        await handle.write(data)
    return str(path)


async def cache_image_from_bytes(data: bytes, ext: str = ".jpg") -> str:
    """Persist validated image bytes for MCP/image-tool consumption."""
    _validate_size(data, "image")
    if not _looks_like_image(data):
        raise ValueError("Refusing to cache non-image bytes as an image")
    return await _write_cached("images", "img", data, _safe_extension(ext, ".jpg"))


async def cache_audio_from_bytes(data: bytes, ext: str = ".ogg") -> str:
    """Persist audio bytes for MCP/tool consumption."""
    _validate_size(data, "audio")
    return await _write_cached("audio", "audio", data, _safe_extension(ext, ".ogg"))


async def cache_document_from_bytes(data: bytes, filename: str) -> str:
    """Persist an MCP resource without allowing path traversal."""
    _validate_size(data, "document")
    name = Path(str(filename or "document")).name.replace("\x00", "").strip()
    if not name or name in {".", ".."}:
        name = "document"
    target_dir = await _cache_dir("documents")
    path = target_dir / f"doc_{uuid.uuid4().hex[:12]}_{name}"
    if target_dir.resolve() not in path.resolve().parents:
        raise ValueError(f"Path traversal rejected: {filename!r}")
    async with aiofiles.open(path, "wb") as handle:
        await handle.write(data)
    return str(path)


async def _cleanup_cache(kind: str, max_age_hours: int = 24) -> int:
    """Delete stale local cache entries; used only by explicit maintenance."""
    import time

    cutoff = time.time() - max(0, int(max_age_hours)) * 3600
    removed = 0
    cache_dir = await _cache_dir(kind)
    for entry in await aiofiles.os.scandir(cache_dir):
        try:
            if entry.is_file() and (await aiofiles.os.stat(entry.path)).st_mtime < cutoff:
                await aiofiles.os.remove(entry.path)
                removed += 1
        except OSError:
            continue
    return removed


async def cleanup_image_cache(max_age_hours: int = 24) -> int:
    return await _cleanup_cache("images", max_age_hours)


async def cleanup_audio_cache(max_age_hours: int = 24) -> int:
    return await _cleanup_cache("audio", max_age_hours)


async def cleanup_document_cache(max_age_hours: int = 24) -> int:
    return await _cleanup_cache("documents", max_age_hours)


def resolve_proxy_url(*_args, **_kwargs) -> str | None:
    """Compatibility placeholder; transport clients own proxy setup now."""
    return None


def proxy_kwargs_for_aiohttp(*_args, **_kwargs) -> tuple[dict, dict]:
    """Compatibility placeholder for removed messaging transports."""
    return {}, {}


def utf16_len(value: str) -> int:
    """Return UTF-16 code-unit length for callers preserving API semantics."""
    return len(str(value).encode("utf-16-le")) // 2


def can_resolve_host(host: str) -> bool:
    """Best-effort DNS probe retained for diagnostics."""
    try:
        socket.getaddrinfo(host, None)
        return True
    except OSError:
        return False


__all__ = [
    "GATEWAY_SECRET_CAPTURE_UNSUPPORTED_MESSAGE",
    "cache_audio_from_bytes",
    "cache_document_from_bytes",
    "cache_image_from_bytes",
    "cleanup_audio_cache",
    "cleanup_document_cache",
    "cleanup_image_cache",
    "is_network_accessible",
    "proxy_kwargs_for_aiohttp",
    "resolve_proxy_url",
    "utf16_len",
]
