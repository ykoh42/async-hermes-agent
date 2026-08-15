"""Single resolver for every media source -> bytes + mime.

All source handling (data:/http(s)/file/local/container) funnels through
:func:`resolve_image_source` so size and magic-byte checks are enforced exactly
once.  Returns raw bytes (not a path): the downstream step is base64 -> data URL
(RFC 2397) and provider base64 content blocks.

Images are the default and the historical purpose. Callers whose argument
takes video opt in via ``permitted=("video",)`` — the same confinement and
credential-guard pipeline applies, and only the type check at the end differs
(extension-table typing plus an mp4 magic sniff, rather than image magic
bytes). Every existing call site keeps the image-only default unchanged.

Local paths are resolved and read through the same awaited filesystem boundary
as file tools. Non-local backends use their native async ``execute`` boundary
for paths outside the explicitly permitted Hermes media caches; a backend that
only exposes synchronous execution fails closed rather than blocking the loop.
"""
from __future__ import annotations

import base64
import asyncio
import inspect
import os
import re
import aiofiles
import aiofiles.os
import aiofiles.tempfile
from dataclasses import dataclass
from pathlib import Path

# Raw-bytes INGEST budget — what the resolver will load before handing off.
# This is deliberately the 50MB download cap (tools/vision_tools._VISION_MAX_DOWNLOAD_BYTES),
# NOT the 20MB provider payload cap. The 20MB cap (_MAX_BASE64_BYTES) is a
# *post-resize* limit enforced at the call sites: an oversized raw image must
# still reach the resizer so it can be downscaled under the payload cap. Capping
# raw bytes at 20MB here would reject every 20-50MB photo before resize can run.
_MAX_INGEST_BYTES = 50 * 1024 * 1024
_realpath = aiofiles.os.wrap(os.path.realpath)


class ImageResolutionError(Exception):
    def __init__(self, message: str, *, src: str = "", origin: str = ""):
        super().__init__(message)
        self.src, self.origin = src, origin


class UnsupportedScheme(ImageResolutionError):
    pass


class SourceUnsafe(ImageResolutionError):  # SSRF / path-allowlist
    pass


class SourceTooLarge(ImageResolutionError):
    pass


class SourceNotFound(ImageResolutionError):
    pass


class NotAnImage(ImageResolutionError):
    pass


@dataclass
class ResolveContext:
    task_id: str | None = None


@dataclass
class ResolvedImage:
    data: bytes
    mime: str
    origin: str  # one of: data | http | file | local | container


# Explicit URL scheme, e.g. "ftp://", "s3://". Bare Windows drive paths
# ("C:\x.png") don't match because they lack the "//".
_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")


async def resolve_image_source(
    src: str,
    ctx: ResolveContext,
    *,
    permitted: tuple = ("image",),
) -> ResolvedImage:
    if not isinstance(src, str) or not src.strip():
        raise SourceNotFound("image_url is required", src=str(src))
    s = src.strip()
    if s.startswith("data:"):
        data, mime = _resolve_data_url(s)
        return _finalize(data, mime, "data", s, permitted)
    if s.startswith(("http://", "https://")):
        reason = await _http_block_reason(s)
        if reason:
            raise SourceUnsafe(reason, src=s)
        return _finalize(await _download_to_bytes(s), "", "http", s, permitted)

    if _SCHEME_RE.match(s) and not s.lower().startswith("file://"):
        raise UnsupportedScheme(
            "Unrecognized image source scheme. Use an http(s) URL, a local "
            "file path, a file:// URI, or a data: URL.",
            src=s,
        )

    # Everything else is a filesystem path — including bare relative names
    # like "pic.png" (accepted on main; a path-shape gate here regressed them).
    candidate = s[len("file://"):] if s.lower().startswith("file://") else s
    p = Path(os.path.expanduser(candidate))
    host_target = await _permitted_host_read_target(p, ctx)
    if (
        host_target is not None
        and await aiofiles.os.path.isfile(host_target)
    ):
        # Shared credential-read guard (agent.file_safety, #57698): refuse
        # secret-bearing files (.env, auth.json, ...) with an intentional,
        # specific error instead of relying on the magic-byte sniff to
        # reject them incidentally. Same chokepoint the image-gen/video-gen
        # provider plugins enforce on model-supplied local paths. Import is
        # best-effort (guard unavailability must not break image loading);
        # a real block always propagates.
        try:
            from agent.file_safety import raise_if_read_blocked
        except Exception:  # noqa: BLE001 — guard unavailable: proceed
            raise_if_read_blocked = None
        if raise_if_read_blocked is not None:
            try:
                await raise_if_read_blocked(str(host_target))
            except ValueError as exc:
                raise SourceUnsafe(str(exc), src=s, origin="file")
        async with aiofiles.open(host_target, "rb") as image_file:
            data = await image_file.read()
        return _finalize(data, "", "file", s, permitted)
    if _is_local_terminal_backend():
        raise SourceNotFound(f"media file not found: '{p}'", src=s, origin="file")
    return await _resolve_container_fallback(p, ctx, s, permitted)


def _resolve_data_url(s: str) -> tuple[bytes, str]:
    header, _, payload = s.partition(",")
    if ";base64" not in header:
        raise NotAnImage("data: URL must be base64-encoded", src=s[:64])
    declared = header[len("data:"):].split(";", 1)[0].strip() or "application/octet-stream"
    # Cheap pre-decode size gate on the encoded length (~4/3 expansion).
    if (len(payload) * 3) // 4 > _MAX_INGEST_BYTES:
        raise SourceTooLarge("data: URL exceeds size limit", src=s[:64])
    try:
        data = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise NotAnImage(f"invalid base64 in data: URL: {exc}", src=s[:64])
    return data, declared  # real mime verified in _finalize via magic bytes


async def _http_block_reason(url: str) -> str | None:
    """Return a human-readable block reason, or None when the URL is allowed.

    Pre-flight short-circuit: policy-blocked URLs are refused BEFORE any
    network I/O. ``_download_image`` re-checks policy internally (per attempt
    and against the final redirect target) — that second evaluation is
    intentional, not redundant: this one guarantees no bytes move for a
    blocked URL; the inner one covers redirects and non-resolver callers.
    Preserves the specific website-policy message so the agent sees *why*.
    """
    from tools.url_safety import is_safe_url
    from tools.website_policy import check_website_access

    if not await is_safe_url(url):
        return "blocked: unsafe or private URL"
    blocked = await check_website_access(url)
    if blocked:
        return blocked.get("message") or "blocked by website policy"
    return None


async def _download_to_bytes(url: str) -> bytes:
    from tools.vision_tools import _download_image

    async with aiofiles.tempfile.NamedTemporaryFile(
        suffix=".img", delete=False
    ) as tf:
        tmp = Path(tf.name)
    try:
        # Enforces the 50MB stream cap, redirect SSRF guard, and website policy.
        await _download_image(url, tmp)
        async with aiofiles.open(tmp, "rb") as image_file:
            return await image_file.read()
    except PermissionError as exc:  # website policy block
        raise SourceUnsafe(str(exc), src=url, origin="http")
    finally:
        try:
            await aiofiles.os.remove(tmp)
        except FileNotFoundError:
            pass


async def _permitted_host_read_target(
    path: Path,
    _context: ResolveContext,
) -> Path | None:
    """Return a safe host-side media path, or None for sandbox reads.

    The local backend may read arbitrary host paths.  Non-local backends may
    only read Hermes-managed media caches on the host; every other path must
    go through the active sandbox's native async ``execute`` method.
    """
    from agent.secret_scope import get_secret

    backend = str(get_secret("TERMINAL_ENV", "local") or "local").strip().lower()
    try:
        real = Path(await _realpath(path))
    except OSError:
        return path if backend in {"", "local"} else None
    if backend in {"", "local"}:
        return real
    from tools.credential_files import from_agent_visible_cache_path

    try:
        real = Path(await _realpath(Path(from_agent_visible_cache_path(str(path)))))
    except OSError:
        return None
    for root in _media_cache_roots():
        try:
            real.relative_to(Path(await _realpath(root)))
            return real
        except (OSError, ValueError):
            continue
    return None


def _is_local_terminal_backend() -> bool:
    from agent.secret_scope import get_secret

    return str(get_secret("TERMINAL_ENV", "local") or "local").strip().lower() in {
        "",
        "local",
    }


def _media_cache_roots() -> list[Path]:
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    return [
        home / "cache",
        home / "images",
        home / "image_cache",
        home / "audio_cache",
        home / "video_cache",
        home / "temp_vision_images",
        home / "temp_video_files",
    ]


def _get_active_env(task_id: str | None):
    if not task_id:
        return None
    try:
        from tools.terminal_tool import get_active_env

        return get_active_env(task_id)
    except Exception:
        return None


async def _ensure_container_env(task_id: str | None) -> None:
    if not task_id:
        return
    try:
        from tools.terminal_tool import ensure_task_env

        if not inspect.iscoroutinefunction(ensure_task_env):
            return
        result = ensure_task_env(task_id)
        if inspect.isawaitable(result):
            await result
    except asyncio.CancelledError:
        raise
    except Exception:
        # The subsequent active-env check is fail-closed and reports the
        # actionable sandbox-unavailable error to the caller.
        return


async def _resolve_container_fallback(
    path: Path,
    ctx: ResolveContext,
    src: str,
    permitted: tuple = ("image",),
) -> ResolvedImage:
    """Read a non-cache path inside the active sandbox using native async I/O."""
    await _ensure_container_env(ctx.task_id)
    env = _get_active_env(ctx.task_id)
    if env is None:
        raise SourceNotFound(
            f"'{path}' is not reachable inside the sandbox and no active "
            "sandbox session is available to read it",
            src=src,
            origin="container",
        )
    import shlex

    execute = getattr(env, "execute", None)
    if not inspect.iscoroutinefunction(execute):
        raise RuntimeError(
            "The configured sandbox backend does not expose a native async "
            "execute method for media reads."
        )
    command = f"head -c {_MAX_INGEST_BYTES + 1} < {shlex.quote(str(path))} | base64 | tr -d '\\n'"
    last_result: dict = {"returncode": 1, "output": ""}
    for attempt in range(2):
        last_result = await execute(command, timeout=30, bounded_capture=True)
        if last_result.get("returncode", 1) == 0:
            break
        if attempt == 0:
            await asyncio.sleep(0.15)
    if last_result.get("returncode", 1) != 0:
        diagnostic = next(
            (line.strip() for line in str(last_result.get("output") or "").splitlines() if line.strip()),
            "",
        )
        suffix = f" ({diagnostic[:200]})" if diagnostic else ""
        raise SourceNotFound(
            f"could not read '{path}' inside the sandbox{suffix}",
            src=src,
            origin="container",
        )
    try:
        data = base64.b64decode(last_result.get("output", ""), validate=True)
    except Exception as exc:
        raise NotAnImage(
            f"sandbox returned non-image data for '{path}': {exc}",
            src=src,
            origin="container",
        ) from exc
    if len(data) > _MAX_INGEST_BYTES:
        raise SourceTooLarge("media exceeds size limit", src=src, origin="container")
    return _finalize(data, "", "container", src, permitted)


def _finalize(
    data: bytes, declared_mime: str, origin: str, src: str, permitted: tuple = ("image",)
) -> ResolvedImage:
    """Intrinsic-correctness chokepoint: ingest byte cap + type check.

    The cap here is the generous 50MB *ingest* budget, not the 20MB provider
    payload cap — a 20-50MB image must survive this step so the call site can
    resize it under the payload cap. See ``_MAX_INGEST_BYTES``.

    Images are typed by magic bytes. Video (opt-in via ``permitted``) is typed
    by the extension table plus an mp4 container sniff: extension typing is
    sufficient because every downstream consumer re-validates — the upload
    gateway signs the content type into its presigned URL and the vendor
    rejects undecodable input — so a wrong guess is a clean rejection there
    rather than a hole here.
    """
    from tools.vision_tools import _detect_image_mime_type_from_bytes

    if len(data) > _MAX_INGEST_BYTES:
        raise SourceTooLarge("media exceeds size limit", src=src, origin=origin)

    sniffed = _detect_image_mime_type_from_bytes(data)
    if sniffed is not None:
        if "image" not in permitted:
            raise NotAnImage("source is an image, but this argument takes a video", src=src, origin=origin)
        return ResolvedImage(data=data, mime=sniffed, origin=origin)

    if "image" in permitted and b"<svg" in data[:4096].lower():
        # Pass SVG through — the vision call sites rasterize it to PNG
        # via _normalize_to_supported_image before embedding (providers
        # only ingest raster images).
        return ResolvedImage(data=data, mime="image/svg+xml", origin=origin)

    if "video" in permitted:
        video_mime = _detect_video_mime(data, src)
        if video_mime is not None:
            return ResolvedImage(data=data, mime=video_mime, origin=origin)
        raise NotAnImage("source is not a recognized video (mp4 expected)", src=src, origin=origin)

    raise NotAnImage("source is not a recognized image", src=src, origin=origin)


def _detect_video_mime(data: bytes, src: str) -> str | None:
    """Video MIME from the extension table, else the mp4/mov container magic.

    The magic fallback covers extensionless sources (data: URLs, URLs with
    query strings): ISO base-media files carry ``ftyp`` at offset 4.
    """
    from urllib.parse import urlsplit

    from tools.vision_tools import _detect_video_mime_type

    path_part = urlsplit(src).path if _SCHEME_RE.match(src) else src
    by_extension = _detect_video_mime_type(Path(path_part))
    if by_extension is not None:
        return by_extension
    if len(data) > 12 and data[4:8] == b"ftyp":
        return "video/mp4"
    return None


async def resolve_local_source_to_data_url(
    src: str,
    task_id: str | None,
    *,
    permitted: tuple = ("image",),
) -> str:
    """Resolve a path-like source through the native confinement boundary.

    URL and data sources are already self-contained and pass through exactly;
    filesystem paths are read from the active local/cache/sandbox boundary and
    returned as a provider-safe data URL. Generation tools use this helper so
    provider plugins never receive an unconstrained host path.
    """
    value = (src or "").strip()
    if not value or value.lower().startswith(("http://", "https://", "data:")):
        return src
    resolved = await resolve_image_source(
        value,
        ResolveContext(task_id=task_id),
        permitted=permitted,
    )
    encoded = base64.b64encode(resolved.data).decode("ascii")
    return f"data:{resolved.mime or 'application/octet-stream'};base64,{encoded}"
