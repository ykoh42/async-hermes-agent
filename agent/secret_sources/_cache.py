"""Shared substrate for external secret-source backends.

Every backend (Bitwarden, 1Password, …) needs the same handful of
security-sensitive primitives:

  * a uniform result object (:class:`FetchResult`),
  * environment-variable name validation (:func:`is_valid_env_name`),
  * a two-layer fetch cache whose disk half writes atomically with ``0600``
    permissions and honours a TTL (:class:`DiskCache`, :class:`CachedFetch`).

These used to live inline inside ``bitwarden.py``.  Pulling them here means
the atomic-write / ``0600`` / TTL logic is audited and fixed in exactly one
place instead of drifting across copy-pasted per-backend modules — each
backend supplies only its own cache-key shape and a serializer for it.

Nothing in this module ever raises out to the caller's hot path: the disk
layer is strictly best-effort (a miss just triggers a refetch), because a
cache problem must never block Hermes startup.
"""

from __future__ import annotations

import aiofiles
import aiofiles.os
import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar
from collections.abc import Callable

__all__ = [
    "FetchResult",
    "CachedFetch",
    "DiskCache",
    "is_valid_env_name",
    "resolve_cache_home",
]


# ---------------------------------------------------------------------------
# Result object + env-name validation — canonical definitions live in
# ``agent.secret_sources.base`` (the SecretSource contract module); re-exported
# here so backends that import from ``_cache`` keep working.
# ---------------------------------------------------------------------------

from agent.secret_sources.base import (  # noqa: E402
    FetchResult,
    is_valid_env_name,
)


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


@dataclass
class CachedFetch:
    """A set of fetched secret values plus when they were fetched."""

    secrets: dict[str, str]
    fetched_at: float

    def is_fresh(self, ttl_seconds: float) -> bool:
        if ttl_seconds <= 0:
            return False
        return (time.time() - self.fetched_at) < ttl_seconds


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------


def resolve_cache_home(home_path: Path | None = None) -> Path:
    """Resolve the Hermes home used for cache paths.

    ``home_path`` is whatever ``load_hermes_dotenv()`` already resolved;
    falling back to ``$HERMES_HOME`` / ``~/.hermes`` keeps direct callers
    (and tests that don't thread a home through) working.
    """
    if home_path is None:
        from hermes_constants import get_hermes_home

        home_path = get_hermes_home()
    return home_path


async def _canonical_cache_home(
    home_path: Path | None = None,
) -> tuple[str, Path]:
    """Return the physical profile identity and its cache path.

    L1 caches need the physical path in their key so direct callers that omit
    ``home_path`` and callers using a symlink alias cannot share or duplicate
    another profile's secret values.  Resolve it at the already-awaited cache
    boundary rather than doing filesystem work in ``resolve_cache_home()``.
    """
    resolved = resolve_cache_home(home_path)
    realpath = aiofiles.os.wrap(os.path.realpath)
    canonical = os.path.normcase(str(await realpath(str(resolved))))
    return canonical, Path(canonical)


K = TypeVar("K")


async def _remove_owned_cache_temp(path: Path) -> None:
    try:
        await aiofiles.os.remove(path)
    except OSError:
        pass


async def _finish_owned_cache_cleanup(
    task: asyncio.Task[None],
) -> asyncio.CancelledError | None:
    """Finish an owned temp-file deletion through repeated cancellation."""
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            await asyncio.shield(task)
            return cancellation
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - caller re-raises
            if task.cancelled():
                raise
            if cancellation is None:
                cancellation = exc


class DiskCache(Generic[K]):
    """Best-effort, profile-aware on-disk cache for fetched secret values.

    One JSON object per backend lives at ``<hermes_home>/cache/<basename>``::

        {"key": "<serialized cache key>", "secrets": {...}, "fetched_at": 1.0}

    The file holds only secret *values* keyed by the serialized cache key —
    never raw auth material.  Backends are responsible for fingerprinting
    tokens/sessions *before* they reach ``key_serializer`` so the token can't
    land in the key.

    Writes are atomic (``mkstemp`` → ``chmod 0600`` → ``os.replace``) and the
    containing ``cache/`` directory is forced to ``0700`` — ``mkdir``'s mode is
    umask-subject, so the chmod is the reliable form.  Both ``read`` and
    ``write`` short-circuit when ``ttl_seconds <= 0``, so setting the TTL to
    zero disables *both* cache layers symmetrically: a user opting out never
    gets secret values written to disk at all.
    """

    def __init__(self, basename: str, *, key_serializer: Callable[[K], str]) -> None:
        self._basename = basename
        self._key_serializer = key_serializer
        # Temp-file prefix derived from the basename so concurrent writers for
        # different backends in the same dir don't collide on the staging name.
        stem = basename.split(".", 1)[0]
        self._tmp_prefix = f".{stem}_"

    def path(self, home_path: Path | None = None) -> Path:
        return resolve_cache_home(home_path) / "cache" / self._basename

    async def read(
        self,
        key: K,
        ttl_seconds: float,
        home_path: Path | None = None,
    ) -> CachedFetch | None:
        """Return a fresh cached entry for ``key``, or None.

        Best-effort: any I/O or parse error, a key mismatch, or a stale entry
        all return None so the caller re-fetches.
        """
        if ttl_seconds <= 0:
            return None
        path = self.path(home_path)
        try:
            async with aiofiles.open(path, encoding="utf-8") as file:
                payload = json.loads(await file.read())
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("key") != self._key_serializer(key):
            return None
        secrets = payload.get("secrets")
        fetched_at = payload.get("fetched_at")
        if not isinstance(secrets, dict) or not isinstance(fetched_at, (int, float)):
            return None
        # JSON permits non-string values; env vars need strings, so coerce by
        # dropping anything that isn't a str→str pair.
        typed: dict[str, str] = {
            k: v
            for k, v in secrets.items()
            if isinstance(k, str) and isinstance(v, str)
        }
        entry = CachedFetch(secrets=typed, fetched_at=float(fetched_at))
        if not entry.is_fresh(ttl_seconds):
            return None
        return entry

    async def write(
        self,
        key: K,
        entry: CachedFetch,
        ttl_seconds: float,
        home_path: Path | None = None,
    ) -> None:
        """Persist ``entry`` for ``key`` atomically at mode ``0600``.

        No-op when ``ttl_seconds <= 0`` (so caching is genuinely off) or on any
        I/O error — the next invocation just re-fetches.
        """
        if ttl_seconds <= 0:
            return
        path = self.path(home_path)
        try:
            cache_dir = path.parent
            await aiofiles.os.makedirs(cache_dir, exist_ok=True)
            # mkdir's mode is umask-subject; chmod the dir to 0700 so cache
            # metadata isn't exposed if HERMES_HOME is ever made traversable.
            try:
                await aiofiles.os.wrap(os.chmod)(cache_dir, 0o700)
            except OSError:
                pass
            payload = {
                "key": self._key_serializer(key),
                "secrets": entry.secrets,
                "fetched_at": entry.fetched_at,
            }
            # Write to a unique sibling and atomically replace the cache file.
            # Explicit chmod keeps the file private regardless of umask.
            tmp = cache_dir / f"{self._tmp_prefix}{uuid.uuid4().hex}.tmp"
            try:
                async with aiofiles.open(tmp, "x", encoding="utf-8") as file:
                    await aiofiles.os.wrap(os.chmod)(tmp, 0o600)
                    await file.write(json.dumps(payload))
                    await file.flush()
                await aiofiles.os.replace(tmp, path)
            except asyncio.CancelledError:
                cleanup = asyncio.create_task(
                    _remove_owned_cache_temp(tmp),
                    name="secret-cache-staging-cleanup",
                )
                await _finish_owned_cache_cleanup(cleanup)
                raise
            except BaseException as error:
                cleanup = asyncio.create_task(
                    _remove_owned_cache_temp(tmp),
                    name="secret-cache-staging-cleanup",
                )
                cancellation = await _finish_owned_cache_cleanup(cleanup)  # noqa: ASYNC120 - cleanup shields cancellation
                if cancellation is not None:
                    raise cancellation from error
                raise
        except OSError:
            pass  # best-effort — a disk-cache miss next invocation is fine

    async def clear(self, home_path: Path | None = None) -> None:
        """Delete the on-disk cache file if present (idempotent)."""
        try:
            await aiofiles.os.remove(self.path(home_path))
        except (FileNotFoundError, OSError):
            pass
