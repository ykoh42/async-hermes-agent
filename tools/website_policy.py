"""Website access policy helpers for URL-capable tools.

This module loads a user-managed website blocklist from ~/.hermes/config.yaml
and optional shared list files. It is intentionally lightweight so web/browser
tools can enforce URL policy without pulling in the heavier CLI config stack.

Policy is cached in memory with a short TTL so config changes take effect
quickly without re-reading the file on every URL check.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiofiles
import aiofiles.os

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_DEFAULT_WEBSITE_BLOCKLIST = {
    "enabled": False,
    "domains": [],
    "shared_files": [],
}

# Cache: parsed policy + timestamp.  Avoids re-reading config.yaml on every
# URL check (a multi-URL extract with 50 pages would otherwise mean 51 YAML parses).
_CACHE_TTL_SECONDS = 30.0
_cached_policy: dict[str, Any] | None = None
_cached_policy_path: str | None = None
_cached_policy_time: float = 0.0
_policy_cache_by_path: dict[str, tuple[dict[str, Any], float]] = {}
_policy_path_aliases: dict[str, str] = {}
_policy_cache_guard = threading.RLock()


def _get_default_config_path() -> Path:
    return get_hermes_home() / "config.yaml"


class WebsitePolicyError(Exception):
    """Raised when a website policy file is malformed."""


def _normalize_host(host: str) -> str:
    return (host or "").strip().lower().rstrip(".")


def _normalize_rule(rule: Any) -> str | None:
    if not isinstance(rule, str):
        return None
    value = rule.strip().lower()
    if not value or value.startswith("#"):
        return None
    if "://" in value:
        parsed = urlparse(value)
        value = parsed.netloc or parsed.path
    value = value.split("/", 1)[0].strip().rstrip(".")
    if value.startswith("www."):
        value = value[4:]
    return value or None


def invalidate_cache() -> None:
    """Force the next ``check_website_access`` call to re-read config."""
    global _cached_policy, _cached_policy_path, _cached_policy_time
    with _policy_cache_guard:
        _policy_cache_by_path.clear()
        _cached_policy = None
        _cached_policy_path = None
        _cached_policy_time = 0.0


async def _canonical_policy_path(path: Path) -> str:
    """Resolve a loop-neutral physical cache key for one policy file."""
    lexical = os.path.normcase(os.fspath(path))
    with _policy_cache_guard:
        cached = _policy_path_aliases.get(lexical)
    if cached is not None:
        return cached
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
    with _policy_cache_guard:
        _policy_path_aliases[lexical] = canonical
    return canonical


def _publish_policy_cache(
    cache_key: str,
    display_path: str,
    policy: dict[str, Any],
    cached_at: float,
) -> None:
    """Store one profile policy and update historical private snapshots."""
    global _cached_policy, _cached_policy_path, _cached_policy_time
    with _policy_cache_guard:
        _policy_cache_by_path[cache_key] = (policy, cached_at)
        _cached_policy = policy
        _cached_policy_path = display_path
        _cached_policy_time = cached_at


def _match_host_against_rule(host: str, pattern: str) -> bool:
    if not host or not pattern:
        return False
    if pattern.startswith("*."):
        return fnmatch.fnmatch(host, pattern)
    return host == pattern or host.endswith(f".{pattern}")


def _extract_host_from_urlish(url: str) -> str:
    parsed = urlparse(url)
    host = _normalize_host(parsed.hostname or parsed.netloc)
    if host:
        return host

    if "://" not in url:
        schemeless = urlparse(f"//{url}")
        host = _normalize_host(schemeless.hostname or schemeless.netloc)
        if host:
            return host

    return ""


async def check_website_access(
    url: str,
    config_path: Path | None = None,
) -> dict[str, str] | None:
    """Async policy check for network-capable tool handlers.

    On a cold cache, YAML and optional shared blocklist files are read through
    ``aiofiles``; parsing/matching stays local and deterministic.  No async
    caller needs to invoke the synchronous policy loader or a thread fallback.
    """
    host = _extract_host_from_urlish(url)
    if not host:
        return None

    try:
        policy = await load_website_blocklist(config_path)
    except WebsitePolicyError as exc:
        if config_path is not None:
            raise
        logger.warning("Website policy config error (failing open): %s", exc)
        return None
    except Exception as exc:
        logger.warning(
            "Unexpected error loading website policy (failing open): %s", exc
        )
        return None

    if not policy.get("enabled"):
        return None
    for rule in policy.get("rules", []):
        pattern = rule.get("pattern", "")
        if _match_host_against_rule(host, pattern):
            logger.info(
                "Blocked URL %s — matched rule '%s' from %s",
                url,
                pattern,
                rule.get("source", "config"),
            )
            return {
                "url": url,
                "host": host,
                "rule": pattern,
                "source": rule.get("source", "config"),
                "message": (
                    f"Blocked by website policy: '{host}' matched rule '{pattern}'"
                    f" from {rule.get('source', 'config')}"
                ),
            }
    return None


async def load_website_blocklist(config_path: Path | None = None) -> dict[str, Any]:
    """Load the website blocklist through native async file I/O."""
    config_path = config_path or _get_default_config_path()
    resolved_path = str(config_path)
    cache_key: str | None = None
    if config_path == _get_default_config_path():
        cache_key = await _canonical_policy_path(config_path)
        with _policy_cache_guard:
            cached = _policy_cache_by_path.get(cache_key)
        if cached is not None:
            cached_policy, cached_at = cached
            if (time.monotonic() - cached_at) < _CACHE_TTL_SECONDS:
                return cached_policy
    if not await aiofiles.os.path.exists(config_path):
        policy = dict(_DEFAULT_WEBSITE_BLOCKLIST)
    else:
        try:
            import yaml
        except ImportError:
            return dict(_DEFAULT_WEBSITE_BLOCKLIST)
        try:
            async with aiofiles.open(config_path, encoding="utf-8") as handle:
                raw_config = await handle.read()
        except FileNotFoundError:
            policy = dict(_DEFAULT_WEBSITE_BLOCKLIST)
            raw_config = None
        except (OSError, UnicodeDecodeError) as exc:
            raise WebsitePolicyError(
                f"Failed to read config file {config_path}: {exc}"
            ) from exc
        if raw_config is None:
            config = {}
        else:
            try:
                config = yaml.safe_load(raw_config) or {}
            except yaml.YAMLError as exc:
                raise WebsitePolicyError(f"Invalid config YAML at {config_path}: {exc}") from exc
        if not isinstance(config, dict):
            raise WebsitePolicyError("config root must be a mapping")
        security = config.get("security") or {}
        if not isinstance(security, dict):
            raise WebsitePolicyError("security must be a mapping")
        blocklist = security.get("website_blocklist") or {}
        if not isinstance(blocklist, dict):
            raise WebsitePolicyError("security.website_blocklist must be a mapping")
        policy = dict(_DEFAULT_WEBSITE_BLOCKLIST)
        policy.update(blocklist)

    raw_domains = policy.get("domains", []) or []
    if not isinstance(raw_domains, list):
        raise WebsitePolicyError("security.website_blocklist.domains must be a list")
    raw_shared_files = policy.get("shared_files", []) or []
    if not isinstance(raw_shared_files, list):
        raise WebsitePolicyError("security.website_blocklist.shared_files must be a list")
    enabled = policy.get("enabled", True)
    if not isinstance(enabled, bool):
        raise WebsitePolicyError("security.website_blocklist.enabled must be a boolean")

    rules: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_rule in raw_domains:
        normalized = _normalize_rule(raw_rule)
        if normalized and ("config", normalized) not in seen:
            rules.append({"pattern": normalized, "source": "config"})
            seen.add(("config", normalized))
    for shared_file in raw_shared_files:
        if not isinstance(shared_file, str) or not shared_file.strip():
            continue
        path = Path(shared_file).expanduser()
        if not path.is_absolute():
            path = get_hermes_home() / path
        try:
            async with aiofiles.open(path, encoding="utf-8") as handle:
                raw_rules = (await handle.read()).splitlines()
        except FileNotFoundError:
            logger.warning("Shared blocklist file not found (skipping): %s", path)
            raw_rules = []
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Failed to read shared blocklist file %s (skipping): %s", path, exc)
            raw_rules = []
        for line in raw_rules:
            normalized = _normalize_rule(line)
            key = (str(path), normalized) if normalized else None
            if normalized and key not in seen:
                rules.append({"pattern": normalized, "source": str(path)})
                seen.add(key)

    result = {"enabled": enabled, "rules": rules}
    if cache_key is not None:
        _publish_policy_cache(
            cache_key,
            resolved_path,
            result,
            time.monotonic(),
        )
    return result
