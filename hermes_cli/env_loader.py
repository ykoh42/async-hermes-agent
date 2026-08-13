"""Native async loading for Hermes dotenv files."""

from __future__ import annotations

import asyncio
import codecs
import io
import logging
import os
import sys
import threading
import weakref
from pathlib import Path

import aiofiles
import aiofiles.os
from dotenv import dotenv_values


_CREDENTIAL_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_KEY")
_WARNED_KEYS: set[str] = set()
_WARNED_UTF32_PATHS: set[Path] = set()
_SECRET_SOURCES: dict[str, str] = {}
_SECRET_SOURCES_BY_HOME: dict[str, dict[str, str]] = {}
_SECRET_SOURCE_VALUES_BY_HOME: dict[str, dict[str, str]] = {}
_SECRET_SOURCE_HOME_ALIASES: dict[str, str] = {}
_APPLIED_HOMES: set[str] = set()
_SECRET_SOURCE_STATE_GUARD = threading.RLock()


class _SecretSourceCacheLocks:
    """Per-loop/profile locks without retaining closed event loops."""

    def __init__(self) -> None:
        self._locks: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop,
            dict[str, weakref.ReferenceType[asyncio.Lock]],
        ] = weakref.WeakKeyDictionary()

    def for_home(self, home_key: str) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        with _SECRET_SOURCE_STATE_GUARD:
            locks = self._locks.setdefault(loop, {})
            lock_ref = locks.get(home_key)
            lock = lock_ref() if lock_ref is not None else None
            if lock is None:
                lock = asyncio.Lock()
                locks[home_key] = weakref.ref(lock)
            return lock

    def clear(self) -> None:
        with _SECRET_SOURCE_STATE_GUARD:
            self._locks.clear()


_SECRET_SOURCE_CACHE_LOCK = _SecretSourceCacheLocks()
logger = logging.getLogger(__name__)


def _lexical_home_key(home: str | os.PathLike) -> str:
    return os.path.normcase(os.path.normpath(str(home)))


def _active_secret_source_home_key() -> str | None:
    """Resolve the active profile to a previously awaited canonical key."""
    from hermes_constants import get_hermes_home_override

    override = get_hermes_home_override()
    if override is not None:
        home = override
    else:
        home = os.environ.get("HERMES_HOME", "").strip()
        if not home:
            # This snapshot helper is deliberately synchronous and must remain
            # environment-only.  Path.home()/expanduser may consult NSS and
            # block; an environment without a resolvable launch home cannot
            # safely inherit provenance from an arbitrary cached profile.
            if sys.platform == "win32":
                local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
                user_profile = os.environ.get("USERPROFILE", "").strip()
                if local_appdata:
                    home = os.path.join(local_appdata, "hermes")
                elif user_profile:
                    home = os.path.join(user_profile, "AppData", "Local", "hermes")
                else:
                    return None
            else:
                user_home = os.environ.get("HOME", "").strip()
                if not user_home:
                    return None
                home = os.path.join(user_home, ".hermes")

    lexical = _lexical_home_key(home)
    with _SECRET_SOURCE_STATE_GUARD:
        canonical = _SECRET_SOURCE_HOME_ALIASES.get(lexical)
    if canonical is not None:
        return canonical
    # A context-local profile override is an isolation boundary. If that
    # profile has not hydrated its source state, fail closed instead of
    # falling back to provenance recorded for the process profile.
    if override is not None:
        return None
    with _SECRET_SOURCE_STATE_GUARD:
        return lexical if lexical in _SECRET_SOURCES_BY_HOME else None


def _active_secret_source_snapshot() -> tuple[dict[str, str], dict[str, str]]:
    """Return active-profile provenance and values without filesystem I/O."""
    key = _active_secret_source_home_key()
    if key is not None:
        with _SECRET_SOURCE_STATE_GUARD:
            return (
                dict(_SECRET_SOURCES_BY_HOME.get(key, {})),
                dict(_SECRET_SOURCE_VALUES_BY_HOME.get(key, {})),
            )
    # The process-wide map is retained only as a compatibility/debug snapshot.
    # It cannot safely authorize subprocess pass-through because it may have
    # been populated while an explicitly supplied, different profile home was
    # loading. No awaited profile state means no provenance.
    return {}, {}


def get_secret_source(env_var: str) -> str | None:
    """Return this profile's external source for an environment variable."""
    sources, _values = _active_secret_source_snapshot()
    return sources.get(env_var)


async def get_secret_source_values(
    hermes_home: str | os.PathLike,
) -> dict[str, str]:
    """Return the immutable external-secret snapshot for one Hermes home."""
    lexical_key = _lexical_home_key(hermes_home)
    home_key = os.path.normcase(
        str(await aiofiles.os.wrap(os.path.realpath)(hermes_home))
    )
    with _SECRET_SOURCE_STATE_GUARD:
        _SECRET_SOURCE_HOME_ALIASES[lexical_key] = home_key
        return dict(_SECRET_SOURCE_VALUES_BY_HOME.get(home_key, {}))


def reset_secret_source_cache() -> None:
    """Forget external-source application state for all Hermes homes."""
    with _SECRET_SOURCE_STATE_GUARD:
        _APPLIED_HOMES.clear()
        _SECRET_SOURCES.clear()
        _SECRET_SOURCES_BY_HOME.clear()
        _SECRET_SOURCE_VALUES_BY_HOME.clear()
        _SECRET_SOURCE_HOME_ALIASES.clear()
    _SECRET_SOURCE_CACHE_LOCK.clear()


def format_secret_source_suffix(env_var: str) -> str:
    """Return a human-readable provenance suffix for a credential."""
    source_name = get_secret_source(env_var)
    if not source_name:
        return ""
    if source_name == "bitwarden":
        return " (from Bitwarden)"
    try:
        from agent.secret_sources.registry import get_source

        source = get_source(source_name)
        if source is not None and source.label:
            return f" (from {source.label})"
    except Exception:
        pass
    return f" (from {source_name})"


def _format_offending_chars(value: str, limit: int = 3) -> str:
    seen: list[str] = []
    for character in value:
        if ord(character) <= 127:
            continue
        label = f"U+{ord(character):04X}"
        if character.isprintable():
            label += f" ({character!r})"
        if label not in seen:
            seen.append(label)
        if len(seen) >= limit:
            break
    return ", ".join(seen)


def _sanitize_loaded_credentials() -> None:
    """Strip invalid non-ASCII bytes from HTTP credential values."""
    for key, value in list(os.environ.items()):
        if not any(key.endswith(suffix) for suffix in _CREDENTIAL_SUFFIXES):
            continue
        try:
            value.encode("ascii")
            continue
        except UnicodeEncodeError:
            pass
        cleaned = value.encode("ascii", errors="ignore").decode("ascii")
        os.environ[key] = cleaned
        if key in _WARNED_KEYS:
            continue
        _WARNED_KEYS.add(key)
        removed = len(value) - len(cleaned)
        detail = _format_offending_chars(value) or "non-printable"
        print(
            f"Warning: {key} contained {removed} non-ASCII character"
            f"{'s' if removed != 1 else ''} ({detail}); stripped before use. "
            "Re-copy the credential from its source if authentication fails.",
            file=sys.stderr,
        )


async def _sanitize_env_file_if_needed(path: Path) -> None:
    """Rewrite UTF-16 dotenv files as UTF-8 and refuse UTF-32 safely."""
    path = Path(path)
    async with aiofiles.open(path, "rb") as handle:
        raw = await handle.read()
    if raw.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        if path not in _WARNED_UTF32_PATHS:
            _WARNED_UTF32_PATHS.add(path)
            logger.warning("Refusing to rewrite UTF-32 dotenv file: %s", path)
        return
    if not raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return
    text = raw.decode("utf-16")
    async with aiofiles.open(path, "w", encoding="utf-8") as handle:
        await handle.write(text)


async def _load_dotenv_file(path: Path, *, override: bool) -> None:
    await _sanitize_env_file_if_needed(path)
    async with aiofiles.open(path, "rb") as handle:
        raw = await handle.read()
    if raw.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        return
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    for key, value in dotenv_values(stream=io.StringIO(text.replace("\x00", ""))).items():
        if value is not None and (override or key not in os.environ):
            os.environ[key] = value
    _sanitize_loaded_credentials()


async def _load_secrets_config(home_path: Path) -> dict:
    """Read the secrets section without invoking the synchronous config cache."""
    config_path = home_path / "config.yaml"
    try:
        async with aiofiles.open(config_path, encoding="utf-8") as file:
            raw_text = await file.read()
        from utils import fast_safe_load

        data = fast_safe_load(raw_text) or {}
    except (FileNotFoundError, OSError):
        return {}
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    secrets = data.get("secrets")
    return secrets if isinstance(secrets, dict) else {}


def _remediation_hint(source_name: str, error_kind, secrets_cfg: dict) -> str:
    try:
        from agent.secret_sources.registry import get_source

        source = get_source(source_name)
        if source is None:
            return ""
        source_config = secrets_cfg.get(source_name)
        if not isinstance(source_config, dict):
            source_config = {}
        return str(source.remediation(error_kind, source_config) or "").strip()
    except Exception:
        return ""


async def _apply_external_secret_sources(home_path: Path) -> None:
    """Fetch and apply every enabled source once per Hermes home."""
    home = Path(home_path)
    lexical_key = _lexical_home_key(home)
    home_key = os.path.normcase(str(await aiofiles.os.wrap(os.path.realpath)(home)))
    with _SECRET_SOURCE_STATE_GUARD:
        _SECRET_SOURCE_HOME_ALIASES[lexical_key] = home_key
    async with _SECRET_SOURCE_CACHE_LOCK.for_home(home_key):
        with _SECRET_SOURCE_STATE_GUARD:
            already_applied = home_key in _APPLIED_HOMES
        if already_applied:
            return
        secrets_config = await _load_secrets_config(home)
        if not secrets_config:
            return

        try:
            from agent.secret_sources.registry import apply_all

            report = await apply_all(secrets_config, home)
        except Exception:
            return
        if not report.sources:
            return

        sources = {
            name: applied.source for name, applied in report.provenance.items()
        }
        values: dict[str, str] = {}
        if report.applied_any:
            _sanitize_loaded_credentials()
            for name, applied in report.provenance.items():
                value = os.environ.get(name)
                if value is not None:
                    values[name] = value
        with _SECRET_SOURCE_STATE_GUARD:
            _APPLIED_HOMES.add(home_key)
            _SECRET_SOURCES_BY_HOME[home_key] = sources
            _SECRET_SOURCES.update(
                {name: sources[name] for name in values if name in sources}
            )
            if values:
                _SECRET_SOURCE_VALUES_BY_HOME[home_key] = values

        for source_report in report.sources:
            if source_report.applied:
                count = len(source_report.applied)
                print(
                    f"  {source_report.label}: applied {count} "
                    f"secret{'s' if count != 1 else ''}",
                    file=sys.stderr,
                )
            if source_report.result.error:
                print(
                    f"  {source_report.label}: {source_report.result.error}",
                    file=sys.stderr,
                )
                hint = _remediation_hint(
                    source_report.name,
                    source_report.result.error_kind,
                    secrets_config,
                )
                if hint:
                    print(f"  {source_report.label}: → {hint}", file=sys.stderr)
            for warning in source_report.result.warnings:
                print(f"  {source_report.label}: {warning}", file=sys.stderr)
        for conflict in report.conflicts:
            print(f"  Secret sources: {conflict}", file=sys.stderr)


async def hydrate_profile_secret_sources(
    hermes_home: str | os.PathLike,
) -> dict[str, str]:
    """Resolve a profile's sources into an isolated environment snapshot."""
    home = Path(hermes_home)
    lexical_key = _lexical_home_key(home)
    home_key = os.path.normcase(str(await aiofiles.os.wrap(os.path.realpath)(home)))
    with _SECRET_SOURCE_STATE_GUARD:
        _SECRET_SOURCE_HOME_ALIASES[lexical_key] = home_key
    async with _SECRET_SOURCE_CACHE_LOCK.for_home(home_key):
        with _SECRET_SOURCE_STATE_GUARD:
            already_applied = home_key in _APPLIED_HOMES
        if already_applied:
            return await get_secret_source_values(home)

        secrets_config = await _load_secrets_config(home)
        if not secrets_config:
            return {}

        try:
            from agent.secret_scope import _is_global_env, load_env_file
            from agent.secret_sources.registry import apply_all

            local_env = {
                name: value
                for name, value in os.environ.items()
                if _is_global_env(name)
            }
            local_env.update(await load_env_file(home / ".env"))
            op_env = home / ".op.env"
            if await aiofiles.os.path.exists(op_env):
                for name, value in (await load_env_file(op_env)).items():
                    local_env.setdefault(name, value)
            local_env["HERMES_HOME"] = str(home)
            report = await apply_all(secrets_config, home, environ=local_env)
        except Exception:
            return {}
        if not report.sources:
            return {}

        sources = {
            name: applied.source for name, applied in report.provenance.items()
        }
        values: dict[str, str] = {}
        for name, applied in report.provenance.items():
            value = local_env.get(name)
            if value is not None:
                values[name] = value
        with _SECRET_SOURCE_STATE_GUARD:
            _APPLIED_HOMES.add(home_key)
            _SECRET_SOURCES_BY_HOME[home_key] = sources
            _SECRET_SOURCES.update(
                {name: sources[name] for name in values if name in sources}
            )
            if values:
                _SECRET_SOURCE_VALUES_BY_HOME[home_key] = values
        return dict(values)


async def load_hermes_dotenv(
    *,
    hermes_home: str | os.PathLike | None = None,
    project_env: str | os.PathLike | None = None,
) -> list[Path]:
    """Load user, project, and managed dotenv files in precedence order."""
    loaded: list[Path] = []
    home = Path(hermes_home or os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    user_env = home / ".env"
    project = Path(project_env) if project_env else None

    if await aiofiles.os.path.exists(user_env):
        await _load_dotenv_file(user_env, override=True)
        loaded.append(user_env)

    op_env = home / ".op.env"
    if await aiofiles.os.path.exists(op_env) and not os.environ.get(
        "OP_SERVICE_ACCOUNT_TOKEN"
    ):
        await _load_dotenv_file(op_env, override=False)

    if project is not None and await aiofiles.os.path.exists(project):
        await _load_dotenv_file(project, override=not loaded)
        loaded.append(project)

    await _apply_external_secret_sources(home)

    managed_override = os.environ.get("HERMES_MANAGED_DIR", "").strip()
    managed_dir = Path(managed_override) if managed_override else Path("/etc/hermes")
    managed_env = managed_dir / ".env"
    if await aiofiles.os.path.isdir(managed_dir) and await aiofiles.os.path.exists(
        managed_env
    ):
        await _load_dotenv_file(managed_env, override=True)
    await _reapply_terminal_config_bridge(home)
    return loaded


async def _reapply_terminal_config_bridge(home: Path) -> None:
    """Re-apply explicit terminal config after dotenv precedence processing."""
    process_home = Path(
        os.getenv("HERMES_HOME", Path.home() / ".hermes")
    )
    realpath = aiofiles.os.wrap(os.path.realpath)
    if await realpath(home) != await realpath(process_home):
        return

    config_path = home / "config.yaml"
    try:
        async with aiofiles.open(config_path, encoding="utf-8") as handle:
            raw_text = await handle.read()
        from utils import fast_safe_load

        raw_config = fast_safe_load(raw_text) or {}
    except (FileNotFoundError, OSError):
        return
    except Exception:
        logger.debug("Could not apply terminal config bridge", exc_info=True)
        return

    terminal = raw_config.get("terminal") if isinstance(raw_config, dict) else None
    if not isinstance(terminal, dict):
        return
    from hermes_cli.config import TERMINAL_CONFIG_ENV_MAP, _terminal_env_value

    for key, value in terminal.items():
        env_name = TERMINAL_CONFIG_ENV_MAP.get(key)
        if env_name is None:
            continue
        if key == "cwd" and str(value or "").strip() in {".", "auto", "cwd"}:
            continue
        if key == "cwd" and isinstance(value, str):
            value = os.path.expanduser(value)
        os.environ[env_name] = _terminal_env_value(value)
