"""Native async loading for Hermes dotenv files."""

from __future__ import annotations

import codecs
import io
import logging
import os
import sys
from pathlib import Path

import aiofiles
import aiofiles.os
from dotenv import dotenv_values


_CREDENTIAL_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_KEY")
_WARNED_KEYS: set[str] = set()
_WARNED_UTF32_PATHS: set[Path] = set()
logger = logging.getLogger(__name__)


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
    if home.resolve() != process_home.resolve():
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
