"""Shared native-async SKILL.md preprocessing helpers."""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
from pathlib import Path

from hermes_cli._subprocess_compat import windows_hide_flags

logger = logging.getLogger(__name__)

_SKILL_TEMPLATE_RE = re.compile(r"\$\{(HERMES_SKILL_DIR|HERMES_SESSION_ID)\}")
_INLINE_SHELL_RE = re.compile(r"!`([^`\n]+)`")
_INLINE_SHELL_MAX_OUTPUT = 4000


async def load_skills_config() -> dict:
    """Load the ``skills`` config section through the async config boundary."""
    try:
        from hermes_cli.config import load_config_readonly

        config = await load_config_readonly()
        skills_config = (config or {}).get("skills")
        if isinstance(skills_config, dict):
            return skills_config
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("Could not read skills config", exc_info=True)
    return {}


def substitute_template_vars(
    content: str,
    skill_dir: Path | None,
    session_id: str | None,
) -> str:
    """Replace the two supported Hermes skill-template variables."""
    if not content:
        return content
    skill_dir_text = str(skill_dir) if skill_dir else None

    def _replace(match: re.Match) -> str:
        token = match.group(1)
        if token == "HERMES_SKILL_DIR" and skill_dir_text:
            return skill_dir_text
        if token == "HERMES_SESSION_ID" and session_id:
            return str(session_id)
        return match.group(0)

    return _SKILL_TEMPLATE_RE.sub(_replace, content)


async def _finish_process_communicate(
    process: asyncio.subprocess.Process,
    communicate_task: asyncio.Task[tuple[bytes | None, bytes | None]],
) -> tuple[bytes | None, bytes | None]:
    """Drain and reap one owned inline-shell process through cancellation."""
    async def drain_or_wait() -> tuple[bytes | None, bytes | None]:
        try:
            return await communicate_task
        except BaseException:
            await process.wait()
            raise

    cleanup_task = asyncio.create_task(drain_or_wait())
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            output = await asyncio.shield(cleanup_task)
            break
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised below
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
    return output


async def run_inline_shell(command: str, cwd: Path | None, timeout: int) -> str:
    """Execute one trusted skill inline-shell snippet without blocking."""
    timeout = max(1, int(timeout))
    try:
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            cwd=str(cwd) if cwd else None,
            stdin=subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=windows_hide_flags(),
        )
    except FileNotFoundError:
        return "[inline-shell error: bash not found]"
    except Exception as exc:
        return f"[inline-shell error: {exc}]"

    communicate_task = asyncio.create_task(process.communicate())
    try:
        stdout, stderr = await asyncio.wait_for(
            asyncio.shield(communicate_task), timeout=timeout
        )
    except TimeoutError:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await _finish_process_communicate(process, communicate_task)
        return f"[inline-shell timeout after {timeout}s: {command}]"
    except asyncio.CancelledError:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        try:
            await _finish_process_communicate(process, communicate_task)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug(
                "Inline-shell cleanup after cancellation failed",
                exc_info=True,
            )
        raise
    except Exception as exc:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        try:
            await _finish_process_communicate(process, communicate_task)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        return f"[inline-shell error: {exc}]"

    output = (stdout or b"").decode("utf-8", errors="replace").rstrip("\n")
    if not output and stderr:
        output = stderr.decode("utf-8", errors="replace").rstrip("\n")
    if len(output) > _INLINE_SHELL_MAX_OUTPUT:
        output = output[:_INLINE_SHELL_MAX_OUTPUT] + "...[truncated]"
    return output


async def expand_inline_shell(
    content: str,
    skill_dir: Path | None,
    timeout: int,
) -> str:
    """Replace inline-shell snippets sequentially in their source order."""
    if "!`" not in content:
        return content
    parts: list[str] = []
    cursor = 0
    for match in _INLINE_SHELL_RE.finditer(content):
        parts.append(content[cursor : match.start()])
        command = match.group(1).strip()
        parts.append(
            await run_inline_shell(command, skill_dir, timeout)
            if command
            else ""
        )
        cursor = match.end()
    parts.append(content[cursor:])
    return "".join(parts)


async def preprocess_skill_content(
    content: str,
    skill_dir: Path | None,
    session_id: str | None = None,
    skills_cfg: dict | None = None,
) -> str:
    """Apply configured template and inline-shell preprocessing."""
    if not content:
        return content
    config = (
        skills_cfg
        if isinstance(skills_cfg, dict)
        else await load_skills_config()
    )
    if config.get("template_vars", True):
        content = substitute_template_vars(content, skill_dir, session_id)
    if config.get("inline_shell", False):
        timeout = int(config.get("inline_shell_timeout", 10) or 10)
        content = await expand_inline_shell(content, skill_dir, timeout)
    return content
