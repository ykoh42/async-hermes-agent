"""Compatibility helpers retained after removing the interactive setup wizard.

The former module contained the full first-run/configuration wizard.  The
wizard is intentionally gone from this runtime fork, but a few existing
maintenance and import paths still use these small prompt/output helpers.  The
module path is kept stable so upstream updates and third-party integrations do
not need a filename migration.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any

from hermes_cli.cli_output import print_error, print_info, print_success, print_warning
from hermes_cli.colors import Colors, color

_BRACKETED_PASTE_PATTERN = re.compile(r"\x1b\[s*200~|\x1b\[s*201~")


def print_header(title: str) -> None:
    print()
    print(color(f"◆ {title}", Colors.CYAN, Colors.BOLD))


def prompt(question: str, default: str | None = None, password: bool = False) -> str:
    """Read a simple value; password mode is intentionally unsupported."""
    del password
    suffix = f" [{default}]" if default else ""
    try:
        value = input(color(f"{question}{suffix}: ", Colors.YELLOW))
    except (KeyboardInterrupt, EOFError):
        return default or ""
    value = _BRACKETED_PASTE_PATTERN.sub("", value or "").strip()
    return value or default or ""


def _curses_prompt_choice(
    question: str,
    choices: list[Any],
    default: int = 0,
    description: str | None = None,
) -> int:
    """Compatibility choice helper without curses or terminal UI."""
    del description
    if not choices:
        return -1
    if not sys.stdin.isatty():
        return default
    print(color(question, Colors.YELLOW))
    for index, choice in enumerate(choices):
        marker = "*" if index == default else " "
        print(f"  {marker} {index + 1}. {choice}")
    raw = prompt("Select", str(default + 1))
    try:
        index = int(raw) - 1
    except ValueError:
        return default
    return index if 0 <= index < len(choices) else default


def prompt_choice(
    question: str,
    choices: list[Any],
    default: int = 0,
    description: str | None = None,
) -> int:
    return _curses_prompt_choice(question, choices, default, description)


def prompt_yes_no(question: str, default: bool = True) -> bool:
    if os.environ.get("HERMES_NONINTERACTIVE", "").lower() in {"1", "true", "yes", "on"}:
        return default
    suffix = "Y/n" if default else "y/N"
    try:
        value = input(color(f"{question} [{suffix}]: ", Colors.YELLOW)).strip().lower()
    except (KeyboardInterrupt, EOFError):
        return default
    if not value:
        return default
    return value in {"y", "yes"}


def prompt_checklist(
    title: str,
    items: list[Any],
    pre_selected: list[Any] | None = None,
) -> list[Any]:
    """Return the existing selection; interactive checklist UI was removed."""
    del title, items
    return list(pre_selected or [])


def _run_xai_oauth_login_from_setup() -> bool:
    """Compatibility hook for optional TTS/tool setup; auth owns the flow."""
    try:
        from hermes_cli.auth import (
            _is_remote_session,
            _save_xai_oauth_tokens,
            _xai_oauth_device_code_login,
            unsuppress_credential_source,
        )

        creds = _xai_oauth_device_code_login(open_browser=not _is_remote_session())
        _save_xai_oauth_tokens(
            creds["tokens"],
            discovery=creds.get("discovery"),
            redirect_uri=creds.get("redirect_uri", ""),
            last_refresh=creds.get("last_refresh"),
            auth_mode="oauth_device_code",
            set_active=False,
        )
        unsuppress_credential_source("xai-oauth", "device_code")
        return True
    except Exception as exc:
        print_warning(f"xAI Grok OAuth login failed: {exc}")
        return False


def is_noninteractive() -> bool:
    return os.environ.get("HERMES_NONINTERACTIVE", "").lower() in {
        "1", "true", "yes", "on"
    }
