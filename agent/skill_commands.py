"""Parse historical slash-skill messages stored in sessions and memory.

The async runtime exposes skills through the native ``skills_list`` and
``skill_view`` tools.  Older Hermes sessions can still contain the expanded
messages produced by classic slash-skill commands, so this module deliberately
keeps only the pure parsing needed to display and store those records safely.
"""

from __future__ import annotations

import re
from typing import Any

# These literals are part of the persisted format emitted by the classic
# runtime.  Keep them byte-stable so existing session previews and memory rows
# remain readable after the async-only migration.
_SKILL_INVOCATION_PREFIX = "[IMPORTANT: The user has invoked the "
_SINGLE_SKILL_MARKER = "The full skill content is loaded below.]"
_SINGLE_SKILL_INSTRUCTION = (
    "The user has provided the following instruction alongside the skill invocation: "
)
_RUNTIME_NOTE = "\n\n[Runtime note:"
_BUNDLE_MARKER = " skill bundle,"
_BUNDLE_USER_INSTRUCTION = "\nUser instruction: "
_BUNDLE_FIRST_SKILL_BLOCK = "\n\n[Loaded as part of the "

_SKILL_NAME_RE = re.compile(re.escape(_SKILL_INVOCATION_PREFIX) + r'"([^"]*)"')

# SQL LIKE pattern matching a historical skill-expanded turn.  The literal
# prefix intentionally contains no LIKE wildcards.
SKILL_SCAFFOLD_SQL_LIKE = _SKILL_INVOCATION_PREFIX + "%"

# Joins head and tail excerpts before ``describe_skill_invocation`` parses
# them.  Never include text from the far side of this marker in a preview.
SKILL_EXCERPT_JOINT = "\x1e"


def extract_user_instruction_from_skill_message(content: Any) -> str | None:
    """Return the actual user instruction from a historical skill turn.

    A non-skill string is returned unchanged.  A bare skill invocation and
    non-string content return ``None`` so memory providers do not persist the
    injected skill body as if it were user-authored text.
    """
    if not isinstance(content, str):
        return None
    if not content.startswith(_SKILL_INVOCATION_PREFIX):
        return content
    if _BUNDLE_MARKER in content:
        return _extract_bundle_user_instruction(content)
    if _SINGLE_SKILL_MARKER in content:
        return _extract_single_skill_user_instruction(content)
    return None


def describe_skill_invocation(content: Any, separator: str = " — ") -> str | None:
    """Render a historical expanded skill turn as its original slash command."""
    if not isinstance(content, str) or not content.startswith(_SKILL_INVOCATION_PREFIX):
        return None

    match = _SKILL_NAME_RE.match(content)
    name = (match.group(1) if match else "").strip()
    label = name if name.startswith("/") else f"/{name}"

    instruction = extract_user_instruction_from_skill_message(content)
    if instruction and instruction is not content:
        instruction = " ".join(instruction.split(SKILL_EXCERPT_JOINT)[0].split())
        if instruction:
            return f"{label}{separator}{instruction}" if name else instruction
    return label if name else None


def _extract_single_skill_user_instruction(message: str) -> str | None:
    marker_index = message.rfind(_SINGLE_SKILL_INSTRUCTION)
    if marker_index < 0:
        return None
    instruction = message[marker_index + len(_SINGLE_SKILL_INSTRUCTION):]
    runtime_index = instruction.find(_RUNTIME_NOTE)
    if runtime_index >= 0:
        instruction = instruction[:runtime_index]
    return instruction.strip() or None


def _extract_bundle_user_instruction(message: str) -> str | None:
    marker_index = message.find(_BUNDLE_USER_INSTRUCTION)
    if marker_index < 0:
        return None
    instruction = message[marker_index + len(_BUNDLE_USER_INSTRUCTION):]
    first_skill_index = instruction.find(_BUNDLE_FIRST_SKILL_BLOCK)
    if first_skill_index >= 0:
        instruction = instruction[:first_skill_index]
    return instruction.strip() or None
