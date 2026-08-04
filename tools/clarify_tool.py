#!/usr/bin/env python3
"""
Clarify Tool Module - Interactive Clarifying Questions

Allows the agent to present structured multiple-choice questions or open-ended
prompts to the user. In CLI mode, choices are navigable with arrow keys. On
messaging platforms, choices are rendered as a numbered list.

Supports both single-select (radio) and multi-select (checkbox) modes via the
``multi_select`` parameter.

The actual user-interaction logic lives in the platform layer (cli.py for CLI,
gateway/run.py for messaging). This module defines the schema, validation, and
a thin dispatcher that delegates to a platform-provided callback.
"""

import inspect
import json
from typing import List, Optional, Callable


# Maximum number of predefined choices the agent can offer.
# A 5th "Other (type your answer)" option is always appended by the UI.
MAX_CHOICES = 4


def _flatten_choice(c) -> str:
    """Coerce a single choice into its user-facing display string.

    The schema declares choices as bare strings, but LLMs sometimes emit
    dict-shaped choices like ``[{"description": "..."}]``. A naive ``str(c)``
    turns the whole dict into its Python repr — ``{'description': '...'}`` —
    which then leaks onto every surface that renders the choice (CLI panel,
    Discord buttons, Telegram numbered list) AND is returned verbatim as the
    user's answer. Normalising here, at the one platform-agnostic entry point,
    fixes the whole class in one place instead of per-adapter.

    Dict unwrap order is the canonical LLM tool-call user-facing keys:
    ``label`` → ``description`` → ``text`` → ``title``. ``name`` and ``value``
    are deliberately excluded — they're component-shaped fields that could
    carry raw enum values or short identifiers, not human-readable labels. A
    dict with none of the canonical keys is dropped (returns ""), since a
    garbage label is worse than no choice at all.
    """
    if c is None:
        return ""
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, dict):
        for key in ("label", "description", "text", "title"):
            v = c.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    if isinstance(c, (list, tuple)):
        return " ".join(_flatten_choice(x) for x in c).strip()
    return str(c).strip()


async def _invoke_callback(callback, question, choices, multi_select):
    """Invoke the platform callback, passing multi_select if supported.

    Uses signature inspection (not a ``TypeError`` retry) to decide whether
    the callback accepts the ``multi_select`` keyword — a retry-on-TypeError
    approach would re-invoke a *compatible* callback that raised TypeError
    internally, potentially prompting the user twice.
    """
    accepts_multi = False
    try:
        sig = inspect.signature(callback)
        params = sig.parameters
        accepts_multi = "multi_select" in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    except (TypeError, ValueError):
        # Builtins / C callables without introspectable signatures:
        # be conservative and use the legacy 2-arg form.
        accepts_multi = False

    if accepts_multi:
        return await callback(question, choices, multi_select=multi_select)
    return await callback(question, choices)


def _parse_multi_select_response(raw_response) -> List[str]:
    """Parse a multi-select response into a list of cleaned choice strings.

    Handles three forms:
      - Already a list  →  stringify + strip each element
      - JSON array      →  parse and strip
      - Comma-separated →  split, strip, drop empties
    """
    if isinstance(raw_response, list):
        return [str(r).strip() for r in raw_response if str(r).strip()]

    raw = str(raw_response).strip()

    # Try JSON array
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(p).strip() for p in parsed if str(p).strip()]
        except json.JSONDecodeError:
            pass

    # Fall back to comma-separated
    return [s.strip() for s in raw.split(",") if s.strip()]




async def clarify_tool(
    question: str,
    choices: Optional[List[str]] = None,
    multi_select: bool = False,
    callback: Optional[Callable] = None,
) -> str:
    """Native async implementation of the model-visible clarify tool."""
    if not question or not question.strip():
        return tool_error("Question text is required.")

    question = question.strip()
    if choices is not None:
        if not isinstance(choices, list):
            return tool_error("choices must be a list of strings.")
        choices = [choice for choice in (_flatten_choice(item) for item in choices) if choice]
        choices = choices[:MAX_CHOICES] or None

    if callback is None:
        return tool_error("Clarify tool is not available in this execution context.")
    if not inspect.iscoroutinefunction(callback):
        return tool_error(
            "Clarify requires an async platform callback in async-hermes-agent."
        )

    try:
        raw_response = await _invoke_callback(
            callback, question, choices, multi_select
        )
    except Exception as exc:
        return tool_error(f"Failed to get user input: {exc}")

    user_response = (
        _parse_multi_select_response(raw_response)
        if multi_select and choices is not None
        else str(raw_response).strip()
    )
    return json.dumps(
        {
            "question": question,
            "choices_offered": choices,
            "user_response": user_response,
        },
        ensure_ascii=False,
    )


def check_clarify_requirements() -> bool:
    """Clarify tool has no external requirements -- always available."""
    return True


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

CLARIFY_SCHEMA = {
    "name": "clarify",
    "description": (
        "Ask the user a question when you need clarification, feedback, or a "
        "decision before proceeding. Supports three modes:\n\n"
        "1. **Single-select multiple choice** — provide up to 4 choices. The user picks one "
        "or types their own answer via a 5th 'Other' option.\n"
        "2. **Multi-select multiple choice** — set multi_select=true. The user can select "
        "multiple options via checkboxes. user_response will be a list of selected choices.\n"
        "3. **Open-ended** — omit choices entirely. The user types a free-form "
        "response.\n\n"
        "CRITICAL: when you are offering options, put each option ONLY in the "
        "`choices` array — NEVER enumerate the options inside the `question` "
        "text. The UI renders `choices` as selectable rows; options written "
        "into the question string render as dead prose the user can't pick. "
        "Right: question='Which deployment target?', choices=['staging', "
        "'prod']. Wrong: question='Which target? 1) staging 2) prod', choices=[].\n\n"
        "Use this tool when:\n"
        "- The task is ambiguous and you need the user to choose an approach\n"
        "- You want post-task feedback ('How did that work out?')\n"
        "- You want to offer to save a skill or update memory\n"
        "- A decision has meaningful trade-offs the user should weigh in on\n\n"
        "Do NOT use this tool for simple yes/no confirmation of dangerous "
        "commands (the terminal tool handles that). Prefer making a reasonable "
        "default choice yourself when the decision is low-stakes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "The question itself, and ONLY the question (e.g. 'Which "
                    "deployment target?'). Do NOT embed the answer options here "
                    "— pass them as separate elements in `choices`."
                ),
            },
            "choices": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": MAX_CHOICES,
                "description": (
                    "REQUIRED whenever you are presenting selectable options: "
                    "each distinct option is its own array element (up to 4). "
                    "The UI renders these as pickable rows and auto-appends an "
                    "'Other (type your answer)' option. Omit this parameter "
                    "entirely ONLY for a genuinely open-ended free-text question."
                ),
            },
            "multi_select": {
                "type": "boolean",
                "description": (
                    "When true, the user can select MULTIPLE options (like checkboxes). "
                    "The user_response will be a list of selected choices. "
                    "When false (default), single selection (radio). "
                    "Has no effect when choices is omitted (open-ended question)."
                ),
            },
        },
        "required": ["question"],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error


async def _handle_clarify(args: dict, **kwargs) -> str:
    """Adapt the registry's JSON-object contract to ``clarify_tool``."""
    return await clarify_tool(
        question=args.get("question", ""),
        choices=args.get("choices"),
        multi_select=args.get("multi_select", False),
        callback=kwargs.get("callback"),
    )


registry.register(
    name="clarify",
    toolset="clarify",
    schema=CLARIFY_SCHEMA,
    handler=_handle_clarify,
    check_fn=check_clarify_requirements,
    emoji="❓",
)
