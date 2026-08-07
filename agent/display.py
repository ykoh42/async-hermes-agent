"""CLI presentation -- spinner, kawaii faces, tool preview formatting.

Pure display functions and classes with no AIAgent dependency.
Used by AIAgent._execute_tool_calls for CLI feedback.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from utils import safe_json_loads
from agent.redact import redact_sensitive_text
from agent.tool_result_classification import file_mutation_result_landed

# ANSI escape codes for coloring tool failure indicators
_RED = "\033[31m"
_RESET = "\033[0m"

logger = logging.getLogger(__name__)

_ANSI_RESET = "\033[0m"


def _display_url(value: Any) -> str:
    """Extract a display-only URL without assuming model argument types."""
    if isinstance(value, dict):
        value = value.get("url") or value.get("href")
    return value.strip() if isinstance(value, str) else ""


# Fixed diff colors, independent of UI themes.
_diff_colors_cached: dict[str, str] | None = None


def _diff_ansi() -> dict[str, str]:
    """Return fixed ANSI escapes for diff display."""
    global _diff_colors_cached
    if _diff_colors_cached is not None:
        return _diff_colors_cached

    # Defaults that work on dark terminals
    dim = "\033[38;2;150;150;150m"
    file_c = "\033[38;2;180;160;255m"
    hunk = "\033[38;2;120;120;140m"
    minus = "\033[38;2;255;255;255;48;2;120;20;20m"
    plus = "\033[38;2;255;255;255;48;2;20;90;20m"

    _diff_colors_cached = {
        "dim": dim, "file": file_c, "hunk": hunk,
        "minus": minus, "plus": plus,
    }
    return _diff_colors_cached


# Module-level helpers — each call resolves from the active skin lazily.
def _diff_dim():   return _diff_ansi()["dim"]
def _diff_file():  return _diff_ansi()["file"]
def _diff_hunk():  return _diff_ansi()["hunk"]
def _diff_minus(): return _diff_ansi()["minus"]
def _diff_plus():  return _diff_ansi()["plus"]
_MAX_INLINE_DIFF_FILES = 6
_MAX_INLINE_DIFF_LINES = 80


# =========================================================================
# Configurable tool preview length (0 = no limit)
# Set once at startup by CLI or gateway from display.tool_preview_length config.
# =========================================================================
_tool_preview_max_len: int = 0  # 0 = unlimited


def set_tool_preview_max_len(n: int) -> None:
    """Set the global max length for tool call previews. 0 = no limit."""
    global _tool_preview_max_len
    _tool_preview_max_len = max(int(n), 0) if n else 0


def get_tool_preview_max_len() -> int:
    """Return the configured max preview length (0 = unlimited)."""
    return _tool_preview_max_len


# =========================================================================
# Fixed display defaults
# =========================================================================

def _get_skin():
    """Compatibility shim for display callers; UI skins are not included."""
    return None


def get_skin_tool_prefix() -> str:
    """Get the fixed tool output prefix."""
    return "┊"


def get_tool_emoji(tool_name: str, default: str = "⚡") -> str:
    """Get the display emoji for a tool.

    Resolution order: tool registry emoji, then *default* fallback.
    """
    # Registry default
    try:
        from tools.registry import registry
        emoji = registry.get_emoji(tool_name, default="")
        if emoji:
            return emoji
    except Exception:
        pass
    # Hardcoded fallback
    return default


# =========================================================================
# Tool preview (one-line summary of a tool call's primary argument)
# =========================================================================

def _oneline(text: str) -> str:
    """Collapse whitespace (including newlines) to single spaces."""
    return " ".join(text.split())


def _truncate_preview(text: str, max_len: int | None) -> str:
    if max_len and max_len > 0 and len(text) > max_len:
        if max_len <= 3:
            return "." * max_len
        return text[:max_len - 3] + "..."
    return text


@dataclass(frozen=True)
class ToolPreview:
    """A compact tool preview plus presentation facts lost to truncation."""

    text: str
    truncated: bool = False
    url: str | None = None


_SHELL_SILENT_HEADS = {"cd", "pushd", "popd", "export", "set", "unset", "source", ".", "true", "false", ":"}
_SHELL_PIPE_TAIL_HEADS = {"head", "tail", "wc", "sort", "uniq"}


def _shell_basename(head: str) -> str:
    return head.rsplit("/", 1)[-1] if head else ""


def _split_shell_words(segment: str) -> list[str]:
    words: list[str] = []
    buf: list[str] = []
    quote: str | None = None

    for i, ch in enumerate(segment):
        if quote:
            buf.append(ch)
            if ch == quote and (i == 0 or segment[i - 1] != "\\"):
                quote = None
            continue

        if ch in {"'", '"'}:
            quote = ch
            buf.append(ch)
            continue

        if ch.isspace():
            if buf:
                words.append("".join(buf))
                buf = []
            continue

        buf.append(ch)

    if buf:
        words.append("".join(buf))

    return words


def _strip_shell_pipe_tail(segment: str) -> str:
    words = _split_shell_words(segment)
    out: list[str] = []

    for i, word in enumerate(words):
        if word == "|" and _shell_basename(words[i + 1] if i + 1 < len(words) else "") in _SHELL_PIPE_TAIL_HEADS:
            break
        out.append(word)

    return " ".join(out).strip()


def _split_shell_compound(command: str) -> list[str]:
    segments: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    i = 0

    while i < len(command):
        ch = command[i]

        if quote:
            buf.append(ch)
            if ch == quote and (i == 0 or command[i - 1] != "\\"):
                quote = None
            i += 1
            continue

        if ch in {"'", '"'}:
            quote = ch
            buf.append(ch)
            i += 1
            continue

        op_len = 2 if command.startswith("&&", i) or command.startswith("||", i) else 1 if ch in {";", "\n"} else 0
        if op_len:
            segment = _strip_shell_pipe_tail("".join(buf).strip())
            if segment:
                segments.append(segment)
            buf = []
            i += op_len
            continue

        buf.append(ch)
        i += 1

    segment = _strip_shell_pipe_tail("".join(buf).strip())
    if segment:
        segments.append(segment)

    return segments


def _shell_head_word(segment: str) -> str:
    words = _split_shell_words(segment)
    index = 0
    while index < len(words) and re.match(r"^[A-Za-z_]\w*=", words[index]):
        index += 1
    return _shell_basename(words[index] if index < len(words) else "")


def _clean_shell_segment(segment: str) -> str:
    words = _split_shell_words(segment)
    out: list[str] = []
    i = 0
    while i < len(words):
        word = words[i]
        if re.match(r"^\d*(?:>>?|<)$", word):
            i += 2
            continue
        if re.match(r"^\d*(?:>&|<&)\d+$", word) or re.match(r"^\d*>&\d+$", word):
            i += 1
            continue
        out.append(word)
        i += 1
    return " ".join(out).strip()


def _is_shell_boundary_echo(segment: str) -> bool:
    words = _split_shell_words(segment)
    if _shell_basename(words[0] if words else "") != "echo":
        return False
    rest = " ".join(words[1:])
    return bool(re.search(r"-{2,}|_exit=|(?:^|\s|=)\$[?{]|PIPESTATUS", rest))


def summarize_shell_command(command: str) -> str:
    """Compact shell wrapper/plumbing for display while preserving raw command elsewhere."""
    original = _oneline(command)
    if not original:
        return ""

    segments = _split_shell_compound(original)
    if len(segments) <= 1:
        return _clean_shell_segment(segments[0] if segments else original) or original

    core: list[str] = []
    for segment in segments:
        cleaned = _clean_shell_segment(segment)
        head = _shell_head_word(cleaned)
        if cleaned and head not in _SHELL_SILENT_HEADS and not _is_shell_boundary_echo(cleaned):
            core.append(cleaned)

    if not core:
        return original
    if len(core) == 1:
        return core[0]

    count = len(core) - 1
    return f"{core[0]} + {count} {'command' if count == 1 else 'commands'}"


def _read_file_line_label(args: dict) -> str:
    offset = args.get("offset")
    limit = args.get("limit")
    if not isinstance(offset, int) or offset <= 0:
        return ""
    if not isinstance(limit, int) or limit <= 1:
        return f"L{offset}"
    return f"L{offset}-{offset + limit - 1}"


def redact_browser_typed_text_for_display(value: Any, typed_text: Any) -> Any:
    """Apply secret redaction to browser_type text in display-facing payloads.

    Backends sometimes echo the attempted input in error strings or fallback
    metadata.  When the raw typed value contains a recognizable secret (API
    key, token, JWT, etc.) the redacted form differs from the raw value, so we
    replace every occurrence of the raw value with its redacted form before a
    browser_type result reaches logs, callbacks, the model, or chat history.

    Normal typed text (search queries, addresses, form fields) matches no
    secret pattern, so it passes through unchanged and stays readable.

    Redaction is forced here regardless of the global ``security.redact_secrets``
    preference: a typed credential leaking into chat history is a security
    boundary, not mere log hygiene.
    """
    if typed_text is None:
        return value
    needle = str(typed_text)
    if needle == "":
        return value
    redacted = redact_sensitive_text(needle, force=True)
    if redacted == needle:
        # Nothing secret-looking in the typed text; leave payload untouched.
        return value
    if isinstance(value, str):
        return value.replace(needle, redacted)
    if isinstance(value, dict):
        return {
            key: redact_browser_typed_text_for_display(item, typed_text)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_browser_typed_text_for_display(item, typed_text) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_browser_typed_text_for_display(item, typed_text) for item in value)
    return value


def redact_tool_args_for_display(tool_name: str, args: dict | None) -> dict | None:
    """Return a copy of tool args safe for logs/progress UI.

    For ``browser_type`` the ``text`` argument is run through the same
    secret-pattern redactor used for logs.  Recognizable credentials (API
    keys, tokens) are masked before the value reaches tool progress
    notifications; normal typed text is left intact for debuggability.
    """
    if not isinstance(args, dict):
        return args
    if tool_name == "browser_type" and isinstance(args.get("text"), str):
        safe_args = dict(args)
        safe_args["text"] = redact_sensitive_text(args["text"], force=True)
        return safe_args
    return args


def _delegate_task_goal_parts(tasks: Any, *, per_goal_len: int) -> tuple[int, list[str]]:
    if not isinstance(tasks, list):
        return 0, []
    goals: list[str] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        raw_goal = task.get("goal")
        goal = "?" if raw_goal is None else _oneline(str(raw_goal))
        goals.append(_truncate_preview(goal or "?", per_goal_len))
    return len(goals), goals


def build_tool_preview(tool_name: str, args: dict, max_len: int | None = None) -> str | None:
    """Build a short preview of a tool call's primary argument for display.

    *max_len* controls truncation.  ``None`` (default) defers to the global
    ``_tool_preview_max_len`` set via config; ``0`` means unlimited.
    """
    if max_len is None:
        max_len = _tool_preview_max_len
    if not args:
        return None
    args = redact_tool_args_for_display(tool_name, args) or args
    primary_args = {
        "terminal": "command", "web_search": "query", "web_extract": "urls",
        "read_file": "path", "write_file": "path", "patch": "path",
        "search_files": "pattern", "browser_navigate": "url",
        "browser_click": "ref", "browser_type": "text",
        "image_generate": "prompt", "text_to_speech": "text",
        "vision_analyze": "question",
        "skill_view": "name", "skills_list": "category",
        "delegate_task": "goal",
        "clarify": "question", "skill_manage": "name",
    }

    # delegate_task: show goal (single) or individual task goals (batch)
    if tool_name == "delegate_task":
        tasks = args.get("tasks")
        if tasks and isinstance(tasks, list):
            task_count, goals = _delegate_task_goal_parts(tasks, per_goal_len=40)
            preview = (
                f"{task_count} tasks: " + " | ".join(goals)
                if goals else f"{len(tasks)} parallel tasks"
            )
            return _truncate_preview(preview, max_len)
        goal = args.get("goal", "")
        if goal is None:
            return None
        preview = _oneline(str(goal))
        return _truncate_preview(preview, max_len) if preview else None

    if tool_name == "process":
        action = args.get("action", "")
        sid = args.get("session_id", "")
        data = args.get("data", "")
        timeout_val = args.get("timeout")
        parts = [str(action) if action else ""]
        if sid:
            parts.append(str(sid)[:16])
        if data:
            parts.append(f'"{_oneline(str(data)[:20])}"')
        if timeout_val and action == "wait":
            parts.append(f"{timeout_val}s")
        parts = [p for p in parts if p]
        return " ".join(parts) if parts else None

    if tool_name == "todo":
        todos_arg = args.get("todos")
        merge = args.get("merge", False)
        if todos_arg is None:
            return "reading task list"
        elif merge:
            return f"updating {len(todos_arg)} task(s)"
        else:
            return f"planning {len(todos_arg)} task(s)"

    if tool_name == "terminal":
        command = args.get("command")
        if command is None:
            return None
        preview = summarize_shell_command(str(command))
        return _truncate_preview(preview, max_len) if preview else None

    if tool_name == "read_file":
        path = args.get("path") or args.get("file") or args.get("filepath")
        if path is None:
            return None
        label = Path(str(path).replace("\\", "/")).name or str(path)
        line_label = _read_file_line_label(args)
        preview = f"{label} {line_label}".strip()
        return _truncate_preview(preview, max_len) if preview else None

    if tool_name == "session_search":
        query = _oneline(args.get("query", ""))
        return f"recall: \"{query[:25]}{'...' if len(query) > 25 else ''}\""

    if tool_name == "memory":
        action = args.get("action", "")
        target = args.get("target", "")
        if action == "add":
            content = _oneline(args.get("content", ""))
            return f"+{target}: \"{content[:25]}{'...' if len(content) > 25 else ''}\""
        elif action == "replace":
            old = _oneline(args.get("old_text") or "") or "<missing old_text>"
            return f"~{target}: \"{old[:20]}\""
        elif action == "remove":
            old = _oneline(args.get("old_text") or "") or "<missing old_text>"
            return f"-{target}: \"{old[:20]}\""
        return action

    if tool_name == "send_message":
        target = args.get("target", "?")
        msg = _oneline(args.get("message", ""))
        if len(msg) > 20:
            msg = msg[:17] + "..."
        return f"to {target}: \"{msg}\""

    if tool_name == "skill_view":
        name = _oneline(str(args.get("name") or ""))
        file_path = args.get("file_path")
        if file_path:
            file_path = _oneline(str(file_path))
            preview = f"{name} → {file_path}" if name else file_path
        else:
            preview = name
        return _truncate_preview(preview, max_len) if preview else None

    key = primary_args.get(tool_name)
    if not key:
        for fallback_key in ("query", "text", "command", "path", "name", "prompt", "code", "goal"):
            if fallback_key in args:
                key = fallback_key
                break

    if not key or key not in args:
        return None

    value = args[key]
    if isinstance(value, list):
        value = value[0] if value else ""

    preview = _oneline(str(value))
    if not preview:
        return None
    if max_len > 0 and len(preview) > max_len:
        preview = preview[:max_len - 3] + "..."
    return preview


def prepare_tool_preview(
    tool_name: str,
    args: dict | None,
    *,
    fallback: str,
    max_len: int,
) -> ToolPreview:
    """Build one canonical compact preview before platform formatting.

    The uncapped preview is rebuilt from the tool arguments when possible so
    an upstream display cap cannot discard its link target.  Platforms then
    receive explicit truncation and URL metadata instead of inferring either
    fact from the rendered text.
    """
    full_text = build_tool_preview(tool_name, args, max_len=0) or fallback
    text = _truncate_preview(full_text, max_len)
    truncated = text != full_text
    url = None
    if truncated:
        candidate = _display_url(full_text)
        try:
            parsed = urlsplit(candidate)
        except ValueError:
            parsed = None
        if parsed and parsed.scheme.lower() in {"http", "https"} and parsed.netloc:
            url = candidate
    return ToolPreview(text=text, truncated=truncated, url=url)


# =========================================================================
# Friendly tool labels (human-phrased verbs for built-in tools)
#
# Turns "web_search <query>" into "Searching the web for <query>" — the
# ChatGPT-style "Searching…/Reading…" surface.  Curated and built-in only:
# we know each core tool's semantics, so the verb is fixed, not computed.
# Custom/plugin/MCP tools have no entry and fall back to the raw preview.
# =========================================================================

# Each entry maps a built-in tool name to its present-participle verb phrase.
# A trailing space-then-preview is appended by build_tool_label() when the
# tool's argument preview is available (e.g. "Reading docs/api.md").
_TOOL_VERBS: dict[str, str] = {
    "web_search": "Searching the web",
    "web_extract": "Reading",
    "browser_navigate": "Browsing",
    "browser_click": "Clicking",
    "browser_type": "Typing",
    "read_file": "Reading",
    "write_file": "Writing",
    "patch": "Editing",
    "search_files": "Searching files",
    "terminal": "Running",
    "image_generate": "Generating image",
    "video_generate": "Generating video",
    "text_to_speech": "Generating speech",
    "vision_analyze": "Looking at the image",
    "session_search": "Searching past sessions",
    "skill_view": "Reading skill",
    "skills_list": "Listing skills",
    "skill_manage": "Updating skill",
    "delegate_task": "Delegating",
    "clarify": "Asking",
    "memory": "Updating memory",
    "todo": "Updating tasks",
}

# Verbs that read better without the raw argument preview appended.
_TOOL_VERBS_NO_PREVIEW: frozenset[str] = frozenset({
    "skills_list",
    "session_search",
})

# Verbs that take a "for" connector before the preview (search-style phrasing):
# "Searching the web for <query>" reads better than "Searching the web <query>".
_TOOL_VERBS_FOR_CONNECTOR: frozenset[str] = frozenset({
    "web_search",
    "search_files",
})

_friendly_tool_labels: bool = True


def set_friendly_tool_labels(enabled: bool) -> None:
    """Toggle friendly human-phrased tool labels (display.friendly_tool_labels)."""
    global _friendly_tool_labels
    _friendly_tool_labels = bool(enabled)


def get_friendly_tool_labels() -> bool:
    """Return whether friendly tool labels are enabled."""
    return _friendly_tool_labels


def get_tool_verb(tool_name: str) -> str | None:
    """Return the friendly verb for a built-in tool, or None.

    Returns None when friendly labels are disabled or the tool has no curated
    verb (custom/plugin/MCP tools).  Callers that already hold a computed
    argument preview can compose ``f"{verb} {preview}"`` themselves; use
    :func:`tool_verb_connector` to pick the right joiner.
    """
    if not _friendly_tool_labels:
        return None
    return _TOOL_VERBS.get(tool_name)


def tool_verb_connector(tool_name: str) -> str:
    """Return the connector between a verb and its preview (" for " or " ")."""
    return " for " if tool_name in _TOOL_VERBS_FOR_CONNECTOR else " "


def verb_drops_preview(tool_name: str) -> bool:
    """Whether the verb should render alone, without the argument preview."""
    return tool_name in _TOOL_VERBS_NO_PREVIEW


def build_status_phrase(tool_name: str, args: dict | None, max_len: int = 49) -> str | None:
    """Build a short present-tense status phrase for platform status surfaces.

    Used by text-rendering "typing" indicators (Slack's
    ``assistant.threads.setStatus`` line) to show what the agent is doing
    right now: ``is running scripts/run_tests.sh…`` instead of a static
    ``is thinking...``.  The phrase is phrased to follow the bot's display
    name ("Hermes is running …"), so it starts lowercase with "is".

    Pass ``args=None`` for a verb-only phrase (``is running…``) — used when
    ``display.live_status`` is ``verb`` to keep argument previews out of
    shared channels.

    Returns None for the ``_thinking`` pseudo-tool and when friendly labels
    are disabled (callers fall back to their static default).  ``max_len``
    caps the total phrase length; Slack truncates its status line around 50
    characters, so the default stays just under that.
    """
    if not tool_name or tool_name == "_thinking":
        return None
    if not _friendly_tool_labels:
        return None

    verb = _TOOL_VERBS.get(tool_name)
    if verb:
        head = f"is {verb[0].lower()}{verb[1:]}"
    else:
        # Custom / plugin / MCP tools: generic but still informative.
        head = f"is using {tool_name}"

    phrase = head
    if args and verb and tool_name not in _TOOL_VERBS_NO_PREVIEW:
        preview = build_tool_preview(tool_name, args, max_len=None)
        if preview:
            # Previews can contain newlines (terminal commands); keep the
            # status to the first line.
            preview = preview.splitlines()[0].strip()
            phrase = f"{head}{tool_verb_connector(tool_name)}{preview}"

    if len(phrase) > max_len - 1:
        phrase = phrase[: max_len - 2].rstrip() + "…"
    else:
        phrase = phrase + "…"
    return phrase


def build_tool_label(tool_name: str, args: dict, max_len: int | None = None) -> str | None:
    """Build a human-phrased status label for a tool call.

    For built-in tools with a known verb (``web_search`` -> "Searching the
    web for ..."), returns the verb optionally followed by the argument
    preview.  For everything else (custom/plugin/MCP tools, or when friendly
    labels are disabled) returns the raw preview, so callers can use this as a
    drop-in replacement for :func:`build_tool_preview`.
    """
    if not _friendly_tool_labels:
        return build_tool_preview(tool_name, args, max_len=max_len)

    verb = _TOOL_VERBS.get(tool_name)
    if not verb:
        return build_tool_preview(tool_name, args, max_len=max_len)

    if tool_name in _TOOL_VERBS_NO_PREVIEW:
        return verb

    preview = build_tool_preview(tool_name, args, max_len=max_len)
    if not preview:
        return verb
    if tool_name in _TOOL_VERBS_FOR_CONNECTOR:
        return f"{verb} for {preview}"
    return f"{verb} {preview}"


# =========================================================================
# Inline diff previews for write actions
# =========================================================================

def extract_edit_diff(
    tool_name: str,
    result: str | None,
    *,
    function_args: dict | None = None,
) -> str | None:
    """Extract a unified diff from a file-edit tool result."""
    if tool_name == "patch" and result:
        data = safe_json_loads(result)
        if isinstance(data, dict):
            diff = data.get("diff")
            if isinstance(diff, str) and diff.strip():
                return diff

    return None


def _emit_inline_diff(diff_text: str, print_fn) -> bool:
    """Emit rendered diff text through the CLI's prompt_toolkit-safe printer."""
    if print_fn is None or not diff_text:
        return False
    try:
        print_fn("  ┊ review diff")
        for line in diff_text.rstrip("\n").splitlines():
            print_fn(line)
        return True
    except Exception:
        return False


def _render_inline_unified_diff(diff: str) -> list[str]:
    """Render unified diff lines in Hermes' inline transcript style."""
    rendered: list[str] = []
    from_file = None
    to_file = None

    for raw_line in diff.splitlines():
        if raw_line.startswith("--- "):
            from_file = raw_line[4:].strip()
            continue
        if raw_line.startswith("+++ "):
            to_file = raw_line[4:].strip()
            if from_file or to_file:
                rendered.append(f"{_diff_file()}{from_file or 'a/?'} → {to_file or 'b/?'}{_ANSI_RESET}")
            continue
        if raw_line.startswith("@@"):
            rendered.append(f"{_diff_hunk()}{raw_line}{_ANSI_RESET}")
            continue
        if raw_line.startswith("-"):
            rendered.append(f"{_diff_minus()}{raw_line}{_ANSI_RESET}")
            continue
        if raw_line.startswith("+"):
            rendered.append(f"{_diff_plus()}{raw_line}{_ANSI_RESET}")
            continue
        if raw_line.startswith(" "):
            rendered.append(f"{_diff_dim()}{raw_line}{_ANSI_RESET}")
            continue
        if raw_line:
            rendered.append(raw_line)

    return rendered


def _split_unified_diff_sections(diff: str) -> list[str]:
    """Split a unified diff into per-file sections."""
    sections: list[list[str]] = []
    current: list[str] = []

    for line in diff.splitlines():
        if line.startswith("--- ") and current:
            sections.append(current)
            current = [line]
            continue
        current.append(line)

    if current:
        sections.append(current)

    return ["\n".join(section) for section in sections if section]


def _summarize_rendered_diff_sections(
    diff: str,
    *,
    max_files: int = _MAX_INLINE_DIFF_FILES,
    max_lines: int = _MAX_INLINE_DIFF_LINES,
) -> list[str]:
    """Render diff sections while capping file count and total line count."""
    sections = _split_unified_diff_sections(diff)
    rendered: list[str] = []
    omitted_files = 0
    omitted_lines = 0

    for idx, section in enumerate(sections):
        if idx >= max_files:
            omitted_files += 1
            omitted_lines += len(_render_inline_unified_diff(section))
            continue

        section_lines = _render_inline_unified_diff(section)
        remaining_budget = max_lines - len(rendered)
        if remaining_budget <= 0:
            omitted_lines += len(section_lines)
            omitted_files += 1
            continue

        if len(section_lines) <= remaining_budget:
            rendered.extend(section_lines)
            continue

        rendered.extend(section_lines[:remaining_budget])
        omitted_lines += len(section_lines) - remaining_budget
        omitted_files += 1 + max(0, len(sections) - idx - 1)
        for leftover in sections[idx + 1:]:
            omitted_lines += len(_render_inline_unified_diff(leftover))
        break

    if omitted_files or omitted_lines:
        summary = f"… omitted {omitted_lines} diff line(s)"
        if omitted_files:
            summary += f" across {omitted_files} additional file(s)/section(s)"
        rendered.append(f"{_diff_hunk()}{summary}{_ANSI_RESET}")

    return rendered


def render_edit_diff_with_delta(
    tool_name: str,
    result: str | None,
    *,
    function_args: dict | None = None,
    print_fn=None,
) -> bool:
    """Render an edit diff inline without taking over the terminal UI."""
    diff = extract_edit_diff(
        tool_name,
        result,
        function_args=function_args,
    )
    if not diff:
        return False
    try:
        rendered_lines = _summarize_rendered_diff_sections(diff)
    except Exception as exc:
        logger.debug("Could not render inline diff: %s", exc)
        return False
    return _emit_inline_diff("\n".join(rendered_lines), print_fn)


_ERROR_SUFFIX_MAX_LEN = 48


def _trim_error(message: str) -> str:
    """Shrink an error message for an inline tool-status suffix."""
    message = message.strip()
    if "File not found:" in message:
        _, _, tail = message.partition("File not found:")
        tail = tail.strip()
        if "/" in tail:
            message = f"File not found: {tail.rsplit('/', 1)[-1]}"
    if len(message) > _ERROR_SUFFIX_MAX_LEN:
        message = message[: _ERROR_SUFFIX_MAX_LEN - 3] + "..."
    return message


def _detect_tool_failure(tool_name: str, result: str | None) -> tuple[bool, str]:
    """Inspect a tool result string for signs of failure.

    Returns ``(is_failure, suffix)`` where *suffix* is a short informational
    tag like ``" [exit 1]"`` for terminal failures, ``" [full]"`` for memory
    overflow, or a trimmed error message (``" [File not found: foo.py]"``).
    On success returns ``(False, "")``.
    """
    if result is None:
        return False, ""
    if file_mutation_result_landed(tool_name, result):
        return False, ""

    data = safe_json_loads(result)

    # Terminal: non-zero exit code is the canonical failure signal.
    if tool_name == "terminal":
        if isinstance(data, dict):
            exit_code = data.get("exit_code")
            if exit_code is not None and exit_code != 0:
                err_msg = data.get("error")
                if err_msg:
                    return True, f" [{_trim_error(str(err_msg))}]"
                return True, f" [exit {exit_code}]"
        return False, ""

    # Memory: distinguish "store full" from real errors.
    if tool_name == "memory":
        if isinstance(data, dict):
            if data.get("success") is False and "exceed the limit" in data.get("error", ""):
                return True, " [full]"

    # Structured error in JSON result (any tool that surfaces {"error": ...}).
    if isinstance(data, dict):
        err = data.get("error") or data.get("message")
        if err and (data.get("success") is False or "error" in data):
            return True, f" [{_trim_error(str(err))}]"

    # Generic heuristic for non-terminal tools
    # Multimodal tool results (dicts with _multimodal=True) are not strings —
    # treat them as successes since failures would be JSON-encoded strings.
    if not isinstance(result, str):
        return False, ""
    lower = result[:500].lower()
    if '"error"' in lower or '"failed"' in lower or result.startswith("Error"):
        return True, " [error]"

    return False, ""


def _get_cute_tool_message(
    tool_name: str, args: dict, duration: float, result: str | None = None,
) -> str:
    """Generate a formatted tool completion line for CLI quiet mode.

    Format: ``| {emoji} {verb:9} {detail}  {duration}``

    When *result* is provided the line is checked for failure indicators.
    Failed tool calls get a red prefix and an informational suffix.
    """
    args = redact_tool_args_for_display(tool_name, args) or args
    dur = f"{duration:.1f}s"
    is_failure, failure_suffix = _detect_tool_failure(tool_name, result)
    skin_prefix = get_skin_tool_prefix()

    def _trunc(s, n=40):
        s = str(s)
        if _tool_preview_max_len == 0:
            return s  # no limit
        limit = _tool_preview_max_len
        return (s[:limit-3] + "...") if len(s) > limit else s

    def _path(p, n=35):
        p = str(p)
        if _tool_preview_max_len == 0:
            return p  # no limit
        limit = _tool_preview_max_len
        return ("..." + p[-(limit-3):]) if len(p) > limit else p

    def _wrap(line: str) -> str:
        """Apply skin tool prefix and failure suffix."""
        if skin_prefix != "┊":
            line = line.replace("┊", skin_prefix, 1)
        if not is_failure:
            return line
        return f"{line}{failure_suffix}"

    if tool_name == "web_search":
        return _wrap(f"┊ 🔍 search    {_trunc(args.get('query', ''), 42)}  {dur}")
    if tool_name == "web_extract":
        urls = args.get("urls", [])
        if urls:
            url = _display_url(urls[0] if isinstance(urls, list) else urls)
            if not url:
                return _wrap(f"┊ 📄 fetch     pages  {dur}")
            domain = url.replace("https://", "").replace("http://", "").split("/")[0]
            extra = f" +{len(urls)-1}" if isinstance(urls, list) and len(urls) > 1 else ""
            return _wrap(f"┊ 📄 fetch     {_trunc(domain, 35)}{extra}  {dur}")
        return _wrap(f"┊ 📄 fetch     pages  {dur}")
    if tool_name == "terminal":
        return _wrap(f"┊ 💻 $         {_trunc(build_tool_preview(tool_name, args) or args.get('command', ''), 42)}  {dur}")
    if tool_name == "process":
        action = args.get("action", "?")
        sid = args.get("session_id", "")[:12]
        labels = {"list": "ls processes", "poll": f"poll {sid}", "log": f"log {sid}",
                  "wait": f"wait {sid}", "kill": f"kill {sid}", "write": f"write {sid}", "submit": f"submit {sid}"}
        return _wrap(f"┊ ⚙️  proc      {labels.get(action, f'{action} {sid}')}  {dur}")
    if tool_name == "read_file":
        return _wrap(f"┊ 📖 read      {_trunc(build_tool_preview(tool_name, args) or args.get('path', ''), 42)}  {dur}")
    if tool_name == "write_file":
        return _wrap(f"┊ ✍️  write     {_path(args.get('path', ''))}  {dur}")
    if tool_name == "patch":
        return _wrap(f"┊ 🔧 patch     {_path(args.get('path', ''))}  {dur}")
    if tool_name == "search_files":
        pattern = _trunc(args.get("pattern", ""), 35)
        target = args.get("target", "content")
        verb = "find" if target == "files" else "grep"
        return _wrap(f"┊ 🔎 {verb:9} {pattern}  {dur}")
    if tool_name == "browser_navigate":
        url = args.get("url", "")
        domain = url.replace("https://", "").replace("http://", "").split("/")[0]
        return _wrap(f"┊ 🌐 navigate  {_trunc(domain, 35)}  {dur}")
    if tool_name == "browser_snapshot":
        mode = "full" if args.get("full") else "compact"
        return _wrap(f"┊ 📸 snapshot  {mode}  {dur}")
    if tool_name == "browser_click":
        return _wrap(f"┊ 👆 click     {args.get('ref', '?')}  {dur}")
    if tool_name == "browser_type":
        return _wrap(f"┊ ⌨️  type      \"{_trunc(args.get('text', ''), 30)}\"  {dur}")
    if tool_name == "browser_scroll":
        d = args.get("direction", "down")
        arrow = {"down": "↓", "up": "↑", "right": "→", "left": "←"}.get(d, "↓")
        return _wrap(f"┊ {arrow}  scroll    {d}  {dur}")
    if tool_name == "browser_back":
        return _wrap(f"┊ ◀️  back      {dur}")
    if tool_name == "browser_press":
        return _wrap(f"┊ ⌨️  press     {args.get('key', '?')}  {dur}")
    if tool_name == "browser_get_images":
        return _wrap(f"┊ 🖼️  images    extracting  {dur}")
    if tool_name == "browser_vision":
        return _wrap(f"┊ 👁️  vision    analyzing page  {dur}")
    if tool_name == "todo":
        todos_arg = args.get("todos")
        merge = args.get("merge", False)
        # Parse result for completion progress
        total = 0
        done = 0
        if result:
            try:
                data = safe_json_loads(result)
                if data:
                    s = data.get("summary", {})
                    total = s.get("total", 0)
                    done = s.get("completed", 0)
            except Exception:
                pass
        if todos_arg is None:
            if total > 0:
                return _wrap(f"┊ 📋 plan      {done}/{total} task(s)  {dur}")
            return _wrap(f"┊ 📋 plan      reading tasks  {dur}")
        elif merge:
            if total > 0 and done > 0:
                return _wrap(f"┊ 📋 plan      update {done}/{total} ✓  {dur}")
            return _wrap(f"┊ 📋 plan      update {len(todos_arg)} task(s)  {dur}")
        else:
            if total > 0 and done > 0:
                return _wrap(f"┊ 📋 plan      {done}/{total} task(s)  {dur}")
            return _wrap(f"┊ 📋 plan      {len(todos_arg)} task(s)  {dur}")
    if tool_name == "session_search":
        return _wrap(f"┊ 🔍 recall    \"{_trunc(args.get('query', ''), 35)}\"  {dur}")
    if tool_name == "memory":
        action = args.get("action", "?")
        target = args.get("target", "")
        if action == "add":
            return _wrap(f"┊ 🧠 memory    +{target}: \"{_trunc(args.get('content', ''), 30)}\"  {dur}")
        elif action == "replace":
            old = args.get("old_text") or ""
            old = old if old else "<missing old_text>"
            return _wrap(f"┊ 🧠 memory    ~{target}: \"{_trunc(old, 20)}\"  {dur}")
        elif action == "remove":
            old = args.get("old_text") or ""
            old = old if old else "<missing old_text>"
            return _wrap(f"┊ 🧠 memory    -{target}: \"{_trunc(old, 20)}\"  {dur}")
        return _wrap(f"┊ 🧠 memory    {action}  {dur}")
    if tool_name == "skills_list":
        return _wrap(f"┊ 📚 skills    list {args.get('category', 'all')}  {dur}")
    if tool_name == "skill_view":
        label = args.get("name", "")
        file_path = args.get("file_path")
        if file_path:
            label = f"{label} → {file_path}" if label else str(file_path)
        return _wrap(f"┊ 📚 skill     {_trunc(label, 44)}  {dur}")
    if tool_name == "image_generate":
        return _wrap(f"┊ 🎨 create    {_trunc(args.get('prompt', ''), 35)}  {dur}")
    if tool_name == "text_to_speech":
        return _wrap(f"┊ 🔊 speak     {_trunc(args.get('text', ''), 30)}  {dur}")
    if tool_name == "vision_analyze":
        return _wrap(f"┊ 👁️  vision    {_trunc(args.get('question', ''), 30)}  {dur}")
    if tool_name == "send_message":
        return _wrap(f"┊ 📨 send      {args.get('target', '?')}: \"{_trunc(args.get('message', ''), 25)}\"  {dur}")
    if tool_name == "delegate_task":
        tasks = args.get("tasks")
        if tasks and isinstance(tasks, list):
            task_count, goals = _delegate_task_goal_parts(tasks, per_goal_len=30)
            detail = " | ".join(goals) if goals else "parallel"
            count_label = task_count or len(tasks)
            return _wrap(f"┊ 🔀 delegate  {count_label}x: {_trunc(detail, 35)}  {dur}")
        return _wrap(f"┊ 🔀 delegate  {_trunc(args.get('goal', ''), 35)}  {dur}")

    preview = build_tool_preview(tool_name, args) or ""
    return _wrap(f"┊ ⚡ {tool_name[:9]:9} {_trunc(preview, 35)}  {dur}")


def get_cute_tool_message(
    tool_name: str, args: dict, duration: float, result: str | None = None,
) -> str:
    """Render a completion label without letting cosmetic failures escape."""
    try:
        return _get_cute_tool_message(tool_name, args, duration, result=result)
    except Exception as exc:  # noqa: BLE001 — display must never abort a turn
        logger.debug("Tool completion label failed for %s: %s", tool_name, exc)
        safe_name = tool_name[:9] if isinstance(tool_name, str) and tool_name else "tool"
        safe_duration = f"{duration:.1f}s" if isinstance(duration, (int, float)) else "done"
        return f"┊ ⚡ {safe_name:9} completed  {safe_duration}"


# =========================================================================
# Honcho session line (one-liner with clickable OSC 8 hyperlink)
# =========================================================================
