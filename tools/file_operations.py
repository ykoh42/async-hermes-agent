"""Pure helpers shared by the native async file-tool handlers.

The former shell-backed ``FileOperations`` hierarchy was a synchronous
compatibility layer for terminal backends that async-hermes-agent no longer
ships.  Keeping these small transformations here preserves the upstream file
location while leaving filesystem ownership with :mod:`tools.file_tools`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, List, Optional


MAX_LINES = 2000
MAX_LINE_LENGTH = 2000
DEFAULT_READ_OFFSET = 1
DEFAULT_READ_LIMIT = 2000
DEFAULT_SEARCH_OFFSET = 0
DEFAULT_SEARCH_LIMIT = 50

_UTF8_BOM = "\ufeff"


def _detect_line_ending(sample: str) -> str | None:
    """Return the dominant line ending in the first 4096 characters."""
    if not sample:
        return None
    head = sample[:4096]
    return "\r\n" if "\r\n" in head else "\n" if "\n" in head else None


def _normalize_line_endings(text: str, target: str) -> str:
    """Normalize newlines to *target* while preserving text content."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if target == "\n":
        return normalized
    if target == "\r\n":
        return normalized.replace("\n", target)
    return text


def _strip_bom(text: str) -> tuple[str, bool]:
    """Return ``(text_without_leading_bom, had_bom)``."""
    if text.startswith(_UTF8_BOM):
        return text[len(_UTF8_BOM):], True
    return text, False


def _has_bom(text: str | None) -> bool:
    """True if ``text`` begins with a UTF-8 BOM."""
    return bool(text) and text.startswith(_UTF8_BOM)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_read_pagination(
    offset: Any = DEFAULT_READ_OFFSET,
    limit: Any = DEFAULT_READ_LIMIT,
) -> tuple[int, int]:
    """Return schema-safe bounds for ``read_file`` pagination."""
    from tools.tool_output_limits import get_max_lines

    normalized_offset = max(1, _coerce_int(offset, DEFAULT_READ_OFFSET))
    normalized_limit = max(1, _coerce_int(limit, DEFAULT_READ_LIMIT))
    return normalized_offset, min(normalized_limit, get_max_lines())


def normalize_search_pagination(
    offset: Any = DEFAULT_SEARCH_OFFSET,
    limit: Any = DEFAULT_SEARCH_LIMIT,
) -> tuple[int, int]:
    """Return schema-safe bounds for ``search_files`` pagination."""
    normalized_offset = max(0, _coerce_int(offset, DEFAULT_SEARCH_OFFSET))
    normalized_limit = max(1, _coerce_int(limit, DEFAULT_SEARCH_LIMIT))
    return normalized_offset, normalized_limit


@dataclass
class ReadResult:
    """Result from reading a file."""

    content: str = ""
    total_lines: int = 0
    file_size: int = 0
    truncated: bool = False
    hint: Optional[str] = None
    is_binary: bool = False
    is_image: bool = False
    base64_content: Optional[str] = None
    mime_type: Optional[str] = None
    dimensions: Optional[str] = None
    error: Optional[str] = None
    similar_files: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            key: value
            for key, value in self.__dict__.items()
            if value is not None and value != []
        }


@dataclass
class WriteResult:
    """Result from writing a file."""

    bytes_written: int = 0
    dirs_created: bool = False
    verified: Optional[bool] = None
    lint: Optional[Dict[str, Any]] = None
    lsp_diagnostics: Optional[str] = None
    error: Optional[str] = None
    warning: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            key: value for key, value in self.__dict__.items() if value is not None
        }


@dataclass
class PatchResult:
    """Result from patching a file."""

    success: bool = False
    diff: str = ""
    files_modified: List[str] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    files_deleted: List[str] = field(default_factory=list)
    lint: Optional[Dict[str, Any]] = None
    lsp_diagnostics: Optional[str] = None
    error: Optional[str] = None
    no_change: bool = False
    note: Optional[str] = None

    def to_dict(self) -> dict:
        result: Dict[str, Any] = {"success": self.success}
        if self.no_change:
            result["no_change"] = True
        if self.note:
            result["note"] = self.note
        if self.diff:
            result["diff"] = self.diff
        if self.files_modified:
            result["files_modified"] = self.files_modified
        if self.files_created:
            result["files_created"] = self.files_created
        if self.files_deleted:
            result["files_deleted"] = self.files_deleted
        if self.lint:
            result["lint"] = self.lint
        if self.lsp_diagnostics:
            result["lsp_diagnostics"] = self.lsp_diagnostics
        if self.error:
            result["error"] = self.error
        return result


@dataclass
class SearchMatch:
    """A single content-search match."""

    path: str
    line_number: int
    content: str
    mtime: float = 0.0


@dataclass
class SearchResult:
    """Canonical Hermes search result."""

    matches: list[SearchMatch] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    total_count: int = 0
    truncated: bool = False
    limit_reason: str | None = None
    warning: str | None = None
    error: str | None = None

    _DENSIFY_MIN_MATCHES: ClassVar[int] = 5

    def _densify_matches(self) -> str | None:
        if len(self.matches) < self._DENSIFY_MIN_MATCHES:
            return None
        lines: list[str] = []
        current_path: str | None = None
        for match in self.matches:
            if match.path != current_path:
                lines.append(match.path)
                current_path = match.path
            lines.append(f"  {match.line_number}: {match.content.rstrip()}")
        return "\n".join(lines)

    def to_dict(self, densify: bool = False) -> dict:
        result: dict[str, object] = {"total_count": self.total_count}
        if self.matches:
            dense = self._densify_matches() if densify else None
            if dense is None:
                result["matches"] = [
                    {
                        "path": match.path,
                        "line": match.line_number,
                        "content": match.content,
                    }
                    for match in self.matches
                ]
            else:
                result["matches_format"] = (
                    "path-grouped: each file path on its own line, followed by "
                    "indented '<line>: <content>' rows for matches in that file"
                )
                result["matches_text"] = dense
        if self.files:
            result["files"] = self.files
        if self.counts:
            result["counts"] = self.counts
        if self.truncated:
            result["truncated"] = True
        if self.limit_reason:
            result["limit_reason"] = self.limit_reason
        if self.warning:
            result["warning"] = self.warning
        if self.error:
            result["error"] = self.error
        return result


@dataclass
class LintResult:
    """Result from linting a file."""

    success: bool = True
    skipped: bool = False
    output: str = ""
    message: str = ""

    def to_dict(self) -> dict:
        if self.skipped:
            return {"status": "skipped", "message": self.message}
        result = {"status": "ok" if self.success else "error", "output": self.output}
        if self.message:
            result["message"] = self.message
        return result


def _lint_json_inproc(content: str) -> tuple[bool, str]:
    """In-process JSON syntax check. Returns ``(ok, error_message)``."""
    import json

    try:
        json.loads(content)
        return True, ""
    except json.JSONDecodeError as exc:
        return False, (
            f"JSONDecodeError: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})"
        )
    except Exception as exc:  # noqa: BLE001 - any parse failure is a lint failure
        return False, f"{type(exc).__name__}: {exc}"


def _lint_yaml_inproc(content: str) -> tuple[bool, str]:
    """In-process YAML syntax check without constructing YAML values."""
    try:
        import yaml
    except ImportError:
        return True, "__SKIP__"
    try:
        for _event in yaml.parse(content):
            pass
        return True, ""
    except yaml.YAMLError as exc:
        return False, f"YAMLError: {exc}"
    except Exception as exc:  # noqa: BLE001 - any parse failure is a lint failure
        return False, f"{type(exc).__name__}: {exc}"


def _lint_toml_inproc(content: str) -> tuple[bool, str]:
    """In-process TOML syntax check using Python 3.11's ``tomllib``."""
    import tomllib

    try:
        tomllib.loads(content)
        return True, ""
    except Exception as exc:  # tomllib raises TOMLDecodeError (ValueError)
        return False, f"{type(exc).__name__}: {exc}"


def _lint_python_inproc(content: str) -> tuple[bool, str]:
    """In-process Python syntax check via ``ast.parse``."""
    import ast

    try:
        ast.parse(content)
        return True, ""
    except SyntaxError as exc:
        location = (
            f" (line {exc.lineno}, column {exc.offset})" if exc.lineno else ""
        )
        return False, f"{type(exc).__name__}: {exc.msg}{location}"
    except Exception as exc:  # noqa: BLE001 - any parse failure is a lint failure
        return False, f"{type(exc).__name__}: {exc}"


LINTERS_INPROC: dict[str, Callable[[str], tuple[bool, str]]] = {
    ".py": _lint_python_inproc,
    ".json": _lint_json_inproc,
    ".yaml": _lint_yaml_inproc,
    ".yml": _lint_yaml_inproc,
    ".toml": _lint_toml_inproc,
}

_FAIL_CLOSED_INPROC_EXTS = frozenset({".json", ".yaml", ".yml", ".toml"})

LINTERS = {
    ".py": "python -m py_compile {file} 2>&1",
    ".js": "node --check {file} 2>&1",
    ".ts": "npx tsc --noEmit {file} 2>&1",
    ".go": "go vet {file} 2>&1",
    ".rs": "rustfmt --check {file} 2>&1",
}

_LINTER_UNUSABLE_PATTERNS = {
    "npx": (
        "this is not the tsc command you are looking for",
        "could not determine executable to run",
        "not found in npm registry",
    ),
    "rustfmt": (
        "no input filename given",
        "error: not a workspace",
    ),
    "go": (
        "cannot find package",
        "go: cannot find main module",
    ),
}


def _looks_like_linter_unusable(base_cmd: str, output: str) -> bool:
    """Return whether output reports a tooling gap rather than a file error."""
    patterns = _LINTER_UNUSABLE_PATTERNS.get(base_cmd)
    if not patterns:
        return False
    lower = output.lower()
    return any(pattern in lower for pattern in patterns)


def _parse_search_context_line(line: str) -> tuple[str, int, str] | None:
    """Parse ``path-line-content`` while allowing ``-<digits>-`` in paths."""
    if not line or line == "--":
        return None
    match = None
    for candidate in re.finditer(r"-(\d+)-", line):
        match = candidate
    if match is None:
        return None
    path = line[:match.start()]
    if not path:
        return None
    return path, int(match.group(1)), line[match.end():]
