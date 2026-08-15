from __future__ import annotations

import asyncio
import inspect
import json
import mimetypes
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Awaitable, Callable

import aiofiles
import aiofiles.os

from agent.model_metadata import estimate_tokens_rough
from hermes_cli._subprocess_compat import windows_hide_flags
from hermes_cli.sizefmt import format_bytes

from abc import ABC, abstractmethod

IS_WINDOWS = os.name == "nt"

# ---------------------------------------------------------------------------
# Plugin context-reference provider API (Issue #26193)
# ---------------------------------------------------------------------------

BUILTIN_PREFIXES = frozenset({"diff", "staged", "file", "folder", "git", "url"})

_context_reference_providers: dict[str, ContextReferenceProvider] = {}


class ContextCompletionItem:
    """A single autocomplete result from a context reference provider."""

    __slots__ = ("text", "display", "meta")

    def __init__(self, text: str, display: str = "", meta: str = "") -> None:
        self.text = text
        self.display = display or text
        self.meta = meta


class ContextReferenceProvider(ABC):
    """Base class for plugin-registered @-prefix context reference providers.

    Plugins subclass this and register via
    ``PluginContext.register_context_reference()``.
    """

    prefix: str = ""  # e.g. "issue", "channel", "doc"
    description: str = ""  # shown in autocomplete meta column

    @abstractmethod
    async def autocomplete(self, query: str, *, limit: int = 10) -> list[ContextCompletionItem]:
        """Return autocomplete items for the given query string."""
        ...

    @abstractmethod
    async def expand(self, target: str) -> str | None:
        """Expand *target* to prompt content.  Return ``None`` to skip."""
        ...


def register_context_reference_provider(provider: ContextReferenceProvider) -> None:
    """Register a plugin context reference provider."""
    if not isinstance(provider, ContextReferenceProvider):
        raise TypeError("provider must be a ContextReferenceProvider instance")
    prefix = provider.prefix.lower().strip()
    if not prefix:
        raise ValueError("prefix must be a non-empty string")
    if prefix in BUILTIN_PREFIXES:
        raise ValueError(f"prefix '{prefix}' is reserved for built-in references")
    if prefix in _context_reference_providers:
        raise ValueError(f"prefix '{prefix}' is already registered")
    _context_reference_providers[prefix] = provider


def get_context_reference_providers() -> dict[str, ContextReferenceProvider]:
    """Return a snapshot of all registered plugin providers."""
    return dict(_context_reference_providers)


_QUOTED_REFERENCE_VALUE = r'(?:`[^`\n]+`|"[^"\n]+"|\'[^\'\n]+\')'
REFERENCE_PATTERN = re.compile(
    rf"(?<![\w/])@(?:(?P<simple>diff|staged)\b|(?P<kind>file|folder|git|url):(?P<value>{_QUOTED_REFERENCE_VALUE}(?::\d+(?:-\d+)?)?|\S+))"
)
# Plugin fallback pattern – catches any @<word>:<value> not handled by the
# built-in regex so that plugin-registered prefixes can be resolved.
_PLUGIN_REFERENCE_PATTERN = re.compile(
    rf"(?<![\w/])@(?P<kind>[a-zA-Z][a-zA-Z0-9_-]*):(?P<value>{_QUOTED_REFERENCE_VALUE}(?::\d+(?:-\d+)?)?|\S+)"
)

TRAILING_PUNCTUATION = ",.;!?"
_NEEDS_QUOTING = re.compile(r"""[\s()\[\]{}<>"'`]""")
_SENSITIVE_HOME_DIRS = (".ssh", ".aws", ".gnupg", ".kube", ".docker", ".azure", ".config/gh")
_SENSITIVE_HERMES_DIRS = (Path("skills") / ".hub",)
_SENSITIVE_HOME_FILES = (
    Path(".ssh") / "authorized_keys",
    Path(".ssh") / "id_rsa",
    Path(".ssh") / "id_ed25519",
    Path(".ssh") / "config",
    Path(".bashrc"),
    Path(".zshrc"),
    Path(".profile"),
    Path(".bash_profile"),
    Path(".zprofile"),
    Path(".netrc"),
    Path(".pgpass"),
    Path(".npmrc"),
    Path(".pypirc"),
)


@dataclass(frozen=True)
class ContextReference:
    raw: str
    kind: str
    target: str
    start: int
    end: int
    line_start: int | None = None
    line_end: int | None = None


@dataclass
class ContextReferenceResult:
    message: str
    original_message: str
    references: list[ContextReference] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    injected_tokens: int = 0
    expanded: bool = False
    blocked: bool = False


def format_reference_value(value: str) -> str:
    """Quote a reference value so ``REFERENCE_PATTERN`` reads it back whole.

    The unquoted alternative in the pattern is ``\\S+``, so a path containing a
    space parses as a truncated ref with the tail left behind as loose text.
    Mirrors ``formatRefValue`` in the desktop's directive-text.tsx.
    """
    if not _NEEDS_QUOTING.search(value):
        return value
    for quote in ("`", '"', "'"):
        if quote not in value:
            return f"{quote}{value}{quote}"
    return value


def parse_context_references(message: str) -> list[ContextReference]:
    refs: list[ContextReference] = []
    if not message:
        return refs

    for match in REFERENCE_PATTERN.finditer(message):
        simple = match.group("simple")
        if simple:
            refs.append(
                ContextReference(
                    raw=match.group(0),
                    kind=simple,
                    target="",
                    start=match.start(),
                    end=match.end(),
                )
            )
            continue

        kind = match.group("kind")
        value = _strip_trailing_punctuation(match.group("value") or "")
        line_start = None
        line_end = None
        target = _strip_reference_wrappers(value)

        if kind == "file":
            target, line_start, line_end = _parse_file_reference_value(value)

        refs.append(
            ContextReference(
                raw=match.group(0),
                kind=kind,
                target=target,
                start=match.start(),
                end=match.end(),
                line_start=line_start,
                line_end=line_end,
            )
        )

    # Second pass: resolve plugin-registered prefixes the built-in pattern missed
    if _context_reference_providers:
        for match in _PLUGIN_REFERENCE_PATTERN.finditer(message):
            kind = match.group("kind")
            if kind in BUILTIN_PREFIXES:
                continue
            # Skip if already captured by the built-in pattern
            if any(r.kind == kind and r.start == match.start() for r in refs):
                continue
            if kind in _context_reference_providers:
                value = _strip_trailing_punctuation(match.group("value") or "")
                refs.append(
                    ContextReference(
                        raw=match.group(0),
                        kind=kind,
                        target=_strip_reference_wrappers(value),
                        start=match.start(),
                        end=match.end(),
                    )
                )

    return refs


async def preprocess_context_references(
    message: str,
    *,
    cwd: str | Path,
    context_length: int,
    url_fetcher: Callable[[str], str | Awaitable[str]] | None = None,
    allowed_root: str | Path | None = None,
) -> ContextReferenceResult:
    """Expand references at the native async public boundary."""
    return await preprocess_context_references_async(
        message,
        cwd=cwd,
        context_length=context_length,
        url_fetcher=url_fetcher,
        allowed_root=allowed_root,
    )


async def preprocess_context_references_async(
    message: str,
    *,
    cwd: str | Path,
    context_length: int,
    url_fetcher: Callable[[str], str | Awaitable[str]] | None = None,
    allowed_root: str | Path | None = None,
) -> ContextReferenceResult:
    refs = parse_context_references(message)
    if not refs:
        return ContextReferenceResult(message=message, original_message=message)

    cwd_path = Path(cwd).expanduser().resolve()
    # Default to the current working directory so @ references cannot escape
    # the active workspace unless a caller explicitly widens the root.
    allowed_root_path = (
        Path(allowed_root).expanduser().resolve() if allowed_root is not None else cwd_path
    )
    warnings: list[str] = []
    blocks: list[str] = []
    injected_tokens = 0

    # Expand all references concurrently. Each _expand_reference is independent
    # (no shared state during expansion) — a message with several @url: refs
    # would otherwise pay one full web_extract round-trip per ref in series.
    # gather preserves positional order, so we reassemble warnings/blocks in the
    # original ref order exactly as the prior serial loop did; the token-budget
    # check below is unchanged (it runs once, after all refs are expanded).
    expanded = await asyncio.gather(
        *(
            _expand_reference(
                ref,
                cwd_path,
                url_fetcher=url_fetcher,
                allowed_root=allowed_root_path,
            )
            for ref in refs
        )
    )
    for warning, block in expanded:
        if warning:
            warnings.append(warning)
        if block:
            blocks.append(block)
            injected_tokens += estimate_tokens_rough(block)

    hard_limit = max(1, int(context_length * 0.50))
    soft_limit = max(1, int(context_length * 0.25))
    if injected_tokens > hard_limit:
        warnings.append(
            f"@ context injection refused: {injected_tokens} tokens exceeds the 50% hard limit ({hard_limit})."
        )
        return ContextReferenceResult(
            message=message,
            original_message=message,
            references=refs,
            warnings=warnings,
            injected_tokens=injected_tokens,
            expanded=False,
            blocked=True,
        )

    if injected_tokens > soft_limit:
        warnings.append(
            f"@ context injection warning: {injected_tokens} tokens exceeds the 25% soft limit ({soft_limit})."
        )

    # Leave the `@file:`/`@folder:` tokens where the user typed them. The token
    # IS the reference, not scaffolding around it: clients render each one as an
    # inline chip, so stripping them left a sentence with a hole in it ("review
    # and ship") and made the desktop re-derive the refs from the attached block
    # to show them as a detached list above the prose.
    final = message
    if warnings:
        final = f"{final}\n\n--- Context Warnings ---\n" + "\n".join(f"- {warning}" for warning in warnings)
    if blocks:
        final = f"{final}\n\n--- Attached Context ---\n\n" + "\n\n".join(blocks)

    return ContextReferenceResult(
        message=final.strip(),
        original_message=message,
        references=refs,
        warnings=warnings,
        injected_tokens=injected_tokens,
        expanded=bool(blocks or warnings),
        blocked=False,
    )


async def _expand_reference(
    ref: ContextReference,
    cwd: Path,
    *,
    url_fetcher: Callable[[str], str | Awaitable[str]] | None = None,
    allowed_root: Path | None = None,
) -> tuple[str | None, str | None]:
    try:
        if ref.kind == "file":
            return await _expand_file_reference(ref, cwd, allowed_root=allowed_root)
        if ref.kind == "folder":
            return await _expand_folder_reference(ref, cwd, allowed_root=allowed_root)
        if ref.kind == "diff":
            return await _expand_git_reference(ref, cwd, ["diff"], "git diff")
        if ref.kind == "staged":
            return await _expand_git_reference(
                ref,
                cwd,
                ["diff", "--staged"],
                "git diff --staged",
            )
        if ref.kind == "git":
            count = max(1, min(int(ref.target or "1"), 10))
            return await _expand_git_reference(
                ref,
                cwd,
                ["log", f"-{count}", "-p"],
                f"git log -{count} -p",
            )
        if ref.kind == "url":
            content = await _fetch_url_content(ref.target, url_fetcher=url_fetcher)
            if not content:
                return f"{ref.raw}: no content extracted", None
            return None, f"🌐 {ref.raw} ({estimate_tokens_rough(content)} tokens)\n{content}"
    except Exception as exc:
        return f"{ref.raw}: {exc}", None

    # Plugin-provided context references
    provider = _context_reference_providers.get(ref.kind)
    if provider is not None:
        try:
            plugin_content = await provider.expand(ref.target)
            if plugin_content is not None:
                return None, f"📌 {ref.raw} ({estimate_tokens_rough(plugin_content)} tokens)\n{plugin_content}"
        except Exception as exc:
            return f"{ref.raw}: plugin expansion error: {exc}", None

    return f"{ref.raw}: unsupported reference type", None


async def _expand_file_reference(
    ref: ContextReference,
    cwd: Path,
    *,
    allowed_root: Path | None = None,
) -> tuple[str | None, str | None]:
    path = _resolve_path(cwd, ref.target, allowed_root=allowed_root)
    await _ensure_reference_path_allowed(path)
    if not await aiofiles.os.path.exists(path):
        return f"{ref.raw}: file not found", None
    if not await aiofiles.os.path.isfile(path):
        return f"{ref.raw}: path is not a file", None
    if await _is_binary_file(path):
        # A binary file can't be inlined as text, but it IS on disk (the agent's
        # tools run where this resolves — the local cwd, or the staged copy in a
        # remote session workspace). Returning a bare "not supported" warning
        # with no content was a dead end: the model saw a failure and gave up
        # (told the user the file type wasn't supported). Instead, hand it an
        # actionable block — the path, type, size, and a nudge to use its tools —
        # so it can read/convert/view the file itself.
        return None, await _binary_reference_block(ref, path)

    async with aiofiles.open(path, encoding="utf-8") as handle:
        text = await handle.read()
    if ref.line_start is not None:
        lines = text.splitlines()
        start_idx = max(ref.line_start - 1, 0)
        end_idx = min(ref.line_end or ref.line_start, len(lines))
        text = "\n".join(lines[start_idx:end_idx])

    lang = _code_fence_language(path)
    label = ref.raw
    return None, f"📄 {label} ({estimate_tokens_rough(text)} tokens)\n```{lang}\n{text}\n```"


async def _expand_folder_reference(
    ref: ContextReference,
    cwd: Path,
    *,
    allowed_root: Path | None = None,
) -> tuple[str | None, str | None]:
    path = _resolve_path(cwd, ref.target, allowed_root=allowed_root)
    await _ensure_reference_path_allowed(path)
    if not await aiofiles.os.path.exists(path):
        return f"{ref.raw}: folder not found", None
    if not await aiofiles.os.path.isdir(path):
        return f"{ref.raw}: path is not a folder", None

    listing = await _build_folder_listing(path, cwd)
    return None, f"📁 {ref.raw} ({estimate_tokens_rough(listing)} tokens)\n{listing}"


async def _expand_git_reference(
    ref: ContextReference,
    cwd: Path,
    args: list[str],
    label: str,
) -> tuple[str | None, str | None]:
    _popen_kwargs = {"creationflags": windows_hide_flags()} if IS_WINDOWS else {}
    try:
        process = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=str(cwd),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **_popen_kwargs,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
    except TimeoutError:
        if process.returncode is None:
            process.kill()
            await process.wait()
        return f"{ref.raw}: git command timed out (30s)", None
    except (FileNotFoundError, OSError) as exc:
        return f"{ref.raw}: {exc}", None
    if process.returncode != 0:
        error = stderr.decode("utf-8", errors="replace").strip() or "git command failed"
        return f"{ref.raw}: {error}", None
    content = stdout.decode("utf-8", errors="replace").strip()
    if not content:
        content = "(no output)"
    return None, f"🧾 {label} ({estimate_tokens_rough(content)} tokens)\n```diff\n{content}\n```"


async def _fetch_url_content(
    url: str,
    *,
    url_fetcher: Callable[[str], str | Awaitable[str]] | None = None,
) -> str:
    fetcher = url_fetcher or _default_url_fetcher
    content = fetcher(url)
    if inspect.isawaitable(content):
        content = await content
    return str(content or "").strip()


async def _default_url_fetcher(url: str) -> str:
    from tools.web_tools import web_extract_tool

    raw = await web_extract_tool([url], format="markdown")
    payload = json.loads(raw)
    docs = payload.get("results", [])
    if not docs:
        return ""
    doc = docs[0]
    return str(doc.get("content") or doc.get("raw_content") or "").strip()


def _resolve_path(cwd: Path, target: str, *, allowed_root: Path | None = None) -> Path:
    path = Path(os.path.expanduser(target))
    if not path.is_absolute():
        path = cwd / path
    resolved = path.resolve()
    if allowed_root is not None:
        try:
            resolved.relative_to(allowed_root)
        except ValueError as exc:
            raise ValueError("path is outside the allowed workspace") from exc
    return resolved


async def _ensure_reference_path_allowed(path: Path) -> None:
    from hermes_constants import get_hermes_home
    home = Path(os.path.expanduser("~")).resolve()
    hermes_home = get_hermes_home().resolve()

    blocked_exact = {home / rel for rel in _SENSITIVE_HOME_FILES}
    blocked_exact.add(hermes_home / ".env")
    blocked_dirs = [home / rel for rel in _SENSITIVE_HOME_DIRS]
    blocked_dirs.extend(hermes_home / rel for rel in _SENSITIVE_HERMES_DIRS)

    if path in blocked_exact:
        raise ValueError("path is a sensitive credential file and cannot be attached")

    for blocked_dir in blocked_dirs:
        try:
            path.relative_to(blocked_dir)
        except ValueError:
            continue
        raise ValueError("path is a sensitive credential or internal Hermes path and cannot be attached")

    # Anchor to the canonical read deny-list (agent/file_safety.get_read_block_error),
    # the single source of truth used by the file/terminal read path. The narrow
    # list above predates that guard and never caught the real credential stores:
    # provider keys (auth.json), Anthropic OAuth tokens (.anthropic_oauth.json),
    # MCP OAuth material (mcp-tokens/), webhook HMAC secrets, and project-local
    # .env files. That gap matters because the gateway feeds UNTRUSTED remote
    # message text into reference expansion, so `@file:~/.hermes/auth.json` from a
    # chat peer would otherwise read the operator's keys straight into context.
    # Routing through the canonical guard closes the gap today and keeps this path
    # protected automatically whenever that deny-list grows.
    try:
        from agent.file_safety import get_read_block_error

        if await get_read_block_error(str(path)) is not None:
            raise ValueError(
                "path is a sensitive credential or internal Hermes path and cannot be attached"
            )
    except ValueError:
        raise
    except Exception:
        # Fail CLOSED on the security path. This guard exists specifically to
        # cover credential stores the narrow list above misses (auth.json,
        # .anthropic_oauth.json, mcp-tokens/, ...). If the canonical lookup
        # ever fails, silently falling through would re-open that exact hole —
        # the gateway feeds untrusted remote text here, so a probe could then
        # attach the operator's keys. Refuse instead: a spurious block on a
        # legitimate file is a recoverable annoyance; a leaked credential is not.
        raise ValueError(
            "path could not be verified against the credential deny-list and cannot be attached"
        )


def _strip_trailing_punctuation(value: str) -> str:
    stripped = value.rstrip(TRAILING_PUNCTUATION)
    while stripped.endswith((")", "]", "}")):
        closer = stripped[-1]
        opener = {")": "(", "]": "[", "}": "{"}[closer]
        if stripped.count(closer) > stripped.count(opener):
            stripped = stripped[:-1]
            continue
        break
    return stripped


def _strip_reference_wrappers(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "`\"'":
        return value[1:-1]
    return value


def _parse_file_reference_value(value: str) -> tuple[str, int | None, int | None]:
    quoted_match = re.match(
        r'^(?P<quote>`|"|\')(?P<path>.+?)(?P=quote)(?::(?P<start>\d+)(?:-(?P<end>\d+))?)?$',
        value,
    )
    if quoted_match:
        line_start = quoted_match.group("start")
        line_end = quoted_match.group("end")
        return (
            quoted_match.group("path"),
            int(line_start) if line_start is not None else None,
            int(line_end or line_start) if line_start is not None else None,
        )

    range_match = re.match(r"^(?P<path>.+?):(?P<start>\d+)(?:-(?P<end>\d+))?$", value)
    if range_match:
        line_start = int(range_match.group("start"))
        return (
            range_match.group("path"),
            line_start,
            int(range_match.group("end") or range_match.group("start")),
        )

    return _strip_reference_wrappers(value), None, None


async def _is_binary_file(path: Path) -> bool:
    mime, _ = mimetypes.guess_type(path.name)
    if mime and not mime.startswith("text/") and not any(
        path.name.endswith(ext) for ext in (".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".js", ".ts")
    ):
        return True
    async with aiofiles.open(path, "rb") as handle:
        chunk = await handle.read(4096)
    return b"\x00" in chunk


async def _build_folder_listing(path: Path, cwd: Path, limit: int = 200) -> str:
    lines = [f"{path.relative_to(cwd)}/"]
    entries = await _iter_visible_entries(path, cwd, limit=limit)
    for entry in entries:
        rel = entry.relative_to(cwd)
        indent = "  " * max(len(rel.parts) - len(path.relative_to(cwd).parts) - 1, 0)
        if await aiofiles.os.path.isdir(entry):
            lines.append(f"{indent}- {entry.name}/")
        else:
            meta = await _file_metadata(entry)
            lines.append(f"{indent}- {entry.name} ({meta})")
    if len(entries) >= limit:
        lines.append("- ...")
    return "\n".join(lines)


async def _iter_visible_entries(path: Path, cwd: Path, limit: int) -> list[Path]:
    rg_entries = await _rg_files(path, cwd, limit=limit)
    if rg_entries is not None:
        output: list[Path] = []
        seen_dirs: set[Path] = set()
        for rel in rg_entries:
            full = cwd / rel
            for parent in full.parents:
                if parent == cwd or parent in seen_dirs or path not in {parent, *parent.parents}:
                    continue
                seen_dirs.add(parent)
                output.append(parent)
            output.append(full)
        existing: list[Path] = []
        for candidate in output:
            if await aiofiles.os.path.exists(candidate):
                existing.append(candidate)
        typed: list[tuple[bool, str, Path]] = []
        for candidate in set(existing):
            typed.append(
                (
                    not await aiofiles.os.path.isdir(candidate),
                    str(candidate),
                    candidate,
                )
            )
        return [candidate for _is_file, _name, candidate in sorted(typed)]

    output = []
    pending = [path]
    while pending and len(output) < limit:
        current = pending.pop(0)
        try:
            names = sorted(await aiofiles.os.listdir(current))
        except OSError:
            continue
        for name in names:
            if name.startswith(".") or name == "__pycache__":
                continue
            candidate = current / name
            if await aiofiles.os.path.isdir(candidate):
                output.append(candidate)
                pending.append(candidate)
            else:
                output.append(candidate)
            if len(output) >= limit:
                return output
    return output


async def _rg_files(path: Path, cwd: Path, limit: int) -> list[Path] | None:
    _popen_kwargs = {"creationflags": windows_hide_flags()} if IS_WINDOWS else {}
    try:
        process = await asyncio.create_subprocess_exec(
            "rg",
            "--files",
            str(path.relative_to(cwd)),
            cwd=str(cwd),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **_popen_kwargs,
        )
        stdout, _stderr = await asyncio.wait_for(process.communicate(), timeout=10)
    except (TimeoutError, FileNotFoundError, OSError):
        if "process" in locals() and process.returncode is None:
            process.kill()
            await process.wait()
        return None
    if process.returncode != 0:
        return None
    files = [
        Path(line.strip())
        for line in stdout.decode("utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    return files[:limit]


def _agent_visible_path(path: Path) -> str:
    """Map a host path to the path the agent's tools can read in the active backend.

    Under a container backend (docker) the gateway host path dangles inside the
    sandbox — the container has its own filesystem and the host path is not
    mounted. Files staged into an auto-mounted cache dir (``images/``,
    ``attachments/``, ...) are translated to their in-container path via the
    existing ``tools.credential_files`` machinery (#76577). Falls back to the
    host path when the backend is local or translation is unavailable.
    """
    try:
        from tools.credential_files import to_agent_visible_cache_path

        return to_agent_visible_cache_path(str(path))
    except Exception:
        return str(path)


async def _binary_reference_block(ref: ContextReference, path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    mime = mime or "application/octet-stream"
    try:
        size = format_bytes(await aiofiles.os.path.getsize(path))
    except OSError:
        size = "unknown size"
    # Mount discovery is awaited before the synchronous path translation. This
    # keeps first-use attachment references equivalent to later references in
    # a running remote session.
    try:
        from tools.credential_files import get_cache_directory_mounts

        await get_cache_directory_mounts()
    except Exception:
        # Path translation is best-effort; retaining the host path preserves
        # local behavior if a backend cannot enumerate its mounts.
        pass
    return (
        f"📎 {ref.raw} ({mime}, {size}) — binary file, not inlined as text. "
        f"It is available on disk at `{_agent_visible_path(path)}`. Use your tools to work with it "
        f"(read or convert it, extract its text, or view/render it as needed); "
        f"do not tell the user the file type is unsupported."
    )


async def _file_metadata(path: Path) -> str:
    if await _is_binary_file(path):
        return f"{await aiofiles.os.path.getsize(path)} bytes"
    try:
        async with aiofiles.open(path, encoding="utf-8") as handle:
            line_count = (await handle.read()).count("\n") + 1
    except Exception:
        return f"{await aiofiles.os.path.getsize(path)} bytes"
    return f"{line_count} lines"


def _code_fence_language(path: Path) -> str:
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".json": "json",
        ".md": "markdown",
        ".sh": "bash",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".toml": "toml",
    }
    return mapping.get(path.suffix.lower(), "")
