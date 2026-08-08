#!/usr/bin/env python3
"""Generate agent-readable indexes from the documentation present on disk.

The generator has no network inputs. Every entry in ``llms.txt`` and
``llms-full.txt`` corresponds to a Markdown or MDX page under ``website/docs``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
WEBSITE_DIRECTORY = SCRIPT_DIRECTORY.parent
DOCS_DIRECTORY = WEBSITE_DIRECTORY / "docs"
STATIC_DIRECTORY = WEBSITE_DIRECTORY / "static"

SITE_BASE = "https://ykoh42.github.io/async-hermes-agent"
REPOSITORY_URL = "https://github.com/ykoh42/async-hermes-agent"

SECTION_LABELS = {
    "": "Overview",
    "getting-started": "Getting Started",
    "user-guide": "User Guide",
    "integrations": "Integrations",
    "guides": "Guides",
    "developer-guide": "Developer Guide",
    "reference": "Reference",
}
SECTION_ORDER = {section: index for index, section in enumerate(SECTION_LABELS)}

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
FRONTMATTER_FIELD_RE = re.compile(
    r"^(title|description|slug):\s*(.*?)\s*$",
    re.MULTILINE,
)
HEADING_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Document:
    relative_path: Path
    route: str
    section: str
    title: str
    description: str
    body: str


def _clean_frontmatter_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _read_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    metadata = {
        key: _clean_frontmatter_value(value)
        for key, value in FRONTMATTER_FIELD_RE.findall(match.group(1))
    }
    return metadata, text[match.end():]


def _default_route(relative_path: Path) -> str:
    route = relative_path.with_suffix("").as_posix()
    if route == "index":
        return ""
    if route.endswith("/index"):
        return route.removesuffix("/index")
    return route


def _document_from_path(path: Path) -> Document:
    relative_path = path.relative_to(DOCS_DIRECTORY)
    metadata, body = _read_frontmatter(path.read_text(encoding="utf-8"))

    route = _default_route(relative_path)
    configured_slug = metadata.get("slug")
    if configured_slug:
        route = configured_slug.strip("/")

    heading = HEADING_RE.search(body)
    fallback_title = (
        heading.group(1)
        if heading
        else relative_path.stem.replace("-", " ").title()
    )
    section = "" if not route else relative_path.parts[0]

    return Document(
        relative_path=relative_path,
        route=route,
        section=section,
        title=metadata.get("title", fallback_title),
        description=metadata.get("description", ""),
        body=body,
    )


def discover_documents() -> list[Document]:
    paths = [
        path
        for path in DOCS_DIRECTORY.rglob("*")
        if path.is_file() and path.suffix in {".md", ".mdx"}
    ]
    documents = [_document_from_path(path) for path in paths]
    return sorted(
        documents,
        key=lambda document: (
            SECTION_ORDER.get(document.section, len(SECTION_ORDER)),
            document.route,
        ),
    )


def _document_url(document: Document) -> str:
    if not document.route:
        return f"{SITE_BASE}/"
    return f"{SITE_BASE}/{document.route}"


def emit_llms_index(documents: list[Document]) -> str:
    """Build the concise llms.txt navigation index."""
    lines = [
        "# Async Hermes Agent",
        "",
        (
            "> A native-async, library-focused distribution of Hermes Agent for "
            "services, tool-using agents, persistent sessions, and trajectory generation."
        ),
        "",
        (
            "Install: `uv pip install "
            f"\"git+{REPOSITORY_URL}.git\"`"
        ),
        "",
        f"Repository: {REPOSITORY_URL}",
        f"Documentation: {SITE_BASE}/",
        "",
    ]

    documents_by_section: dict[str, list[Document]] = defaultdict(list)
    for document in documents:
        documents_by_section[document.section].append(document)

    for section in sorted(
        documents_by_section,
        key=lambda value: (SECTION_ORDER.get(value, len(SECTION_ORDER)), value),
    ):
        label = SECTION_LABELS.get(section, section.replace("-", " ").title())
        lines.extend((f"## {label}", ""))
        for document in documents_by_section[section]:
            entry = f"- [{document.title}]({_document_url(document)})"
            if document.description:
                entry += f": {document.description}"
            lines.append(entry)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def emit_llms_full(documents: list[Document]) -> str:
    """Concatenate all current documentation pages in navigation order."""
    chunks = [
        "# Async Hermes Agent — Full Documentation\n\n",
        (
            "This file contains every Markdown and MDX page currently published in "
            "the Async Hermes Agent documentation.\n\n"
        ),
        f"Canonical site: {SITE_BASE}/\n\n",
        f"Short index: {SITE_BASE}/llms.txt\n\n",
        "---\n\n",
    ]

    for document in documents:
        chunks.append(
            f"<!-- source: website/docs/{document.relative_path.as_posix()} -->\n"
        )
        chunks.append(f"# {document.title}\n\n")
        chunks.append(document.body.rstrip())
        chunks.append("\n\n---\n\n")

    return "".join(chunks).rstrip() + "\n"


def main() -> None:
    documents = discover_documents()
    index = emit_llms_index(documents)
    full = emit_llms_full(documents)

    STATIC_DIRECTORY.mkdir(parents=True, exist_ok=True)
    index_path = STATIC_DIRECTORY / "llms.txt"
    full_path = STATIC_DIRECTORY / "llms-full.txt"
    index_path.write_text(index, encoding="utf-8")
    full_path.write_text(full, encoding="utf-8")

    print(f"Wrote {index_path} ({len(index):,} bytes, {len(documents)} pages)")
    print(f"Wrote {full_path} ({len(full):,} bytes, {len(documents)} pages)")


if __name__ == "__main__":
    main()
