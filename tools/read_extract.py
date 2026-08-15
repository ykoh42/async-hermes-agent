"""Stdlib document-to-text extraction for ``read_file``.

Supports Jupyter notebooks, DOCX, and XLSX without adding hard dependencies.
Malformed documents raise :class:`ExtractionError`; callers can then fall back to
normal text/binary handling.
"""

from __future__ import annotations

import asyncio
import io
import importlib
import json
import posixpath
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import aiofiles
import aiofiles.os

__all__ = [
    "EXTRACTABLE_EXTENSIONS",
    "ExtractionError",
    "extract_document_bytes",
    "extract_document_text",
    "is_extractable_document",
]

EXTRACTABLE_EXTENSIONS = frozenset({".ipynb", ".docx", ".xlsx"})
# These formats are handled by the optional Rust-backed ``anydoc`` binding.
# Keep the binding optional: the retained library must remain useful without
# pulling a large native wheel into every installation.
ANYDOC_EXTENSIONS = frozenset(
    {
        ".doc",
        ".docm",
        ".ppt",
        ".pps",
        ".pot",
        ".pptx",
        ".pptm",
        ".ppsx",
        ".ppsm",
        ".xls",
        ".xlsm",
        ".xlsb",
        ".odt",
        ".ods",
        ".odp",
        ".rtf",
        ".epub",
        ".pdf",
    }
)
MAX_XLSX_BYTES = 50 * 1024 * 1024
# anydoc materializes the complete input in its Rust core. Bound the input
# before reading it so a read_file request cannot pin the process on a huge
# document; the ordinary read_file character budget applies after conversion.
MAX_ANYDOC_BYTES = 50 * 1024 * 1024
MAX_DOCUMENT_BYTES = 50 * 1024 * 1024
_MAX_XLSX_ROWS_PER_SHEET = 5000
_MAX_XLSX_COLS = 256

_NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_NS_S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


class ExtractionError(Exception):
    """Raised when a supported-looking document cannot be rendered as text."""


_ANYDOC_UNSET = object()
_anydoc_module: Any = _ANYDOC_UNSET


def _anydoc() -> Any | None:
    """Return the optional converter, without installing or prompting.

    The upstream CLI can lazily install this dependency.  That path is not
    part of the retained library surface and would perform synchronous
    package-manager work inside an async tool call, so the library only uses a
    converter that is already installed.  Keeping the sentinel unset after a
    failed import also allows a deployment that installs the extra at runtime
    to retry on a later request.
    """
    global _anydoc_module
    if _anydoc_module is not _ANYDOC_UNSET:
        return _anydoc_module
    try:
        _anydoc_module = importlib.import_module("anydoc")
    except Exception:
        return None
    return _anydoc_module


def _extension(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in EXTRACTABLE_EXTENSIONS:
        return ext
    return ext if ext in ANYDOC_EXTENSIONS and _anydoc() is not None else ""


def is_extractable_document(path: str) -> bool:
    return bool(_extension(path))


async def extract_document_text(path: str) -> str:
    ext = _extension(path)
    if not ext:
        raise ExtractionError(f"Unsupported document type: {path!r}")
    if ext in ANYDOC_EXTENSIONS:
        return await _extract_anydoc(path)
    try:
        async with aiofiles.open(path, "rb") as handle:
            data = await handle.read()
    except OSError as exc:
        raise ExtractionError(str(exc)) from exc
    if ext == ".ipynb":
        return _extract_notebook(data, Path(path).name)
    if ext == ".docx":
        return _extract_docx(data)
    if ext == ".xlsx":
        return _extract_xlsx(data)
    raise AssertionError(f"unhandled document extension: {ext}")


async def extract_document_bytes(data: bytes, path: str) -> str:
    """Extract a document already fetched across a file-backend boundary.

    The upstream API is synchronous because its file backend is synchronous.
    The retained runtime keeps the same public name and arguments but makes
    the boundary awaitable: parsing the already-materialized bytes remains
    CPU-only, while optional PDF coverage probing uses the native async
    subprocess/file path below.
    """
    if len(data) > MAX_DOCUMENT_BYTES:
        raise ExtractionError(
            f"Document too large to convert ({len(data):,} bytes, "
            f"limit is {MAX_DOCUMENT_BYTES:,})"
        )
    ext = _extension(path)
    if ext == ".ipynb":
        return _extract_notebook(data, Path(path).name)
    if ext == ".docx":
        return _extract_docx(data)
    if ext == ".xlsx":
        return _extract_xlsx(data)
    if ext in ANYDOC_EXTENSIONS:
        return await _extract_anydoc_bytes(data, path)
    raise ExtractionError(f"Unsupported document type: {path!r}")


async def _extract_anydoc(path: str) -> str:
    """Convert an optional anydoc format after an async bounded file read."""
    module = _anydoc()
    if module is None:
        raise ExtractionError(f"Unsupported document type: {path!r}")
    try:
        async with aiofiles.open(path, "rb") as handle:
            data = await handle.read(MAX_ANYDOC_BYTES + 1)
    except OSError as exc:
        raise ExtractionError(str(exc)) from exc
    if len(data) > MAX_ANYDOC_BYTES:
        raise ExtractionError(
            f"Document too large to convert ({len(data):,} bytes, "
            f"limit is {MAX_ANYDOC_BYTES:,})"
        )
    try:
        # firecrawl-anydoc 0.1.x exposes this bytes API.  Do not fall back to
        # to_markdown(path): that would reintroduce synchronous file I/O in the
        # retained async runtime.  Content detection handles container aliases
        # such as .docm/.xlsm; extension detection is the fallback for formats
        # without a reliable signature (notably CSV).
        convert = module.to_markdown_bytes
        detected = module.format_from_bytes(data)
        if detected is None:
            detected = module.format_from_extension(
                Path(path).suffix.lstrip(".").lower()
            )
        text = convert(data, format=detected)
    except OSError as exc:
        raise ExtractionError(str(exc)) from exc
    except Exception as exc:
        raise ExtractionError(f"{type(exc).__name__}: {exc}") from exc
    if not isinstance(text, str) or not text.strip():
        raise ExtractionError("Document contains no extractable text")
    text = text.rstrip("\n") + "\n"
    if Path(path).suffix.lower() == ".pdf":
        note = await _pdf_coverage_note(path)
        if note:
            # read_file paginates the extraction; a footer at the end of a
            # long document could otherwise be missed entirely by the model.
            text = note + text
    return text


async def _extract_anydoc_bytes(data: bytes, path: str) -> str:
    """Convert bytes through the optional anydoc binding without sync I/O."""
    module = _anydoc()
    if module is None:
        raise ExtractionError(f"Unsupported document type: {path!r}")
    if len(data) > MAX_ANYDOC_BYTES:
        raise ExtractionError(
            f"Document too large to convert ({len(data):,} bytes, "
            f"limit is {MAX_ANYDOC_BYTES:,})"
        )
    try:
        convert = module.to_markdown_bytes
        detected = module.format_from_bytes(data)
        if detected is None:
            detected = module.format_from_extension(
                Path(path).suffix.lstrip(".").lower()
            )
        text = convert(data, format=detected)
    except OSError as exc:
        raise ExtractionError(str(exc)) from exc
    except Exception as exc:
        raise ExtractionError(f"{type(exc).__name__}: {exc}") from exc
    if not isinstance(text, str) or not text.strip():
        raise ExtractionError("Document contains no extractable text")
    text = text.rstrip("\n") + "\n"
    if Path(path).suffix.lower() == ".pdf":
        note = await _pdf_coverage_note_from_bytes(data, display_path=path)
        if note:
            text = note + text
    return text


# ── Scanned-PDF coverage detection ────────────────────────────────────────
# anydoc can successfully convert a mostly-scanned PDF while silently
# omitting image-only pages.  A small native subprocess probe makes that data
# loss visible without adding a synchronous subprocess call to the async path.
PDF_EMPTY_PAGE_CHARS = 20
PDF_COVERAGE_MIN_EMPTY = 2
PDF_COVERAGE_MIN_RATIO = 0.2
PDF_COVERAGE_ABSOLUTE_EMPTY = 10
PDF_PAGE_SCAN_TIMEOUT = 20.0
PDF_GAP_MAP_MAX_ENTRIES = 20
_GAP_CONTEXT_CHARS = 60


async def _pdf_page_texts(path: str) -> list[str] | None:
    executable = await aiofiles.os.wrap(shutil.which)("pdftotext")
    if executable is None:
        return None
    try:
        process = await asyncio.create_subprocess_exec(
            executable,
            path,
            "-",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    async def _reap_after_abort() -> None:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await process.wait()

    try:
        stdout, _ = await asyncio.wait_for(
            process.communicate(), timeout=PDF_PAGE_SCAN_TIMEOUT
        )
    except asyncio.CancelledError:
        await asyncio.shield(_reap_after_abort())
        raise
    except (OSError, subprocess.SubprocessError, TimeoutError):
        await _reap_after_abort()
        return None
    if process.returncode != 0:
        return None
    pages = stdout.decode("utf-8", errors="replace").split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    return pages or None


async def _pdf_page_char_counts(path: str) -> list[int] | None:
    pages = await _pdf_page_texts(path)
    return None if pages is None else [len(page.strip()) for page in pages]


def _group_ranges(pages: list[int]) -> list[list[int]]:
    ranges: list[list[int]] = []
    for page in pages:
        if ranges and page == ranges[-1][1] + 1:
            ranges[-1][1] = page
        else:
            ranges.append([page, page])
    return ranges


def _page_ranges(pages: list[int]) -> str:
    """Render compact page ranges for the legacy coverage-note contract."""
    rendered = [
        f"{start}-{end}" if start != end else str(start)
        for start, end in _group_ranges(pages)
    ]
    if len(rendered) > 12:
        rendered = rendered[:12] + ["…"]
    return ", ".join(rendered)


def _gap_map(counts: list[int], texts: list[str], empty: list[int]) -> str:
    """Describe empty page ranges with the nearest preceding text heading."""
    ranges = _group_ranges(empty)
    lines: list[str] = []
    for start, end in ranges[:PDF_GAP_MAP_MAX_ENTRIES]:
        label = ""
        for previous in range(start - 2, -1, -1):
            if counts[previous] >= PDF_EMPTY_PAGE_CHARS:
                snippet = " ".join(texts[previous].split())[:_GAP_CONTEXT_CHARS]
                label = f' — after "{snippet}" (p{previous + 1})'
                break
        span = f"page {start}" if start == end else f"pages {start}-{end}"
        count = end - start + 1
        lines.append(
            f"  {span} ({count} page{'s' if count != 1 else ''}){label}"
        )
    if len(ranges) > PDF_GAP_MAP_MAX_ENTRIES:
        rest = ranges[PDF_GAP_MAP_MAX_ENTRIES:]
        rest_pages = sum(end - start + 1 for start, end in rest)
        lines.append(f"  … {len(rest)} more gaps ({rest_pages} pages)")
    return "\n".join(lines)


async def _pdf_coverage_note(
    path: str, display_path: str | None = None
) -> str:
    pages = await _pdf_page_texts(path)
    if not pages or len(pages) < 2:
        return ""
    counts = [len(page.strip()) for page in pages]
    empty = [index + 1 for index, count in enumerate(counts) if count < PDF_EMPTY_PAGE_CHARS]
    total = len(counts)
    if len(empty) < PDF_COVERAGE_MIN_EMPTY:
        return ""
    if (
        len(empty) / total < PDF_COVERAGE_MIN_RATIO
        and len(empty) < PDF_COVERAGE_ABSOLUTE_EMPTY
    ):
        return ""
    shown = display_path or path
    return (
        "[EXTRACTION COVERAGE WARNING: "
        f"{len(empty)} of {total} pages in this PDF yielded no text. "
        "Those pages are likely scanned images (or blank) — their content "
        "is MISSING from the extracted text below, even where section "
        "headers appear with empty bodies. Unreadable page ranges: "
        f"{_page_ranges(empty)}. Unreadable gaps, each labeled "
        "with the last text extracted before it:\n"
        f"{_gap_map(counts, pages, empty)}\n"
        "Decide which gaps you actually need — do NOT OCR or render "
        "everything. For the gaps that matter, render just that range with "
        f"`pdftoppm -jpeg -r 150 -f <first> -l <last> '{shown}' /tmp/page` "
        "and inspect each image with the vision_analyze tool, or use the "
        "ocr-and-documents skill (marker-pdf) for bulk OCR of large "
        "ranges.]\n"
    )


async def _pdf_coverage_note_from_bytes(
    data: bytes, display_path: str | None = None
) -> str:
    """Run the path-oriented PDF probe on backend-transferred bytes.

    ``pdftotext`` accepts a path rather than stdin for the per-page probe.
    Materialize a private temporary file through aiofiles so the retained
    async path never performs a blocking file write, then remove it even when
    cancellation or an invalid PDF interrupts the probe.
    """
    temp_path = ""
    try:
        async with aiofiles.tempfile.NamedTemporaryFile(
            suffix=".pdf", mode="wb", delete=False
        ) as handle:
            await handle.write(data)
            # aiofiles' broad stub includes integer file descriptors, but
            # NamedTemporaryFile(delete=False) returns a filesystem path.
            temp_path = str(handle.name)
        return await _pdf_coverage_note(temp_path, display_path=display_path)
    except (OSError, TypeError):
        return ""
    finally:
        if temp_path:
            try:
                await aiofiles.os.remove(temp_path)
            except OSError:
                pass


def _source_text(source) -> str:
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(item for item in source if isinstance(item, str))
    return ""


def _clean_stream_text(text: str) -> str:
    """Normalize ANSI output and carriage-return progress redraws."""
    from tools.ansi_strip import strip_ansi

    cleaned = strip_ansi(text).replace("\r\n", "\n")
    lines: list[str] = []
    for line in cleaned.split("\n"):
        frames = [frame for frame in line.split("\r") if frame]
        lines.append(frames[-1] if frames else "")
    return "\n".join(lines)


_MAX_NOTEBOOK_OUTPUT_CHARS = 20_000


def _base64_size(value: Any) -> int:
    if not isinstance(value, (str, list)):
        return 0
    raw = _source_text(value)
    clean = "".join(char for char in raw if char.isalnum() or char in "+/=")
    padding = min(2, len(clean) - len(clean.rstrip("=")))
    return max(0, (len(clean) * 3) // 4 - padding)


def _notebook_output_text(output: Any) -> str:
    """Render one notebook output without copying raw markup or base64 data."""
    if not isinstance(output, dict):
        return ""
    output_type = output.get("output_type")
    if output_type == "stream":
        body = _clean_stream_text(_source_text(output.get("text", "")))
        return body if body.strip() else ""
    if output_type in {"error", "pyerr"}:
        traceback = output.get("traceback")
        trace = ""
        if isinstance(traceback, list):
            trace = _clean_stream_text(
                "\n".join(item for item in traceback if isinstance(item, str))
            )
        header = f"Error: {output.get('ename', '')}: {output.get('evalue', '')}".rstrip(": ")
        return f"{header}\n{trace}".rstrip()
    if output_type not in {"execute_result", "display_data", "pyout"}:
        return ""

    data = output.get("data")
    if not isinstance(data, dict):
        data = {}
        if isinstance(output.get("text"), (str, list)):
            data["text/plain"] = output["text"]
        for key, mime in (
            ("png", "image/png"),
            ("jpeg", "image/jpeg"),
            ("svg", "image/svg+xml"),
            ("html", "text/html"),
        ):
            if key in output:
                data[mime] = output[key]
    if "application/vnd.jupyter.widget-view+json" in data:
        return "[interactive widget — omitted]"
    for mime in ("text/plain", "text/markdown"):
        if mime in data:
            body = _clean_stream_text(_source_text(data[mime]))
            if body.strip():
                return body
    for mime, value in data.items():
        if isinstance(mime, str) and mime.startswith("image/"):
            return f"[{mime} output — {_base64_size(value):,} bytes, omitted]"
    if "text/html" in data:
        return f"[text/html output — {len(_source_text(data['text/html'])):,} chars, omitted]"
    mimes = ", ".join(str(mime) for mime in data) or "unknown"
    return f"[{mimes} output — omitted]"


def _notebook_outputs(
    cell: dict[str, Any],
    cell_number: int,
    jq_pointer: str = "",
    filename: str = "",
) -> str:
    outputs = cell.get("outputs")
    if not isinstance(outputs, list):
        return ""
    blocks = [
        text
        for text in (_notebook_output_text(output) for output in outputs)
        if text
    ]
    if not blocks:
        return ""
    joined = "\n".join(blocks)
    if len(joined) > _MAX_NOTEBOOK_OUTPUT_CHARS:
        omitted = len(joined) - _MAX_NOTEBOOK_OUTPUT_CHARS
        hint = ""
        if jq_pointer and filename:
            hint = f" — full output: jq -r '{jq_pointer}' {filename}"
        joined = (
            joined[:_MAX_NOTEBOOK_OUTPUT_CHARS]
            + f"\n… [{omitted:,} output chars truncated{hint}]"
        )
    return joined


def _extract_notebook(data: bytes, filename: str = "") -> str:
    try:
        nb = json.loads(data.decode("utf-8", errors="replace"))
    except (ValueError, json.JSONDecodeError) as exc:
        raise ExtractionError(f"Not a valid notebook: {exc}") from exc
    if not isinstance(nb, dict):
        raise ExtractionError("Notebook root is not an object")

    raw_cells = nb.get("cells")
    if isinstance(raw_cells, list):
        cells = [
            (f".cells[{index}].outputs", cell)
            for index, cell in enumerate(raw_cells)
        ]
    else:
        cells = [
            (f".worksheets[{ws_index}].cells[{cell_index}].outputs", cell)
            for ws_index, ws in enumerate(nb.get("worksheets", []))
            if isinstance(ws, dict)
            for cell_index, cell in enumerate(ws.get("cells", []))
        ]
    if not cells:
        raise ExtractionError("Notebook contains no cells")

    counts = {"markdown": 0, "code": 0, "raw": 0}
    labels = {"markdown": "Markdown", "code": "Code", "raw": "Raw"}
    out: list[str] = []
    for jq_pointer, cell in cells:
        if not isinstance(cell, dict):
            continue
        typ = cell.get("cell_type")
        if typ not in labels:
            continue
        counts[typ] += 1
        suffix = f" {counts[typ]}" if typ != "raw" else ""
        out.extend(
            (
                f"# ── {labels[typ]} cell{suffix} ──",
                _source_text(cell.get("source", "")).rstrip("\n"),
                "",
            )
        )
        if typ == "code":
            rendered = _notebook_outputs(cell, counts[typ], jq_pointer, filename)
            if rendered:
                out.extend(
                    (
                        f"# ── Output (cell {counts[typ]}) ──",
                        rendered.rstrip("\n"),
                        "",
                    )
                )
    if not out:
        raise ExtractionError("Notebook contains no readable cells")
    return "\n".join(out).rstrip("\n") + "\n"


def _zip_xml(zf: zipfile.ZipFile, name: str) -> ET.Element:
    try:
        return ET.fromstring(zf.read(name))
    except KeyError as exc:
        raise ExtractionError(f"Missing {name}") from exc
    except ET.ParseError as exc:
        raise ExtractionError(f"Malformed XML in {name}: {exc}") from exc


def _extract_docx(data: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            root = _zip_xml(zf, "word/document.xml")
    except zipfile.BadZipFile as exc:
        raise ExtractionError(f"Not a valid DOCX: {exc}") from exc

    w = f"{{{_NS_W}}}"
    lines: list[str] = []
    for para in root.iter(f"{w}p"):
        buf: list[str] = []
        for node in para.iter():
            if node.tag == f"{w}t":
                buf.append(node.text or "")
            elif node.tag == f"{w}tab":
                buf.append("\t")
            elif node.tag in {f"{w}br", f"{w}cr"}:
                buf.append("\n")
        lines.extend("".join(buf).split("\n"))
    if not any(line.strip() for line in lines):
        raise ExtractionError("DOCX contains no extractable text")
    return "\n".join(lines).rstrip("\n") + "\n"


def _extract_xlsx(data: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = set(zf.namelist())
            shared = _shared_strings(zf, names)
            sheets = _workbook_sheets(zf)
            rels = _workbook_rels(zf, names)
            out: list[str] = []
            for name, state, rid in sheets:
                if state in {"hidden", "veryHidden"}:
                    continue
                part = _sheet_part(rels.get(rid, ""))
                if part not in names:
                    continue
                try:
                    rows = _sheet_rows(zf.read(part), shared)
                except ET.ParseError:
                    continue
                out.append(f"# ── Sheet: {name} ──")
                out.extend("\t".join(row) for row in rows)
                if not rows:
                    out.append("(empty)")
                out.append("")
    except zipfile.BadZipFile as exc:
        raise ExtractionError(f"Not a valid XLSX: {exc}") from exc

    if not out:
        raise ExtractionError("XLSX has no visible sheets with content")
    return "\n".join(out).rstrip("\n") + "\n"


def _shared_strings(zf: zipfile.ZipFile, names: set[str]) -> list[str]:
    if "xl/sharedStrings.xml" not in names:
        return []
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except ET.ParseError:
        return []
    s = f"{{{_NS_S}}}"
    return ["".join(t.text or "" for t in item.iter(f"{s}t")) for item in root.iter(f"{s}si")]


def _workbook_sheets(zf: zipfile.ZipFile) -> list[tuple[str, str, str]]:
    root = _zip_xml(zf, "xl/workbook.xml")
    s, r = f"{{{_NS_S}}}", f"{{{_NS_REL}}}"
    return [
        (sheet.get("name", "Sheet"), sheet.get("state", "visible"), sheet.get(f"{r}id", ""))
        for sheet in root.iter(f"{s}sheet")
    ]


def _workbook_rels(zf: zipfile.ZipFile, names: set[str]) -> dict[str, str]:
    rels_path = "xl/_rels/workbook.xml.rels"
    if rels_path not in names:
        return {}
    try:
        root = ET.fromstring(zf.read(rels_path))
    except ET.ParseError:
        return {}
    rel_tag = f"{{{_NS_PKG_REL}}}Relationship"
    return {rel.get("Id", ""): rel.get("Target", "") for rel in root.iter(rel_tag) if rel.get("Id")}


def _sheet_part(target: str) -> str:
    target = target.lstrip("/")
    return posixpath.normpath(target if target.startswith("xl/") else f"xl/{target}")


def _col_index(ref: str) -> int:
    idx = 0
    for ch in ref:
        if not ch.isalpha():
            break
        idx = idx * 26 + ord(ch.upper()) - ord("A") + 1
    return max(idx - 1, 0)


def _sheet_rows(xml_bytes: bytes, shared: list[str]) -> list[list[str]]:
    root = ET.fromstring(xml_bytes)
    s = f"{{{_NS_S}}}"
    rows: list[list[str]] = []
    for row in root.iter(f"{s}row"):
        if len(rows) >= _MAX_XLSX_ROWS_PER_SHEET:
            break
        cells: dict[int, str] = {}
        max_col = -1
        for cell in row.iter(f"{s}c"):
            col = _col_index(cell.get("r", "")) if cell.get("r") else max_col + 1
            if col >= _MAX_XLSX_COLS:
                continue
            cells[col] = _cell_value(cell, shared, s)
            max_col = max(max_col, col)
        rows.append([cells.get(i, "") for i in range(max_col + 1)] if max_col >= 0 else [])
    while rows and not any(value.strip() for value in rows[-1]):
        rows.pop()
    return rows


def _cell_value(cell: ET.Element, shared: list[str], s: str) -> str:
    value = cell.findtext(f"{s}v") or ""
    typ = cell.get("t", "")
    if typ == "s":
        try:
            return shared[int(value)]
        except (ValueError, IndexError):
            return ""
    if typ == "inlineStr":
        inline = cell.find(f"{s}is")
        return "" if inline is None else "".join(t.text or "" for t in inline.iter(f"{s}t"))
    if typ == "b":
        return "TRUE" if value.strip() in {"1", "true", "TRUE"} else "FALSE"
    if typ == "e":
        return value or "#ERROR"
    return value
