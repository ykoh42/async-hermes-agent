#!/usr/bin/env python3
"""
Tests for structured-document extraction in the read_file tool.

Covers .ipynb / .docx / .xlsx extraction (ported from Kilo-Org/kilocode
#10733, #10737, #10740) and the read_file_tool integration: pagination,
line-numbering, graceful fallback on malformed input, and hidden-sheet
omission.

Run with:  python -m pytest tests/tools/test_read_extract.py -v
"""

import json
import os
import tempfile
import unittest
import zipfile
from unittest.mock import AsyncMock, patch

from blockbuster import BlockBuster

from tools.read_extract import (
    ExtractionError,
    extract_document_bytes,
    extract_document_text,
    is_extractable_document,
)
from tools.file_tools import read_file_tool


# ---------------------------------------------------------------------------
# Fixture builders — construct minimal valid OOXML / notebook files.
# ---------------------------------------------------------------------------

def _write_notebook(path, cells, nbformat=4):
    nb = {"cells": cells, "metadata": {}, "nbformat": nbformat, "nbformat_minor": 5}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(nb, fh)


def _write_docx(path, document_xml):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", document_xml)


def _write_xlsx(path, *, workbook, rels, shared, sheets):
    """sheets: dict of part-name -> xml string."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/workbook.xml", workbook)
        z.writestr("xl/_rels/workbook.xml.rels", rels)
        if shared is not None:
            z.writestr("xl/sharedStrings.xml", shared)
        for part, xml in sheets.items():
            z.writestr(part, xml)


_NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_NS_S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


# ---------------------------------------------------------------------------
# is_extractable_document
# ---------------------------------------------------------------------------

class TestIsExtractable(unittest.TestCase):
    def test_recognized_extensions(self):
        self.assertTrue(is_extractable_document("a.ipynb"))
        self.assertTrue(is_extractable_document("/x/B.DOCX"))
        self.assertTrue(is_extractable_document("report.xlsx"))

    def test_unrecognized_extensions(self):
        self.assertFalse(is_extractable_document("a.py"))
        self.assertFalse(is_extractable_document("a.pdf"))
        self.assertFalse(is_extractable_document("a.txt"))

    def test_anydoc_extensions_track_optional_binding(self):
        """Optional formats are advertised only when the binding is present."""
        from tools import read_extract

        saved = read_extract._anydoc_module
        try:
            read_extract._anydoc_module = None
            self.assertFalse(is_extractable_document("a.pdf"))
            self.assertFalse(is_extractable_document("a.odt"))
            read_extract._anydoc_module = object()
            self.assertTrue(is_extractable_document("a.pdf"))
            self.assertTrue(is_extractable_document("a.epub"))
        finally:
            read_extract._anydoc_module = saved


class TestAnydocAbsent(unittest.IsolatedAsyncioTestCase):
    """The optional converter must fail clearly without changing stdlib paths."""

    def setUp(self):
        from tools import read_extract

        self.read_extract = read_extract
        self.saved = read_extract._anydoc_module
        read_extract._anydoc_module = None

    def tearDown(self):
        self.read_extract._anydoc_module = self.saved

    async def test_extract_raises_unsupported_without_anydoc(self):
        with self.assertRaises(ExtractionError):
            await extract_document_text("/tmp/whatever.pdf")

    async def test_stdlib_formats_unaffected(self):
        self.assertTrue(is_extractable_document("a.ipynb"))
        self.assertTrue(is_extractable_document("a.docx"))
        self.assertTrue(is_extractable_document("a.xlsx"))


class TestAnydocExtraction(unittest.IsolatedAsyncioTestCase):
    """Exercise the optional converter through the native async bytes path."""

    def setUp(self):
        from tools import read_extract

        self.read_extract = read_extract
        self.saved = read_extract._anydoc_module
        self.tmp = tempfile.mkdtemp(prefix="rex_anydoc_")

        class FakeAnydoc:
            @staticmethod
            def format_from_bytes(data):
                return None

            @staticmethod
            def format_from_extension(extension):
                return extension

            @staticmethod
            def to_markdown_bytes(data, *, format=None):
                if data.startswith(b"bad"):
                    raise ValueError("malformed document")
                return f"{format}: {data.decode('utf-8')}\n"

        read_extract._anydoc_module = FakeAnydoc

    def tearDown(self):
        import shutil

        self.read_extract._anydoc_module = self.saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    async def test_rtf_extracts_markdown(self):
        path = os.path.join(self.tmp, "doc.rtf")
        with open(path, "wb") as handle:
            handle.write(b"{\\rtf1 plain body}")
        text = await extract_document_text(path)
        self.assertEqual(text, "rtf: {\\rtf1 plain body}\n")

    async def test_malformed_file_raises_extraction_error(self):
        path = os.path.join(self.tmp, "junk.pdf")
        with open(path, "wb") as handle:
            handle.write(b"bad pdf")
        with self.assertRaises(ExtractionError):
            await extract_document_text(path)

    async def test_size_cap_rejects_before_converter(self):
        path = os.path.join(self.tmp, "large.pdf")
        with open(path, "wb") as handle:
            handle.write(b"0123456789")
        saved_cap = self.read_extract.MAX_ANYDOC_BYTES
        self.read_extract.MAX_ANYDOC_BYTES = 5
        try:
            with self.assertRaises(ExtractionError) as context:
                await extract_document_text(path)
        finally:
            self.read_extract.MAX_ANYDOC_BYTES = saved_cap
        self.assertIn("too large", str(context.exception))

    async def test_stdlib_docx_path_remains_authoritative(self):
        path = os.path.join(self.tmp, "d.docx")
        _write_docx(
            path,
            f'<w:document xmlns:w="{_NS_W}"><w:body>'
            "<w:p><w:r><w:t>hello</w:t></w:r></w:p>"
            "</w:body></w:document>",
        )
        text = await extract_document_text(path)
        self.assertEqual(text, "hello\n")

    async def test_bytes_path_uses_native_converter(self):
        text = await extract_document_bytes(b"{\\rtf1 plain body}", "/remote/doc.rtf")
        self.assertEqual(text, "rtf: {\\rtf1 plain body}\n")

    async def test_bytes_path_preserves_stdlib_docx_extractor(self):
        from io import BytesIO

        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("word/document.xml", (
                f'<w:document xmlns:w="{_NS_W}"><w:body>'
                "<w:p><w:r><w:t>remote body</w:t></w:r></w:p>"
                "</w:body></w:document>"
            ))
        text = await extract_document_bytes(buffer.getvalue(), "/remote/doc.docx")
        self.assertEqual(text, "remote body\n")

    async def test_bytes_path_enforces_document_cap_before_parsing(self):
        saved = self.read_extract.MAX_DOCUMENT_BYTES
        self.read_extract.MAX_DOCUMENT_BYTES = 3
        try:
            with self.assertRaises(ExtractionError) as context:
                await extract_document_bytes(b"1234", "/remote/doc.docx")
        finally:
            self.read_extract.MAX_DOCUMENT_BYTES = saved
        self.assertIn("too large", str(context.exception))


class TestPdfCoverageNote(unittest.IsolatedAsyncioTestCase):
    async def test_mostly_scanned_pdf_warns_with_ranges(self):
        from tools import read_extract

        pages = ["text " * 20, "", "", "text " * 20, "", ""]
        with patch.object(
            read_extract, "_pdf_page_texts", new=AsyncMock(return_value=pages)
        ):
            note = await read_extract._pdf_coverage_note("/tmp/report.pdf")
        self.assertIn("EXTRACTION COVERAGE WARNING", note)
        self.assertIn("4 of 6 pages", note)
        self.assertIn("2-3, 5-6", note)

    async def test_text_pdf_is_silent(self):
        from tools import read_extract

        pages = ["text " * 20] * 10
        with patch.object(
            read_extract, "_pdf_page_texts", new=AsyncMock(return_value=pages)
        ):
            self.assertEqual(
                await read_extract._pdf_coverage_note("/tmp/report.pdf"), ""
            )

    def test_page_ranges_compact(self):
        from tools.read_extract import _page_ranges

        self.assertEqual(_page_ranges([2, 3, 4, 7, 9, 10]), "2-4, 7, 9-10")


# ---------------------------------------------------------------------------
# Notebooks (.ipynb) — #10733
# ---------------------------------------------------------------------------

class TestNotebookExtraction(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rex_nb_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    async def test_markdown_and_code_in_order(self):
        p = os.path.join(self.tmp, "nb.ipynb")
        _write_notebook(p, [
            {"cell_type": "markdown", "source": ["# Title\n", "para"]},
            {"cell_type": "code", "source": "x = 1\nprint(x)",
             "outputs": [{"output_type": "stream", "text": ["1\n"]}],
             "execution_count": 1},
        ])
        text = await extract_document_text(p)
        self.assertIn("# Title", text)
        self.assertIn("print(x)", text)
        # Output payloads must NOT leak into the extracted text.
        self.assertNotIn("output_type", text)
        self.assertNotIn("execution_count", text)
        # Order preserved: markdown before code.
        self.assertLess(text.index("Title"), text.index("print(x)"))


    async def test_empty_cells_raises(self):
        p = os.path.join(self.tmp, "empty.ipynb")
        _write_notebook(p, [])
        with self.assertRaises(ExtractionError):
            await extract_document_text(p)

    async def test_code_outputs_are_rendered_without_raw_payloads(self):
        p = os.path.join(self.tmp, "outputs.ipynb")
        _write_notebook(
            p,
            [
                {
                    "cell_type": "code",
                    "source": "print('hello')",
                    "outputs": [
                        {"output_type": "stream", "text": "hello\rhello\n"},
                        {
                            "output_type": "display_data",
                            "data": {
                                "text/plain": "value: 42",
                                "image/png": "aGVsbG8=",
                            },
                        },
                    ],
                }
            ],
        )
        text = await extract_document_text(p)
        self.assertIn("# ── Output (cell 1) ──", text)
        self.assertIn("hello", text)
        self.assertIn("value: 42", text)
        self.assertNotIn("output_type", text)
        self.assertNotIn("aGVsbG8=", text)

    async def test_oversized_outputs_include_jq_retrieval_hint(self):
        from tools.read_extract import _MAX_NOTEBOOK_OUTPUT_CHARS

        p = os.path.join(self.tmp, "nb_big.ipynb")
        _write_notebook(
            p,
            [
                {"cell_type": "markdown", "source": "# intro"},
                {
                    "cell_type": "code",
                    "source": "spam()",
                    "outputs": [{
                        "output_type": "stream",
                        "text": "x" * (_MAX_NOTEBOOK_OUTPUT_CHARS + 5000),
                    }],
                },
            ],
        )
        text = await extract_document_text(p)
        self.assertIn("output chars truncated", text)
        self.assertIn(
            "— full output: jq -r '.cells[1].outputs' nb_big.ipynb]",
            text,
        )

    async def test_legacy_v3_oversized_outputs_include_jq_hint(self):
        from tools.read_extract import _MAX_NOTEBOOK_OUTPUT_CHARS

        p = os.path.join(self.tmp, "nb_v3_big.ipynb")
        notebook = {
            "worksheets": [{
                "cells": [
                    {"cell_type": "markdown", "source": "# intro"},
                    {
                        "cell_type": "code",
                        "source": "spam()",
                        "outputs": [{
                            "output_type": "stream",
                            "text": "x" * (_MAX_NOTEBOOK_OUTPUT_CHARS + 5000),
                        }],
                    },
                ],
            }],
            "nbformat": 3,
        }
        with open(p, "w", encoding="utf-8") as handle:
            json.dump(notebook, handle)
        text = await extract_document_text(p)
        self.assertIn("output chars truncated", text)
        self.assertIn(
            "— full output: jq -r '.worksheets[0].cells[1].outputs' nb_v3_big.ipynb]",
            text,
        )

    async def test_read_file_tool_extracts_notebook(self):
        p = os.path.join(self.tmp, "integrated.ipynb")
        _write_notebook(
            p,
            [{"cell_type": "code", "source": "value = 42"}],
        )

        blockbuster = BlockBuster()
        blockbuster.activate()
        try:
            result = json.loads(await read_file_tool(p))
        finally:
            blockbuster.deactivate()

        self.assertTrue(result["extracted_document"])
        self.assertIn("value = 42", result["content"])


# ---------------------------------------------------------------------------
# Word documents (.docx) — #10737
# ---------------------------------------------------------------------------

class TestDocxExtraction(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rex_docx_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _doc(self, body):
        return (f'<?xml version="1.0"?><w:document xmlns:w="{_NS_W}">'
                f'<w:body>{body}</w:body></w:document>')

    async def test_paragraphs_and_runs(self):
        p = os.path.join(self.tmp, "d.docx")
        _write_docx(p, self._doc(
            '<w:p><w:r><w:t>Hello </w:t></w:r><w:r><w:t>World</w:t></w:r></w:p>'
            '<w:p><w:r><w:t>Second</w:t></w:r></w:p>'))
        text = await extract_document_text(p)
        self.assertIn("Hello World", text)
        self.assertIn("Second", text)


    async def test_missing_document_xml_raises(self):
        p = os.path.join(self.tmp, "nodoc.docx")
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("other.xml", "<x/>")
        with self.assertRaises(ExtractionError):
            await extract_document_text(p)


# ---------------------------------------------------------------------------
# Excel workbooks (.xlsx) — #10740
# ---------------------------------------------------------------------------

class TestXlsxExtraction(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rex_xlsx_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build(self, path, *, include_hidden=True):
        r = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
        hidden_sheet = (f'<sheet name="Hidden" sheetId="2" state="hidden" '
                        f'xmlns:r="{r}" r:id="rId2"/>') if include_hidden else ""
        workbook = (
            f'<workbook xmlns="{_NS_S}" xmlns:r="{r}"><sheets>'
            f'<sheet name="Data" sheetId="1" r:id="rId1"/>{hidden_sheet}'
            f'</sheets></workbook>')
        rels = (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="x"/>'
            '<Relationship Id="rId2" Target="worksheets/sheet2.xml" Type="x"/>'
            '</Relationships>')
        shared = (f'<sst xmlns="{_NS_S}"><si><t>Name</t></si><si><t>Score</t></si>'
                  f'<si><t>Alice</t></si></sst>')
        sheet1 = (
            f'<worksheet xmlns="{_NS_S}"><sheetData>'
            '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
            '<row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2"><v>95</v></c></row>'
            '</sheetData></worksheet>')
        sheet2 = (f'<worksheet xmlns="{_NS_S}"><sheetData>'
                  '<row r="1"><c r="A1" t="str"><v>SECRETDATA</v></c></row>'
                  '</sheetData></worksheet>')
        _write_xlsx(path, workbook=workbook, rels=rels, shared=shared,
                    sheets={"xl/worksheets/sheet1.xml": sheet1,
                            "xl/worksheets/sheet2.xml": sheet2})

    async def test_visible_sheet_content(self):
        p = os.path.join(self.tmp, "wb.xlsx")
        self._build(p)
        text = await extract_document_text(p)
        self.assertIn("Data", text)        # sheet label
        self.assertIn("Name\tScore", text)  # shared-string header row
        self.assertIn("Alice\t95", text)    # string + numeric cells


    async def test_not_a_zip_raises(self):
        p = os.path.join(self.tmp, "bad.xlsx")
        with open(p, "wb") as fh:
            fh.write(b"nope")
        with self.assertRaises(ExtractionError):
            await extract_document_text(p)


class TestReadFileToolIntegration(unittest.IsolatedAsyncioTestCase):
    """Binary-document failures retain the actionable extractor reason."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rex_int_")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    async def test_corrupt_docx_surfaces_extraction_error(self):
        path = os.path.join(self.tmp, "bad.docx")
        with open(path, "wb") as handle:
            handle.write(b"not a zip")
        result = json.loads(await read_file_tool(path))
        self.assertIn("error", result)
        self.assertIn("extraction failed", result["error"].lower())
        self.assertIn("docx", result["error"].lower())

    async def test_oversized_anydoc_surfaces_size_error(self):
        from tools import read_extract

        saved_module = read_extract._anydoc_module
        saved_cap = read_extract.MAX_ANYDOC_BYTES
        read_extract._anydoc_module = object()
        read_extract.MAX_ANYDOC_BYTES = 5
        try:
            path = os.path.join(self.tmp, "big.pdf")
            with open(path, "wb") as handle:
                handle.write(b"0123456789")
            result = json.loads(await read_file_tool(path))
        finally:
            read_extract._anydoc_module = saved_module
            read_extract.MAX_ANYDOC_BYTES = saved_cap
        self.assertIn("error", result)
        self.assertIn("too large", result["error"].lower())
        self.assertNotIn("cannot read binary file", result["error"].lower())


if __name__ == "__main__":
    unittest.main()
