"""Focused parity tests for the advisory SKILL.md linter."""

from __future__ import annotations

from tools.skill_linter import ERROR, WARNING, format_findings, has_errors, lint_content


CLEAN = """---
name: my-skill
description: Search papers by keyword, author, or ID.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [research]
    related_skills: []
---

# My Skill

## When to Use
- When the user wants research.

Use `read_file` to load a source.
"""


def _rules(content: str):
    return {finding.rule for finding in lint_content(content)}


def test_clean_skill_has_no_findings():
    assert lint_content(CLEAN) == []


def test_linter_flags_conventions_without_blocking_content():
    content = CLEAN.replace(
        "Search papers by keyword, author, or ID.", "A powerful helper."
    ).replace("Use `read_file` to load a source.", "Use `grep` to find a source.")
    findings = lint_content(content)
    assert {"description-marketing", "shell-utility-reference"} <= _rules(
        content
    )
    assert all(finding.severity == WARNING for finding in findings)
    assert not has_errors(findings)


def test_linter_rejects_bad_name_format():
    findings = lint_content(CLEAN.replace("name: my-skill", "name: Bad Skill!"))
    assert "name-format" in _rules(CLEAN.replace("name: my-skill", "name: Bad Skill!"))
    assert has_errors(findings)
    assert any(finding.severity == ERROR for finding in findings)


def test_linter_detects_dangling_references_and_forbidden_files(tmp_path):
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "README.md").write_text("# noise", encoding="utf-8")
    findings = lint_content(
        CLEAN + "\nSee references/missing.md for details.\n",
        skill_dir=skill_dir,
    )
    rules = {finding.rule for finding in findings}
    assert {"dangling-reference", "forbidden-file"} <= rules


def test_format_findings_is_stable():
    output = format_findings(lint_content(CLEAN.replace("name: my-skill", "name: BAD")))
    assert "[name-format]" in output
