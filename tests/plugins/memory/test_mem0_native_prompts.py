"""Byte parity tests for vendored Mem0 additive-extraction prompts."""

from __future__ import annotations

import hashlib

from plugins.memory.mem0._native_prompts import (
    ADDITIVE_EXTRACTION_PROMPT,
    generate_additive_extraction_prompt,
)


def test_additive_prompt_matches_pinned_upstream_bytes():
    assert hashlib.sha256(ADDITIVE_EXTRACTION_PROMPT.encode()).hexdigest() == (
        "ad19187a37813ef77ee156e714c0650e6ec749e0264bdc07d499bc9b24115155"
    )


def test_additive_prompt_builder_matches_pinned_upstream_shape():
    assert generate_additive_extraction_prompt(
        existing_memories=[{"id": "0", "text": "known"}],
        new_messages="user: new\n",
        last_k_messages=[{"role": "user", "content": "prior"}],
        current_date="2026-08-09",
        timestamp="2026-08-08",
        custom_instructions="Keep context",
    ) == (
        "## Summary\n\n\n"
        "## Last k Messages\nuser: prior\n\n\n"
        "## Recently Extracted Memories\n[]\n\n"
        '## Existing Memories\n[{"id": "0", "text": "known"}]\n\n'
        "## New Messages\nuser: new\n\n\n"
        "## Observation Date\n2026-08-08\n\n"
        "## Current Date\n2026-08-09\n\n"
        "## Custom Instructions\nKeep context\n\n"
        "# Output:"
    )
