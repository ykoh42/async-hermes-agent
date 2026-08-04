"""Tests for retained memory-context helpers."""

from agent.memory_manager import (
    build_memory_context_block,
    normalize_tool_schema,
    sanitize_context,
)


def test_normalize_tool_schema_accepts_bare_and_wrapped_functions():
    bare = {"name": "recall", "parameters": {"type": "object"}}
    wrapped = {"type": "function", "function": bare}

    assert normalize_tool_schema(bare) is bare
    assert normalize_tool_schema(wrapped) is bare


def test_normalize_tool_schema_rejects_missing_names():
    assert normalize_tool_schema(None) is None
    assert normalize_tool_schema("recall") is None
    assert normalize_tool_schema({}) is None
    assert normalize_tool_schema({"type": "function", "function": {}}) is None


def test_sanitize_context_removes_internal_memory_fences():
    raw = "before<memory-context>private</memory-context>after"

    assert sanitize_context(raw) == "beforeafter"


def test_build_memory_context_block_strips_existing_fenced_payload():
    wrapped = build_memory_context_block(
        "<memory-context>remember tea</memory-context>"
    )

    assert wrapped.count("<memory-context>") == 1
    assert wrapped.count("</memory-context>") == 1
    assert "remember tea" not in wrapped
