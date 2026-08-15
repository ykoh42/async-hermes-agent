"""Strict v2026.8.3 parity for retained core-tool registry contracts."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from unittest.mock import AsyncMock

import pytest


_UPSTREAM_SCHEMA_HASHES = {
    "write_file": "e3eee8862945e4e923460a83b4b98238a8675224a2c72365d1b3d08a9c06f846",
    "patch": "835e59b0ab3d32a199e975463a148bb2fd934b2482dcc17a22be4d44cefed761",
    "skill_manage": "d6aa0216264b0a62f7f9ec09abc553198fc8c209a950cfeda6c1c485179b3a83",
    "terminal": "b77004e9cd89882279ad870123cf424539f66f1320bacfd098120329a49f62e6",
    "text_to_speech": "fbdcf6e8969c36688e2dae75d213774c9217ea285f80153567a74a8262761c44",
}


def _normalized_schema_hash(schema: dict) -> str:
    """Hash a schema after replacing the profile-dependent Hermes home."""

    from hermes_constants import display_hermes_home

    home = display_hermes_home()

    def normalize(value):
        if isinstance(value, str):
            # Some tools may already be imported during collection, before the
            # hermetic profile fixture changes the active Hermes home.
            normalized = value.replace(home, "/HERMES").replace(
                "~/.hermes", "/HERMES"
            )
            normalized = re.sub(
                r"/[^\s`]*hermes-test-home-[^/\s`]+",
                "/HERMES",
                normalized,
            )
            return re.sub(
                r"/(?:[^\s`]+/)*hermes_test",
                "/HERMES",
                normalized,
            )
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        return value

    encoded = json.dumps(
        normalize(schema),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _parameter_contract(function) -> tuple[list[str], list[object]]:
    signature = inspect.signature(function)
    parameters = list(signature.parameters.values())
    return (
        [parameter.name for parameter in parameters],
        [parameter.default for parameter in parameters],
    )


def test_retained_schema_bytes_match_upstream_v2026_8_3():
    from tools import file_tools, skill_manager_tool, terminal_tool, tts_tool

    schemas = {
        "write_file": file_tools.WRITE_FILE_SCHEMA,
        "patch": file_tools.PATCH_SCHEMA,
        "skill_manage": skill_manager_tool.SKILL_MANAGE_SCHEMA,
        "terminal": terminal_tool.TERMINAL_SCHEMA,
        "text_to_speech": tts_tool.TTS_SCHEMA,
    }

    assert {
        name: _normalized_schema_hash(schema)
        for name, schema in schemas.items()
    } == _UPSTREAM_SCHEMA_HASHES


def test_public_coroutine_signatures_preserve_upstream_args_and_defaults():
    from tools import file_tools, skill_manager_tool, terminal_tool, tts_tool

    expected = {
        file_tools.write_file_tool: (
            ["path", "content", "task_id", "cross_profile", "session_id"],
            [inspect.Parameter.empty, inspect.Parameter.empty, "default", False, None],
        ),
        file_tools.patch_tool: (
            [
                "mode", "path", "old_string", "new_string", "replace_all",
                "patch", "task_id", "cross_profile", "session_id",
            ],
            ["replace", None, None, None, False, None, "default", False, None],
        ),
        skill_manager_tool.skill_manage: (
            [
                "action", "name", "content", "category", "file_path",
                "file_content", "old_string", "new_string", "replace_all",
                "absorbed_into", "task_id", "session_id",
            ],
            [
                inspect.Parameter.empty, inspect.Parameter.empty, None, None,
                None, None, None, None, False, None, None, None,
            ],
        ),
        terminal_tool.terminal_tool: (
            [
                "command", "background", "timeout", "task_id", "session_id",
                "force", "workdir", "pty", "notify_on_complete",
                "watch_patterns",
            ],
            [
                inspect.Parameter.empty, False, None, None, None, False, None,
                False, False, None,
            ],
        ),
        tts_tool.text_to_speech_tool: (
            ["text", "output_path", "speed", "instructions", "provider"],
            [inspect.Parameter.empty, None, None, None, None],
        ),
    }

    for function, contract in expected.items():
        assert inspect.iscoroutinefunction(function)
        assert _parameter_contract(function) == contract
        assert inspect.signature(function).return_annotation in {str, "str"}


def test_registry_handlers_and_checks_preserve_upstream_ownership():
    from tools import file_tools, skill_manager_tool, terminal_tool, tts_tool
    from tools.registry import registry

    expected = {
        "write_file": (
            file_tools.WRITE_FILE_SCHEMA,
            file_tools._handle_write_file,
            file_tools._check_file_reqs,
        ),
        "patch": (
            file_tools.PATCH_SCHEMA,
            file_tools._handle_patch,
            file_tools._check_file_reqs,
        ),
        "skill_manage": (
            skill_manager_tool.SKILL_MANAGE_SCHEMA,
            skill_manager_tool._handle_skill_manage,
            None,
        ),
        "terminal": (
            terminal_tool.TERMINAL_SCHEMA,
            terminal_tool._handle_terminal,
            terminal_tool.check_terminal_requirements,
        ),
        "text_to_speech": (
            tts_tool.TTS_SCHEMA,
            tts_tool._handle_text_to_speech,
            tts_tool.check_tts_requirements,
        ),
    }

    for name, (schema, handler, check_fn) in expected.items():
        entry = registry.get_entry(name)
        assert entry is not None
        assert entry.schema is schema
        assert entry.handler is handler
        assert entry.check_fn is check_fn
        assert inspect.iscoroutinefunction(handler)
        assert [*inspect.signature(handler).parameters] == ["args", "kw"]
        if check_fn is not None:
            assert inspect.iscoroutinefunction(check_fn)


@pytest.mark.asyncio
async def test_registry_definition_order_matches_upstream_sorted_name_order():
    from tools.registry import ToolRegistry

    registry = ToolRegistry()

    async def handler(_args, **_kwargs):
        return "{}"

    def schema(name):
        return {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        }

    registry.register(
        name="zeta",
        toolset="test",
        schema=schema("zeta"),
        handler=handler,
    )
    registry.register(
        name="alpha",
        toolset="test",
        schema=schema("alpha"),
        handler=handler,
    )

    definitions = await registry.get_definitions({"zeta", "alpha"})
    assert [item["function"]["name"] for item in definitions] == [
        "alpha",
        "zeta",
    ]


def test_toolsets_and_model_tools_expose_upstream_names():
    import model_tools
    import toolsets

    core = toolsets._HERMES_CORE_TOOLS
    for name in (
        "write_file", "patch", "skill_manage", "terminal", "text_to_speech",
    ):
        assert core.count(name) == 1
    assert core.index("text_to_speech") == core.index("browser_dialog") + 1
    assert toolsets.TOOLSETS["file"]["tools"] == [
        "read_file", "write_file", "patch", "search_files",
    ]
    assert toolsets.TOOLSETS["skills"]["tools"] == [
        "skills_list", "skill_view", "skill_manage",
    ]
    assert toolsets.TOOLSETS["terminal"]["tools"] == ["terminal", "process"]
    assert toolsets.TOOLSETS["tts"]["tools"] == ["text_to_speech"]
    assert model_tools._LEGACY_TOOLSET_MAP["file_tools"] == [
        "read_file", "write_file", "patch", "search_files",
    ]
    assert model_tools._LEGACY_TOOLSET_MAP["skills_tools"] == [
        "skills_list", "skill_view", "skill_manage",
    ]
    assert model_tools._LEGACY_TOOLSET_MAP["terminal_tools"] == ["terminal"]
    assert model_tools._LEGACY_TOOLSET_MAP["tts_tools"] == ["text_to_speech"]


@pytest.mark.asyncio
async def test_terminal_handler_forwards_upstream_raw_values(monkeypatch):
    from tools import terminal_tool

    execute = AsyncMock(return_value="terminal-result")
    monkeypatch.setattr(terminal_tool, "terminal_tool", execute)

    result = await terminal_tool._handle_terminal(
        {
            "command": None,
            "background": "raw-background",
            "pty": "raw-pty",
            "notify_on_complete": "raw-notify",
        },
        task_id="task",
        session_id="session",
    )

    assert result == "terminal-result"
    execute.assert_awaited_once_with(
        command=None,
        background="raw-background",
        timeout=None,
        task_id="task",
        session_id="session",
        workdir=None,
        pty="raw-pty",
        notify_on_complete="raw-notify",
        watch_patterns=None,
    )


@pytest.mark.asyncio
async def test_file_handlers_forward_upstream_public_contract(monkeypatch):
    from tools import file_tools

    write = AsyncMock(return_value="write-result")
    patch = AsyncMock(return_value="patch-result")
    monkeypatch.setattr(file_tools, "write_file_tool", write)
    monkeypatch.setattr(file_tools, "patch_tool", patch)

    assert await file_tools._handle_write_file(
        {"path": "target", "content": "data", "cross_profile": "yes"},
        task_id="",
        session_id="session",
    ) == "write-result"
    write.assert_awaited_once_with(
        path="target",
        content="data",
        task_id="default",
        cross_profile=True,
        session_id="session",
    )

    assert await file_tools._handle_patch(
        {
            "mode": "patch",
            "path": "target",
            "old_string": "old",
            "new_string": "new",
            "replace_all": "raw-replace-all",
            "patch": "body",
            "cross_profile": "yes",
        },
        task_id="task",
        session_id="session",
    ) == "patch-result"
    patch.assert_awaited_once_with(
        mode="patch",
        path="target",
        old_string="old",
        new_string="new",
        replace_all="raw-replace-all",
        patch="body",
        task_id="task",
        cross_profile=True,
        session_id="session",
    )


@pytest.mark.asyncio
async def test_named_handlers_preserve_upstream_lambda_forwarding(monkeypatch):
    from tools import skill_manager_tool, tts_tool

    skill_manage = AsyncMock(return_value="skill-result")
    text_to_speech = AsyncMock(return_value="tts-result")
    monkeypatch.setattr(skill_manager_tool, "skill_manage", skill_manage)
    monkeypatch.setattr(tts_tool, "text_to_speech_tool", text_to_speech)

    assert await skill_manager_tool._handle_skill_manage(
        {"action": "edit", "name": "sample"},
        ignored="value",
    ) == "skill-result"
    skill_manage.assert_awaited_once_with(
        action="edit",
        name="sample",
        content=None,
        category=None,
        file_path=None,
        file_content=None,
        old_string=None,
        new_string=None,
        replace_all=False,
        absorbed_into=None,
        task_id=None,
        session_id=None,
    )

    assert await tts_tool._handle_text_to_speech(
        {"text": "hello"},
        ignored="value",
    ) == "tts-result"
    text_to_speech.assert_awaited_once_with(
        text="hello",
        output_path=None,
        speed=None,
        instructions=None,
        provider=None,
    )


@pytest.mark.asyncio
async def test_file_handler_validation_keeps_upstream_error_contracts():
    from tools.file_tools import _handle_patch, _handle_write_file

    assert json.loads(await _handle_write_file({}))["error"] == (
        "write_file: missing required field 'path'. Re-emit the tool call with "
        "both 'path' and 'content' set."
    )
    assert "dropped-arg bug" in json.loads(
        await _handle_write_file({"path": "target"})
    )["error"]
    assert json.loads(
        await _handle_write_file({"path": "target", "content": 1})
    )["error"] == "write_file: 'content' must be a string, got int."
    assert json.loads(await _handle_patch({"mode": "unknown"}))["error"] == (
        "Unknown mode: unknown"
    )
    assert json.loads(await _handle_patch({"mode": "replace"}))["error"] == (
        "path required"
    )
    assert json.loads(
        await _handle_patch({"mode": "replace", "path": "target"})
    )["error"] == "old_string and new_string required"
    assert json.loads(await _handle_patch({"mode": "patch"}))["error"] == (
        "patch content required"
    )
