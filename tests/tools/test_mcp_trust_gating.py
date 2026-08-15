"""Upstream MCP trust-gating tests, adapted to native async handlers."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from tools import mcp_tool


@pytest.fixture(autouse=True)
def _clean_trust_state():
    mcp_tool._server_trust_levels.clear()
    mcp_tool._tool_read_only_hints.clear()
    yield
    mcp_tool._server_trust_levels.clear()
    mcp_tool._tool_read_only_hints.clear()


@pytest.fixture
def connected_server():
    server = SimpleNamespace(session=SimpleNamespace(), _rpc_lock=None)
    with (
        patch(
            "tools.mcp_tool._get_connected_server_for_call",
            new=AsyncMock(return_value=server),
        ),
        patch(
            "tools.mcp_tool._call_mcp_tool",
            new=AsyncMock(return_value=json.dumps({"result": "ok"})),
        ) as call_tool,
    ):
        yield call_tool


@pytest.mark.asyncio
async def test_untrusted_write_tool_requires_and_accepts_approval(connected_server):
    mcp_tool._server_trust_levels["srv"] = "untrusted"
    handler = mcp_tool._make_tool_handler("srv", "delete_repo", 30.0)
    with patch(
        "tools.approval.request_elicitation_consent",
        new=AsyncMock(return_value="accept"),
    ) as consent:
        raw = await handler({"repo": "x"})
    consent.assert_awaited_once()
    assert json.loads(raw) == {"result": "ok"}
    connected_server.assert_awaited_once()


@pytest.mark.asyncio
async def test_denied_approval_blocks_rpc(connected_server):
    mcp_tool._server_trust_levels["srv"] = "untrusted"
    handler = mcp_tool._make_tool_handler("srv", "delete_repo", 30.0)
    with patch(
        "tools.approval.request_elicitation_consent",
        new=AsyncMock(return_value="decline"),
    ):
        raw = await handler({"repo": "x"})
    assert "did not approve" in json.loads(raw)["error"]
    connected_server.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_only_hint_skips_untrusted_approval(connected_server):
    mcp_tool._server_trust_levels["srv"] = "untrusted"
    mcp_tool._tool_read_only_hints["srv"] = {"list_repos": True}
    handler = mcp_tool._make_tool_handler("srv", "list_repos", 30.0)
    with patch(
        "tools.approval.request_elicitation_consent",
        new=AsyncMock(),
    ) as consent:
        raw = await handler({})
    consent.assert_not_awaited()
    assert json.loads(raw) == {"result": "ok"}


@pytest.mark.asyncio
async def test_full_and_unconfigured_servers_preserve_legacy_behavior(connected_server):
    handler = mcp_tool._make_tool_handler("srv", "delete_repo", 30.0)
    with patch(
        "tools.approval.request_elicitation_consent",
        new=AsyncMock(),
    ) as consent:
        assert json.loads(await handler({})) == {"result": "ok"}
    consent.assert_not_awaited()

    mcp_tool._server_trust_levels["srv"] = "full"
    with patch(
        "tools.approval.request_elicitation_consent",
        new=AsyncMock(),
    ) as consent:
        assert json.loads(await handler({})) == {"result": "ok"}
    consent.assert_not_awaited()


@pytest.mark.asyncio
async def test_approval_failure_fails_closed(connected_server):
    mcp_tool._server_trust_levels["srv"] = "untrusted"
    handler = mcp_tool._make_tool_handler("srv", "delete_repo", 30.0)
    with patch(
        "tools.approval.request_elicitation_consent",
        new=AsyncMock(side_effect=RuntimeError("approval backend down")),
    ):
        raw = await handler({})
    assert "fail-closed" in json.loads(raw)["error"]
    connected_server.assert_not_awaited()


def test_trust_normalization_fails_closed_for_unknown_values():
    assert mcp_tool._normalize_server_trust(None) == "full"
    assert mcp_tool._normalize_server_trust(" full ") == "full"
    assert mcp_tool._normalize_server_trust("UNTRUSTED") == "untrusted"
    assert mcp_tool._normalize_server_trust("banana") == "untrusted"


def test_annotation_hint_requires_exact_true():
    assert mcp_tool._annotation_read_only_hint(
        SimpleNamespace(annotations={"readOnlyHint": True})
    ) is True
    assert mcp_tool._annotation_read_only_hint(
        SimpleNamespace(annotations={"readOnlyHint": "yes"})
    ) is False
    assert mcp_tool._annotation_read_only_hint(SimpleNamespace(annotations=None)) is False


def test_discovery_records_trust_and_hints():
    tools = [
        SimpleNamespace(name="list", annotations=SimpleNamespace(readOnlyHint=True)),
        SimpleNamespace(name="write", annotations=SimpleNamespace(readOnlyHint=False)),
        SimpleNamespace(name="unknown", annotations=None),
    ]
    mcp_tool._record_tool_trust_metadata("srv", {"trust": "untrusted"}, tools)
    assert mcp_tool._server_trust_levels["srv"] == "untrusted"
    assert mcp_tool._tool_read_only_hints["srv"] == {
        "list": True,
        "write": False,
        "unknown": False,
    }
