"""Tests for MCP tool-handler auth-failure detection.

When a tool call raises UnauthorizedError / OAuthNonInteractiveError /
httpx.HTTPStatusError(401), the handler should:
  1. Ask MCPOAuthManager.handle_401 if recovery is viable.
  2. If yes, trigger MCPServerTask._reconnect_event and retry once.
  3. If no, return a structured needs_reauth error so the model stops
     hallucinating manual refresh attempts.
"""
import json
from unittest.mock import MagicMock

import pytest


pytest.importorskip("mcp.client.auth.oauth2")


def test_is_auth_error_detects_oauth_flow_error():
    from tools.mcp_tool import _is_auth_error
    from mcp.client.auth import OAuthFlowError

    assert _is_auth_error(OAuthFlowError("expired")) is True
