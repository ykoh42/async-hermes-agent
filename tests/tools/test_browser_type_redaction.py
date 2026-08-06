"""Regression tests for browser_type display redaction.

Typed text is passed through the same secret-pattern redactor used for logs:
recognizable credentials (API keys, tokens) are masked in display-facing
output, while normal typed text is left intact.  The raw value is always sent
to the browser backend regardless.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from tools.browser_tool import browser_type


@pytest.mark.asyncio
async def test_browser_type_redacts_api_key_in_output(monkeypatch):
    monkeypatch.delenv("CAMOFOX_URL", raising=False)
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    monkeypatch.setenv("HERMES_REDACT_SECRETS", "true")
    secret = "sk-proj-ABCD1234567890EFGH"

    with patch(
        "tools.browser_tool._run_browser_command",
        new=AsyncMock(return_value={"success": True}),
    ) as mock_run:
        with (
            patch(
                "tools.browser_tool._is_camofox_mode",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "tools.browser_tool._eval_ssrf_guard_active",
                new=AsyncMock(return_value=False),
            ),
        ):
            result = json.loads(
                await browser_type("@apikey", secret, task_id="redaction-test")
            )

    assert result["success"] is True
    assert secret not in json.dumps(result)
    assert result["typed"].startswith("sk-pro")
    # Raw secret still typed into the page.
    mock_run.assert_awaited_once()
    assert mock_run.call_args.args[2] == ["@apikey", secret]


@pytest.mark.asyncio
async def test_browser_type_keeps_normal_text_in_output(monkeypatch):
    monkeypatch.delenv("CAMOFOX_URL", raising=False)
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    monkeypatch.setenv("HERMES_REDACT_SECRETS", "true")
    text = "hello world search query"

    with patch(
        "tools.browser_tool._run_browser_command",
        new=AsyncMock(return_value={"success": True}),
    ) as mock_run:
        with (
            patch(
                "tools.browser_tool._is_camofox_mode",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "tools.browser_tool._eval_ssrf_guard_active",
                new=AsyncMock(return_value=False),
            ),
        ):
            result = json.loads(
                await browser_type("@search", text, task_id="redaction-test")
            )

    assert result["success"] is True
    assert result["typed"] == text
    mock_run.assert_awaited_once()
    assert mock_run.call_args.args[2] == ["@search", text]


@pytest.mark.asyncio
async def test_browser_type_failure_redacts_api_key_in_error(monkeypatch):
    monkeypatch.delenv("CAMOFOX_URL", raising=False)
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    monkeypatch.setenv("HERMES_REDACT_SECRETS", "true")
    secret = "sk-proj-ABCD1234567890EFGH"

    with patch(
        "tools.browser_tool._run_browser_command",
        new=AsyncMock(return_value={
            "success": False,
            "error": f"backend failed while typing {secret}",
            "fallback_warning": f"chrome fallback also saw {secret}",
        }),
    ) as mock_run:
        with (
            patch(
                "tools.browser_tool._is_camofox_mode",
                new=AsyncMock(return_value=False),
            ),
            patch(
                "tools.browser_tool._eval_ssrf_guard_active",
                new=AsyncMock(return_value=False),
            ),
        ):
            raw_result = await browser_type(
                "@apikey", secret, task_id="redaction-test"
            )
        result = json.loads(raw_result)

    assert result["success"] is False
    assert secret not in raw_result
    assert "sk-pro" in raw_result
    mock_run.assert_awaited_once()
    assert mock_run.call_args.args[2] == ["@apikey", secret]
