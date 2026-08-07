"""Native-async regressions for partial search results at the time budget."""

import json

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target", "output_mode", "raw", "expected_key"),
    [
        ("files", "content", "src/a.py\nsrc/b.py\n", "files"),
        ("content", "files_only", "src/a.py\nsrc/b.py\n", "files"),
        (
            "content",
            "content",
            "src/a.py:10:foo\nsrc/b.py:20:foo\n",
            "matches",
        ),
    ],
)
async def test_timeout_returns_partial_results(
    tmp_path,
    monkeypatch,
    target,
    output_mode,
    raw,
    expected_key,
):
    from tools.file_tools import search_tool

    async def fake_run_rg(_arguments):
        return 124, raw, ""

    monkeypatch.setattr("tools.file_tools._run_rg", fake_run_rg)
    output = await search_tool(
        "foo",
        target=target,
        path=str(tmp_path),
        output_mode=output_mode,
        task_id=f"timeout-{target}-{output_mode}",
    )
    payload = json.loads(output.split("\n\n", 1)[0])

    assert payload["truncated"] is True
    assert payload["limit_reason"] == "search_timeout"
    assert len(payload[expected_key]) == 2
    assert "timed out" not in json.dumps(payload).lower()


@pytest.mark.asyncio
async def test_real_rg_error_still_hard_fails(tmp_path, monkeypatch):
    from tools.file_tools import search_tool

    async def fake_run_rg(_arguments):
        return 2, "", "rg: regex parse error:\n"

    monkeypatch.setattr("tools.file_tools._run_rg", fake_run_rg)
    result = json.loads(
        await search_tool("[", path=str(tmp_path), task_id="hard-rg-error")
    )

    assert result["error"] == "Search failed: rg: regex parse error:"
    assert "limit_reason" not in result
