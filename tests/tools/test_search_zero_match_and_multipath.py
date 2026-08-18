"""Tests for search_files zero-match probes and multi-path recovery."""

import json
import os

import pytest

from tools.file_tools import search_tool


@pytest.fixture
def proj(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    d = tmp_path / "proj"
    d.mkdir()
    (d / "a.py").write_text("TOKEN_ALPHA = 'find_me_value'\nother = 1\n")
    (d / "b.py").write_text("x = compute(TOKEN_ALPHA)\n")
    e = tmp_path / "extra"
    e.mkdir()
    (e / "c.txt").write_text("TOKEN_ALPHA appears here too\n")
    return tmp_path


@pytest.mark.asyncio
class TestZeroMatchProbe:
    async def test_case_mismatch_gets_hint(self, proj):
        r = json.loads(await search_tool("token_alpha", path=str(proj / "proj"), task_id="t-zm"))
        assert r["total_count"] == 0
        assert "case-insensitive" in r.get("warning", "")

    async def test_case_mismatch_hint_names_the_files(self, proj):
        r = json.loads(await search_tool("token_alpha", path=str(proj / "proj"), task_id="t-zm"))
        warning = r.get("warning", "")
        assert "a.py" in warning and "b.py" in warning

    async def test_regex_metachar_literal_hint(self, proj):
        d = proj / "proj"
        (d / "meta.py").write_text("result = lookup[key+1]\n")
        r = json.loads(await search_tool("lookup[key+1]", path=str(d), task_id="t-zm"))
        assert r["total_count"] == 0
        assert "literal match" in r.get("warning", "")
        assert "meta.py" in r.get("warning", "")

    async def test_true_zero_match_no_hint(self, proj):
        r = json.loads(await search_tool("zzz_totally_absent_zzz", path=str(proj / "proj"), task_id="t-zm"))
        assert r["total_count"] == 0
        assert "warning" not in r

    async def test_hidden_only_match_gets_hint(self, proj):
        d = proj / "proj"
        (d / ".secretdir").mkdir()
        (d / ".secretdir" / "conf.cfg").write_text("HIDDEN_ONLY_TOKEN = true\n")
        r = json.loads(await search_tool("HIDDEN_ONLY_TOKEN", path=str(d), task_id="t-zm"))
        assert r["total_count"] == 0
        assert "hidden or gitignored" in r.get("warning", "")
        assert "conf.cfg" in r.get("warning", "")

    async def test_probe_path_list_is_capped(self, proj):
        d = proj / "proj"
        for i in range(8):
            (d / f"cap{i}.txt").write_text("capped_case_token = 1\n")
        r = json.loads(await search_tool("CAPPED_CASE_TOKEN", path=str(d), task_id="t-zm"))
        warning = r.get("warning", "")
        assert "case-insensitive" in warning
        assert "+3 more" in warning

    async def test_matching_search_unaffected(self, proj):
        r = json.loads(await search_tool("TOKEN_ALPHA", path=str(proj / "proj"), task_id="t-zm"))
        assert r["total_count"] >= 2
        assert "warning" not in r


@pytest.mark.asyncio
class TestMultiPathRecovery:
    async def test_two_existing_paths_merged(self, proj):
        p = f"{proj / 'proj'} {proj / 'extra'}"
        r = json.loads(await search_tool("TOKEN_ALPHA", path=p, task_id="t-mp"))
        assert "error" not in r
        assert r["total_count"] >= 3
        blob = json.dumps(r)
        assert "a.py" in blob and "c.txt" in blob
        assert "2 entries" in r.get("warning", "") or "searched 2" in r.get("warning", "")

    async def test_missing_path_skipped_with_note(self, proj):
        p = f"{proj / 'proj'} {proj / 'nonexistent_dir'}"
        r = json.loads(await search_tool("TOKEN_ALPHA", path=p, task_id="t-mp"))
        assert "error" not in r
        assert r["total_count"] >= 2
        assert "skipped missing" in r.get("warning", "")

    async def test_comma_separated_paths(self, proj):
        p = f"{proj / 'proj'},{proj / 'extra'}"
        r = json.loads(await search_tool("TOKEN_ALPHA", path=p, task_id="t-mp"))
        assert "error" not in r
        assert r["total_count"] >= 3

    async def test_all_missing_still_errors(self, proj):
        p = f"{proj / 'gone1'} {proj / 'gone2'}"
        r = json.loads(await search_tool("TOKEN_ALPHA", path=p, task_id="t-mp"))
        assert "error" in r

    async def test_single_missing_path_keeps_similar_hint(self, proj):
        # single-path miss must keep the existing "Similar paths" behavior
        r = json.loads(await search_tool("TOKEN_ALPHA", path=str(proj / "pro"), task_id="t-mp"))
        assert "error" in r
        assert "Path not found" in r["error"]

    async def test_files_target_multi_path(self, proj):
        p = f"{proj / 'proj'} {proj / 'extra'}"
        r = json.loads(await search_tool("*.py", path=p, target="files", task_id="t-mp"))
        assert "error" not in r
        blob = json.dumps(r)
        assert "a.py" in blob
