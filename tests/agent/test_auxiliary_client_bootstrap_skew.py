"""Regression for #64333 — auxiliary client must survive a version-skewed
agent.process_bootstrap that lacks build_keepalive_http_client.

Desktop installs can run a runtime whose agent/process_bootstrap.py predates
the helper while newer callers (cron scheduler → auxiliary_client) expect it.
Before the fix the module-level import made every cron job die with
ImportError before any agent logic ran.
"""

import builtins
import logging

import pytest

import agent.auxiliary_client as aux


@pytest.mark.asyncio
async def test_missing_bootstrap_helper_degrades_instead_of_raising(monkeypatch, caplog):
    """ImportError from process_bootstrap uses explicit native async clients."""
    real_import = builtins.__import__

    def _fail_bootstrap(name, *args, **kwargs):
        if name == "agent.process_bootstrap":
            raise ImportError(
                "cannot import name 'build_keepalive_http_client' "
                "from 'agent.process_bootstrap'"
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_bootstrap)
    monkeypatch.setattr(aux, "_WARNED_KEEPALIVE_IMPORT_SKEW", False)

    with caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
        result = await aux._openai_http_client_kwargs("https://api.example.com/v1")
        again = await aux._openai_http_client_kwargs("https://api.example.com/v1")

    assert result["http_client"] is not None
    assert again["http_client"] is not None
    skew_warnings = [
        r for r in caplog.records if "mixed/stale install" in r.getMessage()
    ]
    assert len(skew_warnings) == 1  # warned once, not per call
    await result["http_client"].aclose()
    await again["http_client"].aclose()


@pytest.mark.asyncio
async def test_healthy_bootstrap_still_injects_keepalive_client():
    """With the helper present, the keepalive http_client is injected."""
    result = await aux._openai_http_client_kwargs("https://api.example.com/v1")
    assert "http_client" in result
    assert result["http_client"] is not None
    await result["http_client"].aclose()
