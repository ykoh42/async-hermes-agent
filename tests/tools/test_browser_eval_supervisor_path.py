"""Unit tests for the supervisor-WS fast path in browser_console / _browser_eval.

These exercise the dispatch logic in ``tools.browser_tool._browser_eval`` and
the response shaping in ``CDPSupervisor.evaluate_runtime`` using mocks — no
real browser, no real WebSocket.  Real-CDP coverage lives in
``tests/tools/test_browser_supervisor.py`` (gated on Chrome being installed).
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fast-path dispatch: tools.browser_tool._browser_eval
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _disable_camofox(monkeypatch):
    """Force the non-camofox path so our supervisor branch is reached."""
    import tools.browser_tool as bt

    monkeypatch.setattr(bt, "_is_camofox_mode", AsyncMock(return_value=False))
    monkeypatch.setattr(bt, "_last_session_key", lambda task_id: "test-task")


def _patch_supervisor(monkeypatch, supervisor):
    """Wire SUPERVISOR_REGISTRY.get to return ``supervisor`` for any task_id."""
    import tools.browser_supervisor as bs

    registry = MagicMock()
    registry.get.return_value = supervisor
    monkeypatch.setattr(bs, "SUPERVISOR_REGISTRY", registry)
    return registry


class TestBrowserEvalSupervisorPath:
    """The supervisor fast path replaces the agent-browser subprocess hop."""

    async def test_primitive_result_routes_through_supervisor(self, monkeypatch):
        import tools.browser_tool as bt

        sup = MagicMock()
        sup.evaluate_runtime = AsyncMock(return_value={
            "ok": True,
            "result": 42,
            "result_type": "number",
        })
        _patch_supervisor(monkeypatch, sup)
        # If the subprocess path is hit we want a loud failure.
        monkeypatch.setattr(
            bt, "_run_browser_command",
            lambda *a, **kw: pytest.fail("subprocess path must not run when supervisor is healthy"),
        )

        out = json.loads(await bt._browser_eval("1 + 41"))
        assert out["success"] is True
        assert out["result"] == 42
        assert out["method"] == "cdp_supervisor"
        sup.evaluate_runtime.assert_awaited_once_with("1 + 41")

    async def test_json_string_result_is_parsed(self, monkeypatch):
        """Match agent-browser semantics: JSON-string results get parsed."""
        import tools.browser_tool as bt

        sup = MagicMock()
        sup.evaluate_runtime = AsyncMock(return_value={
            "ok": True,
            "result": '{"a": 1, "b": [2, 3]}',
            "result_type": "string",
        })
        _patch_supervisor(monkeypatch, sup)
        monkeypatch.setattr(
            bt, "_run_browser_command",
            lambda *a, **kw: pytest.fail("subprocess path must not run"),
        )

        out = json.loads(await bt._browser_eval('JSON.stringify({a:1,b:[2,3]})'))
        assert out["success"] is True
        assert out["result"] == {"a": 1, "b": [2, 3]}
        # result_type reflects the parsed Python type, not the raw JS type.
        assert out["result_type"] == "dict"


    async def test_subprocess_reference_chain_error_becomes_guidance(self, monkeypatch):
        """The CLI subprocess can't retry with returnByValue=False, so the
        cryptic 'Object reference chain is too long' CDP error must be turned
        into actionable guidance instead of surfaced raw."""
        import tools.browser_tool as bt

        # No supervisor → subprocess path runs.
        _patch_supervisor(monkeypatch, None)

        def _fake_subprocess(task_id, cmd, args):
            assert cmd == "eval"
            return {
                "success": False,
                "error": "Runtime.evaluate failed: Object reference chain is too long",
            }

        monkeypatch.setattr(bt, "_run_browser_command", AsyncMock(side_effect=_fake_subprocess))

        out = json.loads(await bt._browser_eval("document.body"))
        assert out["success"] is False
        # Raw protocol error must NOT leak through.
        assert "reference chain" not in out["error"].lower()
        # Actionable guidance instead.
        assert "primitive" in out["error"].lower()
        assert "DOM node" in out["error"] or "dom node" in out["error"].lower()


# ---------------------------------------------------------------------------
# Response shaping: CDPSupervisor.evaluate_runtime
# ---------------------------------------------------------------------------


def _make_supervisor_with_cdp(cdp_response):
    """Build a CDPSupervisor instance that mocks ``_cdp`` to return ``cdp_response``.

    Bypasses ``__init__`` entirely so we don't need a real WS connection.  We
    set just the state ``evaluate_runtime`` reads.
    """
    from tools.browser_supervisor import CDPSupervisor

    sup = object.__new__(CDPSupervisor)
    sup._active = True
    sup._page_session_id = "test-session-id"
    sup._run_task = MagicMock()
    sup._run_task.done.return_value = False
    sup._cdp = AsyncMock(return_value=cdp_response)
    return sup


class TestEvaluateRuntimeResponseShaping:
    """CDPSupervisor.evaluate_runtime decodes the Runtime.evaluate response correctly."""

    async def test_primitive_value(self):
        sup = _make_supervisor_with_cdp({
            "id": 1,
            "result": {"result": {"type": "number", "value": 42}},
        })
        out = await sup.evaluate_runtime("1 + 41")
        assert out == {"ok": True, "result": 42, "result_type": "number"}

    async def test_object_value_returned_by_value(self):
        sup = _make_supervisor_with_cdp({
            "id": 1,
            "result": {
                "result": {
                    "type": "object",
                    "value": {"foo": "bar", "n": 7},
                }
            },
        })
        out = await sup.evaluate_runtime('({foo:"bar", n:7})')
        assert out["ok"] is True
        assert out["result"] == {"foo": "bar", "n": 7}
        assert out["result_type"] == "object"


    async def test_no_session_attached_returns_error(self):
        from tools.browser_supervisor import CDPSupervisor

        sup = object.__new__(CDPSupervisor)
        sup._active = True
        sup._page_session_id = None  # ← attach hasn't happened yet
        sup._run_task = MagicMock()
        sup._run_task.done.return_value = False
        out = await sup.evaluate_runtime("1+1")
        assert out["ok"] is False
        assert "session" in out["error"].lower()


def _make_supervisor_with_cdp_fn(cdp_fn):
    """Like ``_make_supervisor_with_cdp`` but lets the test supply a coroutine
    function as ``_cdp`` so behaviour can vary by params (e.g. returnByValue).
    """
    from tools.browser_supervisor import CDPSupervisor

    sup = object.__new__(CDPSupervisor)
    sup._active = True
    sup._page_session_id = "test-session-id"
    sup._run_task = MagicMock()
    sup._run_task.done.return_value = False
    sup._cdp = cdp_fn
    return sup


class TestEvaluateRuntimeDomNodeCrashRetry:
    """returnByValue=True on a DOM node fails CDP serialization with 'Object
    reference chain is too long'.  evaluate_runtime must retry with
    returnByValue=False and return the node's description instead of crashing.
    """

    async def test_reference_chain_crash_retries_without_by_value(self):
        calls = []

        async def _fake_cdp(method, params=None, *, session_id=None, timeout=10.0):
            by_value = (params or {}).get("returnByValue")
            calls.append(by_value)
            if by_value:
                # Mirror _read_loop turning a top-level CDP error into a RuntimeError.
                raise RuntimeError(
                    "CDP error on id=7: {'code': -32000, "
                    "'message': 'Object reference chain is too long'}"
                )
            # returnByValue=False: Chrome returns the node's description, no value.
            return {
                "id": 8,
                "result": {
                    "result": {
                        "type": "object",
                        "subtype": "node",
                        "description": "body",
                    }
                },
            }

        sup = _make_supervisor_with_cdp_fn(_fake_cdp)
        out = await sup.evaluate_runtime("document.body")
        assert out["ok"] is True
        assert out["result"] == "body"
        assert out["result_type"] == "object"
        # First call by_value=True (crashed), retried with by_value=False.
        assert calls == [True, False]

    async def test_unrelated_error_does_not_retry(self):
        calls = []

        async def _fake_cdp(method, params=None, *, session_id=None, timeout=10.0):
            calls.append((params or {}).get("returnByValue"))
            raise RuntimeError("CDP error on id=3: {'message': 'Target closed'}")

        sup = _make_supervisor_with_cdp_fn(_fake_cdp)
        out = await sup.evaluate_runtime("document.body")
        assert out["ok"] is False
        assert "Target closed" in out["error"]
        # No retry for unrelated failures — exactly one call.
        assert calls == [True]
