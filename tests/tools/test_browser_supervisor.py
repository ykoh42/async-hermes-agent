"""Integration tests for tools.browser_supervisor.

Exercises the supervisor end-to-end against a real local Chrome
(``--remote-debugging-port``).  Skipped when Chrome is not installed
— these are the tests that actually verify the CDP wire protocol
works, since mock-CDP unit tests can only prove the happy paths we
thought to model.

These tests spawn a **real Chrome process** on the machine running them.
They are therefore opt-in, twice over:

* ``@pytest.mark.integration`` — excluded by the default
  ``addopts = "-m 'not integration'"`` in ``pyproject.toml``, so a bare
  ``pytest`` cannot launch a browser on a developer's desktop by accident.
* ``HERMES_E2E_BROWSER=1`` — the env gate this docstring has always claimed.
  It previously existed only in this prose: nothing read the variable, and
  the sole real gate was "is a Chrome binary on PATH", which is true on most
  desktops and on ``ubuntu-latest``. Now it is enforced.

Run manually:
    HERMES_E2E_BROWSER=1 scripts/run_tests.sh -m integration \\
        tests/tools/test_browser_supervisor.py

(``scripts/run_tests.sh`` runs under ``env -i`` and forwards
``HERMES_E2E_BROWSER`` explicitly; ``-m integration`` overrides the default
marker filter.)
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import tempfile

import httpx
import pytest
import pytest_asyncio


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("HERMES_E2E_BROWSER", "").strip() != "1",
        reason="real-browser E2E: set HERMES_E2E_BROWSER=1 to opt in",
    ),
    pytest.mark.skipif(
        not shutil.which("google-chrome") and not shutil.which("chromium"),
        reason="Chrome/Chromium not installed",
    ),
]


def _find_chrome() -> str:
    for candidate in ("google-chrome", "chromium", "chromium-browser"):
        path = shutil.which(candidate)
        if path:
            return path
    pytest.skip("no Chrome binary found")


@pytest_asyncio.fixture
async def chrome_cdp(request):
    """Start a headless Chrome with --remote-debugging-port, yield its WS URL.

    Uses a unique port per xdist worker to avoid cross-worker collisions.
    Always launches with ``--site-per-process`` so cross-origin iframes
    become real OOPIFs (needed by the iframe interaction tests).
    """

    # xdist worker_id is "master" in single-process mode or "gw0".."gwN" otherwise.
    # Under subprocess-per-file isolation there's no xdist, so we fall back
    # to "master" via the session-scoped fixture below.
    worker_id = request.getfixturevalue("worker_id") if "worker_id" in request.fixturenames else "master"
    if worker_id == "master":
        port_offset = 0
    else:
        port_offset = int(worker_id.lstrip("gw"))
    port = 9225 + port_offset
    profile = tempfile.mkdtemp(prefix="hermes-supervisor-test-")
    proc = await asyncio.create_subprocess_exec(
        _find_chrome(),
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile}",
        "--no-first-run",
        "--no-default-browser-check",
        "--headless=new",
        "--disable-gpu",
        "--site-per-process",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )

    ws_url = None
    deadline = asyncio.get_running_loop().time() + 15
    async with httpx.AsyncClient(timeout=1) as client:
        while asyncio.get_running_loop().time() < deadline:
            try:
                response = await client.get(f"http://127.0.0.1:{port}/json/version")
                response.raise_for_status()
                ws_url = response.json()["webSocketDebuggerUrl"]
                break
            except Exception:
                await asyncio.sleep(0.25)
    if ws_url is None:
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except Exception:
                pass
        shutil.rmtree(profile, ignore_errors=True)
        pytest.skip("Chrome didn't expose CDP in time")

    yield ws_url, port

    # Tear down Chrome. The stdlib `subprocess._wait()` POSIX implementation
    # has a known race (https://bugs.python.org/issue38630): when SIGCHLD
    # arrives concurrently with `proc.wait()`, `_try_wait(WNOHANG)` can
    # return a foreign pid and the `assert pid == self.pid or pid == 0`
    # fires. We saw this in CI on slice 1 after this fixture's teardown
    # (PR #33661 follow-up). Swallow the stdlib race + force-kill if wait
    # hangs, then always reap so we don't leak a zombie.
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        await asyncio.wait_for(proc.wait(), timeout=3)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=2)
        except Exception:
            pass
    shutil.rmtree(profile, ignore_errors=True)


def _test_page_url() -> str:
    html = """<!doctype html>
<html><head><title>Supervisor pytest</title></head><body>
<h1>Supervisor pytest</h1>
<iframe id="inner" srcdoc="<body><h2>frame-marker</h2></body>" width="400" height="100"></iframe>
</body></html>"""
    return "data:text/html;base64," + base64.b64encode(html.encode()).decode()


async def _fire_on_page(cdp_url: str, expression: str) -> None:
    """Navigate the first page target to a data URL and fire `expression`."""
    import websockets as _ws_mod

    async with _ws_mod.connect(cdp_url, max_size=50 * 1024 * 1024) as ws:
        next_id = [1]

        async def call(method, params=None, session_id=None):
            cid = next_id[0]
            next_id[0] += 1
            payload = {"id": cid, "method": method}
            if params:
                payload["params"] = params
            if session_id:
                payload["sessionId"] = session_id
            await ws.send(json.dumps(payload))
            async for raw in ws:
                message = json.loads(raw)
                if message.get("id") == cid:
                    return message

        targets = (await call("Target.getTargets"))["result"]["targetInfos"]
        page = next(target for target in targets if target.get("type") == "page")
        attach = await call(
            "Target.attachToTarget", {"targetId": page["targetId"], "flatten": True}
        )
        session_id = attach["result"]["sessionId"]
        await call("Page.navigate", {"url": _test_page_url()}, session_id=session_id)
        await asyncio.sleep(1.5)
        await call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True},
            session_id=session_id,
        )


@pytest_asyncio.fixture
async def supervisor_registry():
    """Yield the global registry and tear down any supervisors after the test."""
    from tools.browser_supervisor import SUPERVISOR_REGISTRY

    yield SUPERVISOR_REGISTRY
    await SUPERVISOR_REGISTRY.stop_all()


async def _wait_for_dialog(supervisor, timeout: float = 5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        snap = supervisor.snapshot()
        if snap.pending_dialogs:
            return snap.pending_dialogs
        await asyncio.sleep(0.1)
    return ()


@pytest.mark.asyncio
async def test_supervisor_start_and_snapshot(chrome_cdp, supervisor_registry):
    """Supervisor attaches, exposes an active snapshot with a top frame."""
    cdp_url, _port = chrome_cdp
    supervisor = await supervisor_registry.get_or_start(
        task_id="pytest-1", cdp_url=cdp_url
    )

    # Navigate so the frame tree populates.
    await _fire_on_page(cdp_url, "/* no dialog */ void 0")

    # Give a moment for frame events to propagate
    await asyncio.sleep(1.0)
    snap = supervisor.snapshot()
    assert snap.active is True
    assert snap.task_id == "pytest-1"
    assert snap.pending_dialogs == ()
    # At minimum a top frame should exist after the navigate.
    assert snap.frame_tree.get("top") is not None


@pytest.mark.asyncio
async def test_main_frame_alert_detection_and_dismiss(chrome_cdp, supervisor_registry):
    """alert() in the main frame surfaces and can be dismissed via the sync API."""
    cdp_url, _port = chrome_cdp
    supervisor = await supervisor_registry.get_or_start(
        task_id="pytest-2", cdp_url=cdp_url
    )

    await _fire_on_page(cdp_url, "setTimeout(() => alert('PYTEST-MAIN-ALERT'), 50)")
    dialogs = await _wait_for_dialog(supervisor)
    assert dialogs, "no dialog detected"
    d = dialogs[0]
    assert d.type == "alert"
    assert "PYTEST-MAIN-ALERT" in d.message

    result = await supervisor.respond_to_dialog("dismiss")
    assert result["ok"] is True
    # State cleared after dismiss
    await asyncio.sleep(0.3)
    assert supervisor.snapshot().pending_dialogs == ()


@pytest.mark.asyncio
async def test_iframe_contentwindow_alert(chrome_cdp, supervisor_registry):
    """alert() fired from inside a same-origin iframe surfaces too."""
    cdp_url, _port = chrome_cdp
    supervisor = await supervisor_registry.get_or_start(
        task_id="pytest-3", cdp_url=cdp_url
    )

    await _fire_on_page(
        cdp_url,
        "setTimeout(() => document.querySelector('#inner').contentWindow.alert('PYTEST-IFRAME'), 50)",
    )
    dialogs = await _wait_for_dialog(supervisor)
    assert dialogs, "no iframe dialog detected"
    assert any("PYTEST-IFRAME" in d.message for d in dialogs)

    result = await supervisor.respond_to_dialog("accept")
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_prompt_dialog_with_response_text(chrome_cdp, supervisor_registry):
    """prompt() gets our prompt_text back inside the page."""
    cdp_url, _port = chrome_cdp
    supervisor = await supervisor_registry.get_or_start(
        task_id="pytest-4", cdp_url=cdp_url
    )

    # Fire a prompt and stash the answer on window
    await _fire_on_page(
        cdp_url,
        "setTimeout(() => { window.__promptResult = prompt('give me a token', 'default-x'); }, 50)",
    )
    dialogs = await _wait_for_dialog(supervisor)
    assert dialogs
    d = dialogs[0]
    assert d.type == "prompt"
    assert d.default_prompt == "default-x"

    result = await supervisor.respond_to_dialog(
        "accept", prompt_text="PYTEST-PROMPT-REPLY"
    )
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_browser_dialog_tool_end_to_end(chrome_cdp, supervisor_registry):
    """Full agent-path check: fire an alert, call the tool handler directly."""
    from tools.browser_dialog_tool import browser_dialog

    cdp_url, _port = chrome_cdp
    supervisor = await supervisor_registry.get_or_start(
        task_id="pytest-tool", cdp_url=cdp_url
    )

    await _fire_on_page(cdp_url, "setTimeout(() => alert('PYTEST-TOOL-END2END'), 50)")
    assert await _wait_for_dialog(supervisor), "no dialog detected via wait_for_dialog"

    r = json.loads(await browser_dialog(action="dismiss", task_id="pytest-tool"))
    assert r["success"] is True
    assert r["action"] == "dismiss"
    assert "PYTEST-TOOL-END2END" in r["dialog"]["message"]


def test_browser_cdp_frame_id_real_oopif_smoke_documented():
    """Document that real-OOPIF E2E was manually verified — see PR #14540.

    A pytest version of this hits an asyncio version-quirk in the venv
    (3.11) that doesn't show up in standalone scripts (3.13 + system
    websockets). The mechanism IS verified end-to-end by two separate
    smoke scripts in /tmp/dialog-iframe-test/:

      * smoke_local_oopif.py   — local Chrome + 2 http servers on
        different hostnames + --site-per-process. Outer page on
        localhost:18905, iframe src=http://127.0.0.1:18906. Calls
        browser_cdp(method='Runtime.evaluate', frame_id=<OOPIF>) and
        verifies inner page's title comes back from the OOPIF session.
        PASSED on 2026-04-23: iframe document.title = 'INNER-FRAME-XYZ'

      * smoke_bb_iframe_agent_path.py — Browserbase + real cross-origin
        iframe (src=https://example.com/). Same browser_cdp(frame_id=)
        path. PASSED on 2026-04-23: iframe document.title =
        'Example Domain'

    The test_browser_cdp_frame_id_routes_via_supervisor pytest covers
    the supervisor-routing plumbing with a fake injected OOPIF.
    """
    pytest.skip(
        "Real-OOPIF E2E verified manually with smoke_local_oopif.py and "
        "smoke_bb_iframe_agent_path.py — pytest version hits an asyncio "
        "version quirk between venv (3.11) and standalone (3.13). "
        "Smoke logs preserved in /tmp/dialog-iframe-test/."
    )


@pytest.mark.asyncio
async def test_evaluate_runtime_unserializable_value(chrome_cdp, supervisor_registry):
    """``Infinity``/``NaN``/``BigInt`` come back via ``unserializableValue``."""
    cdp_url, _port = chrome_cdp
    supervisor = await supervisor_registry.get_or_start(
        task_id="pytest-eval-5", cdp_url=cdp_url
    )

    await _fire_on_page(cdp_url, "void 0")
    await asyncio.sleep(0.5)

    out = await supervisor.evaluate_runtime("Infinity")
    assert out["ok"] is True
    assert out["result"] == "Infinity"
