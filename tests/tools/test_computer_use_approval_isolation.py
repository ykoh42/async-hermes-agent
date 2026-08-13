"""Approval callbacks and per-session stores are isolated between tests."""

import json

import pytest
import pytest_asyncio


pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture(autouse=True)
async def _reset_computer_use_state():
    from tools.computer_use import tool as cu_tool

    await cu_tool.reset_backend_for_tests()
    cu_tool.set_approval_callback(None)
    yield
    cu_tool.set_approval_callback(None)
    await cu_tool.reset_backend_for_tests()


async def _install_backend(cu_tool):
    from tools.computer_use.backend import ActionResult, CaptureResult

    class _RecordingBackend:
        def __init__(self):
            self.calls = []

        async def start(self):
            pass

        async def stop(self):
            pass

        async def is_available(self):
            return True

        async def click(self, **kw):
            self.calls.append(("click", kw))
            return ActionResult(ok=True, action="click")

        async def capture(
            self, mode="som", app=None, pid=None, window_id=None
        ):
            return CaptureResult(
                mode=mode,
                width=1,
                height=1,
                png_b64=None,
                elements=[],
                app="X",
                window_title="",
            )

    backend = _RecordingBackend()
    await cu_tool.reset_backend_for_tests()
    cu_tool._backend = backend
    return backend


async def test_a_forgets_a_poisoned_approval_callback():
    from tools.computer_use import tool as cu_tool

    def stale_two_arg_callback(action, args):
        return "approve_once"

    cu_tool.set_approval_callback(stale_two_arg_callback)


async def test_b_still_dispatches_with_default_allow():
    from tools.computer_use import tool as cu_tool

    backend = await _install_backend(cu_tool)
    result = await cu_tool.handle_computer_use(
        {"action": "click", "element": 3}
    )
    call_names = [c[0] for c in backend.calls]
    assert "click" in call_names, f"leaked approval callback: {result!r}"
    payload = json.loads(result) if isinstance(result, str) else result
    assert not (isinstance(payload, dict) and payload.get("error"))
