"""Sibling regression test for #79178: background-PTY stdin must round-trip
surrogateescape content instead of crashing on the strict UTF-8 encode."""
import asyncio
import shlex

import pytest

from tools.process_registry import ProcessRegistry

pytestmark = pytest.mark.asyncio


async def test_write_stdin_pty_surrogateescape_roundtrip(tmp_path):
    registry = ProcessRegistry()
    out = tmp_path / "out.bin"
    script = tmp_path / "read_stdin.py"
    # readline(): a PTY never delivers EOF, so read one line (canonical mode
    # delivers it after the newline we send).
    script.write_text(
        f"import sys\nopen({str(out)!r}, 'wb').write(sys.stdin.buffer.readline())\n"
    )
    session = await registry.spawn_local(
        f"python3 {shlex.quote(str(script))}",
        cwd=str(tmp_path),
        use_pty=True,
    )
    if session._pty_master_fd is None:
        await registry.kill_process(session.id)
        pytest.skip("ptyprocess not available; PTY path not exercised")
    try:
        result = await registry.write_stdin(
            session.id, b"\xff".decode("utf-8", "surrogateescape") + "\n"
        )
        assert result["status"] == "ok", result
        deadline = asyncio.get_running_loop().time() + 10
        while asyncio.get_running_loop().time() < deadline and not out.exists():
            await asyncio.sleep(0.05)
        assert out.read_bytes() == b"\xff\n"
    finally:
        await registry.kill_process(session.id)
