"""Tests for the native-async LSPService.

Drives the service through ``snapshot_baseline`` →
``get_diagnostics_sync`` against the mock LSP server, exercising the
delta filter that ``tools/file_operations._check_lint_delta`` relies
on.
"""
from __future__ import annotations

import sys
import asyncio
from pathlib import Path

import pytest

from agent.lsp.manager import LSPService
from agent.lsp.servers import (
    SERVERS,
    ServerContext,
    ServerDef,
    SpawnSpec,
)


MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


def _install_mock_server(monkeypatch, script: str = "errors", server_id: str = "pyright"):
    """Replace one registered server with a wrapper that spawns the mock.

    We reuse ``pyright`` so .py files route to it.  This keeps the
    test free of any LSP toolchain dependency.
    """
    target_index = next(i for i, s in enumerate(SERVERS) if s.server_id == server_id)
    original = SERVERS[target_index]

    async def _spawn(root: str, ctx: ServerContext) -> SpawnSpec:
        env = {"MOCK_LSP_SCRIPT": script}
        return SpawnSpec(
            command=[sys.executable, MOCK_SERVER],
            workspace_root=root,
            cwd=root,
            env=env,
            initialization_options={},
        )

    async def _root(_file_path: str, workspace: str) -> str:
        return workspace

    replacement = ServerDef(
        server_id=server_id,
        extensions=original.extensions,
        resolve_root=_root,  # always use workspace root
        build_spawn=_spawn,
        seed_first_push=False,
        description="mock " + server_id,
    )
    # Patch the SERVERS list element directly + restore on teardown.
    SERVERS[target_index] = replacement

    yield

    SERVERS[target_index] = original


@pytest.fixture
def mock_pyright(monkeypatch, tmp_path):
    """Install the mock as ``pyright`` and create a fake git workspace."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "pyproject.toml").write_text("")  # so pyright's root resolver finds it
    monkeypatch.chdir(str(repo))
    gen = _install_mock_server(monkeypatch, "errors", "pyright")
    next(gen)
    yield repo
    try:
        next(gen)
    except StopIteration:
        pass






@pytest.mark.asyncio
async def test_service_e2e_delta_filter(mock_pyright):
    """End-to-end: snapshot baseline → wait → delta returned."""
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("print('hi')\n")

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
    )
    try:
        assert await svc.enabled_for(str(f))
        # Baseline first — server pushes 1 error.
        await svc.snapshot_baseline(str(f))
        # Re-poll: same error is in baseline, so delta is empty.
        new_diags = await svc.get_diagnostics_sync(str(f))
        assert new_diags == []
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_service_e2e_delta_filter_with_line_shift(mock_pyright):
    """End-to-end: an edit that shifts the diagnostic's line still
    filters correctly when ``line_shift`` is supplied.

    The mock LSP server emits a fixed error at line 0; for this test
    we don't need to actually shift the server's output — we just
    need to prove that supplying a line_shift through the API works
    and doesn't break the existing delta path.  The unit tests in
    test_delta_key.py cover the shift semantics in detail.
    """
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("print('hi')\n")

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
    )
    try:
        await svc.snapshot_baseline(str(f))
        # Identity shift — should behave exactly like no shift.
        new_diags = await svc.get_diagnostics_sync(
            str(f), line_shift=lambda line: line
        )
        assert new_diags == []
    finally:
        await svc.shutdown()






@pytest.mark.asyncio
async def test_reused_client_refreshes_last_used_and_survives_reap(mock_pyright):
    """A client re-acquired from the cache must have its ``_last_used``
    timestamp refreshed so a subsequent sweep does NOT evict it.

    Covers the timestamp refresh on the existing-client fast path in
    ``_get_or_spawn`` — without it, a client in constant use would be
    reaped ``idle_timeout`` seconds after its FIRST use.
    """
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("")
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
        idle_timeout=60.0,  # sweeps manually below; loop never fires
    )
    try:
        await svc.get_diagnostics_sync(str(f))
        key = next(iter(svc._clients))
        first_used = svc._last_used[key]

        # Age the timestamp past the cutoff, then re-acquire the client.
        svc._last_used[key] = first_used - 120.0
        await svc.get_diagnostics_sync(str(f))
        assert svc._last_used[key] > first_used - 120.0, (
            "re-acquiring a cached client must refresh _last_used"
        )

        # A sweep right after reuse must keep the client.
        await svc._reap_idle_once()
        assert key in svc._clients
        assert svc.get_status()["clients"]
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_reaper_survives_sweep_error(mock_pyright):
    """One failing sweep must not kill the reaper loop — the loop's
    ``except Exception`` guard must swallow the error and keep sweeping."""
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("")
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
        idle_timeout=0.1,
    )
    try:
        # Sabotage the sweep itself so the reaper-loop except branch
        # actually runs (a failing client.shutdown() would be swallowed
        # by gather(return_exceptions=True) and never reach the loop).
        calls = {"n": 0}
        real_reap = svc._reap_idle_once

        async def _flaky_reap():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("sweep sabotage")
            await real_reap()

        svc._reap_idle_once = _flaky_reap  # type: ignore[method-assign]

        await svc.get_diagnostics_sync(str(f))
        assert svc.get_status()["clients"]

        # First sweep raises; later sweeps must still reap the client.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 3.0
        while svc.get_status()["clients"] and loop.time() < deadline:
            await asyncio.sleep(0.02)

        assert calls["n"] >= 2, "reaper loop died after the failing sweep"
        assert svc.get_status()["clients"] == []
        assert svc._idle_reaper_task is not None
        assert not svc._idle_reaper_task.done()
    finally:
        await svc.shutdown()


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_cancel_shared_spawn(mock_pyright):
    repo = mock_pyright
    target = repo / "x.py"
    target.write_text("")
    server = next(item for item in SERVERS if item.server_id == "pyright")
    original_spawn = server.build_spawn
    entered = asyncio.Event()
    release = asyncio.Event()

    async def delayed_spawn(root: str, context: ServerContext) -> SpawnSpec:
        entered.set()
        await release.wait()
        return await original_spawn(root, context)

    server.build_spawn = delayed_spawn
    service = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
    )
    try:
        first = asyncio.create_task(service._get_or_spawn(str(target)))
        await entered.wait()
        second = asyncio.create_task(service._get_or_spawn(str(target)))
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        release.set()
        client = await second
        assert client is not None
        assert client.is_running
        assert len(service._clients) == 1
        assert not service._spawn_tasks
    finally:
        server.build_spawn = original_spawn
        release.set()
        await service.shutdown()


@pytest.mark.asyncio
async def test_shutdown_cancels_and_awaits_owned_spawn(mock_pyright):
    repo = mock_pyright
    target = repo / "x.py"
    target.write_text("")
    server = next(item for item in SERVERS if item.server_id == "pyright")
    original_spawn = server.build_spawn
    entered = asyncio.Event()

    async def blocked_spawn(root: str, context: ServerContext) -> SpawnSpec:
        entered.set()
        await asyncio.Event().wait()
        return await original_spawn(root, context)

    server.build_spawn = blocked_spawn
    service = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
    )
    waiter = asyncio.create_task(service._get_or_spawn(str(target)))
    try:
        await entered.wait()
        await service.shutdown()
        assert await waiter is None
        assert not service._spawn_tasks
        assert not service._spawning
        assert not service._clients
    finally:
        server.build_spawn = original_spawn
        if not waiter.done():
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)


@pytest.mark.asyncio
async def test_concurrent_shutdown_callers_wait_for_same_cleanup():
    service = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="manual",
        idle_timeout=0,
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    class SlowClient:
        async def shutdown(self):
            entered.set()
            await release.wait()

    service._clients[("mock", "/workspace")] = SlowClient()  # type: ignore[assignment]
    first = asyncio.create_task(service.shutdown())
    await entered.wait()
    second = asyncio.create_task(service.shutdown())
    await asyncio.sleep(0)
    assert not second.done()

    release.set()
    await asyncio.gather(first, second)
    assert service._shutdown_task is not None
    assert service._shutdown_task.done()
