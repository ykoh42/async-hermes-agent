"""Tests for native-async LSP singleton lifecycle ownership."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent import lsp as lsp_module


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Force clean process-global state around every lifecycle test."""
    lsp_module._service = None
    lsp_module._service_lock = None
    lsp_module._service_loop = None
    lsp_module._service_shutdown_task = None
    lsp_module._lifecycle_consumers.clear()
    yield
    lsp_module._service = None
    lsp_module._service_lock = None
    lsp_module._service_loop = None
    lsp_module._service_shutdown_task = None
    lsp_module._lifecycle_consumers.clear()


@pytest.mark.asyncio
async def test_get_service_creates_singleton_once(monkeypatch):
    fake_service = MagicMock()
    fake_service.is_active.return_value = True
    create = AsyncMock(return_value=fake_service)
    monkeypatch.setattr(lsp_module.LSPService, "create_from_config", create)

    first = await lsp_module.get_service()
    second = await lsp_module.get_service()
    third = await lsp_module.get_service()

    assert first is fake_service
    assert second is fake_service
    assert third is fake_service
    create.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_shutdown_service_idempotent(monkeypatch):
    fake_service = MagicMock()
    fake_service.is_active.return_value = True
    fake_service.shutdown = AsyncMock()
    monkeypatch.setattr(
        lsp_module.LSPService,
        "create_from_config",
        AsyncMock(return_value=fake_service),
    )

    await lsp_module.get_service()
    await lsp_module.shutdown_service()
    await lsp_module.shutdown_service()

    fake_service.shutdown.assert_awaited_once_with()
    assert lsp_module._service is None
    assert lsp_module._service_lock is None
    assert lsp_module._service_loop is None


@pytest.mark.asyncio
async def test_concurrent_global_shutdown_callers_wait_for_cleanup(monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()
    fake_service = MagicMock()
    fake_service.is_active.return_value = True

    async def slow_shutdown():
        entered.set()
        await release.wait()

    fake_service.shutdown = AsyncMock(side_effect=slow_shutdown)
    monkeypatch.setattr(
        lsp_module.LSPService,
        "create_from_config",
        AsyncMock(return_value=fake_service),
    )
    await lsp_module.get_service()
    first = asyncio.create_task(lsp_module.shutdown_service())
    await entered.wait()
    second = asyncio.create_task(lsp_module.shutdown_service())
    await asyncio.sleep(0)
    assert not second.done()

    release.set()
    await asyncio.gather(first, second)
    fake_service.shutdown.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_cancelled_global_shutdown_finishes_cleanup_before_reraise(
    monkeypatch,
):
    entered = asyncio.Event()
    release = asyncio.Event()
    fake_service = MagicMock()
    fake_service.is_active.return_value = True

    async def slow_shutdown():
        entered.set()
        await release.wait()

    fake_service.shutdown = AsyncMock(side_effect=slow_shutdown)
    monkeypatch.setattr(
        lsp_module.LSPService,
        "create_from_config",
        AsyncMock(return_value=fake_service),
    )
    await lsp_module.get_service()
    shutdown = asyncio.create_task(lsp_module.shutdown_service())
    await entered.wait()

    shutdown.cancel()
    await asyncio.sleep(0)
    shutdown.cancel()
    await asyncio.sleep(0)
    assert not shutdown.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await shutdown
    fake_service.shutdown.assert_awaited_once_with()
    assert lsp_module._service is None
    assert lsp_module._service_shutdown_task is None
    assert lsp_module._service_lock is None
    assert lsp_module._service_loop is None


@pytest.mark.asyncio
async def test_get_service_waits_for_shutdown_before_recreating(monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()
    first_service = MagicMock()
    first_service.is_active.return_value = True
    second_service = MagicMock()
    second_service.is_active.return_value = True

    async def slow_shutdown():
        entered.set()
        await release.wait()

    first_service.shutdown = AsyncMock(side_effect=slow_shutdown)
    create = AsyncMock(side_effect=[first_service, second_service])
    monkeypatch.setattr(lsp_module.LSPService, "create_from_config", create)
    assert await lsp_module.get_service() is first_service
    shutdown = asyncio.create_task(lsp_module.shutdown_service())
    await entered.wait()
    replacement = asyncio.create_task(lsp_module.get_service())
    await asyncio.sleep(0)
    assert not replacement.done()

    release.set()
    await shutdown
    assert await replacement is second_service
    assert create.await_count == 2


@pytest.mark.asyncio
async def test_inactive_service_is_cached_but_not_returned(monkeypatch):
    fake_service = MagicMock()
    fake_service.is_active.return_value = False
    fake_service.shutdown = AsyncMock()
    create = AsyncMock(return_value=fake_service)
    monkeypatch.setattr(lsp_module.LSPService, "create_from_config", create)

    assert await lsp_module.get_service() is None
    assert await lsp_module.get_service() is None
    create.assert_awaited_once_with()
    await lsp_module.shutdown_service()


@pytest.mark.asyncio
async def test_final_agent_lease_owns_service_shutdown(monkeypatch):
    class Owner:
        pass

    first = Owner()
    second = Owner()
    shutdown = AsyncMock()
    monkeypatch.setattr(lsp_module, "shutdown_service", shutdown)

    await lsp_module._retain_lsp_lifecycle(first)
    await lsp_module._retain_lsp_lifecycle(second)
    await lsp_module._release_lsp_lifecycle(first)
    shutdown.assert_not_awaited()

    await lsp_module._release_lsp_lifecycle(second)
    shutdown.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_releasing_same_agent_lease_twice_is_safe(monkeypatch):
    class Owner:
        pass

    owner = Owner()
    shutdown = AsyncMock()
    monkeypatch.setattr(lsp_module, "shutdown_service", shutdown)

    await lsp_module._retain_lsp_lifecycle(owner)
    await lsp_module._release_lsp_lifecycle(owner)
    await lsp_module._release_lsp_lifecycle(owner)

    shutdown.assert_awaited_once_with()


def test_state_only_singleton_can_rebind_between_sequential_loops(monkeypatch):
    service = lsp_module.LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="manual",
        idle_timeout=0,
    )
    create = AsyncMock(return_value=service)
    monkeypatch.setattr(lsp_module.LSPService, "create_from_config", create)

    assert asyncio.run(lsp_module.get_service()) is service
    assert asyncio.run(lsp_module.get_service()) is service
    assert service._has_owned_resources() is False
    asyncio.run(lsp_module.shutdown_service())


def test_started_singleton_rejects_cross_loop_reuse(monkeypatch):
    service = lsp_module.LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=1.0,
        install_strategy="manual",
        idle_timeout=0,
    )
    service._started = True
    monkeypatch.setattr(
        lsp_module.LSPService,
        "create_from_config",
        AsyncMock(return_value=service),
    )

    assert asyncio.run(lsp_module.get_service()) is service
    with pytest.raises(RuntimeError, match="another event loop"):
        asyncio.run(lsp_module.get_service())
