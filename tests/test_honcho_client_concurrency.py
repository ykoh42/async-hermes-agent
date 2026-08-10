"""Concurrency test for get_honcho_client() — the TOCTOU race fix (#24759).

Proves the Honcho client is constructed exactly once when many tasks race the
first call.
"""

import asyncio
from unittest.mock import patch

import pytest
import pytest_asyncio
from honcho import Honcho

from plugins.memory.honcho import client as honcho_client
from plugins.memory.honcho.client import (
    HonchoClientConfig,
    get_honcho_client,
    reset_honcho_client,
)


@pytest_asyncio.fixture(autouse=True)
async def _reset_singleton():
    await reset_honcho_client()
    yield
    await reset_honcho_client()


@pytest.mark.asyncio
async def test_get_honcho_client_builds_once_under_concurrent_first_call():
    config = HonchoClientConfig(
        api_key="test-key",
        workspace_id="ws",
        environment="production",
    )

    with patch.object(Honcho, "model_construct", wraps=Honcho.model_construct) as build:
        results = await asyncio.gather(
            *(get_honcho_client(config) for _ in range(20))
        )

    assert build.call_count == 1
    assert len(results) == 20
    assert all(r is results[0] for r in results), "all threads share one client"


@pytest.mark.asyncio
async def test_reset_allows_rebuild():
    config = HonchoClientConfig(
        api_key="test-key", workspace_id="ws", environment="production"
    )

    c1 = await get_honcho_client(config)
    # Cached: no rebuild.
    assert await get_honcho_client(config) is c1

    await reset_honcho_client()
    c2 = await get_honcho_client(config)
    assert c2 is not c1


@pytest.mark.asyncio
async def test_missing_credentials_still_raises_before_build():
    bad = HonchoClientConfig(api_key="", base_url="", workspace_id="ws")
    with pytest.raises(ValueError):
        await get_honcho_client(bad)
