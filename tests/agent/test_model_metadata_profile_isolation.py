"""Profile and event-loop isolation for model metadata caches."""

from __future__ import annotations

import asyncio
import gc
import json
import weakref
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from agent import model_metadata as metadata
from hermes_constants import (
    reset_hermes_home_override,
    set_hermes_home_override,
)


@contextmanager
def _profile(home):
    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _client(response):
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


@pytest.fixture(autouse=True)
def _reset_private_caches():
    metadata._endpoint_model_metadata_cache.clear()
    metadata._endpoint_model_metadata_cache_time.clear()
    metadata._LOCAL_CTX_PROBE_CACHE.clear()
    yield
    metadata._endpoint_model_metadata_cache.clear()
    metadata._endpoint_model_metadata_cache_time.clear()
    metadata._LOCAL_CTX_PROBE_CACHE.clear()


@pytest.mark.asyncio
async def test_context_cache_serializes_same_profile_rmw(tmp_path):
    """Concurrent saves must retain both rows in one profile file."""
    profile = tmp_path / "profile"

    async def save(model, length):
        with _profile(profile):
            await metadata.save_context_length(model, "https://models.test/v1", length)

    await asyncio.gather(save("model-a", 64_000), save("model-b", 128_000))

    payload = yaml.safe_load(
        (profile / "context_length_cache.yaml").read_text(encoding="utf-8")
    )
    assert payload["context_lengths"] == {
        "model-a@https://models.test/v1": 64_000,
        "model-b@https://models.test/v1": 128_000,
    }


@pytest.mark.asyncio
async def test_context_cache_keeps_concurrent_profiles_separate(tmp_path):
    """The same cache key may resolve differently in two profile homes."""
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"

    async def save(home, length):
        with _profile(home):
            await metadata.save_context_length(
                "account-model", "https://models.test/v1", length
            )

    await asyncio.gather(save(profile_a, 64_000), save(profile_b, 128_000))

    for profile, expected in ((profile_a, 64_000), (profile_b, 128_000)):
        payload = yaml.safe_load(
            (profile / "context_length_cache.yaml").read_text(encoding="utf-8")
        )
        assert payload["context_lengths"] == {
            "account-model@https://models.test/v1": expected
        }


@pytest.mark.asyncio
async def test_context_cache_aliases_share_one_rmw_lock(tmp_path):
    """Two lexical homes resolving to one directory must not lose an update."""
    profile = tmp_path / "profile"
    profile.mkdir()
    alias = tmp_path / "profile-alias"
    alias.symlink_to(profile, target_is_directory=True)

    async def save(home, model, length):
        with _profile(home):
            await metadata.save_context_length(model, "https://models.test/v1", length)

    await asyncio.gather(
        save(profile, "model-a", 64_000),
        save(alias, "model-b", 128_000),
    )

    payload = yaml.safe_load(
        (profile / "context_length_cache.yaml").read_text(encoding="utf-8")
    )
    assert set(payload["context_lengths"]) == {
        "model-a@https://models.test/v1",
        "model-b@https://models.test/v1",
    }


def test_disk_locks_do_not_bind_subsequent_asyncio_run(tmp_path):
    """A contended lock from a closed loop must not be reused by a new loop."""
    profile = tmp_path / "profile"

    async def write_pair(suffix):
        with _profile(profile):
            await asyncio.gather(
                metadata.save_context_length(
                    f"model-{suffix}-a", "https://models.test/v1", 64_000
                ),
                metadata.save_context_length(
                    f"model-{suffix}-b", "https://models.test/v1", 128_000
                ),
                metadata._local_probe_disk_put(
                    "server_type", f"endpoint-{suffix}-a", "ollama"
                ),
                metadata._local_probe_disk_put(
                    "server_type", f"endpoint-{suffix}-b", "vllm"
                ),
            )

    first_loop = asyncio.new_event_loop()
    first_ref = weakref.ref(first_loop)
    try:
        first_loop.run_until_complete(write_pair("first"))
    finally:
        first_loop.close()
        del first_loop
    gc.collect()

    asyncio.run(write_pair("second"))
    assert first_ref() is None


@pytest.mark.asyncio
async def test_local_probe_disk_round_trip_and_concurrent_rmw(tmp_path):
    """A native disk round-trip must retain both concurrent probe rows."""
    profile = tmp_path / "profile"
    with _profile(profile):
        await asyncio.gather(
            metadata._local_probe_disk_put("server_type", "endpoint-a", "ollama"),
            metadata._local_probe_disk_put("server_type", "endpoint-b", "vllm"),
        )
        assert await metadata._local_probe_disk_get(
            "server_type", "endpoint-a"
        ) == "ollama"
        assert await metadata._local_probe_disk_get(
            "server_type", "endpoint-b"
        ) == "vllm"

    payload = json.loads(
        (profile / "cache" / "local_endpoint_probes.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(payload) == {
        "server_type:endpoint-a",
        "server_type:endpoint-b",
    }


@pytest.mark.asyncio
async def test_endpoint_metadata_cache_is_scoped_by_credential(tmp_path):
    """One account's endpoint catalogue must not satisfy another account."""
    response_a = MagicMock(status_code=200)
    response_a.raise_for_status = MagicMock()
    response_a.json.return_value = {
        "data": [{"id": "account-a-model", "context_length": 64_000}]
    }
    response_b = MagicMock(status_code=200)
    response_b.raise_for_status = MagicMock()
    response_b.json.return_value = {
        "data": [{"id": "account-b-model", "context_length": 128_000}]
    }
    endpoint = "https://private.models.test/v1"
    responses = {
        ("Bearer secret-account-a", endpoint + "/models"): response_a,
        ("Bearer secret-account-b", endpoint + "/models"): response_b,
    }
    clients = {}

    def client_for_request(*, headers, **_kwargs):
        authorization = headers["Authorization"]
        client = MagicMock()

        async def get(url):
            return responses[(authorization, url)]

        client.get = AsyncMock(side_effect=get)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        clients[authorization] = client
        return client

    async def fetch(home, api_key):
        with _profile(home):
            return await metadata.fetch_endpoint_model_metadata(
                endpoint, api_key=api_key
            )

    with patch(
        "agent.model_metadata._create_httpx_client",
        new=AsyncMock(side_effect=client_for_request),
    ) as create_client:
        catalogue_a, catalogue_b = await asyncio.gather(
            fetch(tmp_path / "profile-a", "secret-account-a"),
            fetch(tmp_path / "profile-b", "secret-account-b"),
        )
        with _profile(tmp_path / "profile-a"):
            catalogue_a_again = await metadata.fetch_endpoint_model_metadata(
                endpoint, api_key="secret-account-a"
            )

    assert set(catalogue_a) == set(catalogue_a_again) == {"account-a-model"}
    assert set(catalogue_b) == {"account-b-model"}
    assert all(
        "secret-account" not in repr(key)
        for key in metadata._endpoint_model_metadata_cache
    )
    assert create_client.await_count == 2
    assert set(clients) == {
        "Bearer secret-account-a",
        "Bearer secret-account-b",
    }
    assert all(client.get.await_count == 1 for client in clients.values())


@pytest.mark.asyncio
async def test_local_context_probe_cache_is_scoped_by_credential():
    """A credential-specific live context response must not cross accounts."""
    with patch(
        "agent.model_metadata._query_local_context_length_uncached",
        new_callable=AsyncMock,
        side_effect=[64_000, 128_000],
    ) as probe:
        first = await metadata._query_local_context_length(
            "model", "https://private.models.test/v1", api_key="secret-a"
        )
        second = await metadata._query_local_context_length(
            "model", "https://private.models.test/v1", api_key="secret-b"
        )
        first_again = await metadata._query_local_context_length(
            "model", "https://private.models.test/v1", api_key="secret-a"
        )

    assert (first, second, first_again) == (64_000, 128_000, 64_000)
    assert probe.await_count == 2
    assert all("secret-" not in repr(key) for key in metadata._LOCAL_CTX_PROBE_CACHE)
