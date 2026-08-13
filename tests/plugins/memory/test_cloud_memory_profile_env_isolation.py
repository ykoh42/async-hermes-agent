"""Profile-scoped non-key settings for retained cloud memory providers."""

from __future__ import annotations

import asyncio

import pytest

from agent.secret_scope import (
    is_multiplex_active,
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.memory import retaindb, supermemory


@pytest.mark.asyncio
async def test_concurrent_profiles_isolate_supermemory_url_and_container(
    tmp_path, monkeypatch
):
    process_url = "https://process-profile.invalid"
    monkeypatch.setenv("SUPERMEMORY_BASE_URL", process_url)
    monkeypatch.setenv("SUPERMEMORY_CONTAINER_TAG", "process-profile")

    async def load_config(_home):
        return supermemory._default_config()

    clients = []

    class Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            clients.append(self)

        async def initialize(self):
            return None

        async def close(self):
            self.closed = True

    monkeypatch.setattr(supermemory, "_load_supermemory_config", load_config)
    monkeypatch.setattr(supermemory, "_SupermemoryClient", Client)

    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)

    async def initialize_profile(name: str):
        home = tmp_path / name
        home.mkdir()
        home_token = set_hermes_home_override(home)
        scope_token = set_secret_scope(
            {
                "SUPERMEMORY_API_KEY": f"key-{name}",
                "SUPERMEMORY_BASE_URL": f"https://{name}.example/",
                "SUPERMEMORY_CONTAINER_TAG": f"container-{name}",
            }
        )
        provider = supermemory.SupermemoryMemoryProvider()
        try:
            await provider.initialize(f"session-{name}", hermes_home=str(home))
            return provider
        finally:
            reset_secret_scope(scope_token)
            reset_hermes_home_override(home_token)

    try:
        profile_a, profile_b = await asyncio.gather(
            initialize_profile("alpha"),
            initialize_profile("beta"),
        )
    finally:
        set_multiplex_active(previous_multiplex)

    assert profile_a._api_key == "key-alpha"
    assert profile_a._base_url == "https://alpha.example"
    assert profile_a._container_tag == "container_alpha"
    assert profile_b._api_key == "key-beta"
    assert profile_b._base_url == "https://beta.example"
    assert profile_b._container_tag == "container_beta"
    assert process_url not in {client.kwargs["base_url"] for client in clients}

    await asyncio.gather(profile_a.shutdown(), profile_b.shutdown())
    assert all(client.closed for client in clients)


@pytest.mark.asyncio
async def test_concurrent_profiles_isolate_retaindb_url_and_project(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("RETAINDB_API_KEY", "process-key")
    monkeypatch.setenv("RETAINDB_BASE_URL", "https://process-profile.invalid")
    monkeypatch.setenv("RETAINDB_PROJECT", "process-project")

    async def load_config():
        return {}

    clients = []

    class Client:
        def __init__(self, api_key, base_url, project):
            self.api_key = api_key
            self.base_url = base_url
            self.project = project
            self.closed = False
            clients.append(self)

        async def close(self):
            self.closed = True

    class Queue:
        def __init__(self, client, path):
            self.client = client
            self.path = path
            self.closed = False

        async def initialize(self):
            return None

        async def shutdown(self):
            self.closed = True

    monkeypatch.setattr(retaindb, "_load_retaindb_config", load_config)
    monkeypatch.setattr(retaindb, "_Client", Client)
    monkeypatch.setattr(retaindb, "_WriteQueue", Queue)

    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)

    async def initialize_profile(name: str):
        home = tmp_path / name
        home.mkdir()
        home_token = set_hermes_home_override(home)
        scope_token = set_secret_scope(
            {
                "RETAINDB_API_KEY": f"key-{name}",
                "RETAINDB_BASE_URL": f"https://{name}.example/",
                "RETAINDB_PROJECT": f"project-{name}",
            }
        )
        provider = retaindb.RetainDBMemoryProvider()
        try:
            await provider.initialize(f"session-{name}", hermes_home=str(home))
            return provider
        finally:
            reset_secret_scope(scope_token)
            reset_hermes_home_override(home_token)

    try:
        profile_a, profile_b = await asyncio.gather(
            initialize_profile("alpha"),
            initialize_profile("beta"),
        )
    finally:
        set_multiplex_active(previous_multiplex)

    assert [(client.api_key, client.base_url, client.project) for client in clients] == [
        ("key-alpha", "https://alpha.example", "project-alpha"),
        ("key-beta", "https://beta.example", "project-beta"),
    ]

    await asyncio.gather(profile_a.shutdown(), profile_b.shutdown())
    assert all(client.closed for client in clients)


@pytest.mark.asyncio
async def test_supermemory_empty_profile_values_fall_through_like_upstream(
    tmp_path, monkeypatch
):
    """Upstream treats empty environment URL/tag values as absent."""

    config = supermemory._default_config()
    config.update(
        base_url="https://config.example/",
        container_tag="config-container",
    )

    async def load_config(_home):
        return config

    captured = {}

    class Client:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def initialize(self):
            return None

        async def close(self):
            return None

    monkeypatch.setenv("SUPERMEMORY_BASE_URL", "https://process.invalid")
    monkeypatch.setenv("SUPERMEMORY_CONTAINER_TAG", "process-container")
    monkeypatch.setattr(supermemory, "_load_supermemory_config", load_config)
    monkeypatch.setattr(supermemory, "_SupermemoryClient", Client)

    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    scope_token = set_secret_scope(
        {
            "SUPERMEMORY_API_KEY": "key",
            "SUPERMEMORY_BASE_URL": "",
            "SUPERMEMORY_CONTAINER_TAG": "",
        }
    )
    provider = supermemory.SupermemoryMemoryProvider()
    try:
        await provider.initialize("session", hermes_home=str(tmp_path))
    finally:
        reset_secret_scope(scope_token)
        set_multiplex_active(previous_multiplex)

    assert captured["base_url"] == "https://config.example"
    assert captured["container_tag"] == "config_container"
    await provider.shutdown()


@pytest.mark.asyncio
async def test_retaindb_empty_profile_values_fall_through_to_config_like_upstream(
    tmp_path, monkeypatch
):
    """Upstream's ``env or config`` treats an explicit empty value as absent."""

    async def load_config():
        return {
            "base_url": "https://config.example",
            "project": "config-project",
        }

    captured = {}

    class Client:
        def __init__(self, api_key, base_url, project):
            captured.update(api_key=api_key, base_url=base_url, project=project)

        async def close(self):
            return None

    class Queue:
        def __init__(self, _client, _path):
            pass

        async def initialize(self):
            return None

        async def shutdown(self):
            return None

    monkeypatch.setenv("RETAINDB_API_KEY", "process-key")
    monkeypatch.setenv("RETAINDB_BASE_URL", "https://process.invalid")
    monkeypatch.setenv("RETAINDB_PROJECT", "process-project")
    monkeypatch.setattr(retaindb, "_load_retaindb_config", load_config)
    monkeypatch.setattr(retaindb, "_Client", Client)
    monkeypatch.setattr(retaindb, "_WriteQueue", Queue)

    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    scope_token = set_secret_scope(
        {
            "RETAINDB_API_KEY": "key",
            "RETAINDB_BASE_URL": "",
            "RETAINDB_PROJECT": "",
        }
    )
    provider = retaindb.RetainDBMemoryProvider()
    try:
        await provider.initialize("session", hermes_home=str(tmp_path / "profile"))
    finally:
        reset_secret_scope(scope_token)
        set_multiplex_active(previous_multiplex)
    assert captured == {
        "api_key": "key",
        "base_url": "https://config.example",
        "project": "config-project",
    }
    await provider.shutdown()


@pytest.mark.asyncio
async def test_supermemory_repeated_initialize_cancellation_finishes_close(
    tmp_path, monkeypatch
):
    initialize_started = asyncio.Event()
    close_started = asyncio.Event()
    close_release = asyncio.Event()
    clients = []

    class Client:
        def __init__(self, **_kwargs):
            self.closed = False
            clients.append(self)

        async def initialize(self):
            initialize_started.set()
            await asyncio.Event().wait()

        async def close(self):
            close_started.set()
            await close_release.wait()
            self.closed = True

    async def load_config(_home):
        return supermemory._default_config()

    monkeypatch.setenv("SUPERMEMORY_API_KEY", "key")
    monkeypatch.setattr(supermemory, "_load_supermemory_config", load_config)
    monkeypatch.setattr(supermemory, "_SupermemoryClient", Client)

    provider = supermemory.SupermemoryMemoryProvider()
    initialize = asyncio.create_task(
        provider.initialize("session", hermes_home=str(tmp_path))
    )
    await initialize_started.wait()
    initialize.cancel()
    await close_started.wait()
    initialize.cancel()
    await asyncio.sleep(0)
    assert not initialize.done()
    close_release.set()
    with pytest.raises(asyncio.CancelledError):
        await initialize

    assert clients[0].closed is True
    assert provider._client is None
    assert provider._active is False


@pytest.mark.asyncio
async def test_retaindb_repeated_initialize_cancellation_finishes_close(
    tmp_path, monkeypatch
):
    initialize_started = asyncio.Event()
    close_started = asyncio.Event()
    close_release = asyncio.Event()
    clients = []

    class Client:
        def __init__(self, *_args):
            self.closed = False
            clients.append(self)

        async def close(self):
            close_started.set()
            await close_release.wait()
            self.closed = True

    class Queue:
        def __init__(self, *_args):
            pass

        async def initialize(self):
            initialize_started.set()
            await asyncio.Event().wait()

    async def load_config():
        return {}

    monkeypatch.setenv("RETAINDB_API_KEY", "key")
    monkeypatch.setattr(retaindb, "_load_retaindb_config", load_config)
    monkeypatch.setattr(retaindb, "_Client", Client)
    monkeypatch.setattr(retaindb, "_WriteQueue", Queue)

    provider = retaindb.RetainDBMemoryProvider()
    initialize = asyncio.create_task(
        provider.initialize("session", hermes_home=str(tmp_path))
    )
    await initialize_started.wait()
    initialize.cancel()
    await close_started.wait()
    initialize.cancel()
    await asyncio.sleep(0)
    assert not initialize.done()
    close_release.set()
    with pytest.raises(asyncio.CancelledError):
        await initialize

    assert clients[0].closed is True
    assert provider._client is None
    assert provider._queue is None


@pytest.mark.asyncio
async def test_internal_http_clients_finish_close_through_repeated_cancellation():
    super_close_started = asyncio.Event()
    super_close_release = asyncio.Event()

    class SuperSDK:
        def __init__(self):
            self.closed = False

        async def close(self):
            super_close_started.set()
            await super_close_release.wait()
            self.closed = True

    class SuperHTTP:
        def __init__(self):
            self.is_closed = False

        async def aclose(self):
            self.is_closed = True

    super_client = supermemory._SupermemoryClient(
        "key",
        1.0,
        "container",
        base_url="https://profile.example",
    )
    super_sdk = SuperSDK()
    super_http = SuperHTTP()
    super_client._client = super_sdk
    super_client._http_client = super_http
    super_close = asyncio.create_task(super_client.close())
    await super_close_started.wait()
    super_close.cancel()
    super_close.cancel()
    await asyncio.sleep(0)
    assert not super_close.done()
    super_close_release.set()
    with pytest.raises(asyncio.CancelledError):
        await super_close
    assert super_sdk.closed is True
    assert super_http.is_closed is True

    retain_close_started = asyncio.Event()
    retain_close_release = asyncio.Event()

    class RetainHTTP:
        def __init__(self):
            self.closed = False

        async def aclose(self):
            retain_close_started.set()
            await retain_close_release.wait()
            self.closed = True

    retain_client = retaindb._Client(
        "key",
        "https://profile.example",
        "project",
    )
    retain_http = RetainHTTP()
    retain_client._http_client = retain_http
    retain_close = asyncio.create_task(retain_client.close())
    await retain_close_started.wait()
    retain_close.cancel()
    retain_close.cancel()
    await asyncio.sleep(0)
    assert not retain_close.done()
    retain_close_release.set()
    with pytest.raises(asyncio.CancelledError):
        await retain_close
    assert retain_http.closed is True
    assert retain_client._http_client is None


def test_cloud_memory_profile_state_does_not_cross_sequential_event_loops(
    tmp_path, monkeypatch
):
    supermemory_clients = []
    retaindb_clients = []
    loops = []

    class SupermemoryClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            supermemory_clients.append(self)

        async def initialize(self):
            return None

        async def close(self):
            self.closed = True

    class RetainDBClient:
        def __init__(self, api_key, base_url, project):
            self.settings = (api_key, base_url, project)
            self.closed = False
            retaindb_clients.append(self)

        async def close(self):
            self.closed = True

    class Queue:
        def __init__(self, *_args):
            pass

        async def initialize(self):
            return None

        async def shutdown(self):
            return None

    async def load_supermemory_config(_home):
        return supermemory._default_config()

    async def load_retaindb_config():
        return {}

    monkeypatch.setattr(
        supermemory,
        "_load_supermemory_config",
        load_supermemory_config,
    )
    monkeypatch.setattr(supermemory, "_SupermemoryClient", SupermemoryClient)
    monkeypatch.setattr(retaindb, "_load_retaindb_config", load_retaindb_config)
    monkeypatch.setattr(retaindb, "_Client", RetainDBClient)
    monkeypatch.setattr(retaindb, "_WriteQueue", Queue)

    async def exercise(name: str) -> None:
        loops.append(asyncio.get_running_loop())
        home = tmp_path / name
        home.mkdir()
        home_token = set_hermes_home_override(home)
        scope_token = set_secret_scope(
            {
                "SUPERMEMORY_API_KEY": f"super-key-{name}",
                "SUPERMEMORY_BASE_URL": f"https://super-{name}.example",
                "SUPERMEMORY_CONTAINER_TAG": f"super-{name}",
                "RETAINDB_API_KEY": f"retain-key-{name}",
                "RETAINDB_BASE_URL": f"https://retain-{name}.example",
                "RETAINDB_PROJECT": f"retain-{name}",
            }
        )
        super_provider = supermemory.SupermemoryMemoryProvider()
        retain_provider = retaindb.RetainDBMemoryProvider()
        try:
            await super_provider.initialize("session", hermes_home=str(home))
            await retain_provider.initialize("session", hermes_home=str(home))
            await super_provider.shutdown()
            await retain_provider.shutdown()
        finally:
            reset_secret_scope(scope_token)
            reset_hermes_home_override(home_token)

    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    try:
        asyncio.run(exercise("alpha"))
        asyncio.run(exercise("beta"))
    finally:
        set_multiplex_active(previous_multiplex)

    assert loops[0] is not loops[1]
    assert [client.kwargs["api_key"] for client in supermemory_clients] == [
        "super-key-alpha",
        "super-key-beta",
    ]
    assert [client.settings for client in retaindb_clients] == [
        (
            "retain-key-alpha",
            "https://retain-alpha.example",
            "retain-alpha",
        ),
        (
            "retain-key-beta",
            "https://retain-beta.example",
            "retain-beta",
        ),
    ]
    assert all(client.closed for client in supermemory_clients)
    assert all(client.closed for client in retaindb_clients)
