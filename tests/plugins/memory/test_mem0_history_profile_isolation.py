"""Profile isolation for Mem0 OSS history storage."""

from __future__ import annotations

import asyncio

import pytest

from agent.secret_scope import (
    UnscopedSecretError,
    is_multiplex_active,
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.memory.mem0 import _native_memory
from plugins.memory.mem0._native_memory import Memory


@pytest.fixture(autouse=True)
def _restore_multiplex_state():
    previous = is_multiplex_active()
    try:
        yield
    finally:
        set_multiplex_active(previous)


def _native_config() -> dict:
    return {
        "embedder": {"provider": "openai", "config": {}},
        "llm": {"provider": "openai", "config": {}},
        "vector_store": {"provider": "qdrant", "config": {}},
    }


@pytest.mark.asyncio
async def test_concurrent_profiles_use_distinct_history_databases(
    tmp_path,
    monkeypatch,
):
    class Resource:
        async def _initialize(self):
            await asyncio.sleep(0)

        async def reset(self):
            return None

        async def close(self):
            return None

    monkeypatch.setattr(_native_memory, "OpenAIEmbedding", lambda _cfg: Resource())
    monkeypatch.setattr(_native_memory, "OpenAILLM", lambda _cfg: Resource())
    monkeypatch.setattr(_native_memory, "Qdrant", lambda _cfg: Resource())
    monkeypatch.setattr(_native_memory, "NativeNLP", Resource)
    monkeypatch.setattr(
        _native_memory,
        "NativeEntities",
        lambda *_args, **_kwargs: Resource(),
    )
    monkeypatch.setenv("MEM0_DIR", str(tmp_path / "process"))
    set_multiplex_active(True)

    async def initialize(name: str):
        token = set_secret_scope({"MEM0_DIR": str(tmp_path / name)})
        memory = Memory(_native_config())
        try:
            await memory.initialize()
            return memory, memory.db.db_path
        finally:
            reset_secret_scope(token)

    (memory_a, path_a), (memory_b, path_b) = await asyncio.gather(
        initialize("alpha"),
        initialize("beta"),
    )
    try:
        assert {path_a, path_b} == {
            str(tmp_path / "alpha" / "history.db"),
            str(tmp_path / "beta" / "history.db"),
        }
        assert (tmp_path / "alpha" / "history.db").is_file()
        assert (tmp_path / "beta" / "history.db").is_file()
    finally:
        await memory_a.close()
        await memory_b.close()


@pytest.mark.asyncio
async def test_concurrent_profiles_isolate_implicit_history_and_qdrant_storage(
    tmp_path,
    monkeypatch,
):
    class Resource:
        async def _initialize(self):
            await asyncio.sleep(0)

        async def reset(self):
            return None

        async def close(self):
            return None

    class Vector(Resource):
        def __init__(self, config):
            self.config = dict(config)

    monkeypatch.setattr(_native_memory, "OpenAIEmbedding", lambda _cfg: Resource())
    monkeypatch.setattr(_native_memory, "OpenAILLM", lambda _cfg: Resource())
    monkeypatch.setattr(_native_memory, "Qdrant", Vector)
    monkeypatch.setattr(_native_memory, "NativeNLP", Resource)
    monkeypatch.setattr(
        _native_memory,
        "NativeEntities",
        lambda *_args, **_kwargs: Resource(),
    )
    foreign_home = tmp_path / "foreign-home"
    foreign_home.mkdir()
    monkeypatch.setenv("HOME", str(foreign_home))
    monkeypatch.setenv("MEM0_DIR", str(tmp_path / "foreign-mem0"))
    set_multiplex_active(True)

    async def initialize(name: str, scoped: dict[str, str]):
        profile_home = tmp_path / f"profile-{name}"
        profile_home.mkdir()
        config = _native_config()
        home_token = set_hermes_home_override(profile_home)
        secret_token = set_secret_scope(scoped)
        memory = Memory(config)
        try:
            await memory.initialize()
            await memory.db.add_history(
                f"memory-{name}",
                None,
                name,
                "ADD",
            )
            return memory
        finally:
            reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)

    memory_a, memory_b = await asyncio.gather(
        initialize("alpha", {}),
        initialize("beta", {"MEM0_DIR": ""}),
    )
    try:
        expected_a = tmp_path / "profile-alpha"
        expected_b = tmp_path / "profile-beta"
        assert memory_a.db.db_path == str(expected_a / "mem0" / "history.db")
        assert memory_b.db.db_path == str(expected_b / "mem0" / "history.db")
        assert memory_a.vector_store.config["path"] == str(
            expected_a / "mem0_qdrant"
        )
        assert memory_b.vector_store.config["path"] == str(
            expected_b / "mem0_qdrant"
        )
        assert len(await memory_a.db.get_history("memory-alpha")) == 1
        assert await memory_a.db.get_history("memory-beta") == []
        assert len(await memory_b.db.get_history("memory-beta")) == 1
        assert await memory_b.db.get_history("memory-alpha") == []
        assert not (foreign_home / ".mem0" / "history.db").exists()
        assert not (foreign_home / ".hermes" / "mem0_qdrant").exists()
    finally:
        await memory_a.close()
        await memory_b.close()


@pytest.mark.asyncio
async def test_unscoped_multiplex_history_path_fails_closed(monkeypatch):
    def must_not_construct(_config):
        pytest.fail("Mem0 resources were constructed before storage failed closed")

    monkeypatch.setattr(_native_memory, "OpenAIEmbedding", must_not_construct)
    monkeypatch.setattr(_native_memory, "OpenAILLM", must_not_construct)
    monkeypatch.setattr(_native_memory, "Qdrant", must_not_construct)
    monkeypatch.setenv("MEM0_DIR", "/foreign-profile")
    set_multiplex_active(True)
    memory = Memory(_native_config())

    with pytest.raises(UnscopedSecretError, match="MEM0_DIR"):
        await memory.initialize()


@pytest.mark.asyncio
async def test_unscoped_multiplex_implicit_qdrant_path_fails_closed(
    tmp_path,
    monkeypatch,
):
    def must_not_construct(_config):
        pytest.fail("Mem0 resources were constructed before storage failed closed")

    monkeypatch.setattr(_native_memory, "OpenAIEmbedding", must_not_construct)
    monkeypatch.setattr(_native_memory, "OpenAILLM", must_not_construct)
    monkeypatch.setattr(_native_memory, "Qdrant", must_not_construct)
    config = _native_config()
    config["history_db_path"] = str(tmp_path / "history.db")
    set_multiplex_active(True)
    memory = Memory(config)

    with pytest.raises(UnscopedSecretError, match="Qdrant storage"):
        await memory.initialize()


@pytest.mark.asyncio
async def test_empty_scoped_mem0_dir_uses_active_profile_default(
    tmp_path,
    monkeypatch,
):
    paths = []

    class Resource:
        async def _initialize(self):
            return None

        async def close(self):
            return None

    class Database(Resource):
        def __init__(self, path):
            paths.append(path)

    monkeypatch.setattr(_native_memory, "OpenAIEmbedding", lambda _cfg: Resource())
    monkeypatch.setattr(_native_memory, "OpenAILLM", lambda _cfg: Resource())
    monkeypatch.setattr(_native_memory, "Qdrant", lambda _cfg: Resource())
    monkeypatch.setattr(_native_memory, "SQLiteManager", Database)
    monkeypatch.setattr(_native_memory, "NativeNLP", Resource)
    monkeypatch.setattr(
        _native_memory,
        "NativeEntities",
        lambda *_args, **_kwargs: Resource(),
    )
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MEM0_DIR", str(tmp_path / "process"))
    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    set_multiplex_active(True)
    home_token = set_hermes_home_override(profile_home)
    secret_token = set_secret_scope({"MEM0_DIR": ""})
    memory = Memory(_native_config())
    try:
        await memory.initialize()
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)

    assert paths == [str(profile_home / "mem0" / "history.db")]
    await memory.close()


@pytest.mark.asyncio
async def test_single_profile_keeps_legacy_history_and_qdrant_defaults(
    tmp_path,
    monkeypatch,
):
    history_paths = []
    vector_configs = []

    class Resource:
        async def _initialize(self):
            return None

        async def close(self):
            return None

    class Vector(Resource):
        def __init__(self, config):
            vector_configs.append(dict(config))

    class Database(Resource):
        def __init__(self, path):
            history_paths.append(path)

    monkeypatch.setattr(_native_memory, "OpenAIEmbedding", lambda _cfg: Resource())
    monkeypatch.setattr(_native_memory, "OpenAILLM", lambda _cfg: Resource())
    monkeypatch.setattr(_native_memory, "Qdrant", Vector)
    monkeypatch.setattr(_native_memory, "SQLiteManager", Database)
    monkeypatch.setattr(_native_memory, "NativeNLP", Resource)
    monkeypatch.setattr(
        _native_memory,
        "NativeEntities",
        lambda *_args, **_kwargs: Resource(),
    )
    legacy_home = tmp_path / "legacy-home"
    monkeypatch.setenv("HOME", str(legacy_home))
    monkeypatch.delenv("MEM0_DIR", raising=False)
    set_multiplex_active(False)
    memory = Memory(_native_config())

    await memory.initialize()

    assert history_paths == [str(legacy_home / ".mem0" / "history.db")]
    assert vector_configs == [{}]
    await memory.close()


@pytest.mark.asyncio
async def test_multiplex_defaults_canonicalize_profile_home_symlink(
    tmp_path,
    monkeypatch,
):
    history_paths = []
    vector_configs = []

    class Resource:
        async def _initialize(self):
            return None

        async def close(self):
            return None

    class Vector(Resource):
        def __init__(self, config):
            vector_configs.append(dict(config))

    class Database(Resource):
        def __init__(self, path):
            history_paths.append(path)

    monkeypatch.setattr(_native_memory, "OpenAIEmbedding", lambda _cfg: Resource())
    monkeypatch.setattr(_native_memory, "OpenAILLM", lambda _cfg: Resource())
    monkeypatch.setattr(_native_memory, "Qdrant", Vector)
    monkeypatch.setattr(_native_memory, "SQLiteManager", Database)
    monkeypatch.setattr(_native_memory, "NativeNLP", Resource)
    monkeypatch.setattr(
        _native_memory,
        "NativeEntities",
        lambda *_args, **_kwargs: Resource(),
    )
    profile_home = tmp_path / "profile"
    profile_home.mkdir()
    alias = tmp_path / "profile-alias"
    alias.symlink_to(profile_home, target_is_directory=True)
    set_multiplex_active(True)
    home_token = set_hermes_home_override(alias)
    secret_token = set_secret_scope({})
    memory = Memory(_native_config())
    try:
        await memory.initialize()
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)

    assert history_paths == [str(profile_home / "mem0" / "history.db")]
    assert vector_configs == [{"path": str(profile_home / "mem0_qdrant")}]
    await memory.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target_name", "target"),
    [
        pytest.param("client", object(), id="client"),
        pytest.param("url", "explicit-url", id="url"),
        pytest.param("host", "explicit-host", id="host"),
        pytest.param("path", "explicit-path", id="path"),
        pytest.param("path", "", id="empty-path"),
        pytest.param("path", None, id="none-path"),
    ],
)
async def test_explicit_qdrant_target_wins_in_multiplex(
    target_name,
    target,
    tmp_path,
    monkeypatch,
):
    vector_configs = []

    class Resource:
        async def _initialize(self):
            return None

        async def close(self):
            return None

    class Vector(Resource):
        def __init__(self, config):
            vector_configs.append(dict(config))

    monkeypatch.setattr(_native_memory, "OpenAIEmbedding", lambda _cfg: Resource())
    monkeypatch.setattr(_native_memory, "OpenAILLM", lambda _cfg: Resource())
    monkeypatch.setattr(_native_memory, "Qdrant", Vector)
    monkeypatch.setattr(_native_memory, "SQLiteManager", lambda _path: Resource())
    monkeypatch.setattr(_native_memory, "NativeNLP", Resource)
    monkeypatch.setattr(
        _native_memory,
        "NativeEntities",
        lambda *_args, **_kwargs: Resource(),
    )
    config = _native_config()
    config["history_db_path"] = str(tmp_path / "history.db")
    config["vector_store"]["config"][target_name] = target
    set_multiplex_active(True)
    memory = Memory(config)

    await memory.initialize()

    assert vector_configs == [{target_name: target}]
    await memory.close()


@pytest.mark.asyncio
async def test_reset_reuses_initialized_history_path_without_reresolving_scope(
    tmp_path,
    monkeypatch,
):
    paths = []

    class Resource:
        async def _initialize(self):
            return None

        async def reset(self):
            return None

        async def close(self):
            return None

    class Database(Resource):
        def __init__(self, path):
            self.path = path
            paths.append(path)

    monkeypatch.setattr(_native_memory, "OpenAIEmbedding", lambda _cfg: Resource())
    monkeypatch.setattr(_native_memory, "OpenAILLM", lambda _cfg: Resource())
    monkeypatch.setattr(_native_memory, "Qdrant", lambda _cfg: Resource())
    monkeypatch.setattr(_native_memory, "SQLiteManager", Database)
    monkeypatch.setattr(_native_memory, "NativeNLP", Resource)
    monkeypatch.setattr(
        _native_memory,
        "NativeEntities",
        lambda *_args, **_kwargs: Resource(),
    )
    set_multiplex_active(True)
    token = set_secret_scope({"MEM0_DIR": str(tmp_path / "alpha")})
    memory = Memory(_native_config())
    try:
        await memory.initialize()
    finally:
        reset_secret_scope(token)

    # Reset may run after initialization's profile scope has unwound. Mem0's
    # upstream config stores one resolved history_db_path and reuses it.
    await memory.reset()

    expected = str(tmp_path / "alpha" / "history.db")
    assert paths == [expected, expected]
    assert memory._history_db_path == expected
    await memory.close()


@pytest.mark.asyncio
async def test_explicit_history_path_wins_without_reading_mem0_dir(
    tmp_path,
    monkeypatch,
):
    paths = []

    class Resource:
        async def _initialize(self):
            return None

        async def reset(self):
            return None

        async def close(self):
            return None

    class Database(Resource):
        def __init__(self, path):
            paths.append(path)

    monkeypatch.setattr(_native_memory, "OpenAIEmbedding", lambda _cfg: Resource())
    monkeypatch.setattr(_native_memory, "OpenAILLM", lambda _cfg: Resource())
    monkeypatch.setattr(_native_memory, "Qdrant", lambda _cfg: Resource())
    monkeypatch.setattr(_native_memory, "SQLiteManager", Database)
    monkeypatch.setattr(_native_memory, "NativeNLP", Resource)
    monkeypatch.setattr(
        _native_memory,
        "NativeEntities",
        lambda *_args, **_kwargs: Resource(),
    )
    explicit = tmp_path / "explicit" / "history.db"
    config = _native_config()
    config["history_db_path"] = str(explicit)
    config["vector_store"]["config"]["path"] = str(
        tmp_path / "explicit" / "qdrant"
    )
    set_multiplex_active(True)
    memory = Memory(config)

    await memory.initialize()
    await memory.reset()

    assert paths == [str(explicit), str(explicit)]
    await memory.close()


@pytest.mark.asyncio
async def test_repeated_initialize_cancellation_closes_every_created_resource(
    tmp_path,
    monkeypatch,
):
    initialize_started = asyncio.Event()
    all_closes_started = asyncio.Event()
    close_release = asyncio.Event()
    resources = []

    class Resource:
        def __init__(self, *_args, **_kwargs):
            self.closed = False
            self.close_started = False
            resources.append(self)

        async def _initialize(self):
            return None

        async def close(self):
            self.close_started = True
            if all(resource.close_started for resource in resources):
                all_closes_started.set()
            await close_release.wait()
            self.closed = True

    class Vector(Resource):
        async def _initialize(self):
            initialize_started.set()
            await asyncio.Event().wait()

    class Database(Resource):
        def __init__(self, _path):
            super().__init__()

    monkeypatch.setattr(_native_memory, "OpenAIEmbedding", Resource)
    monkeypatch.setattr(_native_memory, "OpenAILLM", Resource)
    monkeypatch.setattr(_native_memory, "Qdrant", Vector)
    monkeypatch.setattr(_native_memory, "SQLiteManager", Database)
    monkeypatch.setattr(_native_memory, "NativeNLP", Resource)
    monkeypatch.setattr(_native_memory, "NativeEntities", Resource)
    monkeypatch.setenv("MEM0_DIR", str(tmp_path / "mem0"))

    memory = Memory(_native_config())
    initialize = asyncio.create_task(memory.initialize())
    await initialize_started.wait()
    initialize.cancel()
    await all_closes_started.wait()
    initialize.cancel()
    await asyncio.sleep(0)
    assert not initialize.done()
    close_release.set()

    with pytest.raises(asyncio.CancelledError):
        await initialize

    assert len(resources) == 6
    assert all(resource.closed for resource in resources)
    assert memory._initialized is False
    assert memory.db is None
