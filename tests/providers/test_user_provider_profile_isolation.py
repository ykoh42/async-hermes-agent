"""User model-provider discovery is isolated by loop and HERMES_HOME."""

from __future__ import annotations

import asyncio
import concurrent.futures
import gc
import sys
import types
import weakref
from pathlib import Path

import pytest

import providers
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from providers.base import ProviderProfile


@pytest.fixture(autouse=True)
def clear_user_provider_states():
    providers._clear_user_provider_states()
    yield
    providers._clear_user_provider_states()


def _write_provider(
    home: Path,
    *,
    directory: str,
    name: str,
    base_url: str,
    aliases: tuple[str, ...] = (),
    env_vars: tuple[str, ...] = (),
) -> Path:
    plugin_dir = home / "plugins" / "model-providers" / directory
    plugin_dir.mkdir(parents=True)
    plugin_dir.joinpath("__init__.py").write_text(
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        "register_provider(ProviderProfile(\n"
        f"    name={name!r},\n"
        f"    aliases={aliases!r},\n"
        f"    env_vars={env_vars!r},\n"
        f"    base_url={base_url!r},\n"
        "    auth_type='api_key',\n"
        "))\n",
        encoding="utf-8",
    )
    return plugin_dir


async def _lookup(home: Path, name: str) -> ProviderProfile | None:
    token = set_hermes_home_override(home)
    try:
        return await providers.get_provider_profile(name)
    finally:
        reset_hermes_home_override(token)


async def _listed(home: Path) -> list[ProviderProfile]:
    token = set_hermes_home_override(home)
    try:
        return await providers.list_providers()
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_sequential_profiles_get_same_name_overrides_and_aliases(tmp_path):
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    _write_provider(
        home_a,
        directory="gmi",
        name="gmi",
        base_url="https://a.example/v1",
        aliases=("profile-a-gmi",),
    )
    _write_provider(
        home_b,
        directory="gmi",
        name="gmi",
        base_url="https://b.example/v1",
        aliases=("profile-b-gmi",),
    )

    profile_a = await _lookup(home_a, "gmi")
    alias_a = await _lookup(home_a, "profile-a-gmi")
    profile_b = await _lookup(home_b, "gmi")
    alias_b = await _lookup(home_b, "profile-b-gmi")

    assert profile_a is alias_a
    assert profile_b is alias_b
    assert profile_a is not None and profile_a.base_url == "https://a.example/v1"
    assert profile_b is not None and profile_b.base_url == "https://b.example/v1"
    assert await _lookup(home_b, "profile-a-gmi") is None
    assert await _lookup(home_a, "profile-b-gmi") is None
    assert providers._REGISTRY["gmi"] is not profile_a
    assert providers._REGISTRY["gmi"] is not profile_b


@pytest.mark.asyncio
async def test_profile_without_override_keeps_bundled_provider(tmp_path):
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    _write_provider(
        home_a,
        directory="gmi",
        name="gmi",
        base_url="https://a.example/v1",
        aliases=("profile-a-only-alias",),
    )
    home_b.mkdir()

    profile_a = await _lookup(home_a, "gmi")
    profile_b = await _lookup(home_b, "gmi")

    assert profile_a is not None and profile_a.base_url == "https://a.example/v1"
    assert profile_b is providers._REGISTRY["gmi"]
    assert await _lookup(home_b, "profile-a-only-alias") is None


@pytest.mark.asyncio
async def test_concurrent_profiles_do_not_share_same_name_module_or_list(tmp_path):
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    _write_provider(
        home_a,
        directory="same-name",
        name="same-name",
        base_url="https://a.example/v1",
        aliases=("same-a",),
    )
    _write_provider(
        home_b,
        directory="same-name",
        name="same-name",
        base_url="https://b.example/v1",
        aliases=("same-b",),
    )

    profile_a, profile_b, listed_a, listed_b = await asyncio.gather(
        _lookup(home_a, "same-name"),
        _lookup(home_b, "same-name"),
        _listed(home_a),
        _listed(home_b),
    )

    assert profile_a is not None and profile_a.base_url == "https://a.example/v1"
    assert profile_b is not None and profile_b.base_url == "https://b.example/v1"
    assert next(p for p in listed_a if p.name == "same-name") is profile_a
    assert next(p for p in listed_b if p.name == "same-name") is profile_b
    module_names = [
        name
        for name in sys.modules
        if name.startswith("_hermes_user_provider_")
        and name.endswith("_same_name")
    ]
    assert len(module_names) == 2
    assert module_names[0] != module_names[1]


@pytest.mark.asyncio
async def test_concurrent_same_profile_uses_one_discovery_and_module(tmp_path):
    home = tmp_path / "profile"
    _write_provider(
        home,
        directory="shared",
        name="profile-shared",
        base_url="https://shared.example/v1",
    )

    first, second = await asyncio.gather(
        _lookup(home, "profile-shared"),
        _lookup(home, "profile-shared"),
    )

    assert first is second
    matching_modules = [
        name
        for name in sys.modules
        if name.startswith("_hermes_user_provider_") and name.endswith("_shared")
    ]
    assert len(matching_modules) == 1


@pytest.mark.asyncio
async def test_symlinked_home_reuses_canonical_profile_state(tmp_path):
    home = tmp_path / "profile"
    alias_home = tmp_path / "profile-link"
    _write_provider(
        home,
        directory="canonical",
        name="canonical-profile",
        base_url="https://canonical.example/v1",
    )
    try:
        alias_home.symlink_to(home, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are unavailable")

    direct = await _lookup(home, "canonical-profile")
    through_alias = await _lookup(alias_home, "canonical-profile")

    assert direct is through_alias
    current_loop_states = providers._USER_PROVIDER_STATES[
        asyncio.get_running_loop()
    ]
    assert len(current_loop_states) == 1


@pytest.mark.asyncio
async def test_a_only_provider_is_not_projected_into_process_globals(tmp_path):
    from hermes_cli import auth, config, models

    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    _write_provider(
        home_a,
        directory="a-only",
        name="a-only-provider",
        base_url="https://a-only.example/v1",
        aliases=("a-only-alias",),
        env_vars=("A_ONLY_API_KEY",),
    )
    home_b.mkdir()

    assert await _lookup(home_a, "a-only-alias") is not None
    # Agent initialization repeats these historical projection calls after
    # discovery. They must still see only the process-shared bundled layer.
    auth._inject_profile_provider_registry()
    config._inject_profile_env_vars()
    models._inject_profile_canonical_providers()
    assert await _lookup(home_b, "a-only-provider") is None
    assert "a-only-provider" not in auth.PROVIDER_REGISTRY
    assert "a-only-alias" not in auth.PROVIDER_REGISTRY
    assert "A_ONLY_API_KEY" not in config.OPTIONAL_ENV_VARS
    assert models.normalize_provider("a-only-alias") == "a-only-alias"


@pytest.mark.asyncio
async def test_cancelled_user_discovery_rolls_back_registry_and_modules(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / "cancelled-profile"
    _write_provider(
        home,
        directory="cancelled",
        name="cancelled-profile",
        base_url="https://complete.example/v1",
        aliases=("cancelled-alias",),
    )
    started = asyncio.Event()
    original_loader = providers._load_source_package

    async def interrupted_loader(module_name, _init_file):
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module
        providers.register_provider(
            ProviderProfile(
                name="partial-profile",
                aliases=("partial-alias",),
                base_url="https://partial.example/v1",
            )
        )
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(providers, "_load_source_package", interrupted_loader)
    task = asyncio.create_task(_lookup(home, "cancelled-profile"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert asyncio.get_running_loop() not in providers._USER_PROVIDER_LOCKS
    assert all(
        "partial-profile" not in state.registry
        for per_loop in providers._USER_PROVIDER_STATES.values()
        for state in per_loop.values()
    )
    assert not any(
        name.startswith("_hermes_user_provider_") and name.endswith("_cancelled")
        for name in sys.modules
    )

    monkeypatch.setattr(providers, "_load_source_package", original_loader)
    completed = await _lookup(home, "cancelled-alias")
    assert completed is not None
    assert completed.name == "cancelled-profile"
    assert completed.base_url == "https://complete.example/v1"


@pytest.mark.asyncio
async def test_failed_plugin_cannot_leave_partial_profile_or_module(tmp_path):
    home = tmp_path / "profile"
    broken = _write_provider(
        home,
        directory="broken",
        name="partial-before-error",
        base_url="https://partial.example/v1",
        aliases=("partial-before-error-alias",),
    )
    with broken.joinpath("__init__.py").open("a", encoding="utf-8") as handle:
        handle.write("raise RuntimeError('broken provider')\n")
    _write_provider(
        home,
        directory="valid",
        name="valid-after-error",
        base_url="https://valid.example/v1",
    )

    valid = await _lookup(home, "valid-after-error")

    assert valid is not None
    assert await _lookup(home, "partial-before-error") is None
    assert await _lookup(home, "partial-before-error-alias") is None
    assert not any(
        name.startswith("_hermes_user_provider_") and name.endswith("_broken")
        for name in sys.modules
    )


@pytest.mark.asyncio
async def test_private_cleanup_unloads_owned_modules_and_source_finders(tmp_path):
    home = tmp_path / "profile"
    _write_provider(
        home,
        directory="cleanup",
        name="cleanup-profile",
        base_url="https://cleanup.example/v1",
    )
    assert await _lookup(home, "cleanup-profile") is not None
    module_name = next(
        name
        for name in sys.modules
        if name.startswith("_hermes_user_provider_") and name.endswith("_cleanup")
    )
    finder = sys.modules[module_name].__hermes_async_source_finder__
    assert finder in sys.meta_path

    providers._clear_user_provider_states()

    assert module_name not in sys.modules
    assert finder not in sys.meta_path
    assert not providers._USER_PROVIDER_STATES
    assert not providers._USER_PROVIDER_LOCKS


def test_separate_event_loops_do_not_reuse_user_plugin_module(tmp_path):
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    _write_provider(
        home_a,
        directory="cross-loop",
        name="cross-loop",
        base_url="https://a.example/v1",
    )
    _write_provider(
        home_b,
        directory="cross-loop",
        name="cross-loop",
        base_url="https://b.example/v1",
    )

    profile_a = asyncio.run(_lookup(home_a, "cross-loop"))
    profile_b = asyncio.run(_lookup(home_b, "cross-loop"))

    assert profile_a is not None and profile_a.base_url == "https://a.example/v1"
    assert profile_b is not None and profile_b.base_url == "https://b.example/v1"
    matching_modules = [
        name
        for name in sys.modules
        if name.startswith("_hermes_user_provider_")
        and name.endswith("_cross_loop")
    ]
    assert len(matching_modules) == 2


def test_concurrent_event_loops_keep_same_name_profiles_isolated(tmp_path):
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    _write_provider(
        home_a,
        directory="threaded-loop",
        name="threaded-loop",
        base_url="https://a.example/v1",
        aliases=("threaded-a",),
    )
    _write_provider(
        home_b,
        directory="threaded-loop",
        name="threaded-loop",
        base_url="https://b.example/v1",
        aliases=("threaded-b",),
    )
    def load(home: Path, alias: str):
        return asyncio.run(_lookup(home, alias))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(load, home_a, "threaded-a")
        future_b = executor.submit(load, home_b, "threaded-b")
        profile_a = future_a.result(timeout=10)
        profile_b = future_b.result(timeout=10)

    assert profile_a is not None and profile_a.base_url == "https://a.example/v1"
    assert profile_b is not None and profile_b.base_url == "https://b.example/v1"
    assert profile_a is not profile_b


def test_completed_discovery_does_not_retain_closed_event_loop(tmp_path):
    home = tmp_path / "collectable-loop"
    _write_provider(
        home,
        directory="collectable",
        name="collectable",
        base_url="https://collectable.example/v1",
    )

    async def load_twice():
        return await asyncio.gather(
            _lookup(home, "collectable"),
            _lookup(home, "collectable"),
        )

    loop = asyncio.new_event_loop()
    loop_ref = weakref.ref(loop)
    try:
        first, second = loop.run_until_complete(load_twice())
        assert first is second
        assert loop not in providers._USER_PROVIDER_LOCKS
        assert not providers._SHARED_DISCOVERY_RUNNING
        assert not providers._SHARED_DISCOVERY_WAITERS
    finally:
        loop.close()
    del loop
    for _ in range(3):
        gc.collect()

    assert loop_ref() is None
