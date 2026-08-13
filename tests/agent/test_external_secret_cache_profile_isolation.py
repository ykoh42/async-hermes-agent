"""Profile and cancellation isolation for bundled external-secret caches."""

from __future__ import annotations

import asyncio
import gc
import json
import weakref
from pathlib import Path

import pytest

from agent.secret_sources import bitwarden, onepassword
from agent.secret_sources._cache import CachedFetch, DiskCache
from hermes_constants import (
    reset_hermes_home_override,
    set_hermes_home_override,
)


@pytest.fixture(autouse=True)
def clear_l1_caches():
    bitwarden._CACHE.clear()
    onepassword._CACHE.clear()
    yield
    bitwarden._CACHE.clear()
    onepassword._CACHE.clear()


@pytest.mark.asyncio
async def test_bitwarden_l1_never_crosses_canonical_profile_home(
    tmp_path,
    monkeypatch,
):
    calls = 0

    async def fetch_live(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return {"API_KEY": f"profile-{calls}"}, []

    monkeypatch.setattr(bitwarden, "_run_bws_list", fetch_live)
    kwargs = {
        "access_token": "same-token",
        "project_id": "same-project",
        "binary": Path("/unused/bws"),
    }
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"

    first, _ = await bitwarden.fetch_bitwarden_secrets(
        **kwargs,
        home_path=home_a,
    )
    second, _ = await bitwarden.fetch_bitwarden_secrets(
        **kwargs,
        home_path=home_b,
    )
    first_again, _ = await bitwarden.fetch_bitwarden_secrets(
        **kwargs,
        home_path=home_a,
    )

    assert first == first_again == {"API_KEY": "profile-1"}
    assert second == {"API_KEY": "profile-2"}
    assert calls == 2


@pytest.mark.asyncio
async def test_onepassword_omitted_home_uses_active_profile(tmp_path, monkeypatch):
    calls = 0

    async def read_live(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return f"profile-{calls}"

    monkeypatch.setattr(onepassword, "_run_op_read", read_live)

    async def fetch(home):
        token = set_hermes_home_override(home)
        try:
            secrets, _ = await onepassword.fetch_onepassword_secrets(
                references={"API_KEY": "op://vault/item/field"},
                binary=Path("/unused/op"),
            )
            return secrets
        finally:
            reset_hermes_home_override(token)

    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    first = await fetch(home_a)
    second = await fetch(home_b)
    first_again = await fetch(home_a)

    assert first == first_again == {"API_KEY": "profile-1"}
    assert second == {"API_KEY": "profile-2"}
    assert calls == 2


@pytest.mark.asyncio
async def test_symlink_alias_reuses_onepassword_l1_entry(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    alias = tmp_path / "profile-alias"
    home.mkdir()
    alias.symlink_to(home, target_is_directory=True)
    calls = 0

    async def read_live(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return "profile-secret"

    monkeypatch.setattr(onepassword, "_run_op_read", read_live)
    kwargs = {
        "references": {"API_KEY": "op://vault/item/field"},
        "binary": Path("/unused/op"),
    }
    direct, _ = await onepassword.fetch_onepassword_secrets(
        **kwargs,
        home_path=home,
    )
    through_alias, _ = await onepassword.fetch_onepassword_secrets(
        **kwargs,
        home_path=alias,
    )

    assert direct == through_alias == {"API_KEY": "profile-secret"}
    assert calls == 1
    assert len(onepassword._CACHE) == 1


@pytest.mark.asyncio
async def test_concurrent_disk_writes_remain_complete_and_leave_no_staging_files(
    tmp_path,
):
    cache = DiskCache("concurrent.json", key_serializer=str)
    entries = [
        CachedFetch(secrets={"VALUE": str(index)}, fetched_at=10**10)
        for index in range(20)
    ]

    await asyncio.gather(
        *(cache.write(str(index), entry, 300, tmp_path) for index, entry in enumerate(entries))
    )

    payload = json.loads(cache.path(tmp_path).read_text(encoding="utf-8"))
    assert payload["secrets"]["VALUE"] == payload["key"]
    assert not list(cache.path(tmp_path).parent.glob(".concurrent_*.tmp"))


@pytest.mark.asyncio
async def test_repeated_cancellation_finishes_owned_disk_staging_cleanup(
    tmp_path,
    monkeypatch,
):
    cache = DiskCache("cancel.json", key_serializer=str)
    replace_started = asyncio.Event()
    remove_started = asyncio.Event()
    allow_remove = asyncio.Event()
    original_remove = bitwarden.aiofiles.os.remove

    async def blocking_replace(_source, _destination):
        replace_started.set()
        await asyncio.Event().wait()

    async def blocking_remove(path):
        if Path(path).name.startswith(".cancel_"):
            remove_started.set()
            await allow_remove.wait()
        await original_remove(path)

    monkeypatch.setattr("agent.secret_sources._cache.aiofiles.os.replace", blocking_replace)
    monkeypatch.setattr("agent.secret_sources._cache.aiofiles.os.remove", blocking_remove)
    task = asyncio.create_task(
        cache.write(
            "key",
            CachedFetch(secrets={"API_KEY": "secret"}, fetched_at=10**10),
            300,
            tmp_path,
        )
    )
    await replace_started.wait()
    task.cancel()
    await remove_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    allow_remove.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert not list(cache.path(tmp_path).parent.glob(".cancel_*.tmp"))


@pytest.mark.asyncio
async def test_repeated_cancellation_cleans_encrypted_bitwarden_staging_file(
    tmp_path,
    monkeypatch,
):
    replace_started = asyncio.Event()
    remove_started = asyncio.Event()
    allow_remove = asyncio.Event()
    original_remove = bitwarden.aiofiles.os.remove

    async def blocking_replace(_source, _destination):
        replace_started.set()
        await asyncio.Event().wait()

    async def blocking_remove(path):
        if Path(path).name.startswith(".bws_cache_enc_"):
            remove_started.set()
            await allow_remove.wait()
        await original_remove(path)

    monkeypatch.setattr(bitwarden.aiofiles.os, "replace", blocking_replace)
    monkeypatch.setattr(bitwarden.aiofiles.os, "remove", blocking_remove)
    task = asyncio.create_task(
        bitwarden._write_encrypted_disk_cache(
            cache_key=("fingerprint", "project", "", str(tmp_path)),
            access_token="bootstrap-token",
            entry=CachedFetch(
                secrets={"API_KEY": "secret"},
                fetched_at=10**10,
            ),
            home_path=tmp_path,
        )
    )
    await replace_started.wait()
    task.cancel()
    await remove_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    allow_remove.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert not list(tmp_path.joinpath("cache").glob(".bws_cache_enc_*.tmp"))


def test_cache_operations_do_not_retain_closed_event_loops(tmp_path):
    cache = DiskCache("loop.json", key_serializer=str)
    loop_refs = []

    async def cycle(index):
        loop_refs.append(weakref.ref(asyncio.get_running_loop()))
        await cache.write(
            str(index),
            CachedFetch(secrets={"VALUE": str(index)}, fetched_at=10**10),
            300,
            tmp_path / str(index),
        )

    asyncio.run(cycle(1))
    asyncio.run(cycle(2))
    gc.collect()

    assert [reference() for reference in loop_refs] == [None, None]
