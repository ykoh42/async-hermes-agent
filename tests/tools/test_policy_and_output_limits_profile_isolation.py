"""Profile isolation for website policy and tool output limit caches."""

from __future__ import annotations

import asyncio
import gc
import threading
import time
import weakref
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import tool_output_limits as limits
from tools import website_policy as policy


def _write_config(
    home: Path,
    *,
    blocked_domain: str | None = None,
    max_bytes: int = 50_000,
) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "security": {
                    "website_blocklist": {
                        "enabled": blocked_domain is not None,
                        "domains": [blocked_domain] if blocked_domain else [],
                    }
                },
                "tool_output": {
                    "max_bytes": max_bytes,
                    "max_lines": max_bytes // 10,
                    "max_line_length": max_bytes // 100,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


async def _in_profile(home: Path, callback):
    token = set_hermes_home_override(home)
    try:
        return await callback()
    finally:
        reset_hermes_home_override(token)


@pytest.fixture(autouse=True)
def reset_profile_caches():
    policy.invalidate_cache()
    limits._reset_tool_output_limits_cache()
    yield
    policy.invalidate_cache()
    limits._reset_tool_output_limits_cache()


@pytest.mark.asyncio
async def test_website_policy_is_sequentially_and_concurrently_profile_local(tmp_path):
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    _write_config(profile_a, blocked_domain="blocked-a.test")
    _write_config(profile_b, blocked_domain="blocked-b.test")

    async def check(url: str):
        return await policy.check_website_access(url)

    assert await _in_profile(
        profile_a,
        lambda: check("https://blocked-a.test"),
    ) is not None
    assert await _in_profile(
        profile_b,
        lambda: check("https://blocked-a.test"),
    ) is None

    policy.invalidate_cache()
    a_result, b_result = await asyncio.gather(
        _in_profile(profile_a, lambda: check("https://blocked-a.test")),
        _in_profile(profile_b, lambda: check("https://blocked-b.test")),
    )
    assert a_result is not None and a_result["rule"] == "blocked-a.test"
    assert b_result is not None and b_result["rule"] == "blocked-b.test"


@pytest.mark.asyncio
async def test_policy_symlink_aliases_share_one_cache_entry(tmp_path):
    profile = tmp_path / "profile"
    alias = tmp_path / "profile-alias"
    _write_config(profile, blocked_domain="blocked.test")
    alias.symlink_to(profile, target_is_directory=True)

    await _in_profile(
        profile,
        lambda: policy.check_website_access("https://blocked.test"),
    )
    await _in_profile(
        alias,
        lambda: policy.check_website_access("https://blocked.test"),
    )

    assert len(policy._policy_cache_by_path) == 1


@pytest.mark.asyncio
async def test_cancelled_policy_read_does_not_publish_partial_cache(
    monkeypatch,
    tmp_path,
):
    profile = tmp_path / "profile"
    _write_config(profile, blocked_domain="blocked.test")
    original_open = policy.aiofiles.open
    entered = asyncio.Event()

    class _CancelledRead:
        async def __aenter__(self):
            entered.set()
            await asyncio.Future()

        async def __aexit__(self, *_args):
            return False

    monkeypatch.setattr(policy.aiofiles, "open", lambda *_args, **_kwargs: _CancelledRead())
    task = asyncio.create_task(
        _in_profile(
            profile,
            lambda: policy.check_website_access("https://blocked.test"),
        )
    )
    await entered.wait()
    task.cancel()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert policy._policy_cache_by_path == {}

    monkeypatch.setattr(policy.aiofiles, "open", original_open)
    blocked = await _in_profile(
        profile,
        lambda: policy.check_website_access("https://blocked.test"),
    )
    assert blocked is not None


@pytest.mark.asyncio
async def test_output_limits_remain_profile_local_after_concurrent_refresh(tmp_path):
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    _write_config(profile_a, max_bytes=111_000)
    _write_config(profile_b, max_bytes=222_000)
    ready = 0
    ready_lock = asyncio.Lock()
    gate = asyncio.Event()

    async def refresh_and_read():
        nonlocal ready
        refreshed = await limits._refresh_tool_output_limits()
        async with ready_lock:
            ready += 1
            if ready == 2:
                gate.set()
        await gate.wait()
        await asyncio.sleep(0)
        return refreshed, limits.get_tool_output_limits()

    result_a, result_b = await asyncio.gather(
        _in_profile(profile_a, refresh_and_read),
        _in_profile(profile_b, refresh_and_read),
    )
    assert result_a[0] == result_a[1]
    assert result_b[0] == result_b[1]
    assert result_a[0]["max_bytes"] == 111_000
    assert result_b[0]["max_bytes"] == 222_000


@pytest.mark.asyncio
async def test_output_limit_symlink_aliases_share_one_profile_entry(tmp_path):
    profile = tmp_path / "profile"
    alias = tmp_path / "profile-alias"
    _write_config(profile, max_bytes=123_000)
    alias.symlink_to(profile, target_is_directory=True)

    direct = await _in_profile(profile, limits._refresh_tool_output_limits)
    alias_value = await _in_profile(alias, limits._refresh_tool_output_limits)

    assert direct == alias_value
    assert len(limits._limits_by_profile) == 1


@pytest.mark.asyncio
async def test_cancelled_output_limit_load_keeps_previous_profile_value(tmp_path):
    profile = tmp_path / "profile"
    _write_config(profile, max_bytes=321_000)
    previous = await _in_profile(profile, limits._refresh_tool_output_limits)
    entered = asyncio.Event()

    async def cancelled_load():
        entered.set()
        await asyncio.Future()

    with patch(
        "hermes_cli.config.load_config_readonly",
        new=AsyncMock(side_effect=cancelled_load),
    ):
        task = asyncio.create_task(
            _in_profile(profile, limits._refresh_tool_output_limits)
        )
        await entered.wait()
        task.cancel()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def read_current():
        return limits.get_tool_output_limits()

    assert await _in_profile(profile, read_current) == previous


def test_profile_cache_state_survives_cross_loop_without_loop_ownership(tmp_path):
    profile = tmp_path / "profile"
    _write_config(profile, blocked_domain="blocked.test", max_bytes=444_000)
    loop_refs = []

    async def run_once():
        loop_refs.append(weakref.ref(asyncio.get_running_loop()))

        async def use_both():
            blocked = await policy.check_website_access("https://blocked.test")
            configured = await limits._refresh_tool_output_limits()
            return blocked, configured

        return await _in_profile(profile, use_both)

    first = asyncio.run(run_once())
    second = asyncio.run(run_once())
    gc.collect()

    assert first == second
    assert first[0] is not None
    assert first[1]["max_bytes"] == 444_000
    assert all(loop_ref() is None for loop_ref in loop_refs)


def test_concurrent_distinct_event_loops_keep_profile_values_separate(tmp_path):
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    _write_config(profile_a, blocked_domain="blocked-a.test", max_bytes=111_000)
    _write_config(profile_b, blocked_domain="blocked-b.test", max_bytes=222_000)
    results = []
    errors = []

    async def use(home: Path, domain: str):
        async def operation():
            return (
                await policy.check_website_access(f"https://{domain}"),
                await limits._refresh_tool_output_limits(),
            )

        return await _in_profile(home, operation)

    def runner(home: Path, domain: str):
        try:
            results.append(asyncio.run(use(home, domain)))
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=runner, args=(profile_a, "blocked-a.test"))
    second = threading.Thread(target=runner, args=(profile_b, "blocked-b.test"))
    first.start()
    second.start()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert {item[0]["rule"] for item in results} == {
        "blocked-a.test",
        "blocked-b.test",
    }
    assert {item[1]["max_bytes"] for item in results} == {111_000, 222_000}
