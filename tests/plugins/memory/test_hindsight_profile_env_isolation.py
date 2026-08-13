"""Profile-scoped environment fallbacks for Hindsight memory."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import aiofiles
import aiofiles.os
import pytest

from agent.secret_scope import (
    UnscopedSecretError,
    is_multiplex_active,
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.memory.hindsight import (
    HindsightMemoryProvider,
    _build_embedded_profile_env,
    _check_local_runtime,
    _embedded_profile_env_path,
    _export_port_health_grace_timeout,
    _load_config,
    _load_simple_env,
    _materialize_embedded_profile_env,
    _scope_implicit_embedded_profile,
)


pytestmark = pytest.mark.asyncio


async def test_local_runtime_probe_uses_credential_scrubbed_child_env(
    monkeypatch,
):
    captured = {}

    async def locate(_name):
        return object()

    class Process:
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def spawn(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return Process()

    monkeypatch.setattr(
        "plugins.memory.hindsight._locate_source_module",
        locate,
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setenv("HINDSIGHT_API_KEY", "foreign-hindsight")
    monkeypatch.setenv("OPENAI_API_KEY", "foreign-openai")
    monkeypatch.setenv("HINDSIGHT_PROBE_MARKER", "foreign-hindsight-marker")
    monkeypatch.setenv("PROBE_RUNTIME_MARKER", "preserved")

    assert await _check_local_runtime() == (True, None)
    assert captured["args"][0:2] == (os.sys.executable, "-c")
    child_env = captured["kwargs"]["env"]
    assert child_env["PROBE_RUNTIME_MARKER"] == "preserved"
    assert "HINDSIGHT_API_KEY" not in child_env
    assert "HINDSIGHT_PROBE_MARKER" not in child_env
    assert "OPENAI_API_KEY" not in child_env


async def test_multiplex_profiles_never_borrow_legacy_shared_config(
    tmp_path,
    monkeypatch,
):
    legacy_home = tmp_path / "legacy-home"
    legacy_path = legacy_home / ".hindsight" / "config.json"
    await aiofiles.os.makedirs(legacy_path.parent)
    async with aiofiles.open(legacy_path, "w") as config_file:
        await config_file.write(
            json.dumps({"mode": "cloud", "apiKey": "foreign-legacy-key"})
        )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: legacy_home))
    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)

    async def load(label):
        home_token = set_hermes_home_override(tmp_path / f"profile-{label}")
        scope_token = set_secret_scope(
            {
                "HINDSIGHT_MODE": "cloud",
                "HINDSIGHT_API_KEY": f"profile-{label}-key",
            }
        )
        try:
            await asyncio.sleep(0)
            return await _load_config()
        finally:
            reset_secret_scope(scope_token)
            reset_hermes_home_override(home_token)

    try:
        alpha, beta = await asyncio.gather(load("alpha"), load("beta"))
    finally:
        set_multiplex_active(previous_multiplex)

    assert alpha["apiKey"] == "profile-alpha-key"
    assert beta["apiKey"] == "profile-beta-key"
    assert "foreign-legacy-key" not in json.dumps((alpha, beta))


async def test_single_profile_keeps_legacy_shared_config_fallback(
    tmp_path,
    monkeypatch,
):
    profile_home = tmp_path / "profile"
    legacy_home = tmp_path / "legacy-home"
    legacy_path = legacy_home / ".hindsight" / "config.json"
    await aiofiles.os.makedirs(legacy_path.parent)
    legacy_config = {"mode": "cloud", "apiKey": "legacy-single-key"}
    async with aiofiles.open(legacy_path, "w") as config_file:
        await config_file.write(json.dumps(legacy_config))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: legacy_home))
    previous_multiplex = is_multiplex_active()
    set_multiplex_active(False)
    home_token = set_hermes_home_override(profile_home)
    try:
        assert await _load_config() == legacy_config
    finally:
        reset_hermes_home_override(home_token)
        set_multiplex_active(previous_multiplex)


async def test_concurrent_profiles_isolate_hindsight_environment_settings(
    tmp_path, monkeypatch
):
    legacy_home = tmp_path / "legacy-home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: legacy_home))
    for key, value in {
        "HINDSIGHT_API_KEY": "process-key",
        "HINDSIGHT_API_URL": "https://process.invalid",
        "HINDSIGHT_MODE": "local_external",
        "HINDSIGHT_TIMEOUT": "999",
        "HINDSIGHT_IDLE_TIMEOUT": "998",
        "HINDSIGHT_BANK_ID": "process-bank",
        "HINDSIGHT_BUDGET": "high",
        "HINDSIGHT_RETAIN_TAGS": "process-tag",
        "HINDSIGHT_RETAIN_OBSERVATION_SCOPES": "combined",
        "HINDSIGHT_RETAIN_SOURCE": "process-source",
        "HINDSIGHT_RETAIN_USER_PREFIX": "ProcessUser",
        "HINDSIGHT_RETAIN_ASSISTANT_PREFIX": "ProcessAssistant",
    }.items():
        monkeypatch.setenv(key, value)

    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)

    async def initialize_profile(name: str, timeout: int):
        home = tmp_path / name
        home.mkdir()
        home_token = set_hermes_home_override(home)
        scope_token = set_secret_scope(
            {
                "HINDSIGHT_API_KEY": f"key-{name}",
                "HINDSIGHT_API_URL": f"https://{name}.example",
                "HINDSIGHT_MODE": "cloud",
                "HINDSIGHT_TIMEOUT": str(timeout),
                "HINDSIGHT_IDLE_TIMEOUT": str(timeout + 1),
                "HINDSIGHT_BANK_ID": f"bank-{name}",
                "HINDSIGHT_BUDGET": "low" if name == "alpha" else "mid",
                "HINDSIGHT_RETAIN_TAGS": f"tag-{name}",
                "HINDSIGHT_RETAIN_OBSERVATION_SCOPES": "per_tag",
                "HINDSIGHT_RETAIN_SOURCE": f"source-{name}",
                "HINDSIGHT_RETAIN_USER_PREFIX": f"User-{name}",
                "HINDSIGHT_RETAIN_ASSISTANT_PREFIX": f"Assistant-{name}",
            }
        )
        provider = HindsightMemoryProvider()
        try:
            await provider.initialize(f"session-{name}", hermes_home=str(home))
            return provider
        finally:
            reset_secret_scope(scope_token)
            reset_hermes_home_override(home_token)

    try:
        profile_a, profile_b = await asyncio.gather(
            initialize_profile("alpha", 111),
            initialize_profile("beta", 222),
        )
    finally:
        set_multiplex_active(previous_multiplex)

    assert (
        profile_a._api_key,
        profile_a._api_url,
        profile_a._timeout,
        profile_a._idle_timeout,
        profile_a._bank_id,
        profile_a._budget,
        profile_a._retain_tags,
        profile_a._observation_scopes,
        profile_a._retain_source,
        profile_a._retain_user_prefix,
        profile_a._retain_assistant_prefix,
    ) == (
        "key-alpha",
        "https://alpha.example",
        111,
        112,
        "bank-alpha",
        "low",
        ["tag-alpha"],
        "per_tag",
        "source-alpha",
        "User-alpha",
        "Assistant-alpha",
    )
    assert (
        profile_b._api_key,
        profile_b._api_url,
        profile_b._timeout,
        profile_b._idle_timeout,
        profile_b._bank_id,
        profile_b._budget,
        profile_b._retain_tags,
        profile_b._observation_scopes,
        profile_b._retain_source,
        profile_b._retain_user_prefix,
        profile_b._retain_assistant_prefix,
    ) == (
        "key-beta",
        "https://beta.example",
        222,
        223,
        "bank-beta",
        "mid",
        ["tag-beta"],
        "per_tag",
        "source-beta",
        "User-beta",
        "Assistant-beta",
    )

    await asyncio.gather(profile_a.shutdown(), profile_b.shutdown())


async def test_embedded_profile_env_uses_task_local_url_and_timeout(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_LLM_BASE_URL", "https://process.invalid")
    monkeypatch.setenv("HINDSIGHT_IDLE_TIMEOUT", "999")
    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)

    async def build(name: str, timeout: int):
        token = set_secret_scope(
            {
                "HINDSIGHT_LLM_API_KEY": f"key-{name}",
                "HINDSIGHT_API_LLM_BASE_URL": f"https://{name}.example/v1",
                "HINDSIGHT_IDLE_TIMEOUT": str(timeout),
            }
        )
        try:
            return _build_embedded_profile_env(
                {"llm_provider": "openai", "llm_model": f"model-{name}"}
            )
        finally:
            reset_secret_scope(token)

    try:
        alpha, beta = await asyncio.gather(build("alpha", 111), build("beta", 222))
    finally:
        set_multiplex_active(previous_multiplex)

    assert alpha["HINDSIGHT_API_LLM_API_KEY"] == "key-alpha"
    assert alpha["HINDSIGHT_API_LLM_BASE_URL"] == "https://alpha.example/v1"
    assert alpha["HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT"] == "111"
    assert beta["HINDSIGHT_API_LLM_API_KEY"] == "key-beta"
    assert beta["HINDSIGHT_API_LLM_BASE_URL"] == "https://beta.example/v1"
    assert beta["HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT"] == "222"


async def test_profile_config_keeps_precedence_over_scoped_environment(
    tmp_path,
) -> None:
    home = tmp_path / "profile"
    config_path = home / "hindsight" / "config.json"
    await aiofiles.os.makedirs(config_path.parent, exist_ok=True)
    config = {
        "mode": "cloud",
        "apiKey": "file-key",
        "api_url": "https://file.example",
        "timeout": 31,
        "idle_timeout": 32,
        "bank_id": "file-bank",
        "recall_budget": "high",
        "retain_tags": "file-tag",
        "observation_scopes": "combined",
        "retain_source": "file-source",
        "retain_user_prefix": "FileUser",
        "retain_assistant_prefix": "FileAssistant",
        "llm_base_url": "https://file-llm.example/v1",
    }
    async with aiofiles.open(config_path, "w", encoding="utf-8") as handle:
        await handle.write(json.dumps(config))

    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    home_token = set_hermes_home_override(home)
    scope_token = set_secret_scope(
        {
            "HINDSIGHT_API_KEY": "env-key",
            "HINDSIGHT_API_URL": "https://env.example",
            "HINDSIGHT_MODE": "local_external",
            "HINDSIGHT_TIMEOUT": "91",
            "HINDSIGHT_IDLE_TIMEOUT": "92",
            "HINDSIGHT_BANK_ID": "env-bank",
            "HINDSIGHT_BUDGET": "low",
            "HINDSIGHT_RETAIN_TAGS": "env-tag",
            "HINDSIGHT_RETAIN_OBSERVATION_SCOPES": "per_tag",
            "HINDSIGHT_RETAIN_SOURCE": "env-source",
            "HINDSIGHT_RETAIN_USER_PREFIX": "EnvUser",
            "HINDSIGHT_RETAIN_ASSISTANT_PREFIX": "EnvAssistant",
            "HINDSIGHT_API_LLM_BASE_URL": "https://env-llm.example/v1",
        }
    )
    provider = HindsightMemoryProvider()
    try:
        await provider.initialize("file-session", hermes_home=str(home))
        assert (
            provider._mode,
            provider._api_key,
            provider._api_url,
            provider._timeout,
            provider._idle_timeout,
            provider._bank_id,
            provider._budget,
            provider._retain_tags,
            provider._observation_scopes,
            provider._retain_source,
            provider._retain_user_prefix,
            provider._retain_assistant_prefix,
            provider._llm_base_url,
        ) == (
            "cloud",
            "file-key",
            "https://file.example",
            31,
            32,
            "file-bank",
            "high",
            ["file-tag"],
            "combined",
            "file-source",
            "FileUser",
            "FileAssistant",
            "https://file-llm.example/v1",
        )
    finally:
        await provider.shutdown()
        reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)
        set_multiplex_active(previous_multiplex)


async def test_single_profile_environment_fallback_remains_unchanged(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "profile"
    legacy_home = tmp_path / "legacy-home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: legacy_home))
    for key, value in {
        "HINDSIGHT_API_KEY": "single-key",
        "HINDSIGHT_API_URL": "https://single.example",
        "HINDSIGHT_MODE": "cloud",
        "HINDSIGHT_TIMEOUT": "41",
        "HINDSIGHT_IDLE_TIMEOUT": "42",
        "HINDSIGHT_BANK_ID": "single-bank",
        "HINDSIGHT_BUDGET": "low",
        "HINDSIGHT_RETAIN_TAGS": "single-tag",
        "HINDSIGHT_RETAIN_OBSERVATION_SCOPES": "per_tag",
        "HINDSIGHT_RETAIN_SOURCE": "single-source",
        "HINDSIGHT_RETAIN_USER_PREFIX": "SingleUser",
        "HINDSIGHT_RETAIN_ASSISTANT_PREFIX": "SingleAssistant",
    }.items():
        monkeypatch.setenv(key, value)

    previous_multiplex = is_multiplex_active()
    set_multiplex_active(False)
    home_token = set_hermes_home_override(home)
    provider = HindsightMemoryProvider()
    try:
        await provider.initialize("single-session", hermes_home=str(home))
        assert (
            provider._api_key,
            provider._api_url,
            provider._timeout,
            provider._idle_timeout,
            provider._bank_id,
            provider._budget,
            provider._retain_tags,
            provider._observation_scopes,
            provider._retain_source,
            provider._retain_user_prefix,
            provider._retain_assistant_prefix,
        ) == (
            "single-key",
            "https://single.example",
            41,
            42,
            "single-bank",
            "low",
            ["single-tag"],
            "per_tag",
            "single-source",
            "SingleUser",
            "SingleAssistant",
        )
    finally:
        await provider.shutdown()
        reset_hermes_home_override(home_token)
        set_multiplex_active(previous_multiplex)


async def test_unscoped_multiplex_configuration_fails_closed(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Path,
        "home",
        classmethod(lambda cls: tmp_path / "legacy-home"),
    )
    monkeypatch.setenv("HINDSIGHT_MODE", "cloud")
    monkeypatch.setenv("HINDSIGHT_API_KEY", "other-profile-key")
    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    scope_token = set_secret_scope(None)
    home_token = set_hermes_home_override(tmp_path / "profile")
    provider = HindsightMemoryProvider()
    try:
        assert await provider.is_available() is False
        with pytest.raises(UnscopedSecretError):
            await provider.initialize("unscoped", hermes_home=str(tmp_path))
        assert provider._daemon_start_task is None
    finally:
        await provider.shutdown()
        reset_hermes_home_override(home_token)
        reset_secret_scope(scope_token)
        set_multiplex_active(previous_multiplex)


async def test_multiplexed_implicit_embedded_profiles_use_canonical_home_names(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_home = tmp_path / "user-home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: shared_home))
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    alias_a = tmp_path / "profile-a-alias"
    await aiofiles.os.makedirs(home_a)
    await aiofiles.os.makedirs(home_b)
    await aiofiles.os.symlink(home_a, alias_a)
    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)

    async def materialize(home: Path, label: str):
        home_token = set_hermes_home_override(home)
        scope_token = set_secret_scope(
            {
                "HINDSIGHT_LLM_API_KEY": f"key-{label}",
                "HINDSIGHT_API_LLM_BASE_URL": f"https://{label}.example/v1",
                "HINDSIGHT_IDLE_TIMEOUT": "71",
            }
        )
        config = {
            "mode": "local_embedded",
            "llm_provider": "openai",
            "llm_model": f"model-{label}",
        }
        try:
            await _scope_implicit_embedded_profile(config)
            path = await _materialize_embedded_profile_env(config)
            return config, path, await _load_simple_env(path)
        finally:
            reset_secret_scope(scope_token)
            reset_hermes_home_override(home_token)

    try:
        alpha, alpha_alias, beta = await asyncio.gather(
            materialize(home_a, "alpha"),
            materialize(alias_a, "alpha"),
            materialize(home_b, "beta"),
        )
    finally:
        set_multiplex_active(previous_multiplex)

    assert alpha[0]["profile"] == alpha_alias[0]["profile"]
    assert alpha[0]["profile"].startswith("hermes-")
    assert beta[0]["profile"].startswith("hermes-")
    assert alpha[0]["profile"] != beta[0]["profile"]
    assert alpha[1] == alpha_alias[1]
    assert alpha[1] != beta[1]
    assert alpha[2]["HINDSIGHT_API_LLM_API_KEY"] == "key-alpha"
    assert beta[2]["HINDSIGHT_API_LLM_API_KEY"] == "key-beta"


async def test_embedded_profile_name_and_child_env_preserve_explicit_parity(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_multiplex = is_multiplex_active()
    explicit = {"profile": "Exact_Profile", "port_health_grace_timeout": 17}
    implicit_single: dict = {}

    set_multiplex_active(False)
    monkeypatch.delenv(
        "HINDSIGHT_EMBED_PORT_HEALTH_GRACE_TIMEOUT",
        raising=False,
    )
    _export_port_health_grace_timeout(explicit)
    assert os.environ["HINDSIGHT_EMBED_PORT_HEALTH_GRACE_TIMEOUT"] == "17.0"
    await _scope_implicit_embedded_profile(implicit_single)
    assert implicit_single == {}
    assert _embedded_profile_env_path(implicit_single).name == "hermes.env"

    set_multiplex_active(True)
    home_token = set_hermes_home_override(tmp_path / "profile")
    scope_token = set_secret_scope(
        {"HINDSIGHT_EMBED_PORT_HEALTH_GRACE_TIMEOUT": "19"}
    )
    monkeypatch.setenv("HINDSIGHT_API_KEY", "other-profile-key")
    captured: dict = {}

    class Process:
        returncode = 0

        async def wait(self):
            return 0

    async def create_process(*args, **kwargs):
        captured["args"] = args
        captured["env"] = kwargs["env"]
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    provider = HindsightMemoryProvider()
    provider._config = explicit
    try:
        await _scope_implicit_embedded_profile(explicit)
        assert explicit["profile"] == "Exact_Profile"
        assert await provider._run_embedded_cli("daemon", "start") == 0
    finally:
        reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)
        set_multiplex_active(previous_multiplex)

    assert captured["args"][3:5] == ("--profile", "Exact_Profile")
    assert captured["env"]["HINDSIGHT_EMBED_PORT_HEALTH_GRACE_TIMEOUT"] == "19"
    assert captured["env"]["HERMES_HOME"] == str(tmp_path / "profile")
    assert "HINDSIGHT_API_KEY" not in captured["env"]


async def test_cancelled_embedded_cli_reaps_only_its_profile_process(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    processes: dict[str, Process] = {}
    profile_names: dict[str, str] = {}

    class Process:
        def __init__(self, profile: str):
            self.profile = profile
            self.returncode = None
            self.started = asyncio.Event()
            self.terminate_started = asyncio.Event()
            self.release = asyncio.Event()
            self.killed = False

        async def wait(self):
            self.started.set()
            await self.release.wait()
            return self.returncode

        def terminate(self):
            self.terminate_started.set()

        def kill(self):
            self.killed = True
            self.returncode = -9
            self.release.set()

    async def create_process(*args, **_kwargs):
        process = Process(args[4])
        processes[process.profile] = process
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

    async def run(home: Path):
        home_token = set_hermes_home_override(home)
        scope_token = set_secret_scope({})
        config = {"mode": "local_embedded"}
        provider = HindsightMemoryProvider()
        try:
            await _scope_implicit_embedded_profile(config)
            profile_names[home.name] = config["profile"]
            provider._config = config
            return await provider._run_embedded_cli("daemon", "start")
        finally:
            reset_secret_scope(scope_token)
            reset_hermes_home_override(home_token)

    task_a = asyncio.create_task(run(tmp_path / "profile-a"))
    task_b = asyncio.create_task(run(tmp_path / "profile-b"))
    try:
        while len(processes) < 2 or len(profile_names) < 2:
            await asyncio.sleep(0)
        process_a = processes[profile_names["profile-a"]]
        process_b = processes[profile_names["profile-b"]]
        await asyncio.gather(process_a.started.wait(), process_b.started.wait())

        task_a.cancel()
        await process_a.terminate_started.wait()
        task_a.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task_a

        assert process_a.killed is True
        assert process_b.terminate_started.is_set() is False
        assert task_b.done() is False
        process_b.returncode = 0
        process_b.release.set()
        assert await task_b == 0
    finally:
        for task in (task_a, task_b):
            if not task.done():
                task.cancel()
        await asyncio.gather(task_a, task_b, return_exceptions=True)
        set_multiplex_active(previous_multiplex)
