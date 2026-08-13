"""Profile and event-loop isolation for native-async Azure identity."""

from __future__ import annotations

import asyncio
import gc
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
from blockbuster import BlockBuster, BlockingError

from agent import secret_scope
from hermes_constants import (
    reset_hermes_home_override,
    set_hermes_home_override,
)


@pytest.fixture
def azure_adapter(monkeypatch):
    from agent import azure_identity_adapter as module

    previous_multiplex = secret_scope.is_multiplex_active()
    secret_token = secret_scope.set_secret_scope(None)
    with module._credential_cache_guard:
        module._credential_caches.clear()
        module._credential_leases.clear()
        module._issued_providers.clear()

    class Credential:
        instances: list["Credential"] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = 0
            self.close_started: asyncio.Event | None = None
            self.allow_close: asyncio.Event | None = None
            self.instances.append(self)

        async def get_token(self, _scope):
            return SimpleNamespace(token="entra-token", expires_on=1234)

        async def close(self):
            if self.close_started is not None:
                self.close_started.set()
            if self.allow_close is not None:
                await self.allow_close.wait()
            self.closed += 1

    def get_bearer_token_provider(credential, scope):
        async def provide():
            return (await credential.get_token(scope)).token

        return provide

    identity = SimpleNamespace(
        DefaultAzureCredential=Credential,
        get_bearer_token_provider=get_bearer_token_provider,
    )
    monkeypatch.setattr(module, "_require_azure_identity", lambda: identity)
    monkeypatch.setattr(module, "has_azure_identity_installed", lambda: True)
    yield module, Credential

    with module._credential_cache_guard:
        module._credential_caches.clear()
        module._credential_leases.clear()
        module._issued_providers.clear()
    secret_scope.reset_secret_scope(secret_token)
    secret_scope.set_multiplex_active(previous_multiplex)


async def _build_for(module, home: Path, secrets: dict[str, str]):
    home_token = set_hermes_home_override(home)
    secret_token = secret_scope.set_secret_scope(secrets)
    try:
        return await module.build_credential(module.EntraIdentityConfig())
    finally:
        secret_scope.reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


async def _build_provider_for(module, home: Path, secrets: dict[str, str]):
    home_token = set_hermes_home_override(home)
    secret_token = secret_scope.set_secret_scope(secrets)
    try:
        return await module.build_token_provider()
    finally:
        secret_scope.reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


async def _reset_for(module, home: Path) -> None:
    home_token = set_hermes_home_override(home)
    try:
        await module.reset_credential_cache()
    finally:
        reset_hermes_home_override(home_token)


@pytest.mark.asyncio
async def test_concurrent_profiles_pin_supported_selectors_and_never_borrow_env(
    azure_adapter,
    monkeypatch,
    tmp_path,
):
    module, credential_type = azure_adapter
    secret_scope.set_multiplex_active(True)
    monkeypatch.setenv("AZURE_CLIENT_ID", "foreign-client")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "foreign-secret")
    monkeypatch.setenv("AZURE_TENANT_ID", "foreign-tenant")
    monkeypatch.setenv("AZURE_FEDERATED_TOKEN_FILE", "/foreign/token")
    monkeypatch.setenv("AZURE_AUTHORITY_HOST", "foreign.authority.invalid")
    monkeypatch.setenv("AZURE_USERNAME", "foreign@example.invalid")

    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    settings_a = {
        "AZURE_CLIENT_ID": "client-a",
        "AZURE_TENANT_ID": "tenant-a",
        "AZURE_AUTHORITY_HOST": "authority-a.invalid",
    }
    settings_b = {
        "AZURE_CLIENT_ID": "client-b",
        "AZURE_TENANT_ID": "tenant-b",
        "AZURE_AUTHORITY_HOST": "authority-b.invalid",
    }

    credential_a, credential_b = await asyncio.gather(
        _build_for(module, profile_a, settings_a),
        _build_for(module, profile_b, settings_b),
    )

    assert credential_a is not credential_b
    assert len(credential_type.instances) == 2
    assert credential_a.kwargs["managed_identity_client_id"] == "client-a"
    assert credential_a.kwargs["authority"] == "authority-a.invalid"
    assert credential_b.kwargs["managed_identity_client_id"] == "client-b"
    assert credential_b.kwargs["authority"] == "authority-b.invalid"
    for credential in (credential_a, credential_b):
        assert credential.kwargs["exclude_environment_credential"] is True
        assert credential.kwargs["exclude_workload_identity_credential"] is True
        assert credential.kwargs["exclude_managed_identity_credential"] is False
        assert credential.kwargs["exclude_shared_token_cache_credential"] is True
        assert credential.kwargs["exclude_visual_studio_code_credential"] is True
        assert credential.kwargs["exclude_cli_credential"] is True
        assert credential.kwargs["exclude_powershell_credential"] is True
        assert credential.kwargs["exclude_developer_cli_credential"] is True

    await _reset_for(module, profile_a)
    await _reset_for(module, profile_b)


@pytest.mark.asyncio
async def test_symlinked_home_and_same_settings_share_one_loop_credential(
    azure_adapter,
    tmp_path,
):
    module, credential_type = azure_adapter
    secret_scope.set_multiplex_active(True)
    real_home = tmp_path / "real-profile"
    real_home.mkdir()
    alias_home = tmp_path / "profile-alias"
    alias_home.symlink_to(real_home, target_is_directory=True)
    settings = {"AZURE_CLIENT_ID": "managed-client"}

    first = await _build_for(module, real_home, settings)
    second = await _build_for(module, alias_home, settings)

    assert first is second
    assert len(credential_type.instances) == 1
    await _reset_for(module, alias_home)
    assert first.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name",
    [
        "AZURE_CLIENT_SECRET",
        "AZURE_CLIENT_CERTIFICATE_PATH",
        "AZURE_FEDERATED_TOKEN_FILE",
        "AZURE_TOKEN_CREDENTIALS",
        "AZURE_USERNAME",
    ],
)
async def test_process_only_scoped_credentials_fail_closed(
    azure_adapter,
    tmp_path,
    name,
):
    module, credential_type = azure_adapter
    secret_scope.set_multiplex_active(True)

    with pytest.raises(RuntimeError, match=name):
        await _build_for(module, tmp_path / "profile", {name: "profile-value"})

    assert credential_type.instances == []


@pytest.mark.asyncio
async def test_unscoped_multiplex_credential_construction_fails_closed(
    azure_adapter,
):
    module, credential_type = azure_adapter
    secret_scope.set_multiplex_active(True)

    with pytest.raises(secret_scope.UnscopedSecretError):
        await module.build_credential(module.EntraIdentityConfig())

    assert credential_type.instances == []


@pytest.mark.asyncio
async def test_interactive_browser_opt_in_fails_instead_of_silent_noop(
    azure_adapter,
):
    module, credential_type = azure_adapter

    with pytest.raises(
        RuntimeError,
        match="InteractiveBrowserCredential is unavailable",
    ):
        await module.build_credential(
            module.EntraIdentityConfig(exclude_interactive_browser=False)
        )

    assert credential_type.instances == []


@pytest.mark.asyncio
async def test_empty_foreign_env_values_cannot_configure_sdk_chain(
    azure_adapter,
    monkeypatch,
    tmp_path,
):
    module, _credential_type = azure_adapter
    secret_scope.set_multiplex_active(True)
    monkeypatch.setenv("AZURE_CLIENT_ID", "foreign-client")
    monkeypatch.setenv("AZURE_TENANT_ID", "foreign-tenant")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "")
    monkeypatch.setenv("AZURE_FEDERATED_TOKEN_FILE", "")

    credential = await _build_for(module, tmp_path / "profile", {})

    assert credential.kwargs["exclude_environment_credential"] is True
    assert credential.kwargs["exclude_workload_identity_credential"] is True
    assert credential.kwargs["managed_identity_client_id"] is None
    await _reset_for(module, tmp_path / "profile")


@pytest.mark.asyncio
async def test_process_global_token_chain_selector_fails_closed_in_multiplex(
    azure_adapter,
    monkeypatch,
    tmp_path,
):
    module, credential_type = azure_adapter
    secret_scope.set_multiplex_active(True)
    monkeypatch.setenv("AZURE_TOKEN_CREDENTIALS", "environmentcredential")

    with pytest.raises(RuntimeError, match="AZURE_TOKEN_CREDENTIALS"):
        await _build_for(module, tmp_path / "profile", {})

    assert credential_type.instances == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (
            {
                "AZURE_CLIENT_ID": "client",
                "AZURE_TENANT_ID": "tenant",
                "AZURE_FEDERATED_TOKEN_FILE": "/token",
            },
            "WorkloadIdentityCredential",
        ),
        (
            {
                "AZURE_CLIENT_ID": "client",
                "AZURE_TENANT_ID": "tenant",
                "AZURE_CLIENT_CERTIFICATE_PATH": "/certificate",
            },
            "CertificateCredential",
        ),
        (
            {"AZURE_USERNAME": "cached@example.invalid"},
            "SharedTokenCacheCredential",
        ),
        (
            {"AZURE_TOKEN_CREDENTIALS": "azureclicredential"},
            "not native-async and profile-safe",
        ),
        (
            {"AZURE_TOKEN_CREDENTIALS": "visualstudiocodecredential"},
            "not native-async and profile-safe",
        ),
    ],
)
async def test_single_profile_non_native_sdk_paths_fail_clearly(
    azure_adapter,
    monkeypatch,
    tmp_path,
    settings,
    message,
):
    module, credential_type = azure_adapter
    secret_scope.set_multiplex_active(False)
    for name, value in settings.items():
        monkeypatch.setenv(name, value)

    home_token = set_hermes_home_override(tmp_path / "profile")
    try:
        with pytest.raises(RuntimeError, match=message):
            await module.build_credential(module.EntraIdentityConfig())
    finally:
        reset_hermes_home_override(home_token)

    assert credential_type.instances == []


@pytest.mark.asyncio
async def test_explicit_managed_identity_excludes_other_configured_paths(
    azure_adapter,
    monkeypatch,
    tmp_path,
):
    module, _credential_type = azure_adapter
    secret_scope.set_multiplex_active(False)
    monkeypatch.setenv("AZURE_TOKEN_CREDENTIALS", "managedidentitycredential")
    monkeypatch.setenv("AZURE_CLIENT_ID", "managed-client")
    monkeypatch.setenv("AZURE_TENANT_ID", "foreign-tenant")
    monkeypatch.setenv("AZURE_CLIENT_CERTIFICATE_PATH", "/foreign/certificate")
    monkeypatch.setenv("AZURE_FEDERATED_TOKEN_FILE", "/foreign/token")
    monkeypatch.setenv("AZURE_USERNAME", "foreign@example.invalid")
    home_token = set_hermes_home_override(tmp_path / "profile")
    try:
        credential = await module.build_credential(module.EntraIdentityConfig())
        await module.reset_credential_cache()
    finally:
        reset_hermes_home_override(home_token)

    assert credential.kwargs["exclude_managed_identity_credential"] is False
    assert all(
        credential.kwargs[name] is True
        for name in module._DEFAULT_CREDENTIAL_EXCLUDES
        if name != "exclude_managed_identity_credential"
    )
    assert credential.closed == 1


@pytest.mark.asyncio
async def test_single_profile_selects_native_client_secret_and_fingerprints_env(
    azure_adapter,
    monkeypatch,
    tmp_path,
):
    module, credential_type = azure_adapter
    secret_scope.set_multiplex_active(False)
    home_token = set_hermes_home_override(tmp_path / "profile")
    monkeypatch.setenv("AZURE_CLIENT_ID", "client")
    monkeypatch.setenv("AZURE_TENANT_ID", "tenant")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "secret-one")
    try:
        first = await module.build_credential(module.EntraIdentityConfig())
        monkeypatch.setenv("AZURE_CLIENT_SECRET", "secret-two")
        second = await module.build_credential(module.EntraIdentityConfig())
    finally:
        await module.reset_credential_cache()
        reset_hermes_home_override(home_token)

    assert first is not second
    for credential in (first, second):
        assert credential.kwargs["exclude_environment_credential"] is False
        assert credential.kwargs["exclude_managed_identity_credential"] is True
        assert all(
            credential.kwargs[name] is True
            for name in module._DEFAULT_CREDENTIAL_EXCLUDES
            if name != "exclude_environment_credential"
        )
    assert len(credential_type.instances) == 2
    assert first.closed == second.closed == 1


@pytest.mark.asyncio
async def test_single_profile_empty_client_secret_does_not_select_environment(
    azure_adapter,
    monkeypatch,
    tmp_path,
):
    module, _credential_type = azure_adapter
    secret_scope.set_multiplex_active(False)
    monkeypatch.setenv("AZURE_CLIENT_ID", "managed-client")
    monkeypatch.setenv("AZURE_TENANT_ID", "")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "")
    home_token = set_hermes_home_override(tmp_path / "profile")
    try:
        credential = await module.build_credential(module.EntraIdentityConfig())
        await module.reset_credential_cache()
    finally:
        reset_hermes_home_override(home_token)

    assert credential.kwargs["exclude_environment_credential"] is True
    assert credential.kwargs["exclude_managed_identity_credential"] is False


@pytest.mark.asyncio
async def test_multiplex_kwargs_select_only_native_managed_identity(monkeypatch):
    identity = pytest.importorskip("azure.identity.aio")
    from agent import azure_identity_adapter as module

    for name in module._AZURE_IDENTITY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    scope_token = secret_scope.set_secret_scope(
            {
                "AZURE_CLIENT_ID": "00000000-0000-0000-0000-000000000001",
                "AZURE_TENANT_ID": "00000000-0000-0000-0000-000000000002",
            }
    )
    try:
        settings, multiplexed = module._identity_environment()
        kwargs = module._credential_kwargs(
            module.EntraIdentityConfig(),
            settings,
            multiplexed=multiplexed,
        )
        isolated = identity.DefaultAzureCredential(**kwargs)
        try:
            assert [type(item).__name__ for item in isolated.credentials] == [
                "ManagedIdentityCredential"
            ]
        finally:
            await isolated.close()
    finally:
        secret_scope.reset_secret_scope(scope_token)
        secret_scope.set_multiplex_active(previous_multiplex)


@pytest.mark.asyncio
async def test_single_profile_kwargs_select_only_native_client_secret(monkeypatch):
    identity = pytest.importorskip("azure.identity.aio")
    from agent import azure_identity_adapter as module

    for name in module._AZURE_IDENTITY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AZURE_CLIENT_ID", "00000000-0000-0000-0000-000000000001")
    monkeypatch.setenv("AZURE_TENANT_ID", "00000000-0000-0000-0000-000000000002")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "profile-secret")
    settings, multiplexed = module._identity_environment()
    kwargs = module._credential_kwargs(
        module.EntraIdentityConfig(),
        settings,
        multiplexed=multiplexed,
    )
    credential = identity.DefaultAzureCredential(**kwargs)
    try:
        assert [type(item).__name__ for item in credential.credentials] == [
            "EnvironmentCredential"
        ]
    finally:
        await credential.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        ({}, "ManagedIdentityCredential"),
        (
            {
                "AZURE_CLIENT_ID": "00000000-0000-0000-0000-000000000001",
                "AZURE_TENANT_ID": "00000000-0000-0000-0000-000000000002",
                "AZURE_CLIENT_SECRET": "profile-secret",
            },
            "EnvironmentCredential",
        ),
    ],
)
async def test_supported_real_sdk_construction_does_not_block_loop(
    monkeypatch,
    tmp_path,
    settings,
    expected,
):
    pytest.importorskip("azure.identity.aio")
    from agent import azure_identity_adapter as module

    for name in module._AZURE_IDENTITY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(False)
    home_token = set_hermes_home_override(tmp_path / expected)
    blocker = BlockBuster()
    blocker.activate()
    try:
        credential = await module.build_credential(module.EntraIdentityConfig())
        assert [type(item).__name__ for item in credential.credentials] == [expected]
    finally:
        try:
            await module.reset_credential_cache()
        finally:
            blocker.deactivate()
            reset_hermes_home_override(home_token)
            secret_scope.set_multiplex_active(previous_multiplex)


def test_pinned_aio_sdk_chain_gap_remains_explicit_release_blocker(
    monkeypatch,
):
    sync_identity = pytest.importorskip("azure.identity")
    async_identity = pytest.importorskip("azure.identity.aio")
    from agent import azure_identity_adapter as module

    for name in module._AZURE_IDENTITY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    sync_credential = sync_identity.DefaultAzureCredential(
        exclude_interactive_browser_credential=False,
    )
    async_credential = async_identity.DefaultAzureCredential(
        exclude_interactive_browser_credential=False,
    )
    try:
        sync_chain = [type(item).__name__ for item in sync_credential.credentials]
        async_chain = [type(item).__name__ for item in async_credential.credentials]
        assert "InteractiveBrowserCredential" in sync_chain
        assert "BrokerCredential" in sync_chain
        assert "InteractiveBrowserCredential" not in async_chain
        assert "BrokerCredential" not in async_chain
    finally:
        sync_credential.close()
        asyncio.run(async_credential.close())


def test_pinned_aio_workload_file_read_is_external_blocker(tmp_path):
    identity = pytest.importorskip("azure.identity.aio")
    token_file = tmp_path / "federated-token"
    token_file.write_text("jwt", encoding="utf-8")
    credential = identity.WorkloadIdentityCredential(
        tenant_id="00000000-0000-0000-0000-000000000001",
        client_id="00000000-0000-0000-0000-000000000002",
        token_file_path=str(token_file),
    )

    async def reproduce_blocking_read():
        blocker = BlockBuster()
        blocker.activate()
        try:
            with pytest.raises(BlockingError):
                credential._get_service_account_token()
        finally:
            blocker.deactivate()
            await credential.close()

    asyncio.run(reproduce_blocking_read())


def test_same_profile_uses_distinct_credentials_across_event_loops(
    azure_adapter,
    tmp_path,
):
    module, credential_type = azure_adapter
    home = tmp_path / "profile"

    async def construct_and_reset():
        home_token = set_hermes_home_override(home)
        try:
            credential = await module.build_credential(
                module.EntraIdentityConfig()
            )
            await module.reset_credential_cache()
            return credential
        finally:
            reset_hermes_home_override(home_token)

    first = asyncio.run(construct_and_reset())
    second = asyncio.run(construct_and_reset())

    assert first is not second
    assert len(credential_type.instances) == 2
    assert first.closed == second.closed == 1
    assert not module._credential_caches


def test_released_credential_cache_does_not_pin_closed_loop(
    azure_adapter,
    tmp_path,
):
    module, _credential_type = azure_adapter
    home = tmp_path / "profile"

    async def construct_and_reset():
        home_token = set_hermes_home_override(home)
        try:
            await module.build_credential(module.EntraIdentityConfig())
            await module.reset_credential_cache()
        finally:
            reset_hermes_home_override(home_token)

    loop = asyncio.new_event_loop()
    loop.run_until_complete(construct_and_reset())
    loop.close()
    loop_ref = weakref.ref(loop)
    del loop
    gc.collect()

    assert loop_ref() is None
    assert not module._credential_caches


def test_provider_release_rejects_wrong_loop_then_owner_can_close(
    azure_adapter,
    tmp_path,
):
    module, _credential_type = azure_adapter
    home = tmp_path / "profile"
    owner_loop = asyncio.new_event_loop()
    provider = owner_loop.run_until_complete(_build_provider_for(module, home, {}))
    credential = provider._hermes_credential

    async def wrong_loop_use_and_release():
        with pytest.raises(RuntimeError, match="event loop that created"):
            await provider()
        with pytest.raises(RuntimeError, match="event loop that created"):
            await module._release_token_provider(provider)

    asyncio.run(wrong_loop_use_and_release())
    assert provider._hermes_released is False
    assert credential.closed == 0

    assert owner_loop.run_until_complete(provider()) == "entra-token"
    owner_loop.run_until_complete(module._release_token_provider(provider))
    with pytest.raises(RuntimeError, match="has been released"):
        owner_loop.run_until_complete(provider())
    owner_loop.close()
    owner_loop_ref = weakref.ref(owner_loop)
    del owner_loop
    gc.collect()
    assert provider._hermes_released is True
    assert provider._hermes_credential is None
    assert credential.closed == 1
    assert owner_loop_ref() is None


@pytest.mark.asyncio
async def test_reset_closes_only_active_canonical_profile(
    azure_adapter,
    tmp_path,
):
    module, _credential_type = azure_adapter
    secret_scope.set_multiplex_active(True)
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    provider_a = await _build_provider_for(module, profile_a, {})
    provider_b = await _build_provider_for(module, profile_b, {})
    credential_a = provider_a._hermes_credential
    credential_b = provider_b._hermes_credential

    await _reset_for(module, profile_a)

    assert credential_a.closed == 1
    assert credential_b.closed == 0
    assert provider_a._hermes_released is True
    assert provider_b._hermes_released is False
    await module._release_token_provider(provider_b)
    assert credential_b.closed == 1


@pytest.mark.asyncio
async def test_reset_finishes_close_under_repeated_cancellation(
    azure_adapter,
    tmp_path,
):
    module, _credential_type = azure_adapter
    home = tmp_path / "profile"
    credential = await _build_for(module, home, {})
    credential.close_started = asyncio.Event()
    credential.allow_close = asyncio.Event()

    task = asyncio.create_task(_reset_for(module, home))
    await credential.close_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    credential.allow_close.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert credential.closed == 1
    assert not module._credential_caches


@pytest.mark.asyncio
async def test_provider_release_finishes_close_under_repeated_cancellation(
    azure_adapter,
    tmp_path,
):
    module, _credential_type = azure_adapter
    provider = await _build_provider_for(module, tmp_path / "profile", {})
    credential = provider._hermes_credential
    credential.close_started = asyncio.Event()
    credential.allow_close = asyncio.Event()

    task = asyncio.create_task(module._release_token_provider(provider))
    await credential.close_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    credential.allow_close.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert credential.closed == 1
    assert provider._hermes_released is True
    assert provider._hermes_credential_loop is None
    assert provider._hermes_credential is None
    assert not module._credential_caches


@pytest.mark.asyncio
async def test_diagnostics_report_only_scoped_identity_environment(
    azure_adapter,
    monkeypatch,
):
    module, _credential_type = azure_adapter
    secret_scope.set_multiplex_active(True)
    monkeypatch.setenv("AZURE_CLIENT_ID", "foreign-client")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "foreign-secret")
    monkeypatch.setenv("AZURE_TENANT_ID", "foreign-tenant")
    scope_token = secret_scope.set_secret_scope(
        {
            "AZURE_CLIENT_ID": "profile-client",
            "AZURE_CLIENT_SECRET": "profile-secret",
            "AZURE_TENANT_ID": "profile-tenant",
        }
    )
    try:
        info = await module.describe_active_credential()
    finally:
        secret_scope.reset_secret_scope(scope_token)

    assert info["tenant_id_env"] == "profile-tenant"
    assert info["env_sources"] == ["EnvironmentCredential (client secret)"]
    assert "AZURE_CLIENT_SECRET" in info["error"]
    assert "profile-secret" not in info["error"]
    assert "foreign" not in repr(info)
