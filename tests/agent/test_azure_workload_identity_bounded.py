"""Bounded projected-token file I/O for Azure Workload Identity."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import secret_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def fake_identity(monkeypatch):
    from agent import azure_identity_adapter as module

    previous_multiplex = secret_scope.is_multiplex_active()
    secret_token = secret_scope.set_secret_scope(None)
    with module._credential_cache_guard:
        module._credential_caches.clear()
        module._credential_leases.clear()
        module._issued_providers.clear()

    class CredentialUnavailableError(Exception):
        pass

    class ClientAuthenticationError(Exception):
        pass

    class ClientAssertionCredential:
        instances: list[ClientAssertionCredential] = []
        fail_mode: str | None = None

        def __init__(self, tenant_id, client_id, func, *, authority):
            self.tenant_id = tenant_id
            self.client_id = client_id
            self.func = func
            self.authority = authority
            self.assertions: list[str] = []
            self.closed = 0
            self.instances.append(self)

        async def get_token(self, *_scopes, **_kwargs):
            if self.fail_mode == "unavailable":
                raise CredentialUnavailableError("workload unavailable")
            if self.fail_mode == "auth":
                raise ClientAuthenticationError("workload rejected")
            assertion = self.func()
            self.assertions.append(assertion)
            return SimpleNamespace(
                token=f"entra:{assertion}",
                expires_on=2_000_000_000,
            )

        async def get_token_info(self, *scopes, **kwargs):
            return await self.get_token(*scopes, **kwargs)

        async def close(self):
            self.closed += 1

    class DefaultAzureCredential:
        instances: list[DefaultAzureCredential] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = 0
            self.instances.append(self)

        async def get_token(self, *_scopes, **_kwargs):
            return SimpleNamespace(token="managed-token", expires_on=2_000_000_000)

        async def close(self):
            self.closed += 1

    class ChainedTokenCredential:
        instances: list[ChainedTokenCredential] = []

        def __init__(self, *credentials):
            self.credentials = credentials
            self.closed = 0
            self.instances.append(self)

        async def get_token(self, *scopes, **kwargs):
            errors = []
            for credential in self.credentials:
                try:
                    return await credential.get_token(*scopes, **kwargs)
                except CredentialUnavailableError as exc:
                    errors.append(exc)
            raise CredentialUnavailableError("all credentials unavailable") from errors[
                -1
            ]

        async def close(self):
            for credential in self.credentials:
                await credential.close()
            self.closed += 1

    identity = SimpleNamespace(
        ClientAssertionCredential=ClientAssertionCredential,
        ChainedTokenCredential=ChainedTokenCredential,
        ClientAuthenticationError=ClientAuthenticationError,
        CredentialUnavailableError=CredentialUnavailableError,
        DefaultAzureCredential=DefaultAzureCredential,
        get_bearer_token_provider=lambda credential, scope: (
            lambda: credential.get_token(scope)
        ),
    )
    monkeypatch.setattr(module, "_require_azure_identity", lambda: identity)
    monkeypatch.setattr(module, "has_azure_identity_installed", lambda: True)
    yield module, identity

    with module._credential_cache_guard:
        module._credential_caches.clear()
        module._credential_leases.clear()
        module._issued_providers.clear()
    secret_scope.reset_secret_scope(secret_token)
    secret_scope.set_multiplex_active(previous_multiplex)


def _credential(module, identity, token_file: Path):
    return module._AsyncWorkloadIdentityCredential(
        identity,
        tenant_id="tenant",
        client_id="client",
        token_file_path=str(token_file),
        authority="authority.invalid",
    )


@pytest.mark.asyncio
async def test_reads_assertion_only_on_initial_and_refresh_windows(
    fake_identity,
    monkeypatch,
    tmp_path,
):
    module, identity = fake_identity
    token_file = tmp_path / "token"
    token_file.write_text("jwt-a", encoding="utf-8")
    read_paths: list[str] = []
    real_open = module.aiofiles.open

    def counted_open(path, *args, **kwargs):
        read_paths.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(module.aiofiles, "open", counted_open)
    clock = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    credential = _credential(module, identity, token_file)
    try:
        first = await credential.get_token("scope")
        token_file.write_text("jwt-b", encoding="utf-8")
        second = await credential.get_token("scope")
        clock[0] = 701.0
        third = await credential.get_token("scope")
    finally:
        await credential.close()

    assert first.token == "entra:jwt-a"
    assert second.token == "entra:jwt-a"
    assert third.token == "entra:jwt-b"
    assert read_paths == [str(token_file), str(token_file)]


@pytest.mark.asyncio
async def test_concurrent_refresh_is_single_flight(
    fake_identity, monkeypatch, tmp_path
):
    module, identity = fake_identity
    token_file = tmp_path / "token"
    token_file.write_text("jwt", encoding="utf-8")
    open_count = 0
    real_open = module.aiofiles.open

    def counted_open(path, *args, **kwargs):
        nonlocal open_count
        open_count += 1
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(module.aiofiles, "open", counted_open)
    credential = _credential(module, identity, token_file)
    try:
        tokens = await asyncio.gather(
            *(credential.get_token("scope") for _ in range(20))
        )
    finally:
        await credential.close()

    assert open_count == 1
    assert {token.token for token in tokens} == {"entra:jwt"}


@pytest.mark.asyncio
async def test_public_azure_client_assertion_uses_cached_projected_token(
    monkeypatch,
    tmp_path,
):
    identity = pytest.importorskip("azure.identity.aio")
    from agent import azure_identity_adapter as module

    token_file = tmp_path / "token"
    token_file.write_text("jwt-public", encoding="utf-8")
    credential = module._AsyncWorkloadIdentityCredential(
        identity,
        tenant_id="tenant",
        client_id="client",
        token_file_path=str(token_file),
        authority="authority.invalid",
    )
    seen: list[str] = []

    async def exchange(scopes, assertion, **_kwargs):
        await asyncio.sleep(0)
        seen.append(assertion)
        return SimpleNamespace(token="entra:jwt-public", expires_on=2_000_000_000)

    monkeypatch.setattr(
        credential._inner._client,
        "get_cached_access_token",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        credential._inner._client,
        "obtain_token_by_jwt_assertion",
        exchange,
    )
    try:
        token = await credential.get_token("scope")
    finally:
        await credential.close()

    assert token.token == "entra:jwt-public"
    assert seen == ["jwt-public"]


@pytest.mark.asyncio
async def test_public_client_assertion_exchanges_with_loopback_token_server(
    tmp_path,
):
    identity = pytest.importorskip("azure.identity.aio")
    from agent import azure_identity_adapter as module

    requests: list[dict[str, str]] = []

    async def token_server(reader, writer):
        try:
            headers = (await reader.readuntil(b"\r\n\r\n")).decode("ascii")
            content_length = next(
                int(line.split(":", 1)[1].strip())
                for line in headers.split("\r\n")
                if line.lower().startswith("content-length:")
            )
            body = (await reader.readexactly(content_length)).decode("ascii")
            requests.append(dict(item.split("=", 1) for item in body.split("&")))
            payload = json.dumps({
                "access_token": "loopback-token",
                "expires_in": 3600,
                "token_type": "Bearer",
            }).encode("utf-8")
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: "
                + str(len(payload)).encode("ascii")
                + b"\r\nConnection: close\r\n\r\n"
                + payload
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(token_server, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    token_file = tmp_path / "token"
    token_file.write_text("jwt-loopback", encoding="utf-8")
    credential = module._AsyncWorkloadIdentityCredential(
        identity,
        tenant_id="tenant",
        client_id="client",
        token_file_path=str(token_file),
        authority="login.microsoftonline.com",
    )
    credential._inner._client._authority = f"http://127.0.0.1:{port}"

    class NoopTokenCache:
        def search(self, *_args, **_kwargs):
            return []

        def add(self, **_kwargs):
            return None

    credential._inner._client._cache = NoopTokenCache()
    credential._inner._client._cae_cache = NoopTokenCache()
    try:
        token = await asyncio.wait_for(credential.get_token("scope"), timeout=2)
    finally:
        await credential.close()
        server.close()
        await server.wait_closed()

    assert token.token == "loopback-token"
    assert len(requests) == 1
    assert requests[0]["client_assertion"] == "jwt-loopback"
    assert requests[0]["client_id"] == "client"
    assert requests[0]["grant_type"] == "client_credentials"
    assert requests[0]["scope"] == "scope"


@pytest.mark.asyncio
async def test_actual_build_credential_uses_public_bounded_chain(
    monkeypatch,
    tmp_path,
):
    pytest.importorskip("azure.identity.aio")
    from agent import azure_identity_adapter as module

    token_file = tmp_path / "token"
    token_file.write_text("jwt-public", encoding="utf-8")
    for name in module._AZURE_IDENTITY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AZURE_CLIENT_ID", "client")
    monkeypatch.setenv("AZURE_TENANT_ID", "tenant")
    monkeypatch.setenv("AZURE_FEDERATED_TOKEN_FILE", str(token_file))
    home_token = set_hermes_home_override(tmp_path / "profile")
    try:
        credential = await module.build_credential(module.EntraIdentityConfig())
        assert type(credential).__name__ == "ChainedTokenCredential"
        assert [type(item).__name__ for item in credential.credentials] == [
            "_AsyncWorkloadIdentityCredential",
            "DefaultAzureCredential",
        ]
        assert [
            type(item).__name__ for item in credential.credentials[1].credentials
        ] == [
            "ManagedIdentityCredential",
        ]
    finally:
        await module.reset_credential_cache()
        reset_hermes_home_override(home_token)


@pytest.mark.asyncio
async def test_workload_chain_precedes_managed_identity_and_preserves_profile_scope(
    fake_identity,
    monkeypatch,
    tmp_path,
):
    module, identity = fake_identity
    secret_scope.set_multiplex_active(True)
    monkeypatch.setenv("AZURE_CLIENT_ID", "foreign-client")
    monkeypatch.setenv("AZURE_TENANT_ID", "foreign-tenant")
    monkeypatch.setenv("AZURE_FEDERATED_TOKEN_FILE", "/foreign/token")
    token_file = tmp_path / "profile-token"
    token_file.write_text("profile-jwt", encoding="utf-8")
    home_token = set_hermes_home_override(tmp_path / "profile")
    scope_token = secret_scope.set_secret_scope({
        "AZURE_CLIENT_ID": "profile-client",
        "AZURE_TENANT_ID": "profile-tenant",
        "AZURE_FEDERATED_TOKEN_FILE": str(token_file),
    })
    workload = managed = None
    try:
        credential = await module.build_credential(module.EntraIdentityConfig())
        assert isinstance(credential, identity.ChainedTokenCredential)
        assert len(credential.credentials) == 2
        workload, managed = credential.credentials
        assert workload._inner.tenant_id == "profile-tenant"
        assert workload._inner.client_id == "profile-client"
        assert managed.kwargs["exclude_managed_identity_credential"] is False
        assert (await credential.get_token("scope")).token == "entra:profile-jwt"
    finally:
        await module.reset_credential_cache()
        if workload is not None and managed is not None:
            assert workload._inner.closed == 1
            assert managed.closed == 1
        secret_scope.reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)


@pytest.mark.asyncio
async def test_explicit_workload_selector_is_profile_safe(
    fake_identity,
    monkeypatch,
    tmp_path,
):
    module, identity = fake_identity
    secret_scope.set_multiplex_active(True)
    token_file = tmp_path / "profile-token"
    token_file.write_text("profile-jwt", encoding="utf-8")
    home_token = set_hermes_home_override(tmp_path / "profile")
    scope_token = secret_scope.set_secret_scope({
        "AZURE_CLIENT_ID": "profile-client",
        "AZURE_TENANT_ID": "profile-tenant",
        "AZURE_FEDERATED_TOKEN_FILE": str(token_file),
        "AZURE_TOKEN_CREDENTIALS": "WorkloadIdentityCredential",
    })
    monkeypatch.setenv("AZURE_CLIENT_ID", "foreign-client")
    try:
        credential = await module.build_credential(module.EntraIdentityConfig())
        assert isinstance(credential, identity.ChainedTokenCredential)
        assert (await credential.get_token("scope")).token == "entra:profile-jwt"
    finally:
        await module.reset_credential_cache()
        secret_scope.reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)


@pytest.mark.asyncio
async def test_concurrent_workload_profiles_do_not_share_assertions(
    fake_identity,
    monkeypatch,
    tmp_path,
):
    module, identity = fake_identity
    secret_scope.set_multiplex_active(True)
    monkeypatch.setenv("AZURE_CLIENT_ID", "foreign-client")
    monkeypatch.setenv("AZURE_TENANT_ID", "foreign-tenant")

    async def build_for(name: str):
        token_file = tmp_path / f"{name}.token"
        token_file.write_text(f"jwt-{name}", encoding="utf-8")
        home_token = set_hermes_home_override(tmp_path / name)
        scope_token = secret_scope.set_secret_scope({
            "AZURE_CLIENT_ID": f"client-{name}",
            "AZURE_TENANT_ID": f"tenant-{name}",
            "AZURE_FEDERATED_TOKEN_FILE": str(token_file),
        })
        try:
            credential = await module.build_credential(module.EntraIdentityConfig())
            token = await credential.get_token("scope")
            return credential, token.token
        finally:
            await module.reset_credential_cache()
            secret_scope.reset_secret_scope(scope_token)
            reset_hermes_home_override(home_token)

    (credential_a, token_a), (credential_b, token_b) = await asyncio.gather(
        build_for("a"),
        build_for("b"),
    )

    assert token_a == "entra:jwt-a"
    assert token_b == "entra:jwt-b"
    assert credential_a is not credential_b
    assert credential_a.credentials[0]._inner.client_id == "client-a"
    assert credential_b.credentials[0]._inner.client_id == "client-b"
    assert len(identity.ChainedTokenCredential.instances) == 2


@pytest.mark.asyncio
async def test_workload_token_proxy_is_explicitly_unsupported(
    fake_identity,
    tmp_path,
):
    module, identity = fake_identity
    secret_scope.set_multiplex_active(True)
    token_file = tmp_path / "profile-token"
    token_file.write_text("profile-jwt", encoding="utf-8")
    home_token = set_hermes_home_override(tmp_path / "profile")
    scope_token = secret_scope.set_secret_scope({
        "AZURE_CLIENT_ID": "profile-client",
        "AZURE_TENANT_ID": "profile-tenant",
        "AZURE_FEDERATED_TOKEN_FILE": str(token_file),
        "AZURE_KUBERNETES_TOKEN_PROXY": "http://proxy.invalid",
    })
    try:
        with pytest.raises(RuntimeError, match="token proxy"):
            await module.build_credential(module.EntraIdentityConfig())
        assert identity.ClientAssertionCredential.instances == []
    finally:
        secret_scope.reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)


@pytest.mark.asyncio
async def test_client_secret_precedes_workload_identity(
    fake_identity, monkeypatch, tmp_path
):
    module, identity = fake_identity
    secret_scope.set_multiplex_active(False)
    monkeypatch.setenv("AZURE_CLIENT_ID", "client")
    monkeypatch.setenv("AZURE_TENANT_ID", "tenant")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "secret")
    token_file = tmp_path / "token"
    token_file.write_text("jwt", encoding="utf-8")
    monkeypatch.setenv("AZURE_FEDERATED_TOKEN_FILE", str(token_file))
    home_token = set_hermes_home_override(tmp_path / "profile")
    try:
        credential = await module.build_credential(module.EntraIdentityConfig())
    finally:
        await module.reset_credential_cache()
        reset_hermes_home_override(home_token)

    assert isinstance(credential, identity.DefaultAzureCredential)
    assert len(identity.ClientAssertionCredential.instances) == 0
    assert credential.kwargs["exclude_environment_credential"] is False


@pytest.mark.asyncio
async def test_unavailable_workload_credential_falls_back_to_managed(
    fake_identity,
    tmp_path,
):
    module, identity = fake_identity
    identity.ClientAssertionCredential.fail_mode = "unavailable"
    token_file = tmp_path / "token"
    token_file.write_text("jwt", encoding="utf-8")
    credential = _credential(module, identity, token_file)
    chain = identity.ChainedTokenCredential(
        credential, identity.DefaultAzureCredential()
    )
    try:
        token = await chain.get_token("scope")
    finally:
        await chain.close()

    assert token.token == "managed-token"


@pytest.mark.asyncio
async def test_authentication_failure_does_not_fall_through(
    fake_identity,
    tmp_path,
):
    module, identity = fake_identity
    identity.ClientAssertionCredential.fail_mode = "auth"
    token_file = tmp_path / "token"
    token_file.write_text("jwt", encoding="utf-8")
    chain = identity.ChainedTokenCredential(
        _credential(module, identity, token_file),
        identity.DefaultAzureCredential(),
    )
    try:
        with pytest.raises(identity.ClientAuthenticationError):
            await chain.get_token("scope")
    finally:
        await chain.close()


@pytest.mark.asyncio
async def test_built_workload_chain_falls_back_only_on_unavailable(
    fake_identity,
    monkeypatch,
    tmp_path,
):
    module, identity = fake_identity
    identity.ClientAssertionCredential.fail_mode = "unavailable"
    token_file = tmp_path / "token"
    token_file.write_text("jwt", encoding="utf-8")
    secret_scope.set_multiplex_active(False)
    home_token = set_hermes_home_override(tmp_path / "profile")
    monkeypatch.setenv("AZURE_CLIENT_ID", "client")
    monkeypatch.setenv("AZURE_TENANT_ID", "tenant")
    monkeypatch.setenv("AZURE_FEDERATED_TOKEN_FILE", str(token_file))
    try:
        credential = await module.build_credential(module.EntraIdentityConfig())
        assert (await credential.get_token("scope")).token == "managed-token"
    finally:
        await module.reset_credential_cache()
        reset_hermes_home_override(home_token)


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", "x" * (64 * 1024 + 1)])
async def test_invalid_assertion_file_fails_closed(fake_identity, tmp_path, content):
    module, identity = fake_identity
    token_file = tmp_path / "token"
    token_file.write_text(content, encoding="utf-8")
    credential = _credential(module, identity, token_file)
    try:
        with pytest.raises((ValueError, identity.CredentialUnavailableError)):
            await credential.get_token("scope")
    finally:
        await credential.close()


@pytest.mark.asyncio
async def test_missing_assertion_file_is_unavailable(fake_identity, tmp_path):
    module, identity = fake_identity
    credential = _credential(module, identity, tmp_path / "missing-token")
    try:
        with pytest.raises(identity.CredentialUnavailableError):
            await credential.get_token("scope")
    finally:
        await credential.close()


@pytest.mark.asyncio
async def test_assertion_read_timeout_is_unavailable(
    fake_identity, monkeypatch, tmp_path
):
    module, identity = fake_identity
    token_file = tmp_path / "token"
    token_file.write_text("jwt", encoding="utf-8")
    monkeypatch.setattr(module, "_WORKLOAD_TOKEN_READ_TIMEOUT_SECONDS", 0.01)
    real_open = module.aiofiles.open

    class SlowFile:
        async def __aenter__(self):
            self._manager = real_open(token_file, mode="r", encoding="utf-8")
            self._file = await self._manager.__aenter__()
            return self

        async def __aexit__(self, *args):
            return await self._manager.__aexit__(*args)

        async def read(self, _size=-1):
            await asyncio.sleep(0.1)
            return await self._file.read()

    monkeypatch.setattr(module.aiofiles, "open", lambda *args, **kwargs: SlowFile())
    credential = _credential(module, identity, token_file)
    try:
        with pytest.raises(identity.CredentialUnavailableError, match="timed out"):
            await credential.get_token("scope")
    finally:
        await credential.close()


@pytest.mark.asyncio
async def test_read_cancellation_is_rethrown_after_owned_read_finishes(
    fake_identity,
    monkeypatch,
    tmp_path,
):
    module, identity = fake_identity
    token_file = tmp_path / "token"
    token_file.write_text("jwt", encoding="utf-8")
    opened = asyncio.Event()
    release = asyncio.Event()
    real_open = module.aiofiles.open

    class SlowFile:
        async def __aenter__(self):
            self._manager = real_open(token_file, mode="r", encoding="utf-8")
            self._file = await self._manager.__aenter__()
            opened.set()
            return self

        async def __aexit__(self, *args):
            return await self._manager.__aexit__(*args)

        async def read(self, size=-1):
            await release.wait()
            return await self._file.read(size)

    monkeypatch.setattr(module.aiofiles, "open", lambda *args, **kwargs: SlowFile())
    credential = _credential(module, identity, token_file)
    task = asyncio.create_task(credential.get_token("scope"))
    await opened.wait()
    task.cancel()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await credential.close()
