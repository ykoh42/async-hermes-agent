from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from blockbuster import BlockBuster

from tools import managed_tool_gateway as gateway
from tools import tool_backend_helpers


@pytest.mark.asyncio
async def test_peek_nous_access_token_reads_auth_store_without_blocking(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "auth.json").write_text(
        json.dumps(
            {
                "providers": {
                    "nous": {
                        "access_token": "cached-token",
                        "expires_at": "2099-01-01T00:00:00Z",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    blocker = BlockBuster()
    blocker.activate()
    try:
        assert await gateway.peek_nous_access_token() == "cached-token"
        assert await gateway.read_nous_access_token() == "cached-token"
    finally:
        blocker.deactivate()


@pytest.mark.asyncio
async def test_gateway_token_override_is_isolated_between_concurrent_profiles(
    monkeypatch,
):
    from agent import secret_scope

    monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "wrong-process-token")
    secret_scope.set_multiplex_active(True)

    async def read(scoped_token):
        token = secret_scope.set_secret_scope(
            {"TOOL_GATEWAY_USER_TOKEN": scoped_token}
        )
        try:
            await asyncio.sleep(0)
            return await gateway.peek_nous_access_token()
        finally:
            secret_scope.reset_secret_scope(token)

    try:
        assert await asyncio.gather(
            read("profile-a-token"),
            read("profile-b-token"),
        ) == ["profile-a-token", "profile-b-token"]
    finally:
        secret_scope.set_multiplex_active(False)


@pytest.mark.asyncio
async def test_gateway_token_override_does_not_borrow_process_token(
    monkeypatch,
    tmp_path,
):
    from agent import secret_scope

    monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "wrong-process-token")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    previous = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        empty_scope = secret_scope.set_secret_scope({})
        try:
            assert await gateway.peek_nous_access_token() is None
        finally:
            secret_scope.reset_secret_scope(empty_scope)

        with pytest.raises(
            secret_scope.UnscopedSecretError,
            match="TOOL_GATEWAY_USER_TOKEN",
        ):
            await gateway.peek_nous_access_token()
    finally:
        secret_scope.set_multiplex_active(previous)


@pytest.mark.asyncio
async def test_read_nous_access_token_refreshes_expiring_token(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "auth.json").write_text(
        json.dumps(
            {
                "providers": {
                    "nous": {
                        "access_token": "stale-token",
                        "expires_at": "2000-01-01T00:00:00Z",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    refresh = AsyncMock(return_value="fresh-token")
    monkeypatch.setattr("hermes_cli.auth.resolve_nous_access_token", refresh)

    assert await gateway.read_nous_access_token() == "fresh-token"
    refresh.assert_awaited_once_with(refresh_skew_seconds=120)


@pytest.mark.asyncio
async def test_resolve_managed_gateway_preserves_upstream_config_shape(
    monkeypatch,
):
    entitled = AsyncMock(return_value=True)
    monkeypatch.setattr(gateway, "managed_nous_tools_enabled", entitled)

    resolved = await gateway.resolve_managed_tool_gateway(
        "fal-queue",
        gateway_builder=lambda vendor: f"https://{vendor}.gateway.test",
        token_reader=AsyncMock(return_value="nous-token"),
    )

    assert resolved == gateway.ManagedToolGatewayConfig(
        vendor="fal-queue",
        gateway_origin="https://fal-queue.gateway.test",
        nous_user_token="nous-token",
        managed_mode=True,
    )
    entitled.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_resolve_managed_gateway_fails_closed_before_token_read(
    monkeypatch,
):
    monkeypatch.setattr(
        gateway,
        "managed_nous_tools_enabled",
        AsyncMock(return_value=False),
    )
    token_reader = AsyncMock(return_value="must-not-be-read")

    assert await gateway.resolve_managed_tool_gateway(
        "fal-queue",
        token_reader=token_reader,
    ) is None
    token_reader.assert_not_awaited()


def test_managed_vendor_endpoints_pin_upstream_gateway_contract(monkeypatch):
    monkeypatch.setenv("TOOL_GATEWAY_DOMAIN", "nousresearch.com")
    monkeypatch.setenv("TOOL_GATEWAY_SCHEME", "https")
    monkeypatch.delenv("TOOL_GATEWAY_URL", raising=False)

    assert gateway.managed_vendor_endpoints("bfl") == {
        "origin": "https://tool-gateway.nousresearch.com",
        "base_url": "https://tool-gateway.nousresearch.com/api/bfl",
        "upload_path": "/api/uploads/bfl",
    }


@pytest.mark.asyncio
async def test_gateway_readiness_uses_cached_token_without_refresh(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "auth.json").write_text(
        json.dumps(
            {
                "providers": {
                    "nous": {
                        "access_token": "expired-but-cached-token",
                        "expires_at": "2000-01-01T00:00:00Z",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        gateway,
        "managed_nous_tools_enabled",
        AsyncMock(return_value=True),
    )
    refresh = AsyncMock(return_value="fresh-token")
    monkeypatch.setattr("hermes_cli.auth.resolve_nous_access_token", refresh)

    assert await gateway.is_managed_tool_gateway_ready("fal-queue") is True
    refresh.assert_not_awaited()


@pytest.mark.asyncio
async def test_managed_gateway_auth_headers_only_trust_configured_origin():
    builder = lambda vendor: "https://tool.gateway.test"
    token_reader = AsyncMock(return_value="live-token")

    assert await gateway.managed_gateway_auth_headers(
        "https://tool.gateway.test/api/krea/run",
        builder,
        token_reader,
    ) == {"Authorization": "Bearer live-token"}
    assert await gateway.managed_gateway_auth_headers(
        "https://attacker.test/api/krea/run",
        builder,
        token_reader,
    ) == {}
    token_reader.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_managed_gateway_auth_headers_reflect_rotated_token():
    tokens = iter(["first-token", "second-token"])
    builder = lambda vendor: "https://tool.gateway.test"
    url = "https://tool.gateway.test/api/krea/run"

    first = await gateway.managed_gateway_auth_headers(
        url,
        builder,
        lambda: next(tokens),
    )
    second = await gateway.managed_gateway_auth_headers(
        url,
        builder,
        lambda: next(tokens),
    )

    assert first == {"Authorization": "Bearer first-token"}
    assert second == {"Authorization": "Bearer second-token"}


@pytest.mark.parametrize(
    ("server_url", "upload_path"),
    [
        ("https://attacker.test/api/krea", "/api/uploads/krea"),
        ("https://tool.gateway.test/api/krea", None),
        ("https://tool.gateway.test/api/krea", "api/uploads/krea"),
    ],
)
def test_managed_media_uploader_rejects_untrusted_or_unrooted_target(
    server_url,
    upload_path,
):
    assert gateway.build_managed_media_uploader(
        server_url,
        upload_path,
        lambda vendor: "https://tool.gateway.test",
    ) is None


@pytest.mark.asyncio
async def test_managed_media_uploader_preserves_protocol_and_closes_clients(
    monkeypatch,
):
    calls = {}

    class Response:
        def __init__(self, status_code, payload=None):
            self.status_code = status_code
            self._payload = payload

        def json(self):
            return self._payload

    class Client:
        def __init__(self, kind):
            self.kind = kind

        async def post(self, url, **kwargs):
            calls["post"] = (url, kwargs)
            return Response(
                200,
                {
                    "uploadUrl": "https://storage.test/presigned",
                    "token": "upload-token",
                },
            )

        async def put(self, url, **kwargs):
            calls["put"] = (url, kwargs)
            return Response(200)

        async def aclose(self):
            calls[f"{self.kind}_closed"] = True

    async def create_httpx_client(**kwargs):
        calls["presign_timeout"] = kwargs["timeout"]
        return Client("presign")

    async def create_ssrf_safe_client(**kwargs):
        calls["put_timeout"] = kwargs["timeout"]
        return Client("put")

    monkeypatch.setattr(
        "agent.ssl_verify._create_httpx_client",
        create_httpx_client,
    )
    monkeypatch.setattr(
        "tools.url_safety.create_ssrf_safe_client",
        create_ssrf_safe_client,
    )
    builder = lambda vendor: "https://tool.gateway.test"
    uploader = gateway.build_managed_media_uploader(
        "https://tool.gateway.test/api/krea",
        "/api/uploads/krea",
        builder,
        AsyncMock(return_value="nous-token"),
    )
    assert uploader is not None

    blocker = BlockBuster()
    blocker.activate()
    try:
        result = await uploader(b"image-bytes", "image/png")
    finally:
        blocker.deactivate()

    assert result == "nous-upload:upload-token"
    assert calls["post"] == (
        "https://tool.gateway.test/api/uploads/krea",
        {
            "headers": {"Authorization": "Bearer nous-token"},
            "json": {"contentType": "image/png", "contentLength": 11},
        },
    )
    assert calls["put"] == (
        "https://storage.test/presigned",
        {
            "content": b"image-bytes",
            "headers": {"Content-Type": "image/png"},
        },
    )
    assert calls["presign_timeout"].connect == 15.0
    assert calls["put_timeout"].read == 60.0
    assert calls["put_timeout"].write == 300.0
    assert calls["presign_closed"] is True
    assert calls["put_closed"] is True


@pytest.mark.asyncio
async def test_managed_nous_tools_enabled_awaits_account_lookup(monkeypatch):
    account = SimpleNamespace(logged_in=True, tool_gateway_entitled=True)
    lookup = AsyncMock(return_value=account)
    monkeypatch.setattr(
        "hermes_cli.nous_account.get_nous_portal_account_info",
        lookup,
    )

    assert await tool_backend_helpers.managed_nous_tools_enabled(
        force_fresh=True
    ) is True
    lookup.assert_awaited_once_with(force_fresh=True)


@pytest.mark.asyncio
async def test_prefers_gateway_uses_awaited_config(monkeypatch):
    load = AsyncMock(return_value={"image_gen": {"use_gateway": "yes"}})
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", load)

    assert await tool_backend_helpers.prefers_gateway("image_gen") is True
    load.assert_awaited_once_with()
