from __future__ import annotations

import base64
import json
import time
from unittest.mock import AsyncMock

import pytest

from hermes_cli import auth


def _jwt_with_exp(exp: int) -> str:
    header = (
        base64
        .urlsafe_b64encode(json.dumps({"alg": "none"}).encode())
        .decode()
        .rstrip("=")
    )
    payload = (
        base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).decode().rstrip("=")
    )
    return f"{header}.{payload}.sig"


def test_xai_oauth_refresh_skew_is_one_hour() -> None:
    assert auth.XAI_ACCESS_TOKEN_REFRESH_SKEW_SECONDS == 3600


def test_xai_oauth_token_expiring_uses_one_hour_skew() -> None:
    token = _jwt_with_exp(int(time.time()) + 30 * 60)

    assert auth._xai_access_token_is_expiring(
        token,
        auth.XAI_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    )


def test_xai_proactive_refresh_skew_short_lived_token() -> None:
    token = _jwt_with_exp(int(time.time()) + 15 * 60)
    skew = auth._xai_proactive_refresh_skew_seconds(token)

    assert skew == 120
    assert not auth._xai_access_token_is_expiring(token, skew)


class _Response:
    def __init__(self, status_code: int, payload, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _Client:
    response: _Response
    posted: dict | None = None

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        del exc_type, exc, tb

    async def post(self, url, *, headers, data):
        type(self).posted = {"url": url, "headers": headers, "data": data}
        return type(self).response


@pytest.mark.asyncio
async def test_refresh_xai_oauth_pure_preserves_success_contract(monkeypatch) -> None:
    _Client.response = _Response(
        200,
        {
            "access_token": "fresh-access",
            "refresh_token": "fresh-refresh",
            "id_token": "fresh-id",
            "expires_in": 900,
            "token_type": "Bearer",
        },
    )
    monkeypatch.setattr(auth.httpx, "AsyncClient", _Client)

    refreshed = await auth.refresh_xai_oauth_pure(
        "stale-access",
        "stale-refresh",
        token_endpoint="https://accounts.x.ai/oauth2/token",
    )

    assert refreshed["access_token"] == "fresh-access"
    assert refreshed["refresh_token"] == "fresh-refresh"
    assert refreshed["id_token"] == "fresh-id"
    assert refreshed["expires_in"] == 900
    assert refreshed["token_type"] == "Bearer"
    assert refreshed["last_refresh"].endswith("Z")
    assert _Client.posted == {
        "url": "https://accounts.x.ai/oauth2/token",
        "headers": {"Content-Type": "application/x-www-form-urlencoded"},
        "data": {
            "grant_type": "refresh_token",
            "client_id": auth.XAI_OAUTH_CLIENT_ID,
            "refresh_token": "stale-refresh",
        },
    }


@pytest.mark.asyncio
async def test_refresh_xai_oauth_pure_preserves_tier_denied_error(monkeypatch) -> None:
    _Client.response = _Response(403, {}, "tier denied")
    monkeypatch.setattr(auth.httpx, "AsyncClient", _Client)

    with pytest.raises(auth.AuthError) as error:
        await auth.refresh_xai_oauth_pure(
            "stale-access",
            "stale-refresh",
            token_endpoint="https://accounts.x.ai/oauth2/token",
        )

    assert error.value.code == "xai_oauth_tier_denied"
    assert error.value.relogin_required is False


@pytest.mark.asyncio
async def test_refresh_xai_oauth_pure_requires_refresh_token() -> None:
    with pytest.raises(auth.AuthError) as error:
        await auth.refresh_xai_oauth_pure("stale-access", "")

    assert error.value.code == "xai_auth_missing_refresh_token"
    assert error.value.relogin_required is True


@pytest.mark.asyncio
async def test_resolve_xai_oauth_runtime_credentials_preserves_public_contract(
    monkeypatch,
) -> None:
    class _Entry:
        id = "xai-entry"
        access_token = "xai-access"
        runtime_api_key = "xai-access"
        runtime_base_url = auth.DEFAULT_XAI_OAUTH_BASE_URL
        last_refresh = "2026-08-08T00:00:00Z"

    class _Pool:
        def has_credentials(self):
            return True

        async def select(self):
            return _Entry()

    monkeypatch.setattr(
        "agent.credential_pool.load_pool",
        AsyncMock(return_value=_Pool()),
    )
    monkeypatch.setattr(
        auth,
        "get_provider_auth_state",
        AsyncMock(return_value={"tokens": {"access_token": "xai-access"}}),
    )

    credentials = await auth.resolve_xai_oauth_runtime_credentials()

    assert credentials == {
        "provider": "xai-oauth",
        "base_url": auth.DEFAULT_XAI_OAUTH_BASE_URL,
        "api_key": "xai-access",
        "source": "hermes-auth-store",
        "last_refresh": "2026-08-08T00:00:00Z",
        "auth_mode": "oauth_device_code",
    }


@pytest.mark.asyncio
async def test_xai_terminal_refresh_failure_quarantines_singleton(
    monkeypatch,
    tmp_path,
) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    auth_path = hermes_home / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {
                    "xai-oauth": {
                        "tokens": {
                            "access_token": "stale-access",
                            "refresh_token": "stale-refresh",
                        }
                    }
                },
                "credential_pool": {
                    "xai-oauth": [
                        {
                            "id": "xai-device",
                            "source": "device_code",
                            "auth_type": "oauth",
                            "access_token": "stale-access",
                            "refresh_token": "stale-refresh",
                            "base_url": auth.DEFAULT_XAI_OAUTH_BASE_URL,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    terminal_error = auth.AuthError(
        "refresh token revoked",
        provider="xai-oauth",
        code="xai_refresh_failed",
        relogin_required=True,
    )
    monkeypatch.setattr(
        auth,
        "refresh_xai_oauth_pure",
        AsyncMock(side_effect=terminal_error),
    )

    from agent.credential_pool import load_pool

    pool = await load_pool("xai-oauth")
    assert await pool.try_refresh_matching(credential_id="xai-device") is None

    stored = json.loads(auth_path.read_text(encoding="utf-8"))
    state = stored["providers"]["xai-oauth"]
    assert "access_token" not in state["tokens"]
    assert "refresh_token" not in state["tokens"]
    assert state["last_auth_error"]["code"] == "xai_refresh_failed"
    assert stored["credential_pool"]["xai-oauth"] == []
