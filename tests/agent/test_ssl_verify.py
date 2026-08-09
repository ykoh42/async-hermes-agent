"""Tests for agent.ssl_verify.resolve_httpx_verify."""

import ssl
import threading

import certifi
import pytest
from blockbuster import BlockBuster

from agent.ssl_verify import _resolve_httpx_client_verify, resolve_httpx_verify

_CA_ENV_VARS = (
    "HERMES_CA_BUNDLE",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def clean_ca_env(monkeypatch):
    for var in _CA_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


async def test_hermes_ca_bundle_returns_ssl_context(clean_ca_env, monkeypatch):
    monkeypatch.setenv("HERMES_CA_BUNDLE", certifi.where())
    result = await resolve_httpx_verify()
    assert isinstance(result, ssl.SSLContext)


async def test_default_without_env_is_true(clean_ca_env):
    assert await resolve_httpx_verify() is True


async def test_ca_directory_falls_back_to_upstream_true(
    clean_ca_env,
    tmp_path,
):
    assert await resolve_httpx_verify(ca_bundle=str(tmp_path)) is True


async def test_ssl_cert_dir_does_not_change_upstream_resolution(
    clean_ca_env,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path))
    assert await resolve_httpx_verify() is True


async def test_client_materializer_preserves_httpx_ssl_cert_dir(
    clean_ca_env,
    tmp_path,
    monkeypatch,
):
    calls = []
    create_default_context = ssl.create_default_context

    def tracked_create_default_context(*args, **kwargs):
        calls.append(kwargs)
        return create_default_context(*args, **kwargs)

    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path))
    monkeypatch.setattr(ssl, "create_default_context", tracked_create_default_context)

    result = await _resolve_httpx_client_verify()

    assert isinstance(result, ssl.SSLContext)
    assert calls == [{"capath": str(tmp_path)}]


async def test_default_context_creation_does_not_block_event_loop(
    clean_ca_env,
    monkeypatch,
):
    event_loop_thread = threading.get_ident()
    create_default_context = ssl.create_default_context
    certifi_where = certifi.where

    def tracked_create_default_context(*args, **kwargs):
        assert threading.get_ident() != event_loop_thread
        return create_default_context(*args, **kwargs)

    def tracked_certifi_where():
        assert threading.get_ident() != event_loop_thread
        return certifi_where()

    monkeypatch.setattr(ssl, "create_default_context", tracked_create_default_context)
    monkeypatch.setattr(certifi, "where", tracked_certifi_where)
    blocker = BlockBuster()
    blocker.activate()
    try:
        result = await _resolve_httpx_client_verify()
    finally:
        blocker.deactivate()

    assert isinstance(result, ssl.SSLContext)
