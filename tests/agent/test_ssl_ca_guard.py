"""Tests for the preventive async SSL CA bundle guard."""

from pathlib import Path
import ssl
import threading

import certifi
import pytest
from blockbuster import BlockBuster

from agent.errors import SSLConfigurationError
from agent.ssl_guard import (
    _CA_BUNDLE_ENV_VARS,
    verify_ca_bundle,
    verify_ca_bundle_with_fallback,
)

pytestmark = pytest.mark.asyncio


async def test_healthy_bundle_passes(monkeypatch):
    for key in (
        "HERMES_CA_BUNDLE",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
    ):
        monkeypatch.delenv(key, raising=False)
    bundle = Path(certifi.where())
    assert bundle.exists()
    assert bundle.stat().st_size > 1024
    await verify_ca_bundle()
    await verify_ca_bundle_with_fallback()


async def test_healthy_bundle_does_not_read_files_on_event_loop(monkeypatch):
    for key in _CA_BUNDLE_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    event_loop_thread = threading.get_ident()
    create_default_context = ssl.create_default_context

    def tracked_create_default_context(*args, **kwargs):
        assert threading.get_ident() != event_loop_thread
        return create_default_context(*args, **kwargs)

    monkeypatch.setattr(ssl, "create_default_context", tracked_create_default_context)
    blocker = BlockBuster()
    blocker.activate()
    try:
        await verify_ca_bundle()
    finally:
        blocker.deactivate()


async def test_empty_certifi_bundle_raises_ssl_error(monkeypatch, tmp_path):
    fake = tmp_path / "empty.pem"
    fake.write_bytes(b"")
    monkeypatch.setattr(certifi, "where", lambda: str(fake))
    with pytest.raises(SSLConfigurationError, match="too small"):
        await verify_ca_bundle()


@pytest.mark.parametrize(
    "env_var",
    [
        "HERMES_CA_BUNDLE",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
    ],
)
async def test_missing_explicit_ca_bundle_raises_before_client(
    monkeypatch,
    tmp_path,
    env_var,
):
    fake = tmp_path / "missing.pem"
    monkeypatch.setenv(env_var, str(fake))
    with pytest.raises(SSLConfigurationError) as exc_info:
        await verify_ca_bundle()
    message = str(exc_info.value)
    assert env_var in message
    assert str(fake) in message
    assert "Repair" in message
