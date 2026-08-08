"""Tests for the preventive async SSL CA bundle guard."""

from pathlib import Path

import certifi
import pytest

from agent.errors import SSLConfigurationError
from agent.ssl_guard import verify_ca_bundle, verify_ca_bundle_with_fallback

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
