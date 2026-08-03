"""Focused tests for GMI Cloud first-class provider wiring."""

from __future__ import annotations

import contextlib
import io
import sys
import types
from argparse import Namespace
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.asyncio

if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv

from hermes_cli.auth import resolve_provider
from hermes_cli.config import load_config
from hermes_cli.models import (
    CANONICAL_PROVIDERS,
    _PROVIDER_LABELS,
    normalize_provider,
)


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    for key in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GLM_API_KEY",
        "KIMI_API_KEY",
        "MINIMAX_API_KEY",
        "GMI_API_KEY",
        "GMI_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


class TestGmiAliases:
    @pytest.mark.parametrize("alias", ["gmi", "gmi-cloud", "gmicloud"])
    async def test_alias_resolves(self, alias, monkeypatch):
        monkeypatch.setenv("GMI_API_KEY", "gmi-test-key")
        assert await resolve_provider(alias) == "gmi"

    async def test_models_normalize_provider(self):
        assert normalize_provider("gmi-cloud") == "gmi"
        assert normalize_provider("gmicloud") == "gmi"

    async def test_providers_normalize_provider(self):
        from hermes_cli.providers import normalize_provider as normalize_provider_in_providers

        assert normalize_provider_in_providers("gmi-cloud") == "gmi"
        assert normalize_provider_in_providers("gmicloud") == "gmi"


class TestGmiConfigRegistry:
    async def test_optional_env_vars_include_gmi(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS

        assert "GMI_API_KEY" in OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["GMI_API_KEY"]["category"] == "provider"
        assert OPTIONAL_ENV_VARS["GMI_API_KEY"]["password"] is True
        assert OPTIONAL_ENV_VARS["GMI_API_KEY"]["url"] == "https://www.gmicloud.ai/"

        assert "GMI_BASE_URL" in OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["GMI_BASE_URL"]["category"] == "provider"
        assert OPTIONAL_ENV_VARS["GMI_BASE_URL"]["password"] is False
        # ENV_VARS_BY_VERSION entries are not needed for providers added after
        # _config_version 22 (the current baseline) — users discover GMI via
        # hermes model, not via upgrade prompts.


class TestGmiModelCatalog:
    async def test_canonical_provider_entry(self):
        slugs = [p.slug for p in CANONICAL_PROVIDERS]
        assert "gmi" in slugs


class TestGmiProvidersModule:
    async def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS

        assert "gmi" in HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS["gmi"]
        assert overlay.transport == "openai_chat"
        assert overlay.extra_env_vars == ("GMI_API_KEY",)
        assert overlay.base_url_override == "https://api.gmi-serving.com/v1"
        assert overlay.base_url_env_var == "GMI_BASE_URL"
        assert not overlay.is_aggregator

    async def test_provider_label(self):
        assert _PROVIDER_LABELS["gmi"] == "GMI Cloud"


class TestGmiModelMetadata:
    async def test_url_to_provider(self):
        from agent.model_metadata import _URL_TO_PROVIDER

        assert _URL_TO_PROVIDER.get("api.gmi-serving.com") == "gmi"


    async def test_known_gmi_endpoint_still_uses_endpoint_metadata(self):
        import agent.model_metadata as model_metadata

        with patch(
            "agent.model_metadata.get_cached_context_length",
            return_value=None,
        ), patch(
            "agent.model_metadata.fetch_endpoint_model_metadata",
            return_value={"anthropic/claude-opus-4.6": {"context_length": 409600}},
        ), patch(
            "agent.models_dev.lookup_models_dev_context",
            return_value=None,
        ), patch(
            "agent.model_metadata.fetch_model_metadata",
            return_value={},
        ):
            result = await model_metadata.get_model_context_length(
                "anthropic/claude-opus-4.6",
                base_url="https://api.gmi-serving.com/v1",
                api_key="gmi-test-key",
                provider="custom",
            )

        assert result == 409600


class TestGmiAuxiliary:
    @pytest.mark.asyncio
    async def test_resolve_provider_client_uses_gmi_aux_default(self, monkeypatch):
        import agent.auxiliary_client as auxiliary_client

        monkeypatch.setenv("GMI_API_KEY", "gmi-test-key")

        with patch("agent.auxiliary_client._create_openai_client") as mock_openai:
            mock_openai.return_value = object()
            client, model = await auxiliary_client.resolve_provider_client("gmi")

        assert client is not None
        assert model == "google/gemini-3.1-flash-lite-preview"
        assert mock_openai.call_args.kwargs["api_key"] == "gmi-test-key"
        assert mock_openai.call_args.kwargs["base_url"] == "https://api.gmi-serving.com/v1"
        # GMI profile declares default_headers with a HermesAgent User-Agent
        # for traffic attribution. The generic profile-fallback branch in
        # resolve_provider_client should carry it through to the OpenAI client.
        headers = mock_openai.call_args.kwargs.get("default_headers", {})
        assert headers.get("User-Agent", "").startswith("HermesAgent/")

    async def test_gmi_profile_declares_hermes_user_agent(self):
        """The GMI plugin sets a HermesAgent/<ver> User-Agent on its profile."""
        from providers import get_provider_profile

        profile = get_provider_profile("gmi")
        assert profile is not None
        ua = profile.default_headers.get("User-Agent", "")
        assert ua.startswith("HermesAgent/"), (
            f"expected GMI profile User-Agent to start with 'HermesAgent/', got {ua!r}"
        )
