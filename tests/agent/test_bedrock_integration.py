"""Integration tests for the AWS Bedrock provider wiring.

Verifies that the Bedrock provider is correctly registered in the
provider registry, model catalog, and runtime resolution pipeline.
These tests do NOT require AWS credentials or a live Bedrock endpoint — all AWS calls
are mocked.

Note: Tests that import ``hermes_cli.auth`` or ``hermes_cli.runtime_provider``
require Python 3.10+ due to ``str | None`` type syntax in the import chain.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestProviderRegistry:
    """Verify Bedrock is registered in PROVIDER_REGISTRY."""

    def test_bedrock_in_registry(self):
        from hermes_cli.auth import PROVIDER_REGISTRY
        assert "bedrock" in PROVIDER_REGISTRY


    def test_bedrock_has_no_api_key_env_vars(self):
        """Bedrock uses the AWS SDK credential chain, not API keys."""
        from hermes_cli.auth import PROVIDER_REGISTRY
        pconfig = PROVIDER_REGISTRY["bedrock"]
        assert pconfig.api_key_env_vars == ()



class TestProviderAliases:
    """Verify Bedrock aliases resolve correctly."""

    def test_aws_alias(self):
        from hermes_cli.models import _PROVIDER_ALIASES
        assert _PROVIDER_ALIASES.get("aws") == "bedrock"





class TestProviderLabels:
    """Verify Bedrock appears in provider labels."""

    def test_bedrock_label(self):
        from hermes_cli.models import _PROVIDER_LABELS
        assert _PROVIDER_LABELS.get("bedrock") == "AWS Bedrock"


class TestModelCatalog:
    """Verify Bedrock has a static model fallback list."""

    def test_bedrock_has_curated_models(self):
        from hermes_cli.models import _PROVIDER_MODELS
        models = _PROVIDER_MODELS.get("bedrock", [])
        assert len(models) > 0

    def test_bedrock_models_include_claude(self):
        from hermes_cli.models import _PROVIDER_MODELS
        models = _PROVIDER_MODELS.get("bedrock", [])
        claude_models = [m for m in models if "anthropic.claude" in m]
        assert len(claude_models) > 0

    def test_bedrock_models_include_nova(self):
        from hermes_cli.models import _PROVIDER_MODELS
        models = _PROVIDER_MODELS.get("bedrock", [])
        nova_models = [m for m in models if "amazon.nova" in m]
        assert len(nova_models) > 0


class TestResolveProvider:
    """Verify resolve_provider() handles bedrock correctly."""

    @pytest.mark.asyncio
    async def test_explicit_bedrock_resolves(self, monkeypatch):
        """When user explicitly requests 'bedrock', it should resolve."""
        # bedrock is in the registry, so resolve_provider should return it
        from hermes_cli.auth import resolve_provider
        result = await resolve_provider("bedrock")
        assert result == "bedrock"

    @pytest.mark.asyncio
    async def test_aws_alias_resolves_to_bedrock(self):
        from hermes_cli.auth import resolve_provider
        result = await resolve_provider("aws")
        assert result == "bedrock"

    @pytest.mark.asyncio
    async def test_auto_detects_bedrock_after_other_provider_sources(self, monkeypatch):
        from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider

        for provider in PROVIDER_REGISTRY.values():
            for env_var in provider.api_key_env_vars:
                monkeypatch.delenv(env_var, raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        has_credentials = AsyncMock(return_value=True)
        with (
            patch(
                "hermes_cli.config.load_config_readonly",
                new=AsyncMock(return_value={}),
            ),
            patch(
                "hermes_cli.auth._load_auth_store",
                new=AsyncMock(return_value={}),
            ),
            patch(
                "agent.bedrock_adapter.has_aws_credentials",
                new=has_credentials,
            ),
        ):
            assert await resolve_provider("auto") == "bedrock"

        has_credentials.assert_awaited_once_with()


class TestRuntimeProvider:
    """Verify resolve_runtime_provider() handles bedrock correctly."""



    @pytest.mark.asyncio
    async def test_bedrock_runtime_no_credentials_raises_on_auto_detect(self, monkeypatch):
        """When bedrock is auto-detected (not explicitly requested) and no
        credentials are found, runtime resolution should raise AuthError."""
        from hermes_cli.runtime_provider import resolve_runtime_provider
        from hermes_cli.auth import AuthError

        # Clear all AWS env vars
        for var in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_PROFILE",
                     "AWS_BEARER_TOKEN_BEDROCK", "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
                     "AWS_WEB_IDENTITY_TOKEN_FILE"]:
            monkeypatch.delenv(var, raising=False)

        with patch("hermes_cli.runtime_provider.resolve_provider", return_value="bedrock"), \
             patch("hermes_cli.runtime_provider._get_model_config", return_value={"provider": "bedrock"}), \
             patch("hermes_cli.runtime_provider.resolve_requested_provider", return_value="auto"), \
             patch("agent.bedrock_adapter.resolve_aws_auth_env_var", new=AsyncMock(return_value=None)):
            with pytest.raises(AuthError, match="No AWS credentials"):
                await resolve_runtime_provider(requested="auto")

    @pytest.mark.asyncio
    async def test_bedrock_runtime_explicit_skips_credential_check(self, monkeypatch):
        """Explicit Bedrock trusts the SDK's IAM/instance-role chain."""
        from hermes_cli.runtime_provider import resolve_runtime_provider

        # No AWS env vars set — but explicit bedrock request should not raise
        for var in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_PROFILE",
                     "AWS_BEARER_TOKEN_BEDROCK"]:
            monkeypatch.delenv(var, raising=False)

        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
        with patch("hermes_cli.runtime_provider.resolve_provider", return_value="bedrock"), \
             patch("hermes_cli.runtime_provider._get_model_config", return_value={"provider": "bedrock"}), \
             patch("agent.bedrock_adapter.resolve_aws_auth_env_var", new=AsyncMock(return_value=None)):
            runtime = await resolve_runtime_provider(requested="bedrock")

        assert runtime["provider"] == "bedrock"
        assert runtime["region"] == "us-west-2"
        assert runtime["api_key"] == "aws-sdk"


# ---------------------------------------------------------------------------
# providers.py integration
# ---------------------------------------------------------------------------

class TestProvidersModule:
    """Verify bedrock is wired into hermes_cli/providers.py."""

    def test_bedrock_alias_in_providers(self):
        from hermes_cli.providers import ALIASES
        assert ALIASES.get("bedrock") is None  # "bedrock" IS the canonical name, not an alias
        assert ALIASES.get("aws") == "bedrock"
        assert ALIASES.get("aws-bedrock") == "bedrock"


    def test_determine_api_mode_from_bedrock_url(self):
        from hermes_cli.providers import determine_api_mode
        assert determine_api_mode(
            "unknown", "https://bedrock-runtime.us-east-1.amazonaws.com"
        ) == "bedrock_converse"



# ---------------------------------------------------------------------------
# Error classifier integration
# ---------------------------------------------------------------------------

class TestErrorClassifierBedrock:
    """Verify Bedrock error patterns are in the global error classifier."""

    def test_throttling_in_rate_limit_patterns(self):
        from agent.error_classifier import _RATE_LIMIT_PATTERNS
        assert "throttlingexception" in _RATE_LIMIT_PATTERNS

    def test_context_overflow_patterns(self):
        from agent.error_classifier import _CONTEXT_OVERFLOW_PATTERNS
        assert "input is too long" in _CONTEXT_OVERFLOW_PATTERNS


# ---------------------------------------------------------------------------
# pyproject.toml async boundary
# ---------------------------------------------------------------------------

class TestPackaging:
    """Verify disabled sync Bedrock dependencies are not published."""

    @staticmethod
    def _optional_dependencies():
        import tomllib
        from pathlib import Path

        content = (Path(__file__).parent.parent.parent / "pyproject.toml").read_text()
        return tomllib.loads(content)["project"]["optional-dependencies"]

    def test_bedrock_extra_is_native_async(self):
        extras = self._optional_dependencies()
        assert extras["bedrock"] == [
            "aiobotocore==3.8.0",
            "anthropic[bedrock]==0.87.0",
        ]

    def test_bedrock_is_not_eager_installed(self):
        import tomllib
        from pathlib import Path

        content = (Path(__file__).parent.parent.parent / "pyproject.toml").read_text()
        dependencies = tomllib.loads(content)["project"]["dependencies"]
        assert not any(
            dep.startswith(("boto3", "urllib3")) for dep in dependencies
        )


# ---------------------------------------------------------------------------
# Model ID dot preservation — regression for #11976
# ---------------------------------------------------------------------------
# AWS Bedrock inference-profile model IDs embed structural dots:
#
#   global.anthropic.claude-opus-4-7
#   us.anthropic.claude-sonnet-4-5-20250929-v1:0
#   apac.anthropic.claude-haiku-4-5
#
# ``agent.anthropic_adapter.normalize_model_name`` converts dots to hyphens
# unless the caller opts in via ``preserve_dots=True``.  Before this fix,
# ``AIAgent._anthropic_preserve_dots`` returned False for the ``bedrock``
# provider, so Claude-on-Bedrock requests went out with
# ``global-anthropic-claude-opus-4-7`` (all dots mangled to hyphens) and
# Bedrock rejected them with:
#
#   HTTP 400: The provided model identifier is invalid.
#
# The fix adds ``bedrock`` to the preserve-dots provider allowlist and
# ``bedrock-runtime.`` to the base-URL heuristic, mirroring the shape of
# the opencode-go fix for #5211 (commit f77be22c), which extended this
# same allowlist.


class TestBedrockPreserveDotsFlag:
    """``AIAgent._anthropic_preserve_dots`` must return True on Bedrock so
    inference-profile IDs survive the normalize step intact."""

    def test_bedrock_provider_preserves_dots(self):
        from types import SimpleNamespace
        agent = SimpleNamespace(provider="bedrock", base_url="")
        from run_agent import AIAgent
        assert AIAgent._anthropic_preserve_dots(agent) is True



    def test_non_bedrock_aws_url_does_not_preserve_dots(self):
        """Unrelated AWS endpoints (e.g. ``s3.us-east-1.amazonaws.com``)
        must not accidentally activate the dot-preservation heuristic —
        the heuristic is scoped to the ``bedrock-runtime.`` substring
        specifically."""
        from types import SimpleNamespace
        agent = SimpleNamespace(
            provider="custom",
            base_url="https://s3.us-east-1.amazonaws.com",
        )
        from run_agent import AIAgent
        assert AIAgent._anthropic_preserve_dots(agent) is False



class TestBedrockModelNameNormalization:
    """End-to-end: ``normalize_model_name`` + the preserve-dots flag
    reproduce the exact production request shape for each Bedrock model
    family, confirming the fix resolves the reporter's HTTP 400."""

    def test_global_anthropic_inference_profile_preserved(self):
        """The reporter's exact model ID."""
        from agent.anthropic_adapter import normalize_model_name
        assert normalize_model_name(
            "global.anthropic.claude-opus-4-7", preserve_dots=True
        ) == "global.anthropic.claude-opus-4-7"



    def test_bedrock_prefix_preserved_without_preserve_dots(self):
        """Bedrock inference profile IDs are auto-detected by prefix and
        always returned unmangled -- ``preserve_dots`` is irrelevant for
        these IDs because the dots are namespace separators, not version
        separators.  Regression for #12295."""
        from agent.anthropic_adapter import normalize_model_name
        assert normalize_model_name(
            "global.anthropic.claude-opus-4-7", preserve_dots=False
        ) == "global.anthropic.claude-opus-4-7"



class TestBedrockBuildAnthropicKwargsEndToEnd:
    """Integration: calling ``build_anthropic_kwargs`` with a Bedrock-
    shaped model ID and ``preserve_dots=True`` produces the unmangled
    model string in the outgoing kwargs — the exact body sent to the
    ``bedrock-runtime.`` endpoint.  This is the integration-level
    regression for the reporter's HTTP 400."""

    def test_bedrock_inference_profile_survives_build_kwargs(self):
        from agent.anthropic_adapter import build_anthropic_kwargs
        kwargs = build_anthropic_kwargs(
            model="global.anthropic.claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=1024,
            reasoning_config=None,
            preserve_dots=True,
        )
        assert kwargs["model"] == "global.anthropic.claude-opus-4-7", (
            "Bedrock inference-profile ID was mangled in build_anthropic_kwargs: "
            f"{kwargs['model']!r}"
        )

    def test_bedrock_model_preserved_without_preserve_dots(self):
        """Bedrock inference profile IDs survive ``build_anthropic_kwargs``
        even without ``preserve_dots=True`` -- the prefix auto-detection
        in ``normalize_model_name`` is the load-bearing piece.
        Regression for #12295."""
        from agent.anthropic_adapter import build_anthropic_kwargs
        kwargs = build_anthropic_kwargs(
            model="global.anthropic.claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=1024,
            reasoning_config=None,
            preserve_dots=False,
        )
        assert kwargs["model"] == "global.anthropic.claude-opus-4-7"


class TestBedrockModelIdDetection:
    """Tests for ``_is_bedrock_model_id`` and the auto-detection that
    makes ``normalize_model_name`` preserve dots for Bedrock IDs
    regardless of ``preserve_dots``.  Regression for #12295."""

    def test_bare_bedrock_id_detected(self):
        from agent.anthropic_adapter import _is_bedrock_model_id
        assert _is_bedrock_model_id("anthropic.claude-opus-4-7") is True






    def test_bare_bedrock_id_preserved_without_flag(self):
        """The primary bug from #12295: ``anthropic.claude-opus-4-7``
        sent to bedrock-mantle via auxiliary clients that don't pass
        ``preserve_dots=True``."""
        from agent.anthropic_adapter import normalize_model_name
        assert normalize_model_name(
            "anthropic.claude-opus-4-7", preserve_dots=False
        ) == "anthropic.claude-opus-4-7"




# ---------------------------------------------------------------------------
# auxiliary_client Bedrock resolution — fix for #13919
# ---------------------------------------------------------------------------
# Before the fix, resolve_provider_client("bedrock", ...) fell through to the
# "unhandled auth_type" warning and returned (None, None), breaking all
# auxiliary tasks (compression, memory, summarization) for Bedrock users.


class TestAuxiliaryClientBedrockResolution:
    """Verify resolve_provider_client handles Bedrock's aws_sdk auth type."""

    @pytest.mark.asyncio
    async def test_bedrock_returns_client_with_credentials(self, monkeypatch):
        """Auxiliary Bedrock returns the native async Anthropic adapter."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
        monkeypatch.setenv("AWS_REGION", "us-west-2")

        mock_anthropic_bedrock = MagicMock()
        with patch("agent.anthropic_adapter.build_anthropic_bedrock_client",
                   new=AsyncMock(return_value=mock_anthropic_bedrock)):
            from agent.auxiliary_client import resolve_provider_client
            client, model = await resolve_provider_client("bedrock", None)

        assert client._real_client is mock_anthropic_bedrock
        assert model == "anthropic.claude-haiku-4-5-20251001-v1:0"

    @pytest.mark.asyncio
    async def test_bedrock_returns_none_without_credentials(self, monkeypatch):
        with patch(
            "agent.bedrock_adapter.has_aws_credentials",
            new=AsyncMock(return_value=False),
        ):
            from agent.auxiliary_client import resolve_provider_client
            assert await resolve_provider_client("bedrock", None) == (None, None)




    @pytest.mark.asyncio
    async def test_bedrock_default_model_is_haiku(self, monkeypatch):
        """The native async path preserves Hermes's default Bedrock model."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")

        with patch("agent.anthropic_adapter.build_anthropic_bedrock_client",
                   new=AsyncMock(return_value=MagicMock())):
            from agent.auxiliary_client import resolve_provider_client
            _, model = await resolve_provider_client("bedrock", None)

        assert model == "anthropic.claude-haiku-4-5-20251001-v1:0"





    @pytest.mark.asyncio
    async def test_bedrock_converse_shim_awaits_native_transport(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIO...MPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")

        from agent.auxiliary_client import BedrockAuxiliaryClient

        client = BedrockAuxiliaryClient("us-east-1", "openai.gpt-oss-20b-1:0")
        response = SimpleNamespace(choices=[])
        with patch(
            "agent.bedrock_adapter.call_converse",
            new=AsyncMock(return_value=response),
        ) as mock_converse:
            result = await client.chat.completions.create(
                model="openai.gpt-oss-20b-1:0",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
        assert result is response
        mock_converse.assert_awaited_once()


class _BedrockClientContext:
    def __init__(self, client):
        self.client = client
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self.client

    async def __aexit__(self, *_args):
        self.exited = True
        return False


class TestAIAgentBedrockDispatch:
    @staticmethod
    def _agent():
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
        agent.api_mode = "bedrock_converse"
        agent.provider = "bedrock"
        agent._disable_streaming = False
        agent._interrupt_requested = False
        agent._fire_stream_delta = lambda _text: None
        agent._fire_tool_gen_started = lambda _name: None
        agent._fire_reasoning_delta = lambda _text: None
        return agent

    @pytest.mark.asyncio
    async def test_nonstream_request_is_awaited_and_client_is_closed(self, monkeypatch):
        agent = self._agent()
        client = SimpleNamespace(
            converse=AsyncMock(
                return_value={
                    "output": {
                        "message": {
                            "role": "assistant",
                            "content": [{"text": "native async"}],
                        }
                    },
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 2, "outputTokens": 3},
                }
            )
        )
        context = _BedrockClientContext(client)
        monkeypatch.setattr(
            "agent.bedrock_adapter._get_bedrock_runtime_client",
            lambda region: context,
        )

        response = await agent._execute_model_request(
            {
                "modelId": "amazon.nova-pro-v1:0",
                "messages": [{"role": "user", "content": [{"text": "hi"}]}],
                "__bedrock_converse__": True,
                "__bedrock_region__": "eu-west-1",
            }
        )

        assert response.choices[0].message.content == "native async"
        client.converse.assert_awaited_once_with(
            modelId="amazon.nova-pro-v1:0",
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
        )
        assert context.entered and context.exited

    @pytest.mark.asyncio
    async def test_stream_request_preserves_callbacks_and_closes_client(self, monkeypatch):
        agent = self._agent()
        text_deltas = []
        reasoning_deltas = []
        first_deltas = []
        agent._fire_stream_delta = text_deltas.append
        agent._fire_reasoning_delta = reasoning_deltas.append

        async def events():
            yield {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"reasoningContent": {"text": "think"}},
                }
            }
            yield {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"text": "answer"},
                }
            }
            yield {"contentBlockStop": {"contentBlockIndex": 0}}
            yield {"messageStop": {"stopReason": "end_turn"}}

        client = SimpleNamespace(
            converse_stream=AsyncMock(return_value={"stream": events()})
        )
        context = _BedrockClientContext(client)
        monkeypatch.setattr(
            "agent.bedrock_adapter._get_bedrock_runtime_client",
            lambda region: context,
        )

        response = await agent._execute_model_request(
            {
                "modelId": "amazon.nova-pro-v1:0",
                "messages": [],
                "__bedrock_region__": "us-east-2",
            },
            use_streaming=True,
            on_first_delta=lambda: first_deltas.append(True),
        )

        assert response.choices[0].message.content == "answer"
        assert response.choices[0].message.reasoning_content == "think"
        assert text_deltas == ["answer"]
        assert reasoning_deltas == ["think"]
        assert first_deltas == [True]
        assert context.entered and context.exited

    @pytest.mark.asyncio
    async def test_cancellation_closes_inflight_client_context(self, monkeypatch):
        agent = self._agent()
        started = asyncio.Event()

        async def converse(**_kwargs):
            started.set()
            await asyncio.Event().wait()

        context = _BedrockClientContext(SimpleNamespace(converse=converse))
        monkeypatch.setattr(
            "agent.bedrock_adapter._get_bedrock_runtime_client",
            lambda region: context,
        )

        task = asyncio.create_task(
            agent._execute_model_request(
                {
                    "modelId": "amazon.nova-pro-v1:0",
                    "messages": [],
                    "__bedrock_region__": "us-east-1",
                }
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert context.entered and context.exited
