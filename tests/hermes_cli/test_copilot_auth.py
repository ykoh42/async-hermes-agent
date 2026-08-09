"""Tests for hermes_cli.copilot_auth — Copilot token validation and resolution."""

import asyncio
import inspect
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

class TestTokenValidation:
    """Token type validation."""

    def test_classic_pat_rejected(self):
        from hermes_cli.copilot_auth import validate_copilot_token
        valid, msg = validate_copilot_token("ghp_abcdefghijklmnop1234")
        assert valid is False
        assert "Classic Personal Access Tokens" in msg
        assert "ghp_" in msg


class TestResolveToken:
    """Token resolution with env var priority."""


    @pytest.mark.asyncio
    async def test_gh_token_second_priority(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.setenv("GH_TOKEN", "gho_gh_second")
        monkeypatch.setenv("GITHUB_TOKEN", "gho_github_third")
        token, source = await resolve_copilot_token()
        assert token == "gho_gh_second"
        assert source == "GH_TOKEN"




    @pytest.mark.asyncio
    async def test_gh_cli_classic_pat_raises(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with patch(
            "hermes_cli.copilot_auth._try_gh_cli_token",
            new=AsyncMock(return_value="ghp_classic"),
        ):
            with pytest.raises(ValueError, match="classic PAT"):
                await resolve_copilot_token()

    @pytest.mark.asyncio
    async def test_invalid_env_var_skips_gh_cli_fallback(self, monkeypatch):
        """When an env var is set but holds an unsupported classic PAT,
        resolve_copilot_token must NOT fall back to ``gh auth token``.

        The user explicitly exported a token; silently substituting one
        from the gh CLI credential store is surprising and the subprocess
        call adds up to 5s of latency on Windows cold starts (#60800).
        Only fall back to the CLI when NO Copilot env var is set at all.
        """
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_classic_pat_nope")
        with patch("hermes_cli.copilot_auth._try_gh_cli_token") as mock_cli:
            token, source = await resolve_copilot_token()
        assert token == ""
        assert source == ""
        mock_cli.assert_not_called()

    @pytest.mark.asyncio
    async def test_gh_cli_cancellation_reaps_subprocess(self, monkeypatch):
        from hermes_cli import copilot_auth

        communicate_started = asyncio.Event()
        release_communicate = asyncio.Event()
        communicate_completed = asyncio.Event()
        process = Mock(returncode=None)

        async def communicate():
            communicate_started.set()
            await release_communicate.wait()
            communicate_completed.set()
            process.returncode = -9
            return b"", b""

        async def wait():
            return process.returncode

        process.communicate = communicate
        process.wait = wait
        process.kill = Mock()
        monkeypatch.setattr(
            copilot_auth,
            "_gh_cli_candidates",
            AsyncMock(return_value=["gh"]),
        )
        monkeypatch.setattr(
            copilot_auth.asyncio,
            "create_subprocess_exec",
            AsyncMock(return_value=process),
        )

        task = asyncio.create_task(copilot_auth._try_gh_cli_token())
        await communicate_started.wait()
        task.cancel()
        while not process.kill.called:
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)

        assert task.done() is False
        release_communicate.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        process.kill.assert_called_once()
        assert communicate_completed.is_set()

    @pytest.mark.asyncio
    async def test_all_env_vars_invalid_skips_gh_cli_fallback(self, monkeypatch):
        """All three env vars set to classic PATs → no gh CLI call."""
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "ghp_one")
        monkeypatch.setenv("GH_TOKEN", "ghp_two")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_three")
        with patch("hermes_cli.copilot_auth._try_gh_cli_token") as mock_cli:
            token, source = await resolve_copilot_token()
        assert token == ""
        assert source == ""
        mock_cli.assert_not_called()


class TestDeviceCodeLogin:
    @pytest.mark.asyncio
    async def test_device_code_flow_is_native_async(self, monkeypatch):
        from hermes_cli import copilot_auth

        responses = [
            {
                "device_code": "device",
                "user_code": "ABCD-EFGH",
                "verification_uri": "https://github.com/login/device",
                "interval": 1,
            },
            {"error": "authorization_pending"},
            {"access_token": "ghu_authorized"},
        ]

        class Client:
            def __init__(self, **_kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def post(self, url, **_kwargs):
                return httpx.Response(
                    200,
                    request=httpx.Request("POST", url),
                    json=responses.pop(0),
                )

        sleep = AsyncMock()
        monkeypatch.setattr(copilot_auth.httpx, "AsyncClient", Client)
        monkeypatch.setattr(copilot_auth.asyncio, "sleep", sleep)

        token = await copilot_auth.copilot_device_code_login(timeout_seconds=30)

        assert token == "ghu_authorized"
        assert sleep.await_count == 2
        assert inspect.iscoroutinefunction(copilot_auth.copilot_device_code_login)


class TestRequestHeaders:
    """Copilot API header generation."""

    def test_default_headers_include_openai_intent(self):
        from hermes_cli.copilot_auth import copilot_request_headers
        headers = copilot_request_headers()
        assert headers["Openai-Intent"] == "conversation-edits"
        assert headers["User-Agent"] == "HermesAgent/1.0"
        assert "Editor-Version" in headers


    def test_no_vision_header_by_default(self):
        from hermes_cli.copilot_auth import copilot_request_headers
        headers = copilot_request_headers()
        assert "Copilot-Vision-Request" not in headers


class TestCopilotDefaultHeaders:
    """The models.py copilot_default_headers uses copilot_auth."""


    def test_agent_turn_explicit(self):
        """Explicitly passing is_agent_turn=True sets x-initiator to 'agent'."""
        from hermes_cli.models import copilot_default_headers
        headers = copilot_default_headers(is_agent_turn=True)
        assert headers["x-initiator"] == "agent"

    def test_param_passthrough_both_values(self):
        """is_agent_turn param correctly maps to x-initiator for both True and False."""
        from hermes_cli.models import copilot_default_headers
        for is_agent, expected in [(True, "agent"), (False, "user")]:
            headers = copilot_default_headers(is_agent_turn=is_agent)
            assert headers["x-initiator"] == expected, (
                f"is_agent_turn={is_agent} should produce x-initiator={expected!r}, "
                f"got {headers['x-initiator']!r}"
            )


class TestApiModeSelection:
    """API mode selection matching opencode's shouldUseCopilotResponsesApi."""

    def test_gpt5_uses_responses(self):
        from hermes_cli.models import _should_use_copilot_responses_api
        assert _should_use_copilot_responses_api("gpt-5.4") is True
        assert _should_use_copilot_responses_api("gpt-5.4-mini") is True
        assert _should_use_copilot_responses_api("gpt-5.3-codex") is True
        assert _should_use_copilot_responses_api("gpt-5.2-codex") is True
        assert _should_use_copilot_responses_api("gpt-5.2") is True
        assert _should_use_copilot_responses_api("gpt-5.1-codex-max") is True

    def test_gpt5_mini_excluded(self):
        from hermes_cli.models import _should_use_copilot_responses_api
        assert _should_use_copilot_responses_api("gpt-5-mini") is False


class TestEnvVarOrder:
    """PROVIDER_REGISTRY has correct env var order."""

    def test_copilot_env_vars_include_copilot_github_token(self):
        from hermes_cli.auth import PROVIDER_REGISTRY
        copilot = PROVIDER_REGISTRY["copilot"]
        assert "COPILOT_GITHUB_TOKEN" in copilot.api_key_env_vars
        # COPILOT_GITHUB_TOKEN should be first
        assert copilot.api_key_env_vars[0] == "COPILOT_GITHUB_TOKEN"
