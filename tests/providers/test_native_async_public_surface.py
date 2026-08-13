"""Pin the retained provider API after removal of duplicate async aliases."""

from __future__ import annotations

import inspect
import subprocess
import sys
import textwrap

import pytest

import providers
from agent import anthropic_adapter, auxiliary_client, chat_completion_helpers
from agent import codex_runtime, gemini_native_adapter
from hermes_cli import auth, runtime_provider
from providers.base import ProviderProfile


@pytest.mark.parametrize(
    "function",
    [
        providers.get_provider_profile,
        providers.list_providers,
        ProviderProfile.fetch_models,
        chat_completion_helpers.direct_api_call,
        chat_completion_helpers.interruptible_api_call,
        chat_completion_helpers.interruptible_streaming_api_call,
        chat_completion_helpers.try_activate_fallback,
        chat_completion_helpers.cleanup_task_resources,
        anthropic_adapter.build_anthropic_client,
        anthropic_adapter.build_anthropic_bedrock_client,
        anthropic_adapter.resolve_anthropic_token,
        anthropic_adapter.create_anthropic_message,
        codex_runtime.run_codex_stream,
        codex_runtime.run_codex_app_server_turn,
        codex_runtime.run_codex_create_stream_fallback,
        auxiliary_client.get_text_auxiliary_client,
        auxiliary_client.resolve_provider_client,
        auxiliary_client.call_llm,
        auxiliary_client.shutdown_cached_clients,
        runtime_provider.resolve_requested_provider,
        runtime_provider.resolve_runtime_provider,
        auth.get_provider_auth_state,
        auth.resolve_provider,
        auth.resolve_api_key_provider_credentials,
        auth.resolve_nous_runtime_credentials,
        auth.resolve_codex_runtime_credentials,
        auth.resolve_xai_oauth_runtime_credentials,
        auth.resolve_qwen_runtime_credentials,
        auth.resolve_minimax_oauth_runtime_credentials,
        auth.resolve_external_process_provider_credentials,
    ],
)
def test_retained_io_surface_is_one_same_name_coroutine(function) -> None:
    assert inspect.iscoroutinefunction(function), function.__qualname__


def test_duplicate_async_aliases_and_wrapper_clients_stay_removed() -> None:
    for name in (
        "async_call_llm",
        "get_async_text_auxiliary_client",
        "AsyncCodexAuxiliaryClient",
        "AsyncAnthropicAuxiliaryClient",
        "AsyncBedrockAuxiliaryClient",
        "neuter_async_httpx_del",
        "cleanup_stale_async_clients",
    ):
        assert not hasattr(auxiliary_client, name)
    assert not hasattr(gemini_native_adapter, "AsyncGeminiNativeClient")


def test_retained_public_signatures_keep_upstream_arguments() -> None:
    assert str(inspect.signature(providers.get_provider_profile)) == (
        "(name: 'str') -> 'ProviderProfile | None'"
    )
    assert str(inspect.signature(ProviderProfile.fetch_models)) == (
        "(self, *, api_key: 'str | None' = None, base_url: 'str | None' = None, "
        "timeout: 'float' = 8.0) -> 'list[str] | None'"
    )
    assert str(inspect.signature(auxiliary_client.get_text_auxiliary_client)) == (
        "(task: 'str' = '', *, main_runtime: 'dict[str, Any] | None' = None) "
        "-> 'tuple[OpenAI | None, str | None]'"
    )
    assert str(inspect.signature(auth.resolve_nous_runtime_credentials)) == (
        "(*, timeout_seconds: 'float' = 15.0, insecure: 'bool | None' = None, "
        "ca_bundle: 'str | None' = None, force_refresh: 'bool' = False) "
        "-> 'dict[str, Any]'"
    )


def test_cold_first_provider_discovery_has_no_blocking_import_fallback() -> None:
    script = textwrap.dedent(
        """
        import asyncio
        import os
        import sys
        import traceback

        from blockbuster import BlockBuster
        import providers

        async def main():
            providers._REGISTRY.clear()
            providers._ALIASES.clear()
            providers._PROVIDER_LIST_CACHE = None
            providers._discovered = False
            for name in tuple(sys.modules):
                if name.startswith((
                    "plugins.model_providers",
                    "_hermes_user_provider",
                )):
                    sys.modules.pop(name, None)
            blocker = BlockBuster()
            blocker.activate()
            try:
                profiles = await providers.list_providers()
                print(f"DISCOVERED={len(profiles)}")
            except BaseException:
                traceback.print_exc()
                return False
            finally:
                blocker.deactivate()
            return True

        succeeded = asyncio.run(main())
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if succeeded else 1)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "DISCOVERED=" in completed.stdout
