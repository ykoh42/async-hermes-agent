"""Public export parity retained from Hermes v2026.8.3."""

from agent import agent_runtime_helpers, chat_completion_helpers


def test_agent_runtime_helpers_preserve_upstream_exports():
    assert agent_runtime_helpers.__all__ == [
        "convert_to_trajectory_format",
        "sanitize_tool_call_arguments",
        "repair_message_sequence",
        "strip_think_blocks",
        "recover_with_credential_pool",
        "try_recover_primary_transport",
        "drop_thinking_only_and_merge_users",
        "restore_primary_runtime",
        "extract_reasoning",
        "dump_api_request_debug",
        "prompt_caching_disabled_from_config",
        "blank_cache_policy_stub",
        "plan_cache_sections_for_destination",
        "anthropic_prompt_cache_policy",
        "create_openai_client",
        "switch_model",
        "invoke_tool",
        "repair_tool_call",
        "sanitize_api_messages",
        "looks_like_codex_intermediate_ack",
        "copy_reasoning_content_for_api",
        "cleanup_dead_connections",
        "extract_api_error_context",
        "apply_pending_steer_to_tool_results",
        "_iter_pool_sockets",
        "force_close_tcp_sockets",
    ]


def test_chat_completion_helpers_preserve_upstream_exports():
    assert chat_completion_helpers.__all__ == [
        "interruptible_api_call",
        "build_api_kwargs",
        "build_assistant_message",
        "try_activate_fallback",
        "handle_max_iterations",
        "cleanup_task_resources",
        "interruptible_streaming_api_call",
    ]
