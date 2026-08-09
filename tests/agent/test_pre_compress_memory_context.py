"""Behavior contracts for memory-provider context in compression prompts."""

import json

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.context_compressor import ContextCompressor


def _make_compressor():
    compressor = ContextCompressor.__new__(ContextCompressor)
    compressor.protect_first_n = 2
    compressor.protect_last_n = 5
    # Set context_length BEFORE the derived budgets: its setter resets the
    # lazily-cached threshold/tail/summary budgets (#32221 lazy init), so
    # assigning it later would clear the explicit values below.
    compressor.context_length = 200_000
    compressor.threshold_percent = 0.80
    compressor.threshold_tokens = 160_000
    compressor.summary_target_ratio = 0.20
    compressor.tail_token_budget = 20_000
    compressor.max_summary_tokens = 10_000
    compressor.quiet_mode = True
    compressor.compression_count = 0
    compressor.last_prompt_tokens = 0
    compressor._previous_summary = None
    compressor._ineffective_compression_count = 0
    compressor._verify_compaction_cleared_threshold = False
    compressor._summary_failure_cooldown_until = 0.0
    compressor.summary_model = None
    compressor.model = "test-model"
    compressor.provider = "test"
    compressor.base_url = "http://localhost"
    compressor.api_key = ""
    compressor.api_mode = "chat_completions"
    return compressor


def _summary_response(content="## Goal\nCompaction complete."):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def _make_host_agent(memory_manager, compressor):
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        provider="openrouter",
        api_mode="chat_completions",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        session_db=None,
        session_id="test-session",
        skip_context_files=True,
        skip_memory=True,
    )
    agent._memory_manager = memory_manager
    agent.context_compressor = compressor
    agent.compression_in_place = False
    agent._compression_feasibility_checked = True
    agent._invalidate_system_prompt = AsyncMock()
    agent._build_system_prompt = AsyncMock(return_value="new-system-prompt")
    return agent


def _host_messages():
    return [{"role": "user", "content": f"message {i}"} for i in range(6)]


def _configure_engine_state(engine):
    engine.compression_count = 1
    engine.last_prompt_tokens = 0
    engine.last_completion_tokens = 0
    engine._last_summary_error = None
    engine._last_compress_aborted = False
    engine._last_summary_auth_failure = False
    engine._last_aux_model_failure_model = None
    engine._last_aux_model_failure_error = None
    engine._last_compression_made_progress = False
    engine._last_summary_fallback_used = False
    engine._last_feasibility_skip = False


def _memory_manager(provider_context):
    manager = MagicMock()
    manager.on_pre_compress = AsyncMock(return_value=provider_context)
    manager.on_session_end = AsyncMock()
    manager.shutdown_all = AsyncMock()
    return manager


def _mock_engine(side_effect):
    engine = MagicMock()
    engine.compress = AsyncMock(side_effect=side_effect)
    engine.on_session_end = AsyncMock()
    _configure_engine_state(engine)
    return engine


@pytest.mark.asyncio
async def test_memory_context_injected_into_initial_summary_prompt_with_focus():
    compressor = _make_compressor()
    turns = [
        {"role": "user", "content": "Fix the auth bug"},
        {"role": "assistant", "content": "Fixed the JWT expiry check."},
    ]
    prompts = []

    async def mock_call_llm(**kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        return _summary_response()

    with patch("agent.context_compressor.call_llm", mock_call_llm):
        await compressor._generate_summary(
            turns,
            focus_topic="authentication",
            memory_context="User uses JWT tokens with a one-hour expiry.",
        )

    assert len(prompts) == 1
    assert "MEMORY PROVIDER CONTEXT" in prompts[0]
    assert "User uses JWT tokens with a one-hour expiry." in prompts[0]
    assert 'FOCUS TOPIC: "authentication"' in prompts[0]


@pytest.mark.asyncio
async def test_memory_context_injected_into_iterative_summary_prompt():
    compressor = _make_compressor()
    compressor._previous_summary = "Previous checkpoint."
    turns = [
        {"role": "user", "content": "Continue the migration"},
        {"role": "assistant", "content": "Migration continued."},
    ]
    prompts = []

    async def mock_call_llm(**kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        return _summary_response("## Goal\nMigration updated.")

    with patch("agent.context_compressor.call_llm", mock_call_llm):
        await compressor._generate_summary(
            turns,
            memory_context="Checkpoint id: ctx-123",
        )

    assert len(prompts) == 1
    assert "PREVIOUS SUMMARY:\nPrevious checkpoint." in prompts[0]
    assert "MEMORY PROVIDER CONTEXT" in prompts[0]
    assert "Checkpoint id: ctx-123" in prompts[0]


@pytest.mark.asyncio
async def test_memory_context_is_strictly_redacted_before_summary_llm(monkeypatch):
    compressor = _make_compressor()
    prefix_secret = "sk-" + "b" * 30
    query_secret = "opaque-query-secret"
    userinfo_value = "opaque-userinfo-value"
    hyphen_client_secret = "HYPHEN_CLIENT_SECRET"
    hyphen_access_secret = "HYPHEN_ACCESS_SECRET"
    hyphen_api_secret = "HYPHEN_API_SECRET"
    encoded_hyphen_secret = "ENCODED_HYPHEN_SECRET"
    prompts = []

    async def mock_call_llm(**kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        return _summary_response()

    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    with patch("agent.context_compressor.call_llm", mock_call_llm):
        await compressor._generate_summary(
            [{"role": "user", "content": "Continue"}],
            memory_context=(
                f"api key: {prefix_secret}\n"
                f"callback: https://example.test/cb?token={query_secret}\n"
                f"endpoint: https://user:{userinfo_value}@example.test/private\n"
                f"hyphen-client: /resume?client-secret={hyphen_client_secret}\n"
                f"hyphen-access: /resume?Access-Token={hyphen_access_secret}\n"
                f"hyphen-api: /resume?api-key={hyphen_api_secret}\n"
                f"encoded-hyphen: /resume?client%2Dsecret={encoded_hyphen_secret}"
            ),
        )

    assert len(prompts) == 1
    prompt = prompts[0]
    assert prefix_secret not in prompt
    assert query_secret not in prompt
    assert userinfo_value not in prompt
    assert hyphen_client_secret not in prompt
    assert hyphen_access_secret not in prompt
    assert hyphen_api_secret not in prompt
    assert encoded_hyphen_secret not in prompt
    assert "token=***" in prompt
    assert "https://user:***@example.test/private" in prompt
    assert "client-secret=***" in prompt
    assert "Access-Token=***" in prompt
    assert "api-key=***" in prompt
    assert "client%2Dsecret=***" in prompt






@pytest.mark.asyncio
async def test_whitespace_memory_context_is_not_injected():
    compressor = _make_compressor()
    turns = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    prompts = []

    async def mock_call_llm(**kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        return _summary_response()

    with patch("agent.context_compressor.call_llm", mock_call_llm):
        await compressor._generate_summary(turns, memory_context="  \n\t ")

    assert len(prompts) == 1
    assert "MEMORY PROVIDER CONTEXT" not in prompts[0]


@pytest.mark.asyncio
async def test_on_pre_compress_result_reaches_engine_with_existing_options():
    manager = _memory_manager("Checkpoint id: ctx-orchestrator")
    received = {}

    async def capture_compress(
        incoming,
        current_tokens=None,
        focus_topic=None,
        force=False,
        memory_context="",
    ):
        received.update(
            current_tokens=current_tokens,
            focus_topic=focus_topic,
            force=force,
            memory_context=memory_context,
        )
        return [incoming[0], incoming[-1]]

    engine = _mock_engine(capture_compress)
    agent = _make_host_agent(manager, engine)
    messages = _host_messages()
    try:
        await agent._compress_context(
            messages,
            "sys",
            approx_tokens=100_000,
            focus_topic="checkpoint continuity",
            force=True,
        )
    finally:
        await agent.close()

    manager.on_pre_compress.assert_awaited_once_with(messages)
    assert received == {
        "current_tokens": 100_000,
        "focus_topic": "checkpoint continuity",
        "force": True,
        "memory_context": "Checkpoint id: ctx-orchestrator",
    }


@pytest.mark.asyncio
async def test_legacy_engine_receives_only_supported_compression_arguments():
    manager = _memory_manager("Checkpoint id: unsupported-by-legacy")
    calls = []

    class StrictLegacyEngine:
        async def compress(self, messages, current_tokens=None):
            calls.append(current_tokens)
            return [messages[0], messages[-1]]

    engine = StrictLegacyEngine()
    _configure_engine_state(engine)
    agent = _make_host_agent(manager, engine)
    try:
        compressed, _prompt = await agent._compress_context(
            _host_messages(),
            "sys",
            approx_tokens=100_000,
            focus_topic="unsupported focus",
            force=True,
        )
    finally:
        await agent.close()

    assert len(compressed) == 2
    assert calls == [100_000]


@pytest.mark.asyncio
async def test_provider_context_is_strictly_sanitized_before_plugin_engine(
    monkeypatch,
):
    prefix_secret = "sk-" + "a" * 30
    query_secret = "opaque-query-secret"
    userinfo_value = "opaque-userinfo-value"
    fragment_secret = "FRAG_SECRET"
    relative_secret = "REL_SECRET"
    encoded_key_secret = "ENC_SECRET"
    hyphen_client_secret = "HYPHEN_CLIENT_SECRET"
    hyphen_access_secret = "HYPHEN_ACCESS_SECRET"
    hyphen_api_secret = "HYPHEN_API_SECRET"
    encoded_hyphen_secret = "ENCODED_HYPHEN_SECRET"
    network_userinfo_secret = "NET_SECRET"
    manager = _memory_manager(
        f"api key: {prefix_secret}\n"
        f"callback: https://example.test/cb?access_token={query_secret}&state=ok\n"
        f"endpoint: https://user:{userinfo_value}@example.test/private\n"
        f"fragment: https://x.test/#access_token={fragment_secret}&view=public\n"
        f"relative: /resume?token={relative_secret}&view=public\n"
        f"encoded: https://x.test/cb?client%5Fsecret={encoded_key_secret}&view=public\n"
        f"hyphen-client: /resume?client-secret={hyphen_client_secret}&view=public\n"
        f"hyphen-access: /resume?Access-Token={hyphen_access_secret}&view=public\n"
        f"hyphen-api: /resume?api-key={hyphen_api_secret}&view=public\n"
        f"encoded-hyphen: /resume?client%2Dsecret={encoded_hyphen_secret}&view=public\n"
        f"network: //user:{network_userinfo_secret}@x.test/path"
    )
    received = []

    async def capture_compress(
        messages, current_tokens=None, memory_context="", **_kwargs
    ):
        received.append(memory_context)
        return [messages[0], messages[-1]]

    engine = _mock_engine(capture_compress)
    agent = _make_host_agent(manager, engine)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    try:
        await agent._compress_context(
            _host_messages(), "sys", approx_tokens=100_000
        )
    finally:
        await agent.close()

    assert len(received) == 1
    context = received[0]
    for secret in (
        prefix_secret,
        query_secret,
        userinfo_value,
        fragment_secret,
        relative_secret,
        encoded_key_secret,
        hyphen_client_secret,
        hyphen_access_secret,
        hyphen_api_secret,
        encoded_hyphen_secret,
        network_userinfo_secret,
    ):
        assert secret not in context
    assert "access_token=***" in context
    assert "https://user:***@example.test/private" in context
    assert "https://x.test/#access_token=***&view=public" in context
    assert "/resume?token=***&view=public" in context
    assert "client%5Fsecret=***&view=public" in context
    assert "client-secret=***&view=public" in context
    assert "Access-Token=***&view=public" in context
    assert "api-key=***&view=public" in context
    assert "client%2Dsecret=***&view=public" in context
    assert "//user:***@x.test/path" in context


@pytest.mark.asyncio
async def test_provider_context_is_bounded_before_plugin_engine():
    manager = _memory_manager(
        "HEAD-SENTINEL" + "x" * 8_000 + "TAIL-SENTINEL"
    )
    received = []

    async def capture_compress(
        messages, current_tokens=None, memory_context="", **_kwargs
    ):
        received.append(memory_context)
        return [messages[0], messages[-1]]

    engine = _mock_engine(capture_compress)
    agent = _make_host_agent(manager, engine)
    try:
        await agent._compress_context(
            _host_messages(), "sys", approx_tokens=100_000
        )
    finally:
        await agent.close()

    assert len(received) == 1
    context = received[0]
    assert len(context) <= 6_000
    assert context.startswith("HEAD-SENTINEL")
    assert context.endswith("TAIL-SENTINEL")
    assert "[memory provider context truncated]" in context


@pytest.mark.asyncio
async def test_internal_engine_type_error_propagates_after_one_call():
    manager = _memory_manager("Checkpoint id: ctx-typeerror")
    calls = []

    class BrokenEngine:
        async def compress(
            self,
            messages,
            current_tokens=None,
            focus_topic=None,
            force=False,
            memory_context="",
        ):
            calls.append(memory_context)
            raise TypeError("engine implementation bug")

    engine = BrokenEngine()
    _configure_engine_state(engine)
    agent = _make_host_agent(manager, engine)
    with pytest.raises(TypeError, match="engine implementation bug"):
        try:
            await agent._compress_context(
                _host_messages(), "sys", approx_tokens=100_000
            )
        finally:
            await agent.close()

    assert calls == ["Checkpoint id: ctx-typeerror"]

