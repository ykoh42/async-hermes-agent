"""Tests for trajectory_compressor.py — config, metrics, and compression logic."""

import base64
import asyncio
import importlib
import io
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

import trajectory_compressor as trajectory_compressor_module
from trajectory_compressor import (
    CompressionConfig,
    TrajectoryMetrics,
    AggregateMetrics,
    TrajectoryCompressor,
)


def _minimal_kimi_assets():
    model = b"\n".join(
        base64.b64encode(bytes([value])) + b" " + str(value).encode("ascii")
        for value in range(256)
    )
    config = json.dumps(
        {
            "added_tokens_decoder": {
                "256": {"content": "[BOS]"},
                "257": {"content": "[EOS]"},
                "258": {"content": "<|im_end|>"},
            }
        }
    ).encode("utf-8")
    return config, model


def _minimal_fast_tokenizer_assets():
    tokenizer = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "[BOS]": 1, "hello": 2, "world": 3},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A",
        special_tokens=[("[BOS]", 1)],
    )
    return b"{}", tokenizer.to_str().encode("utf-8")


def _minimal_sentencepiece_model(*, model_type="unigram"):
    import sentencepiece as spm

    destination = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(
            [
                "hello world",
                "hello native async tokenizer",
                "world of agent trajectories",
            ]
        ),
        model_writer=destination,
        model_type=model_type,
        vocab_size=24,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=-1,
        hard_vocab_limit=False,
    )
    return destination.getvalue()


@pytest.mark.asyncio
async def test_custom_tokenizer_falls_back_to_sentencepiece_on_missing_json(
    monkeypatch,
):
    missing_request = httpx.Request(
        "GET", "https://huggingface.co/org/model/resolve/main/tokenizer.json"
    )
    missing_response = httpx.Response(404, request=missing_request)
    missing = httpx.HTTPStatusError(
        "not found",
        request=missing_request,
        response=missing_response,
    )
    model_bytes = b"serialized-sentencepiece"
    load = AsyncMock(
        side_effect=[
            missing,
            {
                "tokenizer_config.json": b'{"tokenizer_class":"LlamaTokenizer"}',
                "tokenizer.model": model_bytes,
            },
        ]
    )
    monkeypatch.setattr(
        trajectory_compressor_module, "_load_hf_tokenizer_assets", load
    )

    kind, config_bytes, loaded_model = (
        await trajectory_compressor_module._load_custom_tokenizer_assets(
            "org/model"
        )
    )

    assert kind == "sentencepiece"
    assert json.loads(config_bytes) == {"tokenizer_class": "LlamaTokenizer"}
    assert loaded_model == model_bytes
    assert load.await_args_list == [
        (("org/model", ("tokenizer_config.json", "tokenizer.json")),),
        (("org/model", ("tokenizer_config.json", "tokenizer.model")),),
    ]


@pytest.mark.asyncio
async def test_custom_tokenizer_fallback_does_not_hide_auth_errors(monkeypatch):
    request = httpx.Request(
        "GET", "https://huggingface.co/private/model/resolve/main/tokenizer.json"
    )
    response = httpx.Response(401, request=request)
    unauthorized = httpx.HTTPStatusError(
        "unauthorized",
        request=request,
        response=response,
    )
    load = AsyncMock(side_effect=unauthorized)
    monkeypatch.setattr(
        trajectory_compressor_module, "_load_hf_tokenizer_assets", load
    )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await trajectory_compressor_module._load_custom_tokenizer_assets(
            "private/model"
        )

    assert exc_info.value is unauthorized
    assert load.await_count == 1


def test_sentencepiece_llama_encode_adds_upstream_bos_only():
    import sentencepiece as spm

    model_bytes = _minimal_sentencepiece_model(model_type="bpe")
    processor = spm.SentencePieceProcessor()
    assert processor.LoadFromSerializedProto(model_bytes)
    tokenizer = trajectory_compressor_module._build_sentencepiece_tokenizer(
        json.dumps(
            {
                "tokenizer_class": "LlamaTokenizer",
                "add_bos_token": True,
                "add_eos_token": False,
            }
        ).encode(),
        model_bytes,
    )

    assert tokenizer.encode("hello world") == [
        processor.bos_id(),
        *processor.encode("hello world", out_type=int),
    ]


def test_sentencepiece_llama_preserves_repeated_spaces():
    model_bytes = _minimal_sentencepiece_model(model_type="bpe")
    tokenizer = trajectory_compressor_module._build_sentencepiece_tokenizer(
        b'{"tokenizer_class":"LlamaTokenizer","add_bos_token":true}',
        model_bytes,
    )

    single_spaces = tokenizer.encode("hello world")
    repeated_spaces = tokenizer.encode("  hello  world")

    assert repeated_spaces != single_spaces
    assert len(repeated_spaces) > len(single_spaces)


def test_sentencepiece_t5_encode_adds_upstream_eos_only():
    import sentencepiece as spm

    model_bytes = _minimal_sentencepiece_model()
    processor = spm.SentencePieceProcessor()
    assert processor.LoadFromSerializedProto(model_bytes)
    tokenizer = trajectory_compressor_module._build_sentencepiece_tokenizer(
        json.dumps(
            {
                "tokenizer_class": "T5Tokenizer",
                "eos_token": "</s>",
                "extra_ids": 100,
            }
        ).encode(),
        model_bytes,
    )

    assert tokenizer.encode("hello world") == [
        *processor.encode("hello world", out_type=int),
        processor.eos_id(),
    ]


def test_sentencepiece_unknown_family_fails_instead_of_guessing():
    model_bytes = _minimal_sentencepiece_model()

    with pytest.raises(ValueError, match="unsupported SentencePiece tokenizer"):
        trajectory_compressor_module._build_sentencepiece_tokenizer(
            b'{"tokenizer_class":"UnknownTokenizer"}',
            model_bytes,
        )


@pytest.mark.asyncio
async def test_kimi_tokenizer_assets_use_conditional_async_cache(
    tmp_path, monkeypatch
):
    config_bytes, model_bytes = _minimal_kimi_assets()
    etags = {
        "tokenizer_config.json": '"config-v1"',
        "tiktoken.model": '"model-v1"',
    }
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        name = request.url.path.rsplit("/", 1)[-1]
        if request.headers.get("if-none-match") == etags[name]:
            return httpx.Response(304, request=request)
        content = config_bytes if name == "tokenizer_config.json" else model_bytes
        return httpx.Response(
            200,
            content=content,
            headers={"etag": etags[name]},
            request=request,
        )

    real_async_client = httpx.AsyncClient

    def client_factory(**kwargs):
        return real_async_client(
            transport=httpx.MockTransport(handler),
            follow_redirects=kwargs.get("follow_redirects", False),
            timeout=kwargs.get("timeout"),
            headers=kwargs.get("headers"),
        )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        trajectory_compressor_module.httpx,
        "AsyncClient",
        client_factory,
    )

    assert await trajectory_compressor_module._load_kimi_tokenizer_assets() == (
        config_bytes,
        model_bytes,
    )
    assert await trajectory_compressor_module._load_kimi_tokenizer_assets() == (
        config_bytes,
        model_bytes,
    )

    assert len(requests) == 4
    assert all("if-none-match" not in request.headers for request in requests[:2])
    assert [request.headers.get("if-none-match") for request in requests[2:]] == [
        etags["tokenizer_config.json"],
        etags["tiktoken.model"],
    ]


@pytest.mark.asyncio
async def test_kimi_tokenizer_offline_cache_miss_fails_without_network(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    client = MagicMock(side_effect=AssertionError("network must not be used"))
    monkeypatch.setattr(trajectory_compressor_module.httpx, "AsyncClient", client)

    with pytest.raises(RuntimeError, match="HF_HUB_OFFLINE"):
        await trajectory_compressor_module._load_kimi_tokenizer_assets()

    client.assert_not_called()


@pytest.mark.asyncio
async def test_custom_fast_tokenizer_loads_local_directory_without_network(
    tmp_path, monkeypatch
):
    config_bytes, tokenizer_bytes = _minimal_fast_tokenizer_assets()
    tokenizer_directory = tmp_path / "custom-tokenizer"
    tokenizer_directory.mkdir()
    (tokenizer_directory / "tokenizer_config.json").write_bytes(config_bytes)
    (tokenizer_directory / "tokenizer.json").write_bytes(tokenizer_bytes)
    client = MagicMock(side_effect=AssertionError("network must not be used"))
    monkeypatch.setattr(trajectory_compressor_module.httpx, "AsyncClient", client)

    with patch.object(TrajectoryCompressor, "_init_summarizer"):
        compressor = TrajectoryCompressor(
            CompressionConfig(
                tokenizer_name=str(tokenizer_directory),
                trust_remote_code=False,
            )
        )
        assert await compressor.count_tokens("hello world") == 3

    client.assert_not_called()


@pytest.mark.asyncio
async def test_custom_sentencepiece_tokenizer_loads_local_directory(
    tmp_path, monkeypatch
):
    model_bytes = _minimal_sentencepiece_model()
    tokenizer_directory = tmp_path / "custom-sentencepiece"
    tokenizer_directory.mkdir()
    (tokenizer_directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "T5Tokenizer",
                "eos_token": "</s>",
                "extra_ids": 100,
            }
        )
    )
    (tokenizer_directory / "spiece.model").write_bytes(model_bytes)
    client = MagicMock(side_effect=AssertionError("network must not be used"))
    monkeypatch.setattr(trajectory_compressor_module.httpx, "AsyncClient", client)

    with patch.object(TrajectoryCompressor, "_init_summarizer"):
        compressor = TrajectoryCompressor(
            CompressionConfig(
                tokenizer_name=str(tokenizer_directory),
                trust_remote_code=False,
            )
        )
        expected = trajectory_compressor_module._build_sentencepiece_tokenizer(
            (tokenizer_directory / "tokenizer_config.json").read_bytes(),
            model_bytes,
        )
        assert await compressor.count_tokens("hello world") == len(
            expected.encode("hello world")
        )

    client.assert_not_called()


@pytest.mark.asyncio
async def test_custom_tokenizer_rejects_invalid_hub_repository_before_network(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    client = MagicMock(side_effect=AssertionError("network must not be used"))
    monkeypatch.setattr(trajectory_compressor_module.httpx, "AsyncClient", client)

    with pytest.raises(ValueError, match="Invalid Hugging Face"):
        await trajectory_compressor_module._load_fast_tokenizer_assets(
            "https://attacker.invalid/tokenizer"
        )

    client.assert_not_called()


@pytest.mark.asyncio
async def test_generate_summary_kimi_omits_temperature():
    """Kimi models should have temperature omitted — server manages it."""
    config = CompressionConfig(
        summarization_model="kimi-for-coding",
        temperature=0.3,
        summary_target_tokens=100,
        max_retries=1,
    )
    compressor = TrajectoryCompressor.__new__(TrajectoryCompressor)
    compressor.config = config
    compressor.logger = MagicMock()
    compressor._use_call_llm = False
    async_client = MagicMock()
    async_client.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="[CONTEXT SUMMARY]: summary"))]
    ))
    compressor._get_client = MagicMock(return_value=async_client)

    metrics = TrajectoryMetrics()
    result = await compressor._generate_summary("tool output", metrics)

    assert result.startswith("[CONTEXT SUMMARY]:")
    assert "temperature" not in async_client.chat.completions.create.call_args.kwargs






# ---------------------------------------------------------------------------
# CompressionConfig
# ---------------------------------------------------------------------------


class TestCompressionConfig:
    def test_defaults(self):
        config = CompressionConfig()
        assert config.target_max_tokens == 15250
        assert config.summary_target_tokens == 750
        assert config.protect_last_n_turns == 4
        assert config.skip_under_target is True

    @pytest.mark.asyncio
    async def test_from_yaml(self, tmp_path):
        yaml_content = """\
tokenizer:
  name: custom-tokenizer
  trust_remote_code: false
compression:
  target_max_tokens: 10000
  summary_target_tokens: 500
protected_turns:
  first_system: true
  first_human: false
  last_n_turns: 6
summarization:
  model: gpt-4
  temperature: 0.5
  max_retries: 5
output:
  add_summary_notice: false
  output_suffix: _short
processing:
  num_workers: 8
  max_concurrent_requests: 100
  skip_under_target: false
  save_over_limit: false
metrics:
  enabled: false
  per_trajectory: false
  output_file: my_metrics.json
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)
        config = await CompressionConfig.from_yaml(str(yaml_file))
        assert config.tokenizer_name == "custom-tokenizer"
        assert config.trust_remote_code is False
        assert config.target_max_tokens == 10000
        assert config.summary_target_tokens == 500
        assert config.protect_first_human is False
        assert config.protect_last_n_turns == 6
        assert config.summarization_model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_retries == 5
        assert config.add_summary_notice is False
        assert config.output_suffix == "_short"
        assert config.num_workers == 8
        assert config.max_concurrent_requests == 100
        assert config.skip_under_target is False
        assert config.save_over_limit is False
        assert config.metrics_enabled is False
        assert config.metrics_output_file == "my_metrics.json"




# ---------------------------------------------------------------------------
# TrajectoryMetrics
# ---------------------------------------------------------------------------


class TestTrajectoryMetrics:
    def test_to_dict(self):
        m = TrajectoryMetrics()
        m.original_tokens = 10000
        m.compressed_tokens = 5000
        m.tokens_saved = 5000
        m.compression_ratio = 0.5
        m.original_turns = 20
        m.compressed_turns = 10
        m.turns_removed = 10
        m.was_compressed = True
        d = m.to_dict()
        assert d["original_tokens"] == 10000
        assert d["compressed_tokens"] == 5000
        assert d["compression_ratio"] == 0.5
        assert d["was_compressed"] is True
        assert d["compression_region"]["start_idx"] == -1



# ---------------------------------------------------------------------------
# AggregateMetrics
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    def test_empty_to_dict(self):
        agg = AggregateMetrics()
        d = agg.to_dict()
        assert d["summary"]["total_trajectories"] == 0
        assert d["averages"]["avg_compression_ratio"] == 1.0
        assert d["averages"]["avg_tokens_saved_per_compressed"] == 0

    def test_add_compressed_trajectory(self):
        agg = AggregateMetrics()
        m = TrajectoryMetrics()
        m.original_tokens = 20000
        m.compressed_tokens = 10000
        m.tokens_saved = 10000
        m.compression_ratio = 0.5
        m.original_turns = 30
        m.compressed_turns = 15
        m.turns_removed = 15
        m.was_compressed = True
        agg.add_trajectory_metrics(m)
        assert agg.total_trajectories == 1
        assert agg.trajectories_compressed == 1
        assert agg.total_tokens_saved == 10000
        assert len(agg.compression_ratios) == 1






# ---------------------------------------------------------------------------
# TrajectoryCompressor._find_protected_indices
# ---------------------------------------------------------------------------


def _make_compressor(config=None):
    """Create a TrajectoryCompressor with mocked tokenizer and summarizer."""
    if config is None:
        config = CompressionConfig()
    with patch.object(TrajectoryCompressor, '_init_summarizer'):
        compressor = TrajectoryCompressor(config)
    compressor._summarizer_initialized = True
    # Provide a simple token counter for tests (1 token per 4 chars)
    compressor.tokenizer = MagicMock()
    compressor.tokenizer.encode = lambda text: [0] * (len(text) // 4)
    return compressor


class TestFindProtectedIndices:
    def test_basic_trajectory(self):
        tc = _make_compressor()
        trajectory = [
            {"from": "system", "value": "You are an agent."},
            {"from": "human", "value": "Do something."},
            {"from": "gpt", "value": "I will use a tool."},
            {"from": "tool", "value": "Tool result."},
            {"from": "gpt", "value": "More work."},
            {"from": "tool", "value": "Another result."},
            {"from": "gpt", "value": "Work continues."},
            {"from": "tool", "value": "Result 3."},
            {"from": "gpt", "value": "Done."},
            {"from": "human", "value": "Thanks."},
        ]
        protected, start, end = tc._find_protected_indices(trajectory)
        # First system (0), human (1), gpt (2), tool (3) are protected
        assert 0 in protected
        assert 1 in protected
        assert 2 in protected
        assert 3 in protected
        # Last 4 turns (6,7,8,9) are protected
        assert 6 in protected
        assert 7 in protected
        assert 8 in protected
        assert 9 in protected
        # Compressible region should be between head and tail
        assert start >= 4
        assert end <= 6



    def test_no_system_turn(self):
        tc = _make_compressor()
        trajectory = [
            {"from": "human", "value": "hi"},
            {"from": "gpt", "value": "hello"},
            {"from": "tool", "value": "data"},
            {"from": "gpt", "value": "result"},
            {"from": "human", "value": "thanks"},
        ]
        protected, start, end = tc._find_protected_indices(trajectory)
        assert 0 in protected  # first human

    def test_disable_protect_first_system(self):
        config = CompressionConfig()
        config.protect_first_system = False
        tc = _make_compressor(config)
        trajectory = [
            {"from": "system", "value": "sys"},
            {"from": "human", "value": "q"},
            {"from": "gpt", "value": "a"},
            {"from": "tool", "value": "r"},
            {"from": "gpt", "value": "b"},
            {"from": "tool", "value": "r2"},
            {"from": "gpt", "value": "c"},
            {"from": "tool", "value": "r3"},
        ]
        protected, _, _ = tc._find_protected_indices(trajectory)
        assert 0 not in protected  # system not protected


# ---------------------------------------------------------------------------
# TrajectoryCompressor._extract_turn_content_for_summary
# ---------------------------------------------------------------------------


class TestExtractTurnContent:
    def test_basic_extraction(self):
        tc = _make_compressor()
        trajectory = [
            {"from": "gpt", "value": "I will search."},
            {"from": "tool", "value": "Search result: found it."},
            {"from": "gpt", "value": "Great, done."},
        ]
        content = tc._extract_turn_content_for_summary(trajectory, 0, 2)
        assert "[Turn 0 - GPT]" in content
        assert "I will search." in content
        assert "[Turn 1 - TOOL]" in content
        assert "Search result: found it." in content
        # Turn 2 should NOT be included (end is exclusive)
        assert "[Turn 2" not in content

    def test_long_content_truncated(self):
        tc = _make_compressor()
        trajectory = [
            {"from": "tool", "value": "x" * 5000},
        ]
        content = tc._extract_turn_content_for_summary(trajectory, 0, 1)
        assert "...[truncated]..." in content
        assert len(content) < 5000

    def test_empty_range(self):
        tc = _make_compressor()
        trajectory = [{"from": "gpt", "value": "hello"}]
        content = tc._extract_turn_content_for_summary(trajectory, 0, 0)
        assert content == ""


# ---------------------------------------------------------------------------
# TrajectoryCompressor.count_tokens / count_trajectory_tokens
# ---------------------------------------------------------------------------


class TestTokenCounting:

    @pytest.mark.asyncio
    async def test_default_tokenizer_is_loaded_lazily_once(self):
        config_bytes, model_bytes = _minimal_kimi_assets()
        with (
            patch.object(TrajectoryCompressor, "_init_summarizer"),
            patch(
                "trajectory_compressor._load_kimi_tokenizer_assets",
                new=AsyncMock(return_value=(config_bytes, model_bytes)),
            ) as load_assets,
        ):
            tc = TrajectoryCompressor(CompressionConfig())

            assert tc.tokenizer is None
            counts = await asyncio.gather(
                tc.count_tokens("<|im_end|>"),
                tc.count_tokens("<|im_end|>"),
            )
            assert counts == [1, 1]

        load_assets.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_custom_fast_tokenizer_uses_native_async_assets(self):
        config_bytes, tokenizer_bytes = _minimal_fast_tokenizer_assets()
        with (
            patch.object(TrajectoryCompressor, "_init_summarizer"),
            patch(
                "trajectory_compressor._load_fast_tokenizer_assets",
                new=AsyncMock(return_value=(config_bytes, tokenizer_bytes)),
            ) as load_assets,
        ):
            tc = TrajectoryCompressor(
                CompressionConfig(tokenizer_name="org/custom-tokenizer")
            )

            assert await tc.count_tokens("hello world") == 3

        load_assets.assert_awaited_once_with("org/custom-tokenizer")

    @pytest.mark.asyncio
    async def test_custom_python_tokenizer_fails_instead_of_blocking(self):
        _, tokenizer_bytes = _minimal_fast_tokenizer_assets()
        config_bytes = json.dumps(
            {"auto_map": {"AutoTokenizer": ["tokenization_custom.Custom", None]}}
        ).encode("utf-8")
        with (
            patch.object(TrajectoryCompressor, "_init_summarizer"),
            patch(
                "trajectory_compressor._load_fast_tokenizer_assets",
                new=AsyncMock(return_value=(config_bytes, tokenizer_bytes)),
            ),
        ):
            tc = TrajectoryCompressor(
                CompressionConfig(tokenizer_name="org/custom-tokenizer")
            )

            with pytest.raises(RuntimeError, match="requires Python remote code"):
                await tc.count_tokens("hello")

    @pytest.mark.asyncio
    async def test_tokenizer_initialization_preserves_cancellation_and_retries(self):
        config_bytes, model_bytes = _minimal_kimi_assets()
        started = asyncio.Event()
        calls = 0

        async def load_assets():
            nonlocal calls
            calls += 1
            if calls == 1:
                started.set()
                await asyncio.Event().wait()
            return config_bytes, model_bytes

        with (
            patch.object(TrajectoryCompressor, "_init_summarizer"),
            patch(
                "trajectory_compressor._load_kimi_tokenizer_assets",
                side_effect=load_assets,
            ),
        ):
            tc = TrajectoryCompressor(CompressionConfig())
            first = asyncio.create_task(tc.count_tokens("hello"))
            await started.wait()
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first

            assert await tc.count_tokens("<|im_end|>") == 1

        assert calls == 2

    @pytest.mark.asyncio
    async def test_count_tokens_basic(self):
        tc = _make_compressor()
        # Our mock: 1 token per 4 chars
        assert await tc.count_tokens("12345678") == 2

    @pytest.mark.asyncio
    async def test_count_trajectory_tokens(self):
        tc = _make_compressor()
        trajectory = [
            {"from": "system", "value": "12345678"},   # 2 tokens
            {"from": "human", "value": "1234567890ab"}, # 3 tokens
        ]
        assert await tc.count_trajectory_tokens(trajectory) == 5


    @pytest.mark.asyncio
    async def test_count_tokens_fallback_on_error(self):
        tc = _make_compressor()
        tc.tokenizer.encode = MagicMock(side_effect=Exception("fail"))
        # Should fallback to len(text) // 4
        assert await tc.count_tokens("12345678") == 2


class TestGenerateSummary:
    @pytest.mark.asyncio
    async def test_generate_summary_handles_none_content(self):
        tc = _make_compressor()
        async_client = MagicMock()
        async_client.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
        ))
        tc._get_client = MagicMock(return_value=async_client)
        metrics = TrajectoryMetrics()

        summary = await tc._generate_summary("Turn content", metrics)

        assert summary == "[CONTEXT SUMMARY]:"



# ---------------------------------------------------------------------------
# TrajectoryCompressor — compression boundary must not split tool pairs
# ---------------------------------------------------------------------------


def _gpt_with_tool_call(label, tokens):
    """A 'gpt' turn carrying a <tool_call> marker, padded to ~`tokens` tokens."""
    body = f"<tool_call>\n{{\"name\": \"{label}\"}}\n</tool_call>"
    pad = max(0, tokens * 4 - len(body))
    return {"from": "gpt", "value": body + "x" * pad}


def _tool_response(label, tokens):
    """A 'tool' turn carrying a <tool_response> marker, padded to ~`tokens` tokens."""
    body = f"<tool_response>\n{{\"name\": \"{label}\"}}\n</tool_response>"
    pad = max(0, tokens * 4 - len(body))
    return {"from": "tool", "value": body + "x" * pad}


def _count_marker(trajectory, marker):
    return sum(turn["value"].count(marker) for turn in trajectory)


def _paired_trajectory():
    """A 10-turn trajectory of gpt/tool pairs with one oversized middle gpt turn.

    Layout (index): system, human, gpt#0, tool#0, gpt#1(big), tool#1, gpt#2,
    tool#2, gpt(final), human. With ``protect_last_n_turns=2`` the compressible
    region is [4, 8) and the oversized gpt#1 at index 4 is large enough that the
    token-accumulation boundary stops at index 5 — i.e. between gpt#1's
    <tool_call> and tool#1's <tool_response>.
    """
    return [
        {"from": "system", "value": "You are an agent. " * 4},
        {"from": "human", "value": "Please do the task. " * 4},
        _gpt_with_tool_call("a", 12),
        _tool_response("a", 12),
        _gpt_with_tool_call("b", 400),  # oversized — forces a mid-pair boundary
        _tool_response("b", 12),
        _gpt_with_tool_call("c", 12),
        _tool_response("c", 12),
        {"from": "gpt", "value": "<think>\n</think>\nAll done."},
        {"from": "human", "value": "Thanks!"},
    ]


async def _target_that_splits_after_index_4(tc, trajectory):
    """Pick a target so token accumulation breaks right after index 4 (a gpt)."""
    turn_tokens = await tc.count_turn_tokens(trajectory)
    total = sum(turn_tokens)
    # threshold == turn_tokens[4] makes the loop break at compress_until = 5,
    # which lands on the tool turn paired with gpt#1.
    return total - turn_tokens[4] + tc.config.summary_target_tokens


class TestCompressionToolPairIntegrity:
    def _config(self):
        config = CompressionConfig()
        config.protect_last_n_turns = 2
        config.summary_target_tokens = 4
        return config

    @pytest.mark.asyncio
    async def test_compression_does_not_orphan_tool_markers(self):
        tc = _make_compressor(self._config())
        tc._generate_summary = AsyncMock(
            return_value="[CONTEXT SUMMARY]: middle turns summarized."
        )
        trajectory = _paired_trajectory()
        tc.config.target_max_tokens = await _target_that_splits_after_index_4(
            tc, trajectory
        )

        compressed, metrics = await tc.compress_trajectory(trajectory)

        assert metrics.was_compressed
        # Every <tool_call> must keep its matching <tool_response>.
        assert _count_marker(compressed, "<tool_call>") == _count_marker(
            compressed, "<tool_response>"
        )
        # A kept 'tool' turn must always immediately follow its 'gpt' turn —
        # never the inserted summary (a 'human' turn) or another 'tool' turn.
        for i, turn in enumerate(compressed):
            if turn.get("from") == "tool":
                assert i > 0 and compressed[i - 1].get("from") == "gpt"


    def test_snap_boundary_skips_tool_turn_forward(self):
        tc = _make_compressor()
        trajectory = _paired_trajectory()
        # Index 5 is a 'tool' turn; the boundary should move forward to 6.
        assert tc._snap_boundary(trajectory, 5, 4, 8) == 6
        # Index 4 is a 'gpt' turn and already clean.
        assert tc._snap_boundary(trajectory, 4, 4, 8) == 4

    def test_snap_boundary_falls_back_to_backward(self):
        tc = _make_compressor()
        # Protected tail begins on a 'tool' turn at max_idx: no clean boundary
        # ahead, so the boundary must retreat onto the preceding 'gpt' turn.
        trajectory = [
            {"from": "gpt", "value": "<tool_call>a</tool_call>"},
            {"from": "tool", "value": "<tool_response>a</tool_response>"},
        ]
        assert tc._snap_boundary(trajectory, 1, 0, 1) == 0


# ---------------------------------------------------------------------------
# TrajectoryCompressor — compression must never increase the token count
# ---------------------------------------------------------------------------


class TestCompressionNetSavingsGuard:
    """When the compressible middle is no larger than the summary that would
    replace it, compression cannot help — it must be skipped rather than grow
    the trajectory (and burn a summarization call)."""

    def _tiny_middle_trajectory(self):
        # Large protected head (system+human), tiny compressible middle.
        big = "w " * 400  # ~200 tokens each (1 token / 4 chars)
        small = "ok " * 2
        return [
            {"from": "system", "value": big},  # protected (first_system)
            {"from": "human", "value": big},  # protected (first_human)
            {"from": "gpt", "value": small},  # protected (first_gpt)
            {"from": "tool", "value": small},  # protected (first_tool)
            {"from": "gpt", "value": small},  # compressible middle
            {"from": "tool", "value": small},  # compressible middle
            {"from": "gpt", "value": small},  # protected (last 2)
            {"from": "human", "value": small},  # protected (last 2)
        ]

    def _config(self):
        config = CompressionConfig()
        config.protect_last_n_turns = 2
        config.summary_target_tokens = 20
        config.target_max_tokens = 100  # trajectory is far over this
        return config

    @pytest.mark.asyncio
    async def test_skips_compression_when_middle_smaller_than_summary(self):
        tc = _make_compressor(self._config())
        tc._generate_summary = AsyncMock(
            return_value="[CONTEXT SUMMARY]: " + "blah " * 30
        )
        trajectory = self._tiny_middle_trajectory()
        before = sum(await tc.count_turn_tokens(trajectory))

        compressed, metrics = await tc.compress_trajectory(trajectory)

        assert metrics.was_compressed is False
        assert compressed == trajectory
        assert sum(await tc.count_turn_tokens(compressed)) == before
        tc._generate_summary.assert_not_called()
