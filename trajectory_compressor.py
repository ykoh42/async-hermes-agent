#!/usr/bin/env python3
"""
Trajectory Compressor

Post-processes completed agent trajectories to compress them within a target
token budget while preserving training signal quality.

Compression Strategy:
1. Protect first turns (system, human, first gpt, first tool)
2. Protect last N turns (final actions and conclusions)
3. Compress MIDDLE turns only, starting from 2nd tool response
4. Compress only as much as needed to fit under target
5. Replace compressed region with a single human summary message
6. Keep remaining tool calls intact (model continues working after summary)

Usage:
    # Compress a directory of JSONL files
    python trajectory_compressor.py --input=data/my_run
    
    # Compress a single JSONL file
    python trajectory_compressor.py --input=data/trajectories.jsonl
    
    # Compress 15% sample of a file
    python trajectory_compressor.py --input=data/trajectories.jsonl --sample_percent=15
    
    # Compress with custom output and token target
    python trajectory_compressor.py --input=data/trajectories.jsonl --output=compressed.jsonl --target_max_tokens=16000
    
    # Compress 10% sample from a directory
    python trajectory_compressor.py --input=data/my_run --sample_percent=10
"""

import base64
import hashlib
import json
import os
import random
import time
import uuid
import yaml
import logging
import asyncio
import aiofiles
import aiofiles.os
import aiofiles.tempfile
import httpx
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from utils import base_url_host_matches, base_url_hostname
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
from hermes_constants import OPENROUTER_BASE_URL, get_hermes_home
from agent.retry_utils import jittered_backoff
from agent.secret_scope import get_secret


_KIMI_TOKENIZER_NAME = "moonshotai/Kimi-K2-Thinking"
_KIMI_TOKENIZER_REVISION = "main"
_KIMI_TOKENIZER_FILES = ("tokenizer_config.json", "tiktoken.model")
_FAST_TOKENIZER_FILES = ("tokenizer_config.json", "tokenizer.json")
_SENTENCEPIECE_TOKENIZER_FILES = (
    ("tokenizer_config.json", "tokenizer.model"),
    ("tokenizer_config.json", "spiece.model"),
)
_KIMI_NUM_RESERVED_SPECIAL_TOKENS = 256
_KIMI_PAT_STR = "|".join(
    [
        r"[\p{Han}]+",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)


class _KimiTokenizer:
    """CPU-only implementation of Kimi's published tokenizer ``encode`` API."""

    def __init__(self, encoding: Any) -> None:
        self._encoding = encoding

    @staticmethod
    def _split_whitespace_runs(text: str, max_run: int):
        current_length = 0
        current_is_space = text[0].isspace() if text else False
        start = 0
        for index, character in enumerate(text):
            is_space = character.isspace()
            if current_is_space ^ is_space:
                current_length = 1
                current_is_space = is_space
            else:
                current_length += 1
                if current_length > max_run:
                    yield text[start:index]
                    start = index
                    current_length = 1
        yield text[start:]

    def encode(self, text: str) -> List[int]:
        if type(text) is not str:
            raise TypeError("Kimi tokenizer input must be str")
        token_ids: List[int] = []
        for chunk_start in range(0, len(text), 400_000):
            chunk = text[chunk_start : chunk_start + 400_000]
            for substring in self._split_whitespace_runs(chunk, 25_000):
                token_ids.extend(
                    self._encoding.encode(substring, allowed_special="all")
                )
        return token_ids


class _FastTokenizer:
    """Transformers-compatible ``encode`` facade over tokenizer.json."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text, add_special_tokens=True).ids


def _generate_sentencepiece_bpe_merges(
    vocab: Dict[str, int],
    scores: Dict[str, float],
) -> List[Tuple[str, str]]:
    """Reconstruct the BPE merge order encoded by a SentencePiece model."""
    ranked: List[Tuple[str, str, float]] = []
    for merged_piece, score in scores.items():
        candidates: List[Tuple[str, str, float]] = []
        for index in range(1, len(merged_piece)):
            left = merged_piece[:index]
            right = merged_piece[index:]
            if left in vocab and right in vocab:
                candidates.append((left, right, score))
        candidates.sort(key=lambda value: (vocab[value[0]], vocab[value[1]]))
        ranked.extend(candidates)
    ranked.sort(
        key=lambda value: (value[2], len(value[0]), len(value[1])),
        reverse=True,
    )
    return [(left, right) for left, right, _score in ranked]


def _parse_tiktoken_ranks(model_bytes: bytes) -> Dict[bytes, int]:
    """Parse the public ``tiktoken.model`` format without synchronous I/O."""
    ranks: Dict[bytes, int] = {}
    for raw_line in model_bytes.splitlines():
        if not raw_line:
            continue
        token, rank = raw_line.split()
        ranks[base64.b64decode(token)] = int(rank)
    return ranks


def _build_kimi_tokenizer(config_bytes: bytes, model_bytes: bytes) -> _KimiTokenizer:
    """Construct the official Kimi BPE from already-loaded bytes."""
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - pinned core dependency
        raise RuntimeError(
            "Trajectory compression requires tiktoken. Reinstall "
            "async-hermes-agent to restore its pinned dependencies."
        ) from exc

    tokenizer_config = json.loads(config_bytes)
    added_tokens = tokenizer_config.get("added_tokens_decoder") or {}
    token_names = {
        int(token_id): token_data["content"]
        for token_id, token_data in added_tokens.items()
        if isinstance(token_data, dict) and isinstance(token_data.get("content"), str)
    }
    mergeable_ranks = _parse_tiktoken_ranks(model_bytes)
    base_token_count = len(mergeable_ranks)
    special_tokens = {
        token_names.get(token_id, f"<|reserved_token_{token_id}|>"): token_id
        for token_id in range(
            base_token_count,
            base_token_count + _KIMI_NUM_RESERVED_SPECIAL_TOKENS,
        )
    }
    encoding = tiktoken.Encoding(
        name="tiktoken.model",
        pat_str=_KIMI_PAT_STR,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    return _KimiTokenizer(encoding)


def _tokenizer_cache_directory(tokenizer_name: str) -> Path:
    digest = hashlib.sha256(tokenizer_name.encode("utf-8")).hexdigest()[:24]
    return get_hermes_home() / "cache" / "tokenizers" / digest


async def _read_optional_bytes(path: Path) -> bytes | None:
    try:
        async with aiofiles.open(path, "rb") as source:
            return await source.read()
    except FileNotFoundError:
        return None


async def _atomic_write_bytes(path: Path, content: bytes) -> None:
    await aiofiles.os.makedirs(path.parent, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        async with aiofiles.open(temporary, "wb") as destination:
            await destination.write(content)
        await aiofiles.os.replace(temporary, path)
    finally:
        try:
            await aiofiles.os.remove(temporary)
        except FileNotFoundError:
            pass


def _offline_mode_enabled() -> bool:
    return os.getenv("HF_HUB_OFFLINE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _complete_cached_assets(
    cached: Dict[str, bytes | None],
    filenames: Tuple[str, ...],
) -> Dict[str, bytes] | None:
    complete: Dict[str, bytes] = {}
    for name in filenames:
        content = cached.get(name)
        if content is None:
            return None
        complete[name] = content
    return complete


def _validate_hf_repo_id(repo_id: str) -> None:
    """Apply Hugging Face's repository-id shape constraints before HTTP."""
    if len(repo_id) > 96 or repo_id.endswith(".git"):
        raise ValueError(f"Invalid Hugging Face tokenizer repository: {repo_id!r}")
    parts = repo_id.split("/")
    allowed = frozenset(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    )
    if (
        not 1 <= len(parts) <= 2
        or any(
            not part
            or part[0] in "-."
            or part[-1] in "-."
            or "--" in part
            or ".." in part
            or any(character not in allowed for character in part)
            for part in parts
        )
    ):
        raise ValueError(f"Invalid Hugging Face tokenizer repository: {repo_id!r}")


async def _load_hf_tokenizer_assets(
    tokenizer_name: str,
    filenames: Tuple[str, ...],
    *,
    revision: str = "main",
) -> Dict[str, bytes]:
    """Load tokenizer files through native async local or Hub I/O."""
    local_directory = Path(tokenizer_name)
    if await aiofiles.os.path.isdir(local_directory):
        local_assets = {
            name: await _read_optional_bytes(local_directory / name)
            for name in filenames
        }
        complete_local = _complete_cached_assets(local_assets, filenames)
        if complete_local is None:
            missing = [name for name, content in local_assets.items() if content is None]
            raise FileNotFoundError(
                f"Tokenizer directory '{tokenizer_name}' is missing: "
                + ", ".join(missing)
            )
        return complete_local

    _validate_hf_repo_id(tokenizer_name)

    cache_directory = _tokenizer_cache_directory(tokenizer_name)
    cached = {
        name: await _read_optional_bytes(cache_directory / name)
        for name in filenames
    }
    metadata_path = cache_directory / "metadata.json"
    metadata_bytes = await _read_optional_bytes(metadata_path)
    try:
        metadata = json.loads(metadata_bytes) if metadata_bytes else {}
    except (TypeError, ValueError):
        metadata = {}

    cached_assets = _complete_cached_assets(cached, filenames)
    if _offline_mode_enabled():
        if cached_assets is not None:
            return cached_assets
        raise RuntimeError(
            f"Tokenizer '{tokenizer_name}' is not cached and "
            "HF_HUB_OFFLINE is enabled."
        )

    token = get_secret("HF_TOKEN") or get_secret("HUGGING_FACE_HUB_TOKEN")
    common_headers = {"User-Agent": "async-hermes-agent/trajectory-compressor"}
    if token:
        common_headers["Authorization"] = f"Bearer {token}"

    refreshed: Dict[str, bytes] = {}
    etags: Dict[str, str] = {}
    try:
        from agent.ssl_verify import _create_httpx_client

        async with (await _create_httpx_client(
            follow_redirects=True,
            timeout=httpx.Timeout(60.0),
            headers=common_headers,
        )) as client:
            for name in filenames:
                headers = {}
                previous_etag = (metadata.get("etags") or {}).get(name)
                if cached[name] is not None and isinstance(previous_etag, str):
                    headers["If-None-Match"] = previous_etag
                url = (
                    f"https://huggingface.co/{tokenizer_name}/resolve/"
                    f"{revision}/{name}"
                )
                response = await client.get(url, headers=headers)
                if response.status_code == 304:
                    cached_content = cached[name]
                    if cached_content is None or not isinstance(previous_etag, str):
                        raise RuntimeError(
                            f"Invalid conditional response for tokenizer asset {name}"
                        )
                    refreshed[name] = cached_content
                    etags[name] = previous_etag
                    continue
                response.raise_for_status()
                refreshed[name] = response.content
                response_etag = response.headers.get("etag")
                if response_etag:
                    etags[name] = response_etag
    except httpx.TransportError:
        if cached_assets is not None:
            return cached_assets
        raise

    for name, content in refreshed.items():
        if content != cached[name]:
            await _atomic_write_bytes(cache_directory / name, content)
    await _atomic_write_bytes(
        metadata_path,
        json.dumps(
            {
                "tokenizer": tokenizer_name,
                "revision": revision,
                "etags": etags,
            },
            sort_keys=True,
        ).encode("utf-8"),
    )
    return refreshed


async def _load_kimi_tokenizer_assets() -> Tuple[bytes, bytes]:
    """Load Kimi tokenizer files through an async, conditional HTTP cache."""
    assets = await _load_hf_tokenizer_assets(
        _KIMI_TOKENIZER_NAME,
        _KIMI_TOKENIZER_FILES,
        revision=_KIMI_TOKENIZER_REVISION,
    )
    return assets["tokenizer_config.json"], assets["tiktoken.model"]


async def _load_fast_tokenizer_assets(
    tokenizer_name: str,
) -> Tuple[bytes, bytes]:
    """Load a standard Hugging Face tokenizer.json and its configuration."""
    assets = await _load_hf_tokenizer_assets(
        tokenizer_name,
        _FAST_TOKENIZER_FILES,
    )
    return assets["tokenizer_config.json"], assets["tokenizer.json"]


def _is_missing_tokenizer_asset(exc: BaseException) -> bool:
    if isinstance(exc, FileNotFoundError):
        return True
    return (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response.status_code == 404
    )


async def _load_custom_tokenizer_assets(
    tokenizer_name: str,
) -> Tuple[str, bytes, bytes]:
    """Load a tokenizer.json pipeline or a supported SentencePiece model."""
    try:
        config_bytes, tokenizer_bytes = await _load_fast_tokenizer_assets(
            tokenizer_name
        )
        return "fast", config_bytes, tokenizer_bytes
    except (FileNotFoundError, httpx.HTTPStatusError) as exc:
        if not _is_missing_tokenizer_asset(exc):
            raise

    last_missing: BaseException | None = None
    for filenames in _SENTENCEPIECE_TOKENIZER_FILES:
        try:
            assets = await _load_hf_tokenizer_assets(
                tokenizer_name,
                filenames,
            )
            return "sentencepiece", assets["tokenizer_config.json"], assets[
                filenames[1]
            ]
        except (FileNotFoundError, httpx.HTTPStatusError) as exc:
            if not _is_missing_tokenizer_asset(exc):
                raise
            last_missing = exc
    if last_missing is not None:
        raise last_missing
    raise RuntimeError(f"No tokenizer assets found for {tokenizer_name!r}")


def _build_fast_tokenizer(
    config_bytes: bytes,
    tokenizer_bytes: bytes,
) -> _FastTokenizer:
    """Construct the generic Rust tokenizer from already-loaded JSON bytes."""
    tokenizer_config = json.loads(config_bytes)
    if tokenizer_config.get("auto_map", {}).get("AutoTokenizer"):
        raise ValueError(
            "the configured tokenizer requires Python remote code, which has "
            "no native-async file-loading contract"
        )
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - pinned core dependency
        raise RuntimeError(
            "Trajectory compression requires tokenizers. Reinstall "
            "async-hermes-agent to restore its pinned dependencies."
        ) from exc
    return _FastTokenizer(Tokenizer.from_str(tokenizer_bytes.decode("utf-8")))


def _build_sentencepiece_tokenizer(
    config_bytes: bytes,
    model_bytes: bytes,
) -> _FastTokenizer:
    """Build verified Llama/T5 SentencePiece families without file I/O."""
    tokenizer_config = json.loads(config_bytes)
    if tokenizer_config.get("auto_map", {}).get("AutoTokenizer"):
        raise ValueError(
            "the configured tokenizer requires Python remote code, which has "
            "no native-async file-loading contract"
        )
    tokenizer_class = str(tokenizer_config.get("tokenizer_class") or "")
    normalized_class = tokenizer_class.removesuffix("Fast")
    if normalized_class.endswith("LlamaTokenizer"):
        add_bos_token = bool(tokenizer_config.get("add_bos_token", True))
        add_eos_token = bool(tokenizer_config.get("add_eos_token", False))
        tokenizer_family = "llama"
    elif normalized_class in {"T5Tokenizer", "MT5Tokenizer"} or (
        not normalized_class and "extra_ids" in tokenizer_config
    ):
        add_bos_token = False
        add_eos_token = True
        tokenizer_family = "t5"
    else:
        raise ValueError(
            "unsupported SentencePiece tokenizer family: "
            f"{tokenizer_class or 'unknown'}"
        )

    try:
        from sentencepiece import sentencepiece_model_pb2
    except ImportError as exc:  # pragma: no cover - pinned core dependency
        raise RuntimeError(
            "Trajectory compression requires sentencepiece. Reinstall "
            "async-hermes-agent to restore its pinned dependencies."
        ) from exc
    try:
        model_proto_type = getattr(sentencepiece_model_pb2, "ModelProto")
        model_proto = model_proto_type.FromString(model_bytes)
    except Exception as exc:
        raise ValueError("invalid serialized SentencePiece model") from exc
    trainer_spec = model_proto.trainer_spec
    if add_bos_token and trainer_spec.bos_id < 0:
        raise ValueError("SentencePiece tokenizer requires a BOS token")
    if add_eos_token and trainer_spec.eos_id < 0:
        raise ValueError("SentencePiece tokenizer requires an EOS token")
    try:
        from tokenizers import (
            AddedToken,
            Tokenizer,
            decoders,
            models,
            pre_tokenizers,
            processors,
        )
    except ImportError as exc:  # pragma: no cover - pinned core dependency
        raise RuntimeError(
            "Trajectory compression requires tokenizers. Reinstall "
            "async-hermes-agent to restore its pinned dependencies."
        ) from exc
    vocab_scores = [
        (piece.piece, piece.score)
        for piece in model_proto.pieces
    ]
    vocab = {
        piece: index
        for index, (piece, _score) in enumerate(vocab_scores)
    }

    if tokenizer_family == "llama":
        scores = {
            piece: score for piece, score in vocab_scores
        }
        tokenizer = Tokenizer(
            models.BPE(
                vocab=vocab,
                merges=_generate_sentencepiece_bpe_merges(vocab, scores),
                fuse_unk=True,
                byte_fallback=True,
                dropout=None,
            )
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
            replacement="▁",
            prepend_scheme="first",
            split=False,
        )
        decoder_steps: List[Any] = [
            decoders.Replace("▁", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
        ]
        if bool(tokenizer_config.get("add_prefix_space", True)):
            decoder_steps.append(decoders.Strip(content=" ", left=1))
        tokenizer.decoder = decoders.Sequence(decoder_steps)

        special_tokens: List[AddedToken] = []
        template_tokens: List[Tuple[str, int]] = []
        for key, fallback in (
            ("unk_token", "<unk>"),
            ("bos_token", "<s>"),
            ("eos_token", "</s>"),
        ):
            value = tokenizer_config.get(key, fallback)
            if isinstance(value, dict):
                content = str(value.get("content") or fallback)
                normalized = bool(value.get("normalized", key == "unk_token"))
            else:
                content = str(value or fallback)
                normalized = key == "unk_token"
            token_id = vocab.get(content)
            if token_id is None:
                raise ValueError(f"SentencePiece model is missing {key}")
            special_tokens.append(
                AddedToken(content, special=True, normalized=normalized)
            )
            if key != "unk_token":
                template_tokens.append((content, token_id))
        tokenizer.add_special_tokens(special_tokens)

        bos_content, bos_id = template_tokens[0]
        eos_content, eos_id = template_tokens[1]
        single_parts = ["$A"]
        pair_parts = ["$A"]
        post_specials: List[Tuple[str, int]] = []
        if add_bos_token:
            single_parts.insert(0, bos_content)
            pair_parts.insert(0, bos_content)
            post_specials.append((bos_content, bos_id))
        if add_eos_token:
            single_parts.append(eos_content)
            pair_parts.append(eos_content)
            post_specials.append((eos_content, eos_id))
        if add_bos_token:
            pair_parts.extend((f"{bos_content}:1", "$B:1"))
        else:
            pair_parts.append("$B:1")
        if add_eos_token:
            pair_parts.append(f"{eos_content}:1")
        if post_specials:
            tokenizer.post_processor = processors.TemplateProcessing(
                single=" ".join(single_parts),
                pair=" ".join(pair_parts),
                special_tokens=post_specials,
            )
        return _FastTokenizer(tokenizer)

    tokenizer = Tokenizer(
        models.Unigram(
            vocab=vocab_scores,
            unk_id=trainer_spec.unk_id,
            byte_fallback=False,
        )
    )
    precompiled_charsmap = model_proto.normalizer_spec.precompiled_charsmap
    if precompiled_charsmap:
        from tokenizers import normalizers

        tokenizer.normalizer = normalizers.Precompiled(precompiled_charsmap)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Metaspace(
                replacement="▁",
                prepend_scheme="always",
                split=True,
            ),
        ]
    )
    tokenizer.decoder = decoders.Metaspace(
        replacement="▁",
        prepend_scheme="always",
        split=True,
    )
    for key, fallback in (
        ("unk_token", "<unk>"),
        ("eos_token", "</s>"),
        ("pad_token", "<pad>"),
    ):
        value = tokenizer_config.get(key, fallback)
        content = str(value.get("content") or fallback) if isinstance(
            value, dict
        ) else str(value or fallback)
        if content in vocab:
            tokenizer.add_special_tokens(
                [AddedToken(content, special=True, normalized=False)]
            )
    eos_value = tokenizer_config.get("eos_token", "</s>")
    eos_content = str(eos_value.get("content") or "</s>") if isinstance(
        eos_value, dict
    ) else str(eos_value or "</s>")
    eos_id = vocab.get(eos_content)
    if eos_id is None:
        raise ValueError("SentencePiece model is missing eos_token")
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"$A {eos_content}",
        pair=f"$A {eos_content} $B {eos_content}",
        special_tokens=[(eos_content, eos_id)],
    )
    return _FastTokenizer(tokenizer)


async def _finish_owned_task(task: asyncio.Task[Any]) -> Any:
    """Finish one compressor-owned task through repeated cancellation."""
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised below
            if task.cancelled():
                raise
            if cancellation is None:
                cancellation = exc
        except Exception as exc:
            if cancellation is not None:
                raise cancellation from exc
            raise
    if cancellation is not None:
        raise cancellation
    return result


async def _list_jsonl_files(directory: Path) -> List[Path]:
    """Return JSONL files without blocking the event loop on directory I/O."""
    try:
        names = await aiofiles.os.listdir(directory)
    except OSError:
        return []
    paths: list[Path] = []
    for name in names:
        path = directory / name
        if name.endswith(".jsonl") and await aiofiles.os.path.isfile(path):
            paths.append(path)
    return sorted(paths)


def _effective_temperature_for_model(
    model: str,
    requested_temperature: float,
    base_url: Optional[str] = None,
) -> Optional[float]:
    """Apply fixed model temperature contracts to direct client calls.

    Returns ``None`` when the model manages temperature server-side (Kimi);
    callers must omit the ``temperature`` kwarg entirely in that case.
    """
    try:
        from agent.auxiliary_client import _fixed_temperature_for_model, OMIT_TEMPERATURE
    except Exception:
        return requested_temperature

    fixed_temperature = _fixed_temperature_for_model(model, base_url)
    if fixed_temperature is OMIT_TEMPERATURE:
        return None  # caller must omit temperature
    if fixed_temperature is not None:
        return fixed_temperature
    return requested_temperature


@dataclass
class CompressionConfig:
    """Configuration for trajectory compression."""
    # Tokenizer
    tokenizer_name: str = "moonshotai/Kimi-K2-Thinking"
    trust_remote_code: bool = True
    
    # Compression targets
    target_max_tokens: int = 15250
    summary_target_tokens: int = 750
    
    # Protected turns
    protect_first_system: bool = True
    protect_first_human: bool = True
    protect_first_gpt: bool = True
    protect_first_tool: bool = True
    protect_last_n_turns: int = 4
    
    # Summarization (OpenRouter)
    summarization_model: str = "google/gemini-3-flash-preview"
    base_url: str = OPENROUTER_BASE_URL
    api_key_env: str = "OPENROUTER_API_KEY"
    temperature: float = 0.3
    max_retries: int = 3
    retry_delay: int = 2
    
    # Output
    add_summary_notice: bool = True
    summary_notice_text: str = "\n\nSome of your previous tool responses may be summarized to preserve context."
    output_suffix: str = "_compressed"
    
    # Processing
    num_workers: int = 4
    max_concurrent_requests: int = 50  # Max concurrent API calls for summarization
    skip_under_target: bool = True
    save_over_limit: bool = True
    per_trajectory_timeout: int = 300  # Timeout per trajectory in seconds (default: 5 min)
    
    # Metrics
    metrics_enabled: bool = True
    metrics_per_trajectory: bool = True
    metrics_output_file: str = "compression_metrics.json"
    
    @classmethod
    async def from_yaml(cls, yaml_path: str) -> "CompressionConfig":
        """Load configuration from YAML file."""
        async with aiofiles.open(yaml_path, "r", encoding="utf-8") as source:
            data = yaml.safe_load(await source.read()) or {}

        config = cls()

        # Tokenizer
        if 'tokenizer' in data:
            config.tokenizer_name = data['tokenizer'].get('name', config.tokenizer_name)
            config.trust_remote_code = data['tokenizer'].get('trust_remote_code', config.trust_remote_code)
        
        # Compression
        if 'compression' in data:
            config.target_max_tokens = data['compression'].get('target_max_tokens', config.target_max_tokens)
            config.summary_target_tokens = data['compression'].get('summary_target_tokens', config.summary_target_tokens)
        
        # Protected turns
        if 'protected_turns' in data:
            config.protect_first_system = data['protected_turns'].get('first_system', config.protect_first_system)
            config.protect_first_human = data['protected_turns'].get('first_human', config.protect_first_human)
            config.protect_first_gpt = data['protected_turns'].get('first_gpt', config.protect_first_gpt)
            config.protect_first_tool = data['protected_turns'].get('first_tool', config.protect_first_tool)
            config.protect_last_n_turns = data['protected_turns'].get('last_n_turns', config.protect_last_n_turns)
        
        # Summarization
        if 'summarization' in data:
            config.summarization_model = data['summarization'].get('model', config.summarization_model)
            config.base_url = data['summarization'].get('base_url') or config.base_url
            config.api_key_env = data['summarization'].get('api_key_env', config.api_key_env)
            config.temperature = data['summarization'].get('temperature', config.temperature)
            config.max_retries = data['summarization'].get('max_retries', config.max_retries)
            config.retry_delay = data['summarization'].get('retry_delay', config.retry_delay)
        
        # Output
        if 'output' in data:
            config.add_summary_notice = data['output'].get('add_summary_notice', config.add_summary_notice)
            config.summary_notice_text = data['output'].get('summary_notice_text', config.summary_notice_text)
            config.output_suffix = data['output'].get('output_suffix', config.output_suffix)
        
        # Processing
        if 'processing' in data:
            config.num_workers = data['processing'].get('num_workers', config.num_workers)
            config.max_concurrent_requests = data['processing'].get('max_concurrent_requests', config.max_concurrent_requests)
            config.skip_under_target = data['processing'].get('skip_under_target', config.skip_under_target)
            config.save_over_limit = data['processing'].get('save_over_limit', config.save_over_limit)
        
        # Metrics
        if 'metrics' in data:
            config.metrics_enabled = data['metrics'].get('enabled', config.metrics_enabled)
            config.metrics_per_trajectory = data['metrics'].get('per_trajectory', config.metrics_per_trajectory)
            config.metrics_output_file = data['metrics'].get('output_file', config.metrics_output_file)
        
        return config


@dataclass
class TrajectoryMetrics:
    """Metrics for a single trajectory compression."""
    original_tokens: int = 0
    compressed_tokens: int = 0
    tokens_saved: int = 0
    compression_ratio: float = 1.0
    
    original_turns: int = 0
    compressed_turns: int = 0
    turns_removed: int = 0
    
    turns_compressed_start_idx: int = -1
    turns_compressed_end_idx: int = -1
    turns_in_compressed_region: int = 0
    
    was_compressed: bool = False
    still_over_limit: bool = False
    skipped_under_target: bool = False
    
    summarization_api_calls: int = 0
    summarization_errors: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "tokens_saved": self.tokens_saved,
            "compression_ratio": round(self.compression_ratio, 4),
            "original_turns": self.original_turns,
            "compressed_turns": self.compressed_turns,
            "turns_removed": self.turns_removed,
            "compression_region": {
                "start_idx": self.turns_compressed_start_idx,
                "end_idx": self.turns_compressed_end_idx,
                "turns_count": self.turns_in_compressed_region,
            },
            "was_compressed": self.was_compressed,
            "still_over_limit": self.still_over_limit,
            "skipped_under_target": self.skipped_under_target,
            "summarization_api_calls": self.summarization_api_calls,
            "summarization_errors": self.summarization_errors,
        }


@dataclass 
class AggregateMetrics:
    """Aggregate metrics across all trajectories."""
    total_trajectories: int = 0
    trajectories_compressed: int = 0
    trajectories_skipped_under_target: int = 0
    trajectories_still_over_limit: int = 0
    trajectories_failed: int = 0
    
    total_tokens_before: int = 0
    total_tokens_after: int = 0
    total_tokens_saved: int = 0
    
    total_turns_before: int = 0
    total_turns_after: int = 0
    total_turns_removed: int = 0
    
    total_summarization_calls: int = 0
    total_summarization_errors: int = 0
    
    # Distribution stats
    compression_ratios: List[float] = field(default_factory=list)
    tokens_saved_list: List[int] = field(default_factory=list)
    turns_removed_list: List[int] = field(default_factory=list)
    
    processing_start_time: str = ""
    processing_end_time: str = ""
    processing_duration_seconds: float = 0.0
    
    def add_trajectory_metrics(self, metrics: TrajectoryMetrics):
        """Add a trajectory's metrics to the aggregate."""
        self.total_trajectories += 1
        self.total_tokens_before += metrics.original_tokens
        self.total_tokens_after += metrics.compressed_tokens
        self.total_tokens_saved += metrics.tokens_saved
        self.total_turns_before += metrics.original_turns
        self.total_turns_after += metrics.compressed_turns
        self.total_turns_removed += metrics.turns_removed
        self.total_summarization_calls += metrics.summarization_api_calls
        self.total_summarization_errors += metrics.summarization_errors
        
        if metrics.was_compressed:
            self.trajectories_compressed += 1
            self.compression_ratios.append(metrics.compression_ratio)
            self.tokens_saved_list.append(metrics.tokens_saved)
            self.turns_removed_list.append(metrics.turns_removed)
        
        if metrics.skipped_under_target:
            self.trajectories_skipped_under_target += 1
        
        if metrics.still_over_limit:
            self.trajectories_still_over_limit += 1
    
    def to_dict(self) -> Dict[str, Any]:
        avg_compression_ratio = (
            sum(self.compression_ratios) / len(self.compression_ratios) 
            if self.compression_ratios else 1.0
        )
        avg_tokens_saved = (
            sum(self.tokens_saved_list) / len(self.tokens_saved_list)
            if self.tokens_saved_list else 0
        )
        avg_turns_removed = (
            sum(self.turns_removed_list) / len(self.turns_removed_list)
            if self.turns_removed_list else 0
        )
        
        return {
            "summary": {
                "total_trajectories": self.total_trajectories,
                "trajectories_compressed": self.trajectories_compressed,
                "trajectories_skipped_under_target": self.trajectories_skipped_under_target,
                "trajectories_still_over_limit": self.trajectories_still_over_limit,
                "trajectories_failed": self.trajectories_failed,
                "compression_rate": round(self.trajectories_compressed / max(self.total_trajectories, 1), 4),
            },
            "tokens": {
                "total_before": self.total_tokens_before,
                "total_after": self.total_tokens_after,
                "total_saved": self.total_tokens_saved,
                "overall_compression_ratio": round(self.total_tokens_after / max(self.total_tokens_before, 1), 4),
            },
            "turns": {
                "total_before": self.total_turns_before,
                "total_after": self.total_turns_after,
                "total_removed": self.total_turns_removed,
            },
            "averages": {
                "avg_compression_ratio": round(avg_compression_ratio, 4),
                "avg_tokens_saved_per_compressed": round(avg_tokens_saved, 1),
                "avg_turns_removed_per_compressed": round(avg_turns_removed, 2),
            },
            "summarization": {
                "total_api_calls": self.total_summarization_calls,
                "total_errors": self.total_summarization_errors,
                "success_rate": round(1 - (self.total_summarization_errors / max(self.total_summarization_calls, 1)), 4),
            },
            "processing": {
                "start_time": self.processing_start_time,
                "end_time": self.processing_end_time,
                "duration_seconds": round(self.processing_duration_seconds, 2),
            },
        }


class TrajectoryCompressor:
    """
    Compresses agent trajectories to fit within a target token budget.
    
    Compression strategy:
    1. Keep protected head turns (system, human, first gpt+tool)
    2. Keep protected tail turns (last N turns)
    3. From the compressible middle region, compress only as much as needed
    4. Replace compressed turns with a single human summary message
    5. Keep remaining middle turns intact (model continues with tools)
    """
    
    def __init__(self, config: CompressionConfig):
        """Initialize the compressor."""
        self.config = config
        self.aggregate_metrics = AggregateMetrics()
        self.tokenizer = None
        self._tokenizer_init_lock = asyncio.Lock()
        self._initialization_lock = asyncio.Lock()
        self._client_init_lock = asyncio.Lock()
        self._summarizer_initialized = False
        self.client = None
        self.logger = logging.getLogger(__name__)

    async def _initialize(self) -> None:
        """Initialize retained resources in their upstream order."""
        if self.tokenizer is not None and self._summarizer_initialized:
            return
        async with self._initialization_lock:
            if self.tokenizer is None:
                await self._init_tokenizer()
            if not self._summarizer_initialized:
                self._init_summarizer()
                self._summarizer_initialized = True

    async def _init_tokenizer(self) -> None:
        """Load the configured tokenizer without blocking the event loop."""
        if self.tokenizer is not None:
            return
        async with self._tokenizer_init_lock:
            if self.tokenizer is not None:
                return
            try:
                tokenizer_name = self.config.tokenizer_name.strip().rstrip("/")
                if tokenizer_name == _KIMI_TOKENIZER_NAME:
                    if not self.config.trust_remote_code:
                        raise ValueError(
                            "trust_remote_code=true is required by the published "
                            "Kimi tokenizer contract"
                        )
                    config_bytes, model_bytes = await _load_kimi_tokenizer_assets()
                    self.tokenizer = _build_kimi_tokenizer(
                        config_bytes,
                        model_bytes,
                    )
                else:
                    tokenizer_kind, config_bytes, tokenizer_bytes = (
                        await _load_custom_tokenizer_assets(tokenizer_name)
                    )
                    if tokenizer_kind == "fast":
                        self.tokenizer = _build_fast_tokenizer(
                            config_bytes,
                            tokenizer_bytes,
                        )
                    else:
                        self.tokenizer = _build_sentencepiece_tokenizer(
                            config_bytes,
                            tokenizer_bytes,
                        )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load tokenizer '{self.config.tokenizer_name}': {exc}"
                ) from exc
            print(f"✅ Loaded tokenizer: {self.config.tokenizer_name}")
    
    def _init_summarizer(self):
        """Initialize async-only LLM routing metadata for summarization.

        Uses call_llm from the centralized provider router
        which handles auth, headers, and provider detection internally.
        Provider construction is deferred to the first async summary call so
        constructing a compressor never creates a synchronous SDK client.
        """

        provider = self._detect_provider()
        if provider:
            # Store provider for use in _generate_summary calls
            self._llm_provider = provider
            self._use_call_llm = True
            self.client = None
        else:
            # Custom endpoint — retain only lazy client construction metadata.
            self._use_call_llm = False
            api_key = get_secret(self.config.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Missing API key. Set {self.config.api_key_env} "
                    f"environment variable.")
            # AsyncOpenAI is created lazily in _get_client() so construction
            # stays side-effect-free and the transport binds to the caller's
            # active event loop.
            self.client = None
            self._client_api_key = api_key

        print(f"✅ Initialized summarizer client: {self.config.summarization_model}")
        print(f"   Max concurrent requests: {self.config.max_concurrent_requests}")

    async def _get_client(self):
        """Return an AsyncOpenAI client bound to the current event loop.

        Created lazily and reused for this compressor lifecycle.
        """
        if self.client is not None:
            return self.client
        async with self._client_init_lock:
            if self.client is not None:
                return self.client
            from openai import AsyncOpenAI
            from agent.auxiliary_client import _to_openai_base_url
            from agent.ssl_verify import _create_openai_sdk_client

            self.client = await _create_openai_sdk_client(
                AsyncOpenAI,
                api_key=self._client_api_key,
                base_url=_to_openai_base_url(self.config.base_url),
            )
        return self.client

    async def close(self) -> None:
        """Close the owned summarization transport, if one was created."""
        client = self.client
        self.client = None
        if client is not None:
            close_task = asyncio.create_task(
                client.close(),
                name="trajectory-compressor-client-close",
            )
            await _finish_owned_task(close_task)

    def _detect_provider(self) -> str:
        """Detect the provider name from the configured base_url."""
        url = self.config.base_url or ""
        if base_url_host_matches(url, "openrouter.ai"):
            return "openrouter"
        if base_url_host_matches(url, "nousresearch.com"):
            return "nous"
        if (
            base_url_hostname(url) == "chatgpt.com"
            and "/backend-api/codex" in url.lower()
        ):
            return "codex"
        if base_url_host_matches(url, "z.ai"):
            return "zai"
        if (
            base_url_host_matches(url, "moonshot.ai")
            or base_url_host_matches(url, "moonshot.cn")
            or base_url_host_matches(url, "api.kimi.com")
        ):
            return "kimi-coding"
        if base_url_host_matches(url, "arcee.ai"):
            return "arcee"
        if base_url_host_matches(url, "minimaxi.com"):
            return "minimax-cn"
        if base_url_host_matches(url, "minimax.io"):
            return "minimax"
        # Unknown base_url — not a known provider
        return ""
    
    def _count_tokens_initialized(self, text: str) -> int:
        if not text:
            return 0
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise RuntimeError("Tokenizer initialization invariant violated")
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
        return len(text) // 4

    async def count_tokens(self, text: str) -> int:
        """Count tokens with the configured tokenizer."""
        await self._initialize()
        return self._count_tokens_initialized(text)

    async def count_trajectory_tokens(
        self, trajectory: List[Dict[str, str]]
    ) -> int:
        """Count total tokens in a trajectory."""
        await self._initialize()
        return sum(
            self._count_tokens_initialized(turn.get("value", ""))
            for turn in trajectory
        )

    async def count_turn_tokens(
        self, trajectory: List[Dict[str, str]]
    ) -> List[int]:
        """Count tokens for each turn in a trajectory."""
        await self._initialize()
        return [
            self._count_tokens_initialized(turn.get("value", ""))
            for turn in trajectory
        ]
    
    def _find_protected_indices(self, trajectory: List[Dict[str, str]]) -> Tuple[set, int, int]:
        """
        Find indices of protected turns.
        
        Returns:
            Tuple of (protected_set, compressible_start, compressible_end)
        """
        n = len(trajectory)
        protected = set()
        
        # Track first occurrences
        first_system = first_human = first_gpt = first_tool = None
        
        for i, turn in enumerate(trajectory):
            role = turn.get("from", "")
            if role == "system" and first_system is None:
                first_system = i
            elif role == "human" and first_human is None:
                first_human = i
            elif role == "gpt" and first_gpt is None:
                first_gpt = i
            elif role == "tool" and first_tool is None:
                first_tool = i
        
        # Protect first turns
        if self.config.protect_first_system and first_system is not None:
            protected.add(first_system)
        if self.config.protect_first_human and first_human is not None:
            protected.add(first_human)
        if self.config.protect_first_gpt and first_gpt is not None:
            protected.add(first_gpt)
        if self.config.protect_first_tool and first_tool is not None:
            protected.add(first_tool)
        
        # Protect last N turns
        for i in range(max(0, n - self.config.protect_last_n_turns), n):
            protected.add(i)
        
        # Determine compressible region
        # Start after the last protected head turn
        head_protected = [i for i in protected if i < n // 2]
        tail_protected = [i for i in protected if i >= n // 2]
        
        compressible_start = max(head_protected) + 1 if head_protected else 0
        compressible_end = min(tail_protected) if tail_protected else n

        return protected, compressible_start, compressible_end

    @staticmethod
    def _is_boundary_clean(trajectory: List[Dict[str, str]], idx: int) -> bool:
        """Return True if a region boundary at ``idx`` does not split a turn pair.

        In the from/value trajectory format a ``tool`` turn (carrying
        ``<tool_response>`` markers) is always emitted immediately after the
        ``gpt`` turn whose ``<tool_call>`` it answers. A compression boundary
        that lands *on* a ``tool`` turn therefore cuts between a tool call and
        its response. A boundary is only clean when it sits at the very end of
        the trajectory or on a non-``tool`` turn.
        """
        return idx >= len(trajectory) or trajectory[idx].get("from") != "tool"

    @classmethod
    def _snap_boundary(
        cls,
        trajectory: List[Dict[str, str]],
        idx: int,
        min_idx: int,
        max_idx: int,
    ) -> int:
        """Move a compression boundary onto the nearest clean turn boundary.

        Moving forward is preferred so that an orphaned ``tool`` turn is folded
        into the region that already holds its ``gpt`` turn; if no clean
        boundary exists ahead (for example the protected tail itself begins on a
        ``tool`` turn) the boundary is moved backward instead. The result is
        clamped to ``[min_idx, max_idx]``.
        """
        forward = idx
        while forward < max_idx and not cls._is_boundary_clean(trajectory, forward):
            forward += 1
        if cls._is_boundary_clean(trajectory, forward):
            return forward
        backward = idx
        while backward > min_idx and not cls._is_boundary_clean(trajectory, backward):
            backward -= 1
        return backward

    def _extract_turn_content_for_summary(self, trajectory: List[Dict[str, str]], start: int, end: int) -> str:
        """
        Extract content from turns to be summarized.
        
        Args:
            trajectory: Full trajectory
            start: Start index (inclusive)
            end: End index (exclusive)
            
        Returns:
            Formatted string of turn contents for summarization
        """
        parts = []
        for i in range(start, end):
            turn = trajectory[i]
            role = turn.get("from", "unknown")
            value = turn.get("value", "")
            
            # Truncate very long values for the summary prompt
            if len(value) > 3000:
                value = value[:1500] + "\n...[truncated]...\n" + value[-500:]
            
            parts.append(f"[Turn {i} - {role.upper()}]:\n{value}")
        
        return "\n\n".join(parts)

    @staticmethod
    def _coerce_summary_content(content: Any) -> str:
        """Normalize summary-model output to a safe string."""
        if not isinstance(content, str):
            content = str(content) if content else ""
        return content.strip()

    @staticmethod
    def _ensure_summary_prefix(summary: str) -> str:
        """Normalize summary text to include the expected prefix exactly once."""
        text = (summary or "").strip()
        if text.startswith("[CONTEXT SUMMARY]:"):
            return text
        return "[CONTEXT SUMMARY]:" if not text else f"[CONTEXT SUMMARY]: {text}"
    
    async def _generate_summary(self, content: str, metrics: TrajectoryMetrics) -> str:
        """
        Generate a summary of the compressed turns.
        
        Args:
            content: The content to summarize
            metrics: Metrics object to update
            
        Returns:
            Summary string
        """
        prompt = f"""Summarize the following agent conversation turns concisely. This summary will replace these turns in the conversation history.

Write the summary from a neutral perspective describing what the assistant did and learned. Include:
1. What actions the assistant took (tool calls, searches, file operations)
2. Key information or results obtained
3. Any important decisions or findings
4. Relevant data, file names, values, or outputs

Keep the summary factual and informative. Target approximately {self.config.summary_target_tokens} tokens.

---
TURNS TO SUMMARIZE:
{content}
---

Write only the summary, starting with "[CONTEXT SUMMARY]:" prefix."""

        for attempt in range(self.config.max_retries):
            try:
                metrics.summarization_api_calls += 1
                summary_temperature = _effective_temperature_for_model(
                    self.config.summarization_model,
                    self.config.temperature,
                    self.config.base_url,
                )
                
                if getattr(self, '_use_call_llm', False):
                    from agent.auxiliary_client import call_llm
                    response = await call_llm(
                        provider=self._llm_provider,
                        model=self.config.summarization_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=summary_temperature,
                        max_tokens=self.config.summary_target_tokens * 2,
                    )
                else:
                    _create_kwargs = {
                        "model": self.config.summarization_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": self.config.summary_target_tokens * 2,
                    }
                    if summary_temperature is not None:
                        _create_kwargs["temperature"] = summary_temperature
                    client = await self._get_client()
                    response = await client.chat.completions.create(**_create_kwargs)
                
                summary = self._coerce_summary_content(response.choices[0].message.content)
                return self._ensure_summary_prefix(summary)
                
            except Exception as e:
                metrics.summarization_errors += 1
                self.logger.warning(f"Summarization attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(jittered_backoff(attempt + 1, base_delay=self.config.retry_delay, max_delay=30.0))
                else:
                    # Fallback: create a basic summary
                    return "[CONTEXT SUMMARY]: [Summary generation failed - previous turns contained tool calls and responses that have been compressed to save context space.]"
    
    async def compress_trajectory(
        self,
        trajectory: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], TrajectoryMetrics]:
        """
        Compress a single trajectory to fit within the target token budget.
        """
        metrics = TrajectoryMetrics()
        metrics.original_turns = len(trajectory)
        
        # Count tokens per turn
        turn_tokens = await self.count_turn_tokens(trajectory)
        total_tokens = sum(turn_tokens)
        metrics.original_tokens = total_tokens
        
        # Check if compression needed
        if total_tokens <= self.config.target_max_tokens:
            metrics.skipped_under_target = True
            metrics.compressed_tokens = total_tokens
            metrics.compressed_turns = len(trajectory)
            metrics.compression_ratio = 1.0
            return trajectory, metrics
        
        # Find protected regions
        protected, compress_start, compress_end = self._find_protected_indices(trajectory)

        # Snap the head boundary so the compressible region never *starts* on an
        # orphaned <tool_response> whose <tool_call> lives in the protected head.
        compress_start = self._snap_boundary(trajectory, compress_start, compress_start, compress_end)

        # Check if there's anything to compress
        if compress_start >= compress_end:
            metrics.compressed_tokens = total_tokens
            metrics.compressed_turns = len(trajectory)
            metrics.still_over_limit = total_tokens > self.config.target_max_tokens
            return trajectory, metrics
        
        # Calculate how much we need to save
        tokens_to_save = total_tokens - self.config.target_max_tokens
        target_tokens_to_compress = tokens_to_save + self.config.summary_target_tokens
        
        # Accumulate turns from compress_start until we have enough savings
        accumulated_tokens = 0
        compress_until = compress_start
        
        for i in range(compress_start, compress_end):
            accumulated_tokens += turn_tokens[i]
            compress_until = i + 1
            if accumulated_tokens >= target_tokens_to_compress:
                break
        
        # If we still don't have enough savings, compress the entire compressible region
        if accumulated_tokens < target_tokens_to_compress and compress_until < compress_end:
            compress_until = compress_end
            accumulated_tokens = sum(turn_tokens[compress_start:compress_end])

        # Snap the tail boundary so we never cut between a <tool_call> and its
        # <tool_response>: the summary replaces [compress_start, compress_until)
        # and the remainder is kept verbatim, so a boundary on a tool turn would
        # leave an orphaned marker and corrupt the training trajectory.
        compress_until = self._snap_boundary(trajectory, compress_until, compress_start, compress_end)
        if compress_until <= compress_start:
            # Snapping collapsed the region; nothing can be safely compressed.
            metrics.compressed_tokens = total_tokens
            metrics.compressed_turns = len(trajectory)
            metrics.still_over_limit = total_tokens > self.config.target_max_tokens
            return trajectory, metrics

        # If the region we can safely compress is no larger than the summary
        # that would replace it, compression cannot reduce the token count --
        # it would grow the trajectory and still spend a summarization call.
        if (
            sum(turn_tokens[compress_start:compress_until])
            <= self.config.summary_target_tokens
        ):
            metrics.compressed_tokens = total_tokens
            metrics.compressed_turns = len(trajectory)
            metrics.still_over_limit = total_tokens > self.config.target_max_tokens
            return trajectory, metrics

        # Record compression region
        metrics.turns_compressed_start_idx = compress_start
        metrics.turns_compressed_end_idx = compress_until
        metrics.turns_in_compressed_region = compress_until - compress_start

        # Extract content for summary
        content_to_summarize = self._extract_turn_content_for_summary(
            trajectory, compress_start, compress_until
        )

        # Generate summary (ASYNC)
        summary = await self._generate_summary(content_to_summarize, metrics)
        
        # Build compressed trajectory
        compressed = []
        
        # Add head (turns before compression region)
        for i in range(compress_start):
            turn = trajectory[i].copy()
            if turn.get("from") == "system" and self.config.add_summary_notice:
                turn["value"] = turn["value"] + self.config.summary_notice_text
            compressed.append(turn)
        
        # Add summary as human message
        compressed.append({
            "from": "human",
            "value": summary
        })
        
        # Add tail (turns after compression region)
        for i in range(compress_until, len(trajectory)):
            compressed.append(trajectory[i].copy())
        
        # Calculate final metrics
        metrics.compressed_turns = len(compressed)
        metrics.compressed_tokens = await self.count_trajectory_tokens(compressed)
        metrics.turns_removed = metrics.original_turns - metrics.compressed_turns
        metrics.tokens_saved = metrics.original_tokens - metrics.compressed_tokens
        metrics.compression_ratio = metrics.compressed_tokens / max(metrics.original_tokens, 1)
        metrics.was_compressed = True
        metrics.still_over_limit = metrics.compressed_tokens > self.config.target_max_tokens
        
        return compressed, metrics
    
    async def process_entry(self, entry: Dict[str, Any]) -> Tuple[Dict[str, Any], TrajectoryMetrics]:
        """
        Process a single JSONL entry.
        """
        if "conversations" not in entry:
            metrics = TrajectoryMetrics()
            return entry, metrics
        
        trajectory = entry["conversations"]
        compressed_trajectory, metrics = await self.compress_trajectory(trajectory)
        
        # Create new entry with compressed trajectory
        result = entry.copy()
        result["conversations"] = compressed_trajectory
        
        # Add compression metadata if enabled
        if self.config.metrics_per_trajectory and metrics.was_compressed:
            result["compression_metrics"] = metrics.to_dict()
        
        return result, metrics
    
    async def process_directory(self, input_dir: Path, output_dir: Path):
        """
        Async implementation of directory processing with parallel API calls.
        """
        console = Console()
        
        # Record start time
        self.aggregate_metrics.processing_start_time = datetime.now().isoformat()
        start_time = time.time()
        
        # Find all JSONL files
        jsonl_files = await _list_jsonl_files(input_dir)
        
        if not jsonl_files:
            self.logger.warning(f"No JSONL files found in {input_dir}")
            return
        
        # Load ALL entries from all files
        console.print("\n[dim]Loading all entries...[/dim]")
        all_entries = []  # List of (file_path, entry_idx, entry)
        
        for file_path in jsonl_files:
            async with aiofiles.open(file_path, encoding="utf-8") as handle:
                # Upstream's directory path uses enumerate(handle)'s zero-based
                # index (the single-file CLI path intentionally starts at 1).
                line_num = -1
                async for line in handle:
                    line_num += 1
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            all_entries.append((file_path, line_num, entry))
                        except json.JSONDecodeError as exc:
                            self.logger.warning(
                                "Skipping invalid JSON at %s:%d: %s",
                                file_path,
                                line_num,
                                exc,
                            )
        
        total_entries = len(all_entries)
        
        console.print(f"\n{'='*60}")
        console.print(f"📂 Input: {input_dir}")
        console.print(f"📂 Output: {output_dir}")
        console.print(f"📄 Files to process: {len(jsonl_files)}")
        console.print(f"📊 Total trajectories: {total_entries:,}")
        console.print(f"🎯 Target max tokens: {self.config.target_max_tokens:,}")
        console.print(f"📝 Summary target tokens: {self.config.summary_target_tokens}")
        console.print(f"⚡ Max concurrent API calls: {self.config.max_concurrent_requests}")
        console.print(f"{'='*60}\n")
        
        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        # Tracking for progress display across concurrent asyncio tasks.
        progress_lock = asyncio.Lock()
        compressed_count = 0
        skipped_count = 0
        api_calls = 0
        in_flight = 0
        
        # Results storage: {file_path: {entry_idx: (processed_entry, metrics)}}
        results = {f: {} for f in jsonl_files}
        
        # Track timeouts separately
        timeout_count = 0
        
        async def process_single(file_path: Path, entry_idx: int, entry: Dict, 
                                  progress, main_task, status_task):
            """Process a single entry with semaphore rate limiting and timeout."""
            nonlocal compressed_count, skipped_count, api_calls, in_flight, timeout_count
            
            async with semaphore:
                # Track in-flight
                async with progress_lock:
                    in_flight += 1
                
                try:
                    # Apply per-trajectory timeout
                    processed_entry, metrics = await asyncio.wait_for(
                        self.process_entry(entry),
                        timeout=self.config.per_trajectory_timeout
                    )
                    results[file_path][entry_idx] = (processed_entry, metrics)
                    
                    # Update aggregate metrics atomically with the counters.
                    async with progress_lock:
                        self.aggregate_metrics.add_trajectory_metrics(metrics)
                        
                        # Update counters
                        if metrics.was_compressed:
                            compressed_count += 1
                            api_calls += metrics.summarization_api_calls
                        if metrics.skipped_under_target:
                            skipped_count += 1
                        
                        in_flight -= 1
                        
                        # Update progress
                        progress.advance(main_task)
                        progress.update(
                            status_task,
                            description=f"[dim]✅ {compressed_count} compressed | ⏭️ {skipped_count} skipped | ⏱️ {timeout_count} timeout | 🔄 {api_calls} API calls | ⚡ {in_flight} in-flight[/dim]"
                        )
                
                except asyncio.TimeoutError:
                    self.logger.warning(f"Timeout processing entry from {file_path}:{entry_idx} (>{self.config.per_trajectory_timeout}s)")
                    
                    async with progress_lock:
                        self.aggregate_metrics.trajectories_failed += 1
                        timeout_count += 1
                        in_flight -= 1
                        progress.advance(main_task)
                        progress.update(
                            status_task,
                            description=f"[dim]✅ {compressed_count} compressed | ⏭️ {skipped_count} skipped | ⏱️ {timeout_count} timeout | 🔄 {api_calls} API calls | ⚡ {in_flight} in-flight[/dim]"
                        )
                    
                    # Skip this entry entirely (don't include in output)
                    results[file_path][entry_idx] = None
                    
                except Exception as e:
                    self.logger.error(f"Error processing entry from {file_path}:{entry_idx}: {e}")
                    
                    async with progress_lock:
                        self.aggregate_metrics.trajectories_failed += 1
                        in_flight -= 1
                        progress.advance(main_task)
                    
                    # Keep original entry on error
                    results[file_path][entry_idx] = (entry, TrajectoryMetrics())
        
        # Create progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=10  # Higher refresh for async
        ) as progress:
            # Main task for overall progress
            main_task = progress.add_task(
                f"[cyan]Compressing {total_entries:,} trajectories",
                total=total_entries
            )
            
            # Status line task
            status_task = progress.add_task(
                "[dim]Starting...[/dim]",
                total=None
            )
            
            # Create all tasks
            tasks = [
                process_single(file_path, entry_idx, entry, progress, main_task, status_task)
                for file_path, entry_idx, entry in all_entries
            ]
            
            # Run all tasks concurrently (semaphore limits actual concurrency)
            await asyncio.gather(*tasks)
            
            # Remove status task
            progress.remove_task(status_task)
        
        # Write results to output files (preserving original order)
        console.print("\n[dim]Writing output files...[/dim]")
        await aiofiles.os.makedirs(output_dir, exist_ok=True)
        
        for file_path in jsonl_files:
            output_path = output_dir / file_path.name
            file_results = results[file_path]
            
            # Sort by original entry index to preserve order, skip None (timed out) entries
            sorted_entries = [
                file_results[idx][0] 
                for idx in sorted(file_results.keys()) 
                if file_results[idx] is not None
            ]
            
            async with aiofiles.open(output_path, "w", encoding="utf-8") as handle:
                for entry in sorted_entries:
                    await handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        # Record end time
        self.aggregate_metrics.processing_end_time = datetime.now().isoformat()
        self.aggregate_metrics.processing_duration_seconds = time.time() - start_time
        
        # Print summary
        self._print_summary()
        
        # Save metrics
        if self.config.metrics_enabled:
            metrics_path = output_dir / self.config.metrics_output_file
            async with aiofiles.open(metrics_path, "w", encoding="utf-8") as handle:
                await handle.write(json.dumps(self.aggregate_metrics.to_dict(), indent=2))
            console.print(f"\n💾 Metrics saved to {metrics_path}")
    
    def _print_summary(self):
        """Print comprehensive compression summary statistics."""
        m = self.aggregate_metrics.to_dict()
        
        # Calculate some additional stats
        total = m['summary']['total_trajectories']
        compressed = m['summary']['trajectories_compressed']
        skipped = m['summary']['trajectories_skipped_under_target']
        over_limit = m['summary']['trajectories_still_over_limit']
        failed = m['summary']['trajectories_failed']
        
        # Token stats
        tokens_before = m['tokens']['total_before']
        tokens_after = m['tokens']['total_after']
        tokens_saved = m['tokens']['total_saved']
        
        # Calculate percentages
        compressed_pct = (compressed / max(total, 1)) * 100
        skipped_pct = (skipped / max(total, 1)) * 100
        over_limit_pct = (over_limit / max(total, 1)) * 100
        
        print("\n")
        print(f"╔{'═'*70}╗")
        print(f"║{'TRAJECTORY COMPRESSION REPORT':^70}║")
        print(f"╠{'═'*70}╣")
        
        # Trajectories section
        print(f"║{'':2}📁 TRAJECTORIES{' '*54}║")
        print(f"║{'─'*70}║")
        print(f"║{'':4}Total Processed:        {total:>10,}{' '*32}║")
        print(f"║{'':4}├─ Compressed:          {compressed:>10,}  ({compressed_pct:>5.1f}%){' '*18}║")
        print(f"║{'':4}├─ Skipped (under limit):{skipped:>9,}  ({skipped_pct:>5.1f}%){' '*18}║")
        print(f"║{'':4}├─ Still over limit:    {over_limit:>10,}  ({over_limit_pct:>5.1f}%){' '*18}║")
        print(f"║{'':4}└─ Failed:              {failed:>10,}{' '*32}║")
        
        print(f"╠{'═'*70}╣")
        
        # Tokens section
        print(f"║{'':2}🔢 TOKENS{' '*60}║")
        print(f"║{'─'*70}║")
        print(f"║{'':4}Before Compression:     {tokens_before:>15,} tokens{' '*21}║")
        print(f"║{'':4}After Compression:      {tokens_after:>15,} tokens{' '*21}║")
        print(f"║{'':4}Total Saved:            {tokens_saved:>15,} tokens{' '*21}║")
        print(f"║{'':4}Overall Compression:    {m['tokens']['overall_compression_ratio']:>14.1%}{' '*28}║")
        
        if tokens_before > 0:
            savings_pct = (tokens_saved / tokens_before) * 100
            print(f"║{'':4}Space Savings:          {savings_pct:>14.1f}%{' '*28}║")
        
        print(f"╠{'═'*70}╣")
        
        # Turns section
        print(f"║{'':2}💬 CONVERSATION TURNS{' '*48}║")
        print(f"║{'─'*70}║")
        print(f"║{'':4}Before Compression:     {m['turns']['total_before']:>15,} turns{' '*22}║")
        print(f"║{'':4}After Compression:      {m['turns']['total_after']:>15,} turns{' '*22}║")
        print(f"║{'':4}Total Removed:          {m['turns']['total_removed']:>15,} turns{' '*22}║")
        
        print(f"╠{'═'*70}╣")
        
        # Averages section (for compressed trajectories only)
        print(f"║{'':2}📈 AVERAGES (Compressed Trajectories Only){' '*27}║")
        print(f"║{'─'*70}║")
        if compressed > 0:
            print(f"║{'':4}Avg Compression Ratio:  {m['averages']['avg_compression_ratio']:>14.1%}{' '*28}║")
            print(f"║{'':4}Avg Tokens Saved:       {m['averages']['avg_tokens_saved_per_compressed']:>14,.0f}{' '*28}║")
            print(f"║{'':4}Avg Turns Removed:      {m['averages']['avg_turns_removed_per_compressed']:>14.1f}{' '*28}║")
        else:
            print(f"║{'':4}No trajectories were compressed{' '*38}║")
        
        print(f"╠{'═'*70}╣")
        
        # Summarization API section
        print(f"║{'':2}🤖 SUMMARIZATION API{' '*49}║")
        print(f"║{'─'*70}║")
        print(f"║{'':4}API Calls Made:         {m['summarization']['total_api_calls']:>15,}{' '*27}║")
        print(f"║{'':4}Errors:                 {m['summarization']['total_errors']:>15,}{' '*27}║")
        print(f"║{'':4}Success Rate:           {m['summarization']['success_rate']:>14.1%}{' '*28}║")
        
        print(f"╠{'═'*70}╣")
        
        # Processing time section
        duration = m['processing']['duration_seconds']
        if duration > 60:
            time_str = f"{duration/60:.1f} minutes"
        else:
            time_str = f"{duration:.1f} seconds"
        
        throughput = total / max(duration, 0.001)
        
        print(f"║{'':2}⏱️  PROCESSING TIME{' '*51}║")
        print(f"║{'─'*70}║")
        print(f"║{'':4}Duration:               {time_str:>20}{' '*22}║")
        print(f"║{'':4}Throughput:             {throughput:>15.1f} traj/sec{' '*18}║")
        print(f"║{'':4}Started:                {m['processing']['start_time'][:19]:>20}{' '*22}║")
        print(f"║{'':4}Finished:               {m['processing']['end_time'][:19]:>20}{' '*22}║")
        
        print(f"╚{'═'*70}╝")
        
        # Distribution summary if we have data
        if self.aggregate_metrics.compression_ratios:
            ratios = self.aggregate_metrics.compression_ratios
            tokens_saved_list = self.aggregate_metrics.tokens_saved_list
            
            print("\n📊 Distribution Summary:")
            print(f"   Compression ratios: min={min(ratios):.2%}, max={max(ratios):.2%}, median={sorted(ratios)[len(ratios)//2]:.2%}")
            print(f"   Tokens saved:       min={min(tokens_saved_list):,}, max={max(tokens_saved_list):,}, median={sorted(tokens_saved_list)[len(tokens_saved_list)//2]:,}")


async def main(
    input: str,
    output: str = None,
    config: str = "configs/trajectory_compression.yaml",
    target_max_tokens: int = None,
    tokenizer: str = None,
    sample_percent: float = None,
    seed: int = 42,
    dry_run: bool = False,
):
    """
    Compress agent trajectories to fit within a target token budget.
    
    Supports both single JSONL files and directories containing multiple JSONL files.
    Optionally sample a percentage of trajectories before compression.
    
    Args:
        input: Path to JSONL file or directory containing JSONL files
        output: Output path (file for file input, directory for dir input)
                Default: adds "_compressed" suffix to input name
        config: Path to YAML configuration file
        target_max_tokens: Override target token count from config
        tokenizer: Override tokenizer name from config
        sample_percent: Sample this percentage of trajectories (1-100) before compression
        seed: Random seed for sampling reproducibility (default: 42)
        dry_run: Analyze without compressing (just show what would happen)
    
    Examples:
        # Compress a directory (original behavior)
        python trajectory_compressor.py --input=data/my_run
        
        # Compress a single file
        python trajectory_compressor.py --input=data/trajectories.jsonl
        
        # Compress 15% sample of a file
        python trajectory_compressor.py --input=data/trajectories.jsonl --sample_percent=15
        
        # Compress 10% sample with custom output
        python trajectory_compressor.py --input=data/trajectories.jsonl --sample_percent=10 --output=data/sampled_compressed.jsonl
    """
    print("🗜️  Trajectory Compressor")
    print("=" * 60)
    
    # Load configuration
    config_path = Path(config)
    if await aiofiles.os.path.exists(config_path):
        print(f"📋 Loading config from {config}")
        compression_config = await CompressionConfig.from_yaml(config)
    else:
        print(f"⚠️  Config not found at {config}, using defaults")
        compression_config = CompressionConfig()
    
    # Apply CLI overrides
    if target_max_tokens:
        compression_config.target_max_tokens = target_max_tokens
    if tokenizer:
        compression_config.tokenizer_name = tokenizer
    
    # Validate sample_percent
    if sample_percent is not None:
        if sample_percent <= 0 or sample_percent > 100:
            print(f"❌ sample_percent must be between 1 and 100, got {sample_percent}")
            return
        print(f"🎲 Will sample {sample_percent}% of trajectories (seed={seed})")
    
    # Setup paths and determine input type
    input_path = Path(input)
    if not await aiofiles.os.path.exists(input_path):
        print(f"❌ Input not found: {input}")
        return
    
    is_file_input = await aiofiles.os.path.isfile(input_path)
    
    if is_file_input:
        print("📄 Input mode: Single JSONL file")
        
        # For file input, default output is file with _compressed suffix
        if output:
            output_path = Path(output)
        else:
            output_path = input_path.parent / (input_path.stem + compression_config.output_suffix + ".jsonl")
        
        # Load entries from the single file
        entries = []
        async with aiofiles.open(input_path, "r", encoding="utf-8") as source:
            line_num = 0
            async for line in source:
                line_num += 1
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Skipping invalid JSON at line {line_num}: {e}")
        
        total_entries = len(entries)
        print(f"   Loaded {total_entries:,} trajectories from {input_path.name}")
        
        # Sample if requested
        if sample_percent is not None:
            random.seed(seed)
            sample_size = max(1, int(total_entries * sample_percent / 100))
            entries = random.sample(entries, sample_size)
            print(f"   Sampled {len(entries):,} trajectories ({sample_percent}% of {total_entries:,})")
        
        if dry_run:
            print("\n🔍 DRY RUN MODE - analyzing without writing")
            print(f"📄 Would process: {len(entries):,} trajectories")
            print(f"📄 Would output to: {output_path}")
            return
        
        # Create a temporary directory for processing
        async with aiofiles.tempfile.TemporaryDirectory() as temp_dir:
            temp_input_dir = Path(temp_dir) / "input"
            temp_output_dir = Path(temp_dir) / "output"
            await aiofiles.os.makedirs(temp_input_dir)
            
            # Write entries to temp file
            temp_input_file = temp_input_dir / "trajectories.jsonl"
            async with aiofiles.open(
                temp_input_file, "w", encoding="utf-8"
            ) as temporary_input:
                for entry in entries:
                    await temporary_input.write(
                        json.dumps(entry, ensure_ascii=False) + "\n"
                    )
            
            # Initialize compressor and process
            compressor = TrajectoryCompressor(compression_config)
            try:
                await compressor.process_directory(temp_input_dir, temp_output_dir)
            finally:
                await compressor.close()
            
            # Copy result to output path (merge all files in temp_output_dir)
            await aiofiles.os.makedirs(output_path.parent, exist_ok=True)
            async with aiofiles.open(
                output_path, "w", encoding="utf-8"
            ) as output_file:
                for jsonl_file in await _list_jsonl_files(temp_output_dir):
                    async with aiofiles.open(
                        jsonl_file, "r", encoding="utf-8"
                    ) as input_file:
                        async for line in input_file:
                            await output_file.write(line)
            
            # Copy metrics file if it exists
            metrics_file = temp_output_dir / compression_config.metrics_output_file
            if await aiofiles.os.path.exists(metrics_file):
                metrics_output = output_path.parent / (output_path.stem + "_metrics.json")
                async with aiofiles.open(metrics_file, "rb") as metrics_source:
                    metrics_payload = await metrics_source.read()
                async with aiofiles.open(metrics_output, "wb") as metrics_target:
                    await metrics_target.write(metrics_payload)
                print(f"💾 Metrics saved to {metrics_output}")
        
        print("\n✅ Compression complete!")
        print(f"📄 Output: {output_path}")
        
    else:
        # Directory input - original behavior
        print("📁 Input mode: Directory of JSONL files")
        
        if output:
            output_path = Path(output)
        else:
            output_path = input_path.parent / (input_path.name + compression_config.output_suffix)
        
        # If sampling is requested for directory mode, we need to handle it differently
        if sample_percent is not None:
            print(f"\n⚠️  Sampling from directory: will sample {sample_percent}% from each file")
            
            # Create a temp directory with sampled files
            async with aiofiles.tempfile.TemporaryDirectory() as temp_dir:
                temp_input_dir = Path(temp_dir) / "input"
                await aiofiles.os.makedirs(temp_input_dir)
                
                random.seed(seed)
                total_original = 0
                total_sampled = 0
                
                # Sample from each JSONL file
                for jsonl_file in await _list_jsonl_files(input_path):
                    entries = []
                    async with aiofiles.open(
                        jsonl_file, "r", encoding="utf-8"
                    ) as source:
                        async for line in source:
                            line = line.strip()
                            if line:
                                try:
                                    entries.append(json.loads(line))
                                except json.JSONDecodeError:
                                    pass
                    
                    total_original += len(entries)
                    sample_size = max(1, int(len(entries) * sample_percent / 100))
                    sampled_entries = random.sample(entries, min(sample_size, len(entries)))
                    total_sampled += len(sampled_entries)
                    
                    # Write sampled entries
                    temp_file = temp_input_dir / jsonl_file.name
                    async with aiofiles.open(
                        temp_file, "w", encoding="utf-8"
                    ) as sampled_file:
                        for entry in sampled_entries:
                            await sampled_file.write(
                                json.dumps(entry, ensure_ascii=False) + "\n"
                            )
                
                print(f"   Sampled {total_sampled:,} from {total_original:,} total trajectories")
                
                if dry_run:
                    print("\n🔍 DRY RUN MODE - analyzing without writing")
                    print(f"📁 Would process: {temp_input_dir}")
                    print(f"📁 Would output to: {output_path}")
                    return
                
                # Initialize compressor and process the sampled data
                compressor = TrajectoryCompressor(compression_config)
                try:
                    await compressor.process_directory(temp_input_dir, output_path)
                finally:
                    await compressor.close()
        else:
            if dry_run:
                print("\n🔍 DRY RUN MODE - analyzing without writing")
                print(f"📁 Would process: {input_path}")
                print(f"📁 Would output to: {output_path}")
                return
            
            # Initialize compressor and process directly
            compressor = TrajectoryCompressor(compression_config)
            try:
                await compressor.process_directory(input_path, output_path)
            finally:
                await compressor.close()
        
        print("\n✅ Compression complete!")
