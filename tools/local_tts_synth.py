#!/usr/bin/env python3
"""Isolated entry point for local TTS SDKs without coroutine APIs.

Piper and KittenTTS perform model and filesystem I/O from synchronous SDK
methods. The retained async tool launches this program as a profile-scoped
broker with ``asyncio.create_subprocess_exec`` so those SDKs never run on the
agent event loop, loaded models retain upstream's bounded LRU lifetime, and
the child can be cancelled and reaped deterministically.
"""

from __future__ import annotations

import contextlib
import json
import sys
import wave
from pathlib import Path
from typing import Any


_TTS_MODEL_CACHE_MAX = 3
_RESPONSE_PREFIX = "HERMES_TTS_RESPONSE:"
_piper_voice_cache: dict[Any, Any] = {}
_kittentts_model_cache: dict[Any, Any] = {}


def _tts_cache_get_or_load(
    cache: dict[Any, Any],
    key: Any,
    load,
) -> Any:
    """Mirror the retained runtime's exact insertion-ordered LRU contract."""
    if key in cache:
        cache[key] = cache.pop(key)
        return cache[key]
    value = load()
    cache[key] = value
    while len(cache) > _TTS_MODEL_CACHE_MAX:
        cache.pop(next(iter(cache)), None)
    return value


def _synthesize_piper(payload: dict[str, Any]) -> None:
    from piper import PiperVoice

    model_path = str(payload["model_path"])
    output_path = str(payload["output_path"])
    use_cuda = bool(payload.get("use_cuda", False))
    speaker_id = payload.get("speaker_id", 0)
    if isinstance(speaker_id, bool) or not isinstance(speaker_id, int):
        speaker_id = 0

    cache_key = f"{model_path}::cuda={use_cuda}"

    def load_voice():
        return PiperVoice.load(model_path, use_cuda=use_cuda)

    voice = _tts_cache_get_or_load(
        _piper_voice_cache,
        cache_key,
        load_voice,
    )
    syn_config = None
    if payload.get("has_advanced", False):
        try:
            from piper import SynthesisConfig

            syn_config = SynthesisConfig(
                length_scale=float(payload.get("length_scale", 1.0)),
                noise_scale=float(payload.get("noise_scale", 0.667)),
                noise_w_scale=float(payload.get("noise_w_scale", 0.8)),
                volume=float(payload.get("volume", 1.0)),
                normalize_audio=bool(payload.get("normalize_audio", True)),
                speaker_id=speaker_id,
            )
        except ImportError:
            pass

    with wave.open(output_path, "wb") as wav_file:
        if syn_config is None:
            voice.synthesize_wav(str(payload["text"]), wav_file)
        else:
            voice.synthesize_wav(
                str(payload["text"]),
                wav_file,
                syn_config=syn_config,
            )


def _synthesize_kittentts(payload: dict[str, Any]) -> None:
    import soundfile as sf
    from kittentts import KittenTTS

    model_name = payload["model"]

    def load_model():
        return KittenTTS(model_name)

    model = _tts_cache_get_or_load(
        _kittentts_model_cache,
        model_name,
        load_model,
    )
    audio = model.generate(
        str(payload["text"]),
        voice=payload.get("voice", "Jasper"),
        speed=payload.get("speed", 1.0),
        clean_text=payload.get("clean_text", True),
    )
    sf.write(str(payload["output_path"]), audio, 24000)


def _synthesize(provider: str, payload: dict[str, Any]) -> None:
    Path(str(payload["output_path"])).parent.mkdir(parents=True, exist_ok=True)
    if provider == "piper":
        _synthesize_piper(payload)
    elif provider == "kittentts":
        _synthesize_kittentts(payload)
    else:
        raise ValueError(f"unsupported local TTS provider: {provider}")


def _write_response(response: dict[str, Any]) -> None:
    sys.stdout.write(
        _RESPONSE_PREFIX
        + json.dumps(response, ensure_ascii=False, separators=(",", ":"))
        + "\n"
    )
    sys.stdout.flush()


def _run_broker() -> int:
    """Serve serialized synthesis requests while retaining the model LRUs."""
    for raw_request in sys.stdin:
        request_id = ""
        try:
            request = json.loads(raw_request)
            if request.get("command") == "shutdown":
                return 0
            request_id = str(request.get("id", ""))
            provider = str(request["provider"])
            payload = request["payload"]
            if not isinstance(payload, dict):
                raise TypeError("local TTS payload must be an object")
            # Reserve stdout for response frames. Local SDKs commonly print
            # model progress to stdout; route that diagnostic output to the
            # broker's drained stderr pipe so it cannot corrupt framing.
            with contextlib.redirect_stdout(sys.stderr):
                _synthesize(provider, payload)
        except Exception as exc:
            _write_response(
                {
                    "id": request_id,
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        else:
            _write_response({"id": request_id, "ok": True})
    return 0


def _main() -> int:
    if len(sys.argv) != 2:
        print(
            "usage: local_tts_synth.py {--broker|piper|kittentts}",
            file=sys.stderr,
        )
        return 2
    if sys.argv[1] == "--broker":
        return _run_broker()
    if sys.argv[1] not in {"piper", "kittentts"}:
        print(
            "usage: local_tts_synth.py {--broker|piper|kittentts}",
            file=sys.stderr,
        )
        return 2
    payload = json.load(sys.stdin)
    _synthesize(sys.argv[1], payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
