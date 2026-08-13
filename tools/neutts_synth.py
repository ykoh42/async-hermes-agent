#!/usr/bin/env python3
"""Standalone NeuTTS synthesis helper.

Called by tts_tool.py via subprocess to keep the TTS model (~500MB)
in a separate process that exits after synthesis — no lingering memory.

Usage:
    python -m tools.neutts_synth --text "Hello" --out output.wav \
        --ref-audio samples/jo.wav --ref-text samples/jo.txt

Requires: python -m pip install -U neutts[all]
System:   apt install espeak-ng  (or brew install espeak-ng)
"""

import argparse
import struct
import sys
from pathlib import Path


def _write_wav(path: str, samples, sample_rate: int = 24000) -> None:
    """Write mono float samples as PCM16 without optional audio packages."""
    # NeuTTS normally returns a NumPy array or torch Tensor because its own
    # ``[all]`` install includes those runtimes. Keep the fallback usable with
    # plain Python iterables too: tests and minimal downstream integrations can
    # exercise the subprocess boundary without installing either NumPy or
    # soundfile solely to serialize two-byte PCM frames.
    if hasattr(samples, "detach"):
        samples = samples.detach()
    if hasattr(samples, "cpu"):
        samples = samples.cpu()
    if hasattr(samples, "reshape") and hasattr(samples, "tolist"):
        samples = samples.reshape(-1).tolist()
    elif hasattr(samples, "tolist"):
        samples = samples.tolist()

    def flatten(values):
        if isinstance(values, (str, bytes)):
            yield float(values)
            return
        try:
            iterator = iter(values)
        except TypeError:
            yield float(values)
            return
        for value in iterator:
            yield from flatten(value)

    pcm = bytearray()
    for sample in flatten(samples):
        clamped = max(-1.0, min(1.0, sample))
        pcm.extend(struct.pack("<h", int(clamped * 32767)))

    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)
    data_size = len(pcm)

    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, num_channels, sample_rate,
                            byte_rate, block_align, bits_per_sample))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(pcm)


def main():
    parser = argparse.ArgumentParser(description="NeuTTS synthesis helper")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--out", required=True, help="Output WAV path")
    parser.add_argument("--ref-audio", required=True, help="Reference voice audio path")
    parser.add_argument("--ref-text", required=True, help="Reference voice transcript path")
    parser.add_argument("--model", default="neuphonic/neutts-air-q4-gguf",
                        help="HuggingFace backbone model repo")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda/mps)")
    args = parser.parse_args()

    # llama_cpp (backbone) offloads to GPU only for the literal string "gpu";
    # torch (codec) only accepts "cuda". A single --device value can't satisfy
    # both — "cuda" silently no-ops on the backbone, leaving it on CPU.
    backbone_device = "gpu" if args.device == "cuda" else args.device
    codec_device = args.device

    # Validate inputs
    ref_audio = Path(args.ref_audio).expanduser()
    ref_text_path = Path(args.ref_text).expanduser()
    if not ref_audio.exists():
        print(f"Error: reference audio not found: {ref_audio}", file=sys.stderr)
        sys.exit(1)
    if not ref_text_path.exists():
        print(f"Error: reference text not found: {ref_text_path}", file=sys.stderr)
        sys.exit(1)

    ref_text = ref_text_path.read_text(encoding="utf-8").strip()

    # Import and run NeuTTS
    try:
        from neutts import NeuTTS
    except ImportError:
        print("Error: neutts not installed. Run: python -m pip install -U neutts[all]", file=sys.stderr)
        sys.exit(1)

    tts = NeuTTS(
        backbone_repo=args.model,
        backbone_device=backbone_device,
        codec_repo="neuphonic/neucodec",
        codec_device=codec_device,
    )
    ref_codes = tts.encode_reference(str(ref_audio))
    wav = tts.infer(args.text, ref_codes, ref_text)

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import soundfile as sf
        sf.write(str(out_path), wav, 24000)
    except ImportError:
        _write_wav(str(out_path), wav, 24000)

    print(f"OK: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
