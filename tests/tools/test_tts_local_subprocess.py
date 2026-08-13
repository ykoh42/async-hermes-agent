"""Real subprocess coverage for sync-only local TTS SDK boundaries."""

import asyncio

import aiofiles
import pytest

from tools import tts_tool


def test_local_tts_child_entrypoint_is_not_public_api():
    from tools import local_tts_synth

    assert not hasattr(local_tts_synth, "main")
    assert local_tts_synth._main.__module__ == local_tts_synth.__name__


async def _install_fake_sdks(root, *, hang=False):
    async with aiofiles.open(root / "piper.py", "w") as module:
        await module.write(
            "import wave\n"
            "class SynthesisConfig:\n"
            "    def __init__(self, **kwargs): self.kwargs = kwargs\n"
            "class Voice:\n"
            "    def synthesize_wav(self, text, output, syn_config=None):\n"
            "        output.setnchannels(1); output.setsampwidth(2); "
            "output.setframerate(22050); output.writeframes(b'\\0\\0' * 32)\n"
            "class PiperVoice:\n"
            "    @classmethod\n"
            "    def load(cls, model_path, use_cuda=False): return Voice()\n"
        )
    async with aiofiles.open(root / "kittentts.py", "w") as module:
        await module.write(
            "import time\n"
            "class KittenTTS:\n"
            "    def __init__(self, model): self.model = model\n"
            "    def generate(self, text, **kwargs):\n"
            f"        time.sleep({60 if hang else 0})\n"
            "        return [0.0] * 32\n"
        )
    async with aiofiles.open(root / "soundfile.py", "w") as module:
        await module.write(
            "def write(path, audio, samplerate):\n"
            "    with open(path, 'wb') as output:\n"
            "        output.write(b'RIFF\\0\\0\\0\\0WAVE')\n"
        )


@pytest.mark.asyncio
async def test_real_owned_subprocess_runs_piper_and_kittentts(tmp_path, monkeypatch):
    await _install_fake_sdks(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    model = tmp_path / "voice.onnx"
    async with aiofiles.open(model, "wb") as model_file:
        await model_file.write(b"model")

    piper_output = tmp_path / "piper.wav"
    kitten_output = tmp_path / "kitten.wav"
    assert await tts_tool._generate_piper_tts(
        "hello",
        str(piper_output),
        {"piper": {"voice": str(model)}},
    ) == str(piper_output)
    assert await tts_tool._generate_kittentts(
        "hello",
        str(kitten_output),
        {},
    ) == str(kitten_output)
    async with aiofiles.open(piper_output, "rb") as output:
        assert (await output.read()).startswith(b"RIFF")
    async with aiofiles.open(kitten_output, "rb") as output:
        assert (await output.read()).startswith(b"RIFF")


@pytest.mark.asyncio
async def test_local_sdk_cancellation_reaps_child(tmp_path, monkeypatch):
    await _install_fake_sdks(tmp_path, hang=True)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    original_create = asyncio.create_subprocess_exec
    children = []

    async def capture_child(*args, **kwargs):
        process = await original_create(*args, **kwargs)
        if len(args) > 1 and str(args[1]).endswith("local_tts_synth.py"):
            children.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture_child)
    task = asyncio.create_task(
        tts_tool._generate_kittentts(
            "hello",
            str(tmp_path / "cancelled.wav"),
            {},
        )
    )
    while not children:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    assert children[0].returncode is not None
