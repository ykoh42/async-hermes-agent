"""Credential isolation for non-model TTS subprocesses."""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
from pathlib import Path

import aiofiles
import pytest

from agent import secret_scope
from tools import tts_tool


@pytest.fixture(autouse=True)
def _restore_secret_scope():
    previous_multiplex = secret_scope.is_multiplex_active()
    scope_token = secret_scope.set_secret_scope(None)
    try:
        yield
    finally:
        secret_scope.reset_secret_scope(scope_token)
        secret_scope.set_multiplex_active(previous_multiplex)


class _CompletedProcess:
    returncode = 0

    async def wait(self) -> int:
        return self.returncode

    async def communicate(self, input=None) -> tuple[bytes, bytes]:
        del input
        return b"", b""


@pytest.mark.asyncio
async def test_probe_ffmpeg_and_piper_fake_spawns_receive_no_credentials(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setenv("OPENAI_API_KEY", "foreign-openai")
    monkeypatch.setenv("HF_TOKEN", "foreign-hugging-face")
    monkeypatch.setenv("TTS_RUNTIME_MARKER", "preserved")
    captured: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def spawn(*args, **kwargs):
        captured.append((args, kwargs))
        if "piper.download_voices" in args:
            voice = str(args[args.index("piper.download_voices") + 1])
            download_dir = Path(str(args[args.index("--download-dir") + 1]))
            async with aiofiles.open(download_dir / f"{voice}.onnx", "wb") as model:
                await model.write(b"model")
            async with aiofiles.open(
                download_dir / f"{voice}.onnx.json",
                "w",
            ) as config:
                await config.write("{}")
        return _CompletedProcess()

    monkeypatch.setattr(tts_tool.asyncio, "create_subprocess_exec", spawn)

    assert await tts_tool._has_ffmpeg() is True
    assert await tts_tool._check_python_module_available("piper") is True
    voice_path = await tts_tool._resolve_piper_voice_path("test-voice", tmp_path)
    assert voice_path == str(tmp_path / "test-voice.onnx")

    assert len(captured) == 3
    for _args, kwargs in captured:
        child_env = kwargs["env"]
        assert isinstance(child_env, dict)
        assert child_env["TTS_RUNTIME_MARKER"] == "preserved"
        assert "OPENAI_API_KEY" not in child_env
        assert "HF_TOKEN" not in child_env


async def _run_neutts_profile(
    tmp_path: Path,
    label: str,
    hf_token: str,
) -> dict[str, str | None]:
    ref_audio = tmp_path / f"{label}.wav"
    ref_text = tmp_path / f"{label}.txt"
    output = tmp_path / f"{label}-output.wav"
    env_log = tmp_path / f"{label}-env.json"
    async with aiofiles.open(ref_audio, "wb") as audio_file:
        await audio_file.write(b"RIFF")
    async with aiofiles.open(ref_text, "w") as text_file:
        await text_file.write("reference")

    token = secret_scope.set_secret_scope({"HF_TOKEN": hf_token})
    try:
        await asyncio.sleep(0)
        result = await tts_tool._generate_neutts(
            "hello",
            str(output),
            {
                "neutts": {
                    "ref_audio": str(ref_audio),
                    "ref_text": str(ref_text),
                    "model": str(env_log),
                }
            },
        )
    finally:
        secret_scope.reset_secret_scope(token)

    assert result == str(output)
    async with aiofiles.open(env_log) as log_file:
        return json.loads(await log_file.read())


@pytest.mark.asyncio
async def test_real_neutts_children_receive_only_their_profile_hf_token(
    monkeypatch,
    tmp_path: Path,
):
    fake_sdk = tmp_path / "fake-sdk"
    fake_sdk.mkdir()
    async with aiofiles.open(fake_sdk / "neutts.py", "w") as module:
        await module.write(
            "import json, os\n"
            "class NeuTTS:\n"
            "    def __init__(self, backbone_repo, **kwargs):\n"
            "        del kwargs\n"
            "        with open(backbone_repo, 'w') as log:\n"
            "            json.dump({\n"
            "                'hf': os.environ.get('HF_TOKEN'),\n"
            "                'openai': os.environ.get('OPENAI_API_KEY'),\n"
            "            }, log)\n"
            "    def encode_reference(self, path):\n"
            "        return []\n"
            "    def infer(self, text, codes, ref_text):\n"
            "        return [0.0, 0.0]\n"
        )

    monkeypatch.setenv("PYTHONPATH", str(fake_sdk))
    monkeypatch.setenv("HF_TOKEN", "foreign-process-hf")
    monkeypatch.setenv("OPENAI_API_KEY", "foreign-process-openai")
    secret_scope.set_multiplex_active(True)

    profile_a, profile_b = await asyncio.gather(
        _run_neutts_profile(tmp_path, "a", "profile-a-hf"),
        _run_neutts_profile(tmp_path, "b", "profile-b-hf"),
    )

    assert profile_a == {"hf": "profile-a-hf", "openai": None}
    assert profile_b == {"hf": "profile-b-hf", "openai": None}


@pytest.mark.asyncio
async def test_neutts_hf_token_fails_closed_without_profile_scope(
    monkeypatch,
    tmp_path: Path,
):
    spawned = False

    async def spawn(*_args, **_kwargs):
        nonlocal spawned
        await asyncio.sleep(0)
        spawned = True
        return _CompletedProcess()

    monkeypatch.setattr(tts_tool.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setenv("HF_TOKEN", "foreign-process-hf")
    secret_scope.set_multiplex_active(True)

    with pytest.raises(secret_scope.UnscopedSecretError, match="HF_TOKEN"):
        await tts_tool._generate_neutts(
            "hello",
            str(tmp_path / "output.wav"),
            {},
        )

    assert spawned is False


@pytest.mark.asyncio
async def test_single_profile_neutts_preserves_process_hf_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "legacy-process-hf")
    child_env = await tts_tool._non_model_tts_subprocess_env(
        include_hf_token=True,
    )
    assert child_env["HF_TOKEN"] == "legacy-process-hf"


@pytest.mark.asyncio
async def test_multiplex_explicit_empty_hf_token_does_not_borrow_process_value(
    monkeypatch,
):
    monkeypatch.setenv("HF_TOKEN", "foreign-process-hf")
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"HF_TOKEN": ""})
    try:
        child_env = await tts_tool._non_model_tts_subprocess_env(
            include_hf_token=True,
        )
    finally:
        secret_scope.reset_secret_scope(token)

    assert child_env["HF_TOKEN"] == ""


@pytest.mark.asyncio
async def test_neutts_cancellation_reaps_real_child(
    monkeypatch,
    tmp_path: Path,
):
    child_script = tmp_path / "neutts_synth.py"
    started = tmp_path / "started"
    async with aiofiles.open(child_script, "w") as script:
        await script.write(
            "import pathlib, sys, time\n"
            "pathlib.Path(sys.argv[1]).touch()\n"
            "time.sleep(60)\n"
        )

    original_spawn = asyncio.create_subprocess_exec
    children = []

    async def spawn(*args, **kwargs):
        rewritten = tuple(
            str(child_script) if str(arg).endswith("neutts_synth.py") else arg
            for arg in args
        )
        rewritten = (*rewritten[:2], str(started), *rewritten[2:])
        process = await original_spawn(*rewritten, **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(tts_tool.asyncio, "create_subprocess_exec", spawn)
    task = asyncio.create_task(
        tts_tool._generate_neutts(
            "hello",
            str(tmp_path / "output.wav"),
            {},
        )
    )
    while not await aiofiles.os.path.exists(started):
        await asyncio.sleep(0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    assert children[0].returncode is not None


def test_all_non_model_tts_exec_sinks_pass_an_explicit_environment():
    functions = (
        tts_tool._has_ffmpeg,
        tts_tool._ffmpeg_transcode_to_opus,
        tts_tool._generate_gemini_tts,
        tts_tool._check_python_module_available,
        tts_tool._generate_neutts,
        tts_tool._resolve_piper_voice_path,
        tts_tool._finalize_local_wav,
    )
    call_count = 0
    for function in functions:
        tree = ast.parse(inspect.getsource(function))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "create_subprocess_exec":
                continue
            call_count += 1
            assert any(keyword.arg == "env" for keyword in node.keywords), (
                f"{function.__name__} has an inherited-environment subprocess"
            )
    assert call_count == 8
